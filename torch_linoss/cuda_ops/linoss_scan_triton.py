"""
Triton kernels for the LinOSS recurrence

    h_t = M h_{t-1} + f x_t,   y_t = h_t[1],   h_t in R^2,

run independently for every (batch, channel). M (2x2) and f (2,) are per channel and constant in time.

Each program owns one batch element and BLOCK_D channels, and walks over the sequence in chunks of BLOCK_L
steps, keeping the 2-vector state in registers. Inside a chunk the recurrence is evaluated with
`tl.associative_scan` over (matrix, vector) pairs, the same operator as `binary_operator` in the JAX code.
The carried state enters through the first row of the chunk (b_0 += M h_carry), so no matrix powers are
materialized. Steps past the end of the sequence use the identity matrix, so the carried state is exact.

The forward kernel also stores the state at the start of every chunk. The backward kernel walks the chunks in
reverse, runs the adjoint recurrence lambda_t = M^T lambda_{t+1} + e_2 g_t with a reverse scan, recomputes
h_{t-1} from the saved chunk-start state, and accumulates

    dx_t = f . lambda_t,   df = sum_t lambda_t x_t,   dM = sum_t lambda_t h_{t-1}^T,   dh_init = M^T lambda_0.

All arithmetic is fp32, whatever the input dtype.
"""

import triton
import triton.language as tl


@triton.jit
def _combine(a11, a12, a21, a22, a1, a2, b11, b12, b21, b22, b1, b2):
    # Compose two affine maps v -> A v + a: apply (A, a) first, then (B, b).
    c11 = b11 * a11 + b12 * a21
    c12 = b11 * a12 + b12 * a22
    c21 = b21 * a11 + b22 * a21
    c22 = b21 * a12 + b22 * a22
    c1 = b11 * a1 + b12 * a2 + b1
    c2 = b21 * a1 + b22 * a2 + b2
    return c11, c12, c21, c22, c1, c2


@triton.jit
def _linoss_scan_fwd_kernel(
    X, Y, MF, H0, HS, HT,
    L, D, NC,
    sxb, sxd, sxl,
    syb, syd, syl,
    HAS_H0: tl.constexpr,
    BLOCK_L: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    offs_l = tl.arange(0, BLOCK_L)

    # MF: (D, 6) = [m11, m12, m21, m22, f1, f2]
    m11 = tl.load(MF + offs_d * 6 + 0, mask=mask_d, other=1.0)
    m12 = tl.load(MF + offs_d * 6 + 1, mask=mask_d, other=0.0)
    m21 = tl.load(MF + offs_d * 6 + 2, mask=mask_d, other=0.0)
    m22 = tl.load(MF + offs_d * 6 + 3, mask=mask_d, other=1.0)
    f1 = tl.load(MF + offs_d * 6 + 4, mask=mask_d, other=0.0)
    f2 = tl.load(MF + offs_d * 6 + 5, mask=mask_d, other=0.0)

    if HAS_H0:
        h1 = tl.load(H0 + (pid_b * D + offs_d) * 2 + 0, mask=mask_d, other=0.0)
        h2 = tl.load(H0 + (pid_b * D + offs_d) * 2 + 1, mask=mask_d, other=0.0)
    else:
        h1 = tl.zeros([BLOCK_D], dtype=tl.float32)
        h2 = tl.zeros([BLOCK_D], dtype=tl.float32)

    first = (offs_l == 0)[:, None]
    last = (offs_l == BLOCK_L - 1)[:, None]
    for c in range(0, NC):
        # HS: (B, NC, D, 2) state before the first step of chunk c.
        hs_ptr = HS + ((pid_b * NC + c) * D + offs_d) * 2
        tl.store(hs_ptr + 0, h1, mask=mask_d)
        tl.store(hs_ptr + 1, h2, mask=mask_d)

        ls = c * BLOCK_L + offs_l
        valid_l = (ls < L)[:, None]
        mask = valid_l & mask_d[None, :]
        x = tl.load(
            X + pid_b * sxb + ls[:, None] * sxl + offs_d[None, :] * sxd, mask=mask, other=0.0
        ).to(tl.float32)

        a11 = tl.where(valid_l, m11[None, :], 1.0)
        a12 = tl.where(valid_l, m12[None, :], 0.0)
        a21 = tl.where(valid_l, m21[None, :], 0.0)
        a22 = tl.where(valid_l, m22[None, :], 1.0)
        v1 = f1[None, :] * x + tl.where(first, (m11 * h1 + m12 * h2)[None, :], 0.0)
        v2 = f2[None, :] * x + tl.where(first, (m21 * h1 + m22 * h2)[None, :], 0.0)
        a11 = tl.where(first, 0.0, a11)
        a12 = tl.where(first, 0.0, a12)
        a21 = tl.where(first, 0.0, a21)
        a22 = tl.where(first, 0.0, a22)

        _, _, _, _, s1, s2 = tl.associative_scan(
            (a11, a12, a21, a22, v1, v2), 0, _combine
        )
        tl.store(
            Y + pid_b * syb + ls[:, None] * syl + offs_d[None, :] * syd,
            s2.to(Y.dtype.element_ty),
            mask=mask,
        )
        h1 = tl.sum(tl.where(last, s1, 0.0), axis=0)
        h2 = tl.sum(tl.where(last, s2, 0.0), axis=0)

    tl.store(HT + (pid_b * D + offs_d) * 2 + 0, h1, mask=mask_d)
    tl.store(HT + (pid_b * D + offs_d) * 2 + 1, h2, mask=mask_d)


@triton.jit
def _linoss_scan_bwd_kernel(
    X, DY, MF, HS, DHT, DX, DMF, DH0,
    L, D, NC,
    sxb, sxd, sxl,
    sgb, sgd, sgl,
    sdb, sdd, sdl,
    HAS_DHT: tl.constexpr,
    BLOCK_L: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    offs_l = tl.arange(0, BLOCK_L)

    m11 = tl.load(MF + offs_d * 6 + 0, mask=mask_d, other=1.0)
    m12 = tl.load(MF + offs_d * 6 + 1, mask=mask_d, other=0.0)
    m21 = tl.load(MF + offs_d * 6 + 2, mask=mask_d, other=0.0)
    m22 = tl.load(MF + offs_d * 6 + 3, mask=mask_d, other=1.0)
    f1 = tl.load(MF + offs_d * 6 + 4, mask=mask_d, other=0.0)
    f2 = tl.load(MF + offs_d * 6 + 5, mask=mask_d, other=0.0)

    # Adjoint carried in from the chunk after the current one.
    l1 = tl.zeros([BLOCK_D], dtype=tl.float32)
    l2 = tl.zeros([BLOCK_D], dtype=tl.float32)
    acc11 = tl.zeros([BLOCK_D], dtype=tl.float32)
    acc12 = tl.zeros([BLOCK_D], dtype=tl.float32)
    acc21 = tl.zeros([BLOCK_D], dtype=tl.float32)
    acc22 = tl.zeros([BLOCK_D], dtype=tl.float32)
    accf1 = tl.zeros([BLOCK_D], dtype=tl.float32)
    accf2 = tl.zeros([BLOCK_D], dtype=tl.float32)

    first = (offs_l == 0)[:, None]
    last = (offs_l == BLOCK_L - 1)[:, None]
    for cc in range(0, NC):
        c = NC - 1 - cc
        ls = c * BLOCK_L + offs_l
        valid_l = (ls < L)[:, None]
        mask = valid_l & mask_d[None, :]

        # Adjoint: lambda_t = M^T lambda_{t+1} + e_2 g_t, scanned right to left. Rows past L are the identity
        # with zero input, so they pass the (zero) carry through unchanged.
        g = tl.load(
            DY + pid_b * sgb + ls[:, None] * sgl + offs_d[None, :] * sgd, mask=mask, other=0.0
        ).to(tl.float32)
        t11 = tl.where(valid_l, m11[None, :], 1.0)
        t12 = tl.where(valid_l, m21[None, :], 0.0)
        t21 = tl.where(valid_l, m12[None, :], 0.0)
        t22 = tl.where(valid_l, m22[None, :], 1.0)
        w1 = tl.where(last, (m11 * l1 + m21 * l2)[None, :], 0.0)
        w2 = g + tl.where(last, (m12 * l1 + m22 * l2)[None, :], 0.0)
        if HAS_DHT:
            # Gradient of the final state enters the adjoint at t = L - 1.
            at_end = (ls == L - 1)[:, None]
            dht1 = tl.load(DHT + (pid_b * D + offs_d) * 2 + 0, mask=mask_d, other=0.0)
            dht2 = tl.load(DHT + (pid_b * D + offs_d) * 2 + 1, mask=mask_d, other=0.0)
            w1 += tl.where(at_end, dht1[None, :], 0.0)
            w2 += tl.where(at_end, dht2[None, :], 0.0)
        t11 = tl.where(last, 0.0, t11)
        t12 = tl.where(last, 0.0, t12)
        t21 = tl.where(last, 0.0, t21)
        t22 = tl.where(last, 0.0, t22)
        _, _, _, _, lam1, lam2 = tl.associative_scan(
            (t11, t12, t21, t22, w1, w2), 0, _combine, reverse=True
        )

        tl.store(
            DX + pid_b * sdb + ls[:, None] * sdl + offs_d[None, :] * sdd,
            (f1[None, :] * lam1 + f2[None, :] * lam2).to(DX.dtype.element_ty),
            mask=mask,
        )
        x = tl.load(
            X + pid_b * sxb + ls[:, None] * sxl + offs_d[None, :] * sxd, mask=mask, other=0.0
        ).to(tl.float32)
        accf1 += tl.sum(lam1 * x, axis=0)
        accf2 += tl.sum(lam2 * x, axis=0)

        # Previous states h_{t-1}: forward scan seeded with the saved chunk-start state at row 0 and the
        # inputs x_{t-1} at rows >= 1.
        hs_ptr = HS + ((pid_b * NC + c) * D + offs_d) * 2
        hc1 = tl.load(hs_ptr + 0, mask=mask_d, other=0.0)
        hc2 = tl.load(hs_ptr + 1, mask=mask_d, other=0.0)
        prev_mask = valid_l & (offs_l >= 1)[:, None] & mask_d[None, :]
        xp = tl.load(
            X + pid_b * sxb + (ls - 1)[:, None] * sxl + offs_d[None, :] * sxd,
            mask=prev_mask,
            other=0.0,
        ).to(tl.float32)
        u11 = tl.where(first, 0.0, m11[None, :])
        u12 = tl.where(first, 0.0, m12[None, :])
        u21 = tl.where(first, 0.0, m21[None, :])
        u22 = tl.where(first, 0.0, m22[None, :])
        v1 = f1[None, :] * xp + tl.where(first, hc1[None, :], 0.0)
        v2 = f2[None, :] * xp + tl.where(first, hc2[None, :], 0.0)
        _, _, _, _, s1, s2 = tl.associative_scan(
            (u11, u12, u21, u22, v1, v2), 0, _combine
        )
        # lambda is zero on rows past L, so those rows do not contribute.
        acc11 += tl.sum(lam1 * s1, axis=0)
        acc12 += tl.sum(lam1 * s2, axis=0)
        acc21 += tl.sum(lam2 * s1, axis=0)
        acc22 += tl.sum(lam2 * s2, axis=0)

        l1 = tl.sum(tl.where(first, lam1, 0.0), axis=0)
        l2 = tl.sum(tl.where(first, lam2, 0.0), axis=0)

    # h_0 = M h_init + ..., so d h_init = M^T lambda_0.
    tl.store(DH0 + (pid_b * D + offs_d) * 2 + 0, m11 * l1 + m21 * l2, mask=mask_d)
    tl.store(DH0 + (pid_b * D + offs_d) * 2 + 1, m12 * l1 + m22 * l2, mask=mask_d)

    # DMF: (B, D, 6) per-batch partial gradients, summed over batch on the host.
    out = DMF + (pid_b * D + offs_d) * 6
    tl.store(out + 0, acc11, mask=mask_d)
    tl.store(out + 1, acc12, mask=mask_d)
    tl.store(out + 2, acc21, mask=mask_d)
    tl.store(out + 3, acc22, mask=mask_d)
    tl.store(out + 4, accf1, mask=mask_d)
    tl.store(out + 5, accf2, mask=mask_d)
