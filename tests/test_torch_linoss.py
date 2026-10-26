import pytest
import torch
import torch.nn.functional as F

from torch_linoss.cuda_ops import HAS_TRITON, linoss_scan
from torch_linoss.models import LinOSS, LinOSSCuda, LinOSSLayer
from torch_linoss.ops import (
    causal_conv1d_fft,
    discretize,
    fft_causal_conv_lag,
    linoss_kernel,
    linoss_layer_reference,
    linoss_recurrence_reference,
)

DISCS = ["IM", "IMEX"]
cuda = pytest.mark.skipif(
    not (torch.cuda.is_available() and HAS_TRITON), reason="needs CUDA and Triton"
)


def rel_err(a, b):
    a, b = a.double().cpu(), b.double().cpu()
    return ((a - b).abs().max() / (b.abs().max() + 1e-12)).item()


def random_recurrence(P, disc, dtype=torch.float64, zero_A=True):
    A = torch.rand(P, dtype=dtype)
    if zero_A:
        A[0] = -1.0  # relu -> 0: M is a Jordan block
    steps = torch.randn(P, dtype=dtype)
    return discretize(A, steps, disc)


@pytest.mark.parametrize("width", [1, 3, 4, 64, 65, 300])
@pytest.mark.parametrize("activation", [None, "silu"])
def test_causal_conv1d_fft_matches_conv1d(width, activation):
    torch.manual_seed(0)
    batch, dim, L = 2, 5, 300
    x = torch.randn(batch, dim, L, dtype=torch.float64)
    w = torch.randn(dim, width, dtype=torch.float64)
    b = torch.randn(dim, dtype=torch.float64)
    ref = F.conv1d(x, w.unsqueeze(1), b, padding=width - 1, groups=dim)[..., :L]
    if activation:
        ref = F.silu(ref)
    out = causal_conv1d_fft(x, w, b, activation)
    assert rel_err(out, ref) < 1e-10


@pytest.mark.parametrize("disc", DISCS)
@pytest.mark.parametrize("L", [1, 2, 7, 64, 100])
def test_kernel_matches_matrix_powers(disc, L):
    torch.manual_seed(0)
    M, f = random_recurrence(6, disc)
    expected = torch.stack(
        [(torch.linalg.matrix_power(M, k) @ f.unsqueeze(-1))[:, 1, 0] for k in range(L)], -1
    )
    assert rel_err(linoss_kernel(M, f, L), expected) < 1e-12


@pytest.mark.parametrize("disc", DISCS)
def test_fft_conv_matches_recurrence(disc):
    torch.manual_seed(0)
    M, f = random_recurrence(6, disc)
    x = torch.randn(3, 6, 150, dtype=torch.float64)
    ref, _ = linoss_recurrence_reference(x, M, f)
    assert rel_err(fft_causal_conv_lag(x, linoss_kernel(M, f, 150)), ref) < 1e-10


@pytest.mark.parametrize("disc", DISCS)
@pytest.mark.parametrize("path", ["depthwise", "freq"])
def test_layer_matches_complex_reference(disc, path):
    torch.manual_seed(0)
    layer = LinOSSLayer(8, 5, disc, path=path).double()
    u = torch.randn(2, 40, 5, dtype=torch.float64)
    ref = linoss_layer_reference(
        u, layer.A_diag, layer.steps, layer.B, layer.C, layer.D, disc
    )
    assert rel_err(layer(u), ref) < 1e-10


@pytest.mark.parametrize("disc", DISCS)
def test_A_eps_floor(disc):
    # A = relu(A_diag) + A_eps: for positive A_diag it equals the original layer with A_diag + A_eps.
    torch.manual_seed(0)
    layer = LinOSSLayer(8, 5, disc, A_eps=0.1, path="depthwise").double()
    u = torch.randn(2, 40, 5, dtype=torch.float64)
    ref = linoss_layer_reference(
        u, layer.A_diag + 0.1, layer.steps, layer.B, layer.C, layer.D, disc
    )
    assert rel_err(layer(u), ref) < 1e-10
    M, _ = discretize(torch.full((3,), -1.0), torch.zeros(3), disc, A_eps=1e-4)
    M0, _ = discretize(torch.full((3,), -1.0), torch.zeros(3), disc)
    assert not torch.equal(M, M0)  # the floor applies even where relu clamps to 0


@pytest.mark.parametrize("disc", DISCS)
def test_layer_gradcheck(disc):
    torch.manual_seed(0)
    layer = LinOSSLayer(3, 2, disc, path="depthwise").double()
    u = torch.randn(1, 6, 2, dtype=torch.float64, requires_grad=True)
    names = ["A_diag", "steps", "B", "C", "D"]
    params = [getattr(layer, n) for n in names]

    def fn(u, *ps):
        return torch.func.functional_call(layer, dict(zip(names, ps)), (u,))

    inputs = (u,) + tuple(p.detach().clone().requires_grad_() for p in params)
    assert torch.autograd.gradcheck(fn, inputs)


@pytest.mark.parametrize("classification", [True, False])
def test_model_shapes(classification):
    torch.manual_seed(0)
    model = LinOSS(2, 3, 8, 6, 4, classification, output_step=5)
    out = model(torch.randn(2, 20, 3))
    assert out.shape == ((2, 4) if classification else (2, 4, 4))


def test_batchnorm_matches_equinox():
    # Equinox 0.11 normalizes with the running statistics in training, seeded by the first batch.
    from torch_linoss.models.common import BatchNorm

    torch.manual_seed(0)
    bn = BatchNorm(3).double()
    x1, x2 = torch.randn(50, 3, dtype=torch.float64) * 5 + 2, torch.randn(50, 3, dtype=torch.float64)
    m1, v1 = x1.mean(0), x1.var(0, unbiased=False)
    assert torch.allclose(bn(x1), (x1 - m1) / torch.sqrt(v1 + 1e-5))
    m2 = 0.01 * x2.mean(0) + 0.99 * m1
    v2 = 0.01 * x2.var(0, unbiased=False) + 0.99 * v1
    assert torch.allclose(bn(x2), (x2 - m2) / torch.sqrt(v2 + 1e-5))
    bn.eval()
    assert torch.allclose(bn(x1), (x1 - m2) / torch.sqrt(v2 + 1e-5))


def test_batch_shared_dropout():
    # The JAX code broadcasts the dropout key over the batch: one mask for every sample.
    from torch_linoss.models.common import BatchSharedDropout

    drop = BatchSharedDropout(0.5).train()
    out = drop(torch.ones(4, 30, 8))
    assert torch.equal(out, out[:1].expand_as(out))
    assert set(out.unique().tolist()) <= {0.0, 2.0}
    assert torch.equal(drop.eval()(out), out)


def test_unknown_discretization():
    with pytest.raises(ValueError):
        LinOSSLayer(4, 4, "RK4")


@cuda
@pytest.mark.parametrize("disc", DISCS)
@pytest.mark.parametrize("shape", [(1, 3, 1), (2, 6, 64), (3, 37, 200), (4, 128, 1000)])
@pytest.mark.parametrize("transposed", [False, True])
def test_scan_matches_reference(disc, shape, transposed):
    torch.manual_seed(0)
    batch, D, L = shape
    M, f = random_recurrence(D, disc)
    x = torch.randn(batch, D, L, dtype=torch.float64)
    h0 = torch.randn(batch, D, 2, dtype=torch.float64)
    g = torch.randn(batch, D, L, dtype=torch.float64)
    gT = torch.randn(batch, D, 2, dtype=torch.float64)

    leaves = [t.clone().requires_grad_() for t in (x, M, f, h0)]
    y, hT = linoss_recurrence_reference(leaves[0], leaves[1], leaves[2], leaves[3])
    ((y * g).sum() + (hT * gT).sum()).backward()

    xc = x.float().cuda()
    if transposed:  # (batch, L, D) memory layout viewed as (batch, D, L)
        xc = xc.transpose(1, 2).contiguous().transpose(1, 2)
    cl = [xc.requires_grad_()] + [t.float().cuda().requires_grad_() for t in (M, f, h0)]
    yc, hTc = linoss_scan(cl[0], cl[1], cl[2], h0=cl[3], return_final_state=True)
    ((yc * g.float().cuda()).sum() + (hTc * gT.float().cuda()).sum()).backward()

    tol = 2e-4 if disc == "IMEX" else 5e-5  # IMEX does not decay, errors accumulate over L
    assert rel_err(yc, y) < tol
    assert rel_err(hTc, hT) < tol
    for c, r in zip(cl, leaves):
        assert rel_err(c.grad, r.grad) < tol


@cuda
def test_scan_bf16_input():
    torch.manual_seed(0)
    M, f = random_recurrence(16, "IM", dtype=torch.float32)
    x = torch.randn(2, 16, 300, device="cuda")
    ref = linoss_scan(x, M.cuda(), f.cuda())
    out = linoss_scan(x.bfloat16(), M.cuda(), f.cuda())
    assert out.dtype == torch.bfloat16
    assert rel_err(out, ref) < 2e-2


@cuda
@pytest.mark.parametrize("disc", DISCS)
@pytest.mark.parametrize("classification", [True, False])
def test_cuda_model_matches_reference_model(disc, classification):
    torch.manual_seed(0)
    ref = LinOSS(2, 3, 16, 8, 4, classification, output_step=7, discretization=disc).cuda()
    model = LinOSSCuda.from_reference(ref)
    x = torch.randn(4, 300, 3, device="cuda")

    ref.train(), model.train()
    torch.manual_seed(1)
    out_ref = ref(x)
    out_ref.square().sum().backward()
    torch.manual_seed(1)  # same dropout masks
    out = model(x)
    out.square().sum().backward()
    assert rel_err(out, out_ref) < 1e-4
    grads_ref = dict(ref.named_parameters())
    for name, p in model.named_parameters():
        assert rel_err(p.grad, grads_ref[name].grad) < 1e-3, name

    ref.eval()
    model.freeze()
    with torch.no_grad():
        assert rel_err(model(x), ref(x)) < 1e-4


@cuda
def test_scan_torch_compile():
    torch.manual_seed(0)
    model = LinOSSCuda(2, 3, 16, 8, 4, True).cuda()
    x = torch.randn(4, 100, 3, device="cuda")
    compiled = torch.compile(model, fullgraph=True)
    model.eval()
    with torch.no_grad():
        assert rel_err(compiled(x), model(x)) < 1e-5


@cuda
def test_export_roundtrip():
    torch.manual_seed(0)
    model = LinOSSCuda(2, 3, 16, 8, 4, True).cuda().freeze()
    x = torch.randn(4, 100, 3, device="cuda")
    batch = torch.export.Dim("batch", min=1, max=1024)
    ep = torch.export.export(model, (x,), dynamic_shapes=({0: batch},))
    with torch.no_grad():
        assert rel_err(ep.module()(x[:3]), model(x[:3])) < 1e-5
