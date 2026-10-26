"""
Building blocks shared by the reference model (`linoss.py`, pure-PyTorch ops) and the custom-kernel
model (`linoss_cuda.py`). Both models have identical parameter names, so a state_dict trained with one
loads into the other.

Parameter names follow the JAX code (`models/LinOSS.py`): linear_encoder, blocks.{i}.ssm.{A_diag, B, C, D,
steps}, blocks.{i}.glu, linear_layer.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_linoss.ops import check_discretization, discretize


class LinOSSLayerBase(nn.Module):
    """LinOSS SSM layer: u (batch, L, H) -> (batch, L, H).

    The complex input/output matrices are stored as real/imag pairs, like the JAX code, and applied as
    real matrices: Bu = [B_re; B_im] u gives 2P real channels that share the per-state recurrence, and
    Re(C y) = [C_re, -C_im] [y_re; y_im]. Subclasses implement `mix`, the recurrence over those channels.
    """

    def __init__(self, ssm_size, H, discretization, A_eps=0.0):
        super().__init__()
        check_discretization(discretization)
        self.A_eps = A_eps
        self.ssm_size = ssm_size
        self.H = H
        self.discretization = discretization
        self.A_diag = nn.Parameter(torch.rand(ssm_size))
        self.B = nn.Parameter((torch.rand(ssm_size, H, 2) * 2 - 1) / math.sqrt(H))
        self.C = nn.Parameter((torch.rand(H, ssm_size, 2) * 2 - 1) / math.sqrt(ssm_size))
        self.D = nn.Parameter(torch.randn(H))
        self.steps = nn.Parameter(torch.rand(ssm_size))

    def recurrence(self):
        """(M (P, 2, 2), f (P, 2)) of the discretized oscillators."""
        return discretize(self.A_diag, self.steps, self.discretization, self.A_eps)

    def in_proj_weight(self):
        """(2P, H) real input matrix [B_re; B_im]."""
        return torch.cat([self.B[..., 0], self.B[..., 1]], dim=0)

    def out_proj_weight(self):
        """(H, 2P) real output matrix [C_re, -C_im]."""
        return torch.cat([self.C[..., 0], -self.C[..., 1]], dim=1)

    def mix(self, x):
        """Apply the recurrence to x (batch, 2P, L) and return the positions (batch, 2P, L)."""
        raise NotImplementedError

    def forward(self, u):
        x = F.linear(u, self.in_proj_weight())  # (batch, L, 2P)
        y = self.mix(x.transpose(1, 2))  # (batch, 2P, L)
        out = F.linear(y.transpose(1, 2).to(u.dtype), self.out_proj_weight())
        return out + self.D * u


class GLU(nn.Module):
    """w1(x) * sigmoid(w2(x)) with both projections fused into one matmul."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, 2 * output_dim)

    def forward(self, x):
        return F.glu(self.proj(x), dim=-1)


class BatchNorm(nn.Module):
    """Equinox 0.11 `BatchNorm(channelwise_affine=False)`, which the JAX model uses. On (N, C) input.

    Unlike torch's BatchNorm, it normalizes with the *running* statistics in training too:

        running = batch                                     (first training call)
        running = (1 - momentum) * batch + momentum * running   (afterwards; momentum = 0.99)
        out = (x - running_mean) / sqrt(running_var + eps)

    Gradients flow only through the (1 - momentum) * batch part. On LinOSS's large activations this
    slowly-moving normalization is what keeps early training stable; torch's batch statistics make the
    logits blow up after the first Adam step and training stalls.
    """

    def __init__(self, num_features, momentum=0.99, eps=1e-5):
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.register_buffer("initialized", torch.zeros((), dtype=torch.bool))
        self._initialized = False  # Python mirror of the buffer: no device sync in forward.

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
        self._initialized = bool(self.initialized)

    def forward(self, x):
        if self.training:
            xf = x.float()
            batch_mean = xf.mean(0)
            batch_var = (xf - batch_mean).square().mean(0).clamp_min(0.0)
            if self._initialized:
                mean = (1 - self.momentum) * batch_mean + self.momentum * self.running_mean
                var = (1 - self.momentum) * batch_var + self.momentum * self.running_var
            else:
                mean, var = batch_mean, batch_var
            with torch.no_grad():
                self.running_mean.copy_(mean)
                self.running_var.copy_(var)
                self.initialized.fill_(True)
            self._initialized = True
        else:
            mean, var = self.running_mean, self.running_var
        return ((x - mean) * torch.rsqrt(var + self.eps)).to(x.dtype)


class BatchSharedDropout(nn.Module):
    """Dropout with one mask shared by every sample of the batch, on (batch, L, H).

    The JAX training code vmaps the model over the batch with the dropout key broadcast
    (`in_axes=(0, None, None)` in `calc_output`), so all samples of a batch get the same mask.
    """

    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        keep = 1.0 - self.p
        mask = torch.empty((1,) + x.shape[1:], device=x.device, dtype=x.dtype).bernoulli_(keep)
        return x * mask / keep


class LinOSSBlock(nn.Module):
    """BatchNorm -> LinOSS layer -> GELU -> dropout -> GLU -> dropout -> residual, on (batch, L, H)."""

    def __init__(
        self, layer_cls, ssm_size, H, discretization, drop_rate=0.05, dropout_mask="batch", A_eps=0.0
    ):
        super().__init__()
        self.norm = BatchNorm(H)
        self.ssm = layer_cls(ssm_size, H, discretization, A_eps=A_eps)
        self.glu = GLU(H, H)
        if dropout_mask == "batch":
            self.drop = BatchSharedDropout(drop_rate)
        elif dropout_mask == "sample":
            self.drop = nn.Dropout(drop_rate)
        else:
            raise ValueError(f"dropout_mask must be 'batch' or 'sample', got {dropout_mask!r}")

    def forward(self, x):
        skip = x
        batch, L, H = x.shape
        # Statistics over batch and time per channel, as in the JAX code. (batch * L, H) is a free view.
        x = self.norm(x.reshape(batch * L, H)).view(batch, L, H)
        x = self.ssm(x)
        # jax.nn.gelu defaults to the tanh approximation.
        x = self.drop(F.gelu(x, approximate="tanh"))
        x = self.drop(self.glu(x))
        return skip + x


class LinOSSNet(nn.Module):
    """Encoder -> LinOSS blocks -> head. Input (batch, L, N).

    Classification: mean over time, then a linear layer returning logits.
    Regression: every output_step-th step, then a linear layer and tanh, giving (batch, L // output_step, out).
    """

    layer_cls = None

    def __init__(
        self,
        num_blocks,
        N,
        ssm_size,
        H,
        output_dim,
        classification,
        output_step=1,
        discretization="IM",
        drop_rate=0.05,
        dropout_mask="batch",
        A_eps=0.0,
    ):
        super().__init__()
        self.config = dict(
            num_blocks=num_blocks,
            N=N,
            ssm_size=ssm_size,
            H=H,
            output_dim=output_dim,
            classification=classification,
            output_step=output_step,
            discretization=discretization,
            drop_rate=drop_rate,
            dropout_mask=dropout_mask,
            A_eps=A_eps,
        )
        self.classification = classification
        self.output_step = output_step
        self.linear_encoder = nn.Linear(N, H)
        self.blocks = nn.ModuleList(
            [
                LinOSSBlock(
                    self.layer_cls, ssm_size, H, discretization, drop_rate, dropout_mask, A_eps
                )
                for _ in range(num_blocks)
            ]
        )
        self.linear_layer = nn.Linear(H, output_dim)

    def forward(self, x):
        x = self.linear_encoder(x)
        for block in self.blocks:
            x = block(x)
        if self.classification:
            return self.linear_layer(x.mean(dim=1))
        x = x[:, self.output_step - 1 :: self.output_step]
        return torch.tanh(self.linear_layer(x))
