"""
Mamba-3 baseline with the same encoder and output heads as `LinOSSNet`, so a comparison changes only the
sequence mixer. Uses the official `mamba_ssm.Mamba3` block (SISO; its full-sequence path is Triton).

Each block is the standard pre-norm residual block of Mamba: x + Mamba3(RMSNorm(x)).
"""

import torch
import torch.nn as nn


class Mamba3Net(nn.Module):
    """Encoder -> Mamba-3 blocks -> final RMSNorm -> head. Input (batch, L, N).

    Classification: mean over time, then a linear layer returning logits.
    Regression: every output_step-th step, then a linear layer and tanh.

    Mamba-3 arguments keep their library defaults (d_state=128, expand=2, headdim=64). If expand * H is
    smaller than headdim (e.g. H=16), headdim is reduced to expand * H, i.e. a single head.
    """

    def __init__(
        self,
        num_blocks,
        N,
        H,
        output_dim,
        classification,
        output_step=1,
        d_state=128,
        expand=2,
        headdim=64,
    ):
        super().__init__()
        from mamba_ssm import Mamba3

        headdim = min(headdim, expand * H)
        self.config = dict(
            num_blocks=num_blocks,
            N=N,
            H=H,
            output_dim=output_dim,
            classification=classification,
            output_step=output_step,
            d_state=d_state,
            expand=expand,
            headdim=headdim,
        )
        self.classification = classification
        self.output_step = output_step
        self.linear_encoder = nn.Linear(N, H)
        self.norms = nn.ModuleList([nn.RMSNorm(H) for _ in range(num_blocks)])
        self.mixers = nn.ModuleList(
            [
                Mamba3(d_model=H, d_state=d_state, expand=expand, headdim=headdim, layer_idx=i, n_layer=num_blocks)
                for i in range(num_blocks)
            ]
        )
        self.norm_f = nn.RMSNorm(H)
        self.linear_layer = nn.Linear(H, output_dim)

    def forward(self, x):
        x = self.linear_encoder(x)
        for norm, mixer in zip(self.norms, self.mixers):
            x = x + mixer(norm(x))
        x = self.norm_f(x)
        if self.classification:
            return self.linear_layer(x.mean(dim=1))
        x = x[:, self.output_step - 1 :: self.output_step]
        return torch.tanh(self.linear_layer(x))
