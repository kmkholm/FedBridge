"""
BridgeMamba-KAN — pure-PyTorch reference implementation (CPU-friendly).

Components:
  SelectiveSSM   simplified S6 block (diagonal A, input-dependent dt/B/C),
                 sequential scan — numerically faithful to Mamba's recurrence,
                 slower than the CUDA kernel but exact enough for CPU dev and
                 for the Colab run we swap in mamba-ssm's fused kernel.
  BiMamba        forward + time-reversed SelectiveSSM, gated, concatenated.
  KANLayer       Kolmogorov–Arnold layer with learnable B-spline edge
                 functions (efficient-KAN formulation: silu base + spline).
  BridgeMambaKAN embedding -> N x BiMamba -> masked mean+last pooling
                 (+ bridge-context vector) -> KAN head -> logit.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveSSM(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16):
        super().__init__()
        self.d_model, self.d_state = d_model, d_state
        self.in_proj = nn.Linear(d_model, 2 * d_model)          # x and gate z
        self.dt_proj = nn.Linear(d_model, d_model)
        self.B_proj = nn.Linear(d_model, d_state)
        self.C_proj = nn.Linear(d_model, d_state)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1).float())
                                  .repeat(d_model, 1))          # (D, N)
        self.D = nn.Parameter(torch.ones(d_model))
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:        # x: (B, L, D)
        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)
        dt = F.softplus(self.dt_proj(x_in))                     # (B, L, D)
        Bm = self.B_proj(x_in)                                  # (B, L, N)
        Cm = self.C_proj(x_in)                                  # (B, L, N)
        A = -torch.exp(self.A_log)                              # (D, N)

        B_, L, D = x_in.shape
        h = x_in.new_zeros(B_, D, self.d_state)
        ys = []
        for t in range(L):
            dA = torch.exp(dt[:, t].unsqueeze(-1) * A)          # (B, D, N)
            dBx = (dt[:, t] * x_in[:, t]).unsqueeze(-1) * Bm[:, t].unsqueeze(1)
            h = dA * h + dBx
            ys.append((h * Cm[:, t].unsqueeze(1)).sum(-1))      # (B, D)
        y = torch.stack(ys, dim=1) + self.D * x_in
        return self.out_proj(y * F.silu(z))


class BiMamba(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16):
        super().__init__()
        self.fwd = SelectiveSSM(d_model, d_state)
        self.bwd = SelectiveSSM(d_model, d_state)
        self.norm = nn.LayerNorm(d_model)
        self.mix = nn.Linear(2 * d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        y = torch.cat([self.fwd(h), self.bwd(h.flip(1)).flip(1)], dim=-1)
        return x + self.mix(y)


class KANLayer(nn.Module):
    """Edge-wise learnable functions: silu base + B-spline (grid G, order k)."""

    def __init__(self, d_in: int, d_out: int, grid: int = 5, k: int = 3,
                 x_range: float = 3.0):
        super().__init__()
        self.d_in, self.d_out, self.k = d_in, d_out, k
        n_knots = grid + 2 * k + 1
        knots = torch.linspace(-x_range - k * (2 * x_range / grid),
                               x_range + k * (2 * x_range / grid), n_knots)
        self.register_buffer("knots", knots)
        self.n_basis = grid + k
        self.base_w = nn.Parameter(torch.empty(d_out, d_in))
        self.spline_w = nn.Parameter(torch.empty(d_out, d_in, self.n_basis))
        nn.init.kaiming_uniform_(self.base_w, a=math.sqrt(5))
        nn.init.normal_(self.spline_w, std=0.1 / math.sqrt(d_in))

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:      # x: (B, d_in)
        t = self.knots
        x = x.unsqueeze(-1)                                     # (B, d_in, 1)
        b = ((x >= t[:-1]) & (x < t[1:])).float()               # order 0
        for p in range(1, self.k + 1):
            left = (x - t[: -p - 1]) / (t[p:-1] - t[: -p - 1]) * b[..., :-1]
            right = (t[p + 1:] - x) / (t[p + 1:] - t[1:-p]) * b[..., 1:]
            b = left + right
        return b[..., : self.n_basis]                           # (B, d_in, n_basis)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(F.silu(x), self.base_w)
        spl = torch.einsum("bin,oin->bo", self.b_splines(torch.tanh(x) * 3.0),
                           self.spline_w)
        return base + spl


class BridgeMambaKAN(nn.Module):
    def __init__(self, d_feat: int, d_ctx: int, d_model: int = 48,
                 n_layers: int = 2, d_state: int = 16, kan_hidden: int = 24):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(d_feat, d_model), nn.SiLU(),
                                   nn.Linear(d_model, d_model))
        self.blocks = nn.ModuleList(BiMamba(d_model, d_state) for _ in range(n_layers))
        self.norm = nn.LayerNorm(d_model)
        d_pool = 2 * d_model + d_ctx
        self.kan1 = KANLayer(d_pool, kan_hidden)
        self.kan2 = KANLayer(kan_hidden, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor,
                ctx: torch.Tensor) -> torch.Tensor:
        # x: (B, L, F)   mask: (B, L) bool, True = real step   ctx: (B, C)
        h = self.embed(x)
        for blk in self.blocks:
            h = blk(h)
        h = self.norm(h)
        m = mask.unsqueeze(-1).float()
        mean = (h * m).sum(1) / m.sum(1).clamp(min=1)
        last = h[:, -1]                                         # right-aligned
        z = torch.cat([mean, last, ctx], dim=-1)
        return self.kan2(self.kan1(z)).squeeze(-1)
