"""Masked message-passing EPV network, implemented using plain PyTorch."""
import torch
from torch import nn

from .data import FEATURES


class GraphEPV(nn.Module):
    def __init__(self, hidden=48, layers=2, edges_enabled=True):
        super().__init__()
        self.config = {"hidden": hidden, "layers": layers, "edges_enabled": edges_enabled}
        self.edges_enabled = edges_enabled
        self.encoder = nn.Sequential(nn.Linear(len(FEATURES), hidden), nn.SiLU())
        self.messages = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * hidden + 4, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
            for _ in range(layers)])
        self.updates = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.LayerNorm(hidden))
            for _ in range(layers)])
        self.head = nn.Sequential(nn.Linear(3 * hidden + 2, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, x, mask):
        if x.ndim != 3 or x.shape[-1] != len(FEATURES) or mask.shape != x.shape[:2]:
            raise ValueError("Expected x[B,N,F] and mask[B,N]")
        mask = mask.bool()
        x = torch.where(mask[..., None], x, torch.zeros_like(x))
        h = self.encoder(x) * mask[..., None]
        n = x.shape[1]
        relative = x[:, None, :, :2] - x[:, :, None, :2]
        distance = relative.square().sum(-1, keepdim=True).sqrt()
        same_team = (x[:, :, None, 4] * x[:, None, :, 4] + x[:, :, None, 5] * x[:, None, :, 5])[..., None]
        edges = mask[:, :, None] & mask[:, None, :] & ~torch.eye(n, dtype=torch.bool, device=x.device)[None]
        for message, update in zip(self.messages, self.updates):
            if self.edges_enabled:
                inputs = torch.cat((h[:, :, None].expand(-1, -1, n, -1),
                                    h[:, None, :].expand(-1, n, -1, -1), relative, distance, same_team), -1)
                messages = message(inputs) * edges[..., None]
                aggregate = messages.sum(2) / edges.sum(2).clamp_min(1)[..., None]
            else:
                aggregate = torch.zeros_like(h)
            h = (h + update(torch.cat((h, aggregate), -1))) * mask[..., None]
        pools, counts = [], []
        for feature in (4, 5, 6):
            weights = x[..., feature] * mask
            pools.append((h * weights[..., None]).sum(1) / weights.sum(1).clamp_min(1)[:, None])
            counts.append(weights.sum(1, keepdim=True) / 11)
        return self.head(torch.cat(pools + counts[:2], -1)).squeeze(-1).tanh()

    def symmetric(self, x, mask):
        mirror = x.clone()
        mirror[..., 1] *= -1
        mirror[..., 3] *= -1
        return .5 * (self(x, mask) + self(mirror, mask))


class BallEPV(nn.Module):
    """Ball-only reference using exactly the same target, split and snapshots."""
    def __init__(self, hidden=32):
        super().__init__()
        self.config = {"hidden": hidden}
        self.head = nn.Sequential(nn.Linear(5, hidden), nn.SiLU(),
                                  nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, x, mask):
        safe = torch.where(mask[..., None], x, torch.zeros_like(x))
        ball = safe[..., 6:7] * mask[..., None]
        features = (safe[..., [0, 1, 2, 3, 9]] * ball).sum(1)
        return self.head(features).squeeze(-1).tanh()

    def symmetric(self, x, mask):
        mirror = x.clone()
        mirror[..., 1] *= -1
        mirror[..., 3] *= -1
        return .5 * (self(x, mask) + self(mirror, mask))
