"""Two nonnegative conditional xG expectations, relative to possession internally."""
import torch
from torch import nn

from .model import GraphEPV, BallEPV
from .data import FEATURES


class TeamGraphEPV(GraphEPV):
    def __init__(self, hidden=48, layers=2, edges_enabled=True):
        super().__init__(hidden, layers, edges_enabled)
        # Two expectations plus unused remaining mass enforce their sum <= 1.
        # The third coordinate is NOT a predicted probability of no shot.
        self.head[-1] = nn.Linear(hidden, 3)

    def forward(self, x, mask):
        if x.ndim != 3 or x.shape[-1] != len(FEATURES) or mask.shape != x.shape[:2]:
            raise ValueError('Expected x[B,N,F] and mask[B,N]')
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
        logits = self.head(torch.cat(pools + counts[:2], -1))
        return logits.softmax(-1)[..., :2]


class TeamBallEPV(BallEPV):
    def __init__(self, hidden=32):
        super().__init__(hidden)
        self.head[-1] = nn.Linear(hidden, 3)

    def forward(self, x, mask):
        safe = torch.where(mask[..., None], x, torch.zeros_like(x))
        ball = safe[..., 6:7] * mask[..., None]
        features = (safe[..., [0, 1, 2, 3, 9]] * ball).sum(1)
        return self.head(features).softmax(-1)[..., :2]
