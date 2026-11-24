import torch
from typing import Tuple


class AffineCameraOptModule(torch.nn.Module):
    def __init__(self, n: int = None, fixed_indices: list[int] = [], lr: float = 1e-4):
        super().__init__()
        if n is None:
            raise ValueError("Parameter 'n' must be provided and cannot be None.")

        self.embeds = torch.nn.Embedding(n, 2 * 4)
        self.fixed_indices = fixed_indices
        self.optimizer = torch.optim.Adam(self.embeds.parameters(), lr=lr, eps=1e-15, betas=(0.9, 0.999))

        self.identity_A = torch.zeros(2, 3, dtype=torch.float32, requires_grad=False)
        self.identity_b = torch.zeros(2, 1, dtype=torch.float32, requires_grad=False)

    def zero_init(self):
        torch.nn.init.zeros_(self.embeds.weight)

    def forward(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if idx in self.fixed_indices:
            return self.identity_A, self.identity_b

        idx = torch.tensor([idx], dtype=torch.int, device=self.embeds.weight.device)
        deltas = self.embeds(idx)
        deltaA = deltas[..., :6].reshape(2, 3)
        deltab = deltas[..., 6:].reshape(2, 1)
        return deltaA, deltab

    def optimizer_step(self):
        self.optimizer.step()
        self.optimizer.zero_grad()
