import torch


class SunlightModule(torch.nn.Module):
    def __init__(self, n: int = None, color_dim: int = 3, lr: float = 2.5e-3, init_ambient: float = 0.2):
        super().__init__()
        if n is None:
            raise ValueError("Parameter 'n' must be provided and cannot be None.")
        self.n = n
        self.lr = lr
        self.color_dim = color_dim

        # Per camera shadow coefficients (N, color_dim)
        ambient = torch.logit(torch.full((n, color_dim), init_ambient, dtype=torch.float32))
        self.ambient = torch.nn.Parameter(ambient)

        self.optimizer = torch.optim.Adam(
            [self.ambient],
            lr=lr,
            eps=1e-15,
            betas=(0.9, 0.999),
        )

    def ambient_mean(self):
        return torch.sigmoid(self.ambient).mean().item()

    def forward(self, cam_idx: int, sunlit: torch.Tensor):
        """Calculates the illumination based on the sunlit areas and ambient light"""
        cam_ambient = torch.sigmoid(self.ambient[cam_idx])
        l_A_S = sunlit + (1.0 - sunlit) * cam_ambient
        return l_A_S

    def optimizer_step(self):
        self.optimizer.step()
        self.optimizer.zero_grad()
