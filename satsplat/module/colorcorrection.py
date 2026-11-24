import torch


class ColorCorrectionModule(torch.nn.Module):
    def __init__(self, n: int = None, color_dim: int = 3, lr: float = 2.5e-3):
        super().__init__()
        if n is None:
            raise ValueError("Parameter 'n' must be provided and cannot be None.")
        self.n = n
        self.lr = lr
        self.color_dim = color_dim

        # Per camera color affine coefficients (N,3,4)
        color_affine = torch.zeros((n, color_dim, color_dim + 1), dtype=torch.float32)
        for i in range(color_dim):
            # Initialize the affine transformation matrix for each camera
            color_affine[:, i, i] = 1.0  # Set diagonal to 1 for identity transformation

        self.color_affine = torch.nn.Parameter(color_affine, requires_grad=True)

        self.optimizer = torch.optim.Adam(
            [self.color_affine],
            lr=lr,
            eps=1e-15,
            betas=(0.9, 0.999),
        )

    def forward(self, cam_idx: int, in_colors: torch.Tensor):
        """Applies color correction to the albedo based on the shadow map and camera index."""
        M = self.color_affine[cam_idx]
        assert in_colors.shape[-1] == self.color_dim, f"RGB input should have {self.color_dim} channels"
        # Apply color affine transformation
        return torch.einsum("ij,...j->...i", M[..., : self.color_dim], in_colors) + M[..., self.color_dim]

    def optimizer_step(self):
        self.optimizer.step()
        self.optimizer.zero_grad()
