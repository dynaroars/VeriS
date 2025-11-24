import torch
import torch.nn as nn
from PIL import Image
import numpy as np


def load_image(path: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    tensor = torch.tensor(np.array(img), dtype=torch.float32) / 255.0
    return tensor.permute(2, 0, 1)


def to_pil(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.clamp(0.0, 1.0)
    array = (tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(array)


class RotationPerturbationLayer(nn.Module):

    def __init__(self, image: torch.Tensor):
        super().__init__()
        image = image.contiguous().clone()
        self.C, self.H, self.W = image.shape
        self.num_pixels = self.H * self.W

        self.register_buffer("image", image)
        self.register_buffer("flat_image", image.view(1, self.C, -1))

        c_x = (self.W - 1) / 2.0
        c_y = (self.H - 1) / 2.0
        self.register_buffer("center", torch.tensor([c_x, c_y], dtype=torch.float32))

        xs = torch.arange(self.W, dtype=torch.float32)
        ys = torch.arange(self.H, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

        x_rel = (grid_x - c_x).reshape(1, -1)
        y_rel = (grid_y - c_y).reshape(1, -1)
        coords = torch.cat([x_rel, y_rel], dim=0)  # [2, N]
        self.register_buffer("relative_coords", coords)

        self.register_buffer("x_coords", xs)
        self.register_buffer("y_coords", ys)

    def _prepare_angles(self, theta, degrees: bool) -> torch.Tensor:
        theta = torch.as_tensor(theta, dtype=torch.float32, device=self.image.device)
        if theta.ndim == 0:
            theta = theta.unsqueeze(0)
        theta = theta.view(-1)
        if degrees:
            theta = torch.deg2rad(theta)
        return theta

    @staticmethod
    def _tent_weights(offset: torch.Tensor) -> torch.Tensor:
        z = 1.0 - offset.abs()
        return 0.5 * (z + z.abs())

    def _bilinear_sample(self, src_x: torch.Tensor, src_y: torch.Tensor) -> torch.Tensor:
        batch_size = src_x.shape[0]
        num_coords = src_x.shape[1]
        
        # Compute separable triangular weights
        dx = src_x.unsqueeze(-1) - self.x_coords  # [B, N, W]
        dy = src_y.unsqueeze(-1) - self.y_coords  # [B, N, H]

        weights_x = self._tent_weights(dx)  # [B, N, W]
        weights_y = self._tent_weights(dy)  # [B, N, H]

        weights_xy = weights_y.unsqueeze(-1) * weights_x.unsqueeze(-2)  # [B, N, H, W]
        weights_flat = weights_xy.reshape(batch_size, num_coords, -1)  # [B, N, H*W]

        base = self.flat_image.expand(batch_size, -1, -1)  # [B, C, H*W]
        base_transposed = base.permute(0, 2, 1)  # [B, H*W, C]

        sampled = torch.bmm(weights_flat, base_transposed)  # [B, N, C]
        sampled = sampled.permute(0, 2, 1)  # [B, C, N]
        return sampled

    def rotate(self, theta, degrees: bool = True) -> torch.Tensor:
        theta = self._prepare_angles(theta, degrees)
        batch_size = theta.shape[0]

        cos_theta = torch.cos(theta).view(batch_size, 1, 1)
        sin_theta = torch.sin(theta).view(batch_size, 1, 1)

        rotation_matrices = torch.cat(
            [
                torch.cat([cos_theta, sin_theta], dim=-1),
                torch.cat([-sin_theta, cos_theta], dim=-1),
            ],
            dim=-2,
        )  # [B, 2, 2]

        coords = self.relative_coords.unsqueeze(0)  # [1, 2, N]
        rotated = rotation_matrices @ coords  # [B, 2, N]

        src_x = rotated[:, 0, :] + self.center[0]
        src_y = rotated[:, 1, :] + self.center[1]

        samples = self._bilinear_sample(src_x, src_y)
        return samples.view(batch_size, self.C, self.H, self.W)

    def forward(self, theta, degrees: bool = True) -> torch.Tensor:
        return self.rotate(theta, degrees=degrees)

if __name__ == "__main__":
    theta_degrees = torch.tensor([0.0, 45])

    img_tensor = torch.randn(3, 32, 32)
    layer = RotationPerturbationLayer(img_tensor)
    rotated = layer(theta_degrees, degrees=True)
    print(rotated.shape)


