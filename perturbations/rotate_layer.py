import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision
import torch


def abs2relu(x: torch.Tensor) -> torch.Tensor:
    return 2 * torch.relu(x) - x

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

    @staticmethod
    def _tent_weights(offset: torch.Tensor) -> torch.Tensor:
        z = 1.0 - abs2relu(offset)
        return 0.5 * (z + abs2relu(z))

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

    def forward(self, theta) -> torch.Tensor:
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

if __name__ == "__main__":
    torch.manual_seed(37)
    theta_degrees = torch.tensor([0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0])
    
    # dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = torchvision.datasets.MNIST(root='data', download=True, transform=transform, train=False)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    for img, _ in dataloader:
        img_tensor = img.squeeze(0)
        break
    
    layer = RotationPerturbationLayer(img_tensor)
    theta_radians = torch.deg2rad(theta_degrees)
    rotated = layer(theta_radians)
    print(rotated.shape)
    print(rotated.sum().item())
    
    
    images = [('Original', img_tensor)]
    for i in range(len(theta_degrees)):
        images.append((f"Theta: {theta_degrees[i]}", rotated[i]))

    n_rows = 2
    n_cols = len(images) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 5 * n_rows / 2))
    # Flatten axes if it's a 2D array to make indexing easier,  or index it as axes[row, col]
    for i, (title, img) in enumerate(images):
        row = i // n_cols
        col = i % n_cols
        
        # If img is a torch tensor (C, H, W), we permute to (H, W, C) for matplotlib
        axes[row, col].imshow(img.permute(1, 2, 0).cpu().numpy())
        axes[row, col].set_title(title)
        axes[row, col].axis('off') # Optional: hides the x/y ticks

    plt.tight_layout()
    plt.savefig('data/rotate_layer.png', dpi=300, bbox_inches='tight')


