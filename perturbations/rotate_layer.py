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

        # Use a Linear layer instead of bmm/expand
        self.image_layer = nn.Linear(self.num_pixels, self.C, bias=False)
        with torch.no_grad():
            self.image_layer.weight.copy_(self.flat_image.squeeze(0))
            self.image_layer.weight.requires_grad_(False)

        c_x = (self.W - 1) / 2.0
        c_y = (self.H - 1) / 2.0
        self.register_buffer("center", torch.tensor([c_x, c_y], dtype=torch.float32))

        xs = torch.arange(self.W, dtype=torch.float32)
        ys = torch.arange(self.H, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

        x_rel = (grid_x - c_x).reshape(1, -1)
        y_rel = (grid_y - c_y).reshape(1, -1)
        self.register_buffer("x_rel", x_rel)
        self.register_buffer("y_rel", y_rel)

        self.register_buffer("x_coords", xs)
        self.register_buffer("y_coords", ys)

    @staticmethod
    def _tent_weights(offset: torch.Tensor) -> torch.Tensor:
        z = 1.0 - abs2relu(offset)
        return 0.5 * (z + abs2relu(z))

    def _bilinear_sample(self, src_x: torch.Tensor, src_y: torch.Tensor) -> torch.Tensor:
        num_coords = src_x.shape[1]
        
        # Compute separable triangular weights
        dx = src_x.unsqueeze(-1) - self.x_coords  # [B, N, W]
        dy = src_y.unsqueeze(-1) - self.y_coords  # [B, N, H]

        weights_x = self._tent_weights(dx)  # [B, N, W]
        weights_y = self._tent_weights(dy)  # [B, N, H]

        weights_x = weights_x.view(-1, num_coords, 1, self.W)
        weights_y = weights_y.view(-1, num_coords, self.H, 1)

        weights_xy = weights_y * weights_x  # [B, N, H, W]
        weights_flat = weights_xy.view(-1, num_coords, self.num_pixels)  # [B, N, H*W]

        sampled = self.image_layer(weights_flat)  # [B, N, C]
        sampled = sampled.permute(0, 2, 1)  # [B, C, N]
        return sampled

    def forward(self, theta) -> torch.Tensor:
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        # x' = x*cos + y*sin
        src_x = (cos_theta * self.x_rel) + (sin_theta * self.y_rel) + self.center[0]
        # y' = -x*sin + y*cos
        src_y = (-sin_theta * self.x_rel) + (cos_theta * self.y_rel) + self.center[1]

        samples = self._bilinear_sample(src_x, src_y)
        return samples.view(theta.shape[0], self.C, self.H, self.W)

if __name__ == "__main__":
    torch.manual_seed(37)
    theta_degrees = torch.tensor([0.0, 30.0, 45.0, 60.0, 90.0, 120.0]).view(-1, 1)
    
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
    perturbed = layer(theta_radians)
    
    print("Output Shape:", perturbed.shape, [_.sum().item() for _ in perturbed])
    
    images = [('Original', img_tensor)]
    for i in range(len(theta_degrees)):
        images.append((f'theta = {theta_degrees[i].item()}', perturbed[i]))

    fig, axes = plt.subplots(1, len(images), figsize=(12, 3))
    for i, (title, img) in enumerate(images):
        axes[i].imshow(img.permute(1, 2, 0).cpu().numpy())
        axes[i].set_title(title)
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig('figures/rotate_layer.png', dpi=300, bbox_inches='tight')
    

    torch.onnx.export(
        layer,
        theta_radians,
        "data/rotate_layer.onnx",
        opset_version=12,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'},
        }
    )


