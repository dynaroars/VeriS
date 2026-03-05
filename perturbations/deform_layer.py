import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision
import torch

def abs2relu(x: torch.Tensor) -> torch.Tensor:
    return 2 * torch.relu(x) - x

class DeformationPerturbationLayer(nn.Module):
    def __init__(self, image: torch.Tensor, displacement_type='localized_bulge', max_displacement=1.0, center=None):
        super().__init__()
        image = image.contiguous().clone()
        self.C, self.H, self.W = image.shape
        self.num_pixels = self.H * self.W
        
        self.displacement_type = displacement_type
        self.max_displacement = max_displacement
        self.custom_center = center

        self.register_buffer("image", image)
        self.register_buffer("flat_image", image.view(1, self.C, -1))

        # Use a Linear layer instead of bmm/expand
        self.image_layer = nn.Linear(self.num_pixels, self.C, bias=False)
        with torch.no_grad():
            self.image_layer.weight.copy_(self.flat_image.squeeze(0))
            self.image_layer.weight.requires_grad_(False)

        xs = torch.arange(self.W, dtype=torch.float32)
        ys = torch.arange(self.H, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

        self.register_buffer("x_coords", xs)
        self.register_buffer("y_coords", ys)
        
        # Base coordinates for the target grid flattened to [1, N]
        self.register_buffer("base_x", grid_x.reshape(1, -1))
        self.register_buffer("base_y", grid_y.reshape(1, -1))
        
        # Generate and register the base displacement pattern
        self.register_buffer("base_displacement", self._create_base_displacement(grid_x, grid_y))

    def _create_base_displacement(self, grid_x, grid_y) -> torch.Tensor:
        """Creates a fixed [1, 2, N] displacement field pattern to be scaled by w."""
        base_disp = torch.zeros((1, 2, self.num_pixels))
        
        if self.custom_center is not None:
            center_x, center_y = self.custom_center
        else:
            center_x, center_y = (self.W - 1) / 2.0, (self.H - 1) / 2.0
        
        if self.displacement_type == 'translation':
            base_disp[0, 0, :] = self.max_displacement
        elif self.displacement_type == 'sine_ripple':
            base_disp[0, 0, :] = (self.max_displacement * torch.sin(grid_y / 1.5)).reshape(-1)
        elif self.displacement_type == 'expansion':
            # Negative sign flips the sampling so the object expands instead of shrinking
            base_disp[0, 0, :] = (-(grid_x - center_x) * self.max_displacement).reshape(-1)
            base_disp[0, 1, :] = (-(grid_y - center_y) * self.max_displacement).reshape(-1)
        elif self.displacement_type == 'localized_bulge':
            # Negative sign for expanding the object
            dx = -(grid_x - center_x)
            dy = -(grid_y - center_y)
            
            # Gaussian envelope to decay the displacement near the edges
            sigma = min(self.H, self.W) / 4.0 
            gaussian = torch.exp(-(dx**2 + dy**2) / (2 * sigma**2))
            
            base_disp[0, 0, :] = (dx * gaussian * self.max_displacement).reshape(-1)
            base_disp[0, 1, :] = (dy * gaussian * self.max_displacement).reshape(-1)
        else:
            raise ValueError(f"Unknown displacement type: {self.displacement_type}")
            
        return base_disp

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

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        """
        w: [B, 1] tensor representing the intensity of the deformation.
        """
        # Scale the base displacement field by the scalar w
        displacement_field = w.unsqueeze(1) * self.base_displacement # [B, 2, N]
        
        dx = displacement_field[:, 0, :] # [B, N]
        dy = displacement_field[:, 1, :] # [B, N]

        # Calculate source coordinates for bilinear sampling
        src_x = self.base_x + dx
        src_y = self.base_y + dy

        samples = self._bilinear_sample(src_x, src_y)
        return samples.view(w.shape[0], self.C, self.H, self.W)
    
if __name__ == "__main__":
    torch.manual_seed(37)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = torchvision.datasets.MNIST(root='data', download=True, transform=transform, train=False)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    for img, _ in dataloader:
        img_tensor = img.squeeze(0)
        break
        
    # Test the layer with the new "localized_bulge" type
    layer = DeformationPerturbationLayer(img_tensor, displacement_type='localized_bulge', max_displacement=1.0, center=(16.0, 7.0))
    
    # Define an interval of scalar inputs w
    w_values = torch.tensor([[0.0], [0.2], [0.4], [0.6], [0.8], [1.0]]) # [B, 1]
    
    deformed = layer(w_values)
    print("Output Shape:", deformed.shape)
    
    images = [
        ('Original', img_tensor),
        ('w = 0.0', deformed[0]), 
        ('w = 0.2', deformed[1]), 
        ('w = 0.4', deformed[2]), 
        ('w = 0.6', deformed[3]), 
        ('w = 0.8', deformed[4]), 
        ('w = 1.0', deformed[5]),
    ]

    fig, axes = plt.subplots(1, len(images), figsize=(12, 3))
    for i, (title, img) in enumerate(images):
        axes[i].imshow(img.permute(1, 2, 0).cpu().numpy())
        axes[i].set_title(title)
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig('figures/deform_layer.png', dpi=300, bbox_inches='tight')
    

    torch.onnx.export(
        layer,
        w_values,
        "data/deform_layer.onnx",
        opset_version=12,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'},
        }
    )


