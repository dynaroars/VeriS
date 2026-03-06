import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision
import torch

class LightnessPerturbationLayer(nn.Module):
    def __init__(self, image: torch.Tensor, lightness_type='spotlight', max_delta=1.0, center=None):
        super().__init__()
        image = image.contiguous().clone()
        self.C, self.H, self.W = image.shape
        
        self.lightness_type = lightness_type
        self.max_delta = max_delta
        self.custom_center = center

        # Store the original image expanded to [1, C, H, W]
        self.register_buffer("image", image.unsqueeze(0))

        xs = torch.arange(self.W, dtype=torch.float32)
        ys = torch.arange(self.H, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        
        # Generate and register the base lightness delta map
        self.register_buffer("base_lightness_map", self._create_base_map(grid_x, grid_y))

    def _create_base_map(self, grid_x, grid_y) -> torch.Tensor:
        """Creates a fixed [1, C, H, W] lightness modification map to be scaled by w."""
        base_map = torch.zeros((1, 1, self.H, self.W))
        
        if self.custom_center is not None:
            center_x, center_y = self.custom_center
        else:
            center_x, center_y = (self.W - 1) / 2.0, (self.H - 1) / 2.0
            
        if self.lightness_type == 'spotlight':
            # Localized brightness change (like a flashlight)
            dx = grid_x - center_x
            dy = grid_y - center_y
            
            # Gaussian envelope to decay the brightness change near the edges
            sigma = min(self.H, self.W) / 4.0 
            gaussian = torch.exp(-(dx**2 + dy**2) / (2 * sigma**2))
            
            base_map[0, 0, :, :] = gaussian * self.max_delta
            
        elif self.lightness_type == 'gradient_x':
            # Linear gradient of brightness from left to right
            normalized_x = grid_x / (self.W - 1)
            base_map[0, 0, :, :] = normalized_x * self.max_delta
            
        else:
            raise ValueError(f"Unknown lightness type: {self.lightness_type}")
            
        # Expand to match channel dimension [1, C, H, W]
        return base_map.expand(1, self.C, self.H, self.W)

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        """
        w: [B, 1] tensor representing the intensity of the lightness perturbation.
        """
        w_expanded = w.unsqueeze(1).unsqueeze(2) # [B, 1, 1, 1]
        perturbed = self.image + (w_expanded * self.base_lightness_map) # [B, 1, H, W]
        return perturbed

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
        
    # Test the layer with the "spotlight" type targeting the center
    # This will simulate a localized glow/lightness increase

    
    # Define an interval of scalar inputs w
    w_values = torch.tensor([[0.0], [0.2], [0.4], [0.6], [0.8], [1.0]]) # [B, 1]
    max_delta_dict = {
        'spotlight': 1.5,
        'gradient_x': 1.5,
    }
    
    lightness_types = ['spotlight', 'gradient_x']
    fig = plt.figure(figsize=(12, 3 * len(lightness_types)))
    # Create a vertical grid of subfigures (one for each row)
    subfigs = fig.subfigures(len(lightness_types), 1)

    for idx, lightness_type in enumerate(lightness_types):
        # Add the centered subtitle for this specific row
        subfigs[idx].suptitle(lightness_type.replace('_', ' ').capitalize(), fontsize=14, fontweight='bold')
        
        # Create the axes inside this row's subfigure
        num_cols = len(w_values) + 1
        axes = subfigs[idx].subplots(1, num_cols)
        
        layer = LightnessPerturbationLayer(
            image=img_tensor, 
            lightness_type=lightness_type, 
            max_delta=max_delta_dict[lightness_type],
        )
        perturbed = layer(w_values)
        
        print(f"{perturbed.shape=}")
        for i, _ in enumerate(perturbed):
            print(f'\t+ w={w_values[i].item():.02f}, sum={_.sum().item():.02f}, min={torch.min(_).item():.02f}, max={torch.max(_).item():.02f}')
        
        images = [('Original', img_tensor)]
        for i in range(len(w_values)):
            images.append((f'w = {w_values[i].item():.01f}', perturbed[i]))
            
        for i, (title, img) in enumerate(images):
            axes[i].imshow(img.permute(1, 2, 0).cpu().numpy())
            axes[i].set_title(title)
            axes[i].axis('off')


    plt.tight_layout()
    plt.savefig('figures/lightness_layer.png', dpi=300, bbox_inches='tight')
    
    torch.onnx.export(
        layer,
        w_values,
        "data/lightness_layer.onnx",
        opset_version=12,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'},
        }
    )
