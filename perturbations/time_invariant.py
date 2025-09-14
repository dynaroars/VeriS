import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def get_kernel(perturbation_type, kernel_size):
        """Create different 1D kernels based on perturbation type."""
        kernel = torch.zeros(kernel_size, dtype=torch.float32)
        
        if perturbation_type == 'shift':
            # [0,..,0,1] Shift/Delay
            kernel[-1] = 1.0
            
        elif perturbation_type == 'lowpass':
            # [1/N,...,1/N] Low pass
            kernel.fill_(1.0 / kernel_size)
            
        elif perturbation_type == 'echo':
            # [1, 0,...,0,a] Echo/Comb Filter
            kernel[0] = 1.0
            kernel[-1] = 0.5  # echo strength
            
        elif perturbation_type == 'highpass':
            # [0, -1, 2, -1, 0] Sharpening
            center = kernel_size // 2
            kernel[center] = 2.0
            if center > 0:
                kernel[center - 1] = -1.0
            if center < kernel_size - 1:
                kernel[center + 1] = -1.0
                
        elif perturbation_type == 'gaussian':
            # Gaussian smoothing
            center = kernel_size // 2
            sigma = kernel_size / 6.0  # standard deviation
            indices = torch.arange(kernel_size, dtype=torch.float32)
            kernel = torch.exp(-0.5 * ((indices - center) / sigma) ** 2)
            # Normalize
            kernel = kernel / kernel.sum()
            
        else:
            raise ValueError(f"Unknown perturbation type: {perturbation_type}")
        
        return kernel
    
    
class TimeInvariantPerturbationLayer(nn.Module):

    def __init__(self, input_signal, perturbation_type='shift', kernel_size=5):
        super().__init__()
        self.x = input_signal.clone()  # [1, T]
        self.T = len(input_signal)
        self.perturbation_type = perturbation_type
        self.kernel_size = kernel_size
        
        self.kernel = self._create_kernel()
        self.pad = torch.zeros((1, self.kernel_size // 2), dtype=self.x.dtype)
        self.padded_signal = torch.cat([self.pad, self.x, self.pad], dim=1)
        self.residual = self.apply_perturbation(self.padded_signal) - self.x # [1, T]
        
        # Create linear layer: L(z) = z * residual + x
        # Input: z [B, 1], Output: [B, T]
        self.linear_layer = nn.Linear(1, self.T, bias=True)
        
        # Set weights and bias for the linear layer
        with torch.no_grad():
            # Weight is the residual signal, shape [T, 1]
            self.linear_layer.weight.data = self.residual.transpose(0, 1)  # [T, 1]
            # Bias is the original signal
            self.linear_layer.bias.data = self.x.flatten()
        
    def _create_kernel(self):
        """Create different 1D kernels based on perturbation type."""
        kernel = get_kernel(self.perturbation_type, self.kernel_size)
        return kernel
    
    def apply_perturbation(self, signal):
        """Apply the perturbation kernel to the signal using 1D convolution."""
        signal_reshaped = signal.unsqueeze(0) # [1, 1, T+2*pad]
        kernel_reshaped = self.kernel.unsqueeze(0).unsqueeze(0)  # [1, 1, K]
        perturbed = torch.nn.functional.conv1d(signal_reshaped, kernel_reshaped, padding=0) # [1, 1, T]
        return perturbed.squeeze(0)  # [1, T]
    
    def forward(self, z):
        # Forward pass implementing L(z) = z * (P(x) - x) + x, where P(x) is the perturbed signal
        # Use linear layer: L(z) = z * residual + x
        # z: [B, 1], linear_layer.weight: [T, 1], linear_layer.bias: [T]
        # Result: [B, T]
        return self.linear_layer(z)
    


def create_test_signal(T=64):
    """Create a test signal with multiple frequency components."""
    n = torch.arange(T, dtype=torch.float32)
    x = (
        torch.sin(2 * torch.pi * n / 16.0) +           # Main frequency
        0.5 * torch.sin(2 * torch.pi * n / 8.0) +     # Higher frequency
        0.3 * torch.cos(2 * torch.pi * n / 32.0)      # Lower frequency
    )
    
    # Add a distinctive pulse in the middle
    pulse_center = T // 2
    pulse_width = 5
    for i in range(-pulse_width, pulse_width + 1):
        idx = pulse_center + i
        if 0 <= idx < T:
            x[idx] += 0.8 * torch.exp(-torch.tensor((i**2) / (2 * (pulse_width/2)**2)))
    
    return x.unsqueeze(0)


def plot_perturbation_test(layer, z_values, save_path=None):
    """Plot the signal with z = 0, 0.5, 1.0 for the perturbation."""
    sns.set_style("whitegrid")
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    time_axis = np.arange(layer.x.shape[1])
    colors = sns.color_palette()
    
    # Plot signals for different z values
    ys = layer(z_values).squeeze(1)
    
    for i, z in enumerate(z_values):
        if z == 0.0:
            # z = 0: original signal
            ax.plot(time_axis, layer.x[0].numpy(), color='k', linewidth=2, label=f'z={z.item():.1f}', alpha=1.0)
        else:
            # z = 0.5, 1.0: interpolated signals
            ax.plot(time_axis, ys[i].detach().numpy(), color=colors[i], linewidth=2, label=f'z={z.item():.1f}', alpha=0.8)
    
    ax.set_title(f'{layer.perturbation_type.capitalize()} Perturbation (K={layer.kernel_size})')
    ax.set_xlabel('Time sample')
    ax.set_ylabel('Amplitude')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")


@torch.no_grad()
def main():
    torch.manual_seed(42)
    np.random.seed(42)

    T = 64
    x = create_test_signal(T)
    
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    Pz = TimeInvariantPerturbationLayer(x, perturbation_type='echo', kernel_size=3)
    print(f'{x=}')
    print(f'{Pz.padded_signal=}')
    print(f'{Pz.kernel=}')
    z = torch.tensor([[1.0]])
    print(f'{Pz(z)=}')
    exit()
    
    # Test different perturbation types
    perturbation_types = ['shift', 'lowpass', 'echo', 'highpass', 'gaussian']
    kernel_sizes = [5, 7]
    
    for pert_type in perturbation_types:
        for k_size in kernel_sizes:
            if k_size % 2 == 1:  # Only odd kernel sizes
                print(f"\nTesting {pert_type} with kernel size {k_size}")
                layer = TimeInvariantPerturbationLayer(x, perturbation_type=pert_type, kernel_size=k_size)
                print(f"Kernel: {layer.kernel}")
                z_batch = torch.tensor([[0.0], [0.1], [0.5], [0.8], [1.0]])  # [3, 1]
                plot_perturbation_test(layer, z_batch, save_path=f"perturbation_{pert_type}_k{k_size}.png")
                break

    torch.onnx.export(
        layer,
        torch.zeros(1, 1),
        'time_invariant_perturbation.onnx',
        opset_version=12,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'},
        }
    )

if __name__ == "__main__":
    main()
    