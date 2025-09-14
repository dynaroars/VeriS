import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import math
import os

from perturbations.time_varying import TimeVaryingPerturbationLayer
from .utils import *

def plot_time_varying_example(layer, z_values, save_path=None):
    """Plot the polynomial time-warp perturbation results."""
    sns.set_style("whitegrid")
    
    
    time_axis = np.arange(layer.x.shape[1])
    colors = sns.color_palette("Reds", len(z_values))
    
    # Plot 1: Displacement fields for different z values
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    for i, z in enumerate(z_values):
        u = layer.get_displacement_field(torch.tensor([z]).unsqueeze(0))
        u_single = u[0]
        ax.plot(time_axis, u_single.numpy(), color=colors[i], linewidth=1.5, 
               label=f'u(z={z.item():.1f})', alpha=0.7, marker='o', markersize=4)
    ax.set_xlabel('Sample')
    ax.set_ylabel('Amplitude')
    ax.legend(fancybox=True)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-2, 2)
    ax.set_xticks(time_axis)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'ltv_displacement_field.pdf'), dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.close()
    # Plot 2: Effect of different z values (z fixed)
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    ax.plot(time_axis, layer.x.squeeze().numpy(), 'k-', linewidth=2, label=f'$x$', alpha=0.8, marker='o', markersize=4)
    
    ys = layer(z_values)
    for i, z in enumerate(z_values):
        y_single = ys[i]
        ax.plot(time_axis, y_single.numpy(), color=colors[i], linewidth=1.5, 
                label=f'$\hat{{x}}$ (z={z.item():.1f})', alpha=0.7, marker='o', markersize=4)
    
    ax.set_xlabel('Sample')
    ax.set_ylabel('Amplitude')
    ax.legend(fancybox=True)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(time_axis)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'ltv_perturbed_signal.pdf'), dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")


@torch.no_grad()
def main():
    torch.manual_seed(42)
    np.random.seed(42)

    x = torch.tensor([1.0, 2.0, 3.0, 4.0]).view(1, -1)
    u = torch.tensor([0.2, -1.3, 0.4, -1.2]).view(1, -1)
    
    layer = TimeVaryingPerturbationLayer(x, displacement_type='sinusoidal', max_displacement=3.0, window_size=x.shape[1])
    layer.coeffs = u
    layer.create_s_layer()
    
    print(f"{layer.coeffs=}")
    print(f"{layer.coeffs.shape=}, {layer.coeffs=}")

    z = torch.tensor([[0.2], [0.5], [1.0]])  # [B, 1]
    y = layer(z)
    print(f"{x.shape=}, {z.shape=}, {y.shape=}")
    
    print(f"{y=}")
    plot_time_varying_example(layer, z, save_path="../../figure/")
    # exit()
    

if __name__ == "__main__":
    main()
