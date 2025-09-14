import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import math
import os

from perturbations.time_invariant import TimeInvariantPerturbationLayer
from perturbations.time_varying import TimeVaryingPerturbationLayer
from .utils import *

def plot_invariant_vs_varying(x, z, save_path=None):
    sns.set_style("whitegrid")
    time_axis = np.arange(x.shape[1])
    colors = sns.color_palette("bright", 3)
    
    # Plot 1: Invariant perturbation
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    ax.plot(time_axis, x.flatten().numpy(), color='k', linewidth=2, label=f'raw', alpha=1.0)
    
    for i, kernel_type in enumerate(['highpass', 'lowpass', 'echo']):
        layer_invariant = TimeInvariantPerturbationLayer(x, perturbation_type=kernel_type, kernel_size=7)
        y_invariant = layer_invariant(torch.tensor(z).view(1, 1))
        ax.plot(time_axis, y_invariant[0].numpy(), color=colors[i], linewidth=1.5, label=f'{kernel_type}', alpha=0.8)
    ax.set_xlabel('Sample')
    ax.set_ylabel('Amplitude')
    ax.legend(fancybox=True)
    ax.grid(True, alpha=0.3)
    # ax.set_xticks(time_axis)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'example_time_invariant.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Varying perturbation
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    ax.plot(time_axis, x.flatten().numpy(), color='k', linewidth=2, label=f'raw', alpha=1.0)
    for i, displacement_type in enumerate(['linear', 'gaussian', 'sinusoidal']):
        layer_varying = TimeVaryingPerturbationLayer(x, displacement_type=displacement_type, max_displacement=5.0, window_size=x.shape[1]//1)
        y_varying = layer_varying(torch.tensor(z).view(1, 1))
        ax.plot(time_axis, y_varying[0].numpy(), color=colors[i], linewidth=1.5, label=f'{displacement_type}', alpha=0.8)
    ax.set_xlabel('Sample')
    ax.set_ylabel('Amplitude')
    ax.legend(fancybox=True)
    ax.grid(True, alpha=0.3)
    # ax.set_xticks(time_axis)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'example_time_varying.pdf'), dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.close()


@torch.no_grad()
def main():
    torch.manual_seed(42)
    np.random.seed(42)

    T = 64
    x = create_test_signal(T)
    
    z = 1.0  # [B, 1]
    plot_invariant_vs_varying(x, z, save_path="../../figure/")
    
    

if __name__ == "__main__":
    main()
