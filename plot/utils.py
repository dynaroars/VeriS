import matplotlib
import matplotlib.pyplot as plt
import torch
import math

matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 11
# plt.rc('axes', labelsize=35)


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
            x[idx] += 0.8 * math.exp(-(i**2) / (2 * (pulse_width/2)**2))
    
    return x.unsqueeze(0)
