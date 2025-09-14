import numpy as np
import torch
import tqdm


def find_neighborhood_bounds(input_array, kernel, kernel_bounds=None):
    """
    Find upper and lower bounds for convolution output when kernel values can vary.
    
    This function computes bounds for convolution output x' = conv(x, K) when 
    each kernel element K[i] can vary between K_min and K_max.

    Args:
        input_array: 1D numpy array (fixed input image/signal)
        kernel: 1D numpy array representing the kernel shape/pattern (used for size)
        kernel_bounds: tuple (K_min, K_max) where each kernel element can vary between these scalar values

    Returns:
        upper_bounds: Tensor of maximum possible convolution values at each position
        lower_bounds: Tensor of minimum possible convolution values at each position
    """
    if len(kernel) % 2 == 0:
        raise ValueError("Kernel size must be odd")

    # Convert input to torch tensor
    input_tensor = torch.tensor(input_array, dtype=torch.float32)
    n = len(input_tensor)
    kernel_size = len(kernel)
    half_kernel = kernel_size // 2
    
    K_min, K_max = kernel_bounds
    
    # Pad input to handle boundaries
    padded_input = torch.nn.functional.pad(input_tensor, (half_kernel, half_kernel), mode='constant', value=0)
    
    # Create neighborhoods using unfold operation
    # unfold(dimension, size, step) - extracts sliding windows
    neighborhoods = padded_input.unfold(0, kernel_size, 1)  # Shape: (n, kernel_size)
    
    # Split neighborhoods into positive and negative parts
    positive_parts = torch.clamp(neighborhoods, min=0)
    negative_parts = torch.clamp(neighborhoods, max=0)
    
    # For upper bounds:
    upper_bounds = positive_parts.sum(dim=1) * K_max + negative_parts.sum(dim=1) * K_min
    
    # For lower bounds:
    lower_bounds = positive_parts.sum(dim=1) * K_min + negative_parts.sum(dim=1) * K_max
    
    # Ensure bounds are valid (lower <= upper)
    assert torch.all(upper_bounds >= lower_bounds), "Upper bounds must be >= lower bounds"
    
    return upper_bounds, lower_bounds



def find_neighborhood_bounds_old(input_array, kernel, kernel_bounds=None):
    """
    Find upper and lower bounds for each position using neighborhood kernels.

    Args:
        input_array: 1D numpy array
        kernel: 1D numpy array representing the kernel weights

    Returns:
        upper_bounds: Array of maximum values in each neighborhood
        lower_bounds: Array of minimum values in each neighborhood
    """
    if len(kernel) % 2 == 0:
        raise ValueError("Kernel size must be odd")

    n = len(input_array)
    upper_bounds = np.zeros(n)
    lower_bounds = np.zeros(n)

    half_kernel = len(kernel) // 2

    for i in range(n):
        # Define neighborhood boundaries
        start = max(0, i - half_kernel)
        end = min(n, i + half_kernel + 1)

        # Extract neighborhood
        neighborhood = input_array[start:end]

        # Calculate the corresponding kernel slice
        # Adjust kernel start/end to match the neighborhood
        kernel_start = max(0, half_kernel - i)
        kernel_end = kernel_start + len(neighborhood)
        kernel_slice = np.array(kernel)[kernel_start:kernel_end]
        max_kernel = np.where(kernel_slice > 0, kernel_slice, 0)
        min_kernel = np.where(kernel_slice < 0, kernel_slice, 0)

        # Find max and min in weighted neighborhood
        upper_bounds[i] = np.sum(neighborhood * max_kernel)
        lower_bounds[i] = np.sum(neighborhood * min_kernel)

    return upper_bounds, lower_bounds


# Example usage
if __name__ == "__main__":
    # Example 1D array
    test_array = np.array([1.0, 3.0, 2.0, 5.0, 4.0, 1.5, 6.0, 2.5])
    kernel = [-100, 100, -100]
    upper_bounds, lower_bounds = find_neighborhood_bounds(test_array, kernel)
    print(f"Upper bounds:   {upper_bounds}")
    print(f"Lower bounds:   {lower_bounds}")
    