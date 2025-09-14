import numpy as np
import os


def find_neighborhood_bounds(input_array, kernel_size):
    """
    Find upper and lower bounds for each position using neighborhood kernels.

    Args:
        input_array: 1D numpy array
        kernel_size: Size of the neighborhood kernel (must be odd)

    Returns:
        upper_bounds: Array of maximum values in each neighborhood
        lower_bounds: Array of minimum values in each neighborhood
    """
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be odd")

    n = len(input_array)
    upper_bounds = np.zeros(n)
    lower_bounds = np.zeros(n)

    half_kernel = kernel_size // 2

    for i in range(n):
        # Define neighborhood boundaries
        start = max(0, i - half_kernel)
        end = min(n, i + half_kernel + 1)

        # Extract neighborhood
        neighborhood = input_array[start:end]

        # Find max and min in neighborhood
        upper_bounds[i] = np.max(neighborhood)
        lower_bounds[i] = np.min(neighborhood)

    return upper_bounds, lower_bounds


def create_vnnlib_spec(
    input_array, upper_bounds, lower_bounds, num_classes=2, target_class=0, output_file="spec.vnnlib"
):
    """
    Create a VNN-LIB specification file with input bounds.

    Args:
        input_array: Original 1D input array
        upper_bounds: Upper bounds for each position
        lower_bounds: Lower bounds for each position
        num_classes: Number of output classes
        target_class: Class that should remain the highest prediction (default: 0)
        output_file: Output VNN-LIB file path
    """
    n = len(input_array)

    with open(output_file, "w") as f:
        # Write header
        f.write("; VNN-LIB specification for min/max kernel perturbation\n")
        f.write(f"; Input size: {n}\n")
        f.write(f"; Number of classes: {num_classes}\n")
        f.write(f"; Target class to remain permanent: {target_class}\n")

        # Declare input variables
        f.write("; Input variables\n")
        for i in range(n):
            f.write(f"(declare-const X_{i} Real)\n")
        f.write("\n")

        # Input constraints (bounds)
        f.write("; Input constraints\n")
        for i in range(n):
            f.write(f"(assert (>= X_{i} {lower_bounds[i]:.6f}))\n")
            f.write(f"(assert (<= X_{i} {upper_bounds[i]:.6f}))\n")
        f.write("\n")

        # Output variables
        f.write("; Output variables\n")
        for i in range(num_classes):
            f.write(f"(declare-const Y_{i} Real)\n")
        f.write("\n")

        # Create assertion that ensures the target class remains the highest
        f.write(f"; Assertion: Class {target_class} should remain the highest probability\n")
        for i in range(num_classes):
            if i != target_class:
                f.write(f"(assert (>= Y_{target_class} Y_{i}))\n")
        f.write("\n")


def process_array_and_create_spec(input_array, kernel_size=5, num_classes=2, target_class=0, output_file="spec.vnnlib"):
    """
    Complete pipeline: process array and create VNN-LIB spec.

    Args:
        input_array: 1D numpy array to process
        kernel_size: Size of neighborhood kernel (must be odd)
        num_classes: Number of output classes
        target_class: Class that should remain the highest prediction (default: 0)
        output_file: Output VNN-LIB file path

    Returns:
        upper_bounds: Array of upper bounds
        lower_bounds: Array of lower bounds
    """
    # Step 1: Receive 1D array (already done)
    print(f"Input array: {input_array}")
    print(f"Array length: {len(input_array)}")

    # Step 2: Find upper bounds using max kernel
    print(f"\nStep 2: Finding upper bounds with kernel size {kernel_size}")
    upper_bounds, _ = find_neighborhood_bounds(input_array, kernel_size)
    print(f"Upper bounds: {upper_bounds}")

    # Step 3: Find lower bounds using min kernel
    print(f"\nStep 3: Finding lower bounds with kernel size {kernel_size}")
    _, lower_bounds = find_neighborhood_bounds(input_array, kernel_size)
    print(f"Lower bounds: {lower_bounds}")

    # Step 4: Create VNN-LIB spec file
    print(f"\nStep 4: Creating VNN-LIB specification file: {output_file}")
    print(f"Target class to remain permanent: {target_class}")
    create_vnnlib_spec(input_array, upper_bounds, lower_bounds, num_classes, target_class, output_file)
    print(f"VNN-LIB specification created successfully!")

    return upper_bounds, lower_bounds


# Example usage
if __name__ == "__main__":
    # Example 1D array
    test_array = np.array([1.0, 3.0, 2.0, 5.0, 4.0, 1.5, 6.0, 2.5])

    # Process array and create spec with 3 classes, ensuring class 1 remains permanent
    upper_bounds, lower_bounds = process_array_and_create_spec(
        test_array, kernel_size=3, num_classes=3, target_class=1, output_file="min_max_kernel_spec.vnnlib"
    )

    print(f"\nSummary:")
    print(f"Original array: {test_array}")
    print(f"Upper bounds:   {upper_bounds}")
    print(f"Lower bounds:   {lower_bounds}")
