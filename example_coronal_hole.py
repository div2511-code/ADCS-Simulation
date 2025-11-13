"""
Example script demonstrating how to use the coronal hole structure module
with custom parameters and analysis.
"""

from coronal_hole_structure import (
    CoronalHoleParams,
    plot_coronal_hole_structure,
    solve_pressure_distribution,
    temperature,
    pressure_base
)
import numpy as np
import matplotlib.pyplot as plt


def example_basic():
    """Run with default parameters"""
    print("Example 1: Default parameters")
    print("-" * 60)
    params = CoronalHoleParams()
    fig, data = plot_coronal_hole_structure(params, save_fig=True,
                                           filename='example_default.png')
    plt.close()


def example_custom_parameters():
    """Run with custom parameters"""
    print("\nExample 2: Custom parameters")
    print("-" * 60)

    params = CoronalHoleParams()

    # Modify parameters
    params.T_CH = 7e5      # Cooler coronal hole
    params.T_C = 2e6       # Hotter closed region
    params.p_CH = 0.2e-1   # Lower CH pressure
    params.p_C = 4.0e-1    # Higher closed pressure
    params.z_max = 150.0   # Extend to 150 Mm

    print(f"Custom T_CH: {params.T_CH/1e6:.2f} MK")
    print(f"Custom T_C:  {params.T_C/1e6:.2f} MK")
    print(f"Custom z_max: {params.z_max} Mm")

    fig, data = plot_coronal_hole_structure(params, save_fig=True,
                                           filename='example_custom.png')
    plt.close()


def example_data_analysis():
    """Example of analyzing the computed data"""
    print("\nExample 3: Data analysis")
    print("-" * 60)

    params = CoronalHoleParams()
    A_grid, z_grid, p_grid, T_grid = solve_pressure_distribution(params, n_A=100, n_z=200)

    # Find scale height at different A values
    print("\nScale heights at different flux values:")
    A_samples = [0.2, 0.5, 0.8, 1.0]

    for A_val in A_samples:
        idx = np.argmin(np.abs(A_grid[0, :] - A_val))

        # Fit exponential to get scale height
        p_profile = p_grid[:, idx]
        z_profile = z_grid[:, idx] * 1e6  # Convert to meters

        # Use first 50 points for fit
        valid = p_profile > 0
        if np.sum(valid) > 10:
            log_p = np.log(p_profile[valid][:50])
            z_fit = z_profile[valid][:50]

            # Linear fit to log(p) vs z gives -1/H as slope
            coeffs = np.polyfit(z_fit, log_p, 1)
            H_fitted = -1 / coeffs[0] / 1e6  # Convert to Mm

            T_val = T_grid[0, idx]
            print(f"  A = {A_val:.1f}: H = {H_fitted:.1f} Mm, T = {T_val/1e6:.2f} MK")


def example_comparison_plot():
    """Create comparison plot for different parameter sets"""
    print("\nExample 4: Parameter comparison")
    print("-" * 60)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Different temperature contrasts
    params_list = [
        {'T_CH': 8e5, 'T_C': 1.5e6, 'label': 'Standard'},
        {'T_CH': 7e5, 'T_C': 2.0e6, 'label': 'High contrast'},
        {'T_CH': 9e5, 'T_C': 1.3e6, 'label': 'Low contrast'},
    ]

    # Plot temperature profiles
    ax = axes[0]
    for p_dict in params_list:
        params = CoronalHoleParams()
        params.T_CH = p_dict['T_CH']
        params.T_C = p_dict['T_C']

        A_grid, z_grid, p_grid, T_grid = solve_pressure_distribution(params, n_A=50, n_z=100)

        # Plot for A=0.5 (middle field line)
        idx = np.argmin(np.abs(A_grid[0, :] - 0.5))
        ax.plot(T_grid[:, idx] / 1e6, z_grid[:, idx],
               label=p_dict['label'], linewidth=2)

    ax.set_xlabel('Temperature (MK)')
    ax.set_ylabel('Height z (Mm)')
    ax.set_title('Temperature Profiles (A=0.5)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot base temperature functions
    ax = axes[1]
    A_range = np.linspace(0.01, 1.0, 100)

    for p_dict in params_list:
        params = CoronalHoleParams()
        params.T_CH = p_dict['T_CH']
        params.T_C = p_dict['T_C']

        T_base = temperature(A_range, params)
        ax.plot(A_range, T_base / 1e6, label=p_dict['label'], linewidth=2)

    ax.set_xlabel('Flux Function A')
    ax.set_ylabel('Temperature (MK)')
    ax.set_title('Base Temperature T(A)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('example_comparison.png', dpi=300, bbox_inches='tight')
    print("Comparison plot saved as 'example_comparison.png'")
    plt.close()


if __name__ == "__main__":
    print("="*60)
    print("CORONAL HOLE STRUCTURE - USAGE EXAMPLES")
    print("="*60)

    # Run examples
    example_basic()
    example_custom_parameters()
    example_data_analysis()
    example_comparison_plot()

    print("\n" + "="*60)
    print("All examples completed successfully!")
    print("="*60)
    print("\nGenerated files:")
    print("  - example_default.png")
    print("  - example_custom.png")
    print("  - example_comparison.png")
