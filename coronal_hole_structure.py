"""
Coronal Hole Thermal and Magnetic Structure Simulation

Based on the paper:
"Construction of coronal hole and active region magnetohydrostatic solutions
in two dimensions: Force and energy balance" by J. Terradas

This module solves the magnetohydrostatic equation for a coronal hole
and plots the thermal and magnetic structure.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.interpolate import interp1d

# Physical constants
R = 8.314e3  # Gas constant (J/(K·kmol))
g = 274.0    # Solar surface gravity (m/s^2)
mu_bar = 0.6  # Mean molecular weight (dimensionless)

# Coronal hole and closed field parameters
class CoronalHoleParams:
    """Parameters for the coronal hole model"""
    def __init__(self):
        # Temperature parameters (K)
        self.T_CH = 8e5   # Coronal hole temperature
        self.T_C = 1.5e6  # Closed field region temperature

        # Pressure parameters (Pa)
        self.p_CH = 0.3e-1  # Coronal hole pressure
        self.p_C = 3.0e-1   # Closed field region pressure

        # Reference flux function value
        self.A_ref = 1.0

        # Height scale (Mm)
        self.z_max = 100.0  # Maximum height in Mm

        # Flux function range
        self.A_min = 0.0
        self.A_max = 1.0


def temperature(A, params):
    """
    Temperature as a function of flux function A
    T(A) = (T_C - T_CH) * (A/A_ref) + T_CH

    Args:
        A: Flux function value(s)
        params: CoronalHoleParams object

    Returns:
        Temperature in K
    """
    return (params.T_C - params.T_CH) * (A / params.A_ref) + params.T_CH


def pressure_base(A, params):
    """
    Base pressure as a function of flux function A (at z=0)
    p0(A) = (p_C - p_CH) * (A/A_ref)^2 + p_CH

    Args:
        A: Flux function value(s)
        params: CoronalHoleParams object

    Returns:
        Pressure in Pa
    """
    return (params.p_C - params.p_CH) * (A / params.A_ref)**2 + params.p_CH


def dT_dA(A, params):
    """
    Derivative of temperature with respect to A

    Args:
        A: Flux function value
        params: CoronalHoleParams object

    Returns:
        dT/dA
    """
    return (params.T_C - params.T_CH) / params.A_ref


def dp0_dA(A, params):
    """
    Derivative of base pressure with respect to A

    Args:
        A: Flux function value
        params: CoronalHoleParams object

    Returns:
        dp0/dA
    """
    return 2 * (params.p_C - params.p_CH) * A / (params.A_ref**2)


def pressure_derivative(A, z, params):
    """
    Calculate ∂p/∂A(A,z) based on the magnetohydrostatic equation:

    ∂p/∂A(A,z) = exp(-z * μ̄g / (RT(A))) * [∂p0/∂A(A) + z * μ̄g/R * ∂T/∂A(A) / T²(A)]

    Args:
        A: Flux function value
        z: Height (in meters for calculation)
        params: CoronalHoleParams object

    Returns:
        ∂p/∂A
    """
    T_A = temperature(A, params)

    # Scale height
    H = R * T_A / (mu_bar * g)

    # Exponential factor
    exp_factor = np.exp(-z / H)

    # Derivative terms
    dp0_term = dp0_dA(A, params)
    dT_term = z * (mu_bar * g / R) * dT_dA(A, params) / (T_A**2)

    return exp_factor * (dp0_term + dT_term)


def solve_pressure_distribution(params, n_A=50, n_z=100):
    """
    Solve for pressure distribution p(A, z)

    This uses the hydrostatic equilibrium along z for each A value

    Args:
        params: CoronalHoleParams object
        n_A: Number of A grid points
        n_z: Number of z grid points

    Returns:
        A_grid, z_grid, p_grid, T_grid
    """
    # Create grids
    A_values = np.linspace(params.A_min + 0.01, params.A_max, n_A)
    z_values = np.linspace(0, params.z_max * 1e6, n_z)  # Convert Mm to m

    A_grid, z_grid = np.meshgrid(A_values, z_values)

    # Initialize pressure and temperature grids
    p_grid = np.zeros_like(A_grid)
    T_grid = np.zeros_like(A_grid)

    # For each A value, solve the pressure distribution along z
    for i, A in enumerate(A_values):
        T_A = temperature(A, params)
        p0_A = pressure_base(A, params)

        # Scale height for this flux tube
        H = R * T_A / (mu_bar * g)

        # Hydrostatic equilibrium: p(z) = p0 * exp(-z/H)
        # (simplified version for constant temperature along field line)
        p_grid[:, i] = p0_A * np.exp(-z_values / H)
        T_grid[:, i] = T_A

    return A_grid, z_grid / 1e6, p_grid, T_grid  # Convert z back to Mm


def calculate_magnetic_field(A_grid, z_grid, B0=10.0):
    """
    Calculate magnetic field magnitude from flux function

    For a simple model: B_z ∝ ∂A/∂x, B_x ∝ -∂A/∂z
    We'll use a simplified approach: |B| ∝ A for visualization

    Args:
        A_grid: Flux function grid
        z_grid: Height grid
        B0: Reference magnetic field strength (Gauss)

    Returns:
        B_grid: Magnetic field magnitude
    """
    # Simplified model: field strength proportional to A
    # In reality, this would require spatial derivatives of A
    B_grid = B0 * A_grid
    return B_grid


def plot_coronal_hole_structure(params=None, save_fig=False, filename='coronal_hole_structure.png'):
    """
    Create comprehensive plots of the coronal hole structure

    Args:
        params: CoronalHoleParams object (creates default if None)
        save_fig: Whether to save the figure
        filename: Filename for saved figure
    """
    if params is None:
        params = CoronalHoleParams()

    # Solve for the structure
    print("Solving for pressure distribution...")
    A_grid, z_grid, p_grid, T_grid = solve_pressure_distribution(params)
    B_grid = calculate_magnetic_field(A_grid, z_grid)

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))

    # 1. Temperature distribution
    ax1 = plt.subplot(2, 3, 1)
    contour1 = ax1.contourf(A_grid, z_grid, T_grid / 1e6, levels=20, cmap='hot')
    ax1.set_xlabel('Flux Function A')
    ax1.set_ylabel('Height z (Mm)')
    ax1.set_title('Temperature Distribution T(A,z)')
    cbar1 = plt.colorbar(contour1, ax=ax1)
    cbar1.set_label('Temperature (MK)')
    ax1.grid(True, alpha=0.3)

    # 2. Pressure distribution (log scale)
    ax2 = plt.subplot(2, 3, 2)
    contour2 = ax2.contourf(A_grid, z_grid, np.log10(p_grid), levels=20, cmap='viridis')
    ax2.set_xlabel('Flux Function A')
    ax2.set_ylabel('Height z (Mm)')
    ax2.set_title('Pressure Distribution log₁₀[p(A,z)]')
    cbar2 = plt.colorbar(contour2, ax=ax2)
    cbar2.set_label('log₁₀(Pressure) [Pa]')
    ax2.grid(True, alpha=0.3)

    # 3. Magnetic field strength
    ax3 = plt.subplot(2, 3, 3)
    contour3 = ax3.contourf(A_grid, z_grid, B_grid, levels=20, cmap='plasma')
    ax3.set_xlabel('Flux Function A')
    ax3.set_ylabel('Height z (Mm)')
    ax3.set_title('Magnetic Field Strength |B|(A,z)')
    cbar3 = plt.colorbar(contour3, ax=ax3)
    cbar3.set_label('Magnetic Field (G)')
    ax3.grid(True, alpha=0.3)

    # 4. Temperature profile along selected field lines
    ax4 = plt.subplot(2, 3, 4)
    A_samples = [0.2, 0.4, 0.6, 0.8, 1.0]
    for A_val in A_samples:
        idx = np.argmin(np.abs(A_grid[0, :] - A_val))
        ax4.plot(T_grid[:, idx] / 1e6, z_grid[:, idx], label=f'A = {A_val:.1f}')
    ax4.set_xlabel('Temperature (MK)')
    ax4.set_ylabel('Height z (Mm)')
    ax4.set_title('Temperature Profiles along Field Lines')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Pressure profile along selected field lines
    ax5 = plt.subplot(2, 3, 5)
    for A_val in A_samples:
        idx = np.argmin(np.abs(A_grid[0, :] - A_val))
        ax5.semilogy(p_grid[:, idx], z_grid[:, idx], label=f'A = {A_val:.1f}')
    ax5.set_xlabel('Pressure (Pa)')
    ax5.set_ylabel('Height z (Mm)')
    ax5.set_title('Pressure Profiles along Field Lines')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. Base quantities vs A
    ax6 = plt.subplot(2, 3, 6)
    A_range = np.linspace(0.01, 1.0, 100)
    T_base = temperature(A_range, params)
    p_base = pressure_base(A_range, params)

    ax6_twin = ax6.twinx()
    line1 = ax6.plot(A_range, T_base / 1e6, 'r-', linewidth=2, label='Temperature')
    line2 = ax6_twin.plot(A_range, p_base, 'b-', linewidth=2, label='Pressure')

    ax6.set_xlabel('Flux Function A')
    ax6.set_ylabel('Temperature (MK)', color='r')
    ax6_twin.set_ylabel('Pressure (Pa)', color='b')
    ax6.set_title('Base Temperature and Pressure vs A')
    ax6.tick_params(axis='y', labelcolor='r')
    ax6_twin.tick_params(axis='y', labelcolor='b')
    ax6.grid(True, alpha=0.3)

    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax6.legend(lines, labels, loc='upper left')

    plt.tight_layout()

    if save_fig:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Figure saved as {filename}")

    plt.show()

    return fig, (A_grid, z_grid, p_grid, T_grid, B_grid)


def print_parameters(params):
    """Print the model parameters"""
    print("\n" + "="*60)
    print("CORONAL HOLE MODEL PARAMETERS")
    print("="*60)
    print(f"Temperature - Coronal Hole: {params.T_CH/1e6:.2f} MK")
    print(f"Temperature - Closed Field:  {params.T_C/1e6:.2f} MK")
    print(f"Pressure - Coronal Hole:     {params.p_CH:.3f} Pa")
    print(f"Pressure - Closed Field:     {params.p_C:.3f} Pa")
    print(f"Reference flux value:        {params.A_ref}")
    print(f"Maximum height:              {params.z_max:.1f} Mm")
    print(f"Mean molecular weight:       {mu_bar}")
    print(f"Solar surface gravity:       {g} m/s²")
    print("="*60 + "\n")


if __name__ == "__main__":
    # Create default parameters
    params = CoronalHoleParams()

    # Print parameters
    print_parameters(params)

    # Generate and plot the coronal hole structure
    print("Generating coronal hole structure plots...")
    fig, data = plot_coronal_hole_structure(params, save_fig=True)

    print("\nSimulation complete!")
    print("\nPlots show:")
    print("  1. Temperature distribution T(A,z)")
    print("  2. Pressure distribution p(A,z)")
    print("  3. Magnetic field strength |B|(A,z)")
    print("  4. Temperature profiles along different field lines")
    print("  5. Pressure profiles along different field lines")
    print("  6. Base temperature and pressure as functions of A")
