# Coronal Hole Magnetohydrostatic Structure Simulation

This module implements the magnetohydrostatic equations for coronal holes based on the work of J. Terradas: *"Construction of coronal hole and active region magnetohydrostatic solutions in two dimensions: Force and energy balance"*.

## Overview

Coronal holes are regions of the solar corona with open magnetic field lines, characterized by:
- Lower temperatures (~0.8 MK) compared to closed field regions (~1.5 MK)
- Lower densities and pressures
- High-speed solar wind sources

## Mathematical Model

### Governing Equation

The magnetohydrostatic pressure equation solved is:

```
∂p/∂A(A,z) = exp(-z·μ̄g/RT(A)) · [∂p₀/∂A(A) + z·(μ̄g/R)·∂T/∂A(A)/T²(A)]
```

Where:
- `p(A,z)`: Pressure as a function of flux function A and height z
- `A`: Magnetic flux function (labels field lines)
- `z`: Height above the solar surface
- `μ̄`: Mean molecular weight
- `g`: Solar surface gravity (274 m/s²)
- `R`: Gas constant
- `T(A)`: Temperature function

### Boundary Conditions

**Temperature function:**
```
T(A) = (T_C - T_CH)·(A/A_ref) + T_CH
```

**Base pressure function (at z=0):**
```
p₀(A) = (p_C - p_CH)·(A/A_ref)² + p_CH
```

Where:
- `T_CH = 0.8 MK`: Coronal hole temperature
- `T_C = 1.5 MK`: Closed field region temperature
- `p_CH = 0.03 Pa`: Coronal hole pressure
- `p_C = 0.3 Pa`: Closed field region pressure
- `A_ref = 1.0`: Reference flux value

## Installation

### Requirements

```bash
pip install numpy scipy matplotlib
```

Or use the provided requirements file:
```bash
pip install -r requirements_coronal.txt
```

## Usage

### Basic Usage

Run the simulation with default parameters:

```bash
python coronal_hole_structure.py
```

This will:
1. Print the model parameters
2. Solve the magnetohydrostatic equations
3. Generate 6 subplots showing different aspects of the structure
4. Save the figure as `coronal_hole_structure.png`

### Custom Parameters

```python
from coronal_hole_structure import CoronalHoleParams, plot_coronal_hole_structure

# Create custom parameters
params = CoronalHoleParams()
params.T_CH = 7e5    # Adjust coronal hole temperature
params.T_C = 2e6     # Adjust closed field temperature
params.p_CH = 0.2e-1 # Adjust pressures
params.p_C = 2.5e-1
params.z_max = 150.0 # Extend height to 150 Mm

# Generate plots
fig, data = plot_coronal_hole_structure(params, save_fig=True,
                                       filename='custom_coronal_hole.png')
```

### Accessing the Data

```python
from coronal_hole_structure import solve_pressure_distribution, CoronalHoleParams

params = CoronalHoleParams()
A_grid, z_grid, p_grid, T_grid = solve_pressure_distribution(params)

# A_grid: Flux function values (2D array)
# z_grid: Height values in Mm (2D array)
# p_grid: Pressure in Pa (2D array)
# T_grid: Temperature in K (2D array)

# Example: Get pressure at specific location
i, j = 50, 25  # Grid indices
print(f"At A={A_grid[i,j]:.2f}, z={z_grid[i,j]:.1f} Mm:")
print(f"  Pressure: {p_grid[i,j]:.3e} Pa")
print(f"  Temperature: {T_grid[i,j]/1e6:.2f} MK")
```

## Output Plots

The code generates 6 plots:

1. **Temperature Distribution T(A,z)**: Contour plot showing how temperature varies with flux function and height
2. **Pressure Distribution log₁₀[p(A,z)]**: Log-scale pressure contours
3. **Magnetic Field Strength |B|(A,z)**: Simplified magnetic field distribution
4. **Temperature Profiles**: Temperature vs height for different field lines (different A values)
5. **Pressure Profiles**: Pressure vs height for different field lines (semi-log plot)
6. **Base Quantities**: Temperature and pressure at the base (z=0) as functions of A

## Physical Interpretation

- **Small A values (A→0)**: Represent coronal hole field lines
  - Lower temperature (~0.8 MK)
  - Lower pressure
  - Larger scale heights (slower pressure decrease with height)

- **Large A values (A→1)**: Represent closed field region
  - Higher temperature (~1.5 MK)
  - Higher pressure
  - Smaller scale heights (faster pressure decrease)

## Key Features

- ✅ Solves 2D magnetohydrostatic equilibrium
- ✅ Implements Terradas et al. formulation
- ✅ Generates comprehensive visualization suite
- ✅ Customizable physical parameters
- ✅ Exports high-resolution figures
- ✅ Provides access to raw numerical data

## References

- Terradas, J. (2009). "Construction of coronal hole and active region magnetohydrostatic solutions in two dimensions: Force and energy balance"
- Solar coronal physics and magnetohydrostatics
- Coronal hole observations and modeling

## Notes

- Heights are in megameters (Mm): 1 Mm = 1000 km
- Temperatures are in Kelvin (K) or megakelvin (MK): 1 MK = 10⁶ K
- Pressures are in Pascals (Pa)
- The simulation assumes:
  - Hydrostatic equilibrium along field lines
  - Temperature constant along individual field lines
  - Simplified magnetic field topology

## Future Enhancements

Potential improvements:
- [ ] Full 2D spatial grid (x, z) instead of (A, z)
- [ ] Include energy equation for variable temperature along field lines
- [ ] Add magnetic field line visualization
- [ ] Implement more complex flux function geometries
- [ ] Include solar wind acceleration effects
- [ ] Add time-dependent evolution

## License

This code is provided for educational and research purposes.
