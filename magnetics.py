# magnetics.py
import numpy as np
from scipy.special import ellipk, ellipe

# --- Constants ---
mu_0 = 4 * np.pi * 1e-7  # Vacuum permeability

def calculate_b_field_off_axis(coil_points, current_per_path, r_max, z_max, num_r_points=30, num_z_points=30):
    """
    Calculates the magnetic field (Br, Bz) in the r-z plane from coil path points.
    It models the continuous coil path as a series of discrete circular current loops.
    """
    # Create the grid for calculation. Start r slightly off-axis to avoid singularity at r=0.
    r_grid = np.linspace(1e-6, r_max, num_r_points)
    z_grid = np.linspace(-z_max, z_max, num_z_points)
    R_mesh, Z_mesh = np.meshgrid(r_grid, z_grid)
    
    Br_total = np.zeros_like(R_mesh, dtype=float)
    Bz_total = np.zeros_like(R_mesh, dtype=float)

    # Treat each point in the coil path as the center of a small, circular current loop
    # The current is per path, so we divide by number of points to get current per loop element
    # This is an approximation; a true line current integral would be more complex.
    if len(coil_points) == 0:
        return Br_total.T, Bz_total.T, r_grid, z_grid
        
    current_per_loop_element = current_per_path # Assume each point represents one turn element of the path

    loop_radii = np.sqrt(coil_points[:, 0]**2 + coil_points[:, 1]**2)
    loop_z = coil_points[:, 2]

    # Pre-calculate elliptic integrals for all grid points and loops
    for r_idx in range(num_r_points):
        for z_idx in range(num_z_points):
            r_point = R_mesh[z_idx, r_idx]
            z_point = Z_mesh[z_idx, r_idx]
            
            # Sum contributions from all coil loop elements
            # This is vectorized over the coil points for better performance
            R_loops, z_loops = loop_radii, loop_z
            
            dz_sq = (z_point - z_loops)**2
            m_num_den = (R_loops + r_point)**2 + dz_sq
            
            # Avoid division by zero if a grid point is exactly on a coil
            m_num_den[m_num_den == 0] = 1e-12 

            m = (4 * R_loops * r_point) / m_num_den
            m = np.clip(m, 0, 1) # Ensure m is in [0, 1] for elliptic functions
            
            K_m, E_m = ellipk(m), ellipe(m)
            
            common_den = (R_loops - r_point)**2 + dz_sq + 1e-12
            common_factor = mu_0 * current_per_loop_element / (2 * np.pi * np.sqrt(m_num_den))
            
            # Sum Bz and Br contributions from all coil points
            Bz_p = np.sum(common_factor * (K_m + (R_loops**2 - r_point**2 - dz_sq) / common_den * E_m))
            
            # --- CORRECTED: Handle r_point=0 case explicitly, although grid avoids it. ---
            # This term is singular at r_point=0, but Br should be 0 on-axis by symmetry.
            if r_point > 0:
                Br_p = np.sum(common_factor * ((z_point - z_loops) / r_point) * \
                        (-K_m + (R_loops**2 + r_point**2 + dz_sq) / common_den * E_m))
            else:
                Br_p = 0.0

            Br_total[z_idx, r_idx], Bz_total[z_idx, r_idx] = Br_p, Bz_p
            
    # The meshgrid and loop structure mean we need to transpose the result
    # to align with (r, z) axes conventions in plotting.
    return Br_total.T, Bz_total.T, r_grid, z_grid

def analyze_field_ripple(b_magnitude_map):
    """Calculates on-axis field ripple from a B-field magnitude map (in r, z)."""
    if b_magnitude_map is None or b_magnitude_map.shape[0] == 0 or b_magnitude_map.shape[1] == 0:
        return 0
    
    # On-axis field is the first row of the map (index 0 of the 'r' dimension)
    on_axis_field = b_magnitude_map[0, :]
    if on_axis_field.size == 0: return 0
    
    b_max, b_min = np.max(on_axis_field), np.min(on_axis_field)
    
    return ((b_max - b_min) / (b_max + b_min)) * 100 if (b_max + b_min) > 0 else 0