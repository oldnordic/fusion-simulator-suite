# coil_generator.py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- New Ovoid Fibonacci Spiral Coil Generator ---
def define_ovoid_surface_radius(z_coord, max_radius_coil_assembly, half_height_coil_assembly):
    """
    Defines the radius of a symmetrical ovoid/spheroid at a given height z_coord.
    """
    if abs(z_coord) > half_height_coil_assembly + 1e-9:
        return 0.0
    
    if half_height_coil_assembly == 0:
        return max_radius_coil_assembly if abs(z_coord) < 1e-9 else 0.0

    normalized_z_squared = (z_coord / half_height_coil_assembly)**2
    # Ensure argument of sqrt is non-negative
    sqrt_arg = max(0.0, 1.0 - normalized_z_squared)
    
    radius_at_z = max_radius_coil_assembly * np.sqrt(sqrt_arg)
    return radius_at_z

def generate_fibonacci_spiral_coils(
    max_radius_coil_assembly, total_height_coil_assembly,
    num_spirals_set1, pitch_set1, num_turns_set1,
    num_spirals_set2, pitch_set2, num_turns_set2,
    points_per_coil=200
):
    """
    Generates 3D coordinates for two sets of intersecting spiral coil paths
    on the surface of a symmetrical ovoid.
    """
    all_coil_paths_data = []
    half_height = total_height_coil_assembly / 2.0

    def generate_single_set(num_spirals, pitch, num_turns, angular_offset_base=0.0):
        if not (num_spirals > 0 and pitch != 0 and num_turns > 0):
            return
            
        for i in range(num_spirals):
            coil_points = []
            initial_angular_offset_rad = angular_offset_base + i * (2 * np.pi / num_spirals)
            
            # Start from the south pole
            start_z = -half_height
            if pitch < 0: # If pitch is negative, the spiral naturally goes down, so we start at the top
                start_z = half_height
            
            # Generate u values that correspond to the full path length
            u_values = np.linspace(0, 2 * np.pi * num_turns, points_per_coil)
            
            for u_param in u_values:
                # Parametric definition of z based on pitch and turns
                z = start_z + (pitch / (2 * np.pi)) * u_param
                z = np.clip(z, -half_height, half_height)

                current_surface_r = define_ovoid_surface_radius(z, max_radius_coil_assembly, half_height)
                
                # --- CORRECTED: Refined pole handling to avoid duplicate points ---
                if current_surface_r < 1e-9:
                    # This point is at a pole. Add the pole point only if it's the start or end.
                    pole_z = np.sign(z) * half_height
                    is_start_or_end = (u_param == u_values[0] or u_param == u_values[-1])
                    if not coil_points or (is_start_or_end and not np.allclose(coil_points[-1], [0.0, 0.0, pole_z])):
                        coil_points.append([0.0, 0.0, pole_z])
                    continue

                angle_xy = initial_angular_offset_rad + u_param
                x = current_surface_r * np.cos(angle_xy)
                y = current_surface_r * np.sin(angle_xy)
                
                # Avoid adding duplicate points if calculation stalls
                if coil_points and np.allclose(coil_points[-1], [x, y, z]):
                    continue
                coil_points.append([x, y, z])
            
            if coil_points:
                all_coil_paths_data.append(np.array(coil_points))

    # Generate Set 1 Spirals (e.g., right-handed)
    generate_single_set(num_spirals_set1, pitch_set1, num_turns_set1)
    
    # Generate Set 2 Spirals (e.g., left-handed) with a slight phase shift for better coverage
    offset = (np.pi / num_spirals_set2) if num_spirals_set1 == num_spirals_set2 else 0
    generate_single_set(num_spirals_set2, pitch_set2, num_turns_set2, angular_offset_base=offset)
            
    return all_coil_paths_data

def plot_fibonacci_coils_with_chamber(
    coil_paths_list,
    coil_assembly_max_radius,
    coil_assembly_total_height,
    plasma_chamber_radius_fraction=0.5,
    title="Fibonacci Spiral Coils (Ovoid) with Reactor Chamber",
    ax=None,
    num_s1_for_coloring=0
):
    """
    Plots the generated list of coil paths and a central chamber on a given or new 3D axis.
    """
    if ax is None:
        fig = plt.figure(figsize=(10, 12))
        ax = fig.add_subplot(111, projection='3d')
        show_plot = True
    else:
        show_plot = False

    for i, path_array in enumerate(coil_paths_list):
        if isinstance(path_array, np.ndarray) and path_array.ndim == 2 and path_array.shape[0] > 1 and path_array.shape[1] == 3:
            color = 'blue' if i < num_s1_for_coloring else 'red'
            ax.plot(path_array[:, 0], path_array[:, 1], path_array[:, 2], color=color, linewidth=1.2)
    
    # Add a sphere in the middle to represent the "glowing reactor chamber"
    plasma_chamber_radius = coil_assembly_max_radius * plasma_chamber_radius_fraction
    u_sphere = np.linspace(0, 2 * np.pi, 50)
    v_sphere = np.linspace(0, np.pi, 50)
    x_sphere = plasma_chamber_radius * np.outer(np.cos(u_sphere), np.sin(v_sphere))
    y_sphere = plasma_chamber_radius * np.outer(np.sin(u_sphere), np.sin(v_sphere))
    z_sphere = plasma_chamber_radius * np.outer(np.ones(np.size(u_sphere)), np.cos(v_sphere))
    ax.plot_surface(x_sphere, y_sphere, z_sphere, color='orange', alpha=0.5, rstride=4, cstride=4, linewidth=0)

    ax.set_xlabel('X axis (m)')
    ax.set_ylabel('Y axis (m)')
    ax.set_zlabel('Z axis (m)')
    ax.set_title(title)
    
    ax.set_xlim([-coil_assembly_max_radius * 1.1, coil_assembly_max_radius * 1.1])
    ax.set_ylim([-coil_assembly_max_radius * 1.1, coil_assembly_max_radius * 1.1])
    ax.set_zlim([-coil_assembly_total_height / 2 * 1.1, coil_assembly_total_height / 2 * 1.1])
    
    ax.set_box_aspect((
        coil_assembly_max_radius * 2.2, 
        coil_assembly_max_radius * 2.2, 
        coil_assembly_total_height * 1.1
    ))

    if show_plot:
        plt.show(block=True)

if __name__ == '__main__':
    print("--- Demonstrating New Ovoid Fibonacci Spiral Coil Generator ---")
    ovoid_max_radius = 0.7
    ovoid_total_height = 2.0
    num_s1 = 7
    total_turns_s1 = 5.0
    pitch_s1 = ovoid_total_height / total_turns_s1
    num_s2 = 7
    total_turns_s2 = 5.0
    pitch_s2 = -ovoid_total_height / total_turns_s2 # Negative for opposite winding direction

    print(f"\nGenerating Ovoid Fibonacci Coils:")
    print(f"  Assembly: Max Radius={ovoid_max_radius:.2f}m, Total Height={ovoid_total_height:.2f}m")
    print(f"  Set 1: {num_s1} spirals, {total_turns_s1:.1f} turns each, Pitch={pitch_s1:.2f}")
    print(f"  Set 2: {num_s2} spirals, {total_turns_s2:.1f} turns each, Pitch={pitch_s2:.2f}")

    fibonacci_coil_paths = generate_fibonacci_spiral_coils(
        ovoid_max_radius, ovoid_total_height,
        num_s1, pitch_s1, total_turns_s1,
        num_s2, pitch_s2, total_turns_s2,
        points_per_coil=300 
    )

    if fibonacci_coil_paths:
        print(f"\nGenerated {len(fibonacci_coil_paths)} coil paths for the ovoid structure.")
        plot_fibonacci_coils_with_chamber(
            fibonacci_coil_paths,
            ovoid_max_radius,
            ovoid_total_height,
            plasma_chamber_radius_fraction=0.4,
            title="Ovoid Fibonacci Coils & Reactor Chamber",
            num_s1_for_coloring=num_s1
        )
    else:
        print("Ovoid Fibonacci coil generation failed or produced no paths.")