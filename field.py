# field.py
import numpy as np

try:
    from coil_generator import generate_fibonacci_spiral_coils
    FIBONACCI_OVOID_GENERATOR_AVAILABLE = True
except ImportError:
    print("Warning: coil_generator.py not found or function could not be imported.")
    print("         The 'ovoid_fibonacci_reactor' geometry will not be available.")
    FIBONACCI_OVOID_GENERATOR_AVAILABLE = False

def get_coil_points(params):
    """
    Generates 3D coil point coordinates based on a geometry specified in params.
    Supports "fibonacci" (spherical), "cylinder", and "ovoid_fibonacci_reactor".
    """
    geometry = params.get("geometry", "fibonacci")
    
    if geometry == "ovoid_fibonacci_reactor":
        if not FIBONACCI_OVOID_GENERATOR_AVAILABLE:
            print("Error: 'ovoid_fibonacci_reactor' selected, but generator is unavailable.")
            return np.empty((0,3))
        
        try:
            # Consolidate required parameters for the generator
            gen_params = {
                "max_radius_coil_assembly": params["max_radius_coil_assembly"],
                "total_height_coil_assembly": params["total_height_coil_assembly"],
                "num_spirals_set1": int(params["num_spirals_set1"]),
                "pitch_set1": params["pitch_set1"],
                "num_turns_set1": params["num_turns_set1"],
                "num_spirals_set2": int(params["num_spirals_set2"]),
                "pitch_set2": params["pitch_set2"],
                "num_turns_set2": params["num_turns_set2"],
                "points_per_coil": int(params.get("points_per_coil", 200))
            }
            coil_paths_list = generate_fibonacci_spiral_coils(**gen_params)
            
            if coil_paths_list:
                # Return all points from all coil paths concatenated
                return np.vstack(coil_paths_list)
            else:
                print("Warning: 'ovoid_fibonacci_reactor' generation returned no paths.")
                return np.empty((0,3))
                
        except KeyError as e:
            print(f"Error: Missing required parameter for 'ovoid_fibonacci_reactor': {e}")
            return np.empty((0,3))
        except Exception as e:
            print(f"An unexpected error occurred during ovoid generation in field.py: {e}")
            return np.empty((0,3))

    # --- Fallback to simpler geometries if not ovoid ---
    points = []
    num_coils = int(params.get("num_coils", 10))
    if num_coils == 0:
        return np.empty((0,3))

    if geometry == "fibonacci":
        # Distributes points on a sphere using a golden angle spiral logic
        coil_radius = params.get("coil_radius", 1.0)
        coil_extent_z = params.get("coil_pitch", coil_radius) 
        golden_angle = np.pi * (3. - np.sqrt(5.))
        z_max_abs = min(coil_extent_z / 2.0, coil_radius)
        z_vals = np.linspace(-z_max_abs, z_max_abs, num_coils)
        
        for i in range(num_coils):
            z = z_vals[i]
            radius_at_z = np.sqrt(max(0, coil_radius**2 - z**2))
            theta = golden_angle * i
            x = radius_at_z * np.cos(theta)
            y = radius_at_z * np.sin(theta)
            points.append((x, y, z))

    else: # Default to a simple cylinder
        coil_radius = params.get("coil_radius", 1.0)
        coil_pitch_total = params.get("coil_pitch", 1.0)

        for i in range(num_coils):
            theta = 2 * np.pi * i / num_coils
            x = coil_radius * np.cos(theta)
            y = coil_radius * np.sin(theta)
            # Distribute z along the total height
            z_pos = coil_pitch_total * ((i / (num_coils - 1)) - 0.5) if num_coils > 1 else 0
            points.append((x, y, z_pos))
            
    return np.array(points)

if __name__ == '__main__':
    print("--- Testing field.py ---")

    print("\n1. Testing 'fibonacci' geometry (spherical point distribution):")
    params_fib_sphere = {"geometry": "fibonacci", "num_coils": 20, "coil_radius": 0.5, "coil_pitch": 0.8}
    fib_points = get_coil_points(params_fib_sphere)
    print(f"  Generated {fib_points.shape[0]} points." if fib_points.size > 0 else "  Generation failed.")

    print("\n2. Testing 'cylinder' geometry:")
    params_cylinder = {"geometry": "cylinder", "num_coils": 15, "coil_radius": 0.6, "coil_pitch": 1.5}
    cyl_points = get_coil_points(params_cylinder)
    print(f"  Generated {cyl_points.shape[0]} points." if cyl_points.size > 0 else "  Generation failed.")

    if FIBONACCI_OVOID_GENERATOR_AVAILABLE:
        print("\n3. Testing 'ovoid_fibonacci_reactor' geometry:")
        params_ovoid_reactor = {
            "geometry": "ovoid_fibonacci_reactor",
            "max_radius_coil_assembly": 0.7, "total_height_coil_assembly": 2.0,
            "num_spirals_set1": 7, "pitch_set1": 0.4, "num_turns_set1": 5.0,
            "num_spirals_set2": 7, "pitch_set2": -0.4, "num_turns_set2": 5.0,
            "points_per_coil": 50
        }
        reactor_coil_points = get_coil_points(params_ovoid_reactor)
        if reactor_coil_points.size > 0:
            print(f"  Generated {reactor_coil_points.shape[0]} total points for ovoid reactor.")
            print("  Visual validation available by running main.py GUI.")
        else:
            print("  Failed to generate points for 'ovoid_fibonacci_reactor'.")
    else:
        print("\nSkipping 'ovoid_fibonacci_reactor' test as generator is not available.")