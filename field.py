# field.py
import numpy as np

def get_coil_points(params):
    """Generates 3D coil point coordinates for visualization."""
    geometry = params.get("geometry", "fibonacci")
    num_coils = int(params["num_coils"])
    coil_radius = params["coil_radius"]
    coil_pitch = params["coil_pitch"]
    points = []
    if geometry == "fibonacci":
        golden_angle = np.pi * (3. - np.sqrt(5.))
        z_vals = np.linspace(-coil_pitch / 2., coil_pitch / 2., num_coils)
        for i in range(num_coils):
            z = z_vals[i]
            radius = np.sqrt(max(coil_radius**2 - z**2, 0))
            theta = golden_angle * i
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            points.append((x, y, z))
    else: # Default to a simple cylinder if not fibonacci
        for i in range(num_coils):
            theta = 2 * np.pi * i / num_coils
            x = coil_radius * np.cos(theta)
            y = coil_radius * np.sin(theta)
            z = coil_pitch * (i - num_coils / 2)
            points.append((x, y, z))
            
    return np.array(points)
