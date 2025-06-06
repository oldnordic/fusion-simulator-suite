# physics.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk

# --- Constants ---
e_charge = 1.602e-19  # Electron charge in C
m_p = 1.672e-27       # Proton mass in kg
m_D = 3.344e-27       # Deuteron mass in kg
mu_0 = 4 * np.pi * 1e-7 # Vacuum permeability
k_boltzmann = 1.3806e-23 # Boltzmann constant

# ==============================================================================
# SECTION 1: 0D REACTOR PHYSICS MODELS
# ==============================================================================
def fusion_reactivity_dd(temp_keV):
    """
    Calculates the D-D fusion reactivity <σv> using the Bosch-Hale formulation.
    Input: temp_keV (float) - Ion temperature in keV
    Output: <σv> in m³/s
    """
    if temp_keV <= 0: return 0.0
    B_G = 31.397
    # Coefficients for the analytical fit
    c = [1.5136e-2, 2.5247e-3, 3.0313e-3, -1.1793e-4, 8.7183e-6, -2.4839e-7, 2.7051e-9]
    theta = temp_keV / (1 - (temp_keV * (c[2] + temp_keV * (c[4] + temp_keV * c[6]))) / (1 + temp_keV * (c[1] + temp_keV * (c[3] + temp_keV * c[5]))))
    zeta = (B_G**2 / (4 * theta))**(1/3)
    # The reactivity formula
    return 5.68e-18 * (zeta**2 * np.exp(-3 * zeta)) / (temp_keV**(2/3))

def calculate_bremsstrahlung_loss(n_e_m3, T_e_keV, Z_eff):
    """
    Calculates Bremsstrahlung radiation power density.
    Output: Power density in MW/m³
    """
    T_e_eV = T_e_keV * 1000
    # Standard formula for Bremsstrahlung power density in W/m³
    power_density_W_m3 = 5.35e-37 * (n_e_m3**2) * Z_eff * np.sqrt(T_e_eV)
    return power_density_W_m3 / 1e6  # Convert to MW/m³

def calculate_gyro_bohm_transport_loss(params):
    """
    Estimates transport loss based on gyro-Bohm scaling. This is a physics-based
    model for turbulent transport, replacing the simple tau_E assumption.
    The Geometry Factor G represents the confinement improvement of the specific
    magnetic geometry over a simple tokamak.
    """
    n_i = params['ion_density_m3']
    T_i_keV = params['ion_temperature_keV']
    B = params['magnetic_field_T']
    V = params['plasma_volume_m3']
    G = params['transport_geometry_factor_G']
    
    # Estimate minor radius 'a' from volume, assuming toroidal aspect ratio of 3 (R=3a)
    # V = 2 * pi^2 * R * a^2 = 2 * pi^2 * (3a) * a^2 = 6 * pi^2 * a^3
    a = (V / (6 * np.pi**2))**(1/3)
    if a == 0: return float('inf')

    # Gyro-Bohm diffusion coefficient scaling: D_gB ~ (rho_s / a) * (c_s * rho_s)
    # This leads to P_loss ~ n * T^2.5 / (B^2 * a^2)
    # A simplified but physically motivated scaling for power loss density
    # The coefficient is empirical and tuned to give reasonable tau_E for known machines
    p_transport_density_MW_m3 = 8e-7 * (n_i/1e20)**1.0 * (T_i_keV)**2.5 / (B**2 * a**2)

    # The Geometry Factor G improves confinement, reducing transport loss
    if G <= 0: return float('inf')
    return (p_transport_density_MW_m3 / G) * V

def calculate_plasma_beta(params):
    """
    Calculates plasma beta (β), the ratio of plasma pressure to magnetic pressure.
    A key indicator for MHD stability.
    """
    n_i = params['ion_density_m3']
    T_i_keV = params['ion_temperature_keV']
    B = params['magnetic_field_T']

    # Assuming T_e ~ T_i for this 0D model
    T_joules = T_i_keV * 1000 * e_charge
    # Total plasma pressure (ions + electrons)
    plasma_pressure = 2 * n_i * k_boltzmann * (T_joules / e_charge / 1000)
    magnetic_pressure = B**2 / (2 * mu_0)
    
    if magnetic_pressure == 0: return float('inf')
    
    beta = plasma_pressure / magnetic_pressure
    return beta * 100 # Return as a percentage

def calculate_power_balance(params):
    """
    Main function to calculate the full power balance, stability, and performance metrics.
    """
    n_i = params['ion_density_m3']
    T_i_keV = params['ion_temperature_keV']
    V = params['plasma_volume_m3']
    A = params['plasma_surface_area_m2']
    Z_eff = params['Z_eff']
    alpha_n = params['density_profile_alpha']
    alpha_T = params['temp_profile_alpha']

    # --- Profile Effects ---
    # Approximates the effect of peaked profiles on fusion rate
    # For profiles like n(r)=n0*(1-r^2/a^2)^alpha, the fusion power is enhanced
    # This is a common approximation for 0D models.
    profile_factor = (1 + alpha_n + alpha_T) * (1 + alpha_n)**2 / (1 + 2*alpha_n + alpha_T)
    
    # --- Power Generation ---
    reactivity = fusion_reactivity_dd(T_i_keV)
    E_dd_avg_joules = 3.6e6 * e_charge  # Average energy per D-D reaction (MeV to Joules)
    P_fusion_MW = (0.5 * (n_i**2) * reactivity * E_dd_avg_joules * V * profile_factor) / 1e6
    
    # --- Power Losses ---
    P_rad_MW = calculate_bremsstrahlung_loss(n_i, T_i_keV, Z_eff) * V
    P_transport_MW = calculate_gyro_bohm_transport_loss(params)

    # --- Derived Performance Metrics ---
    P_loss_total_MW = P_rad_MW + P_transport_MW
    # Required heating power is the total loss (assuming no alpha heating for this definition)
    P_heating_MW = P_loss_total_MW
    Q_value = P_fusion_MW / P_heating_MW if P_heating_MW > 1e-9 else float('inf')

    # Thermal energy content of the plasma
    W_th_joules = 1.5 * (2 * n_i) * (T_i_keV * 1000 * e_charge) * V
    confinement_time_s = W_th_joules / (P_loss_total_MW * 1e6) if P_loss_total_MW > 1e-9 else float('inf')
    
    triple_product = n_i * confinement_time_s * T_i_keV
    triple_product_1e20 = triple_product / 1e20 # For easier display

    # --- Stability and Engineering ---
    plasma_beta_percent = calculate_plasma_beta(params)
    wall_load_MW_m2 = (P_transport_MW + P_rad_MW) / A if A > 0 else float('inf') # Heat flux on PFCs

    return {
        "P_fusion_MW": P_fusion_MW,
        "P_rad_MW": P_rad_MW,
        "P_transport_MW": P_transport_MW,
        "P_heating_MW": P_heating_MW,
        "Q_value": Q_value,
        "confinement_time_s": confinement_time_s,
        "Triple_Product_1e20": triple_product_1e20,
        "plasma_beta_percent": plasma_beta_percent,
        "wall_load_MW_m2": wall_load_MW_m2,
    }

# ==============================================================================
# SECTION 2: SIMPLIFIED PHYSICS FOR PARTICLE VISUALIZER
# This section remains unchanged as it's for illustrative purposes.
# ==============================================================================
def lorentz_force_vis(t, y, B_scale):
    pos, vel = y[:3], y[3:]
    B = np.array([0, 0, B_scale])
    acc = (e_charge / m_D) * np.cross(vel, B)
    return np.concatenate((vel, acc))

def simulate_particle_for_vis(params, coil_points):
    pos0 = np.random.uniform(-0.04, 0.04, 3)
    speed = params['max_velocity'] * np.random.rand()
    direction = np.random.randn(3)
    direction /= np.linalg.norm(direction)
    y0 = np.concatenate((pos0, direction * speed))
    t_span = (0, params['sim_time'])
    t_eval = np.linspace(*t_span, 200)
    sol = solve_ivp(lorentz_force_vis, t_span, y0, t_eval=t_eval, args=(params['B_strength'],), rtol=1e-5)
    return sol

def calculate_simple_q_value(trajectories):
    initial_ke, final_ke = 0, 0
    for sol in trajectories:
        initial_ke += 0.5 * m_D * np.sum(sol.y[3:, 0]**2)
        final_ke += 0.5 * m_D * np.sum(sol.y[3:, -1]**2)
    return initial_ke, final_ke, final_ke / initial_ke if initial_ke > 0 else 0

def plot_trajectories_for_vis(trajectories, coil_points, filename, info_text):
    plt.close("all")
    fig = plt.figure(num="Figure 1: Trajectories")
    ax = fig.add_subplot(111, projection='3d')
    for sol in trajectories: ax.plot(sol.y[0], sol.y[1], sol.y[2])
    ax.set_title("Trajectories in Magnetic Field")
    ax.set_xlabel("X"), ax.set_ylabel("Y"), ax.set_zlabel("Z")
    fig.text(0.5, 0.05, info_text, ha='center', fontsize=10)
    plt.savefig(filename); plt.show(block=False)

def plot_energy_vs_time_for_vis(trajectories, filename):
    plt.close("all")
    fig = plt.figure(num="Figure 2: Energy")
    ax = fig.add_subplot(111)
    for i, sol in enumerate(trajectories):
        energy = 0.5 * m_D * np.sum(sol.y[3:]**2, axis=0)
        ax.plot(np.arange(len(sol.t)), energy, label=f"Particle {i}")
    ax.set_title("Kinetic Energy vs Time")
    ax.set_xlabel("Timestep"); ax.set_ylabel("Kinetic Energy (J)")
    ax.legend(); plt.savefig(filename); plt.show(block=False)

def animate_trajectories(trajectories, filename, root_window):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Particle Trajectory Animation")
    max_range = 0.2
    ax.set_xlim([-max_range, max_range]), ax.set_ylim([-max_range, max_range]), ax.set_zlim([-max_range, max_range])
    
    lines = [ax.plot([], [], [], lw=2)[0] for _ in trajectories]
    predicted_lines = [ax.plot([], [], [], lw=1, ls='--', color='gray', alpha=0.7)[0] for _ in trajectories]

    canvas = FigureCanvasTkAgg(fig, master=root_window)
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def init():
        for line, pred_line in zip(lines, predicted_lines):
            line.set_data([], []); line.set_3d_properties([])
            pred_line.set_data([], []); pred_line.set_3d_properties([])
        return lines + predicted_lines

    def update(frame):
        for i, (sol, line, pred_line) in enumerate(zip(trajectories, lines, predicted_lines)):
            if frame < sol.y.shape[1]:
                line.set_data(sol.y[0, :frame+1], sol.y[1, :frame+1])
                line.set_3d_properties(sol.y[2, :frame+1])
                pred_line.set_data(sol.y[0, frame:], sol.y[1, frame:])
                pred_line.set_3d_properties(sol.y[2, frame:])
        return lines + predicted_lines
    
    num_frames = trajectories[0].y.shape[1]
    ani = animation.FuncAnimation(fig, update, frames=num_frames, init_func=init, blit=False, interval=30)
    
    ani.save(filename, writer='pillow', fps=30)
    canvas.draw()
