# physics.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk

# --- Constants ---
e_charge = 1.602e-19
m_D = 3.344e-27

# ==============================================================================
# SECTION 1: REALISTIC PHYSICS FOR FEASIBILITY ANALYSIS
# ==============================================================================
def fusion_reactivity_dd(temp_keV):
    if temp_keV <= 0: return 0.0
    B_G = 31.397
    c = [1.5136e-2, 2.5247e-3, 3.0313e-3, -1.1793e-4, 8.7183e-6, -2.4839e-7, 2.7051e-9]
    theta = temp_keV / (1 - (temp_keV * (c[2] + temp_keV * (c[4] + temp_keV * c[6]))) / (1 + temp_keV * (c[1] + temp_keV * (c[3] + temp_keV * c[5]))))
    zeta = (B_G**2 / (4 * theta))**(1/3)
    return 5.68e-18 * (zeta**2 * np.exp(-3 * zeta)) / (temp_keV**(2/3))

def calculate_bremsstrahlung_loss(n_e_m3, T_e_keV, Z_eff):
    T_e_eV = T_e_keV * 1000
    power_density_W_m3 = 5.35e-37 * (n_e_m3**2) * Z_eff * np.sqrt(T_e_eV)
    return power_density_W_m3 / 1e6

def get_required_heating_power(params):
    n_i, T_i_keV, tau_E, volume, Z_eff = params['ion_density_m3'], params['ion_temperature_keV'], params['confinement_time_s'], params['plasma_volume_m3'], params['Z_eff']
    W_th = 1.5 * (2 * n_i) * (T_i_keV * 1000 * e_charge) * volume
    P_transport_MW = (W_th / tau_E) / 1e6 if tau_E > 0 else float('inf')
    P_rad_MW = calculate_bremsstrahlung_loss(n_i, T_i_keV, Z_eff) * volume
    return P_transport_MW + P_rad_MW

def calculate_power_balance(params):
    n_i, T_i_keV, P_heating_MW, volume = params['ion_density_m3'], params['ion_temperature_keV'], params['P_heating_MW'], params['plasma_volume_m3']
    reactivity = fusion_reactivity_dd(T_i_keV)
    E_dd_avg_joules = 3.6 * 1e6 * e_charge
    P_fusion_MW = (0.5 * (n_i**2) * reactivity * E_dd_avg_joules * volume) / 1e6
    P_rad_MW = calculate_bremsstrahlung_loss(n_i, T_i_keV, params['Z_eff']) * volume
    W_th = 3 * n_i * (T_i_keV * 1000 * e_charge) * volume
    P_transport_MW = (W_th / params['confinement_time_s']) / 1e6 if params['confinement_time_s'] > 0 else float('inf')
    Q_value = P_fusion_MW / P_heating_MW if P_heating_MW > 1e-9 else float('inf')
    triple_product = n_i * params['confinement_time_s'] * T_i_keV / 1e20
    return {"P_fusion_MW": P_fusion_MW, "P_rad_MW": P_rad_MW, "P_transport_MW": P_transport_MW, "P_heating_MW": P_heating_MW, "Q_value": Q_value, "Triple_Product": triple_product}

# ==============================================================================
# SECTION 2: SIMPLIFIED PHYSICS FOR VISUALIZER
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
    plt.close("all") # Close previous plot windows
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
    """Saves a GIF and displays the animation in a new Toplevel window."""
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Particle Trajectory Animation")
    max_range = 0.2
    ax.set_xlim([-max_range, max_range]), ax.set_ylim([-max_range, max_range]), ax.set_zlim([-max_range, max_range])
    
    # Create lines for past path (solid) and predicted path (dashed)
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
                # Update past path
                line.set_data(sol.y[0, :frame+1], sol.y[1, :frame+1])
                line.set_3d_properties(sol.y[2, :frame+1])
                # Update predicted future path
                pred_line.set_data(sol.y[0, frame:], sol.y[1, frame:])
                pred_line.set_3d_properties(sol.y[2, frame:])
        return lines + predicted_lines
    
    num_frames = trajectories[0].y.shape[1]
    ani = animation.FuncAnimation(fig, update, frames=num_frames, init_func=init, blit=False, interval=30)
    
    # Save the animation as a GIF
    ani.save(filename, writer='pillow', fps=30)
    canvas.draw()
