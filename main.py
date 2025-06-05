# main.py
import os
import time
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from multiprocessing import Pool, cpu_count
import csv
import numpy as np
import matplotlib.pyplot as plt

from physics import (
    calculate_power_balance, get_required_heating_power,
    simulate_particle_for_vis, calculate_simple_q_value,
    plot_trajectories_for_vis, plot_energy_vs_time_for_vis,
    animate_trajectories, e_charge, m_D
)
from report import generate_html_report, generate_pdf_report
from field import get_coil_points

# This helper function must be at the top level for multiprocessing to work
def run_particle_sim_worker(args):
    """Helper function to run a single particle simulation for the pool."""
    vis_params, coil_points = args
    return simulate_particle_for_vis(vis_params, coil_points)

class FusionSimulatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fusion Simulation Suite")
        self.root.geometry("700x800")

        # --- Data & Parameters ---
        self.output_dir = "sim_output_" + time.strftime("%Y%m%d_%H%M%S")
        os.makedirs(self.output_dir, exist_ok=True)
        self.initialize_presets()

        # --- Create Tabs ---
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(pady=10, padx=10, fill="both", expand=True)
        self.tab1 = ttk.Frame(self.notebook)
        self.tab2 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab1, text='Reactor Feasibility Analysis')
        self.notebook.add(self.tab2, text='Particle Visualizer')

        self.create_feasibility_tab()
        self.create_visualizer_tab()

        # --- Add clean exit handler ---
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def on_closing(self):
        """Handle the event of closing the window."""
        if messagebox.askokcancel("Quit", "Do you want to quit the application?"):
            plt.close('all')  # Close all matplotlib figures
            self.root.destroy() # Destroy the main tkinter window

    def initialize_presets(self):
        """Defines the preset parameters for different reactor concepts."""
        self.presets = {
            "Custom": {},
            "ITER (Projected)": {
                'ion_density_m3': 1e20, 'ion_temperature_keV': 20.0, 'confinement_time_s': 3.7,
                'magnetic_field_T': 5.3, 'plasma_volume_m3': 840.0, 'Z_eff': 1.8
            },
            "SPARC (Projected)": {
                'ion_density_m3': 4e20, 'ion_temperature_keV': 25.0, 'confinement_time_s': 1.0,
                'magnetic_field_T': 12.2, 'plasma_volume_m3': 10.0, 'Z_eff': 1.5
            },
            "Fibonacci Whitepaper Claim": {
                'ion_density_m3': 1e20, 'ion_temperature_keV': 0.01, 'confinement_time_s': 5.0,
                'magnetic_field_T': 1.0, 'plasma_volume_m3': 2.2, 'Z_eff': 1.0
            }
        }
        self.phys_params = self.presets["ITER (Projected)"].copy()

    def create_feasibility_tab(self):
        main_frame = ttk.Frame(self.tab1, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        preset_frame = ttk.Frame(main_frame)
        preset_frame.pack(fill=tk.X, expand=True, pady=5)
        ttk.Label(preset_frame, text="Load Preset:").pack(side=tk.LEFT)
        self.preset_var = tk.StringVar(value="ITER (Projected)")
        preset_menu = ttk.Combobox(preset_frame, textvariable=self.preset_var, values=list(self.presets.keys()))
        preset_menu.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        preset_menu.bind("<<ComboboxSelected>>", self.load_preset)

        input_frame = ttk.LabelFrame(main_frame, text="Plasma & Machine Parameters", padding="10")
        input_frame.pack(fill=tk.X, expand=True)
        self.phys_entries = {}
        param_labels = {
            'ion_density_m3': "Ion Density (n) [m⁻³]:", 'ion_temperature_keV': "Ion Temperature (T) [keV]:",
            'confinement_time_s': "Energy Confinement Time (τE) [s]:", 'magnetic_field_T': "Magnetic Field (B) [T]:",
            'plasma_volume_m3': "Plasma Volume (V) [m³]:", 'Z_eff': "Effective Charge (Z_eff):"
        }
        for i, (key, label) in enumerate(param_labels.items()):
            ttk.Label(input_frame, text=label).grid(row=i, column=0, sticky=tk.W, pady=2)
            entry = ttk.Entry(input_frame, width=20)
            entry.grid(row=i, column=1, sticky=tk.EW, pady=2)
            self.phys_entries[key] = entry
        input_frame.columnconfigure(1, weight=1)

        ttk.Button(main_frame, text="Calculate Power Balance", command=self.run_physics_simulation).pack(fill=tk.X, pady=10)
        results_frame = ttk.LabelFrame(main_frame, text="Power Balance & Q-Value Results", padding="10")
        results_frame.pack(fill=tk.X, expand=True, pady=5)
        self.phys_result_labels = {}
        result_keys = [
            ("P_fusion_MW", "Fusion Power [MW]:"), ("P_rad_MW", "Radiation Loss [MW]:"),
            ("P_transport_MW", "Transport Loss [MW]:"), ("P_heating_MW", "Required Heating Power [MW]:"),
            ("Q_value", "Q-Value (P_fusion / P_heating):"), ("Triple_Product", "Lawson Criterion (nτT):")
        ]
        for i, (key, label) in enumerate(result_keys):
            ttk.Label(results_frame, text=label).grid(row=i, column=0, sticky=tk.W)
            lbl = ttk.Label(results_frame, text="N/A", font=("Courier", 10))
            lbl.grid(row=i, column=1, sticky=tk.W)
            self.phys_result_labels[key] = lbl

        self.phys_export_button = ttk.Menubutton(main_frame, text="Export Physics Report", state=tk.DISABLED)
        export_menu = tk.Menu(self.phys_export_button, tearoff=0)
        export_menu.add_command(label="Export as PDF", command=lambda: self.export_phys_report('pdf'))
        export_menu.add_command(label="Export as HTML", command=lambda: self.export_phys_report('html'))
        export_menu.add_command(label="Export as CSV", command=lambda: self.export_phys_report('csv'))
        self.phys_export_button["menu"] = export_menu
        self.phys_export_button.pack(fill=tk.X, pady=10)
        
        self.load_preset()

    def create_visualizer_tab(self):
        self.vis_params = {'num_particles': 10, 'num_coils': 7, 'coil_radius': 0.2, 'coil_pitch': 1.0, 'B_strength': 5.0, 'sim_time': 1e-05, 'max_velocity': 1e6}
        self.vis_trajectories = []
        
        frame = ttk.Frame(self.tab2, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)

        params_frame = ttk.LabelFrame(frame, text="Visualizer Parameters", padding="10")
        params_frame.pack(fill=tk.X)
        self.vis_entries = {}
        vis_labels = {'num_particles': "Number of particles", 'num_coils': "Number of coils", 'coil_radius': "Coil radius (m)", 'coil_pitch': "Coil pitch (m)", 'B_strength': "Magnetic field strength (T)", 'sim_time': "Simulation time (s)", 'max_velocity': "Max initial velocity (m/s)"}
        for i, (key, label) in enumerate(vis_labels.items()):
            ttk.Label(params_frame, text=label).grid(row=i, column=0, sticky=tk.W, pady=2)
            entry = ttk.Entry(params_frame)
            entry.insert(0, str(self.vis_params[key]))
            entry.grid(row=i, column=1, sticky="ew", pady=2)
            self.vis_entries[key] = entry
        
        button_frame = ttk.Frame(frame, padding="5")
        button_frame.pack(fill=tk.X)
        ttk.Button(button_frame, text="Run Simulation", command=self.run_visualization_simulation).pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        ttk.Button(button_frame, text="Import from Feasibility", command=self.import_from_feasibility).pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        
        plot_button_frame = ttk.Frame(frame, padding="5")
        plot_button_frame.pack(fill=tk.X)
        self.vis_plot_energy_button = ttk.Button(plot_button_frame, text="Plot Energy-Time", command=self.plot_vis_energy, state=tk.DISABLED)
        self.vis_plot_energy_button.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        self.vis_animate_button = ttk.Button(plot_button_frame, text="Animate Trajectories", command=self.animate_vis, state=tk.DISABLED)
        self.vis_animate_button.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)

        progress_frame = ttk.LabelFrame(frame, text="Simulation Progress", padding="10")
        progress_frame.pack(fill=tk.X, pady=10)
        self.progress_bar = ttk.Progressbar(progress_frame, orient='horizontal', mode='determinate')
        self.progress_bar.pack(fill=tk.X, expand=True, pady=2)
        self.time_label = ttk.Label(progress_frame, text="Time Remaining: N/A")
        self.time_label.pack(pady=2)
        self.vis_status_label = ttk.Label(frame, text="Ready.")
        self.vis_status_label.pack(pady=5)

    def load_preset(self, event=None):
        preset_name = self.preset_var.get()
        if preset_name == "Custom": return
        self.phys_params = self.presets[preset_name].copy()
        for key, entry in self.phys_entries.items():
            entry.delete(0, tk.END)
            entry.insert(0, f"{self.phys_params[key]:.1e}" if 'density' in key else str(self.phys_params[key]))

    def import_from_feasibility(self):
        """Copies relevant parameters from the Feasibility tab to the Visualizer tab."""
        try:
            # Update params dict from Feasibility entry fields first
            for key, entry in self.phys_entries.items():
                self.phys_params[key] = float(entry.get())
            
            # 1. Copy Magnetic Field
            b_field = self.phys_params.get('magnetic_field_T', 5.0)
            self.vis_entries['B_strength'].delete(0, tk.END)
            self.vis_entries['B_strength'].insert(0, str(b_field))

            # 2. Calculate and copy Max Velocity from Temperature
            temp_keV = self.phys_params.get('ion_temperature_keV', 15.0)
            temp_joules = temp_keV * 1000 * e_charge
            # v_rms = sqrt(3*k*T/m), so use a multiple of this for max velocity
            v_rms = np.sqrt(3 * temp_joules / m_D)
            max_velocity = 2 * v_rms # Set max velocity to twice the RMS speed
            self.vis_entries['max_velocity'].delete(0, tk.END)
            self.vis_entries['max_velocity'].insert(0, f"{max_velocity:.2e}")
            
            messagebox.showinfo("Import Complete", "Magnetic Field and Max Velocity have been updated from the Feasibility tab.")

        except (ValueError, KeyError) as e:
            messagebox.showerror("Import Failed", f"Could not import parameters. Please ensure Feasibility values are valid numbers.\nError: {e}")


    def run_physics_simulation(self):
        try:
            for key, entry in self.phys_entries.items(): self.phys_params[key] = float(entry.get())
        except ValueError: messagebox.showerror("Invalid Input", "Please ensure all parameters are valid numbers."); return
        self.preset_var.set("Custom")
        self.phys_params['P_heating_MW'] = get_required_heating_power(self.phys_params)
        self.phys_results = calculate_power_balance(self.phys_params)
        for key, label in self.phys_result_labels.items():
            if key in self.phys_results: label.config(text=f"{self.phys_results[key]:.4e}")
        self.phys_export_button.config(state=tk.NORMAL)

    def export_phys_report(self, format_type):
        filename = filedialog.asksaveasfilename(initialdir=self.output_dir, title=f"Save {format_type.upper()} Report", defaultextension=f".{format_type}")
        if not filename: return
        try:
            if format_type == 'pdf': generate_pdf_report(filename, self.phys_params, self.phys_results)
            elif format_type == 'html': generate_html_report(filename, self.phys_params, self.phys_results)
            elif format_type == 'csv':
                with open(filename, 'w', newline='') as f:
                    writer = csv.writer(f); writer.writerow(['Parameter', 'Value']); writer.writerows(self.phys_params.items()); writer.writerow([]); writer.writerow(['Result', 'Value']); writer.writerows(self.phys_results.items())
            messagebox.showinfo("Export Successful", f"Report saved to:\n{filename}")
        except Exception as e: messagebox.showerror("Export Failed", f"An error occurred: {e}")

    def run_visualization_simulation(self):
        try:
            for key, entry in self.vis_entries.items(): self.vis_params[key] = float(entry.get())
        except ValueError: messagebox.showerror("Invalid Input", "Please ensure all visualizer parameters are valid numbers."); return
        
        num_particles = int(self.vis_params['num_particles'])
        self.vis_status_label.config(text=f"Starting simulation for {num_particles} particles...")
        self.progress_bar['value'] = 0
        self.progress_bar['maximum'] = num_particles
        self.time_label.config(text="Time Remaining: Estimating...")
        self.root.update()

        vis_params_full = {**self.vis_params, 'geometry': 'fibonacci'}
        coil_points = get_coil_points(vis_params_full)
        tasks = [(vis_params_full, coil_points) for _ in range(num_particles)]
        
        self.vis_trajectories = []
        completed_tasks = 0
        start_time = time.time()

        with Pool(processes=cpu_count()) as pool:
            for result in pool.imap_unordered(run_particle_sim_worker, tasks):
                self.vis_trajectories.append(result)
                completed_tasks += 1
                
                self.progress_bar['value'] = completed_tasks
                elapsed_time = time.time() - start_time
                avg_time_per_task = elapsed_time / completed_tasks
                remaining_tasks = num_particles - completed_tasks
                time_remaining = avg_time_per_task * remaining_tasks
                self.time_label.config(text=f"Time Remaining: {time_remaining:.1f}s")
                self.root.update_idletasks()
        
        initial_ke, final_ke, vis_q_value = calculate_simple_q_value(self.vis_trajectories)
        self.vis_info_text = f"Energy in = {initial_ke:.2e} J, Out = {final_ke:.2e} J, Q = {vis_q_value:.2f}"
        self.vis_status_label.config(text=f"Simulation complete. Q-value: {vis_q_value:.2f}")
        self.vis_plot_energy_button.config(state=tk.NORMAL)
        self.vis_animate_button.config(state=tk.NORMAL)
        plot_trajectories_for_vis(self.vis_trajectories, coil_points, os.path.join(self.output_dir, "vis_trajectories.png"), self.vis_info_text)

    def plot_vis_energy(self):
        plot_energy_vs_time_for_vis(self.vis_trajectories, os.path.join(self.output_dir, "vis_energy_vs_time.png"))

    def animate_vis(self):
        animation_window = tk.Toplevel(self.root)
        animation_window.title("Particle Animation")
        animation_window.geometry("800x600")
        self.vis_status_label.config(text="Generating animation...")
        self.root.update()
        
        gif_path = os.path.join(self.output_dir, f"vis_animation_{time.strftime('%H%M%S')}.gif")
        animate_trajectories(self.vis_trajectories, gif_path, animation_window)
        messagebox.showinfo("Animation Saved", f"Animation saved to:\n{gif_path}")
        self.vis_status_label.config(text=f"Simulation complete.")

if __name__ == '__main__':
    root = tk.Tk()
    app = FusionSimulatorApp(root)
    root.mainloop()
