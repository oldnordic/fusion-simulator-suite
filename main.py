# main.py
import os
import time
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from multiprocessing import Pool, cpu_count
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import csv
import shutil

# Import from your other modules
from physics import calculate_power_balance
from report import generate_html_report, generate_pdf_report
from coil_generator import generate_fibonacci_spiral_coils, plot_fibonacci_coils_with_chamber
from magnetics import calculate_b_field_off_axis

# Helper class for creating tooltips
class Tooltip:
    def __init__(self, widget, text):
        self.widget, self.text, self.tooltip = widget, text, None
        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)

    def show_tooltip(self, event=None):
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25
        self.tooltip = tk.Toplevel(self.widget)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.wm_geometry(f"+{x}+{y}")
        label = tk.Label(self.tooltip, text=self.text, background="#ffffe0", relief='solid', 
                         borderwidth=1, wraplength=200, justify='left', font=("tahoma", "8", "normal"))
        label.pack(ipadx=1)

    def hide_tooltip(self, event=None):
        if self.tooltip:
            self.tooltip.destroy()
        self.tooltip = None

def run_scan_worker(params):
    """Helper function for parallel processing in 2D sweep."""
    return calculate_power_balance(params)

class FusionSimulatorApp:
    def __init__(self, root_window):
        self.root = root_window
        self.root.title("Fusion Simulation Suite v7.2 - Ovoid Reactor Design")
        self.root.geometry("1250x900") 
        
        self.output_dir = "sim_output_" + time.strftime("%Y%m%d_%H%M%S")
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.initialize_presets()
        self.phys_results = {}
        self.sweep_results = None
        
        self.designed_coil_points = None 
        self.designed_ovoid_coil_paths_list = None 
        self.designed_ovoid_coil_params = None 
        self.coil_plot_image_path = None 
        self.current_coil_design_type = None 
        
        self.notebook = ttk.Notebook(root_window)
        self.notebook.pack(pady=10, padx=10, fill="both", expand=True)
        
        self.tab_feasibility = ttk.Frame(self.notebook)
        self.tab_coil_designer = ttk.Frame(self.notebook) 
        self.tab_magnetics = ttk.Frame(self.notebook)
        self.tab_sweep = ttk.Frame(self.notebook)
        
        self.notebook.add(self.tab_feasibility, text='Reactor Feasibility')
        self.notebook.add(self.tab_coil_designer, text='Ovoid Coil Designer') 
        self.notebook.add(self.tab_magnetics, text='Magnetic Field Analyzer')
        self.notebook.add(self.tab_sweep, text='2D Parametric Sweep')
        
        self.create_feasibility_tab()
        self.create_ovoid_coil_designer_tab() 
        self.create_magnetics_tab()
        self.create_sweep_tab()
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def on_closing(self):
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            plt.close('all') 
            if self.coil_plot_image_path and os.path.exists(self.coil_plot_image_path):
                try:
                    os.remove(self.coil_plot_image_path)
                except Exception as e:
                    print(f"Error removing temp coil plot '{self.coil_plot_image_path}': {e}")
            self.root.destroy()

    def initialize_presets(self):
        self.presets = {"Custom": {}}
        self.presets["ITER-like"] = { 
            'fuel_cycle': 'D-T', 'transport_model': 'ITER98', 'h_factor': 1.0, 
            'ion_density_m3': 1.0e20, 'ion_temperature_keV': 20.0, 'magnetic_field_T': 5.3,
            'plasma_volume_m3': 840.0, 'plasma_surface_area_m2': 1200.0, 
            'divertor_wetted_area_m2': 10.0, 'plasma_current_MA': 15.0, 'q_safety_factor': 3.0,
            'density_profile_alpha': 0.5, 'temp_profile_alpha': 1.2, 
            'sputtering_yield': 2e-4, 'Z_eff': 1.8
        }
        self.presets["Compact High-Field"] = {
            'fuel_cycle': 'D-T', 'transport_model': 'Gyro-Bohm', 
            'transport_geometry_factor_G': 8.0, 'ion_density_m3': 2.5e20, 
            'ion_temperature_keV': 25.0, 'magnetic_field_T': 12.0, 
            'plasma_volume_m3': 50.0, 'plasma_surface_area_m2': 300.0, 
            'divertor_wetted_area_m2': 5.0, 'plasma_current_MA': 10.0, 'q_safety_factor': 4.5,
            'density_profile_alpha': 0.8, 'temp_profile_alpha': 1.5, 
            'sputtering_yield': 1e-4, 'Z_eff': 1.6
        }
        self.presets["Ovoid Fibonacci Concept"] = { 
            'fuel_cycle': 'D-D', 'transport_model': 'Gyro-Bohm',
            'h_factor': 1.5, 'transport_geometry_factor_G': 10.0, 
            'ion_density_m3': 4.0e20, 'ion_temperature_keV': 25.0, 
            'magnetic_field_T': 8.0, 
            'plasma_volume_m3': 2.0, 
            'plasma_surface_area_m2': 10.0, 
            'divertor_wetted_area_m2': 1.0, 
            'plasma_current_MA': 5.0, 
            'q_safety_factor': 5.0,
            'density_profile_alpha': 1.0, 'temp_profile_alpha': 1.0,
            'sputtering_yield': 1e-5, 'Z_eff': 1.6
        }

    def create_feasibility_tab(self):
        main_frame = ttk.Frame(self.tab_feasibility, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        left_frame = ttk.Frame(main_frame)
        right_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        top_frame = ttk.Frame(left_frame)
        top_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(top_frame, text="Load Preset:").pack(side=tk.LEFT, padx=(0,5))
        self.preset_var = tk.StringVar(value="ITER-like") 
        preset_menu = ttk.Combobox(top_frame, textvariable=self.preset_var, 
                                   values=list(self.presets.keys()), state="readonly")
        preset_menu.pack(side=tk.LEFT, fill=tk.X, expand=True)
        preset_menu.bind("<<ComboboxSelected>>", self.load_preset)

        input_frame = ttk.LabelFrame(left_frame, text="Machine & Plasma Parameters", padding=10)
        input_frame.pack(fill=tk.X)
        self.phys_entries = {}
        param_info = { 
            'fuel_cycle': ("Fuel Cycle:", "D-T or D-D"), 
            'transport_model': ("Transport Model:", "Physics model for heat loss."), 
            'h_factor': ("H-Factor (for ITER98):", "Confinement enhancement over L-mode."), 
            'transport_geometry_factor_G': ("G-Factor (for Gyro-Bohm):", "Geometric factor for Gyro-Bohm transport."),
            'ion_density_m3': ("Core Density (n₀) [m⁻³]:", "Central ion density."), 
            'ion_temperature_keV': ("Core Temp (T₀) [keV]:", "Central ion temperature."),
            'magnetic_field_T': ("Avg. B-Field (B):", "Effective toroidal magnetic field in plasma."), 
            'plasma_volume_m3': ("Plasma Volume (V) [m³]:", "Volume of the confined plasma (inner chamber)."),
            'plasma_surface_area_m2': ("Plasma Surface Area (A) [m²]:", "Surface area of the confined plasma."), 
            'divertor_wetted_area_m2': ("Divertor Wetted Area [m²]:", "Area for power exhaust."),
            'plasma_current_MA': ("Plasma Current (Ip) [MA]:", "Required for some confinement types."), 
            'q_safety_factor': ("Safety Factor (q95):", "Stability measure related to field line pitch."),
            'density_profile_alpha': ("Density Peaking (α_n):", "Exponent for n(r) = n₀(1-(r/a)²)ᴬⁿ."), 
            'temp_profile_alpha': ("Temp. Peaking (α_T):", "Exponent for T(r) = T₀(1-(r/a)²)ᴬᵗ."),
            'sputtering_yield': ("Sputtering Yield (avg.):", "Avg. impurities per incident particle."),
            'Z_eff': ("Effective Charge (Z_eff base):", "Measure of plasma purity before PMI effects.")
        }
        for i, (key, (label_text, tooltip_text)) in enumerate(param_info.items()):
            label = ttk.Label(input_frame, text=label_text)
            label.grid(row=i, column=0, sticky=tk.W, pady=2)
            Tooltip(label, tooltip_text)
            var = tk.StringVar()
            
            widget_args = {"textvariable": var}
            if key == 'fuel_cycle':
                widget = ttk.Combobox(input_frame, **widget_args, values=['D-T', 'D-D'], state='readonly')
            elif key == 'transport_model':
                widget = ttk.Combobox(input_frame, **widget_args, values=['Gyro-Bohm', 'ITER98', 'Neo-classical'], state='readonly')
            else:
                widget = ttk.Entry(input_frame, width=20, **widget_args)
            
            widget.grid(row=i, column=1, sticky=tk.EW, pady=2)
            self.phys_entries[key] = var
        input_frame.columnconfigure(1, weight=1)
        
        button_frame = ttk.Frame(left_frame)
        button_frame.pack(fill=tk.X, pady=10)
        ttk.Button(button_frame, text="▶ Calculate Power Balance", command=self.run_physics_simulation, style="Accent.TButton").pack(side=tk.LEFT, expand=True, fill=tk.X)
        self.export_button = ttk.Menubutton(button_frame, text="Export Report", state=tk.DISABLED)
        self.export_button.pack(side=tk.LEFT, padx=(5,0))
        export_menu = tk.Menu(self.export_button, tearoff=0)
        export_menu.add_command(label="Export as PDF", command=lambda: self.export_report('pdf'))
        export_menu.add_command(label="Export as HTML", command=lambda: self.export_report('html'))
        self.export_button["menu"] = export_menu
        
        self.load_preset() 

        results_frame = ttk.LabelFrame(right_frame, text="Performance & Stability", padding=10)
        results_frame.pack(fill=tk.X)
        self.phys_result_labels = {}
        result_keys = [
            ("Q_value", "Q-Value (Fusion Gain):"), ("P_fusion_MW", "Fusion Power (MW):"),
            ("plasma_beta_percent", "Plasma Beta (%):"), ("greenwald_fraction", "Greenwald Fraction:"),
            ("confinement_time_s", "Energy Confinement (s):"), ("troyon_beta_percent", "Troyon Beta Limit (%):"),
            ("P_heating_MW", "External Heating (MW):"), ("final_Z_eff", "Final Z_eff (incl. PMI):")
        ]
        for i, (key, label_text) in enumerate(result_keys):
            col = (i % 2) * 2
            ttk.Label(results_frame, text=label_text).grid(row=i//2, column=col, sticky=tk.W)
            lbl = ttk.Label(results_frame, text="N/A", font=("Courier", 10, "bold"))
            lbl.grid(row=i//2, column=col+1, sticky=tk.W, padx=5)
            self.phys_result_labels[key] = lbl
        results_frame.columnconfigure((1,3), weight=1)

        loads_frame = ttk.LabelFrame(right_frame, text="Component Heat Loads", padding=10)
        loads_frame.pack(fill=tk.X, pady=5)
        load_keys = [
            ("avg_wall_load_MW_m2", "Avg. Wall Load (MW/m²):"),
            ("peak_divertor_load_MW_m2", "Peak Divertor Load (MW/m²):")
        ]
        for i, (key, label_text) in enumerate(load_keys):
            ttk.Label(loads_frame, text=label_text).grid(row=i, column=0, sticky=tk.W)
            lbl = ttk.Label(loads_frame, text="N/A", font=("Courier", 10, "bold"))
            lbl.grid(row=i, column=1, sticky=tk.W, padx=5)
            self.phys_result_labels[key] = lbl
        loads_frame.columnconfigure(1, weight=1)

        chart_frame = ttk.LabelFrame(right_frame, text="Power Balance", padding=10)
        chart_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.power_fig, self.power_ax = plt.subplots(figsize=(5, 3))
        self.power_canvas = FigureCanvasTkAgg(self.power_fig, master=chart_frame)
        self.power_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.power_fig.tight_layout()

    def create_ovoid_coil_designer_tab(self):
        frame = ttk.Frame(self.tab_coil_designer, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)

        input_plot_frame = ttk.Frame(frame)
        input_plot_frame.pack(fill=tk.BOTH, expand=True)

        input_frame_container = ttk.Frame(input_plot_frame)
        input_frame_container.pack(side=tk.LEFT, fill=tk.Y, padx=(0,10), anchor='nw')
        
        input_frame = ttk.LabelFrame(input_frame_container, text="Ovoid Fibonacci Coil Parameters", padding=10)
        input_frame.pack(fill=tk.X, pady=5)

        self.ovoid_coil_entries = {
            "max_radius_coil_assembly": tk.DoubleVar(value=0.7),
            "total_height_coil_assembly": tk.DoubleVar(value=2.0),
            "num_spirals_set1": tk.IntVar(value=7),
            "num_turns_set1": tk.DoubleVar(value=5.0),
            "num_spirals_set2": tk.IntVar(value=7),
            "num_turns_set2": tk.DoubleVar(value=5.0),
            "points_per_coil": tk.IntVar(value=200),
            "plasma_chamber_radius_fraction": tk.DoubleVar(value=0.4),
            "total_current_MA_coil_assembly": tk.DoubleVar(value=50.0)
        }
        
        param_labels_tooltips = [
            ("Max Radius (Coil Assembly) [m]:", "max_radius_coil_assembly", "Maximum radius of the ovoid coil structure at its widest point."),
            ("Total Height (Coil Assembly) [m]:", "total_height_coil_assembly", "Total height of the ovoid coil structure."),
            ("Num Spirals (Set 1):", "num_spirals_set1", "Number of coils in the first spiral set (e.g., right-handed)."),
            ("Num Turns (Set 1):", "num_turns_set1", "Number of full 360-degree turns each spiral in Set 1 makes."),
            ("Num Spirals (Set 2):", "num_spirals_set2", "Number of coils in the second spiral set (e.g., left-handed)."),
            ("Num Turns (Set 2):", "num_turns_set2", "Number of full 360-degree turns each spiral in Set 2 makes."),
            ("Points per Coil:", "points_per_coil", "Number of points to define each coil path (resolution)."),
            ("Plasma Chamber Radius Fraction:", "plasma_chamber_radius_fraction", "Fraction of coil assembly max radius for visualizing inner plasma chamber."),
            ("Total Current in Coil Assembly (MA):", "total_current_MA_coil_assembly", "Total effective current for the entire coil assembly (for B-field estimation).")
        ]

        for i, (label_text, key, tooltip_text) in enumerate(param_labels_tooltips):
            label = ttk.Label(input_frame, text=label_text)
            label.grid(row=i, column=0, sticky="w", pady=3, padx=5)
            Tooltip(label, tooltip_text)
            entry = ttk.Entry(input_frame, textvariable=self.ovoid_coil_entries[key], width=15)
            entry.grid(row=i, column=1, pady=3, padx=5, sticky="ew")
        
        input_frame.columnconfigure(1, weight=1)

        ttk.Button(input_frame_container, text="Generate Ovoid Coil Geometry", 
                   command=self.run_ovoid_coil_generation, style="Accent.TButton").pack(fill=tk.X, pady=10, anchor='n')

        plot_frame = ttk.LabelFrame(input_plot_frame, text="Coil Geometry Visualization", padding="10")
        plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.ovoid_coil_fig = plt.figure(figsize=(7, 8)) 
        self.ovoid_coil_ax = self.ovoid_coil_fig.add_subplot(111, projection='3d')
        self.ovoid_coil_canvas = FigureCanvasTkAgg(self.ovoid_coil_fig, master=plot_frame)
        self.ovoid_coil_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.ovoid_coil_ax.set_xlabel("X (m)"); self.ovoid_coil_ax.set_ylabel("Y (m)"); self.ovoid_coil_ax.set_zlabel("Z (m)")
        self.ovoid_coil_ax.set_title("Ovoid Coil Assembly")
        self.ovoid_coil_fig.tight_layout()

    def run_ovoid_coil_generation(self):
        try:
            params_gui = {key: var.get() for key, var in self.ovoid_coil_entries.items()}
            
            if params_gui['num_turns_set1'] <= 0 or params_gui['num_turns_set2'] <= 0:
                messagebox.showerror("Input Error", "Number of turns for coil sets must be greater than zero.")
                return

            pitch_s1 = params_gui['total_height_coil_assembly'] / params_gui['num_turns_set1']
            pitch_s2 = -params_gui['total_height_coil_assembly'] / params_gui['num_turns_set2']

            self.designed_ovoid_coil_params = {
                "geometry": "ovoid_fibonacci_reactor", 
                "max_radius_coil_assembly": params_gui['max_radius_coil_assembly'],
                "total_height_coil_assembly": params_gui['total_height_coil_assembly'],
                "num_spirals_set1": int(params_gui['num_spirals_set1']),
                "pitch_set1": pitch_s1,
                "num_turns_set1": params_gui['num_turns_set1'],
                "num_spirals_set2": int(params_gui['num_spirals_set2']),
                "pitch_set2": pitch_s2,
                "num_turns_set2": params_gui['num_turns_set2'],
                "points_per_coil": int(params_gui['points_per_coil']),
                "total_current_MA_coil_assembly": params_gui['total_current_MA_coil_assembly']
            }
            
            self.designed_ovoid_coil_paths_list = generate_fibonacci_spiral_coils(
                max_radius_coil_assembly=params_gui['max_radius_coil_assembly'],
                total_height_coil_assembly=params_gui['total_height_coil_assembly'],
                num_spirals_set1=int(params_gui['num_spirals_set1']),
                pitch_set1=pitch_s1,
                num_turns_set1=params_gui['num_turns_set1'],
                num_spirals_set2=int(params_gui['num_spirals_set2']),
                pitch_set2=pitch_s2,
                num_turns_set2=params_gui['num_turns_set2'],
                points_per_coil=int(params_gui['points_per_coil'])
            )

            if not self.designed_ovoid_coil_paths_list:
                messagebox.showerror("Error", "Ovoid coil generation failed or produced no paths."); return

            self.designed_coil_points = np.vstack(self.designed_ovoid_coil_paths_list) 
            self.current_coil_design_type = "ovoid_fibonacci"

            self.ovoid_coil_ax.clear() 
            plot_fibonacci_coils_with_chamber( 
                ax=self.ovoid_coil_ax, 
                coil_paths_list=self.designed_ovoid_coil_paths_list,
                coil_assembly_max_radius=params_gui['max_radius_coil_assembly'],
                coil_assembly_total_height=params_gui['total_height_coil_assembly'],
                plasma_chamber_radius_fraction=params_gui['plasma_chamber_radius_fraction'],
                num_s1_for_coloring=int(params_gui['num_spirals_set1']) 
            )
            self.ovoid_coil_canvas.draw()
            
            self.coil_plot_image_path = os.path.join(self.output_dir, "current_ovoid_coil_design.png")
            self.ovoid_coil_fig.savefig(self.coil_plot_image_path, dpi=150, bbox_inches='tight')

            messagebox.showinfo("Success", "Ovoid Fibonacci coil geometry generated and visualized. Plot saved for reporting.")
        except (ValueError, tk.TclError, ZeroDivisionError) as e:
            messagebox.showerror("Input Error", f"Invalid ovoid coil parameters: {e}"); return
        except Exception as e:
            messagebox.showerror("Generation Error", f"An unexpected error occurred during ovoid coil generation: {e}"); return

    def create_magnetics_tab(self):
        frame = ttk.Frame(self.tab_magnetics, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        ttk.Button(frame, text="Calculate B-Field Map from Designed Ovoid Coil", 
                   command=self.run_magnetic_field_analysis, style="Accent.TButton").pack(fill=tk.X, pady=5)
        self.mag_status = ttk.Label(frame, text="Design an ovoid coil in the 'Ovoid Coil Designer' tab first.")
        self.mag_status.pack(pady=5)
        
        plot_frame = ttk.LabelFrame(frame, text="Off-Axis Magnetic Field", padding="10")
        plot_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.mag_fig, self.mag_ax = plt.subplots(figsize=(7, 7)) 
        self.mag_canvas = FigureCanvasTkAgg(self.mag_fig, master=plot_frame)
        self.mag_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.mag_fig.tight_layout()

    def run_magnetic_field_analysis(self):
        if self.current_coil_design_type!= "ovoid_fibonacci" or \
           self.designed_coil_points is None or \
           self.designed_ovoid_coil_params is None or \
           not self.designed_ovoid_coil_paths_list: 
            messagebox.showerror("Error", "Please design the Ovoid Fibonacci coil first in the 'Ovoid Coil Designer' tab."); return

        self.mag_status.config(text="Calculating B-field... this may take a moment.")
        self.root.update_idletasks() 
        
        try:
            ovoid_design_params = self.designed_ovoid_coil_params
            total_current_MA_assembly = ovoid_design_params.get('total_current_MA_coil_assembly', 50.0)
            
            num_total_individual_coils = len(self.designed_ovoid_coil_paths_list)
            if num_total_individual_coils == 0:
                messagebox.showerror("Error", "No coil paths available for B-field calculation."); return
            
            current_per_individual_coil_path_Amps = (total_current_MA_assembly * 1e6) / num_total_individual_coils
            
            r_calc_max = ovoid_design_params['max_radius_coil_assembly'] * 1.5 
            z_calc_max = ovoid_design_params['total_height_coil_assembly'] / 2 * 1.5

            Br, Bz, r_grid, z_grid = calculate_b_field_off_axis(
                self.designed_coil_points, 
                current_per_individual_coil_path_Amps, 
                r_calc_max, 
                z_calc_max,
                num_r_points=50, 
                num_z_points=50 
            )
            
            B_mag = np.sqrt(Br**2 + Bz**2)
            self.mag_ax.clear()
            contour = self.mag_ax.contourf(r_grid, z_grid, B_mag.T, levels=30, cmap='plasma')
            
            if hasattr(self, 'mag_cbar') and self.mag_cbar: 
                self.mag_cbar.remove()
            self.mag_cbar = self.mag_fig.colorbar(contour, ax=self.mag_ax, label='Magnetic Field Strength |B| (T)')
            
            skip_val = max(1, Br.shape[0] // 20) if Br.shape[0] > 0 else 1
            max_b_val = np.max(B_mag)
            # --- CORRECTED: Prevent division by zero if field is zero and removed double transpose on Br/Bz.
            scale_val = max_b_val * 30 if max_b_val > 0 else 1
            quiver_r_grid = r_grid[::skip_val, ::skip_val]
            quiver_z_grid = z_grid[::skip_val, ::skip_val]
            # --- NOTE: The returned Br, Bz from `calculate_b_field_off_axis` are already transposed. No need for .T here.
            self.mag_ax.quiver(quiver_r_grid, quiver_z_grid, 
                               Br[::skip_val, ::skip_val], Bz[::skip_val, ::skip_val], 
                               color='white', scale=scale_val, width=0.003)
            
            for coil_path_arr in self.designed_ovoid_coil_paths_list:
                r_coords = np.sqrt(coil_path_arr[:,0]**2 + coil_path_arr[:,1]**2)
                z_coords = coil_path_arr[:,2]
                self.mag_ax.plot(r_coords, z_coords, 'k.', markersize=0.5, alpha=0.2) 
                self.mag_ax.plot(-r_coords, z_coords, 'k.', markersize=0.5, alpha=0.2)

            self.mag_ax.set_xlabel("Radius (m)"); self.mag_ax.set_ylabel("Height (m)"); self.mag_ax.set_title("Off-Axis Magnetic Field Map (Ovoid Coils)")
            self.mag_ax.set_aspect('equal', adjustable='box')
            self.mag_ax.set_xlim([-r_calc_max * 0.8, r_calc_max * 0.8]) 
            self.mag_ax.set_ylim([-z_calc_max * 0.8, z_calc_max * 0.8])
            self.mag_canvas.draw()
            self.mag_status.config(text="B-field calculation complete.")

        except Exception as e:
            messagebox.showerror("Magnetic Analysis Error", f"An error occurred: {e}")
            self.mag_status.config(text="Error during B-field calculation.")

    def create_sweep_tab(self):
        frame = ttk.Frame(self.tab_sweep, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        setup_frame = ttk.LabelFrame(frame, text="2D Scan Setup", padding="10")
        setup_frame.pack(fill=tk.X, pady=5)
        param_keys = [
            'ion_density_m3', 'ion_temperature_keV', 'magnetic_field_T', 'h_factor',
            'plasma_current_MA', 'q_safety_factor', 'density_profile_alpha', 'temp_profile_alpha'
        ]
        
        ttk.Label(setup_frame, text="X-Axis:").grid(row=0, column=0, sticky=tk.W)
        self.scan_x_param = ttk.Combobox(setup_frame, values=param_keys, state="readonly")
        self.scan_x_param.grid(row=0, column=1, columnspan=2, sticky=tk.EW, pady=2)
        self.scan_x_param.set('ion_temperature_keV')
        ttk.Label(setup_frame, text="Range:").grid(row=1, column=0, sticky=tk.W)
        self.scan_x_start = ttk.Entry(setup_frame, width=10); self.scan_x_start.grid(row=1, column=1)
        self.scan_x_start.insert(0, "10")
        self.scan_x_end = ttk.Entry(setup_frame, width=10); self.scan_x_end.grid(row=1, column=2)
        self.scan_x_end.insert(0, "30")
        
        ttk.Label(setup_frame, text="Y-Axis:").grid(row=0, column=3, sticky=tk.W, padx=(10,0))
        self.scan_y_param = ttk.Combobox(setup_frame, values=param_keys, state="readonly")
        self.scan_y_param.grid(row=0, column=4, columnspan=2, sticky=tk.EW, pady=2)
        self.scan_y_param.set('ion_density_m3')
        ttk.Label(setup_frame, text="Range:").grid(row=1, column=3, sticky=tk.W, padx=(10,0))
        self.scan_y_start = ttk.Entry(setup_frame, width=10); self.scan_y_start.grid(row=1, column=4)
        self.scan_y_start.insert(0, "1.0e20")
        self.scan_y_end = ttk.Entry(setup_frame, width=10); self.scan_y_end.grid(row=1, column=5)
        self.scan_y_end.insert(0, "5.0e20")
        
        ttk.Label(setup_frame, text="Resolution (NxN):").grid(row=2, column=0, sticky=tk.W)
        self.scan_steps = ttk.Entry(setup_frame, width=10)
        self.scan_steps.grid(row=2, column=1, columnspan=2, sticky=tk.EW)
        self.scan_steps.insert(0, "20")
        
        setup_frame.columnconfigure((1,2,4,5), weight=1)

        button_frame = ttk.Frame(setup_frame)
        button_frame.grid(row=3, column=0, columnspan=6, sticky=tk.EW, pady=10)
        self.run_sweep_button = ttk.Button(button_frame, text="▶ Run 2D Sweep", command=self.run_2d_sweep, style="Accent.TButton")
        self.run_sweep_button.pack(side=tk.LEFT, expand=True, fill=tk.X)
        self.save_sweep_button = ttk.Button(button_frame, text="Save Sweep Results", command=self.export_sweep_results, state=tk.DISABLED)
        self.save_sweep_button.pack(side=tk.LEFT, padx=(5,0))
        
        plot_frame = ttk.LabelFrame(frame, text="Q-Value Heatmap", padding="10")
        plot_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.sweep_fig, self.sweep_ax = plt.subplots(figsize=(7, 6))
        self.sweep_canvas = FigureCanvasTkAgg(self.sweep_fig, master=plot_frame)
        self.sweep_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.sweep_fig.tight_layout()

    def load_preset(self, event=None):
        selected_preset_name = self.preset_var.get()
        params_to_load = self.presets.get(selected_preset_name, {})
        
        if selected_preset_name == "Custom": return

        for key, var in self.phys_entries.items():
            var.set(str(params_to_load.get(key, var.get())))

        if selected_preset_name == "Ovoid Fibonacci Concept":
            self.ovoid_coil_entries["max_radius_coil_assembly"].set(0.7)
            self.ovoid_coil_entries["total_height_coil_assembly"].set(2.0)
            self.ovoid_coil_entries["num_spirals_set1"].set(7)
            self.ovoid_coil_entries["num_turns_set1"].set(5.0)
            self.ovoid_coil_entries["num_spirals_set2"].set(7)
            self.ovoid_coil_entries["num_turns_set2"].set(5.0)
            self.ovoid_coil_entries["total_current_MA_coil_assembly"].set(50.0)
            self.ovoid_coil_entries["plasma_chamber_radius_fraction"].set(0.4)

    def get_current_phys_params(self):
        params = {}
        for key, var in self.phys_entries.items():
            val_str = var.get()
            try:
                if key not in ['fuel_cycle', 'transport_model']:
                    params[key] = float(val_str) if val_str else 0.0
                else:
                    params[key] = val_str if val_str else "" 
            except ValueError: 
                params[key] = 0.0 if key not in ['fuel_cycle', 'transport_model'] else ""
                messagebox.showwarning("Input Warning", f"Invalid value for {key}. Using default.")
        return params
        
    def run_physics_simulation(self):
        try:
            current_params = self.get_current_phys_params()
        except (ValueError, tk.TclError) as e:
            messagebox.showerror("Input Error", f"Invalid physics parameters: {e}"); return
        
        self.preset_var.set("Custom") 
        self.phys_results = calculate_power_balance(current_params)
        
        for key, label_widget in self.phys_result_labels.items():
            val = self.phys_results.get(key)
            if val is not None and isinstance(val, (int, float)):
                if abs(val) > 1e4 or (abs(val) < 1e-3 and val!= 0): 
                    label_widget.config(text=f"{val:.3e}")
                else:
                    label_widget.config(text=f"{val:.4f}") 
            elif val is not None:
                label_widget.config(text=str(val))
            else:
                label_widget.config(text="N/A")
                
        self.update_power_chart()
        self.export_button.config(state=tk.NORMAL)

    def update_power_chart(self):
        res = self.phys_results
        sources = {'$P_{\\alpha}$': res.get('P_alpha_MW', 0)} 
        losses = {'$P_{rad}$': res.get('P_rad_MW', 0), '$P_{transport}$': res.get('P_transport_MW', 0)}
        
        self.power_ax.clear()
        self.power_ax.bar(sources.keys(), sources.values(), color='#4CAF50', label='Heating (Alpha)')
        self.power_ax.bar(losses.keys(), losses.values(), color='#D32F2F', label='Losses')
        
        p_heating_val = res.get('P_heating_MW', 0)
        if p_heating_val > 1e-3 : 
             self.power_ax.bar(['$P_{ext\_heat}$'], [p_heating_val], color='#FFC107', label='External Heating Req.')

        self.power_ax.set_ylabel("Power (MW)")
        self.power_ax.set_title("Plasma Power Balance")
        self.power_ax.legend(fontsize='small')
        self.power_ax.grid(axis='y', linestyle='--', alpha=0.7)
        self.power_fig.tight_layout()
        self.power_canvas.draw()
        
    def run_2d_sweep(self):
        try:
            base_params = self.get_current_phys_params()
            x_key, y_key = self.scan_x_param.get(), self.scan_y_param.get()
            steps = int(self.scan_steps.get())
            if steps < 2: messagebox.showerror("Input Error", "Resolution (NxN) must be at least 2."); return
            x_vals = np.linspace(float(self.scan_x_start.get()), float(self.scan_x_end.get()), steps)
            y_vals = np.linspace(float(self.scan_y_start.get()), float(self.scan_y_end.get()), steps)
        except (ValueError, tk.TclError) as e:
            messagebox.showerror("Input Error", f"Invalid scan parameters: {e}"); return
        if x_key == y_key:
            messagebox.showerror("Input Error", "Scan parameters (X and Y axes) must be different."); return

        tasks = [dict(base_params, **{x_key: x_val, y_key: y_val}) for y_val in y_vals for x_val in x_vals]
        
        self.run_sweep_button.config(state=tk.DISABLED)
        progress_popup = tk.Toplevel(self.root)
        progress_popup.title("Sweep Progress")
        ttk.Label(progress_popup, text=f"Running {len(tasks)} simulations...").pack(padx=20, pady=10)
        progress_bar = ttk.Progressbar(progress_popup, orient="horizontal", length=300, mode="determinate", maximum=len(tasks))
        progress_bar.pack(padx=20, pady=10)
        
        try:
            # --- CORRECTED: Use map_async for a responsive GUI ---
            with Pool(processes=max(1, cpu_count() - 1)) as pool:
                async_result = pool.map_async(run_scan_worker, tasks)
                # Monitor progress without freezing the GUI
                while not async_result.ready():
                    remaining = async_result._number_left
                    progress_bar['value'] = len(tasks) - remaining
                    self.root.update_idletasks()
                    time.sleep(0.1)
                
                results_list = async_result.get()

            if progress_popup.winfo_exists(): progress_popup.destroy() 
        except Exception as e:
            messagebox.showerror("Sweep Error", f"Error during parallel processing: {e}")
            if progress_popup.winfo_exists(): progress_popup.destroy()
            self.run_sweep_button.config(state=tk.NORMAL)
            return
        
        self.run_sweep_button.config(state=tk.NORMAL)
        q_values_flat = [res.get('Q_value', 0) if isinstance(res, dict) else 0 for res in results_list]
        q_values = np.array(q_values_flat).reshape(len(y_vals), len(x_vals))
        q_values_for_plot = np.nan_to_num(q_values, nan=0.0, posinf=50, neginf=0.0)

        self.sweep_ax.clear()
        vmax_val = np.percentile(q_values_for_plot[q_values_for_plot <= 100], 99.8) if np.any(q_values_for_plot <=100) else 10
        if vmax_val == 0 and np.any(q_values_for_plot > 0): vmax_val = np.max(q_values_for_plot)
        if vmax_val == 0: vmax_val = 10

        im = self.sweep_ax.imshow(q_values_for_plot, extent=[x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()], 
                                  origin='lower', aspect='auto', cmap='viridis', vmin=0, vmax=vmax_val)
        CS = self.sweep_ax.contour(x_vals, y_vals, q_values_for_plot, levels=[1, 2, 3, 4], colors='white', linestyles='dashed')
        self.sweep_ax.clabel(CS, inline=True, fontsize=9, fmt='Q=%.0f')
        self.sweep_ax.set_xlabel(x_key.replace('_', ' ').title()); self.sweep_ax.set_ylabel(y_key.replace('_', ' ').title()); self.sweep_ax.set_title("Q-Value Map")
        
        if hasattr(self, 'sweep_cbar') and self.sweep_cbar: self.sweep_cbar.remove()
        self.sweep_cbar = self.sweep_fig.colorbar(im, ax=self.sweep_ax, label="Q-Value", extend='max' if vmax_val < np.max(q_values_for_plot[np.isfinite(q_values_for_plot)]) else 'neither')
        
        self.sweep_fig.tight_layout(); self.sweep_canvas.draw()
        self.sweep_results = {'x': x_vals, 'y': y_vals, 'q': q_values, 'x_key': x_key, 'y_key': y_key}
        self.save_sweep_button.config(state=tk.NORMAL); messagebox.showinfo("Scan Complete", "2D parameter sweep finished.")

    def export_report(self, format_type):
        filename = filedialog.asksaveasfilename(
            initialdir=self.output_dir, title=f"Save {format_type.upper()} Report", 
            defaultextension=f".{format_type}",
            filetypes=((f"{format_type.upper()} files", f"*.{format_type}"),("All files", "*.*"))
        )
        if not filename: return
        
        try:
            report_params = self.get_current_phys_params()
            image_to_pass_to_report = None 

            if self.current_coil_design_type == "ovoid_fibonacci" and self.designed_ovoid_coil_params:
                for key, value in self.designed_ovoid_coil_params.items():
                    report_params[key] = value 
                
                if self.coil_plot_image_path and os.path.exists(self.coil_plot_image_path):
                    image_to_pass_to_report = self.coil_plot_image_path
                else:
                    print(f"Warning: Coil plot image not found at {self.coil_plot_image_path} for report.")

            if not self.phys_results:
                messagebox.showwarning("Export Warning", "No physics simulation results to report. Please run a simulation first.")
                return

            if format_type == 'pdf':
                generate_pdf_report(filename, report_params, self.phys_results, image_path=image_to_pass_to_report)
            else: # html
                html_image_path_relative = None
                if image_to_pass_to_report:
                    try:
                        report_dir = os.path.dirname(filename)
                        base_image_name = os.path.basename(image_to_pass_to_report)
                        destination_image_path = os.path.join(report_dir, base_image_name)
                        if not os.path.exists(destination_image_path) or not os.path.samefile(image_to_pass_to_report, destination_image_path):
                            shutil.copy(image_to_pass_to_report, destination_image_path)
                        html_image_path_relative = base_image_name 
                    except Exception as e_copy:
                        print(f"Could not copy image for HTML report: {e_copy}. Using absolute path or no image.")
                        html_image_path_relative = image_to_pass_to_report 
                
                generate_html_report(filename, report_params, self.phys_results, image_path=html_image_path_relative)
                
            messagebox.showinfo("Export Successful", f"Report saved to:\n{filename}")
        except Exception as e:
            messagebox.showerror("Export Failed", f"An error occurred during report generation: {e}")
            print(f"Report export error details: {type(e).__name__}, {e}")

    def export_sweep_results(self):
        if not self.sweep_results: messagebox.showerror("Error", "No sweep results to save."); return
        basename_suggestion = os.path.join(self.output_dir, 
                                           f"sweep_{self.sweep_results['x_key']}_vs_{self.sweep_results['y_key']}")
        
        filename_png = filedialog.asksaveasfilename(
            initialdir=self.output_dir, title="Save Sweep Heatmap Image",
            initialfile=basename_suggestion + ".png",
            defaultextension=".png", filetypes=[("PNG Image", "*.png")])
        if not filename_png: return 

        base_png_name, _ = os.path.splitext(filename_png)
        filename_csv = filedialog.asksaveasfilename(
            initialdir=os.path.dirname(filename_png), title="Save Sweep Data", 
            initialfile=base_png_name + ".csv", 
            defaultextension=".csv", filetypes=[("CSV file", "*.csv")])
        if not filename_csv: return 
        
        try:
            self.sweep_fig.savefig(filename_png, dpi=300, bbox_inches='tight')
            
            header_info_row = [f"X-Axis -> {self.sweep_results['x_key']}", ""]
            header_values_row = [f"Y-Axis ({self.sweep_results['y_key']}) / X-Axis ({self.sweep_results['x_key']})"] + \
                                [f"{x_val:.4g}" for x_val in self.sweep_results['x']]
            
            with open(filename_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header_info_row)
                writer.writerow(header_values_row)
                for i, y_val in enumerate(self.sweep_results['y']):
                    row_data = [f"{y_val:.4g}"] + \
                               [f"{self.sweep_results['q'][i,j]:.4g}" if np.isfinite(self.sweep_results['q'][i,j]) else str(self.sweep_results['q'][i,j]) 
                                for j in range(len(self.sweep_results['x']))]
                    writer.writerow(row_data)

            messagebox.showinfo("Export Successful", f"Sweep heatmap and data saved as:\n{filename_png}\n{filename_csv}")
        except Exception as e:
            messagebox.showerror("Export Failed", f"An error occurred during sweep export: {e}")

if __name__ == '__main__':
    root = tk.Tk()
    style = ttk.Style(root)
    available_themes = style.theme_names()
    # A more robust theme selection
    if 'clam' in available_themes:
        style.theme_use("clam")
    elif 'vista' in available_themes:
        style.theme_use("vista")
    elif 'aqua' in available_themes:
        style.theme_use("aqua")
    
    style.configure("Accent.TButton", foreground="white", background="#0078D4", font=('Helvetica', 10, 'bold'))
    style.configure("TButton", font=('Helvetica', 10))
    style.configure("TLabel", font=('Helvetica', 10))
    style.configure("TEntry", font=('Helvetica', 10))
    style.configure("TCombobox", font=('Helvetica', 10))
    style.configure("TMenubutton", font=('Helvetica', 10))
    style.configure("TLabelframe", font=('Helvetica', 10))
    style.configure("TNotebook.Tab", font=('Helvetica', 10, 'bold'))
    style.configure("TLabelframe.Label", font=('Helvetica', 10, 'bold'))

    app = FusionSimulatorApp(root)
    root.mainloop()