# physics.py
import numpy as np

# --- Constants ---
e_charge = 1.602e-19
m_D = 3.344e-27  # Deuterium mass
m_T = 5.008e-27  # Tritium mass
m_p = 1.672e-27  # Proton mass for scaling
mu_0 = 4 * np.pi * 1e-7
k_boltzmann = 1.3806e-23
epsilon_0 = 8.854e-12

def get_radial_grid(params):
    """
    Derives minor radius 'a' and a radial grid.
    This uses an approximation for a toroidal plasma.
    """
    vol = params.get('plasma_volume_m3', 0)
    area = params.get('plasma_surface_area_m2', 0)
    
    if vol <= 0 or area <= 0:
        return np.array([0]), 0

    # Estimate major radius R and minor radius a from volume and surface area
    # V = 2 * pi^2 * R * a^2
    # A = 4 * pi^2 * R * a
    # From these, R = A^2 / (8 * pi^2 * V) and a = 2V / A
    R_major_est = area**2 / (8 * np.pi**2 * vol)
    a = (2 * vol) / area
    
    return np.linspace(0, a, 101), a

def create_radial_profile(r, r_max, central_value, alpha):
    """Creates a parabolic-like radial profile."""
    if r_max <= 0: return np.zeros_like(r)
    # Ensure the profile doesn't become complex for r > r_max
    norm_r_sq = (r / r_max)**2
    return central_value * (1 - np.minimum(norm_r_sq, 1.0))**alpha

def fusion_reactivity(T_keV, fuel_cycle='D-T'):
    """
    Calculates fusion reactivity <sigma*v> in m^3/s using Bosch-Hale fits.
    """
    T_keV = np.maximum(T_keV, 1e-3) # Avoid issues at T=0
    if fuel_cycle == 'D-T':
        # D-T reactivity fit coefficients
        B_G, c = 34.3827, [1.17302e-1, 1.51361e-2, 2.56434e-1, -1.43699e-3, 1.30938e-5, 0, 4.88168e-7]
        theta = T_keV / (1 - (T_keV * (c[2] + T_keV * (c[4] + T_keV * c[6]))) / (1 + T_keV * (c[1] + T_keV * (c[3] + T_keV * c[5]))))
        theta = np.maximum(theta, 1e-6)
        zeta = (B_G**2 / (4 * theta))**(1/3)
        # Convert from cm^3/s to m^3/s
        return (1.1e-12 * (theta**(-2/3)) * np.exp(-3 * zeta)) / 1e6 
    else: # D-D average reactivity fit
        B_G, c = 31.397, [1.5136e-2, 2.5247e-3, 3.0313e-3, -1.1793e-4, 8.7183e-6, -2.4839e-7, 2.7051e-9]
        theta = T_keV / (1 - (T_keV * (c[2] + T_keV * (c[4] + T_keV * c[6]))) / (1 + T_keV * (c[1] + T_keV * (c[3] + T_keV * c[5]))))
        theta = np.maximum(theta, 1e-6)
        zeta = (B_G**2 / (4 * theta))**(1/3)
        # Convert from cm^3/s to m^3/s
        return (5.68e-18 * (zeta**2 * np.exp(-3 * zeta)) / (T_keV**(2/3))) / 1e6

def calculate_pmi_effects(params, n_profile, tau_E):
    """Estimates the increase in Z_eff due to plasma-material interaction (sputtering)."""
    sputtering_yield = params.get('sputtering_yield', 0.0)
    base_Z_eff = params.get('Z_eff', 1.0)
    if tau_E <= 0 or sputtering_yield == 0: return base_Z_eff
    
    total_ions = np.mean(n_profile) * params['plasma_volume_m3']
    particle_flux_rate = total_ions / tau_E
    impurity_influx_rate = particle_flux_rate * sputtering_yield
    
    # Assuming Tungsten (W) as the impurity from the divertor/wall
    Z_impurity = 74 
    # This factor is a simplified empirical term for impurity concentration in the core
    impurity_confinement_factor_C = 1e-21 
    delta_Z_eff = impurity_confinement_factor_C * impurity_influx_rate * (Z_impurity**2 - Z_impurity)
    
    return base_Z_eff + delta_Z_eff

def calculate_transport_loss(params, n_profile, T_profile_keV, r_grid, a, R_major):
    """Calculates power loss due to heat transport based on selected model."""
    T_profile_J = T_profile_keV * 1000 * e_charge
    m_ion = (m_D + m_T) / 2 if params['fuel_cycle'] == 'D-T' else m_D
    B_T = params['magnetic_field_T']
    
    # Prevent division by zero if profiles are zero
    mean_T_J = np.mean(T_profile_J)
    if mean_T_J <= 0: return 0.0
    
    if params.get('transport_model') == 'Neo-classical':
        coll_log = 24 - np.log(np.sqrt(np.mean(n_profile))/(np.mean(T_profile_keV)*1000))
        nu_ii = (np.mean(n_profile) * (e_charge**4) * coll_log) / (4*np.pi*(epsilon_0**2) * np.sqrt(m_ion) * (mean_T_J**1.5))
        rho_i_theta = np.sqrt(2 * m_ion * mean_T_J) / (e_charge * (B_T * (a / (R_major * params['q_safety_factor']))))
        chi_base_profile = np.full_like(T_profile_J, nu_ii * (rho_i_theta**2) * (R_major/a)**1.5)
    elif params.get('transport_model') == 'ITER98':
        P_loss_est_MW = params.get('P_heating_MW_est', 50.0)
        # Global scaling law for tau_E based on ITER98(y,2)
        tau_E = 0.0562 * params['h_factor'] * (params['plasma_current_MA']**0.93) * (B_T**0.15) * (P_loss_est_MW**-0.69) * \
                ((np.mean(n_profile)/1e19)**0.41) * ((m_ion/m_p)**0.19) * (R_major**1.97) * ((a/R_major)**0.58) * \
                (params.get('kappa_elongation', 1.7)**0.78)
        # Effective chi from global tau_E
        chi_base_profile = np.full_like(T_profile_J, (a**2) / (6 * tau_E) if tau_E > 0 else float('inf'))
    else: # Gyro-Bohm (Default)
        rho_i_profile = np.sqrt(2 * m_ion * T_profile_J) / (e_charge * B_T)
        # Avoid division by zero for rho_i at the center
        rho_i_profile[rho_i_profile == 0] = 1e-9
        chi_base_profile = (T_profile_J / (e_charge * B_T)) * (rho_i_profile / a) * params.get('transport_geometry_factor_G', 1.0)

    grad_T = np.gradient(T_profile_J, r_grid)
    grad_T[-1] = grad_T[-2] # Avoid edge effects from gradient
    q_profile = -n_profile * chi_base_profile * grad_T
    
    # Return heat flux at the second to last grid point (at r~a) multiplied by area
    return (params['plasma_surface_area_m2'] * q_profile[-2]) / 1e6 if len(q_profile) > 1 else 0

def calculate_stability_and_penalty(params, a, n_profile, T_profile_J):
    """Calculates key stability metrics like Greenwald fraction and Beta limit."""
    Ip_MA = params.get('plasma_current_MA', 0)
    B_T = params['magnetic_field_T']
    
    n_avg_1e20 = np.mean(n_profile) / 1e20
    n_greenwald_1e20 = Ip_MA / (np.pi * a**2) if a > 0 else float('inf')
    density_ratio = n_avg_1e20 / n_greenwald_1e20 if n_greenwald_1e20 > 0 else float('inf')
    
    # Plasma beta = plasma pressure / magnetic pressure
    plasma_pressure = np.mean(2 * n_profile * T_profile_J) # Factor of 2 for ions and electrons
    magnetic_pressure = (B_T**2) / (2 * mu_0)
    beta_percent = (plasma_pressure / magnetic_pressure) * 100 if magnetic_pressure > 0 else 0
    
    beta_N_troyon = 2.8 # Troyon limit coefficient for tokamaks
    beta_troyon_percent = beta_N_troyon * (Ip_MA / (a * B_T)) if a > 0 and B_T > 0 else 0
    
    return {
        "greenwald_density_1e20": n_greenwald_1e20, "greenwald_fraction": density_ratio,
        "troyon_beta_percent": beta_troyon_percent, "plasma_beta_percent": beta_percent
    }

def calculate_power_balance(params):
    """Main function to calculate the power balance of the fusion plasma."""
    r, a = get_radial_grid(params)
    if a <= 0: return {}
    R_major = params['plasma_volume_m3'] / (2 * np.pi**2 * a**2)
    
    n_i_profile = create_radial_profile(r, a, params['ion_density_m3'], params['density_profile_alpha'])
    T_i_profile_keV = create_radial_profile(r, a, params['ion_temperature_keV'], params['temp_profile_alpha'])
    T_joules = T_i_profile_keV * 1000 * e_charge
    
    # --- CORRECTED: The differential volume element dV must be used correctly inside the integral. ---
    # dV = (2*pi*R) * (2*pi*r*dr) for a torus. We assume R is constant R_major.
    # dV = 4 * pi^2 * R_major * r * dr. The term `4 * pi^2 * R_major * r` is the integrand multiplier.
    dV_dr = 4 * np.pi**2 * R_major * r

    # Fusion Power
    reactivity = fusion_reactivity(T_i_profile_keV, params['fuel_cycle'])
    E_fus_J, E_alpha_J, factor = (17.6e6*e_charge, 3.5e6*e_charge, 0.25) if params['fuel_cycle']=='D-T' else (3.6e6*e_charge, 1.0e6*e_charge, 0.5)
    fusion_power_density = factor * (n_i_profile**2) * reactivity * E_fus_J
    P_fusion_MW = np.trapz(fusion_power_density * dV_dr, r) / 1e6
    P_alpha_MW = P_fusion_MW * (E_alpha_J / E_fus_J)

    # Thermal Stored Energy
    thermal_energy_density = 3 * n_i_profile * T_joules # W = 1.5*n_e*T_e + 1.5*n_i*T_i -> 3*n*T for n_e=n_i, T_e=T_i
    W_th_joules = np.trapz(thermal_energy_density * dV_dr, r)
    
    stability = calculate_stability_and_penalty(params, a, n_i_profile, T_joules)

    # Iterative loop for transport models that depend on power loss (like ITER98)
    p_loss_est_MW = 50.0 # Initial guess
    for _ in range(5): # Iterate a few times to converge
        p_heat_flow_MW = max(1.0, P_alpha_MW + P_heating_MW if 'P_heating_MW' in locals() else p_loss_est_MW)
        
        # Estimate confinement time and Z_eff based on this power flow
        tau_E_est = W_th_joules / (p_heat_flow_MW * 1e6) if p_heat_flow_MW > 0 else float('inf')
        final_Z_eff = calculate_pmi_effects(params, n_i_profile, tau_E_est)
        
        # Radiation Loss (Bremsstrahlung)
        rad_power_density = 1.69e-38 * (n_i_profile**2) * final_Z_eff * np.sqrt(T_i_profile_keV) # Z_eff^2*sqrt(T) form, more accurate
        P_rad_MW = np.trapz(rad_power_density * dV_dr, r) / 1e6
        
        # Transport Loss
        params['P_heating_MW_est'] = p_heat_flow_MW
        P_transport_MW = calculate_transport_loss(params, n_i_profile, T_i_profile_keV, r, a, R_major)
        
        p_loss_new_MW = P_rad_MW + P_transport_MW
        if abs(p_loss_est_MW - p_loss_new_MW) < 0.1: break # Converged
        p_loss_est_MW = 0.5 * p_loss_est_MW + 0.5 * p_loss_new_MW # Dampen oscillation
    else: # If loop finishes without break
        p_loss_est_MW = p_loss_new_MW

    P_loss_MW = p_loss_est_MW
    P_heating_MW = max(0, P_loss_MW - P_alpha_MW)
    Q_value = P_fusion_MW / P_heating_MW if P_heating_MW > 1e-9 else float('inf')
    confinement_time_s = W_th_joules / (P_loss_MW * 1e6) if P_loss_MW > 0 else float('inf')
    
    return {
        "P_fusion_MW": P_fusion_MW, "P_alpha_MW": P_alpha_MW, "P_rad_MW": P_rad_MW,
        "P_transport_MW": P_transport_MW, "P_heating_MW": P_heating_MW, "Q_value": Q_value,
        "confinement_time_s": confinement_time_s, "final_Z_eff": final_Z_eff,
        "avg_wall_load_MW_m2": (P_rad_MW * 0.8) / params['plasma_surface_area_m2'] if params['plasma_surface_area_m2'] > 0 else 0,
        "peak_divertor_load_MW_m2": (P_transport_MW + P_rad_MW * 0.2) / params['divertor_wetted_area_m2'] if params['divertor_wetted_area_m2'] > 0 else 0,
        **stability
    }