import numpy as np
import pandas as pd

def calculate_lateral_pressure(gamma, D, mu, K, z):
    """
    Calculate lateral pressure p(z) using Janssen's Silo Theory.
    p(z) = (gamma * D / (4 * mu * K)) * (1 - exp(-(4 * mu * K * z) / D))
    """
    # Handle division by zero or very small values gracefully
    denominator = 4 * mu * K
    term1 = (gamma * D) / denominator
    term2 = 1 - np.exp(-(denominator * z) / D)
    return term1 * term2

def calculate_hoop_stress(p_z, D, t):
    """
    Calculate hoop stress sigma_h
    sigma_h = (p_z * D) / (2 * t)
    """
    return (p_z * D) / (2 * t)

def generate_synthetic_data(num_samples=10000, add_noise=True):
    """
    Generates synthetic engineering dataset based on Janssen's theory.
    """
    np.random.seed(42)
    # Define ranges for parameters based on typical engineering values
    H = np.random.uniform(10, 50, num_samples) # Silo height (m)
    D = np.random.uniform(5, 20, num_samples) # Silo diameter (m)
    t = np.random.uniform(0.1, 0.5, num_samples) # Wall thickness (m)
    gamma = np.random.uniform(8, 16, num_samples) # Bulk density (kN/m^3)
    mu = np.random.uniform(0.3, 0.6, num_samples) # Wall friction coefficient
    K = np.random.uniform(0.4, 0.6, num_samples) # Lateral pressure coefficient
    
    # Depth position z must be between 0 and H
    z = np.random.uniform(0, H)
    
    p_z = calculate_lateral_pressure(gamma, D, mu, K, z)
    sigma_h = calculate_hoop_stress(p_z, D, t)
    
    if add_noise:
        # Add 1% Gaussian noise to simulate real-world sensor/measurement variations
        p_z += np.random.normal(0, 0.01 * np.mean(p_z), num_samples)
        sigma_h += np.random.normal(0, 0.01 * np.mean(sigma_h), num_samples)
        
    data = pd.DataFrame({
        'Height_H': H,
        'Diameter_D': D,
        'Thickness_t': t,
        'Density_gamma': gamma,
        'Friction_mu': mu,
        'Pressure_Coeff_K': K,
        'Depth_z': z,
        'Lateral_Pressure_p': p_z,
        'Hoop_Stress_sigma_h': sigma_h
    })
    
    return data
