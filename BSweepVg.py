import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hankel2

# =============================================================================
# Physical constants and parameters
# =============================================================================
freqs = np.linspace(10.99e9, 11.01e9, 3)  # Frequency range from 1 GHz to 20 GHz
Bexts = np.linspace(600e-3, 450e-3, 1000)  # Frequency range from 1 GHz to 20 GHz

# =============================================================================
# Initialize arrays for fields and magnetization
# =============================================================================
kEffArray = np.zeros([len(Bexts), len(freqs)], dtype=complex)
vg = np.zeros([len(Bexts)], dtype=float)

for j, Bext in enumerate(Bexts):
    for i,f in enumerate(freqs):
        c = 3e8                      # Speed of light in vacuum (m/s)
        mu0 = 4 * np.pi * 1e-7       # Permeability of free space (H/m)
        epsilon0 = 1/(mu0 * c**2)    # Permittivity of free space (F/m)
        epsilonR = 15 - 1j*3.7e-5    # Relative permittivity (dimensionless) - YIG - https://doi.org/10.1016/j.materresbull.2024.112994
        # epsilonR = 10 - 1j*1e7     # Relative permittivity (dimensionless) - Metal
        epsilon = epsilon0 * epsilonR

        # RF source parameters
        omega = 2 * np.pi * f        # Angular frequency (rad/s)
        k = omega / c                # Free-space wavenumber (rad/m)

        gamma = 2 * np.pi * 28e9     # Gyromagnetic ratio (rad/s/T)
        Msat = 86e3                  # Saturation magnetization (A/m)
        alpha = 0.01                 # Gilbert damping factor
        K1 = 16779                   #J/m3 anisotropy constant, assuming out-of-plane PMA
        omegaH = gamma * Bext
        omegaM = gamma * mu0 * Msat   
        omegaEff = gamma * mu0 * (Msat - 2*K1/Msat/mu0)  # Effective frequency
        omegaFMR = np.sqrt(omegaH*(omegaH - 2*gamma*K1/Msat + omegaM)) #FMR frequency
        FMR = omegaFMR/(2*np.pi) * 1e-9   #FMR frequency in GHz


        I0 = 75e-3                    # Current amplitude (in amperes)

        # =============================================================================
        # Effective permeability calculations
        # =============================================================================
        muEffzz = mu0 * (1 + (( 1j * alpha * omega + omegaH) * omegaM)/(-omega**2 + omegaH * (omegaEff + omegaH) + 1j * alpha * omega * (omegaEff + 2 * omegaH)))
        muEff = muEffzz
        # =============================================================================
        # Calculate the effective wavenumber
        # =============================================================================
        kEff = omega * np.sqrt(muEff * epsilon) #Effective wavenumber
        kEffArray[j, i] = kEff
    vg[j] = np.gradient(2 * np.pi * freqs, kEffArray[j, :])[1]  # rad/s divided by rad/m gives m/s


plt.figure(figsize=(8, 6))
plt.semilogy(Bexts*1e3, np.abs(vg)/1e3, '.')
plt.xlabel('External Magnetic Field (mT)')
plt.ylabel('Group Velocity (µm/ns)')

# =============================================================================
# export
# =============================================================================

