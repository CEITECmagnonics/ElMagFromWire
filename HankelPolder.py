import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hankel2

# =============================================================================
# Physical constants and parameters
# =============================================================================
freqs = np.linspace(2e9, 20e9, 500)  # Frequency range from 1 GHz to 20 GHz
# Define a range of radial distances (avoid r=0 to prevent singularity)
r_vals = np.logspace(-6, -1, 2000)  # from 1 nm to 0.1 m
A_z = np.zeros([len(r_vals), len(freqs)], dtype=complex)
E_z = np.zeros([len(r_vals), len(freqs)], dtype=complex)
H_phi = np.zeros([len(r_vals), len(freqs)], dtype=complex)

for i,f in enumerate(freqs):
    c = 3e8                      # Speed of light in vacuum (m/s)
    mu0 = 4 * np.pi * 1e-7       # Permeability of free space (H/m)
    epsilon0 = 1/(mu0 * c**2)      # Permittivity of free space (F/m)

    # RF source parameters
    omega = 2 * np.pi * f        # Angular frequency (rad/s)
    k = omega / c                # Free-space wavenumber (rad/m)

    gamma = 2 * np.pi * 28e9  # Gyromagnetic ratio (rad/s/T)
    Msat = 86e3                # Saturation magnetization (A/m)
    Bext = 550e-3             # External magnetic field (T)
    alpha = 0.01            # Gilbert damping factor
    K1 = 16779 #J/m3 anisotropy constant, assuming out-of-plane PMA
    omegaH = gamma * Bext + 1j*alpha*omega  #
    omegaM = gamma * mu0 * Msat  # 
    omegaFMR = np.sqrt(omegaH*(omegaH - 2*gamma*K1/Msat + omegaM)) #FMR frequency
    FMR = omegaFMR/(2*np.pi) * 1e-9 #FMR frequency in GHz


    I0 = 75e-3                    # Current amplitude (in amperes)

    muEff = mu0 * ((omegaH + omegaM)**2 - omega**2 )/ (omegaFMR**2 - omega**2)  #Effective permeability

    epsEff = 1/(muEff * c**2)   
    kEff = omega * np.sqrt(muEff * epsEff) #Effective wavenumber

    # =============================================================================
    # Field expressions for an infinitely long wire (line source)
    # =============================================================================
    # The magnetic vector potential (only nonzero component is A_z):
    #   A_z(r) = (μ0 I0 / 4) * H_0^(2)(k r)
    # Electric field (assuming phasor exp(+iωt) convention):
    #   E_z(r) = -i ω A_z(r)
    # Magnetic field (only nonzero component is H_φ):
    #   H_φ(r) = I0 k / 4 * H_1^(2)(k r)
    #
    # These solutions naturally capture both the near-field (reactive) and far-field
    # (radiative) behavior, with the Hankel functions handling the cylindrical wave
    # propagation from an infinite line source.

    # Compute the vector potential A_z, electric field E_z, and magnetic field H_phi
    A_z[:,i] = (muEff * I0 / 4) * hankel2(0, kEff * r_vals)
    E_z[:,i]  = -1j * omega * A_z[:,i] 
    H_phi[:,i]  = (muEff * I0 * kEff / 4) * hankel2(1, kEff * r_vals)

# =============================================================================
# Plotting the field magnitudes versus distance
# =============================================================================
fig, axs = plt.subplots(3, 1, figsize=(10, 5), sharex=True)

# Plot magnetic field H_phi
axs[0].plot(r_vals * 1e6, np.abs(H_phi[:,0::50]) * mu0 * 1e3)
axs[0].set_ylabel('Magnetic Field (mT)')
axs[0].set_title('Field Magnitudes from an Infinitely Long Antenna (Line Source)')
# axs[0].set_yscale('log')
axs[0].set_xscale('log')
axs[0].grid(True, which="both", linestyle="--", alpha=0.7)

# Plot electric field E_z
axs[1].plot(r_vals * 1e6, np.abs(E_z[:,0::50]))
axs[1].set_ylabel('Electric Field (V/m)')
# axs[1].set_yscale('log')
axs[1].set_xscale('log')
axs[1].grid(True, which="both", linestyle="--", alpha=0.7)

# Add a new panel for the phases of H_phi and E_z
axs_phase = fig.axes[2]
axs_phase.plot(r_vals * 1e6, np.angle(H_phi[:,0::50]))
axs_phase.plot(r_vals * 1e6, np.angle(E_z[:,0::50]), linestyle='--')
axs_phase.set_xlabel('Radial Distance r (µm)')
axs_phase.set_ylabel('Phase (rad)')
axs_phase.grid(True, which="both", linestyle="--", alpha=0.7)

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
index = np.argmin(np.abs(r_vals - 40e-6)) # Find the index of the closest value to  um
plt.plot(freqs*1e-9, np.transpose(np.abs(H_phi[0::50,:])) * mu0 * 1e3, label='H_phi')
plt.xlabel('Frequency (GHz)')
plt.ylabel('Magnetic Field (mT)')

plt.show()

plt.figure(figsize=(10, 5))
index = np.argmin(np.abs(r_vals - 15e-6)) # Find the index of the closest value to  um
plt.plot(freqs*1e-9, np.transpose(np.abs(H_phi[index,:])) * mu0 * 1e3, label='H_phi')
plt.xlabel('Frequency (GHz)')
plt.ylabel('Magnetic Field (mT)')
ExpHpiIndex = np.abs(H_phi[index,:]) * mu0 * 1e3

plt.show()

plt.figure(figsize=(10, 5))
index = np.argmin(np.abs(r_vals - 15e-6)) # Find the index of the closest value to  um
plt.plot(freqs*1e-9, np.transpose(np.angle(H_phi[index,:])), label='H_phi')
plt.xlabel('Frequency (GHz)')
plt.ylabel('Phase (rad)')

plt.show()

plt.figure(figsize=(10, 5))
plt.pcolormesh(r_vals* 1e6, freqs * 1e-9, np.log(np.transpose(np.abs(H_phi) * mu0 * 1e3)), shading='auto')
plt.ylabel('Frequency (GHz)')
plt.xlabel('r (µm)')
plt.colorbar(label='Magnetic Field (mT)')
plt.show()



expBmT = np.abs(H_phi) * mu0 * 1e3 #Prepare for export
expEkVom = np.abs(E_z) * 1e-3 #Prepare for export
expPhH = np.angle(H_phi) #Prepare for export
expPhE = np.angle(E_z) #Prepare for export
expR = r_vals * 1e6 #Prepare for export2