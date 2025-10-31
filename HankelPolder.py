import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hankel2

# =============================================================================
# Physical constants and parameters
# =============================================================================
freqs = np.linspace(1e9, 20e9, 501)  # Frequency range from 1 GHz to 20 GHz
# freqs = [7e9, 11.02e9, 15e9]
# Define a range of radial distances (avoid r=0 to prevent singularity)
r_vals = np.logspace(-6, -1, 500)  # from 1 nm to 0.1 m
# r_vals = np.linspace(1e-6, 1000e-6, 100)  # from 1 µm to 0.1 m

# =============================================================================
# Initialize arrays for fields and magnetization
# =============================================================================
A_z = np.zeros([len(r_vals), len(freqs)], dtype=complex)
E_z = np.zeros([len(r_vals), len(freqs)], dtype=complex)
H_z = np.zeros([len(r_vals), len(freqs)], dtype=complex)
Mx = np.zeros([len(r_vals), len(freqs)], dtype=complex)  # Magnetization array
Mz = np.zeros([len(r_vals), len(freqs)], dtype=complex)  # Magnetization array
muEffArrayxx = np.zeros([len(freqs)], dtype=complex)
muEffArrayxz = np.zeros([len(freqs)], dtype=complex)
muEffArrayzz = np.zeros([len(freqs)], dtype=complex)
kEffArray = np.zeros([len(freqs)], dtype=complex)

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
    Bext = 550e-3                # External magnetic field (T)
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
    muEffxx = mu0 * (1 + (1j* alpha * omega + omegaEff + omegaH) *omegaM)/(-omega**2 + omegaH * (omegaEff + omegaH) + 1j * alpha * omega * (omegaEff + 2 * omegaH))
    muEffxz = -mu0 * 1j * omega * omegaM/(-omega**2 + omegaH * (omegaEff + omegaH) + 1j * alpha * omega * (omegaEff + 2 * omegaH))
    muEffzz = mu0 * (1 + (( 1j * alpha * omega + omegaH) * omegaM)/(-omega**2 + omegaH * (omegaEff + omegaH) + 1j * alpha * omega * (omegaEff + 2 * omegaH)))
    muEffArrayxx[i] = muEffxx  
    muEffArrayxz[i] = muEffxz  
    muEffArrayzz[i] = muEffzz
    muEff = muEffzz
    # =============================================================================
    # Calculate the effective wavenumber
    # =============================================================================
    kEff = omega * np.sqrt(muEff * epsilon) #Effective wavenumber
    kEffArray[i] = kEff

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

    # Compute the vector potential A_z, electric field E_z, and magnetic field H_z
    A_z[:,i] = (mu0 * I0 / 4) * hankel2(0, kEff * r_vals)
    E_z[:,i]  = -1j*omega * A_z[:,i] 
    H_z[:,i]  = (I0 * kEff / 4) * hankel2(1, kEff * r_vals)

    Mx[:,i] = (muEffxz/mu0) * H_z[:,i]  # Magnetization
    Mz[:,i] = (muEffzz/mu0 - 1) * H_z[:,i]  # Magnetization

# =============================================================================
# Plotting the field magnitudes versus distance
# =============================================================================
fig, axs = plt.subplots(5, 1, figsize=(10, 10), sharex=True)
fig.suptitle('Field Magnitudes and Phases on distance', fontsize=14)

# Plot magnetic field H_z
axs[0].plot(r_vals * 1e6, np.abs(H_z[:,0::50]) * mu0 * 1e3, label='H_z')
axs[0].set_ylabel('H (mT)')
# axs[0].set_yscale('log')
axs[0].set_xscale('log')
axs[0].grid(True, which="both", linestyle="--", alpha=0.7)

# Plot electric field E_z
axs[1].plot(r_vals * 1e6, np.abs(E_z[:,0::50]))
axs[1].set_ylabel('Electric Field (V/m)')
# axs[1].set_yscale('log')
axs[1].set_xscale('log')
axs[1].grid(True, which="both", linestyle="--", alpha=0.7)

# Add a new panel for the phases of H_z and E_z
axs_phase = fig.axes[2]
axs_phase.plot(r_vals * 1e6, np.angle(H_z[:,0::50]), label='H_z phase')
axs_phase.plot(r_vals * 1e6, np.angle(E_z[:,0::50]), linestyle=':', label='E_z phase')
axs_phase.set_ylabel('Phase (rad)')
axs_phase.grid(True, which="both", linestyle="--", alpha=0.7)

# Add a new panel for magnetization
axs_mag = fig.axes[3]
axs_mag.plot(r_vals * 1e6, np.abs(Mx), '-', r_vals * 1e6, np.abs(Mz), '--')
axs_mag.legend(['Mx', 'Mz'])
axs_mag.set_ylabel('Magnetization (A/m)')
axs_mag.set_yscale('log')
axs_mag.grid(True, which="both", linestyle="--", alpha=0.7)
axs_mag.set_ylim(1, 8e4)  # Set y-axis limit for magnetization

axs_angle= fig.axes[4]
axs_angle.plot(r_vals * 1e6, np.rad2deg(np.atan(np.abs(Mz)/Msat)))
axs_angle.set_xlabel('Radial Distance r (µm)')
axs_angle.set_ylabel('Magnetization angle (deg)')
axs_angle.set_xscale('log')
axs_angle.set_yscale('log')
axs_angle.grid(True, which="both", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()


# =============================================================================
# Plotting the frequency response of the fields and magnetization
# =============================================================================

fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
fig.suptitle('Frequency Response of Magnetization', fontsize=14)
# Get index for a specific distance
InvDistance = 40e-6 # Sweeps in distance of in meters
index = np.argmin(np.abs(r_vals - InvDistance)) # Find the index of the closest value in m

axs_mag = fig.axes[0]
axs_mag.plot(freqs*1e-9, np.abs(Mx[index,:]), label='Mx')
axs_mag.plot(freqs*1e-9, np.abs(Mz[index,:]), label='Mz')
axs_mag.set_ylabel('M (A/m)')
axs_mag.vlines(FMR, 0, 1, color='red', linestyle='--', label='FMR frequency')
ExpHziIndex = np.abs(H_z[index,:]) * mu0 * 1e3
axs_mag.set_yscale('log')
axs_mag.legend(loc='lower right')
plt.legend()
ymin, ymax = axs_mag.get_ylim()
axs_mag.vlines(FMR, ymin, ymax, color='red', linestyle='--', label='FMR frequency')

axs_mag_phase = fig.axes[1]
axs_mag_phase.plot(freqs*1e-9, np.transpose(np.angle(Mx[index,:])), label='Mx Phase')
axs_mag_phase.plot(freqs*1e-9, np.transpose(np.angle(Mz[index,:])), label='Mz Phase')
axs_mag_phase.vlines(FMR, 0, 1, color='red', linestyle='--', label='FMR frequency')
axs_mag_phase.legend(loc='lower right')
axs_mag_phase.set_ylabel('∠M (rad)')
ymin, ymax = axs_mag_phase.get_ylim()
axs_mag_phase.vlines(FMR, ymin, ymax, color='red', linestyle='--', label='FMR frequency')

axs_mag_elli = fig.axes[2]
axs_mag_elli.plot(freqs*1e-9, np.transpose(np.abs(Mz[index,:])/np.abs(Mx[index,:])), label='Mz/Mx ellipticity')
axs_mag_elli.vlines(FMR, 0, 1, color='red', linestyle='--', label='FMR frequency')
axs_mag_elli.legend(loc='lower right')
axs_mag_elli.set_ylabel('Ellipticity (Mz/Mx)')
ymin, ymax = axs_mag_elli.get_ylim()
axs_mag_elli.vlines(FMR, ymin, ymax, color='red', linestyle='--', label='FMR frequency')
axs_mag_elli.set_xlabel('Frequency (GHz)')

# =============================================================================
# Plotting the effective permeability
# =============================================================================

fig, axs = plt.subplots(3, 1, figsize=(10, 5), sharex=True)
fig.suptitle('Effective Permeability and dispersion', fontsize=14)
axs[0].plot(freqs * 1e-9, np.real(muEffArrayxz/mu0), label='real')
axs[0].plot(freqs * 1e-9, np.imag(muEffArrayxz/mu0), label='imag')
axs[0].plot(freqs * 1e-9, np.abs(muEffArrayxz/mu0), label='abs')
ymin, ymax = axs[0].get_ylim()
axs[0].vlines(FMR, ymin, ymax, color='red', linestyle='--', label='FMR frequency')
axs[0].set_ylabel('Susceptibility ()')
axs[0].set_title('xz')
axs[0].legend(loc='right')
axs[1].plot(freqs * 1e-9, np.real(muEffArrayzz/mu0))
axs[1].plot(freqs * 1e-9, np.imag(muEffArrayzz/mu0))
axs[1].plot(freqs * 1e-9, np.abs(muEffArrayzz/mu0))
ymin, ymax = axs[1].get_ylim()
axs[1].vlines(FMR, ymin, ymax, color='red', linestyle='--', label='FMR frequency')
axs[1].set_ylabel('Susceptibility ()')
axs[1].set_title('zz')
axs[2].plot(freqs * 1e-9, np.real(muEffArrayxx/mu0))
axs[2].plot(freqs * 1e-9, np.imag(muEffArrayxx/mu0))
axs[2].plot(freqs * 1e-9, np.abs(muEffArrayxx/mu0))
ymin, ymax = axs[2].get_ylim()
axs[2].vlines(FMR, ymin, ymax, color='red', linestyle='--', label='FMR frequency')
axs[2].set_ylabel('Susceptibility ()')
axs[2].set_title('xx')
axs[2].set_xlabel('Frequency (GHz)')

# =============================================================================
# Plotting the dispersion relation
# =============================================================================
fig, axs = plt.subplots(2, 1, figsize=(10, 5), sharex=False)
plt.subplots_adjust(hspace=0.4)  # Increase vertical spacing between subplots
fig.suptitle('Dispersion Relation and Group Velocity', fontsize=14)

axs[0].plot(np.real(kEffArray), freqs*1e-9, '.', label='Re(k_eff)')
axs[0].plot(np.imag(kEffArray), freqs*1e-9, '.', label='Im(k_eff)')
axs[0].plot(freqs*2*np.pi/(c/np.sqrt(np.real(epsilonR))), freqs*1e-9, label='light line')
axs[0].axhline(FMR, color='red', linestyle='--', label='FMR frequency')
axs[0].set_ylabel('Frequency (GHz)')
axs[0].set_xlabel('Effective Wavenumber (rad/m)')
axs[0].legend()

vg1 = np.gradient(2 * np.pi * freqs, kEffArray)  # rad/s divided by rad/m gives m/s

axs[1].plot(freqs*1e-9, np.abs(vg1)/1e3, '.')
ymin, ymax = axs[1].get_ylim()
axs[1].vlines(FMR, ymin, ymax, color='red', linestyle='--', label='FMR frequency')
xmin, xmax = axs[1].get_xlim()
axs[1].hlines(c/np.sqrt(np.real(epsilonR))*1e-3, xmin, xmax, color='black', linestyle='--', label='Speed of light')
axs[1].legend(loc='lower right')
axs[1].set_xlabel('Frequency (GHz)')
axs[1].set_ylabel('Group Velocity (µm/ns)')
axs[1].set_yscale('log')
plt.show()


# =============================================================================
# export
# =============================================================================
expRekEff = np.real(kEffArray)  # Prepare for export
expImkEff = np.imag(kEffArray)  # Prepare for export
expFreq = freqs * 1e-9  # Prepare for export

expBmT = np.abs(H_z) * mu0 * 1e3 #Prepare for export
expEkVom = np.abs(E_z) * 1e-3 #Prepare for export
expMx = np.abs(Mx) #Prepare for export
expMz = np.abs(Mz) #Prepare for export
expPhH = np.angle(H_z) #Prepare for export
expPhE = np.angle(E_z) #Prepare for export
expPMx = np.angle(Mx) #Prepare for export
expPMz = np.angle(Mz) #Prepare for export
expR = r_vals * 1e6 #Prepare for export2

expREMuRXX = np.real(muEffArrayxx/mu0)
expIMuRXX = np.imag(muEffArrayxx/mu0)
expAbsMuRXX = np.abs(muEffArrayxx/mu0)

expREMuRXZ = np.real(muEffArrayxz/mu0)
expIMuRXZ = np.imag(muEffArrayxz/mu0)
expAbsMuRXZ = np.abs(muEffArrayxz/mu0)

expREMuRZZ = np.real(muEffArrayzz/mu0)
expIMuRZZ = np.imag(muEffArrayzz/mu0)
expAbsMuRZZ = np.abs(muEffArrayzz/mu0)
