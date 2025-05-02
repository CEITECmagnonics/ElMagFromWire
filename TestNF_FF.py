import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.special import hankel2

# =============================================================================
# Physical constants and parameters
# =============================================================================
c = 3e8                      # Speed of light in vacuum (m/s)
mu0 = 4 * np.pi * 1e-7       # Permeability of free space (H/m)
epsilon0 = 1 / (mu0 * c**2)    # Permittivity of free space (F/m)

# RF source parameters
f = 10e9                      # Frequency in Hz (10 GHz here)
omega = 2 * np.pi * f         # Angular frequency (rad/s)
k = omega / c                 # Free-space wavenumber (rad/m)

I0 = 75e-3                    # Current amplitude (in amperes)

# =============================================================================
# Field expressions for an infinitely long wire (line source)
# =============================================================================
# For an infinite line source, the phasor expressions are:
#
#   Magnetic vector potential: A_z(r) = (μ0 I0 / 4) * H₀^(2)(k r)
#   Electric field: E_z(r) = -i ω A_z(r)   (assuming phasor exp(+iωt) convention)
#   Magnetic field: H_φ(r) = (I0 k / 4) * H₁^(2)(k r)
#
# These expressions capture both reactive (near-field) and radiative (far-field)
# behavior through the Hankel functions.

# Define a range of radial distances (avoid r = 0 to prevent singularity)
r_vals = np.logspace(-6, -1, 10000)  # from 1 nm to 0.1 m

# Compute the phasor fields (complex values)
A_z = (mu0 * I0 / 4) * hankel2(0, k * r_vals)
E_z = -1j * omega * A_z
H_phi = (I0 * k / 4) * hankel2(1, k * r_vals)

# =============================================================================
# Animation: time evolution of the fields
# =============================================================================
# For the animation we will show:
# - The real part of H_phi (scaled to mT) versus radial distance
# - The real part of E_z (in V/m) versus radial distance
# Both fields are computed as the real parts of the phasor multiplied by the
# time-varying factor exp(iωt), so that:
#     H_phi_time(r, t) = Re{ H_phi(r) * exp(iωt) }
#     E_z_time(r, t)   = Re{ E_z(r) * exp(iωt) }

# Set up the figure with two subplots (sharing the x-axis)
fig, axs = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

# Initialize the plot for H_phi
# Multiply by mu0 and 1e3 to convert H_phi to mT for a more convenient scale.
line1, = axs[0].plot(r_vals * 1e6, np.abs(H_phi * np.exp(1j * 0)) * mu0 * 1e3, 
                       label=r'$H_\phi$ (mT)')
axs[0].set_ylabel('Magnetic Field (mT)')
axs[0].set_title('Time Evolution of Fields from an Infinitely Long Antenna')
axs[0].grid(True, which="both", linestyle="--", alpha=0.7)
axs[0].set_xscale('log')  # Log scale for better visibility of near-field effects
axs[0].legend()
axs[0].set_ylim(-20, 20)  # Set y-limits for better visibility

# Initialize the plot for E_z
line2, = axs[1].plot(r_vals * 1e6, np.abs(E_z * np.exp(1j * 0)), 
                       label=r'$E_z$ (V/m)')
axs[1].set_xlabel('Radial Distance r (µm)')
axs[1].set_ylabel('Electric Field (V/m)')
axs[1].grid(True, which="both", linestyle="--", alpha=0.7)
axs[1].set_xscale('log')  # Log scale for better visibility of near-field effects
axs[1].legend()
axs[1].set_ylim(-10000, 10000)  # Set y-limits for better visibility

# Add a text element to display the current time in the first plot
time_text = axs[0].text(0.05, 0.9, '', transform=axs[0].transAxes, fontsize=10)

# Define animation parameters
nframes = 200               # Number of frames per period of oscillation
T = 1 / f                   # Period of the RF signal (seconds)
dt = T / nframes            # Time step for each frame

# The update function computes the instantaneous field values for a given frame.
def update(frame):
    t = frame * dt  # current time
    # Multiply the phasor fields by exp(iωt) to get the time-domain values.
    H_current = np.real(H_phi * np.exp(1j * omega * t))
    E_current = np.real(E_z * np.exp(1j * omega * t))
    # Update the plotted y-data for each field
    line1.set_ydata(H_current * mu0 * 1e3)  # scale to mT for H_phi
    line2.set_ydata(E_current)
    # Update time annotation (using scientific notation because the period is very short)
    time_text.set_text('Time = {:.2e} s'.format(t))
    return line1, line2, time_text

# Create the animation
# ani = animation.FuncAnimation(fig, update, frames=nframes, interval=50, blit=True)

# plt.tight_layout()
# plt.show()

# # =============================================================================
# # Exporting the Animation
# # =============================================================================
# # To export/save the animation as a video file (e.g., MP4), uncomment and run the following:
# #
# #   ani.save('fields_animation.mp4', writer='ffmpeg', fps=30)
# #
# # Alternatively, to save as a GIF (if Pillow is installed), you can use:
# #
# ani.save('fields_animation.gif', writer='pillow', fps=30)
# #
# # Note: The export requires that you have ffmpeg or pillow installed and available.