import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

c = 299792458 # m/s Speed of light constatnt
eps0 = 8.854187817e-12 # F/m Vacuum permittivity constant
omegaFMR = 10.8e9 * 2 * np.pi # rad Hz FMR frequency
gamma = 28.02495224e9 * 2 * np.pi # rad Hz/T Gyromagnetic ratio
Ms = 86e3 # A/m Saturation magnetization
theta = 0 # rad Angle between the magnetization and the external field

omega = np.linspace(10.799 * 2 * np.pi * 1e9, 10.801 * 2 * np.pi * 1e9, 1000) # Hz Frequency range
k2 = omega**2 * eps0/c**2 * (-omega)/(omegaFMR-omega) # Wavevector 
k = np.sqrt(abs(k2))

# # implicit equation
# def solve_k(omega_val):
#     def equation(k):
#         return omegaFMR - omega_val - gamma * Ms * np.cos(theta) * (eps0 * omega_val**2 / c**2) / (k**2 + eps0 * omega_val**2 / c**2)
#     kInit = np.sqrt(abs(omega_val**2 * eps0/c**2 * (-omega_val)/(omegaFMR-omega_val))) # Wavevector 
#     k_initial_guess = abs(kInit) # Initial guess for k
#     k_solution = fsolve(equation, k_initial_guess)
#     return k_solution[0]

# k = np.array([solve_k(omega_val) for omega_val in omega])



plt.figure()
plt.plot(k/1e6, omega/2/np.pi/1e9)
plt.ylabel('Frequency (GHz)')
plt.xlabel('Wavevector (rad/µm)')
plt.show()

# Calculate the group velocity
vg = np.gradient(omega, k)

plt.figure()
plt.plot(omega/2/np.pi/1e9, vg/c)
plt.xlabel('Frequency (GHz)')
plt.ylabel('Fraction of light\'s velocity ()')
plt.ylim(0, 1)
plt.show()
