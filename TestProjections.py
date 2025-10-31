import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

phi=np.linspace(0, 2*np.pi, 1000)
theta=np.linspace(0, np.pi, 1000)
Ku1 = 0.1  # Example value for Ku1
Ku2 = 0  # Example value for Ku2
phi_grid, theta_grid = np.meshgrid(phi, theta)
E = Ku1 * np.cos(phi_grid) * np.cos(theta_grid) + Ku2 * np.sin(theta_grid)**2 * np.cos(phi_grid)**2


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Convert spherical coordinates to Cartesian for plotting
x = np.sin(theta_grid) * np.cos(phi_grid)
y = np.sin(theta_grid) * np.sin(phi_grid)
z = np.cos(theta_grid)

# Adjust the radius based on the energy density E
r = E / np.max(E)  # Normalize E to use as a scaling factor for the radius
x_scaled = abs(r) * x
y_scaled = abs(r) * y
z_scaled = abs(r) * z

# Plot the surface with the scaled coordinates
surf = ax.plot_surface(x_scaled, y_scaled, z_scaled, facecolors=plt.cm.viridis(E / np.max(E)), edgecolor='none')

# Add color bar for energy density
m = plt.cm.ScalarMappable(cmap='viridis')
m.set_array(E)
plt.colorbar(m, ax=ax, label='Energy Density')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Energy Density Distribution in Spherical Coordinates')
plt.show()

