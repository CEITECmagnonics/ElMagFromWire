
# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import SpinWaveToolkit as SWT

# %matplotlib ipympl

# Define parameters
k=np.linspace(10,10e6,1000) # Wavenumber region (rad/m)
Bext=25e-3                  # External magnetic field (T)
d= 95e-9                   # Thin film thickness (m)
# material=SWT.NiFe           # material (imported from SpinWaveToolkit)
material=SWT.Material(Ms=747e3, Aex=8.1e-12, alpha=70e-4, gamma=30.4*2* np.pi*1e9) #Ms -BLS fitting Mie_PPSW_02, Aex, gamma - Krcma et. al., alpha - SWT
N=3                         # Number of modes to display (N=3 will give n=0,1,2)
angle=np.deg2rad(45)        # Specific angle (only if you're not plotting for BV or DE) (rad)

# Calculate the DispersionCharacteristic, SingleLayer
DispChar_BV=SWT.SingleLayer(kxi = k, theta = np.pi/2, phi = 0, d = d, boundary_cond = 1, Bext = Bext, material = material)
DispChar_DE=SWT.SingleLayer(kxi = k, theta = np.pi/2, phi = np.pi/2, d = d, boundary_cond = 1, Bext = Bext, material = material)
DispChar_FV=SWT.SingleLayer(kxi = k, theta = 0, phi = 0, d = d, boundary_cond = 1, Bext = Bext, material = material)
DispChar_angle=SWT.SingleLayer(kxi = k, theta = np.pi/2, phi = angle, d = d, boundary_cond = 1, Bext = Bext, material = material)

DispChar_pin=SWT.SingleLayer(kxi = k, theta = np.pi/2, phi = 0, d = d, boundary_cond = 4, Bext = Bext, material = material, dp=1e7)

# Calculate and plot the dispersion
fontsize=11
fig, ax = plt.subplots()
for i in range(N):
    Disp_BV=DispChar_BV.GetDispersion(n=i)  # Calculate the Backward Volume dispersion
    Disp_DE=DispChar_DE.GetDispersion(n=i)  # Calculate the Damon Eshbach dispersion
    Disp_FV=DispChar_FV.GetDispersion(n=i)  # Calculate the Forward Volume dispersion
    Disp_angle=DispChar_angle.GetDispersion(n=i) # Calculate the angle dispersion

    Disp_pin=DispChar_pin.GetDispersion(n=i)  # Calculate the partially pinned dispersion
    
    ax.plot(k*1e-6,Disp_BV/2/np.pi*1e-9 ,label="BV n="+str(i))         # Plot the Backward Volume dispersion
    # ax.plot(k*1e-6,Disp_DE/2/np.pi*1e-9 ,label="DE n="+str(i))       # Plot the Damon Eshbach dispersion
    # ax.plot(k*1e-6,Disp_FV/2/np.pi*1e-9 ,label="FV n="+str(i))       # Plot the Forward Volume dispersion
    # ax.plot(k*1e-6,Disp_angle/2/np.pi*1e-9 ,label="angle n="+str(i)) # Plot the angle dispersion 

    # ax.plot(k*1e-6,Disp_pin/2/np.pi*1e-9 ,label="part. pinned n="+str(i))       # Plot the partially pinned dispersion
    
    # ax.set_xlim(0,max(k)*1e-6)
    # ax.set_ylim(ymin=2)

    ax.set_xlabel('Wavenumber (rad$\cdot$µm$^{{-1}}$)', fontsize=fontsize)
    ax.set_ylabel("Frequency (GHz)", fontsize=fontsize)
    ax.legend(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)  # X-axis tick labels font size
    plt.yticks(fontsize=fontsize)