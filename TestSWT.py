import numpy as np
import SpinWaveToolkit as SWT

kxi = np.linspace(1e-6, 150e6, 150)

PyChar = SWT.SingleLayer(Bext=20e-3, kxi=kxi, theta=np.pi/2,
                         phi=np.pi/2, d=30e-9, weff=2e-6,
                         boundary_cond=2, material=SWT.NiFe)
DispPy = PyChar.GetDispersion()*1e-9/(2*np.pi)  # GHz
vgPy = PyChar.GetGroupVelocity()*1e-3  # km/s
lifetimePy = PyChar.GetLifetime()*1e9  # ns
decLen = PyChar.GetDecLen()*1e6  # um