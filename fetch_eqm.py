'''This script demonstrates some ways to fetch AUG equilibria in Python. Other ways may be preferred by others, depending on their application of interest.

'''
from omfit_classes.omfit_eqdsk import OMFITgeqdsk
import matplotlib.pyplot as plt
plt.ion()
import aug_sfutils as sf
import numpy as np

shot = 40470
time_s = 4.
eqm_diag = 'EQI'  # name of equilibrium shotfile

# ------- Method #1: use tools in the aug_sfutils package
# e.g. plot flux surface contours (from G. Tardini)
# see more examples at https://www.aug.ipp.mpg.de/~git/aug_sfutils/
equ = sf.EQU(shot, diag=eqm_diag)
rhop = [0.1, 1]
r2, z2 = sf.rho2rz(equ, rhop, t_in=3, coord_in='rho_pol')
n_theta = len(r2[0][-1])
theta = np.linspace(0, 2*np.pi, 2*n_theta)
r1, z1 = sf.rhoTheta2rz(equ, rhop, theta, t_in=3, coord_in='rho_pol')

plt.figure(1, figsize=(10, 7))
for jrho, rho in enumerate(rhop):
    plt.subplot(1, len(rhop), jrho + 1, aspect='equal')
    plt.title(r'$\rho$ = %8.4f' %rho)
    plt.plot(r2[0][jrho], z2[0][jrho], 'go', label='rho2rz')
    plt.plot(r1[0, :, jrho], z1[0, :, jrho], 'r+', label='rhoTheta2rz')
    plt.legend()

# --------- Method #2: use combination of aug_sfutils and OMFITgeqdsk
g = OMFITgeqdsk("").from_aug_sfutils(shot=shot, time=time_s, eq_shotfile=eqm_diag)

plt.figure();
plt.plot(g['AuxQuantities']['R'], g['AuxQuantities']['PSI'])
plt.xlabel('R [m]')
plt.ylabel(r'\Psi')

# change cocos, if needed
new_cocos = 5
g = g.cocosify(new_cocos,True,True)

# save gEQDSK file
g.filename = f'geqdsk_{shot}_{int(time_s*1000)}.cocos{new_cocos}'
g.save(raw=True)

# visualize equilibrium at chosen time
g.plot()
