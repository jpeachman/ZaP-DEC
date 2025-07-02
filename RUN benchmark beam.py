'''
Benchmark beam case

'''

#%% Header
# My files:
from ZAP_DEC import ZAP_DEC
from particles import Particle, super_maxwellian
from domain import Domain
from electrodes import Electrode

# Packages:
import numpy as np
import matplotlib.pyplot as plt

#%% Constants
# mp = 1.6726e-27 # proton mass
# e  = 1.6022e-19 # proton charge

#%% Define eletrodes and domain
left_wall = Electrode('wall')
ground = Electrode('screen')
reflect = Electrode('screen')
rib1 = Electrode('ribbon', ribbon_angle=12)
rib2 = Electrode('ribbon', ribbon_angle=13.2)
rib3 = Electrode('ribbon', ribbon_angle=14.9)
right_wall = Electrode('wall')

electrodes = [left_wall, ground, reflect, rib1, rib2, rib3, right_wall]
positions = np.array([-.5, -0.24,   0, 0.13, 1.39, 2.65, 3.91])
potentials= np.array([  0,     0, -14, 94.4,  132,  180,  243])*1000 # Volts

dz=0.01
domain = Domain(electrodes, potentials, positions, dz=dz, SI=True)

#%% Particles 
zrange = [-.5, -.24]
yrange = [.25, .25]
ymax = 2.5
a0 = 7 # degrees, the beam angle
kT = 19.25*1000 # eV was 25.6

Dpart = super_maxwellian(450, 1.43E+12, yrange, zrange, kT, \
        uz=4e6, mass=2, q=1, theta=a0, ymax=ymax, SI=True, beam=True)

Tpart = super_maxwellian(450, 1.09e12, yrange, zrange, kT, \
        uz=3.27e6, mass=3, q=1, theta=a0, ymax=ymax, SI=True, beam=True)
    
Hepart = super_maxwellian(100, 2.03e10, yrange, zrange, kT, \
        uz=4e6, mass=4, q=2, theta=a0, ymax=ymax, SI=True, beam=True)

particles = Dpart+Tpart+Hepart


#%% Analysis
dt = dz/6e6/2
# dt = .000001 # Need to figure out for SI units
test = ZAP_DEC(particles, domain, dt=dt)
test.run(dt*10000)

#%% Plot
fig, ax = plt.subplots(dpi=600, figsize=(10,2))
test.trajectories(ax, kV=True);
# plt.savefig("benchmark beam.pdf", bbox_inches="tight")

for idx, el in enumerate(electrodes):   
    print(f'Electrode {idx}, ({el.style}), efficiency = {el.efficiency()*100:.1f}%')
    
print(f'Total efficiency: {test.total_efficiency()*100:0.1f}%')