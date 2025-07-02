'''
DT Fusion Case
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
mp = 1.6726e-27 # proton mass
e  = 1.6022e-19 # proton charge

#%% Define eletrodes and domain
left_wall = Electrode('wall')
ground = Electrode('screen')
reflect = Electrode('screen')
rib1 = Electrode('ribbon', ribbon_angle=7.7)
rib2 = Electrode('ribbon', ribbon_angle=9.9)
rib3 = Electrode('ribbon', ribbon_angle=17.1)
right_wall = Electrode('wall')

electrodes = [left_wall, ground, reflect, rib1, rib2, rib3, right_wall]
positions = np.array([-.5, -0.2,   0, 0.2, 1.5, 2.8, 4])
potentials= np.array([  0,     0, -10, 50,  150,  250,  350])*1000 # Volts

dz=0.01
domain = Domain(electrodes, potentials, positions, dz=dz, SI=True)

#%% Test particle
a0 = 7 # degrees, the beam angle
W0 = .200e6 * e # Twice the max grid voltage bc Z=2 for alphas

vz = np.sqrt(2*W0/(4*mp)) # Find the trajectory that hits the back wall
vy = vz*np.tan(a0*np.pi/180)

test_particle = [Particle(pos0=(.05,-.15), v0=(vy,vz), ymax=1, \
                              mass=4, charge=2, SI=True, KEx=False)]

dt = dz/(2*vz) # Takes 2 time steps to cross one grid-space
test = ZAP_DEC(test_particle, domain, dt=dt)
test.run(dt*10000, report=True)

#%% Plot test particle
fig, ax = plt.subplots(dpi=600, figsize=(10,2))
test.trajectories(ax, kV=True, linewidth=1);

for idx, el in enumerate(electrodes):   
    print(f'Electrode {idx}, ({el.style}), efficiency = {el.efficiency()*100:.1f}%')
    
print(f'Total efficiency: {test.total_efficiency()*100:0.1f}%')
# plt.savefig('periodic.pdf', bbox_inches="tight")

#%% Particles 
zrange = [-.4, -.25]
yrange = [.25, .25]
ymax = 2.5
kT = .7e6 # eV was 25.6

# Dpart = super_maxwellian(450, 1.43E+12, yrange, zrange, kT, \
#         uz=4e6, mass=2, q=1, theta=a0, ymax=ymax, SI=True, beam=True)

# Tpart = super_maxwellian(450, 1.09e12, yrange, zrange, kT, \
#         uz=3.27e6, mass=3, q=1, theta=a0, ymax=ymax, SI=True, beam=True)

# (N, density, yrange, zrange, kT, uz, mass=1, q=1, theta=0, ymax=[], SI=False, beam=False)


#!!!! I USED LOWER DENSITY!!!
Hepart = super_maxwellian(100, 3.5e10, yrange, zrange, kT, \
        uz=3.9e6, mass=4, q=2, theta=a0, ymax=ymax, SI=True, beam=False)

particles = Hepart  #Dpart+Tpart+


#%% Analysis
vmax = np.max([particle.vz for particle in particles])
dt = dz/(2*vmax)

test = ZAP_DEC(particles, domain, dt=dt)
test.run(dt*10000)

#%% Plot
fig, ax = plt.subplots(dpi=600, figsize=(10,2))
test.trajectories(ax, kV=True);
# plt.savefig("benchmark beam.pdf", bbox_inches="tight")

for idx, el in enumerate(electrodes):   
    print(f'Electrode {idx}, ({el.style}), efficiency = {el.efficiency()*100:.1f}%')
    
print(f'Total efficiency: {test.total_efficiency()*100:0.1f}%')