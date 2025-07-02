'''
DT Fusion case per takagaki

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
rib1 = Electrode('ribbon', ribbon_angle=7.4)
rib2 = Electrode('ribbon', ribbon_angle=9.1)
rib3 = Electrode('ribbon', ribbon_angle=12.8)
right_wall = Electrode('wall')

electrodes = [left_wall, ground, reflect, rib1, rib2, rib3, right_wall]
positions = np.array([-.5, -0.24,   0, 0.13, 1.39, 2.65, 3.91])
potentials= np.array([  0,     0, -20, 100, 400, 700, 1000])*1000 # Volts

dz=0.01
domain = Domain(electrodes, potentials, positions, dz=dz, SI=True)

#%% Single test particle
a0 = 7 # degrees, the beam angle

W0 = 2000*1000*e
vz = np.sqrt(W0*2/(4*mp)) # Find the trajectory that hits the back wall
vy = vz*np.tan(a0*np.pi/180)

test_particle = [Particle(pos0=(.05,-.15), v0=(vy,vz), ymax=.3, \
                              mass=4, charge=2, SI=True, KEx=False)]

dt = dz/(2*vz) # Takes 2 time steps to cross one grid-space
test = ZAP_DEC(test_particle, domain, dt=dt)
test.run(dt*10000, report=True)

#%% Plot
fig, ax = plt.subplots(dpi=600, figsize=(10,2))
test.trajectories(ax, kV=True, linewidth=1);

for idx, el in enumerate(electrodes):   
    print(f'Electrode {idx}, ({el.style}), efficiency = {el.efficiency()*100:.1f}%')
    
print(f'Total efficiency: {test.total_efficiency()*100:0.1f}%')
plt.savefig('periodic.pdf', bbox_inches="tight")

#%% Particles 
zrange = [-.5, -.24]
yrange = [.25, .25]
ymax = 2.5
a0 = 7 # degrees, the beam angle
kTalpha = 700*1000 # eV was 25.6
kTother = 10*1000 # eV was 25.6

Dpart = super_maxwellian(100, 2E19, yrange, zrange, kTother, \
        uz=78480, mass=2, q=1, theta=a0, ymax=ymax, SI=True, beam=True)

Tpart = super_maxwellian(100, 2E19, yrange, zrange, kTother, \
        uz=78480, mass=3, q=1, theta=a0, ymax=ymax, SI=True, beam=True)
    
Hepart = super_maxwellian(800, 2.03E10, yrange, zrange, kTalpha, \
        uz=3924000, mass=4, q=2, theta=a0, ymax=ymax, SI=True, beam=True)

particles = Dpart+Tpart#+Hepart


#%% Analysis
# dt = dz/6e6/2
# dt = .000001 # Need to figure out for SI units
test = ZAP_DEC(particles, domain, dt=dt)
test.run(dt*1000)

#%% Plot
fig, ax = plt.subplots(dpi=600, figsize=(10,2))
test.trajectories(ax, kV=True);
# plt.savefig("benchmark beam.pdf", bbox_inches="tight")

for idx, el in enumerate(electrodes):   
    print(f'Electrode {idx}, ({el.style}), efficiency = {el.efficiency()*100:.1f}%')
    
print(f'Total efficiency: {test.total_efficiency()*100:0.1f}%')