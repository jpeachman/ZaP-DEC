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
rib1 = Electrode('ribbon', ribbon_angle=10.9)
rib2 = Electrode('ribbon', ribbon_angle=15.8)
right_wall = Electrode('wall')

electrodes = [left_wall, ground, reflect, rib1, rib2, right_wall]
positions = np.array([-.15, -0.05,   0, 0.05, 0.3, 0.55])
potentials= np.array([  0,      0, -100,   15,  80, 160])

dz=0.001
domain = Domain(electrodes, potentials, positions, dz=dz, SI=True)

#%% Single test particle

a0 = 10 # degrees, the beam angle

vz = np.sqrt(90*e*2/mp) # Find the trajectory that hits the back wall
vy = vz*np.tan(a0*np.pi/180)

test_particle = [Particle(pos0=(.05,-.15), v0=(vy,vz), ymax=.3, \
                              mass=1, charge=1, SI=True, KEx=False)]

dt = dz/vz # Takes 2 time steps to cross one grid-space
test = ZAP_DEC(test_particle, domain, dt=dt)
test.run(dt*10000, report=True)

    
#%% Plot
fig, ax = plt.subplots(dpi=600, figsize=(10,2))
test.trajectories(ax, kV=False, linewidth=1);

for idx, el in enumerate(electrodes):   
    print(f'Electrode {idx}, ({el.style}), efficiency = {el.efficiency()*100:.1f}%')
    
print(f'Total efficiency: {test.total_efficiency()*100:0.1f}%')
plt.savefig('periodic.pdf', bbox_inches="tight")

#%%
# rib2.map_opacity(plot=True)

#%% Particles for maxwellian (no end wall)
zrange = [-.15, -.05]
yrange = [.2, .2]
ymax = .5
kT = 20 # eV

particles = super_maxwellian(1000, 4e12, yrange, zrange, kT, \
        uz=103000, mass=1, q=1, theta=a0, ymax=ymax, SI=True, beam=False)


#%% Analysis, stop for space charge

test = ZAP_DEC(particles, domain, dt=dt)
test.run(dt*1000)

#%% Plots
fig, ax = plt.subplots(dpi=600, figsize=(10,2))
test.trajectories(ax, kV=False);

potential = domain.phi
cz = domain.z

ax2 = ax.twinx()
ax2.plot(cz, potential, label='Potential, $\phi_z$, eV', color='blue')
ax2.set_ylabel('Potential, $\phi_z$, V')#; ax2.set_ylim(2000,6000)
plt.savefig('zap space charge.pdf', bbox_inches="tight")

#%% Finish

test = ZAP_DEC(particles, domain, dt=dt)
test.run(dt*1000000)

#%% Plots
fig, ax = plt.subplots(dpi=600, figsize=(10,2))
test.trajectories(ax, kV=False);

for idx, el in enumerate(electrodes):   
    print(f'Electrode {idx}, ({el.style}), efficiency = {el.efficiency()*100:.1f}%')
    
print(f'Total efficiency: {test.total_efficiency()*100:0.1f}%')