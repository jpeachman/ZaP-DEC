#%% Header
# My files:
from ZAP_DEC import ZAP_DEC
from particles import Particle#, rand_particles
from domain import Domain
from electrodes import Electrode

# Packages:
import numpy as np
import matplotlib.pyplot as plt

#%% Define eletrodes and domain
a0 = 7 # degrees, the beam angle

ribbon_angle = [12, 13.2, 14.9]
rib1 = Electrode('ribbon', ribbon_angle=ribbon_angle[0])
rib2 = Electrode('ribbon', ribbon_angle=ribbon_angle[1])
rib3 = Electrode('ribbon', ribbon_angle=ribbon_angle[2])

screen = Electrode('screen')
wall = Electrode('wall')

electrodes = [wall, screen, screen, rib1, rib2, rib3, wall]
positions = np.array([-.5, -0.24,   0, 0.13, 1.39, 2.65, 3.91])
potentials= np.array([  0,     0, -14, 94.4,  132,  180,  243])*1000 # Volts

dz=0.01
domain = Domain(electrodes, potentials, positions, dz=dz, SI=True)

#%% Particles 
ymax = 2.2
mp = 1.6726e-27 # proton mass
e  = 1.6022e-19 # proton charge
energies = np.array([94+132, 132+180, 180+243, 1000])*1000*e/2
vz = np.sqrt(2*energies/mp)
vy = vz*np.tan(a0*np.pi/180)
particles = []
for j in range(len(energies)):
    particles.append(Particle(pos0=(.1,-.4), v0=(vy[j],vz[j]), ymax=ymax, SI=True))

#%% Analysis
dt = dz/np.max(vz)/2
# dt = .000001 # Need to figure out for SI units
test = ZAP_DEC(particles, domain, dt=dt)
test.run(dt*5000)

#%% Plot
fig, ax = plt.subplots(dpi=600, figsize=(10,2))
test.trajectories(ax, kV=True);

#%% Outputs

for j, particle in enumerate(particles):
    print(f'\nParticle {j}:')
    particle.remaining()
    particle.eff()