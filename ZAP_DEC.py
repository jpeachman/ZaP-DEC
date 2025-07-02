"""****************************
OVERVIEW OF THE ZAP_DEC PROGRAM
- ZAP_DEC is used to study DEC from a Flow Z-Pinch with VB collectors.
- This is a 1D PIC code, but particle trjectories are tracked in 2D.
- Tracking 2D trajectories allows visulization of the angular dependency of collection.
- The Z direction is perpendicular to the retarding electric field. Y is perp to Z.
- Finite domain in Z, periodic in Y. Ey=0 everywhere.

*******************************
USING THE CODE:
- A run script should be made to actually run cases. You don't run it from this file.

*******************************
The ZAP_DEC() class:
- This actually runs the analysis, stores inputs and outputs, etc.
- It ties together the Particle(), Domain(), and Electrode() classes
    - A list of Particle() objects is called particles
    - A list of Electrode() objects is called electrodes

List of particles can be created manually, or with one of the functions in Particles.py
    
ZAP_DEC.run() actually runs the PIC algorithm by:
    1) Weighting charge to the domain cells & solving fields using Domain.update()
    2) Push each particle, and check each particle for collection.
    3) Increment time forward.
    
Particle absorption, energy collection:
    Every electrode has some opacity.
        'Screen' has fixed opacity.
        'Collectors' have angular dependent opacity.
        'Wall' (at zmin and zmax, respectively) have 100% opacity.
    
    Every 'particle' is actually superparticle representing many smaller particles.
    When a particle impact an electrode, charge and mass are multiplied by the opacity.
    The energy wasted is equal to the kinetic energy of the particle during impact.
    The energy collected is equal to (q_particle) * (electrode_voltage)
        Note1: Particles hitting Ground at zmin are absorbed with 0% collection.
        Note2: Particles hitting electron reflector collect negative energy!
        
    Each particle object keeps track of:
        - Current kinetic energy, charge, and mass
        - Energy absorbed
        - Energy wasted
    When only 1% of the particle mass remains, the particle is completely lost.

Some visualization functions are here to easily plot the results of a particular case.
"""

import numpy as np
import matplotlib.pyplot as plt

class ZAP_DEC:
    def __init__(self, particles, domain, dt=1):
        domain.update(particles) # Solve for cell potentials given particle positions.
        
        self.particles = np.array(particles) # An array of particle objects
        self.domain = domain
        self.dt = dt
        self.tpoints = [0]
        self.count_absorbed = 0
        
    def run(self, tspan, report=False):
        particles = self.particles
        npart = len(particles) # How many particles are there
        count_absorbed = self.count_absorbed # When this == npart, stop run.
        
        domain = self.domain
        elec_pos = domain.positions
        potentials = domain.potentials
        zmin, zmax = domain.zmin, domain.zmax
        
        dt = self.dt
        t = self.tpoints[-1]
        
        if t >= tspan:
            raise ValueError('Chosen "tspan" has already been simulated. Increase tspan.')
            return
        
        while t <= tspan:
            if t+dt > tspan:
                break
            
            for idx, particle in enumerate(particles):
                if particle.absorbed == True:
                    continue # try the next particle
                    
                if t == 0:
                    particle.pull(dt) # Find vx_old
                
                particle.push(dt)
                z0, z1 = particle.z[-2], particle.z[-1]
                left, right = min(z0,z1), max(z0,z1)
                    
                # Check for collection
                if z1 <= zmin:
                    incident_electrode=[0]
                elif z1 >= zmax:
                    incident_electrode=[-1]
                else:
                    incident_electrode = [elec_idx for elec_idx, \
                                         pos in enumerate(elec_pos) if left < pos < right]
                for elec_idx in incident_electrode:
                    # If no grids are crossed, nothing happens.
                    # If one or more grids are crossed, absorb at each.
                    opacity = domain.electrodes[elec_idx].incident(particle.vy,\
                                                                   particle.vz[-1])
                    absorbed, collected, wasted = particle.collect(opacity,\
                                                                   potentials[elec_idx])
                    if absorbed:
                        count_absorbed += 1
                    domain.electrodes[elec_idx].collected += collected
                    domain.electrodes[elec_idx].wasted += wasted
                    if report:
                        print(f'Particle hit electrode {elec_idx} at angle '
                              f'{particle.angle():.1f} deg.')
                        
            t += dt # Advance time
            self.tpoints.append(t)        
            
            # Advance fields to n+1
            self.domain.update(particles)
            
            # Check collection
            if count_absorbed == npart:
                print(f'All particles absorbed at t={t}')
                break
        
        self.count_absorbed = count_absorbed
    
    # methods below plot results for a given ZAP_DEC object
    def trajectories(self, ax, kV=False, linewidth=0.1):
        particles = self.particles
        n = len(particles)
        ymax = particles[0].ymax
        zmax = self.domain.zmax
        zmin = self.domain.zmin
        
        # Unique color for each particle:
        colors = plt.cm.viridis(np.linspace(0, .9, n)) #tab10
        idx = 0
        
        for particle in particles:
            particle.trajectory(ax, color=colors[idx], label=idx, linewidth=linewidth)
            idx += 1
        
        self.draw_domain(ax, ymax, kV=kV)
        ax.set_xlim(zmin,zmax); ax.set_xlabel('z\' (meters)')
        
        ax.set_ylim(0,ymax); #ax.set_ylabel('y')
        ax.set_yticks([])
        
    def fields(self, ymax='particle', figsize=(10,3), rightspine=1.1):
        
        if ymax == 'particle':
            ymax = self.particles[0].ymax
            # User can override in case there are no particles (vacuum field)
            
        # Plot the positions of each particle, the potential, and the E field
        pz = np.array([particle.z[-1] for particle in self.particles])
        py = np.array([particle.y[-1] for particle in self.particles])
        
        cz = self.domain.z
        potential = self.domain.phi
        E = self.domain.E
        
        # Particle positions
        fig, ax1 = plt.subplots(dpi=600, layout='constrained', figsize=figsize)
        ax1.plot(pz, py, label='Particles', color='orange', marker='o', linewidth=0)
        ax1.set_xlabel('z')#; ax1.set_xticks([1,2,3,4]); ax1.grid('True')
        ax1.set_ylabel('y'); ax1.set_ylim(0,ymax)

        # Potential
        ax2 = ax1.twinx()
        ax2.plot(cz, potential, label='Potential, $\phi_z$', color='green')
        ax2.spines['right'].set_position(('axes', 1.0)) # adjust right spine position
        ax2.set_ylabel('Potential, $\phi_z$')#; ax2.set_ylim(2000,6000)

        # E-Field
        ax3 = ax1.twinx()
        ax3.plot(cz, E, label='Electric Field, $E_z$', color='blue')
        ax3.plot(cz, np.zeros_like(E), label='E=0', color='blue', linestyle='--')
        ax3.spines['right'].set_position(('axes', rightspine)) # adjust right spine 
        ax3.set_ylabel('Electric Field, $E_z$')#; ax3.set_ylim(2,20)
        
        # Legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines3, labels3 = ax3.get_legend_handles_labels()
        lines = lines1 + lines2 + lines3; labels = labels1 + labels2 + labels3
        ax1.legend(lines, labels, frameon=False, fontsize='small',
                   ncol = 4, bbox_to_anchor=(0.5, -0.1), loc='upper center')
        
        return ax1, ax2, ax3, ymax
    
    def potential_field(self, ax1, ymax, keV=False):
        # Plot the positions of each particle and the potential
        pz = np.array([particle.z[-1] for particle in self.particles])
        py = np.array([particle.y[-1] for particle in self.particles])
        
        cz = self.domain.z
        potential = self.domain.phi
        
        # Potential
        if keV == True:
            ax1.plot(cz, potential/1000, label='Potential, $\phi_z$, keV', color='green')
            ax1.set_ylabel('Potential, $\phi_z$, kV')#; ax2.set_ylim(2000,6000)
        else:
            ax1.plot(cz, potential, label='Potential, $\phi_z$, eV', color='green')
            ax1.set_ylabel('Potential, $\phi_z$, V')#; ax2.set_ylim(2000,6000)
        # ax1.spines['right'].set_position(('axes', 1.0)) # adjust right spine position
        
        
        # Particle positions
        ax2 = ax1.twinx()
        ax2.plot(pz, py, label='Particles', color='blue', \
                 marker='o', markersize=.2, linewidth=0)
        ax2.set_xlabel('z\'')#; ax1.set_xticks([1,2,3,4]); ax1.grid('True')
        ax2.set_ylim(0,ymax)
        
        zmax = self.domain.zmax
        zmin = self.domain.zmin
        ax1.set_xlim(zmin,zmax); ax1.set_xlabel('z\' (meters)') 
        ax2.set_ylim(0,ymax); #ax.set_ylabel('y')
        ax2.set_yticks([])
        
        # Legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines = lines1 + lines2; labels = labels1 + labels2
        ax1.legend(lines, labels, frameon=False, fontsize='small',
                   ncol = 4, bbox_to_anchor=(0.8, -0.2), loc='upper center')
        
        return ax1, ax2
        
    def draw_domain(self, ax, ymax, kV=False):
        """
        Draws the domain and the grids, labeling grids with potentials
        """
        positions  = self.domain.positions
        potentials = self.domain.potentials
        if kV:
            scale=1/1000
        else:
            scale=1
        
        for idx in range(len(positions)):
            style = self.domain.electrodes[idx].style
            if   style=='wall':   linestyle='-'
            elif style=='screen': linestyle=':'
            elif style=='ribbon': linestyle='-.'
            if kV == True:
                label = f'  ${potentials[idx]*scale:+.0f}$ kV'
            else:
                label = f'  ${potentials[idx]*scale:+.0f}$ V'
            ax.plot([positions[idx], positions[idx]], [0, ymax], \
                    linestyle=linestyle, color='black')
            ax.text(positions[idx], ymax, label, fontsize=8,\
                    rotation=90, ha='center')
                
    def total_efficiency(self):
        collected = 0
        total = 0
        for particle in self.particles:
            collected += particle.energy_collected
            total += particle.KE0
        return collected/total