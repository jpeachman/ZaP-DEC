'''
The domain is the computational grid on which the PIC analysis runs.
Particles have their own domain in z and y, and the z domains should match.
Since collectors can also be called "grids", I use the words "cells, nodes, and domain".

To run a Particle-in-cell analysis, we need cells. We weight particle charge to the cells,
then solve for the electric field between the nodes. A cell is the space between nodes.

Different regions of the domain are calculated differently:
    Z < 0: Quasineutral Zone 
        This is where particles are normally spawned. Only ions are simulated directly. 
        A neutralizing background of electrons is sim'd by subtracting avg charge density.
        
    Z > 0: Space Charge Zones
        Electrons are screened from these zones, so space charge effects are possible.
    
    Zones are seperated by electrodes. 
    For example (See Fig 10 & 13 of [1]):
        'Ground' at zmin (sets reference voltage for the system)
        'Electron-reflector' at z=0 (Screen electrode with negative voltage)
        'Collector(s)': Electrodes at z>0 (positive voltage)
        'Wall': at zmax (positive voltage)
        
    This class only calculates fields in the presence of particles.
    Particle collisions are calculated by ZAP_DEC after particle.push().
    
    [1]Barr, W. L., R. J. Burleigh, W. L. Dexter, R. W. Moir, and R. R. Smith. 
       “A Preliminary Engineering Design of a ‘Venetian Blind’ Direct Energy Converter 
       for Fusion Reactors.” IEEE Transactions on Plasma Science 2, no. 2 (June 1974)
       https://doi.org/10.1109/TPS.1974.6593737.
'''
#%% Header
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve, splu

class Domain:
    def __init__(self, electrodes, potentials=[0, -10, 100], \
                 positions=[-1, 0, 1], dz=0.1, SI=False):
        # 'electrodes' is a list of Electrode class objects. Stores opacity information.
        # 'positions'  is a vector of z-coordinates for each corresponding electrode. 0 is required.
        # 'potentials' is the 'voltage' of each corresponding electrode.
        # dz is the spacing of cell nodes on the computational domain.
        
        # Check inputs for errors:
        if not all([len(electrodes) == len(potentials), \
                    len(electrodes) == len(positions)]):
            raise ValueError('Each electrode must have a position and potential.')
            
        if not all([electrodes[0].style == 'wall', electrodes[-1].style == 'wall']):
            raise ValueError('The first and last electrodes must be walls.')
                
        if 0 not in positions:
            raise ValueError('One electrode must be located at z=0. '
                             'This should be the electron reflector.')
            
        if any(pos+dz in positions for pos in positions):
            raise ValueError('Minimum seperation between electrodes is 2*dz')
        
        # Initialize cell nodes globally
        zmin, zmax = positions[0], positions[-1]  # These are domain endpoints
        z = np.arange(zmin, zmax+dz, dz)
        z = np.round(z, decimals=8) # Stupid floating point errors
        
        if dz != np.round(z[1] - z[0], decimals=8):
           raise Exception('Linspace did not match user-specified dz')
        
        if not all(pos in z for pos in positions):
            raise ValueError("Every electrode position must be on the computational grid "
                             "(divisible by dz).")
        
        # Define each zone and make laplacian operator for each:
        nzones = (len(positions)-1)
        zones = np.zeros((nzones,2), dtype=int)
            # zones is a 2D array where each row reads 
                # [zone number, left z index, right z index]
            # Each z-index corresponds to the electrode position at the zone boundary
        LU = [] 
            # List of LU objects. These are LU decompositions of laplacian for each zone.
        
        for i in range(nzones):
            zones[i,0] = np.argwhere(z==positions[i])[0,0]
            zones[i,1] = np.argwhere(z==positions[i+1])[0,0]
            m = zones[i,1]-zones[i,0]+1 # Number of points in the zone
            
            d1 = np.ones(m) # A diagonal of 1s
            A = spdiags([d1, -2*d1, d1],
                        [-1,     0,  1], (m-2,m-2), format='csc') 
            # Size is m-2 because grid potentials are known.

            A /= dz**2
            LU.append(splu(A)) # I can reuse this many times.
        
        # Store attributes
        self.electrodes = electrodes
        self.potentials = potentials
        self.positions = positions
        self.z, self.dz = z, dz
        self.zmin, self.zmax = zmin, zmax
        
        self.nzones = nzones
        self.zones = zones
        self.LU = LU
        self.SI = SI
        

    #%% Update function
    def update(self, particles):
        """
        This method takes the following steps:
            1) Weights charge from particles to the grid
            2) Electric field solve on grid (piecewise by zone)
            3) Weights Electric field to the particles

        Input is a list of particles of Particle class.
        """
        # =============================================================================
        # 1) WEIGHT TO GRID
        # =============================================================================
        
        zCells = self.z   # Positions of grid points
        rho = np.zeros_like(zCells) # Charge density at each gridpoint
        dz = self.dz
        
        for particle in particles:
            zPart = particle.z[-1] # Current position of the particle
            q = particle.q         # Current charge of the particle
            # Find the indexes above and below
            idx_above = np.searchsorted(zCells, zPart)
            idx_below = idx_above-1

            # If particle is beyond highest grid point, it must have been collected
            if idx_above > len(zCells)-1:
                # No charge to weight here
                # print('Warning: Particle which should have been collected is off grid')
                return

            # If particle is exactly on a gridpoint, idx_above will match
            # If exactly on the left endpoint, idx_below will be -1
            if zCells[idx_above] == zPart:
                rho[idx_above] += q/dz
                break
            
            # Weight charge
            a = zPart - zCells[idx_below] # This is why I needed 'if' just above
            b = dz-a
            
            rho[idx_below] += q*b/(dz**2)
            rho[idx_above] += q*a/(dz**2)        

        # Add neutralizing background charge for z<0
        idx_z0 = np.argwhere(zCells==0)[0,0] # Corresponds to z=0
        if idx_z0 > 2: # Doesn't make sense otherwise...
            rho[1:idx_z0] -= np.mean(rho[1:idx_z0]) 
            # I exclude charge weighted to the endpoints (electrodes)
        
        self.rho = rho # Store this for debugging purposes.
        
        # =============================================================================
        # FIELD SOLVE
        # =============================================================================
        # Electric potential (phi) solve: ∇^2(phi) = -rho
        phi = np.array([self.potentials[0]])
        if self.SI:
            E0 = 8.85419e-12 # F/M
        else:
            E0 = 1
        
        for i in range(self.nzones):
        
            pleft, pright = self.potentials[i], self.potentials[i+1]
            idx_left, idx_right = self.zones[i,0], self.zones[i,1]
            
            b = -rho[idx_left+1:idx_right]/E0 # Truncate the end points (potential known) 
            b[0]  -= pleft/dz**2   # RHS of poissons eq for boundary points
            b[-1] -= pright/dz**2 
            phi = np.append(phi, self.LU[i].solve(b)) # potentials at interior points
            phi = np.append(phi, pright) # potential at right side of domain
        
        self.phi = phi # Save the potentials
        
        """
        Electric field solve: E=−∇phi
        np.gradient uses central difference on interior points, fwd/back diff at ends
        """
        E = -np.gradient(phi,dz) # Takes the gradient with spacing dz
        self.E = E
        
        # =============================================================================
        # WEIGHT ELECTRIC FIELD TO PARTICLES
        # =============================================================================
        
        for particle in particles:
            zPart = particle.z[-1] # Current position of the particle
            # Find the indexes above and below
            idx_above = np.searchsorted(zCells, zPart)
            idx_below = idx_above-1
        
            # If particle is beyond highest grid point, it is between x[-1] and x[0]
            if idx_above > len(zCells)-1:
                raise ValueError('Particle is outside of the grid')
        
            # If particle is exactly on a gridpoint, idx_above will match
            # If exactly on the left endpoint, idx_below will be -1
            if zCells[idx_above] == zPart:
                particle.E = E[idx_above]
                break
            
            # Weight field
            a = zPart - zCells[idx_below]
            b = self.dz-a
            
            particle.E  = E[idx_below]*b/dz
            particle.E += E[idx_above]*a/dz