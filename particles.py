import numpy as np
# import matplotlib.pyplot as plt

empty = np.array([np.nan, np.nan]) # Used to fill vectors in various places

class Particle: # Track a particle's motion in response to forces
    """
    Initial position (pos0) and velocity (v0) are tuples or vectors: (y,z), (vy, vz).
    The y dimension is periodic, from 0 to ymax [m]. 
    There are no forces in y, so vy is constant always. These can be superparticles. 
    
    Mass and charge are normalized to proton mass and e, respectively.
    If SI=True, mass and charge get modified internally (specify # nucleons and # protons)

    KEx accounts for energy loss in the x dimension by doubling energy loss in y.
        Keep off for beams. Consider using for maxwellians.
    """
    def __init__(self, pos0=(0,0), v0=(1,1), ymax=1, \
                 mass=1, charge=1, SI=False, KEx=False):
        if len(pos0)!=2 or len(v0)!=2:
            raise ValueError('pos0 (y,z) and v0 (vy,vz) must have length 2.')    
        
        self.z = [pos0[1]]
        self.y = [pos0[0]]
        
        self.vy = v0[0]   # Always constant
        self.vz = [v0[1]] # List of vz
        
        self.ymax = ymax

        if SI:
            mass   *= 1.6726e-27 # proton mass
            charge *= 1.6022e-19 # proton charge
            
        self.m, self.m0 = mass, mass # Current mass and initial mass
        self.q = charge
        self.E = 0 # V/m, Electric field felt by particle in z direction
        
        if KEx:
            self.KE0 = .5*mass*( v0[1]**2 + 2*v0[0]**2 ) 
            # Initial KE of particle. vx estimated.
        else:
            self.KE0 = .5*mass*( v0[1]**2 + v0[0]**2 ) 
            # Initial KE of particle. vx is not accounted for.
        
        self.cross = [empty]           # History of when and where I cross y boundaries.
                                       # Contains coords of crossing over last timestep.
                                       # nan, nan means no crossing
        """
        Track whether this particlar particle has been collected or lost (absorbed)
        Electric energy collected
        """
        self.absorbed=False
        self.energy_collected=0
      
    def pull(self, dt):
        # Integrate backwards by half of a timestep to find vx_old
        # This only needs to happen when vx_old is unknown
        # NOTE: Use same dt as push(), the 1/2 is added in the function!
        vz = self.vz[-1]
        accel = self.E*self.q/self.m
        self.vz_old = vz - dt/2*accel
    
    def push(self, dt):
        # if self.absorbed == True:
        #     return # This code was redundant with ZAP_DEC
            
        y1, z1 = self.y[-1], self.z[-1]
        
        # Leapfrog velocity, then position by 1 timestep. (They are 1/2 step apart)
        accel = self.E*self.q/self.m
        vz_new = self.vz_old + dt*accel
        y2 = y1 + self.vy*dt
        z2 = z1 + vz_new*dt

        # Periodic BCs (y only)
        ymax = self.ymax # Note: ymin is always 0
        
        if y2 >= ymax:
            f = (ymax-y1)/(y2-y1)
            zc = f*(z2-z1)+z1 # This is where the trajectory crossed the boundary
            cross = np.array([ymax,zc]) # 1 means particle entered ymax boundary
        elif y2 < 0:
            f = -y1/(y2-y1)
            zc = f*(z2-z1)+z1
            cross = np.array([0,zc]) # 0 means particle entered ymin boundary
        else:
            cross = empty # no crossing
        
        y2 %= ymax  # remainder wraps over domain
        
        # Update centered velocity (For plots, not used in algorithm)
        self.vz.append(( vz_new + self.vz_old )/2)
        
        # Prepare for the next timestep
        self.y.append(y2)
        self.z.append(z2)
        self.vz_old = vz_new
        self.cross.append(cross.copy())
        
    def collect(self, opacity, potential):
        collected = self.q * potential * opacity

        wasted = .5*self.m*opacity * (self.vy**2+self.vz[-1]**2)
        
        self.m *= (1-opacity)
        self.q *= (1-opacity)
        self.energy_collected += collected
        
        if self.m/self.m0 < .01:
            self.absorbed = True # If only 1% of the mass is left, lets call it a day.
        return self.absorbed, collected, wasted
    
    def remaining(self):
        remain = self.m/self.m0*100
        print(f'{remain:.1f}% of particle mass remaining')
        
    def eff(self):
        eff = self.energy_collected/self.KE0
        print(f'{eff*100:.1f}% of particle energy collected')
        
    def trajectory(self, ax, color='C0', label='', linewidth=0.1):
        """
        Plots the trajectory of a single particle.
        Plotting is broken into segments delineated by periodic boundary crossings.
        All segments and the final marker have the same color, chosen at random.
        """
        posA = np.array([self.y, self.z]) # Positions
        crossA = np.array(self.cross) # y-crossings
        idxs = np.argwhere(~np.isnan(crossA[:,0])).flatten() # Indices of crossings
            # ~ is the bitwise NOT operator. Converts false to true.
            # This find indices of non-NAN values in first column of crossA.
                    
        if len(idxs)==0: # If no crossings, this function is very simple
            y, z = posA[0,:], posA[1,:]     
            ax.plot(z, y, color=color, label=label, linewidth=linewidth)
            ax.plot(z[-1], y[-1], color=color, marker='o', markersize=0.5) 
            # Final location (collected or lost here)
            return
        
        """
        First plot a line from start point until the index where it has crossed.
        Overwrite the last point with the crossing location.
        New start point has same crossing location but on the next boundary.
        """
        start_idx = 0 # The start index. This changes every loop.
        start_pt = posA[:,0] # The start point. This changes every loop.
        ymax = self.ymax

        for idx in idxs:
            y, z = posA[0,start_idx:idx+1], posA[1,start_idx:idx+1] 
            # Plot until it crosses
            y[0], z[0] = start_pt # Overwrite first point to be start point.
            y[-1], z[-1] = crossA[idx,:] # Overight final point as crossing entrance
            ax.plot(z, y, color=color, linewidth=linewidth) # Plot the segment
            
            # Find crossing exit for next segment
            start_idx = idx.copy()
            start_pt = crossA[idx,:]
            if start_pt[0]==0:
                start_pt[0] = ymax
            else:
                start_pt[0] = 0
            
        # Plot the final segment after the last crossing
        y, z = posA[0,start_idx:], posA[1,start_idx:]
        y[0], z[0] = start_pt # Overwrite first point to be start point.
        ax.plot(z, y, color=color, label=label, linewidth=linewidth)
        ax.plot(z[-1], y[-1], color=color, marker='o', markersize=0.5) 
        # Final location (collected or lost here)
        
    def angle(self, idx=-1):
        if idx > len(self.vz)-1:
            idx = -1
        angle = np.arctan2(self.vy, self.vz[idx])*180/np.pi
        return angle
        
def super_maxwellian(N, density, yrange, zrange, kT, uz, \
                     mass=1, q=1, theta=0, ymax=[], SI=False, beam=False):   
    '''
    Makes a maxwellian distribution of N superparticles!
    kT is the temperature. If SI is on, this is in eV.
    uz is the drift velocity (m/s in SI)
    theta specifies the incidence angle of the beam on the grids
    mass and q are for actual particles, NOT superparticles.
        Should always be normalized to proton mass & charge.
    beam=True makes temperature only apply in z. Beam is cold in x and y.
    '''
    if ymax==[]:
        ymax = yrange[1]
    
    # Superparticals
    Nplasma = density*(max(zrange)-min(zrange))
    scale = Nplasma/N
    
    # Find v_thermal of the actual particles
    if SI:
        mp = 1.6726e-27 # proton mass
        e  = 1.6022e-19 # proton charge
    else:
        mp = 1
        e = 1
    v_th = np.sqrt(kT*e/(mass*mp))
    
    # Positions and Velocities
    y =  np.random.uniform(yrange[0], yrange[1], N)
    z =  np.random.uniform(zrange[0], zrange[1], N)
    vz = np.random.normal(uz, v_th, N)
    if beam:
        vy =  np.zeros(N) # kTy = 0
        KEx=False # kTx = 0
    else:
        vy = np.random.normal(0, v_th, N)
        KEx=True # kTx = kTy
    
    # Rotate velocity vectors by given angle
    theta *= np.pi/180
    vz_prime = vz*np.cos(theta) - vy*np.sin(theta)
    vy_prime = vz*np.sin(theta) + vy*np.cos(theta)
    
    pos0 = np.array([y, z]).T
    v0 = np.array([vy_prime,vz_prime]).T
    
    particles = [Particle(pos0[i], v0[i], ymax, mass*scale, q*scale, SI=SI, KEx=KEx)\
                 for i in range(N)]
    # Note: particles is always defined with normalized mass and charge
    return particles