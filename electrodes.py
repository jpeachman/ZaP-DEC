import numpy as np
import matplotlib.pyplot as plt
from PIL import Image # Used for animation

#%%

class Electrode():
    """
    Defines an electrode, which is then abstracted as a plane.
    -------------------------
    TYPES:
        'screen' electrodes have fixed opacity
        'ribbon' electrodes have angular dependant transparency
    PARAMETERS
        fixed_opacity : Only works for 'screen' electrodes
        ribbon_angle : float, DEGREES
            Angle of the ribbons
        ribbon_L : float
            Length of the ribbons in a ribbon grid.
        pitch : float
            Spacing between ribbons, measured from leading edge of each ribbon.
    -------------------------
    METHODS
    
        incident(vy, vz, print_out=False) : 
            Particles hit ribbons at incident angle determined by velocities. 
            Returns fraction of particles absorbed.
            vy, vz: Vertical and horizontal velocities respectively.
            print_out : Optional argument to print the opacity
    
        map_opacity(plot=False, animate=False) : 
            Calls opacity() in a loop with 1 degree increments.
            Arg plot=True shows the map, animate=True shows all the plots from opacity().
            WARNING: Animate makes a lot of files (359x .png, 1x .gif)
    
        opacity(theta, plot=False) : 
            Calculates opacity at a given angle. Arg plot=True visualizes this.
            theta : float, RADIANS (For all methods)
                Angle of particle velocity vector. 0 degrees is horizontal to the right.
            
        plot_vec() : 
            Helper function used for plotting in opacity(). 
            Not intended to be called externally.
    """
    
    def __init__(self, style, # Style options are 'screen' and 'ribbon' 
                 fixed_opacity=.01, ribbon_angle=5, ribbon_L=1, pitch=1/2.5):
                # defaults are close to those in barr et.al. (1974)
                # "1% of total beam current impionges on the grid" pg 80 (10 in pdf)
        
        self.style=style
        
        if style=='ribbon':
            self.ribbon_angle = float(ribbon_angle)
            self.ribbon_L = float(ribbon_L)
            self.pitch = float(pitch)
            self.angles, self.opacities = self.map_opacity()
        
        elif style=='screen':
            self.fixed_opacity = fixed_opacity

        elif style=='wall':
            self.fixed_opacity = 1
            
        else:
            raise ValueError('Style must be ribbon, screen, or wall')
            
        # For tracking collection efficiency
        self.collected = 0
        self.wasted = 0
            
    def incident(self, vy, vz, print_out=False):
        # This method will be called every time a particle hits an electrode
        
        theta = np.arctan2(vy, vz)
        
        if self.style=='ribbon':
            opacity = np.interp(theta, self.angles, self.opacities)
            if print_out == True:
                print(f'Interpolated Opacity: {opacity*100:.1f}%')  
            return opacity
            
        if self.style=='screen' or 'wall':
            return self.fixed_opacity

    def map_opacity(self, plot=False, animate=False):
        Angles = np.arange(-180, 180, 1)*np.pi/180 # Evaluated once per degree
        Opacities = []
        file_names = []
        
        for theta in Angles:
            opacity = self.opacity(theta, plot=animate)
            # plot=animate because we only need to plot at this step if we plan to animate
            Opacities.append(opacity)

            if animate==True:
                file_name = f"frame_{theta:03d}.png"
                plt.savefig(file_name, dpi=300)
                plt.close()  # Close the figure to free up memory
                file_names.append(file_name) 
        
        if plot==True:
            fig, ax = plt.subplots(dpi=600)
            ax.plot(Angles*180/np.pi, Opacities)
            ax.set_title('Opacity vs Incident Angle of Particles\n'
                f'{self.ribbon_L} long, '
                f'{self.ribbon_angle}$\degree$ angle, {self.pitch} pitch.')
            ax.set_xlabel('Angle, degrees')
            ax.set_ylabel('Opacity')
        
        if animate==True:
            images = [Image.open(file_name) for file_name in file_names]
            images[0].save("ribbon_animation.gif", save_all=True, \
                           append_images=images[1:], duration=100, loop=0)
            
        return Angles, np.array(Opacities)

    def opacity(self, theta, plot=False):
        ribbon_angle = self.ribbon_angle
        ribbon_angle *= np.pi/180 # Radians
        
        ribbon_L = self.ribbon_L
        pitch = self.pitch
        
        # Unit vec perpendicular to theta
        view_vec = np.array([np.cos(theta +np.pi/2), np.sin(theta +np.pi/2)]) 
        
        # Draw ribbon and its projection
        ribbon_vec = np.array([ribbon_L * np.cos(ribbon_angle), \
                               ribbon_L*np.sin(ribbon_angle)])
        ribbon_proj = np.dot(view_vec, ribbon_vec)*view_vec
        
        # Show the particle flow direction
        particle_vec = np.array([ribbon_L * np.cos(theta), ribbon_L * np.sin(theta)])
        
        # Project the pitch between ribbons (max size of each opening)
        if ribbon_proj[1] < 0:
            pitch *= -1
        pitch_vec = np.array([0,pitch])
        pitch_proj = np.dot(view_vec, pitch_vec)*view_vec
        
        # Compute magnitudes of projections
        mag_ribbon_proj = np.linalg.norm(ribbon_proj)
        mag_pitch_proj = np.linalg.norm(pitch_proj)
        
        # Compute opacity, which is 1-transparency
        if mag_pitch_proj == 0:
            opacity = 1
        else:
            opacity = mag_ribbon_proj/mag_pitch_proj
            if opacity > 1:
                opacity = 1
        
        if plot==True:
            fig, ax = plt.subplots(dpi=600)
            
            self.plt_vec(ax, ribbon_vec, label='Ribbon', color='k', pitch=pitch, lw=3)
            ax.arrow(0,0,particle_vec[0],particle_vec[1], color='green', \
                     head_width=0.1, head_length=0.1, label='Flow Direction' )
            
            self.plt_vec(ax, ribbon_proj, label='Ribbon_Proj', color='r')
            self.plt_vec(ax, ribbon_proj, origin=ribbon_vec, color='r', ls=':')
            self.plt_vec(ax, pitch_proj, label='Pitch_Proj', ls=':', color='C0', lw=2)
            self.plt_vec(ax, pitch_proj, origin=(0,pitch), ls=':', color='C0')
            
            ax.set_aspect('equal')
            ax.set_xlim(-1.25*ribbon_L,1.25*ribbon_L)
            ax.legend(frameon=False, bbox_to_anchor=(1.05, 0.5), loc='center left')
        
        return opacity
    
    def plt_vec(self, ax, vec, origin=(0,0), pitch=0, **kwargs):
        """
        Convenience function for plotting a vector.
        Providing pitch will make 2x vectors, vertically spaced by pitch.
        **kwargs is any argument for ax.plot(); i.e. color, label
        """
        ax.plot( (origin[0],vec[0]), (origin[1],vec[1]), **kwargs)
        
        if pitch != 0:
            if "label" in kwargs:
                kwargs["label"] = None
            ax.plot( (origin[0],vec[0]), (origin[1]+pitch,vec[1]+pitch), **kwargs)
            ax.plot( (origin[0],vec[0]), (origin[1]-pitch,vec[1]-pitch), **kwargs)
            
    def efficiency(self):
        total = self.wasted+self.collected
        if total == 0:
            efficiency = 0
        else:
            efficiency = self.collected / total
        return efficiency