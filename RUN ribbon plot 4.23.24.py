#%% Header
# My files:
from ZAP_DEC import ZAP_DEC
from particles import Particle, super_maxwellian
from domain import Domain
from electrodes import Electrode

# Packages:
import numpy as np
import matplotlib.pyplot as plt


#%% Define eletrodes and domain

rib1 = Electrode('ribbon', ribbon_angle=20, pitch=1/2)

"""
opacity(theta, plot=False) : 
    Calculates opacity at a given angle. Arg plot=True visualizes this.
    theta : float, DEGREES (For all methods)
        Angle of particle velocity vector. 0 degrees is horizontal to the right.
"""
angle = 30*np.pi/180
rib1.opacity(angle, plot=True)