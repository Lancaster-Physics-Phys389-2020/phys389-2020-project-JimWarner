"""
Module containing the elliptical galaxy class
"""

import random
import astropy.units as u
import numpy as np
from particles.particle import Particle
from galaxies.galaxy import Galaxy

class Elliptical(Galaxy):
    """
    Class used to initialise and control any elliptical galaxies
    """
    
    def init_particles(self):
        """
        Generates a list of particles at random positions within a spherical region
        """
        galaxy_diameter = (u.lyr).to(u.m) * 100000
        particle_mass = 10**9 * (u.M_sun).to(u.kg) / self.particle_count

        particles = []
        for i in range(self.particle_count):
            # TODO: Make spherical not cubic region
            particle_pos = np.array([
                random.random(), random.random(), random.random()
            ]) * galaxy_diameter
            # TODO: Init elliptical galaxy particle momentum
            particle = Particle(particle_pos, mass = particle_mass)
            particles.append(particle)

        return np.array(particles)
