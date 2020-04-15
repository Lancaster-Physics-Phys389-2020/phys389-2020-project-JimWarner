"""
Module containing the elliptical galaxy class
"""

import random
import astropy.units as u
import numpy as np
from particles.particle import Particle
from galaxies.galaxy import Galaxy
from octtree import OctTree, barnes_hut_approximation

class Elliptical(Galaxy):
    """
    Class used to initialise and control any elliptical galaxies
    """

    def init_particles(self):
        """
        Generates a list of particles at random positions within a spherical
            region and gives them approximatly the momentum required to orbit
            the central point of the galaxy
        """
        self.radius = 50000 * (u.lyr).to(u.m)
        self.mass = 1e12 * (u.M_sun).to(u.kg)
        self.particle_mass = self.mass / self.particle_count

        particles = self.add_orbits(self.get_locations())

        for i, _ in enumerate(particles):
            particles[i].position += self.position

        for particle in particles:
            if (particle.position > 0.0).any():
                momentum_boost = np.array([-10000 * self.particle_mass] * 3)
            else:
                momentum_boost = np.array([10000 * self.particle_mass] * 3)

            particle.momentum += momentum_boost * u.N * u.s

        return particles

    def get_locations(self):
        """
        Calculates the locations for the particles and returns a list
            of particles in those locations
        """

        particles = []

        for _ in range(self.particle_count):
            phi = random.random()*2*np.pi
            costheta = random.random() * 2 - 1
            theta = np.arccos(costheta)

            distance = random.random() ** 0.7 * self.radius

            x_pos = distance * np.sin(theta) * np.cos(phi)
            y_pos = distance * np.sin(theta) * np.sin(phi)
            z_pos = distance * np.cos(theta)

            particle_pos = np.array([x_pos, y_pos, z_pos])
            particle = Particle(particle_pos, mass=self.particle_mass)
            particles.append(particle)

        return particles

    def add_orbits(self, particles):
        """
        Calculates the momentum that each particle would need to
            orbit if the current force was constant and adds it
            to them

        Takes particles list as a parameter
        Returns particles list
        """

        # Sort particles by distance from centre of galaxy
        particles.sort(key=lambda p: np.sum(p.position**2))
        particle_tree = OctTree(radius=self.radius)

        # Work through particles in order of distance
        for i in range(1, len(particles)):
            distance_from_centre = np.sqrt(
                particles[i].position.dot(particles[i].position) ** 2
            )

            # Insert previous particle to octtree
            particle_tree.insert_new_particle(particles[i - 1])

            if i > 3:
                force = barnes_hut_approximation(
                    particle_tree, particles[i]).value
                speed = np.sqrt(np.sqrt(np.sum(force**2)) *
                                distance_from_centre / particles[i].mass)
            else:
                speed = 0.0 * u.m / u.s

            direction = np.cross([
                particles[i].position[0].value,
                particles[i].position[1].value,
                0.0
            ], [0.0, 0.0, 1.0])

            # Normalise direction vecter
            direction = direction / np.sqrt(np.sum(direction ** 2))
            momentum = (speed * direction * self.particle_mass).value
            particles[i].momentum = momentum * u.N * u.s

        return particles
