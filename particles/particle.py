
"""
Module containing the particle class
"""

import numpy as np
import astropy.units as u

class Particle:
    """
    Class to model a massive particle in a gravitational field.
    Uses numpy arrays to store the position amd momentum

    Stores velocity as momentum

    mass in kg
    position in m
    momentum in kg m / s
    """

    def __init__(self, position=np.array([0.0, 0.0, 0.0], dtype=float),
                 momentum=np.array([0.0, 0.0, 0.0], dtype=float), mass=1.0):
        """
        Creates a particle with a position, momentum and mass

        Arguments:
        position => The initial position vector of the particle
                    (default [0.0, 0.0, 0.0])
        momentum => The initial momentum vector of the particle
                    (default [0.0, 0.0, 0.0])
        mass => The mass of the particle (default 1.0)
        """

        # Reject particles with 0 or negative mass
        assert mass > 0.0

        self.position = np.array(position, dtype=float) * u.m
        self.momentum = np.array(momentum, dtype=float) * u.N * u.s
        self.mass = mass * u.kg

    def __repr__(self):
        """
        Returns a string containing the mass,
            position and momentum of the particle
        """
        return "Particle: Mass: {0}, Position: {1}, Momentum: {2}".format(
            self.mass, self.position, self.momentum)

    def velocity(self):
        """
        Calculates and returns the velocity of the particle in m / s
        """
        return self.momentum / self.mass

    def kinetic_energy(self):
        """
        Calculates and returns the kinetic energy of the particle in J
        """
        return self.momentum.dot(self.momentum) / (2 * self.mass)

    def move_particle(self, delta_t):
        """
        Moves particle along its velocity vector for time delta_t
        """
        self.position += self.velocity() * delta_t

    def apply_force(self, force, delta_t):
        """
        Updates momentum based on the force being applied and
            the time step across which it is being applied

        Uses delta_P = F * delta_t
        """
        self.momentum += force * delta_t

    @staticmethod
    def vector_between(particle1, particle2):
        """
        The vector between particle 1 and particle 2
        """
        return particle2.position - particle1.position
