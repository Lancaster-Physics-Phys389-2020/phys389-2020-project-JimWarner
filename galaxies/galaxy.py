"""
Module containing the base galaxy abstract class which the other galaxy
    classes will inherit from
"""

from abc import ABC, abstractmethod
import numpy as np
import astropy.units as u

class Galaxy(ABC):
    """
    The abstract class that the other galaxy classes are built on
    """

    def __init__(self, position=np.array([0.0, 0.0, 0.0], dtype=float),
                 particle_count=1):
        """
        Creates the galaxy with its particle list
        """
        # Check particle number is positive
        assert particle_count > 0

        self.particle_count = particle_count
        self.position = np.array(position, dtype=float) * u.m
        self.particles = self.init_particles()

    @abstractmethod
    def init_particles(self):
        """
        Calculates the locations and types of particles in the galaxy and
            returns a list containing the particles
        """

    def get_particles(self):
        """
        Returns a list of the particles in the galaxy
        """
        return self.particles
