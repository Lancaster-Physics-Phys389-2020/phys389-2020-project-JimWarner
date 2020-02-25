"""
Module containing the functions required to model a gravitational field

Masses in kg
Positions in m
Forces in N
"""

from astropy.constants import G
import numpy as np
from particles.particle import Particle

def get_force(particle1, particle2):
    """
    Calculates the force on particle1 by particle2
    F_g = (G m_1 m_2)/(r.r) * unit vec r

    Returns force in N
    """

    pos_diff = Particle.vector_between(particle1, particle2)
    return ((G * particle1.mass * particle2.mass) /
            (np.dot(pos_diff, pos_diff) ** (3/2))) * pos_diff
