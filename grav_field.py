"""
Module containing the functions required to model a gravitational field

Masses in kg
Positions in m
Forces in N
"""

from astropy.constants import G # pylint: disable=no-name-in-module
import numpy as np

def get_force(particle1, particle2):
    """
    Calculates the force on particle1 by particle2
    F_g = (G m_1 m_2)/(r.r) * unit vec r

    Returns force in N
    """

    pos_diff = particle2.position - particle1.position
    return ((G * particle1.mass * particle2.mass)/np.dot(pos_diff, pos_diff)**(3/2))*pos_diff
