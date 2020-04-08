"""
Module containing the functions required to model a gravitational field

Masses in kg
Positions in m
Forces in N
"""

from astropy.constants import G
import astropy.units as u
import numpy as np
from particles.particle import Particle

def get_force(particle1, particle2):
    """
    Calculates the force on particle1 by particle2
    F_g = (G m_1 m_2)/(r.r) * unit vec r

    Returns force in N
    """

    pos_diff = Particle.vector_between(particle1, particle2)
    distance = np.sqrt(pos_diff.dot(pos_diff))

    # If particles in the exact same place return 0
    if (pos_diff == 0.0).all():
        return np.array([0.0] * 3)

    force = ((G * particle1.mass * particle2.mass) / (distance ** 3)) * pos_diff 
    
    # Soften force if particles are close
    if distance.value < 1e19:
        #print(pos_diff, distance)
        force = force * (distance / (1e19 * u.m)) ** 3
    
    return force

def get_potential_energy(particle1, particle2):
    """
    Calculates the potential energy of particle1 in the field of particle2

    Returns the potential energy in J
    """
    pos_diff = Particle.vector_between(particle1, particle2)
    return -1 * (G * particle1.mass * particle2.mass) / np.sqrt(
        np.dot(pos_diff, pos_diff))

def get_centre_of_mass(particles):
    """
    Calculates the centre of mass of a list of particles

    Assumes that the particle are all of equal mass
    """

    positions = np.array([i.position.value for i in particles]).T
    #print(positions)
    com = np.array([positions[0].mean(), positions[1].mean(), positions[2].mean()])
    return com * u.m
