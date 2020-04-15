"""
Module containing the functions required to initialise the
    universe simulation
"""

from random import random
import numpy as np
from particles.particle import Particle

def get_particles(particle_count):
    """
    Calculates the locations of the particles in the simulation
        and returns a list of them
    """

    particles = []

    total_mass = 3e48
    size = 1e25

    row_length = int(np.round(particle_count ** (1/3)))

    particle_mass = total_mass / particle_count
    seperation = size / (row_length)
    centre = np.array([size, size, size]) / 2

    for i in range(row_length):
        for j in range(row_length):
            for k in range(row_length):
                particle_pos_x = seperation * (i + 0.5 + (random() - 0.5))
                particle_pos_y = seperation * (j + 0.5 + (random() - 0.5))
                particle_pos_z = seperation * (k + 0.5 + (random() - 0.5))
                particle_pos = np.array([
                    particle_pos_x,
                    particle_pos_y,
                    particle_pos_z
                ]) - centre

                particles.append(
                    Particle(position=particle_pos, mass=particle_mass)
                )

    # Check particle_count is cubic
    assert len(particles) == particle_count

    return particles
