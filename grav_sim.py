"""
Module containing the main code to setup and run the simulation
"""

from particles.particle import Particle
import grav_field

PARTICLE1 = Particle([0, 3.4, 5.6], [8.6, 2.8, 4], 4)
PARTICLE2 = Particle([0, 2.4, 4.9], [7.3, 1.1, 3.2], 7)
print(PARTICLE1, ", Velocity: ", PARTICLE1.velocity())
print(PARTICLE2, ", Velocity: ", PARTICLE2.velocity())
print(grav_field.get_force(PARTICLE1, PARTICLE2))
