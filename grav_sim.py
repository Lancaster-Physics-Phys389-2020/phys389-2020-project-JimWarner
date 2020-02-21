"""
Module containing the main code to setup and run the simulation
"""

from particles.particle import Particle

particle = Particle([1.2, 3.4, 5.6], [8.6, 2.8, 4], 4)
print(particle, ", Velocity: ", particle.velocity())
