"""
Module containing the main code to setup and run the simulation
"""

import astropy.units as u
from particles.particle import Particle
import grav_field

PARTICLE1 = Particle([0, 3.4, 5.6], [8.6, 2.8, 4], 4)
PARTICLE2 = Particle([0, 2.4, 4.9], [7.3, 1.1, 3.2], 7)
print(PARTICLE1, ", Velocity: ", PARTICLE1.velocity())
print(PARTICLE2, ", Velocity: ", PARTICLE2.velocity().unit.decompose())
print("Force: ", grav_field.get_force(PARTICLE1, PARTICLE2).to(u.N))
print("Kinetic Energy: ", PARTICLE1.kinetic_energy().to(u.J))
print(PARTICLE1.momentum.dot(PARTICLE1.momentum)/(PARTICLE1.mass * 2))
