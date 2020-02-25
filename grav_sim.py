"""
Module containing the main code to setup and run the simulation
"""

import astropy.units as u
from particles.particle import Particle
import grav_field

PARTICLE1 = Particle([0, 6800000, 0], [7.5 * 1000 * 4, 0, 0], 4)
PARTICLE2 = Particle(mass=(1 * u.M_earth).to(u.kg).value)
TIMESTEP = 1 * u.s

for i in range(90 * 60):
    force_12 = grav_field.get_force(PARTICLE1, PARTICLE2)
    PARTICLE1.apply_force(force_12, TIMESTEP)
    PARTICLE2.apply_force(-force_12, TIMESTEP)
    PARTICLE1.move_particle(TIMESTEP)
    PARTICLE2.move_particle(TIMESTEP)
    print("Pos1: {}, Pos2: {}, Mom1: {}, Mom2: {}, Mom_Tot: {}".format(
        PARTICLE1.position, PARTICLE2.position, PARTICLE1.momentum,
        PARTICLE2.momentum, PARTICLE1.momentum + PARTICLE2.momentum
    ))
