"""
Module containing the main code to setup and run the simulation
"""

import astropy.units as u
from particles.particle import Particle
import grav_field

# Test model based on the international space station
PARTICLE1 = Particle([0, 6800000, 0], [7.5 * 1000 * 4, 0, 0], 4)
PARTICLE2 = Particle(mass=(1 * u.M_earth).to(u.kg).value)
TIMESTEP = (100 * u.ms).to(u.s)

# Run for one orbit of 88 minutes
for i in range(int(88 * 60 / TIMESTEP.value) + 1):
    force_12 = grav_field.get_force(PARTICLE1, PARTICLE2)
    PARTICLE1.apply_force(force_12, TIMESTEP)
    PARTICLE2.apply_force(-force_12, TIMESTEP)
    PARTICLE1.move_particle(TIMESTEP)
    PARTICLE2.move_particle(TIMESTEP)
    pot_energy = grav_field.get_potential_energy(PARTICLE1, PARTICLE2)
    p1_energy = PARTICLE1.kinetic_energy() + pot_energy
    p2_energy = PARTICLE2.kinetic_energy() + pot_energy
    print("Minute: {:.2f}, Pos1: {}, Pos2: {}, E1: {}, KE2: {}, KE_Tot: {}".
          format(i * TIMESTEP.value / 60,
                 PARTICLE1.position, PARTICLE2.position,
                 p1_energy, p2_energy,
                 p1_energy + p2_energy))
