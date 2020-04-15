"""
The test module for the grav_field module
"""

import pytest # pylint: disable=unused-import
import numpy as np
import astropy.units as u
from astropy.constants import G
from particles.particle import Particle
from grav_field import get_force, get_potential_energy

@pytest.mark.parametrize("power,expected",
                         [
                             (1e21, [0.0, 0.48, -0.36]),
                             (1e14, np.array([0.0, 0.48, -0.36]) * 1 / 360000)
                         ])
def test_get_force(power, expected):
    """
    Tests grav_field.get_force()
    """
    particle1 = Particle(position=[0, 0, 3 * power], mass=3 * power)
    particle2 = Particle(position=[0, 4 * power, 0], mass=5 * power)
    force = get_force(particle1, particle2)
    # Floating point arithmetic has small rounding errors
    assert(np.abs(force - G * np.array(expected) *
                  (u.kg ** 2 / u.m ** 2)).value < [1e-16, 1e-16, 1e-16]).all()

def test_get_potential_energy():
    """
    Tests grav_field.get_potential_energy()
    """
    particle1 = Particle(position=[0, 0, 3], mass=3.0)
    particle2 = Particle(position=[0, 4, 0], mass=5.0)
    pot_energy = get_potential_energy(particle1, particle2)
    # Floating point arithmetic has small rounding errors
    assert np.abs(pot_energy - G * -3 * (u.kg ** 2 / u.m)).value < 1e-15
