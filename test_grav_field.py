"""
The test module for the grav_field module
"""

import pytest # pylint: disable=unused-import
import numpy as np
import astropy.units as u
from astropy.constants import G
from particles.particle import Particle
from grav_field import get_force

def test_get_force():
    """
    Tests grav_field.get_force()
    """
    particle1 = Particle(position=[0, 0, 3], mass=3.0)
    particle2 = Particle(position=[0, 4, 0], mass=5.0)
    force = get_force(particle1, particle2)
    # Floating point arithmetic has small rounding errors
    assert(np.abs(force - G * [0.0, 0.48, -0.36] *
                  (u.kg ** 2 / u.m ** 2)).value < [1e-16, 1e-16, 1e-16]).all()
