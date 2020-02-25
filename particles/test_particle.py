"""
The test module for the Particle class
"""

import astropy.units as u
import pytest
from .particle import Particle

@pytest.mark.parametrize("mom_input,mass_input,expected",
                         [
                             ([2, 4, 6], 2, [1, 2, 3] * u.m / u.s),
                             ([-1, 8, 4.2], 4, [-0.25, 2, 1.05] * u.m / u.s)
                         ])
def test_velocity(mom_input, mass_input, expected):
    """
    Checks that the velocity method returns the correct velocity
        with a given mass and momentum
    """
    particle = Particle(momentum=mom_input, mass=mass_input)
    assert (particle.velocity() == expected).all()


@pytest.mark.parametrize("mom_input,mass_input,expected",
                         [
                             ([2, 4, 6], 2.0, 14.0 * u.J),
                             ([-1, 8, 4.2], 4.0, 10.33 * u.J)
                         ])
def test_kinetic_energy(mom_input, mass_input, expected):
    """
    Checks that the kinetic energy method returns the correct kinetic energy
        with a given mass and momentum
    """
    particle = Particle(momentum=mom_input, mass=mass_input)
    assert (particle.kinetic_energy() == expected)
