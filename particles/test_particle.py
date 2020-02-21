"""
The test module for the Particle class
"""

import pytest
from .particle import Particle

@pytest.mark.parametrize("mom_input,mass_input,expected",
                         [
                             ([2, 4, 6], 2, [1, 2, 3]),
                             ([-1, 8, 4.2], 4, [-0.25, 2, 1.05])
                         ])
def test_velocity(mom_input, mass_input, expected):
    """
    Checks that the velocity method returns the correct velocity
        with a given mass and momentum
    """
    particle = Particle(momentum=mom_input, mass=mass_input)
    assert (particle.velocity() == expected).all()
