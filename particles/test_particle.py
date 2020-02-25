"""
The test module for the Particle class
"""

import astropy.units as u
import numpy as np
import pytest
from particles.particle import Particle

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
    assert particle.kinetic_energy() == expected

@pytest.mark.parametrize("mom_input,time_input,expected",
                         [
                             ([1.2, 2.4, 4.8] * u.N * u.s, 0.8 * u.s,
                              [0.96, 1.92, 3.84] * u.m),
                             ([-1.86, 2.94, 0.34] * u.N * u.s, 1.7 * u.s,
                              [-3.162, 4.998, 0.578] * u.m)
                         ])
def test_move_particle(mom_input, time_input, expected):
    """
    Checks that the move_particle method correctly updates the position
    Test particle has a mass of 1.0 kg starts at the origin
        with momentum mom_input
    Test force is applied for time_input s
    """
    particle = Particle(momentum=mom_input)
    particle.move_particle(time_input)
    assert(np.abs(particle.position - expected).value <
           [1e-15, 1e-15, 1e-15]).all()

@pytest.mark.parametrize("force_input,expected",
                         [
                             ([1.3, 2.9, 4.7] * u.N,
                              [0.65, 1.45, 2.35] * u.N * u.s),
                             ([-3.4, 2.1, 0.2] * u.N,
                              [-1.7, 1.05, 0.1] * u.N * u.s)
                         ])
def test_apply_force(force_input, expected):
    """
    Checks that the apply force method correctly updates the momentum
    Test particle has a mass of 1.0 kg starts stationary at the origin
    Test force is applied for 0.5 s
    """
    particle = Particle()
    particle.apply_force(force_input, 0.5 * u.s)
    assert(np.abs(particle.momentum - expected).value <
           [1e-16, 1e-16, 1e-16]).all()

@pytest.mark.parametrize("pos1_input, pos2_input, expected",
                         [
                             ([2, 4, 6], [8, 6, 4], [6, 2, -2] * u.m),
                             ([-1, 8, 4.2], [-9.3, 8.0, 4.3],
                              [-8.3, 0, 0.1] * u.m)
                         ])
def test_vector_between(pos1_input, pos2_input, expected):
    """
    Checks that the vector_between method calculates the correct vector
    """
    particle1 = Particle(position=pos1_input)
    particle2 = Particle(position=pos2_input)
    assert(np.abs(Particle.vector_between(particle1, particle2) -
                  expected).value < [1e-15, 1e-15, 1e-15]).all()
