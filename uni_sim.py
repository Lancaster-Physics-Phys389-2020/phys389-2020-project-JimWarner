"""
Module containing the functions required to perform and collect data
    about the particle mesh simulation.
"""

import multiprocessing
import random
import time
import astropy.units as u
import numpy as np
import universe
import sim

def move_particles(timestep):
    """
    Moves the particles by the provided timestep accounting for
        the periodic boundary conditions
    """
    for i in range(len(sim.PARTICLES)):
        sim.PARTICLES[i].move_particle(timestep)

        # Periodic boundary conditions
        for j in range(3):
            if sim.PARTICLES[i].position[j].value > sim.RADIUS:
                sim.PARTICLES[i].position[j] = (
                    sim.PARTICLES[i].position[j].value % sim.RADIUS - sim.RADIUS
                ) * u.m
            elif sim.PARTICLES[i].position[j].value < -1 * sim.RADIUS:
                sim.PARTICLES[i].position[j] = (
                    sim.PARTICLES[i].position[j].value % sim.RADIUS
                ) * u.m

def apply_forces(timestep):
    """
    Calculates and applies the forces over the provided timestep
        using the particle mesh method
    """
    results = sim.get_forces_mesh()
    for index, force in enumerate(results):
        sim.PARTICLES[index].apply_force(force, timestep)

def check_forces(pool):
    """
    Compares the forces calculated by the particle mesh method to
        those calculated by the octtree on 25 particles
    """
    sample_size = 25

    sim.send_data(pool)

    if sample_size > sim.PARTICLE_COUNT:
        sample_size = sim.PARTICLE_COUNT

    indices = random.sample(range(len(sim.PARTICLES)), sample_size)

    pool.imap(sim.get_forces_tree, indices)
    results = []
    for arr in sim.get_results(pool):
        results += arr
    tree_forces = sorted(results, key=lambda item: item[0])
    tree_forces = np.array([i[1] for i in tree_forces])

    mesh_forces = np.array(sim.get_forces_mesh())[indices]

    factors = np.abs(mesh_forces / tree_forces)
    factors = np.array([np.sum(i) / 3 for i in factors])

    return [factors.mean(), factors.std()]

def get_timing(pool):
    """
    Measures and returns the time taken to calculate the forces
        on all the particles using both the particle mesh method
        and the octtree
    """
    sim.send_data(pool)

    start_tree = time.time()
    pool.imap(sim.get_forces_tree, range(sim.PARTICLE_COUNT))
    sim.get_results(pool)
    end_tree = time.time()

    start_mesh = time.time()
    sim.get_forces_mesh()
    end_mesh = time.time()

    return [end_tree - start_tree, end_mesh - start_mesh]

def run(args):
    """
    Runs the particle mesh simulation as according to the provided
        arguments
    """

    processes = args.processes

    energy_interval = args.energy_interval
    timing_interval = args.time_interval

    timestep_list = [0.0]
    mom_list = []
    energy_list = []
    timing_list = []
    force_factor_list = []
    start_timestep = 0

    if args.resume:
        (start_timestep, force_factor_list, mom_list,
         energy_list, timing_list, timestep_list) = sim.read_data()
    else:
        sim.PARTICLE_COUNT = args.particle_no

        # Model requires the number of particles to be cubic
        sim.PARTICLE_COUNT = (int(sim.PARTICLE_COUNT ** (1/3)) + 1) ** 3
        print("Modelling with {} particles".format(sim.PARTICLE_COUNT))

        sim.PARTICLES = universe.get_particles(sim.PARTICLE_COUNT)

        sim.TIMESTEP = (args.timestep * u.yr).to(u.s)
        sim.LENGTH = args.length
        sim.RADIUS = 5e24

    param_c = [0.6756, -0.1756, -0.1756, 0.6756]
    param_d = [1.3512, -1.7024, 1.3512]

    with multiprocessing.Pool(processes=processes) as pool:
        if start_timestep == 0:
            if not energy_interval is None:
                energy_list.append(sim.get_energy(pool))

            if args.force:
                force_factor_list.append(check_forces(pool))

            if not timing_interval is None:
                timing_list.append(get_timing(pool))

            if args.momentum:
                momenta = np.array([i.momentum for i in sim.PARTICLES]).T
                mom_list.append(
                    np.sum(momenta[0]) +
                    np.sum(momenta[1]) +
                    np.sum(momenta[2])
                )

        for timestep_no in range(start_timestep, sim.LENGTH):
            start = time.time()

            if args.images:
                sim.write_image(timestep_no, "univ_sim")

            timestep_no += 1

            #move
            move_particles(sim.TIMESTEP * param_c[0])
            #force
            apply_forces(sim.TIMESTEP * param_d[0])
            #move
            move_particles(sim.TIMESTEP * param_c[1])
            #force
            apply_forces(sim.TIMESTEP * param_d[1])
            #move
            move_particles(sim.TIMESTEP * param_c[2])
            #force
            apply_forces(sim.TIMESTEP * param_d[2])
            #move
            move_particles(sim.TIMESTEP * param_c[3])

            if args.force:
                force_factor_list.append(check_forces(pool))

            if args.momentum:
                momenta = np.array([i.momentum for i in sim.PARTICLES]).T
                mom_list.append(
                    np.sum(momenta[0]) +
                    np.sum(momenta[1]) +
                    np.sum(momenta[2])
                )

            if not energy_interval is None:
                if timestep_no % energy_interval == 0:
                    energy_list.append(sim.get_energy(pool))

            if not timing_interval is None:
                if timestep_no % timing_interval == 0:
                    timing_list.append(get_timing(pool))

                timestep_list.append(
                    np.round(timestep_no * sim.TIMESTEP.to(u.yr).value)
                )

            if not args.save_interval is None:
                if timestep_no % args.save_interval == 0:
                    sim.save_data(timestep_no,
                                  force_factor_list,
                                  mom_list,
                                  energy_list,
                                  timing_list,
                                  timestep_list)

            end = time.time()
            print("Time: {:.04}, Time step took: {}s".format
                  ((timestep_no * sim.TIMESTEP).to(u.yr), end - start))

    sim.write_data(args,
                   force_factor_list,
                   mom_list,
                   energy_list,
                   timing_list,
                   timestep_list)
