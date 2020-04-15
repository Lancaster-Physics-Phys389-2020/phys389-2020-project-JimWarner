"""
Module containing all the method used in both the octtree and particle
    mesh simulations and all the global variables
"""

import multiprocessing
import copy
import pickle
import numpy as np
import astropy.units as u
import png
import pandas as pd
import matplotlib.pyplot as plt
import grav_field
from octtree import OctTree, barnes_hut_approximation
from particle_mesh import ParticleMesh

GOT_DATA = False
FORCE_LIST = []
TOTAL_PE = 0.0
PARTICLES = None
TIMESTEP = None
RET_QUEUE = multiprocessing.Queue()
PARTICLE_QUEUE = multiprocessing.Queue()
RADIUS = None
PARTICLE_COUNT = None
PARTICLE_TREE = None
LENGTH = None
PARAM_C = [0.6756, -0.1756, -0.1756, 0.6756]
PARAM_D = [1.3512, -1.7024, 1.3512]

def get_forces_raw(index):
    """
    Calculates the force on PARTICLES[index] using through brute force
        and appends it to FORCE_LIST
    """
    global FORCE_LIST
    global PARTICLES

    total_force = 0.0
    for j, particle in enumerate(PARTICLES):
        if j == index:
            continue
        total_force += grav_field.get_force(PARTICLES[index], particle)

    FORCE_LIST.append((index, total_force))


def get_forces_tree(index):
    """
    Calculates the force on PARTICLES[index] using the octree and appends
        it to FORCE_LIST
    """
    global PARTICLES
    global PARTICLE_TREE

    force = barnes_hut_approximation(PARTICLE_TREE, PARTICLES[index])
    FORCE_LIST.append((index, force))

def get_forces_mesh():
    """
    Calculates the force on PARTICLES[index] using the particle mesh
        method and appends it to FORCE_LIST
    """
    global PARTICLES
    global RADIUS

    mesh = ParticleMesh(grid_points=300, radius=RADIUS)

    mesh.insert_particles(PARTICLES)
    mesh.calculate_potential()
    mesh.calculate_force()

    forces = mesh.get_forces(PARTICLES)
    return forces

def get_energies(indices):
    """
    Calculates the potential energy of PARTICLES[indices[0]] in the field
        produced PARTICLES[indices[1]] and adds double that to TOTAL_PE
    """
    global TOTAL_PE

    potential_energy = grav_field.get_potential_energy(
        PARTICLES[indices[0]], PARTICLES[indices[1]]
    ).value
    TOTAL_PE += potential_energy * 2

def get_data(_):
    """
    Reads a copy of the PARTICLES list from PARTICLE_QUEUE and produces
        an OctTree to store as local copies
    Sets local GOT_DATA

    Called by child processes
    """
    global PARTICLE_TREE
    global PARTICLES
    global PARTICLE_QUEUE
    global GOT_DATA

    if not GOT_DATA:
        PARTICLES = PARTICLE_QUEUE.get()
        PARTICLE_TREE = OctTree(radius=RADIUS)
        PARTICLE_TREE.build_tree(PARTICLES)
        GOT_DATA = True


def send_data(pool):
    """
    Puts a deepcopy of the PARTICLES list into PARTICLE_QUEUE for each
        child process then calls get_data with each child process

    Called by main process
    """
    global PARTICLES
    global PARTICLE_QUEUE

    for _ in range(pool._processes):
        PARTICLE_QUEUE.put(copy.deepcopy(PARTICLES))

    while True:
        pool.map(get_data, range(pool._processes))
        if PARTICLE_QUEUE.empty():
            break

def send_results(_):
    """
    Sends a deepcopy of the local FORCE_LIST onto RET_QUEUE and clears
        the local versions of both GOT_DATA and FORCE_LIST

    Called by child processes
    """
    global FORCE_LIST
    global RET_QUEUE
    global GOT_DATA

    RET_QUEUE.put(copy.deepcopy(FORCE_LIST))
    FORCE_LIST.clear()
    GOT_DATA = False

def get_results(pool):
    """
    Calls send_results with each child process and reads the results
        from RET_QUEUE and returns all the results as a list of lists

    Called by main process
    """
    global RET_QUEUE

    force_list = []

    while len(force_list) < pool._processes:
        pool.map_async(send_results, range(pool._processes))
        for _ in range(pool._processes):
            results = RET_QUEUE.get()
            if len(results) > 0:
                force_list.append(results)

    return force_list

def send_energy_results(_):
    """
    Sends a copy of the local TOTAL_PE onto RET_QUEUE and clears the
        local versions of both GOT_DATA and TOTAL_PE

    Called by child processes
    """
    global TOTAL_PE
    global RET_QUEUE
    global GOT_DATA

    RET_QUEUE.put(TOTAL_PE)
    TOTAL_PE = 0.0
    GOT_DATA = False

def get_energy_results(pool):
    """
    Calls send_results with each child process and reads the results
        from RET_QUEUE and returns the sum of all the results

    Called by main process
    """
    global RET_QUEUE

    pe_list = []

    while len(pe_list) < pool._processes:
        pool.map_async(send_energy_results, range(pool._processes))
        for _ in range(pool._processes):
            result = RET_QUEUE.get()
            if result != 0.0:
                pe_list.append(result)

    return np.sum(pe_list)

def get_energy(pool):
    """
    Calculates the total potential energy and kinetic energy of the
        particles in the PARTICLES list then returns them in a list
        of [total energy, total potential, total kinetic]
    """
    global PARTICLES

    send_data(pool)

    energies = [i.kinetic_energy().value for i in PARTICLES]
    total_ke = np.sum(energies)

    indices = []
    for row in [[(i, j) for j in range(i + 1, len(PARTICLES))]
                for i in range(len(PARTICLES))]:
        for item in row:
            indices.append(item)

    pool.map(get_energies, indices)
    total_pe = get_energy_results(pool)
    total_energy = total_ke + total_pe

    return [total_energy, total_ke, total_pe]

def write_image(timestep_no, name):
    """
    Writes the locations of all the particles in the PARTICLES list to
        a 2000 x 2000 px png image with brightness representing distance
        from the closest edge of the region

    Requires a frames foldes in the current directory to save images in
    """

    img_dims = 2000
    img = [[0] * img_dims for i in range(img_dims)]
    for particle in PARTICLES:
        img_x = int(
            (particle.position[0].value / (RADIUS * 2) + 0.5) * img_dims
        )
        img_y = int(
            (particle.position[1].value / (RADIUS * 2) + 0.5) * img_dims
        )
        img_z = int(
            (particle.position[2].value / (RADIUS * 2) + 0.5) * 255
        )

        if 0 < img_x < img_dims and 0 < img_y < img_dims and 0 < img_z < 256:
            img[img_x][img_y] = img_z

    png.from_array(
        img, 'L'
    ).save(
        "frames/{}_p{}_t{}_l{}_pic{:05}.png"
        .format(
            name, PARTICLE_COUNT, TIMESTEP / (u.yr).to(u.s), LENGTH, timestep_no
        )
    )


def save_data(timestep_no, force_factor_list, mom_list,
              energy_list, timing_list, timestep_list):
    """
    Writes the required variables to a pickled file that can be read
        later to resume the simulation from this point
    """
    pickle.dump(PARTICLES,
                open("sim_data_particles.p", 'wb'))
    pickle.dump([timestep_no, TIMESTEP, RADIUS, PARTICLE_COUNT, LENGTH],
                open("sim_data.p", 'wb'))
    pickle.dump(force_factor_list,
                open("sim_data_force.p", 'wb'))
    pickle.dump(mom_list,
                open("sim_data_mom.p", 'wb'))
    pickle.dump(energy_list,
                open("sim_data_energy.p", 'wb'))
    pickle.dump(timing_list,
                open("sim_data_timing.p", 'wb'))
    pickle.dump(timestep_list,
                open("sim_data_timestep.p", 'wb'))

def read_data():
    """
    Reads the global variables from the pickeled files written to by
        save_data to resume the simulation from the last saved point
    """
    global PARTICLES
    global TIMESTEP
    global RADIUS
    global PARTICLE_COUNT
    global LENGTH

    PARTICLES = pickle.load(open("sim_data_particles.p", 'rb'))
    force_factor_list = pickle.load(open("sim_data_force.p", 'rb'))
    mom_list = pickle.load(open("sim_data_mom.p", 'rb'))
    energy_list = pickle.load(open("sim_data_energy.p", 'rb'))
    timing_list = pickle.load(open("sim_data_timing.p", 'rb'))
    timestep_list = pickle.load(open("sim_data_timestep.p", 'rb'))
    data = pickle.load(open("sim_data.p", 'rb'))
    TIMESTEP = data[1]
    RADIUS = data[2]
    PARTICLE_COUNT = data[3]
    LENGTH = data[4]
    return (data[0], force_factor_list, mom_list,
            energy_list, timing_list, timestep_list)

def write_data(args, force_factor_list, mom_list,
               energy_list, timing_list, timestep_list):
    """
    Writes the data recorded by the simulation to graphs and a .csv file
    """
    energy_interval = args.energy_interval
    timing_interval = args.time_interval

    name = None
    if args.simulation == 'g':
        name = "gal"
    elif args.simulation == 'u':
        name = "univ"

    if args.momentum:
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        mom_list = np.array(mom_list).T
        ax1.plot(timestep_list, mom_list, label="Total Momentum")
        ax1.legend()
        ax1.set_ylabel("Momentum / Ns")
        ax1.set_xlabel("Time / yr")
        fig1.savefig("{}_sim_momentum_p{}_t{}_l{}.png".format
                     (name, PARTICLE_COUNT, TIMESTEP /
                      (u.yr).to(u.s), LENGTH
                     ), dpi=500)

    if not timing_interval is None:
        timing_list = np.array(timing_list).T

    if not energy_interval is None:
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)

        energy_list = np.array(energy_list).T
        timecodes = [timestep_list[energy_interval * i]
                     for i in range(LENGTH // energy_interval + 1)]
        ax2.plot(timecodes, energy_list[0], label="Total Energy")
        ax2.plot(timecodes, energy_list[1], label="Kinetic Energy")
        ax2.plot(timecodes, energy_list[2], label="Potential Energy")
        ax2.legend()
        ax2.set_ylabel("Energy / J")
        ax2.set_xlabel("Time / yr")
        fig2.savefig("{}_sim_energy_p{}_t{}_l{}.png".format
                     (name, PARTICLE_COUNT, TIMESTEP /
                      (u.yr).to(u.s), LENGTH
                     ), dpi=500)

    if args.force:
        force_factor_list = np.array(force_factor_list).T
        fig3 = plt.figure()
        ax3 = fig3.add_subplot(111)
        ax3.errorbar(timestep_list, force_factor_list[0], force_factor_list[1])
        ax3.set_ylabel("Relative Force")
        ax3.set_xlabel("Time / yr")
        fig3.savefig("{}_sim_force_error_p{}_t{}_l{}.png".format
                     (name, PARTICLE_COUNT, TIMESTEP /
                      (u.yr).to(u.s), LENGTH
                     ), dpi=500)

    data_columns = []
    data = []
    data_width = 0
    data_length = 0
    data_index = 0

    # Initialise empty data array
    if args.momentum:
        data_width += 1
        data_length = len(mom_list)

    if args.force:
        data_width += 2
        data_length = len(force_factor_list[0])

    if not energy_interval is None:
        data_width += 3

    if not timing_interval is None:
        data_width += 2

    if data_length == 0:
        data_length = len(timestep_list)

    data = [[np.nan for _ in range(data_width)] for _ in range(data_length)]

    # Fill data array
    if args.momentum:
        for (i, mom) in enumerate(mom_list):
            data[i][data_index] = mom
        data_index += 1
        data_columns.append("Momentum / Ns")

    if args.force:
        for i in range(len(force_factor_list[0])):
            data[i][data_index] = force_factor_list[0][i]
            data[i][data_index + 1] = force_factor_list[1][i]
        data_index += 2
        data_columns.append("Mean_Force_Factor")
        data_columns.append("Force_Factor_Std_Dev")

    if not energy_interval is None:
        for i in range(len(energy_list[0])):
            data[energy_interval * i][data_index] = energy_list[0][i]
            data[energy_interval * i][data_index + 1] = energy_list[1][i]
            data[energy_interval * i][data_index + 2] = energy_list[2][i]
        data_index += 3
        data_columns.append("Total_Energy / J")
        data_columns.append("Total_Kinetic_Energy / J")
        data_columns.append("Total_Potential_Energy / J")

    if not timing_interval is None:
        for i in range(len(timing_list[0])):
            data[timing_interval * i][data_index] = timing_list[0][i]
            data[timing_interval * i][data_index + 1] = timing_list[1][i]
        data_index += 2
        if args.simulation == 'g':
            data_columns.append("Brute_Force_Timing / s")
        data_columns.append("Octtree_Timing / s")
        if args.simulation == 'u':
            data_columns.append("Mesh_Timing / s")

    if (args.momentum or args.force or not energy_interval is None
            or not timing_interval is None):
        dataframe = pd.DataFrame(data,
                                 index=timestep_list,
                                 columns=data_columns)
        dataframe.index.name = "Time / yr"
        dataframe.to_csv("{}_sim_data_p{}_t{}_l{}.csv"
                         .format(name, PARTICLE_COUNT,
                                 TIMESTEP / (u.yr).to(u.s), LENGTH))
