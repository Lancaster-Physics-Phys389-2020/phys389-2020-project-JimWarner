from galaxies.elliptical import Elliptical
import numpy as np
import astropy.units as u
import grav_field
import multiprocessing
from octtree import OctTree, barnes_hut_approximation
from particle_mesh import ParticleMesh
import png
import copy
import random
import pandas as pd
from arguments import parse_arguments
import sys
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from particles.particle import Particle

GOT_DATA = False
FORCE_LIST = []
TOTAL_PE = 0.0
particles = None
timestep = None
ret_queue = None
particle_queue = None
radius = None
particle_count = None
energy_list = []
timing_list = []

def get_forces_raw(index):
    global FORCE_LIST

    #print(index)
    total_force = 0.0
    
    # Brute force
    for j in range(len(particles)):
        if j == index:
            continue
        total_force += grav_field.get_force(particles[index], particles[j])
        
    FORCE_LIST.append((index, total_force))


def get_forces_tree(index):        
    # Octtree
    force = barnes_hut_approximation(particle_tree, particles[index])
    #print(particle.momentum)
    #particle.apply_force(force, timestep)
    #particle.move_particle(timestep)
    
    #print("Process: {}, Index: {}, Force: {}".format(os.getpid(), index, total_force))
    
    #print(total_force)
    FORCE_LIST.append((index, force))
    #return (index, force)

def get_energies(indices):
    global TOTAL_PE
    
    (i1, i2) = indices
    pe = grav_field.get_potential_energy(particles[i1], particles[i2]).value
    
    #ret_queue.put(pe)
    #return pe
    TOTAL_PE += pe * 2
    
def get_data(index):
    global particle_tree
    global particles
    global GOT_DATA
    
    #print("GETTING DATA", index, multiprocessing.current_process()._identity)
    
    if not GOT_DATA:
        particles = particle_queue.get()
        particle_tree = OctTree(radius=radius)
        particle_tree.build_tree(particles)
        GOT_DATA = True
        
        
def send_data(pool):
    global particles
    loops = 0
    
    trans_start=time.time()
    for i in range(pool._processes):
        particle_queue.put(copy.deepcopy(particles))
        
    #print(particle_queue.empty())

    while True:
        loops += 1
        pool.map(get_data, range(pool._processes))
        if particle_queue.empty():
            break
        
    #print("Transfer took {}s. {} loops".format(time.time()-trans_start, loops))

def send_results(index):
    global FORCE_LIST
    global ret_queue
    global GOT_DATA
    
    #if len(FORCE_LIST) > 0:
     #   print("SENDING RESULTS",index, multiprocessing.current_process()._identity)
        #print(FORCE_LIST)
    
    ret_queue.put(copy.deepcopy(FORCE_LIST))
    FORCE_LIST.clear()
    GOT_DATA = False
    
def get_results(pool):
    global ret_queue
    
    loops = 0
    force_list = []
    
    #print("GETTING RESULTS")
    
    trans_start=time.time()
    while len(force_list) < pool._processes:
        pool.map(send_results, range(pool._processes))
        loops += 1
        for i in range(pool._processes):
            results = ret_queue.get()
            #print(force_list)
            if len(results) > 0:
                force_list.append(results)

    #print("Recieved data {}s. {} loops".format(time.time()-trans_start, loops))

    return force_list

def send_energy_results(index):
    global TOTAL_PE
    global ret_queue
    global GOT_DATA
    
    #if TOTAL_PE != 0.0:
    #print("SENDING ENERGY RESULTS", index, multiprocessing.current_process()._identity)
    #print(FORCE_LIST)
    
    ret_queue.put(TOTAL_PE)
    TOTAL_PE = 0.0
    GOT_DATA = False
    
def get_energy_results(pool):
    global ret_queue
    
    loops = 0
    pe_list = []
    
    #print("GETTING ENERGY RESULTS")
    
    trans_start=time.time()
    while len(pe_list) < pool._processes:
        pool.map(send_energy_results, range(pool._processes))
        loops += 1
        for i in range(pool._processes):
            result = ret_queue.get()
            #print(results)
            if result != 0.0:
                pe_list.append(result)

    #print("Recieved data {}s. {} loops".format(time.time()-trans_start, loops))

    return np.sum(pe_list)
    
def record_energy(pool):
    global particles
    global energy_list
    
    send_data(pool)
    
    #print("Getting Energy")
    #calc_start = time.time()
    energies = [i.kinetic_energy().value for i in particles]
    total_ke = np.sum(energies)
    
    indices = []
    for row in [[(i, j) for j in range(i + 1, len(particles))]
                for i in range(len(particles))]:
        for item in row:
            indices.append(item)
            
    pool.map(get_energies, indices)#, len(particles) // pool._processes)
    total_pe = get_energy_results(pool)
    #for i in range(pool._processes):
    #    pe  = ret_queue.get()
    #    total_energy += 2 * pe
    
    total_energy = total_ke + total_pe
    
    energy_list.append([total_energy, total_ke, total_pe])
    #calc_end = time.time()
    #print("Calculation took: {}s".format(calc_end - calc_start))
    
def move_particles(pool, timestep):
    global particles
    
    for i in range(len(particles)):
        particles[i].move_particle(timestep)
        
def apply_forces(pool, timestep):
    global particles
    
    #print("SENDING DATA")
    
    send_data(pool)
    
    #print("APPLY FORCES")
    
    pool.imap(get_forces_tree, range(len(particles)))
    
    #print("RETRIEVING FORCES")
    
    results = get_results(pool)
    for array in results:
        for index, force in array:
            #(index, force) = ret_queue.get()
            particles[index].apply_force(force, timestep)
            
def write_image(particles, t):
    img_dims = 2000
    img = [[0] * img_dims for i in range(img_dims)]
    for particle in particles:
        img_x = int((particle.position[0].value / (radius * 2) + 0.5) * img_dims)
        img_y = int((particle.position[1].value / (radius * 2) + 0.5) * img_dims)
        img_z = int((particle.position[2].value / (radius * 2) + 0.5) * 255)
        #print(img_x, img_y, img_z)
        
        if img_x > 0 and img_x < img_dims and img_y > 0 and img_y < img_dims and img_z > 0 and img_z < 256:
            img[img_x][img_y] = img_z
            
    #for row in img:
    #    print(row)
    png.from_array(img, 'L').save("frames/gal_pic{:05}.png".format(t))
    
def check_forces(pool):
    global particles
    global particle_count

    sample_size = 100
    
    send_data(pool)

    #print("CHECKING FORCES")

    if sample_size > particle_count:
        sample_size = particle_count
     
    indices = random.sample(range(len(particles)), sample_size)
    #print(indices)

    pool.imap(get_forces_raw, indices)
    results = []
    for arr in get_results(pool):
        results += arr
    raw_forces = sorted(results, key=lambda item: item[0])
    raw_forces = np.array([i[1] for i in raw_forces])

    #print(results)
    #print(raw_forces)
    
    pool.imap(, indices)
    results = []
    for arr in get_results(pool):
        results += arr
    approx_forces = sorted(results, key=lambda item: item[0])
    approx_forces = np.array([i[1] for i in approx_forces])

    #print(approx_forces)

    factors = np.abs(approx_forces / raw_forces)
    #print(factors)
    factors = np.array([np.sum(i) / 3 for i in factors])
    #print(factors)
    return [factors.mean(), factors.std()]

def get_timing(pool):
    global particles
    global particle_count

    send_data(pool)

    start_raw = time.time()
    pool.imap(get_forces_raw, range(particle_count))
    get_results(pool)
    end_raw = time.time()

    start_tree = time.time()
    pool.imap(get_forces_tree, range(particle_count))
    get_results(pool)
    end_tree = time.time()

    return [end_raw - start_raw, end_tree - start_tree]
    
def run(args):
    global particles
    global timestep
    global ret_queue
    global particle_queue
    global radius
    global particle_count
    global energy_list

    processes = args.processes
    
    #tree = OctTree(radius = 1000)
    #p1 = Particle([1, 1, 1])
    #p2 = Particle([2, 2, 2])
    #tree.build_tree([p1, p2])
    #print(sys.getsizeof(1e32))
    #sys.exit()
        
    particle_count = args.particle_no

    count_modifier = processes
    if count_modifier % 2 == 1:
        count_modifier *= 2

    # Guarantees the number of particles divides by the number of processes
    particle_count = particle_count + count_modifier - particle_count % count_modifier
    particles_per_galaxy = particle_count // 2

    print("Modelling with {} particles".format(particles_per_galaxy * 2))
    
    #gal = Elliptical(particle_count=100)
    
    #count = 400

    #print("Starting init")
    #tree_start = time.time()
    gal1 = Elliptical(particle_count=particles_per_galaxy, position=[5e20, 5e20, 5e20])
    gal2 = Elliptical(particle_count=particles_per_galaxy, position=[-5e20, -5e20, -5e20])
    particles = gal1.particles + gal2.particles

    #print(len(particles))
    
    #tree_end = time.time()
    #print("Init took: {}s".format(tree_end - tree_start))

    #particles = gal.particles

    #sys.exit()
    
    #pm = ParticleMesh(count, 5e21)
    #pm.insert_particles(particles)
    #print(pm.grid)
    #pm.calculate_potential()
    #print(pm.grid)
    #
    #arr = []
    #index = count // 2
    #for i in range(len(pm.grid[index])):
    #    for j in range(len(pm.grid[index][i])):
    #        arr.append([i, j, pm.grid[index][i][j]])
    #    
    #arr = np.array(arr).T
    #
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(xs=arr[0], ys=arr[1], zs=arr[2], s=2)
    #plt.show()
    #sys.exit(0)

    

    #for i in gal.particles:
    #    print(i)

    #sys.exit(0)
    
    timestep = (args.timestep * u.yr).to(u.s)
    length = args.length
    radius = 1e22
    energy_interval = args.energy_interval
    timing_interval = args.time_interval

    ret_queue=multiprocessing.Queue()
    particle_queue=multiprocessing.Queue()

    #com = grav_field.get_centre_of_mass(particles)
    #for particle in particles:
    #    particle.position -= com

    #particles = gal.particles
    particle_tree = None
    poses = []
    temp = []
    for p in particles:
        temp.append([p.position[0].value, p.position[1].value, p.position[2].value])
    poses.append(temp)
    timestep_list = [0.0]
    mom_list = []
    timing_list = []
    force_factor_list = []
        
    
    c = [0.6756, -0.1756, -0.1756, 0.6756]
    d = [1.3512, -1.7024, 1.3512]
    
    with multiprocessing.Pool(processes=processes) as pool:
        if not (energy_interval is None):
            record_energy(pool)

        if args.force:
            force_factor_list.append(check_forces(pool))

        if not (timing_interval is None):
            timing_list.append(get_timing(pool))

        if args.momentum:
            momentums = np.array([i.momentum for i in particles]).T
            mom_list.append(np.sum(momentums[0]) + np.sum(momentums[1]) + np.sum(momentums[2]))

        for t in range(length):
            start = time.time()            
            write_image(particles, t)

            t+=1
            
            #mesh = ParticleMesh(count, 1e21)
            #mesh.insert_particles(particles)
            #mesh.calculate_potential()
            #mesh.calculate_force()
            #forces = mesh.get_forces(particles)
            #for i in range(len(particles)):
            #    if i < len(forces):
            #        particles[i].apply_force(forces[i], timestep)
            #    particles[i].move_particle(timestep)
            #print(np.abs(forces).mean())
            
            #com = grav_field.get_centre_of_mass(particles)
            #for particle in particles:
            #    particle.position -= com
         
            #print("Starting Tree")
            #tree_start = time.time()
            #particle_tree = OctTree(radius=radius)
            #particle_tree.build_tree(particles)
            #tree_end = time.time()
            #print("Tree build took: {}s".format(tree_end - tree_start))
                       
            #print("Starting Calculation")
            #tree_start = time.time()

            #move
            move_particles(pool, timestep * c[0])
            #force
            apply_forces(pool, timestep * d[0])
            #move
            move_particles(pool, timestep * c[1])
            #force
            apply_forces(pool, timestep * d[1])
            #move
            move_particles(pool, timestep * c[2])
            #force
            apply_forces(pool, timestep * d[2])
            #move
            move_particles(pool, timestep * c[3])
            
            #tree_end = time.time()
            #print("Calculation took: {}s".format(tree_end - tree_start))

            #    print(np.abs(forces).mean())
            #temp = []
            #for p in particles:
            #    temp.append([p.position[0].value, p.position[1].value, p.position[2].value])
            #poses.append(temp)

            #print("Starting Write")
            #write_start = time.time()
            #write_image(particles, t)
            #write_end = time.time()
            #print("Write took: {}s".format(write_end - write_start))
                
            #shared_particles = particles

            if args.force:
                force_factor_list.append(check_forces(pool))

            #mom_list_list = []
            #for i in range(len(particles)):
            #    mom_list_list.append([t, np.sqrt(particles[i].momentum[0]**2 + particles[i].momentum[1]**2 + particles[i].momentum[2]**2).value])
            #mom_list.append(np.array(mom_list_list).T)
        
            #positions = np.array([i.position for i in particles])

            if args.momentum:
                momentums = np.array([i.momentum for i in particles]).T
                mom_list.append(np.sum(momentums[0]) + np.sum(momentums[1]) + np.sum(momentums[2]))

            if not (energy_interval is None):
                if t % energy_interval == 0:
                    record_energy(pool)

            if not (timing_interval is None):
                if t % timing_interval == 0:
                    timing_list.append(get_timing(pool))

            timestep_list.append(np.round(t * timestep.to(u.yr).value))
        
            #print("Time: {:.04}, COM: {:.04}, Total Mom: {}, P0: {}".format((t * timestep).to(u.yr), positions.mean(), [np.sum(momentums[0]), np.sum(momentums[1]), np.sum(momentums[2])], particles[1]))
            end = time.time()
            print("Time: {:.04}, Time step took: {}s".format((t * timestep).to(u.yr), end - start))

            
    if args.momentum:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        #for p in poses:
        #    print(p)
        #poses=np.array(poses).T
        #print(len(poses[0]))
        #print(poses)
        #for i in range(len(poses[0])):
        #    ax.scatter(xs=poses[0][i], ys=poses[1][i], zs=poses[2][i], s=2)
        #    ax.plot(xs=poses[0][i], ys=poses[1][i], zs=poses[2][i])

        mom_list=np.array(mom_list).T
        ax.plot(timestep_list, mom_list)
        fig.set_size_inches(8, 8)
        fig.savefig("momentum.png", dpi=500)
            
    #mom_list = np.array(mom_list).T
    #print(mom_list)
    #for mom_list_list in mom_list:

    if not (timing_interval is None):
        timing_list = np.array(timing_list).T
        
    if not (energy_interval is None):
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        
        energy_list = np.array(energy_list).T
        timecodes = [timestep_list[energy_interval * i] for i in range(length // energy_interval + 1)]
        ax2.plot(timecodes, energy_list[0])
        ax2.plot(timecodes, energy_list[1])
        ax2.plot(timecodes, energy_list[2])
        fig2.set_size_inches(8, 8)
        fig2.savefig("energy.png", dpi=500)

    if args.force:
        force_factor_list = np.array(force_factor_list).T
        fig3 = plt.figure()
        ax3 = fig3.add_subplot(111)
        ax3.errorbar(timestep_list, force_factor_list[0], force_factor_list[1])
        fig3.set_size_inches(8, 8)
        fig3.savefig("force_error.png", dpi=500)
 
    #ax.set_xlim3d(-1e21, 1e21)
    #ax.set_ylim3d(-1e21, 1e21)
    #ax.set_zlim3d(-1e21, 1e21)
    #plt.show()

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
        
    if not (energy_interval is None):
        data_width += 3
        
    if not (timing_interval is None):
        data_width += 2
    
    data = [[np.nan for _ in range(data_width)] for _ in range(data_length)]
    
    # Fill data array
    if args.momentum:
        for i in range(len(mom_list)):
            data[i][data_index] = mom_list[i]
        data_index += 1
        data_columns.append("Momentum / Ns")
        
    if args.force:
        for i in range(len(force_factor_list[0])):
            data[i][data_index] = force_factor_list[0][i]
            data[i][data_index + 1] = force_factor_list[1][i]
        data_index += 2
        data_columns.append("Mean_Force_Factor")
        data_columns.append("Force_Factor_Std_Dev")

    if not (energy_interval is None):
        for i in range(length // energy_interval + 1):
            data[energy_interval * i][data_index] = energy_list[i][0]
            data[energy_interval * i][data_index + 1] = energy_list[i][1]
            data[energy_interval * i][data_index + 2] = energy_list[i][2]
        data_index += 3
        data_columns.append("Total_Energy / J")
        data_columns.append("Total_Kinetic_Energy / J")
        data_columns.append("Total_Potential_Energy / J")
        
    if not (timing_interval is None):
        for i in range(length // timing_interval + 1):
            data[timing_interval * i][data_index] = timing_list[i][0]
            data[timing_interval * i][data_index + 1] = timing_list[i][1]
        data_index += 2
        data_columns.append("Brute_Force_Timing / s")
        data_columns.append("Octtree_Timing / s")
                   
    #    data = [[mom_list[1][i], np.nan] for i in range(len(mom_list[1]))]
    #    energy_data = [energy_list[1][i] for i in range(len(energy_list[1]))]
    #    for i in range(length // energy_interval + 1):
    #        data[energy_interval * i][1] = energy_data[i]

    if args.momentum or args.force or not (energy_interval is None):
        df = pd.DataFrame(data, index=timestep_list, columns=data_columns)
        df.index.name = "Time / yr"
        df.to_csv("data.csv")
