import numpy as np
import astropy.units as u
import grav_field
import multiprocessing
from octtree import OctTree, barnes_hut_approximation
from particle_mesh import ParticleMesh
import png
import copy
#import random
#import pandas as pd
import time
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#from particles.particle import Particle
import pickle

GOT_DATA = False
FORCE_LIST = []
TOTAL_PE = 0.0
particles = None
timestep = None
ret_queue=multiprocessing.Queue()
particle_queue=multiprocessing.Queue()
radius = None
particle_count = None
particle_tree = None
length = None

def get_forces_raw(index):
    global FORCE_LIST
    global particles

    #print(index)
    total_force = 0.0
    
    # Brute force
    for j in range(len(particles)):
        if j == index:
            continue
        total_force += grav_field.get_force(particles[index], particles[j])
        
    FORCE_LIST.append((index, total_force))


def get_forces_tree(index):
    global particles
    global particle_tree

    #print(particle_tree.particle)
    
    # Octtree
    force = barnes_hut_approximation(particle_tree, particles[index])
    #print(particle.momentum)
    #particle.apply_force(force, timestep)
    #particle.move_particle(timestep)
    
    #print("Process: {}, Index: {}, Force: {}".format(os.getpid(), index, total_force))
    
    #print(total_force)
    FORCE_LIST.append((index, force))
    #return (index, force)

def get_forces_mesh():
    global particles
    global radius

    mesh = ParticleMesh(grid_points=300, radius=radius)

    mesh.insert_particles(particles)
    mesh.calculate_potential()
    mesh.calculate_force()
    
    forces = mesh.get_forces(particles)
#    print(forces, len(particles))
    return forces

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
    global particle_queue
    global GOT_DATA
    
    #print("GETTING DATA", index, multiprocessing.current_process()._identity)
    
    if not GOT_DATA:
        particles = particle_queue.get()
        #for p in particles:
         #   print(p.position)
        particle_tree = OctTree(radius=radius)
        #print(particle_tree)
        particle_tree.build_tree(particles)
        #print(len(particles), radius)
        GOT_DATA = True
        
        
def send_data(pool):
    global particles
    global particle_queue
    
    loops = 0

    #print("Starting transfer")
    
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
    
    #trans_start=time.time()
    while len(force_list) < pool._processes:
        pool.map_async(send_results, range(pool._processes))
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
    
    #trans_start=time.time()
    while len(pe_list) < pool._processes:
        pool.map_async(send_energy_results, range(pool._processes))
        loops += 1
        for i in range(pool._processes):
            result = ret_queue.get()
            #print(results)
            if result != 0.0:
                pe_list.append(result)

    #print("Recieved data {}s. {} loops".format(time.time()-trans_start, loops))

    return np.sum(pe_list)
    
def get_energy(pool):
    global particles
    global energy_list
    #print("Getting Energy", 1)
    send_data(pool)
    
    #print("Getting Energy", 2)
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
    
    return [total_energy, total_ke, total_pe]
    #calc_end = time.time()
    #print("Calculation took: {}s".format(calc_end - calc_start))

def write_image(particles, t, name):
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
    png.from_array(img, 'L').save("frames/{}_p{}_t{}_l{}_pic{:05}.png".format(name, particle_count, timestep / (u.yr).to(u.s), length, t))


def save_data(t):
    pickle.dump(particles, open("sim_data_particles.p", 'wb'))
    pickle.dump([t, timestep, radius, particle_count, length], open("sim_data.p", 'wb'))

def read_data():
    global particles
    global timestep
    global radius
    global particle_count
    global length
    
    particles = pickle.load(open("sim_data_particles.p", 'rb'))
    data = pickle.load(open("sim_data.p", 'rb'))
    timestep = data[1]
    radius = data[2]
    particle_count = data[3]
    length = data[4]
    return data[0]

            
