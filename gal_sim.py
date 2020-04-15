from galaxies.elliptical import Elliptical
import numpy as np
import astropy.units as u
#import grav_field
import multiprocessing
#from octtree import OctTree, barnes_hut_approximation
#from particle_mesh import ParticleMesh
#import png
#import copy
import random
import pandas as pd
#from arguments import parse_arguments
#import sys
import time
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#import os
#from particles.particle import Particle
import pickle

import sim
    
def move_particles(timestep):
    #global sim.particles
    
    for i in range(len(sim.particles)):
        sim.particles[i].move_particle(timestep)
        
def apply_forces(pool, timestep):
    #global sim.particles
    
    #print("SENDING DATA")
    
    sim.send_data(pool)
    
    #print("APPLY FORCES")
    
    pool.imap(sim.get_forces_tree, range(len(sim.particles)))
    
    #print("RETRIEVING FORCES")
    
    results = sim.get_results(pool)
    for array in results:
        for index, force in array:
            #(index, force) = ret_queue.get()
            sim.particles[index].apply_force(force, timestep)
    
def check_forces(pool):
    #global sim.particles
    #global sim.particle_count

    sample_size = 25
    
    sim.send_data(pool)

    #print("CHECKING FORCES")

    if sample_size > sim.particle_count:
        sample_size = sim.particle_count
     
    indices = random.sample(range(len(sim.particles)), sample_size)
    #print(indices)

    pool.imap(sim.get_forces_raw, indices)
    results = []
    for arr in sim.get_results(pool):
        results += arr
    raw_forces = sorted(results, key=lambda item: item[0])
    raw_forces = np.array([i[1] for i in raw_forces])

    #print(results)
    #print(raw_forces)
    
    pool.imap(sim.get_forces_tree, indices)
    results = []
    for arr in sim.get_results(pool):
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
    #global particles
    #global particle_count

    sim.send_data(pool)

    start_raw = time.time()
    pool.imap(sim.get_forces_raw, range(sim.particle_count))
    sim.get_results(pool)
    end_raw = time.time()

    start_tree = time.time()
    pool.imap(sim.get_forces_tree, range(sim.particle_count))
    sim.get_results(pool)
    end_tree = time.time()

    return [end_raw - start_raw, end_tree - start_tree]
    
def run(args):
    #global sim.particles
    #global sim.timestep
    #global sim.ret_queue
    #global sim.particle_queue
    #global sim.radius
    #global sim.particle_count
    #global sim.energy_list

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
        start_timestep = sim.read_data()
        timestep_list = pickle.load(open("sim_data_timestep.p", 'rb'))
        timing_list = pickle.load(open("sim_data_timing.p", 'rb'))
        energy_list = pickle.load(open("sim_data_energy.p", 'rb'))
        mom_list = pickle.load(open("sim_data_mom.p", 'rb'))
        force_factor_list = pickle.load(open("sim_data_force.p", 'rb'))
    else:
        
        #tree = OctTree(radius = 1000)
        #p1 = Particle([1, 1, 1])
        #p2 = Particle([2, 2, 2])
        #tree.build_tree([p1, p2])
        #print(sys.getsizeof(1e32))
        #sys.exit()
        
        sim.particle_count = args.particle_no

        count_modifier = processes
        if count_modifier % 2 == 1:
            count_modifier *= 2

        # Guarantees the number of particles divides by the number of processes
        sim.particle_count = sim.particle_count + count_modifier - sim.particle_count % count_modifier
        particles_per_galaxy = sim.particle_count // 2

        print("Modelling with {} particles".format(sim.particle_count))
    
        #gal = Elliptical(particle_count=100)
    
        #count = 400

        #print("Starting init")
        #tree_start = time.time()
        gal1 = Elliptical(particle_count=particles_per_galaxy, position=[5e20, 5e20, 5e20])
        gal2 = Elliptical(particle_count=particles_per_galaxy, position=[-5e20, -5e20, -5e20])
        sim.particles = gal1.particles + gal2.particles

        sim.timestep = (args.timestep * u.yr).to(u.s)
        sim.length = args.length
        sim.radius = 1e22

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
    
    
    

    #com = grav_field.get_centre_of_mass(particles)
    #for particle in particles:
    #    particle.position -= com

    #particles = gal.particles
    #poses = []
    #temp = []
    #for p in particles:
    #    temp.append([p.position[0].value, p.position[1].value, p.position[2].value])
    #poses.append(temp)
    
    
        
    
    c = [0.6756, -0.1756, -0.1756, 0.6756]
    d = [1.3512, -1.7024, 1.3512]
    
    with multiprocessing.Pool(processes=processes) as pool:
        if start_timestep == 0:
            if not (energy_interval is None):
                energy_list.append(sim.get_energy(pool))

            if args.force:
                force_factor_list.append(check_forces(pool))

            if not (timing_interval is None):
                timing_list.append(get_timing(pool))

            if args.momentum:
                momentums = np.array([i.momentum for i in sim.particles]).T
                mom_list.append(np.sum(momentums[0]) + np.sum(momentums[1]) + np.sum(momentums[2]))

        for t in range(start_timestep, sim.length):
            start = time.time()            
            sim.write_image(sim.particles, t, "gal_sim")

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
            move_particles(sim.timestep * c[0])
            #force
            apply_forces(pool, sim.timestep * d[0])
            #move
            move_particles(sim.timestep * c[1])
            #force
            apply_forces(pool, sim.timestep * d[1])
            #move
            move_particles(sim.timestep * c[2])
            #force
            apply_forces(pool, sim.timestep * d[2])
            #move
            move_particles(sim.timestep * c[3])
            
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
                momentums = np.array([i.momentum for i in sim.particles]).T
                mom_list.append(np.sum(momentums[0]) + np.sum(momentums[1]) + np.sum(momentums[2]))

            if not (energy_interval is None):
                if t % energy_interval == 0:
                    energy_list.append(sim.get_energy(pool))

            if not (timing_interval is None):
                if t % timing_interval == 0:
                    timing_list.append(get_timing(pool))

            timestep_list.append(np.round(t * sim.timestep.to(u.yr).value))

            if not (args.save_interval is None):
                if t % args.save_interval == 0:
                    pickle.dump(force_factor_list, open("sim_data_force.p", 'wb'))
                    pickle.dump(mom_list, open("sim_data_mom.p", 'wb'))
                    pickle.dump(energy_list, open("sim_data_energy.p", 'wb'))
                    pickle.dump(timing_list, open("sim_data_timing.p", 'wb'))
                    pickle.dump(timestep_list, open("sim_data_timestep.p", 'wb'))
                    sim.save_data(t)
        
            #print("Time: {:.04}, COM: {:.04}, Total Mom: {}, P0: {}".format((t * timestep).to(u.yr), positions.mean(), [np.sum(momentums[0]), np.sum(momentums[1]), np.sum(momentums[2])], particles[1]))
            end = time.time()
            print("Time: {:.04}, Time step took: {}s".format((t * sim.timestep).to(u.yr), end - start))

            
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

        mom_list = np.array(mom_list).T
        ax.plot(timestep_list, mom_list, label="Total Momentum")
        ax.legend()
        ax.set_ylabel("Momentum / Ns")
        ax.set_xlabel("Time / yr")
        #fig.set_size_inches(8, 8)
        fig.savefig("gal_sim_momentum_p{}_t{}_l{}.png".format(sim.particle_count, sim.timestep / (u.yr).to(u.s), sim.length), dpi=500)
            
    #mom_list = np.array(mom_list).T
    #print(mom_list)
    #for mom_list_list in mom_list:

    if not (timing_interval is None):
        timing_list = np.array(timing_list).T
        
    if not (energy_interval is None):
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        
        energy_list = np.array(energy_list).T
        timecodes = [timestep_list[energy_interval * i] for i in range(sim.length // energy_interval + 1)]
        ax2.plot(timecodes, energy_list[0], label="Total Energy")
        ax2.plot(timecodes, energy_list[1], label="Kinetic Energy")
        ax2.plot(timecodes, energy_list[2], label="Potential Energy")
        ax2.legend()
        ax2.set_ylabel("Energy / J")
        ax2.set_xlabel("Time / yr")
        #fig2.set_size_inches(8, 8)
        fig2.savefig("gal_sim_energy_p{}_t{}_l{}.png".format(sim.particle_count, sim.timestep / (u.yr).to(u.s), sim.length), dpi=500)

    if args.force:
        force_factor_list = np.array(force_factor_list).T
        fig3 = plt.figure()
        ax3 = fig3.add_subplot(111)
        ax3.errorbar(timestep_list, force_factor_list[0], force_factor_list[1])
        ax3.set_ylabel("Relative Force")
        ax3.set_xlabel("Time / yr")
        #fig3.set_size_inches(8, 8)
        fig3.savefig("gal_sim_force_error_p{}_t{}_l{}.png".format(sim.particle_count, sim.timestep / (u.yr).to(u.s), sim.length), dpi=500)
 
    #ax.set_xlim3d(-1e21, 1e21)
    #ax.set_ylim3d(-1e21, 1e21)
    #ax.set_zlim3d(-1e21, 1e21)
    #plt.show()

    data_columns = []
    data = []
    data_width = 0
    data_length = 0
    data_index = 0

    #if args.momentum or args.force:
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

    if data_length == 0:
        data_length = len(timestep_list)
    
    data = [[np.nan for _ in range(data_width)] for _ in range(data_length)]

    #print(timing_list)
    #print(data_width, data_length)
    #print(data)
    
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
        for i in range(len(energy_list[0])):
            data[energy_interval * i][data_index] = energy_list[0][i]
            data[energy_interval * i][data_index + 1] = energy_list[1][i]
            data[energy_interval * i][data_index + 2] = energy_list[2][i]
        data_index += 3
        data_columns.append("Total_Energy / J")
        data_columns.append("Total_Kinetic_Energy / J")
        data_columns.append("Total_Potential_Energy / J")
        
    if not (timing_interval is None):
        for i in range(len(timing_list[0])):
            data[timing_interval * i][data_index] = timing_list[0][i]
            data[timing_interval * i][data_index + 1] = timing_list[1][i]
        data_index += 2
        data_columns.append("Brute_Force_Timing / s")
        data_columns.append("Octtree_Timing / s")
    #elif not (timing_interval is None):
    #    data = timing_list
    #    data_columns.append("Brute_Force_Timing / s")
    #    data_columns.append("Octtree_Timing / s")
                   
    #    data = [[mom_list[1][i], np.nan] for i in range(len(mom_list[1]))]
    #    energy_data = [energy_list[1][i] for i in range(len(energy_list[1]))]
    #    for i in range(length // energy_interval + 1):
    #        data[energy_interval * i][1] = energy_data[i]

    if args.momentum or args.force or not (energy_interval is None) or not (timing_interval is None):
        df = pd.DataFrame(data, index=timestep_list, columns=data_columns)
        df.index.name = "Time / yr"
        df.to_csv("gal_sim_data_p{}_t{}_l{}.csv".format(sim.particle_count, sim.timestep / (u.yr).to(u.s), sim.length))
