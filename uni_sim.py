import particle_mesh
import universe
import astropy.units as u
import multiprocessing
import sim
import random
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
import pickle

def move_particles(timestep):
#    global particles
#    global radius
    
    for i in range(len(sim.particles)):
        sim.particles[i].move_particle(timestep)
        # Periodic boundary conditions
        for j in range(3):
            if sim.particles[i].position[j].value > sim.radius:
                sim.particles[i].position[j] = (sim.particles[i].position[j].value % sim.radius - sim.radius) * u.m
            elif sim.particles[i].position[j].value < -sim.radius:
                sim.particles[i].position[j] = (sim.particles[i].position[j].value % sim.radius) * u.m

def apply_forces(timestep):
    #global particles
    
    results = sim.get_forces_mesh()
    for index, force in enumerate(results):
        sim.particles[index].apply_force(force, timestep)

def check_forces(pool):
    #global sim.particles
    #global sim.particle_count

    sample_size = 25
    
    sim.send_data(pool)

    if sample_size > sim.particle_count:
        sample_size = sim.particle_count
     
    indices = random.sample(range(len(sim.particles)), sample_size)
    #print(indices)

    pool.imap(sim.get_forces_tree, indices)
    results = []
    for arr in sim.get_results(pool):
        results += arr
    tree_forces = sorted(results, key=lambda item: item[0])
    tree_forces = np.array([i[1] for i in tree_forces])
    
    mesh_forces = np.array(sim.get_forces_mesh())[indices]
    
    factors = np.abs(mesh_forces / tree_forces)
    #print(factors)
    factors = np.array([np.sum(i) / 3 for i in factors])
    #print(factors)
    return [factors.mean(), factors.std()]

def get_timing(pool):
    sim.send_data(pool)
    
    start_tree = time.time()
    pool.imap(sim.get_forces_tree, range(sim.particle_count))
    sim.get_results(pool)
    end_tree = time.time()

    start_mesh = time.time()
    sim.get_forces_mesh()
    end_mesh = time.time()

    return [end_tree - start_tree, end_mesh - start_mesh]

def run(args):

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
        sim.particle_count = args.particle_no

        # Model requires the number of particles to be cubic
        sim.particle_count = (int(sim.particle_count ** (1/3)) + 1) ** 3
        print("Modelling with {} particles".format(sim.particle_count))

        sim.particles = universe.get_particles(sim.particle_count)

        sim.timestep = (args.timestep * u.yr).to(u.s)
        sim.length = args.length
        sim.radius = 5e24
        
    

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
            sim.write_image(sim.particles, t, "univ_sim")

            t+=1

            #move
            move_particles(sim.timestep * c[0])
            #force
            apply_forces(sim.timestep * d[0])
            #move
            move_particles(sim.timestep * c[1])
            #force
            apply_forces(sim.timestep * d[1])
            #move
            move_particles(sim.timestep * c[2])
            #force
            apply_forces(sim.timestep * d[2])
            #move
            move_particles(sim.timestep * c[3])

            if args.force:
                force_factor_list.append(check_forces(pool))

            if args.momentum:
                momentums = np.array([i.momentum for i in sim.particles]).T
                mom_list.append(np.sum(momentums[0]) + np.sum(momentums[1]) + np.sum(momentums[2]))

            if not (energy_interval is None):
                if t % energy_interval == 0:
                   energy_list.append(sim.get_energy(pool))

            if not (timing_interval is None):
                if t % timing_interval == 0:
                    timing_list.append(get_timing(pool))

            timestep_list.append(np.round(t * sim.timestep.to(u.yr).value))#

            if not (args.save_interval is None):
                if t % args.save_interval == 0:
                    pickle.dump(force_factor_list, open("sim_data_force.p", 'wb'))
                    pickle.dump(mom_list, open("sim_data_mom.p", 'wb'))
                    pickle.dump(energy_list, open("sim_data_energy.p", 'wb'))
                    pickle.dump(timing_list, open("sim_data_timing.p", 'wb'))
                    pickle.dump(timestep_list, open("sim_data_timestep.p", 'wb'))
                    sim.save_data(t)

            end = time.time()
            print("Time: {:.04}, Time step took: {}s".format((t * sim.timestep).to(u.yr), end - start))

    if args.momentum:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        mom_list=np.array(mom_list).T
        ax.plot(timestep_list, mom_list, label="Total Momentum")
        ax.legend()
        ax.set_ylabel("Momentum / Ns")
        ax.set_xlabel("Time / yr")
        #fig.set_size_inches(8, 8)
        fig.savefig("univ_sim_momentum_p{}_t{}_l{}.png".format(sim.particle_count, sim.timestep / (u.yr).to(u.s), sim.length), dpi=500)

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
        fig2.savefig("univ_sim_energy_p{}_t{}_l{}.png".format(sim.particle_count, sim.timestep / (u.yr).to(u.s), sim.length), dpi=500)

    if args.force:
        force_factor_list = np.array(force_factor_list).T
        fig3 = plt.figure()
        ax3 = fig3.add_subplot(111)
        ax3.errorbar(timestep_list, force_factor_list[0], force_factor_list[1])
        ax3.set_ylabel("Relative Force")
        ax3.set_xlabel("Time / yr")
        #fig3.set_size_inches(8, 8)
        fig3.savefig("univ_sim_force_error_p{}_t{}_l{}.png".format(sim.particle_count, sim.timestep / (u.yr).to(u.s), sim.length), dpi=500)

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
        data_columns.append("Octtree_Timing / s")
        data_columns.append("Mesh_Timing / s")

    if args.momentum or args.force or not (energy_interval is None) or not (timing_interval is None):
        df = pd.DataFrame(data, index=timestep_list, columns=data_columns)
        df.index.name = "Time / yr"
        df.to_csv("univ_sim_data_p{}_t{}_l{}.csv".format(sim.particle_count, sim.timestep  / (u.yr).to(u.s), sim.length))
