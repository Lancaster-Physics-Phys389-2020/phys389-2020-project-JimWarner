"""
Module containing the main code to setup and run the simulation
"""

import astropy.units as u
from particles.particle import Particle
import grav_field

from particle_mesh import ParticleMesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sys

# Test model based on the international space station
PARTICLE1 = Particle([0.0, 6800000, 0.0], [7.5 * 1000 * 4000 * 3e8, 0, 0], 4000)
PARTICLE2 = Particle(position=[0.0, 0.0, 10000], mass=(1 * u.M_earth).to(u.kg).value)
TIMESTEP = (10000 * 1e-8 * u.ms).to(u.s)

print(PARTICLE2.position)

def get_force_raw(particle1, particle2):
    force_12 = grav_field.get_force(particle1, particle2)
    return [force_12, -force_12]

def get_force_mesh(particle1, particle2):
    mesh = ParticleMesh(200, 1e9)
    mesh.insert_particles([particle1, particle2])
    mesh.calculate_potential()
    mesh.calculate_force()
    return mesh.get_forces([particle1, particle2])

POSES = []

def show_potential(particles):
    count = 200
    pm = ParticleMesh(count, 1e8)
    pm.insert_particles(particles)
    #print(pm.grid)
    pm.calculate_potential()
    #pm.calculate_force()

    #grid = [[[1/np.sqrt(np.sum(np.abs(pm.grid[i][j][k]))) for k in range(count)] for j in range(count)] for i in range(count)]
    #for i in range(pm.grid_points):
    #    for j in range(pm.grid_points):
    #        for k in range(pm.grid_points):
    #            grid[i][j][k] = np.sum(np.abs(pm.grid[i][j][k]))
    #print(pm.grid)
    #pm.grid = grid
    
    arr = []
    index = count // 2
    for i in range(len(pm.grid[index])):
        for j in range(len(pm.grid[index][i])):
            arr.append([i, j, pm.grid[index][i][j]])
            #if pm.grid[i][index][j][2] > 0:
            #    arr.append([i, j, 1])
            #elif pm.grid[index][i][j][1] == 0:
            #    arr.append([i, j, 0])
            #else:
            #    arr.append([i, j, -1])
                
    arr = np.array(arr).T
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs=arr[0], ys=arr[1], zs=arr[2], s=2)
    #ax.set_zlim3d(0, 5e-8)
    plt.show()
    sys.exit(0)

#show_potential([PARTICLE1, PARTICLE2])
    
# Run for one orbit of 88 minutes
for i in range(int(88 * 60 * 2e-9 / TIMESTEP.value) + 1):
    
    #forces = get_force_raw(PARTICLE1, PARTICLE2)
    forces = get_force_mesh(PARTICLE1, PARTICLE2)

    print(forces[0])
    print(forces[1])
    
    PARTICLE1.apply_force(forces[0], TIMESTEP)
    PARTICLE2.apply_force(forces[1], TIMESTEP)
    PARTICLE1.move_particle(TIMESTEP)
    PARTICLE2.move_particle(TIMESTEP)

    POSES.append([[PARTICLE1.position[0].value, PARTICLE1.position[1].value, PARTICLE1.position[2].value], [PARTICLE2.position[0].value, PARTICLE2.position[1].value, PARTICLE2.position[2].value]])
    
    pot_energy = grav_field.get_potential_energy(PARTICLE1, PARTICLE2)
    p1_energy = PARTICLE1.kinetic_energy() + pot_energy
    p2_energy = PARTICLE2.kinetic_energy() + pot_energy
    print(i, "Minute: {:.2f}, Pos1: {}, Pos2: {}, E1: {}, E2: {}, E_Tot: {}".
          format(i * TIMESTEP.value / 60,
                 PARTICLE1.position, PARTICLE2.position,
                 p1_energy, p2_energy,
                 p1_energy + p2_energy))

    if i == 100:
        break

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#for p in poses:
#    print(p)
POSES=np.array(POSES).T
#print(len(poses[0]))
#print(poses)
for i in range(len(POSES[0])):
    ax.scatter(xs=POSES[0][i], ys=POSES[1][i], zs=POSES[2][i], s=2)
    ax.plot(xs=POSES[0][i], ys=POSES[1][i], zs=POSES[2][i])
    
#mom_list = np.array(mom_list).T
#print(mom_list)
#for mom_list_list in mom_list:
#    fig2 = plt.figure()
#    ax2 = fig2.add_subplot(111)
#    ax2.scatter(x=mom_list_list[0], y=mom_list_list[1], s=2)

ax.set_xlim3d(-1e7, 1e7)
ax.set_ylim3d(-1e7, 1e7)
ax.set_zlim3d(-1e7, 1e7)
plt.show()
fig.set_size_inches(8, 8)
fig.savefig("test_orbit.png", dpi=500)
