from galaxies.elliptical import Elliptical
import numpy as np
import astropy.units as u
import grav_field

gal = Elliptical(particle_count=100)

for i in gal.particles:
    print(i)

TIMESTEP = (100 * u.yr).to(u.s)
LENGTH = (10**8) * u.yr
for t in range(int(LENGTH/TIMESTEP)+1):
    for i in range(gal.particle_count):
        for j in range(i, gal.particle_count):
            if i == j:
                continue

            force_ij = grav_field.get_force(gal.particles[i], gal.particles[j])
            gal.particles[i].apply_force(force_ij, TIMESTEP)
            gal.particles[j].apply_force(-force_ij, TIMESTEP)

    for i in range(gal.particle_count):
        gal.particles[i].move_particle(TIMESTEP)
        
    positions = np.array([i.position for i in gal.particles])
    momentums = np.array([i.momentum for i in gal.particles]).T
    
    print("Time: {:.04}, COM: {:.04}, Total Mom: {}, P0: {}".format((t * TIMESTEP).to(u.yr), positions.mean(), [np.sum(momentums[0]), np.sum(momentums[1]), np.sum(momentums[2])], gal.particles[0]))
