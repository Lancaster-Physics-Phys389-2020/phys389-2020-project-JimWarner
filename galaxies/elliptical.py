"""
Module containing the elliptical galaxy class
"""

import random
import astropy.units as u
from astropy.constants import G
import numpy as np
from particles.particle import Particle
from galaxies.galaxy import Galaxy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from octtree import OctTree, barnes_hut_approximation

class Elliptical(Galaxy):
    """
    Class used to initialise and control any elliptical galaxies
    """
    
    def init_particles(self):
        """
        Generates a list of particles at random positions within a spherical region
        """
        galaxy_radius = (u.lyr).to(u.m) * 50000
        galaxy_mass = 1e12 * (u.M_sun).to(u.kg) 
        particle_mass = galaxy_mass / self.particle_count
        
        particles = []
        poses = []

        #particles.append(Particle([0.0, 0.0, 0.0], mass=galaxy_mass*0.1, momentum=[0.0, 0.0, 0.0]))
        
        for i in range(1, self.particle_count):
            phi = random.random()*2*np.pi
            costheta = random.random() * 2 - 1
            theta = np.arccos(costheta)

            r = random.random() ** 0.7 * galaxy_radius
            
            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta)
            
            particle_pos = np.array([x, y, z])
            #poses.append(particle_pos)
            particle = Particle(particle_pos, mass = particle_mass)
            particles.append(particle)

        # sort particles by distance from centre of galaxy
        particles.sort(key=lambda p:np.sum(p.position**2))
        particle_tree = OctTree(radius=galaxy_radius)

        for i in range(1, len(particles)):
            #print(i)
            inner_mass = particle_mass * np.sqrt(i)

            distance_from_centre = np.sqrt(particles[i].position[0]**2 + particles[i].position[1]**2 + particles[i].position[2] ** 2)

            particle_tree.insert_new_particle(particles[i - 1])
            
            if i > 3:
                force = barnes_hut_approximation(particle_tree, particles[i]).value
                speed = np.sqrt(np.sqrt(np.sum(force**2)) * distance_from_centre / particles[i].mass)
            else:
                speed = 0.0 * u.m / u.s

            #print(speed)
                
            #print(galaxy_radius , distance_from_centre)
            speed = speed# / np.sqrt(galaxy_radius / distance_from_centre)
            #print(speed)
            direction = np.cross([particles[i].position[0].value, particles[i].position[1].value, 0.0], [0.0, 0.0, 1.0])
            direction = direction / np.sqrt(np.sum(direction ** 2))
            momentum = (speed * direction * particle_mass).value
            momentum_boost = []

            #print(particles[i].momentum)
            particles[i].position += self.position
            particles[i].momentum = momentum * u.N * u.s
            #print(particles[i].momentum)

        for particle in particles:
            #print(particle.position, (particle.position > 0.0).any())
            if (particle.position > 0.0).any():
                momentum_boost = np.array([-10000 * particle_mass] * 3)
            else:
                momentum_boost = np.array([10000 * particle_mass] * 3)

            #print(particle_pos, (particle_pos > 0.0).any(), momentum_boost)

            particle.momentum += momentum_boost * u.N * u.s
            

        #poses=np.array(poses).T

        #fig = plt.figure()
        #ax = fig.add_subplot(111, projection='3d')

        #ax.scatter(xs=poses[0], ys=poses[1], zs=poses[2], s=2)
        #plt.show()
        return particles
