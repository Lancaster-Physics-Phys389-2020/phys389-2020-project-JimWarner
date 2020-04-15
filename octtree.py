"""
Module containing the OctTree class and the Barnes-Hut approximation method
    which uses it.
"""

import numpy as np
import astropy.units as u
import grav_field
from particles.particle import Particle

class OctTree:
    """
    Class used to initialise and control the octtree used to accelerate the
        force calculations.

    It is a logical tree which subdivides the particles spatially and so can
        approximate many distant particles as a single one at their centre
        of mass.
    """

    def __init__(self, centre=np.array([0.0, 0.0, 0.0], dtype=float), radius=0):

        self.centre = np.array(centre)
        self.radius = radius
        self.particle = None
        self.children = [None] * 8

    def build_tree(self, particles):
        """
        Constructs an octree containing the particles provided.

        Only inserts particles within volume contained by the base node as
            any particles outside that will not have a noticable impact on
            the other particles.
        """

        for particle in particles:
            # Check particle is inside box
            if (np.abs(particle.position).value < self.radius).all():
                self.insert_particle(particle)

        # Tidy up empty nodes
        self.kill_empty_leaves()
        self.compute_masses()

    def get_new_centre(self, i):
        """
        Calculates and returns the coordinates of the centre of the child
            node with the provided index
        """
        new_centre = [self.centre[0], self.centre[1], self.centre[2]]
        if i & 4 == 4:
            new_centre[0] += self.radius * 0.5
        else:
            new_centre[0] -= self.radius * 0.5

        if i & 2 == 2:
            new_centre[1] += self.radius * 0.5
        else:
            new_centre[1] -= self.radius * 0.5

        if i & 1 == 1:
            new_centre[2] += self.radius * 0.5
        else:
            new_centre[2] -= self.radius * 0.5

        return new_centre

    def insert_particle(self, particle):
        """
        Inserts a single particle into the tree.

        Recursively tries to insert the particle into the correct octant
            until it reaches a leaf node, where it either gets inserted or
            moves the previous particle into a lower layer of the tree and
            inserts itself into that layer.
        """

        if self.is_leaf_node_quick():
            if self.particle is None:
                # If self.particle is None the octant is not initialised so
                #     insert the new particle here
                self.particle = particle
            else:
                # If the octant has been initialised need to generate its
                #     child nodes and move the old particle to one of them
                #     as well as add the new particle
                old_particle = self.particle
                self.particle = None

                # Create the 8 children
                for i in range(8):
                    new_centre = self.get_new_centre(i)
                    self.children[i] = OctTree(centre=new_centre,
                                               radius=(self.radius / 2.0))

                # Insert particles into new children
                old_octant = self.get_octant_containing_particle(old_particle)
                self.children[old_octant].insert_particle(old_particle)

                octant = self.get_octant_containing_particle(particle)
                self.children[octant].insert_particle(particle)
        else:
            octant = self.get_octant_containing_particle(particle)
            self.children[octant].insert_particle(particle)

    def insert_new_particle(self, particle):
        """
        Inserts a new Particle into an already constructed tree.
        Does not use any shortcuts or initialise anything unecessary.
        """

        if self.is_leaf_node():
            if self.particle is None:
                # If no particle just insert particle
                self.particle = particle
            else:
                # If leaf node has a particle
                #     need to replace with com particle
                old_particle = self.particle
                total_mass = particle.mass + old_particle.mass
                com = (particle.position * particle.mass +
                       old_particle.position * old_particle.mass) / total_mass
                self.particle = Particle(position=com.value,
                                         mass=total_mass.value)

                # Create and insert particles into child nodes
                old_octant = self.get_octant_containing_particle(old_particle)
                new_centre = self.get_new_centre(old_octant)
                self.children[old_octant] = OctTree(centre=new_centre,
                                                    radius=(self.radius / 2.0))
                self.children[old_octant].insert_new_particle(old_particle)

                octant = self.get_octant_containing_particle(particle)
                new_centre = self.get_new_centre(octant)
                self.children[octant] = OctTree(centre=new_centre,
                                                radius=(self.radius / 2.0))
                self.children[octant].insert_new_particle(particle)
        else:
            # If not a leaf node add to com particle and move on
            old_particle = self.particle
            total_mass = particle.mass + old_particle.mass
            com = (particle.position * particle.mass +
                   old_particle.position * old_particle.mass) / total_mass
            self.particle.position = com.value * u.m
            self.particle.mass = total_mass.value * u.kg

            octant = self.get_octant_containing_particle(particle)
            # If child is uninitialised need to initialise it
            if self.children[octant] is None:
                new_centre = self.get_new_centre(octant)
                self.children[octant] = OctTree(centre=new_centre,
                                                radius=(self.radius/2.0))
            self.children[octant].insert_new_particle(particle)

    def compute_masses(self):
        """
        Calculates the values and locations for the centre of mass particles
        """
        # Only run on nodes without a particle
        if self.particle is None:
            total_mass = 0.0
            pos = np.array([0.0, 0.0, 0.0])
            for i in range(len(self.children)):
                # Ignore uninitialised nodes
                if self.children[i] is None:
                    continue

                # Run on child nodes
                self.children[i].compute_masses()
                total_mass += self.children[i].particle.mass.value
                pos += (self.children[i].particle.mass.value *
                        self.children[i].particle.position.value)

            self.particle = Particle(position=(pos / total_mass),
                                     mass=total_mass)

    def get_octant_containing_particle(self, particle):
        """
        Calculates and returns which of the octants in the current branch
            contains the region the provided particle is in.
        """

        pos = particle.position.value
        #print(pos)
        #print(self.centre)

        octant = 0
        if pos[0] >= self.centre[0]:
            octant |= 4
        if pos[1] >= self.centre[1]:
            octant |= 2
        if pos[2] >= self.centre[2]:
            octant |= 1

        return octant

    def is_leaf_node(self):
        """
        Checks whether the current OctTree object is a leaf node and returns
            the answer as a boolean
        """

        for child in self.children:
            # If any of the child nodes have been initialised the current
            #     object is not a leaf
            if child is not None:
                return False
        return True

    def is_leaf_node_quick(self):
        """
        Checks whether the first child node has been initialised as in the
            initialisation process either all the nodes will have been
            initialised or none will.
        Only used in initialisation.
        """

        return self.children[0] is None

    def kill_empty_leaves(self):
        """
        Assignes all unused leaf nodes to None to allow easier checking.
        """
        for i in range(len(self.children)):
            # Only called at end of initialisation so can use quick check
            if self.children[i].is_leaf_node_quick():
                # If leaf node with no particle kill it
                if self.children[i].particle is None:
                    self.children[i] = None
            else:
                self.children[i].kill_empty_leaves()


def barnes_hut_approximation(node, particle):
    """
    Calculates the force on a particle from all the other particles in the
        provided octtree using the barnes hut approximation.
    """
    # Approximation parameter
    theta = 0.5
    total_force = np.array([0.0, 0.0, 0.0]) * u.N

    if node.is_leaf_node():
        # Calculate the force from the one particle
        total_force += grav_field.get_force(particle, node.particle)
    else:
        vector = Particle.vector_between(particle, node.particle).value
        distance_to_particle = np.sqrt(vector.dot(vector))
        if (2 * node.radius) / distance_to_particle < theta:
            # Particle is distant, can use approximation
            total_force += grav_field.get_force(particle, node.particle)
        else:
            # Particle is close, must be more accurate
            for child in node.children:
                if child is not None:
                    total_force += barnes_hut_approximation(child, particle)

    return total_force
