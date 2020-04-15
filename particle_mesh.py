import numpy as np
from numpy.fft import fftn, ifftn
import astropy.units as u

class ParticleMesh:
    """
    Class used to initialise and control the mesh used to accelerate the force calculations.

    The particle masses are applied to the mesh which then undergoes a fourier transformation to produce the density that can be used to calculate the force for each particle
    """

    def __init__(self, grid_points, radius, centre=np.array([0.0, 0.0, 0.0], dtype=float)):

        self.centre = np.array(centre) + np.array([radius] * 3)
        self.radius = radius
        self.grid_length = radius * 2 / (grid_points - 1)
        self.grid_points = grid_points
        self.grid = np.array([[[0.0] * grid_points] * grid_points] * grid_points, dtype=np.double)

    def insert_particles(self, particles):
        """
        Inserts the particles into the mesh using the cloud in cell method where each particle assigns its mass to its nearest grid points proportionally to distance.
        """

        #print("inserting particles")
        
        for particle in particles:
            # Check particle is not outside box
            if (np.abs(particle.position).value > self.radius).any():
                continue
            
            # Base is the grid point closest to 0 adjacent to the particle
            grid_pos = self.get_grid_pos(particle)
            base_x = int(grid_pos[0])
            base_y = int(grid_pos[1])
            base_z = int(grid_pos[2])

            split = self.get_grid_split(particle)

            mass = particle.mass.value

            self.grid[base_x    ][base_y    ][base_z    ] += mass * split[0]
            self.grid[base_x    ][base_y    ][base_z + 1] += mass * split[1]
            self.grid[base_x    ][base_y + 1][base_z    ] += mass * split[2]
            self.grid[base_x    ][base_y + 1][base_z + 1] += mass * split[3]
            self.grid[base_x + 1][base_y    ][base_z    ] += mass * split[4]
            self.grid[base_x + 1][base_y    ][base_z + 1] += mass * split[5]
            self.grid[base_x + 1][base_y + 1][base_z    ] += mass * split[6]
            self.grid[base_x + 1][base_y + 1][base_z + 1] += mass * split[7]

        self.grid = self.grid / np.sum(self.grid)

    def calculate_potential(self):
        """
        Calculates the gravitational potential at each grid point
        """

        #print("Calculating potential")
        
        x = np.array([[[i for i in range(self.grid_points)]]])
        y = np.array([[[i] for i in range(self.grid_points)]])
        z = np.array([[[i]] for i in range(self.grid_points)])
        
        
        k_squared = np.sin(np.pi*x/self.grid_points)**2 + np.sin(np.pi*y/self.grid_points)**2 + np.sin(np.pi*z/self.grid_points)**2
        
        # Avoid the divide by 0
        k_squared[0][0][0] = 1.0
        
        g_hat = -1/k_squared
        
        # Fix the earlier edit
        g_hat[0][0][0] = 0.0
        
        rho_hat = fftn(self.grid)

        #print(g_hat)
        #print(rho_hat)
        
        phi_hat = rho_hat * g_hat
        
        # Assign potential to grid
        self.grid = ifftn(phi_hat).real
            
    def calculate_force(self):
        """
        Calculates the force at each grid point on the inside of the space.

        Uses the gradient of the potential calculated by self.calculate_potential()
        """

        # Transposed as the gradient is returned as 3 arrays for the 3 dimentions
        self.grid = -1 * np.array(np.gradient(self.grid), dtype=np.double).T / (self.grid_length)

                    
    def get_forces(self, particles):
        """
        Calculates the appropriate force on each of the particles and applies it to
            them with the provided timestep
        """

        forces = []
        
        for particle in particles:
            # Check particle is not outside box
            if (np.abs(particle.position).value > self.radius).any():
                #print(particle.position)
                continue
            
            # Base is the grid point closest to 0 adjacent to the particle
            grid_pos = self.get_grid_pos(particle)
            base_x = int(grid_pos[0])
            base_y = int(grid_pos[1])
            base_z = int(grid_pos[2])

            split = self.get_grid_split(particle)

            #print(split)
            #print(np.sum(split))

            force = np.array([0.0] * 3)
            force += self.grid[base_x    ][base_y    ][base_z    ] * split[0]
            force += self.grid[base_x    ][base_y    ][base_z + 1] * split[1]
            force += self.grid[base_x    ][base_y + 1][base_z    ] * split[2]
            force += self.grid[base_x    ][base_y + 1][base_z + 1] * split[3]
            force += self.grid[base_x + 1][base_y    ][base_z    ] * split[4]
            force += self.grid[base_x + 1][base_y    ][base_z + 1] * split[5]
            force += self.grid[base_x + 1][base_y + 1][base_z    ] * split[6]
            force += self.grid[base_x + 1][base_y + 1][base_z + 1] * split[7]

            force = force * u.N * particle.mass.value * 1e16

            #print(force)
            
            forces.append(force)

        return forces

    def get_grid_pos(self, particle):
        """
        Calculates the grid position of the provided particle
        """
        return (particle.position.value + self.centre) / self.grid_length
        
            
    def get_grid_split(self, particle):
        """
        Calculates the weighting each of the 8 nearest gridpoints of the given
            particle will have when it is split between them
        """

        # Base is the grid point closest to 0 adjacent to the particle
        grid_pos = self.get_grid_pos(particle)
        base_x = int(grid_pos[0]) + 0.5
        base_y = int(grid_pos[1]) + 0.5
        base_z = int(grid_pos[2]) + 0.5
        
        # Split is the position of the particle between base and base + 1
        x_split = grid_pos[0] - base_x
        y_split = grid_pos[1] - base_y
        z_split = grid_pos[2] - base_z

        split = [0.0] * 8
        
        split[7] = (1 - x_split) * (1 - y_split) * (1 - z_split)
        split[6] = (1 - x_split) * (1 - y_split) *      z_split
        split[5] = (1 - x_split) *      y_split  * (1 - z_split)
        split[4] = (1 - x_split) *      y_split  *      z_split
        split[3] =      x_split  * (1 - y_split) * (1 - z_split)
        split[2] =      x_split  * (1 - y_split) *      z_split
        split[1] =      x_split  *      y_split  * (1 - z_split)
        split[0] =      x_split  *      y_split  *      z_split

        return split
