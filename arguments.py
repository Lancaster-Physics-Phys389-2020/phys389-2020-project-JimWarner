import argparse
import multiprocessing
import sys

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run some simulations")

    #sims = parser.add_argument_group("Simulations")
    #parser.add_mutually_exclusive_group(required=True)
    #sims.add_argument("g",# "galaxies",
                      #help="Run the galaxy collision simulation",
                      #action="store_true")
    #sims.add_argument("u",# "universe",
                      #help="Run the universe evolution simulation",
                      #action="store_true")
    
    parser.add_argument("simulation",
                        help="The simulation to be run. g=Galaxy Collision, u=Universe evolution",
                        choices=["g", "u"])

    parser.add_argument("particle_no",
                        help="The number of particles used in the simulation",
                        type=int)

    parser.add_argument("timestep",
                        help="The length of each timestep in years",
                        type=float)

    parser.add_argument("length",
                        help="The length of the simulation in timesteps",
                        type=int)
    
    parser.add_argument("-p", "--processes",
                        help="The number of processes the program can use",
                        type=int,
                        default=multiprocessing.cpu_count())

    parser.add_argument("-m", "--momentum",
                        help="Record the momentum and plot it in a graph",
                        action="store_true")

    parser.add_argument("-e", "--energy",
                        help="Record the kinetic and potential energy every ENERGY_INTERVAL timesteps and plot them in a graph",
                        dest="energy_interval",
                        type=int)
    
    parser.add_argument("-f", "--force",
                        help="Calculate the accuracy of the forces as a fraction and plot this in a graph",
                        action="store_true")

    parser.add_argument("-t", "--time",
                        dest="time_interval",
                        help="Calculates the time taken for the force calculations at points throughout the simulation seperated by TIME_INTERVAL timesteps",
                        type=int)
    
    parser.add_argument("-i", "--images",
                        help="Save an image of the particles for each timestep",
                        action="store_true")

    parser.add_argument("-w", "--write",
                        dest="write_interval",
                        help="Writes the particle data to sim_data.csv every WRITE_INTERVAL timesteps.",
                        type=int)

    parser.add_argument("-r", "--read",
                        help="Reads initial conditions from .csv file")

    args = parser.parse_args()
    
    return args
