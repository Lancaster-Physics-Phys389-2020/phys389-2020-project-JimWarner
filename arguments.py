"""
Module containing the functions required to handle the command line arguments
"""

import argparse
import multiprocessing

def parse_arguments():
    """
    Parses the command line arguments using the argparse package and returns
        the results as a single object`
    """
    parser = argparse.ArgumentParser(
        description="Runs n-body gravitational simulations of either the " +
        "collision between two elliptical galaxies or the evolution of the " +
        "universe over a large scale. The galaxy collision simulation " +
        "calculates the forces using tree codes and the universe simulation " +
        "uses the particle mesh method."
    )

    parser.add_argument("simulation",
                        help="The simulation to be run. " +
                        "g=Galaxy Collision, u=Universe evolution",
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
                        help="The number of processes the program can use. " +
                        "Defaults to the number of logical cores.",
                        type=int,
                        default=multiprocessing.cpu_count())

    parser.add_argument("-m", "--momentum",
                        help="Record the momentum and plot it in a graph",
                        action="store_true")

    parser.add_argument("-e", "--energy",
                        help="Record the kinetic and potential energy " +
                        "every ENERGY_INTERVAL timesteps and plot them " +
                        "in a graph",
                        dest="energy_interval",
                        type=int)

    parser.add_argument("-f", "--force",
                        help="Calculate the accuracy of the forces as " +
                        "a fraction and plot this in a graph",
                        action="store_true")

    parser.add_argument("-t", "--time",
                        dest="time_interval",
                        help="Calculates the time taken for the force " +
                        "calculations at points throughout the simulation " +
                        "seperated by TIME_INTERVAL timesteps",
                        type=int)

    parser.add_argument("-i", "--images",
                        help="Save an image of the particles for each timestep",
                        action="store_true")

    parser.add_argument("-s", "--save",
                        dest="save_interval",
                        help="Writes the particle data to sim_data.p files " +
                        "every SAVE_INTERVAL timesteps. RECOMMENDED",
                        type=int)

    parser.add_argument("-r", "--resume",
                        help="Resumes a stopped run by reading the files " +
                        "previously written by the save flag",
                        action="store_true")

    args = parser.parse_args()

    return args
