"""
Module containing the main code to setup and run the simulation
"""

from arguments import parse_arguments
import gal_sim
import uni_sim

if __name__ == "__main__":
    ARGS = parse_arguments()

    if ARGS.simulation == "g":
        gal_sim.run(ARGS)

    if ARGS.simulation == "u":
        uni_sim.run(ARGS)
