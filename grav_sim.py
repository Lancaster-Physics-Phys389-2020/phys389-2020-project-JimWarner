"""
Module containing the main code to setup and run the simulation
"""

from arguments import parse_arguments
import gal_sim
import uni_sim

if __name__ == "__main__":
    args = parse_arguments()

    if args.simulation == "g":
        gal_sim.run(args)

    if args.simulation == "u":
        uni_sim.run(args)
