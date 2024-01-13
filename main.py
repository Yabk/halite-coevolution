"""The following script takes a config file as the first argument, runs the coevolution
and saves the results in the same directory as the config file."""
import os
import pickle
import sys
from pathlib import Path

from tools.config import parse_config
from tools.stats import plot_coevolution, plot_logbook

if __name__ == "__main__":
    # take config file as the first argument
    config_file = sys.argv[1]

    # get the directory of the config file
    dir_path = os.path.dirname(config_file)

    # parse config
    coevolution = parse_config(config_file)

    # run coevolution
    print(f"Running the coevolution for {coevolution.generations} generations")
    coevolution.run()

    print("Done! Saving the results...")

    # save the both populations
    with open(os.path.join(dir_path, "ship-population.pkl"), "wb") as f:
        pickle.dump(coevolution.ship_subevolution.population, f)
    with open(os.path.join(dir_path, "yard-population.pkl"), "wb") as f:
        pickle.dump(coevolution.yard_subevolution.population, f)
    # as well as both HoFs
    with open(os.path.join(dir_path, "ship-hof.pkl"), "wb") as f:
        pickle.dump(coevolution.ship_subevolution.hof, f)
    with open(os.path.join(dir_path, "yard-hof.pkl"), "wb") as f:
        pickle.dump(coevolution.yard_subevolution.hof, f)

    # save the plots
    ship_logbook = coevolution.ship_subevolution.logbook
    yard_logbook = coevolution.yard_subevolution.logbook
    plot_coevolution(
        ship_logbook,
        yard_logbook,
        title="First run",
        filename=os.path.join(dir_path, "coevolution.png"),
    )
    plot_logbook(
        ship_logbook,
        title="Ship subevolution",
        filename=os.path.join(dir_path, "ship.png"),
    )
    plot_logbook(
        yard_logbook,
        title="Yard subevolution",
        filename=os.path.join(dir_path, "yard.png"),
    )

    # create 2 new empty files with filenames of highest achieved fitness for both yard and ship
    # this is used to easily find the best fitness of the run when you have a lot of runs in a lot of directories
    Path(
        os.path.join(
            dir_path, "ship-" + str(coevolution.ship_subevolution.hof[0].fitness[0])
        )
    ).touch()
    Path(
        os.path.join(
            dir_path, "yard-" + str(coevolution.yard_subevolution.hof[0].fitness[0])
        )
    ).touch()

    print("Done!")
