import pickle

import matplotlib.pyplot as plt
import numpy as np
from deap.tools import Logbook, Statistics


def get_stats() -> Statistics:
    """Initialize Statistics object"""
    stats = Statistics(key=lambda ind: ind.fitness)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("median", np.median)
    stats.register("min", min)
    stats.register("max", max)
    return stats


def plot_logbook(logbook: Logbook, title: str = None, filename: str = None):
    """Plot average, best and worst fitness over generations
    :param logbook: Logbook containing the data (avg, max, min and std in each generation)
    :param title: Title of the plot
    :param filename: Filename to save the plot to (if None, the plot is shown)"""
    gen = logbook.select("gen")
    fit_avg = logbook.select("avg")
    fit_max = logbook.select("max")
    fit_min = logbook.select("min")
    fit_median = logbook.select("median")

    plt.figure(figsize=(12, 8))
    plt.title(title)
    plt.plot(gen, fit_avg, label="Average Fitness")
    plt.plot(gen, fit_max, label="Max Fitness")
    plt.plot(gen, fit_min, label="Min Fitness")
    plt.plot(gen, fit_median, label="Median Fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend(loc="lower right")

    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()


def plot_coevolution(
    ship_logbook: Logbook,
    yard_logbook: Logbook,
    ship_generations_per_tick: int,
    yard_generations_per_tick: int,
    title: str = None,
    filename: str = None,
):
    """Plot average fitness over generations for both subevolutions
    :param ship_logbook: Logbook containing the data for the ship subevolution
    :param yard_logbook: Logbook containing the data for the yard subevolution
    :param ship_generations_per_tick: Number of generations per coevolution tick for the ship subevolution
    :param yard_generations_per_tick: Number of generations per coevolution tick for the yard subevolution
    :param title: Title of the plot
    :param filename: Filename to save the plot to (if None, the plot is shown)"""
    ship_gen = [g / ship_generations_per_tick for g in ship_logbook.select("gen")]
    yard_gen = [g / yard_generations_per_tick for g in yard_logbook.select("gen")]
    ship_fit_avg = ship_logbook.select("avg")
    yard_fit_avg = yard_logbook.select("avg")

    plt.figure(figsize=(12, 8))
    plt.title(title)
    plt.plot(ship_gen, ship_fit_avg, label="Ship Average Fitness")
    plt.plot(yard_gen, yard_fit_avg, label="Yard Average Fitness")
    plt.xlabel("Coevolution generation")
    plt.ylabel("Fitness")
    plt.legend(loc="lower right")

    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()


if __name__ == "__main__":
    with open("../ship-logbook5.pkl", "rb") as f:
        ship_logbook = pickle.load(f)
    with open("../yard-logbook5.pkl", "rb") as f:
        yard_logbook = pickle.load(f)

    plot_logbook(ship_logbook, title="ship", filename=None)
    plot_logbook(yard_logbook, title="yard", filename=None)
    plot_coevolution(ship_logbook, yard_logbook, title=None, filename=None)
