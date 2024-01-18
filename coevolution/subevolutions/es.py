import pickle
from typing import Callable

from deap import base, tools

from coevolution.coevolution import Subevolution
from models.individual import Individual


class ESevolution(Subevolution):
    """Evolution strategy implemented as a subevolution of a coevolutionary algorithm"""

    def __init__(
        self,
        toolbox: base.Toolbox,
        stats: tools.Statistics,
        evaluate: Callable[[Individual], tuple[float, ...]],
        hof_maxsize: int = 10,
        generations_per_tick: int = 10,
        mu: int = 1,
        lmbd: int = 4,
        elitism: bool = True,
        parents_file: str | None = None,
    ):
        """Initialize the sub-evolution.
        :param toolbox: toolbox containing functions for the ES
        :param statistics: object used for logging statistics for each generation
        :param evaluate: evaluation function
        :param hof_maxsize: maximum size of the hall of fame
        :param generations_per_tick: number of generations to run in each tick of the main coevolution
        :param mu: number of parents to select from the population
        :param lmbd: number of children to generate
        :param elitism: if True, the best individual from the previous generation is automatically promoted to the next generation
        :param parents_file: file containing the initial parents

        toolbox should contain the following functions:
        - Individual() -> Individual
        - mutate(individual: Individual) -> None
        """
        super().__init__(toolbox, stats, hof_maxsize, generations_per_tick)
        self.lmbd = lmbd
        self.mu = mu
        self.elitism = elitism
        self.generations_per_tick = generations_per_tick

        # initialize the population
        if parents_file is not None:
            with open(parents_file, "rb") as f:
                self.parents = pickle.load(f)
            for individual in self.parents:
                individual.fitness = evaluate(individual)
            self.logbook.record(gen=self.generation, **self.stats.compile(self.parents))
            self.hof.update(self.parents)
        else:
            population = [self.toolbox.Individual() for _ in range(self.lmbd)]
            for individual in population:
                individual.fitness = evaluate(individual)
            self.logbook.record(gen=self.generation, **self.stats.compile(population))
            self.hof.update(population)
            population.sort(key=lambda i: i.fitness, reverse=True)
            self.parents = population[: self.mu]

    @property
    def representative(self) -> Individual:
        """Return the current best individual in the generation"""
        return self.parents[0]

    @property
    def population(self) -> list[Individual]:
        """Return the current population"""
        return self.parents

    def tick(self, evaluate: Callable[[Individual], tuple[float, ...]]) -> None:
        """Run the algorithm for generations_per_tick generations"""
        for _ in range(self.generations_per_tick):
            self.generation += 1
            new_population = self.parents[:] if self.elitism else []
            while len(new_population) < self.lmbd:
                for parent in self.parents:
                    child = self.toolbox.clone(parent)
                    self.toolbox.mutate(child)
                    child.fitness = evaluate(child)
                    new_population.append(child)
                    if len(new_population) >= self.lmbd:
                        break
            new_population.sort(key=lambda i: i.fitness, reverse=True)
            self.parents = new_population[: self.mu]
            self.hof.update(self.parents)
            record = self.stats.compile(new_population)
            self.logbook.record(gen=self.generation, **record)
