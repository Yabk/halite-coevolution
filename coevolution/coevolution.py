from abc import ABC, abstractmethod
from typing import Callable

from deap import base, tools

from evaluators.halite_evaluator import HaliteEvaluator
from models.individual import Individual


class Subevolution(ABC):
    """Class containing one population of individuals in a coevolutionary algorithm"""

    def __init__(
        self,
        toolbox: base.Toolbox,
        stats: tools.Statistics,
        hof_maxsize: int = 10,
        generations_per_tick: int = 1,
    ):
        """Initialize the sub-evolution
        :param toolbox: toolbox containing functions for the Subevolution
        :param statistics: object used for logging statistics for each generation
        :param hof_maxsize: maximum size of the hall of fame
        :param generations_per_tick: number of generations to run in each tick of the main coevolution
        """
        self.toolbox = toolbox
        self.stats = stats
        self.logbook = tools.Logbook()
        self.hof = tools.HallOfFame(hof_maxsize)
        self.generations_per_tick = generations_per_tick
        self.generation = 0

    @property
    @abstractmethod
    def representative(self) -> Individual:
        """Return the representative -> current best individual, as used in potter2000"""

    @property
    @abstractmethod
    def population(self) -> list[Individual]:
        """Return the current population"""

    @abstractmethod
    def tick(self, evaluate: Callable[[Individual], tuple[float, ...]]) -> None:
        """Evaluate the individuals and generate a new population"""


class Coevolution:
    """Coevolutionary algorithm"""

    def __init__(
        self,
        evaluator: HaliteEvaluator,
        ship_subevolution: Subevolution,
        yard_subevolution: Subevolution,
        generations: int = 1000,
    ):
        """Initialize the coevolutionary algorithm"""
        self.evaluator = evaluator
        self.ship_subevolution = ship_subevolution
        self.yard_subevolution = yard_subevolution
        self.generations = generations

    def run(self) -> tools.Logbook:
        """Run the coevolution"""
        for generation in range(self.generations):

            def ship_evaluator(individual: Individual) -> tuple[float, ...]:
                return self.evaluator.evaluate(
                    individual, self.yard_subevolution.representative
                )

            def yard_evaluator(individual: Individual) -> tuple[float, ...]:
                return self.evaluator.evaluate(
                    self.ship_subevolution.representative, individual
                )

            self.ship_subevolution.tick(ship_evaluator)
            self.yard_subevolution.tick(yard_evaluator)
            self.evaluator.reset_seeds()
            print(f"Generation {generation} complete")
            print(f"Best ship: {self.ship_subevolution.representative.fitness}")
            print(f"Best yard: {self.yard_subevolution.representative.fitness}")
