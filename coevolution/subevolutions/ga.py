from typing import Callable

from deap import base, tools

from coevolution.coevolution import Subevolution
from models.individual import Individual


class GAevolution(Subevolution):
    """GA implemented as a subevolution of a coevolutionary algorithm"""

    def __init__(
        self,
        toolbox: base.Toolbox,
        stats: tools.Statistics,
        evaluate: Callable[[Individual], tuple[float, ...]],
        hof_maxsize: int = 10,
        generations_per_tick: int = 1,
        population_size: int = 10,
    ):
        """Initialize the sub-evolution

        toolbox should contain the following functions:
        - Individual() -> Individual
        - select(population) -> Individual, Individual
        - mate(Individual, Individual) -> None
        - mutate(Individual) -> None
        """
        super().__init__(toolbox, stats, hof_maxsize, generations_per_tick)
        self.population_size = population_size

        # initialize the population
        self.pop = [self.toolbox.Individual() for _ in range(self.population_size)]
        for individual in self.pop:
            individual.fitness = evaluate(individual)
        self.logbook.record(gen=self.generation, **self.stats.compile(self.population))

        self.hof.update(self.population)
        self.current_best = self.hof[0]

    @property
    def representative(self) -> Individual:
        """Return the current best individual in the generation"""
        return self.current_best

    @property
    def population(self) -> list[Individual]:
        """Return the current population"""
        return self.pop

    def tick(self, evaluate: Callable[[Individual], tuple[float, ...]]) -> None:
        """Run the algorithm for generations_per_tick generations"""
        for _ in range(self.generations_per_tick):
            new_pop = []
            self.generation += 1
            while len(new_pop) < self.population_size:
                parent1, parent2 = self.toolbox.select(self.pop)
                child1, child2 = self.toolbox.clone(parent1), self.toolbox.clone(
                    parent2
                )
                self.toolbox.mate(child1, child2)
                self.toolbox.mutate(child1)
                self.toolbox.mutate(child2)
                child1.fitness = evaluate(child1)
                child2.fitness = evaluate(child2)
                new_pop.append(child1)
                new_pop.append(child2)

            self.hof.update(new_pop)
            self.pop = new_pop
            self._set_current_best()
            record = self.stats.compile(self.population)
            self.logbook.record(gen=self.generation, **record)

    def _set_current_best(self) -> None:
        """Set the current best individual"""
        self.current_best = self.pop[0]
        for individual in self.pop:
            if individual.fitness > self.current_best.fitness:
                self.current_best = individual
