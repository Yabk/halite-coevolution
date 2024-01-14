from __future__ import annotations

import operator
import random

import matplotlib.pyplot as plt
import networkx as nx
from deap.gp import (
    PrimitiveSetTyped,
    PrimitiveTree,
    compile,
    cxOnePoint,
    genGrow,
    genHalfAndHalf,
    graph,
    mutNodeReplacement,
    mutUniform,
)

from .individual import Individual


class GPIndividual(Individual):
    """A genetic programming individual"""

    def __init__(
        self,
        pset: PrimitiveSetTyped,
        output_num: int,
        min_depth: int = 1,
        max_depth: int = 3,
    ):
        super().__init__()
        self.pset = pset
        self.trees = []
        for i in range(output_num):
            self.trees.append(
                PrimitiveTree(genHalfAndHalf(pset=pset, min_=min_depth, max_=max_depth))
            )

    def evaluate(self, inputs):
        """Evaluate given inputs and return the outputs for each tree"""
        return [compile(tree, self.pset)(*inputs) for tree in self.trees]

    def mutate(self, max_depth: int = 3, probability: float = None) -> None:
        """Mutate the individual using mutUniform() and genGrow(). If the height after mutation
        is greater than max_depth, the mutation is discarded and simple node replacement is used.
        :param max_depth: maximum depth of the newly generated subtree
        :param probability: probability of mutating each tree. If None, the probability is 1 / number of trees.
        """
        if probability is None:
            probability = 1 / len(self.trees)
        for i in range(len(self.trees)):
            if random.random() < probability:
                expr = lambda pset, type_: genGrow(
                    pset, min_=0, max_=max_depth, type_=type_
                )
                tree_copy = PrimitiveTree(self.trees[i].copy())
                mutUniform(tree_copy, expr, self.pset)
                # check if height of the tree is less than max_depth
                if tree_copy.height <= max_depth:
                    self.trees[i] = tree_copy
                # if it is, do a simple mutNodeReplacement instead
                else:
                    mutNodeReplacement(self.trees[i], self.pset)

    def __getitem__(self, key: int) -> PrimitiveTree:
        return self.trees[key]

    def visualize(self, shorten_consts: bool = True) -> None:
        """Visualize the trees"""
        for i, tree in enumerate(self.trees):
            nodes, edges, labels = graph(tree)
            # reverse the edges so we can have correct direction
            edges = [(v, u) for u, v in edges]

            if shorten_consts:
                for k, v in labels.items():
                    if isinstance(v, float):
                        labels[k] = f"{v:.2f}"

            g = nx.DiGraph()
            g.add_nodes_from(nodes)
            g.add_edges_from(edges)
            #            pos = nx.spring_layout(g)
            pos = nx.kamada_kawai_layout(g)

            plt.title(f"Tree {i}")
            nx.draw(
                g,
                pos,
                with_labels=True,
                labels=labels,
                arrows=True,
                arrowsize=20,
                node_size=1200,
            )
            plt.show()

    @staticmethod
    def macro_cx(ind1: GPIndividual, ind2: GPIndividual) -> None:
        """Perform macro crossover - exchange whole trees between individuals.
        Each tree has a 50% chance of being exchanged.
        """
        for i in range(len(ind1.trees)):
            if random.random() < 0.5:
                ind1.trees[i], ind2.trees[i] = ind2.trees[i], ind1.trees[i]

    @staticmethod
    def micro_cx(ind1: GPIndividual, ind2: GPIndividual) -> None:
        """Perform micro crossover - perform crossover on single tree, leaving the other intact."""
        tree_index = random.randrange(len(ind1.trees))
        cxOnePoint(ind1.trees[tree_index], ind2.trees[tree_index])

    @staticmethod
    def if_then_else(condition: bool, output1: float, output2: float) -> float:
        return output1 if condition else output2

    @staticmethod
    def p_div(a: float, b: float) -> float:
        """Protected division"""
        if abs(b) < 1e-6:
            return 0
        return a / b

    @staticmethod
    def _ephermal() -> float:
        return random.uniform(-1, 1)

    @staticmethod
    def generate_pset(in_types: list[type]) -> PrimitiveSetTyped:
        pset = PrimitiveSetTyped("main", in_types, float)
        pset.addPrimitive(operator.xor, [bool, bool], bool)
        pset.addPrimitive(operator.and_, [bool, bool], bool)
        pset.addPrimitive(operator.or_, [bool, bool], bool)
        pset.addPrimitive(operator.mul, [float, float], float)
        pset.addPrimitive(operator.add, [float, float], float)
        pset.addPrimitive(operator.sub, [float, float], float)
        pset.addPrimitive(max, [float, float], float)
        pset.addPrimitive(operator.ge, [float, float], bool)
        pset.addPrimitive(operator.neg, [float], float)
        pset.addPrimitive(GPIndividual.p_div, [float, float], float)
        pset.addPrimitive(GPIndividual.if_then_else, [bool, float, float], float)
        pset.addTerminal(0.0, float)
        pset.addTerminal(1, bool)
        pset.addTerminal(0, bool)
        pset.addEphemeralConstant("eph0-1", GPIndividual._ephermal, float)

        return pset


if __name__ == "__main__":
    pset = GPIndividual.generate_pset([float, float, float])
    ind = GPIndividual(pset, 2)

    ind.visualize()

    ind.mutate(probability=1)

    ind.visualize()
