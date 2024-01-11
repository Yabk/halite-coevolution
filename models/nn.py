from __future__ import annotations

import random
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike

from .individual import Individual


class NNIndividual(Individual):
    """Neural network individual"""

    def __init__(
        self,
        layer_sizes: tuple[int, ...],
        activations: tuple[Callable[[ArrayLike], ArrayLike], ...],
    ):
        """Initialize the neural network individual

        :param layer_sizes: list of layer sizes (including number of inputs as the first element)
        :param activations: list of activation functions for each layer
        """
        super().__init__()
        self.layers = [
            NNLayer(layer_sizes[i], layer_sizes[i + 1], activations[i])
            for i in range(len(layer_sizes) - 1)
        ]

    def evaluate(self, inputs: list[float]) -> list[float]:
        """Evaluate given inputs and return the outputs"""
        outputs = inputs
        for layer in self.layers:
            outputs = layer.propagate(outputs)
        return list(outputs)

    def serialize(self) -> list[float]:
        """Serialize the individual into a list of floats"""
        serialized = []
        for layer in self.layers:
            serialized.extend(layer.weights.flatten())
            serialized.extend(layer.biases)
        return serialized

    def deserialize(self, serialized: list[float]) -> NNIndividual:
        """Deserialize the individual from a list of floats (weights and biases)"""
        new = self.copy()
        for layer in new.layers:
            layer.weights = np.array(serialized[: layer.weights.size]).reshape(
                layer.weights.shape
            )
            serialized = serialized[layer.weights.size :]
            layer.biases = np.array(serialized[: layer.biases.size])
            serialized = serialized[layer.biases.size :]
        return new

    def mutate(self, probability: float = 0.1, sigma: float = 0.1) -> None:
        """Mutate the individual"""
        for layer in self.layers:
            layer.mutate(probability, sigma)

    def point_crossover(
        self, other: NNIndividual, k: int = 2
    ) -> tuple[NNIndividual, NNIndividual]:
        """K-point crossover"""
        if k < 1:
            raise ValueError(f"k must be greater than 0, got {k}")
        new = self.copy()
        i1 = self.serialize()
        i2 = other.serialize()

        points = sorted(random.sample(range(1, len(i1)), k))

        c1 = []
        c2 = []
        switch = False
        prev_point = 0
        for point in points:
            if switch:
                c1 += i2[prev_point:point]
                c2 += i1[prev_point:point]
            else:
                c1 += i1[prev_point:point]
                c2 += i2[prev_point:point]
            prev_point = point
            switch = not switch
        if switch:
            c1 += i2[prev_point:]
            c2 += i1[prev_point:]
        else:
            c1 += i1[prev_point:]
            c2 += i2[prev_point:]

        return self.deserialize(c1), self.deserialize(c2)


class NNLayer:
    """Neural network layer"""

    def __init__(
        self,
        input_num: int,
        output_num: int,
        activation: Callable[[ArrayLike], ArrayLike],
    ):
        """Initialize the neural network layer

        :param input_num: number of inputs
        :param output_num: number of outputs
        :param activation: activation function
        """
        self.activation = activation
        self.weights = np.random.randn(output_num, input_num) * 0.1
        self.biases = np.random.randn(output_num) * 0.1

    def propagate(self, inputs: ArrayLike) -> ArrayLike:
        """Propagate the inputs through the layer"""
        return self.activation(self.weights @ inputs + self.biases)

    def mutate(self, probability: float = 0.1, sigma: float = 0.1) -> None:
        """Mutate the layer"""
        self.weights += (
            np.random.randn(*self.weights.shape)
            * sigma
            * np.random.binomial(1, probability, self.weights.shape)
        )
        self.biases += (
            np.random.randn(*self.biases.shape)
            * sigma
            * np.random.binomial(1, probability, self.biases.shape)
        )


class Activations:
    """Activation functions"""

    @staticmethod
    def relu(x: ArrayLike) -> ArrayLike:
        return np.maximum(0, x)

    @staticmethod
    def identity(x: ArrayLike) -> ArrayLike:
        return x


if __name__ == "__main__":
    nn1 = NNIndividual((2, 3, 3), (Activations.relu, Activations.identity))
    nn2 = NNIndividual((2, 3, 3), (Activations.relu, Activations.identity))

    c1, c2 = nn1.point_crossover(nn2)
