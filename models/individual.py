"""Module containing abstract class for learning models (individuals) and their factories"""
from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from copy import deepcopy


class Individual(ABC):
    """Abstract individual class"""

    def __init__(self):
        """Initialize the individual"""
        self.fitness = None

    @abstractmethod
    def evaluate(self, inputs: list) -> list[float]:
        """Evaluate given inputs and return the outputs"""

    def predict(self, inputs: list) -> int:
        """Predict the action given the inputs"""
        outputs = self.evaluate(inputs)
        return outputs.index(max(outputs))

    def copy(self) -> Individual:
        """Return a copy of self"""
        return deepcopy(self)

    def to_file(self, path: str) -> None:
        """Save the individual to a file"""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def from_file(path: str) -> Individual:
        """Load the individual from a file"""
        with open(path, "rb") as f:
            return pickle.load(f)
