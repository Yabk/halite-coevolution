import operator
import random
from typing import Callable

from .individual import Individual

PROTECTED_DIVISION_EPSILON = 1e-5


def protected_div(a: float, b: float) -> float:
    """Protected division, as in miller2009"""
    if abs(b) > PROTECTED_DIVISION_EPSILON:
        return a / b
    return a


def or_operator(a: int | float, b: int | float) -> int | float:
    if type(a) is int and type(b) is int:
        return a | b
    else:
        return a * b


def xor_operator(a: int | float, b: int | float) -> int | float:
    if type(a) is int and type(b) is int:
        return a ^ b
    else:
        return a * b


def notneg_operator(a: int | float, b: int | float) -> int | float:
    if type(a) is int:
        return 0 if a else 1
    else:
        return -a


class CGPIndividual(Individual):
    """Cartesian genetic programming individual"""

    FUNCTIONS = (
        operator.add,
        operator.sub,
        operator.mul,
        protected_div,
        or_operator,
        xor_operator,
        notneg_operator,
    )

    def __init__(
        self,
        input_num: int,
        output_num: int,
        grid_width: int,
        grid_height: int,
        functions: list[Callable] = FUNCTIONS,
        constants: list[float] = (0, 1, 10, 1000),
        levels_back: int = 1,
    ):
        super().__init__()
        self.input_num, self.output_num = input_num, output_num
        self.width, self.height = grid_width, grid_height
        self.genotype = [0] * (grid_width * grid_height * 3 + output_num)
        self.functions = functions
        self.constants = constants
        self.levels_back = levels_back

        # indices contained in each column of the grid
        self.indices = []
        self.indices.append(list(range(input_num + len(constants))))
        for column in range(1, grid_width + 1):
            self.indices.append(
                list(
                    range(
                        input_num + len(constants) + (column - 1) * grid_height,
                        input_num + len(constants) + column * grid_height,
                    )
                )
            )

        # valid input array (each element contains valid inputs for the corresponding column of the grid)
        self.valid_inputs = []
        for column in range(grid_width):
            self.valid_inputs.append(self.indices[0][:])
            for c in range(max(1, column + 1 - levels_back), column + 1):
                self.valid_inputs[-1].extend(self.indices[c])

        self._generate_genotype()

    def evaluate(self, inputs: list[float]) -> list[float]:
        """Evaluate given inputs and return the outputs"""
        to_evaluate = self._list_active_nodes()

        terminals = inputs + list(self.constants)
        outputs = {}
        for i, value in enumerate(terminals):
            outputs[i] = value
        for module_index in to_evaluate:
            x, y, function_index = self._get_module(module_index)
            if x < self.input_num + len(self.constants):
                x_value = terminals[x]
            else:
                x_value = outputs[x]
            if y < self.input_num + len(self.constants):
                y_value = terminals[y]
            else:
                y_value = outputs[y]
            outputs[module_index] = self.functions[function_index](x_value, y_value)

        return [outputs[self.genotype[i]] for i in range(-self.output_num, 0, 1)]

    def _get_module(self, module_index: int) -> tuple[int, int, int]:
        """Get the module at the given index"""

        # raise exception if module_index is index of an input or a constant
        if module_index < self.input_num + len(self.constants):
            raise ValueError(f"Invalid module index {module_index} (input or constant)")
        # raise exception if module_index is too large
        if module_index >= self.width * self.height + self.input_num + len(
            self.constants
        ):
            raise ValueError(f"Invalid module index {module_index} (too large)")

        # return the module (x, y, function_index)
        genotype_index = module_index - self.input_num - len(self.constants)
        return tuple(self.genotype[genotype_index * 3 : (genotype_index + 1) * 3])

    def _generate_genotype(self):
        """Generate the genotype"""
        for c in range(self.width):
            valid_inputs = self.valid_inputs[c]
            for r in range(self.height):
                pos = (c * self.height + r) * 3
                self.genotype[pos] = random.choice(valid_inputs)
                self.genotype[pos + 1] = random.choice(valid_inputs)
                self.genotype[pos + 2] = random.randrange(len(self.functions))

        for i in range(self.output_num):
            self.genotype[-i - 1] = random.randrange(
                self.input_num + len(self.constants) + self.width * self.height
            )

    def _list_active_nodes(self) -> list[int, ...]:
        """Returns a list of indices of active nodes (sorted ascending)"""
        active = set(
            [
                self.genotype[i]
                for i in range(-1, -self.output_num - 1, -1)
                if self.genotype[i] >= self.input_num + len(self.constants)
            ]
        )
        to_process = active.copy()

        while to_process:
            module_index = to_process.pop()
            x, y, function_index = self._get_module(module_index)
            if x >= self.input_num + len(self.constants) and x not in active:
                to_process.add(x)
                active.add(x)
            if y >= self.input_num + len(self.constants) and y not in active:
                to_process.add(y)
                active.add(y)

        active = list(active)
        active.sort()

        return active

    def mutate(self) -> None:
        """Goldman mutation - mutate until you hit an active gene"""
        # list all the active genes
        active = self._list_active_nodes()

        done = False
        while not done:
            pos = random.randrange(len(self.genotype))
            if pos < self.width * self.height * 3:
                module_index = pos // 3 + self.input_num + len(self.constants)
                done = module_index in active
                if pos % 3 == 2:
                    self.genotype[pos] = random.randrange(len(self.functions))
                else:
                    column = (
                        module_index - self.input_num - len(self.constants)
                    ) // self.height
                    self.genotype[pos] = random.choice(self.valid_inputs[column])
            else:
                done = True
                self.genotype[pos] = random.randrange(
                    self.input_num + len(self.constants) + self.width * self.height
                )


if __name__ == "__main__":
    cgp = CGPIndividual(2, 1, 4, 2, constants=[], levels_back=2)
    print(cgp.evaluate([2, 5]))
