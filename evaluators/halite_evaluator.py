import random
from typing import Callable

from kaggle_environments import make
from kaggle_environments.envs.halite.helpers import (
    Board,
    Cell,
    Ship,
    ShipAction,
    Shipyard,
    ShipyardAction,
    board_agent,
)

from models.individual import Individual


class HaliteEvaluator:
    """Evaluator for Halite"""

    SHIP_ACTIONS = list(ShipAction)
    SHIP_ACTIONS.append(None)
    YARD_ACTIONS = list(ShipyardAction)
    YARD_ACTIONS.append(None)

    def __init__(
        self,
        enemy_agent: str | Callable = "random",
        board_size: int = 5,
        repeat_evaluation: int = 5,
        debug: bool = False,
    ):
        self.seeds = [random.randint(0, int(1e7)) for _ in range(repeat_evaluation)]
        self.enemy_agent = enemy_agent
        self.environment = make(
            "halite",
            configuration={"size": board_size, "startingHalite": 1000},
            debug=debug,
        )
        if (
            enemy_agent is not None
        ):  # if enemy agent is None, we are evaluating a single agent
            self.environment.reset(2)
        else:
            self.environment.reset(1)

    def reset_seeds(self):
        self.seeds = [random.randint(0, int(1e7)) for _ in range(len(self.seeds))]

    @staticmethod
    def _extract_cell_feature(
        board: Board, cell: Cell, for_ship: bool = True
    ) -> list[float]:
        """Extract features from the given cell"""
        features = []
        if cell.ship is not None:
            if cell.ship.player_id == board.current_player.id:
                features.append(1)
                features.append(0)
            else:
                features.append(0)
                features.append(1)
        else:
            features.append(0)
            features.append(0)
        if for_ship:
            if cell.shipyard is not None:
                if cell.shipyard.player_id == board.current_player.id:
                    features.append(1)
                    features.append(0)
                else:
                    features.append(0)
                    features.append(1)
            else:
                features.append(0)
                features.append(0)
            features.append(cell.halite / board.configuration.max_cell_halite)
        return features

    @staticmethod
    def extract_features(board: Board, entity: Ship | Shipyard) -> list[float]:
        """Extract features from the board for the given entity"""
        cell = entity.cell
        if isinstance(entity, Ship):
            features = [
                entity.halite / board.configuration.convert_cost,
                cell.halite / board.configuration.max_cell_halite,
                1 if cell.shipyard is not None else 0,
            ]
            for_ship = True
        elif isinstance(entity, Shipyard):
            features = [
                1
                if cell.ship is not None
                else 0,  # 1 if friendly ship on current cell, 0 otherwise
                board.current_player.halite
                / board.configuration.convert_cost,  # current player halite divided by
                # shipyard cost
            ]
            for_ship = False
        else:
            raise TypeError(f"Invalid entity type: {type(entity)}")
        features.extend(
            HaliteEvaluator._extract_cell_feature(board, cell.north, for_ship)
        )
        features.extend(
            HaliteEvaluator._extract_cell_feature(board, cell.east, for_ship)
        )
        features.extend(
            HaliteEvaluator._extract_cell_feature(board, cell.south, for_ship)
        )
        features.extend(
            HaliteEvaluator._extract_cell_feature(board, cell.west, for_ship)
        )

        return features

    @staticmethod
    def agent_factory(
        ship_ind: Individual, shipyard_ind: Individual, punishment=None
    ) -> Callable:
        """Create an agent from ship and shipyard individual individuals"""
        if punishment is None:
            punishment = [0]

        @board_agent
        def agent(board: Board) -> None:
            for ship in board.current_player.ships:
                features = HaliteEvaluator.extract_features(board, ship)
                output = ship_ind.evaluate(features)
                ship.next_action = HaliteEvaluator.SHIP_ACTIONS[
                    output.index(max(output))
                ]
                if (
                    ship.next_action == ShipAction.CONVERT
                    and ship.cell.shipyard is not None
                ):
                    output[output.index(max(output))] = 0
                    ship.next_action = HaliteEvaluator.SHIP_ACTIONS[
                        output.index(max(output))
                    ]

            for shipyard in board.current_player.shipyards:
                features = HaliteEvaluator.extract_features(board, shipyard)
                shipyard.next_action = HaliteEvaluator.YARD_ACTIONS[
                    shipyard_ind.predict(features)
                ]

        return agent

    def evaluate(self, ship: Individual, shipyard: Individual) -> tuple[float, ...]:
        """Evaluate a single individual"""
        punishment = [0]
        agent = HaliteEvaluator.agent_factory(ship, shipyard, punishment)

        fitness = 0
        for seed in self.seeds:
            self.environment.configuration.randomSeed = seed
            if self.enemy_agent is not None:
                game_run = self.environment.run([agent, self.enemy_agent])
                result = game_run[-1][0]
            else:
                result = self.environment.run([agent])[-1][0]
            if result["status"] == "error":
                raise RuntimeError(f"Error in game run")
            try:
                fitness += result["reward"]
            except TypeError:
                fitness = result["reward"]
        try:
            fitness -= punishment[0]
        except TypeError:
            fitness = -punishment[0]
        fitness /= len(self.seeds)

        return (fitness,)

    @staticmethod
    def ship_in_types() -> list[type]:
        """Return input types for ship individual"""
        # fmt: off
        return [
            float,  # ship halite / convert cost
            float,  # cell halite / max cell halite
            bool,  # 1 if cell has friendly shipyard, 0 otherwise
            bool, bool, bool, bool, float,  # north cell features
            bool, bool, bool, bool, float,  # east cell features
            bool, bool, bool, bool, float,  # south cell features
            bool, bool, bool, bool, float,  # west cell features
        ]
        # fmt: on
        # cell features
        #  1 if cell has friendly ship, 0 otherwise
        #  1 if cell has enemy ship, 0 otherwise
        #  1 if cell has friendly shipyard, 0 otherwise
        #  1 if cell has enemy shipyard, 0 otherwise
        #  cell halite / max cell halite

    @staticmethod
    def yard_in_types() -> list[type]:
        """Return input types for shipyard individual"""
        # fmt: off
        return [
            bool,  # 1 if cell has friendly ship, 0 otherwise
            float,  # current player halite / convert cost
            bool, bool,  # north cell features
            bool, bool,  # east cell features
            bool, bool,  # south cell features
            bool, bool,  # west cell features
        ]
        # fmt: on
        # cell features
        #  1 if cell has friendly ship, 0 otherwise
        #  1 if cell has enemy ship, 0 otherwise
