import tomllib
from typing import Callable

from deap.base import Toolbox
from deap.tools import selTournament

from coevolution.coevolution import Coevolution, Subevolution
from coevolution.subevolutions.es import ESevolution
from coevolution.subevolutions.ga import GAevolution
from evaluators.halite_evaluator import HaliteEvaluator
from evaluators.halite_strategies.mnmnao import agent as mnmnao
from evaluators.halite_strategies.naive import agent as naive
from models.cgp import CGPIndividual
from models.gp import GPIndividual
from models.individual import Individual
from models.nn import Activations, NNIndividual
from tools.stats import get_stats


def parse_nn_config(nn_config: dict, for_ship: bool) -> dict:
    """Parse NNIndividual configuration.
    :param nn_config: dictionary containing configuration for NNIndividual
    :param for_ship: True if config is for ship, False if it is for yard
    :return: dictionary containing NNIndividual generator and mutation/crossover functions (if applicable)
    """
    parsed = {}

    activations = [Activations.relu for i in range(len(nn_config["layer_sizes"]))]
    activations.append(Activations.identity)
    if for_ship:
        layer_sizes = (
            [len(HaliteEvaluator.ship_in_types())]
            + nn_config["layer_sizes"]
            + [len(HaliteEvaluator.SHIP_ACTIONS)]
        )
    else:
        layer_sizes = (
            [len(HaliteEvaluator.yard_in_types())]
            + nn_config["layer_sizes"]
            + [len(HaliteEvaluator.YARD_ACTIONS)]
        )
    parsed["Individual"] = lambda: NNIndividual(layer_sizes, activations)

    parsed["mutate"] = lambda individual: individual.mutate(**nn_config["mutation"])

    if "crossover" in nn_config:
        parsed["mate"] = lambda i1, i2: i1.point_crossover(i2, **nn_config["crossover"])

    return parsed


def parse_cgp_config(cgp_config: dict, for_ship: bool) -> dict:
    """Parse CGPIndividual configuration.
    :param cgp_config: dictionary containing configuration for CGPIndividual
    :param for_ship: True if config is for ship, False if it is for yard
    :return: dictionary containing CGPIndividual generator and mutation/crossover functions (if applicable)
    """
    parsed = {}

    if for_ship:
        input_num = len(HaliteEvaluator.ship_in_types())
        output_num = len(HaliteEvaluator.SHIP_ACTIONS)
    else:
        input_num = len(HaliteEvaluator.yard_in_types())
        output_num = len(HaliteEvaluator.YARD_ACTIONS)
    parsed["Individual"] = lambda: CGPIndividual(
        input_num,
        output_num,
        cgp_config["grid_width"],
        cgp_config["grid_height"],
        levels_back=cgp_config["levels_back"],
    )

    parsed["mutate"] = lambda individual: individual.mutate()

    return parsed


def parse_gp_config(gp_config: dict, for_ship: bool) -> dict:
    """Parse GPIndividual configuration.
    :param gp_config: dictionary containing configuration for GPIndividual
    :param for_ship: True if config is for ship, False if it is for yard
    :return: dictionary containing GPIndividual generator and mutation/crossover functions (if applicable)
    """
    parsed = {}

    if for_ship:
        output_num = len(HaliteEvaluator.SHIP_ACTIONS)
        pset = GPIndividual.generate_pset(HaliteEvaluator.ship_in_types())
    else:
        output_num = len(HaliteEvaluator.YARD_ACTIONS)
        pset = GPIndividual.generate_pset(HaliteEvaluator.yard_in_types())
    parsed["Individual"] = lambda: GPIndividual(
        pset, output_num, gp_config["min_depth"], gp_config["max_depth"]
    )

    parsed["mutate"] = lambda individual: individual.mutate(**gp_config["mutation"])

    if "mate" in gp_config:
        if gp_config["mate"] == "micro":
            parsed["mate"] = lambda i1, i2: GPIndividual.micro_cx(i1, i2)
        elif gp_config["mate"] == "macro":
            parsed["mate"] = lambda i1, i2: GPIndividual.macro_cx(i1, i2)
        else:
            raise Exception(f"Unknown crossover type: {gp_config['mate']}")

    return parsed


def parse_model_config(config: dict, for_ship: bool) -> dict:
    """Parse model configuration.
    :param config: dictionary containing configuration for subevolution
    :param for_ship: True if config is for ship, False if it is for yard
    :return: dictionary containing model generator and mutation/crossover functions (if applicable)
    """
    if config["model"] == "NN":
        parsed = parse_nn_config(config["nn"], for_ship)
    elif config["model"] == "CGP":
        parsed = parse_cgp_config(config["cgp"], for_ship)
    elif config["model"] == "GP":
        parsed = parse_gp_config(config["gp"], for_ship)
    else:
        raise Exception(f"Unknown model type: {config['model']}")

    return parsed


def parse_ga_config(
    ga_config: dict, for_ship: bool, evaluate: Callable[[Individual], tuple[float, ...]]
) -> GAevolution:
    """Parse GAevolution configuration.
    :param ga_config: dictionary containing configuration for GAevolution
    :param for_ship: True if config is for ship, False if it is for yard
    :param evaluate: evaluation function
    :return: GAevolution instance
    """
    parsed = parse_model_config(ga_config, for_ship)

    toolbox = Toolbox()
    toolbox.register("Individual", parsed["Individual"])
    toolbox.register("mutate", parsed["mutate"])
    toolbox.register("mate", parsed["mate"])
    toolbox.register("select", selTournament, k=2, tournsize=ga_config["tournsize"])

    ga = GAevolution(
        toolbox,
        get_stats(),
        evaluate,
        population_size=ga_config["population_size"],
        generations_per_tick=ga_config["generations_per_tick"],
        save_best=ga_config["save_best"],
        population_file=ga_config.get("population_file", None),
    )

    return ga


def parse_es_config(
    es_config: dict, for_ship: bool, evaluate: Callable[[Individual], tuple[float, ...]]
) -> ESevolution:
    """Parse ESevolution configuration.
    :param es_config: dictionary containing configuration for ESevolution
    :param for_ship: True if config is for ship, False if it is for yard
    :param evaluate: evaluation function
    :return: ESevolution instance
    """
    parsed = parse_model_config(es_config, for_ship)

    toolbox = Toolbox()
    toolbox.register("Individual", parsed["Individual"])
    toolbox.register("mutate", parsed["mutate"])

    es = ESevolution(
        toolbox,
        get_stats(),
        evaluate,
        mu=es_config["mu"],
        lmbd=es_config["lmbd"],
        elitism=es_config["elitism"],
        generations_per_tick=es_config["generations_per_tick"],
        parents_file=es_config.get("parents_file", None),
    )

    return es


def parse_subevolution_config(
    subevolution_config: dict,
    for_ship: bool,
    evaluate: Callable[[Individual], tuple[float, ...]],
) -> Subevolution:
    if subevolution_config["alg"] == "GA":
        return parse_ga_config(subevolution_config, for_ship, evaluate)
    elif subevolution_config["alg"] == "ES":
        return parse_es_config(subevolution_config, for_ship, evaluate)
    else:
        raise Exception(f"Unknown subevolution type: {subevolution_config['alg']}")


def parse_config(config_file: str) -> Coevolution:
    with open(config_file, "rb") as f:
        config = tomllib.load(f)

    if config["enemy_agent"] == "random":
        enemy_agent = "random"
    elif config["enemy_agent"] == "naive":
        enemy_agent = naive
    elif config["enemy_agent"] == "mnmnao":
        enemy_agent = mnmnao
    else:
        raise Exception(f"Unknown enemy agent: {config['enemy_agent']}")

    evaluator = HaliteEvaluator(
        enemy_agent=enemy_agent, repeat_evaluation=config["repeat_evaluation"]
    )
    yard_model = parse_model_config(config["yard"], False)
    ship_subevolution = parse_subevolution_config(
        config["ship"], True, lambda i: evaluator.evaluate(i, yard_model["Individual"])
    )
    yard_subevolution = parse_subevolution_config(
        config["yard"],
        False,
        lambda i: evaluator.evaluate(ship_subevolution.representative, i),
    )

    coevolution = Coevolution(
        evaluator,
        ship_subevolution,
        yard_subevolution,
        generations=config["generations"],
    )
    return coevolution


if __name__ == "__main__":
    config_file = "../example.toml"

    coevolution = parse_config(config_file)
