# Halite Coevolution

This project is an attempt to use evolutionary algorithms and coevolution to
learn an effective strategy for playing Halite. The problem of playing the game
is decomposed into two strategies; strategy for controlling the ships and
strategy for controlling the shipyards. With that decomposition the problem is
transformed into multi-agent optimization problem on which we can apply
cooperative coevolution. Strategies for controlling ships and shipyards are
evolved in separate populations (hence coevolution). It uses different
evolutionary algorithms (GA/ES) and models (NN/CGP/GP) for the evolution of
both the strategies.

## Installation

Install the required packages

```pip install -r requirements.txt```

## Usage
To run the coevolution, you need to provide a configuration file in TOML format.  
An example configuration is provided in `example.toml`.

You can run the coevolution with the following command:

```python main.py example.toml```

After running the coevolution, the results are stored in the current directory in the following files:
- `ship-population.pkl`: Last population of the ship strategies (pickled)
- `yard-population.pkl`: Last population of the shipyard strategies (pickled)
- `ship-hof.pkl`: Hall of fame of the ship strategies (pickled)
- `yard-hof.pkl`: Hall of fame of the shipyard strategies (pickled)
- `ship-logbook.pkl`: Logbook of the ship subevolution (pickled)
- `yard-logbook.pkl`: Logbook of the shipyard subevolution (pickled)
- `coevolution.png`: Plot of the average fitness of the both subevolutions over generations
- `ship.png`: Plot of the average, median, best and worst fitness of the ship subevolution over generations
- `yard.png`: Plot of the average, median, best and worst fitness of the shipyard subevolution over generations
- `best_fit.txt`: The best fitness values of the ship and shipyard subevolutions in the whole run

(It is recommended to run the coevolution in a separate directory, as it will create a lot of files.)
