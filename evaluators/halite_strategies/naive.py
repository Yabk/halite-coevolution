"""This code is from the following github repo: https://github.com/eashang1/halite_agents
I only added board_agent decorator to the agent function, all credit goes to the original author."""

from random import choice

from kaggle_environments.envs.halite.helpers import *

# Lists (some of) the objects

# board = Board(obs,config)
# board.cells
# board.ships
# board.shipyards
# board.players
# board.current_player
# board.current_player_id
# board.opponents
# board.configuration
# board.observation
# board.step
# board.next()
#
# cell.position
# cell.halite
# cell.ship
# cell.ship_id
# cell.shipyard
# cell.shipyard_id
# cell.north
# cell.south
# cell.east
# cell.west
#
# ship.id
# ship.halite
# ship.position
# ship.cell
# ship.player
# ship.player_id
# ship.next_action
#
# shipyard.id
# shipyard.position
# shipyard.cell
# shipyard.player
# shipyard.player_id
# shipyard.next_action
#
# player.id
# player.is_current_player
# player.halite
# player.next_actions
# player.ships
# player.ship_ids
# player.shipyards
# player.shipyard_ids


# Create position objects
class Position:
    x = 0
    y = 0

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y


# Calculates Manhattan distance
def manhattan_dist(p1, p2):
    xdif = min(abs(p1.x - p2.x), 21 - abs(p1.x - p2.x))
    ydif = min(abs(p1.y - p2.y), 21 - abs(p1.y - p2.y))
    return xdif + ydif


# Returns the distance to the nearest shipyard
def dist_shipyard(ship, shipyards):
    return min(
        map(
            lambda shipyard: manhattan_dist(ship.position, shipyard.position), shipyards
        ),
        default=10000,
    )


# Returns the closest shipyard
def closest_shipyard(ship, shipyards):
    return min(
        shipyards, key=lambda shipyard: manhattan_dist(ship.position, shipyard.position)
    )


# Moves a ship to deposit to the nearest shipyard
def deposit(ship, shipyards):
    east = Position((ship.position.x + 1) % 21, ship.position.y)
    west = Position((ship.position.x - 1) % 21, ship.position.y)
    north = Position(ship.position.x, (ship.position.y + 1) % 21)
    south = Position(ship.position.x, (ship.position.y - 1) % 21)

    shipyard = closest_shipyard(ship, shipyards)
    best = min(
        [east, west, north, south],
        key=lambda new_position: manhattan_dist(new_position, shipyard.position),
    )

    if east is best:
        ship.next_action = ShipAction.EAST
    elif west is best:
        ship.next_action = ShipAction.WEST
    elif north is best:
        ship.next_action = ShipAction.NORTH
    elif south is best:
        ship.next_action = ShipAction.SOUTH


# Moves a ship away from the nearest shipyard
def move_away(ship, shipyards):
    if dist_shipyard(ship, shipyards) == 0:
        ship.next_action = choice(
            [ShipAction.NORTH, ShipAction.EAST, ShipAction.SOUTH, ShipAction.WEST]
        )
    else:
        deposit(ship, shipyards)
        if ship.next_action is ShipAction.EAST:
            ship.next_action = ShipAction.WEST
        elif ship.next_action is ShipAction.WEST:
            ship.next_action = ShipAction.EAST
        elif ship.next_action is ShipAction.NORTH:
            ship.next_action = ShipAction.SOUTH
        elif ship.next_action is ShipAction.SOUTH:
            ship.next_action = ShipAction.NORTH


def nearby_ships(shipyard, ships):
    for ship in ships:
        if manhattan_dist(ship.position, shipyard.position) < 1:
            return True
    return False


# Checks if a ship is crashing
def crash(ship, ships):
    new_pos = next_pos(ship)
    for other in ships:
        if other is ship:
            continue
        check = next_pos(other)
        if new_pos == check:
            return True
    return False


# Returns the next position of a ship given it's next action
def next_pos(ship):
    if ship.next_action is ShipAction.NORTH:
        return Position(ship.position.x, (ship.position.y + 1) % 21)
    elif ship.next_action is ShipAction.SOUTH:
        return Position(ship.position.x, (ship.position.y - 1) % 21)
    elif ship.next_action is ShipAction.EAST:
        return Position((ship.position.x + 1) % 21, ship.position.y)
    elif ship.next_action is ShipAction.WEST:
        return Position((ship.position.x - 1) % 21, ship.position.y)
    else:
        return ship.position


# Avoids crashes
def avoid_crash(ship, ships):
    if ship.next_action is None:
        return

    if crash(ship, ships):
        ship.next_action = ShipAction.NORTH
    if crash(ship, ships):
        ship.next_action = ShipAction.SOUTH
    if crash(ship, ships):
        ship.next_action = ShipAction.EAST
    if crash(ship, ships):
        ship.next_action = ShipAction.WEST
    if crash(ship, ships):
        ship.next_action = None


@board_agent
def agent(board: Board):
    # def agent(obs, config):
    # Note that this board is a torus
    #    board = Board(obs, config)
    me = board.current_player

    cost = 0
    # Set actions for each ship
    for ship in me.ships:
        ship.next_action = None

        if (
            len(me.shipyards) > 0
            and me.halite - cost >= 2000
            and board.step <= 300
            and dist_shipyard(ship, me.shipyards) >= 8
        ) or (len(me.shipyards) == 0 and me.halite - cost >= 500):
            # if no shipyards or Manhattan distance of at least 9, make a shipyard
            ship.next_action = ShipAction.CONVERT
            cost += 500
        elif board.cells[ship.position].halite >= 50:
            # if on halite, mine it
            ship.next_action = None
        elif (ship.halite >= 500 or (ship.halite >= 75 and board.step >= 360)) and len(
            me.shipyards
        ):
            # if halite > 1000, move to deposit
            deposit(ship, me.shipyards)
            avoid_crash(ship, me.ships)
        elif dist_shipyard(ship, me.shipyards) <= 3:
            # if near shipyard, move away
            move_away(ship, me.shipyards)
            avoid_crash(ship, me.ships)
        else:
            # random choice of direction
            ship.next_action = choice(
                [ShipAction.NORTH, ShipAction.EAST, ShipAction.SOUTH, ShipAction.WEST]
            )

    # Set actions for each shipyard
    for shipyard in me.shipyards:
        shipyard.next_action = None
        if (len(me.ships) == 0 and me.halite - cost >= 500) or (
            me.halite - cost >= 750
            and board.step <= 300
            and not nearby_ships(shipyard, me.ships)
        ):
            # if we have no ships or a lot of halite and many turns left make more while you can
            shipyard.next_action = ShipyardAction.SPAWN
            cost += 500

    for ship in me.ships:
        avoid_crash(ship, me.ships)

    return me.next_actions
