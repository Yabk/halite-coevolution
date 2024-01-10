"""This code is from the following github gist: https://gist.github.com/minaminao/cbe4a3d514ea9ae924cb5d59181d78cc
All the credit goes to the original authors.
"""
# https://github.com/Kaggle/kaggle-environments/blob/master/kaggle_environments/envs/halite/helpers.py)
import datetime
import math
import random
import sys
import time
from collections import deque
from heapq import heappop, heappush
from operator import pos

import numpy as np
from kaggle_environments.envs.halite.helpers import *
from tqdm import tqdm_notebook as tqdm

"""
Global constants
NOTE: 関数内で変更しても変わらない
"""
LOG = False
LOG_FLOW_ADD_EDGE = True

MAX_STEPS = 400

# Attack mode
ATTACK_START_STEP = 100
ATTACK_END_STEP = 300

MAX_BFS_DISTANCE = 10

# No need to change
INF = 10000000000
EPS = 1e-5

# 船が移動する4方向
DIRECTIONS = [ShipAction.NORTH, ShipAction.EAST, ShipAction.SOUTH, ShipAction.WEST]
# ユーザーの行動
ACTIONS = [
    ShipAction.NORTH,
    ShipAction.EAST,
    ShipAction.SOUTH,
    ShipAction.WEST,
    None,
]  # TODO: ShipAction.CONVERT

# 所持していい船の最大数
MAX_SHIP_NUM = 50
# spawn と止めるターン数
STOP_SPAWN = 300

"""
Global variables
"""

# Logging
# 動作環境だと一意なシード値になってしまうため
rnd = random.Random(time.time())
# TODO: botnameをつける __file__は使えない
if LOG:
    log_filename_time_prefix = datetime.datetime.now().strftime("%m%d_%H%M%S")
    log_filename = f"{log_filename_time_prefix}_{rnd.randint(0, 10000)}.log"

progress_bar = None


"""
Structures and Functions
"""


class MinCostFlow:
    """
    最小費用流
    """

    INF = 10**18
    EPS = 1e-6
    ABORT = "ABORT"

    def __init__(self, N):
        self.N = N
        self.G = [[] for _ in range(N)]
        self.edge_num = 0

    def add_edge(self, fr, to, cap, cost):
        forward = [to, cap, cost, None]
        backward = forward[3] = [fr, 0, -cost, forward]
        self.G[fr].append(forward)
        self.G[to].append(backward)
        self.edge_num += 2

    def flow(self, s, t, f, start_time=None, exit_time=None):
        N = self.N
        G = self.G
        INF = MinCostFlow.INF
        EPS = MinCostFlow.EPS

        res = 0
        H = [0] * N
        prv_v = [0] * N
        prv_e = [None] * N

        d0 = [INF] * N
        dist = [INF] * N

        while f > EPS:
            dist[:] = d0
            dist[s] = 0
            que = [(0, s)]

            if start_time and time.time() - start_time > exit_time:
                return self.ABORT

            while que:
                c, v = heappop(que)
                if dist[v] + EPS < c:
                    continue
                r0 = dist[v] + H[v]
                for e in G[v]:
                    w, cap, cost, _ = e
                    if cap > EPS and r0 + cost - H[w] + EPS < dist[w]:
                        dist[w] = r = r0 + cost - H[w]
                        prv_v[w] = v
                        prv_e[w] = e
                        heappush(que, (r, w))
            if dist[t] == INF:
                return None

            for i in range(N):
                H[i] += dist[i]

            d = f
            v = t
            while v != s:
                d = min(d, prv_e[v][1])
                v = prv_v[v]
            f -= d
            res += d * H[t]
            v = t
            while v != s:
                e = prv_e[v]
                e[1] -= d
                e[3][1] += d
                v = prv_v[v]
        return res


TURNS_OPTIMAL = np.array(
    [
        [0, 2, 3, 4, 4, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 8],
        [0, 1, 2, 3, 3, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 7, 7, 7],
        [0, 0, 2, 2, 3, 3, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 7],
        [0, 0, 1, 2, 2, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6],
        [0, 0, 0, 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6],
        [0, 0, 0, 0, 0, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
)


def limit(x, a, b):
    if x < a:
        return a
    if x > b:
        return b
    return x


def num_turns_to_mine(carrying, halite, rt_travel):
    # How many turns should we plan on mining?
    # C=carried halite, H=halite in square, rt_travel=steps to square and back
    # to shipyard
    if carrying < 0.01:
        ch = 0
    elif halite < 30:
        ch = TURNS_OPTIMAL.shape[0] - 1
    else:
        ch = int(math.log(carrying / halite) * 2.5 + 5.5)
        ch = limit(ch, 0, TURNS_OPTIMAL.shape[0] - 1)
    rt_travel = int(limit(rt_travel, 0, TURNS_OPTIMAL.shape[1] - 1))

    return TURNS_OPTIMAL[ch, rt_travel]


def halite_per_turn(carrying, halite, travel, min_mine=1):
    # compute return from going to a cell containing halite, using optimal number of mining steps
    # returns halite mined per turn, optimal number of mining steps
    # Turns could be zero, meaning it should just return to a shipyard
    # (subject to min_mine)
    turns = num_turns_to_mine(carrying, halite, travel)
    if turns < min_mine:
        turns = min_mine
    mined = carrying + (1 - 0.75**turns) * halite
    return mined / (travel + turns)


def log(message):
    """
    messageをlog_fileに書き込む
    """
    if not LOG:
        return
    with open(f"logs/{log_filename}", "a") as log_file:
        log_file.write(message + "\n")


def spawn_step(ship: Ship):
    """
    shipが生成されたターンを返す
    """
    return int(ship.id.split("-")[0])


# 状態初期化したかどうか
# 途中からゲームを始めるとき用
initialized = False


def agent(obs, config):
    """
    obs
    - obs.step: 0 ~ MAX_STEPS - 2

    config
    - dict
    - configじゃなくてboard.configurationを使うべし
    """

    global initialized
    global progress_bar

    # boardに全情報が詰まっている
    board = Board(obs, config)

    """
    時間計測系
    """
    start_time = time.time()
    if not initialized:
        EXIT_TIME = board.configuration.agent_timeout * 0.8
    else:
        EXIT_TIME = board.configuration.act_timeout * 0.8

    def log_time(message, logging=True):
        elapsed_time = time.time() - start_time
        if logging:
            log(f"time: {elapsed_time: .5f} {message}")
        if elapsed_time >= EXIT_TIME:
            print(
                f"Step {obs.step} time over: {elapsed_time}, {message}", file=sys.stderr
            )
            return True
        return False

    """
    情報加工系
    obsとconfigからboardを構築する
    """
    me = board.current_player

    # Constants
    BOARD_SIZE = board.configuration.size  # 21

    # プログレスバーの初期化
    if me.id == 0:
        if not progress_bar:
            progress_bar = tqdm(total=board.configuration.episode_steps)
            progress_bar.update(obs.step + 1)
        progress_bar.update(1)

    if not initialized:
        initialized = True
        if LOG:
            print(f"player {me.id}'s log filename: {log_filename}")
        log(f"player_id: {me.id}")

    log(f"step {obs.step + 1}")

    # index to cell 高速化用
    index_to_cell = {
        point.to_index(BOARD_SIZE): cell for point, cell in board.cells.items()
    }

    # 障害物 index => (max_cargo, count)
    # 例えば、100 だったら、最小 100 haliteを持つ敵がいる。
    # haliteを120持つshipはNG haliteを80持つshipはOK と判定するための辞書
    # TODO: remove
    index_to_ship_max_cargo: Dict[int, (int, int)] = {}

    # 小さいほうが強い
    rank_of_ship_num = 0
    for opponent in board.opponents:
        if len(me.ships) < len(opponent.ships):
            rank_of_ship_num += 1

    """
    Methods
    """

    def calc_distance_between_points(p1x, p1y, p2x, p2y):
        """
        2点間の距離を計算
        """
        dx = min((p1x - p2x) % BOARD_SIZE, (p2x - p1x) % BOARD_SIZE)
        dy = min((p1y - p2y) % BOARD_SIZE, (p2y - p1y) % BOARD_SIZE)
        return dx + dy

    def calc_chebyshev_distance_between_points(p1x, p1y, p2x, p2y):
        """
        2点間のChebyshev距離を計算
        """
        dx = min((p1x - p2x) % BOARD_SIZE, (p2x - p1x) % BOARD_SIZE)
        dy = min((p1y - p2y) % BOARD_SIZE, (p2y - p1y) % BOARD_SIZE)
        return max(dx, dy)

    def calc_min_xy_distance_between_points(p1x, p1y, p2x, p2y):
        """
        2点間のChebyshev距離を計算
        """
        dx = min((p1x - p2x) % BOARD_SIZE, (p2x - p1x) % BOARD_SIZE)
        dy = min((p1y - p2y) % BOARD_SIZE, (p2y - p1y) % BOARD_SIZE)
        return min(dx, dy)

    def translate(ax, ay, dx, dy):
        """
        (ax, ay) + (dx, dy) -> (bx, by)
        """
        return (ax + dx + BOARD_SIZE) % BOARD_SIZE, (ay + dy + BOARD_SIZE) % BOARD_SIZE

    def index_to_xy(index):
        """
        index -> (x, y)
        """
        y, x = divmod(index, BOARD_SIZE)
        return x, (BOARD_SIZE - y - 1)

    def xy_to_index(x, y):
        """
        (x, y) -> index
        """
        return (BOARD_SIZE - y - 1) * BOARD_SIZE + x

    def is_opposite_ship_or_shipyard(point_index) -> bool:
        """
        pointにshipかshipyardがいるかどうか
        """
        cell = index_to_cell[point_index]
        return (cell.ship and cell.ship.player_id != me.id) or (
            cell.shipyard and cell.shipyard.player_id != me.id
        )

    def update_index_to_ship_max_cargo(index, cargo: int):
        """ """
        if index in index_to_ship_max_cargo:
            max_cargo, count = index_to_ship_max_cargo[index]
            if cargo < max_cargo:
                index_to_ship_max_cargo[index] = (cargo, 1)
            elif cargo == max_cargo:
                index_to_ship_max_cargo[index] = (cargo, count + 1)
        else:
            index_to_ship_max_cargo[index] = (cargo, 1)

    (SHIP_LOSE, SHIP_DRAW, SHIP_WIN) = range(3)

    def check_max_cargo_condition(ship_halite, index):
        """
        あるshipがpへ行っても船が壊されないかどうか
        TODO: Rename
        """
        if not index in index_to_ship_max_cargo:
            return SHIP_WIN
        if ship_halite < index_to_ship_max_cargo[index][0]:
            return SHIP_WIN
        elif ship_halite > index_to_ship_max_cargo[index][0]:
            return SHIP_LOSE
        return SHIP_DRAW

    if log_time(""):
        return me.next_actions

    # 各マスの制限を更新
    # 相手のshipに対して
    # TODO: 周辺4マスは制限強いので緩くする
    for ship_id, ship in board.ships.items():
        if ship.player_id != me.id:
            update_index_to_ship_max_cargo(
                ship.position.to_index(BOARD_SIZE), ship.halite
            )
            for direction in DIRECTIONS:
                next_point = ship.position.translate(direction.to_point(), BOARD_SIZE)
                update_index_to_ship_max_cargo(
                    next_point.to_index(BOARD_SIZE), ship.halite
                )

    # 各地点について、その地点から最短の拠点までの距離
    # 拠点が一つも存在しなければINFになる
    distance_to_my_nearest_shipyard = [
        [0 for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)
    ]
    distance_to_opponent_nearest_shipyard = [
        [0 for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)
    ]
    chebyshev_distance_to_my_nearest_shipyard = [
        [0 for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)
    ]
    min_xy_distance_to_my_nearest_shipyard = [
        [0 for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)
    ]
    distance_to_opponent_nearest_ship = [
        [0 for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)
    ]
    distance_to_my_nearest_ship = [
        [0 for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)
    ]
    distances_to_my_nearest_shipyard = [
        [[] for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)
    ]
    for x in range(BOARD_SIZE):
        for y in range(BOARD_SIZE):
            min_num = INF
            for shipyard in me.shipyards:
                min_num = min(
                    min_num,
                    calc_distance_between_points(
                        shipyard.position.x, shipyard.position.y, x, y
                    ),
                )
            distance_to_my_nearest_shipyard[x][y] = min_num

            min_num = INF
            for shipyard in me.shipyards:
                min_num = min(
                    min_num,
                    calc_chebyshev_distance_between_points(
                        shipyard.position.x, shipyard.position.y, x, y
                    ),
                )
            chebyshev_distance_to_my_nearest_shipyard[x][y] = min_num

            min_num = INF
            for shipyard in me.shipyards:
                min_num = min(
                    min_num,
                    calc_min_xy_distance_between_points(
                        shipyard.position.x, shipyard.position.y, x, y
                    ),
                )
            min_xy_distance_to_my_nearest_shipyard[x][y] = min_num

            min_num = INF
            for opponent in board.opponents:
                for ship in opponent.ships:
                    min_num = min(
                        min_num,
                        calc_distance_between_points(
                            ship.position.x, ship.position.y, x, y
                        ),
                    )
            distance_to_opponent_nearest_ship[x][y] = min_num

            min_num = INF
            for opponent in board.opponents:
                for shipyard in opponent.shipyards:
                    min_num = min(
                        min_num,
                        calc_distance_between_points(
                            shipyard.position.x, shipyard.position.y, x, y
                        ),
                    )
            distance_to_opponent_nearest_shipyard[x][y] = min_num

            min_num = INF
            for ship in me.ships:
                distance = calc_distance_between_points(
                    ship.position.x, ship.position.y, x, y
                )
                if distance == 0:
                    continue
                min_num = min(min_num, distance)
            distance_to_my_nearest_ship[x][y] = min_num

            for shipyard in me.shipyards:
                distances_to_my_nearest_shipyard[x][y].append(
                    calc_distance_between_points(
                        shipyard.position.x, shipyard.position.y, x, y
                    )
                )
            distances_to_my_nearest_shipyard[x][y].sort()

    # 相手のshipyardに対して
    for _shipyard_id, shipyard in board.shipyards.items():
        if shipyard.player_id != me.id:
            shipyard_index = shipyard.position.to_index(BOARD_SIZE)
            shipyard_destroyer_max_cargo = 0  # 0 だと障害物

            if (
                rank_of_ship_num <= 1
                and distance_to_my_nearest_shipyard[shipyard.position.x][
                    shipyard.position.y
                ]
                <= 3
            ) or obs.step >= 375:
                shipyard_destroyer_max_cargo = 1

                if shipyard_index in index_to_ship_max_cargo:
                    index_to_ship_max_cargo.pop(shipyard_index)
                update_index_to_ship_max_cargo(
                    shipyard_index, shipyard_destroyer_max_cargo
                )

                for next_direction in DIRECTIONS:
                    neighbor_point_index = shipyard.position.translate(
                        next_direction.to_point(), BOARD_SIZE
                    ).to_index(BOARD_SIZE)
                    if neighbor_point_index in index_to_ship_max_cargo:
                        index_to_ship_max_cargo.pop(neighbor_point_index)
                    update_index_to_ship_max_cargo(
                        neighbor_point_index, shipyard_destroyer_max_cargo
                    )

    RANGE_CAMP = math.ceil(math.sqrt(5 * len(me.ships) / max(1, len(me.shipyards)) / 2))
    log(f"{len(me.ships)},{len(me.shipyards)} => RANGE_CAMP: {RANGE_CAMP}")
    CAMP_HALITE = 100

    log("index_to_ship_max_cargo")
    for k, v in index_to_ship_max_cargo.items():
        log(f"  {index_to_xy(k)} {v}")

    def is_minable_halite_area_in_attack_mode(x, y):
        """
        Attack mode中にhaliteをマイニングしていいエリア
        TODO: なめらかにしたい
        """
        if distance_to_my_nearest_shipyard[x][y] <= 2:
            return distance_to_opponent_nearest_ship[x][y] >= 1
        if distance_to_my_nearest_shipyard[x][y] <= 4:
            return distance_to_opponent_nearest_ship[x][y] >= 2
        if distance_to_my_nearest_shipyard[x][y] <= 8:
            return distance_to_opponent_nearest_ship[x][y] >= 3
        return False

    def bfs(ship: Ship, point_from: Point, max_distance=1000):
        """
        shipから全点への最短経路を求める
        point_fromの条件チェックはしない(TODO: 本当に良い？)
        """
        point_directions = [direction.to_point() for direction in DIRECTIONS]
        q = deque()
        # (position, distance, parent, prev direction)
        q.append((point_from.to_index(BOARD_SIZE), 0, -1, -1))
        index_to_info = {}
        while len(q):
            point_u_index, distance, parent_index, direction = q.popleft()
            point_u_x, point_u_y = index_to_xy(point_u_index)

            if point_u_index in index_to_info:
                continue
            index_to_info[point_u_index] = (distance, parent_index, direction)

            if distance == max_distance:
                continue

            for direction, point_direction in zip(DIRECTIONS, point_directions):
                point_v_x, point_v_y = translate(
                    point_u_x, point_u_y, point_direction.x, point_direction.y
                )
                point_v_index = xy_to_index(point_v_x, point_v_y)

                # 自分の拠点はcontinueしない。2つ目のbfsではmax_cargoを考慮
                if not (
                    index_to_cell[point_v_index].shipyard
                    and index_to_cell[point_v_index].shipyard.player_id != me.id
                ):
                    # 強い船が周囲1マス以内にいるならcontinue
                    if (
                        check_max_cargo_condition(ship.halite, point_v_index)
                        == SHIP_LOSE
                    ):
                        continue
                    # 互角の船が周囲1マス以内にいるならcontinue
                    if (
                        check_max_cargo_condition(ship.halite, point_v_index)
                        == SHIP_DRAW
                    ):
                        # and is_opposite_ship_or_shipyard(point_v_index):
                        continue

                if point_v_index in index_to_info:
                    continue

                q.append((point_v_index, distance + 1, point_u_index, direction))
        return index_to_info

    def bfs_one_target(
        ship: Ship, point_from: Point, point_to: Point, check_max_cargo=True
    ):
        """
        ある点 から ある点 への最短経路を求める
        point_fromの条件チェックはしない
        """
        point_directions = [direction.to_point() for direction in DIRECTIONS]
        q = deque()
        point_from_index = point_from.to_index(BOARD_SIZE)
        point_to_index = point_to.to_index(BOARD_SIZE)
        # (position, distance, parent, prev direction)
        q.append((point_from_index, 0, -1, -1))
        index_to_info = {}
        while len(q):
            point_u_index, distance, parent_index, direction = q.popleft()
            point_u_x, point_u_y = index_to_xy(point_u_index)
            if point_u_index in index_to_info:
                continue
            index_to_info[point_u_index] = (distance, parent_index, direction)

            if point_to_index in index_to_info:
                break

            for direction, point_direction in zip(DIRECTIONS, point_directions):
                point_v_x, point_v_y = translate(
                    point_u_x, point_u_y, point_direction.x, point_direction.y
                )
                point_v_index = xy_to_index(point_v_x, point_v_y)

                # TODO: refine
                if (
                    check_max_cargo
                    and check_max_cargo_condition(ship.halite, point_v_index)
                    != SHIP_WIN
                ):
                    continue
                if point_v_index in index_to_info:
                    continue
                q.append((point_v_index, distance + 1, point_u_index, direction))

        if point_to_index in index_to_info:
            res = []
            point_v_index = point_to_index
            while point_v_index != point_from_index:
                distance, point_u_index, direction = index_to_info[point_v_index]
                res.append((point_u_index, direction))
                point_v_index = point_u_index
            res = res[::-1]
            return True, res
        else:
            return False, []

    tmp_halite = me.halite

    log(f"len(me.ships): {len(me.ships)}")

    # shipをconvertすべきタイミングかどうか

    def should_ship_convert():
        if len(me.ships) < 12:
            n = 1
        elif len(me.ships) < 14:
            n = 2
        elif len(me.ships) < 23:
            n = 3
        elif len(me.ships) < 33:
            n = 4
        else:
            n = 5

        if obs.step < 200:
            n = min(n, 5)
        elif obs.step < 250:
            n = min(n, 4)
        elif obs.step < 300:
            n = min(n, 3)
        elif obs.step < 350:
            n = min(n, 2)
        else:
            n = min(n, 1)

        return len(me.shipyards) < n

    ship_convert = should_ship_convert()
    log(f"ship_convert {ship_convert}")

    """
    Cell scoring
    """
    if log_time("cell scoring start"):
        return me.next_actions

    # セルの評価 (近くにhaliteがあれば高くなる。自セルは含まない。)
    cell_score = [[0 for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    cell_score_and_index = []

    RANGE_DISTANCE = 3
    for ax in range(BOARD_SIZE):
        for ay in range(BOARD_SIZE):
            log(
                f"({ax}, {ay}) cell halite: {index_to_cell[xy_to_index(ax,ay)].halite}, ship halite: {index_to_cell[xy_to_index(ax,ay)].ship.halite if index_to_cell[xy_to_index(ax,ay)].ship else None}"
            )
            if is_opposite_ship_or_shipyard(xy_to_index(ax, ay)):
                continue
            for dx in range(-RANGE_DISTANCE, RANGE_DISTANCE + 1):
                for dy in range(-RANGE_DISTANCE, RANGE_DISTANCE + 1):
                    if abs(dx) + abs(dy) > RANGE_DISTANCE:
                        continue
                    bx, by = translate(ax, ay, dx, dy)
                    distance = calc_distance_between_points(ax, ay, bx, by)
                    if distance == 0:
                        continue
                    cell_score[ax][ay] += (
                        index_to_cell[xy_to_index(bx, by)].halite / distance
                    )
            cell_score_and_index.append((cell_score[ax][ay], (ax, ay)))
    cell_score_and_index.sort()
    cell_score_and_index.reverse()
    log(f"cell_score_and_index: {cell_score_and_index[:5]}")
    if log_time("cell scoring end"):
        return me.next_actions

    """
    shipyardの行動
    """

    if log_time("shipyard start"):
        return me.next_actions

    # board_total_halite = sum(obs.halite)
    score_and_shipyard_ids = []
    for shipyard in me.shipyards:
        score_and_shipyard_ids.append(
            (cell_score[shipyard.position.x][shipyard.position.y], shipyard.id)
        )
    score_and_shipyard_ids.sort()
    score_and_shipyard_ids.reverse()

    # 作るなら早めに作ったほうが良い
    # 躊躇する必要はない
    spawn_ship_num = 0
    # expected_ship_halite = 754.2045976147749 + obs.step*- \
    #     2.28367203 + board_total_halite * 0.00364342
    # log(f"expected ship halite: {expected_ship_halite}")
    for _, shipyard_id in score_and_shipyard_ids:
        shipyard = board.shipyards[shipyard_id]
        shipyard_index = shipyard.position.to_index(BOARD_SIZE)

        # ship convert するならその余地を残しておく
        if ship_convert and tmp_halite < 1000:
            break

        spawn = False
        # >= 1000 は拠点攻撃マン対策
        if tmp_halite >= 1000 or (obs.step < ATTACK_START_STEP and tmp_halite >= 500):
            spawn |= (
                obs.step < STOP_SPAWN and len(me.ships) + spawn_ship_num < MAX_SHIP_NUM
            )  # and expected_ship_halite > 500
            spawn |= len(me.ships) + spawn_ship_num < 2

        if spawn:
            shipyard.next_action = ShipyardAction.SPAWN
            spawn_ship_num += 1
            tmp_halite -= 500
            update_index_to_ship_max_cargo(shipyard_index, -1)  # -1にしないと拠点守るくんと競合してしまう
        else:
            shipyard.next_action = None

    if log_time("shipyard end"):
        return me.next_actions

    """
    ターゲットの割当
    """

    # 最小費用流インデックス構築
    ship_id_to_mcf_node_id = {}
    mcf_id_to_ship_id = {}
    for ship_index, ship_id in enumerate(me.ship_ids):
        ship_id_to_mcf_node_id[ship_id] = ship_index
        mcf_id_to_ship_id[ship_index] = ship_id

    BASE_COST = 10000

    def calc_cost(x):
        return x + BASE_COST + 1

    """
    フローの計算量
    N = 30 + 400 + 2 = 432
    M = 30
    M * (N*M) * log N = 30 * 432 * 30 * 9 = 3,499,200
    """
    mcf_node_num = len(me.ships) + BOARD_SIZE * BOARD_SIZE + 2
    ship_num = len(me.ships)
    start_mcf_node_id = mcf_node_num - 1
    goal_mcf_node_id = mcf_node_num - 2

    def build_mcf(first=False):
        """
        最小費用流のグラフの初期化
        """
        mcf = MinCostFlow(mcf_node_num)
        for i in range(len(me.ships)):
            mcf.add_edge(start_mcf_node_id, i, 1, calc_cost(0))  # fr, to, cap, cost
        for i in range(BOARD_SIZE * BOARD_SIZE):
            point = Point.from_index(i, BOARD_SIZE)
            if first:
                # 拠点
                if distance_to_my_nearest_shipyard[point.x][point.y] == 0:
                    if obs.step < ATTACK_END_STEP:
                        mcf.add_edge(ship_num + i, goal_mcf_node_id, 5, calc_cost(0))
                    else:
                        mcf.add_edge(ship_num + i, goal_mcf_node_id, 100, calc_cost(0))
                    if (
                        100 <= obs.step or len(me.ships) >= 2
                    ) and distance_to_my_nearest_shipyard[point.x][point.y] == 0:
                        mcf.add_edge(
                            ship_num + i, goal_mcf_node_id, 1, calc_cost(-1000)
                        )
                elif (
                    is_opposite_ship_or_shipyard(point.to_index(BOARD_SIZE))
                    and ATTACK_START_STEP <= obs.step < ATTACK_END_STEP
                ):
                    if distance_to_opponent_nearest_shipyard[point.x][point.y] == 0:
                        mcf.add_edge(ship_num + i, goal_mcf_node_id, 2, calc_cost(0))
                    elif is_minable_halite_area_in_attack_mode(point.x, point.y):
                        mcf.add_edge(ship_num + i, goal_mcf_node_id, 2, calc_cost(0))
                    else:
                        mcf.add_edge(ship_num + i, goal_mcf_node_id, 4, calc_cost(0))
                else:
                    mcf.add_edge(ship_num + i, goal_mcf_node_id, 1, calc_cost(0))
            else:
                if obs.step < 390:
                    if (
                        100 <= obs.step or len(me.ships) >= 2
                    ) and distance_to_my_nearest_shipyard[point.x][point.y] == 0:
                        mcf.add_edge(
                            ship_num + i, goal_mcf_node_id, 1, calc_cost(-1000)
                        )
                    else:
                        mcf.add_edge(ship_num + i, goal_mcf_node_id, 1, calc_cost(0))
                else:
                    mcf.add_edge(ship_num + i, goal_mcf_node_id, 10, calc_cost(0))
        return mcf

    mcf = build_mcf(True)

    if log_time("flow1 adding edge start"):
        return me.next_actions

    def fast_hash(x):
        return x * 107 % 31

    # bfsして各船に対して各セルの評価値を定める
    # haliteが多いほど高く、距離が遠いほど低く
    for ship in me.ships:
        if log_time("flow1 ship loop", False):
            return me.next_actions
        ship_index = ship_id_to_mcf_node_id[ship.id]
        if obs.step <= 15:
            index_to_info = bfs(ship, ship.position, MAX_BFS_DISTANCE * 2)
        else:
            index_to_info = bfs(ship, ship.position, MAX_BFS_DISTANCE)

        for target_point_index, (
            distance,
            _parent_index,
            _direction,
        ) in index_to_info.items():
            target_point = Point.from_index(target_point_index, BOARD_SIZE)
            # TODO: refine
            # if (not check_max_cargo_condition(ship.halite, target_point_index)) and is_opposite_ship_or_shipyard(target_point_index):
            # BFSは相打ち以外
            if check_max_cargo_condition(ship.halite, target_point_index) != SHIP_WIN:
                # spawnされてると自分の拠点上にはいけないため、SHIP_LOSE でここに入る
                continue

            if len(me.shipyards) == 0 and MAX_STEPS - (obs.step + 1) < distance:
                continue
            if (
                len(me.shipyards) > 0
                and MAX_STEPS - (obs.step + 1)
                < distance
                + distance_to_my_nearest_shipyard[target_point.x][target_point.y]
            ):
                continue
            # distance == 0 だと stay するから、その分を引かなくちゃいけない
            if (
                len(me.shipyards) > 0
                and MAX_STEPS - (obs.step + 2)
                < distance
                + distance_to_my_nearest_shipyard[target_point.x][target_point.y]
                and distance == 0
            ):
                continue

            # 自分の拠点だった場合
            if distance_to_my_nearest_shipyard[target_point.x][target_point.y] == 0:
                # shipが居ると distance=0 でゼロ除算になるため場合分け
                if distance > 0:
                    expected_halite = ship.halite / distance
                else:
                    expected_halite = 0
            else:
                target_position_halite = 0
                target_cell = index_to_cell[target_point_index]

                if (
                    obs.step <= 20
                    and len(me.shipyards) <= 1
                    and distance_to_my_nearest_shipyard[target_point.x][target_point.y]
                    < 10 - spawn_step(ship)
                ):
                    continue

                if obs.step < ATTACK_START_STEP and target_cell.halite < 100:
                    continue

                if ATTACK_START_STEP <= obs.step < ATTACK_END_STEP:
                    # TODO: 相手の船のhaliteをhalite_per_turnに入れない (絶対値的には良くない)

                    # if ship.halite == 0 and distance_to_my_nearest_shipyard[target_point.x][target_point.y] <= RANGE_CAMP and target_cell.halite == 0 and (target_point_index + obs.step) % 3 == 0:
                    #     target_position_halite += 300 / \
                    #         distance_to_my_nearest_shipyard[target_point.x][target_point.y]

                    # if is_minable_halite_area_in_attack_mode(target_point.x, target_point.y):
                    #     # 割る数が以外と重要
                    #     target_position_halite += obs.halite[target_point.to_index(
                    #         BOARD_SIZE)] / 3
                    # elif ship.halite == 0 and distance_to_my_nearest_shipyard[target_point.x][target_point.y] <= RANGE_CAMP and target_cell.ship and target_cell.ship.player_id != me.id:
                    #     target_position_halite += 200
                    # else:
                    #     # 500未満じゃないと拠点作られて船を失ってしまう
                    #     if spawn_step(ship) % 2 == 1 and ship.halite == 0 \
                    #             and target_cell.ship and target_cell.ship.player_id != me.id and 0 < target_cell.ship.halite < 500:
                    #         target_position_halite += target_cell.ship.halite

                    if is_minable_halite_area_in_attack_mode(
                        target_point.x, target_point.y
                    ):
                        if obs.halite[target_point_index] > 400 / pow(
                            distance_to_my_nearest_shipyard[target_point.x][
                                target_point.y
                            ],
                            0.5,
                        ):
                            target_position_halite += obs.halite[target_point_index]

                    if ship.halite == 0:
                        # 500未満じゃないと拠点作られて船を失ってしまう
                        if (
                            target_cell.ship
                            and target_cell.ship.player_id != me.id
                            and 0 <= target_cell.ship.halite < 500
                        ):
                            halite = target_cell.ship.halite
                            if (
                                distance_to_my_nearest_shipyard[target_point.x][
                                    target_point.y
                                ]
                                <= RANGE_CAMP
                            ):
                                halite = max(100, halite)
                            target_position_halite += halite

                    # 陣地ロジック
                    # if ship.halite == 0:
                    #     if distance_to_nearest_shipyard[target_point.x][target_point.y] <= RANGE_CAMP and obs.halite[target_point_index] >= CAMP_HALITE:
                    #         target_position_halite += obs.halite[target_point_index]
                    #     elif distance_to_nearest_shipyard[target_point.x][target_point.y] <= RANGE_CAMP:
                    #         if target_cell.ship and target_cell.ship.player_id != me.id and target_cell.ship.halite > ship.halite:
                    #             target_position_halite += 100 * \
                    #                 distance_to_nearest_shipyard[target_point.x][target_point.y]
                    #         if len(me.shipyards) > 0:
                    #             target_position_halite += obs.halite[target_point_index] / 10
                else:
                    target_position_halite += obs.halite[target_point_index]

                if (
                    375 <= obs.step
                    and target_cell.shipyard
                    and target_cell.shipyard.player_id != me.id
                ):
                    target_position_halite += 500

                # 近くにいるshipyardは破壊
                elif (
                    rank_of_ship_num <= 1
                    and distance_to_my_nearest_shipyard[target_point.x][target_point.y]
                    <= 3
                    and target_cell.shipyard
                    and target_cell.shipyard.player_id != me.id
                ):
                    target_position_halite += 500

                # TODO: 距離2以内にいるshipは破壊

                # 0だったら目的地として相応しくない
                # 2回目のフローでは0のマスも候補になるからcontinueしても良い
                if target_position_halite == 0:
                    continue

                if len(me.shipyards) == 0:
                    # 拠点に帰らないので、cargo は 0 にしないと動かなくなってしまう
                    expected_halite = halite_per_turn(
                        0, target_position_halite, distance
                    )
                else:
                    expected_halite = halite_per_turn(
                        ship.halite,
                        target_position_halite,
                        distance_to_my_nearest_shipyard[target_point.x][target_point.y]
                        + distance,
                    )

            if expected_halite > 0:
                mcf.add_edge(
                    ship_index,
                    ship_num + target_point_index,
                    1,
                    calc_cost(-expected_halite),
                )
                if LOG_FLOW_ADD_EDGE:
                    log(
                        f"flow1 add edge: {ship.position} => {target_point}, expected_halite {expected_halite}"
                    )

    if log_time("flow1 adding edge end"):
        return me.next_actions

    log(f"flow1 edge num: {mcf.edge_num}")
    if (
        mcf.flow(start_mcf_node_id, goal_mcf_node_id, ship_num, start_time, EXIT_TIME)
        == MinCostFlow.ABORT
    ):
        log_time("abort flow1")
        return me.next_actions

    if log_time("flow1 end"):
        return me.next_actions

    mcf_node_id_to_target_index_and_score = {}
    for ship_index in range(len(mcf.G)):
        edges = mcf.G[ship_index]
        for to, cap, cost, _ in edges:
            if (
                ship_index < ship_num
                and ship_num <= to < mcf_node_num - 2
                and cap < MinCostFlow.EPS
            ):
                next_index = to - ship_num
                ship_id = mcf_id_to_ship_id[ship_index]
                ship = board.ships[ship_id]
                mcf_node_id_to_target_index_and_score[ship_index] = (next_index, cost)
                next_point = Point.from_index(next_index, BOARD_SIZE)
                log(
                    f"flow1: id {ship_id}, halite {ship.halite}, {ship.position} => target {next_point}, cost {cost}"
                )

    # グラフの初期化
    mcf = build_mcf()

    if log_time("flow2 adding edge start"):
        return me.next_actions

    for ship in me.ships:
        if log_time("flow2 ship loop", False):
            return me.next_actions

        ship_id = ship.id
        ship_node_id = ship_id_to_mcf_node_id[ship_id]
        ship_index = ship.position.to_index(BOARD_SIZE)

        # 移動候補の評価値の初期化
        for next_direction in DIRECTIONS:
            target_point = ship.position.translate(
                next_direction.to_point(), BOARD_SIZE
            )
            target_point_index = target_point.to_index(BOARD_SIZE)
            target_cell = index_to_cell[target_point_index]
            if check_max_cargo_condition(ship.halite, target_point_index) == SHIP_WIN:
                cost = (
                    distance_to_my_nearest_shipyard[target_point.x][target_point.y]
                    if distance_to_my_nearest_shipyard[target_point.x][target_point.y]
                    != INF
                    else 0
                )
                mcf.add_edge(
                    ship_node_id, ship_num + target_point_index, 1, calc_cost(cost)
                )
                if LOG_FLOW_ADD_EDGE:
                    log(
                        f"flow2 add edge: {ship.position} => {target_point}, cost {cost}, distance 1"
                    )
            # 行けるっちゃいける
            elif (
                check_max_cargo_condition(ship.halite, target_point_index) == SHIP_DRAW
            ):
                mcf.add_edge(
                    ship_node_id,
                    ship_num + target_point_index,
                    1,
                    calc_cost(BASE_COST / 2 - 1),
                )
                if LOG_FLOW_ADD_EDGE:
                    log(
                        f"flow2 add edge: {ship.position} => {target_point}, cost {BASE_COST/2-1}, distance 1"
                    )
            # shipyardの条件がないと spawn した船と衝突してしまう
            elif not is_opposite_ship_or_shipyard(target_point_index) and not (
                target_cell.shipyard and target_cell.shipyard.player_id == me.id
            ):
                mcf.add_edge(
                    ship_node_id,
                    ship_num + target_point_index,
                    1,
                    calc_cost(BASE_COST / 2),
                )
                if LOG_FLOW_ADD_EDGE:
                    log(
                        f"flow2 add edge: {ship.position} => {target_point}, cost {BASE_COST/2}, distance 1"
                    )
            elif (
                target_point_index in index_to_ship_max_cargo
                and index_to_ship_max_cargo[target_point_index][1] == 1
                and not (
                    target_cell.shipyard and target_cell.shipyard.player_id == me.id
                )
            ):
                mcf.add_edge(
                    ship_node_id,
                    ship_num + target_point_index,
                    1,
                    calc_cost(BASE_COST / 2 + 1),
                )
                if LOG_FLOW_ADD_EDGE:
                    log(
                        f"flow2 add edge: {ship.position} => {target_point}, cost {BASE_COST/2+1}, distance 1"
                    )
            else:
                mcf.add_edge(
                    ship_node_id, ship_num + target_point_index, 1, calc_cost(BASE_COST)
                )
                if LOG_FLOW_ADD_EDGE:
                    log(
                        f"flow2 add edge: {ship.position} => {target_point}, cost {BASE_COST}, distance 1"
                    )

        # 居座るかどうか
        if check_max_cargo_condition(ship.halite, ship_index) == SHIP_WIN:
            if ATTACK_START_STEP <= obs.step < ATTACK_END_STEP:
                if (
                    is_minable_halite_area_in_attack_mode(
                        ship.position.x, ship.position.y
                    )
                    or index_to_cell[ship_index].shipyard
                ):
                    cost = (
                        distance_to_my_nearest_shipyard[ship.position.x][
                            ship.position.y
                        ]
                        if distance_to_my_nearest_shipyard[ship.position.x][
                            ship.position.y
                        ]
                        != INF
                        else 0
                    )
                    mcf.add_edge(
                        ship_node_id, ship_num + ship_index, 1, calc_cost(cost)
                    )
                    if LOG_FLOW_ADD_EDGE:
                        log(
                            f"flow2 add edge: {ship.position} => {ship.position}, cost {cost}, distance 0"
                        )
                else:
                    mcf.add_edge(
                        ship_node_id, ship_num + ship_index, 1, calc_cost(BASE_COST / 3)
                    )
                    if LOG_FLOW_ADD_EDGE:
                        log(
                            f"flow2 add edge: {ship.position} => {ship.position}, cost {BASE_COST/3}, distance 0"
                        )
            else:
                cost = (
                    distance_to_my_nearest_shipyard[ship.position.x][ship.position.y]
                    if distance_to_my_nearest_shipyard[ship.position.x][ship.position.y]
                    != INF
                    else 0
                )
                mcf.add_edge(ship_node_id, ship_num + ship_index, 1, calc_cost(cost))
                if LOG_FLOW_ADD_EDGE:
                    log(
                        f"flow2 add edge: {ship.position} => {ship.position}, cost {cost}, distance 0"
                    )
        # 拠点守る
        # 相手が隣にいるけど、shipyardに居る
        elif (
            check_max_cargo_condition(ship.halite, ship_index) == SHIP_DRAW
            and index_to_cell[ship_index].shipyard
        ):
            mcf.add_edge(ship_node_id, ship_num + ship_index, 1, calc_cost(0))
            if LOG_FLOW_ADD_EDGE:
                log(
                    f"flow2 add edge: {ship.position} => {ship.position}, cost 0, distance 0"
                )
        # 相手が隣にいたら評価を悪く
        else:
            mcf.add_edge(ship_node_id, ship_num + ship_index, 1, calc_cost(BASE_COST))
            if LOG_FLOW_ADD_EDGE:
                log(
                    f"flow2 add edge: {ship.position} => {ship.position}, cost {BASE_COST}, distance 0"
                )

        # 船にターゲットが指定されていれば
        if ship_node_id in mcf_node_id_to_target_index_and_score:
            target_index, score = mcf_node_id_to_target_index_and_score[ship_node_id]
            score -= BASE_COST + 1  # expected_halite
            target_point = Point.from_index(target_index, BOARD_SIZE)
            # target から逆算
            index_to_info = bfs(ship, target_point)

            for next_action in ACTIONS:
                if next_action:
                    next_point = ship.position.translate(
                        next_action.to_point(), BOARD_SIZE
                    )
                else:
                    next_point = ship.position
                    if (
                        ATTACK_START_STEP <= obs.step < ATTACK_END_STEP
                        and not is_minable_halite_area_in_attack_mode(
                            next_point.x, next_point.y
                        )
                    ):
                        continue

                next_point_index = next_point.to_index(BOARD_SIZE)

                if next_point_index in index_to_info:
                    distance, _, _ = index_to_info[next_point_index]
                    log(f"{ship.position} {next_point} {distance}")
                    if (
                        check_max_cargo_condition(ship.halite, next_point_index)
                        == SHIP_WIN
                    ):
                        # TODO: refine
                        # halite_per_turn などを考慮して厳密に

                        mcf.add_edge(
                            ship_node_id,
                            ship_num + next_point_index,
                            1,
                            calc_cost(score / (distance + 1)),
                        )
                        if LOG_FLOW_ADD_EDGE:
                            log(
                                f"flow2 add edge: {ship.position} => {next_point}, cost {score / (distance+1)}, distance {distance}"
                            )

    if log_time("flow2 adding edge end"):
        return me.next_actions

    log(f"flow2 edge num: {mcf.edge_num}")
    if (
        mcf.flow(start_mcf_node_id, goal_mcf_node_id, ship_num, start_time, EXIT_TIME)
        == MinCostFlow.ABORT
    ):
        log_time("abort flow2")
        return me.next_actions

    if log_time("flow2 end"):
        return me.next_actions

    mcf_node_id_to_target_index_and_score = {}
    for ship_index in range(len(mcf.G)):
        edges = mcf.G[ship_index]
        for to, cap, cost, _ in edges:
            if (
                ship_index < ship_num
                and ship_num <= to < mcf_node_num - 2
                and cap < MinCostFlow.EPS
            ):
                next_index = to - ship_num
                ship_id = mcf_id_to_ship_id[ship_index]
                ship = board.ships[ship_id]
                mcf_node_id_to_target_index_and_score[ship_index] = (next_index, cost)
                next_point = Point.from_index(next_index, BOARD_SIZE)
                log(
                    f"flow2: id {ship_id}, halite {ship.halite}, {ship.position} => next {next_point}, cost {cost}"
                )

    for ship_index, (
        next_position_id,
        score,
    ) in mcf_node_id_to_target_index_and_score.items():
        ship_id = mcf_id_to_ship_id[ship_index]
        ship = board.ships[ship_id]

        next_position = Point.from_index(next_position_id, BOARD_SIZE)

        # TODO: refine
        if score > BASE_COST + 1:
            _success, path = bfs_one_target(ship, ship.position, next_position, False)
        else:
            _success, path = bfs_one_target(ship, ship.position, next_position)

        if len(path) == 0:
            next_action = None
        else:
            next_action = path[0][1]
        ship.next_action = next_action

    for ship in me.ships:
        if ship.next_action:
            next_position = ship.position.translate(
                ship.next_action.to_point(), BOARD_SIZE
            )
        else:
            next_position = ship.position

        update_index_to_ship_max_cargo(next_position.to_index(BOARD_SIZE), ship.halite)

    """
    Convert ship to shipyard
    """
    if ship_convert:
        v = []
        log(f"convert: ")
        for ship in me.ships:
            # 拠点があればスキップ
            if index_to_cell[ship.position.to_index(BOARD_SIZE)].shipyard:
                continue

            if (
                20 <= obs.step
                and distance_to_my_nearest_ship[ship.position.x][ship.position.y] != INF
                and distance_to_opponent_nearest_ship[ship.position.x][ship.position.y]
                <= distance_to_my_nearest_ship[ship.position.x][ship.position.y]
            ):
                continue

            log(f"  candidate: {ship.id} {ship.position} halite {ship.halite}")
            if ship.halite + tmp_halite >= 500:
                if (
                    len(
                        distances_to_my_nearest_shipyard[ship.position.x][
                            ship.position.y
                        ]
                    )
                    == 0
                ):
                    v.append((cell_score[ship.position.x][ship.position.y], ship.id))
                elif (
                    len(
                        distances_to_my_nearest_shipyard[ship.position.x][
                            ship.position.y
                        ]
                    )
                    == 1
                ):
                    if (
                        6
                        <= distances_to_my_nearest_shipyard[ship.position.x][
                            ship.position.y
                        ][0]
                        <= 7
                    ):
                        v.append(
                            (cell_score[ship.position.x][ship.position.y], ship.id)
                        )
                else:
                    if (
                        6
                        <= distances_to_my_nearest_shipyard[ship.position.x][
                            ship.position.y
                        ][0]
                        <= 7
                        and 6
                        <= distances_to_my_nearest_shipyard[ship.position.x][
                            ship.position.y
                        ][1]
                        <= 7
                    ):
                        v.append(
                            (cell_score[ship.position.x][ship.position.y], ship.id)
                        )

        v.sort()
        v.reverse()
        log(f"v {v}")

        if len(v):
            distance, ship_id = v[0]
            if 4 <= distance:
                ship = board.ships[ship_id]
                tmp_halite -= 500 - ship.halite
                ship.next_action = ShipAction.CONVERT

    log(str(me.next_actions))

    log(f"time: {time.time() - start_time}")
    log("")

    return me.next_actions
