import copy
import random
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


# =====================================================
# Environment
# =====================================================

SIZE = 50

EMPTY = 0
WALL = 1
MAIN_A = 2
MAIN_B = 3
WHITE = 4


class TwoRoomWorld:

    def __init__(self):
        self.grid = np.zeros((SIZE, SIZE), dtype=np.int32)
        self.logs = []
        self.build_world()
        self.reset()

    # -------------------------------------------------
    # 2部屋 + トンネル
    # -------------------------------------------------

    def build_world(self):
        self.grid[:] = EMPTY

        # 外壁
        self.grid[0, :] = WALL
        self.grid[-1, :] = WALL
        self.grid[:, 0] = WALL
        self.grid[:, -1] = WALL

        # 中央壁
        for y in range(1, SIZE - 1):
            if not (22 <= y <= 27):
                self.grid[y, 25] = WALL

    # -------------------------------------------------
    # 初期化
    # -------------------------------------------------

    def reset(self):
        self.A = [10, 10]
        self.B = [40, 40]
        self.WHITE = [25, 25]

        # 0 Main
        # 1 White
        self.mode = 0
        self.done = False
        self.step_count = 0
        self.logs = []

        return self.observe()

    # -------------------------------------------------
    # 状態取得
    # -------------------------------------------------

    def observe(self):
        return np.array(
            [
                self.A[0] / SIZE,
                self.A[1] / SIZE,
                self.B[0] / SIZE,
                self.B[1] / SIZE,
                self.WHITE[0] / SIZE,
                self.WHITE[1] / SIZE,
                self.mode,
            ],
            dtype=float,
        )

    # -------------------------------------------------
    # 移動
    # -------------------------------------------------

    def move(self, obj, dx, dy):
        nx = obj[0] + dx
        ny = obj[1] + dy

        if self.grid[ny, nx] != WALL:
            obj[0] = nx
            obj[1] = ny

    # -------------------------------------------------
    # Action
    # -------------------------------------------------

    def step(self, action):
        before = self.observe()

        if action == 4:
            self.mode = 1 - self.mode
        else:
            moves = {
                0: (0, -1),
                1: (0, 1),
                2: (-1, 0),
                3: (1, 0),
                5: (0, -1),
                6: (0, 1),
                7: (-1, 0),
                8: (1, 0),
            }

            if action in moves:
                dx, dy = moves[action]

                # Main操作
                if self.mode == 0:
                    self.move(self.A, dx, dy)
                    # AとBは逆
                    self.move(self.B, -dx, -dy)

                # White操作
                else:
                    if action >= 5:
                        self.move(self.WHITE, dx, dy)

        after = self.observe()
        self.step_count += 1

        # 重なり判定
        if self.A == self.B:
            self.done = True

        # ログ
        self.logs.append(
            {
                "step": self.step_count,
                "state": before,
                "action": action,
                "next_state": after,
                "A": self.A.copy(),
                "B": self.B.copy(),
                "WHITE": self.WHITE.copy(),
                "mode": self.mode,
                "done": self.done,
            }
        )

        return after

    # -------------------------------------------------
    # 描画
    # -------------------------------------------------

    def render(self):
        img = self.grid.copy()
        img[self.A[1], self.A[0]] = MAIN_A
        img[self.B[1], self.B[0]] = MAIN_B
        img[self.WHITE[1], self.WHITE[0]] = WHITE

        plt.figure(figsize=(6, 6))
        plt.imshow(img, cmap="gray_r")
        plt.title(f"step={self.step_count} mode={self.mode}")
        plt.grid()
        plt.show()


# =====================================================
# Hippocampus
# 経験記憶 (Action付きMemoryCell)
# =====================================================


class MemoryCell:

    def __init__(self, cell_id, state):
        self.id = cell_id
        self.state = np.array(state, dtype=float)
        
        # 構造変更: actionごとの遷移先管理 {action: {next_id: strength}}
        self.links = {}

        self.visits = 0
        self.energy = 1.0

    def similarity(self, state):
        d = np.linalg.norm(self.state - state)
        return 1 / (d + 0.001)


class Hippocampus:

    def __init__(self):
        self.cells = []
        self.max_id = 0

    def encode(self, state):
        best = None
        score = 0

        for c in self.cells:
            s = c.similarity(state)
            if s > score:
                score = s
                best = c

        # 近い記憶がある
        if score > 10:
            best.visits += 1
            best.energy = min(1.5, best.energy + 0.05)
            return best

        # 新規記憶
        cell = MemoryCell(self.max_id, state)
        self.max_id += 1
        self.cells.append(cell)
        return cell

    def recall(self, state):
        if len(self.cells) == 0:
            return None

        scores = [c.similarity(state) for c in self.cells]
        return self.cells[np.argmax(scores)]


# =====================================================
# Cortex
# Action条件付き因果遷移モデル
# =====================================================


class Cortex:

    def __init__(self, hippocampus):
        self.hippo = hippocampus

    # -----------------------------------------
    # 経験から因果関係を学習
    # -----------------------------------------
    def learn_transition(self, s1, action, s2):
        c1 = self.hippo.encode(s1)
        c2 = self.hippo.encode(s2)

        if action not in c1.links:
            c1.links[action] = {}

        if c2.id not in c1.links[action]:
            c1.links[action][c2.id] = 0.1
        else:
            # Hebbian的強化
            c1.links[action][c2.id] += 0.05

    # -----------------------------------------
    # アクションを考慮した次状態予測
    # -----------------------------------------
    def predict_next(self, state, action):
        cell = self.hippo.recall(state)
        if cell is None:
            return None

        if action not in cell.links or len(cell.links[action]) == 0:
            return None

        candidates = list(cell.links[action].keys())
        weights = np.array([cell.links[action][x] for x in candidates])
        weights /= weights.sum()

        next_id = np.random.choice(candidates, p=weights)
        target = [c for c in self.hippo.cells if c.id == next_id][0]
        return target.state


# =====================================================
# Planner
# 内部シミュレーション (Action指定対応)
# =====================================================


class Planner:

    def __init__(self, cortex):
        self.cortex = cortex

    def imagine(self, state, steps=10):
        trajectory = [state]
        current = state

        for i in range(steps):
            # 内部シミュレーションではランダムまたは目的別のアクションを選択
            action = np.random.randint(0, 9)
            nxt = self.cortex.predict_next(current, action)
            if nxt is None:
                break
            trajectory.append(nxt)
            current = nxt

        return trajectory

    def simulate(self, state, trials=5, steps=10):
        worlds = []
        for i in range(trials):
            worlds.append(self.imagine(state, steps))
        return worlds


# =====================================================
# 解析・可視化関数群
# =====================================================


def collect_experience(env, cortex, episodes=50, steps=200):
    for ep in range(episodes):
        env.reset()
        for t in range(steps):
            state = env.observe()
            action = np.random.randint(0, 9)
            env.step(action)
            next_state = env.observe()

            cortex.learn_transition(state, action, next_state)

            if env.done:
                break


def plot_memory_graph(hippocampus, max_nodes=200):
    G = nx.DiGraph()
    cells = hippocampus.cells[:max_nodes]

    for c in cells:
        G.add_node(c.id, energy=c.energy)

    for c in cells:
        for action, sublinks in c.links.items():
            for nxt, w in sublinks.items():
                if nxt < max_nodes:
                    G.add_edge(c.id, nxt, weight=w, action=action)

    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)
    energy = [G.nodes[n]["energy"] for n in G.nodes]

    nx.draw_networkx_nodes(G, pos, node_size=40, node_color=energy)
    nx.draw_networkx_edges(G, pos, alpha=0.2)
    plt.title("Hippocampus Memory Graph (Action-Conditioned)")
    plt.axis("off")
    plt.show()


def plot_transition_network(hippocampus, threshold=0.15):
    G = nx.DiGraph()

    for c in hippocampus.cells:
        G.add_node(c.id)
        for action, sublinks in c.links.items():
            for nxt, w in sublinks.items():
                if w > threshold:
                    G.add_edge(c.id, nxt, weight=w, action=action)

    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=1)
    nx.draw(G, pos, node_size=20, arrows=True, alpha=0.5)
    plt.title("Action-Conditioned State Transition Network")
    plt.show()


def plot_memory_activation(hippocampus):
    visits = np.array([c.visits for c in hippocampus.cells])
    energy = np.array([c.energy for c in hippocampus.cells])

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].hist(visits, bins=40)
    ax[0].set_title("Memory Visits")

    ax[1].hist(energy, bins=40)
    ax[1].set_title("Memory Energy")
    plt.show()


def analyze_trajectory(trajectory):
    traj = np.array(trajectory)

    plt.figure(figsize=(6, 6))
    plt.plot(traj[:, 0], traj[:, 1], marker="o")
    plt.gca().invert_yaxis()
    plt.title("Imagined Future Trajectory")
    plt.grid()
    plt.show()

    distance = np.sum(
        np.linalg.norm(np.diff(traj[:, :2], axis=0), axis=1)
    )
    print("Trajectory length:", distance)


def check_goal_path(trajectory, goal_distance=0.05):
    for i, state in enumerate(trajectory):
        A = np.array([state[0], state[1]])
        B = np.array([state[2], state[3]])
        d = np.linalg.norm(A - B)

        if d < goal_distance:
            print("GOAL predicted step:", i)
            return True

    print("No goal in simulation")
    return False


def run_analysis(env, hippocampus, cortex, planner):
    print("Memory cells:", len(hippocampus.cells))

    plot_memory_graph(hippocampus)
    plot_transition_network(hippocampus)
    plot_memory_activation(hippocampus)

    state = env.observe()
    futures = planner.simulate(state, trials=5, steps=30)

    for f in futures:
        analyze_trajectory(f)
        check_goal_path(f)


# =====================================================
# 実行部分
# =====================================================

if __name__ == "__main__":
    env = TwoRoomWorld()
    hippo = Hippocampus()
    cortex = Cortex(hippo)
    planner = Planner(cortex)

    # 経験収集
    collect_experience(env, cortex, episodes=200, steps=100)

    # 解析
    run_analysis(env, hippo, cortex, planner)
