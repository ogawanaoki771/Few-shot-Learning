import random
import copy
import cv2
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.interpolate import splprep, splev
from shapely.geometry import LineString
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
# =========================================================
# 🖼️ 1. 基本設定
# =========================================================
IMG_SIZE = 50
SEED = 2026
rng = np.random.default_rng(SEED)
# =========================================================
# 🎨 2. 訓練/テストでファミリーを完全に分離した多様な図形生成器
# =========================================================
def generate_diverse_shapes(num_samples, img_size, seed, family_type="train"):
    np.random.seed(seed)
    random.seed(seed)
    
    X_coords = [] 
    y = []
    
    for i in range(num_samples):
        label = i % 2  # 0: Closed, 1: Open
        is_closed = (label == 0)
        
        valid_curve_found = False
        attempts = 0
        
        while not valid_curve_found and attempts < 100:
            attempts += 1
            img = np.zeros((img_size, img_size), dtype=np.uint8)
            cx = random.randint(15, img_size - 15)
            cy = random.randint(15, img_size - 15)
            
            pts_list = None
            if family_type == "train":
                shape_type = (i + random.randint(0, 2)) % 5  # 0～4
            else:
                shape_type = (i % 5) + 5                   # 5～9 (テスト用 5ファミリー)
            
            # ---------------------------------------------------------
            # 閉曲線 (Closed: label == 0)
            # ---------------------------------------------------------
            if is_closed:
                if shape_type == 0:
                    a, b = random.uniform(8, 16), random.uniform(6, 14)
                    angle = random.uniform(0, 2 * np.pi)
                    n_pts = random.randint(40, 80)
                    thetas = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
                    x_raw = a * np.cos(thetas) + np.random.normal(0, 0.6, n_pts)
                    y_raw = b * np.sin(thetas) + np.random.normal(0, 0.6, n_pts)
                    x_rot = x_raw * np.cos(angle) - y_raw * np.sin(angle) + cx
                    y_rot = x_raw * np.sin(angle) + y_raw * np.cos(angle) + cy
                    x_pts = np.append(x_rot, x_rot[0])
                    y_pts = np.append(y_rot, y_rot[0])
                    tck, _ = splprep([x_pts, y_pts], s=random.uniform(0.1, 1.0), per=True, k=3)
                    x_curve, y_curve = splev(np.linspace(0, 1, 120), tck)
                    pts_list = np.vstack((x_curve, y_curve)).T
                    
                elif shape_type == 1:
                    n_v = random.randint(4, 9)
                    thetas = np.sort(np.random.uniform(0, 2 * np.pi, n_v))
                    radii = np.random.uniform(6, 16, n_v)
                    x_v = cx + radii * np.cos(thetas) + np.random.normal(0, 0.4, n_v)
                    y_v = cy + radii * np.sin(thetas) + np.random.normal(0, 0.4, n_v)
                    x_pts = np.append(x_v, x_v[0])
                    y_pts = np.append(y_v, y_v[0])
                    tck, _ = splprep([x_pts, y_pts], s=0.5, per=True, k=1)
                    x_curve, y_curve = splev(np.linspace(0, 1, 100), tck)
                    pts_list = np.vstack((x_curve, y_curve)).T
                    
                elif shape_type == 2:
                    num_teeth = random.randint(3, 8)
                    n_pts = num_teeth * 4
                    thetas = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
                    r_outer, r_inner = random.uniform(10, 16), random.uniform(4, 9)
                    radii = np.where(np.arange(n_pts) % 2 == 0, r_outer, r_inner) + np.random.normal(0, 0.3, n_pts)
                    x_v = cx + radii * np.cos(thetas)
                    y_v = cy + radii * np.sin(thetas)
                    pts_list = np.vstack((x_v, y_v)).T
                    
                elif shape_type == 3:
                    n_ctrl = random.randint(5, 10)
                    thetas = np.linspace(0, 2 * np.pi, n_ctrl, endpoint=False)
                    radii = np.random.uniform(7, 15, n_ctrl)
                    x_ctrl = cx + radii * np.cos(thetas)
                    y_ctrl = cy + radii * np.sin(thetas)
                    x_pts = np.append(x_ctrl, x_ctrl[0])
                    y_pts = np.append(y_ctrl, y_ctrl[0])
                    tck, _ = splprep([x_pts, y_pts], s=1.0, per=True, k=3)
                    x_curve, y_curve = splev(np.linspace(0, 1, 140), tck)
                    pts_list = np.vstack((x_curve, y_curve)).T
                    
                elif shape_type == 4:
                    n_waves = random.randint(3, 6)
                    thetas = np.linspace(0, 2 * np.pi, 80, endpoint=False)
                    r0, amp = random.uniform(8, 12), random.uniform(2, 5)
                    radii = r0 + amp * np.sin(n_waves * thetas) + np.random.normal(0, 0.2, 80)
                    x_ctrl = cx + radii * np.cos(thetas)
                    y_ctrl = cy + radii * np.sin(thetas)
                    x_pts = np.append(x_ctrl, x_ctrl[0])
                    y_pts = np.append(y_ctrl, y_ctrl[0])
                    tck, _ = splprep([x_pts, y_pts], s=0.2, per=True, k=3)
                    x_curve, y_curve = splev(np.linspace(0, 1, 140), tck)
                    pts_list = np.vstack((x_curve, y_curve)).T
                    
                elif shape_type == 5:
                    thetas = np.linspace(0, 2 * np.pi, 90, endpoint=False)
                    radii = 11 + 4 * np.sin(random.randint(4, 7) * thetas) * np.cos(2 * thetas)
                    x_v = cx + radii * np.cos(thetas) + np.random.normal(0, 0.5, 90)
                    y_v = cy + radii * np.sin(thetas) + np.random.normal(0, 0.5, 90)
                    pts_list = np.vstack((x_v, y_v)).T
                    
                elif shape_type == 6:
                    thetas = np.linspace(0, 2 * np.pi, 80, endpoint=False)
                    a_card = random.uniform(6, 10)
                    radii = a_card * (1 - np.cos(thetas)) + random.uniform(0, 3)
                    x_v = cx + radii * np.cos(thetas)
                    y_v = cy + radii * np.sin(thetas)
                    pts_list = np.vstack((x_v, y_v)).T
                    
                elif shape_type == 7:
                    t = np.linspace(0, 2 * np.pi, 80)
                    scale = random.uniform(10, 15)
                    denom = 1.0 + np.sin(t)**2
                    x_raw = scale * np.cos(t) / denom
                    y_raw = scale * np.sin(t) * np.cos(t) / denom
                    x_v = cx + x_raw + np.random.normal(0, 0.3, 80)
                    y_v = cy + y_raw + np.random.normal(0, 0.3, 80)
                    pts_list = np.vstack((x_v, y_v)).T
                    
                elif shape_type == 8:
                    t = np.linspace(0, 2 * np.pi, 80, endpoint=False)
                    radii = 12 / (np.abs(np.cos(t))**3 + np.abs(np.sin(t))**3)**(1/3)
                    radii = np.clip(radii, 5, 16) + np.random.normal(0, 0.3, 80)
                    x_v = cx + radii * np.cos(t)
                    y_v = cy + radii * np.sin(t)
                    pts_list = np.vstack((x_v, y_v)).T
                    
                else:
                    t = np.linspace(0.5, 2 * np.pi, 80)
                    radii = 4 + 2 * t
                    x_v = cx + radii * np.cos(t * 2)
                    y_v = cy + radii * np.sin(t * 2)
                    pts_list = np.vstack((x_v, y_v)).T
            # ---------------------------------------------------------
            # 開曲線 (Open: label == 1)
            # ---------------------------------------------------------
            else:
                if shape_type == 0:
                    turns = random.uniform(1.2, 2.5)
                    theta_max = turns * 2 * np.pi
                    n_pts = 60
                    thetas = np.linspace(0.2, theta_max, n_pts)
                    r0, b = random.uniform(2, 5), random.uniform(1.0, 2.0)
                    radii = r0 + b * thetas + np.random.normal(0, 0.3, n_pts)
                    x_ctrl = cx + radii * np.cos(thetas)
                    y_ctrl = cy + radii * np.sin(thetas)
                    tck, _ = splprep([x_ctrl, y_ctrl], s=0.5, per=False, k=3)
                    x_curve, y_curve = splev(np.linspace(0, 1, 100), tck)
                    pts_list = np.vstack((x_curve, y_curve)).T
                    
                elif shape_type == 1:
                    span = random.uniform(np.pi, 1.8 * np.pi)
                    start_a = random.uniform(0, 2 * np.pi)
                    n_pts = 50
                    thetas = np.linspace(start_a, start_a + span, n_pts)
                    r = random.uniform(8, 15)
                    x_v = cx + r * np.cos(thetas) + np.random.normal(0, 0.3, n_pts)
                    y_v = cy + r * np.sin(thetas) + np.random.normal(0, 0.3, n_pts)
                    tck, _ = splprep([x_v, y_v], s=0.2, per=False, k=3)
                    x_curve, y_curve = splev(np.linspace(0, 1, 90), tck)
                    pts_list = np.vstack((x_curve, y_curve)).T
                    
                elif shape_type == 2:
                    n_pts = 50
                    t = np.linspace(-2.0, 2.0, n_pts)
                    x_raw = t * random.uniform(5, 9)
                    y_raw = np.sin(t * random.uniform(1.0, 2.5)) * random.uniform(8, 14)
                    angle = random.uniform(0, 2 * np.pi)
                    x_v = x_raw * np.cos(angle) - y_raw * np.sin(angle) + cx + np.random.normal(0, 0.3, n_pts)
                    y_v = x_raw * np.sin(angle) + y_raw * np.cos(angle) + cy + np.random.normal(0, 0.3, n_pts)
                    tck, _ = splprep([x_v, y_v], s=0.5, per=False, k=3)
                    x_curve, y_curve = splev(np.linspace(0, 1, 90), tck)
                    pts_list = np.vstack((x_curve, y_curve)).T
                    
                elif shape_type == 3:
                    n_pts = random.randint(4, 8)
                    x_ctrl = np.linspace(cx - 14, cx + 14, n_pts) + np.random.normal(0, 2.0, n_pts)
                    y_ctrl = cy + np.random.uniform(-12, 12, n_pts)
                    tck, _ = splprep([x_ctrl, y_ctrl], s=1.0, per=False, k=2)
                    x_curve, y_curve = splev(np.linspace(0, 1, 90), tck)
                    pts_list = np.vstack((x_curve, y_curve)).T
                    
                elif shape_type == 4:
                    n_pts = random.randint(6, 12)
                    x_v = np.random.uniform(cx - 15, cx + 15, n_pts)
                    y_v = np.random.uniform(cy - 15, cy + 15, n_pts)
                    tck, _ = splprep([x_v, y_v], s=1.5, per=False, k=3)
                    x_curve, y_curve = splev(np.linspace(0, 1, 100), tck)
                    pts_list = np.vstack((x_curve, y_curve)).T
                    
                elif shape_type == 5:
                    n_pts = 50
                    t = np.linspace(-np.pi, np.pi, n_pts)
                    x_raw = t * 8
                    y_raw = 10 * np.cos(t) + 4 * np.sin(3 * t)
                    angle = random.uniform(0, np.pi)
                    x_v = x_raw * np.cos(angle) - y_raw * np.sin(angle) + cx
                    y_v = x_raw * np.sin(angle) + y_raw * np.cos(angle) + cy
                    pts_list = np.vstack((x_v, y_v)).T
                    
                elif shape_type == 6:
                    t = np.linspace(-3, 3, 50)
                    x_raw = t * 5
                    y_raw = (t**2) * random.uniform(1.5, 3.0) - 10
                    angle = random.uniform(0, 2 * np.pi)
                    x_v = x_raw * np.cos(angle) - y_raw * np.sin(angle) + cx
                    y_v = x_raw * np.sin(angle) + y_raw * np.cos(angle) + cy
                    pts_list = np.vstack((x_v, y_v)).T
                    
                elif shape_type == 7:
                    n_pts = 7
                    x_ctrl = np.linspace(cx - 12, cx + 12, n_pts)
                    y_ctrl = cy + np.array([0, 10, -10, 12, -12, 8, 0]) * random.uniform(0.8, 1.2)
                    tck, _ = splprep([x_ctrl, y_ctrl], s=0.0, per=False, k=1)
                    x_curve, y_curve = splev(np.linspace(0, 1, 80), tck)
                    pts_list = np.vstack((x_curve, y_curve)).T
                    
                elif shape_type == 8:
                    t = np.linspace(0.5, 3.5, 50)
                    r_c = random.uniform(3, 6)
                    x_raw = r_c * (t - np.sin(t)) * 3 - 15
                    y_raw = r_c * (1 - np.cos(t)) * 3 - 5
                    x_v = cx + x_raw + np.random.normal(0, 0.2, 50)
                    y_v = cy + y_raw + np.random.normal(0, 0.2, 50)
                    pts_list = np.vstack((x_v, y_v)).T
                    
                else:
                    t = np.linspace(0, 2*np.pi, 60)
                    x_raw = t * 6 - 18
                    y_raw = 8 * np.sin(t) * np.cos(t * 0.5)
                    x_v = cx + x_raw
                    y_v = cy + y_raw
                    pts_list = np.vstack((x_v, y_v)).T
                    
            if pts_list is not None and len(pts_list) > 2:
                jitter = np.random.normal(0, 0.35, pts_list.shape)
                pts_list += jitter
                line = LineString(pts_list)
                if line.is_simple:
                    valid_curve_found = True
                    pts_cv = pts_list.astype(np.int32).reshape((-1, 1, 2))
                    thickness = random.choice([1, 1, 2])
                    cv2.polylines(img, [pts_cv], isClosed=is_closed, color=255, thickness=thickness)
                    
        py_indices, px_indices = np.where(img > 0)
        current_sample_nodes = [(float(px + 0.5), float(py + 0.5)) for px, py in zip(px_indices, py_indices)]
        
        if len(current_sample_nodes) > 0:
            X_coords.append(current_sample_nodes)
            y.append(label)
            
    return X_coords, np.array(y)
# =========================================================
# 🦠 3. 線構造適応細胞クラス
# =========================================================
class LineAdaptiveCell:
    def __init__(self, cell_id):
        self.id = cell_id
        self.anchor_x = rng.uniform(5, 45)
        self.anchor_y = rng.uniform(5, 45)
        self.geom_sensitivity = rng.uniform(1.0, 5.0)
        
        self.g_relational_bias = rng.uniform(0.0, 1.0)
        self.g_attention_bias  = rng.uniform(0.0, 1.0)
        self.g_resonance_gain = rng.uniform(0.5, 3.5)
        self.g_synaptic_delay = rng.choice([1, 2, 3])
        self.g_learning_rate = rng.uniform(0.05, 0.35)
        self.g_mutation_rate = rng.uniform(0.05, 0.25)
        self.energy = rng.uniform(0.85, 1.0)
        self.age = 0
        self.target_label = rng.choice([0, 1]) 
        self.links = {}  
        self.reset_dynamic_state()
        
    def reset_dynamic_state(self):
        self.activation_history = [0.0] * 5
        self.last_activation = 0.0
        
    def access(self, coordinate_points, id_to_cell, enable_links=True):
        if len(coordinate_points) == 0:
            return 0.0
            
        pts = np.array(coordinate_points) 
        dists = np.hypot(pts[:, 0] - self.anchor_x, pts[:, 1] - self.anchor_y)
        
        near_line_count = np.sum(dists < 7.0)
        visual_response = np.tanh(near_line_count / (self.geom_sensitivity * 10.0 + 1e-9))
        
        if self.g_attention_bias > 0.4 and near_line_count > 0:
            near_pts = pts[dists < 7.0]
            line_mean = np.mean(near_pts, axis=0)
            self.anchor_x = self.anchor_x * 0.90 + line_mean[0] * 0.10
            self.anchor_y = self.anchor_y * 0.90 + line_mean[1] * 0.10
            
        relational_response = 0.0
        if enable_links:
            neighbors = [id_to_cell[nid] for nid in self.links.keys() if nid in id_to_cell]
            weights = [self.links[nid] for nid in self.links.keys() if nid in id_to_cell]
            if neighbors and len(weights) > 0:
                neighbor_delayed_acts = []
                for n in neighbors:
                    delay_idx = min(self.g_synaptic_delay, len(n.activation_history) - 1)
                    neighbor_delayed_acts.append(n.activation_history[-delay_idx])
                relational_response = np.sum(np.array(neighbor_delayed_acts) * np.array(weights)) / (np.sum(weights) + 1e-9)
            
        g_r = self.g_relational_bias if enable_links else 0.0
        final_score = (1.0 - g_r) * visual_response + g_r * relational_response
        return float(np.clip(final_score, 0.0, 1.0))
        
    def update_history(self, act):
        self.activation_history.append(act)
        if len(self.activation_history) > 5:
            self.activation_history.pop(0)
        self.last_activation = act
        
    def adapt(self, coordinate_points, act_strength, current_label=None):
        self.age += 1
        if len(coordinate_points) == 0:
            return
            
        if current_label is None or current_label == self.target_label:
            pts = np.array(coordinate_points)
            dists = np.hypot(pts[:, 0] - self.anchor_x, pts[:, 1] - self.anchor_y)
            if np.min(dists) < 16.0:
                closest_pt = pts[np.argmin(dists)]
                lr = self.g_learning_rate * max(act_strength, 0.2) * (0.8 if current_label is None else 1.0)
                self.anchor_x += lr * (closest_pt[0] - self.anchor_x)
                self.anchor_y += lr * (closest_pt[1] - self.anchor_y)
            if current_label is not None:
                self.energy = min(1.5, self.energy + act_strength * 0.3)
        else:
            self.energy -= 0.05
            
    def divide(self, next_id):
        child = LineAdaptiveCell(next_id)
        child.anchor_x = float(np.clip(self.anchor_x + rng.normal(0, 3.0), 0.0, IMG_SIZE))
        child.anchor_y = float(np.clip(self.anchor_y + rng.normal(0, 3.0), 0.0, IMG_SIZE))
        child.geom_sensitivity = float(max(0.5, self.geom_sensitivity + rng.normal(0, 0.2)))
        
        m_rate = self.g_mutation_rate
        child.g_relational_bias = float(np.clip(self.g_relational_bias + rng.normal(0, m_rate), 0.0, 1.0))
        child.g_attention_bias  = float(np.clip(self.g_attention_bias + rng.normal(0, m_rate), 0.0, 1.0))
        child.g_resonance_gain  = float(np.clip(self.g_resonance_gain + rng.normal(0, m_rate), 0.1, 5.0))
        child.g_synaptic_delay  = rng.choice([1, 2, 3]) if rng.random() < m_rate else self.g_synaptic_delay
        child.g_learning_rate   = float(np.clip(self.g_learning_rate + rng.normal(0, m_rate * 0.2), 0.01, 0.5))
        child.g_mutation_rate   = float(np.clip(self.g_mutation_rate + rng.normal(0, 0.02), 0.01, 0.4))
        child.target_label      = self.target_label if rng.random() > 0.08 else 1 - self.target_label
        self.energy *= 0.50
        return child
# =========================================================
# 🧪 4. 生態系管理クラス（🚀 逐次・非破壊的TTA対応）
# =========================================================
class StochasticMetaEcosystem:
    def __init__(self, init_count=60):
        self.pool = [LineAdaptiveCell(i) for i in range(init_count)]
        self.max_id = init_count
        
    def cycle(self, sample_coords, label, steps=8, is_training=True):
        for c in self.pool:
            c.reset_dynamic_state()
            
        for t in range(steps):
            id_map = {c.id: c for c in self.pool}
            for c in self.pool:
                score = c.access(sample_coords, id_map)
                c.update_history(score)
                
            for i, c1 in enumerate(self.pool):
                for j, c2 in enumerate(self.pool):
                    if i != j and c1.last_activation > 0.2 and c2.last_activation > 0.2:
                        w = c1.links.get(c2.id, 0.1)
                        c1.links[c2.id] = np.clip(w + 0.25 * (c1.last_activation * c2.last_activation), 0.0, 1.0)
                        
        new_cells = []
        for c in self.pool:
            if is_training:
                if c.energy > 0.90 and len(self.pool) < 75:
                    self.max_id += 1
                    new_cells.append(c.divide(self.max_id))
                c.adapt(sample_coords, c.last_activation, label)
                c.energy *= 0.95
            else:
                # テスト時（TTA）はその場のサンプルに対して一時的な適応を行う（ベースを汚さないようコピー上で実行される）
                c.adapt(sample_coords, c.last_activation, current_label=None)
                if c.last_activation > 0.3:
                    c.energy = min(1.2, c.energy + 0.02)
            
        if is_training:
            self.pool.extend(new_cells)
            self.pool = [c for c in self.pool if c.energy > 0.05 or c.age < 4]
            while len(self.pool) < 30:
                self.max_id += 1
                self.pool.append(LineAdaptiveCell(self.max_id))
            
    def extract_features(self, X_coords_data, steps=12, is_training=False):
        features = []
        for sample in X_coords_data:
            if not is_training:
                # 🛠️ 【要望対応】テスト時は初期訓練済みエコシステムをコピーし、
                # その場限りの推論時適応（TTA）を行って特徴抽出する（元モデルは更新されない）
                working_eco = copy.deepcopy(self)
                working_eco.cycle(sample, label=None, steps=steps, is_training=False)
                target_pool = working_eco.pool
            else:
                self.cycle(sample, label=None, steps=steps, is_training=True)
                target_pool = self.pool
            
            act_now = np.array([c.last_activation for c in target_pool])
            act_prev = np.array([c.activation_history[-2] if len(c.activation_history) >= 2 else 0.0 for c in target_pool])
                
            feat_row = [
                np.mean(act_now), np.max(act_now), np.std(act_now),
                np.mean(act_prev), np.max(act_prev), np.std(act_prev)
            ]
            for q in [10, 25, 50, 75, 90]:
                feat_row.append(np.percentile(act_now, q))
                feat_row.append(np.percentile(act_prev, q))
                
            for k_val in [3, 8, 15]:
                k_now = min(k_val, len(act_now))
                k_prev = min(k_val, len(act_prev))
                feat_row.append(np.mean(np.partition(act_now, -k_now)[-k_now:]))
                feat_row.append(np.mean(np.partition(act_prev, -k_prev)[-k_prev:]))
                
            counts_now, _ = np.histogram(act_now, bins=8, range=(0.0, 1.0))
            counts_prev, _ = np.histogram(act_prev, bins=8, range=(0.0, 1.0))
            feat_row.extend(counts_now / (len(act_now) + 1e-9))
            feat_row.extend(counts_prev / (len(act_prev) + 1e-9))
            
            features.append(feat_row)
            
        return np.array(features)
# =========================================================
# 🏃 5. 評価実行関数
# =========================================================
def evaluate_concept_learning(eco, X_tr, y_tr, X_te, y_te):
    X_tr_feats = eco.extract_features(X_tr, steps=8, is_training=True)
    # テストデータに対しては、各サンプルごとにモデルをコピーして逐次的にTTAを実行
    X_te_feats = eco.extract_features(X_te, steps=12, is_training=False)
    
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr_feats)
    X_te_scaled = scaler.transform(X_te_feats)
    
    clf = LogisticRegression(C=0.5, max_iter=2000, random_state=SEED).fit(X_tr_scaled, y_tr)
    y_pred = clf.predict(X_te_scaled)
    
    return {
        "Accuracy": accuracy_score(y_te, y_pred),
        "Precision": precision_score(y_te, y_pred, zero_division=0),
        "Recall": recall_score(y_te, y_pred, zero_division=0),
        "F1": f1_score(y_te, y_pred, zero_division=0)
    }
# =========================================================
# 📊 6. 研究用内部表現ダッシュボードプロット関数
# =========================================================
def plot_bp015_research_dashboard(eco, sample_coords, sample_label, eco_before=None, X_test_coords=None, y_test=None):
    fig = plt.figure(figsize=(13, 11))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    
    # ダッシュボード表示用には、単一サンプルに対する一時的TTAエコシステムを生成する
    demo_eco = copy.deepcopy(eco)
    demo_eco.cycle(sample_coords, label=None, steps=12, is_training=False)
    
    ax1 = plt.subplot(2, 2, 1)
    x_after = [c.anchor_x for c in demo_eco.pool]
    y_after = [c.anchor_y for c in demo_eco.pool]
    colors_after = ["navy" if c.target_label == 0 else "crimson" for c in demo_eco.pool]
    
    if eco_before:
        x_before = [c.anchor_x for c in eco_before.pool]
        y_before = [c.anchor_y for c in eco_before.pool]
        ax1.scatter(x_before, y_before, c="gray", alpha=0.35, s=30, label="Before", zorder=1)
    
    ax1.scatter(x_after, y_after, c=colors_after, s=60, edgecolors="k", alpha=0.85, label="After (TTA)", zorder=2)
    ax1.set_xlim(0, IMG_SIZE)
    ax1.set_ylim(IMG_SIZE, 0)
    ax1.set_aspect("equal")
    ax1.set_title("1. Cell Anchors with TTA (Navy:Closed, Crimson:Open)", fontsize=11, fontweight="bold")
    ax1.legend(loc="upper right")
    ax1.grid(True, linestyle=":", alpha=0.4)
    
    ax2 = plt.subplot(2, 2, 2)
    if len(sample_coords) > 0:
        pts = np.array(sample_coords)
        ax2.scatter(pts[:, 0], pts[:, 1], c="gray", s=15, alpha=0.4, label="Input Shape")
        
    id_map = {c.id: c for c in demo_eco.pool}
    acts = [c.access(sample_coords, id_map) for c in demo_eco.pool]
    
    sc = ax2.scatter(x_after, y_after, c=acts, cmap="plasma", s=70, edgecolors="k", zorder=3)
    plt.colorbar(sc, ax=ax2, label="Activation Value")
    
    label_name = "Closed" if sample_label == 0 else "Open"
    ax2.set_xlim(0, IMG_SIZE)
    ax2.set_ylim(IMG_SIZE, 0)
    ax2.set_aspect("equal")
    ax2.set_title(f"2. Activation Map on Sample ({label_name})", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.4)
    
    ax3 = plt.subplot(2, 2, 3)
    G = nx.Graph()
    pos = {}
    
    for c in demo_eco.pool:
        G.add_node(c.id, label=c.target_label)
        pos[c.id] = (c.anchor_x, -c.anchor_y)
        
    for c in demo_eco.pool:
        for target_id, weight in c.links.items():
            if weight > 0.15 and target_id in G:
                G.add_edge(c.id, target_id, weight=weight)
                
    node_colors = ["navy" if G.nodes[n].get('label', 0) == 0 else "crimson" for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=50, ax=ax3, alpha=0.85)
    nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color="gray", ax=ax3)
    
    ax3.set_title("3. Cell Synaptic Link Network", fontsize=11, fontweight="bold")
    ax3.axis("off")
    
    ax4 = plt.subplot(2, 2, 4)
    if X_test_coords is not None and y_test is not None:
        X_feats = eco.extract_features(X_test_coords, steps=12, is_training=False)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_feats)
        
        for lbl, col, name in zip([0, 1], ["navy", "crimson"], ["Closed", "Open"]):
            mask = (y_test == lbl)
            if np.any(mask):
                ax4.scatter(X_pca[mask, 0], X_pca[mask, 1], c=col, label=name, alpha=0.8, edgecolors="k", s=60)
                
        ax4.set_xlabel("PCA Component 1")
        ax4.set_ylabel("PCA Component 2")
        evr = sum(pca.explained_variance_ratio_) * 100
        ax4.set_title(f"4. Feature Space PCA (Explained Var: {evr:.1f}%)", fontsize=11, fontweight="bold")
        ax4.legend()
        ax4.grid(True, linestyle=":", alpha=0.4)
        
    plt.suptitle("Line Adaptive Ecosystem Internal State Dashboard (Non-destructive TTA)", fontsize=14, fontweight="bold", y=0.98)
    plt.show()
# =========================================================# 📊 6.5 画素数 vs ロジスティック分類器出力のプロット関数# =========================================================
def plot_pixel_vs_logistic_output(eco, X_te, y_te, X_tr, y_tr):
    # 特徴量抽出とスケーリング
    X_tr_feats = eco.extract_features(X_tr, steps=8, is_training=True)
    X_te_feats = eco.extract_features(X_te, steps=12, is_training=False)
    
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr_feats)
    X_te_scaled = scaler.transform(X_te_feats)
    
    # ロジスティック分類器の学習と予測確率の取得
    clf = LogisticRegression(C=0.5, max_iter=2000, random_state=SEED).fit(X_tr_scaled, y_tr)
    # クラス1（開曲線）である確率を取得
    y_probs = clf.predict_proba(X_te_scaled)[:, 1]
    
    # 各サンプルの画素数を計算（座標リストの長さ）
    pixel_counts = [len(sample) for sample in X_te]
    
    # プロット作成
    plt.figure(figsize=(9, 6))
    
    # 閉曲線 (y_te == 0) : 丸マーカー
    closed_mask = (y_te == 0)
    plt.scatter(
        np.array(pixel_counts)[closed_mask], 
        y_probs[closed_mask], 
        marker='o', color='navy', s=60, alpha=0.8, label='Closed Curve (Circle)'
    )
    
    # 開曲線 (y_te == 1) : バツ印マーカー
    open_mask = (y_te == 1)
    plt.scatter(
        np.array(pixel_counts)[open_mask], 
        y_probs[open_mask], 
        marker='x', color='crimson', s=60, alpha=0.8, label='Open Curve (Cross)'
    )
    
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.6, label='Decision Boundary (0.5)')
    plt.xlabel('Number of Pixels (Pixel Count)', fontsize=12, fontweight='bold')
    plt.ylabel('Logistic Classifier Output (Probability of Open)', fontsize=12, fontweight='bold')
    plt.title('Pixel Count vs. Logistic Classifier Output', fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.ylim(-0.05, 1.05)
    plt.show()
# =========================================================
# 🚀 7. メイン実行ブロック
# =========================================================
if __name__ == "__main__":
    print("=========================================================")
    print("[BP015 Diverse Shape Experiment] Non-destructive TTA Evaluation")
    print("=========================================================")
    
    X_train, y_train = generate_diverse_shapes(num_samples=12, img_size=IMG_SIZE, seed=101, family_type="train")
    X_test, y_test   = generate_diverse_shapes(num_samples=34, img_size=IMG_SIZE, seed=999, family_type="test")
    
    print(f" |- Train Family : {len(X_train)} shapes (Closed: {sum(y_train==0)}, Open: {sum(y_train==1)})")
    print(f" |- Test Family  : {len(X_test)} shapes (Closed: {sum(y_test==0)}, Open: {sum(y_test==1)})\n")
    
    eco = StochasticMetaEcosystem(init_count=200)
    eco_before = copy.deepcopy(eco)
    
    print("Ecosystem Learning Progress on Train Family...")
    for epoch in range(2):
        for idx in range(len(X_train)):
            eco.cycle(X_train[idx], label=y_train[idx], steps=8, is_training=True)
            
    print("Evaluating on Unseen Test Family with NON-DESTRUCTIVE Test-Time Adaptation...")
    metrics = evaluate_concept_learning(eco, X_train, y_train, X_test, y_test)
    print("\n" + "="*60)
    print("BP015 Evaluation Results (Non-destructive TTA on Unseen Families)")
    print("="*60)
    print(f" |- Accuracy  : {metrics['Accuracy'] * 100:.2f} %")
    print(f" |- Precision : {metrics['Precision'] * 100:.2f} %")
    print(f" |- Recall    : {metrics['Recall'] * 100:.2f} %")
    print(f" |- F1 Score  : {metrics['F1'] * 100:.2f} %")
    print("="*60)
    
    print("\nGenerating Internal Representation Dashboard...")
    print("\nGenerating Pixel Count vs. Logistic Output Plot...")
    plot_pixel_vs_logistic_output(eco, X_test, y_test, X_train, y_train)
    plot_bp015_research_dashboard(
        eco=eco, 
        sample_coords=X_test[0], 
        sample_label=y_test[0], 
        eco_before=eco_before, 
        X_test_coords=X_test, 
        y_test=y_test
    )
