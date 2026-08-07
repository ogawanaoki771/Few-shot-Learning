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
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
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
# 🔍 3. 前処理と可視化の実装
# =========================================================
def resample_curve(pts, num_points=100):
    line = LineString(pts)
    distances = np.linspace(0, line.length, num_points)
    sampled_points = np.array([line.interpolate(d).coords[0] for d in distances])
    return sampled_points
def preprocess_coords(X_coords, num_points=100):
    X_processed = []
    for i, pts in enumerate(X_coords):
        # 最初のサンプルだけ、入力と出力の挙動を可視化する
        if i == 2:
            resampled = resample_curve(pts, num_points=num_points)
            
            plt.figure(figsize=(10, 5))
            
            plt.subplot(121)
            pts_arr = np.array(pts)
            plt.plot(pts_arr[:, 0], pts_arr[:, 1], '-r', linewidth=0.5)
            plt.scatter(pts_arr[:, 0], pts_arr[:, 1], s=5, c='black')
            plt.title("Input to resample_curve\n(np.where order)")
            plt.gca().invert_yaxis()
            plt.axis('equal')
            
            plt.subplot(122)
            plt.plot(resampled[:, 0], resampled[:, 1], '-b', linewidth=1)
            plt.scatter(resampled[:, 0], resampled[:, 1], s=5, c='blue')
            plt.title("Output of resample_curve\n(Interpolated)")
            plt.gca().invert_yaxis()
            plt.axis('equal')
            
            plt.tight_layout()
            plt.show()
        else:
            resampled = resample_curve(pts, num_points=num_points)
        
        # 重心合わせ
        centroid = np.mean(resampled, axis=0)
        centered = resampled - centroid
        
        # スケール正規化
        max_dist = np.max(np.sqrt(np.sum(centered**2, axis=1)))
        if max_dist > 0:
            normalized = centered / max_dist
        else:
            normalized = centered
            
        X_processed.append(normalized.flatten())
        
    return np.array(X_processed)
# =========================================================
# 🚀 4. 実行
# =========================================================
print("--- データ生成中 ---")
train_X_raw, train_y = generate_diverse_shapes(num_samples=12, img_size=IMG_SIZE, seed=SEED, family_type="train")
test_X_raw, test_y = generate_diverse_shapes(num_samples=10000, img_size=IMG_SIZE, seed=SEED+1, family_type="test")
print("--- 前処理および最初のサンプルの可視化実行中 ---")
X_train = preprocess_coords(train_X_raw, num_points=100)
X_test = preprocess_coords(test_X_raw, num_points=100)
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=12, random_state=SEED)),
    ("svm", SVC(kernel="rbf", C=1, gamma="scale", random_state=SEED))
])
print("--- モデル学習中 ---")
pipeline.fit(X_train, train_y)
y_pred = pipeline.predict(X_test)
acc = accuracy_score(test_y, y_pred)
prec = precision_score(test_y, y_pred, zero_division=0)
rec = recall_score(test_y, y_pred, zero_division=0)
f1 = f1_score(test_y, y_pred, zero_division=0)
print("\n================== 評価結果 ==================")
print(f"Accuracy  : {acc:.4f}")
print(f"Precision : {prec:.4f}")
print(f"Recall    : {rec:.4f}")
print(f"F1-score  : {f1:.4f}")
print("==============================================")
