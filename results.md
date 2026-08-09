## 実験概要
本実験では、stable_shuffle.pyを用いて未知のデータファミリー（Unseen Families）に対する非破壊的TTA（Test-Time Adaptation）手法の評価を実施しました。自己交差などの不具合を持つ図形をあらかじめ除外しているため、実際の有効実験枚数は生成指定枚数から減少しています。
- IMG_SIZE = 50
- SEED = 2026

### 実験条件
データセット構成
生成時の指定枚数と、不具合のある図形を除外した実際の有効実験枚数は以下の通りです。

Train Family (学習用)

生成指定枚数: 12枚 (seed=101)

有効実験枚数: 12枚（内訳: Closed 6枚、Open 6枚）

Test Family (テスト用)

生成指定枚数: 1,200枚 (seed=999)

有効実験枚数: 1,080枚（内訳: Closed 480枚、Open 600枚）

### 実行設定
使用スクリプト: stable_shuffle.py

実行時間: 約14分（※評価スコアが早期に表示される仕様のため、表示確認時点で手動にて早期終了）

### 評価結果
BP015による評価結果（Non-destructive TTA on Unseen Families）は以下の通りです。

Accuracy (正解率): 75.83%

Precision (適合率): 77.29%

Recall (再現率): 80.00%

F1 Score: 78.62%
