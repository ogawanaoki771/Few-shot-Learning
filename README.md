# Few-shot Learningによる未知図形への幾何学概念学習
## 「ARC3に似たようなゲームを作成しました。まずはデモをご覧ください。」
## デモのリンク：[https://github.com/ogawanaoki771/fewshot-game/blob/main/game/new5.py](https://github.com/ogawanaoki771/Few-shot-Learning/blob/main/X.py)

## 実装の詳細についてはTest-Time Adaptation.mdが参考になると思います。

## 概要

このリポジトリでは、**Few-shot Learning**の設定において、極めて少数の学習例から幾何学的概念を獲得し、未知の図形へ一般化できるかを検証します。

対象タスクは、50×50の二値画像に描かれた単一曲線について、

* 開曲線 (Open Curve)
* 閉曲線 (Closed Curve)

を分類する二値分類問題です。

一般的な深層学習とは異なり、以下の条件を満たすことを目標としています。

* 学習画像は12枚のみ（各クラス6枚）
* 事前学習なし
* 転移学習なし
* 外部データセットなし
* Data Augmentationなし
* 人手設計特徴量なし
* ただし、入力が最初から座標列や点列として与えられる場合は、その表現をそのまま利用することは許容します。
* 座標をランダムにシャッフルし、順序に依存しない[＞＞ /blob/main/stable_shuffle.py ](https://github.com/ogawanaoki771/Few-shot-Learning/blob/main/stable_shuffle.py)
* 具体的には、輪郭追跡、細線化（Skeletonization）、Euler数、Hu Moments、Fourier Descriptor、HOG、SIFTなどを入力特徴として利用する方法は対象外とします。

さらに、学習時には存在しない形状ファミリーのみを用いて評価を行い、未知形状への一般化能力を検証します。

---

# 実験設定

## 学習データとテストデータの可視化

学習データとテストデータの生成コードは
```
dataset.py
```
にあります。
実行すると、
訓練データが12枚、
テストデータが40枚、
生成されて描画されます。

## 学習データ

* Training Images: 12
* Open Curves: 6
* Closed Curves: 6

## テストデータ

* 学習時とは異なる形状ファミリー
* 未知図形のみで評価

---

# 評価方法

以下の指標を用いて評価します。

* Accuracy
* Precision
* Recall
* F1-score

評価はテストデータに対してのみ実施し、学習時にテストデータは使用していません。

---

# データリークについて

本研究では、データリークを避けるために以下を徹底しています。

* テストで使われる形状は学習に一切使用しない
* 学習・評価で形状ファミリーを分離
* 評価時には未知形状のみを使用
* 事前学習・転移学習・外部データ利用なし

もし評価プロトコルに問題やデータリークの可能性がありましたら、ご指摘いただけますと幸いです。

---

# 実行方法

```bash
python demo.py　
```

---

# リポジトリ構成

```
Bongard-FewShot/
│
├── README.md
├── demo.py
├── requirements.txt
├── LICENSE
├── input.py
├── SVM.py
├── Test-Time Adaptation.md
├── stable.py
└── .gitignore
```

### demo.py

提案手法を簡単に試すためのデモプログラムです。

アルゴリズムの概要や基本的な動作を確認することを目的としています。

---

### stable.py

実験・評価用の安定版実装です。

デモ版より安定した分類性能が得られるよう調整しています。

#### 主な調整項目

- `SEED`：実験の再現性を制御
- `steps`：学習・推論時の反復回数（例：18、28）
- これらのパラメータを変更することで、性能や安定性への影響を比較できます。
---

# レビューのお願い

本研究は現在も改善を続けています。

特に以下の観点からレビューいただけますと大変助かります。

* 評価方法の妥当性
* データリークの有無
* 実験設計
* Few-shot Learningとしての位置付け
* 改善すべき点

IssueやPull Request、メール等でご意見をいただけますと幸いです。

---

# ライセンス

MIT License
