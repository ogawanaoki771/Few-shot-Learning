# Test-Time Adaptation (TTA)

本実験では、未知のテストサンプルに対して **Test-Time Adaptation (TTA)** を行います。

TTAは、テスト時に入力されたサンプルに合わせてモデルの内部状態を一時的に適応させる方法です。

本実装では、テストサンプルの**正解ラベルを使用せず** 、入力された曲線の座標だけを利用して適応を行います。

---

## 1. この実験におけるTTA

通常の学習では、

```text
入力
 ↓
モデル
 ↓
予測
 ↓
正解ラベルを使って学習
```

という流れになります。

一方、本実験のTTAでは、

```text
学習済みシステム
        ↓
   テストサンプル
        ↓
  ラベルなしで適応
        ↓
   特徴量を抽出
        ↓
 Logistic Regression
        ↓
      予測
```

という流れになります。

重要なのは、**テストサンプルの正解ラベルを適応処理には使用しない**ことです。

---

## 2. 実装

テスト時には、学習済みシステムをコピーします。

```python
working_eco = copy.deepcopy(self)
```

そのコピーに対してTTAを実行します。

```python
working_eco.cycle(
    sample,
    label=None,
    steps=steps,
    is_training=False
)
```

この処理は `extract_features()` 内で行われています。

```python
if not is_training:
    working_eco = copy.deepcopy(self)

    working_eco.cycle(
        sample,
        label=None,
        steps=steps,
        is_training=False
    )

    target_pool = working_eco.pool
```

したがって、TTAによる変更はコピー上でのみ行われ、元の学習済みシステムには残りません。

---

## 3. `label=None` の意味

TTAでは、

```python
label=None
```

としています。

これは、テストサンプルの正解ラベルを適応処理に渡さないことを意味します。

`adapt()` では、

```python
if current_label is None or current_label == self.target_label:
```

となっているため、`current_label=None` の場合にはラベルとの一致・不一致による更新は行われません。

つまり、

```text
テスト入力
   ↓
座標情報だけを使用
   ↓
Cellを適応
```

となります。

正解ラベル

```text
Closed / Open
```

はTTAの入力には使用されません。

---

## 4. `is_training=False` の意味

ここは `label=None` とは別の役割を持っています。

`is_training=False` の場合、`cycle()` はテスト用の分岐に入ります。

```python
else:
    c.adapt(
        sample_coords,
        c.last_activation,
        current_label=None
    )

    if c.last_activation > 0.3:
        c.energy = min(
            1.2,
            c.energy + 0.02
        )
```

この分岐では、Cellの一時的な適応は行いますが、学習時に行われるCellの分裂や削除は行いません。

---

## 5. 学習時との違い

### 学習時

```python
cycle(
    sample,
    label,
    steps=28,
    is_training=True
)
```

学習時には、

* 正解ラベルを利用できる
* Cellのアンカーを適応する
* Cell間のリンクを更新する
* 条件を満たしたCellが分裂する
* エネルギーに基づいてCellが削除される
* システム自体が更新される

という処理が行われます。

```text
Training
    ↓
入力 + 正解ラベル
    ↓
Cellの適応
    ↓
リンク更新
    ↓
分裂・淘汰
    ↓
学習済みシステム
```

---

### TTA時

```python
cycle(
    sample,
    label=None,
    steps=12,
    is_training=False
)
```

TTAでは、

* 正解ラベルを使用しない
* 入力サンプルに対してアンカーを適応する
* Cellの活動状態を利用する
* Cellの分裂を行わない
* Cellの削除を行わない
* コピーしたシステム上で適応する

という処理になります。

```text
Test Sample
    ↓
label=None
    ↓
入力だけで適応
    ↓
一時的なCell状態
    ↓
特徴量抽出
    ↓
予測
    ↓
working_ecoを破棄
```
---
## 6. `label=None` と `is_training=False` は別物

この2つは似て見えますが、**役割がまったく異なります**。

```python
working_eco.cycle(
    sample,
    label=None,
    steps=steps,
    is_training=False
)
```

この1行では、

* `label=None` → **正解ラベルを適応に使わない**
* `is_training=False` → **`cycle()` のテスト用処理を選択する**

という2つの指定を同時に行っています。

---

### 6.1 `label=None` は「ラベルを渡さない」という指定

まず、

```python
label=None
```

は、テストサンプルの正解ラベルを `cycle()` に渡さないことを意味します。

例えばテストサンプルが実際には、

```text
Closed
```

だったとしても、

```python
label=None
```

として渡されます。

つまり、`cycle()` の内部では、

```text
Test Sample
    ↓
座標データ
    +
label = None
```

という状態になります。

---

### 6.2 `label=None` が `adapt()` にどう伝わるか

`cycle()` の学習分岐では、

```python
c.adapt(
    sample_coords,
    c.last_activation,
    label
)
```

としています。

つまり、

```python
label=None
```

なら、

```python
c.adapt(
    sample_coords,
    c.last_activation,
    current_label=None
)
```

として呼び出されます。

`adapt()` では、

```python
if current_label is None or current_label == self.target_label:
```

となっています。

したがって、

```python
current_label is None
```

の場合は、ラベルの一致・不一致を判定しません。

つまり、

```text
current_label = Closed
        ↓
target_label と比較
        ↓
一致 / 不一致によって処理が変わる
```

のではなく、

```text
current_label = None
        ↓
ラベル比較を行わない
        ↓
ラベルによる適応をしない
```

となります。

したがって、`label=None` の役割は、

> **正解ラベルを教師信号として使用しない**

ことです。

---

### 6.3 ただし、`label=None` だけではTTAにならない

ここが重要です。

例えば、

```python
cycle(
    sample,
    label=None,
    steps=12,
    is_training=True
)
```

とした場合を考えます。

この場合でも、

```python
is_training=True
```

なので、`cycle()` は学習側の分岐に入ります。

```python
if is_training:
```

そのため、

* Cellの適応
* Cellの分裂
* Energyの更新
* Cellの削除

など、学習時の処理が実行されます。

つまり、

```text
label=None
```

は

**「ラベルを使わない」**

というだけであり、

**「学習処理を止める」**

という意味ではありません。

---

## 6.4 `is_training=False` は処理の分岐を変える

一方、

```python
is_training=False
```

は `cycle()` の処理そのものを変更します。

`cycle()` では、

```python
if is_training:
    ...
else:
    ...
```

と分岐しています。

`False` の場合は、テスト時用の処理が実行されます。

```python
else:
    c.adapt(
        sample_coords,
        c.last_activation,
        current_label=None
    )

    if c.last_activation > 0.3:
        c.energy = min(
            1.2,
            c.energy + 0.02
        )
```

ここでは明示的に、

```python
current_label=None
```

が指定されています。

したがって、TTA時には正解ラベルを使わずに `adapt()` が実行されます。

---

## 6.5 `is_training=False` にすると何が変わるのか

学習時の分岐には、

```python
if c.energy > 0.90 and len(self.pool) < 75:
    self.max_id += 1
    new_cells.append(c.divide(self.max_id))
```

があります。

これはCellの**分裂**です。

さらに学習処理の最後には、

```python
self.pool.extend(new_cells)

self.pool = [
    c for c in self.pool
    if c.energy > 0.05 or c.age < 4
]
```

があります。

これはCellの**追加と削除**です。

しかし、

```python
is_training=False
```

の場合、この学習側の処理には入りません。

そのためTTAでは、

```text
Cellのアンカー適応
        ↓
行う

Cellの分裂
        ↓
行わない

Cellの削除
        ↓
行わない
```

となります。

---

## 6.6 したがって、2つの指定はこう考える

| 項目           | `label=None` | `is_training=False` |
| ------------ | ------------ | ------------------- |
| 正解ラベルを渡さない   | **決める**      | 直接は決めない             |
| ラベルによる適応をしない | **決める**      | `label=None` の場合に実現 |
| アンカー適応       | 決めない         | **行う**              |
| Cell分裂       | 決めない         | **行わない**            |
| Cell削除       | 決めない         | **行わない**            |
| TTA用の処理分岐    | 決めない         | **決める**             |

つまり、

```python
label=None
```

は、

> **「教師ラベルを使わない」**

という指定。

一方、

```python
is_training=False
```

は、

> **「`cycle()` のテスト時用の処理を実行する」**

という指定です。

---

## 6.7 実際のTTAでは2つを組み合わせる

現在のコードでは、

```python
working_eco.cycle(
    sample,
    label=None,
    steps=steps,
    is_training=False
)
```

としています。

この組み合わせによって、

```text
label=None
    ↓
正解ラベルを使用しない

is_training=False
    ↓
学習時の分裂・削除を行わない
    ↓
テスト用の適応を実行する
```

という動作になります。

したがって、

```text
                 Test Sample
                     │
                     ▼
              label = None
                     │
                     ▼
          正解ラベルを使わない
                     │
                     ▼
          is_training = False
                     │
                     ▼
          ┌─────────────────┐
          │   Test-time     │
          │   adaptation    │
          └────────┬────────┘
                   │
             アンカー適応
                   │
                   ▼
             Cell活動状態
                   │
                   ▼
               特徴量抽出
                   │
                   ▼
          Logistic Regression
```

という流れになります。

---

## 6.8 さらに重要な点：`deepcopy()`

ただし、もう一つ重要な条件があります。

現在のTTAでは、

```python
working_eco = copy.deepcopy(self)
```

としてから、

```python
working_eco.cycle(...)
```

を実行しています。

つまり、

```text
学習済み eco
      │
      │ deepcopy
      ▼
working_eco
      │
      │ TTA
      ▼
一時的に適応された状態
```

です。

そのため、TTAによるアンカーの変化などは、

```text
working_eco
```

には反映されますが、

```text
元の eco
```

には反映されません。

---

## 6.9 3つを分けて考える

この実装では、次の3つを別々に考えると理解しやすくなります。

### ① `label=None`

**教師信号の制御**

```text
正解ラベルを使う？
        ↓
      使わない
```

---

### ② `is_training=False`

**処理モードの制御**

```text
学習時の構造更新をする？
        ↓
      しない

テスト時の適応をする？
        ↓
      する
```

---

### ③ `copy.deepcopy(self)`

**適応結果を元モデルに残すかどうかの制御**

```text
TTAによる変更
        ↓
コピーにだけ反映
        ↓
元の学習済みモデルは維持
```

---

## 6.10 この実装のTTAを一言で表すと

したがって、今回のTTAは単に、

> 「`is_training=False` にした」

というものではありません。

正確には、

> **正解ラベルを与えず、学習済みシステムをコピーし、そのコピーに対してテストサンプルの入力だけを用いた一時的な適応を行い、その状態から特徴量を抽出する**

という仕組みです。

```text
           学習済みシステム
                  │
             deepcopy
                  │
                  ▼
          ┌──────────────┐
          │ working_eco  │
          └──────┬───────┘
                 │
          Test Sample
                 │
          label = None
                 │
                 ▼
       is_training = False
                 │
                 ▼
        ラベルなしTTA
                 │
        ┌────────┴────────┐
        │                 │
    アンカー適応       Cell活動
        │                 │
        └────────┬────────┘
                 ▼
             特徴量抽出
                 │
                 ▼
       Logistic Regression
                 │
                 ▼
              Prediction

       ※ 元の eco は変更しない
```

このように、**`label=None` は「何を使わないか」を決め、`is_training=False` は「どの処理を実行するか」を決め、`deepcopy()` は「その変更をどこに残すか」を決めている**、と整理すると、このコードのTTA構造がかなり明確になります。

---



## 7. なぜ `deepcopy()` が必要なのか

TTAでは、テストサンプルごとに独立した適応を行いたいと考えています。

そのため、

```python
working_eco = copy.deepcopy(self)
```

によって学習済みモデルをコピーします。

例えば、

```text
学習済みモデル　(eco)
     │
     ├── Test A → working_eco_A → TTA
     │
     ├── Test B → working_eco_B → TTA
     │
     └── Test C → working_eco_C → TTA
```

となります。

Test Aへの適応がTest Bの推論状態に残ることはありません。

これは、テストサンプル間で状態が蓄積してしまうことを避けるためです。

---

## 8. この実験での「非破壊的TTA」

本実装では、

>  **学習済みシステムを変更せず、そのコピーに対してテストサンプルごとの一時的な適応を行う**

という方式を採用しています。

そのため、本実験ではこれを

**Non-destructive Test-Time Adaptation**

と呼んでいます。

```text
                 学習
                  ↓
          ┌─────────────┐
          │  Trained ECO │
          └──────┬──────┘
                 │
          ┌──────┴──────┐
          │             │
        copy           copy
          ↓             ↓
       Test A         Test B
          ↓             ↓
        TTA            TTA
          ↓             ↓
      Features        Features
          │             │
          └──────┬──────┘
                 ↓
        Logistic Regression
                 ↓
              Prediction
```

---

## 9. この実験でのTTAの目的

本実験では、訓練データとテストデータで図形ファミリーを分離しています。

```text
Training Families
    ↓
  Closed / Open
    ↓
学習済みシステム

        ↓

Unseen Test Families
    ↓
ラベルなしTTA
    ↓
特徴量抽出
    ↓
分類
```

したがって、TTAによって

> **未知の図形に対して、入力そのものから内部表現を適応させることで、開曲線・閉曲線の分類に利用できる表現を得られるか**

を検証します。

---

## 10. 注意点

この実装で `is_training=False` としたからといって、**テスト時に何も更新されないわけではありません。**

TTAでは、

```python
c.adapt(...)
```

によってCellの内部状態が適応します。

ただし、その適応はコピーされた `working_eco` 上で行われます。

したがって、

```text
「適応しない」
```

ではなく、

```text
「適応するが、元の学習済みモデルを変更しない」
```

という設計です。

また、`label=None` によって正解ラベルは適応処理に渡されません。

---

## まとめ

本実装のTTAは、

> **正解ラベルを使わず、テストサンプルの入力だけを用いて、学習済みシステムのコピーを一時的に適応させ、その内部活動から特徴量を抽出する方法**

です。

重要なのは次の3点です。

1. **`label=None`**

   * テスト時の正解ラベルを適応に使用しない

2. **`is_training=False`**

   * Cellの分裂・削除など、学習時の構造更新を行わない

3. **`copy.deepcopy()`**

   * TTAによる変更を元の学習済みエコシステムに残さない

したがって、この実験におけるTTAは、

```text
ラベルなし
+
一時的な内部適応
+
元モデルを変更しない
```

という3つの特徴を持ちます。
