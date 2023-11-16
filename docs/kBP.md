## 集合・定数

$P$: 候補パスの集合（length_limitを超えないすべての単純パス） \
$k$: 選ぶパスの本数 \
$l_{i}$: パスiの長さ \
$s_{ij}$: パス$i$とパス$j$の類似度 \
$\alpha$: 類似度をどの程度重視するかの係数 \
$\theta_{min}$: 類似度の最小値 \
$\theta_{max}$: 類似度の最大値

## 変数

$x_i$: パス$i$を選ぶかどうか \
$y_{ij}$: パス$i$とパス$j$を同時に選ぶかどうか \
$\theta$: 選んだパス集合の類似度の合計

## 目的関数

$\min \sum_{p \in P} l_p x_p$

## 制約条件

1. 選ぶパスの本数は$k$本

    $\sum_{p \in P} x_i = \min(k, |P|)$

2. y_ijの表現

    $y_{ij} \leq x_i, \forall i, j \in P, i \neq j$

    $y_{ij} \leq x_j, \forall i, j \in P, i \neq j$

    $ y_{ij} \geq x_i + x_j - 1, \forall i, j \in P, i \neq j$

3. 選んだパス集合の類似度の合計

    $\sum_{i, j \in P, i \neq j} s_{ij} y_{ij} \leq \alpha \theta_{min} + (1 - \alpha) \theta_{max}$
