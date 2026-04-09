---
title: "Reinforcement Learning: Chapter 4 Proofs: DP"
category: "Reinforcement Learning"
weight: 14
date: 2026-04-09
---

# Chapter 4 Proofs: DP

> Sutton & Barto ??援먯옱?먯꽌 ?앸왂???섑븰??利앸챸 ?곸꽭 ?뺣━

---
## 紐⑹감

1. [Banach Fixed Point Theorem](#1-banach-fixed-point-theorem)
2. [Policy Improvement Theorem ?꾩쟾??利앸챸](#2-policy-improvement-theorem-?꾩쟾??利앸챸)
3. [琯-soft Policy Improvement 利앸챸](#3-琯-soft-policy-improvement-利앸챸)
4. [Importance Sampling Unbiasedness ?꾨???利앸챸](#4-importance-sampling-unbiasedness-?꾨???利앸챸)
5. [Weighted IS Consistency 利앸챸](#5-weighted-is-consistency-利앸챸)

---

## 1. Banach Fixed Point Theorem

> **?꾩튂**: Chapter 4 ??Policy Evaluation, Value Iteration ?섎졃???섑븰??洹쇨굅

Chapter 4?먯꽌 Policy Evaluation怨?Value Iteration???섎졃???ㅻ챸????"Banach Fixed Point Theorem???곸슜?쒕떎"怨좊쭔 ?덉뒿?덈떎. ???뺣━ ?먯껜瑜?泥섏쓬遺??利앸챸?⑸땲??

### ?뺣━

$(X, d)$媛 ?꾨퉬 嫄곕━ 怨듦컙(complete metric space)?닿퀬, $T: X \to X$媛 ?ㅼ쓬??留뚯”?섎뒗 $\gamma$-contraction ($0 \leq \gamma < 1$)?대㈃:

$$d(Tx, Ty) \leq \gamma\, d(x, y), \quad \forall x, y \in X$$

$T$??**?좎씪??怨좎젙??fixed point) $x^*$**瑜?媛吏硫? ?꾩쓽??$x_0 \in X$?먯꽌 ?쒖옉??諛섎났??$x_k = T^k x_0$? $x^*$濡??섎졃?⑸땲??

### 利앸챸

#### Step 1: $\{x_k\}$媛 Cauchy ?섏뿴?꾩쓣 蹂댁엫

?꾩쓽??$m > n$??????쇨컖遺?깆떇??諛섎났 ?곸슜?⑸땲??

$$d(x_m, x_n) \leq \sum_{k=n}^{m-1} d(x_{k+1}, x_k)$$

媛???뿉 contraction 議곌굔??$k$踰?諛섎났 ?곸슜?⑸땲??

$$d(x_{k+1}, x_k) = d(Tx_k, Tx_{k-1}) \leq \gamma\, d(x_k, x_{k-1}) \leq \gamma^2\, d(x_{k-1}, x_{k-2}) \leq \cdots \leq \gamma^k\, d(x_1, x_0)$$

?곕씪??

$$d(x_m, x_n) \leq \sum_{k=n}^{m-1} \gamma^k\, d(x_1, x_0) \leq d(x_1, x_0) \sum_{k=n}^{\infty} \gamma^k = \frac{\gamma^n}{1-\gamma}\, d(x_1, x_0)$$

$\gamma < 1$?대?濡?$n \to \infty$?대㈃ $\frac{\gamma^n}{1-\gamma} \to 0$. ?곕씪??$\{x_k\}$??**Cauchy ?섏뿴**?낅땲??

#### Step 2: 洹뱁븳 $x^*$媛 怨좎젙?먯엫??蹂댁엫

$X$媛 ?꾨퉬?대?濡?Cauchy ?섏뿴? $X$ ?덉뿉???섎졃?⑸땲?? $x_k \to x^* \in X$.

$T$??contraction?대?濡??곗냽?닿퀬, ?곕씪??

$$Tx^* = T\!\left(\lim_{k \to \infty} x_k\right) = \lim_{k \to \infty} Tx_k = \lim_{k \to \infty} x_{k+1} = x^*$$

利?$x^*$??$T$??怨좎젙?먯엯?덈떎.

#### Step 3: 怨좎젙?먯쓽 ?좎씪??
$x^*$? $y^*$媛 紐⑤몢 怨좎젙?먯씠??媛?뺥빀?덈떎:

$$d(x^*, y^*) = d(Tx^*, Ty^*) \leq \gamma\, d(x^*, y^*)$$

$$(1 - \gamma)\, d(x^*, y^*) \leq 0$$

$\gamma < 1$?대?濡?$1 - \gamma > 0$?닿퀬, $d \geq 0$?대?濡?

$$d(x^*, y^*) = 0 \implies x^* = y^* \qquad \blacksquare$$

### RL?먯꽌???곸슜

| RL ?뚭퀬由ъ쬁 | ?곗궛??$T$ | 嫄곕━ 怨듦컙 | 怨좎젙??|
|---|---|---|---|
| Policy Evaluation | $T^\pi v = \sum_a \pi \sum_{s',r} p[r + \gamma v']$ | $(\mathbb{R}^{\|\mathcal{S}\|}, \|\cdot\|_\infty)$ | $v_\pi$ |
| Value Iteration | $T^* v = \max_a \sum_{s',r} p[r + \gamma v']$ | $(\mathbb{R}^{\|\mathcal{S}\|}, \|\cdot\|_\infty)$ | $v_*$ |

??寃쎌슦 紐⑤몢 $\gamma < 1$?대㈃ Banach ?뺣━??議곌굔??留뚯”?섏뿬 ?섎졃??蹂댁옣?⑸땲??

### ?섎졃 ?띾룄

$k$踰?諛섎났 ??怨좎젙?먭퉴吏??嫄곕━:

$$\|x_k - x^*\|_\infty \leq \frac{\gamma^k}{1-\gamma}\, \|x_1 - x_0\|_\infty$$

$\gamma$媛 ?묒쓣?섎줉, $k$媛 ?댁닔濡??ㅼ감媛 **湲고븯湲됱닔?곸쑝濡?媛먯냼**?⑸땲??

---
