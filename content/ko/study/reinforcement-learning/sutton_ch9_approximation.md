---
title: "Reinforcement Learning: Chapter 9 On-policy Prediction with Approximation"
category: Reinforcement Learning
weight: 9
date: 2026-04-23
---

# Reinforcement Learning: Chapter 9 On-policy Prediction with Approximation

> Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed.) — Chapter 9 핵심 정리

---

## 목차

1. [왜 함수 근사가 필요한가](#1-왜-함수-근사가-필요한가)
2. [Value-Function Approximation의 정식화](#2-value-function-approximation의-정식화)
3. [Prediction Objective: Mean Squared Value Error](#3-prediction-objective-mean-squared-value-error)
4. [Stochastic Gradient Descent (SGD)](#4-stochastic-gradient-descent-sgd)
5. [Semi-Gradient Methods](#5-semi-gradient-methods)
6. [Linear Methods](#6-linear-methods)
7. [Linear TD(0)의 수렴과 TD Fixed Point](#7-linear-td0의-수렴과-td-fixed-point)
8. [Feature Construction](#8-feature-construction)
9. [Nonlinear Function Approximation: ANN](#9-nonlinear-function-approximation-ann)
10. [Least-Squares TD (LSTD)](#10-least-squares-td-lstd)
11. [전체 요약 및 이후 챕터와의 연결](#11-전체-요약-및-이후-챕터와의-연결)

---

## 1. 왜 함수 근사가 필요한가

Chapter 2–8까지는 value function을 **테이블(table)** 로 표현했습니다. 즉 상태 $s$마다 $V(s)$ 값을 배열의 한 칸에 저장하는 방식입니다.

### Tabular 방식의 한계

- **거대한 상태 공간**: 체스 $10^{45}$, 바둑 $10^{170}$, 연속 상태 → 테이블 저장 불가능
- **일반화 부재**: 한 상태에서 학습한 정보가 유사한 다른 상태로 전이되지 않음
- **경험 부족 상태**: 방문하지 못한 상태는 학습 불가

### 해결책: 파라미터화된 근사

$$\hat{v}(s, \mathbf{w}) \approx v_\pi(s), \quad \mathbf{w} \in \mathbb{R}^d, \quad d \ll |\mathcal{S}|$$

- $\mathbf{w}$: 가중치 벡터 (파라미터)
- 한 상태를 업데이트하면 **유사한 상태들의 추정값도 함께 바뀜** → **일반화(generalization)**

```
Tabular                  Function Approximation
─────────                ──────────────────────
V: [v₁, v₂, ..., v_|S|]  w: [w₁, w₂, ..., w_d]  (d << |S|)
상태별 독립적 저장         상태들이 w를 공유 → 일반화
```

---

## 2. Value-Function Approximation의 정식화

### 교시 예제(training example)로서의 업데이트

DP, MC, TD에서의 업데이트는 모두 다음 형태였습니다:

$$V(S_t) \leftarrow V(S_t) + \alpha\left[U_t - V(S_t)\right]$$

여기서 $U_t$는 **target(목표값)** 입니다:

| 알고리즘 | Target $U_t$ |
|---|---|
| Monte Carlo | $G_t$ (실제 return) |
| TD(0) | $R_{t+1} + \gamma V(S_{t+1})$ |
| DP | $\mathbb{E}_\pi[R_{t+1} + \gamma V(S_{t+1}) \mid S_t]$ |

이를 **지도학습의 입력-출력 쌍** $\,S_t \mapsto U_t\,$로 보면, value prediction 문제는 일반적인 **function approximation 문제** 로 바뀝니다.

---

## 3. Prediction Objective: Mean Squared Value Error

Tabular에서는 모든 상태의 값을 정확히 맞출 수 있었지만, 함수 근사에서는 $d < |\mathcal{S}|$이므로 **일부 상태의 정확도를 희생**해야 합니다. 어떤 상태를 더 중요하게 여길지 명시해야 합니다.

### On-policy distribution $\mu(s)$

$\mu(s) \geq 0$, $\sum_s \mu(s) = 1$: 상태 $s$의 상대적 중요도. 일반적으로 policy $\pi$ 하에서의 **방문 빈도(stationary distribution)** 를 사용합니다.

### 목적 함수: Mean Squared Value Error $\overline{VE}$

$$\boxed{\overline{VE}(\mathbf{w}) \doteq \sum_{s \in \mathcal{S}} \mu(s)\left[v_\pi(s) - \hat{v}(s, \mathbf{w})\right]^2}$$

이상적으로는 **전역 최소점 $\mathbf{w}^*$** 를 찾고 싶지만, 비선형 근사에서는 **국소 최적(local optimum)** 에 수렴하는 것으로 만족해야 합니다.

---

## 4. Stochastic Gradient Descent (SGD)

### Gradient Descent 업데이트

$\overline{VE}$를 $\mathbf{w}$에 대해 최소화하려면 gradient descent:

$$\mathbf{w}_{t+1} = \mathbf{w}_t - \frac{1}{2}\alpha \nabla_{\mathbf{w}}\left[v_\pi(S_t) - \hat{v}(S_t, \mathbf{w}_t)\right]^2$$

$$= \mathbf{w}_t + \alpha\left[v_\pi(S_t) - \hat{v}(S_t, \mathbf{w}_t)\right]\nabla_{\mathbf{w}} \hat{v}(S_t, \mathbf{w}_t)$$

실제로는 $v_\pi(S_t)$를 모르므로 샘플 target $U_t$로 대체:

$$\boxed{\mathbf{w}_{t+1} = \mathbf{w}_t + \alpha\left[U_t - \hat{v}(S_t, \mathbf{w}_t)\right]\nabla_{\mathbf{w}} \hat{v}(S_t, \mathbf{w}_t)}$$

### Monte Carlo SGD

$U_t = G_t$는 $v_\pi(S_t)$의 **unbiased estimator** 이므로, Robbins-Monro 조건을 만족하는 $\alpha_t$ 하에서 $\mathbf{w}$는 local optimum에 수렴합니다.

### Gradient MC Algorithm

```
Input: policy π, α, 미분 가능한 ŷ(s, w)
Initialize: w ← 0 (또는 임의)

Loop for each episode:
    Generate episode S₀, A₀, R₁, ..., S_T following π
    For each t = 0, 1, ..., T-1:
        G ← Σ_{k=t+1}^{T} γ^{k-t-1} R_k
        w ← w + α[G - ŷ(S_t, w)] ∇ŷ(S_t, w)
```

---

## 5. Semi-Gradient Methods

### Bootstrapping Target의 문제

TD(0)의 target $U_t = R_{t+1} + \gamma \hat{v}(S_{t+1}, \mathbf{w}_t)$는 **$\mathbf{w}$에 의존**합니다. 진짜 gradient는

$$\nabla_{\mathbf{w}}\left[U_t - \hat{v}(S_t, \mathbf{w})\right]^2$$

에서 $U_t$의 $\mathbf{w}$ 의존성까지 미분해야 하지만, 이를 **무시**하고 아래처럼 계산합니다:

$$\boxed{\mathbf{w}_{t+1} = \mathbf{w}_t + \alpha\left[R_{t+1} + \gamma \hat{v}(S_{t+1}, \mathbf{w}_t) - \hat{v}(S_t, \mathbf{w}_t)\right]\nabla_{\mathbf{w}} \hat{v}(S_t, \mathbf{w}_t)}$$

이를 **semi-gradient** 방법이라 부릅니다.

### 특징

| | True Gradient (MC) | Semi-Gradient (TD) |
|---|---|---|
| Target $\mathbf{w}$ 의존 | 없음 | 있음 (무시) |
| 수렴 보장 | Local optimum | 선형에서만 (TD fixed point) |
| 분산 | 높음 | 낮음 |
| Bootstrap 이득 | 없음 | 있음 (온라인, 빠름) |

---

## 6. Linear Methods

### 선형 근사

각 상태 $s$를 **특징 벡터(feature vector)** $\mathbf{x}(s) = (x_1(s), \ldots, x_d(s))^\top$로 표현:

$$\hat{v}(s, \mathbf{w}) \doteq \mathbf{w}^\top \mathbf{x}(s) = \sum_{i=1}^{d} w_i x_i(s)$$

### Gradient

$$\nabla_{\mathbf{w}} \hat{v}(s, \mathbf{w}) = \mathbf{x}(s)$$

따라서 Semi-Gradient TD(0) 업데이트가 단순해집니다:

$$\boxed{\mathbf{w}_{t+1} = \mathbf{w}_t + \alpha\left[R_{t+1} + \gamma \mathbf{w}_t^\top \mathbf{x}_{t+1} - \mathbf{w}_t^\top \mathbf{x}_t\right]\mathbf{x}_t}$$

### 선형의 장점

- **전역 최적 = 국소 최적**: $\overline{VE}$가 $\mathbf{w}$에 대해 **이차 함수(quadratic)** → 유일한 최소점
- 수학적 분석 용이 → 수렴 증명 가능
- 효율적인 계산

---

## 7. Linear TD(0)의 수렴과 TD Fixed Point

### 기댓값 업데이트 분석

Semi-gradient TD(0)의 기댓값 업데이트:

$$\mathbb{E}[\mathbf{w}_{t+1} \mid \mathbf{w}_t] = \mathbf{w}_t + \alpha\left(\mathbf{b} - \mathbf{A}\mathbf{w}_t\right)$$

여기서:

$$\mathbf{A} \doteq \mathbb{E}\left[\mathbf{x}_t(\mathbf{x}_t - \gamma \mathbf{x}_{t+1})^\top\right] \in \mathbb{R}^{d \times d}$$

$$\mathbf{b} \doteq \mathbb{E}\left[R_{t+1} \mathbf{x}_t\right] \in \mathbb{R}^{d}$$

### TD Fixed Point

수렴점은 $\mathbf{A}\mathbf{w} = \mathbf{b}$를 만족합니다:

$$\boxed{\mathbf{w}_{TD} = \mathbf{A}^{-1}\mathbf{b}}$$

### 수렴 조건과 VE 경계

On-policy distribution $\mu$ 하에서 $\mathbf{A}$는 **positive definite** 이고, $\mathbf{w}_t \to \mathbf{w}_{TD}$가 보장됩니다. 이때:

$$\overline{VE}(\mathbf{w}_{TD}) \leq \frac{1}{1-\gamma} \min_{\mathbf{w}} \overline{VE}(\mathbf{w})$$

즉, TD fixed point의 오차는 **최적 오차의 $\frac{1}{1-\gamma}$ 배** 이내로 bounded됩니다. MC(진짜 gradient)가 $\min \overline{VE}$에 수렴하는 것에 비해 약간 열등하지만, bootstrap의 분산 감소 이득이 이를 상쇄합니다.

---

## 8. Feature Construction

선형 방법의 성능은 **특징(feature) 설계** 에 크게 좌우됩니다.

### 8.1 Polynomials

상태 $s = (s_1, s_2) \in \mathbb{R}^2$에 대해 $n$차 polynomial basis:

$$\mathbf{x}(s) = (1, s_1, s_2, s_1 s_2, s_1^2, s_2^2, s_1^2 s_2, \ldots)$$

- **장점**: 단순
- **단점**: 차원 $n$이 커지면 특징 수가 **$(n+1)^k$** 로 폭발

### 8.2 Fourier Basis

$s \in [0, 1]^k$일 때:

$$x_i(s) = \cos(\pi \mathbf{c}^i \cdot s), \quad \mathbf{c}^i \in \{0, 1, \ldots, n\}^k$$

- 전역적 표현, 이론적 성능이 polynomial보다 우수

### 8.3 Coarse Coding

특징이 상태 공간의 **겹치는 영역(receptive field)** 을 나타냅니다. 각 수용 영역에 속하면 1, 아니면 0.

- 수용 영역 크기 → **일반화의 폭**
- 수용 영역 수 → **해상도**

### 8.4 Tile Coding

Coarse coding의 효율적 특수 형태. 여러 개의 **타일링(tiling)** 을 서로 offset해서 쌓습니다.

```
Tiling 1:  ┌─┬─┬─┐
           ├─┼─┼─┤
           └─┴─┴─┘
Tiling 2 (offset):
             ┌─┬─┬─┐
             ├─┼─┼─┤
             └─┴─┴─┘
```

- 각 상태는 **타일링 수만큼의 타일** 에 속함 → 희소(sparse) 이진 벡터
- 계산 효율 탁월 (해시 테이블로 $O(\text{#tilings})$)
- 일반화와 해상도를 **독립적으로** 조절 가능

### 8.5 Radial Basis Function (RBF)

$$x_i(s) = \exp\left(-\frac{\|s - c_i\|^2}{2\sigma_i^2}\right)$$

연속적인 응답(continuous response)을 제공하지만 계산 비용이 tile coding보다 큼. 저차원 smoothness가 중요할 때 유용.

---

## 9. Nonlinear Function Approximation: ANN

### 동기

선형 방법은 사람이 feature를 직접 설계해야 합니다. **ANN(Artificial Neural Network)** 은 feature 자체를 학습합니다.

### Semi-Gradient Update (ANN)

$$\mathbf{w}_{t+1} = \mathbf{w}_t + \alpha\,\delta_t\,\nabla_{\mathbf{w}} \hat{v}(S_t, \mathbf{w}_t)$$

$$\delta_t = R_{t+1} + \gamma \hat{v}(S_{t+1}, \mathbf{w}_t) - \hat{v}(S_t, \mathbf{w}_t)$$

$\nabla_{\mathbf{w}} \hat{v}$는 **backpropagation** 으로 계산.

### 주의 사항

- 수렴 보장 없음 — local optimum 또는 발산 가능
- **Deadly Triad** (Ch.11): Function Approximation + Bootstrapping + Off-policy 가 동시에 있으면 발산 위험
- Chapter 16 DQN 등에서는 **Experience Replay**, **Target Network** 로 안정화

---

## 10. Least-Squares TD (LSTD)

### 동기

Linear TD(0)는 sample 마다 점진적으로 $\mathbf{w}_{TD} = \mathbf{A}^{-1}\mathbf{b}$에 수렴합니다. 데이터가 고정돼 있다면 $\mathbf{A}$, $\mathbf{b}$를 직접 추정해 한 번에 풀 수 있습니다.

### 추정량

$$\hat{\mathbf{A}}_t \doteq \sum_{k=0}^{t-1} \mathbf{x}_k(\mathbf{x}_k - \gamma\mathbf{x}_{k+1})^\top + \varepsilon \mathbf{I}$$

$$\hat{\mathbf{b}}_t \doteq \sum_{k=0}^{t-1} R_{k+1}\mathbf{x}_k$$

$$\mathbf{w}_t = \hat{\mathbf{A}}_t^{-1}\hat{\mathbf{b}}_t$$

### LSTD vs. Semi-Gradient TD

| | Semi-Gradient TD | LSTD |
|---|---|---|
| Iteration당 계산 | $O(d)$ | $O(d^2)$ (Sherman-Morrison) |
| 메모리 | $O(d)$ | $O(d^2)$ |
| 데이터 효율성 | 낮음 | 높음 (모든 데이터 즉시 활용) |
| Step-size $\alpha$ | 필요 | 불필요 |
| 점진적 적응 | 빠름 | 느림 (망각 메커니즘 필요) |

---

## 11. 전체 요약 및 이후 챕터와의 연결

### Chapter 9 구조 요약

```
Tabular의 한계 (거대한 상태 공간, 일반화 부재)
        ↓
함수 근사: ŷ(s, w) ≈ v_π(s)
        ↓
Objective: VE(w) = Σ μ(s)[v_π(s) - ŷ(s,w)]²
        ↓
SGD                     Semi-Gradient
 ├── Gradient MC         ├── Semi-Gradient TD(0)
 │   (true gradient)     │   (bootstrap, biased)
 │   수렴: local min      │   수렴: TD fixed point
        ↓
Linear: ŷ = w^T x(s)
 ├── 전역 최적 = 국소 최적
 ├── w_TD = A⁻¹b
 └── Feature: Poly, Fourier, Tile, RBF
        ↓
Nonlinear (ANN): feature 학습, but 수렴 보장 없음
        ↓
LSTD: 데이터 효율적, O(d²)
```

### 핵심 수식 한눈에 보기

**Prediction Objective**:
$$\overline{VE}(\mathbf{w}) = \sum_s \mu(s)\left[v_\pi(s) - \hat{v}(s,\mathbf{w})\right]^2$$

**General SGD Update**:
$$\mathbf{w}_{t+1} = \mathbf{w}_t + \alpha\left[U_t - \hat{v}(S_t,\mathbf{w}_t)\right]\nabla_{\mathbf{w}}\hat{v}(S_t,\mathbf{w}_t)$$

**Semi-Gradient TD(0)**:
$$\mathbf{w}_{t+1} = \mathbf{w}_t + \alpha\left[R_{t+1} + \gamma\hat{v}(S_{t+1},\mathbf{w}_t) - \hat{v}(S_t,\mathbf{w}_t)\right]\nabla_{\mathbf{w}}\hat{v}(S_t,\mathbf{w}_t)$$

**Linear TD Fixed Point**:
$$\mathbf{w}_{TD} = \mathbf{A}^{-1}\mathbf{b}, \qquad \overline{VE}(\mathbf{w}_{TD}) \leq \frac{1}{1-\gamma}\min_{\mathbf{w}}\overline{VE}(\mathbf{w})$$

### 이후 챕터로의 연결

| 챕터 | 주제 | Ch.9와의 관계 |
|---|---|---|
| Ch.10 | On-policy Control with Approximation | Semi-gradient SARSA, Mountain Car |
| Ch.11 | Off-policy Methods with Approximation | Deadly Triad, Gradient-TD |
| Ch.12 | Eligibility Traces | TD($\lambda$) + 함수 근사 |
| Ch.13 | Policy Gradient | Value 근사가 아닌 **policy** 직접 근사 |
| Ch.16 | Applications (DQN, AlphaGo) | 딥러닝 + RL의 실제 구현 |

---

> **다음 챕터로**: Chapter 10에서는 함수 근사를 **control** 문제로 확장하여, Semi-gradient SARSA와 Mountain Car 같은 연속 상태 벤치마크를 다룹니다.
