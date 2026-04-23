---
title: "Reinforcement Learning: Chapter 10 On-policy Control with Approximation"
category: Reinforcement Learning
weight: 10
date: 2026-04-23
---

# Reinforcement Learning: Chapter 10 On-policy Control with Approximation

> Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed.) — Chapter 10 핵심 정리

---

## 목차

1. [Prediction → Control로의 확장](#1-prediction--control로의-확장)
2. [Episodic Semi-Gradient SARSA](#2-episodic-semi-gradient-sarsa)
3. [Mountain Car: 연속 상태 제어 벤치마크](#3-mountain-car-연속-상태-제어-벤치마크)
4. [n-step Semi-Gradient SARSA](#4-n-step-semi-gradient-sarsa)
5. [Average Reward: Continuing Tasks의 목적함수](#5-average-reward-continuing-tasks의-목적함수)
6. [Differential Semi-Gradient SARSA](#6-differential-semi-gradient-sarsa)
7. [전체 요약 및 이후 챕터와의 연결](#7-전체-요약-및-이후-챕터와의-연결)

---

## 1. Prediction → Control로의 확장

Chapter 9에서는 **주어진 policy $\pi$** 의 value function $v_\pi$를 함수 근사로 추정하는 **prediction** 문제를 다뤘습니다. Chapter 10은 이를 **control** 로 확장합니다.

### Action-Value Function 근사

Control에서는 $v_\pi(s)$ 대신 **$q_\pi(s, a)$** 를 근사해야 합니다. 모델($p$)이 없으면 greedy policy improvement에 행동-가치 함수가 필요하기 때문입니다.

$$\hat{q}(s, a, \mathbf{w}) \approx q_\pi(s, a), \quad \mathbf{w} \in \mathbb{R}^d$$

### GPI + 함수 근사

```
Prediction (Evaluation)          Control
────────────────────────         ─────────────────────────
ŷ(s, w) ≈ v_π(s)                ŷ(s, a, w) ≈ q_π(s, a)
Semi-gradient TD update  →→→    Semi-gradient SARSA update
                                 + ε-greedy improvement
```

---

## 2. Episodic Semi-Gradient SARSA

### 업데이트 규칙

$$\boxed{\mathbf{w}_{t+1} = \mathbf{w}_t + \alpha\left[R_{t+1} + \gamma\hat{q}(S_{t+1}, A_{t+1}, \mathbf{w}_t) - \hat{q}(S_t, A_t, \mathbf{w}_t)\right]\nabla_{\mathbf{w}}\hat{q}(S_t, A_t, \mathbf{w}_t)}$$

종료 시각 $T$에서 $\hat{q}(S_T, \cdot, \mathbf{w}) \doteq 0$으로 처리합니다.

### 알고리즘

```
Input: 미분 가능한 q̂(s, a, w), α, ε
Initialize: w ← 0

Loop for each episode:
    S ← initial state
    A ← ε-greedy(S, w)

    Loop for each step:
        R, S' ← take action A
        If S' is terminal:
            w ← w + α[R - q̂(S, A, w)] ∇q̂(S, A, w)
            break
        A' ← ε-greedy(S', w)
        w ← w + α[R + γq̂(S', A', w) - q̂(S, A, w)] ∇q̂(S, A, w)
        S ← S', A ← A'
```

---

## 3. Mountain Car: 연속 상태 제어 벤치마크

### 문제 설정

Mountain Car는 함수 근사 control의 **표준 벤치마크** 입니다.

- **상태**: $(x, \dot{x})$ — 위치 $x \in [-1.2, 0.5]$, 속도 $\dot{x} \in [-0.07, 0.07]$
- **행동**: 왼쪽(-1), 중립(0), 오른쪽(+1) 가속
- **보상**: 매 step마다 $-1$ (목표: 가능한 빨리 정상 도달)
- **목표**: $x \geq 0.5$ (우측 정상)

```
       목표 ↗
      /     \
_____/       \_____
← 출발
```

### 핵심 어려움

- 중력이 강해 **직접 가속만으로는 정상 도달 불가**
- 왼쪽으로 먼저 후진해 **운동 에너지를 축적** 해야 함 (비직관적 전략)
- 희소 보상(sparse reward)에 가까움 — 탐색 전략이 중요

### Tile Coding 적용

상태 $(x, \dot{x})$를 tile coding으로 특징화:

$$\hat{q}(s, a, \mathbf{w}) = \mathbf{w}^\top \mathbf{x}(s, a)$$

- 8개 tiling, 각 $8 \times 8$ 격자 → 총 $8 \times 64 \times 3 = 1536$개 특징
- 각 상태-행동 쌍에서 **8개의 타일** 활성화 → 희소 이진 벡터

---

## 4. n-step Semi-Gradient SARSA

### n-step Return

$$G_{t:t+n} \doteq R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{n-1}R_{t+n} + \gamma^n \hat{q}(S_{t+n}, A_{t+n}, \mathbf{w}_{t+n-1})$$

### 업데이트

$$\mathbf{w}_{t+n} = \mathbf{w}_{t+n-1} + \alpha\left[G_{t:t+n} - \hat{q}(S_t, A_t, \mathbf{w}_{t+n-1})\right]\nabla_{\mathbf{w}}\hat{q}(S_t, A_t, \mathbf{w}_{t+n-1})$$

### n의 영향

| $n$ | 특성 |
|---|---|
| $n=1$ | Semi-gradient SARSA (낮은 분산, bootstrap 의존) |
| $n=\infty$ | Monte Carlo (unbiased, 높은 분산) |
| 중간 $n$ | 실제로 최적 성능 — bias-variance tradeoff |

---

## 5. Average Reward: Continuing Tasks의 목적함수

### Episodic vs. Continuing

에피소드가 끝나지 않는 **continuing task** 에서는 discounted return $\sum \gamma^k R$이 적절하지 않습니다 ($\gamma < 1$ 강제 필요). 대신 **평균 보상(average reward)** 을 사용합니다.

### Average Reward 정의

$$\boxed{r(\pi) \doteq \lim_{h \to \infty} \frac{1}{h}\sum_{t=1}^{h}\mathbb{E}\left[R_t \mid S_0, A_{0:t-1} \sim \pi\right] = \sum_s \mu_\pi(s)\sum_a \pi(a \mid s)\sum_{s',r} p(s',r \mid s,a)\, r}$$

$\mu_\pi$: policy $\pi$ 하에서의 stationary distribution.

### Differential Return

평균 보상 $r(\pi)$를 기준으로 한 **차분 보상(differential reward):**

$$\delta_t \doteq R_{t+1} - \bar{R}_t + \hat{v}(S_{t+1}, \mathbf{w}) - \hat{v}(S_t, \mathbf{w})$$

여기서 $\bar{R}_t$는 평균 보상의 추정치 (별도로 업데이트).

---

## 6. Differential Semi-Gradient SARSA

Continuing task를 위한 control 알고리즘:

### 업데이트 규칙

$$\delta_t = R_{t+1} - \bar{R} + \hat{q}(S_{t+1}, A_{t+1}, \mathbf{w}) - \hat{q}(S_t, A_t, \mathbf{w})$$

$$\mathbf{w} \leftarrow \mathbf{w} + \alpha\,\delta_t\,\nabla_{\mathbf{w}}\hat{q}(S_t, A_t, \mathbf{w})$$

$$\bar{R} \leftarrow \bar{R} + \beta\,\delta_t$$

- $\alpha$: value function 학습률
- $\beta$: 평균 보상 추정 학습률

---

## 7. 전체 요약 및 이후 챕터와의 연결

### Chapter 10 구조 요약

```
Ch.9 Prediction (v̂)
        ↓
Ch.10 Control (q̂)
        ├── Episodic Semi-Gradient SARSA
        │       w ← w + α[R + γq̂(S',A',w) - q̂(S,A,w)] ∇q̂(S,A,w)
        │       + ε-greedy improvement
        │
        ├── Mountain Car 벤치마크
        │       연속 상태, Tile Coding 적용
        │
        ├── n-step Semi-Gradient SARSA
        │       n 크기로 bias-variance tradeoff 조절
        │
        └── Average Reward (Continuing Tasks)
                r(π) = Σ_s μ_π(s) Σ_a π·Σ_{s',r} p·r
                Differential SARSA
```

### 핵심 수식 한눈에 보기

**Episodic Semi-Gradient SARSA**:
$$\mathbf{w} \leftarrow \mathbf{w} + \alpha\left[R + \gamma\hat{q}(S',A',\mathbf{w}) - \hat{q}(S,A,\mathbf{w})\right]\nabla_{\mathbf{w}}\hat{q}(S,A,\mathbf{w})$$

**n-step Return**:
$$G_{t:t+n} = \sum_{k=1}^{n}\gamma^{k-1}R_{t+k} + \gamma^n\hat{q}(S_{t+n},A_{t+n},\mathbf{w})$$

**Average Reward**:
$$r(\pi) = \sum_s \mu_\pi(s)\sum_a \pi(a|s)\sum_{s',r} p(s',r|s,a)\,r$$

### 이후 챕터와의 연결

| 챕터 | 주제 | 연결점 |
|---|---|---|
| Ch.11 | Off-policy + 함수 근사 | Semi-gradient의 수렴 문제 심화 |
| Ch.12 | Eligibility Traces | n-step을 λ로 통합 |
| Ch.13 | Policy Gradient | q̂ 없이 policy 직접 최적화 |

---

> **다음 챕터로**: Chapter 11에서는 off-policy + 함수 근사 + bootstrapping이 동시에 결합될 때 발생하는 **Deadly Triad** 문제와 이를 해결하는 Gradient-TD 계열 방법을 다룹니다.
