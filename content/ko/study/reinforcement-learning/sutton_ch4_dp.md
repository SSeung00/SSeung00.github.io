---
title: "Reinforcement Learning: Chapter 4 Dynamic Programming"
category: Reinforcement Learning
weight: 4
date: 2026-04-09
---

# Reinforcement Learning: Chapter 4 Dynamic Programming

> Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed.) — Chapter 4 핵심 정리

---

## 목차

1. [Dynamic Programming이란](#1-dynamic-programming이란)
2. [Policy Evaluation: Prediction 문제](#2-policy-evaluation-prediction-문제)
3. [Policy Evaluation 수렴 증명](#3-policy-evaluation-수렴-증명)
4. [Policy Improvement Theorem 유도](#4-policy-improvement-theorem-유도)
5. [Policy Iteration](#5-policy-iteration)
6. [Value Iteration](#6-value-iteration)
7. [Generalized Policy Iteration (GPI)](#7-generalized-policy-iteration-gpi)
8. [DP의 효율성과 한계](#8-dp의-효율성과-한계)
9. [전체 요약 및 이후 챕터와의 연결](#9-전체-요약-및-이후-챕터와의-연결)

---

## 1. Dynamic Programming이란

**Dynamic Programming (DP)**은 Chapter 3에서 정의한 Bellman Equation을 **반복적으로 적용**하여 value function을 계산하는 방법입니다.

### 전제 조건

- 환경의 완전한 모델, 즉 전이 함수 $p(s', r \mid s, a)$를 알고 있어야 합니다
- 상태 공간 $\mathcal{S}$와 행동 공간 $\mathcal{A}$가 유한(finite)해야 합니다

이 두 조건은 현실에서 종종 충족하기 어렵습니다. 그럼에도 DP를 배우는 이유는, 이후 모든 RL 알고리즘이 **DP의 아이디어를 근사하거나 샘플로 대체하는 방식**으로 설계되기 때문입니다.

### Chapter 4의 두 핵심 문제

```
DP
├── Prediction (평가 문제)
│       "주어진 π에 대해 v_π를 계산하라"
│       → Policy Evaluation
│
└── Control (제어 문제)
        "최적 policy π_*를 찾아라"
        → Policy Iteration
        → Value Iteration
```

---

## 2. Policy Evaluation: Prediction 문제

### 문제 설정

고정된 policy $\pi$가 주어졌을 때, 모든 상태 $s$에 대해 $v_\pi(s)$를 계산하는 문제입니다.

Chapter 3에서 Bellman Equation은 $v_\pi$를 유일하게 결정하는 연립 선형 방정식임을 보였습니다:

$$v_\pi(s) = \sum_a \pi(a \mid s) \sum_{s', r} p(s', r \mid s, a)\left[r + \gamma v_\pi(s')\right], \quad \forall s \in \mathcal{S}$$

작은 $|\mathcal{S}|$에서는 직접 역행렬로 풀 수 있지만, 일반적으로는 **반복적 방법(iterative method)**을 씁니다.

### Iterative Policy Evaluation

임의의 초기값 $v_0$에서 출발하여 Bellman Equation을 업데이트 규칙으로 반복 적용합니다:

$$\boxed{v_{k+1}(s) \leftarrow \sum_a \pi(a \mid s) \sum_{s', r} p(s', r \mid s, a)\left[r + \gamma v_k(s')\right]}$$

이를 **Bellman expectation backup**이라 부릅니다. $k \to \infty$이면 $v_k \to v_\pi$로 수렴합니다.

### 알고리즘

```
Input: policy π, threshold θ (수렴 판정 기준)
Initialize: V(s) ← 0 for all s ∈ S, V(terminal) ← 0

Loop:
    Δ ← 0
    For each s ∈ S:
        v ← V(s)
        V(s) ← Σ_a π(a|s) Σ_{s',r} p(s',r|s,a) [r + γV(s')]
        Δ ← max(Δ, |v - V(s)|)
Until Δ < θ

Output: V ≈ v_π
```

**두 가지 구현 방식**:

| 방식 | 설명 | 특징 |
|---|---|---|
| **Two-array** | $v_k$와 $v_{k+1}$ 별도 유지 | 수학적으로 깔끔, 병렬화 가능 |
| **In-place** | $v$를 즉시 덮어씀 | 실제로 더 빠르게 수렴, 실용적 |

In-place 방식이 실용적으로 선호됩니다. 새로운 값이 즉시 다음 계산에 사용되어 수렴이 빨라지기 때문입니다.

---

## 3. Policy Evaluation 수렴 증명

### Bellman Expectation Operator 정의

Bellman expectation backup을 연산자 $T^\pi$로 표현합니다:

$$(T^\pi v)(s) \doteq \sum_a \pi(a \mid s) \sum_{s', r} p(s', r \mid s, a)\left[r + \gamma v(s')\right]$$

따라서 업데이트 규칙은 $v_{k+1} = T^\pi v_k$이고, 수렴 목표는 $T^\pi v_\pi = v_\pi$ (고정점, fixed point).

### $T^\pi$는 $\gamma$-contraction임을 증명

임의의 두 value function $u$, $v$에 대해 $\|T^\pi u - T^\pi v\|_\infty \leq \gamma \|u - v\|_\infty$를 보입니다.

$$|(T^\pi u)(s) - (T^\pi v)(s)|$$

$$= \left|\sum_a \pi(a|s)\sum_{s',r} p(s',r|s,a)\,\gamma\left[u(s') - v(s')\right]\right|$$

$$\leq \sum_a \pi(a|s)\sum_{s',r} p(s',r|s,a)\,\gamma\left|u(s') - v(s')\right|$$

$$\leq \gamma \|u - v\|_\infty \underbrace{\sum_a \pi(a|s)\sum_{s',r} p(s',r|s,a)}_{= 1} = \gamma \|u - v\|_\infty$$

$s$에 대해 supremum을 취하면:

$$\boxed{\|T^\pi u - T^\pi v\|_\infty \leq \gamma \|u - v\|_\infty}$$

### Banach Fixed Point Theorem 적용

$\gamma < 1$이면 $T^\pi$는 완비 거리 공간 $(\mathbb{R}^{|\mathcal{S}|}, \|\cdot\|_\infty)$ 위의 **contraction mapping**입니다.

**Banach Fixed Point Theorem**: 완비 거리 공간에서 contraction mapping은 **유일한 고정점**을 가지며, 임의의 초기값에서 반복하면 반드시 그 고정점으로 수렴합니다.

따라서:
- $v_\pi$는 $T^\pi$의 **유일한 고정점**
- 임의의 $v_0$에서 시작해도 $v_k = (T^\pi)^k v_0 \to v_\pi$

**수렴 속도**: $k$번 반복 후 오차의 상한:

$$\|v_k - v_\pi\|_\infty \leq \gamma^k \|v_0 - v_\pi\|_\infty$$

$\gamma$가 작을수록, $k$가 클수록 오차가 기하급수적으로 감소합니다.

---

## 4. Policy Improvement Theorem 유도

### 동기

$v_\pi$를 구했다면, 더 좋은 policy를 만들 수 있을까요?

현재 policy $\pi$가 있고, 상태 $s$에서 $\pi$가 선택하는 행동 대신 **행동 $a$를 한 번 취한 뒤 이후에는 다시 $\pi$를 따른다**면 그 기대 return은:

$$q_\pi(s, a) = \sum_{s', r} p(s', r \mid s, a)\left[r + \gamma v_\pi(s')\right]$$

이것이 $v_\pi(s)$보다 크다면, $s$에서 $a$를 선택하는 것이 더 낫습니다.

### Policy Improvement Theorem

**정리**: 두 deterministic policy $\pi$, $\pi'$에 대해, 모든 $s \in \mathcal{S}$에서:

$$q_\pi(s, \pi'(s)) \geq v_\pi(s)$$

이면:

$$v_{\pi'}(s) \geq v_\pi(s), \quad \forall s \in \mathcal{S}$$

### 증명

조건 $q_\pi(s, \pi'(s)) \geq v_\pi(s)$에서 출발합니다:

$$v_\pi(s) \leq q_\pi(s, \pi'(s)) = \mathbb{E}\left[R_{t+1} + \gamma v_\pi(S_{t+1}) \mid S_t=s, A_t=\pi'(s)\right]$$

$v_\pi(S_{t+1}) \leq q_\pi(S_{t+1}, \pi'(S_{t+1}))$을 대입 (조건이 모든 $s$에서 성립하므로):

$$\leq \mathbb{E}\left[R_{t+1} + \gamma q_\pi(S_{t+1}, \pi'(S_{t+1})) \mid S_t=s, A_t=\pi'(s)\right]$$

$$= \mathbb{E}\left[R_{t+1} + \gamma R_{t+2} + \gamma^2 v_\pi(S_{t+2}) \mid S_t=s, A_t=\pi'(s)\right]$$

이 치환을 계속 반복합니다 ($\gamma^k v_\pi(S_{t+k}) \to 0$ as $k \to \infty$):

$$\leq \mathbb{E}\left[R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots \mid S_t=s, A_t=\pi'(s)\right] = v_{\pi'}(s)$$

$$\boxed{v_\pi(s) \leq v_{\pi'}(s), \quad \forall s \in \mathcal{S}} \qquad \checkmark$$

### Greedy Policy Improvement

모든 상태에서 동시에 greedy하게 개선하면:

$$\pi'(s) \doteq \arg\max_a q_\pi(s, a) = \arg\max_a \sum_{s', r} p(s', r \mid s, a)\left[r + \gamma v_\pi(s')\right]$$

**언제 개선이 멈추는가?** $v_{\pi'} = v_\pi$이면 모든 $s$에서:

$$v_\pi(s) = \max_a \sum_{s', r} p(s', r \mid s, a)\left[r + \gamma v_\pi(s')\right] = v_*(s)$$

즉, 개선이 더 이상 일어나지 않으면 $\pi$는 이미 **optimal policy**입니다.

---

## 5. Policy Iteration

### 아이디어

Policy Evaluation과 Policy Improvement를 **교대로 반복**합니다:

$$\pi_0 \xrightarrow{E} v_{\pi_0} \xrightarrow{I} \pi_1 \xrightarrow{E} v_{\pi_1} \xrightarrow{I} \pi_2 \xrightarrow{E} \cdots \xrightarrow{I} \pi_* \xrightarrow{E} v_*$$

- $E$: Policy Evaluation — $\pi$ 고정 → $v_\pi$ 계산 (Bellman equation 반복)
- $I$: Policy Improvement — $v_\pi$ 고정 → $\pi' \geq \pi$ (greedy update)

### 알고리즘

```
1. Initialization
   V(s) ← arbitrary, π(s) ← arbitrary for all s ∈ S
   V(terminal) ← 0

2. Policy Evaluation
   Loop:
       Δ ← 0
       For each s ∈ S:
           v ← V(s)
           V(s) ← Σ_a π(a|s) Σ_{s',r} p(s',r|s,a) [r + γV(s')]
           Δ ← max(Δ, |v - V(s)|)
   Until Δ < θ

3. Policy Improvement
   policy-stable ← true
   For each s ∈ S:
       old-action ← π(s)
       π(s) ← argmax_a Σ_{s',r} p(s',r|s,a) [r + γV(s')]
       If old-action ≠ π(s): policy-stable ← false

4. If policy-stable: return V ≈ v_*, π ≈ π_*
   Else: go to step 2
```

### 수렴 보장

유한 MDP에서 결정론적 policy의 수는 유한합니다 ($|\mathcal{A}|^{|\mathcal{S}|}$개). Policy Improvement Theorem에 의해 각 iteration마다 policy가 **strictly 개선되거나 이미 optimal**이므로, Policy Iteration은 **유한 번의 iteration 후 반드시 수렴**합니다.

---

## 6. Value Iteration

### Policy Iteration의 비효율성

Policy Iteration의 Policy Evaluation 단계는 수렴할 때까지 반복합니다. 하지만 최적 policy를 찾기 위해 $v_\pi$를 **정확하게** 구할 필요가 있을까요? Policy Evaluation을 단 한 번의 sweep으로 줄이면 어떻게 될까요?

이 아이디어를 극단까지 밀면 **Value Iteration**이 됩니다.

### Bellman Optimality Backup

Bellman Optimality Equation을 업데이트 규칙으로 직접 사용합니다:

$$\boxed{v_{k+1}(s) \leftarrow \max_a \sum_{s', r} p(s', r \mid s, a)\left[r + \gamma v_k(s')\right]}$$

Policy Evaluation의 $\sum_a \pi(a \mid s)$ 대신 $\max_a$를 취하는 것이 유일한 차이입니다.

### 알고리즘

```
Initialize: V(s) ← arbitrary for all s ∈ S, V(terminal) ← 0

Loop:
    Δ ← 0
    For each s ∈ S:
        v ← V(s)
        V(s) ← max_a Σ_{s',r} p(s',r|s,a) [r + γV(s')]
        Δ ← max(Δ, |v - V(s)|)
Until Δ < θ

Output policy:
π(s) ← argmax_a Σ_{s',r} p(s',r|s,a) [r + γV(s')]
```

### Value Iteration의 수렴 증명

Bellman Optimality Operator $T^*$를 정의합니다:

$$(T^* v)(s) \doteq \max_a \sum_{s', r} p(s', r \mid s, a)\left[r + \gamma v(s')\right]$$

$T^*$ 역시 $\gamma$-contraction임을 보입니다.

$\max$ 연산자의 성질 $|\max_a f(a) - \max_a g(a)| \leq \max_a |f(a) - g(a)|$를 적용합니다:

$$|(T^* u)(s) - (T^* v)(s)| \leq \max_a \left|\sum_{s',r} p(s',r|s,a)\,\gamma\left[u(s') - v(s')\right]\right|$$

$$\leq \gamma \max_a \sum_{s',r} p(s',r|s,a) \|u - v\|_\infty = \gamma \|u - v\|_\infty$$

따라서:

$$\boxed{\|T^* u - T^* v\|_\infty \leq \gamma \|u - v\|_\infty}$$

$T^*$도 $\gamma$-contraction이므로, $v_*$는 $T^*$의 유일한 고정점이고 임의의 $v_0$에서 수렴이 보장됩니다.

### Policy Iteration vs. Value Iteration 비교

| | Policy Iteration | Value Iteration |
|---|---|---|
| 핵심 업데이트 | $\sum_a \pi(a \mid s)\,[\cdots]$ | $\max_a\,[\cdots]$ |
| Evaluation 방식 | 수렴할 때까지 반복 | 단 1회 sweep |
| Iteration당 비용 | 높음 | 낮음 |
| 총 Iteration 수 | 적음 | 많음 |
| 수렴 보장 | 유한 번 | $\gamma^k$로 기하급수 수렴 |

---

## 7. Generalized Policy Iteration (GPI)

### GPI의 핵심 아이디어

Policy Iteration과 Value Iteration을 **하나의 통합된 관점**으로 바라봅니다.

```
         Evaluation
    π ──────────────→ v_π
    ↑                   |
    |   Improvement     |
    └───────────────────┘
      greedy(v) → π'
```

두 과정이 **상호작용**하면서 서로를 향해 수렴합니다:
- Evaluation: $v$를 현재 $\pi$에 맞게 끌어당김
- Improvement: $\pi$를 현재 $v$에 대해 greedy하게 끌어당김

### GPI의 일반성

Evaluation과 Improvement의 **세밀함(granularity)**을 자유롭게 조절할 수 있습니다:

| 알고리즘 | Evaluation 깊이 | Improvement 빈도 |
|---|---|---|
| Policy Iteration | 완전 수렴까지 | Evaluation 완료 후 1회 |
| Value Iteration | 1 sweep | 매 sweep마다 |
| Asynchronous DP | 일부 상태만 | 비동기적 |
| TD / Q-learning | 1 step bootstrap | 매 step마다 |

이후 등장하는 MC, TD, Q-learning 등 **모든 RL 알고리즘은 GPI의 변형**입니다.

---

## 8. DP의 효율성과 한계

### 효율성

DP는 policy space를 직접 탐색하는 것보다 **압도적으로 효율적**입니다:

- Brute-force: 모든 policy 열거 → $|\mathcal{A}|^{|\mathcal{S}|}$개 (지수적)
- Policy Iteration: **다항식(polynomial) 시간** 내 수렴

### 한계: Curse of Dimensionality

DP의 각 sweep은 **모든 상태 $s \in \mathcal{S}$**를 방문합니다. 상태 공간이 크면:

- 각 iteration의 계산량: $O(|\mathcal{S}|^2 |\mathcal{A}|)$
- 연속 상태 공간에서는 직접 적용 불가능
- 상태 변수가 $n$개이고 각각 $k$개의 값을 가지면 $|\mathcal{S}| = k^n$ — 차원이 늘수록 지수 폭발

### Asynchronous DP

전체 상태를 한꺼번에 업데이트하는 대신 **일부 상태를 선택적·비동기적으로** 업데이트합니다:

- 자주 방문하는 상태를 더 자주 업데이트
- 실시간으로 다른 작업과 병렬 수행 가능
- 수렴 보장은 유지됨 (모든 상태가 무한히 자주 업데이트된다는 조건 하에)

이 아이디어는 이후 TD learning의 online 업데이트와 자연스럽게 연결됩니다.

---

## 9. 전체 요약 및 이후 챕터와의 연결

### Chapter 4 구조 요약

```
Chapter 3: Bellman Equation 정의
        ↓
Chapter 4: Bellman Equation을 반복 적용
        │
        ├── Prediction: Policy Evaluation
        │       v_{k+1}(s) ← Σ_a π Σ_{s',r} p [r + γv_k(s')]
        │       수렴 근거: T^π는 γ-contraction (Banach)
        │
        ├── Control
        │   ├── Policy Iteration
        │   │       E단계 (수렴까지) + I단계 교대 반복
        │   │       수렴 근거: Policy Improvement Theorem
        │   │
        │   └── Value Iteration
        │           v_{k+1}(s) ← max_a Σ_{s',r} p [r + γv_k(s')]
        │           수렴 근거: T^*도 γ-contraction (Banach)
        │
        └── GPI: 모든 RL 알고리즘의 통합 관점
```

### 핵심 수식 한눈에 보기

**Policy Evaluation (Bellman Expectation Backup)**:
$$v_{k+1}(s) \leftarrow \sum_a \pi(a|s) \sum_{s',r} p(s',r|s,a)\left[r + \gamma v_k(s')\right]$$

**Value Iteration (Bellman Optimality Backup)**:
$$v_{k+1}(s) \leftarrow \max_a \sum_{s',r} p(s',r|s,a)\left[r + \gamma v_k(s')\right]$$

**Greedy Policy Improvement**:
$$\pi'(s) \leftarrow \arg\max_a \sum_{s',r} p(s',r|s,a)\left[r + \gamma v_\pi(s')\right]$$

**$\gamma$-Contraction** (수렴의 근거):
$$\|T^\pi u - T^\pi v\|_\infty \leq \gamma\|u-v\|_\infty, \qquad \|T^* u - T^* v\|_\infty \leq \gamma\|u-v\|_\infty$$

### 이후 챕터로의 연결

DP의 핵심 한계는 **모델 $p$가 필요하다**는 점입니다. Chapter 5, 6은 이 한계를 샘플로 대체하여 제거합니다:

| 챕터 | 방법 | DP와의 관계 |
|---|---|---|
| Ch.5 Monte Carlo | 에피소드 샘플로 $v_\pi$ 추정 | Policy Evaluation을 샘플 평균으로 대체 |
| Ch.6 TD Learning | 매 step마다 bootstrap 업데이트 | DP(bootstrap) + MC(샘플)의 결합 |
| Ch.6 Q-learning | $\max_a$ 업데이트를 샘플로 근사 | Value Iteration을 샘플로 대체 |
| Ch.9~10 Function Approx. | 신경망으로 $v$, $q$ 근사 | DP + 함수 근사 → DQN 등으로 이어짐 |

### DP, MC, TD 세 방법의 핵심 비교

```
                모델 필요?    Bootstrap?    에피소드 완료 필요?
DP              ✅ 필요        ✅ 사용        ❌ 불필요
Monte Carlo     ❌ 불필요      ❌ 미사용      ✅ 필요
TD Learning     ❌ 불필요      ✅ 사용        ❌ 불필요
```

TD Learning이 DP와 MC의 **장점을 모두 취하는** 방법임을 이 표로 미리 파악할 수 있습니다.

---

> **다음 챕터로**: Chapter 5에서는 모델 $p$를 모르는 상황에서, 실제 에피소드 샘플을 통해 $v_\pi$와 $q_\pi$를 추정하는 **Monte Carlo Methods**를 다룹니다.
