---
title: "Reinforcement Learning: Chapter 5 Monte Carlo Methods"
category: Reinforcement Learning
weight: 5
date: 2026-04-09
---

# Reinforcement Learning: Chapter 5 Monte Carlo Methods

> Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed.) — Chapter 5 핵심 정리

---

## 목차

1. [Monte Carlo Methods란](#1-monte-carlo-methods란)
2. [Monte Carlo Policy Evaluation](#2-monte-carlo-policy-evaluation)
3. [Monte Carlo Estimation of Action Values](#3-monte-carlo-estimation-of-action-values)
4. [Monte Carlo Control: GPI with MC](#4-monte-carlo-control-gpi-with-mc)
5. [Exploring Starts](#5-exploring-starts)
6. [On-policy vs. Off-policy](#6-on-policy-vs-off-policy)
7. [Off-policy Prediction: Importance Sampling](#7-off-policy-prediction-importance-sampling)
8. [Importance Sampling 수렴 분석](#8-importance-sampling-수렴-분석)
9. [Off-policy Control](#9-off-policy-control)
10. [전체 요약 및 이후 챕터와의 연결](#10-전체-요약-및-이후-챕터와의-연결)

---

## 1. Monte Carlo Methods란

### DP의 한계를 어떻게 넘는가

Chapter 4의 Dynamic Programming은 강력하지만 결정적인 한계가 있습니다:

> **환경의 완전한 모델 $p(s', r \mid s, a)$를 알아야 한다.**

현실에서 이 모델을 정확히 아는 경우는 드뭅니다. **Monte Carlo (MC) Methods​**는 이 한계를 다음 방식으로 극복합니다:

> **실제 환경과 상호작용하여 얻은 샘플 에피소드만으로 $v_\pi$, $q_\pi$를 추정한다.**

### MC의 핵심 아이디어

$v_\pi(s) = \mathbb{E}_\pi[G_t \mid S_t = s]$이므로, 상태 $s$를 방문할 때마다 실제로 받은 return $G_t$를 기록하고 **평균​**을 내면 됩니다.

```
DP  : p(s',r|s,a)를 알고 → 기댓값을 계산
MC  : 실제 에피소드를 돌리고 → 샘플 평균으로 추정
```

### MC의 전제 조건

- **에피소드 단위(episodic)​**로 작동 — 에피소드가 끝나야 return을 계산 가능
- 모든 에피소드가 반드시 **종료(terminate)​**해야 함
- 충분히 많은 에피소드가 필요 (수렴은 보장되지만 속도는 느릴 수 있음)

---

## 2. Monte Carlo Policy Evaluation

### 문제 설정

고정된 policy $\pi$ 하에서 에피소드를 반복적으로 수집하고, 각 상태 $s$의 $v_\pi(s)$를 추정합니다.

에피소드 하나의 구조:

$$S_0, A_0, R_1, S_1, A_1, R_2, \ldots, S_{T-1}, A_{T-1}, R_T, S_T \text{ (terminal)}$$

상태 $s$가 time step $t$에 방문되었다면, 그 시점부터의 실제 return:

$$G_t = R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{T-t-1} R_T$$

이를 충분히 많이 모아 평균을 내면:

$$v_\pi(s) \approx \frac{1}{N(s)} \sum_{i=1}^{N(s)} G_t^{(i)}$$

대수의 법칙에 의해 $N(s) \to \infty$이면 $\hat{v}(s) \to v_\pi(s)$.

### First-visit vs. Every-visit MC

하나의 에피소드에서 상태 $s$가 여러 번 방문될 수 있습니다. 이를 처리하는 방식:

| 방식 | 설명 | 특징 |
|---|---|---|
| **First-visit MC** | 에피소드 내 첫 번째 방문의 $G_t$만 사용 | 이론적으로 unbiased, 분석 용이 |
| **Every-visit MC** | 모든 방문의 $G_t$를 사용 | 더 많은 데이터 활용, biased이지만 consistent |

두 방법 모두 $N(s) \to \infty$에서 $v_\pi(s)$로 수렴합니다.

### First-visit MC 알고리즘

```
Input: policy π
Initialize: V(s) ← arbitrary, Returns(s) ← empty list for all s

Loop (에피소드 반복):
    π를 따라 에피소드 생성: S_0, A_0, R_1, ..., S_T
    G ← 0
    For t = T-1, T-2, ..., 0:
        G ← γG + R_{t+1}
        If S_t가 S_0,...,S_{t-1}에 나타나지 않았다면:  ← first-visit 조건
            Returns(S_t).append(G)
            V(S_t) ← average(Returns(S_t))
```

### Incremental 업데이트

매번 리스트를 저장하지 않고 Chapter 2의 incremental update rule을 그대로 활용합니다:

$$V(S_t) \leftarrow V(S_t) + \frac{1}{N(S_t)}\left[G_t - V(S_t)\right]$$

Nonstationary 환경에서는 constant step-size $\alpha$를 사용합니다:

$$V(S_t) \leftarrow V(S_t) + \alpha\left[G_t - V(S_t)\right]$$

---

## 3. Monte Carlo Estimation of Action Values

### 왜 $q_\pi$가 필요한가

모델 $p$가 없으면 $v_\pi$만으로는 greedy policy improvement를 할 수 없습니다:

$$\pi'(s) = \arg\max_a \underbrace{\sum_{s',r} p(s',r|s,a)[r + \gamma v_\pi(s')]}_{\text{모델 p 필요!}}$$

반면 $q_\pi(s, a)$를 알면 모델 없이도 바로 improvement가 가능합니다:

$$\pi'(s) = \arg\max_a q_\pi(s, a) \quad \leftarrow \text{모델 불필요}$$

따라서 모델이 없는 MC에서는 $v_\pi$ 대신 **$q_\pi(s, a)$를 추정​**하는 것이 핵심입니다.

### State-Action Pair 방문

$q_\pi(s, a)$를 추정하려면 상태 $s$에서 행동 $a$를 취한 이후의 return을 수집해야 합니다. 즉, **(state, action) pair $(s, a)$의 방문​**을 카운트합니다.

문제: deterministic policy $\pi$를 따르면 많은 $(s, a)$ pair가 **전혀 방문되지 않아** 추정 불가능합니다. 이것이 다음 섹션의 **Exploring Starts​**와 **ε-soft policy​**의 동기입니다.

---

## 4. Monte Carlo Control: GPI with MC

### GPI 구조를 MC에 적용

Chapter 4의 GPI 틀을 그대로 유지하되, Evaluation 단계를 MC로 대체합니다:

```
π_0 →(MC Eval)→ q_π0 →(Greedy)→ π_1 →(MC Eval)→ q_π1 → ... → π_* , q_*
```

- **Evaluation**: MC로 $q_{\pi_k}$ 추정
- **Improvement**: $\pi_{k+1}(s) \leftarrow \arg\max_a q_{\pi_k}(s, a)$

### 두 가지 수렴 조건

MC Control이 $\pi_*$로 수렴하려면:

1. **무한한 에피소드**: 각 $(s, a)$ pair가 무한히 많이 방문되어야 $q_\pi$ 추정이 수렴
2. **Policy가 greedy로 수렴**: Evaluation과 Improvement가 균형을 맞춰야 함

이 두 조건을 동시에 만족하는 것이 MC Control의 핵심 과제입니다.

---

## 5. Exploring Starts

### 문제: 방문하지 않은 (s, a) pair

Deterministic policy를 따르면 각 상태에서 단 하나의 행동만 선택됩니다. 따라서 다른 행동들의 $q$ 값을 추정할 수가 없습니다.

### 해결책: Exploring Starts 가정

**에피소드의 시작 상태와 시작 행동을 무작위로 선택​**하여, 모든 $(s, a)$ pair가 시작점이 될 수 있도록 보장합니다:

$$P(S_0 = s, A_0 = a) > 0, \quad \forall s \in \mathcal{S},\ a \in \mathcal{A}(s)$$

이 가정 하에서 충분히 많은 에피소드를 수집하면 모든 $(s, a)$ pair를 방문하게 됩니다.

### MC Control with Exploring Starts (알고리즘)

```
Initialize:
    π(s) ← arbitrary deterministic for all s
    Q(s,a) ← arbitrary for all s, a
    Returns(s,a) ← empty list for all s, a

Loop (에피소드 반복):
    (S_0, A_0)를 무작위 선택 (exploring starts)
    π를 따라 에피소드 생성: S_0,A_0,R_1,...,S_T

    G ← 0
    For t = T-1, T-2, ..., 0:
        G ← γG + R_{t+1}
        If (S_t, A_t)가 앞서 나타나지 않았다면:
            Returns(S_t, A_t).append(G)
            Q(S_t, A_t) ← average(Returns(S_t, A_t))
            π(S_t) ← argmax_a Q(S_t, a)   ← 즉시 greedy update
```

### Exploring Starts의 한계

실제 환경에서는 시작 조건을 자유롭게 선택할 수 없는 경우가 많습니다. 이를 해결하는 두 가지 방향이 on-policy와 off-policy 방법입니다.

---

## 6. On-policy vs. Off-policy

탐색 문제를 해결하는 두 가지 근본적으로 다른 접근법입니다.

| | On-policy | Off-policy |
|---|---|---|
| **핵심 아이디어** | 개선하려는 policy 자체로 탐색 | 별도의 탐색 policy를 사용 |
| **사용 policy** | 하나 (탐색 + 개선 동시) | 둘 (behavior + target) |
| **대표 방법** | ε-greedy, ε-soft | Importance Sampling |
| **장점** | 단순함 | 더 유연한 탐색 가능 |
| **단점** | 탐색과 착취의 트레이드오프 | 높은 분산 |

### On-policy: ε-soft Policy

Deterministic greedy policy 대신 **ε-soft policy​**를 사용합니다:

$$\pi(a \mid s) = \begin{cases} 1 - \varepsilon + \dfrac{\varepsilon}{|\mathcal{A}(s)|} & \text{if } a = \arg\max_{a'} Q(s, a') \\ \dfrac{\varepsilon}{|\mathcal{A}(s)|} & \text{otherwise} \end{cases}$$

모든 행동에 최소 $\frac{\varepsilon}{|\mathcal{A}(s)|}$의 확률을 부여하여 탐색을 보장합니다.

### ε-soft Policy에서의 Improvement

**정리**: ε-soft policy에서 greedy improvement를 하면 항상 개선됩니다.

$\pi'$를 $Q_\pi$에 대한 ε-greedy policy라 하면:

$$q_\pi(s, \pi'(s)) = \sum_a \pi'(a|s)\, q_\pi(s,a)$$

$$= \frac{\varepsilon}{|\mathcal{A}|} \sum_a q_\pi(s,a) + (1-\varepsilon) \max_a q_\pi(s,a)$$

$$\geq \frac{\varepsilon}{|\mathcal{A}|} \sum_a q_\pi(s,a) + (1-\varepsilon) \sum_a \frac{\pi(a|s) - \frac{\varepsilon}{|\mathcal{A}|}}{1-\varepsilon} q_\pi(s,a)$$

$$= \sum_a \pi(a|s)\, q_\pi(s,a) = v_\pi(s)$$

따라서 Policy Improvement Theorem에 의해 $v_{\pi'} \geq v_\pi$. $\checkmark$

단, 수렴 지점은 $v_*$가 아닌 **ε-soft policy 중 최선​**인 $v_{\pi_\varepsilon^*}$입니다. ε-greedy를 쓰는 한 완전한 $v_*$에는 도달할 수 없습니다.

---

## 7. Off-policy Prediction: Importance Sampling

### 문제 설정

- **Target policy $\pi$**: 학습하고 싶은 policy (개선 대상)
- **Behavior policy $b$**: 실제 환경과 상호작용하며 데이터를 수집하는 policy

$b$로 수집한 에피소드로 $\pi$의 value function을 추정하려면, 두 policy의 **분포 차이를 보정​**해야 합니다.

**커버리지 조건 (Coverage Assumption)**:

$$\pi(a \mid s) > 0 \implies b(a \mid s) > 0, \quad \forall s, a$$

$\pi$가 선택할 수 있는 모든 행동을 $b$도 선택할 수 있어야 합니다.

### Importance Sampling Ratio 유도

에피소드의 time step $t$부터 $T$까지의 궤적 $\tau = (S_t, A_t, S_{t+1}, A_{t+1}, \ldots, S_T)$에 대해, $\pi$ 하에서의 확률과 $b$ 하에서의 확률의 비율을 계산합니다.

$\pi$ 하에서 궤적 $\tau$의 확률:

$$P^\pi(\tau) = \prod_{k=t}^{T-1} \pi(A_k \mid S_k)\, p(S_{k+1} \mid S_k, A_k)$$

$b$ 하에서의 확률:

$$P^b(\tau) = \prod_{k=t}^{T-1} b(A_k \mid S_k)\, p(S_{k+1} \mid S_k, A_k)$$

두 확률의 비율에서 전이 확률 $p$가 약분됩니다:

$$\boxed{\rho_{t:T-1} \doteq \frac{P^\pi(\tau)}{P^b(\tau)} = \prod_{k=t}^{T-1} \frac{\pi(A_k \mid S_k)}{b(A_k \mid S_k)}}$$

이것이 **importance sampling ratio​**입니다. 모델 $p$가 없어도 계산 가능하다는 점이 핵심입니다.

### Importance Sampling으로 $v_\pi$ 추정

$b$에서 얻은 return $G_t$를 $\rho_{t:T-1}$로 가중하면 $v_\pi(s)$의 unbiased estimate를 얻습니다:

$$\mathbb{E}_b\left[\rho_{t:T-1} G_t \mid S_t = s\right] = v_\pi(s)$$

**증명**:

$$\mathbb{E}_b\left[\rho_{t:T-1} G_t \mid S_t = s\right] = \mathbb{E}_b\left[\frac{P^\pi(\tau)}{P^b(\tau)} G_t \;\Big|\; S_t = s\right]$$

$$= \sum_\tau P^b(\tau) \cdot \frac{P^\pi(\tau)}{P^b(\tau)} \cdot G_t(\tau) = \sum_\tau P^\pi(\tau) \cdot G_t(\tau) = \mathbb{E}_\pi[G_t \mid S_t = s] = v_\pi(s) \checkmark$$

---

## 8. Importance Sampling 수렴 분석

### Ordinary IS vs. Weighted IS

$s$를 방문한 time step 집합을 $\mathcal{T}(s)$, 각 방문에서의 return을 $G_t$, IS ratio를 $\rho_t \doteq \rho_{t:T(t)-1}$로 표기합니다.

**Ordinary (Simple) IS**:

$$V(s) = \frac{\sum_{t \in \mathcal{T}(s)} \rho_t G_t}{|\mathcal{T}(s)|}$$

**Weighted IS**:

$$V(s) = \frac{\sum_{t \in \mathcal{T}(s)} \rho_t G_t}{\sum_{t \in \mathcal{T}(s)} \rho_t}$$

### 두 방법의 편향-분산 트레이드오프

**Ordinary IS**:
- **Unbiased**: $\mathbb{E}[V(s)] = v_\pi(s)$
- **분산 무한대 가능**: $\rho_t$가 매우 크거나 작을 때 분산이 폭발할 수 있음

**Weighted IS**:
- **Biased** (유한 샘플에서): 분자와 분모의 기댓값의 비율 $\neq$ 비율의 기댓값
- **Consistent**: 샘플 수 $\to \infty$이면 $v_\pi(s)$로 수렴
- **분산 유한**: 실제로는 분산이 훨씬 작아 **실용적으로 크게 선호​**됨

### Ordinary IS의 분산이 무한할 수 있음을 직관적으로 이해

$\pi$와 $b$가 크게 다를 때, $\rho_t = \prod_{k=t}^{T-1} \frac{\pi(A_k)}{b(A_k)}$는 에피소드 길이가 길수록 기하급수적으로 커지거나 작아질 수 있습니다. 예를 들어 매 step마다 $\frac{\pi}{b} = 2$이면 $T-t = 10$ step짜리 에피소드에서 $\rho = 2^{10} = 1024$. 이런 극단적인 가중치가 분산을 폭발시킵니다.

### Incremental Weighted IS 업데이트

에피소드가 쌓일수록 $V_n(s)$를 효율적으로 업데이트하는 점화식을 유도합니다.

$n$번째 방문까지의 누적 가중치 $C_n = \sum_{k=1}^n W_k$ ($W_k = \rho_{t_k}$)로 정의하면:

$$V_{n+1} = V_n + \frac{W_{n+1}}{C_{n+1}}\left[G_{n+1} - V_n\right]$$

$$C_{n+1} = C_n + W_{n+1}$$

이 형태는 Chapter 2의 incremental update rule $Q \leftarrow Q + \frac{1}{n}[R - Q]$와 동일한 구조입니다. 가중치 $\frac{1}{n}$ 대신 $\frac{W_{n+1}}{C_{n+1}}$로 대체된 것입니다.

---

## 9. Off-policy Control

### Off-policy MC Control 알고리즘

- **Target policy $\pi$**: deterministic greedy (개선 대상)
- **Behavior policy $b$**: 탐색을 위한 soft policy (예: ε-soft)

```
Initialize:
    Q(s,a) ← arbitrary for all s, a
    π(s) ← argmax_a Q(s,a) for all s  (greedy target policy)
    C(s,a) ← 0 for all s, a

Loop (에피소드 반복):
    b ← any soft policy (e.g., ε-greedy based on Q)
    b를 따라 에피소드 생성: S_0,A_0,R_1,...,S_T

    G ← 0
    W ← 1
    For t = T-1, T-2, ..., 0:
        G ← γG + R_{t+1}
        C(S_t, A_t) ← C(S_t, A_t) + W
        Q(S_t, A_t) ← Q(S_t, A_t) + W/C(S_t,A_t) [G - Q(S_t,A_t)]
        π(S_t) ← argmax_a Q(S_t, a)

        If A_t ≠ π(S_t): break   ← target policy와 다른 행동이면 중단
        W ← W × π(A_t|S_t) / b(A_t|S_t)
```

### 조기 종료(early exit)의 의미

Target policy $\pi$가 deterministic이면, $A_t \neq \pi(S_t)$인 순간부터의 IS ratio에 $\pi(A_t \mid S_t) = 0$이 포함되어 $W = 0$이 됩니다. 즉, 그 이후 데이터는 $\pi$ 추정에 기여하지 않으므로 즉시 다음 에피소드로 넘어가는 것이 효율적입니다.

이 특성은 Off-policy MC의 **데이터 효율성을 크게 저하​**시킬 수 있습니다 — 에피소드 후반부의 데이터 대부분이 버려집니다. 이 문제는 Chapter 6의 **TD learning​**으로 해결됩니다.

---

## 10. 전체 요약 및 이후 챕터와의 연결

### Chapter 5 구조 요약

```
Chapter 4: DP (모델 필요)
        ↓ 모델 없이 샘플로 대체
Chapter 5: Monte Carlo Methods
        │
        ├── Prediction (Policy Evaluation)
        │       v_π(s) ← 에피소드 return G_t의 샘플 평균
        │       First-visit / Every-visit MC
        │
        ├── Control (GPI with MC)
        │   ├── Exploring Starts
        │   │       모든 (s,a) pair에서 시작 보장
        │   │
        │   ├── On-policy: ε-soft policy
        │   │       탐색 + 착취를 하나의 policy로
        │   │
        │   └── Off-policy: Importance Sampling
        │           behavior policy b로 수집
        │           IS ratio ρ로 분포 보정 → v_π 추정
        │           Ordinary IS (unbiased, 고분산)
        │           Weighted IS (biased, 저분산, 실용적)
        │
        └── Off-policy Control
                Target π (greedy) + Behavior b (soft)
                Incremental weighted IS 업데이트
```

### 핵심 수식 한눈에 보기

**MC Value Estimation (Incremental)**:
$$V(S_t) \leftarrow V(S_t) + \alpha\left[G_t - V(S_t)\right]$$

**Importance Sampling Ratio**:
$$\rho_{t:T-1} = \prod_{k=t}^{T-1} \frac{\pi(A_k \mid S_k)}{b(A_k \mid S_k)}$$

**Ordinary IS Estimator**:
$$V(s) = \frac{\sum_{t \in \mathcal{T}(s)} \rho_t\, G_t}{|\mathcal{T}(s)|}$$

**Weighted IS Estimator**:
$$V(s) = \frac{\sum_{t \in \mathcal{T}(s)} \rho_t\, G_t}{\sum_{t \in \mathcal{T}(s)} \rho_t}$$

**Incremental Weighted IS Update**:
$$C \leftarrow C + W, \qquad V \leftarrow V + \frac{W}{C}\left[G - V\right]$$

### DP, MC, TD 비교 재확인

```
                모델 필요?    Bootstrap?    에피소드 완료 필요?
DP              ✅ 필요        ✅ 사용        ❌ 불필요
Monte Carlo     ❌ 불필요      ❌ 미사용      ✅ 필요
TD Learning     ❌ 불필요      ✅ 사용        ❌ 불필요
```

### MC의 핵심 한계 → Chapter 6으로

MC가 DP보다 낫지만, 여전히 두 가지 한계가 있습니다:

1. **에피소드가 끝나야 업데이트 가능** — 긴 에피소드나 continuing task에서 비효율적
2. **Off-policy에서 높은 분산** — IS ratio가 에피소드 길이에 따라 기하급수적으로 커짐

**TD learning​**은 이 두 문제를 모두 해결합니다:

- 에피소드가 끝나기 전에 **매 step마다 bootstrap​**으로 업데이트
- IS ratio를 **단 1 step​**에 대해서만 계산 → 분산 대폭 감소

| 문제 | MC의 해결 방식 | TD의 해결 방식 |
|---|---|---|
| 모델 불필요 | ✅ 샘플 사용 | ✅ 샘플 사용 |
| 에피소드 완료 | ✅ 필요 | ❌ 불필요 |
| IS 분산 | 높음 (긴 trajectory) | 낮음 (1-step ratio) |

---

> **다음 챕터로**: Chapter 6에서는 에피소드가 끝나기를 기다리지 않고 매 step마다 업데이트하는 **Temporal Difference (TD) Learning**, 그리고 그로부터 파생되는 **SARSA​**와 **Q-learning​**을 다룹니다.
