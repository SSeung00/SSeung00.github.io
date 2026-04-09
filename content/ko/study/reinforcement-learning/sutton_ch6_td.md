---
title: "Reinforcement Learning: Chapter 6 Temporal Difference Learning"
category: "Reinforcement Learning"
weight: 6
date: 2026-04-09
---

# Reinforcement Learning Chapter 6: Temporal Difference Learning

> Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed.) — Chapter 6 핵심 정리

---

## 목차

1. [TD Learning이란](#1-td-learning이란)
2. [TD(0): 가장 단순한 TD 방법](#2-td0-가장-단순한-td-방법)
3. [TD vs. MC vs. DP 비교](#3-td-vs-mc-vs-dp-비교)
4. [TD(0)의 수렴 증명](#4-td0의-수렴-증명)
5. [SARSA: On-policy TD Control](#5-sarsa-on-policy-td-control)
6. [Q-learning: Off-policy TD Control](#6-q-learning-off-policy-td-control)
7. [SARSA vs. Q-learning: Cliff Walking 예시](#7-sarsa-vs-q-learning-cliff-walking-예시)
8. [Expected SARSA](#8-expected-sarsa)
9. [Maximization Bias와 Double Q-learning](#9-maximization-bias와-double-q-learning)
10. [전체 요약 및 이후 챕터와의 연결](#10-전체-요약-및-이후-챕터와의-연결)

---

## 1. TD Learning이란

### MC와 DP의 한계를 동시에 극복

Chapter 5의 Monte Carlo는 모델이 없어도 되지만, 에피소드가 끝날 때까지 기다려야 합니다. Chapter 4의 DP는 매 step마다 업데이트할 수 있지만, 완전한 모델이 필요합니다.

**Temporal Difference (TD) Learning​**은 이 두 한계를 동시에 극복합니다:

```
DP   : 모델 O, bootstrap O, 에피소드 완료 불필요
MC   : 모델 X, bootstrap X, 에피소드 완료 필요
TD   : 모델 X, bootstrap O, 에피소드 완료 불필요  ← 두 장점 결합
```

### TD의 핵심 아이디어: Bootstrap

MC는 실제 return $G_t$를 target으로 씁니다:

$$V(S_t) \leftarrow V(S_t) + \alpha\left[G_t - V(S_t)\right]$$

TD는 **다음 상태의 추정값으로 현재를 업데이트​**합니다:

$$V(S_t) \leftarrow V(S_t) + \alpha\left[R_{t+1} + \gamma V(S_{t+1}) - V(S_t)\right]$$

$G_t$ 대신 $R_{t+1} + \gamma V(S_{t+1})$을 **TD target​**으로 씁니다. 에피소드가 끝나기 전에, 단 한 step만 진행하면 업데이트할 수 있습니다.

---

## 2. TD(0): 가장 단순한 TD 방법

### TD(0) 업데이트 규칙

$$\boxed{V(S_t) \leftarrow V(S_t) + \alpha\left[\underbrace{R_{t+1} + \gamma V(S_{t+1})}_{\text{TD target}} - V(S_t)\right]}$$

괄호 안의 항:

$$\delta_t \doteq R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$$

를 **TD error​**라 부릅니다. 현재 추정값과 1-step bootstrap target 사이의 차이입니다.

### 알고리즘

```
Input: policy π, step-size α ∈ (0,1], discount γ
Initialize: V(s) ← arbitrary for all s, V(terminal) ← 0

Loop (에피소드 반복):
    S ← 초기 상태
    Loop (step 반복, S가 terminal이 아닐 때까지):
        A ← π에 따라 행동 선택
        R, S' ← 환경에서 관측
        V(S) ← V(S) + α[R + γV(S') - V(S)]
        S ← S'
```

MC와의 차이: 에피소드가 끝나기 전에 **매 step마다** $V$를 업데이트합니다.

### TD target의 의미

$R_{t+1} + \gamma V(S_{t+1})$는 $v_\pi(S_t)$의 **biased estimate​**입니다:

- $V(S_{t+1})$이 아직 수렴하지 않았으면 정확하지 않음 (bias)
- 하지만 전체 에피소드를 기다리지 않아도 됨 (variance 감소)

이것이 TD의 **bias-variance tradeoff​**입니다. MC는 unbiased이지만 분산이 크고, TD는 biased이지만 분산이 작습니다.

---

## 3. TD vs. MC vs. DP 비교

### 업데이트 target 비교

세 방법의 차이는 **무엇을 target으로 쓰느냐​**에 있습니다:

$$\text{DP target:} \quad \sum_a \pi(a|s) \sum_{s',r} p(s',r|s,a)\left[r + \gamma V(s')\right]$$

$$\text{MC target:} \quad G_t = R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{T-t-1}R_T$$

$$\text{TD target:} \quad R_{t+1} + \gamma V(S_{t+1})$$

| | Target | 모델 | 에피소드 완료 | Bias | Variance |
|---|---|---|---|---|---|
| **DP** | 완전 기댓값 (bootstrap) | ✅ 필요 | ❌ | 없음 | 없음 |
| **MC** | 완전 return (샘플) | ❌ | ✅ | 없음 | 높음 |
| **TD** | 1-step bootstrap (샘플) | ❌ | ❌ | 있음 | 낮음 |

### Backup Diagram 비교

```
DP (전체 트리 전개)      MC (에피소드 끝까지)    TD (1-step)
        s                       s                    s
       /|\                      |                    |
      / | \                     |                    |
     s' s' s'                   |                   s'
    (모든 가능한 다음 상태)    (실제 궤적)        (다음 상태 1개)
```

---

## 4. TD(0)의 수렴 증명

### 수렴 정리

**정리**: 고정된 policy $\pi$ 하에서, step-size $\alpha_t$가 Robbins-Monro 조건을 만족하면:

$$\sum_{t=0}^{\infty} \alpha_t = \infty, \qquad \sum_{t=0}^{\infty} \alpha_t^2 < \infty$$

TD(0)은 $v_\pi$로 확률 1 수렴합니다.

### 핵심 아이디어: TD(0)를 선형 방정식으로 표현

기댓값 업데이트(expected update)를 분석합니다. $v_\pi$에서 TD target의 기댓값:

$$\mathbb{E}_\pi\left[R_{t+1} + \gamma V(S_{t+1}) \mid S_t = s\right] = \sum_a \pi(a|s)\sum_{s',r} p(s',r|s,a)\left[r + \gamma V(s')\right]$$

$= (T^\pi V)(s)$ (Bellman expectation operator)

따라서 TD(0)의 기댓값 업데이트는:

$$\mathbb{E}[V(S_t) + \alpha\,\delta_t \mid S_t = s] = V(s) + \alpha\left[(T^\pi V)(s) - V(s)\right]$$

$V = v_\pi$일 때, $T^\pi v_\pi = v_\pi$이므로 $\delta_t$의 기댓값은 0입니다. 즉, $v_\pi$는 TD(0) 업데이트의 **고정점​**입니다.

### $\gamma$-Contraction 적용

Chapter 4에서 $T^\pi$가 $\gamma$-contraction임을 증명했습니다:

$$\|T^\pi u - T^\pi v\|_\infty \leq \gamma \|u - v\|_\infty$$

TD(0)은 $T^\pi$를 샘플로 근사하므로, 확률적 근사 이론(stochastic approximation theory)에 의해 Robbins-Monro 조건 하에서 고정점 $v_\pi$로 수렴이 보장됩니다.

### TD vs. MC의 수렴 비교

| | 수렴 대상 | 수렴 조건 | 수렴 속도 |
|---|---|---|---|
| **MC** | $v_\pi$ | 충분한 방문 횟수 | 느림 (고분산) |
| **TD(0)** | $v_\pi$ | Robbins-Monro 조건 | 빠름 (저분산) |

단, **constant $\alpha$** 사용 시 TD(0)은 $v_\pi$에 정확히 수렴하지 않고 그 근방에서 진동합니다.

---

## 5. SARSA: On-policy TD Control

### State-Value → Action-Value

Control 문제에서는 모델 없이 policy를 개선해야 하므로 $V(s)$ 대신 $Q(s, a)$를 추정합니다.

TD(0)를 action-value로 확장하면:

$$\boxed{Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha\left[R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)\right]}$$

업데이트에 필요한 다섯 가지 요소 $(S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1})$의 이름을 따서 **SARSA​**라 부릅니다.

### 알고리즘

```
Initialize: Q(s,a) ← arbitrary for all s,a; Q(terminal,·) ← 0

Loop (에피소드 반복):
    S ← 초기 상태
    A ← Q에 대한 ε-greedy로 선택

    Loop (step 반복, S가 terminal이 아닐 때까지):
        R, S' ← 환경에서 관측
        A' ← Q에 대한 ε-greedy로 선택 (S'에서)
        Q(S,A) ← Q(S,A) + α[R + γQ(S',A') - Q(S,A)]
        S ← S', A ← A'
```

### On-policy의 의미

SARSA는 **행동을 선택하는 policy와 평가하는 policy가 동일​**합니다. ε-greedy로 행동하면서 동시에 ε-greedy policy의 $q$ 값을 추정합니다.

**수렴 조건**:
- 모든 $(s, a)$ pair가 무한히 방문됨
- $\varepsilon \to 0$ (GLIE: Greedy in the Limit with Infinite Exploration)
- Robbins-Monro step-size 조건

이 조건들이 만족되면 SARSA는 최적 policy $\pi_*$와 $q_*$로 수렴합니다.

---

## 6. Q-learning: Off-policy TD Control

### Q-learning 업데이트 규칙

$$\boxed{Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha\left[R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a') - Q(S_t, A_t)\right]}$$

SARSA와의 유일한 차이: $Q(S_{t+1}, A_{t+1})$ 대신 $\max_{a'} Q(S_{t+1}, a')$.

### Off-policy인 이유

Q-learning의 TD target은:

$$R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a')$$

이것은 **greedy policy (target policy)​**의 action-value를 직접 추정합니다. 실제 행동 $A_{t+1}$이 무엇이든 상관없이, 항상 $S_{t+1}$에서 최대 Q 값을 사용합니다.

따라서 행동 수집에 어떤 behavior policy $b$를 써도 (충분한 탐색만 보장된다면) target policy $\pi_* = \text{greedy}$의 $q_*$를 추정합니다. **Importance Sampling 없이 off-policy​**가 가능한 이유입니다.

### 알고리즘

```
Initialize: Q(s,a) ← arbitrary for all s,a; Q(terminal,·) ← 0

Loop (에피소드 반복):
    S ← 초기 상태
    Loop (step 반복, S가 terminal이 아닐 때까지):
        A ← Q에 대한 ε-greedy로 선택  ← behavior policy
        R, S' ← 환경에서 관측
        Q(S,A) ← Q(S,A) + α[R + γ max_{a'} Q(S',a') - Q(S,A)]
        S ← S'
```

### Q-learning의 수렴 증명 스케치

Q-learning의 업데이트를 operator $T^*$의 샘플 근사로 볼 수 있습니다:

$$\mathbb{E}\left[\max_{a'} Q(S_{t+1}, a') \;\Big|\; S_t, A_t\right] = \sum_{s',r} p(s',r|S_t,A_t)\left[r + \gamma \max_{a'} Q(s', a')\right] = (T^* Q)(S_t, A_t)$$

$T^*$는 $\gamma$-contraction이고, $q_*$는 $T^*$의 유일한 고정점입니다 (Chapter 4). 따라서 Robbins-Monro 조건과 충분한 방문 횟수 조건 하에서:

$$Q \to q_* \qquad \text{확률 1}$$

---

## 7. SARSA vs. Q-learning: Cliff Walking 예시

### 환경 설정

```
┌───────────────────────────┐
│  . . . . . . . . . . . .  │
│  S C C C C C C C C C C G  │
└───────────────────────────┘
S: 시작, G: 목표, C: 절벽 (보상 -100, 시작으로 리셋)
일반 이동: 보상 -1
```

### 두 알고리즘의 학습된 경로 차이

**SARSA (on-policy)**:
- ε-greedy로 탐색하므로 가끔 절벽 근처에서 실수 가능성이 있음을 인식
- **절벽에서 멀리 떨어진 안전한 경로** 학습
- 에피소드당 평균 보상은 낮지만 (돌아가므로), **실제 운용 시 안전**

**Q-learning (off-policy)**:
- 항상 $\max_{a'}$로 업데이트 → greedy policy 기준으로 최적
- 탐색(ε-random)으로 인한 절벽 추락을 Q 업데이트에 반영하지 않음
- **절벽 바로 옆 최단 경로** 학습
- 이론적으로 최적이지만 **ε-greedy 실행 중에는 자주 추락**

### 핵심 통찰

| | SARSA | Q-learning |
|---|---|---|
| 학습 target | 현재 policy (ε-greedy) | Greedy policy |
| 학습된 경로 | 안전한 우회로 | 위험한 최단 경로 |
| 수렴 대상 | $q_{\pi_\varepsilon}$ (ε-soft 최적) | $q_*$ (진짜 최적) |
| 실제 성능 | 탐색 중에도 안정적 | 이론적 최적, 탐색 중 불안정 |

> SARSA는 **탐색하면서 발생하는 실수까지 고려​**한 policy를 학습합니다. Q-learning은 **탐색을 무시한 이상적 최적 policy​**를 학습합니다.

---

## 8. Expected SARSA

### 아이디어

SARSA의 업데이트에서 $Q(S_{t+1}, A_{t+1})$은 $A_{t+1}$의 **단일 샘플​**입니다. 이 분산을 줄이기 위해 $A_{t+1}$에 대한 **기댓값​**을 직접 계산합니다:

$$\boxed{Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha\left[R_{t+1} + \gamma \sum_{a'} \pi(a' \mid S_{t+1})\, Q(S_{t+1}, a') - Q(S_t, A_t)\right]}$$

### SARSA, Q-learning, Expected SARSA의 관계

세 알고리즘을 하나의 스펙트럼으로 이해할 수 있습니다:

$$\text{SARSA: } Q(S_{t+1}, A_{t+1}) \quad\leftarrow\quad A_{t+1} \sim \pi(\cdot|S_{t+1}) \text{ 샘플}$$

$$\text{Expected SARSA: } \sum_{a'} \pi(a'|S_{t+1})\,Q(S_{t+1},a') \quad\leftarrow\quad \text{기댓값}$$

$$\text{Q-learning: } \max_{a'} Q(S_{t+1}, a') \quad\leftarrow\quad \pi = \text{greedy일 때 Expected SARSA}$$

즉, **Q-learning은 target policy가 greedy일 때의 Expected SARSA​**입니다.

### Expected SARSA의 장점

- SARSA보다 **분산이 낮음** — $A_{t+1}$ 샘플링의 불확실성 제거
- On-policy로도, off-policy로도 사용 가능
- 계산량은 SARSA보다 약간 높지만 (기댓값 계산), 보통 성능이 더 좋음

---

## 9. Maximization Bias와 Double Q-learning

### Maximization Bias란

Q-learning (그리고 greedy action selection 일반)에서 나타나는 체계적 편향입니다.

**문제**: $\max_{a'} Q(s', a')$는 $\max_{a'} q(s', a')$의 **overestimate​**입니다.

직관적으로, $Q(s', a')$는 $q(s', a')$의 노이즈 있는 추정값입니다. 여러 행동 중 최댓값을 취하면 **노이즈의 양의 방향 편차가 선택​**될 가능성이 높아집니다.

### 수학적 확인

$q(s', a) = 0$이고 $Q(s', a) \sim \mathcal{N}(0, \sigma^2)$이라 가정하면:

$$\mathbb{E}\left[\max_a Q(s', a)\right] > \max_a \mathbb{E}[Q(s', a)] = 0$$

$\max$ 연산자와 기댓값은 교환되지 않습니다 (Jensen's inequality: $f$가 convex이면 $f(\mathbb{E}[X]) \leq \mathbb{E}[f(X)]$, 여기서 $\max$는 convex).

### Double Q-learning

**핵심 아이디어**: 행동을 **선택​**하는 Q와 행동을 **평가​**하는 Q를 분리합니다.

두 개의 독립적인 추정값 $Q_1$, $Q_2$를 유지합니다:
- $Q_1$으로 greedy action 선택: $A^* = \arg\max_a Q_1(S_{t+1}, a)$
- $Q_2$로 그 행동의 가치 평가: $Q_2(S_{t+1}, A^*)$

$$\boxed{Q_1(S_t, A_t) \leftarrow Q_1(S_t, A_t) + \alpha\left[R_{t+1} + \gamma Q_2(S_{t+1}, \arg\max_a Q_1(S_{t+1},a)) - Q_1(S_t, A_t)\right]}$$

각 step에서 $Q_1$과 $Q_2$ 중 하나를 50%로 무작위 선택하여 업데이트합니다.

### Double Q-learning이 bias를 줄이는 이유

$Q_1$과 $Q_2$가 독립적으로 추정되면:

$$\mathbb{E}\left[Q_2(s', \arg\max_a Q_1(s',a))\right] \approx q(s', \arg\max_a q(s',a))$$

선택과 평가에 독립적인 데이터를 사용하므로 양의 편향이 상쇄됩니다.

---

## 10. 전체 요약 및 이후 챕터와의 연결

### Chapter 6 구조 요약

```
Chapter 5: MC (모델 X, bootstrap X, 에피소드 완료 필요)
        ↓ bootstrap 도입 → 매 step 업데이트
Chapter 6: TD Learning
        │
        ├── Prediction: TD(0)
        │       δ_t = R_{t+1} + γV(S_{t+1}) - V(S_t)  ← TD error
        │       수렴 근거: T^π의 γ-contraction + 확률적 근사
        │
        ├── Control
        │   ├── SARSA (On-policy)
        │   │       (S,A,R,S',A') → Q(S,A) 업데이트
        │   │       탐색 policy = 평가 policy (ε-greedy)
        │   │
        │   ├── Q-learning (Off-policy)
        │   │       max_{a'} Q(S',a') → IS 없이 off-policy
        │   │       Value Iteration의 샘플 근사
        │   │
        │   └── Expected SARSA
        │           Σ_{a'} π(a'|S') Q(S',a')
        │           Q-learning = greedy target일 때 Expected SARSA
        │
        └── Double Q-learning
                Maximization bias 해결
                선택(Q_1)과 평가(Q_2) 분리
```

### 핵심 수식 한눈에 보기

**TD error**:
$$\delta_t \doteq R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$$

**TD(0) update**:
$$V(S_t) \leftarrow V(S_t) + \alpha\,\delta_t$$

**SARSA**:
$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha\left[R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)\right]$$

**Q-learning**:
$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha\left[R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a') - Q(S_t, A_t)\right]$$

**Expected SARSA**:
$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha\left[R_{t+1} + \gamma \sum_{a'} \pi(a'|S_{t+1})\,Q(S_{t+1}, a') - Q(S_t, A_t)\right]$$

**Double Q-learning**:
$$Q_1(S_t, A_t) \leftarrow Q_1(S_t, A_t) + \alpha\left[R_{t+1} + \gamma\, Q_2\!\left(S_{t+1}, \arg\max_a Q_1(S_{t+1},a)\right) - Q_1(S_t, A_t)\right]$$

### 세 방법 최종 비교

```
                모델 필요?    Bootstrap?    에피소드 완료?    Bias    Variance
DP              ✅             ✅             ❌               낮음    낮음
Monte Carlo     ❌             ❌             ✅               낮음    높음
TD              ❌             ✅             ❌               있음    낮음
```

### 이후 챕터로의 연결

Chapter 6까지는 **표(table) 형태​**로 $V(s)$, $Q(s,a)$를 저장했습니다. 상태·행동 공간이 크거나 연속적이면 이 방식은 불가능합니다. 이후 챕터들은 이 한계를 넘어갑니다:

| 챕터 | 주제 | Chapter 6와의 관계 |
|---|---|---|
| Ch.7 n-step TD | n-step return | TD(0)과 MC 사이의 스펙트럼 |
| Ch.8 Planning | Dyna-Q | Q-learning + 모델 기반 계획 |
| Ch.9~10 Function Approx. | 신경망으로 $Q$ 근사 | Q-learning + 함수 근사 → DQN |
| Ch.13 Policy Gradient | $\pi$를 직접 최적화 | Actor-Critic = TD error + policy gradient |

특히 **TD error $\delta_t$​**는 Chapter 13의 Actor-Critic에서 **advantage function의 추정값​**으로 그대로 재등장합니다. Chapter 6의 TD error 개념을 잘 이해해두면 이후 deep RL 알고리즘들을 훨씬 자연스럽게 이해할 수 있습니다.

---

> **다음 챕터로**: Chapter 7에서는 1-step TD와 Monte Carlo 사이의 스펙트럼을 이어주는 **n-step TD** 방법을 다루며, 얼마나 먼 미래까지 bootstrap할지를 하이퍼파라미터 $n$으로 조절합니다.
