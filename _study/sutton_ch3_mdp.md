---
layout: post
title: "Reinforcement Learning: Chapter 3 Finite Markov Decision Processes"
category: Reinforcement Learning
date: 2026-04-09
---

# Reinforcement Learning: Chapter 3 Finite Markov Decision Processes

> Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed.) — Chapter 3 핵심 정리

---

## 목차

1. [MDP란 무엇인가](#1-mdp란-무엇인가)
2. [MDP의 수학적 정의](#2-mdp의-수학적-정의)
3. [Goals and Rewards: 보상 가설](#3-goals-and-rewards-보상-가설)
4. [Return: 누적 보상의 정의](#4-return-누적-보상의-정의)
5. [Episodic vs. Continuing Tasks의 통합](#5-episodic-vs-continuing-tasks의-통합)
6. [Policy와 Value Function](#6-policy와-value-function)
7. [Bellman Equation 유도](#7-bellman-equation-유도)
8. [Optimal Policy와 Optimal Value Function](#8-optimal-policy와-optimal-value-function)
9. [Bellman Optimality Equation 유도](#9-bellman-optimality-equation-유도)
10. [전체 요약 및 이후 챕터와의 연결](#10-전체-요약-및-이후-챕터와의-연결)

---

## 1. MDP란 무엇인가

Chapter 2의 k-armed Bandit은 **상태(state)가 없고**, 행동이 미래에 영향을 미치지 않는 단순한 문제였습니다. Chapter 3에서는 이를 완전한 강화학습 문제로 확장합니다.

**Markov Decision Process (MDP)**의 핵심 특징:

- Agent는 **상태(state) $S_t$**를 관측하고
- **행동(action) $A_t$**을 선택하며
- 환경은 **보상(reward) $R_{t+1}$**과 **다음 상태 $S_{t+1}$**을 돌려줌
- 행동이 **미래 상태와 보상에 영향**을 미침 ← Bandit과의 결정적 차이

```
        행동 A_t
Agent ──────────────→ Environment
      ←──────────────
        상태 S_{t+1}
        보상 R_{t+1}
```

이 상호작용이 매 time step $t = 0, 1, 2, \ldots$마다 반복되며 **궤적(trajectory)**을 형성합니다:

$$S_0, A_0, R_1, S_1, A_1, R_2, S_2, A_2, R_3, \ldots$$

---

## 2. MDP의 수학적 정의

### Markov Property

MDP의 핵심 가정은 **Markov 성질**입니다:

$$P(S_{t+1}, R_{t+1} \mid S_t, A_t) = P(S_{t+1}, R_{t+1} \mid S_0, A_0, \ldots, S_t, A_t)$$

> **"현재 상태가 미래를 결정하는 데 충분하다 — 과거는 현재 상태에 이미 요약되어 있다."**

이 성질이 성립하기 때문에 과거 전체 이력을 기억할 필요 없이 **현재 상태만으로 최적 결정**이 가능합니다.

### 전이 함수 $p$

MDP의 dynamics를 완전히 기술하는 함수:

$$\boxed{p(s', r \mid s, a) \doteq P(S_{t+1}=s', R_{t+1}=r \mid S_t=s, A_t=a)}$$

이 하나의 함수로부터 필요한 모든 양을 유도할 수 있습니다:

**상태 전이 확률**:

$$p(s' \mid s, a) = \sum_{r \in \mathcal{R}} p(s', r \mid s, a)$$

**기대 보상**:

$$r(s, a) = \mathbb{E}[R_{t+1} \mid S_t=s, A_t=a] = \sum_{r \in \mathcal{R}} r \sum_{s' \in \mathcal{S}} p(s', r \mid s, a)$$

**다음 상태에서의 기대 보상**:

$$r(s, a, s') = \mathbb{E}[R_{t+1} \mid S_t=s, A_t=a, S_{t+1}=s'] = \frac{\sum_{r} r \cdot p(s', r \mid s, a)}{p(s' \mid s, a)}$$

### MDP의 구성 요소

$$\text{MDP} = (\mathcal{S},\ \mathcal{A},\ p,\ \mathcal{R},\ \gamma)$$

| 기호 | 의미 |
|---|---|
| $\mathcal{S}$ | 상태 공간 (유한) |
| $\mathcal{A}(s)$ | 상태 $s$에서 가능한 행동의 집합 |
| $p(s', r \mid s, a)$ | 전이 확률 (dynamics) |
| $\mathcal{R} \subset \mathbb{R}$ | 보상의 집합 |
| $\gamma \in [0, 1]$ | 할인율 (discount factor) |

---

## 3. Goals and Rewards: 보상 가설

강화학습의 목표는 다음 **보상 가설(Reward Hypothesis)**로 요약됩니다:

> **"Agent의 목표는 미래에 받을 누적 보상의 기댓값을 최대화하는 것으로 표현될 수 있다."**

### 보상 설계의 원칙

보상은 **무엇을 달성하길 원하는가**를 표현해야 하며, **어떻게 달성하는가**를 표현해서는 안 됩니다.

예시:
- 체스: 이기면 +1, 지면 -1, 그 외 0 (중간 전략은 agent가 학습)
- 로봇 보행: 빨리 걸을수록 양의 보상 (특정 걸음걸이를 지정하지 않음)
- ❌ 잘못된 예: 체스에서 상대방 말을 잡을 때마다 보상 → agent가 승리보다 말 잡기를 목표로 삼을 수 있음

---

## 4. Return: 누적 보상의 정의

### Episodic Task

명확한 종료(terminal state)가 있는 task. 각 **에피소드(episode)**는 독립적입니다.

단순 return:

$$G_t = R_{t+1} + R_{t+2} + \cdots + R_T$$

### Continuing Task와 할인율 $\gamma$

종료가 없는 task에서는 $G_t$가 발산할 수 있습니다. 이를 해결하기 위해 **할인율 $\gamma \in [0, 1)$**을 도입:

$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$

### 할인율의 의미

- $\gamma = 0$: 완전 근시안적 — 지금 당장의 보상만 고려
- $\gamma \to 1$: 미래 보상을 현재와 동등하게 고려
- $\gamma < 1$: $k$ 스텝 후 보상은 현재 가치의 $\gamma^k$배

**$\gamma < 1$이면 bounded 보상에서 $G_t$가 수렴함을 증명**:

보상이 $|R_t| \leq R_{\max}$로 bounded이면:

$$|G_t| \leq \sum_{k=0}^{\infty} \gamma^k R_{\max} = \frac{R_{\max}}{1-\gamma} < \infty \quad (\gamma < 1)$$

등비급수 공식 $\sum_{k=0}^{\infty} \gamma^k = \frac{1}{1-\gamma}$ 사용.

### Return의 점화식

$$\boxed{G_t = R_{t+1} + \gamma G_{t+1}}$$

**유도**:

$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots$$

$$= R_{t+1} + \gamma(R_{t+2} + \gamma R_{t+3} + \cdots)$$

$$= R_{t+1} + \gamma G_{t+1}$$

이 점화식은 이후 **Bellman Equation** 유도의 직접적인 출발점이 됩니다.

---

## 5. Episodic vs. Continuing Tasks의 통합

Terminal state에서 $G_T = 0$으로 정의하면, 두 종류의 task를 하나의 notation으로 통합할 수 있습니다:

$$G_t = \sum_{k=t+1}^{T} \gamma^{k-t-1} R_k$$

- Episodic: $T < \infty$, $\gamma = 1$ 가능
- Continuing: $T = \infty$, $\gamma < 1$ 필수

---

## 6. Policy와 Value Function

### Policy $\pi$

Policy는 상태를 행동의 확률 분포로 매핑하는 함수입니다:

$$\pi(a \mid s) = P(A_t = a \mid S_t = s)$$

- **Deterministic policy**: $\pi(s) = a$ (확률 아닌 직접 행동 반환)
- **Stochastic policy**: $\pi(a \mid s) \in [0, 1]$, $\sum_a \pi(a \mid s) = 1$

### State-Value Function $v_\pi(s)$

Policy $\pi$를 따를 때 상태 $s$에서의 **기대 return**:

$$\boxed{v_\pi(s) \doteq \mathbb{E}_\pi[G_t \mid S_t = s] = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \;\Big|\; S_t = s\right]}$$

### Action-Value Function $q_\pi(s, a)$

Policy $\pi$를 따를 때 상태 $s$에서 행동 $a$를 취한 후의 **기대 return**:

$$\boxed{q_\pi(s, a) \doteq \mathbb{E}_\pi[G_t \mid S_t = s, A_t = a] = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \;\Big|\; S_t = s, A_t = a\right]}$$

### $v_\pi$와 $q_\pi$의 관계

$$v_\pi(s) = \sum_a \pi(a \mid s)\, q_\pi(s, a)$$

$$q_\pi(s, a) = \sum_{s', r} p(s', r \mid s, a)\left[r + \gamma v_\pi(s')\right]$$

---

## 7. Bellman Equation 유도

### $v_\pi$의 Bellman Equation

**Chapter 3에서 가장 중요한 유도입니다.**

$v_\pi(s)$를 점화식 $G_t = R_{t+1} + \gamma G_{t+1}$을 이용해 전개합니다:

$$v_\pi(s) = \mathbb{E}_\pi[G_t \mid S_t = s]$$

$$= \mathbb{E}_\pi[R_{t+1} + \gamma G_{t+1} \mid S_t = s]$$

$$= \mathbb{E}_\pi[R_{t+1} + \gamma v_\pi(S_{t+1}) \mid S_t = s]$$

기댓값을 명시적으로 전개하면 ($A_t \sim \pi(\cdot \mid s)$, $(S_{t+1}, R_{t+1}) \sim p(\cdot \mid s, A_t)$):

$$= \sum_a \pi(a \mid s) \sum_{s', r} p(s', r \mid s, a)\left[r + \gamma v_\pi(s')\right]$$

$$\boxed{v_\pi(s) = \sum_a \pi(a \mid s) \sum_{s', r} p(s', r \mid s, a)\left[r + \gamma v_\pi(s')\right]}$$

### Bellman Equation의 의미

이 식은 **현재 상태의 가치를 다음 상태들의 가치로 표현**합니다:

```
v_π(s) = (현재 보상의 기대값) + γ × (다음 상태 가치의 기대값)
       = 즉각 보상 + 할인된 미래 가치
```

이를 **backup diagram**으로 시각화하면:

```
        s
       /|\
      / | \
     a  a  a      ← 행동 선택 (π에 따라)
    /|\ | /|\
   s' s' s' s'    ← 상태 전이 (p에 따라)
   r  r  r  r     ← 보상
```

### $q_\pi$의 Bellman Equation

같은 방식으로 유도합니다:

$$q_\pi(s, a) = \mathbb{E}_\pi[R_{t+1} + \gamma G_{t+1} \mid S_t=s, A_t=a]$$

$$= \sum_{s', r} p(s', r \mid s, a)\left[r + \gamma \mathbb{E}_\pi[G_{t+1} \mid S_{t+1}=s']\right]$$

$$= \sum_{s', r} p(s', r \mid s, a)\left[r + \gamma \sum_{a'} \pi(a' \mid s') q_\pi(s', a')\right]$$

$$\boxed{q_\pi(s, a) = \sum_{s', r} p(s', r \mid s, a)\left[r + \gamma \sum_{a'} \pi(a' \mid s')\, q_\pi(s', a')\right]}$$

### Bellman Equation은 연립 선형 방정식

$|\mathcal{S}| = n$이면, $v_\pi$에 대한 Bellman equation은 **$n$개의 미지수를 가진 $n$개의 선형 방정식**입니다:

$$\mathbf{v}_\pi = \mathbf{r}_\pi + \gamma \mathbf{P}_\pi \mathbf{v}_\pi$$

여기서:
- $\mathbf{v}_\pi \in \mathbb{R}^n$: 각 상태의 가치 벡터
- $\mathbf{r}_\pi \in \mathbb{R}^n$: 각 상태에서의 기대 즉각 보상
- $\mathbf{P}_\pi \in \mathbb{R}^{n \times n}$: 정책 $\pi$ 하의 전이 행렬

이를 직접 풀면:

$$(\mathbf{I} - \gamma \mathbf{P}_\pi)\mathbf{v}_\pi = \mathbf{r}_\pi$$

$$\mathbf{v}_\pi = (\mathbf{I} - \gamma \mathbf{P}_\pi)^{-1} \mathbf{r}_\pi$$

$\gamma < 1$이면 $(\mathbf{I} - \gamma \mathbf{P}_\pi)$는 항상 가역(invertible)입니다. 하지만 상태 공간이 크면 역행렬 계산이 비현실적 → 이후 챕터에서 **반복적(iterative) 방법**으로 해결합니다.

---

## 8. Optimal Policy와 Optimal Value Function

### Policy의 부분 순서

Policy $\pi \geq \pi'$의 의미:

$$\pi \geq \pi' \iff v_\pi(s) \geq v_{\pi'}(s), \quad \forall s \in \mathcal{S}$$

**정리**: 유한 MDP에서는 항상 **최적 정책(optimal policy) $\pi_*$**가 존재하며, 모든 $s$에서 다른 모든 policy보다 좋거나 같습니다.

### Optimal Value Functions

**Optimal state-value function**:

$$\boxed{v_*(s) \doteq \max_\pi v_\pi(s), \quad \forall s \in \mathcal{S}}$$

**Optimal action-value function**:

$$\boxed{q_*(s, a) \doteq \max_\pi q_\pi(s, a), \quad \forall s \in \mathcal{S},\ a \in \mathcal{A}(s)}$$

### $v_*$와 $q_*$의 관계

$$v_*(s) = \max_a q_*(s, a)$$

$$q_*(s, a) = \mathbb{E}\left[R_{t+1} + \gamma v_*(S_{t+1}) \mid S_t=s, A_t=a\right]$$

$$= \sum_{s', r} p(s', r \mid s, a)\left[r + \gamma v_*(s')\right]$$

---

## 9. Bellman Optimality Equation 유도

### $v_*$의 Bellman Optimality Equation

최적 정책 하에서는 행동을 **가치가 최대인 방향으로만** 선택합니다. 따라서 $\pi$의 평균 대신 $\max$를 취합니다:

$$v_*(s) = \max_a q_*(s, a)$$

$$= \max_a \mathbb{E}[R_{t+1} + \gamma v_*(S_{t+1}) \mid S_t=s, A_t=a]$$

$$\boxed{v_*(s) = \max_a \sum_{s', r} p(s', r \mid s, a)\left[r + \gamma v_*(s')\right]}$$

### $q_*$의 Bellman Optimality Equation

$$q_*(s, a) = \mathbb{E}\left[R_{t+1} + \gamma \max_{a'} q_*(S_{t+1}, a') \;\Big|\; S_t=s, A_t=a\right]$$

$$\boxed{q_*(s, a) = \sum_{s', r} p(s', r \mid s, a)\left[r + \gamma \max_{a'} q_*(s', a')\right]}$$

### Bellman Equation vs. Bellman Optimality Equation 비교

| | Bellman Equation | Bellman Optimality Equation |
|---|---|---|
| 대상 | $v_\pi$, $q_\pi$ (특정 policy) | $v_*$, $q_*$ (최적) |
| 행동 선택 | $\sum_a \pi(a \mid s)$ (평균) | $\max_a$ |
| 방정식 형태 | 선형 → 직접 풀기 가능 | 비선형 → 일반적으로 반복법 필요 |
| 해의 유일성 | 고정된 $\pi$에 대해 유일 | 유한 MDP에서 유일 |

### Bellman Optimality Equation은 비선형

$\max$ 연산자 때문에 선형 방정식이 아닙니다. 일반적인 닫힌 해(closed-form solution)가 없으므로 이후 챕터들에서 다양한 반복적 방법을 통해 해결합니다:

- **Chapter 4**: Dynamic Programming (모델 알고 있을 때)
- **Chapter 5**: Monte Carlo Methods (모델 모를 때, episodic)
- **Chapter 6**: TD Learning (모델 모를 때, online)

### Optimal Policy 도출

$v_*$를 구하면 optimal policy는 간단히 유도됩니다:

$$\pi_*(s) = \arg\max_a \sum_{s', r} p(s', r \mid s, a)\left[r + \gamma v_*(s')\right]$$

$q_*$를 구하면 더욱 단순합니다 (모델 $p$ 불필요):

$$\pi_*(s) = \arg\max_a q_*(s, a)$$

이것이 $q_*$ (action-value function)가 실용적으로 더 유용한 이유입니다.

---

## 10. 전체 요약 및 이후 챕터와의 연결

### Chapter 3 구조 요약

```
Bandit (Ch.2)
    │  상태 추가, 행동이 미래에 영향
    ↓
MDP 정의: (S, A, p, R, γ)
    │
    ├── Markov Property: 현재 상태로 미래를 결정
    ├── 전이 함수 p(s', r | s, a)
    │
    ├── Return G_t = R_{t+1} + γG_{t+1}  ← 핵심 점화식
    │
    ├── Policy π(a|s)
    ├── v_π(s), q_π(s,a)  ← value functions
    │
    ├── Bellman Equation  ← 현재↔미래 가치 연결
    │       v_π = Σ_a π Σ_{s',r} p [r + γv_π(s')]
    │
    ├── v_*(s), q_*(s,a)  ← optimal value functions
    │
    └── Bellman Optimality Equation
            v_* = max_a Σ_{s',r} p [r + γv_*(s')]
```

### 핵심 수식 한눈에 보기

**Return 점화식**:
$$G_t = R_{t+1} + \gamma G_{t+1}$$

**Bellman Equation** ($v_\pi$):
$$v_\pi(s) = \sum_a \pi(a|s) \sum_{s',r} p(s',r|s,a)\left[r + \gamma v_\pi(s')\right]$$

**Bellman Equation** ($q_\pi$):
$$q_\pi(s,a) = \sum_{s',r} p(s',r|s,a)\left[r + \gamma \sum_{a'}\pi(a'|s')\,q_\pi(s',a')\right]$$

**Bellman Optimality Equation** ($v_*$):
$$v_*(s) = \max_a \sum_{s',r} p(s',r|s,a)\left[r + \gamma v_*(s')\right]$$

**Bellman Optimality Equation** ($q_*$):
$$q_*(s,a) = \sum_{s',r} p(s',r|s,a)\left[r + \gamma \max_{a'} q_*(s',a')\right]$$

### 이후 챕터로의 연결

| 챕터 | 주제 | Chapter 3와의 연결 |
|---|---|---|
| Ch.4 Dynamic Programming | Policy Evaluation, Value Iteration | Bellman Equation을 반복적으로 적용하여 $v_\pi$, $v_*$ 계산 |
| Ch.5 Monte Carlo | 샘플 기반 가치 추정 | $q_\pi$ 추정 → greedy improvement |
| Ch.6 TD Learning | Q-learning, SARSA | Bellman Optimality Eq.을 샘플로 근사 |
| Ch.13 Policy Gradient | 직접 $\pi$ 최적화 | $v_\pi$, $q_\pi$를 기반으로 gradient 계산 |

### Bellman Equation이 RL의 중심인 이유

```
Bellman Equation
    │
    ├── Policy Evaluation → Prediction 문제
    │       "주어진 π로 v_π를 어떻게 계산하나?"
    │
    └── Bellman Optimality Eq → Control 문제
            "v_* 또는 q_*를 어떻게 찾나?"
                ├── Dynamic Programming (Ch.4)
                ├── Monte Carlo (Ch.5)
                └── TD / Q-learning (Ch.6~)
```

강화학습의 거의 모든 알고리즘은 결국 **Bellman Equation의 변형이거나, 그것을 근사하는 방법**입니다.

---

> **다음 챕터로**: Chapter 4에서는 환경의 dynamics $p$를 완전히 알고 있을 때, Bellman Equation을 반복적으로 적용하여 $v_\pi$와 $v_*$를 계산하는 **Dynamic Programming** 방법을 다룹니다.
