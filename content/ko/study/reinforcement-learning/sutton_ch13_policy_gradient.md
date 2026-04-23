---
title: "Reinforcement Learning: Chapter 13 Policy Gradient Methods"
category: Reinforcement Learning
weight: 13
date: 2026-04-23
---

# Reinforcement Learning: Chapter 13 Policy Gradient Methods

> Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed.) — Chapter 13 핵심 정리

---

## 목차

1. [Policy Gradient의 동기](#1-policy-gradient의-동기)
2. [Policy Parameterization](#2-policy-parameterization)
3. [Policy Gradient Theorem 유도](#3-policy-gradient-theorem-유도)
4. [REINFORCE: Monte Carlo Policy Gradient](#4-reinforce-monte-carlo-policy-gradient)
5. [REINFORCE with Baseline](#5-reinforce-with-baseline)
6. [Actor-Critic Methods](#6-actor-critic-methods)
7. [Continuous Action Space](#7-continuous-action-space)
8. [Chapter 2 Gradient Bandit와의 연결](#8-chapter-2-gradient-bandit와의-연결)
9. [전체 요약 및 이후 챕터와의 연결](#9-전체-요약-및-이후-챕터와의-연결)

---

## 1. Policy Gradient의 동기

### Value-based 방법의 한계

Chapter 6–12의 방법들은 value function $q_\pi$나 $v_\pi$를 추정한 뒤, **greedy하게** policy를 도출했습니다.

| 문제점 | 설명 |
|---|---|
| **연속 행동 공간** | $\arg\max_a q(s,a)$를 계산하기 어려움 |
| **확률적 policy 필요** | 바위-가위-보 같은 게임에서 deterministic policy는 최적이 아님 |
| **수렴 불안정** | ε-greedy의 비연속적 policy 변화 |
| **고차원 행동 공간** | $\max$ 연산의 계산 비용 폭발 |

### Policy Gradient의 핵심 아이디어

Value function 대신 **policy 자체를 파라미터화** 하고, 목적함수를 직접 gradient ascent로 최적화합니다.

$$\pi(a \mid s, \boldsymbol{\theta}), \quad \boldsymbol{\theta} \in \mathbb{R}^d$$

$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t + \alpha\,\widehat{\nabla J(\boldsymbol{\theta}_t)}$$

---

## 2. Policy Parameterization

### Softmax in Action Preferences

이산 행동 공간에서 가장 일반적인 방법:

$$\pi(a \mid s, \boldsymbol{\theta}) \doteq \frac{e^{h(s, a, \boldsymbol{\theta})}}{\sum_b e^{h(s, b, \boldsymbol{\theta})}}$$

$h(s, a, \boldsymbol{\theta})$: 상태-행동 선호도(preference), 선형 또는 신경망으로 표현.

### Softmax vs. ε-Greedy

| | ε-Greedy | Softmax Policy |
|---|---|---|
| 표현 | Deterministic + 무작위 | 진짜 확률 분포 |
| 수렴 | Deterministic에 근접 | Deterministic도 가능 ($h$ 극단화) |
| 미분 가능성 | ❌ | ✅ (gradient 계산 가능) |
| 확률적 최적 policy | ❌ | ✅ |

### Log-Derivative Trick

Policy gradient 계산의 핵심:

$$\nabla_{\boldsymbol{\theta}}\pi(a \mid s, \boldsymbol{\theta}) = \pi(a \mid s, \boldsymbol{\theta})\,\nabla_{\boldsymbol{\theta}}\ln\pi(a \mid s, \boldsymbol{\theta})$$

$$\boxed{\nabla_{\boldsymbol{\theta}}\ln\pi(a \mid s, \boldsymbol{\theta}) = \frac{\nabla_{\boldsymbol{\theta}}\pi(a \mid s, \boldsymbol{\theta})}{\pi(a \mid s, \boldsymbol{\theta})}}$$

이를 **score function** 또는 **log-likelihood gradient** 라고 합니다.

---

## 3. Policy Gradient Theorem 유도

### 목적함수

Episodic task에서:

$$J(\boldsymbol{\theta}) \doteq v_{\pi_{\boldsymbol{\theta}}}(s_0)$$

$s_0$: 고정된 시작 상태.

### 유도의 어려움

$J(\boldsymbol{\theta})$를 $\boldsymbol{\theta}$로 미분하면 두 가지가 함께 변합니다:

1. **행동 선택 확률** $\pi(a \mid s, \boldsymbol{\theta})$
2. **상태 방문 분포** $\mu_\pi(s)$ (policy에 따라 달라짐)

$\mu_\pi(s)$의 미분은 다루기가 매우 어렵습니다. **Policy Gradient Theorem** 은 이를 깔끔하게 해결합니다.

### Policy Gradient Theorem

$$\boxed{\nabla J(\boldsymbol{\theta}) \propto \sum_s \mu_\pi(s)\sum_a q_\pi(s, a)\,\nabla_{\boldsymbol{\theta}}\pi(a \mid s, \boldsymbol{\theta})}$$

$\mu_\pi(s)$의 gradient를 계산할 필요가 없습니다.

### 증명 스케치

$v_\pi(s) = \sum_a \pi(a \mid s)q_\pi(s, a)$를 $\boldsymbol{\theta}$로 미분:

$$\nabla v_\pi(s) = \sum_a \left[\nabla\pi(a \mid s)q_\pi(s, a) + \pi(a \mid s)\nabla q_\pi(s, a)\right]$$

$\nabla q_\pi(s, a) = \nabla\sum_{s'} p(s' \mid s, a)\left[r + \gamma v_\pi(s')\right]$에 재귀적으로 대입하면:

$$\nabla v_\pi(s) = \sum_{s'}\sum_{k=0}^{\infty}P(s \to s', k \text{ steps}, \pi)\sum_a \nabla\pi(a \mid s')q_\pi(s', a)$$

$\mu_\pi(s') \propto \sum_k P(s_0 \to s', k)$로 정의하면 정리 성립. $\square$

### Log-Derivative Trick 적용

$$\nabla J(\boldsymbol{\theta}) \propto \mathbb{E}_\pi\left[q_\pi(S_t, A_t)\,\nabla_{\boldsymbol{\theta}}\ln\pi(A_t \mid S_t, \boldsymbol{\theta})\right]$$

---

## 4. REINFORCE: Monte Carlo Policy Gradient

### 유도

Policy Gradient Theorem에서:

$$\nabla J(\boldsymbol{\theta}) \propto \mathbb{E}_\pi\left[G_t\,\nabla_{\boldsymbol{\theta}}\ln\pi(A_t \mid S_t, \boldsymbol{\theta})\right]$$

$q_\pi(S_t, A_t)$를 **Monte Carlo return $G_t$** 로 대체 (unbiased estimate).

### 업데이트 규칙

$$\boxed{\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t + \alpha\,G_t\,\nabla_{\boldsymbol{\theta}}\ln\pi(A_t \mid S_t, \boldsymbol{\theta}_t)}$$

### 알고리즘

```
Input: 미분 가능한 π(a|s,θ), α
Initialize: θ ← 0

Loop for each episode:
    Generate episode S₀, A₀, R₁, ..., S_T following π(·|·,θ)
    For t = 0, 1, ..., T-1:
        G ← Σ_{k=t+1}^{T} γ^{k-t-1} R_k
        θ ← θ + α G ∇ln π(A_t|S_t, θ)
```

### 업데이트 직관

| 상황 | $G_t$ | 업데이트 방향 |
|---|---|---|
| 좋은 보상 | 크다 | $A_t$ 선택 확률 **증가** |
| 나쁜 보상 | 작다 (음수 가능) | $A_t$ 선택 확률 **감소** |

### 특성

- **장점**: 진짜 gradient descent (unbiased), 이론적으로 local optimum 수렴
- **단점**: 에피소드 종료 후에만 업데이트, **높은 분산**

---

## 5. REINFORCE with Baseline

### 분산 감소

임의의 baseline $b(s)$를 빼도 gradient가 **편향되지 않음** 을 증명합니다:

$$\mathbb{E}_\pi\left[b(S_t)\,\nabla_{\boldsymbol{\theta}}\ln\pi(A_t \mid S_t, \boldsymbol{\theta})\right] = 0$$

**증명**: 

$$\sum_a b(s)\,\nabla\pi(a \mid s) = b(s)\,\nabla\sum_a \pi(a \mid s) = b(s)\,\nabla 1 = 0 \quad \checkmark$$

따라서:

$$\boxed{\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t + \alpha\left[G_t - b(S_t)\right]\nabla_{\boldsymbol{\theta}}\ln\pi(A_t \mid S_t, \boldsymbol{\theta}_t)}$$

### 최적 baseline: $b(s) = v_\pi(s)$

$b(s) = v_\pi(s)$로 설정하면 $(G_t - v_\pi(S_t))$는 **advantage function** $A_\pi(s, a)$의 추정치입니다:

$$A_\pi(s, a) \doteq q_\pi(s, a) - v_\pi(s)$$

- $A > 0$: 해당 행동이 평균보다 좋음 → 확률 증가
- $A < 0$: 해당 행동이 평균보다 나쁨 → 확률 감소

실제로는 $\hat{v}(S_t, \mathbf{w})$를 별도로 학습해 baseline으로 사용합니다.

---

## 6. Actor-Critic Methods

### 동기: REINFORCE의 분산 문제 해결

REINFORCE는 에피소드 전체 return $G_t$를 사용합니다. 이를 **1-step TD estimate** 로 대체하면 온라인 업데이트가 가능하고 분산이 낮아집니다.

### One-step Actor-Critic

$$\delta_t = R_{t+1} + \gamma\hat{v}(S_{t+1}, \mathbf{w}) - \hat{v}(S_t, \mathbf{w}) \quad \text{(TD error)}$$

$$\mathbf{w}_{t+1} = \mathbf{w}_t + \alpha^w\,\delta_t\,\nabla_{\mathbf{w}}\hat{v}(S_t, \mathbf{w}_t) \quad \text{(Critic update)}$$

$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t + \alpha^\theta\,\delta_t\,\nabla_{\boldsymbol{\theta}}\ln\pi(A_t \mid S_t, \boldsymbol{\theta}_t) \quad \text{(Actor update)}$$

### 이름의 유래

```
Actor  ← policy π(a|s, θ)          행동 선택
Critic ← value function v̂(s, w)    행동 평가
```

- **Critic** 이 TD error $\delta_t$를 계산
- **Actor** 가 $\delta_t$를 신호로 policy를 개선

### Actor-Critic with Eligibility Traces

$$\mathbf{e}^\theta_t = \gamma\lambda^\theta\,\mathbf{e}^\theta_{t-1} + \nabla_{\boldsymbol{\theta}}\ln\pi(A_t \mid S_t, \boldsymbol{\theta}_t)$$
$$\mathbf{e}^w_t = \gamma\lambda^w\,\mathbf{e}^w_{t-1} + \nabla_{\mathbf{w}}\hat{v}(S_t, \mathbf{w}_t)$$

$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t + \alpha^\theta\,\delta_t\,\mathbf{e}^\theta_t$$
$$\mathbf{w}_{t+1} = \mathbf{w}_t + \alpha^w\,\delta_t\,\mathbf{e}^w_t$$

### REINFORCE vs. Actor-Critic 비교

| | REINFORCE | REINFORCE+Baseline | Actor-Critic |
|---|---|---|---|
| Update 시점 | 에피소드 종료 후 | 에피소드 종료 후 | 매 step (온라인) |
| Gradient 편향 | 없음 | 없음 | 있음 (semi-gradient) |
| 분산 | 높음 | 중간 | 낮음 |
| Bootstrap | ❌ | ❌ | ✅ |

---

## 7. Continuous Action Space

### 연속 행동의 Policy 파라미터화

행동 $a \in \mathbb{R}$인 경우 softmax 대신 **Gaussian policy** 를 사용:

$$\pi(a \mid s, \boldsymbol{\theta}) \doteq \frac{1}{\sigma(s, \boldsymbol{\theta})\sqrt{2\pi}}\exp\left(-\frac{(a - \mu(s, \boldsymbol{\theta}))^2}{2\sigma(s, \boldsymbol{\theta})^2}\right)$$

- $\mu(s, \boldsymbol{\theta})$: 평균 (선형 또는 신경망)
- $\sigma(s, \boldsymbol{\theta})$: 표준편차 (탐색 정도 조절)

### Score Function (Gaussian)

$$\nabla_{\boldsymbol{\theta}}\ln\pi(a \mid s, \boldsymbol{\theta}) = \frac{(a - \mu)\mathbf{x}_\mu(s)}{\sigma^2}$$

($\mathbf{x}_\mu$: 평균 파라미터에 대한 특징 벡터)

---

## 8. Chapter 2 Gradient Bandit와의 연결

Ch.2의 Gradient Bandit은 Ch.13 Policy Gradient의 **직접적인 원형** 입니다. 상태가 없는 ($|\mathcal{S}| = 1$) 특수 케이스로 볼 수 있습니다.

| 개념 | Gradient Bandit (Ch.2) | Policy Gradient (Ch.13) |
|---|---|---|
| 상태 | 없음 | 상태 $s$ 있음 |
| 파라미터 | $H_t(a)$ | $\boldsymbol{\theta}$ |
| Policy | $\text{softmax}(H_t)$ | $\pi(a \mid s, \boldsymbol{\theta})$ |
| Update 신호 | $R_t - \bar{R}_t$ | $G_t - b(S_t)$ |
| Gradient | $\frac{\partial \pi_t(x)}{\partial H_t(a)}$ | $\nabla_{\boldsymbol{\theta}}\ln\pi(a \mid s, \boldsymbol{\theta})$ |

Log-derivative trick의 동일한 구조:

$$\frac{\partial \pi_t(x)}{\partial H_t(a)} = \pi_t(x)(\mathbf{1}_{x=a} - \pi_t(a)) = \pi_t(x)\cdot\frac{\partial\ln\pi_t(x)}{\partial H_t(a)}$$

---

## 9. 전체 요약 및 이후 챕터와의 연결

### Chapter 13 구조 요약

```
Value-based의 한계 (연속 행동, 확률적 최적 policy 등)
        ↓
Policy Parameterization: π(a|s,θ)
        ↓
Policy Gradient Theorem:
  ∇J(θ) ∝ E_π[q_π(S,A) ∇lnπ(A|S,θ)]
        ↓
REINFORCE: G_t 로 q_π 대체 (unbiased, 고분산)
        ↓
REINFORCE + Baseline: b(s) 빼도 gradient 불변 → 분산 감소
        ↓
Actor-Critic: TD error δ_t 로 q_π 근사 (온라인, 저분산, 편향)
  Actor: θ ← θ + α δ_t ∇lnπ
  Critic: w ← w + α δ_t ∇v̂
        ↓
Continuous Action: Gaussian Policy
```

### 핵심 수식 한눈에 보기

**Policy Gradient Theorem**:
$$\nabla J(\boldsymbol{\theta}) \propto \mathbb{E}_\pi\left[q_\pi(S_t,A_t)\,\nabla_{\boldsymbol{\theta}}\ln\pi(A_t \mid S_t,\boldsymbol{\theta})\right]$$

**REINFORCE**:
$$\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \alpha G_t\,\nabla_{\boldsymbol{\theta}}\ln\pi(A_t \mid S_t,\boldsymbol{\theta})$$

**REINFORCE with Baseline**:
$$\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \alpha(G_t - b(S_t))\,\nabla_{\boldsymbol{\theta}}\ln\pi(A_t \mid S_t,\boldsymbol{\theta})$$

**One-step Actor-Critic**:
$$\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \alpha^\theta\delta_t\,\nabla_{\boldsymbol{\theta}}\ln\pi(A_t \mid S_t,\boldsymbol{\theta})$$
$$\mathbf{w} \leftarrow \mathbf{w} + \alpha^w\delta_t\,\nabla_{\mathbf{w}}\hat{v}(S_t,\mathbf{w})$$

### 이후 챕터와의 연결

| 챕터 | 주제 | 연결점 |
|---|---|---|
| Ch.15 | Neuroscience | TD error와 도파민, policy gradient와 신경 회로 |
| Ch.16 | Applications | A3C, PPO, TRPO: Actor-Critic의 발전형 |

### 현대 RL과의 연결

Policy Gradient는 오늘날 가장 강력한 RL 알고리즘들의 기반입니다:

| 알고리즘 | Ch.13 기반 | 개선점 |
|---|---|---|
| A2C/A3C | Actor-Critic | 비동기 병렬 학습 |
| PPO | REINFORCE+Baseline | Clipped surrogate objective로 안정화 |
| TRPO | Policy Gradient | Trust region으로 step 크기 제한 |
| SAC | Actor-Critic | Entropy 정규화, 연속 행동 |

---

> **이 챕터로**: Chapter 13은 Sutton & Barto 2부의 마지막 알고리즘 챕터입니다. 이후 Ch.14–16은 심리학·신경과학과의 연결, 그리고 게임·로보틱스 등 실제 응용을 다룹니다.
