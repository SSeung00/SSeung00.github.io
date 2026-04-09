---
title: "Reinforcement Learning: Chapter 7 n-step Bootstrapping"
category: "Reinforcement Learning"
weight: 7
date: 2026-04-09
---

# Reinforcement Learning Chapter 7: n-step Bootstrapping

> Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed.) — Chapter 7 핵심 정리

---

## 목차

1. [n-step TD란: TD와 MC 사이의 스펙트럼](#1-n-step-td란-td와-mc-사이의-스펙트럼)
2. [n-step Return 정의와 유도](#2-n-step-return-정의와-유도)
3. [n-step TD Prediction](#3-n-step-td-prediction)
4. [n-step SARSA: On-policy Control](#4-n-step-sarsa-on-policy-control)
5. [Off-policy n-step TD: Importance Sampling](#5-off-policy-n-step-td-importance-sampling)
6. [n-step Tree Backup Algorithm](#6-n-step-tree-backup-algorithm)
7. [n-step Q(σ): 통합 알고리즘](#7-n-step-qσ-통합-알고리즘)
8. [전체 요약 및 이후 챕터와의 연결](#8-전체-요약-및-이후-챕터와의-연결)

---

## 1. n-step TD란: TD와 MC 사이의 스펙트럼

### 복습: TD(0)와 MC의 target

Chapter 6의 TD(0)는 **1-step 앞**만 보고 bootstrap합니다:

$$\text{TD(0) target:} \quad G_t^{(1)} = R_{t+1} + \gamma V(S_{t+1})$$

Chapter 5의 MC는 **에피소드 끝까지** 실제 보상을 사용합니다:

$$\text{MC target:} \quad G_t^{(\infty)} = R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{T-t-1} R_T$$

**n-step TD**는 이 둘 사이 어딘가에 위치합니다. $n$ 스텝 앞까지 실제 보상을 사용하고, 그 이후는 추정값으로 bootstrap합니다:

$$\text{n-step target:} \quad G_t^{(n)} = R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{n-1} R_{t+n} + \gamma^n V(S_{t+n})$$

### 스펙트럼

$$\underbrace{G_t^{(1)}}_{\text{TD(0)}} \quad\longrightarrow\quad G_t^{(2)} \quad\longrightarrow\quad \cdots \quad\longrightarrow\quad \underbrace{G_t^{(\infty)}}_{\text{MC}}$$

$n$이 커질수록:
- **Bias 감소**: 추정값 $V$에 대한 의존도 감소
- **Variance 증가**: 더 많은 실제 보상의 곱이 누적
- **업데이트 지연 증가**: $n$ 스텝을 기다려야 업데이트 가능

최적의 $n$은 문제마다 다르며, 이것이 n-step TD를 하이퍼파라미터로 갖는 이유입니다.

---

## 2. n-step Return 정의와 유도

### n-step Return 정의

$$\boxed{G_t^{(n)} \doteq R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{n-1} R_{t+n} + \gamma^n V(S_{t+n})}$$

$$= \sum_{k=0}^{n-1} \gamma^k R_{t+k+1} + \gamma^n V(S_{t+n})$$

에피소드가 $t+n$ 이전에 종료되면 ($T \leq t+n$):

$$G_t^{(n)} = G_t \quad (\text{실제 return, MC와 동일})$$

### 점화식으로 표현

$G_t^{(n)}$을 점화식으로 쓰면 계산이 편리합니다:

$$G_t^{(n)} = R_{t+1} + \gamma G_{t+1}^{(n-1)}$$

**유도**:

$$G_t^{(n)} = R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{n-1} R_{t+n} + \gamma^n V(S_{t+n})$$

$$= R_{t+1} + \gamma \underbrace{\left(R_{t+2} + \cdots + \gamma^{n-2} R_{t+n} + \gamma^{n-1} V(S_{t+n})\right)}_{= G_{t+1}^{(n-1)}}$$

$$= R_{t+1} + \gamma\, G_{t+1}^{(n-1)} \checkmark$$

경계 조건: $G_t^{(0)} = V(S_t)$, $G_t^{(\infty)} = G_t$ (실제 return).

### n-step Return의 오차 분석

$V = v_\pi$라 가정하면, $n$-step return은 $v_\pi(S_t)$의 **unbiased estimate**입니다:

$$\mathbb{E}_\pi\left[G_t^{(n)} \mid S_t\right] = v_\pi(S_t), \quad \text{if } V = v_\pi$$

하지만 $V \neq v_\pi$이면 bias가 존재하고, 그 크기는 $\gamma^n \|V - v_\pi\|_\infty$에 비례합니다. $n$이 클수록 $\gamma^n$이 작아지므로 bias가 줄어듭니다.

---

## 3. n-step TD Prediction

### 업데이트 규칙

$$\boxed{V(S_t) \leftarrow V(S_t) + \alpha\left[G_t^{(n)} - V(S_t)\right]}$$

단, 이 업데이트는 time step $t + n$ 이후에야 가능합니다 — $G_t^{(n)}$ 계산에 $S_{t+n}$이 필요하기 때문입니다.

### 알고리즘

```
Input: policy π, step-size α, n
Initialize: V(s) ← arbitrary, V(terminal) ← 0

Loop (에피소드 반복):
    S_0 ← 초기 상태
    T ← ∞
    t ← 0

    Loop:
        if t < T:
            A_t ~ π(·|S_t)로 행동 선택
            R_{t+1}, S_{t+1} ← 환경에서 관측
            if S_{t+1} is terminal: T ← t + 1

        τ ← t - n + 1   ← 업데이트 대상 time step
        if τ ≥ 0:
            G ← Σ_{k=τ+1}^{min(τ+n, T)} γ^{k-τ-1} R_k
            if τ + n < T: G ← G + γ^n V(S_{τ+n})
            V(S_τ) ← V(S_τ) + α[G - V(S_τ)]

        t ← t + 1
    Until τ = T - 1
```

### 업데이트 타이밍

```
time:    0   1   2   3   4   5  ...  T
         S₀  S₁  S₂  S₃  S₄  S₅
              R₁  R₂  R₃  R₄  R₅

n=1 업데이트: t=0은 t=1에서 (1 step 후)
n=2 업데이트: t=0은 t=2에서 (2 step 후)
n=3 업데이트: t=0은 t=3에서 (3 step 후)
```

$n$이 클수록 업데이트가 더 늦게 일어납니다. 에피소드가 끝날 때 아직 처리 못 한 초기 step들을 일괄 업데이트합니다.

---

## 4. n-step SARSA: On-policy Control

### Action-value로 확장

Prediction의 $V(S_{t+n})$을 $Q(S_{t+n}, A_{t+n})$으로 교체합니다:

$$G_t^{(n)} \doteq R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{n-1} R_{t+n} + \gamma^n Q(S_{t+n}, A_{t+n})$$

업데이트 규칙:

$$\boxed{Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha\left[G_t^{(n)} - Q(S_t, A_t)\right]}$$

### n=1, n=∞ 경계 확인

- $n=1$: $G_t^{(1)} = R_{t+1} + \gamma Q(S_{t+1}, A_{t+1})$ → **1-step SARSA (Chapter 6)**
- $n=\infty$: $G_t^{(\infty)} = G_t$ → **MC Control (Chapter 5)**

n-step SARSA는 이 두 극단을 연속적으로 잇는 알고리즘군입니다.

### Expected n-step SARSA

마지막 step의 $Q(S_{t+n}, A_{t+n})$ 대신 기댓값을 사용하면 분산이 줄어듭니다:

$$G_t^{(n)} = R_{t+1} + \cdots + \gamma^{n-1} R_{t+n} + \gamma^n \sum_{a'} \pi(a' \mid S_{t+n})\, Q(S_{t+n}, a')$$

---

## 5. Off-policy n-step TD: Importance Sampling

### n-step에서의 IS ratio

Chapter 5의 off-policy MC에서 IS ratio는 전체 에피소드 궤적에 대해 계산했습니다. n-step에서는 **$n$ 스텝에 해당하는 구간**에만 적용합니다:

$$\rho_{t:t+n-1} \doteq \prod_{k=t}^{\min(t+n-1,\, T-1)} \frac{\pi(A_k \mid S_k)}{b(A_k \mid S_k)}$$

업데이트 규칙:

$$\boxed{V(S_t) \leftarrow V(S_t) + \alpha\,\rho_{t:t+n-1}\left[G_t^{(n)} - V(S_t)\right]}$$

### MC IS vs. n-step IS의 분산 비교

MC에서 IS ratio는 전체 에피소드 길이 $T$에 대한 곱이었습니다:

$$\rho_{t:T-1} = \prod_{k=t}^{T-1} \frac{\pi(A_k)}{b(A_k)}$$

n-step에서는 **$n$ 개의 항**만 곱합니다:

$$\rho_{t:t+n-1} = \prod_{k=t}^{t+n-1} \frac{\pi(A_k)}{b(A_k)}$$

$n \ll T$이면 IS ratio의 분산이 **극적으로 감소**합니다. 이것이 n-step off-policy TD가 MC off-policy보다 실용적인 핵심 이유입니다.

### Off-policy n-step SARSA

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha\,\rho_{t+1:t+n}\left[G_t^{(n)} - Q(S_t, A_t)\right]$$

여기서 IS ratio의 시작이 $t+1$인 이유: $A_t$는 이미 선택된 행동이므로 보정이 필요 없고, $A_{t+1}$부터 $A_{t+n}$까지의 행동 선택에 대해서만 보정합니다.

---

## 6. n-step Tree Backup Algorithm

### IS 없이 Off-policy를 달성하는 방법

n-step IS는 분산을 줄였지만 여전히 남아 있습니다. **Tree Backup**은 IS ratio를 **전혀 사용하지 않고** off-policy n-step 업데이트를 수행합니다.

### 핵심 아이디어

실제로 선택한 행동의 보상만 사용하되, **선택하지 않은 행동들의 기여도를 기댓값으로 처리**합니다.

1-step Tree Backup은 Expected SARSA와 동일합니다:

$$G_t^{(1)} = R_{t+1} + \gamma \sum_{a'} \pi(a' \mid S_{t+1})\, Q(S_{t+1}, a')$$

2-step Tree Backup은 다음과 같습니다:

$$G_t^{(2)} = R_{t+1} + \gamma \sum_{a \neq A_{t+1}} \pi(a \mid S_{t+1})\, Q(S_{t+1}, a) + \gamma\,\pi(A_{t+1} \mid S_{t+1})\left[R_{t+2} + \gamma \sum_{a'} \pi(a' \mid S_{t+2})\, Q(S_{t+2}, a')\right]$$

이를 일반화하면, n-step Tree Backup return의 점화식:

$$G_t^{(n)} = R_{t+1} + \gamma \sum_{a \neq A_{t+1}} \pi(a \mid S_{t+1})\, Q(S_{t+1}, a) + \gamma\,\pi(A_{t+1} \mid S_{t+1})\, G_{t+1}^{(n-1)}$$

### 직관적 이해

```
        S_t
       /   \
    A_t    다른 행동들 ← π(a|S_t)·Q(S_t,a)로 즉시 처리
      |
    S_{t+1}
   /    \
A_{t+1} 다른 행동들 ← π(a|S_{t+1})·Q(S_{t+1},a)로 즉시 처리
   |
  ...
```

실제로 선택된 행동 경로(가운데 줄기)를 따라가면서, 각 분기점에서 **선택되지 않은 가지들을 기댓값으로 즉시 처리(prune)**합니다.

---

## 7. n-step Q(σ): 통합 알고리즘

### 네 가지 알고리즘의 통합

Chapter 7의 핵심 통찰은 지금까지 나온 n-step 알고리즘들이 **하나의 파라미터 $\sigma$로 통합**된다는 점입니다.

각 step $t$에서 **샘플링 여부를 결정하는 파라미터 $\sigma_t \in [0, 1]$**:

- $\sigma_t = 1$: IS ratio를 사용한 샘플링 (n-step SARSA 방식)
- $\sigma_t = 0$: 기댓값 사용 (Tree Backup 방식)

### Q(σ) Return 정의

$$G_t^{(n)} \doteq R_{t+1} + \gamma\left[\sigma_{t+1}\rho_{t+1} + (1 - \sigma_{t+1})\pi(A_{t+1} \mid S_{t+1})\right] \cdot \left(G_{t+1}^{(n-1)} - Q(S_{t+1}, A_{t+1})\right) + \gamma \sum_a \pi(a \mid S_{t+1})\, Q(S_{t+1}, a)$$

이를 간결하게 쓰면:

$$\boxed{G_t^{(n)} = R_{t+1} + \gamma \bar{V}_{t+1} + \gamma\left[\sigma_{t+1}\rho_{t+1} + (1-\sigma_{t+1})\pi(A_{t+1}|S_{t+1})\right]\left(G_{t+1}^{(n-1)} - Q(S_{t+1}, A_{t+1})\right)}$$

여기서 $\bar{V}_t \doteq \sum_a \pi(a \mid S_t)\, Q(S_t, a)$는 상태 $S_t$에서의 기대 action-value입니다.

### $\sigma$에 따른 알고리즘 스펙트럼

| $\sigma$ 설정 | 대응 알고리즘 |
|---|---|
| 모든 step에서 $\sigma = 1$ | n-step SARSA |
| 모든 step에서 $\sigma = 0$ | n-step Tree Backup |
| $\sigma$ 자유롭게 혼합 | n-step Q(σ) |
| $n=1$, $\sigma=0$ | Expected SARSA |
| $n=1$, $\sigma=0$, $\pi$ = greedy | Q-learning |

이 통합 관점은 알고리즘들의 공통 구조를 드러내며, 문제에 따라 $\sigma$를 조절하는 유연성을 제공합니다.

---

## 8. 전체 요약 및 이후 챕터와의 연결

### Chapter 7 구조 요약

```
Chapter 6: TD(0) ← 1-step bootstrap
Chapter 5: MC   ← ∞-step (실제 return)
        ↓ 두 극단을 하나의 파라미터 n으로 연결
Chapter 7: n-step Bootstrapping
        │
        ├── n-step Return
        │       G_t^(n) = Σ γ^k R_{t+k+1} + γ^n V(S_{t+n})
        │       점화식: G_t^(n) = R_{t+1} + γ G_{t+1}^(n-1)
        │
        ├── n-step TD Prediction
        │       V(S_t) ← V(S_t) + α[G_t^(n) - V(S_t)]
        │       (t+n 이후에 업데이트)
        │
        ├── n-step SARSA (On-policy Control)
        │       Q 버전: G_t^(n)에 Q(S_{t+n}, A_{t+n}) 사용
        │
        ├── Off-policy n-step TD (IS 사용)
        │       ρ_{t:t+n-1} 로 분포 보정
        │       IS ratio가 n항 곱 → MC보다 분산 대폭 감소
        │
        ├── Tree Backup (IS 없이 off-policy)
        │       선택 안 된 가지 → π(a|s)·Q(s,a)로 즉시 처리
        │
        └── Q(σ): n-step 알고리즘 통합
                σ=1: 샘플링 (SARSA)
                σ=0: 기댓값 (Tree Backup)
                σ 혼합: 유연한 중간 형태
```

### 핵심 수식 한눈에 보기

**n-step Return**:
$$G_t^{(n)} = \sum_{k=0}^{n-1} \gamma^k R_{t+k+1} + \gamma^n V(S_{t+n})$$

**n-step Return 점화식**:
$$G_t^{(n)} = R_{t+1} + \gamma\, G_{t+1}^{(n-1)}, \qquad G_t^{(0)} = V(S_t)$$

**n-step TD update**:
$$V(S_t) \leftarrow V(S_t) + \alpha\left[G_t^{(n)} - V(S_t)\right]$$

**Off-policy n-step IS ratio**:
$$\rho_{t:t+n-1} = \prod_{k=t}^{t+n-1} \frac{\pi(A_k \mid S_k)}{b(A_k \mid S_k)}$$

**Tree Backup return (점화식)**:
$$G_t^{(n)} = R_{t+1} + \gamma \sum_{a \neq A_{t+1}} \pi(a|S_{t+1})\,Q(S_{t+1},a) + \gamma\,\pi(A_{t+1}|S_{t+1})\,G_{t+1}^{(n-1)}$$

### n-step의 bias-variance 트레이드오프 요약

$$\underbrace{n=1}_{\substack{\text{High bias} \\ \text{Low variance} \\ \text{Fast update}}} \quad\longrightarrow\quad \underbrace{n=\infty}_{\substack{\text{Low bias} \\ \text{High variance} \\ \text{Slow update}}}$$

최적 $n$은 문제에 따라 다르며, 실전에서는 $n \in \{4, 8, 16\}$ 정도가 자주 좋은 성능을 보입니다.

### 이후 챕터로의 연결

| 챕터 | 주제 | Chapter 7과의 관계 |
|---|---|---|
| Ch.8 Planning (Dyna) | 모델 기반 계획 | n-step return을 모델로 생성한 샘플에 적용 |
| Ch.9~10 Function Approx. | 신경망으로 $V$, $Q$ 근사 | n-step return을 target으로 함수 근사 학습 |
| Ch.12 Eligibility Traces | TD(λ) | n-step return의 기하급수 가중 평균 → λ로 통합 |

특히 **Chapter 12의 TD(λ)**는 n-step return을 $\lambda$로 가중 평균하는 방법으로, Chapter 7이 그 직접적인 기반입니다:

$$G_t^\lambda = (1-\lambda) \sum_{n=1}^{\infty} \lambda^{n-1} G_t^{(n)}$$

모든 $n$-step return을 동시에 사용하는 이 아이디어가 eligibility traces의 본질입니다.

---

> **다음 챕터로**: Chapter 8에서는 실제 환경 샘플 외에 **학습된 모델로 가상의 경험을 생성**하여 계획(planning)과 학습(learning)을 결합하는 **Dyna 아키텍처**를 다룹니다.
