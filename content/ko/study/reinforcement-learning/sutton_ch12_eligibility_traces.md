---
title: "Reinforcement Learning: Chapter 12 Eligibility Traces"
category: Reinforcement Learning
weight: 12
date: 2026-04-23
---

# Reinforcement Learning: Chapter 12 Eligibility Traces

> Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed.) — Chapter 12 핵심 정리

---

## 목차

1. [Eligibility Traces의 직관](#1-eligibility-traces의-직관)
2. [λ-return: n-step의 통합](#2-λ-return-n-step의-통합)
3. [Forward vs. Backward View](#3-forward-vs-backward-view)
4. [TD(λ): Offline λ-return Algorithm](#4-tdλ-offline-λ-return-algorithm)
5. [Semi-gradient TD(λ) with Eligibility Traces](#5-semi-gradient-tdλ-with-eligibility-traces)
6. [True Online TD(λ)](#6-true-online-tdλ)
7. [Sarsa(λ)](#7-sarsaλ)
8. [Off-policy Traces: GTD(λ), Emphatic TD(λ)](#8-off-policy-traces-gtdλ-emphatic-tdλ)
9. [전체 요약 및 이후 챕터와의 연결](#9-전체-요약-및-이후-챕터와의-연결)

---

## 1. Eligibility Traces의 직관

### 문제: 신용 할당(Credit Assignment)

보상은 일련의 상태-행동 이후 지연되어 도착합니다. 어느 상태·행동이 보상의 **원인(credit)** 인지 어떻게 알 수 있을까요?

**Eligibility trace** $\mathbf{e}_t$는 각 파라미터(또는 상태)가 최근 방문에서 얼마나 **자격(eligible)** 이 있는지를 추적합니다.

```
시간 →    s₁ → s₂ → s₃ → s₄ → R
                               ↑
           이 보상의 원인은?
           e(s₄) 가장 크고, e(s₃), e(s₂), e(s₁) 순으로 감소
```

### 메커니즘

방문한 상태의 trace가 **축적(accumulate)** 되고, 매 step마다 $\gamma\lambda$ 비율로 **감쇠(decay)** 됩니다:

$$\mathbf{e}_t \leftarrow \gamma\lambda\,\mathbf{e}_{t-1} + \nabla_{\mathbf{w}}\hat{v}(S_t, \mathbf{w}_t)$$

TD error $\delta_t$가 발생하면 trace에 비례해 **모든 파라미터를 동시에** 업데이트:

$$\mathbf{w}_t \leftarrow \mathbf{w}_t + \alpha\,\delta_t\,\mathbf{e}_t$$

---

## 2. λ-return: n-step의 통합

### n-step Return 복습

$$G_{t:t+n} = \sum_{k=1}^{n}\gamma^{k-1}R_{t+k} + \gamma^n\hat{v}(S_{t+n}, \mathbf{w}_{t+n-1})$$

### λ-return: 기하급수 가중 평균

$$\boxed{G_t^\lambda \doteq (1-\lambda)\sum_{n=1}^{\infty}\lambda^{n-1}G_{t:t+n}}$$

- $n$-step return에 $(1-\lambda)\lambda^{n-1}$로 **지수적으로 감소하는 가중치** 를 부여
- 가중치 합 = 1: $(1-\lambda)\sum_{n=1}^{\infty}\lambda^{n-1} = 1$ ✓

### 가중치 구조 시각화

```
n=1: (1-λ)·λ⁰
n=2: (1-λ)·λ¹
n=3: (1-λ)·λ²
     ↓ λ가 클수록 먼 미래의 return에 더 큰 비중
```

### 특수 경우

| $\lambda$ | 동치 방법 |
|---|---|
| $\lambda = 0$ | TD(0): $G_t^0 = R_{t+1} + \gamma\hat{v}(S_{t+1})$ |
| $\lambda = 1$ | Monte Carlo: $G_t^1 = G_t$ |
| 중간 $\lambda$ | n-step들의 가중 평균 |

---

## 3. Forward vs. Backward View

### Forward View (이론적)

시각 $t$에서 **미래를 내다보며** $G_t^\lambda$를 계산합니다.

$$\mathbf{w}_{t+1} = \mathbf{w}_t + \alpha\left[G_t^\lambda - \hat{v}(S_t, \mathbf{w}_t)\right]\nabla_{\mathbf{w}}\hat{v}(S_t, \mathbf{w}_t)$$

- **문제**: 에피소드가 끝나야 $G_t^\lambda$를 계산 가능 → 온라인 업데이트 불가

### Backward View (구현 가능)

Eligibility trace를 이용해 **온라인으로** forward view를 근사합니다.

- 현재 TD error $\delta_t$를 계산
- 과거 모든 상태에 trace 비례 업데이트
- **에피소드가 끝나지 않아도 즉시 업데이트 가능**

$$\text{Forward View} \approx \text{Backward View (offline TD(λ)에서 완전히 동치)}$$

---

## 4. TD(λ): Offline λ-return Algorithm

### 알고리즘

```
Input: policy π, α, λ, 미분 가능한 v̂(s, w)
Initialize: w ← 0

Loop for each episode:
    Generate full episode: S₀, R₁, S₁, ..., S_T
    For t = 0, 1, ..., T-1:
        G_t^λ ← (1-λ) Σ_{n=1}^{T-t} λ^{n-1} G_{t:t+n}
        w ← w + α[G_t^λ - v̂(S_t, w)] ∇v̂(S_t, w)
```

에피소드 후 backward view와 **동일한 업데이트** 를 생성합니다.

---

## 5. Semi-gradient TD(λ) with Eligibility Traces

### Eligibility Trace 업데이트

$$\boxed{\mathbf{e}_t \doteq \gamma\lambda\,\mathbf{e}_{t-1} + \nabla_{\mathbf{w}}\hat{v}(S_t, \mathbf{w}_t), \quad \mathbf{e}_0 = \mathbf{0}}$$

### 파라미터 업데이트

$$\delta_t = R_{t+1} + \gamma\hat{v}(S_{t+1}, \mathbf{w}_t) - \hat{v}(S_t, \mathbf{w}_t)$$

$$\boxed{\mathbf{w}_{t+1} = \mathbf{w}_t + \alpha\,\delta_t\,\mathbf{e}_t}$$

### 선형 TD(λ)의 수렴

선형 근사 $\hat{v}(s, \mathbf{w}) = \mathbf{w}^\top\mathbf{x}(s)$ + on-policy에서:

$$\mathbf{e}_t = \gamma\lambda\,\mathbf{e}_{t-1} + \mathbf{x}(S_t)$$

수렴점 $\mathbf{w}_{TD(\lambda)}$는 TD(0) fixed point와 MC 최적 사이에 위치:

$$\overline{VE}(\mathbf{w}_{TD(\lambda)}) \leq \frac{1-\lambda\gamma}{1-\gamma}\min_{\mathbf{w}}\overline{VE}(\mathbf{w})$$

$\lambda \to 1$이면 경계가 $\min \overline{VE}$ (MC 수준)에 가까워집니다.

### λ 선택에 따른 성능 비교

| $\lambda$ | 분산 | 편향 | 실용적 성능 |
|---|---|---|---|
| $0$ (TD(0)) | 낮음 | 높음 | bootstrap 의존 |
| $0.9$–$0.95$ | 중간 | 낮음 | 실제로 최적 성능 근처 |
| $1$ (MC) | 높음 | 없음 | 느린 수렴 |

---

## 6. True Online TD(λ)

### Semi-gradient TD(λ)의 한계

Semi-gradient TD(λ)의 backward view는 **이론적으로** offline λ-return과 동치이지만, **online(즉시 업데이트)** 시에는 미묘한 차이가 발생합니다. 파라미터 $\mathbf{w}$가 episode 중간에 계속 바뀌기 때문입니다.

### True Online TD(λ): 완전한 동치

**True Online TD(λ)** 는 online 업데이트에서도 online λ-return과 **정확히 동치** 가 되도록 trace를 수정합니다.

### Dutch Trace

$$\mathbf{e}_t \doteq \gamma\lambda\,\mathbf{e}_{t-1} + \left(1 - \alpha\gamma\lambda\,\mathbf{e}_{t-1}^\top\mathbf{x}_t\right)\mathbf{x}_t$$

### 업데이트

$$\delta_t = R_{t+1} + \gamma\mathbf{w}_t^\top\mathbf{x}_{t+1} - \mathbf{w}_t^\top\mathbf{x}_t$$

$$\mathbf{w}_{t+1} = \mathbf{w}_t + \alpha\left[\delta_t + \mathbf{w}_t^\top\mathbf{x}_t - \mathbf{w}_{t-1}^\top\mathbf{x}_t\right]\mathbf{e}_t - \alpha\left[\mathbf{w}_t^\top\mathbf{x}_t - \mathbf{w}_{t-1}^\top\mathbf{x}_t\right]\mathbf{x}_t$$

### Semi-gradient TD(λ) vs. True Online TD(λ)

| | Semi-gradient TD(λ) | True Online TD(λ) |
|---|---|---|
| Offline λ-return과 동치 | ✅ | ✅ |
| Online λ-return과 동치 | ❌ | ✅ |
| Trace 종류 | Accumulating | Dutch |
| 계산 복잡도 | $O(d)$ | $O(d)$ |
| 실용적 성능 | 좋음 | 더 좋음 |

---

## 7. Sarsa(λ)

### Action-Value에 Trace 적용

Control 문제를 위해 $v_\pi$ 대신 $q_\pi$를 추정합니다.

$$\mathbf{e}_t = \gamma\lambda\,\mathbf{e}_{t-1} + \nabla_{\mathbf{w}}\hat{q}(S_t, A_t, \mathbf{w}_t)$$

$$\delta_t = R_{t+1} + \gamma\hat{q}(S_{t+1}, A_{t+1}, \mathbf{w}_t) - \hat{q}(S_t, A_t, \mathbf{w}_t)$$

$$\mathbf{w}_{t+1} = \mathbf{w}_t + \alpha\,\delta_t\,\mathbf{e}_t$$

### Sarsa(λ) vs. Sarsa(0)

Mountain Car 실험에서 $\lambda > 0$인 Sarsa(λ)가 Sarsa(0)보다 **훨씬 빠른 학습** 을 보입니다. 보상이 지연되는 문제에서 trace가 credit assignment를 크게 개선하기 때문입니다.

---

## 8. Off-policy Traces: GTD(λ), Emphatic TD(λ)

### Off-policy와 Trace의 결합

Eligibility trace를 off-policy에 적용할 때, importance sampling ratio를 trace에 반영해야 합니다.

### GTD(λ)

Gradient-TD(Ch.11)에 trace를 결합:

$$\mathbf{e}_t = \gamma\lambda\rho_t\,\mathbf{e}_{t-1} + \rho_t\mathbf{x}_t$$

$$\mathbf{w}_{t+1} = \mathbf{w}_t + \alpha\,\delta_t\,\mathbf{e}_t - \alpha\gamma\lambda\left(\mathbf{e}_t^\top\mathbf{v}_t\right)\mathbf{x}_{t+1}$$

$$\mathbf{v}_{t+1} = \mathbf{v}_t + \beta\left(\delta_t - \mathbf{v}_t^\top\mathbf{x}_t\right)\mathbf{e}_t$$

- Off-policy에서도 **수렴 보장**

### Emphatic TD(λ)

Ch.11의 Emphatic-TD에 trace 결합. 수렴 보장 + 실용적 성능 개선.

---

## 9. 전체 요약 및 이후 챕터와의 연결

### Chapter 12 구조 요약

```
n-step Return의 한계 (n 선택 필요)
        ↓
λ-return: G_t^λ = (1-λ)Σ λ^{n-1} G_{t:t+n}
  λ=0 → TD(0), λ=1 → MC
        ↓
Forward View (이론)  ←→  Backward View (구현)
                              ↓
                    Eligibility Trace e_t
                    e_t ← γλe_{t-1} + ∇v̂(S_t,w)
                    w ← w + αδ_t e_t
                              ↓
                    Semi-gradient TD(λ)
                    True Online TD(λ) (더 정확)
                    Sarsa(λ) (control)
                    GTD(λ), Emphatic TD(λ) (off-policy)
```

### 핵심 수식 한눈에 보기

**λ-return**:
$$G_t^\lambda = (1-\lambda)\sum_{n=1}^{\infty}\lambda^{n-1}G_{t:t+n}$$

**Eligibility Trace**:
$$\mathbf{e}_t = \gamma\lambda\,\mathbf{e}_{t-1} + \nabla_{\mathbf{w}}\hat{v}(S_t,\mathbf{w}_t)$$

**TD(λ) 업데이트**:
$$\mathbf{w}_{t+1} = \mathbf{w}_t + \alpha\,\delta_t\,\mathbf{e}_t$$

**VE 경계**:
$$\overline{VE}(\mathbf{w}_{TD(\lambda)}) \leq \frac{1-\lambda\gamma}{1-\gamma}\min_{\mathbf{w}}\overline{VE}(\mathbf{w})$$

### 이후 챕터와의 연결

| 챕터 | 주제 | 연결점 |
|---|---|---|
| Ch.13 | Policy Gradient | Trace 대신 policy 파라미터 직접 업데이트 |
| Ch.15 | Neuroscience | TD error와 도파민 신호의 대응 |
| Ch.16 | DQN, A3C | 실제 구현에서 λ=0 또는 λ>0 선택 |

---

> **다음 챕터로**: Chapter 13에서는 value function을 거치지 않고 **policy 자체를 파라미터화하여 직접 최적화** 하는 Policy Gradient Methods를 다룹니다.
