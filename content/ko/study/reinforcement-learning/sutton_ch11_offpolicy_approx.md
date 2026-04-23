---
title: "Reinforcement Learning: Chapter 11 Off-policy Methods with Approximation"
category: Reinforcement Learning
weight: 11
date: 2026-04-23
---

# Reinforcement Learning: Chapter 11 Off-policy Methods with Approximation

> Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed.) — Chapter 11 핵심 정리

---

## 목차

1. [Off-policy + 함수 근사의 도전](#1-off-policy--함수-근사의-도전)
2. [Semi-Gradient Off-policy TD](#2-semi-gradient-off-policy-td)
3. [Deadly Triad: 발산의 3조건](#3-deadly-triad-발산의-3조건)
4. [발산 예시: Baird's Counterexample](#4-발산-예시-bairds-counterexample)
5. [Gradient-TD Methods](#5-gradient-td-methods)
6. [Emphatic-TD Methods](#6-emphatic-td-methods)
7. [전체 요약 및 이후 챕터와의 연결](#7-전체-요약-및-이후-챕터와의-연결)

---

## 1. Off-policy + 함수 근사의 도전

### 지금까지의 흐름

| 챕터 | 방법 | 함수 근사 | Off-policy |
|---|---|---|---|
| Ch.5–6 | MC, TD (tabular) | ❌ | ✅ (안정) |
| Ch.9–10 | Semi-gradient TD | ✅ | ❌ (on-policy) |
| **Ch.11** | **Off-policy + 근사** | ✅ | ✅ ← **위험** |

### 왜 어려운가?

Off-policy에서 함수 근사를 쓰면 두 가지 문제가 얽힙니다.

1. **분포 불일치**: $\overline{VE}$의 가중치 $\mu$가 behavior policy $b$의 방문 분포인데, target policy $\pi$의 value를 추정하려 함
2. **Bootstrap + 근사의 불안정성**: TD error의 gradient를 정확히 계산하지 않아 수렴 보장이 사라짐

---

## 2. Semi-Gradient Off-policy TD

### Importance Sampling Ratio

$$\rho_t \doteq \frac{\pi(A_t \mid S_t)}{b(A_t \mid S_t)}$$

### Semi-Gradient TD(0) (Off-policy)

$$\mathbf{w}_{t+1} = \mathbf{w}_t + \alpha\,\rho_t\,\delta_t\,\nabla_{\mathbf{w}}\hat{v}(S_t, \mathbf{w}_t)$$

$$\delta_t = R_{t+1} + \gamma\hat{v}(S_{t+1}, \mathbf{w}_t) - \hat{v}(S_t, \mathbf{w}_t)$$

### 문제: 불안정성

$\rho_t$가 gradient에 곱해지지만 target $\hat{v}(S_{t+1}, \mathbf{w}_t)$에는 곱해지지 않는 **비대칭 구조** 로 인해, 일반적인 함수 근사에서 발산 가능합니다.

---

## 3. Deadly Triad: 발산의 3조건

**Deadly Triad** 는 RL 함수 근사에서 발산을 유발하는 세 조건이 **동시에** 충족될 때 발생합니다.

$$\boxed{\text{Deadly Triad} = \text{Function Approximation} + \text{Bootstrapping} + \text{Off-policy}}$$

| 조건 | 설명 | 단독으로는? |
|---|---|---|
| **Function Approximation** | 선형 또는 비선형 근사 | 안전 |
| **Bootstrapping** | 다른 추정값으로 업데이트 (TD, DP) | 안전 |
| **Off-policy** | behavior ≠ target policy | 안전 |
| **셋 모두** | → 발산 가능 | **위험** |

### 각 조건을 제거하면?

- Bootstrapping 제거 → MC (수렴하지만 느림, episodic 한정)
- Off-policy 제거 → On-policy TD (Ch.9–10, 안정)
- 함수 근사 제거 → Tabular off-policy TD (Ch.6, 안정)

실용적으로는 세 조건 모두 **필요하므로**, 해결책이 필요합니다.

---

## 4. 발산 예시: Baird's Counterexample

### 설정

7개 상태, 2개 행동(점선/실선)이 있는 MDP:

```
상태 1~6: 점선 행동 → 상태 7로 이동 (확률 6/7)
          실선 행동 → 자기 자신으로 이동
상태 7:   점선/실선 모두 → 균등 상태로 이동
```

- **Behavior policy $b$**: 점선/실선 각 50%
- **Target policy $\pi$**: 항상 점선

### 선형 근사

$$\hat{v}(s, \mathbf{w}) = \mathbf{w}^\top \phi(s)$$

(각 상태마다 특수한 특징 벡터 $\phi(s)$ 설계)

### 결과

Semi-gradient off-policy TD를 적용하면 **$\mathbf{w}$의 모든 성분이 발산(diverge)** 합니다. $\overline{VE}$는 iteration마다 증가합니다. 이 예시는 Deadly Triad 세 조건 모두 충족 시 안정성 보장이 없음을 **명확히 증명** 합니다.

---

## 5. Gradient-TD Methods

Deadly Triad를 해결하기 위해 **진짜 gradient descent** 를 수행하는 방법입니다.

### 목적함수: Projected Bellman Error (PBE)

$$\overline{PBE}(\mathbf{w}) \doteq \left\|\Pi T^\pi \hat{v}_{\mathbf{w}} - \hat{v}_{\mathbf{w}}\right\|_\mu^2$$

- $\Pi$: $\mu$-가중 $L_2$ 노름 기준 함수 근사 공간으로의 투영(projection)
- $T^\pi$: Bellman expectation operator

### GTD2 (Gradient-TD 2)

보조 가중치 벡터 $\mathbf{v} \in \mathbb{R}^d$를 도입하여 두 시스템을 동시에 업데이트:

$$\delta_t = R_{t+1} + \gamma\mathbf{w}_t^\top \mathbf{x}_{t+1} - \mathbf{w}_t^\top \mathbf{x}_t$$

$$\mathbf{v}_{t+1} = \mathbf{v}_t + \beta\left(\delta_t - \mathbf{v}_t^\top \mathbf{x}_t\right)\mathbf{x}_t$$

$$\boxed{\mathbf{w}_{t+1} = \mathbf{w}_t + \alpha\left(\mathbf{x}_t - \gamma\mathbf{x}_{t+1}\right)\left(\mathbf{v}_t^\top \mathbf{x}_t\right)}$$

### TDC (TD with Gradient Correction)

$$\boxed{\mathbf{w}_{t+1} = \mathbf{w}_t + \alpha\delta_t\mathbf{x}_t - \alpha\gamma\mathbf{x}_{t+1}\left(\mathbf{v}_t^\top\mathbf{x}_t\right)}$$

$$\mathbf{v}_{t+1} = \mathbf{v}_t + \beta\left(\delta_t - \mathbf{v}_t^\top\mathbf{x}_t\right)\mathbf{x}_t$$

### GTD2 vs. TDC vs. Semi-gradient TD

| | Semi-gradient TD | GTD2 | TDC |
|---|---|---|---|
| 수렴 보장 (off-policy) | ❌ | ✅ | ✅ |
| 계산 복잡도 | $O(d)$ | $O(d)$ | $O(d)$ |
| 파라미터 | 1개 벡터 | 2개 벡터 | 2개 벡터 |
| 수렴 속도 | 빠름 | 느림 | TDC가 GTD2보다 빠름 |

---

## 6. Emphatic-TD Methods

### 동기: 분포 불일치 직접 해결

Off-policy의 핵심 문제는 $\mu$(behavior 분포)와 $\pi$(target 분포)의 불일치입니다. **Emphatic-TD** 는 업데이트 강조도(emphasis)를 조정해 이를 보정합니다.

### Emphasis $M_t$

$$M_t = \lambda i(S_t) + (1-\lambda)\,\rho_{t-1}\gamma M_{t-1}$$

- $i(s)$: 상태 $s$의 관심도(interest), 사용자 정의
- $\rho_{t-1}$: 직전 step의 importance sampling ratio

### Emphatic-TD 업데이트

$$\mathbf{w}_{t+1} = \mathbf{w}_t + \alpha M_t\,\rho_t\,\delta_t\,\nabla_{\mathbf{w}}\hat{v}(S_t, \mathbf{w}_t)$$

- 중요한 상태에는 **더 큰 emphasis** → 분포 불일치 보정
- 선형 근사에서 수렴 증명 가능

---

## 7. 전체 요약 및 이후 챕터와의 연결

### Chapter 11 구조 요약

```
Off-policy + 함수 근사
        ↓
Deadly Triad (세 조건 동시 충족 → 발산 위험)
  ├── Function Approximation
  ├── Bootstrapping
  └── Off-policy
        ↓
발산 예시: Baird's Counterexample
        ↓
해결책
  ├── Gradient-TD (GTD2, TDC): 진짜 gradient descent
  │       PBE 목적함수 최소화, 2개 가중치 벡터
  └── Emphatic-TD: emphasis로 분포 불일치 보정
```

### 핵심 수식 한눈에 보기

**TDC 업데이트**:
$$\mathbf{w} \leftarrow \mathbf{w} + \alpha\delta_t\mathbf{x}_t - \alpha\gamma\mathbf{x}_{t+1}(\mathbf{v}^\top\mathbf{x}_t)$$
$$\mathbf{v} \leftarrow \mathbf{v} + \beta(\delta_t - \mathbf{v}^\top\mathbf{x}_t)\mathbf{x}_t$$

**Emphatic-TD**:
$$\mathbf{w} \leftarrow \mathbf{w} + \alpha M_t\rho_t\delta_t\nabla_{\mathbf{w}}\hat{v}(S_t,\mathbf{w})$$

### 이후 챕터와의 연결

| 챕터 | 주제 | 연결점 |
|---|---|---|
| Ch.12 | Eligibility Traces | Gradient-TD + λ-return 결합 |
| Ch.13 | Policy Gradient | off-policy 문제를 policy 직접 최적화로 우회 |
| Ch.16 | DQN | Experience Replay로 Deadly Triad 완화 |

---

> **다음 챕터로**: Chapter 12에서는 TD(0)부터 Monte Carlo까지를 **하나의 파라미터 $\lambda$** 로 통합하는 Eligibility Traces를 다룹니다.
