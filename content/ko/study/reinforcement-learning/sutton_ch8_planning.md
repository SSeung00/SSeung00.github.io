---
title: "Reinforcement Learning: Chapter 8 Planning and Learning with Tabular Methods"
category: "Reinforcement Learning"
weight: 8
date: 2026-04-09
---

# Reinforcement Learning Chapter 8: Planning and Learning with Tabular Methods

> Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed.) — Chapter 8 핵심 정리

---

## 목차

1. [Model-based RL이란](#1-model-based-rl이란)
2. [Model의 종류: Distribution vs. Sample Model](#2-model의-종류-distribution-vs-sample-model)
3. [Dyna 아키텍처](#3-dyna-아키텍처)
4. [Dyna-Q 알고리즘](#4-dyna-q-알고리즘)
5. [환경 변화에 대한 대응: Dyna-Q+](#5-환경-변화에-대한-대응-dyna-q)
6. [Prioritized Sweeping](#6-prioritized-sweeping)
7. [Planning을 Decision Time에: MCTS](#7-planning을-decision-time에-mcts)
8. [Rollout Algorithm](#8-rollout-algorithm)
9. [Expected vs. Sample Updates 비교](#9-expected-vs-sample-updates-비교)
10. [전체 요약 및 이후 챕터와의 연결](#10-전체-요약-및-이후-챕터와의-연결)

---

## 1. Model-based RL이란

### 두 가지 RL의 접근 방향

지금까지 배운 방법들(TD, MC)은 **model-free** 방법입니다. 환경과 직접 상호작용하여 얻은 경험만으로 policy나 value function을 학습합니다.

**Model-based RL​**은 여기에 **환경 모델(world model)​**을 추가합니다:

```
Model-free RL:
    실제 환경 → 경험 → value/policy 업데이트

Model-based RL:
    실제 환경 → 경험 → 모델 학습
                  ↓         ↓
              value/policy  시뮬레이션 경험 (planning)
              업데이트      → value/policy 업데이트
```

### Model-based RL의 핵심 장점

**데이터 효율성(sample efficiency)**:

실제 환경과의 상호작용은 비용이 큽니다 (로봇 마모, 시간, 위험). 한 번 수집한 경험으로 모델을 학습하면, 그 모델을 이용해 **무한히 많은 가상 경험​**을 생성할 수 있습니다.

### Planning vs. Learning의 구분

| | Planning | Learning |
|---|---|---|
| 데이터 소스 | 모델 (시뮬레이션) | 실제 환경 경험 |
| 목적 | 모델로부터 value/policy 개선 | 모델 자체 또는 value/policy 학습 |
| 예시 | Value Iteration, MCTS | TD, MC |

Chapter 8의 핵심 통찰:

> **Planning과 Learning은 동일한 업데이트 연산을 수행하며, 데이터 소스만 다를 뿐이다.**

---

## 2. Model의 종류: Distribution vs. Sample Model

### Distribution Model (완전 모델)

모든 가능한 다음 상태와 보상의 **완전한 확률 분포​**를 제공합니다:

$$p(s', r \mid s, a), \quad \forall s' \in \mathcal{S},\ r \in \mathcal{R}$$

- Chapter 4 DP에서 사용한 형태
- 기댓값 계산 가능 → **Expected Update** 수행
- 저장 및 계산 비용이 큼

### Sample Model (샘플 모델)

실제 환경처럼 $(s, a)$가 주어지면 **하나의 $(s', r)$ 샘플​**을 생성합니다:

$$\text{Model}(s, a) \to (s', r)$$

- Distribution model이 있으면 샘플링으로 구현 가능
- 역은 성립하지 않음 (샘플만 있으면 완전 분포를 알 수 없음)
- **Sample Update** 수행 → 계산 비용 낮음

### 학습된 모델 (Learned Model)

실제 환경을 모를 때, 경험으로부터 모델을 학습합니다:

$$\hat{p}(s', r \mid s, a) \approx p(s', r \mid s, a)$$

학습된 모델은 **불완전하고 편향될 수 있습니다**. 이로 인해 model-based RL은 추가적인 오차 원천(model error)을 갖습니다.

---

## 3. Dyna 아키텍처

### 세 가지 경험의 흐름

Dyna는 실제 경험(direct RL)과 시뮬레이션 경험(planning)을 **통합하는 아키텍처​**입니다.

```
                 실제 환경
                    │
                    │ (S, A, R, S')
                    ↓
        ┌───────────────────────┐
        │      Direct RL        │  ← 실제 경험으로 직접 Q 업데이트
        │  (e.g., Q-learning)   │
        └───────────────────────┘
                    │
        ┌───────────────────────┐
        │     Model Learning    │  ← 경험으로 모델 업데이트
        │  Model(s,a) → (s',r)  │
        └───────────────────────┘
                    │
        ┌───────────────────────┐
        │       Planning        │  ← 모델로 시뮬레이션 → Q 업데이트
        │  (n번 반복 Q-learning) │
        └───────────────────────┘
```

### 핵심 아이디어

실제 환경과 한 번 상호작용할 때마다:
1. Q를 직접 업데이트 (direct RL)
2. 모델을 업데이트 (model learning)
3. 모델로 $n$번 시뮬레이션 → Q를 $n$번 추가 업데이트 (planning)

$n$이 클수록 실제 경험 한 번당 더 많은 가치 전파가 일어납니다.

---

## 4. Dyna-Q 알고리즘

### 알고리즘

```
Initialize: Q(s,a) ← arbitrary, Model(s,a) ← empty

Loop (에피소드 반복):
    S ← 현재 상태

    Loop (step 반복):
        (a) A ← ε-greedy(S, Q)
        (b) R, S' ← 실제 환경에서 실행
        (c) Q(S,A) ← Q(S,A) + α[R + γ max_a' Q(S',a') - Q(S,A)]
                                        ← Direct RL (Q-learning)
        (d) Model(S,A) ← (R, S')       ← Model Learning
        (e) Loop n번:                   ← Planning
                S_sim ← 방문한 적 있는 상태 중 무작위 선택
                A_sim ← S_sim에서 취한 적 있는 행동 중 무작위 선택
                R_sim, S'_sim ← Model(S_sim, A_sim)
                Q(S_sim, A_sim) ← Q(S_sim, A_sim)
                                  + α[R_sim + γ max_a' Q(S'_sim,a')
                                  - Q(S_sim, A_sim)]
        S ← S'
```

### Planning step의 효과

$n=0$이면 순수 Q-learning입니다. $n > 0$이면 각 실제 step마다 $n$번의 추가 업데이트가 일어나, **마치 $n+1$배 많은 실제 경험을 한 것처럼** 빠르게 value function이 수렴합니다.

### 수렴 비교 직관

미로 탐색 문제에서:
- $n=0$ (Q-learning): 목표에 도달한 경험이 역방향으로 천천히 전파됨
- $n=5$: 한 번의 성공 경험에서 5번의 추가 업데이트 → 훨씬 빠른 역방향 전파
- $n=50$: 거의 최적에 가까운 경로를 매우 빠르게 학습

---

## 5. 환경 변화에 대한 대응: Dyna-Q+

### 문제: Stale Model

환경이 변했는데 모델이 오래된 경험을 기반으로 하고 있다면, 잘못된 planning이 일어납니다.

두 가지 환경 변화 유형:
- **쉬운 변화 (shortcut)**: 원래 없던 지름길이 생김 → 기존 policy가 여전히 작동하므로 발견하기 어려움
- **어려운 변화 (blockage)**: 원래 있던 경로가 막힘 → 기존 policy가 실패하므로 자연스럽게 재탐색 발생

### Dyna-Q+ 아이디어

오랫동안 시도하지 않은 (state, action) pair에 **탐색 보너스(exploration bonus)​**를 부여합니다:

$$R^+ = R + \kappa \sqrt{\tau}$$

- $\tau$: 해당 $(s, a)$ pair가 마지막으로 방문된 이후 경과 time step 수
- $\kappa$: 탐색 보너스의 강도 (작은 양수)

오래 방문하지 않은 (s, a)일수록 높은 보너스 → 자연스럽게 재탐색 유도.

### 효과

- 환경이 변해 생긴 새로운 지름길(shortcut)을 빠르게 발견
- $\kappa$가 너무 크면 불필요한 탐색 과잉 → 하이퍼파라미터 튜닝 필요

---

## 6. Prioritized Sweeping

### 문제: 비효율적인 무작위 Planning

Dyna-Q의 planning step에서 업데이트할 (s, a)를 **무작위​**로 선택합니다. 하지만 모든 상태가 동등하게 중요하지는 않습니다.

**아이디어**: **TD error가 큰 (s, a)를 먼저 업데이트​**하면 더 효율적입니다.

$$|\delta| = \left|R + \gamma \max_{a'} Q(S', a') - Q(S, A)\right|$$

$|\delta|$가 크다는 것은 현재 추정값이 많이 틀렸다는 신호입니다.

### 알고리즘

```
Initialize: Q(s,a) ← 0, Model(s,a) ← empty
PQueue ← 우선순위 큐 (TD error 크기 기준)

Loop:
    (a) 실제 환경에서 (S, A, R, S') 수집
    (b) Model(S, A) ← (R, S') 업데이트
    (c) |δ| ← |R + γ max_a' Q(S',a') - Q(S,A)| 계산
        if |δ| > θ: PQueue에 (S,A) 삽입 (우선순위 = |δ|)

    (d) Loop n번 (priority sweeping):
            (S,A) ← PQueue에서 가장 높은 우선순위 꺼냄
            R, S' ← Model(S, A)
            Q(S,A) 업데이트
            For (S̄, Ā)가 S로 이어지는 모든 predecessor:
                R̄ ← Model(S̄, Ā)에서 예측 보상
                |δ̄| ← |R̄ + γ max_a' Q(S,a') - Q(S̄,Ā)| 계산
                if |δ̄| > θ: PQueue에 (S̄,Ā) 삽입
```

### Backward Propagation 효과

목표 상태에서의 보상이 발생하면:
1. 목표 직전 상태의 TD error가 커짐 → 우선순위 큐에 삽입
2. 그 상태를 업데이트하면 → 그 상태로 이어지는 상태들의 TD error가 커짐
3. 연쇄적으로 **역방향으로 빠르게 전파**

```
보상 발생
    ↓
목표 상태 업데이트 → 큰 TD error
    ↓
predecessor 추가 → 그것들도 TD error 커짐
    ↓
그것들의 predecessor 추가 → ...
    ↓
효율적인 역방향 전파
```

### Dyna-Q vs. Prioritized Sweeping 비교

| | Dyna-Q | Prioritized Sweeping |
|---|---|---|
| 업데이트 순서 | 무작위 | TD error 크기 순 |
| 계산량 (같은 수렴 수준) | 많은 planning step 필요 | 훨씬 적은 step으로 수렴 |
| 구현 복잡도 | 단순 | 우선순위 큐, predecessor 추적 필요 |
| 적합한 환경 | 소규모 | 중·대규모 상태 공간 |

---

## 7. Planning을 Decision Time에: MCTS

### Background Planning vs. Decision-time Planning

지금까지 본 planning은 **background planning​**입니다 — 실제 행동을 선택하기 전, 백그라운드에서 미리 계산하여 Q table을 개선합니다.

**Decision-time planning​**은 다릅니다 — **지금 이 상태에서 어떤 행동을 할지 결정하는 바로 그 순간​**에 계산을 수행합니다.

```
Background Planning:   미리 Q table 구축 → 행동 선택 시 Q table 참조
Decision-time Planning: 행동 선택 시점에 → 현재 상태에서 집중 탐색
```

### Monte Carlo Tree Search (MCTS)

**아이디어**: 현재 상태 $s$에서 시작하여 시뮬레이션을 반복하면서 **탐색 트리를 점진적으로 확장​**합니다.

MCTS의 4단계 반복:

```
1. Selection (선택)
   현재 트리에서 leaf node까지 UCB 기준으로 내려감

2. Expansion (확장)
   leaf node에서 새로운 자식 노드 1개 추가

3. Simulation (시뮬레이션, Rollout)
   새 노드에서 임의의 policy로 에피소드 끝까지 실행

4. Backpropagation (역전파)
   시뮬레이션 결과를 트리 위로 전파하여 통계 업데이트
```

### UCB for Trees (UCT)

트리 내에서 자식 노드를 선택하는 기준 (Chapter 2의 UCB 재활용):

$$\text{UCT}(s, a) = \bar{Q}(s, a) + c\sqrt{\frac{\ln N(s)}{N(s, a)}}$$

- $\bar{Q}(s, a)$: 노드 $(s, a)$를 통과한 시뮬레이션들의 평균 return
- $N(s)$: 상태 $s$ 방문 횟수
- $N(s, a)$: $(s, a)$에서 탐색한 횟수
- $c$: 탐색 강도 파라미터

**이 구조가 자연스러운 이유**: 많이 방문하지 않은 노드(불확실성이 높은 방향)를 자동으로 탐색합니다.

### MCTS의 장점

- **큰 상태 공간에서도 작동**: 현재 상태 중심으로 집중 탐색하므로 전체 상태 공간을 열거할 필요 없음
- **모델만 있으면 됨**: 실제 환경 없이 시뮬레이션만으로 행동 선택
- **AlphaGo, AlphaZero**: MCTS + 신경망(policy + value)의 조합으로 바둑 정복

---

## 8. Rollout Algorithm

### Rollout이란

MCTS의 Simulation 단계처럼, 어떤 상태에서 **고정된 rollout policy $\pi_{\text{ro}}$를 따라 에피소드 끝까지 실행​**하고 그 return을 사용하는 방법입니다.

### Rollout의 핵심 성질

**정리**: Rollout policy $\pi_{\text{ro}}$보다 나쁘지 않은 policy는 반드시 만들 수 있습니다.

각 상태 $s$에서 모든 행동 $a$에 대해 rollout을 수행하면:

$$\hat{q}(s, a) \approx q_{\pi_{\text{ro}}}(s, a)$$

이를 기반으로 greedy하게 행동을 선택하면:

$$\pi'(s) = \arg\max_a \hat{q}(s, a)$$

Policy Improvement Theorem에 의해 $\pi' \geq \pi_{\text{ro}}$.

### MC Tree Search와 Rollout의 관계

```
순수 Rollout:
    현재 상태 → 각 행동에 대해 rollout → greedy 선택
    (트리 없음, 단순)

MCTS:
    순수 Rollout + 점진적 트리 확장 + UCB 선택
    (트리 내부: 정교한 value 추정, 트리 외부: rollout)
```

MCTS는 rollout의 반복과 트리 확장을 결합하여, 중요한 상태-행동 쌍에 대해 점점 더 정확한 추정값을 축적합니다.

---

## 9. Expected vs. Sample Updates 비교

### 세 가지 업데이트 차원

Chapter 8은 업데이트 방법을 세 가지 축으로 정리합니다:

**축 1**: State value vs. Action value
- $V(s)$ 업데이트 vs. $Q(s, a)$ 업데이트

**축 2**: Expected update vs. Sample update
- Expected: 모든 가능한 결과를 평균 (distribution model 필요)
- Sample: 하나의 샘플 결과 사용 (sample model로 충분)

**축 3**: 깊이 (depth)
- 1-step, n-step, 끝까지

### 6가지 1-step 업데이트 분류

| | State value ($V$) | Action value ($Q$) |
|---|---|---|
| **Expected** | DP Policy Evaluation | Q-learning (Expected SARSA) |
| **Sample** | TD(0) | SARSA, Q-learning (sample) |

### Expected vs. Sample Update의 트레이드오프

**Expected Update**:
$$Q(s, a) \leftarrow \sum_{s', r} p(s', r | s, a)\left[r + \gamma \max_{a'} Q(s', a')\right]$$

- 분산 없음 (정확한 기댓값)
- 계산 비용: $O(|\mathcal{S}| \cdot |\mathcal{A}|)$ (모든 다음 상태 계산)
- Distribution model 필요

**Sample Update**:
$$Q(s, a) \leftarrow Q(s, a) + \alpha\left[r + \gamma \max_{a'} Q(s', a') - Q(s, a)\right]$$

- 분산 있음 (단일 샘플)
- 계산 비용: $O(1)$ (단일 샘플)
- Sample model으로 충분

**어느 것이 더 나은가?** 상태 공간의 크기에 따라 다릅니다:

- 분기 인수(branching factor, $|\mathcal{S}|$)가 작으면: Expected update가 유리 (분산 없음)
- 분기 인수가 크면: 같은 계산 비용으로 Sample update를 여러 번 수행하는 것이 유리

---

## 10. 전체 요약 및 이후 챕터와의 연결

### Chapter 8 구조 요약

```
Model-free (Ch.5~7)       Model-based (Ch.8)
경험 → value/policy        경험 → 모델 → planning → value/policy
                                  ↑
                           학습된 모델 (불완전할 수 있음)

Chapter 8의 핵심 방법들
        │
        ├── Dyna-Q
        │       Direct RL + Model Learning + Planning 통합
        │       n planning steps per real step
        │
        ├── Dyna-Q+
        │       탐색 보너스 r + κ√τ → 환경 변화 대응
        │
        ├── Prioritized Sweeping
        │       TD error 기반 우선순위 → 효율적 역방향 전파
        │
        ├── Decision-time Planning
        │   ├── Rollout: rollout policy로 즉각 개선
        │   └── MCTS: 트리 확장 + UCT + Rollout 결합
        │
        └── Expected vs. Sample Updates
                분기 인수에 따라 효율적 방법 다름
```

### 핵심 수식 한눈에 보기

**Dyna-Q Planning Update** (Q-learning과 동일, 데이터 소스만 다름):
$$Q(S_{\text{sim}}, A_{\text{sim}}) \leftarrow Q(S_{\text{sim}}, A_{\text{sim}}) + \alpha\left[R_{\text{sim}} + \gamma \max_{a'} Q(S'_{\text{sim}}, a') - Q(S_{\text{sim}}, A_{\text{sim}})\right]$$

**Dyna-Q+ 탐색 보너스**:
$$R^+ = R + \kappa\sqrt{\tau}$$

**Prioritized Sweeping 우선순위**:
$$\text{priority}(s, a) = \left|R + \gamma \max_{a'} Q(S', a') - Q(s, a)\right|$$

**UCT (MCTS 트리 내 선택)**:
$$\text{UCT}(s, a) = \bar{Q}(s, a) + c\sqrt{\frac{\ln N(s)}{N(s, a)}}$$

### RL 방법의 전체 지형도

Chapter 8은 RL 방법의 전체 스펙트럼을 하나로 통합하는 위치에 있습니다:

```
                   Model 없음          Model 있음
                      │                    │
                      ▼                    ▼
Tabular          MC, TD, Q-learning    Dyna-Q, MCTS
Function Approx  DQN, A3C              AlphaZero, Dreamer
```

### 이후 챕터로의 연결

| 챕터 | 주제 | Chapter 8과의 관계 |
|---|---|---|
| Ch.9~10 Function Approx. | 신경망으로 $V$, $Q$ 근사 | Dyna + 함수 근사 → world model RL |
| Ch.12 Eligibility Traces | TD(λ) | n-step return의 λ-가중 평균 |
| Ch.13 Policy Gradient | REINFORCE, Actor-Critic | MCTS + policy gradient → AlphaZero |

특히 **Dyna 아키텍처​**는 최근 deep RL의 **world model 기반 방법들** (Dreamer, MuZero 등)의 직접적인 원형입니다. 모델을 학습하고 그것으로 planning하는 핵심 아이디어가 신경망 기반으로 확장된 것입니다.

---

> **다음 챕터로**: Chapter 9에서는 상태 공간이 너무 크거나 연속적이어서 표(table)로 표현할 수 없을 때, **함수 근사(function approximation)​**를 사용해 $v_\pi$와 $q_\pi$를 근사하는 방법을 다룹니다.
