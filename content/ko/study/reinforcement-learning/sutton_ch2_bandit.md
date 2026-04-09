---
title: "Reinforcement Learning: Chapter 2 Multi-armed Bandits"
category: Reinforcement Learning
weight: 2
date: 2026-04-09
---

# Reinforcement Learning: Chapter 2 Multi-armed Bandits

> Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed.) — Chapter 2 핵심 정리

---

## 목차

1. [문제 정의: k-armed Bandit](#1-문제-정의-k-armed-bandit)
2. [Action-Value 추정: Sample Average](#2-action-value-추정-sample-average)
3. [Exploration vs. Exploitation](#3-exploration-vs-exploitation)
4. [Incremental Update Rule 유도](#4-incremental-update-rule-유도)
5. [Nonstationary 환경: Constant Step-Size와 지수 가중 평균](#5-nonstationary-환경-constant-step-size와-지수-가중-평균)
6. [Step-Size 수렴 조건 (Robbins-Monro)](#6-step-size-수렴-조건-robbins-monro)
7. [Optimistic Initial Values](#7-optimistic-initial-values)
8. [UCB: Upper Confidence Bound](#8-ucb-upper-confidence-bound)
9. [Gradient Bandit Algorithm](#9-gradient-bandit-algorithm)
10. [전체 요약 및 Chapter 13과의 연결](#10-전체-요약-및-chapter-13과의-연결)

---

## 1. 문제 정의: k-armed Bandit

**k-armed Bandit​**은 강화학습의 핵심 딜레마를 가장 단순한 형태로 포착한 문제입니다.

- $k$개의 행동(action) $a \in \{1, \ldots, k\}$ 중 하나를 매 time step마다 선택
- 행동 $a$를 선택하면 확률 분포 $p(r \mid a)$에서 보상 $R_t$가 주어짐
- **목표**: 총 기대 보상을 최대화

각 행동의 **진짜 가치(true value)​**는 다음과 같이 정의됩니다:

$$q_*(a) \doteq \mathbb{E}[R_t \mid A_t = a]$$

이 $q_*(a)$를 정확히 알면 항상 최적 행동을 선택할 수 있지만, 실제로는 **추정값 $Q_t(a)$​**만 가질 수 있습니다.

---

## 2. Action-Value 추정: Sample Average

가장 자연스러운 추정 방법은 지금까지 받은 보상의 **표본 평균​**입니다:

$$Q_t(a) \doteq \frac{\sum_{i=1}^{t-1} R_i \cdot \mathbf{1}_{A_i = a}}{\sum_{i=1}^{t-1} \mathbf{1}_{A_i = a}}$$

대수의 법칙에 의해 충분히 많이 시도하면 $Q_t(a) \xrightarrow{p} q_*(a)$.

---

## 3. Exploration vs. Exploitation

강화학습을 지도학습과 구별하는 **가장 핵심적인 특성​**이 여기서 등장합니다.

| 전략 | 설명 | 문제점 |
|---|---|---|
| **Greedy** | 항상 $\arg\max_a Q_t(a)$ 선택 | 탐색 없음 → 지역 최적에 갇힘 |
| **ε-Greedy** | 확률 $\varepsilon$으로 무작위, $1-\varepsilon$으로 greedy | 무차별적 탐색 |
| **UCB** | 불확실성이 높은 행동 우선 탐색 | 비정상 환경에서 취약 |
| **Gradient Bandit** | 선호도(preference) 기반 선택 | 보상의 절대값이 아닌 상대값에 의존 |

$$\boxed{\text{Exploration vs. Exploitation: 동시에 완벽히 할 수 없다}}$$

---

## 4. Incremental Update Rule 유도

매번 처음부터 평균을 계산하는 것은 메모리와 연산이 $O(n)$으로 증가합니다. 이를 $O(1)$의 점화식으로 바꿉니다.

### 유도

$Q_{n+1} = \frac{1}{n}\sum_{i=1}^{n} R_i$에서 출발합니다:

$$Q_{n+1} = \frac{1}{n}\left(R_n + \sum_{i=1}^{n-1} R_i\right)$$

$$= \frac{1}{n}\left(R_n + (n-1) \cdot \underbrace{\frac{1}{n-1}\sum_{i=1}^{n-1} R_i}_{= Q_n}\right)$$

$$= \frac{1}{n}\left(R_n + (n-1)Q_n\right) = \frac{1}{n}\left(R_n + nQ_n - Q_n\right)$$

$$\boxed{Q_{n+1} = Q_n + \frac{1}{n}\left[R_n - Q_n\right]}$$

### 일반 업데이트 패턴

이 형태는 책 전체를 관통하는 **핵심 패턴​**입니다:

$$\text{NewEstimate} \leftarrow \text{OldEstimate} + \underbrace{\text{StepSize}}_{\alpha} \times \underbrace{[\text{Target} - \text{OldEstimate}]}_{\text{prediction error}}$$

$[R_n - Q_n]$은 **예측 오차(prediction error)​**로, 이 개념은 TD learning까지 그대로 이어집니다.

---

## 5. Nonstationary 환경: Constant Step-Size와 지수 가중 평균

환경이 시간에 따라 변한다면, 오래된 보상보다 **최근 보상에 더 큰 가중치​**를 주어야 합니다. 이를 위해 $\frac{1}{n}$ 대신 고정된 $\alpha \in (0, 1]$을 사용합니다.

### 지수 가중 평균 유도

$$Q_{n+1} = Q_n + \alpha[R_n - Q_n] = (1-\alpha)Q_n + \alpha R_n$$

이를 재귀적으로 전개합니다:

$$= (1-\alpha)\left[(1-\alpha)Q_{n-1} + \alpha R_{n-1}\right] + \alpha R_n$$

$$= (1-\alpha)^2 Q_{n-1} + \alpha(1-\alpha)R_{n-1} + \alpha R_n$$

$$\vdots$$

$$\boxed{Q_{n+1} = (1-\alpha)^n Q_1 + \sum_{i=1}^{n} \alpha(1-\alpha)^{n-i} R_i}$$

### 가중치 합 검증

이것이 진짜 가중 평균인지 확인합니다 (가중치 합 = 1):

$$\underbrace{(1-\alpha)^n}_{\text{초기값}} + \sum_{i=1}^{n} \alpha(1-\alpha)^{n-i}$$

등비급수 공식 $\sum_{i=1}^{n} r^{i-1} = \frac{1-r^n}{1-r}$을 적용하면:

$$= (1-\alpha)^n + \alpha \cdot \frac{1-(1-\alpha)^n}{1-(1-\alpha)} = (1-\alpha)^n + 1 - (1-\alpha)^n = 1 \checkmark$$

**직관**: $i$가 과거일수록 $(1-\alpha)^{n-i}$이 exponential하게 작아짐 → 오래된 보상은 빠르게 망각.

---

## 6. Step-Size 수렴 조건 (Robbins-Monro)

### 두 조건

확률적 근사(stochastic approximation) 이론에서, $Q_n \to q_*$로 수렴하기 위한 필요충분조건:

$$\sum_{n=1}^{\infty} \alpha_n = \infty \tag{조건 1}$$

$$\sum_{n=1}^{\infty} \alpha_n^2 < \infty \tag{조건 2}$$

### 각 조건의 직관

업데이트를 분해하면:

$$Q_{n+1} = Q_n + \alpha_n \underbrace{[q_*(a) - Q_n]}_{\text{신호}} + \alpha_n \underbrace{[R_n - q_*(a)]}_{\text{노이즈}}$$

- **조건 1** ($\sum \alpha_n = \infty$): 신호 누적량이 무한해야 초기값과 무관하게 $q_*$까지 도달 가능
- **조건 2** ($\sum \alpha_n^2 < \infty$): 노이즈 누적 분산이 유한해야 진동 없이 수렴 가능

### 주요 step-size 비교

| $\alpha_n$ | $\sum \alpha_n$ | $\sum \alpha_n^2$ | 수렴? | 비고 |
|---|---|---|---|---|
| $\frac{1}{n}$ | $\infty$ ✅ | $\frac{\pi^2}{6} < \infty$ ✅ | ✅ | Stationary에 적합 |
| $\alpha$ (상수) | $\infty$ ✅ | $\infty$ ❌ | ❌ | Nonstationary에 유용 (추적 가능) |

### 조화급수 발산 증명 ($\sum \frac{1}{n} = \infty$)

묶음법(Cauchy condensation):

$$1 + \frac{1}{2} + \underbrace{\left(\frac{1}{3}+\frac{1}{4}\right)}_{> \frac{1}{2}} + \underbrace{\left(\frac{1}{5}+\cdots+\frac{1}{8}\right)}_{> \frac{1}{2}} + \underbrace{\left(\frac{1}{9}+\cdots+\frac{1}{16}\right)}_{> \frac{1}{2}} + \cdots = \infty$$

### $\sum \frac{1}{n^2} < \infty$ 증명

비교 판정법:

$$\frac{1}{n^2} \leq \frac{1}{n(n-1)} = \frac{1}{n-1} - \frac{1}{n}$$

$$\therefore \sum_{n=2}^{\infty} \frac{1}{n^2} \leq \sum_{n=2}^{\infty}\left(\frac{1}{n-1} - \frac{1}{n}\right) = 1 < \infty \checkmark$$

---

## 7. Optimistic Initial Values

$Q_1(a)$를 실제 기댓값보다 **낙관적으로 높게** 초기화하면, 어떤 행동을 선택해도 실망(보상 < 기댓값)하게 되어 자연스럽게 **모든 행동을 탐색​**하게 됩니다.

```
초기값 Q₁(a) = +5  (실제 q*(a) ≈ 0 수준)
→ 어떤 행동을 해도 R < Q → prediction error < 0
→ 다른 행동 시도 → 결국 전체 탐색
```

**장단점**:
- ✅ 추가 파라미터 없이 초기 탐색 강제
- ❌ Nonstationary 환경에서는 부적합 (초기 편향이 지속적으로 영향)
- ❌ 초기값이 얼마나 낙관적이어야 하는지 사전에 알기 어려움

---

## 8. UCB: Upper Confidence Bound

### 동기: ε-Greedy의 한계

ε-greedy는 탐색할 때 **모든 행동을 동일하게** 취급합니다. 하지만 직관적으로, 이미 충분히 시도한 행동보다 **적게 시도한(불확실한) 행동​**을 우선 탐색하는 것이 더 합리적입니다.

### UCB 공식

$$A_t = \arg\max_a \left[Q_t(a) + c\sqrt{\frac{\ln t}{N_t(a)}}\right]$$

- $N_t(a)$: 시각 $t$까지 행동 $a$가 선택된 횟수
- $c > 0$: 탐색 정도 조절 파라미터
- $\sqrt{\frac{\ln t}{N_t(a)}}$: **불확실성(신뢰 상한 폭)**

### Hoeffding's Inequality로부터의 유도

**Hoeffding's Inequality**: 보상이 $[0,1]$에 bounded일 때:

$$P\left(q_*(a) > Q_t(a) + u\right) \leq e^{-2N_t(a) u^2}$$

$u = c\sqrt{\frac{\ln t}{N_t(a)}}$로 설정하면:

$$P\left(q_*(a) > Q_t(a) + c\sqrt{\frac{\ln t}{N_t(a)}}\right) \leq e^{-2N_t(a) \cdot c^2 \cdot \frac{\ln t}{N_t(a)}} = e^{-2c^2 \ln t} = t^{-2c^2}$$

즉, **진짜 가치가 신뢰 상한을 초과할 확률이 $t^{-2c^2}$로 빠르게 감소** — UCB 항은 $q_*(a)$의 "거의 확실한" 상한입니다.

### UCB 항의 거동 분석

| 상황 | UCB 항 크기 | 의미 |
|---|---|---|
| $N_t(a)$ 작음 | 크다 | 적게 시도한 행동 → 탐색 유도 |
| $N_t(a)$ 충분히 큼 | 작다 | 충분히 탐색됨 → 착취로 전환 |
| $t$ 증가 | $\ln t$로 천천히 증가 | 시간이 지나도 탐색 욕구 유지 |

### Regret Bound

**Regret​**을 정의하면:

$$\text{Regret}_T = \sum_{t=1}^T \left[q_*(a^*) - q_*(A_t)\right]$$

UCB는 이론적으로 $O(\ln T)$ regret을 달성합니다. ε-greedy의 $O(\varepsilon T)$보다 점근적으로 훨씬 우수합니다.

---

## 9. Gradient Bandit Algorithm

### 기본 아이디어

보상값 자체를 추정하는 대신, 각 행동에 대한 **수치 선호도(numerical preference) $H_t(a)$​**를 학습합니다. 선택 확률은 softmax로 정의됩니다:

$$\pi_t(a) \doteq \frac{e^{H_t(a)}}{\sum_{b=1}^k e^{H_t(b)}}$$

### 전체 유도 로드맵

```
목표: E[Rₜ] = Σₐ πₜ(a)·q*(a) 최대화 (gradient ascent)
        ↓
∂E[Rₜ]/∂Hₜ(a) = Σₓ q*(x) · ∂πₜ(x)/∂Hₜ(a)
        ↓
핵심: softmax 편미분 계산
        ↓
baseline 추가 → gradient 불변 증명
        ↓
기댓값 → 샘플 근사 → 최종 update rule
```

### Step 1: Softmax 편미분

분모를 $Z = \sum_b e^{H_t(b)}$로 표기합니다.

**Case 1: $x = a$ (자기 자신에 대해 미분)**

$$\frac{\partial \pi_t(a)}{\partial H_t(a)} = \frac{e^{H_t(a)} \cdot Z - e^{H_t(a)} \cdot e^{H_t(a)}}{Z^2} = \frac{e^{H_t(a)}}{Z}\cdot\frac{Z - e^{H_t(a)}}{Z} = \pi_t(a)(1 - \pi_t(a))$$

**Case 2: $x \neq a$ (다른 행동에 대해 미분)**

$$\frac{\partial \pi_t(x)}{\partial H_t(a)} = \frac{0 \cdot Z - e^{H_t(x)} \cdot e^{H_t(a)}}{Z^2} = -\frac{e^{H_t(x)}}{Z}\cdot\frac{e^{H_t(a)}}{Z} = -\pi_t(x)\pi_t(a)$$

**두 경우를 통합**:

$$\boxed{\frac{\partial \pi_t(x)}{\partial H_t(a)} = \pi_t(x)\left(\mathbf{1}_{x=a} - \pi_t(a)\right)}$$

### Step 2: Gradient 계산

$$\frac{\partial \mathbb{E}[R_t]}{\partial H_t(a)} = \sum_x q_*(x) \cdot \pi_t(x)\left(\mathbf{1}_{x=a} - \pi_t(a)\right)$$

### Step 3: Baseline 추가 — 편향 없음 증명

임의의 baseline $B_t$ ($x$와 무관)를 빼도 gradient가 0이 됨을 보입니다:

$$\sum_x B_t \cdot \pi_t(x)\left(\mathbf{1}_{x=a} - \pi_t(a)\right) = B_t\left[\pi_t(a) - \pi_t(a)\underbrace{\sum_x \pi_t(x)}_{=1}\right] = 0$$

따라서:

$$\frac{\partial \mathbb{E}[R_t]}{\partial H_t(a)} = \sum_x (q_*(x) - B_t) \cdot \pi_t(x)\left(\mathbf{1}_{x=a} - \pi_t(a)\right)$$

### Step 4: 기댓값 → 샘플 근사

$$= \mathbb{E}\left[(q_*(A_t) - B_t)\left(\mathbf{1}_{A_t=a} - \pi_t(a)\right)\right]$$

$q_*(A_t)$는 관측 불가능하므로 샘플 보상 $R_t$로 대체, baseline으로 $\bar{R}_t$ (보상의 이동 평균) 사용:

$$\frac{\partial \mathbb{E}[R_t]}{\partial H_t(a)} \approx (R_t - \bar{R}_t)\left(\mathbf{1}_{A_t=a} - \pi_t(a)\right)$$

### 최종 Update Rule

$$\boxed{H_{t+1}(a) \leftarrow H_t(a) + \alpha(R_t - \bar{R}_t)\left(\mathbf{1}_{A_t=a} - \pi_t(a)\right)}$$

풀어 쓰면:

$$H_{t+1}(A_t) \leftarrow H_t(A_t) + \alpha(R_t - \bar{R}_t)(1 - \pi_t(A_t))$$

$$H_{t+1}(a) \leftarrow H_t(a) - \alpha(R_t - \bar{R}_t)\pi_t(a), \quad \forall a \neq A_t$$

### Baseline의 역할

| 경우 | 업데이트 방향 |
|---|---|
| $R_t > \bar{R}_t$ (평균보다 좋은 보상) | $A_t$의 선호도 **증가**, 나머지 **감소** |
| $R_t < \bar{R}_t$ (평균보다 나쁜 보상) | $A_t$의 선호도 **감소**, 나머지 **증가** |
| $R_t = \bar{R}_t$ | 업데이트 없음 |

Baseline 없이 항상 양의 $R_t$만 쓰면 선택한 행동의 선호도가 **항상 증가하는 bias** 발생. Baseline이 이를 **상대적 비교​**로 교정합니다.

---

## 10. 전체 요약 및 Chapter 13과의 연결

### Chapter 2 구조 요약

```
k-armed Bandit 문제 정의
        ↓
Action-value 추정 (sample average)
        ↓
Exploration 전략들
  ├── ε-Greedy       ── 단순, 범용
  ├── Optimistic Init ── stationary 한정
  ├── UCB             ── 불확실성 기반, 이론적으로 강력
  └── Gradient Bandit ── 선호도 기반, policy gradient의 원형
        ↓
Nonstationary → constant α (지수 가중 평균)
        ↓
Associative Search → full RL로의 연결
```

### Gradient Bandit → REINFORCE (Chapter 13) 연결

Chapter 2의 Gradient Bandit은 Chapter 13 Policy Gradient의 **직접적인 원형​**입니다:

| 개념 | Gradient Bandit (Ch.2) | REINFORCE (Ch.13) |
|---|---|---|
| 상태 | 없음 | 상태 $s$ 있음 |
| 파라미터 | $H_t(a)$ | $\theta$ (파라미터화된 $\pi(a \mid s, \theta)$) |
| 업데이트 신호 | $R_t - \bar{R}_t$ | $G_t - b(s_t)$ (return - baseline) |
| 핵심 미분 | $\frac{\partial \pi_t(x)}{\partial H_t(a)}$ | $\nabla_\theta \ln \pi(a \mid s, \theta)$ (log-derivative trick) |

특히 **log-derivative trick​**은 동일한 softmax 편미분에서 바로 유도됩니다:

$$\frac{\partial \pi_t(x)}{\partial H_t(a)} = \pi_t(x)(\mathbf{1}_{x=a} - \pi_t(a)) = \pi_t(x) \cdot \frac{\partial \ln \pi_t(x)}{\partial H_t(a)}$$

### 핵심 수식 한눈에 보기

$$\underbrace{Q_{n+1} = Q_n + \frac{1}{n}[R_n - Q_n]}_{\text{Incremental update (stationary)}}$$

$$\underbrace{Q_{n+1} = (1-\alpha)^n Q_1 + \sum_{i=1}^{n}\alpha(1-\alpha)^{n-i}R_i}_{\text{Exponential recency-weighted average (nonstationary)}}$$

$$\underbrace{A_t = \arg\max_a\left[Q_t(a) + c\sqrt{\frac{\ln t}{N_t(a)}}\right]}_{\text{UCB action selection}}$$

$$\underbrace{H_{t+1}(a) \leftarrow H_t(a) + \alpha(R_t - \bar{R}_t)(\mathbf{1}_{A_t=a} - \pi_t(a))}_{\text{Gradient Bandit update}}$$

---

> **다음 챕터로**: Chapter 3에서는 상태(state)가 등장하고, 행동이 다음 상태에 영향을 미치는 **완전한 MDP(Markov Decision Process)** 설정으로 확장됩니다.
