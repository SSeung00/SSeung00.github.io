---
title: "Unit 4 — 기다리지 않고 배우기: Temporal Difference"
date: 2026-04-13
category: "Reinforcement Learning"
weight: 5
---

## [Hook] 끝까지 기다려야 할까?

Unit 3에서 Monte Carlo를 배웠다.

경험의 평균으로 가치를 추정하는 방법. 모델이 없어도 된다.
블랙잭처럼 에피소드가 짧은 문제에서 잘 작동한다.

그런데 Unit 3 마지막에 이런 질문을 남겼다.

**에피소드가 매우 길거나, 아예 끝이 없다면?**

체스 한 판이 200 수. MC는 200 수를 전부 두고 결과(승/패)가 나온 뒤에야 첫 수의 가치를 업데이트한다.

그 200 수 동안, 이미 충분한 정보가 있음에도 **아무것도 배우지 않는다.**

더 깊은 문제가 있다. MC의 업데이트 수식을 다시 보자:

$$V(s) \leftarrow V(s) + \alpha\bigl[\underbrace{G}_{\text{에피소드 끝까지 기다려야 앎}} - V(s)\bigr]$$

$G$ 를 알려면 에피소드가 끝나야 한다. 이것이 MC의 구조적 제약이다.

**한 발짝만 내딛고 바로 업데이트할 수는 없을까?**

---

## [Fail] MC는 에피소드 중간에 아무것도 배우지 않는다

아래 시뮬레이터에서 MC와 TD가 **똑같은 에피소드** 를 경험하는 모습을 비교해보자.

{{< simulator src="/simulator/unit-4-td/mc-vs-td.html" height="600px" title="Simulator — MC vs TD 수렴 비교" >}}

탐구해볼 것들:

- **1 에피소드** 를 눌러보자. 에이전트가 한 칸씩 이동할 때마다:
  - **왼쪽(MC) 패널**: 에피소드가 끝날 때까지 값이 전혀 변하지 않는다. "FROZEN" 상태.
  - **오른쪽(TD) 패널**: 매 step마다 해당 셀이 즉시 업데이트된다.

- **100 에피소드** 를 실행한 뒤 RMSE 그래프를 비교하자.
  TD는 같은 에피소드 수로 더 낮은 오차에 도달하는가?

- **γ를 바꿔보자.** γ = 0.5 vs γ = 0.99일 때 수렴 속도는 어떻게 달라지는가?

MC가 에피소드 중간에 멈춰 있는 것 — 이것이 MC의 핵심 제약이다.

---

## [Idea] 추측으로 추측을 수정한다

핵심 아이디어는 이것이다.

> 에피소드 끝까지 기다리지 말고,
> **지금 받은 보상 + 다음 상태의 현재 추정값** 으로 바로 업데이트하자.

$$\underbrace{V(s)}_{\text{현재 추정}} \leftarrow V(s) + \alpha\!\left[\underbrace{R + \gamma V(s')}_{\text{TD 목표}} - V(s)\right]$$

$G$ (실제 리턴) 대신 $R + \gamma V(s')$ (1-step 부트스트랩 추정값)을 사용한다.
이 업데이트는 **한 번의 전이** $(s, R, s')$ 만 있으면 즉시 실행할 수 있다.

이것이 **TD(0)** 이다. (0은 "0 step 앞을 부트스트래핑한다"는 뜻)

$\bigl[R + \gamma V(s') - V(s)\bigr]$ 를 **TD 오차(TD error)** 라고 부른다.
현재 추정값과 더 나은 1-step 추정값의 차이다.

### 운전 비유

서울에서 부산까지 운전한다. 출발 전 "4시간 걸릴 것"이라고 예측했다.

**MC:** 부산에 도착한 뒤 "5시간 걸렸다"고 기록하고 다음 여행 예측을 수정한다.

**TD:** 대전 톨게이트를 지나며 "여기서 부산까지 보통 3.5시간, 이미 2시간 왔으니 총 5.5시간"이라고 **중간에 예측을 수정** 한다.

도착하지 않아도 계속 더 나은 추정이 이루어진다.

---

## [Form] 수식으로 압축

### 1. TD(0) — 예측

임의의 정책 $\pi$ 의 가치 함수를 추정한다.

$$\boxed{V(s) \leftarrow V(s) + \alpha\bigl[R + \gamma V(s') - V(s)\bigr]}$$

```
초기화: V(s) = 0  for all s

에피소드마다:
  s ← 시작 상태
  while s가 종료 상태가 아닐 동안:
    a ← π(s)로 행동 선택
    s', R ← 환경에서 수신
    V(s) ← V(s) + α[R + γV(s') - V(s)]   ← 즉시 업데이트
    s ← s'
```

에피소드 끝을 기다리지 않고, **매 step마다** 업데이트한다.

---

### 2. SARSA — On-policy TD Control

$V(s)$ 대신 **행동 가치 $Q(s,a)$** 를 학습한다.
이름 SARSA는 업데이트에 쓰이는 다섯 요소 $S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1}$ 에서 왔다.

$$\boxed{Q(s,a) \leftarrow Q(s,a) + \alpha\bigl[R + \gamma\, Q(s', a') - Q(s,a)\bigr]}$$

$a'$ 는 **ε-greedy 정책으로 선택한 다음 행동** 이다.

```
초기화: Q(s,a) = 0  for all s, a

에피소드마다:
  s ← 시작;  a ← ε-greedy(Q, s)

  while s가 종료가 아닐 동안:
    s', R ← a 실행
    a' ← ε-greedy(Q, s')        ← 다음 행동도 현재 정책으로
    Q(s,a) ← Q(s,a) + α[R + γQ(s',a') - Q(s,a)]
    s ← s';  a ← a'
```

**On-policy:** 탐험(ε-greedy)을 포함한 실제 행동 정책의 가치를 학습한다.

---

### 3. Q-Learning — Off-policy TD Control

SARSA와 한 줄 차이. $a'$ 를 현재 정책으로 선택하는 대신 **greedy max** 를 사용한다.

$$\boxed{Q(s,a) \leftarrow Q(s,a) + \alpha\bigl[R + \gamma \max_{a'} Q(s', a') - Q(s,a)\bigr]}$$

```
초기화: Q(s,a) = 0  for all s, a

에피소드마다:
  s ← 시작 상태

  while s가 종료가 아닐 동안:
    a ← ε-greedy(Q, s)           ← 탐험을 위해 ε-greedy
    s', R ← a 실행
    Q(s,a) ← Q(s,a) + α[R + γ·max_a' Q(s',a') - Q(s,a)]  ← greedy 목표
    s ← s'
```

**Off-policy:** 탐험(ε-greedy)으로 행동하면서도, **최적 정책(greedy)** 의 가치를 학습한다.
행동 정책과 목표 정책이 분리된다.

---

### 4. MC vs TD 비교

| | Monte Carlo | TD(0) |
|---|---|---|
| **업데이트 시점** | 에피소드 종료 후 | 매 step |
| **업데이트 목표** | 실제 리턴 $G$ | 추정값 $R + \gamma V(s')$ |
| **부트스트래핑** | ✗ | ✓ |
| **편향(Bias)** | 없음 | 있음 (초기 추정값에 의존) |
| **분산(Variance)** | 높음 | 낮음 |
| **연속 태스크** | ✗ 불가 | ✓ 가능 |

**편향-분산 트레이드오프:**
MC는 불편추정량이지만 에피소드마다 리턴이 크게 달라 분산이 높다.
TD는 추정값에 의존하므로 편향이 있지만 분산이 낮아 수렴이 빠르다.

<details>
<summary>더 알고 싶다면 — TD의 수렴 보장</summary>

**TD(0)의 수렴:**
테이블 표현(tabular)과 충분히 작은 α 에서, 모든 상태를 무한히 방문하면 $V \to V_\pi$ 가 보장된다.

**SARSA의 수렴:**
GLIE(Greedy in the Limit with Infinite Exploration) 조건 — ε→0 하면서 모든 (s,a)를 무한히 방문 — 하에서 최적 정책으로 수렴한다.

**Q-Learning의 수렴:**
모든 (s,a)를 무한히 방문하고, step size가 Robbins-Monro 조건 $\sum \alpha_t = \infty, \sum \alpha_t^2 < \infty$ 을 만족하면 최적 Q*로 수렴한다.

</details>

---

## [Next] 다음 질문

TD(0)은 1 step 앞을 본다. MC는 에피소드 전체를 본다.

이 둘 사이 어딘가가 최적일 수 있다.

**2 step 앞을 보면? 4 step? 8 step?**

$$G_t^{(n)} = R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{n-1}R_{t+n} + \gamma^n V(S_{t+n})$$

n = 1이면 TD(0), n = ∞이면 MC.
그 사이 어딘가에 "가장 빠르게 수렴하는" 최적의 n이 있다.

이것이 **n-step TD** 이고, 이를 응용하면 **Dyna-Q** — 수집한 경험으로 모델을 만들어 반복 학습하는 알고리즘 — 로 이어진다.

다음 Unit에서는 n-step 스펙트럼과 계획(Planning)을 배운다.
