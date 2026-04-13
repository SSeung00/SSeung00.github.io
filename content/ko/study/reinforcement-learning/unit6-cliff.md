---
title: "Unit 6 — 안전과 최적: On/Off-Policy"
date: 2026-04-13
category: "Reinforcement Learning"
weight: 7
---

## [Hook] 탐험의 위험을 감수할 것인가

Unit 4에서 SARSA(On-policy)와 Q-Learning(Off-policy)을 배웠다.

수식 하나 차이다.

- SARSA: $Q(s,a) \leftarrow Q(s,a) + \alpha[R + \gamma Q(s', \varepsilon\text{-greedy}(s')) - Q(s,a)]$
- Q-Learning: $Q(s,a) \leftarrow Q(s,a) + \alpha[R + \gamma \max_{a'} Q(s', a') - Q(s,a)]$

SARSA는 다음 행동을 **현재 ε-greedy 정책으로** 선택한다.
Q-Learning은 다음 행동을 **greedy 최선** 으로 선택한다.

작은 차이처럼 보인다. 하지만 특정 환경에서는 **전혀 다른 경로** 를 학습한다.

**위험한 절벽이 있다면?**

---

## [Fail] 최적 경로가 항상 최선은 아니다

Sutton & Barto의 고전적 예제: **Cliff Walking.**

4×12 격자 세계. 아래 줄 한가운데가 절벽이다.
출발은 왼쪽 아래, 목표는 오른쪽 아래.
절벽에 떨어지면 즉시 출발점으로 돌아가고 −100 패널티를 받는다.

{{< simulator src="/simulator/unit-6-cliff/cliff-walking.html" height="620px" title="Simulator — SARSA vs Q-Learning (Cliff Walking)" >}}

**"둘 다 비교" 탭** 으로 실험해보자.

탐구해볼 것들:

- **100 에피소드 이상** 학습시켜보자. 두 알고리즘의 경로가 어떻게 달라지는가?
  - Q-Learning: 절벽 바로 위를 따라 걷는 최단 경로 (−13)
  - SARSA: 절벽에서 한 줄 위를 우회하는 안전한 경로 (약 −17)

- **보상 그래프** 를 보자. 학습 중 누가 더 높은 평균 보상을 받는가?
  Q-Learning은 이론적 최적값(-13)에 수렴하지만, 탐험 중 절벽에 자주 떨어진다.
  SARSA는 탐험 실수를 감안한 정책을 학습해 학습 중 더 안정적이다.

- **ε를 0.3으로 높이면**: SARSA가 절벽에서 더 멀리 우회하는가?
  ε가 클수록 탐험 실수 가능성이 높으므로, SARSA는 더 보수적인 경로를 학습한다.

- **ε를 0으로 줄이면**: 두 알고리즘의 경로가 같아지는가?
  ε=0에서는 탐험이 없어 SARSA와 Q-Learning이 같은 정책으로 수렴한다.

---

## [Idea] On-policy vs Off-policy — 누구의 가치를 학습하는가

**On-policy (SARSA):**
실제로 따르는 정책(ε-greedy)의 가치를 학습한다.

"나는 가끔 실수한다(ε). 그 실수를 포함한 내 행동의 가치를 정확히 알고 싶다."

절벽 근처에서 ε 확률로 절벽에 떨어질 수 있으므로,
SARSA는 절벽 근처 상태의 Q값을 낮게 평가한다.
결국 절벽에서 멀어지는 정책을 학습한다.

**Off-policy (Q-Learning):**
현재 따르는 정책(ε-greedy)과 무관하게, **최적 정책(greedy)** 의 가치를 학습한다.

"실제로 내가 어떻게 행동하든, 이 상태에서 최선이 뭔지만 알고 싶다."

max 연산으로 탐험 실수와 무관하게 최적 Q값을 추정한다.
결국 이론적 최단 경로(절벽 바로 위)를 학습하지만, 학습 중에는 절벽에 자주 빠진다.

---

## [Form] 수식으로 압축

### 1. SARSA vs Q-Learning 수식 비교

**SARSA (On-policy):**

$$Q(s,a) \leftarrow Q(s,a) + \alpha\bigl[R + \gamma\, Q(s', \underbrace{a'}_{\varepsilon\text{-greedy}}) - Q(s,a)\bigr]$$

**Q-Learning (Off-policy):**

$$Q(s,a) \leftarrow Q(s,a) + \alpha\bigl[R + \gamma\, \underbrace{\max_{a'} Q(s', a')}_{\text{greedy}} - Q(s,a)\bigr]$$

한 줄 차이: $Q(s', a')$ (다음 정책으로 선택한 행동) vs $\max_{a'} Q(s', a')$ (탐욕적 최선).

---

### 2. On-policy vs Off-policy 개념 정리

**On-policy:** 학습에 사용하는 정책 = 실제 따르는 정책

| 특징 | 내용 |
|---|---|
| 학습 대상 | 현재 ε-greedy 정책의 가치 |
| 탐험 반영 | ✓ (탐험 실수가 Q값에 반영됨) |
| 안전성 | 위험 환경에서 더 보수적 |
| 수렴 | ε → 0 이면 최적으로 수렴 |

**Off-policy:** 학습에 사용하는 정책(목표 정책) ≠ 실제 따르는 정책(행동 정책)

| 특징 | 내용 |
|---|---|
| 학습 대상 | 탐험과 무관한 최적 greedy 정책의 가치 |
| 탐험 반영 | ✗ (greedy max로 탐험 무시) |
| 최적성 | 이론적 최적 경로 학습 |
| 유연성 | 다른 에이전트의 데이터도 학습 가능 |

**Cliff Walking에서의 수렴값:**

SARSA: ε=0.1 하에서 절벽 위 한 줄을 우회 → 에피소드당 약 −17
Q-Learning: 절벽 바로 옆 최단 경로 → 에피소드당 −13 (이론적 최적)

---

### 3. Off-policy의 더 넓은 의미

Off-policy 학습이 가능하다는 것은 중요한 실용적 장점을 준다.

**① 과거 경험 재사용 (Experience Replay)**
과거 정책으로 수집한 데이터로도 현재 최적 정책을 학습할 수 있다.
Unit 5(DQN, 부록)의 Replay Buffer가 이 원리를 사용한다.

**② 전문가 데이터 학습 (Imitation)**
인간 전문가가 행동한 데이터로 에이전트를 학습시킬 수 있다.
인간이 행동 정책, 에이전트가 목표 정책.

**③ 탐험과 최적화 분리**
탐험은 랜덤하게 하고, 최적 정책의 가치는 별도로 학습한다.

<details>
<summary>더 알고 싶다면 — 중요도 샘플링과 Off-policy의 이론적 기반</summary>

MC에서 Off-policy를 정확히 구현하려면 **중요도 샘플링(Importance Sampling)** 이 필요하다.

행동 정책 $b$ 로 수집한 데이터로 목표 정책 $\pi$ 를 평가할 때:
$$V_\pi(s) \approx \frac{\sum_t \rho_t G_t}{\sum_t \rho_t}, \quad \rho_t = \prod_{k} \frac{\pi(A_k|S_k)}{b(A_k|S_k)}$$

$\rho$ 는 정책 비율의 누적곱. 에피소드가 길수록 분산이 폭발적으로 커진다.

Q-Learning이 이 문제를 우아하게 회피하는 방법: $\max$ 연산.
$\max_a Q(s',a)$ 는 greedy 정책 $\pi$ 의 기대값이고, greedy 정책에서는 $\rho = 1$ 이므로 중요도 보정이 필요 없다.

이것이 Q-Learning이 Off-policy이면서도 구현이 단순한 이유다.

</details>

---

## [Next] Tabular RL의 한계

Unit 0부터 Unit 6까지, Tabular RL의 핵심 도구를 모두 배웠다.

| Unit | 방법 | 핵심 |
|---|---|---|
| 0 | Bandit | 탐험 vs 이용 |
| 1-2 | MDP, DP | 모델 기반 최적화 |
| 3 | Monte Carlo | 샘플 기반, 에피소드 |
| 4 | TD, SARSA, Q-Learning | 즉시 업데이트 |
| 5 | n-step, Dyna-Q | 스펙트럼, 계획 |
| 6 | Cliff Walking | On/Off-policy |

이 모든 방법의 공통 전제: **상태 공간이 표로 저장 가능하다.**

실제 문제를 보자.

- CartPole: 4개 연속 변수 → 무한 상태
- 아타리 게임: 화면 픽셀 → 천문학적 상태
- 로봇 제어: 관절 각도, 속도 → 연속 공간

**상태를 표에 저장할 수 없다면?**

다음 Unit에서는 CartPole을 통해 Tabular 방법의 한계를 직접 확인하고,
신경망과 함수 근사가 왜 필요한지 — Part II로 가는 문을 연다.
