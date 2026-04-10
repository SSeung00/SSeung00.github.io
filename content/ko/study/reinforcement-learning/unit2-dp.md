---
title: "Unit 2 — 지도를 완전히 활용하기: 동적 프로그래밍"
date: 2026-04-10
category: "Reinforcement Learning"
weight: 3
---

## [Hook] 평가할 수 있다면, 개선할 수도 있다

Unit 1에서 중요한 도구를 하나 얻었다.

**가치 함수(Value Function):** 지금 이 위치가 얼마나 좋은가를 숫자 하나로 표현하는 것.
그리고 Bellman 반복으로 임의의 정책에 대한 가치를 계산할 수 있다는 것도 배웠다.

그런데 Unit 1에서 한 가지 질문을 열어두었다.

임의의 정책을 *평가* 하는 법은 알았다.
그렇다면 *최적* 정책은 어떻게 찾는가?

시뮬레이터에서 화살표를 손으로 하나씩 바꾸는 건 현실적이지 않다.
16개 셀, 4가지 방향 조합이면 $4^{16}$ 가지 정책이 가능하다.

모든 정책을 다 평가해보고 가장 좋은 것을 고를 수는 없다.

---

## [Fail] 평가만 해서는 정책이 나아지지 않는다

아래 시뮬레이터에서 일부러 나쁜 정책을 만들어보자.

**화살표 여러 개를 목표 반대 방향(↑ 또는 ←)으로 바꾼 뒤**,
"이 Policy 평가하기 → 수렴까지 실행"을 눌러 Value를 계산해보자.

{{< simulator src="/simulator/unit-1-gridworld/gridworld-policy.html" height="580px" title="Unit 1 Simulator B — 나쁜 Policy를 평가해보기" >}}

Value가 매우 낮게(= 큰 음수) 나오는 것이 보일 것이다.

이 시뮬레이터는 가치를 *계산* 해줄 뿐, 정책을 *개선* 해주지는 않는다.
화살표를 어떻게 바꿔야 더 나아질지는 우리가 직접 판단해야 한다.

이 한계를 넘는 방법이 있을까?

---

## [Idea] 가치를 보면 최적 행동이 보인다

핵심 아이디어는 단순하다.

가치 함수 $V(s)$ 가 계산되어 있다면, 각 위치에서 **가장 가치가 높은 다음 상태로 가는 행동** 이 최선이다.

예를 들어 위치 (2, 3) 에서 네 방향으로 한 발짝 이동했을 때의 가치를 비교하면:
- 위(2,3)→(1,3): $-1 + \gamma V(1,3)$
- 아래(2,3)→(3,3): $-1 + \gamma V(3,3) = -1 + 0 = -1$  ← 최선

(3,3)이 목표이므로 아래로 가는 것이 가장 좋다.

이것이 **정책 개선(Policy Improvement)** 이다.

$$\pi'(s) = \arg\max_a \bigl[R(s,a) + \gamma V_\pi(s')\bigr]$$

현재 가치 함수에서 탐욕적으로(Greedy) 행동을 선택해 새 정책을 만든다.

그리고 이 새 정책으로 다시 가치를 계산하면?
새 가치는 이전보다 나빠질 수 없음이 수학적으로 보장된다.

이것이 **정책 반복(Policy Iteration)** 의 핵심 사이클이다:

```
반복:
  1. Policy Evaluation   — 현재 정책의 가치를 수렴까지 계산
  2. Policy Improvement  — 가치에서 탐욕적으로 새 정책 생성
  정책이 더 이상 바뀌지 않으면 → 최적 정책 도달
```

그런데 한 가지 의문이 생긴다.
Policy Evaluation을 매번 완전히 수렴시켜야 할까?
어차피 바로 개선할 텐데, 한 번만 업데이트하고 바로 개선하면 어떨까?

이 극단적인 방법이 **가치 반복(Value Iteration)** 이다:

$$V_{k+1}(s) = \max_a \bigl[R(s,a) + \gamma V_k(s')\bigr]$$

평가와 개선을 한 번에 합쳐버린 업데이트. 더 빠르다.

---

## [See] Policy Iteration vs Value Iteration을 직접 비교하자

아래 시뮬레이터에서 두 알고리즘을 탭으로 선택해서 비교할 수 있다.

{{< simulator src="/simulator/unit-2-dp/dp-iteration.html" height="620px" title="Simulator — Policy Iteration / Value Iteration" >}}

탐구해볼 것들:

**Policy Iteration 탭:**
- **1 Iteration** 을 눌러보자. 파란 flashing(Evaluation 단계) 후 주황 flashing(Improvement 단계)이 온다.
  어느 셀의 화살표가 바뀌었는가? 처음에는 많이 바뀌다가 나중에는 거의 안 바뀐다.
- **수렴까지** 를 눌러보자. 몇 번의 Iteration으로 최적에 도달했는가?
- **수렴 곡선** 그래프를 보자. 로그 스케일에서 직선으로 감소한다 — 기하급수적 수렴.

**Value Iteration 탭으로 전환:**
- 이번엔 Policy Improvement 단계가 없다. 각 Sweep이 평가+개선을 동시에 한다.
- **1 Iteration** 을 반복해서 누르며 Value가 Goal에서 퍼져나가는 것을 확인하자.
- Sweep 수와 최종 Policy를 Policy Iteration과 비교해보자.

**γ를 바꿔보자:**
- γ = 0.5: 미래를 크게 할인 → 수렴이 빠르지만, 먼 목표의 영향이 약해진다.
- γ = 0.99: 미래를 거의 할인하지 않음 → 수렴이 느리지만 더 정확한 장기 계획.

---

## [Form] 수식으로 압축

### Bellman 최적 방정식

단순히 어떤 정책의 가치가 아니라, **최적** 가치를 직접 정의할 수 있다.

$$V^*(s) = \max_a \left[ R(s,a) + \gamma \sum_{s'} P(s'|s,a)\, V^*(s') \right]$$

이것이 **Bellman 최적 방정식** 이다.
$V^*$ 는 이 방정식의 유일한 고정점(fixed point) 이다.

결정론적 환경(이번 Gridworld)에서는 $\sum_{s'} P(s'|s,a)$ 가 1이므로 단순화된다:

$$\boxed{V^*(s) = \max_a \bigl[R(s,a) + \gamma\, V^*(s')\bigr]}$$

### Policy Iteration

```
초기화: V(s) = 0, π(s) = 임의 정책

반복:
  # Policy Evaluation
  while max|ΔV| > θ:
    V(s) ← R(s, π(s)) + γ · V(s')

  # Policy Improvement
  changed = False
  for all s:
    π_new(s) = argmax_a [R(s,a) + γ · V(s')]
    if π_new(s) ≠ π(s): changed = True
  π ← π_new
  if not changed: break   # 수렴
```

**정책 개선 정리(Policy Improvement Theorem):** 탐욕적으로 개선된 정책 $\pi'$ 에 대해 $V_{\pi'}(s) \geq V_\pi(s)$ 가 모든 $s$ 에서 성립한다. 따라서 반복은 단조적으로 개선되고, 유한 MDP에서 반드시 수렴한다.

### Value Iteration

```
초기화: V(s) = 0

while max|ΔV| > θ:
  for all s:
    V(s) ← max_a [R(s,a) + γ · V(s')]

정책 추출:
  π*(s) = argmax_a [R(s,a) + γ · V(s')]
```

Policy Iteration의 Evaluation 단계를 단 1 sweep으로 줄인 것.
이론적으로 Policy Iteration보다 느리게 보이지만, 실제로는 비슷하거나 더 빠른 경우가 많다.

### 두 알고리즘 비교

| | Policy Iteration | Value Iteration |
|---|---|---|
| **업데이트** | $V(s) = R + \gamma V_\pi(s')$ | $V(s) = \max_a [R + \gamma V(s')]$ |
| **Evaluation 단계** | 수렴까지 반복 | 단 1 sweep |
| **수렴 기준** | 정책이 안 바뀔 때 | $\max|\Delta V| < \theta$ |
| **Iteration 수** | 적음 | 많음 |
| **Sweep당 비용** | 높음 | 낮음 |
| **총 계산량** | 비슷 | 비슷 |

<details>
<summary>더 알고 싶다면 — 수렴 보장과 계산 복잡도</summary>

**Policy Iteration의 수렴:**
유한 MDP에서 정책의 수는 유한하다 ($|A|^{|S|}$ 개).
각 Iteration은 정책을 단조적으로 개선하고, 동일한 정책이 두 번 나타날 수 없으므로 반드시 종료된다.
실제로는 매우 빠르게 수렴한다 — 이론적 상한보다 훨씬 적은 Iteration으로 끝나는 것이 관측된다.

**Value Iteration의 수렴:**
Bellman 최적 연산자 $\mathcal{T}^*$ 는 $\gamma$-contraction이므로 $V_k \to V^*$ 가 보장된다.
오차 경계: $k$ 회 sweep 후 $\|V_k - V^*\|_\infty \leq \frac{\gamma^k}{1-\gamma} \|V_0 - V^*\|_\infty$

**계산 복잡도:**
Policy Evaluation 1 sweep: $O(|S|^2 |A|)$ (일반 MDP), $O(|S||A|)$ (결정론적).
Policy Iteration: 보통 Iteration 수가 매우 적어 (Gridworld는 4~5회) 전체 비용이 낮다.
Value Iteration: sweep당 비용이 적지만 수렴까지 더 많은 sweep이 필요하다.

</details>

---

## [Next] 다음 질문

동적 프로그래밍은 강력하다.
최적 정책을 보장하고, 수렴도 증명할 수 있다.

그런데 한 가지 전제가 있다.

**MDP 모델을 알아야 한다.** 즉, $P(s'|s,a)$ 와 $R(s,a)$ 를 알아야 한다.

Gridworld에서는 명확했다. 규칙이 있고, 어디서 뭘 하면 어디로 가는지 안다.

하지만 현실에서는 어떨까?

바둑 에이전트는 바둑의 모든 상태 전이를 미리 계산할 수 없다.
로봇은 실제 물리 세계의 모든 역학을 정확히 모른다.
주식 트레이딩 에이전트는 시장의 전이 확률을 모른다.

**모델이 없다면?** 직접 경험해보면서 배워야 한다.
에피소드를 끝까지 실행한 뒤 돌아보는 방법이 있다.

다음 Unit에서는 블랙잭을 직접 플레이하면서,
규칙을 모르는 상태에서 가치를 추정하는 **Monte Carlo** 방법을 배운다.
