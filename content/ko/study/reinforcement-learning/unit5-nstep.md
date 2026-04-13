---
title: "Unit 5 — 얼마나 앞을 봐야 할까: n-step TD와 Dyna-Q"
date: 2026-04-13
category: "Reinforcement Learning"
weight: 6
---

## [Hook] 1 step과 무한 step 사이

Unit 4에서 TD(0)을 배웠다.

1 step만 보고 즉시 업데이트. 빠르다. 연속 태스크도 된다.

그런데 1 step의 추정값 $R + \gamma V(s')$ 는 여전히 추정이다.
$V(s')$ 가 틀려 있으면, TD 목표도 틀리다.

반면 MC는 끝까지 기다린다. 실제 리턴 $G$ 를 쓰니까 추정이 없다.
하지만 너무 느리다 — 에피소드가 끝날 때까지.

**1 step과 무한 step 사이 어딘가에 최적이 있지 않을까?**

$$G_t^{(n)} = R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{n-1}R_{t+n} + \gamma^n V(S_{t+n})$$

$n$ 을 조절하면 TD(0)(n=1)부터 MC(n=∞)까지 **연속적으로 이어진다.**

이것이 **n-step TD** 다.

---

## [Fail] 1 step이 항상 최선은 아니다

아래 시뮬레이터에서 여러 n 값이 동시에 학습하는 과정을 보자.

n = 1 (TD), 2, 4, 8, 16 — 모두 **똑같은 에피소드** 를 경험한다.

{{< simulator src="/simulator/unit-5-nstep/nstep-compare.html" height="560px" title="Simulator A — n-step TD 스펙트럼 비교" >}}

탐구해볼 것들:

- **100 에피소드** 를 실행해보자. 어떤 n이 가장 빠르게 수렴하는가?
- n=1 (TD) vs n=16 (≈MC) 를 비교하자. 초반에는 누가 빠른가?
- **α (학습률)** 을 높이면 최적 n이 바뀌는가?
- 보통 중간 값의 n(4~8 정도)이 가장 빠르게 수렴하는 것을 확인해보자.

왜 중간 값이 최선일까?
- n이 작으면: 편향이 크다 (1-step 추정에 의존)
- n이 크면: 분산이 크다 (긴 궤적의 노이즈 누적)
- **중간 n**: 편향과 분산의 균형점

---

## [Idea] 수집한 경험을 버리지 말자: Dyna-Q

n-step TD는 여전히 한 가지 비효율성이 있다.

**환경과의 실제 상호작용** 이 비싸다면?

로봇 물리 실험: 팔이 부러질 수 있다.
자율주행 테스트: 사고가 날 수 있다.
실제 환경을 탐험하는 것 자체가 비용이다.

그런데 이미 경험한 상태에서 어떤 일이 일어났는지는 안다.
기억 속의 경험으로 **가상 연습(Planning)** 을 할 수 있다.

```
환경에서 한 번 실제로 행동
→ (s, a, r, s') 를 모델에 저장
→ 이 모델에서 k번 가상 연습
→ 실제로는 1번 경험했지만, k+1번 학습 효과
```

이것이 **Dyna-Q** 다.

실제 경험으로 모델을 만들고, 모델 안에서 계획(Planning)을 반복한다.
k = 0이면 순수 Q-Learning, k가 클수록 모델 기반 학습에 가까워진다.

---

## [See] 계획 스텝 수에 따라 수렴 속도가 달라진다

아래 시뮬레이터에서 k (계획 스텝 수)를 조절하며 Dyna-Q의 효과를 확인하자.

{{< simulator src="/simulator/unit-5-nstep/dyna-q.html" height="600px" title="Simulator B — Dyna-Q (계획 스텝 비교)" >}}

탐구해볼 것들:

- **k=0** (순수 Q-Learning): 목표를 찾기까지 에피소드 수가 많다.
  "Real steps" 카운터가 빠르게 올라간다.

- **k=10 ~ k=50**: 동일한 Real steps 수에서 훨씬 빨리 목표를 찾는다.
  "Simulated steps" 카운터가 real steps의 k배만큼 올라가는 것을 확인하자.

- **장애물이 있는 경로**: 에이전트가 처음에 좋지 않은 경로를 학습했다가,
  계획을 통해 더 좋은 경로를 점점 발견하는 것을 Q-값 화살표로 확인하자.

- k가 50일 때: real steps 10번으로도 k=0보다 훨씬 적은 에피소드 만에 수렴한다.

---

## [Form] 수식으로 압축

### 1. n-step 리턴

$$G_t^{(n)} \doteq R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{n-1}R_{t+n} + \gamma^n V(S_{t+n})$$

$$V(S_t) \leftarrow V(S_t) + \alpha\bigl[G_t^{(n)} - V(S_t)\bigr]$$

- n = 1: TD(0) — 1-step 부트스트랩
- n = ∞: MC — 실제 리턴 $G_t$
- n = 중간: 편향-분산 최적점

업데이트를 위해 n step 앞의 결과를 기다려야 하므로, TD(0)보다 구현이 약간 복잡하다.

---

### 2. n-step 리턴 비교표

| n | 업데이트 목표 | 편향 | 분산 | 기다림 |
|---|---|---|---|---|
| 1 (TD) | $R + \gamma V(s')$ | 높음 | 낮음 | 즉시 |
| 중간 | $\sum \gamma^k R + \gamma^n V$ | 중간 | 중간 | n step |
| ∞ (MC) | 실제 $G$ | 없음 | 높음 | 에피소드 끝 |

---

### 3. Dyna-Q 알고리즘

```
초기화: Q(s,a) = 0, Model = {}

매 스텝:
  1. 현재 s에서 ε-greedy로 행동 a 선택
  2. s', R ← 환경 실행 (실제 경험)
  3. Q-update:
       Q(s,a) ← Q(s,a) + α[R + γ·max_a' Q(s',a') - Q(s,a)]
  4. Model[s,a] ← (R, s')   ← 결정론적 모델 저장
  5. Planning (k번 반복):
       (sp, ap) ← 기존 방문한 (s,a) 중 랜덤 선택
       (Rp, s'p) ← Model[sp, ap]
       Q(sp,ap) ← Q(sp,ap) + α[Rp + γ·max_a' Q(s'p,a') - Q(sp,ap)]
```

실제 경험 1번 → Q-업데이트 k+1번 (1 real + k simulated).

---

### 4. Model-based vs Model-free

| | Model-free (Q-Learning) | Model-based (Dyna-Q) |
|---|---|---|
| **모델** | 불필요 | 경험으로 직접 구축 |
| **데이터 효율** | 낮음 (경험 재사용 없음) | 높음 (경험 k번 재활용) |
| **계산량** | 낮음 | 높음 (k번 계획) |
| **환경 변화** | 빠른 적응 | 느림 (구 모델 남음) |

**핵심 insight:** 실제 환경 상호작용이 비싸고 시뮬레이션이 싸다면, Dyna-Q의 k를 높일수록 이득.
반대로 환경이 자주 변하면 구 모델의 부작용이 생긴다.

<details>
<summary>더 알고 싶다면 — TD(λ)와 Eligibility Traces</summary>

n-step 리턴을 **모든 n에 대해 지수 가중 평균** 으로 섞으면 TD(λ) 다.

$$G_t^\lambda = (1-\lambda)\sum_{n=1}^{\infty}\lambda^{n-1}G_t^{(n)}$$

$\lambda \in [0,1]$: λ=0은 TD(0), λ=1은 MC.

효율적 구현을 위해 **Eligibility Traces** (적격 흔적)를 사용한다.
각 상태에 흔적 값 $e(s)$ 를 유지한다:
- 방문할 때마다 $e(s) \mathrel{+}= 1$ (또는 ×λγ)
- 매 step마다 $e(s) \mathrel{\times}= \lambda\gamma$ (감쇠)
- TD 오차 $\delta$ 를 모든 상태에 흔적 비례로 역전파

$$V(s) \leftarrow V(s) + \alpha\, \delta_t\, e_t(s)$$

이렇게 하면 n-step 리턴을 따로 저장하지 않고도 O(|S|)로 TD(λ)를 구현할 수 있다.

</details>

---

## [Next] 다음 질문

n-step TD와 Dyna-Q로 Tabular RL의 핵심 도구들을 대부분 배웠다.

그런데 특정 환경에서 On-policy와 Off-policy의 차이가 **전혀 다른 행동** 을 만들어낸다.

위험한 절벽이 있는 환경을 생각해보자.

- On-policy(SARSA): 탐험 실수가 두렵다 → 절벽에서 멀리 우회한다
- Off-policy(Q-Learning): 최적 경로를 추구한다 → 절벽 바로 옆을 걷는다

이 차이는 단순한 알고리즘 차이가 아니다.
**"누구의 관점에서 가치를 평가하느냐"** 의 차이다.

다음 Unit에서는 Cliff Walking 환경에서 이 두 알고리즘을 정면 비교한다.
