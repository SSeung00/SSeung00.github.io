---
title: "Unit 5 — 표를 버리고 신경망으로: DQN"
date: 2026-04-13
category: "Reinforcement Learning"
weight: 6
---

## [Hook] Q-table의 벽

Unit 4에서 Q-Learning을 배웠다.

Gridworld — 20개 상태, 4가지 행동. Q-table 크기: 80.
테이블에 값을 써넣으면 끝이다.

그런데 이런 환경을 생각해보자.

**아타리 게임:** 화면은 $84 \times 84$ 픽셀, 각 픽셀이 256 단계의 밝기를 갖는다.
상태 공간의 크기: $256^{84 \times 84}$. 우주의 원자 수보다 훨씬 크다.

이 상태 공간에 테이블을 만들 수 없다.
설령 만들 수 있어도, 같은 게임 화면을 두 번 정확히 같은 픽셀로 마주칠 확률은 0에 가깝다.
**보지 못한 상태에서도 좋은 Q값을 추정해야 한다.**

여기서 핵심 아이디어가 나온다.

$$Q(s,a) \approx Q(s,a;\,\boldsymbol\theta)$$

Q-table을 **신경망으로 근사** 한다.
비슷한 상태는 비슷한 Q값을 출력하도록 신경망이 일반화(generalization)해준다.

이것이 **함수 근사(Function Approximation)** 이고, 신경망을 쓰면 **DQN(Deep Q-Network)** 이다.

---

## [Fail] 신경망 Q-Learning을 그냥 하면 불안정하다

Q-Learning에 신경망을 끼워넣으면 될 것 같다.

$$\mathcal{L}(\boldsymbol\theta) = \bigl[\underbrace{R + \gamma \max_{a'} Q(s',a';\,\boldsymbol\theta)}_{\text{TD 목표}} - Q(s,a;\,\boldsymbol\theta)\bigr]^2$$

이 손실을 역전파로 최소화하면 된다.

하지만 실제로 이렇게 하면 두 가지 큰 문제가 생긴다.

**문제 1 — 상관된 샘플 (Correlated Samples)**

에이전트가 게임을 플레이하면서 수집한 $(s_t, a_t, r_t, s_{t+1})$ 들은 시간 순서대로 강하게 연결되어 있다.
한 에피소드 동안 에이전트가 오른쪽 구역을 탐험하면, 그 시간 동안의 모든 샘플이 오른쪽 구역 데이터다.
신경망은 이 데이터에 과적합되어 다른 구역에서의 Q값을 망각한다.

이것이 신경망 학습의 **파국적 망각(Catastrophic Forgetting)** 이다.

**문제 2 — 움직이는 목표 (Moving Target)**

TD 목표값 $R + \gamma \max_{a'} Q(s',a';\boldsymbol\theta)$ 를 보자.
$\boldsymbol\theta$ 를 업데이트하면 목표값도 같이 바뀐다.

지도학습에서 레이블(정답)은 고정되어 있다.
TD는 자기 자신이 만든 예측으로 목표를 삼는다 — 예측이 바뀌면 목표도 바뀐다.
움직이는 표적을 쫓는 것처럼 학습이 불안정해진다.

---

## [Idea] DQN의 두 가지 해결책

DeepMind는 2015년 이 두 문제를 모두 해결하는 방법을 발표했다.

### 해결책 1 — Experience Replay (경험 재현)

에이전트가 경험한 $(s,a,r,s',\text{done})$ 튜플을 **버리지 않고 버퍼에 저장** 한다.
학습할 때는 버퍼에서 **무작위로 미니배치를 샘플링** 해서 사용한다.

```
버퍼 B에 (s, a, r, s', done) 저장
학습 시:
  버퍼에서 32개 랜덤 샘플링
  신경망 업데이트
```

무작위 샘플링으로 두 가지 효과가 생긴다:
1. **시간 상관 제거**: 연속된 순간이 아니라 임의 시점의 경험을 섞어 학습
2. **데이터 재사용**: 과거 경험을 반복해서 학습에 활용 (데이터 효율 증가)

### 해결책 2 — Target Network (목표 네트워크)

주 네트워크($\boldsymbol\theta$) 와 **별도의 고정된 목표 네트워크($\boldsymbol\theta^-$)** 를 유지한다.
TD 목표를 계산할 때는 목표 네트워크를 사용한다.

$$\mathcal{L}(\boldsymbol\theta) = \bigl[R + \gamma \max_{a'} Q(s',a';\,\boldsymbol\theta^-) - Q(s,a;\,\boldsymbol\theta)\bigr]^2$$

주 네트워크는 매 step 업데이트되지만,
목표 네트워크는 **일정 주기(예: 50 step)** 마다 주 네트워크의 가중치를 복사해서 갱신한다.

목표가 자주 바뀌지 않으니 학습이 훨씬 안정해진다.

---

## [See] 구성 요소를 켜고 끄며 효과를 비교하자

아래 시뮬레이터는 **신경망(Input 20→Hidden 32→Output 4)** 으로 Q값을 근사한다.
같은 Gridworld를 사용하지만, Q-table 대신 신경망이 Q값을 계산한다.

{{< simulator src="/simulator/unit-5-dqn/dqn-gridworld.html" height="660px" title="Simulator — DQN (Experience Replay / Target Network 비교)" >}}

**4가지 조합을 순서대로 실험해보자** (각각 리셋 후 100 에피소드 이상 실행):

| 조합 | Replay | Target | 예상 결과 |
|------|--------|--------|-----------|
| ① | OFF | OFF | 가장 불안정, Loss 진동 심함 |
| ② | ON  | OFF | 샘플 상관 해결, 그러나 목표 불안정 |
| ③ | OFF | ON  | 목표 안정, 그러나 망각 발생 |
| ④ | ON  | ON  | **풀 DQN**: 가장 안정적 수렴 |

탐구해볼 것들:

- **Loss 곡선** 이 조합에 따라 얼마나 다른가?
  ①번(둘 다 OFF)에서는 Loss가 크게 진동하거나 발산할 수 있다.
- **Replay Buffer 시각화** 를 보자.
  색깔 점들이 다양한 행동(파랑/초록/주황/빨강)과 종료 여부(빨간 진한 점 = 구덩이, 초록 진한 점 = 목표)를 나타낸다.
- **α (학습률)** 을 0.1 이상으로 높이면 ①번 조합에서 발산이 더 잘 보인다.
- Q-값 화살표가 올바른 방향으로 수렴하는 데 Q-table(Unit 4)보다 더 많은 에피소드가 필요한가?

---

## [Form] 수식으로 압축

### 1. 선형 함수 근사 (Linear Approximation)

신경망을 쓰기 전에, 가장 간단한 형태를 보자.

상태 $s$ 를 특성 벡터 $\boldsymbol\phi(s) \in \mathbb{R}^d$ 로 표현한다.
Q값을 선형 함수로 근사한다:

$$Q(s,a;\,\boldsymbol w) = \boldsymbol w_a^\top \boldsymbol\phi(s)$$

업데이트:
$$\boldsymbol w_a \leftarrow \boldsymbol w_a + \alpha\bigl[R + \gamma \max_{a'}\boldsymbol w_{a'}^\top\boldsymbol\phi(s') - \boldsymbol w_a^\top\boldsymbol\phi(s)\bigr]\boldsymbol\phi(s)$$

SGD(확률적 경사 하강법)로 TD 오차를 줄이는 방향으로 $\boldsymbol w$ 를 업데이트한다.

이번 시뮬레이터에서 사용한 것처럼, one-hot 특성을 쓰면 선형 모델은 Q-table과 동일하다.
**함수 근사의 힘은 보지 못한 상태 사이를 보간(interpolate)하는 일반화 능력에서 나온다.**

---

### 2. DQN 알고리즘 전체

```
초기화:
  주 네트워크 Q(s,a; θ) 랜덤 초기화
  목표 네트워크 Q(s,a; θ⁻) ← θ⁻ = θ
  리플레이 버퍼 B = {} (최대 크기 N)

매 스텝:
  1. 현재 상태 s에서 ε-greedy로 행동 a 선택
  2. 환경 실행 → (s', r, done)
  3. (s, a, r, s', done) → 버퍼 B에 저장
  4. B에서 미니배치 {(sⱼ, aⱼ, rⱼ, s'ⱼ, doneⱼ)} 샘플링
  5. TD 목표 계산 (목표 네트워크 사용):
       yⱼ = rⱼ                               if doneⱼ
       yⱼ = rⱼ + γ · max_a' Q(s'ⱼ, a'; θ⁻)  otherwise
  6. 손실 최소화:
       L(θ) = mean[(yⱼ - Q(sⱼ, aⱼ; θ))²]
       θ ← θ - α · ∇_θ L(θ)
  7. 매 C 스텝마다: θ⁻ ← θ  (목표 네트워크 갱신)
```

---

### 3. DQN의 핵심 구성 요소 정리

| 구성 요소 | 문제 | 해결 방법 |
|---|---|---|
| **Experience Replay** | 연속 샘플의 상관관계 | 버퍼에서 랜덤 샘플링 |
| **Target Network** | 움직이는 TD 목표 | 주기적으로만 갱신되는 별도 네트워크 |
| **ε-greedy 감소** | 탐험 vs 이용 균형 | 학습 초반 ε 높게, 점차 감소 |
| **Reward Clipping** | 보상 스케일 차이 | 보상을 [−1, +1] 범위로 클리핑 |
| **프레임 스태킹** | 부분 관측(Partial obs.) | 4프레임을 쌓아 속도 등 정보 제공 |

---

### 4. 왜 DQN이 작동하는가 — 이론적 관점

**수렴 보장:**
선형 함수 근사 + TD는 특정 조건에서 수렴이 보장된다.
하지만 비선형 함수(신경망) + 부트스트래핑 + Off-policy 샘플링의 조합은 수렴이 이론적으로 보장되지 않는다 — **deadly triad** 라고 부른다.

실용적으로는 Experience Replay + Target Network가 이 불안정성을 충분히 억제해 학습을 성공시킨다.

**일반화:**
신경망은 유사한 상태에 대해 유사한 Q값을 출력한다.
아타리 게임에서 공의 위치가 약간 다른 두 프레임은 비슷한 Q값을 가져야 한다.
신경망의 귀납적 편향(inductive bias)이 이 일반화를 자연스럽게 실현한다.

<details>
<summary>더 알고 싶다면 — DQN 이후의 발전</summary>

2015년 DQN 이후 수많은 개선이 이루어졌다.

**Double DQN (2016)**
DQN은 $\max_a Q(s',a;\theta^-)$ 로 목표를 계산하는데, 이 max 연산이 Q값을 과대추정(overestimation)한다.
행동 선택은 주 네트워크로, Q값 평가는 목표 네트워크로 분리해 편향을 줄인다:
$$y = R + \gamma \, Q\!\left(s',\, \arg\max_{a'} Q(s',a';\theta),\; \theta^-\right)$$

**Dueling DQN (2016)**
Q값을 상태 가치 $V(s)$ 와 행동 어드밴티지 $A(s,a)$ 로 분리한다:
$$Q(s,a;\theta) = V(s;\theta_V) + A(s,a;\theta_A) - \frac{1}{|A|}\sum_{a'} A(s,a';\theta_A)$$
행동과 무관한 상태 가치를 별도로 학습해 표현력을 높인다.

**Prioritized Experience Replay (2016)**
모든 경험을 동등하게 샘플링하지 않고, TD 오차가 큰 경험을 더 자주 샘플링한다.
"놀라운" 경험에서 더 많이 배운다.

**Rainbow DQN (2017)**
위의 개선들을 모두 합친 모델. 대부분의 아타리 게임에서 인간 수준을 크게 상회한다.

</details>

---

## [Next] 다음 질문

DQN은 강력하다. 아타리 49개 게임을 인간 수준으로 플레이했다.

하지만 한 가지 근본적인 제약이 있다.

**행동 공간이 이산적(discrete)이어야 한다.**

아타리 게임 — 8방향 이동, 발사 버튼. 유한한 조합이다.

이제 다른 문제를 생각해보자.

로봇 팔을 제어한다. 각 관절의 토크는 연속적인 값이다. $[-1.0, 1.0]$ 범위의 실수.
$\max_a Q(s,a)$ 를 구하려면 연속 공간에서 최적화 문제를 풀어야 한다 — DQN으로는 직접 할 수 없다.

**연속 행동 공간에서는 어떻게 할까?**

Q값을 최대화하는 행동을 직접 출력하는 **정책 네트워크(Policy Network)** 를 따로 학습하면 된다.
Actor가 행동을 선택하고, Critic이 그 행동을 평가한다.

이것이 **Actor-Critic** 과 **Policy Gradient** 방법이다.
다음 Unit에서는 REINFORCE부터 시작해 이 아이디어를 배운다.
