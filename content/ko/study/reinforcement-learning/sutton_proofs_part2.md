---
title: "RL Proofs Part 2: 심화편"
category: "Reinforcement Learning"
date: 2026-04-09
weight: 19
---

# Reinforcement Learning 핵심 수학적 증명 (Part 2: 심화편)

> Sutton & Barto Ch.6~7 — 교재에서 생략된 수학적 증명 완전 정리

---

## 목차

1. [TD(0) 수렴 증명 — ODE 방법론](#1-td0-수렴-증명--ode-방법론)
2. [Q-learning 수렴 증명 — Lyapunov 방법](#2-q-learning-수렴-증명--lyapunov-방법)
3. [n-step Return 오차 감소 성질 증명](#3-n-step-return-오차-감소-성질-증명)
4. [TD(λ)와 n-step Return의 등가성 증명](#4-tdλ와-n-step-return의-등가성-증명)
5. [Maximization Bias의 Jensen 부등식 증명](#5-maximization-bias의-jensen-부등식-증명)

---

## 1. TD(0) 수렴 증명 — ODE 방법론

> **위치**: Chapter 6 — TD(0)가 $v_\pi$로 수렴하는 이론적 근거

### 배경

Chapter 4에서는 Banach 정리로 $T^\pi$의 반복이 $v_\pi$로 수렴함을 보였습니다. 하지만 TD(0)는 **기댓값이 아닌 샘플 하나**로 업데이트합니다. 이 확률적(stochastic) 업데이트가 여전히 수렴하는지는 별도의 이론이 필요합니다.

### 설정

TD(0) 업데이트를 벡터 형태로 씁니다 ($s$번째 성분이 $V(s)$):

$$\mathbf{V}_{t+1}(S_t) = \mathbf{V}_t(S_t) + \alpha_t\left[R_{t+1} + \gamma \mathbf{V}_t(S_{t+1}) - \mathbf{V}_t(S_t)\right]$$

이를 일반적인 확률적 근사(stochastic approximation) 형태로 씁니다:

$$\mathbf{V}_{t+1} = \mathbf{V}_t + \alpha_t\left[F(\mathbf{V}_t) + \varepsilon_t\right]$$

여기서:
- $F(\mathbf{V}) \doteq (T^\pi \mathbf{V}) - \mathbf{V}$ (기댓값 업데이트 방향)
- $\varepsilon_t \doteq \left[R_{t+1} + \gamma \mathbf{V}_t(S_{t+1}) - \mathbf{V}_t(S_t)\right] - F(\mathbf{V}_t)(S_t)$ (노이즈)

### Robbins-Monro 조건

수렴을 보장하는 step-size 조건:

$$\sum_{t=0}^{\infty} \alpha_t = \infty, \qquad \sum_{t=0}^{\infty} \alpha_t^2 < \infty$$

### 수렴 증명

#### Step 1: 고정점 확인

$F(\mathbf{V}^*) = 0$인 $\mathbf{V}^*$를 구합니다:

$$F(\mathbf{V}^*) = T^\pi \mathbf{V}^* - \mathbf{V}^* = 0 \iff T^\pi \mathbf{V}^* = \mathbf{V}^* \iff \mathbf{V}^* = \mathbf{v}_\pi$$

$\mathbf{v}_\pi$는 $T^\pi$의 유일한 고정점 (Banach 정리)이므로 $F$의 유일한 영점입니다.

#### Step 2: 안정성 조건 — $F$의 Jacobian 분석

$$F(\mathbf{V}) = T^\pi \mathbf{V} - \mathbf{V} = (\mathbf{r}_\pi + \gamma \mathbf{P}_\pi \mathbf{V}) - \mathbf{V}$$

$F$의 Jacobian:

$$J_F = \gamma \mathbf{P}_\pi - \mathbf{I}$$

이 행렬의 고유값을 분석합니다. $\mathbf{P}_\pi$는 확률 행렬(stochastic matrix)이므로 모든 고유값 $\lambda_i$에 대해 $|\lambda_i| \leq 1$.

$J_F$의 고유값은 $\gamma\lambda_i - 1$이므로:

$$\text{Re}(\gamma\lambda_i - 1) \leq \gamma|\lambda_i| - 1 \leq \gamma \cdot 1 - 1 = \gamma - 1 < 0 \quad (\gamma < 1)$$

모든 고유값의 실수부가 음수 → $\mathbf{v}_\pi$는 **점근적으로 안정한 평형점**.

#### Step 3: 노이즈 조건

$\varepsilon_t$가 martingale difference임을 확인합니다:

$$\mathbb{E}[\varepsilon_t \mid \mathcal{F}_t] = \mathbb{E}\left[R_{t+1} + \gamma \mathbf{V}_t(S_{t+1}) - \mathbf{V}_t(S_t) \mid \mathcal{F}_t\right] - F(\mathbf{V}_t)(S_t) = 0$$

(기댓값 업데이트의 정의에 의해 정확히 상쇄)

#### Step 4: Kushner-Clark 보조정리 적용

위 세 조건 (고정점 존재, Jacobian 안정성, martingale difference 노이즈)과 Robbins-Monro step-size 조건이 모두 만족되면:

$$\mathbf{V}_t \xrightarrow{a.s.} \mathbf{v}_\pi \qquad \blacksquare$$

### ODE와의 연결 (직관)

확률적 근사 이론의 핵심 통찰: step-size $\alpha_t \to 0$이면 이산 업데이트는 다음 상미분방정식(ODE)의 연속 해로 수렴합니다:

$$\dot{\mathbf{V}}(t) = F(\mathbf{V}(t)) = T^\pi \mathbf{V}(t) - \mathbf{V}(t)$$

이 ODE의 안정 평형점이 $\mathbf{v}_\pi$이므로 (Step 2), 확률적 업데이트도 $\mathbf{v}_\pi$로 수렴합니다.

---

## 2. Q-learning 수렴 증명 — Lyapunov 방법

> **위치**: Chapter 6 — Q-learning이 $q_*$로 수렴하는 이론적 근거 (Watkins & Dayan, 1992)

### 정리

모든 $(s, a) \in \mathcal{S} \times \mathcal{A}$가 무한히 방문되고, step-size $\alpha_t$가 Robbins-Monro 조건을 만족하면:

$$Q_t(s, a) \xrightarrow{a.s.} q_*(s, a), \quad \forall s, a$$

### 핵심 도구: Lyapunov 함수

$L(Q) = \|Q - q_*\|_\infty$를 Lyapunov 함수로 사용합니다.

수렴을 보이려면 기댓값 업데이트 방향이 항상 $L$을 감소시킴을 보여야 합니다.

### 증명

#### Step 1: 기댓값 업데이트 방향 분석

Q-learning의 기댓값 업데이트:

$$\bar{Q}(s,a) = (1-\alpha)\,Q(s,a) + \alpha \sum_{s',r} p(s',r|s,a)\left[r + \gamma\max_{a'} Q(s',a')\right]$$

$$= (1-\alpha)\,Q(s,a) + \alpha\,(T^* Q)(s,a)$$

#### Step 2: $\|\bar{Q} - q_*\|_\infty$ 상한

$$|\bar{Q}(s,a) - q_*(s,a)| = |(1-\alpha)(Q(s,a) - q_*(s,a)) + \alpha((T^*Q)(s,a) - (T^*q_*)(s,a))|$$

(마지막 변환: $T^* q_* = q_*$ 사용)

삼각 부등식:

$$\leq (1-\alpha)|Q(s,a) - q_*(s,a)| + \alpha|(T^*Q)(s,a) - (T^*q_*)(s,a)|$$

$T^*$의 $\gamma$-contraction 성질 ($\|T^*Q - T^*q_*\|_\infty \leq \gamma\|Q - q_*\|_\infty$) 적용:

$$\leq (1-\alpha)\|Q - q_*\|_\infty + \alpha\gamma\|Q - q_*\|_\infty$$

$$= (1 - \alpha(1-\gamma))\|Q - q_*\|_\infty$$

$s, a$에 대해 supremum을 취하면:

$$\|\bar{Q} - q_*\|_\infty \leq (1-\alpha(1-\gamma))\|Q - q_*\|_\infty$$

#### Step 3: 수렴 결론

$\alpha \in (0,1)$이고 $\gamma < 1$이면 $\alpha(1-\gamma) > 0$이므로:

$$1 - \alpha(1-\gamma) < 1$$

기댓값 업데이트마다 $\|Q - q_*\|_\infty$가 $(1-\alpha(1-\gamma))$ 비율로 **strictly 감소**합니다.

Robbins-Monro 조건과 함께 확률적 근사 이론을 적용하면:

$$\|Q_t - q_*\|_\infty \xrightarrow{a.s.} 0 \qquad \blacksquare$$

### TD(0) 수렴과의 비교

| | TD(0) | Q-learning |
|---|---|---|
| 수렴 대상 | $v_\pi$ (고정 policy) | $q_*$ (optimal) |
| 핵심 도구 | Jacobian 안정성 | $T^*$ contraction + Lyapunov |
| 조건 | Robbins-Monro | Robbins-Monro + 모든 (s,a) 방문 |

---

## 3. n-step Return 오차 감소 성질 증명

> **위치**: Chapter 7 — n-step return이 실제로 $v_\pi$에 수렴하는 이론적 근거

### 정리

임의의 value function $V$에 대해:

$$\max_s \left|\mathbb{E}_\pi\left[G_{t:t+n} \mid S_t = s\right] - v_\pi(s)\right| \leq \gamma^n \max_s |V(s) - v_\pi(s)|$$

### 핵심 보조 결과

**Bellman equation의 $n$회 전개**: $v_\pi$의 Bellman equation을 귀납적으로 $n$번 적용하면:

$$v_\pi(s) = \mathbb{E}_\pi\left[\sum_{k=0}^{n-1} \gamma^k R_{t+k+1} + \gamma^n v_\pi(S_{t+n}) \;\Big|\; S_t = s\right] \quad \cdots (*)$$

이것을 먼저 증명합니다.

**귀납 증명**:

$n=1$: Bellman equation의 정의 자체입니다.

$n \to n+1$: $n$회 전개 결과 $(*)$에서 $\gamma^n v_\pi(S_{t+n})$ 부분에 Bellman equation을 한 번 더 적용합니다:

$$\gamma^n v_\pi(S_{t+n}) = \gamma^n \mathbb{E}_\pi[R_{t+n+1} + \gamma v_\pi(S_{t+n+1}) \mid S_{t+n}]$$

기댓값의 탑 성질(tower property)에 의해:

$$\mathbb{E}_\pi[\gamma^n v_\pi(S_{t+n}) \mid S_t=s] = \mathbb{E}_\pi[\gamma^n R_{t+n+1} + \gamma^{n+1} v_\pi(S_{t+n+1}) \mid S_t=s]$$

$(*)$에 대입하면 $n+1$회 전개 공식이 성립합니다. $\checkmark$

### 본 증명

n-step return의 정의:

$$G_{t:t+n} = \sum_{k=0}^{n-1} \gamma^k R_{t+k+1} + \gamma^n V(S_{t+n})$$

기댓값을 취하면:

$$\mathbb{E}_\pi[G_{t:t+n} \mid S_t=s] = \mathbb{E}_\pi\left[\sum_{k=0}^{n-1} \gamma^k R_{t+k+1} + \gamma^n V(S_{t+n}) \;\Big|\; S_t=s\right]$$

보조 결과 $(*)$와의 차이:

$$\mathbb{E}_\pi[G_{t:t+n} \mid S_t=s] - v_\pi(s) = \mathbb{E}_\pi\left[\gamma^n (V(S_{t+n}) - v_\pi(S_{t+n})) \mid S_t=s\right]$$

(두 식에서 $\sum \gamma^k R_{t+k+1}$ 항이 상쇄됨)

절댓값과 기댓값의 Jensen 부등식 ($|\mathbb{E}[X]| \leq \mathbb{E}[|X|]$) 적용:

$$\left|\mathbb{E}_\pi[G_{t:t+n} \mid S_t=s] - v_\pi(s)\right| \leq \gamma^n \mathbb{E}_\pi\left[|V(S_{t+n}) - v_\pi(S_{t+n})| \mid S_t=s\right]$$

$$\leq \gamma^n \max_{s'} |V(s') - v_\pi(s')|$$

$s$에 대해 supremum을 취하면:

$$\max_s \left|\mathbb{E}_\pi[G_{t:t+n} \mid S_t=s] - v_\pi(s)\right| \leq \gamma^n \|V - v_\pi\|_\infty \qquad \blacksquare$$

### 해석

이 부등식이 말하는 것:

- $n$이 클수록 bootstrap 오차가 $\gamma^n$으로 빠르게 감소
- $V$가 $v_\pi$와 멀수록 ($\|V - v_\pi\|_\infty$가 클수록) 오차도 비례해서 커짐
- $\gamma = 0.9$, $n = 10$이면 오차 상한이 $0.9^{10} \approx 0.349$배로 감소
- 이것이 n-step TD가 TD(0)보다 좋은 이유: 더 정확한 target 사용

---

## 4. TD(λ)와 n-step Return의 등가성 증명

> **위치**: Chapter 7 심화 — TD(λ)가 모든 n-step return의 λ-가중 평균임을 증명

### λ-return 정의

$$G_t^\lambda \doteq (1-\lambda)\sum_{n=1}^{\infty} \lambda^{n-1} G_{t:t+n}$$

### 증명 1: 가중치의 합이 1임을 확인

$$\sum_{n=1}^{\infty} (1-\lambda)\lambda^{n-1} = (1-\lambda) \cdot \frac{1}{1-\lambda} = 1 \qquad (\lambda \in [0,1)) \checkmark$$

### 증명 2: $\lambda = 0$이면 TD(0)와 동일

$$G_t^{\lambda=0} = (1-0)\sum_{n=1}^{\infty} 0^{n-1} G_{t:t+n}$$

$0^{n-1} = 0$ for $n > 1$, $0^0 = 1$ for $n = 1$이므로:

$$= 1 \cdot G_{t:t+1} = R_{t+1} + \gamma V(S_{t+1}) \qquad \checkmark$$

### 증명 3: $\lambda = 1$이면 MC return과 동일 (유한 에피소드)

에피소드 길이가 $T$이면 $t + n \geq T$인 모든 $n$에 대해 $G_{t:t+n} = G_t$ (bootstrap 항 없음).

$G_t^\lambda$를 두 부분으로 나눕니다:

$$G_t^\lambda = (1-\lambda)\sum_{n=1}^{T-t-1} \lambda^{n-1} G_{t:t+n} + (1-\lambda)\sum_{n=T-t}^{\infty} \lambda^{n-1} G_t$$

두 번째 항:

$$(1-\lambda)\sum_{n=T-t}^{\infty} \lambda^{n-1} G_t = G_t \cdot (1-\lambda) \cdot \frac{\lambda^{T-t-1}}{1-\lambda} = G_t \cdot \lambda^{T-t-1}$$

따라서:

$$G_t^\lambda = (1-\lambda)\sum_{n=1}^{T-t-1} \lambda^{n-1} G_{t:t+n} + \lambda^{T-t-1} G_t$$

$\lambda \to 1$ 극한에서:
- 첫 항: $(1-\lambda) \to 0$이 각 항을 소멸
- 둘째 항: $\lambda^{T-t-1} \to 1^{T-t-1} = 1$

$$\lim_{\lambda \to 1} G_t^\lambda = G_t \qquad \checkmark$$

### 증명 4: $G_t^\lambda$의 재귀 공식 유도

이것이 실제 구현에서 핵심입니다.

$$G_t^\lambda = (1-\lambda)\sum_{n=1}^{\infty} \lambda^{n-1} G_{t:t+n}$$

$G_{t:t+n}$을 점화식 $G_{t:t+n} = R_{t+1} + \gamma G_{t+1:t+n}$으로 분해합니다:

$$= (1-\lambda)\sum_{n=1}^{\infty} \lambda^{n-1} \left[R_{t+1} + \gamma G_{t+1:t+n}\right]$$

$$= (1-\lambda)R_{t+1}\sum_{n=1}^{\infty}\lambda^{n-1} + \gamma(1-\lambda)\sum_{n=1}^{\infty}\lambda^{n-1} G_{t+1:t+n}$$

$$= R_{t+1} + \gamma(1-\lambda)\sum_{n=1}^{\infty}\lambda^{n-1} G_{t+1:t+n}$$

$n \geq 2$이면 $G_{t+1:t+n} = G_{t+1:(t+1)+(n-1)}$이므로 인덱스를 $m = n-1$로 치환합니다:

$$= R_{t+1} + \gamma\left[(1-\lambda)G_{t+1:t+1} + (1-\lambda)\sum_{m=1}^{\infty}\lambda^m G_{t+1:(t+1)+m}\right]$$

$G_{t+1:t+1} = V(S_{t+1})$이고 두 번째 항은 $\lambda \cdot G_{t+1}^\lambda$이므로:

$$\boxed{G_t^\lambda = R_{t+1} + \gamma\left[(1-\lambda)V(S_{t+1}) + \lambda G_{t+1}^\lambda\right]}$$

이 재귀 공식이 **eligibility traces**의 backward view와 동등합니다.

### TD(λ) Update와의 연결

$$V(S_t) \leftarrow V(S_t) + \alpha\left[G_t^\lambda - V(S_t)\right]$$

재귀 공식을 이용하면 이것을 각 step의 TD error $\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$의 기하급수 합으로 표현할 수 있습니다:

$$G_t^\lambda - V(S_t) = \sum_{k=t}^{T-1} (\gamma\lambda)^{k-t}\, \delta_k$$

**증명**: $e_t = G_t^\lambda - V(S_t)$를 재귀 공식으로 전개합니다:

$$e_t = R_{t+1} + \gamma(1-\lambda)V(S_{t+1}) + \gamma\lambda G_{t+1}^\lambda - V(S_t)$$

$$= \underbrace{R_{t+1} + \gamma V(S_{t+1}) - V(S_t)}_{\delta_t} + \gamma\lambda(G_{t+1}^\lambda - V(S_{t+1}))$$

$$= \delta_t + \gamma\lambda\, e_{t+1}$$

이 재귀를 풀면:

$$e_t = \sum_{k=t}^{T-1} (\gamma\lambda)^{k-t} \delta_k \qquad \blacksquare$$

이것이 **TD(λ)의 forward view = backward view의 등가성**입니다.

---

## 5. Maximization Bias의 Jensen 부등식 증명

> **위치**: Chapter 6 — Q-learning의 체계적 overestimation과 Double Q-learning의 근거

### 정리 1: Maximization Bias의 존재

$Q(s, a_i)$가 $q(s, a_i)$의 독립적인 unbiased 추정치이면:

$$\mathbb{E}\left[\max_{i} Q(s, a_i)\right] \geq \max_i q(s, a_i) = \max_i \mathbb{E}[Q(s, a_i)]$$

### 증명

$\max$ 함수가 convex임을 먼저 확인합니다.

$f(x_1, \ldots, x_k) = \max_i x_i$는 $k$개의 선형 함수 $f_i(x) = x_i$의 pointwise maximum이므로 convex입니다.

**Jensen's inequality**: $\phi$가 convex이면 $\mathbb{E}[\phi(X)] \geq \phi(\mathbb{E}[X])$.

$\phi = \max$, $X = (Q(s,a_1), \ldots, Q(s,a_k))$로 놓으면:

$$\mathbb{E}\left[\max_i Q(s,a_i)\right] \geq \max_i \mathbb{E}[Q(s,a_i)] = \max_i q(s,a_i) \qquad \blacksquare$$

**등호 조건**: $Q(s,a_i)$가 상수 (분산 = 0)일 때만 등호 성립. 노이즈가 있는 한 항상 overestimate.

### 정리 2: Double Q-learning의 Bias 제거

$Q_1, Q_2$가 독립이고 $\mathbb{E}[Q_j(s,a)] = q(s,a)$이면:

$$\mathbb{E}\left[Q_2\!\left(s, \arg\max_a Q_1(s,a)\right)\right] \leq \max_a q(s,a)$$

### 증명

$a^* = \arg\max_a Q_1(s,a)$로 정의합니다. $a^*$는 $Q_1$의 함수이므로 $Q_2$와 독립입니다.

$$\mathbb{E}\left[Q_2(s, a^*)\right] = \mathbb{E}_{a^*}\left[\mathbb{E}_{Q_2}[Q_2(s, a^*) \mid a^*]\right]$$

조건부 기댓값에서 $a^*$가 고정되면:

$$\mathbb{E}_{Q_2}[Q_2(s, a^*) \mid a^*] = q(s, a^*)$$

따라서:

$$\mathbb{E}[Q_2(s, a^*)] = \mathbb{E}_{a^*}[q(s, a^*)] = \mathbb{E}\left[q\!\left(s, \arg\max_a Q_1(s,a)\right)\right]$$

$a^* = \arg\max_a Q_1(s,a)$는 noise가 있는 $Q_1$으로 선택되므로, **진짜 최적 행동을 선택할 보장이 없습니다**:

$$\mathbb{E}[q(s, a^*)] \leq \max_a q(s,a) \qquad \blacksquare$$

### 두 추정량의 비교

| | Q-learning ($\max Q$) | Double Q ($Q_2(\arg\max Q_1)$) |
|---|---|---|
| **기댓값** | $\geq \max_a q(s,a)$ (overestimate) | $\leq \max_a q(s,a)$ (underestimate 가능) |
| **편향 방향** | 양의 편향 (항상) | 음의 편향 (가능) |
| **등호 조건** | 분산 = 0 | $Q_1$이 최적 행동을 올바르게 식별할 때 |

실용적으로 음의 편향(underestimate)이 양의 편향(overestimate)보다 훨씬 안전합니다. Overestimate는 suboptimal action을 optimal로 착각하게 하여 수렴을 방해하는 반면, Underestimate는 탐색을 조금 더 유도하는 정도의 영향만 미칩니다.

### Maximization Bias의 크기 정량화

$Q(s,a_i) \sim \mathcal{N}(q_i, \sigma^2)$ i.i.d., $k$개 행동, $q_i = 0$ (모두 동일)이면:

$$\mathbb{E}\left[\max_i Q(s,a_i)\right] = \mathbb{E}[Z_{(k)}]$$

$Z_{(k)}$는 $k$개 표준 정규 확률변수의 최댓값입니다. 순서통계량 이론으로부터:

$$\mathbb{E}[Z_{(k)}] \approx \sigma\sqrt{2\ln k}$$

| $k$ | $\mathbb{E}[\max Q] / \sigma$ | 실제 최적값 |
|---|---|---|
| 2 | $\approx 0.564$ | 0 |
| 5 | $\approx 1.163$ | 0 |
| 10 | $\approx 1.539$ | 0 |
| 100 | $\approx 2.508$ | 0 |

행동 수 $k$가 늘수록 bias가 $\sqrt{2\ln k}$로 증가합니다. 이것이 많은 행동을 가진 환경에서 Double Q-learning이 특히 중요한 이유입니다.

---

## 전체 증명 구조 요약

```
Banach Fixed Point Theorem (Part 1, Section 1)
        │
        ├── Policy Evaluation 수렴 (T^π contraction)
        ├── Value Iteration 수렴 (T^* contraction)
        │
Policy Improvement Theorem (Part 1, Section 2)
        │
        ├── Policy Iteration 수렴 (유한 policy 공간 + 단조 개선)
        └── ε-soft 버전 (Part 1, Section 3)

IS Unbiasedness (Part 1, Section 4)
        │
        └── Weighted IS Consistency (Part 1, Section 5)

TD(0) 수렴 (Part 2, Section 1) — ODE 방법론
        │
        ├── Jacobian 안정성 (γP_π - I의 고유값)
        └── Robbins-Monro + martingale difference noise

Q-learning 수렴 (Part 2, Section 2) — Lyapunov
        │
        └── T^* contraction → Lyapunov 감소 → 수렴

n-step 오차 감소 (Part 2, Section 3)
        │
        └── Bellman n회 전개 → γ^n ||V - v_π||_∞ 상한

TD(λ) = n-step 평균 (Part 2, Section 4)
        │
        ├── λ=0 → TD(0), λ=1 → MC
        └── 재귀 공식 → Σ(γλ)^k δ_k (eligibility traces)

Maximization Bias (Part 2, Section 5)
        │
        ├── max의 convexity → Jensen 부등식
        └── Double Q: 독립성 → unbiased selection
```

---

## 핵심 수식 한눈에 보기

**TD(0)의 ODE** ($\alpha \to 0$ 극한):
$$\dot{\mathbf{V}} = T^\pi \mathbf{V} - \mathbf{V}, \qquad J_F = \gamma\mathbf{P}_\pi - \mathbf{I},\quad \text{Re}(\gamma\lambda_i - 1) < 0$$

**Q-learning Lyapunov 감소**:
$$\|\bar{Q} - q_*\|_\infty \leq (1 - \alpha(1-\gamma))\|Q - q_*\|_\infty$$

**n-step 오차 상한**:
$$\max_s\left|\mathbb{E}_\pi[G_{t:t+n}|S_t=s] - v_\pi(s)\right| \leq \gamma^n\|V - v_\pi\|_\infty$$

**TD(λ) 재귀 공식**:
$$G_t^\lambda = R_{t+1} + \gamma\left[(1-\lambda)V(S_{t+1}) + \lambda G_{t+1}^\lambda\right]$$

**TD(λ) Forward = Backward**:
$$G_t^\lambda - V(S_t) = \sum_{k=t}^{T-1}(\gamma\lambda)^{k-t}\delta_k$$

**Maximization Bias (Jensen)**:
$$\mathbb{E}[\max_i Q_i] \geq \max_i \mathbb{E}[Q_i] = \max_i q_i$$
