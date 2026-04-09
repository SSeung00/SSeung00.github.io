---
title: "RL Proofs Part 1: 기초편"
category: "Reinforcement Learning"
date: 2026-04-09
weight: 18
---

# Reinforcement Learning 핵심 수학적 증명 (Part 1: 기초편)

> Sutton & Barto Ch.2~6 — 교재에서 생략된 수학적 증명 완전 정리

---

## 목차

1. [Banach Fixed Point Theorem](#1-banach-fixed-point-theorem)
2. [Policy Improvement Theorem 완전한 증명](#2-policy-improvement-theorem-완전한-증명)
3. [ε-soft Policy Improvement 증명](#3-ε-soft-policy-improvement-증명)
4. [Importance Sampling Unbiasedness 엄밀한 증명](#4-importance-sampling-unbiasedness-엄밀한-증명)
5. [Weighted IS Consistency 증명](#5-weighted-is-consistency-증명)

---

## 1. Banach Fixed Point Theorem

> **위치**: Chapter 4 — Policy Evaluation, Value Iteration 수렴의 수학적 근거

Chapter 4에서 Policy Evaluation과 Value Iteration의 수렴을 설명할 때 "Banach Fixed Point Theorem을 적용한다"고만 했습니다. 이 정리 자체를 처음부터 증명합니다.

### 정리

$(X, d)$가 완비 거리 공간(complete metric space)이고, $T: X \to X$가 다음을 만족하는 $\gamma$-contraction ($0 \leq \gamma < 1$)이면:

$$d(Tx, Ty) \leq \gamma\, d(x, y), \quad \forall x, y \in X$$

$T$는 **유일한 고정점(fixed point) $x^*$**를 가지며, 임의의 $x_0 \in X$에서 시작한 반복열 $x_k = T^k x_0$은 $x^*$로 수렴합니다.

### 증명

#### Step 1: $\{x_k\}$가 Cauchy 수열임을 보임

임의의 $m > n$에 대해 삼각부등식을 반복 적용합니다:

$$d(x_m, x_n) \leq \sum_{k=n}^{m-1} d(x_{k+1}, x_k)$$

각 항에 contraction 조건을 $k$번 반복 적용합니다:

$$d(x_{k+1}, x_k) = d(Tx_k, Tx_{k-1}) \leq \gamma\, d(x_k, x_{k-1}) \leq \gamma^2\, d(x_{k-1}, x_{k-2}) \leq \cdots \leq \gamma^k\, d(x_1, x_0)$$

따라서:

$$d(x_m, x_n) \leq \sum_{k=n}^{m-1} \gamma^k\, d(x_1, x_0) \leq d(x_1, x_0) \sum_{k=n}^{\infty} \gamma^k = \frac{\gamma^n}{1-\gamma}\, d(x_1, x_0)$$

$\gamma < 1$이므로 $n \to \infty$이면 $\frac{\gamma^n}{1-\gamma} \to 0$. 따라서 $\{x_k\}$는 **Cauchy 수열**입니다.

#### Step 2: 극한 $x^*$가 고정점임을 보임

$X$가 완비이므로 Cauchy 수열은 $X$ 안에서 수렴합니다: $x_k \to x^* \in X$.

$T$는 contraction이므로 연속이고, 따라서:

$$Tx^* = T\!\left(\lim_{k \to \infty} x_k\right) = \lim_{k \to \infty} Tx_k = \lim_{k \to \infty} x_{k+1} = x^*$$

즉 $x^*$는 $T$의 고정점입니다.

#### Step 3: 고정점의 유일성

$x^*$와 $y^*$가 모두 고정점이라 가정합니다:

$$d(x^*, y^*) = d(Tx^*, Ty^*) \leq \gamma\, d(x^*, y^*)$$

$$(1 - \gamma)\, d(x^*, y^*) \leq 0$$

$\gamma < 1$이므로 $1 - \gamma > 0$이고, $d \geq 0$이므로:

$$d(x^*, y^*) = 0 \implies x^* = y^* \qquad \blacksquare$$

### RL에서의 적용

| RL 알고리즘 | 연산자 $T$ | 거리 공간 | 고정점 |
|---|---|---|---|
| Policy Evaluation | $T^\pi v = \sum_a \pi \sum_{s',r} p[r + \gamma v']$ | $(\mathbb{R}^{\|\mathcal{S}\|}, \|\cdot\|_\infty)$ | $v_\pi$ |
| Value Iteration | $T^* v = \max_a \sum_{s',r} p[r + \gamma v']$ | $(\mathbb{R}^{\|\mathcal{S}\|}, \|\cdot\|_\infty)$ | $v_*$ |

두 경우 모두 $\gamma < 1$이면 Banach 정리의 조건을 만족하여 수렴이 보장됩니다.

### 수렴 속도

$k$번 반복 후 고정점까지의 거리:

$$\|x_k - x^*\|_\infty \leq \frac{\gamma^k}{1-\gamma}\, \|x_1 - x_0\|_\infty$$

$\gamma$가 작을수록, $k$가 클수록 오차가 **기하급수적으로 감소**합니다.

---

## 2. Policy Improvement Theorem 완전한 증명

> **위치**: Chapter 4 — Policy Iteration의 핵심 이론적 근거

### 정리

두 deterministic policy $\pi$, $\pi'$에 대해:

$$q_\pi(s, \pi'(s)) \geq v_\pi(s), \quad \forall s \in \mathcal{S}$$

이면:

$$v_{\pi'}(s) \geq v_\pi(s), \quad \forall s \in \mathcal{S}$$

### 증명

$v_\pi(s) \leq q_\pi(s, \pi'(s))$에서 출발합니다.

$q_\pi$의 정의를 전개합니다:

$$v_\pi(s) \leq q_\pi(s, \pi'(s)) = \mathbb{E}\left[R_{t+1} + \gamma v_\pi(S_{t+1}) \mid S_t=s, A_t=\pi'(s)\right]$$

**핵심 단계**: $v_\pi(S_{t+1}) \leq q_\pi(S_{t+1}, \pi'(S_{t+1}))$을 대입합니다 (가정이 모든 상태에서 성립하므로):

$$\leq \mathbb{E}\left[R_{t+1} + \gamma\, q_\pi(S_{t+1}, \pi'(S_{t+1})) \mid S_t=s, A_t=\pi'(s)\right]$$

$q_\pi(S_{t+1}, \pi'(S_{t+1}))$을 다시 전개합니다:

$$= \mathbb{E}\left[R_{t+1} + \gamma \mathbb{E}\left[R_{t+2} + \gamma v_\pi(S_{t+2}) \mid S_{t+1}, A_{t+1}=\pi'(S_{t+1})\right] \mid S_t=s, A_t=\pi'(s)\right]$$

$$= \mathbb{E}\left[R_{t+1} + \gamma R_{t+2} + \gamma^2 v_\pi(S_{t+2}) \mid S_t=s, A_t=\pi'(s)\right]$$

이 치환을 무한히 반복합니다:

$$\leq \mathbb{E}\left[R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots \mid S_t=s, A_t=\pi'(s)\right]$$

모든 이후 행동이 $\pi'$를 따르므로:

$$= \mathbb{E}_{\pi'}\left[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \mid S_t=s\right] = v_{\pi'}(s)$$

$$\therefore\quad v_\pi(s) \leq v_{\pi'}(s) \qquad \blacksquare$$

**무한 전개의 엄밀성**: $|\gamma^k v_\pi(S_{t+k})| \leq \gamma^k \|v_\pi\|_\infty \to 0$ ($\gamma < 1$이고 유한 MDP에서 $v_\pi$가 bounded이므로).

### 수렴 지점

$v_{\pi'} = v_\pi$이면 (더 이상 개선이 없으면) 모든 $s$에서:

$$v_\pi(s) = \max_a \sum_{s',r} p(s',r|s,a)\left[r + \gamma v_\pi(s')\right] = v_*(s)$$

이것은 정확히 **Bellman Optimality Equation**입니다. 즉, 개선이 멈추면 이미 최적입니다.

---

## 3. ε-soft Policy Improvement 증명

> **위치**: Chapter 5 — On-policy MC Control, ε-greedy policy의 이론적 보장

### 배경

Exploring Starts 없이 탐색을 보장하기 위해 ε-soft policy를 사용합니다. 이때 Policy Improvement Theorem이 여전히 성립하는지 증명해야 합니다.

### 정리

현재 policy $\pi$가 ε-soft이고, $\pi'$가 $Q_\pi$에 대한 ε-greedy policy이면:

$$v_{\pi'}(s) \geq v_\pi(s), \quad \forall s \in \mathcal{S}$$

### 증명

$a^*(s) = \arg\max_a Q_\pi(s, a)$로 정의합니다.

**ε-greedy policy의 정의**:

$$\pi'(a \mid s) = \begin{cases} 1 - \varepsilon + \dfrac{\varepsilon}{|\mathcal{A}|} & a = a^*(s) \\[6pt] \dfrac{\varepsilon}{|\mathcal{A}|} & a \neq a^*(s) \end{cases}$$

**$q_\pi(s, \pi'(s))$ 계산** (stochastic policy이므로 기댓값):

$$\sum_a \pi'(a|s)\, Q_\pi(s,a) = \frac{\varepsilon}{|\mathcal{A}|}\sum_a Q_\pi(s,a) + (1-\varepsilon)\, Q_\pi(s, a^*(s))$$

**$v_\pi(s)$의 상한 계산**:

현재 policy $\pi$도 ε-soft이므로, $\pi(a|s) - \frac{\varepsilon}{|\mathcal{A}|} \geq 0$이고 그 합은 $1 - \varepsilon$입니다.

$$v_\pi(s) = \sum_a \pi(a|s)\, Q_\pi(s,a) = \frac{\varepsilon}{|\mathcal{A}|}\sum_a Q_\pi(s,a) + \sum_a \underbrace{\left(\pi(a|s) - \frac{\varepsilon}{|\mathcal{A}|}\right)}_{\geq\, 0,\ \text{합} = 1-\varepsilon} Q_\pi(s,a)$$

두 번째 항에서 $Q_\pi(s,a) \leq \max_{a'} Q_\pi(s,a') = Q_\pi(s, a^*(s))$이므로:

$$\sum_a \left(\pi(a|s) - \frac{\varepsilon}{|\mathcal{A}|}\right) Q_\pi(s,a) \leq (1-\varepsilon)\, Q_\pi(s, a^*(s))$$

따라서:

$$v_\pi(s) \leq \frac{\varepsilon}{|\mathcal{A}|}\sum_a Q_\pi(s,a) + (1-\varepsilon)\, Q_\pi(s, a^*(s)) = \sum_a \pi'(a|s)\, Q_\pi(s,a)$$

즉 $v_\pi(s) \leq q_\pi(s, \pi'(s))$가 모든 $s$에서 성립합니다.

이제 Policy Improvement Theorem (Section 2)을 적용하면:

$$v_{\pi'}(s) \geq v_\pi(s) \qquad \blacksquare$$

### 중요한 한계

단, ε-soft policy 하에서의 수렴 지점은 $v_*$가 아닌 **ε-soft policy 중 최선**인 $v_{\pi^*_\varepsilon}$입니다.

$$v_{\pi^*_\varepsilon}(s) = \frac{\varepsilon}{|\mathcal{A}|}\sum_a q_{\pi^*_\varepsilon}(s,a) + (1-\varepsilon)\max_a q_{\pi^*_\varepsilon}(s,a)$$

$\varepsilon \to 0$의 극한에서만 $v_{\pi^*_\varepsilon} \to v_*$가 됩니다.

---

## 4. Importance Sampling Unbiasedness 엄밀한 증명

> **위치**: Chapter 5 — Off-policy MC의 핵심 이론적 근거

### 배경

Behavior policy $b$로 수집한 데이터로 target policy $\pi$의 value function을 추정합니다. IS estimator가 실제로 $v_\pi(s)$의 unbiased estimate인지 엄밀히 증명합니다.

### 정리

$$\mathbb{E}_b\left[\rho_{t:T-1}\, G_t \mid S_t = s\right] = v_\pi(s)$$

단, coverage 조건 $\pi(a \mid s) > 0 \implies b(a \mid s) > 0$이 성립해야 합니다.

### 증명

에피소드 궤적 $\tau = (A_t, S_{t+1}, R_{t+1}, A_{t+1}, \ldots, S_T)$의 공간을 $\Omega$라 하면:

$$\mathbb{E}_b[\rho_{t:T-1} G_t \mid S_t=s] = \sum_{\tau \in \Omega} P^b(\tau \mid S_t=s) \cdot \rho(\tau) \cdot G(\tau)$$

$b$ 하에서의 궤적 확률을 명시적으로 씁니다:

$$P^b(\tau \mid S_t=s) = \prod_{k=t}^{T-1} b(A_k \mid S_k)\, p(S_{k+1}, R_{k+1} \mid S_k, A_k)$$

IS ratio:

$$\rho(\tau) = \rho_{t:T-1} = \prod_{k=t}^{T-1} \frac{\pi(A_k \mid S_k)}{b(A_k \mid S_k)}$$

두 항을 곱합니다:

$$P^b(\tau \mid S_t=s) \cdot \rho(\tau) = \prod_{k=t}^{T-1} \underbrace{b(A_k \mid S_k) \cdot \frac{\pi(A_k \mid S_k)}{b(A_k \mid S_k)}}_{= \pi(A_k \mid S_k)} \cdot p(S_{k+1}, R_{k+1} \mid S_k, A_k)$$

$$= \prod_{k=t}^{T-1} \pi(A_k \mid S_k)\, p(S_{k+1}, R_{k+1} \mid S_k, A_k) = P^\pi(\tau \mid S_t=s)$$

따라서:

$$\mathbb{E}_b[\rho_{t:T-1} G_t \mid S_t=s] = \sum_{\tau \in \Omega} P^\pi(\tau \mid S_t=s) \cdot G(\tau) = \mathbb{E}_\pi[G_t \mid S_t=s] = v_\pi(s) \qquad \blacksquare$$

### 핵심 통찰

- $b(A_k \mid S_k)$가 분자·분모에서 **정확히 약분**됩니다
- 환경의 전이 확률 $p(S_{k+1} \mid S_k, A_k)$도 약분됩니다 → **모델 불필요**
- Coverage 조건이 없으면 $b(A_k|S_k) = 0$인 분모가 등장하여 증명이 깨집니다

### IS ratio의 기댓값이 1임을 보임

위 증명의 특수 경우로 $G(\tau) = 1$이면:

$$\mathbb{E}_b[\rho_{t:T-1}] = \sum_\tau P^b(\tau) \cdot \frac{P^\pi(\tau)}{P^b(\tau)} = \sum_\tau P^\pi(\tau) = 1$$

이 성질은 이후 Weighted IS의 consistency 증명에서 핵심적으로 사용됩니다.

---

## 5. Weighted IS Consistency 증명

> **위치**: Chapter 5 — Weighted IS가 $v_\pi$로 수렴함을 보임

### 배경

Ordinary IS는 unbiased이지만 분산이 무한대가 될 수 있습니다. Weighted IS는 biased이지만 consistent (수렴 보장)합니다. 이 consistency를 증명합니다.

### 정리

상태 $s$의 $n$번째 방문까지의 Weighted IS estimator:

$$V_n(s) = \frac{\sum_{k=1}^{n} W_k\, G_k}{\sum_{k=1}^{n} W_k}, \qquad W_k = \rho_{t_k: T_k - 1}$$

는 $n \to \infty$이면 $v_\pi(s)$로 수렴합니다 (확률 1).

### 증명

$\mu = v_\pi(s)$로 표기합니다. $V_n - \mu$를 분해합니다:

$$V_n - \mu = \frac{\sum_k W_k G_k}{\sum_k W_k} - \mu = \frac{\sum_k W_k(G_k - \mu)}{\sum_k W_k}$$

분자와 분모를 $n$으로 나눕니다:

$$V_n - \mu = \frac{\frac{1}{n}\sum_k W_k(G_k - \mu)}{\frac{1}{n}\sum_k W_k}$$

**분모의 수렴**: 대수의 법칙(LLN)에 의해:

$$\frac{1}{n}\sum_k W_k \xrightarrow{a.s.} \mathbb{E}_b[W] = \mathbb{E}_b[\rho] = 1$$

(마지막 등호는 Section 4에서 증명한 $\mathbb{E}_b[\rho] = 1$)

**분자의 수렴**: 마찬가지로 LLN에 의해:

$$\frac{1}{n}\sum_k W_k(G_k - \mu) \xrightarrow{a.s.} \mathbb{E}_b[W(G - \mu)] = \mathbb{E}_b[\rho(G - \mu)]$$

IS unbiasedness (Section 4)에 의해:

$$\mathbb{E}_b[\rho G] = v_\pi(s) = \mu$$

따라서:

$$\mathbb{E}_b[\rho(G - \mu)] = \mathbb{E}_b[\rho G] - \mu\, \mathbb{E}_b[\rho] = \mu - \mu \cdot 1 = 0$$

**결합**:

$$V_n - \mu \xrightarrow{a.s.} \frac{0}{1} = 0$$

$$\therefore\quad V_n \xrightarrow{a.s.} v_\pi(s) \qquad \blacksquare$$

### Ordinary IS와의 비교

| | Ordinary IS | Weighted IS |
|---|---|---|
| **Bias** | 없음 (Unbiased) | 있음 (유한 $n$에서) |
| **Consistency** | ✅ | ✅ |
| **분산** | 무한대 가능 | 항상 유한 |
| **실용성** | 낮음 | 높음 |

Weighted IS의 분산이 유한한 이유: 분모 $\sum W_k$가 커지면 분자 $\sum W_k G_k$도 함께 커지므로 극단적인 $W_k$가 자동으로 normalize됩니다.

---

## 핵심 수식 한눈에 보기

**Banach 수렴 속도**:
$$\|x_k - x^*\|_\infty \leq \frac{\gamma^k}{1-\gamma}\|x_1 - x_0\|_\infty$$

**$\gamma$-Contraction** ($T^\pi$와 $T^*$ 공통):
$$\|Tu - Tv\|_\infty \leq \gamma\|u - v\|_\infty$$

**Policy Improvement의 귀납 부등식 체인**:
$$v_\pi(s) \leq q_\pi(s,\pi') \leq \mathbb{E}[R + \gamma q_\pi(S',\pi')] \leq \cdots \leq v_{\pi'}(s)$$

**IS ratio (전이 확률 약분 후)**:
$$\rho_{t:T-1} = \prod_{k=t}^{T-1}\frac{\pi(A_k|S_k)}{b(A_k|S_k)}, \qquad \mathbb{E}_b[\rho] = 1$$

**Weighted IS 분해**:
$$V_n - \mu = \frac{\frac{1}{n}\sum W_k(G_k - \mu)}{\frac{1}{n}\sum W_k} \xrightarrow{a.s.} \frac{0}{1} = 0$$

---

> **Part 2로**: TD(0) 수렴 증명 (ODE 방법론), Q-learning 수렴 증명 (Lyapunov), n-step 오차 감소, TD(λ)와 n-step 등가성, Maximization Bias의 Jensen 부등식 증명을 다룹니다.
