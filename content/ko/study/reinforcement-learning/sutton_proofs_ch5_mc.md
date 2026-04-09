---
title: "Reinforcement Learning: Chapter 5 Proofs: MC"
category: "Reinforcement Learning"
weight: 15
date: 2026-04-09
---

# Chapter 5 Proofs: MC

> Sutton & Barto ??援먯옱?먯꽌 ?앸왂???섑븰??利앸챸 ?곸꽭 ?뺣━

---
## 2. Policy Improvement Theorem ?꾩쟾??利앸챸

> **?꾩튂**: Chapter 4 ??Policy Iteration???듭떖 ?대줎??洹쇨굅

### ?뺣━

??deterministic policy $\pi$, $\pi'$?????

$$q_\pi(s, \pi'(s)) \geq v_\pi(s), \quad \forall s \in \mathcal{S}$$

?대㈃:

$$v_{\pi'}(s) \geq v_\pi(s), \quad \forall s \in \mathcal{S}$$

### 利앸챸

$v_\pi(s) \leq q_\pi(s, \pi'(s))$?먯꽌 異쒕컻?⑸땲??

$q_\pi$???뺤쓽瑜??꾧컻?⑸땲??

$$v_\pi(s) \leq q_\pi(s, \pi'(s)) = \mathbb{E}\left[R_{t+1} + \gamma v_\pi(S_{t+1}) \mid S_t=s, A_t=\pi'(s)\right]$$

**?듭떖 ?④퀎**: $v_\pi(S_{t+1}) \leq q_\pi(S_{t+1}, \pi'(S_{t+1}))$????낇빀?덈떎 (媛?뺤씠 紐⑤뱺 ?곹깭?먯꽌 ?깅┰?섎?濡?:

$$\leq \mathbb{E}\left[R_{t+1} + \gamma\, q_\pi(S_{t+1}, \pi'(S_{t+1})) \mid S_t=s, A_t=\pi'(s)\right]$$

$q_\pi(S_{t+1}, \pi'(S_{t+1}))$???ㅼ떆 ?꾧컻?⑸땲??

$$= \mathbb{E}\left[R_{t+1} + \gamma \mathbb{E}\left[R_{t+2} + \gamma v_\pi(S_{t+2}) \mid S_{t+1}, A_{t+1}=\pi'(S_{t+1})\right] \mid S_t=s, A_t=\pi'(s)\right]$$

$$= \mathbb{E}\left[R_{t+1} + \gamma R_{t+2} + \gamma^2 v_\pi(S_{t+2}) \mid S_t=s, A_t=\pi'(s)\right]$$

??移섑솚??臾댄븳??諛섎났?⑸땲??

$$\leq \mathbb{E}\left[R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots \mid S_t=s, A_t=\pi'(s)\right]$$

紐⑤뱺 ?댄썑 ?됰룞??$\pi'$瑜??곕Ⅴ誘濡?

$$= \mathbb{E}_{\pi'}\left[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \mid S_t=s\right] = v_{\pi'}(s)$$

$$\therefore\quad v_\pi(s) \leq v_{\pi'}(s) \qquad \blacksquare$$

**臾댄븳 ?꾧컻???꾨???*: $|\gamma^k v_\pi(S_{t+k})| \leq \gamma^k \|v_\pi\|_\infty \to 0$ ($\gamma < 1$?닿퀬 ?좏븳 MDP?먯꽌 $v_\pi$媛 bounded?대?濡?.

### ?섎졃 吏??
$v_{\pi'} = v_\pi$?대㈃ (???댁긽 媛쒖꽑???놁쑝硫? 紐⑤뱺 $s$?먯꽌:

$$v_\pi(s) = \max_a \sum_{s',r} p(s',r|s,a)\left[r + \gamma v_\pi(s')\right] = v_*(s)$$

?닿쾬? ?뺥솗??**Bellman Optimality Equation**?낅땲?? 利? 媛쒖꽑??硫덉텛硫??대? 理쒖쟻?낅땲??

---

## 3. 琯-soft Policy Improvement 利앸챸

> **?꾩튂**: Chapter 5 ??On-policy MC Control, 琯-greedy policy???대줎??蹂댁옣

### 諛곌꼍

Exploring Starts ?놁씠 ?먯깋??蹂댁옣?섍린 ?꾪빐 琯-soft policy瑜??ъ슜?⑸땲?? ?대븣 Policy Improvement Theorem???ъ쟾???깅┰?섎뒗吏 利앸챸?댁빞 ?⑸땲??

### ?뺣━

?꾩옱 policy $\pi$媛 琯-soft?닿퀬, $\pi'$媛 $Q_\pi$?????琯-greedy policy?대㈃:

$$v_{\pi'}(s) \geq v_\pi(s), \quad \forall s \in \mathcal{S}$$

### 利앸챸

$a^*(s) = \arg\max_a Q_\pi(s, a)$濡??뺤쓽?⑸땲??

**琯-greedy policy???뺤쓽**:

$$\pi'(a \mid s) = \begin{cases} 1 - \varepsilon + \dfrac{\varepsilon}{|\mathcal{A}|} & a = a^*(s) \\[6pt] \dfrac{\varepsilon}{|\mathcal{A}|} & a \neq a^*(s) \end{cases}$$

**$q_\pi(s, \pi'(s))$ 怨꾩궛** (stochastic policy?대?濡?湲곕뙎媛?:

$$\sum_a \pi'(a|s)\, Q_\pi(s,a) = \frac{\varepsilon}{|\mathcal{A}|}\sum_a Q_\pi(s,a) + (1-\varepsilon)\, Q_\pi(s, a^*(s))$$

**$v_\pi(s)$???곹븳 怨꾩궛**:

?꾩옱 policy $\pi$??琯-soft?대?濡? $\pi(a|s) - \frac{\varepsilon}{|\mathcal{A}|} \geq 0$?닿퀬 洹??⑹? $1 - \varepsilon$?낅땲??

$$v_\pi(s) = \sum_a \pi(a|s)\, Q_\pi(s,a) = \frac{\varepsilon}{|\mathcal{A}|}\sum_a Q_\pi(s,a) + \sum_a \underbrace{\left(\pi(a|s) - \frac{\varepsilon}{|\mathcal{A}|}\right)}_{\geq\, 0,\ \text{?? = 1-\varepsilon} Q_\pi(s,a)$$

??踰덉㎏ ??뿉??$Q_\pi(s,a) \leq \max_{a'} Q_\pi(s,a') = Q_\pi(s, a^*(s))$?대?濡?

$$\sum_a \left(\pi(a|s) - \frac{\varepsilon}{|\mathcal{A}|}\right) Q_\pi(s,a) \leq (1-\varepsilon)\, Q_\pi(s, a^*(s))$$

?곕씪??

$$v_\pi(s) \leq \frac{\varepsilon}{|\mathcal{A}|}\sum_a Q_\pi(s,a) + (1-\varepsilon)\, Q_\pi(s, a^*(s)) = \sum_a \pi'(a|s)\, Q_\pi(s,a)$$

利?$v_\pi(s) \leq q_\pi(s, \pi'(s))$媛 紐⑤뱺 $s$?먯꽌 ?깅┰?⑸땲??

?댁젣 Policy Improvement Theorem (Section 2)???곸슜?섎㈃:

$$v_{\pi'}(s) \geq v_\pi(s) \qquad \blacksquare$$

### 以묒슂???쒓퀎

?? 琯-soft policy ?섏뿉?쒖쓽 ?섎졃 吏?먯? $v_*$媛 ?꾨땶 **琯-soft policy 以?理쒖꽑**??$v_{\pi^*_\varepsilon}$?낅땲??

$$v_{\pi^*_\varepsilon}(s) = \frac{\varepsilon}{|\mathcal{A}|}\sum_a q_{\pi^*_\varepsilon}(s,a) + (1-\varepsilon)\max_a q_{\pi^*_\varepsilon}(s,a)$$

$\varepsilon \to 0$??洹뱁븳?먯꽌留?$v_{\pi^*_\varepsilon} \to v_*$媛 ?⑸땲??

---

## 4. Importance Sampling Unbiasedness ?꾨???利앸챸

> **?꾩튂**: Chapter 5 ??Off-policy MC???듭떖 ?대줎??洹쇨굅

### 諛곌꼍

Behavior policy $b$濡??섏쭛???곗씠?곕줈 target policy $\pi$??value function??異붿젙?⑸땲?? IS estimator媛 ?ㅼ젣濡?$v_\pi(s)$??unbiased estimate?몄? ?꾨???利앸챸?⑸땲??

### ?뺣━

$$\mathbb{E}_b\left[\rho_{t:T-1}\, G_t \mid S_t = s\right] = v_\pi(s)$$

?? coverage 議곌굔 $\pi(a \mid s) > 0 \implies b(a \mid s) > 0$???깅┰?댁빞 ?⑸땲??

### 利앸챸

?먰뵾?뚮뱶 沅ㅼ쟻 $\tau = (A_t, S_{t+1}, R_{t+1}, A_{t+1}, \ldots, S_T)$??怨듦컙??$\Omega$???섎㈃:

$$\mathbb{E}_b[\rho_{t:T-1} G_t \mid S_t=s] = \sum_{\tau \in \Omega} P^b(\tau \mid S_t=s) \cdot \rho(\tau) \cdot G(\tau)$$

$b$ ?섏뿉?쒖쓽 沅ㅼ쟻 ?뺣쪧??紐낆떆?곸쑝濡??곷땲??

$$P^b(\tau \mid S_t=s) = \prod_{k=t}^{T-1} b(A_k \mid S_k)\, p(S_{k+1}, R_{k+1} \mid S_k, A_k)$$

IS ratio:

$$\rho(\tau) = \rho_{t:T-1} = \prod_{k=t}^{T-1} \frac{\pi(A_k \mid S_k)}{b(A_k \mid S_k)}$$

????쓣 怨깊빀?덈떎:

$$P^b(\tau \mid S_t=s) \cdot \rho(\tau) = \prod_{k=t}^{T-1} \underbrace{b(A_k \mid S_k) \cdot \frac{\pi(A_k \mid S_k)}{b(A_k \mid S_k)}}_{= \pi(A_k \mid S_k)} \cdot p(S_{k+1}, R_{k+1} \mid S_k, A_k)$$

$$= \prod_{k=t}^{T-1} \pi(A_k \mid S_k)\, p(S_{k+1}, R_{k+1} \mid S_k, A_k) = P^\pi(\tau \mid S_t=s)$$

?곕씪??

$$\mathbb{E}_b[\rho_{t:T-1} G_t \mid S_t=s] = \sum_{\tau \in \Omega} P^\pi(\tau \mid S_t=s) \cdot G(\tau) = \mathbb{E}_\pi[G_t \mid S_t=s] = v_\pi(s) \qquad \blacksquare$$

### ?듭떖 ?듭같

- $b(A_k \mid S_k)$媛 遺꾩옄쨌遺꾨え?먯꽌 **?뺥솗???쎈텇**?⑸땲??- ?섍꼍???꾩씠 ?뺣쪧 $p(S_{k+1} \mid S_k, A_k)$???쎈텇?⑸땲????**紐⑤뜽 遺덊븘??*
- Coverage 議곌굔???놁쑝硫?$b(A_k|S_k) = 0$??遺꾨え媛 ?깆옣?섏뿬 利앸챸??源⑥쭛?덈떎

### IS ratio??湲곕뙎媛믪씠 1?꾩쓣 蹂댁엫

??利앸챸???뱀닔 寃쎌슦濡?$G(\tau) = 1$?대㈃:

$$\mathbb{E}_b[\rho_{t:T-1}] = \sum_\tau P^b(\tau) \cdot \frac{P^\pi(\tau)}{P^b(\tau)} = \sum_\tau P^\pi(\tau) = 1$$

???깆쭏? ?댄썑 Weighted IS??consistency 利앸챸?먯꽌 ?듭떖?곸쑝濡??ъ슜?⑸땲??
