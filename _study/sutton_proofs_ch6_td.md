---
layout: post
title: "Reinforcement Learning: Chapter 6 Proofs: TD"
category: "Reinforcement Learning"
date: 2026-04-09
---

# Chapter 6 Proofs: TD

> Sutton & Barto ??援먯옱?먯꽌 ?앸왂???섑븰??利앸챸 ?곸꽭 ?뺣━

---
## 紐⑹감

1. [TD(0) ?섎졃 利앸챸 ??ODE 諛⑸쾿濡?(#1-td0-?섎졃-利앸챸--ode-諛⑸쾿濡?
2. [Q-learning ?섎졃 利앸챸 ??Lyapunov 諛⑸쾿](#2-q-learning-?섎졃-利앸챸--lyapunov-諛⑸쾿)
3. [n-step Return ?ㅼ감 媛먯냼 ?깆쭏 利앸챸](#3-n-step-return-?ㅼ감-媛먯냼-?깆쭏-利앸챸)
4. [TD(貫)? n-step Return???깃???利앸챸](#4-td貫?-n-step-return???깃???利앸챸)
5. [Maximization Bias??Jensen 遺?깆떇 利앸챸](#5-maximization-bias??jensen-遺?깆떇-利앸챸)

---

## 1. TD(0) ?섎졃 利앸챸 ??ODE 諛⑸쾿濡?
> **?꾩튂**: Chapter 6 ??TD(0)媛 $v_\pi$濡??섎졃?섎뒗 ?대줎??洹쇨굅

### 諛곌꼍

Chapter 4?먯꽌??Banach ?뺣━濡?$T^\pi$??諛섎났??$v_\pi$濡??섎졃?⑥쓣 蹂댁??듬땲?? ?섏?留?TD(0)??**湲곕뙎媛믪씠 ?꾨땶 ?섑뵆 ?섎굹**濡??낅뜲?댄듃?⑸땲?? ???뺣쪧??stochastic) ?낅뜲?댄듃媛 ?ъ쟾???섎졃?섎뒗吏??蹂꾨룄???대줎???꾩슂?⑸땲??

### ?ㅼ젙

TD(0) ?낅뜲?댄듃瑜?踰≫꽣 ?뺥깭濡??곷땲??($s$踰덉㎏ ?깅텇??$V(s)$):

$$\mathbf{V}_{t+1}(S_t) = \mathbf{V}_t(S_t) + \alpha_t\left[R_{t+1} + \gamma \mathbf{V}_t(S_{t+1}) - \mathbf{V}_t(S_t)\right]$$

?대? ?쇰컲?곸씤 ?뺣쪧??洹쇱궗(stochastic approximation) ?뺥깭濡??곷땲??

$$\mathbf{V}_{t+1} = \mathbf{V}_t + \alpha_t\left[F(\mathbf{V}_t) + \varepsilon_t\right]$$

?ш린??
- $F(\mathbf{V}) \doteq (T^\pi \mathbf{V}) - \mathbf{V}$ (湲곕뙎媛??낅뜲?댄듃 諛⑺뼢)
- $\varepsilon_t \doteq \left[R_{t+1} + \gamma \mathbf{V}_t(S_{t+1}) - \mathbf{V}_t(S_t)\right] - F(\mathbf{V}_t)(S_t)$ (?몄씠利?

### Robbins-Monro 議곌굔

?섎졃??蹂댁옣?섎뒗 step-size 議곌굔:

$$\sum_{t=0}^{\infty} \alpha_t = \infty, \qquad \sum_{t=0}^{\infty} \alpha_t^2 < \infty$$

### ?섎졃 利앸챸

#### Step 1: 怨좎젙???뺤씤

$F(\mathbf{V}^*) = 0$??$\mathbf{V}^*$瑜?援ы빀?덈떎:

$$F(\mathbf{V}^*) = T^\pi \mathbf{V}^* - \mathbf{V}^* = 0 \iff T^\pi \mathbf{V}^* = \mathbf{V}^* \iff \mathbf{V}^* = \mathbf{v}_\pi$$

$\mathbf{v}_\pi$??$T^\pi$???좎씪??怨좎젙??(Banach ?뺣━)?대?濡?$F$???좎씪???곸젏?낅땲??

#### Step 2: ?덉젙??議곌굔 ??$F$??Jacobian 遺꾩꽍

$$F(\mathbf{V}) = T^\pi \mathbf{V} - \mathbf{V} = (\mathbf{r}_\pi + \gamma \mathbf{P}_\pi \mathbf{V}) - \mathbf{V}$$

$F$??Jacobian:

$$J_F = \gamma \mathbf{P}_\pi - \mathbf{I}$$

???됰젹??怨좎쑀媛믪쓣 遺꾩꽍?⑸땲?? $\mathbf{P}_\pi$???뺣쪧 ?됰젹(stochastic matrix)?대?濡?紐⑤뱺 怨좎쑀媛?$\lambda_i$?????$|\lambda_i| \leq 1$.

$J_F$??怨좎쑀媛믪? $\gamma\lambda_i - 1$?대?濡?

$$\text{Re}(\gamma\lambda_i - 1) \leq \gamma|\lambda_i| - 1 \leq \gamma \cdot 1 - 1 = \gamma - 1 < 0 \quad (\gamma < 1)$$

紐⑤뱺 怨좎쑀媛믪쓽 ?ㅼ닔遺媛 ?뚯닔 ??$\mathbf{v}_\pi$??**?먭렐?곸쑝濡??덉젙???됲삎??*.

#### Step 3: ?몄씠利?議곌굔

$\varepsilon_t$媛 martingale difference?꾩쓣 ?뺤씤?⑸땲??

$$\mathbb{E}[\varepsilon_t \mid \mathcal{F}_t] = \mathbb{E}\left[R_{t+1} + \gamma \mathbf{V}_t(S_{t+1}) - \mathbf{V}_t(S_t) \mid \mathcal{F}_t\right] - F(\mathbf{V}_t)(S_t) = 0$$

(湲곕뙎媛??낅뜲?댄듃???뺤쓽???섑빐 ?뺥솗???곸뇙)

#### Step 4: Kushner-Clark 蹂댁“?뺣━ ?곸슜

????議곌굔 (怨좎젙??議댁옱, Jacobian ?덉젙?? martingale difference ?몄씠利?怨?Robbins-Monro step-size 議곌굔??紐⑤몢 留뚯”?섎㈃:

$$\mathbf{V}_t \xrightarrow{a.s.} \mathbf{v}_\pi \qquad \blacksquare$$

### ODE????곌껐 (吏곴?)

?뺣쪧??洹쇱궗 ?대줎???듭떖 ?듭같: step-size $\alpha_t \to 0$?대㈃ ?댁궛 ?낅뜲?댄듃???ㅼ쓬 ?곷?遺꾨갑?뺤떇(ODE)???곗냽 ?대줈 ?섎졃?⑸땲??

$$\dot{\mathbf{V}}(t) = F(\mathbf{V}(t)) = T^\pi \mathbf{V}(t) - \mathbf{V}(t)$$

??ODE???덉젙 ?됲삎?먯씠 $\mathbf{v}_\pi$?대?濡?(Step 2), ?뺣쪧???낅뜲?댄듃??$\mathbf{v}_\pi$濡??섎졃?⑸땲??

---

## 4. TD(貫)? n-step Return???깃???利앸챸

> **?꾩튂**: Chapter 7 ?ы솕 ??TD(貫)媛 紐⑤뱺 n-step return??貫-媛以??됯퇏?꾩쓣 利앸챸

### 貫-return ?뺤쓽

$$G_t^\lambda \doteq (1-\lambda)\sum_{n=1}^{\infty} \lambda^{n-1} G_{t:t+n}$$

### 利앸챸 1: 媛以묒튂???⑹씠 1?꾩쓣 ?뺤씤

$$\sum_{n=1}^{\infty} (1-\lambda)\lambda^{n-1} = (1-\lambda) \cdot \frac{1}{1-\lambda} = 1 \qquad (\lambda \in [0,1)) \checkmark$$

### 利앸챸 2: $\lambda = 0$?대㈃ TD(0)? ?숈씪

$$G_t^{\lambda=0} = (1-0)\sum_{n=1}^{\infty} 0^{n-1} G_{t:t+n}$$

$0^{n-1} = 0$ for $n > 1$, $0^0 = 1$ for $n = 1$?대?濡?

$$= 1 \cdot G_{t:t+1} = R_{t+1} + \gamma V(S_{t+1}) \qquad \checkmark$$

### 利앸챸 3: $\lambda = 1$?대㈃ MC return怨??숈씪 (?좏븳 ?먰뵾?뚮뱶)

?먰뵾?뚮뱶 湲몄씠媛 $T$?대㈃ $t + n \geq T$??紐⑤뱺 $n$?????$G_{t:t+n} = G_t$ (bootstrap ???놁쓬).

$G_t^\lambda$瑜???遺遺꾩쑝濡??섎닏?덈떎:

$$G_t^\lambda = (1-\lambda)\sum_{n=1}^{T-t-1} \lambda^{n-1} G_{t:t+n} + (1-\lambda)\sum_{n=T-t}^{\infty} \lambda^{n-1} G_t$$

??踰덉㎏ ??

$$(1-\lambda)\sum_{n=T-t}^{\infty} \lambda^{n-1} G_t = G_t \cdot (1-\lambda) \cdot \frac{\lambda^{T-t-1}}{1-\lambda} = G_t \cdot \lambda^{T-t-1}$$

?곕씪??

$$G_t^\lambda = (1-\lambda)\sum_{n=1}^{T-t-1} \lambda^{n-1} G_{t:t+n} + \lambda^{T-t-1} G_t$$

$\lambda \to 1$ 洹뱁븳?먯꽌:
- 泥??? $(1-\lambda) \to 0$??媛???쓣 ?뚮㈇
- ?섏㎏ ?? $\lambda^{T-t-1} \to 1^{T-t-1} = 1$

$$\lim_{\lambda \to 1} G_t^\lambda = G_t \qquad \checkmark$$

### 利앸챸 4: $G_t^\lambda$???ш? 怨듭떇 ?좊룄

?닿쾬???ㅼ젣 援ы쁽?먯꽌 ?듭떖?낅땲??

$$G_t^\lambda = (1-\lambda)\sum_{n=1}^{\infty} \lambda^{n-1} G_{t:t+n}$$

$G_{t:t+n}$???먰솕??$G_{t:t+n} = R_{t+1} + \gamma G_{t+1:t+n}$?쇰줈 遺꾪빐?⑸땲??

$$= (1-\lambda)\sum_{n=1}^{\infty} \lambda^{n-1} \left[R_{t+1} + \gamma G_{t+1:t+n}\right]$$

$$= (1-\lambda)R_{t+1}\sum_{n=1}^{\infty}\lambda^{n-1} + \gamma(1-\lambda)\sum_{n=1}^{\infty}\lambda^{n-1} G_{t+1:t+n}$$

$$= R_{t+1} + \gamma(1-\lambda)\sum_{n=1}^{\infty}\lambda^{n-1} G_{t+1:t+n}$$

$n \geq 2$?대㈃ $G_{t+1:t+n} = G_{t+1:(t+1)+(n-1)}$?대?濡??몃뜳?ㅻ? $m = n-1$濡?移섑솚?⑸땲??

$$= R_{t+1} + \gamma\left[(1-\lambda)G_{t+1:t+1} + (1-\lambda)\sum_{m=1}^{\infty}\lambda^m G_{t+1:(t+1)+m}\right]$$

$G_{t+1:t+1} = V(S_{t+1})$?닿퀬 ??踰덉㎏ ??? $\lambda \cdot G_{t+1}^\lambda$?대?濡?

$$\boxed{G_t^\lambda = R_{t+1} + \gamma\left[(1-\lambda)V(S_{t+1}) + \lambda G_{t+1}^\lambda\right]}$$

???ш? 怨듭떇??**eligibility traces**??backward view? ?숇벑?⑸땲??

### TD(貫) Update????곌껐

$$V(S_t) \leftarrow V(S_t) + \alpha\left[G_t^\lambda - V(S_t)\right]$$

?ш? 怨듭떇???댁슜?섎㈃ ?닿쾬??媛?step??TD error $\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$??湲고븯湲됱닔 ?⑹쑝濡??쒗쁽?????덉뒿?덈떎:

$$G_t^\lambda - V(S_t) = \sum_{k=t}^{T-1} (\gamma\lambda)^{k-t}\, \delta_k$$

**利앸챸**: $e_t = G_t^\lambda - V(S_t)$瑜??ш? 怨듭떇?쇰줈 ?꾧컻?⑸땲??

$$e_t = R_{t+1} + \gamma(1-\lambda)V(S_{t+1}) + \gamma\lambda G_{t+1}^\lambda - V(S_t)$$

$$= \underbrace{R_{t+1} + \gamma V(S_{t+1}) - V(S_t)}_{\delta_t} + \gamma\lambda(G_{t+1}^\lambda - V(S_{t+1}))$$

$$= \delta_t + \gamma\lambda\, e_{t+1}$$

???ш?瑜??硫?

$$e_t = \sum_{k=t}^{T-1} (\gamma\lambda)^{k-t} \delta_k \qquad \blacksquare$$

?닿쾬??**TD(貫)??forward view = backward view???깃???*?낅땲??
