---
title: "Reinforcement Learning: Chapter 7 Proofs: n-step"
category: "Reinforcement Learning"
weight: 17
date: 2026-04-09
---

# Chapter 7 Proofs: n-step

> Sutton & Barto ??援먯옱?먯꽌 ?앸왂???섑븰??利앸챸 ?곸꽭 ?뺣━

---
## 2. Q-learning ?섎졃 利앸챸 ??Lyapunov 諛⑸쾿

> **?꾩튂**: Chapter 6 ??Q-learning??$q_*$濡??섎졃?섎뒗 ?대줎??洹쇨굅 (Watkins & Dayan, 1992)

### ?뺣━

紐⑤뱺 $(s, a) \in \mathcal{S} \times \mathcal{A}$媛 臾댄븳??諛⑸Ц?섍퀬, step-size $\alpha_t$媛 Robbins-Monro 議곌굔??留뚯”?섎㈃:

$$Q_t(s, a) \xrightarrow{a.s.} q_*(s, a), \quad \forall s, a$$

### ?듭떖 ?꾧뎄: Lyapunov ?⑥닔

$L(Q) = \|Q - q_*\|_\infty$瑜?Lyapunov ?⑥닔濡??ъ슜?⑸땲??

?섎졃??蹂댁씠?ㅻ㈃ 湲곕뙎媛??낅뜲?댄듃 諛⑺뼢????긽 $L$??媛먯냼?쒗궡??蹂댁뿬???⑸땲??

### 利앸챸

#### Step 1: 湲곕뙎媛??낅뜲?댄듃 諛⑺뼢 遺꾩꽍

Q-learning??湲곕뙎媛??낅뜲?댄듃:

$$\bar{Q}(s,a) = (1-\alpha)\,Q(s,a) + \alpha \sum_{s',r} p(s',r|s,a)\left[r + \gamma\max_{a'} Q(s',a')\right]$$

$$= (1-\alpha)\,Q(s,a) + \alpha\,(T^* Q)(s,a)$$

#### Step 2: $\|\bar{Q} - q_*\|_\infty$ ?곹븳

$$|\bar{Q}(s,a) - q_*(s,a)| = |(1-\alpha)(Q(s,a) - q_*(s,a)) + \alpha((T^*Q)(s,a) - (T^*q_*)(s,a))|$$

(留덉?留?蹂?? $T^* q_* = q_*$ ?ъ슜)

?쇨컖 遺?깆떇:

$$\leq (1-\alpha)|Q(s,a) - q_*(s,a)| + \alpha|(T^*Q)(s,a) - (T^*q_*)(s,a)|$$

$T^*$??$\gamma$-contraction ?깆쭏 ($\|T^*Q - T^*q_*\|_\infty \leq \gamma\|Q - q_*\|_\infty$) ?곸슜:

$$\leq (1-\alpha)\|Q - q_*\|_\infty + \alpha\gamma\|Q - q_*\|_\infty$$

$$= (1 - \alpha(1-\gamma))\|Q - q_*\|_\infty$$

$s, a$?????supremum??痍⑦븯硫?

$$\|\bar{Q} - q_*\|_\infty \leq (1-\alpha(1-\gamma))\|Q - q_*\|_\infty$$

#### Step 3: ?섎졃 寃곕줎

$\alpha \in (0,1)$?닿퀬 $\gamma < 1$?대㈃ $\alpha(1-\gamma) > 0$?대?濡?

$$1 - \alpha(1-\gamma) < 1$$

湲곕뙎媛??낅뜲?댄듃留덈떎 $\|Q - q_*\|_\infty$媛 $(1-\alpha(1-\gamma))$ 鍮꾩쑉濡?**strictly 媛먯냼**?⑸땲??

Robbins-Monro 議곌굔怨??④퍡 ?뺣쪧??洹쇱궗 ?대줎???곸슜?섎㈃:

$$\|Q_t - q_*\|_\infty \xrightarrow{a.s.} 0 \qquad \blacksquare$$

### TD(0) ?섎졃怨쇱쓽 鍮꾧탳

| | TD(0) | Q-learning |
|---|---|---|
| ?섎졃 ???| $v_\pi$ (怨좎젙 policy) | $q_*$ (optimal) |
| ?듭떖 ?꾧뎄 | Jacobian ?덉젙??| $T^*$ contraction + Lyapunov |
| 議곌굔 | Robbins-Monro | Robbins-Monro + 紐⑤뱺 (s,a) 諛⑸Ц |

---

## 3. n-step Return ?ㅼ감 媛먯냼 ?깆쭏 利앸챸

> **?꾩튂**: Chapter 7 ??n-step return???ㅼ젣濡?$v_\pi$???섎졃?섎뒗 ?대줎??洹쇨굅

### ?뺣━

?꾩쓽??value function $V$?????

$$\max_s \left|\mathbb{E}_\pi\left[G_{t:t+n} \mid S_t = s\right] - v_\pi(s)\right| \leq \gamma^n \max_s |V(s) - v_\pi(s)|$$

### ?듭떖 蹂댁“ 寃곌낵

**Bellman equation??$n$???꾧컻**: $v_\pi$??Bellman equation??洹?⑹쟻?쇰줈 $n$踰??곸슜?섎㈃:

$$v_\pi(s) = \mathbb{E}_\pi\left[\sum_{k=0}^{n-1} \gamma^k R_{t+k+1} + \gamma^n v_\pi(S_{t+n}) \;\Big|\; S_t = s\right] \quad \cdots (*)$$

?닿쾬??癒쇱? 利앸챸?⑸땲??

**洹??利앸챸**:

$n=1$: Bellman equation???뺤쓽 ?먯껜?낅땲??

$n \to n+1$: $n$???꾧컻 寃곌낵 $(*)$?먯꽌 $\gamma^n v_\pi(S_{t+n})$ 遺遺꾩뿉 Bellman equation????踰????곸슜?⑸땲??

$$\gamma^n v_\pi(S_{t+n}) = \gamma^n \mathbb{E}_\pi[R_{t+n+1} + \gamma v_\pi(S_{t+n+1}) \mid S_{t+n}]$$

湲곕뙎媛믪쓽 ???깆쭏(tower property)???섑빐:

$$\mathbb{E}_\pi[\gamma^n v_\pi(S_{t+n}) \mid S_t=s] = \mathbb{E}_\pi[\gamma^n R_{t+n+1} + \gamma^{n+1} v_\pi(S_{t+n+1}) \mid S_t=s]$$

$(*)$????낇븯硫?$n+1$???꾧컻 怨듭떇???깅┰?⑸땲?? $\checkmark$

### 蹂?利앸챸

n-step return???뺤쓽:

$$G_{t:t+n} = \sum_{k=0}^{n-1} \gamma^k R_{t+k+1} + \gamma^n V(S_{t+n})$$

湲곕뙎媛믪쓣 痍⑦븯硫?

$$\mathbb{E}_\pi[G_{t:t+n} \mid S_t=s] = \mathbb{E}_\pi\left[\sum_{k=0}^{n-1} \gamma^k R_{t+k+1} + \gamma^n V(S_{t+n}) \;\Big|\; S_t=s\right]$$

蹂댁“ 寃곌낵 $(*)$???李⑥씠:

$$\mathbb{E}_\pi[G_{t:t+n} \mid S_t=s] - v_\pi(s) = \mathbb{E}_\pi\left[\gamma^n (V(S_{t+n}) - v_\pi(S_{t+n})) \mid S_t=s\right]$$

(???앹뿉??$\sum \gamma^k R_{t+k+1}$ ??씠 ?곸뇙??

?덈뙎媛믨낵 湲곕뙎媛믪쓽 Jensen 遺?깆떇 ($|\mathbb{E}[X]| \leq \mathbb{E}[|X|]$) ?곸슜:

$$\left|\mathbb{E}_\pi[G_{t:t+n} \mid S_t=s] - v_\pi(s)\right| \leq \gamma^n \mathbb{E}_\pi\left[|V(S_{t+n}) - v_\pi(S_{t+n})| \mid S_t=s\right]$$

$$\leq \gamma^n \max_{s'} |V(s') - v_\pi(s')|$$

$s$?????supremum??痍⑦븯硫?

$$\max_s \left|\mathbb{E}_\pi[G_{t:t+n} \mid S_t=s] - v_\pi(s)\right| \leq \gamma^n \|V - v_\pi\|_\infty \qquad \blacksquare$$

### ?댁꽍

??遺?깆떇??留먰븯??寃?

- $n$???댁닔濡?bootstrap ?ㅼ감媛 $\gamma^n$?쇰줈 鍮좊Ⅴ寃?媛먯냼
- $V$媛 $v_\pi$? 硫?섎줉 ($\|V - v_\pi\|_\infty$媛 ?댁닔濡? ?ㅼ감??鍮꾨??댁꽌 而ㅼ쭚
- $\gamma = 0.9$, $n = 10$?대㈃ ?ㅼ감 ?곹븳??$0.9^{10} \approx 0.349$諛곕줈 媛먯냼
- ?닿쾬??n-step TD媛 TD(0)蹂대떎 醫뗭? ?댁쑀: ???뺥솗??target ?ъ슜

---
