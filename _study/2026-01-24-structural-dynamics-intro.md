---
layout: post
title: "Structural Dynamics Intro"
category: Structural Dynamics
date: 2026-01-24
---

구조동역학의 기본 방정식은 다음과 같이 정리할 수 있습니다.

$$
M \ddot{x}(t) + C \dot{x}(t) + K x(t) = f(t)
$$

모드 해석을 위해 고유치 문제를 고려하면,

$$
\left( K - \omega^2 M \right) \phi = 0
$$

행렬을 포함한 예시는 다음과 같습니다.

$$
\begin{bmatrix}
2 & -1 \\
-1 & 2
\end{bmatrix}
\begin{bmatrix}
\phi_1 \\
\phi_2
\end{bmatrix}
=
\lambda
\begin{bmatrix}
1 & 0 \\
0 & 1
\end{bmatrix}
\begin{bmatrix}
\phi_1 \\
\phi_2
\end{bmatrix}
$$

Python으로 간단히 고유치 계산을 수행할 수 있습니다.

```python
import numpy as np

K = np.array([[2.0, -1.0], [-1.0, 2.0]])
M = np.eye(2)
eigvals, eigvecs = np.linalg.eig(np.linalg.inv(M) @ K)
print(eigvals)
```
