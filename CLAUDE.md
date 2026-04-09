# Claude Code — Blog Contribution Guide

이 파일은 Claude Code가 세션 시작 시 자동으로 읽는 가이드입니다.
이 블로그에 콘텐츠를 추가하거나 수정할 때 반드시 아래 규칙을 따르세요.

---

## 블로그 기본 정보

- **빌드 시스템**: Hugo v0.160+ (Extended) — Jekyll이 아님
- **URL**: https://sseung00.github.io
- **언어**: 한국어(기본) / 영어 (다국어 지원)
- **콘텐츠 위치**: `content/ko/` (한국어), `content/en/` (영어)
- **배포**: `git push origin main` → GitHub Actions가 자동 빌드·배포

---

## 콘텐츠 파일 위치

```
content/ko/
├── study/
│   ├── reinforcement-learning/   ← RL 챕터 노트 (weight로 순서 지정)
│   └── structural-dynamics/      ← 구조동역학 노트
├── hobby/                        ← 취미 글
├── journal/                      ← 일기·기록
└── portfolio.md                  ← 포트폴리오 (레이아웃 수정은 layouts/portfolio/single.html)
```

---

## Front Matter 규칙

```yaml
---
title: "제목"
date: 2026-04-10
category: "Reinforcement Learning"   # Study 글의 카테고리 필터에 사용
weight: 9                             # RL 챕터 순서 지정 시 (숫자 클수록 뒤)
---
```

- `layout:` 필드는 **쓰지 않습니다** (Hugo는 layouts/ 폴더로 자동 결정)
- `section:` 필드는 일반 글에 **불필요** (폴더 구조로 자동 결정)

---

## Markdown 작성 규칙

### ❗ Rule 1 — Bold(`**`) 뒤에 한글 조사가 오는 경우

Hugo(Goldmark)는 CommonMark를 엄격하게 구현합니다.
닫는 `**` 뒤에 바로 한글 조사(`을/를/이/가/은/는/의/로/와/과` 등)가 오면
bold가 렌더링되지 않고 `**텍스트**를` 그대로 화면에 출력됩니다.

**해결책: 닫는 `**` 뒤에 공백 하나를 추가하세요.**

```markdown
# ❌ 깨짐
**궤적(trajectory)**을 형성합니다
**상태 $S_t$**를 관측하고
**MDP**의 핵심 특징

# ✅ 정상 (** 뒤에 공백)
**궤적(trajectory)** 을 형성합니다
**상태 $S_t$** 를 관측하고
**MDP** 의 핵심 특징
```

> **원리**: CommonMark 스펙상 닫는 `**`의 직전이 `)`, `$` 같은 ASCII 구두점이고
> 직후가 한글이면 "우측 경계 구분자"로 인정되지 않아 bold 처리가 무시됩니다.
> 공백을 넣으면 이 규칙을 우회합니다.

---

### ❗ Rule 2 — 수식(`$...$`, `$$...$$`) 안에서 특수문자

Hugo는 `$...$`, `$$...$$` 내부를 MathJax 전에 먼저 처리하지 **않도록**
passthrough extension이 설정되어 있습니다. 따라서 수식은 기존과 동일하게
LaTeX 문법 그대로 작성하면 됩니다. 별도 이스케이프 불필요.

```markdown
# ✅ 그대로 써도 됨
$$\;\Big|\; S_t = s$$
$\mathbb{E}_\pi[G_t \mid S_t = s]$
$$\boxed{v_\pi(s) \doteq \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1}\right]}$$
```

---

### Rule 3 — 이미지 경로

```markdown
![설명](/images/파일명.png)   ← assets/images/ 에 파일을 두고 /images/로 참조
```

---

### Rule 4 — 내부 링크

```markdown
[Study](/study/)              ← 한국어 기본 경로
[Study](/en/study/)           ← 영어 경로 명시 시
```

---

## 외부에서 받은 Markdown 파일 수정 방법

ChatGPT, Gemini 등 다른 AI나 외부 소스에서 받은 마크다운 파일을
이 블로그에 추가할 때는 `scripts/fix-markdown.ps1` 스크립트를 실행하세요.

```powershell
# 단일 파일 수정
.\scripts\fix-markdown.ps1 -File "content/ko/study/새파일.md"

# 폴더 내 전체 파일 수정
.\scripts\fix-markdown.ps1 -Dir "content/ko/study/reinforcement-learning"
```

스크립트가 자동으로 처리하는 것:
1. `layout:` front matter 제거
2. `**단어**조사` 패턴에 ZWS 삽입 (Bold 깨짐 방지)
3. 결과 요약 출력

---

## 새 글 추가 워크플로우

```bash
# 1. 파일 작성
content/ko/study/새파일.md

# 2. (외부에서 받은 파일이면) 스크립트 실행
.\scripts\fix-markdown.ps1 -File "content/ko/study/새파일.md"

# 3. 로컬 확인 (선택)
hugo server -D

# 4. 배포
git add content/ko/study/새파일.md
git commit -m "Add: 새 글 제목"
git push origin main
```

---

## 절대 하지 말 것

- `_study/`, `_hobby/`, `_journal/` 폴더에 파일 추가 (Jekyll 잔재, 무시됨)
- `index.html`, `study.html` 등 루트 HTML 파일 수정 (Hugo 레이아웃 사용)
- `public/` 폴더 직접 수정 (Hugo 빌드 자동 생성)
- front matter에 `layout:` 추가
