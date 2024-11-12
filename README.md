# 더 지니어스: 흑과백 with AI 🎲

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)]()
[![Pygame](https://img.shields.io/badge/Pygame-2.0.1-green)]()

*더 지니어스* 시리즈의 전략 게임 "흑과백"을 파이썬으로 구현한 프로젝트입니다. 강화학습으로 훈련된 AI와 대결하며 당신의 전략을 시험해보세요!

## 📌 목차
- [소개](#소개)
- [주요 기능](#주요-기능)
- [게임 규칙](#게임-규칙)
- [설치 방법](#설치-방법)
- [실행 방법](#실행-방법)
- [AI 모델](#ai-모델)
- [기술 스택](#기술-스택)
- [향후 계획](#향후-계획)

## 🌟 소개
**더 지니어스: 흑과백 with AI**는 tvN의 인기 예능 프로그램 '더 지니어스'의 대표적인 게임인 '흑과백'을 강화학습을 접목시켜 구현한 프로젝트입니다. Pygame으로 구현된 게임 인터페이스와 강화학습 기반의 AI를 통해 전략적이고 몰입도 높은 게임 경험을 제공합니다.

## ✨ 주요 기능
- 🤖 강화학습 기반의 지능형 AI 플레이어
- 🎮 직관적인 게임 인터페이스
- 🎯 게임 내 재시작 기능

## 🎯 게임 규칙

1. **기본 설정**
   - 각 플레이어는 0-9까지의 숫자 카드 보유
   - 흑(짝수)와 백(홀수)로 구분

2. **게임 진행**
   - 매 라운드 각자 카드 1장 선택
   - 이전 라운드의 승자가 먼저 선택, 카드의 색을 공개
   - 선택한 카드는 재사용 불가
   - 총 9라운드 진행

3. **점수 계산**
   - 내가 선택한 카드가 더 큰 값이면, 점수를 1 얻습니다.


## ⚙️ 설치 방법

1. 저장소 클론
```bash
git clone https://github.com/username/genius-black-and-white.git
cd genius-black-and-white
```

2. 가상환경 설정
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. 패키지 설치
```bash
pip install -r requirements.txt
```

## 🎮 실행 방법

1. 게임 실행
```bash
python main.py
```

2. 게임 모드 선택

   - AI 대전 모드
   - 향후 추가 예정
 

## 🤖 AI 모델

- **알고리즘**: Deep Q-Network (DQN)
- **학습 데이터**: 10,0000+ 게임 플레이
- **특징**:
  - 
  - 
  - 

## 🛠 기술 스택

- **게임 엔진**: Pygame 2.0.1
- **AI 프레임워크**: Torch 2.4.1
- **데이터 처리**: NumPy, Pandas
- **개발 환경**: Python 3.11+

## 🚀 향후 계획

- [ ] AI 난이도 다양화
- [ ] 게임 UI/UX 개선