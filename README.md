# 🎲 더 지니어스 : 흑과백 with AI 🎲

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)]()
[![Pygame](https://img.shields.io/badge/Pygame-2.6.0-green)]()

*더 지니어스* 시리즈의 데스매치 게임 "흑과백"을 파이썬으로 구현한 프로젝트입니다. 강화학습으로 훈련된 AI와 대결하며 당신의 전략을 시험해보세요!

## 📌 목차
- [소개](#소개)
- [데모 비디오](#데모-비디오)
- [주요 기능](#주요-기능)
- [게임 규칙](#게임-규칙)
- [실행 방법](#실행-방법)
- [AI 모델](#ai-모델)
- [기술 스택](#기술-스택)
- [향후 계획](#향후-계획)

## 🌟 소개
**더 지니어스: 흑과백 with AI**는 tvN의 인기 예능 프로그램 '더 지니어스'의 데스매치 게임인 '흑과백'을 강화학습을 접목시켜 구현한 프로젝트입니다. Pygame으로 구현된 게임 인터페이스와 강화학습 기반의 AI를 통해 전략적이고 몰입도 높은 게임 경험을 제공합니다.

## 📺 데모 비디오

[![Watch the video on youtube](https://github.com/kstone-285/oss_project/blob/main/bnw.png)](https://youtu.be/-MmGbWzX8nk)

## ✨ 주요 기능
- 🤖 **강화학습** 기반의 지능형 AI 플레이어
- 👯 AI와 상대하지 않고 두 명이서도 즐길 수 있는 **2P모드**
- 🎮 직관적인 게임 인터페이스와 실제 더 지니어스를 플레이하는 듯한 몰입감
- 🕰 게임 내 **모드 선택 및 재시작** 기능

## 🎯 게임 규칙

1. **기본 설정**
   - 각 플레이어는 **0-8** 까지의 숫자 타일을 보유합니다.
   - 보유한 타일은 그 숫자에따라 **흑(짝수타일)과 백(홀수타일)**로 구분됩니다.
   - 게임 시작 전, 자신의 타일 배치를 마음대로 변경할 수 있습니다.
   - 당연하게도, **역심리**를 이용해 배치를 안하고 상대하셔도 좋습니다.

2. **게임 진행**
   - 기본적으로 매 라운드 각자 제시할 타일을 1개 선택합니다.
   - 이전 라운드의 승자가 먼저 타일을 선택하고, 상대 플레이어는 제시한 **타일의 색깔**만 알 수 있습니다.
   - 선택한 타일은 라운드가 끝난 후 소멸됩니다.
   - 총 9라운드로 진행하고, 5점을 먼저 선취하거나, 9라운드 이후 점수가 더 높은 플레이어가 승리합니다.

3. **점수 계산**
   - 내가 제시한 타일이 더 큰 값이면, 점수를 1 얻습니다.

4. **제작자의 팁**
   - **질 때는 큰 차이**로, **이길 때는 작은 차이**로 이겨야 합니다!
   

## 🎮 실행 방법

1. blacknwhite.py 파일 다운로드 후 실행

   - AI 대전 모드
   - 2P 대전 모드
   - 둘 중 원하는 모드를 선택하여 플레이하시면 됩니다.
 

## 🤖 강화학습에 사용된 AI 모델과 고찰, 후기

- **알고리즘**: **Deep Q-Network (DQN), Actor-critic (A2C), Proximal Policy Optimization (PPO)**를 사용했습니다.
- **학습 데이터**: 각 모델마다 **50,000+** 에피소드를 플레이하여 학습했습니다.
- **각 AI 모델의 특징**:
  - ***DQN***
    - **DQN**은 가장 처음 적용해본 모델로, 가장 접근성이 좋고 배우기 쉬운 모델이었기에 선택했습니다.<br>
    그러나 DQN은 흑과백과 같은 **확률기반, 장기전략**이 중요한 게임에서는 적합하지 않았습니다.<br>
    해당 이유는 크게 3가지 입니다.  

      1. DQN은 한 번의 행동에 따른 **즉각적인 보상을 기반으로 학습**하는 경향이 있었습니다. <br><br>
         이런 류의 게임은 개별 행동에 대한 리워드보다는 전체 전략, 상대방보다 얼마나 작은 차이로 이기고 큰차이로 지는지에 대한 보상이 중요합니다.<br><br>
      2. **확률적인 요소**와 **심리전 요소**의 **미반영**이 있었습니다.<br><br>
         매 라운드, 매 에피소드마다 상대방의 행동은 확률적으로 변화하고, 심리패턴이나 블러핑에 대한 반응 또한 존재해야 합니다.  
         DQN은 이를 확률적으로 학습하지 않고, 단순하게 **평균적인 패턴에 최적화**되는 경향이 있었습니다. 추가로, 첫 AI다 보니 상대방의 패를 항상 random으로 돌려 더욱 제대로 학습하지 못했습니다.<br><br>
      3. **에피소드 기반 최적화**를 진행하기 때문입니다.<br><br>
         DQN은 일반적으로 **마르코프 결정 과정**을 기반으로 설계되었고, 이는 게임의 전체적인 맥락을 파악하지 못하며, 해당 에피소드 내에서만 학습하는 결과를 낳습니다.
         이로인해 고정된 패턴에 갇혀 학습을 진행한 결과가 일련의 **선택순서**였고, 이는 AI 모델에 적합하지 않다 판단하여 폐기했습니다.<br><br>

  - ***A2C***
    - **A2C**는 **정책 기반 학습**과 **가치 기반 학습**을 결합한 알고리즘으로, DQN보다는 확률을 사용하기에 해당 게임에 더 적합하다고 판단해 적용했습니다.  
      이 알고리즘은 특히 **확률적 요소**가 많고 **장기적인 전략**이 중요한 게임에서 더욱 효과적이었습니다.  
      당연히 DQN보다는 성능이 좋았지만, 게임에 적용했을 때의 **장점**과 **한계**는 다음과 같습니다.  

      - **장점:**
         1. **장기적인 전략 학습**  <br>
            A2C는 **정책 네트워크**와 **가치 네트워크**를 동시에 학습하여, **장기적인 보상**을 고려한 전략을 세울 수 있습니다.  
            게임과 같은 **확률적, 전략적 요소**를 고려하는 데 있어 유리하며, 한 번의 행동에 대한 보상뿐만 아니라 **전체 게임의 승리**를 목표로 학습할 수 있습니다.  
            이로 인해 **개별 행동**이 아닌 **전체적인 전략**을 학습하는 데 유리한 특성을 가지고 있습니다.  

         2. **정책 기반 학습의 장점**  <br>
            A2C는 **정책 기반 학습**을 사용하여 **지속적인 정책 개선**을 가능하게 합니다.  
            특히 **탐험(exploration)**과 **활용(exploitation)**을 적절하게 사용하며 학습할 수 있기 때문에, **확률적**이고 **다양한 패턴**을 가진 대결에서 유리한 결과를 얻을 수 있었습니다.  
            예를 들자면, 상대방이 다양한 방식으로 **블러핑**을 할 때, 이런 변수를 잘 캐치하며 학습할 수 있었습니다.  

      - **한계:**
         1. **정책 업데이트의 불안정성**  <br>
            A2C는 **정책**과 **가치**를 업데이트하는 방식에서 **변동성**이 있을 수 있습니다.  
            때때로 **정책 네트워크**가 너무 급격하게 업데이트되어 **불안정한 학습**을 초래할 수 있습니다.  
            특히 **흑과백** 게임에서처럼, **전략의 미세한 차이**가 승패를 가를 때, 정책 업데이트의 불안정성으로 인해 **일관된 성능**을 보장하기 어려웠습니다.  

         2. **탐험과 활용의 균형 문제**  <br>
            A2C는 **탐험과 활용의 균형**을 잘 맞추는 것이 중요합니다.  
            게임에서 일정한 패턴에 머물러 있다면 **탐험 부족**으로 인해 **새로운 전략**을 학습하는 데 어려움을 겪을 수 있습니다.  
            **흑과백** 게임에서는 상대방의 전략을 예측하는 것이 중요한데, **탐험 부족**은 이를 방해할 수 있습니다.  

         3. **복잡한 하이퍼파라미터 튜닝**  <br>
            A2C는 **하이퍼파라미터**에 민감한 알고리즘입니다. **학습률**, **탐험 비율**, **배치 크기** 등 다양한 파라미터들이 **모델의 성능에 큰 영향을 미칩니다.**  
            **흑과백**과 같은 게임에서 이를 잘 조정하지 않으면 **훈련이 제대로 되지 않거나 성능이 떨어지는 결과**를 초래할 수 있었습니다.
  - ***PPO***
    - 

## 🛠 기술 스택

- **게임 엔진**: Pygame 2.6.0
- **AI 프레임워크**: Torch 2.4.1
- **데이터 처리**: NumPy, Torch, ....
- **개발 환경**: Python 3.11+

## 🚀 향후 계획

- [ ] AI 난이도 다양화
- [ ] 게임 UI/UX 개선
