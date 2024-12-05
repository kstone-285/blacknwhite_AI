import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random

class ActorCritic(nn.Module):

    def __init__(self, state_size, action_size):

        super(ActorCritic, self).__init__()
        # 공통 특성 추출 레이어
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        
        # Actor network (정책)
        self.actor = nn.Linear(64, action_size)
        
        # Critic network (가치)
        self.critic = nn.Linear(64, 1)
    
    def forward(self, state):

        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        # Actor: 유효한 액션에 대한 확률 분포 출력
        action_probs = F.softmax(self.actor(x), dim=-1)
        
        # Critic: 상태 가치 추정
        state_value = self.critic(x)
        
        return action_probs, state_value

class PolicyGradientAgent:

    def __init__(self, state_size, action_size):

        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = ActorCritic(state_size, action_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        self.gamma = 0.99
        self.eps = 1e-8
        
        # 경험 저장
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []  # action_probs 대신 log_probs 사용
        self.values = []

    def act(self, state, valid_actions):

        # 입력 상태를 텐서로 변환
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)
        state = state.to(self.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)

        self.model.eval()
        with torch.no_grad():

            action_probs, state_value = self.model(state)
            
            # 유효하지 않은 액션의 확률을 0으로 만들어줌
            action_probs = action_probs.squeeze()
            mask = torch.zeros_like(action_probs)
            for action in valid_actions:
                mask[action] = 1
            masked_probs = action_probs * mask
            
            # 확률 정규화
            masked_probs = masked_probs / (masked_probs.sum() + self.eps)
            
            # 확률적 선택
            try:
                action_dist = torch.distributions.Categorical(masked_probs)
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action)
            except:
                # 문제 발생시 첫 번째 유효한 액션 선택
                action = torch.tensor(valid_actions[0])
                log_prob = torch.log(masked_probs[action] + self.eps)

        # 경험 저장
        self.states.append(state)
        self.actions.append(action)
        self.values.append(state_value)
        self.log_probs.append(log_prob)  # log_prob 저장
        
        return action.item()

    def process_episode(self, final_reward):

        self.model.train()
        
        # 리턴 계산
        returns = []
        R = final_reward
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        
        # 텐서로 변환
        returns = torch.FloatTensor(returns).to(self.device)
        
        states_tensor = torch.cat(self.states)
        action_probs, state_values = self.model(states_tensor)
        advantages = returns - state_values.squeeze()
        
        # loss 계산
        log_probs_tensor = torch.stack(self.log_probs)
        policy_loss = -(log_probs_tensor * advantages.detach()).mean()
        value_loss = F.mse_loss(state_values.squeeze(), returns)
        loss = policy_loss + 0.5 * value_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def end_episode(self, final_reward):

        try:
            self.process_episode(final_reward)
        except Exception as e:
            print(f"Error in end_episode: {str(e)}")
        finally:
            self.clear_memory()

    def clear_memory(self):

        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []

    def add_reward(self, reward):
        
        self.rewards.append(reward)