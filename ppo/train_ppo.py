import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math
from environment import BlackWhiteEnv

def get_valid_actions(env, player_tiles):
    
    # 사용 가능한 타일 반환
    valid_actions = [tile for tile, used in player_tiles if not used]
    return valid_actions

class PolicyNetwork(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=1)

class ValueNetwork(nn.Module):

    def __init__(self, state_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class PPOAgent:

    def __init__(self, state_dim, action_dim, learning_rate=3e-4, 
                 gamma=0.99, epsilon=0.2, epochs=3, entropy_coef=0.01):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 정책 및 가치 네트워크 초기화
        self.policy_net = PolicyNetwork(state_dim, action_dim).to(self.device)
        self.value_net = ValueNetwork(state_dim).to(self.device)
        
        # 옵티마이저 및 학습률 스케줄러 설정
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=learning_rate)
        self.lr_scheduler_policy = optim.lr_scheduler.ExponentialLR(
            self.policy_optimizer, gamma=0.99
        )
        self.lr_scheduler_value = optim.lr_scheduler.ExponentialLR(
            self.value_optimizer, gamma=0.99
        )
        
        # 하이퍼파라미터
        self.gamma = gamma
        self.epsilon = epsilon
        self.epochs = epochs
        self.entropy_coef = entropy_coef
        
        # 메모리 초기화
        self.memory = []
    
    def select_action(self, state, valid_actions):
        """주어진 상태에서 유효한 행동 중 하나를 선택"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_probs = self.policy_net(state)
        
        # 유효한 행동에 대해 확률 마스킹
        action_probs = action_probs.cpu().detach().numpy()[0]
        
        # 유효한 행동들의 확률만 추출 및 재정규화
        valid_action_probs = action_probs[valid_actions]
        
        if valid_action_probs.sum() > 0:
            valid_action_probs = valid_action_probs / valid_action_probs.sum()
            action_in_valid_subset = np.random.choice(len(valid_action_probs), p=valid_action_probs)
            action = valid_actions[action_in_valid_subset]
        else:
            # 모든 유효 행동의 확률이 0이라면 균등 분포로 선택
            action = np.random.choice(valid_actions)
        
        return action
    
    def store_transition(self, state, action, reward, next_state, done):
        """에이전트의 메모리에 트랜지션 저장"""
        self.memory.append((state, action, reward, next_state, done))
    
    def update(self):
        """PPO 알고리즘을 사용해 에이전트 업데이트"""
        if len(self.memory) == 0:
            return
        
        states = torch.FloatTensor([m[0] for m in self.memory]).to(self.device)
        actions = torch.LongTensor([m[1] for m in self.memory]).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor([m[2] for m in self.memory]).to(self.device)
        next_states = torch.FloatTensor([m[3] for m in self.memory]).to(self.device)
        dones = torch.FloatTensor([m[4] for m in self.memory]).to(self.device)
        
        returns = self.compute_returns(rewards, dones)
        values = self.value_net(states).squeeze()
        advantages = returns - values.detach()

        for _ in range(self.epochs):
            action_probs = self.policy_net(states)
            
            # 새로운 정책 확률 계산
            selected_probs = action_probs.gather(1, actions)
            new_log_probs = torch.log(selected_probs + 1e-8).squeeze()
            
            # 이전 정책 확률 계산
            old_probs = selected_probs.detach()
            old_log_probs = torch.log(old_probs + 1e-8).squeeze()
            
            # 확률 비율
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # Surrogate 손실 계산
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # 엔트로피 정규화
            entropy = -(action_probs * torch.log(action_probs + 1e-8)).mean()
            policy_loss -= self.entropy_coef * entropy
            
            # 가치 손실 계산
            value = self.value_net(states).squeeze()
            value_loss = F.mse_loss(value, returns)
            
            # 네트워크 업데이트
            self.policy_optimizer.zero_grad()
            policy_loss.backward()  
            self.policy_optimizer.step()
            
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()
            
            # 학습률 감쇠
            self.lr_scheduler_policy.step()
            self.lr_scheduler_value.step()
        
        self.memory.clear()
    
    def compute_returns(self, rewards, dones):
        """할인된 반환값 계산"""
        returns = []
        R = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            R = reward + self.gamma * R * (1.0 - done)
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        return returns

class SelfPlayTrainer:
    def __init__(self, env_class):
        sample_env = env_class()
        self.state_dim = len(sample_env._get_state())
        self.action_dim = 9  # 타일 번호 0~8
        self.agent1 = PPOAgent(self.state_dim, self.action_dim)
        self.agent2 = PPOAgent(self.state_dim, self.action_dim)
        self.env_class = env_class
        
        # 최적 모델 추적을 위한 변수들
        self.best_model1 = None
        self.best_model2 = None
        self.best_win_rate = 0.5

    def calculate_rewards(self, env, reward):
        """대칭적이고 공정한 보상 계산"""
        tile_diff = abs(env.player1_last_tile - env.player2_last_tile)
        max_reward = 2.7
        decay_rate = 0.22
        
        if reward > 0:  # 플레이어 2 승리
            winner_reward = max_reward - (tile_diff * decay_rate)
            loser_reward = -winner_reward
        elif reward < 0:  # 플레이어 1 승리
            winner_reward = max_reward - (tile_diff * decay_rate)
            loser_reward = -winner_reward
        else:  # 무승부
            winner_reward = loser_reward = 0
        
        return winner_reward, loser_reward

    def train(self, num_episodes=20000):
        """에이전트들 간의 자가대전 학습"""
        episode_rewards1 = []
        episode_rewards2 = []
        player1_wins = 0
        player2_wins = 0
        total_games = num_episodes
        
        initial_epsilon = 0.2
        min_epsilon = 0.05

        for episode in range(1, num_episodes + 1):
            # 적응적 엡실론 감소
            epsilon = max(
                min_epsilon, 
                initial_epsilon * (1 - episode / num_episodes)
            )
            self.agent1.epsilon = epsilon
            self.agent2.epsilon = epsilon

            env = self.env_class()
            state = env.reset()
            episode_reward1 = 0
            episode_reward2 = 0
            done = False
            
            while not done:
                # Player 1's turn
                valid_actions1 = get_valid_actions(env, env.player1_tiles)
                if not valid_actions1:
                    break
                action1 = self.agent1.select_action(state, valid_actions1)
                
                # Player 2's turn
                valid_actions2 = get_valid_actions(env, env.player2_tiles)
                if not valid_actions2:
                    break
                action2 = self.agent2.select_action(state, valid_actions2)
                
                # Environment step
                next_state, reward, done = env.step(action2, action1)
                
                # 대칭적 보상 계산
                round_reward1, round_reward2 = self.calculate_rewards(env, reward)
                
                self.agent1.store_transition(state, action1, round_reward1, next_state, done)
                self.agent2.store_transition(state, action2, round_reward2, next_state, done)
                
                episode_reward1 += round_reward1
                episode_reward2 += round_reward2
                
                state = next_state
            
            # 게임 결과에 따른 최종 보상
            if env.player1_score >= 5:
                player1_wins += 1
                final_reward1 = 10.0 * (env.player1_score - env.player2_score)
                final_reward2 = -final_reward1
            elif env.player2_score >= 5:
                player2_wins += 1
                final_reward2 = 10.0 * (env.player2_score - env.player1_score)
                final_reward1 = -final_reward2
            else:
                final_reward1, final_reward2 = 0, 0
            
            episode_reward1 += final_reward1
            episode_reward2 += final_reward2            

            episode_rewards1.append(episode_reward1)
            episode_rewards2.append(episode_reward2)
            
            # 에이전트 업데이트
            self.agent1.update()
            self.agent2.update()
            
            # 주기적 모델 평가 및 최적 모델 추적
            if episode % 500 == 0:
                current_win_rate1 = player1_wins / episode
                current_win_rate2 = player2_wins / episode
                
                if current_win_rate1 > self.best_win_rate:
                    self.best_model1 = self.agent1.policy_net.state_dict()
                    self.best_win_rate = current_win_rate1
                
                if current_win_rate2 > self.best_win_rate:
                    self.best_model2 = self.agent2.policy_net.state_dict()
                    self.best_win_rate = current_win_rate2
                
                # 성능 과도하게 저하 시 최적 모델로 리셋
                if current_win_rate1 < 0.2:
                    self.agent1.policy_net.load_state_dict(self.best_model2)
                if current_win_rate2 < 0.2:
                    self.agent2.policy_net.load_state_dict(self.best_model1)
            
            # 주기적 로깅
            if episode % 100 == 0:
                avg_reward1 = np.mean(episode_rewards1[-100:])
                avg_reward2 = np.mean(episode_rewards2[-100:])
                win_rate1 = player1_wins / episode
                win_rate2 = player2_wins / episode
                print(f"Episode: {episode}/{num_episodes}")
                print(f"Avg Rewards (Last 100): Agent1 = {avg_reward1:.2f}, Agent2 = {avg_reward2:.2f}")
                print(f"Win Rate: Agent1 = {win_rate1:.2%}, Agent2 = {win_rate2:.2%}")
                print("------------------------")
        
        # 학습 종료 후 전체 통계 출력
        avg_reward1_total = np.mean(episode_rewards1)
        avg_reward2_total = np.mean(episode_rewards2)
        win_rate1_total = player1_wins / total_games
        win_rate2_total = player2_wins / total_games
        
        print("\n--- Training Completed ---")
        print(f"Total Episodes: {num_episodes}")
        print(f"Overall Avg Rewards: Agent1 = {avg_reward1_total:.2f}, Agent2 = {avg_reward2_total:.2f}")
        print(f"Overall Win Rate: Agent1 = {win_rate1_total:.2%}, Agent2 = {win_rate2_total:.2%}")
        
        # 모델 저장
        self.save_models(num_episodes)
    
    def save_models(self, episode):

        try:
            torch.save(self.agent1.policy_net.state_dict(), 'agent1_policy_ppo.pth')
            torch.save(self.agent2.policy_net.state_dict(), f'agent2_policy_ppo.pth')
            print("Models saved successfully")
        except Exception as e:
            print(f"Error saving model: {str(e)}")

if __name__ == "__main__":
    
    trainer = SelfPlayTrainer(BlackWhiteEnv)
    trainer.train(num_episodes=20000)