import torch
import numpy as np
from a2c.policy_network import ActorCritic
from dqn.dqn_agent import DQN
from ppo.train_ppo import PPOAgent, PolicyNetwork


class PolicyAIPlayer:

    def __init__(self, model_path):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ActorCritic(21, 9).to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
    
    def get_action(self, state, valid_actions):

        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_probs, _ = self.model(state)
            
        # 유효한 액션만 고려
        action_probs = action_probs.squeeze()
        mask = torch.zeros_like(action_probs)
        for action in valid_actions:
            mask[action] = 1
        masked_probs = action_probs * mask
        
        # 확률 정규화
        masked_probs = masked_probs / (masked_probs.sum() + 1e-8)
        
        # 확률적 선택
        return torch.multinomial(masked_probs, 1).item()
    
class AIPlayer:

    def __init__(self, model_path):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(21, 9).to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
    def get_action(self, state, valid_actions):

        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_values = self.model(state)
            
        valid_action_values = action_values.cpu().numpy()[0]
        valid_action_values = {action: valid_action_values[action] for action in valid_actions}
        return max(valid_action_values, key=valid_action_values.get)
    
class PPOAIPlayer:

    def __init__(self, model_path):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = PolicyNetwork(21, 9).to(self.device)
        self.policy_net.load_state_dict(torch.load(model_path, map_location=self.device))
        self.policy_net.eval()
        
    def get_action(self, state, valid_actions):

        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_probs = self.policy_net(state).squeeze().cpu().numpy()
        
        # 유효한 행동들에 대해서만 확률을 필터링
        valid_action_probs = {action: action_probs[action] for action in valid_actions}
        total_prob = sum(valid_action_probs.values())
        
        # 확률 재정규화
        if total_prob > 0:
            valid_action_probs = {k: v / total_prob for k, v in valid_action_probs.items()}
            actions, probabilities = zip(*valid_action_probs.items())
            selected_action = np.random.choice(actions, p=probabilities)
        else:
            # 모든 유효 행동의 확률이 0인 경우 균등 분포로 선택
            selected_action = np.random.choice(valid_actions)
        
        return selected_action
