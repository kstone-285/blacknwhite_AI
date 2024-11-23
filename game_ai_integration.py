import torch
from policy_network import ActorCritic


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