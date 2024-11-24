from environment import BlackWhiteEnv
from policy_network import PolicyGradientAgent
import torch
import numpy as np

def get_valid_actions(env, player_tiles):
    valid_actions = []
    for tile, used in player_tiles:
        if not used:
            valid_actions.append(tile)
    return valid_actions

def state_to_tensor(state):
    # numpy array를 torch tensor로 변환
    return torch.FloatTensor(np.array(state))

def train_policy_agents():
    env = BlackWhiteEnv()
    state_size = 21  # 9(내 타일) + 9(상대 타일) + 3(라운드,점수)
    action_size = 9  # 0-8 타일 선택
    
    agent1 = PolicyGradientAgent(state_size, action_size)
    agent2 = PolicyGradientAgent(state_size, action_size)
    
    max_reward = 3.0
    min_penalty = -1.6
    decay_rate = 0.26
    n_episodes = 15000
    
    for e in range(n_episodes):
        state = env.reset()
        state = state_to_tensor(state)
        episode_reward1 = 0
        episode_reward2 = 0
        
        while True:
            try:
                # Player 1's turn
                valid_actions1 = get_valid_actions(env, env.player1_tiles)
                if not valid_actions1:
                    break
                action1 = agent1.act(state, valid_actions1)
                
                # Player 2's turn
                valid_actions2 = get_valid_actions(env, env.player2_tiles)
                if not valid_actions2:
                    break
                action2 = agent2.act(state, valid_actions2)
                
                # Environment step
                next_state, reward, done = env.step(action2, action1)
                next_state = state_to_tensor(next_state)
                
                # 라운드 결과에 따른 보상 계산
                tile_diff = abs(env.player1_last_tile - env.player2_last_tile)
                if reward > 0:  # Player 2 win
                    round_reward2 = max_reward - tile_diff * decay_rate
                    round_reward1 = min_penalty + tile_diff * decay_rate
                elif reward < 0:  # Player 1 win
                    round_reward2 = min_penalty + tile_diff * decay_rate
                    round_reward1 = max_reward - tile_diff * decay_rate
                else : # draw
                    round_reward1 = min_penalty
                    round_reward2 = min_penalty
                
                agent1.add_reward(round_reward1)
                agent2.add_reward(round_reward2)
                
                episode_reward1 += round_reward1
                episode_reward2 += round_reward2
                
                state = next_state
                
                if done:
                    break
            
            except Exception as e:
                print(f"Error during episode: {str(e)}")
                break
        
        try:
            # 에피소드 종료 시 보상 계산
            if env.player1_score >= 5:
                final_reward1 = 10.0 * (env.player1_score - env.player2_score)
                final_reward2 = -final_reward1
            elif env.player2_score >= 5:
                final_reward2 = 10.0 * (env.player2_score - env.player1_score)
                final_reward1 = -final_reward2
            else:
                final_reward1, final_reward2 = 0, 0
            
            agent1.end_episode(final_reward1)
            agent2.end_episode(final_reward2)
            
            if (e + 1) % 100 == 0:
                print(f"Episode: {e+1}/{n_episodes}")
                print(f"Score: {env.player1_score}-{env.player2_score}")
                print(f"Rewards: {episode_reward1:.2f} vs {episode_reward2:.2f}")
                print("------------------------")
        
        except Exception as e:
            print(f"Error during training: {str(e)}")
            continue
    
    # 학습된 모델 저장
    try:
        torch.save(agent2.model.state_dict(), 'black_white_policy_agent2.pth')
        torch.save(agent1.model.state_dict(), 'black_white_policy_agent1.pth')
        print("Model saved successfully")
    except Exception as e:
        print(f"Error saving model: {str(e)}")


if __name__ == "__main__":
    train_policy_agents()