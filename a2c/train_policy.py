from environment import BlackWhiteEnv
from policy_network import PolicyGradientAgent
import torch
import numpy as np

def get_valid_actions(env, player_tiles):
    valid_actions = [tile for tile, used in player_tiles if not used]
    return valid_actions

def state_to_tensor(state):
    return torch.FloatTensor(np.array(state))

def train_policy_agents():
    env = BlackWhiteEnv()
    state_size = 21  # 9(내 타일) + 9(상대 타일) + 3(라운드,점수)
    action_size = 9  # 0-8 타일 선택
    
    agent1 = PolicyGradientAgent(state_size, action_size)
    agent2 = PolicyGradientAgent(state_size, action_size)
    
    max_reward = 2.7
    min_penalty = -1.5
    decay_rate = 0.22
    n_episodes = 30000
    
    episode_rewards1 = []
    episode_rewards2 = []
    player1_wins = 0
    player2_wins = 0
    
    for e in range(n_episodes):
        state = env.reset()
        state = state_to_tensor(state)
        episode_reward1 = 0
        episode_reward2 = 0
        
        while True:
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
            
            # Calculate round rewards
            tile_diff = abs(env.player1_last_tile - env.player2_last_tile)
            if reward > 0:  # Player 2 wins
                round_reward2 = max_reward - tile_diff * decay_rate
                round_reward1 = min_penalty + tile_diff * decay_rate
            elif reward < 0:  # Player 1 wins
                round_reward2 = min_penalty + tile_diff * decay_rate
                round_reward1 = max_reward - tile_diff * decay_rate
            else:  # Draw
                round_reward1 = 0
                round_reward2 = 0
            
            agent1.add_reward(round_reward1)
            agent2.add_reward(round_reward2)
            
            episode_reward1 += round_reward1
            episode_reward2 += round_reward2
            
            state = next_state
            
            if done:
                break
        
        # Episode end rewards
        if env.player1_score >= 5:
            player1_wins += 1
            final_reward1 = 12.0 * (env.player1_score - env.player2_score)
            final_reward2 = -final_reward1
        elif env.player2_score >= 5:
            player2_wins += 1
            final_reward2 = 12.0 * (env.player2_score - env.player1_score)
            final_reward1 = -final_reward2
        else:
            final_reward1, final_reward2 = 0, 0
        
        agent1.end_episode(final_reward1)
        agent2.end_episode(final_reward2)
        
        episode_rewards1.append(episode_reward1)
        episode_rewards2.append(episode_reward2)
        
        # Logging every 100 episodes
        if (e + 1) % 100 == 0:
            avg_reward1 = np.mean(episode_rewards1[-100:])
            avg_reward2 = np.mean(episode_rewards2[-100:])
            print(f"Episode: {e + 1}/{n_episodes}")
            print(f"Avg Rewards (Last 100): Player1 = {avg_reward1:.2f}, Player2 = {avg_reward2:.2f}")
            print(f"Win Rate: Player1 = {player1_wins / (e + 1):.2%}, Player2 = {player2_wins / (e + 1):.2%}")
            print("------------------------")
    
    # Save models
    torch.save(agent1.model.state_dict(), 'black_white_policy_agent1.pth')
    torch.save(agent2.model.state_dict(), 'black_white_policy_agent2.pth')
    print("Models saved successfully")

    # Evaluation after training
    evaluate_agents(agent1, agent2, env)

def evaluate_agents(agent1, agent2, env, n_games=1000):
    player1_wins = 0
    player2_wins = 0
    total_rewards1 = 0
    total_rewards2 = 0
    
    for _ in range(n_games):
        state = env.reset()
        state = state_to_tensor(state)
        game_reward1 = 0
        game_reward2 = 0
        
        while True:
            valid_actions1 = get_valid_actions(env, env.player1_tiles)
            if not valid_actions1:
                break
            action1 = agent1.act(state, valid_actions1)
            
            valid_actions2 = get_valid_actions(env, env.player2_tiles)
            if not valid_actions2:
                break
            action2 = agent2.act(state, valid_actions2)
            
            next_state, reward, done = env.step(action2, action1)
            next_state = state_to_tensor(next_state)
            
            game_reward1 += reward if reward < 0 else 0
            game_reward2 += reward if reward > 0 else 0
            
            state = next_state
            
            if done:
                break
        
        total_rewards1 += game_reward1
        total_rewards2 += game_reward2
        if env.player1_score > env.player2_score:
            player1_wins += 1
        elif env.player2_score > env.player1_score:
            player2_wins += 1
    
    print("\n--- Evaluation Results ---")
    print(f"Win Rate: Player1 = {player1_wins / n_games:.2%}, Player2 = {player2_wins / n_games:.2%}")
    print(f"Avg Rewards: Player1 = {total_rewards1 / n_games:.2f}, Player2 = {total_rewards2 / n_games:.2f}")

if __name__ == "__main__":
    train_policy_agents()
