import torch
from environment import BlackWhiteEnv
from dqn_agent import DQNAgent

def get_valid_actions(env, player_tiles):

    # 사용 가능한 타일을 받는 함수
    valid_actions = []
    for tile, used in player_tiles:
        if not used:
            valid_actions.append(tile)
    return valid_actions

def train_agents():

    env = BlackWhiteEnv()
    state_size = 21  # 9(내 타일) + 9(상대 타일) + 3(라운드,점수)
    action_size = 9  # 0-8 타일 선택
    
    agent1 = DQNAgent(state_size, action_size)
    agent2 = DQNAgent(state_size, action_size)
    
    batch_size = 32
    n_episodes = 10000
    
    for e in range(n_episodes):
        state = env.reset()
        total_reward1 = 0
        total_reward2 = 0
        
        while True:
            # 플레이어1 턴
            valid_actions1 = get_valid_actions(env, env.player1_tiles)
            action1 = agent1.act(state, valid_actions1)
            
            # 플레이어2 턴
            valid_actions2 = get_valid_actions(env, env.player2_tiles)
            action2 = agent2.act(state, valid_actions2)
            
            next_state, reward, done = env.step(action2, action1)
            
            # 경험을 저장
            agent1.remember(state, action1, -reward, next_state, done)
            agent2.remember(state, action2, reward, next_state, done)
            
            state = next_state
            total_reward1 -= reward
            total_reward2 += reward
            
            if done:
                print(f"Episode: {e+1}/{n_episodes}, Score: {env.player1_score}-{env.player2_score}")
                break
                
            # 트레이닝
            agent1.replay(batch_size)
            agent2.replay(batch_size)
            
        if e % 100 == 0:
            agent1.update_target_model()
            agent2.update_target_model()
            
    # 학습된 모델 저장
    torch.save(agent2.model.state_dict(), 'black_white_ai.pth')

if __name__ == "__main__" :

    train_agents()
