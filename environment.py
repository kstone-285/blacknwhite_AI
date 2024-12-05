import numpy as np
import torch

class BlackWhiteEnv:

    def __init__(self):
        self.reset()

    def reset(self):

        # 각 플레이어의 타일 상태 (0-8 숫자, 사용 여부)
        tiles = np.arange(9)
        np.random.shuffle(tiles)  # 타일 무작위 배치
        self.player1_tiles = [(tile, False) for tile in tiles]
        np.random.shuffle(tiles)  # 타일 무작위 배치
        self.player2_tiles = [(tile, False) for tile in tiles]
        
        self.current_round = 0
        self.player1_score = 0
        self.player2_score = 0
        self.player1_last_tile = None
        self.player2_last_tile = None
        
        # 상태 벡터 초기화
        return self._get_state()
        
    def _get_state(self):
        
        # 상태: [내 남은 타일, 상대 사용한 타일, 현재 라운드, 내 점수, 상대 점수]
        state = []
        # 내 남은 타일들
        for tile, used in self.player2_tiles:
            state.append(0 if used else 1)
        # 상대가 사용한 타일들
        for tile, used in self.player1_tiles:
            state.append(1 if used else 0)
        # 현재 라운드와 점수 상태 추가
        state.extend([self.current_round, self.player1_score, self.player2_score])
        return np.array(state, dtype=np.float32)
    
    def step(self, player2_action, player1_action):
        reward = 0
        done = False
        
        # Player 2 (AI)의 행동 처리
        if not self._process_action(self.player2_tiles, player2_action, is_player2=True):
            return self._get_state(), -10, True  # 잘못된 액션 선택시 패널티
         
        # Player 1 (AI)의 행동 처리
        if not self._process_action(self.player1_tiles, player1_action, is_player2=False):
            return self._get_state(), 10, True  # 상대가 잘못된 액션 선택시 보상
            
        # 승패 판정
        if player2_action > player1_action:
            self.player2_score += 1
            reward = 1
        elif player2_action < player1_action:
            self.player1_score += 1
            reward = -1
            
        self.current_round += 1
        
        # 게임 종료 조건 체크
        if self.player1_score >= 5 or self.player2_score >= 5 or self.current_round >= 9:
            done = True
            if self.player2_score > self.player1_score:
                reward = 10
            elif self.player2_score < self.player1_score:
                reward = -10
            else:
                reward = 0
                
        return self._get_state(), reward, done

    def _process_action(self, tiles, action, is_player2):
        for i, (tile, used) in enumerate(tiles):
            if tile == action and not used:
                tiles[i] = (tile, True)
                if is_player2:
                    self.player2_last_tile = tile
                else:
                    self.player1_last_tile = tile
                return True
        return False
