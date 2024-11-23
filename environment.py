import numpy as np
import torch

class BlackWhiteEnv:
    def __init__(self):
        self.reset()
        
    def reset(self):
        # 각 플레이어의 타일 상태 (0-8 숫자, used 여부)
        self.player1_tiles = [(i, False) for i in range(9)]
        self.player2_tiles = [(i, False) for i in range(9)]
        self.current_round = 0
        self.player1_score = 0
        self.player2_score = 0
        self.player1_last_tile = None  # Player 1의 마지막 타일
        self.player2_last_tile = None  # Player 2의 마지막 타일
        # 현재 상태를 표현하는 벡터 반환
        return self._get_state()
        
    def _get_state(self):
        # 상태: [내 남은 타일들, 상대 사용한 타일들, 현재 라운드, 내 점수, 상대 점수]
        state = []
        # 내 남은 타일들 (9개 타일에 대해 있으면 1, 없으면 0)
        for i in range(9):
            found = False
            for tile, used in self.player2_tiles:
                if tile == i and not used:
                    state.append(1)
                    found = True
                    break
            if not found:
                state.append(0)
        
        # 상대가 사용한 타일들 (9개 타일에 대해 사용했으면 1, 아니면 0)
        for i in range(9):
            found = False
            for tile, used in self.player1_tiles:
                if tile == i and used:
                    state.append(1)
                    found = True
                    break
            if not found:
                state.append(0)
                
        # 현재 라운드, 점수 상태 추가
        state.extend([self.current_round, self.player1_score, self.player2_score])
        
        return np.array(state, dtype=np.float32)
    
    def step(self, action, opponent_action):
        # action: 0-8 사이의 숫자 (선택한 타일 번호)
        reward = 0
        done = False
        
        # 유효한 액션인지 확인
        valid_action = False
        for i, (tile, used) in enumerate(self.player2_tiles):
            if tile == action and not used:
                self.player2_tiles[i] = (tile, True)
                self.player2_last_tile = tile  # Player 2의 마지막 사용 타일 기록
                valid_action = True
                break
                
        if not valid_action:
            return self._get_state(), -10, True  # 잘못된 액션 선택시 패널티
         
        # 상대방 액션 처리
        valid_opponent = False
        for i, (tile, used) in enumerate(self.player1_tiles):
            if tile == opponent_action and not used:
                self.player1_tiles[i] = (tile, True)
                self.player1_last_tile = tile  # Player 1의 마지막 사용 타일 기록
                valid_opponent = True
                break
                
        if not valid_opponent:
            return self._get_state(), 10, True  # 상대가 잘못된 액션 선택시 보상
            
        # 승패 판정
        if action > opponent_action:
            self.player2_score += 1
            reward = 1
        elif action < opponent_action:
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
                
        return self._get_state(), reward, done
