import pygame
from enum import Enum

# 색상 정의
DARK_RED = (139, 0, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GOLD = (255, 215, 0)
GRAY = (128, 128, 128)
LIGHT_RED = (220, 20, 60)

# 게임 상태 열거형
class GameState(Enum):
    MAIN_MENU = 0
    SETUP_PLAYER1 = 1
    SETUP_PLAYER2 = 2
    PLAYER1_TURN = 3
    PLAYER2_TURN = 4
    ANIMATING = 5
    ROUND_END = 6
    GAME_OVER = 7

# 애니메이션 파라미터
MOVE_SPEED = 4
CENTER_X = 1000 // 2 - 40
PLAYER1_CENTER_Y = 800 // 2 + 50
PLAYER2_CENTER_Y = 800 // 2 - 170
CARD_VERTICAL_GAP = 120

class Tile:
    def __init__(self, number, is_black, is_player1=True):
        self.number = number
        self.is_black = is_black
        self.is_player1 = is_player1
        self.rect = pygame.Rect(0, 0, 80, 120)
        self.dragging = False
        self.original_pos = None
        self.target_pos = None
        self.used = False

    def move_to_target(self):
        # 타일을 목표 위치로 이동
        if self.target_pos:
            if abs(self.rect.y - self.target_pos[1]) > MOVE_SPEED:
                direction = 1 if self.target_pos[1] > self.rect.y else -1
                self.rect.y += MOVE_SPEED * direction
                return False
            else:
                self.rect.y = self.target_pos[1]
            
            if abs(self.rect.x - self.target_pos[0]) > MOVE_SPEED:
                direction = 1 if self.target_pos[0] > self.rect.x else -1
                self.rect.x += MOVE_SPEED * direction
                return False
            else:
                self.rect.x = self.target_pos[0]
            
            if self.rect.topleft == self.target_pos:
                return True
            else:
                return False
        return True

    def draw(self, surface, hide_number=False):
        # 타일 그리기
        if not self.used:
            color = BLACK if self.is_black else WHITE
            pygame.draw.rect(surface, color, self.rect)
            pygame.draw.rect(surface, GOLD, self.rect, 2)
            if not hide_number:
                font = pygame.font.Font("images\\HeirofLightBold.ttf", 80)
                text = font.render(str(self.number), True, WHITE if self.is_black else BLACK)
                text_rect = text.get_rect(center=(self.rect.centerx, self.rect.centery - 3))
                surface.blit(text, text_rect)
