import pygame
import random
import math
import time

# Initialize pygame
pygame.init()

# Colors
DARK_RED = (139, 0, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GOLD = (255, 215, 0)

# Screen setup
screen_width, screen_height = 1000, 800
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Black and White")

# Fonts
title_font = pygame.font.Font("C:\\Users\\Kstone\\mypractice\\opensourceproj\\HeirofLightBold.ttf", 40)
main_font = pygame.font.Font("C:\\Users\\Kstone\\mypractice\\opensourceproj\\HeirofLightBold.ttf", 20)
score_font = pygame.font.Font("C:\\Users\\Kstone\\mypractice\\opensourceproj\\HeirofLightBold.ttf", 20)

# Game states
class GameState:
    SETUP = 0
    PLAYER_TURN = 1
    AI_TURN = 2
    ANIMATING = 3
    ROUND_END = 4
    GAME_OVER = 5

# Animation parameters
MOVE_SPEED = 4
CENTER_X = screen_width // 2 - 40
PLAYER_CENTER_Y = screen_height // 2 + 50
AI_CENTER_Y = screen_height // 2 - 170
CARD_VERTICAL_GAP = 120

class Tile:
    def __init__(self, number, is_black, is_player=True):
        self.number = number
        self.is_black = is_black
        self.is_player = is_player
        self.rect = pygame.Rect(0, 0, 80, 120)
        self.dragging = False
        self.original_pos = None
        self.target_pos = None
        self.used = False
        self.revealed = False

    def move_to_target(self):
        if self.target_pos:
            # 먼저 수직 이동
            if abs(self.rect.y - self.target_pos[1]) > MOVE_SPEED:
                direction = 1 if self.target_pos[1] > self.rect.y else -1
                self.rect.y += MOVE_SPEED * direction
                return False
            else:
                self.rect.y = self.target_pos[1]

            # 수직 이동이 완료되면 수평 이동
            if abs(self.rect.x - self.target_pos[0]) > MOVE_SPEED:
                direction = 1 if self.target_pos[0] > self.rect.x else -1
                self.rect.x += MOVE_SPEED * direction
                return False
            else:
                self.rect.x = self.target_pos[0]

            # 목표 위치에 도달했는지 확인
            if self.rect.topleft == self.target_pos:
                return True
            else:
                return False
        return True

    def draw(self, surface):
        if not self.used:
            color = BLACK if self.is_black else WHITE
            pygame.draw.rect(surface, color, self.rect)
            pygame.draw.rect(surface, GOLD, self.rect, 2)
            if self.is_player or self.revealed:
                text = main_font.render(str(self.number), True, WHITE if self.is_black else BLACK)
                text_rect = text.get_rect(center=self.rect.center)
                surface.blit(text, text_rect)

class BlackWhiteGame:
    def __init__(self):
        self.reset_game()

    def reset_game(self):
        self.state = GameState.SETUP
        self.player_tiles = [Tile(i, i % 2 == 0, True) for i in range(9)]
        self.ai_tiles = [Tile(i, i % 2 == 0, False) for i in range(9)]
        random.shuffle(self.ai_tiles)
        
        tile_start_x = (screen_width - (9 * 90)) // 2
        self.player_slots = [pygame.Rect(tile_start_x + i * 90, 600, 80, 120) for i in range(9)]
        self.ai_positions = [pygame.Rect(tile_start_x + i * 90, 80, 80, 120) for i in range(9)]
        
        self.player_positions = [None] * 9
        for i, tile in enumerate(self.player_tiles):
            tile.rect = self.player_slots[i].copy()
            self.player_positions[i] = tile
        
        for tile, pos in zip(self.ai_tiles, self.ai_positions):
            tile.rect = pos.copy()
        
        self.player_score = 0
        self.ai_score = 0
        self.round = 1
        self.selected_tile = None
        self.ai_selected_tile = None
        self.round_result = None
        self.dragging_tile = None
        self.dragging_start_pos = None
        self.dragging_start_index = None
        self.player_first = True

    def find_slot(self, pos):
        for i, slot in enumerate(self.player_slots):
            if slot.collidepoint(pos):
                return i
        return None

    def handle_event(self, event):
        if self.state == GameState.SETUP:
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for i, tile in enumerate(self.player_positions):
                    if tile and tile.rect.collidepoint(event.pos):
                        self.dragging_tile = tile
                        self.dragging_start_index = i
                        self.player_positions[i] = None
                        self.dragging_start_pos = tile.rect.topleft
                        break
                
                start_button = pygame.Rect(screen_width // 2 - 75, 470, 150, 50)
                if start_button.collidepoint(event.pos):
                    self.state = GameState.PLAYER_TURN
            
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                if self.dragging_tile:
                    target_slot_index = self.find_slot(event.pos)
                    if target_slot_index is not None:
                        target_tile = self.player_positions[target_slot_index]
                        if target_tile:
                            target_tile.rect = self.player_slots[self.dragging_start_index].copy()
                            self.player_positions[self.dragging_start_index] = target_tile
                        self.dragging_tile.rect = self.player_slots[target_slot_index].copy()
                        self.player_positions[target_slot_index] = self.dragging_tile
                    else:
                        self.dragging_tile.rect.topleft = self.dragging_start_pos
                        self.player_positions[self.dragging_start_index] = self.dragging_tile
                    self.dragging_tile = None
                    self.dragging_start_index = None
            
            elif event.type == pygame.MOUSEMOTION:
                if self.dragging_tile:
                    self.dragging_tile.rect.topleft = (event.pos[0] - 40, event.pos[1] - 60)
        
        elif self.state == GameState.PLAYER_TURN:
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for tile in self.player_positions:
                    if tile and not tile.used and tile.rect.collidepoint(event.pos):
                        self.selected_tile = tile
                        self.selected_tile.original_pos = self.selected_tile.rect.topleft
                        self.selected_tile.target_pos = (CENTER_X, PLAYER_CENTER_Y)
                        self.state = GameState.ANIMATING
                        break

    def ai_select_tile(self):
        available_ai_tiles = [t for t in self.ai_tiles if not t.used]
        self.ai_selected_tile = random.choice(available_ai_tiles)
        self.ai_selected_tile.original_pos = self.ai_selected_tile.rect.topleft
        self.ai_selected_tile.target_pos = (CENTER_X, AI_CENTER_Y)
        self.state = GameState.ANIMATING

    def update(self):
        if self.state == GameState.AI_TURN:
            self.ai_select_tile()
        elif self.state == GameState.ANIMATING:
            player_done = self.selected_tile.move_to_target() if self.selected_tile else True
            ai_done = self.ai_selected_tile.move_to_target() if self.ai_selected_tile else True
            
            if player_done and ai_done:
                if self.selected_tile and self.ai_selected_tile:
                    self.selected_tile.used = True
                    self.ai_selected_tile.used = True
                    self.ai_selected_tile.revealed = True
                    
                    if self.selected_tile.number > self.ai_selected_tile.number:
                        self.player_score += 1
                        self.round_result = "당신의 승리입니다."
                        self.player_first = True
                    elif self.selected_tile.number < self.ai_selected_tile.number:
                        self.ai_score += 1
                        self.round_result = "AI의 승리입니다."
                        self.player_first = False
                    else:
                        self.round_result = "무승부입니다. 누구의 점수도 증가하지 않습니다."
                    
                    self.state = GameState.ROUND_END
                    
                    if self.check_game_over():
                        self.state = GameState.GAME_OVER
                elif self.selected_tile:
                    self.state = GameState.AI_TURN
                elif self.ai_selected_tile:
                    self.state = GameState.PLAYER_TURN

    def check_game_over(self):
        return self.player_score >= 5 or self.ai_score >= 5 or self.round >= 9

    def draw(self, screen):
        screen.fill(DARK_RED)

        title = title_font.render("더 지니어스 : 흑과백 (with AI)", True, GOLD)
        screen.blit(title, (screen_width // 2 - title.get_width() // 2, 10))

        player_score = score_font.render(f"플레이어 : {self.player_score}", True, WHITE)
        ai_score = score_font.render(f"AI : {self.ai_score}", True, WHITE)
        screen.blit(player_score, (50, 350))
        screen.blit(ai_score, (screen_width - 150, 350))

        for tile in self.ai_tiles:
            tile.draw(screen)

        for slot in self.player_slots:
            pygame.draw.rect(screen, GOLD, slot, 1)
        for tile in self.player_positions:
            if tile and tile != self.dragging_tile:
                tile.draw(screen)

        if self.dragging_tile:
            self.dragging_tile.draw(screen)

        if self.state == GameState.SETUP:
            start_button = pygame.Rect(screen_width // 2 - 75, 470, 150, 50)
            pygame.draw.rect(screen, GOLD, start_button)
            start_text = main_font.render("게임 시작", True, BLACK)
            screen.blit(start_text, (start_button.centerx - start_text.get_width() // 2, start_button.centery - start_text.get_height() // 2))
            instruction = score_font.render("타일을 드래그해 배치하십시오.", True, WHITE)
            screen.blit(instruction, (screen_width // 2 - instruction.get_width() // 2, 320))

        elif self.state in [GameState.ROUND_END, GameState.GAME_OVER]:
            if self.state == GameState.GAME_OVER:
                game_over_text = title_font.render("게임이 종료되었습니다.", True, GOLD)
                winner_text = None
                if self.player_score > self.ai_score:
                    winner_text = title_font.render("당신의 승리입니다.", True, WHITE)
                elif self.ai_score > self.player_score:
                    winner_text = title_font.render("AI의 승리입니다.", True, WHITE)
                else:
                    winner_text = title_font.render("무승부입니다.", True, WHITE)
                
                screen.blit(game_over_text, (screen_width // 2 - game_over_text.get_width() // 2, 270))
                screen.blit(winner_text, (screen_width // 2 - winner_text.get_width() // 2, 350))
                
                restart_text = main_font.render("Press R to restart", True, WHITE)
                screen.blit(restart_text, (screen_width // 2 - restart_text.get_width() // 2, 410))
            else:
                if self.round_result:
                    result_text = main_font.render(self.round_result, True, GOLD)
                    time.sleep(0.5)
                    screen.blit(result_text, (screen_width // 2 - result_text.get_width() // 2, 380))
                continue_text = score_font.render("Click to continue", True, WHITE)
                screen.blit(continue_text, (screen_width // 2 - continue_text.get_width() // 2, 410))

        if self.state in [GameState.PLAYER_TURN, GameState.AI_TURN]:
            turn_text = main_font.render("당신의 차례입니다." if self.state == GameState.PLAYER_TURN else "AI의 차례입니다.", True, WHITE)
            screen.blit(turn_text, (screen_width // 2 - turn_text.get_width() // 2, 380))

        round_text = main_font.render(f"Round: {self.round}", True, WHITE)
        screen.blit(round_text, (screen_width // 2 - round_text.get_width() // 2, 350))

def main():
    clock = pygame.time.Clock()
    game = BlackWhiteGame()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if game.state == GameState.ROUND_END:
                    game.round += 1
                    game.selected_tile = None
                    game.ai_selected_tile = None
                    game.state = GameState.PLAYER_TURN if game.player_first else GameState.AI_TURN
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r and game.state == GameState.GAME_OVER:
                    game.reset_game()
            
            game.handle_event(event)

        game.update()
        game.draw(screen)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()