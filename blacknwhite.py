import pygame
import random
import math

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
    PLAYING = 1
    ANIMATING = 2
    ROUND_END = 3
    GAME_OVER = 4

# Animation parameters
MOVE_SPEED = 10
CENTER_X = screen_width // 2 - 40
PLAYER_CENTER_Y = screen_height // 2 + 50    # 플레이어 카드는 아래쪽으로
AI_CENTER_Y = screen_height // 2 - 170       # AI 카드는 위쪽으로
CARD_VERTICAL_GAP = 120                      # 카드 간 수직 간격

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
        self.should_remove = False

    def move_to_target(self):
        if self.target_pos:
            dx = self.target_pos[0] - self.rect.x
            dy = self.target_pos[1] - self.rect.y
            distance = math.sqrt(dx**2 + dy**2)
            
            if distance < MOVE_SPEED:
                self.rect.topleft = self.target_pos
                return True
            else:
                move_x = (dx / distance) * MOVE_SPEED
                move_y = (dy / distance) * MOVE_SPEED
                self.rect.x += move_x
                self.rect.y += move_y
                return False
        return True

    def draw(self, surface):
        if not self.should_remove:
            color = BLACK if self.is_black else WHITE
            pygame.draw.rect(surface, color, self.rect)
            pygame.draw.rect(surface, GOLD, self.rect, 2)
            
            # AI 카드의 숫자는 절대 보여주지 않음
            if self.is_player:
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
        self.round = 0
        self.selected_tile = None
        self.ai_selected_tile = None
        self.round_result = None
        self.dragging_tile = None
        self.dragging_start_pos = None
        self.dragging_start_index = None

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
                    self.state = GameState.PLAYING
                    
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
        elif self.state == GameState.PLAYING:
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for tile in self.player_positions:
                    if tile and not tile.used and tile.rect.collidepoint(event.pos):
                        self.selected_tile = tile
                        self.selected_tile.original_pos = self.selected_tile.rect.topleft
                        available_ai_tiles = [t for t in self.ai_tiles if not t.used]
                        self.ai_selected_tile = random.choice(available_ai_tiles)
                        self.ai_selected_tile.original_pos = self.ai_selected_tile.rect.topleft
                        
                        # Set targets for vertical alignment
                        self.selected_tile.target_pos = (CENTER_X, PLAYER_CENTER_Y)
                        self.ai_selected_tile.target_pos = (CENTER_X, AI_CENTER_Y)
                        self.state = GameState.ANIMATING
                        break

    def update(self):
        if self.state == GameState.ANIMATING:
            player_done = self.selected_tile.move_to_target()
            ai_done = self.ai_selected_tile.move_to_target()
            
            if player_done and ai_done:
                self.selected_tile.used = True
                self.ai_selected_tile.used = True
                
                # 결과 결정 (AI 카드는 공개하지 않음)
                if self.selected_tile.number > self.ai_selected_tile.number:
                    self.player_score += 1
                    self.round_result = "당신의 승리입니다."
                elif self.selected_tile.number < self.ai_selected_tile.number:
                    self.ai_score += 1
                    self.round_result = "AI의 승리입니다."
                else:
                    self.round_result = "무승부입니다. 누구의 점수도 증가하지 않습니다."
                
                self.round += 1
                self.state = GameState.ROUND_END
                
                if self.check_game_over():
                    self.state = GameState.GAME_OVER
                    self.round_result = None

    def check_game_over(self):
        return self.player_score >= 5 or self.ai_score >= 5 or self.round >= 9

    def draw(self, screen):
        screen.fill(DARK_RED)
        
        # Title
        title = title_font.render("더 지니어스 : 흑과백 (with AI)", True, GOLD)
        screen.blit(title, (screen_width // 2 - title.get_width() // 2, 10))
        
        # Scores
        player_score = score_font.render(f"플레이어 : {self.player_score}", True, WHITE)
        ai_score = score_font.render(f"AI : {self.ai_score}", True, WHITE)
        screen.blit(player_score, (50, 350))
        screen.blit(ai_score, (screen_width - 150, 350))

        # Draw AI tiles
        for tile in self.ai_tiles:
            if not tile.should_remove:
                tile.draw(screen)

        # Draw player slots and tiles
        for slot in self.player_slots:
            pygame.draw.rect(screen, GOLD, slot, 1)

        for tile in self.player_positions:
            if tile and tile != self.dragging_tile and not tile.should_remove:
                tile.draw(screen)

        if self.dragging_tile:
            self.dragging_tile.draw(screen)

        if self.state == GameState.SETUP:
            start_button = pygame.Rect(screen_width // 2 - 75, 470, 150, 50)
            pygame.draw.rect(screen, GOLD, start_button)
            start_text = main_font.render("게임 시작", True, BLACK)
            screen.blit(start_text, (start_button.centerx - start_text.get_width() // 2, 
                                    start_button.centery - start_text.get_height() // 2))
            
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
                
                # 게임 오버 텍스트를 더 위쪽으로 이동
                screen.blit(game_over_text, (screen_width // 2 - game_over_text.get_width() // 2, 270))
                screen.blit(winner_text, (screen_width // 2 - winner_text.get_width() // 2, 350))
                
                restart_text = main_font.render("Press R to restart", True, WHITE)
                screen.blit(restart_text, (screen_width // 2 - restart_text.get_width() // 2, 410))
            else:
                if self.round_result:
                    result_text = main_font.render(self.round_result, True, GOLD)
                    screen.blit(result_text, (screen_width // 2 - result_text.get_width() // 2, 360))
                continue_text = score_font.render("Click to continue", True, WHITE)
                screen.blit(continue_text, (screen_width // 2 - continue_text.get_width() // 2, 410))

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
                    game.selected_tile.should_remove = True
                    game.ai_selected_tile.should_remove = True
                    game.state = GameState.PLAYING
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