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
    SETUP_PLAYER1 = 0
    SETUP_PLAYER2 = 1
    PLAYER1_TURN = 2
    PLAYER2_TURN = 3
    ANIMATING = 4
    ROUND_END = 5
    GAME_OVER = 6

# Animation parameters
MOVE_SPEED = 4
CENTER_X = screen_width // 2 - 40
PLAYER1_CENTER_Y = screen_height // 2 + 50
PLAYER2_CENTER_Y = screen_height // 2 - 170
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
        if not self.used:
            color = BLACK if self.is_black else WHITE
            pygame.draw.rect(surface, color, self.rect)
            pygame.draw.rect(surface, GOLD, self.rect, 2)
            if not hide_number:
                text = main_font.render(str(self.number), True, WHITE if self.is_black else BLACK)
                text_rect = text.get_rect(center=self.rect.center)
                surface.blit(text, text_rect)

class BlackWhiteGame:
    def __init__(self):
        self.reset_game()

    def reset_game(self):
        self.state = GameState.SETUP_PLAYER1
        self.player1_tiles = [Tile(i, i % 2 == 0, True) for i in range(9)]
        self.player2_tiles = [Tile(i, i % 2 == 0, False) for i in range(9)]
        tile_start_x = (screen_width - (9 * 90)) // 2
        self.player1_slots = [pygame.Rect(tile_start_x + i * 90, 600, 80, 120) for i in range(9)]
        self.player2_slots = [pygame.Rect(tile_start_x + i * 90, 80, 80, 120) for i in range(9)]
        self.player1_positions = [None] * 9
        self.player2_positions = [None] * 9
        for i, tile in enumerate(self.player1_tiles):
            tile.rect = self.player1_slots[i].copy()
            self.player1_positions[i] = tile
        for i, tile in enumerate(self.player2_tiles):
            tile.rect = self.player2_slots[i].copy()
            self.player2_positions[i] = tile
        self.player1_score = 0
        self.player2_score = 0
        self.round = 1
        self.selected_tile_player1 = None
        self.selected_tile_player2 = None
        self.round_result = None
        self.dragging_tile = None
        self.dragging_start_pos = None
        self.dragging_start_index = None
        self.current_player = 1

    def reset_round(self):
        self.selected_tile_player1 = None
        self.selected_tile_player2 = None
        self.round_result = None
        self.state = GameState.PLAYER1_TURN if self.current_player == 1 else GameState.PLAYER2_TURN

    def find_slot(self, pos, slots):
        for i, slot in enumerate(slots):
            if slot.collidepoint(pos):
                return i
        return None

    def handle_event(self, event):
        if self.state in [GameState.SETUP_PLAYER1, GameState.SETUP_PLAYER2]:
            current_positions = self.player1_positions if self.state == GameState.SETUP_PLAYER1 else self.player2_positions
            current_slots = self.player1_slots if self.state == GameState.SETUP_PLAYER1 else self.player2_slots
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for i, tile in enumerate(current_positions):
                    if tile and tile.rect.collidepoint(event.pos):
                        self.dragging_tile = tile
                        self.dragging_start_index = i
                        current_positions[i] = None
                        self.dragging_start_pos = tile.rect.topleft
                        break
                start_button = pygame.Rect(screen_width // 2 - 75, 470, 150, 50)
                if start_button.collidepoint(event.pos):
                    if self.state == GameState.SETUP_PLAYER1:
                        self.state = GameState.SETUP_PLAYER2
                    else:
                        self.state = GameState.PLAYER1_TURN
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                if self.dragging_tile:
                    target_slot_index = self.find_slot(event.pos, current_slots)
                    if target_slot_index is not None:
                        target_tile = current_positions[target_slot_index]
                        if target_tile:
                            target_tile.rect = current_slots[self.dragging_start_index].copy()
                            current_positions[self.dragging_start_index] = target_tile
                        self.dragging_tile.rect = current_slots[target_slot_index].copy()
                        current_positions[target_slot_index] = self.dragging_tile
                    else:
                        self.dragging_tile.rect.topleft = self.dragging_start_pos
                        current_positions[self.dragging_start_index] = self.dragging_tile
                    self.dragging_tile = None
                    self.dragging_start_index = None
            elif event.type == pygame.MOUSEMOTION:
                if self.dragging_tile:
                    self.dragging_tile.rect.topleft = (event.pos[0] - 40, event.pos[1] - 60)
        elif self.state in [GameState.PLAYER1_TURN, GameState.PLAYER2_TURN]:
            current_positions = self.player1_positions if self.state == GameState.PLAYER1_TURN else self.player2_positions
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for tile in current_positions:
                    if tile and not tile.used and tile.rect.collidepoint(event.pos):
                        selected_tile = tile
                        selected_tile.original_pos = selected_tile.rect.topleft
                        selected_tile.target_pos = (CENTER_X, PLAYER1_CENTER_Y if self.state == GameState.PLAYER1_TURN else PLAYER2_CENTER_Y)
                        if self.state == GameState.PLAYER1_TURN:
                            self.selected_tile_player1 = selected_tile
                        else:
                            self.selected_tile_player2 = selected_tile
                        self.state = GameState.ANIMATING
                        break
        elif self.state == GameState.ROUND_END:
            if event.type == pygame.MOUSEBUTTONDOWN:
                self.round += 1
                self.reset_round()

    def update(self):
        if self.state == GameState.ANIMATING:
            animation_done = False
            if self.selected_tile_player1 and not self.selected_tile_player2:
                animation_done = self.selected_tile_player1.move_to_target()
                if animation_done:
                    self.state = GameState.PLAYER2_TURN
            elif self.selected_tile_player2 and not self.selected_tile_player1:
                animation_done = self.selected_tile_player2.move_to_target()
                if animation_done:
                    self.state = GameState.PLAYER1_TURN
            elif self.selected_tile_player1 and self.selected_tile_player2:
                animation_done_player1 = self.selected_tile_player1.move_to_target()
                animation_done_player2 = self.selected_tile_player2.move_to_target()
                if animation_done_player1 and animation_done_player2:
                    self.compare_tiles()
                    self.state = GameState.ROUND_END

        if self.check_game_over():
            self.state = GameState.GAME_OVER

    def compare_tiles(self):
        player1_tile = self.selected_tile_player1
        player2_tile = self.selected_tile_player2

        time.sleep(0.4)

        if player1_tile.number > player2_tile.number:
            self.player1_score += 1
            self.round_result = "플레이어 1의 승리입니다."
            self.current_player = 1
        elif player1_tile.number < player2_tile.number:
            self.player2_score += 1
            self.round_result = "플레이어 2의 승리입니다."
            self.current_player = 2
        else:
            self.round_result = "무승부입니다. 누구의 점수도 증가하지 않습니다."
        player1_tile.used = True
        player2_tile.used = True

    def check_game_over(self):
        return self.player1_score >= 5 or self.player2_score >= 5 or self.round >= 9

    def draw(self, screen):
        screen.fill(DARK_RED)
        title = title_font.render("더 지니어스 : 흑과백 (Player vs Player)", True, GOLD)
        screen.blit(title, (screen_width // 2 - title.get_width() // 2, 10))

        player1_score = score_font.render(f"플레이어 1 : {self.player1_score}", True, WHITE)
        player2_score = score_font.render(f"플레이어 2 : {self.player2_score}", True, WHITE)
        screen.blit(player1_score, (50, 350))
        screen.blit(player2_score, (screen_width - 150, 350))

        for tile in self.player2_tiles:
            tile.draw(screen, hide_number=(self.state != GameState.PLAYER2_TURN and self.state != GameState.SETUP_PLAYER2))

        for slot in self.player1_slots:
            pygame.draw.rect(screen, GOLD, slot, 1)

        for tile in self.player1_tiles:
            tile.draw(screen, hide_number=(self.state != GameState.PLAYER1_TURN and self.state != GameState.SETUP_PLAYER1))

        if self.dragging_tile:
            self.dragging_tile.draw(screen)

        round_text = main_font.render(f"Round: {self.round}", True, WHITE)
        screen.blit(round_text, (screen_width // 2 - round_text.get_width() // 2, 350))

        if self.state in [GameState.SETUP_PLAYER1, GameState.SETUP_PLAYER2]:
            start_button = pygame.Rect(screen_width // 2 - 75, 470, 150, 50)
            pygame.draw.rect(screen, GOLD, start_button)
            start_text = main_font.render("Next", True, BLACK)
            screen.blit(start_text, (start_button.centerx - start_text.get_width() // 2, start_button.centery - start_text.get_height() // 2))
            setup_text = main_font.render(f"플레이어 {1 if self.state == GameState.SETUP_PLAYER1 else 2} 타일 배치", True, WHITE)
            screen.blit(setup_text, (screen_width // 2 - setup_text.get_width() // 2, 320))
        elif self.state in [GameState.PLAYER1_TURN, GameState.PLAYER2_TURN]:
            turn_text = main_font.render(f"플레이어 {1 if self.state == GameState.PLAYER1_TURN else 2}의 차례입니다.", True, WHITE)
            screen.blit(turn_text, (screen_width // 2 - turn_text.get_width() // 2, 380))
        elif self.state == GameState.ROUND_END:
            if self.round_result:
                result_text = main_font.render(self.round_result, True, GOLD)
                screen.blit(result_text, (screen_width // 2 - result_text.get_width() // 2, 380))
            continue_text = score_font.render("Click to continue", True, WHITE)
            screen.blit(continue_text, (screen_width // 2 - continue_text.get_width() // 2, 410))
        elif self.state == GameState.GAME_OVER:
            screen.fill(DARK_RED)
            game_over_text = title_font.render("게임이 종료되었습니다.", True, GOLD)
            winner_text = None
            if self.player1_score > self.player2_score:
                winner_text = title_font.render("플레이어 1의 승리입니다.", True, WHITE)
            elif self.player2_score > self.player1_score:
                winner_text = title_font.render("플레이어 2의 승리입니다.", True, WHITE)
            else:
                winner_text = title_font.render("무승부입니다.", True, WHITE)
            screen.blit(game_over_text, (screen_width // 2 - game_over_text.get_width() // 2, 310))
            screen.blit(winner_text, (screen_width // 2 - winner_text.get_width() // 2, 400))
            restart_text = main_font.render("Press R to restart", True, WHITE)
            screen.blit(restart_text, (screen_width // 2 - restart_text.get_width() // 2, 490))

def main():

    pygame.mixer.init()
    pygame.mixer.music.load('y2mate.mp3')
    pygame.mixer.music.set_volume(0.15)
    pygame.mixer.music.play(-1)

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
                    game.reset_round()
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