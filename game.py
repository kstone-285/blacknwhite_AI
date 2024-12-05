import pygame
import random
import time
import numpy as np
import torch
from utils import Tile, GameState, DARK_RED, WHITE, BLACK, GOLD, GRAY, LIGHT_RED
from utils import CENTER_X, PLAYER1_CENTER_Y, PLAYER2_CENTER_Y
from game_ai_integration import PPOAIPlayer, AIPlayer, PolicyAIPlayer

class BlackWhiteGame:

    def __init__(self, is_ai_mode=True):
        self.is_ai_mode = is_ai_mode
        self.ai_player = PPOAIPlayer('trained_AI\\agent1_policy_ppo.pth') if is_ai_mode else None
        self.reset_game()
        self.menu_particles = self.create_menu_particles()
        self.title_glow = 0
        self.title_glow_direction = 1
        self.screen_width, self.screen_height = 1000, 800
        
        # 폰트 설정
        self.title_font = pygame.font.Font("images\\HeirofLightBold.ttf", 40)
        self.main_font = pygame.font.Font("images\\HeirofLightBold.ttf", 20)
        self.score_font = pygame.font.Font("images\\HeirofLightBold.ttf", 20)
        self.large_font = pygame.font.Font("images\\HeirofLightBold.ttf", 80)

    def create_menu_particles(self):
        # 메뉴 파티클 생성 로직
        particles = []
        for _ in range(200):
            x = random.randint(0, 1000)
            y = random.randint(0, 800)
            size = random.randint(1, 3)
            speed_x = random.uniform(-0.5, 0.5)
            speed_y = random.uniform(-0.5, 0.5)
            particles.append({
                'pos': [x, y],
                'size': size,
                'speed_x': speed_x,
                'speed_y': speed_y
            })
        return particles

    def update_menu_particles(self):
        # 메뉴 파티클 업데이트 로직
        for particle in self.menu_particles:
            particle['pos'][0] += particle['speed_x']
            particle['pos'][1] += particle['speed_y']
            if particle['pos'][0] < 0:
                particle['pos'][0] = 1000
            elif particle['pos'][0] > 1000:
                particle['pos'][0] = 0
            if particle['pos'][1] < 0:
                particle['pos'][1] = 800
            elif particle['pos'][1] > 800:
                particle['pos'][1] = 0

    def draw_menu_particles(self, screen):
        # 메뉴 파티클 그리기
        for particle in self.menu_particles:
            pygame.draw.circle(screen, GOLD, (int(particle['pos'][0]), int(particle['pos'][1])), particle['size'])

    def draw_main_menu(self, screen):

        if not hasattr(self, 'gradient_value'):
            self.gradient_value = 0
            self.gradient_direction = 1
            self.last_update = time.time()

        now = time.time()
        time_diff = now - self.last_update

        if time_diff >= 0.04:  
            self.gradient_value += self.gradient_direction * 2 
            if self.gradient_value >= 50: 
                self.gradient_value = 50
                self.gradient_direction = -1
            elif self.gradient_value <= 0: 
                self.gradient_value = 0
                self.gradient_direction = 1
            self.last_update = now

        background_color = (self.gradient_value, self.gradient_value, self.gradient_value)
        screen.fill(background_color)
        
        self.update_menu_particles()
        self.draw_menu_particles(screen)

        self.title_glow += 2 * self.title_glow_direction
        if self.title_glow > 200 or self.title_glow < 0:
            self.title_glow_direction *= -1
 
        title_surface = self.large_font.render("더 지니어스", True, GOLD)
        title_glow = self.large_font.render("더 지니어스", True, GRAY)
        
        screen.blit(title_glow, (self.screen_width // 2 - title_surface.get_width() // 2 - 48, 180 - 2))
        screen.blit(title_surface, (self.screen_width // 2 - title_surface.get_width() // 2 - 50, 180))

        subtitle = self.title_font.render(":  흑과백", True, WHITE)
        subtitle_glow = self.title_font.render(":  흑과백", True, GRAY)
        screen.blit(subtitle_glow, (self.screen_width // 2 + title_surface.get_width() // 2 - 28, 230 - 2))        
        screen.blit(subtitle, (self.screen_width // 2 + title_surface.get_width() // 2 - 30 , 230))

        mouse_pos = pygame.mouse.get_pos()
        
        # AI 모드 버튼
        vs_ai_rect = pygame.Rect(self.screen_width // 2 - 120, 350, 250, 60)
        vs_ai_color = GOLD if vs_ai_rect.collidepoint(mouse_pos) else LIGHT_RED
        pygame.draw.rect(screen, vs_ai_color, vs_ai_rect, border_radius=10)
        vs_ai_text = self.main_font.render("AI 대결", True, BLACK)
        screen.blit(vs_ai_text, (vs_ai_rect.centerx - vs_ai_text.get_width() // 2, vs_ai_rect.centery - vs_ai_text.get_height() // 2))

        # 2P 모드 버튼
        vs_player_rect = pygame.Rect(self.screen_width // 2 - 120, 450, 250, 60)
        vs_player_color = GOLD if vs_player_rect.collidepoint(mouse_pos) else LIGHT_RED
        pygame.draw.rect(screen, vs_player_color, vs_player_rect, border_radius=10)
        vs_player_text = self.main_font.render("2P 모드", True, BLACK)
        screen.blit(vs_player_text, (vs_player_rect.centerx - vs_player_text.get_width() // 2, vs_player_rect.centery - vs_player_text.get_height() // 2))

        # 게임 종료 버튼
        exit_rect = pygame.Rect(self.screen_width // 2 - 120, 550, 250, 60)
        exit_color = GOLD if exit_rect.collidepoint(mouse_pos) else LIGHT_RED
        pygame.draw.rect(screen, exit_color, exit_rect, border_radius=10)
        exit_text = self.main_font.render("게임 종료", True, BLACK)
        screen.blit(exit_text, (exit_rect.centerx - exit_text.get_width() // 2, exit_rect.centery - exit_text.get_height() // 2))

        return vs_ai_rect, vs_player_rect, exit_rect


    def get_state_for_ai(self):
        # AI를 위한 게임 상태 정보 생성
        state = []
        for i in range(9):
            found = False
            for tile in self.player2_tiles:
                if not tile.used and tile.number == i:
                    state.append(1)
                    found = True
                    break
            if not found:
                state.append(0)
        
        for i in range(9):
            found = False
            for tile in self.player1_tiles:
                if tile.used:
                    state.append(1)
                    found = True
                    break
            if not found:
                state.append(0)
        
        state.extend([self.round, self.player1_score, self.player2_score])
        return np.array(state, dtype=np.float32)

    def reset_game(self):
        # 게임 초기화 로직
        self.state = GameState.MAIN_MENU
        self.player1_tiles = [Tile(i, i % 2 == 0, True) for i in range(9)]
        self.player2_tiles = [Tile(i, i % 2 == 0, False) for i in range(9)]
        if self.is_ai_mode:
            random.shuffle(self.player2_tiles)
        
        tile_start_x = (1000 - (9 * 90)) // 2
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
        # 라운드 초기화
        self.selected_tile_player1 = None
        self.selected_tile_player2 = None
        self.round_result = None
        self.state = GameState.PLAYER1_TURN if self.current_player == 1 else GameState.PLAYER2_TURN

    def find_slot(self, pos, slots):
        # 주어진 위치에 해당하는 슬롯 찾기
        for i, slot in enumerate(slots):
            if slot.collidepoint(pos):
                return i
        return None

    def handle_event(self, event):

        if self.is_ai_mode and self.state == GameState.PLAYER2_TURN:
            # AI의 턴
            valid_actions = []
            for tile in self.player2_tiles:
                if not tile.used:
                    valid_actions.append(tile.number)
            
            state = self.get_state_for_ai()
            action = self.ai_player.get_action(state, valid_actions)
            
            # 선택된 타일 찾기
            for tile in self.player2_tiles:
                if not tile.used and tile.number == action:
                    selected_tile = tile
                    selected_tile.original_pos = selected_tile.rect.topleft
                    selected_tile.target_pos = (CENTER_X, PLAYER2_CENTER_Y)
                    self.selected_tile_player2 = selected_tile
                    self.state = GameState.ANIMATING
                    break

        elif self.state in [GameState.SETUP_PLAYER1, GameState.SETUP_PLAYER2]:
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
                start_button = pygame.Rect(self.screen_width // 2 - 75, 470, 150, 50)
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
        
        if player1_tile is None or player2_tile is None:
            self.round_result = "오류: 선택된 타일이 없습니다."
            return
        
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
        return self.player1_score >= 5 or self.player2_score >= 5 or self.round >= 10

    def draw(self, screen):
        
        screen.fill(DARK_RED)
        title = self.title_font.render("더 지니어스 : 흑과백 " + ("( VS AI )" if self.is_ai_mode else "( 2P 모드 )"), True, GOLD)
        screen.blit(title, (self.screen_width // 2 - title.get_width() // 2, 10))

        player1_score = self.score_font.render(f"플레이어 1 : {self.player1_score}", True, WHITE)
        player2_score = self.score_font.render(f"플레이어 2 : {self.player2_score}", True, WHITE)
        screen.blit(player1_score, (50, 350))
        screen.blit(player2_score, (self.screen_width - 150, 350))

        for tile in self.player2_tiles:
            tile.draw(screen, 
                hide_number=(self.is_ai_mode or 
                            (not self.is_ai_mode and self.state == GameState.SETUP_PLAYER1) or
                            (not self.is_ai_mode and self.state not in [GameState.PLAYER2_TURN, GameState.SETUP_PLAYER2, GameState.SETUP_PLAYER1])))

        for slot in self.player1_slots:
            pygame.draw.rect(screen, GOLD, slot, 1)

        for tile in self.player1_tiles:
            tile.draw(screen, hide_number=(self.state != GameState.PLAYER1_TURN and self.state != GameState.SETUP_PLAYER1))

        if self.dragging_tile:
            self.dragging_tile.draw(screen)

        round_text = self.main_font.render(f"Round: {self.round}", True, WHITE)
        screen.blit(round_text, (self.screen_width // 2 - round_text.get_width() // 2, 350))

        if self.state in [GameState.SETUP_PLAYER1, GameState.SETUP_PLAYER2]:

            if(self.is_ai_mode and self.state == GameState.SETUP_PLAYER2) : 
                self.state = GameState.PLAYER1_TURN

            start_button = pygame.Rect(self.screen_width // 2 - 75, 470, 150, 50)
            start_button_color = GOLD if start_button.collidepoint(pygame.mouse.get_pos()) else GRAY
            pygame.draw.rect(screen, start_button_color, start_button)
            start_text = self.main_font.render("NEXT", True, BLACK)
            screen.blit(start_text, (start_button.centerx - start_text.get_width() // 2, start_button.centery - start_text.get_height() // 2 - 2))
            setup_text = self.main_font.render(f"플레이어 {1 if self.state == GameState.SETUP_PLAYER1 else 2} 타일 배치", True, WHITE)

            screen.blit(setup_text, (self.screen_width // 2 - setup_text.get_width() // 2, 320))

        elif self.state in [GameState.PLAYER1_TURN, GameState.PLAYER2_TURN]:
            turn_text = self.main_font.render(f"플레이어 {1 if self.state == GameState.PLAYER1_TURN else 2}의 차례입니다.", True, WHITE)
            screen.blit(turn_text, (self.screen_width // 2 - turn_text.get_width() // 2, 380))
        elif self.state == GameState.ROUND_END:
            if self.round_result:
                result_text = self.main_font.render(self.round_result, True, GOLD)
                screen.blit(result_text, (self.screen_width // 2 - result_text.get_width() // 2, 380))
            continue_text = self.score_font.render("Click to continue", True, WHITE)
            screen.blit(continue_text, (self.screen_width // 2 - continue_text.get_width() // 2, 410))
        elif self.state == GameState.GAME_OVER:
            screen.fill(DARK_RED)
            game_over_text = self.title_font.render("게임이 종료되었습니다.", True, GOLD)
            winner_text = None
            if self.player1_score > self.player2_score:
                winner_text = self.title_font.render("플레이어 1의 승리입니다.", True, WHITE)
            elif self.player2_score > self.player1_score:
                winner_text = self.title_font.render("플레이어 2의 승리입니다.", True, WHITE)
            else:
                winner_text = self.title_font.render("무승부입니다.", True, WHITE)
            screen.blit(game_over_text, (self.screen_width // 2 - game_over_text.get_width() // 2, 310))
            screen.blit(winner_text, (self.screen_width // 2 - winner_text.get_width() // 2, 400))
            restart_text = self.main_font.render("Press R to restart", True, WHITE)
            screen.blit(restart_text, (self.screen_width // 2 - restart_text.get_width() // 2, 490))