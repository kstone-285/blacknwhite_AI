import pygame
from game import BlackWhiteGame
from utils import GameState

# Pygame 초기화
pygame.init()

# 화면 설정
screen_width, screen_height = 1000, 800
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("흑과백")

# 게임 인스턴스 생성
pygame.mixer.music.load('images\\y2mate.mp3')
pygame.mixer.music.set_volume(0.1)
pygame.mixer.music.play(-1)

clock = pygame.time.Clock()
game = BlackWhiteGame()
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
             running = False

        if game.state == GameState.MAIN_MENU:
            
            vs_ai_rect, vs_player_rect, exit_rect = game.draw_main_menu(screen)

            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                    
                if vs_ai_rect.collidepoint(mouse_pos):
                    game = BlackWhiteGame(is_ai_mode=True)
                    game.state = GameState.SETUP_PLAYER1
                elif vs_player_rect.collidepoint(mouse_pos):
                    game = BlackWhiteGame(is_ai_mode=False)
                    game.state = GameState.SETUP_PLAYER1
                elif exit_rect.collidepoint(mouse_pos):
                    running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if game.state == GameState.ROUND_END:
                game.round += 1
                game.reset_round()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r and game.state == GameState.GAME_OVER:
                game.reset_game()
        game.handle_event(event)

    if game.state == GameState.MAIN_MENU:
        game.draw_main_menu(screen)
    else:
        game.update()
        game.draw(screen)
        
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
