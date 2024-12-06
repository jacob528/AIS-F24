import pygame
from nn import Agent
import numpy as np

# Initialize Pygame
pygame.init()

# Constants
BOARD_SIZE = 8  # 8x8 board
CELL_SIZE = 80  # Cell size in pixels
SCREEN_SIZE = BOARD_SIZE * CELL_SIZE
WHITE, BLACK, EMPTY, HIGHLIGHT = 1, -1, 0, 2

# Ruleset for training / testing purposes
RULESET = "AI v AI"
#RULESET = "P v AI"
#RULESET = "P v P"

# Font builder
font = pygame.font.Font(None, 36)

# Colors
GREEN_COLOR = (34, 139, 34)
WHITE_COLOR = (255, 255, 255)
BLACK_COLOR = (0, 0, 0)
BLUE_COLOR = (0, 0, 255)
HIGHLIGHT_COLOR = (255, 215, 0)

# Initialize screen
screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
pygame.display.set_caption("Othello")

# Initialize an empty 8x8 board
board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)

# Place initial pieces
board[3][3] = WHITE
board[3][4] = BLACK
board[4][3] = BLACK
board[4][4] = WHITE

def draw_dialogue():
    screen.fill(BLUE_COLOR)

    # Draw pick color dialogue
    input_box = pygame.Rect((150, 100), (350, 150))
    white_button = pygame.Rect((175, 200), (80, 25))
    black_button = pygame.Rect((400, 200), (80, 25))
    response_box = pygame.Rect((150, 400), (100, 75))
    
    input_text = font.render("Choose your color:", True, GREEN_COLOR)
    white_text = font.render("White", True, BLACK_COLOR)
    black_text = font.render("Black", True, WHITE_COLOR)
    
    pygame.draw.rect(screen, BLUE_COLOR, input_box)
    pygame.draw.rect(screen, WHITE_COLOR, white_button)
    pygame.draw.rect(screen, BLACK_COLOR, black_button)

    screen.blit(input_text, (input_box.x + 75, input_box.y))
    screen.blit(white_text, (white_button.x, white_button.y))
    screen.blit(black_text, (black_button.x, black_button.y))

    continueFlag = False
    pygame.display.flip()

    while continueFlag == False:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                continueFlag = True
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if black_button.collidepoint(event.pos):
                    response_text = font.render("Good Luck!", True, BLACK_COLOR)
                    pygame.draw.rect(screen, BLUE_COLOR, response_box)
                    screen.blit(response_text, (response_box.x, response_box.y))
                    pygame.display.flip()
                    return False            # black is not AI
                elif white_button.collidepoint(event.pos):
                    response_text = font.render("Good Luck!", True, BLACK_COLOR)
                    pygame.draw.rect(screen, BLUE_COLOR, response_box)
                    screen.blit(response_text, (response_box.x, response_box.y))
                    pygame.display.flip()
                    return True             # black is AI
                else:
                    response_text = font.render("Click a box", True, GREEN_COLOR)
                    pygame.draw.rect(screen, BLUE_COLOR, response_box)
                    screen.blit(response_text, (response_box.x, response_box.y))
                    pygame.display.flip()

def draw_board():
    screen.fill(GREEN_COLOR)
    
    # Draw grid
    for x in range(1, BOARD_SIZE):
        pygame.draw.line(screen, BLACK_COLOR, (x * CELL_SIZE, 0), (x * CELL_SIZE, SCREEN_SIZE))
        pygame.draw.line(screen, BLACK_COLOR, (0, x * CELL_SIZE), (SCREEN_SIZE, x * CELL_SIZE))

    # Draw pieces
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if board[row][col] == WHITE:
                pygame.draw.circle(screen, WHITE_COLOR, (col * CELL_SIZE + CELL_SIZE // 2, row * CELL_SIZE + CELL_SIZE // 2), CELL_SIZE // 2 - 5)
            elif board[row][col] == BLACK:
                pygame.draw.circle(screen, BLACK_COLOR, (col * CELL_SIZE + CELL_SIZE // 2, row * CELL_SIZE + CELL_SIZE // 2), CELL_SIZE // 2 - 5)

def is_valid_move(board, row, col, color):
    if board[row][col] != EMPTY:
        return False
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]
    for dr, dc in directions:
        r, c = row + dr, col + dc
        count = 0
        while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r][c] == -color:
            r += dr
            c += dc
            count += 1
        if count > 0 and 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r][c] == color:
            return True
    return False

def place_piece(board, row, col, color):
    board[row][col] = color
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]
    for dr, dc in directions:
        r, c = row + dr, col + dc
        flip_positions = []
        while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r][c] == -color:
            flip_positions.append((r, c))
            r += dr
            c += dc
        if flip_positions and 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r][c] == color:
            for fr, fc in flip_positions:
                board[fr][fc] = color
    return True

def get_score(board):
    white_score = np.sum(board == WHITE)
    black_score = np.sum(board == BLACK)
    return white_score, black_score

def is_game_over(board):
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if board[row][col] == EMPTY and (is_valid_move(board, row, col, BLACK) or is_valid_move(board, row, col, WHITE)):
                return False
    return True

def handle_player_move(board, current_color):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False  # Stop running
        elif event.type == pygame.MOUSEBUTTONDOWN:
            x, y = event.pos
            row, col = y // CELL_SIZE, x // CELL_SIZE
            if is_valid_move(board, row, col, current_color):
                place_piece(board, row, col, current_color)
                return -current_color  # Switch turns
    return current_color

def handle_ai_move(agent, board):
    row, col = agent.placeMove(board)
    if is_valid_move(board, row, col, agent.color):
        place_piece(board, row, col, agent.color)
        return True
    return False

def end_game(board):
    white_score, black_score = get_score(board)
    print("Game Over")
    print(f"White: {white_score}, Black: {black_score}")
    return False  # Stop running

def main():
    clock = pygame.time.Clock()
    current_color = BLACK
    running = True

    if RULESET == "P v AI":
        blackIsAI = draw_dialogue()

    print(f"RULESET: {RULESET}")

    while running:
        draw_board()
        pygame.display.flip()

        if RULESET == "P v P":
            for event in pygame.event.get():

                if event.type == pygame.QUIT:
                    running = False  # Stop running

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = event.pos
                    row, col = y // CELL_SIZE, x // CELL_SIZE

                    if is_valid_move(board, row, col, current_color):
                        place_piece(board, row, col, current_color)
                        current_color = -current_color  # Switch turns


        elif RULESET == "AI v AI":
            agents = [Agent(), Agent()]
            for agent in agents:
                if handle_ai_move(agent, board) and is_game_over(board):
                    running = end_game(board)
                    break


        else: # RULESET == "P v AI"
            player_turn = True  # True = Player's turn, False = AI's turn
            agent = Agent()

            if blackIsAI:
                player_turn = False  # AI goes first

            if player_turn:
                current_color = handle_player_move(board, current_color)
                if not current_color:  # If handle_player_move returned False
                    running = False
            else:
                if handle_ai_move(agent, board) and is_game_over(board):
                    running = end_game(board)

            # Toggle player turn
            player_turn = not player_turn

        '''
        if is_game_over(board):
            running = False  # Exit the loop to end the game
            white_score, black_score = get_score(board)
            print("Game Over")
            print(f"White: {white_score}, Black: {black_score}")
        '''

        clock.tick(30)
    pygame.quit()

if __name__ == "__main__":
    main()