import pygame
from NewCombos import generate_all_unique_valid_states, check_winner

class TicTacToePygame:
    def __init__(self):
        pygame.init()
        
        # Screen dimensions
        self.screen_size = 600
        self.cell_size = self.screen_size // 3
        self.screen = pygame.display.set_mode((self.screen_size, self.screen_size + 100))
        pygame.display.set_caption("Tic Tac Toe")
        
        # Colors
        self.white = (255, 255, 255)
        self.black = (0, 0, 0)
        self.red = (255, 0, 0)
        self.blue = (0, 0, 255)
        
        # Fonts
        self.font = pygame.font.Font(None, 80)
        self.reset_font = pygame.font.Font(None, 40)
        
        # Game state
        self.board = [0] * 9
        self.current_turn = 1  # 1 for X, 2 for O
        self.running = True
        
        # Pre-computed valid states
        self.valid_states = generate_all_unique_valid_states()

    def draw_board(self):
        self.screen.fill(self.white)

        # Draw grid
        for i in range(1, 3):
            pygame.draw.line(self.screen, self.black, (0, i * self.cell_size), (self.screen_size, i * self.cell_size), 5)
            pygame.draw.line(self.screen, self.black, (i * self.cell_size, 0), (i * self.cell_size, self.screen_size), 5)

        # Draw marks (X and O)
        for i, cell in enumerate(self.board):
            x = (i % 3) * self.cell_size
            y = (i // 3) * self.cell_size
            if cell == 1:
                text = self.font.render("X", True, self.red)
                self.screen.blit(text, (x + self.cell_size // 4, y + self.cell_size // 8))
            elif cell == 2:
                text = self.font.render("O", True, self.blue)
                self.screen.blit(text, (x + self.cell_size // 4, y + self.cell_size // 8))

        # Draw reset button
        reset_text = self.reset_font.render("Reset", True, self.black)
        pygame.draw.rect(self.screen, self.red, (self.screen_size // 3, self.screen_size + 10, self.screen_size // 3, 60))
        self.screen.blit(reset_text, (self.screen_size // 3 + 50, self.screen_size + 20))

    def check_click(self, pos):
        x, y = pos

        if y < self.screen_size:  # Inside game grid
            row, col = y // self.cell_size, x // self.cell_size
            idx = row * 3 + col
            if self.board[idx] == 0:  # Valid move
                self.board[idx] = self.current_turn

                winner = check_winner(self.board)
                if winner:
                    self.display_winner(winner)
                elif all(space != 0 for space in self.board):
                    self.display_draw()
                else:
                    self.current_turn = 2 if self.current_turn == 1 else 1
        else:  # Check if reset button was clicked
            if self.screen_size // 3 <= x <= 2 * self.screen_size // 3 and self.screen_size + 10 <= y <= self.screen_size + 70:
                self.reset_board()

    def display_winner(self, winner):
        self.draw_board()
        winner_text = "X Wins!" if winner == 1 else "O Wins!"
        text = self.font.render(winner_text, True, self.black)
        self.screen.blit(text, (self.screen_size // 6, self.screen_size // 3))
        pygame.display.flip()
        pygame.time.wait(2000)
        self.reset_board()

    def display_draw(self):
        self.draw_board()
        draw_text = "It's a Draw!"
        text = self.font.render(draw_text, True, self.black)
        self.screen.blit(text, (self.screen_size // 6, self.screen_size // 3))
        pygame.display.flip()
        pygame.time.wait(2000)
        self.reset_board()

    def reset_board(self):
        self.board = [0] * 9
        self.current_turn = 1

    def run(self):
        while self.running:
            self.draw_board()
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self.check_click(event.pos)

        pygame.quit()

if __name__ == "__main__":
    game = TicTacToePygame()
    game.run()
