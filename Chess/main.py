import random
import chess
import chess.engine
import pygame
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os
from model import ThunderByteCNN

# define piece images
PIECE_IMAGES = {
    'p': 'blackpawn.png',
    'n': 'blackknight.png',
    'b': 'blackbishop.png',
    'r': 'blackrook.png',
    'q': 'blackqueen.png',
    'k': 'blackking.png',
    'P': 'whitepawn.png',
    'N': 'whiteknight.png',
    'B': 'whitebishop.png',
    'R': 'whiterook.png',
    'Q': 'whitequeen.png',
    'K': 'whiteking.png',
}

# initialize pygame
pygame.init()

# set up display dimensions
SQUARE_SIZE = 64  # Each square is now 64x64 pixels
BOARD_SIZE = SQUARE_SIZE * 8
PANEL_WIDTH = 200
WIDTH = BOARD_SIZE + PANEL_WIDTH
HEIGHT = BOARD_SIZE
WINDOW = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("ThunderByte Chess")

# colors
LIGHT_SQUARE = (240, 217, 181)
DARK_SQUARE = (181, 136, 99)


#Monte Carlo Tree Search Algorithm for Model
#Allows the CNN to explore different random moves and figure out how good a move is (q values)
#Makes the CNN play better in simple terms
class MCTS:
    def __init__(self, model, simulations=1000):
        self.model = model
        self.simulations = simulations
        self.q_values = {}
        self.visit_counts = {}

    def select_move(self, board):
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            raise ValueError("No legal moves available.")
        
        move_scores = {}
        for move in legal_moves:
            board.push(move)
            fen = board.fen()
            score = self.q_values.get(fen, 0) / max(1, self.visit_counts.get(fen, 1))
            move_scores[move] = score
            board.pop()
        
        best_move = max(move_scores, key=move_scores.get)
        return best_move

    def simulate(self, board):
        if board.is_game_over():
            return self.evaluate_terminal(board)
        
        fen = board.fen()
        if fen not in self.q_values:
            self.q_values[fen] = 0
            self.visit_counts[fen] = 0
            return self.evaluate(board)

        legal_moves = list(board.legal_moves)
        move = random.choice(legal_moves)
        board.push(move)
        reward = -self.simulate(board)
        board.pop()

        self.q_values[fen] += reward
        self.visit_counts[fen] += 1
        return reward

    def evaluate(self, board):
        board_tensor = encode_board(board)
        policy, nn_eval = self.model(board_tensor)

        # Filter legal moves
        legal_moves = list(board.legal_moves)
        legal_move_indices = [move.to_square for move in legal_moves]
        move_probs = torch.softmax(policy, dim=1).squeeze(0)[legal_move_indices]

        # Normalize probabilities
        move_probs = move_probs / move_probs.sum()
        
        white_material, black_material = calculate_material(board)
        material_eval = (white_material - black_material) / 10.0

        return nn_eval.item() * 0.7 + material_eval * 0.3


    def evaluate_terminal(self, board):
        if board.is_checkmate():
            return -1  # Loss
        elif board.is_stalemate() or board.is_insufficient_material():
            return 0  # Draw
        return 1  # Win

#Function to generate a random board for neural network to analyze
def generate_board(board, depth):
    if (depth == 0): 
        return board
    else:
        move = random.choice(list(board.legal_moves))
        board.push(move)
        print()
        print("Move:", move)
        print(board)
        generate_board(board, depth - 1)

def debug_board_state(board):
    print("Current board FEN:", board.fen())
    print("Legal moves:", list(board.legal_moves))

def draw_board(window, board):
    """
    Draws the chessboard and pieces onto the Pygame window.
    
    Parameters:
        window (pygame.Surface): The Pygame window to draw on.
        board (chess.Board): The chess board state from python-chess.
    """
    # Loop through all squares (0 to 63)
    for rank in range(8):
        for file in range(8):
            square = rank * 8 + file
            
            # Determine the color of the square
            is_light_square = (rank + file) % 2 == 0
            color = LIGHT_SQUARE if is_light_square else DARK_SQUARE
            
            # Draw the square
            pygame.draw.rect(
                window,
                color,
                pygame.Rect(file * SQUARE_SIZE, rank * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
            )
            
            # Draw the piece if one is on the square
            piece = board.piece_at(square)
            if piece:
                piece_image = PIECE_IMAGES.get(piece.symbol())
                if piece_image:
                    window.blit(
                        piece_image,
                        (file * SQUARE_SIZE, rank * SQUARE_SIZE)
                    )

def draw_board_with_panel(window, board, player_input):
    """
    Draw the chessboard and the side panel for player input.
    
    Parameters:
        window (pygame.Surface): The Pygame window to draw on.
        board (chess.Board): The chess board state from python-chess.
        player_input (str): Text input from the player for moves.
    """
    # Draw the chessboard
    for rank in range(8):
        for file in range(8):
            square_color = LIGHT_SQUARE if (rank + file) % 2 == 0 else DARK_SQUARE
            pygame.draw.rect(window, square_color, (file * SQUARE_SIZE, rank * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
            
            # Draw pieces
            piece = board.piece_at(chess.square(file, 7 - rank))  # Convert to 0-based indexing
            if piece:
                piece_image = PIECE_IMAGES[piece.symbol()]
                window.blit(piece_image, (file * SQUARE_SIZE, rank * SQUARE_SIZE))
    
    # Draw the side panel
    pygame.draw.rect(
        window,
        (50, 50, 50),  # Gray color
        (BOARD_SIZE, 0, PANEL_WIDTH, HEIGHT)  # Spanning the right-hand side
    )  # Background for panel
    font = pygame.font.Font(None, 36)
    text = font.render("Player Move:", True, (255, 255, 255))
    window.blit(text, (BOARD_SIZE + 20, 20))
    
    # Render player input
    input_text = font.render(player_input, True, (200, 200, 200))
    window.blit(input_text, (WIDTH - 120, 60))

def handle_player_move(board, move_input):
    try:
        move = chess.Move.from_uci(move_input)
        if move in board.legal_moves:
            board.push(move)
            return True
        else:
            print(f"Illegal move: {move_input} - Not in board.legal_moves.")
            return False
    except ValueError:
        print(f"Invalid move format: {move_input}")
        return False


# load piece images
def load_images():
    for piece, filename in PIECE_IMAGES.items():
        try:
            # Load the image file associated with the piece
            image = pygame.image.load(f'./assets/{filename}')
            # Scale the image to fit the square size
            image = pygame.transform.scale(image, (SQUARE_SIZE, SQUARE_SIZE))
            # Update the dictionary with the Pygame surface
            PIECE_IMAGES[piece] = image
        except FileNotFoundError:
            print(f"Error: Image file for {piece} not found at './assets/{filename}'!")

player_input = ""

# Add this function to draw the button
# Add this function to draw the button
def draw_button(window, x, y, width, height, text):
    """Draw a button with specified dimensions and text."""
    pygame.draw.rect(window, (90, 130, 180), (x, y, width, height))  # Button background
    pygame.draw.rect(window, (50, 50, 50), (x, y, width, height), 2)  # Button border
    font = pygame.font.Font(None, 28)
    text_surface = font.render(text, True, (255, 255, 255))  # Button text
    text_rect = text_surface.get_rect(center=(x + width // 2, y + height // 2))
    window.blit(text_surface, text_rect)

# Position button in the side panel
button_x = BOARD_SIZE + 20  # Align with the side panel
button_y = 120  # Below the Player Move text
button_width = 160
button_height = 40

def encode_board(board):
    """
    Encodes a chess board position into a tensor suitable for input to a neural network.
    """
    planes = torch.zeros((14, 8, 8))
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            piece_type = piece.piece_type - 1
            color_offset = 0 if piece.color == chess.WHITE else 6
            plane = piece_type + color_offset
            planes[plane, square // 8, square % 8] = 1
    return planes.unsqueeze(0)


#Determine the Material on both sides 
def calculate_material(board):
    # Define the piece values
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0
    }

    # Initialize the material score
    white_material = 0
    black_material = 0

    # Iterate over all pieces on the board
    for square, piece in board.piece_map().items():
        # Get the value of the piece type (e.g., PAWN, KNIGHT, etc.)
        piece_value = piece_values.get(piece.piece_type, 0)

        # Add to the appropriate color's total
        if piece.color == chess.WHITE:
            white_material += piece_value
        else:
            black_material += piece_value

    return white_material, black_material

def apply_legal_move_mask(policy, legal_move_mask):
    """
    Apply the legal move mask to the policy output to ensure that only legal moves are considered.
    
    Parameters:
    - policy: The policy output from the neural network (size: [batch_size, num_possible_moves])
    - legal_move_mask: A binary mask where 1 indicates a legal move and 0 indicates an illegal move
    
    Returns:
    - masked_policy: The masked policy output with illegal moves set to a very low value (e.g., -inf)
    """
    masked_policy = policy + (1 - legal_move_mask) * -1e9  # Set illegal moves to a very low value
    return masked_policy

#Self play function
def self_play(model, simulations=10):
    board = chess.Board()
    data = []
    for _ in range(simulations):
        mcts = MCTS(model)
        try:
            move = mcts.select_move(board)
            if move not in board.legal_moves:
                raise ValueError(f"Move {move} is illegal.")
            
            # Collect training data
            board_tensor = encode_board(board)
            data.append((board_tensor, move))
            
            # Apply move and display
            board.push(move)
            draw_board(WINDOW, board)
            pygame.display.flip()
            pygame.time.wait(500)
        except ValueError as e:
            print(f"Error during MCTS simulation: {e}")
            break
    return data


#Training loop
def train_model(model, optimizer, criterion, epochs=5, simulations=10):
    for epoch in range(epochs):
        training_data = []
        for _ in range(simulations):
            training_data.extend(self_play(model, simulations=1))

        # Prepare dataset
        X = torch.cat([x[0] for x in training_data])
        y = torch.tensor([1 if result == "1-0" else -1 if result == "0-1" else 0 for _, result in training_data])

        optimizer.zero_grad()
        predicted_policies, predicted_value = model(X)
        policy_loss = nn.CrossEntropyLoss()(predicted_policies, y.argmax(dim=1))
        value_loss = nn.MSELoss()(predicted_value.squeeze(), y.float())
        loss = policy_loss + value_loss

        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss.item():.4f}")


def display_move_slowly(board, window, move):
    try:
        if move not in board.legal_moves:
            raise ValueError(f"Move {move} is not legal on the current board.")
        
        board.push(move)  # Apply the move
        draw_board(window, board)
        pygame.display.flip()
        pygame.time.wait(500)  # Pause for half a second
    except ValueError as e:
        print(f"Error in move display: {e}")


def train_and_display_moves(model, optimizer, criterion, board, window):
    """Train the model and display moves on the UI during training."""
    training_data = self_play(model, simulations=10)  # Self-play for data generation

    for data in training_data:
        encoded_board, _, _ = data  # Extract board from data
        best_move = data[0]  # Get the best move from the data (ensure it's a chess.Move)

        # Ensure best_move is a chess.Move
        if isinstance(best_move, torch.Tensor):
            best_move = chess.Move.from_uci(best_move.item())  # Convert from UCI string if needed

        # Display the move slowly on the board
        display_move_slowly(board, window, best_move, speed=50)

        # After each move, perform the training step
        train_model(model, optimizer, criterion, epochs=1, simulations=1)



#Main Function
if __name__ == "__main__":
    # Check if assets folder exists
    if not os.path.exists('assets'):
        print("Error: 'assets' folder not found!")
    else:
        print("'assets' folder is accessible!")

    # Initialize the board and load images
    board = chess.Board()
    load_images()

    model = ThunderByteCNN()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.MSELoss()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    if handle_player_move(board, player_input):
                        player_input = ""  # Clear input on valid move
                    else:
                        print("Invalid or illegal move.")
                elif event.key == pygame.K_BACKSPACE:
                    player_input = player_input[:-1]  # Remove last character
                else:
                    player_input += event.unicode  # Add typed character to input
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Check if mouse click is within the button bounds
                mouse_x, mouse_y = event.pos
                if button_x <= mouse_x <= button_x + button_width and button_y <= mouse_y <= button_y + button_height:
                    train_and_display_moves(model, optimizer, criterion, board, WINDOW)

        # Redraw the board, side panel, and button
        draw_board_with_panel(WINDOW, board, player_input)
        
        # Draw the "Train" button
        draw_button(WINDOW, button_x, button_y, button_width, button_height, "Train")
        
        pygame.display.flip()

    pygame.quit()