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
WIDTH, HEIGHT = 640, 512
SQUARE_SIZE  = WIDTH // 8
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
        for _ in range(self.simulations):
            self.simulate(board)

        legal_moves = list(board.legal_moves)
        move_scores = {}
        for move in legal_moves:
            board.push(move)
            move_scores[move] = self.q_values.get(board.fen(), 0) / self.visit_counts.get(board.fen(), 1)
            board.pop()

        best_move = max(move_scores, key=move_scores.get)
        return best_move

    def simulate(self, board):
        if board.is_game_over():
            return self.evaluate_terminal(board)
        
        board_fen = board.fen()
        if board_fen not in self.q_values:
            self.q_values[board_fen] = 0
            self.visit_counts[board_fen] = 0
            return self.evaluate(board)

        legal_moves = list(board.legal_moves)
        move = random.choice(legal_moves)
        board.push(move)
        reward = -self.simulate(board)
        board.pop()

        self.q_values[board_fen] += reward
        self.visit_counts[board_fen] += 1
        return reward

    def evaluate(self, board):
        board_tensor = encode_board(board)
        policy, nn_eval = self.model(board_tensor)

        # Convert policy to move probabilities
        legal_moves = list(board.legal_moves)
        move_probs = torch.softmax(policy, dim=1).squeeze(0)  # Softmax for move probabilities

        # Calculate material evaluation
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
    pygame.draw.rect(window, (50, 50, 50), (WIDTH - 128, 0, 128, HEIGHT))  # Background for panel
    font = pygame.font.Font(None, 36)
    text = font.render("Player Move:", True, (255, 255, 255))
    window.blit(text, (WIDTH - 120, 20))
    
    # Render player input
    input_text = font.render(player_input, True, (200, 200, 200))
    window.blit(input_text, (WIDTH - 120, 60))

def handle_player_move(board, move_input):
    """
    Processes a move input by the player and updates the board.
    
    Parameters:
        board (chess.Board): The current chess board state.
        move_input (str): Player's move in UCI or algebraic notation.
    
    Returns:
        bool: True if the move is valid, False otherwise.
    """
    try:
        move = chess.Move.from_uci(move_input)
        if move in board.legal_moves:
            board.push(move)
            return True
        else:
            print("Illegal move!")
            return False
    except ValueError:
        print("Invalid move format!")
        return False


#Function to encode the different piece boards with 0s and 1s
def encode_board(board):
    # 14 planes, 8x8 board
    # 12 planes for each piece on the board
    # 1 board for current turn (all 0s is black, 1s is white)
    # 1 board for castling rights/rules etc...
    planes = torch.zeros((14, 8, 8))

    #Loop through each square, check for pieces at each square, encode the planes with 0s and 1s
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            plane = (piece.piece_type - 1) + (0 if piece.color == chess.WHITE else 6)
            planes[plane, square // 8, square % 8] = 1
        else:
            # Optional: You can add a comment or logic here for clarity
            pass  # No piece on this square, leave it as zeros
    # After encoding, return planes with unsqueeze (which adds a dimension to the planes tensor)
    return planes.unsqueeze(0)


#Determine the Material on both sides 
def calculate_material(board):
    # Define the piece values
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9
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

#Self play function
def self_play(model, simulations=10):
    board = chess.Board()
    data = []

    for _ in range(simulations):
        if board.is_game_over():
            break
        encoded_board = encode_board(board)
        
        # Evaluate each move using the model and store move probabilities
        move_evals = []
        for move in board.legal_moves:
            board.push(move)
            policy, move_eval = model(encode_board(board))
            move_evals.append((move, move_eval.item(), policy))
            board.pop()

        # Select the best move based on the neural network evaluation
        best_move = max(move_evals, key=lambda x: x[1])[0]
        best_policy = move_evals[0][2]  # Store the policy vector for the selected move

        data.append((encoded_board, best_policy, board.result()))  # Store board, policy, and result
        board.push(best_move)

    return data


#Training loop
def train_model(model, optimizer, criterion, epochs=5, simulations=10):
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        training_data = []
        for _ in range(simulations):
            training_data.extend(self_play(model, simulations=1))

        # Prepare dataset
        X = torch.cat([x[0] for x in training_data])  # Encoded boards
        policies = torch.stack([x[1] for x in training_data])  # Policy vectors
        y = torch.tensor([1 if result == "1-0" else -1 if result == "0-1" else 0 for _, _, result in training_data])  # Game results

        # Training step
        optimizer.zero_grad()
        
        # Forward pass
        predicted_policies, predicted_value = model(X)
        
        # Calculate policy loss (cross-entropy)
        policy_loss = nn.CrossEntropyLoss()(predicted_policies, policies.argmax(dim=1))  # Argmax to get the target move
        
        # Calculate value loss (mean squared error)
        value_loss = nn.MSELoss()(predicted_value.squeeze(), y.float())
        
        # Total loss
        loss = policy_loss + value_loss
        loss.backward()
        optimizer.step()

        print(f"Loss: {loss.item():.4f}")

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
def draw_button(window, x, y, width, height, text):
    """Draw a button with specified dimensions and text."""
    pygame.draw.rect(window, (70, 130, 180), (x, y, width, height))  # Button background
    font = pygame.font.Font(None, 24)
    text_surface = font.render(text, True, (255, 255, 255))  # Button text
    text_rect = text_surface.get_rect(center=(x + width // 2, y + height // 2))
    window.blit(text_surface, text_rect)

# Add this function to handle training
def train_and_display_moves(model, optimizer, criterion, board, window):
    """Train the model and display moves on the UI during training."""
    training_data = self_play(model, simulations=10)  # Self-play for data generation

    for data in training_data:
        board_tensor, _, _ = data  # Extract board from data
        board_copy = board.copy()
        # Decode board_tensor back to a python-chess Board (if necessary)

        # Draw the board to reflect moves
        draw_board(window, board_copy)
        pygame.display.flip()
        pygame.time.wait(500)  # Delay to visualize the move

    # Perform training step
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
                if 500 <= mouse_x <= 600 and 400 <= mouse_y <= 440:  # Button bounds
                    train_and_display_moves(model, optimizer, criterion, board, WINDOW)

        # Redraw the board, side panel, and button
        draw_board_with_panel(WINDOW, board, player_input)
        draw_button(WINDOW, 500, 400, 100, 40, "Train")  # Button for training
        pygame.display.flip()

    pygame.quit()