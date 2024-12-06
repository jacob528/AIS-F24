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
        for _ in range(self.simulations):
            self.simulate(board)

        legal_moves = list(board.legal_moves)
        move_scores = {}
        for move in legal_moves:
            board.push(move)
            move_scores[move] = self.q_values.get(board.fen(), 0) / self.visit_counts.get(board.fen(), 1)
            board.pop()

        # Retry if the selected move is illegal
        best_move = None
        while best_move is None or best_move not in legal_moves:
            best_move = max(move_scores, key=move_scores.get)

        if best_move in legal_moves:
            return best_move
        else:
            raise ValueError(f"Illegal move selected: {best_move}")


    def simulate(self, board):
        if board.is_game_over():
            return self.evaluate_terminal(board)
        
        board_fen = board.fen()
        if board_fen not in self.q_values:
            self.q_values[board_fen] = 0
            self.visit_counts[board_fen] = 0
            return self.evaluate(board)

        legal_moves = list(board.legal_moves)
        
        # Ensure the selected move is legal
        move = random.choice(legal_moves)  # Choose a legal move randomly
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
            print(f"Illegal move: {move_input}")
            return False
    except ValueError:
        print("Invalid move format!")
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

#Function to encode the different piece boards with 0s and 1s
def encode_board(board):
    planes = torch.zeros((14, 8, 8))  # 14 planes for each piece type (12 pieces + 2 additional planes for empty squares)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            piece_type = piece.piece_type - 1  # Piece type (1-6), so subtract 1 to index from 0
            color_offset = 0 if piece.color == chess.WHITE else 6  # Offset for color (0 for white, 6 for black)
            plane = piece_type + color_offset  # Determine the correct plane
            planes[plane, square // 8, square % 8] = 1  # Mark the square with the piece type
    return planes.unsqueeze(0)  # Add a batch dimension to the tensor

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
def self_play(model, simulations=10, mcts_simulations=1000):
    board = chess.Board()
    data = []
    mcts = MCTS(model, simulations=mcts_simulations)

    for _ in range(simulations):
        if board.is_game_over():
            break
        encoded_board = encode_board(board)
        best_move = mcts.select_move(board)
        if best_move in board.legal_moves:
            board.push(best_move)
            data.append((encoded_board, best_move, board.result()))
        else:
            print(f"Illegal move detected: {best_move}")
            break
        
        display_move_slowly(board, WINDOW, best_move)  # Visualize the move during self-play

    return data

#Training loop
def train_model(model, optimizer, criterion, epochs=5, simulations=10):
    for epoch in range(epochs):
        training_data = []
        for _ in range(simulations):
            training_data.extend(self_play(model, simulations=1))

        # Prepare dataset
        X = torch.cat([x[0] for x in training_data])  # Encoded boards
        policies = torch.cat([x[1].unsqueeze(0) for x in training_data])  # Policy vectors
        y = torch.tensor(
            [1 if result == "1-0" else -1 if result == "0-1" else 0 for _, _, result in training_data]
        )  # Game results

        if X.shape[0] != policies.shape[0] or X.shape[0] != y.shape[0]:
            raise ValueError("Inconsistent batch sizes.")

        # Training step
        optimizer.zero_grad()

        # Forward pass
        predicted_policies, predicted_value = model(X)

        # Calculate policy loss (cross-entropy)
        target_moves = policies.argmax(dim=1)
        policy_loss = nn.CrossEntropyLoss()(predicted_policies, target_moves)

        # Calculate value loss (mean squared error)
        value_loss = nn.MSELoss()(predicted_value.squeeze(), y.float())

        # Total loss
        loss = policy_loss + value_loss
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss.item():.4f}")

def display_move_slowly(board, window, move):
    # Placeholder for visualizing moves
    print(f"Processing move: {move}")
    board.push(move)  # Apply the move
    draw_board(window, board)  # Redraw the board
    pygame.display.flip()
    pygame.time.wait(500)  # Wait half a second to show the move

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