import random
import os

def generate_all_unique_valid_states():
    """Generate all unique, valid Tic-Tac-Toe states (avoiding symmetries)."""
    initial_board = tuple([EMPTY] * 9)  # Start with an empty board
    visited = set()  # Store canonical forms of the visited states

    # Start DFS with X's turn (player 1 goes first)
    dfs(initial_board, X, visited)

    return visited

def generate_tic_tac_toe_data():
    """
    Generate all possible valid Tic Tac Toe board states and their optimal moves.
    This function assumes the model is training to maximize wins or blocks.
    """
    def is_winning(board, player):
        """Check if the given player has won on the board."""
        win_conditions = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
            [0, 4, 8], [2, 4, 6]              # Diagonals
        ]
        return any(all(board[pos] == player for pos in condition) for condition in win_conditions)

    def find_optimal_move(board, player):
        """Find the best move for the player on the given board."""
        for move in range(9):
            if board[move] == 0:  # Check if the move is valid
                # Simulate the move
                board[move] = player
                if is_winning(board, player):
                    board[move] = 0  # Undo the move
                    return move  # Winning move
                board[move] = 0  # Undo the move

        # No winning move, block the opponent
        opponent = -player
        for move in range(9):
            if board[move] == 0:
                board[move] = opponent
                if is_winning(board, opponent):
                    board[move] = 0
                    return move  # Block opponent's winning move
                board[move] = 0

        # Otherwise, pick the first available spot
        for move in range(9):
            if board[move] == 0:
                return move

        return None  # No moves left

    data = []
    
    def generate_states(board, player):
        """Recursively generate valid board states and their optimal moves."""
        if is_winning(board, 1) or is_winning(board, -1) or 0 not in board:
            return  # Terminal state (win, loss, or draw)

        optimal_move = find_optimal_move(board, player)
        if optimal_move is not None:
            data.append((board[:], optimal_move))

        for move in range(9):
            if board[move] == 0:
                board[move] = player
                generate_states(board, -player)
                board[move] = 0

    # Start generating from an empty board
    empty_board = [0] * 9
    generate_states(empty_board, 1)
    return data

# Generate the data
data = generate_tic_tac_toe_data()


# Get the current working directory
current_directory = os.getcwd()

# Specify the path for the file in the current directory
file_path = os.path.join("TicTacToe", "tic_tac_toe_data.txt")

# Write to the file in the current directory
with open(file_path, "w") as f:
    for board, move in data:
        f.write(f"{board}, {move}\n")


print(f"Generated {len(data)} training examples.")
