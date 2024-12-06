import itertools

# Constants for the board values
EMPTY = 0
X = 1
O = 2

# Define win conditions (rows, columns, diagonals)
WIN_COMBINATIONS = [
    [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
    [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
    [0, 4, 8], [2, 4, 6]              # Diagonals
]

def check_winner(board):
    """Check if there is a winner on the board."""
    for combo in WIN_COMBINATIONS:
        if board[combo[0]] == board[combo[1]] == board[combo[2]] != EMPTY:
            return board[combo[0]]  # Return the winning player's marker
    return None

def is_valid_board(board):
    """Check if a board is a valid game state."""
    x_count = board.count(X)
    o_count = board.count(O)

    # Rule 1: The number of X's must be equal to or one more than the number of O's
    if x_count < o_count or x_count > o_count + 1:
        return False

    # Rule 2: Only one player can have a winning combination
    winner = check_winner(board)
    if winner == X and x_count != o_count + 1:
        return False
    if winner == O and x_count != o_count:
        return False

    return True

def rotate_90(board):
    """Rotate the board 90 degrees clockwise."""
    return (board[6], board[3], board[0],
            board[7], board[4], board[1],
            board[8], board[5], board[2])

def reflect_horizontal(board):
    """Reflect the board horizontally."""
    return (board[2], board[1], board[0],
            board[5], board[4], board[3],
            board[8], board[7], board[6])

def generate_symmetries(board):
    """Generate all symmetries of the board (rotations + reflections)."""
    symmetries = set()

    # Original board and all rotations
    rotated_90 = rotate_90(board)
    rotated_180 = rotate_90(rotated_90)
    rotated_270 = rotate_90(rotated_180)

    symmetries.add(board)
    symmetries.add(rotated_90)
    symmetries.add(rotated_180)
    symmetries.add(rotated_270)

    # Reflections and their rotations
    reflected_h = reflect_horizontal(board)

    symmetries.add(reflected_h)
    symmetries.add(rotate_90(reflected_h))
    symmetries.add(rotate_90(rotate_90(reflected_h)))
    symmetries.add(rotate_90(rotate_90(rotate_90(reflected_h))))

    return symmetries

def canonical_form(board):
    """Return the lexicographically smallest representation of the board."""
    symmetries = generate_symmetries(board)
    return min(symmetries)

def dfs(board, turn, visited):
    """Explore all game states using Depth-First Search (DFS)."""
    # Skip if the board is already visited
    canonical_state = canonical_form(board)
    if canonical_state in visited:
        return

    visited.add(canonical_state)

    # If the game is over, no need to explore further
    if check_winner(board) or EMPTY not in board:
        return

    # Try placing X (1) or O (2) in all empty spaces
    for i in range(9):
        if board[i] == EMPTY:
            new_board = list(board)
            new_board[i] = turn
            new_turn = O if turn == X else X
            if is_valid_board(new_board):
                dfs(tuple(new_board), new_turn, visited)

def generate_all_unique_valid_states():
    """Generate all unique, valid Tic-Tac-Toe states (avoiding symmetries)."""
    initial_board = tuple([EMPTY] * 9)  # Start with an empty board
    visited = set()  # Store canonical forms of the visited states

    # Start DFS with X's turn (player 1 goes first)
    dfs(initial_board, X, visited)

    return visited

def write_to_file(visited, filename="tic_tac_toe_valid_states.txt"):
    """Write all unique, valid states to a file."""
        
    with open(filename, "w") as f:
        for state in visited:
            f.write(' '.join(map(str, state)) + '\n')

def main():
    unique_valid_states = generate_all_unique_valid_states()
    print(f"Total number of unique, valid states: {len(unique_valid_states)}")
    write_to_file(unique_valid_states)

if __name__ == "__main__":
    main()
