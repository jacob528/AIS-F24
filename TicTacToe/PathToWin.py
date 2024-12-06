from itertools import product

def is_winner(board, player):
    """Checks if the given player has won on the board."""
    win_conditions = [
        # Rows
        [0, 1, 2], [3, 4, 5], [6, 7, 8],
        # Columns
        [0, 3, 6], [1, 4, 7], [2, 5, 8],
        # Diagonals
        [0, 4, 8], [2, 4, 6]
    ]
    return any(all(board[i] == player for i in condition) for condition in win_conditions)

def is_valid_board(board):
    """Checks if the board is a valid Tic-Tac-Toe state."""
    x_count = board.count("X")
    o_count = board.count("O")

    # Rule 1: X must be equal to or one more than O
    if not (x_count == o_count or x_count == o_count + 1):
        return False

    # Rule 2: Both players cannot win simultaneously
    x_wins = is_winner(board, "X")
    o_wins = is_winner(board, "O")
    if x_wins and o_wins:
        return False

    # Rule 3: If X wins, X must have one more move than O
    if x_wins and x_count != o_count + 1:
        return False

    # Rule 4: If O wins, X and O must have the same number of moves
    if o_wins and x_count != o_count:
        return False

    return True

def generate_winning_sequences():
    """Generates valid winning boards with their game sequences."""
    symbols = [" ", "X", "O"]
    all_boards = product(symbols, repeat=9)

    winning_sequences = []  # To store winning boards with game sequences

    for board in all_boards:
        if is_valid_board(board):
            if is_winner(board, "X") or is_winner(board, "O"):
                sequence = []  # Track game sequence
                current_board = [" "] * 9  # Start from an empty board
                turn = 1

                # Recreate the game sequence by comparing board states
                for i, cell in enumerate(board):
                    if cell != " ":
                        move = (i // 3, i % 3)  # Convert index to row, col format
                        player = "X" if turn % 2 == 1 else "O"
                        sequence.append(f"Turn {turn}: {move} - {player}")
                        current_board[i] = cell
                        turn += 1

                # Store the winning board with its sequence
                winning_sequences.append((board, sequence))

    # Print example sequences
    print(f"Example Winning Sequences:")
    for idx, (board, sequence) in enumerate(winning_sequences[:5]):
        print(f"\nWinning Game #{idx + 1}")
        for i in range(0, 9, 3):
            print(f"{board[i]} | {board[i+1]} | {board[i+2]}")
            if i < 6:
                print("--+---+--")
        print("Winning Game Sequence:")
        for move in sequence:
            print(move)

    print(f"\nTotal winning boards: {len(winning_sequences)}")
    return winning_sequences

# Run the function
winning_sequences = generate_winning_sequences()
