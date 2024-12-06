import numpy as np

class ReversiGame:
    def __init__(self):
        self.board = np.zeros((8, 8), dtype=int)
        self.board[3, 3] = 1
        self.board[4, 4] = 1
        self.board[3, 4] = -1
        self.board[4, 3] = -1
        self.current_player = 1  # 1 for player 1, -1 for player 2

    def reset(self):
        self.__init__()
        return self.board

    def display(self):
        print("  " + " ".join([str(i) for i in range(8)]))
        for i, row in enumerate(self.board):
            print(i, " ".join(["." if x == 0 else "O" if x == 1 else "X" for x in row]))

    def is_valid_move(self, move):
        row, col = divmod(move, 8)
        if self.board[row, col] != 0:
            return False

        for dr, dc in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
            r, c = row + dr, col + dc
            if self._can_capture(r, c, dr, dc):
                return True
        return False

    def get_valid_moves(self):
        valid_moves = []
        for i in range(64):
            if self.is_valid_move(i):
                valid_moves.append(i)
        return valid_moves

    def get_random_valid_move(self):
        valid_moves = self.get_valid_moves()
        return np.random.choice(valid_moves) if valid_moves else None

    def step(self, move):
        if not self.is_valid_move(move):
            return self.board, -1, True  # Invalid move penalty

        row, col = divmod(move, 8)
        self.board[row, col] = self.current_player

        for dr, dc in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
            self._capture_pieces(row, col, dr, dc)

        self.current_player *= -1
        done = self.is_game_over()
        reward = self._get_reward() if done else 0

        return self.board.copy(), reward, done

    def is_game_over(self):
        if len(self.get_valid_moves()) > 0:
            return False

        self.current_player *= -1
        if len(self.get_valid_moves()) > 0:
            self.current_player *= -1
            return False

        self.current_player *= -1
        return True

    def _can_capture(self, row, col, dr, dc):
        r, c = row, col
        seen_opponent = False
        while 0 <= r < 8 and 0 <= c < 8:
            if self.board[r, c] == -self.current_player:
                seen_opponent = True
            elif self.board[r, c] == self.current_player and seen_opponent:
                return True
            elif self.board[r, c] == 0:
                break
            r += dr
            c += dc
        return False

    def _capture_pieces(self, row, col, dr, dc):
        r, c = row + dr, col + dc
        pieces_to_flip = []
        while 0 <= r < 8 and 0 <= c < 8:
            if self.board[r, c] == -self.current_player:
                pieces_to_flip.append((r, c))
            elif self.board[r, c] == self.current_player:
                for pr, pc in pieces_to_flip:
                    self.board[pr, pc] = self.current_player
                return
            else:
                break
            r += dr
            c += dc

    def _get_reward(self):
        player_score = np.sum(self.board == 1)
        opponent_score = np.sum(self.board == -1)
        if player_score > opponent_score:
            return 1 if self.current_player == 1 else -1
        elif player_score < opponent_score:
            return -1 if self.current_player == 1 else 1
        else:
            return 0