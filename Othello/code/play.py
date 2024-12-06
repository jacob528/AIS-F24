import tkinter as tk
from tkinter import messagebox
import torch
from model import ReversiNet
from game import ReversiGame

class ReversiApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Reversi: Play against AI")
        self.board_size = 8
        self.cell_size = 60
        self.game = ReversiGame()

        # 加载模型
        self.model = ReversiNet(64, 64)
        self.model.load_state_dict(torch.load("reversi_model.pth", weights_only=True))
        self.model.eval()

        # 创建画布
        self.canvas = tk.Canvas(
            root, width=self.board_size * self.cell_size, height=self.board_size * self.cell_size
        )
        self.canvas.pack()

        # 绘制棋盘
        self.draw_board()
        self.update_board()

        # 绑定点击事件
        self.canvas.bind("<Button-1>", self.on_click)

    def draw_board(self):
        """绘制棋盘格子"""
        for i in range(self.board_size):
            for j in range(self.board_size):
                x0 = j * self.cell_size
                y0 = i * self.cell_size
                x1 = x0 + self.cell_size
                y1 = y0 + self.cell_size
                self.canvas.create_rectangle(x0, y0, x1, y1, outline="black")

    def highlight_valid_moves(self):
        """高亮当前玩家的合法落子位置"""
        self.canvas.delete("highlight")
        valid_moves = self.game.get_valid_moves()
        for move in valid_moves:
            row, col = divmod(move, self.board_size)
            x0 = col * self.cell_size + self.cell_size // 4
            y0 = row * self.cell_size + self.cell_size // 4
            x1 = x0 + self.cell_size // 2
            y1 = y0 + self.cell_size // 2
            self.canvas.create_oval(x0, y0, x1, y1, outline="green", width=2, tags="highlight")

    def update_board(self):
        """更新棋盘并高亮合法落子位置"""
        self.canvas.delete("piece")
        for i in range(self.board_size):
            for j in range(self.board_size):
                piece = self.game.board[i, j]
                if piece != 0:
                    x0 = j * self.cell_size + self.cell_size // 4
                    y0 = i * self.cell_size + self.cell_size // 4
                    x1 = x0 + self.cell_size // 2
                    y1 = y0 + self.cell_size // 2
                    color = "white" if piece == 1 else "black"
                    self.canvas.create_oval(x0, y0, x1, y1, fill=color, tags="piece")
        self.highlight_valid_moves()

    def on_click(self, event):
        """处理玩家点击棋盘事件"""
        col = event.x // self.cell_size
        row = event.y // self.cell_size
        move = row * self.board_size + col

        # 检查是否是合法落子
        if self.game.is_valid_move(move):
            self.game.step(move)
            self.update_board()

            if not self.game.is_game_over():
                self.ai_turn()

            # 检查游戏是否结束
            if self.game.is_game_over():
                self.end_game()

    def ai_turn(self):
        """AI 决策并更新棋盘"""
        state_tensor = torch.tensor((self.game.board + 1) / 2, dtype=torch.float32).flatten().unsqueeze(0)
        q_values = self.model(state_tensor)
        move = torch.argmax(q_values).item()

        if not self.game.is_valid_move(move):
            move = self.game.get_random_valid_move()

        if move is not None:
            self.game.step(move)
            self.update_board()

    def end_game(self):
        """游戏结束处理"""
        player_score = (self.game.board == 1).sum()
        ai_score = (self.game.board == -1).sum()
        result = f"Game Over!\nPlayer (White): {player_score}, AI (Black): {ai_score}\n"
        if player_score > ai_score:
            result += "You Win!"
        elif player_score < ai_score:
            result += "AI Wins!"
        else:
            result += "It's a Draw!"
        messagebox.showinfo("Game Over", result)
        self.root.quit()


if __name__ == "__main__":
    root = tk.Tk()
    app = ReversiApp(root)
    root.mainloop()
