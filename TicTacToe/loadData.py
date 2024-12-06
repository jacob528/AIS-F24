import torch
import os
from torch.utils.data import Dataset, DataLoader

# Define a custom Dataset class for Tic Tac Toe
class TicTacToeDataset(Dataset):
    def __init__(self, file_path):
        """Initialize the dataset by loading and parsing the data from the file."""
        self.data = []
        # Specify the path for the file in the current directory
        file_path = os.path.join("TicTacToe", "tic_tac_toe_data.txt")

        with open(file_path, "r") as f:
            for line in f:
                board_str = line.strip()
                move_str = board_str[-1]
                the_board1 = board_str[1:-4]
                the_board = the_board1.split(",")
                board = list(map(int,the_board))
                move = int(move_str)
                self.data.append((board, move))
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Return a single data sample."""
        board, move = self.data[idx]
        board_tensor = torch.tensor(board, dtype=torch.float32)
        move_tensor = torch.tensor(move, dtype=torch.long)
        return board_tensor, move_tensor

# Load the dataset from the file
dataset = TicTacToeDataset("tic_tac_toe_data.txt")

# Optional: Split into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Use DataLoader for batching
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

print(f"Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples.")
