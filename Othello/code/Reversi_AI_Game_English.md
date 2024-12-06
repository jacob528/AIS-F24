
---

# Reversi AI Game

## Project Overview
Reversi (Othello) is a classic board strategy game. This project implements an AI for Reversi using a **Reinforcement Learning Policy Network**. Users can choose to play against the AI or allow the AI to train itself. A graphical user interface (GUI) is provided for an intuitive gaming experience with the AI.

---

## Features
- **AI Self-Training**:
  - Train the AI model using reinforcement learning (Q-learning).
  - Support for saving and loading trained models.
- **Human vs AI Gameplay**:
  - Includes a graphical user interface (GUI).
  - Highlights valid moves for the player.
  - Dynamically updates the game board.
- **Easy Extensibility**:
  - Built using PyTorch, allowing flexibility in adjusting the model architecture and training strategies.

---

## Project Structure

```
reversi_ai_project/
│
├── game.py              # Reversi game logic
├── train.py             # AI model training
├── play.py              # Graphical user interface (GUI)
├── model.py             # Policy neural network (Q-learning)
├── replay_buffer.py     # Replay buffer
├── main.py              # Entry point for training or gameplay
├── reversi_model.pth    # Saved model (generated after training)
└── README.md            # Documentation
```

---

## Installation Guide

### Requirements

1. Install Python 3.7 or later.
2. Install dependencies:
   ```bash
   pip install torch tkinter numpy
   ```

---

## Usage

### 1. Run the Project

Run the main program from the terminal:
```bash
python main.py
```

### 2. Instructions

#### **Main Menu**
After launching the program, you can choose:
1. `Train AI`: Train the AI to generate a strategy model.
2. `Play against AI`: Enter the GUI to play against the AI.

#### **AI Training**
Select `1` to start training the AI:
- The training loss is displayed every 100 games.
- The trained model is saved as `reversi_model.pth`.

#### **Human vs AI Gameplay**
Select `2` to launch the GUI:
- White (O) is the player, and Black (X) is the AI.
- Click valid positions on the board to make a move.
- Invalid moves will not respond.
- The board will highlight valid positions for the player.

---

## Game Rules
1. The board size is 8x8.
2. Players take turns:
   - White (player) moves first, followed by Black (AI).
   - Each move must flip at least one opponent's piece; otherwise, the move is invalid.
3. The game ends:
   - When neither side has valid moves.
   - The player with the most pieces on the board wins.

---

## Notes

1. If the training loss is too high or the model performs poorly, consider:
   - Increasing the number of training iterations.
   - Adjusting the reward function.
2. If the model file is lost, retraining is required.

---

## Future Improvements

- Add a player-vs-player mode.
- Implement a model evaluation tool to analyze training effectiveness.
- Enhance the GUI with features like board backgrounds and dynamic effects.

---

## License
This project is open-sourced under the MIT License. Feel free to use and modify it!

---

If you have any questions or suggestions, please feel free to contact us or submit an issue!
