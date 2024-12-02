# Chess Model (ThunderByte Chess)

This chess model is created by Jacob L, and James W. It is a chess model we named "ThunderByte", and it is a convolutional neural network that
uses Monte Carlo Tree Search to evaluate positions and explore moves, and self-play reinforcement learning similar to/inspired by Alpha-Zero. 

## Components of the Model

- ThunderByte Model Class: 
The neural network takes in as input a state of the board that is encoded into a tensor, processes it through
the convolutional layers and has both a policy and value head.
The Policy Head output is the probability distribution of the position across all moves (reLU)
The Value Head outputs the prediction of this game using the tanh bounds (from 1 to -1, 1 being win, 0 being draw, -1 being lose)


- MCTS Class (Monte Carlo Tree Search):
The Monte Carlo Tree Search simulates different moves from positions based on the ThunderByte model's predictions 
As the MCTS occurs, it selects the move with the highest visit count and explores lines.


- Self Play function:
The self-play reinforcement learning method similar to alphazero is having the model face itself, and for each move
the MCTS pushes a move based on visit count, and the board's states and outcomes are saved as training data so the model can improve.

- Training Loop:
The training loop generates data through self play and updates the neural network based on the data (board outcomes/states).


## How to train

### Example initialization
model = ThunderByteCNN()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.MSELoss()

### Train the model
train_model(model, optimizer, criterion, epochs=10, simulations=10)








## Resources Used






