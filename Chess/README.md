# Chess Model (ThunderByte Chess)

This chess model is created by Jacob L, and James W. It is a chess model we named "ThunderByte", and it is a convolutional neural network that
uses Monte Carlo Tree Search to evaluate positions and explore moves, and self-play reinforcement learning similar to/inspired by Alpha-Zero. 

## Procedure for building Chess Model

### Build the convolutional neural network

A convolutional neural network is a specific type of neural network that is very good at image/audio recognition.

For chess, it is very good to use a convolutional neural network because it also exceeds at pattern recognition, and 
spatial relationships which make up all fo chess. There are many, many different patters in chess, like the first few moves of an opening
that people can study, threats, checkmating patterns, and pawn structures.

In convolutional neural networks, there is a concept of kerneling/filtering which our model uses in it's layers:

Kernels are small matrices that perform convolutional operations (essentially matrix multiplication) that keep track of things like
a rook holding down the file and other attacking/defending patterns, and a filter is just the combination of these kernels, and they work
to extract diverse features from a baord (edges of the board, the center of the board, different patterns essentially) 
(This is also very similar to span and basis in linear algebra, just have different applications)

Our model also has two heads which are pivotal to its function: 
A Policy and Value Head, which we build to get different distributions and figure out if a position is winning position.
These are very pivotal for data analysis later and training.

### Build MCTS

Monte Carlo Tree Search is a very important topic in logic, and it is very important for our model to be similar to AlphaZero 
in implementation. MCTS essentially handles how our model thinks about/evaluates positions and explores different move sets (Like an expanding
binary tree).

Our model is assigned with a reward that changes our policy and value heads. This can be done by finding the material weighed by its current 
data that it will get by exploring different positions and pushing moves to the board.

There is a qvalue (which in qlearning a type of reinforcment learning, is the total expected reqard our model (agent) recieves in the position) and visit count which are assigned to our model to better track each position our model encounters as it is searching.


### Training Step/UI

Our training loop works by taking in our model, optimizer, criterion, epochs and simulation count.

First we do a loop over epochs (our current training iterations) to allow our model to eventually improve over time.

Then we generate our training data by grabbing from our self play function and current training data, 
and we also need our encoded board and policy vectors which are used to calculate loss for the model's policy and value heads. This loss is important because we evaluate how well our model did based on if the policies match (cross-entropy loss), and if our model is accurate in predicting a games outcome (with mean squared error).

Our important part is backpropogation, which goes back and adjust the layer's biases and weights using the gradient of our losses. We are trying to minimize these losses to improve, and we also track the total loss with a print statement.

Our UI is important for us to track the training progress in a more viewable format. The chess library in python already contains a way to see different moves on the chess board, but through pygame it is better to look at and see the process unfold.

## Components/Review of the Chess Model

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


## How to train (Work in Progress)

(There is supposed to be a button that will appear on the UI when running the game that will allow you to train and watch
the training step in real time, but this is still in progress)

Add the following code to main:

### Example initialization
model = ThunderByteCNN()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.MSELoss()

### Train the model
train_model(model, optimizer, criterion, epochs=10, simulations=10)
-adjust simulations as neccesary

## Papers / Resources Used
- Papers:
Chess AI: Competing Paradigms for Machine Intelligence https://arxiv.org/pdf/2109.11602

- Resources/Videos:
How Alpha Zero used Reinforcement Learning to Master Chess (12.5) https://youtu.be/ikDgyD7nVI8?si=uAjWJxLoDmAp0Lzk
Creating a Chess AI with TensorFlow - https://www.youtube.com/watch?v=ffzvhe97J4Q (for inspiration, not using tensorflow)
Neural Network Series - https://www.youtube.com/@3blue1brown/playlists
Train Your Own Chess AI - https://towardsdatascience.com/train-your-own-chess-ai-66b9ca8d71e4

## Note
This document is subject to change as the training process continues, and as the code updates.





