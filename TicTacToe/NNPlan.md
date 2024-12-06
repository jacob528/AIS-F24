Neural Network Layout

Input Layer:

- Number of neurons: 9
- Purpose: Encodes the Tic Tac Toe board state.
- Each neuron represents a position on the board.
- Encoding example:
- 1 for your pieces
- -1 for opponent pieces
- 0 for empty spaces

Hidden Layers:

- Number of layers: 2
- Number of neurons per layer: 16
- Activation function: Use ReLU (Rectified Linear Unit) for simplicity and good performance.
- Purpose: Learn patterns and strategies in the game. These layers will extract features such as winning paths, blocking moves, or optimal positions.

Output Layer:

- Number of neurons: 9
- Activation function: Softmax
- Purpose:
    - Outputs a probability distribution over the 9 positions.
    - The neuron with the highest probability indicates the model's suggested move.