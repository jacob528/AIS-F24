# Othello Project

This othello AI is a project done by Jacob & Ricardo. We are going to use a reinforcement learning approach to training our neural network, and deploying this model inside a pygame environment of the board game.


## Papers we have looked at so far:
- [Application of Reinforcement Learning to the Game Othello](https://www.sciencedirect.com/science/article/pii/S0305054806002553#:~:text=We%20describe%20how%20reinforcement%20learning%20can%20be%20combined,state%20space%E2%80%94learning%20to%20play%20the%20game%20of%20Othello)
- [Reinforcement Learning in the Game of Othello: Learning Against a Fixed Opponent and Learning from Self-Play](https://www.ai.rug.nl/%7Emwiering/GROUP/ARTICLES/paper-othello.pdf)

## Useful articles / videos:
- [What is DQN?](https://medium.com/data-science-in-your-pocket/deep-q-networks-dqn-explained-with-examples-and-codes-in-reinforcement-learning-928b97efa792)
- [Explanantion of Monte Carlo](https://www.youtube.com/watch?v=ZljeS5aHzuE)

## Procedure for building a reinforcement learning algorithm for Othello:

### 1. define an environment for our agent to explore

An environment is the place where various states of the game exist. Our model will analyze potential, legal states to decide which move is best to play. 

This consists of building the board game in python as a 2 dimensional collection of cells in various states. The three states our model will have to understand are black, white, and empty. It can process these states as integers -1, 0, 1. This can be represented in a visual medium to a client as a board built in pygame.

The various actions a model can take in our environment consists of all legal moves for a player's turn. In Othello, a legal move is defined as a move that flips at least one of the opponent's pieces over. If this cannot be achieved, control is passed to the opponent without the player making a move.

Our reward function should consider a variety of statistics, such as # of pieces flipped, controlling edges/corners, and mobility of future moves. There are a variety of implementations, such as decision trees that look at future turns with the highest reward value to make the best move that predicts your opponent's move. Part of our research should include understanding the different reward functions.

### 2. Choosing a reinforcement learning algorithm

There are quite a few reinforcement learning algorithms out there. The one we are going with is called a Deep Q-Learning Network (DQN). This builds off of a formal logic construct called the Markov Decision Process (MDP), which makes choices in an analytical manner that can be represented with math.

A MDP utilizes these 5 processes: Actions, States, the Transition Function, the Reward Function, and the Discount variable. The MDP works inside of an environment, and controls the actions of an agent. 
- The set *S* contains all possible states *s* our agent can exist in. Imagine the state being a board configuration, a location, or a certain position your agent can exist in your game. When an Agent is in a state, it stays there until some action moves it from that state.

- The set *A* contains all possible actions *a* our agent can take. Agents take actions to change states. An action has an associated value that determines how effective the action was. It is determined by the reward function.

- The transition function determines the probability of moving to state *s*' given action *a* in state *s*. In our model, this is represented by the weights and biases of the neural network, so it can be ignored.

- The reward function is the analytical approach to differentiating states. Based on being in some state *s*, V(s) computes the reward for being in this state. If you take action *a* in state *s*, then you can compute the reward of being in the resulting states with Q(s, a). This Q is the same Q in our algorithm name, and represents a function that computes the

## how we are implementing this

Our DQN needs the following things:
- A Neural Network that approximates the Policy Function
- A Reward Function that computes the reward for (s, a)
- A loss function that computes the loss between a chosen a from Qp(s) and the associated state-action pair input to a Qt

### The Policy Function

In MDP, the policy determines which action *a* to take in state *s*. This action is taken, and then the process is repeated until the end state is reached. The policy works to maximize the reward associated with (*s*, *a*). 

Because Othello has such an involved set of states and rewards, this policy must be approximated with a neural network. The neural network (Q) takes as an input some state *s*, and returns a vector of Q-values, with the index in this vector corresponding to an action *a*

Q(*s*) = \[Q<sub>0</sub>, Q<sub>1</sub>, Q<sub>2</sub>, ... \], 
