## our neural network
import numpy as np
import torch
import torch.nn as nn

class Agent:
    
    def __init__(self, color, model):
        Agent.color = color
        Agent.model = model

    def placeMove(self, board):
        
        x, y = self.determineMove(board.flatten())
        board[x][y] = self.color

        # flatten the board to 1D array,
        # and pass it thorugh the model
        return board
    
    def determineMove(self, input_array):
        # pass input array into NN
        # take output and return as tuple
        return (0, 0)
