# Rough Draft (will be changed) - Teaching AI to play PONG using DQN

## Authors: Jovanny Aguilar & Asher Schalet

## Description
This project implements a Deep Q-Network (DQN) to train an AI agent to play the game Pong. Although subject to change, intitial thought was to use OpenAI's Gymnasium environment for Pong and a Convolutional Neural Network (CNN) to approximate Q-values for the agent's actions. The model is designed to process game frames, extract spatial and temporal features, and learn optimization through reinforcement learning.

## Features
1. Preprocessing Game Frames:
    -Converts raw RGB frames to greyscale for reduced complexity
    -Resizes the frames to an 84x84

2. Deep Q-Network (DQN):
    -A CNN model approximates Q values for each possible action
    -Convolutional Layers for feature extraction
    -Fully connected layers for action-value mapping
    -ReLU activation functions for non-linearity and efficient learning

3. Integration of Open AI Gymnasium
    -Uses the "ALE/Pong-v4" environment to simulate the game for training and evaluation.

4. Test Code
    -Tests environment and preprocess_frame function to check if Pong game simulates (It did not sadly)
    -Prints out processed frame shape (which was expected to be an (84,84))

## How it works
- The environment is initialized using OpenAI's Gymnasium "ALE/Pong-v4" environment
- Game frames are preprocessed to simplify the input data for the agent
- The DQN model processes 4 stacked frames to predict Q-values for possible actions.
- The Q-values guide the agent's decisions, aiming to maximize the reward by learning from game interactions

## Warning
-As initially stated in the description, this code, along with this README.md, is subject to change. Considering the Pong simulation from OpenAI Gymnasium is not working, the immidiate goal at the moment is to create a functioning physics engine for the Pong game in lieu of the imported environment. From there, the rest of the steps in regards to training the AI agent will be implemented.