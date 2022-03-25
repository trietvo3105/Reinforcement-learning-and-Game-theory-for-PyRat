# Template file to create an AI for the game PyRat
# http://formations.telecom-bretagne.eu/pyrat

###############################
# Team name to be displayed in the game 
TEAM_NAME = "Q-Learner"

###############################
# When the player is performing a move, it actually sends a character to the main program
# The four possibilities are defined here
MOVE_DOWN = 'D'
MOVE_LEFT = 'L'
MOVE_RIGHT = 'R'
MOVE_UP = 'U'

###############################
# Please put your imports here

import numpy as np
import random as rd
import pickle
import time
import torch
import torch.nn as nn

###############################
# Please put your global variables here

# Global variables
global model, exp_replay, input_tm1, action, score

# Function to create a numpy array representation of the maze

def input_of_parameters(player, maze, opponent, mazeHeight, mazeWidth, piecesOfCheese):
    im_size = (2 * mazeHeight - 1, 2 * mazeWidth - 1, 1)
    canvas = np.zeros(im_size)
    (x,y) = player
    center_x, center_y = mazeWidth-1, mazeHeight-1
    for (x_cheese,y_cheese) in piecesOfCheese:
        canvas[y_cheese + center_y - y, x_cheese + center_x - x, 0] = 1
#    (x_enemy, y_enemy) = opponent
#    canvas[y_enemy+center_y-y,x_enemy+center_x-x,1] = 1
#    canvas[center_y,center_x,2] = 1
    canvas = np.expand_dims(canvas, axis=0)
    return canvas


class NLinearModels(nn.Module):
    def __init__(self, x_example, number_of_regressors=4):
        super(NLinearModels, self).__init__()
        in_features = x_example.reshape(-1).shape[0]
        # limit = np.sqrt(6 / (in_features + number_of_regressors))
        self.linear = nn.Linear(in_features, number_of_regressors)
        # self.W = np.random.uniform(-limit,limit, size=(in_features, number_of_regressors)) #HE INITIALIZATION
        # self.bias = np.zeros(number_of_regressors)
    
    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        return self.linear(x)

    # def predict(self, x):
    #     x = torch.FloatTensor(x)
    #     x = x.reshape(x.shape[0], -1)
    #     return self.forward(x)

    def load(self):
        self.load_state_dict(torch.load('save_rl/weights.pt'))

    def save(self):
        torch.save(self.state_dict(), 'save_rl/weights.pt')

    
###############################
# Preprocessing function
# The preprocessing function is called at the start of a game
# It can be used to perform intensive computations that can be
# used later to move the player in the maze.
###############################
# Arguments are:
# mazeMap : dict(pair(int, int), dict(pair(int, int), int))
# mazeWidth : int
# mazeHeight : int
# playerLocation : pair(int, int)
# opponentLocation : pair(int,int)
# piecesOfCheese : list(pair(int, int))
# timeAllowed : float
###############################
# This function is not expected to return anything
def preprocessing(mazeMap, mazeWidth, mazeHeight, playerLocation, opponentLocation, piecesOfCheese, timeAllowed):
    global model,exp_replay,input_tm1, action, score
    input_tm1 = input_of_parameters(playerLocation, mazeMap, opponentLocation, mazeHeight, mazeWidth, piecesOfCheese)    
    action = -1
    score = 0
    model = NLinearModels(input_tm1[0])
    model.load()
    


###############################
# Turn function
# The turn function is called each time the game is waiting
# for the player to make a decision (a move).
###############################
# Arguments are:
# mazeMap : dict(pair(int, int), dict(pair(int, int), int))
# mazeWidth : int
# mazeHeight : int
# playerLocation : pair(int, int)
# opponentLocation : pair(int, int)
# playerScore : float
# opponentScore : float
# piecesOfCheese : list(pair(int, int))
# timeAllowed : float
###############################
# This function is expected to return a move
def turn(mazeMap, mazeWidth, mazeHeight, playerLocation, opponentLocation, playerScore, opponentScore, piecesOfCheese, timeAllowed):    
    global model,input_tm1, action, score
    input_t = input_of_parameters(playerLocation, mazeMap, opponentLocation, mazeHeight, mazeWidth, piecesOfCheese)    
    input_tm1 = torch.FloatTensor(input_t)   
    output = model(input_tm1.unsqueeze(dim=0))
    action = torch.argmax(output[0]).item()
    score = playerScore
    return [MOVE_LEFT, MOVE_RIGHT, MOVE_UP, MOVE_DOWN][action]

def postprocessing (mazeMap, mazeWidth, mazeHeight, playerLocation, opponentLocation, playerScore, opponentScore, piecesOfCheese, timeAllowed):
    pass    
