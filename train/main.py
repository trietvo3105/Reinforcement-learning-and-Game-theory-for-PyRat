### Author: Carlos Lassance, Myriam Bontonou, Nicolas Farrugia
### Lab Session on Reinforcement learning
### The goal of this short lab session is to quickly put in practice a simple way to use reinforcement learning to train an agent to play PyRat.
### We perform Q-Learning using a simple regressor to predict the Q-values associated with each of the four possible movements. 
### This regressor is implemented with pytorch
### You will be using Experience Replay: while playing, the agent will 'remember' the moves he performs, and will subsequently train itself to 
### predict what should be the next move, depending on how much reward is associated with the moves.

### Usage : python main.py
### Change the parameters directly in this file. 

### GOAL : complete this file (main.py) in order to perform the full training and testing procedure using reinforcement learning and experience replay.

### When training is finished, copy both the AIs/rl_reload.py file and the save_rl folder into your pyrat folder, and run a pyrat game with the 
### appropriate parameters using the rl_reload.py as AI.

import json
import numpy as np
import time
import random
import pickle
from tqdm import tqdm
from AIs import manh, rl_reload
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

### The game.py file describes the simulation environment, including the generation of reward and the observation that is fed to the agent.
import game

### The rl.py file describes the reinforcement learning procedure, including Q-learning, Experience replay, and a pytorch model to learn the Q-function.
### SGD is used to approximate the Q-function.
import rl



### This set of parameters can be changed in your experiments.
### Definitions :
### - An iteration of training is called an Epoch. It correspond to a full play of a PyRat game. 
### - An experience is a set of  vectors < s, a, r, s’ > describing the consequence of being in state s, doing action a, receiving reward r, and ending up in state s'.
###   Look at the file rl.py to see how the experience replay buffer is implemented. 
### - A batch is a set of experiences we use for training during one epoch. We draw batches from the experience replay buffer.


epoch = 10000  # Total number of epochs that will be done

max_memory = 1000  # Maximum number of experiences we are storing
number_of_batches = 8  # Number of batches per epoch
batch_size = 32  # Number of experiences we use for training per batch
width = 21  # Size of the playing field
height = 15  # Size of the playing field
cheeses = 40  # Number of cheeses in the game
opponent = manh  # AI used for the opponent

### If load, then the last saved result is loaded and training is continued. Otherwise, training is performed from scratch starting with random parameters.
load = False
save = True


env = game.PyRat(width=width, height=height, opponent=opponent, cheeses=cheeses)
exp_replay = rl.ExperienceReplay(max_memory=max_memory)

device = 'cpu' #torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = rl.NLinearModels(env.observe()[0]).to(device)
model = rl.ConvNet(env.observe()[0]).to(device)


### There are a few methods ( = functions) that you should look into.

### For the game.Pyrat() class, you should check out the .observe and the .act methods which will be used to get the environment, and send an action. 

### For the exp_replay() class, you should check the remember and get_batch methods.

### For the NLinearModels class, you should check the train_on_batch function for training, and the forward method for inference (to estimate all Q values given a state).

if load:
    model.load()

def eps_greedy_action(model, state, eps_start, eps_end, decay_rate, eps_step):
    # This is the function for decayed-epsilon-greedy action-selection implementation
    ## Formulate the decaying of epsilon 
    eps = eps_end + (eps_start-eps_end)*np.exp(-eps_step/decay_rate)
    ## Action-selection based on epsilon value
    if np.random.rand() > eps:
         with torch.no_grad():
              Q = model(state.unsqueeze(dim=0))[0] 
            ## Pick the next action that maximizes the Q value
              action = torch.argmax(Q).item()
    else: 
      ## Pick the next action randomly
        action = np.random.choice(4)
    return action

def play(model, epochs, train=True):

    global win_rate, total_loss
    win_cnt = 0
    lose_cnt = 0
    draw_cnt = 0
    win_hist = []
    cheeses = []
    steps = 0.
    last_W = 0
    last_D = 0
    last_L = 0
    loss = 0
    
    win_rate = []
    total_loss = []
    
    # Parameters of decayed-epsilon-greedy q-learning
    eps_start = 1.0
    eps_end = 0.01
    decay_rate = 200 # 0.0001
    eps_step = 0
    
    # Define a loss function and optimizer
    ### CODE TO BE COMPLETED
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.1)

    for e in tqdm(range(epochs)):
        env.reset()
        game_over = False

        # Get the current state of the environment
        ### CODE TO BE COMPLETED
        input_t = torch.FloatTensor(env.observe())

        model.eval()
        # Play a full game until game is over
        while not game_over:
            eps_step += 1
            # Do not forget to transform the input of model into torch tensor
            # using input = torch.FloatTensor(input).
            input_curr = input_t.clone()
            # Predict the Q value for the current state
            #### CODE TO BE COMPLETED
            # with torch.no_grad():
            #   Q_t = model(input_curr.unsqueeze(dim=0))[0] # shape = torch[1,4]
            #   action_t = torch.argmax(Q_t).item()
            # # # Pick the next action that maximizes the Q value
            # ### CODE TO BE COMPLETED
            #   if np.random.rand() > 0.01:
            #       action_t = torch.argmax(Q_t).item()
            #   else:
            #       action_t = np.random.choice(4)
            action_t = eps_greedy_action(model, input_curr, eps_start, eps_end, decay_rate, eps_step)
            # Apply action, get rewards and new state
            ### CODE TO BE COMPLETED
            input_t, reward, game_over = env.act(action_t) 
            input_t = torch.FloatTensor(input_t)

            # Statistics
            if game_over:
                steps += env.round
                if env.score > env.enemy_score:
                    win_cnt += 1
                elif env.score == env.enemy_score:
                    draw_cnt += 1
                else:
                    lose_cnt += 1
                cheese = env.score

            # Create an experience array using previous state, the performed action, the obtained reward and the new state. The vector has to be in this order.
            # Store in the experience replay buffer an experience and end game.
            # Do not forget to transform the previous state and the new state into torch tensor.
            ### CODE TO BE COMPLETED
            exp_replay.remember([input_curr,action_t,reward,input_t],game_over)
            # input_t = input_t1
        win_hist.append(win_cnt)  # Statistics
        cheeses.append(cheese)  # Statistics

        if train:
            model.train()
            # Train using experience replay. For each batch, get a set of experiences (state, action, new state) that were stored in the buffer. 
            # Use this batch to train the model.
            ### CODE TO BE COMPLETED
            loss_per_epoch = 0
            for i in range(number_of_batches):
                inputs, targets = exp_replay.get_batch(model, batch_size=batch_size)
                # print(inputs.permute((0,3,1,2)).shape, targets.shape)
                loss_batch = rl.train_on_batch(model, inputs, targets, criterion, optimizer)
                loss_per_epoch += loss_batch
            loss += loss_per_epoch
            if (e+1) % 100 == 0:
                total_loss.append(loss_per_epoch)    
            

        if (e+1) % 100 == 0:  # Statistics every 100 epochs
            cheese_np = np.array(cheeses)
            win_rate_ = win_cnt/(win_cnt+draw_cnt+lose_cnt)
            win_rate.append(win_rate_)
            string = "Epoch {:03d}/{:03d} | Cheese count {} | Last 100 Cheese {}| W/D/L {}/{}/{} | 100 W/D/L {}/{}/{} | 100 Steps {}, win rate={:.2f}".format(
                        e,epochs, cheese_np.sum(), 
                        cheese_np[-100:].sum(), win_cnt, draw_cnt, lose_cnt, 
                        win_cnt-last_W, draw_cnt-last_D, lose_cnt-last_L, steps/100, win_rate_)
            print(string)
            # print('Loss = ', loss_per_epoch)
            # print('Eps=', eps_end + (eps_start-eps_end)*np.exp(-eps_step/decay_rate))
            steps = 0.
            last_W = win_cnt
            last_D = draw_cnt
            last_L = lose_cnt
                        

# Save win rate and losses
model_name = 'FCN_64_16_decay200' # name of the model to be saved
method = 'eps' # name of the method along side with RL (decayed greedy algorithm)
name = model_name + '_' + method
print("Training")
play(model,epoch, True)
if save:
    model.save()
    np.save('save_rl/wr_train_'+name,win_rate)
    np.save('save_rl/loss_'+name,total_loss)

print("Training done")
print("Testing")
play(model, epoch, False)
if save:
    np.save('save_rl/wr_test_'+name,win_rate)
print("Testing done")





