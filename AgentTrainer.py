# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 08:34:23 2022

@author: Ajay Rao
"""

import os
from collections import namedtuple
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

os.chdir('E:/Learning/Reinforcement Learning/Titanic')

from TitanicEnv import Titanic

# Initialize values
HIDDEN_SIZE = 128 # Size of the hidden layer in the neural network
BATCH_SIZE = 20 # Size of the batch before training the agent
PERCENTILE = 70 # Only the top 30 percentile of records are considered for training the agent

# The number of iterations before stopping the agent training
# if the desired performance level is not reached earlier
MaxIter = 500 

# The agent class is a MLP
class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
            )
        
    def forward(self, x):
        return self.net(x)

# Initialize the environment, with 500 steps per episode
env = Titanic(500)

# The observation size as defined in the environment
# will be used to set the size of neural network
obs_size = env.observation_space

# The number of possible actions as defined by the environment
# will be used to set the size of neural network
n_actions = env.action_space

# Initialize the neural network
net = Net(obs_size, HIDDEN_SIZE, n_actions)

# Initalize the objective function. CrossEntropyLoss is used because is a categorical classification
# Binary Cross Entropy is not defined so this code could be generalized for multi-class classification
objective = nn.CrossEntropyLoss()
# Adam Optimizer with a learning rate of 0.01
optimizer = optim.Adam(params = net.parameters(), lr = 0.01)

# Initialize lists to keep track of performance
itrList = []
lList = []
mReward = []
bReward = []    

# Initialize tuples to keep track of reward, observation and action    
Episode = namedtuple('Episode', field_names = ['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names = ['observation', 'action'])


def iterate_batches(env, net, batch_size):
    '''
    

    Parameters
    ----------
    env : The environment
    net : The neural network agent
    batch_size : The size of the batch before yielding to outer loop

    Yields
    ------
    batch : reward, observation and actions for each step

    '''
    batch = []
    episode_reward = 0.0
    nStep = 0
    episode_steps = []
    obs = env.get_observation()
    sm = nn.Softmax(dim = 1)
    while True:
        obs_v = torch.FloatTensor([obs])
        act_probs_v = sm(net(obs_v))
        act_probs = act_probs_v.data.numpy()[0]
        action = np.random.choice(len(act_probs), p = act_probs)
        next_obs, reward, is_done = env.step(action)
        episode_reward += reward
        nStep += 1
        step = EpisodeStep(observation = obs, action = action)
        episode_steps.append(step)
        if is_done == 1:
            e = Episode(reward = episode_reward * 100 / nStep, steps = episode_steps)
            batch.append(e)
            episode_reward = 0.0
            nStep = 0
            episode_steps = []
            env.reset()
            next_obs = env.get_observation()
            if len(batch) == batch_size:
                yield batch
                batch = []
        obs = next_obs
        
        
def filter_batch(batch, percentile):
    '''
    

    Parameters
    ----------
    batch : The batch with reward, observations and actions
    percentile : The percentile of observations to keep for training the agent

    Returns
    -------
    train_obs_v : The list of observations
    train_act_v : The list of actions
    reward_bound : The minimum reward as per the percentile cutoff 
    reward_mean : The mean rewards of the records left after filteration

    '''
    rewards = list(map(lambda s: s.reward, batch))
    reward_bound = np.percentile(rewards, percentile)
    reward_mean = float(np.mean(rewards))
    
    train_obs = []
    train_act = []
    for reward, steps in batch:
        if reward < reward_bound:
            continue
        train_obs.extend(map(lambda step: step.observation, steps))
        train_act.extend(map(lambda step: step.action, steps))
        
    train_obs_v = torch.FloatTensor(train_obs)
    train_act_v = torch.LongTensor(train_act)
    return train_obs_v, train_act_v, reward_bound, reward_mean


for iter_no, batch in enumerate(iterate_batches(
        env, net, BATCH_SIZE)):
    '''
    Iterate over batches and train the neural network.
    Break the loop if the mean reward goes above 84% accuracy or
    max of 500 iterations are run
    '''
    obs_v, acts_v, reward_b, reward_m = filter_batch(batch, PERCENTILE)
    optimizer.zero_grad()
    action_scores_v = net(obs_v)
    loss_v = objective(action_scores_v, acts_v)
    loss_v.backward()
    optimizer.step()
    
    itrList.append(iter_no)
    lList.append(loss_v.item())
    mReward.append(reward_m)
    bReward.append(reward_b)
    
    print("%d: loss=%.3f, rw_mean=%.3f, rw_bound=%.3f" % (iter_no, loss_v.item(), reward_m, reward_b))
    
    if reward_m > 84:
        print('Solved!')
        break
    if iter_no == MaxIter:
        print('Iterations crossed %d, environment not solved!' % (MaxIter))
        break
    
# Store results of the iterations in a pandas dataframe
resultDT = pd.DataFrame({'iterN': itrList,
                         'loss': lList,
                         'MeanReward': mReward,
                         'BoundReward': bReward})

# Exports the results to csv
resultDT.to_csv("result.csv")

# Store the trained model
torch.save(net, "trainedAgent")