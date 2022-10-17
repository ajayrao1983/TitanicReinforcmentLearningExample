# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 08:19:55 2022

@author: Ajay Rao
"""

import os
import numpy as np
import pandas as pd

os.chdir('E:/Learning/Reinforcement Learning/Titanic')

class Titanic:
    def __init__(self, stp):
        # Load data
        self.DT = pd.read_csv('Data/train.csv')
        
        # Convert Age to Categorical dummies
        self.DT['Age_Missing'] = np.where(self.DT['Age'].isnull(), 1, 0)
        self.DT['Age_Children'] = np.where(self.DT['Age'] <= 18, 1, 0)
        self.DT['Age_Old'] = np.where(self.DT['Age'] > 65, 1, 0)
        self.DT['Age_Adults'] = np.where((self.DT['Age_Missing'] + self.DT['Age_Children'] + self.DT['Age_Old']) == 0, 1, 0)
        
        # Keep only the first letter of 'Cabin'
        self.DT['Cabin2'] = self.DT['Cabin'].str.slice(start = 0, stop = 1)
        
        # Drop attributes that are not required
        self.DT.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Age'], axis = 1, inplace = True)
        # Drop NAs
        self.DT.dropna(axis = 0, how = 'any', inplace = True)
        
        # Create a copy with only numeric variables, for standardizing
        tempDT = self.DT.drop(['Survived', 'Sex', 'Pclass', 'Embarked', 'Cabin2',
                               'Age_Missing', 'Age_Children', 'Age_Adults', 'Age_Old'], axis = 1)
        
        # Create a copy of standardized numeric variables
        self.tempDT1 = (tempDT - tempDT.mean()) / tempDT.std()
        del tempDT
        
        # Create the final dataset by merging the standardized numeric variables
        # the label variable - 'Survived'
        # Categorical dummies for - 'Sex', 'Cabin', 'Pclass' and 'Embarked'
        self.DT2 = pd.concat([
            self.DT[['Survived']],
            self.tempDT1,
            self.DT[['Age_Missing', 'Age_Children', 'Age_Adults', 'Age_Old']],
            pd.get_dummies(self.DT.Sex, prefix = 'Sex', drop_first = True),
            pd.get_dummies(self.DT.Pclass, prefix = 'Pclass', drop_first = True),
            pd.get_dummies(self.DT.Cabin2, prefix = 'Cabin', drop_first = True, dummy_na = True),
            pd.get_dummies(self.DT.Embarked, prefix = 'Embarked', drop_first = True, dummy_na = True)],
            axis = 1)
        del self.DT
        
        # The observation space is the number of attributes excluding 'Survived'
        self.observation_space = self.DT2.shape[1] - 1
        
        # The agent can only take one action with two values i.e. 'Survived' or 'Not Survived'
        self.action_space = 2
        
        # Keeps track of total steps per episode
        self.stps = stp
        
        # Keeps track of number of steps left in the episode
        self.steps_left = stp
        
        
    def reset(self):
        # Resets the number of steps left to total steps in the episode
        self.steps_left = self.stps
        
        
    def get_observation(self):
        # Sample one of the observation from the dataset randomly
        self.DT3 = self.DT2.sample(n = 1, replace = False, axis = 0)
        # Store the label as answer to generate reward later on
        self.ans = self.DT3['Survived'].iloc[0]
        # Store the attributes other than the label as the observation set for the agent
        self.obs = self.DT3.drop(['Survived'], axis = 1).to_numpy()[0]
        # Return observation set to the agent
        return self.obs
    
    
    def get_actions(self):
        # This is not used by the agent, but the user can query this to check
        # the possible actions that are expected by the environment
        return [0, 1]
    
    
    def is_done(self):
        # Checks if the episode is over i.e. no more steps are left in the episode
        return self.steps_left <= 0
    
    
    def step(self, action):
        # Returns a reward of '1' if the action matches the answer, else returns '0'
        reward = 1 if action == self.ans else 0
        # Pull the next observation
        next_obs = self.get_observation()
        # Decrements the steps left
        self.steps_left -= 1
        # Returns the next observation, reward and a flag of whether the episode is done or not
        if self.is_done():
            return next_obs, reward, 1
        else:
            return next_obs, reward, 0
        