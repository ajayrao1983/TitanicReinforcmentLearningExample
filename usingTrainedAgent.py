# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 12:08:00 2022

@author: Ajay Rao
"""


import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

os.chdir('E:/Learning/Reinforcement Learning/Titanic')

# Define the neural network class
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
    
# Load the trained agent    
net = torch.load('trainedAgent')

# Load and process training and test dataset
trainDT = pd.read_csv('Data/train.csv')
trainDT.drop
trainDT['Age_Missing'] = np.where(trainDT['Age'].isnull(), 1, 0)
trainDT['Age_Children'] = np.where(trainDT['Age'] <= 18, 1, 0)
trainDT['Age_Old'] = np.where(trainDT['Age'] > 65, 1, 0)
trainDT['Age_Adults'] = np.where((trainDT['Age_Missing'] + trainDT['Age_Children'] + trainDT['Age_Old']) == 0, 1, 0)
trainDT['Cabin2'] = trainDT['Cabin'].str.slice(start = 0, stop = 1)

trainDT.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Age'], axis = 1, inplace = True)
trainDT.dropna(axis = 0, how = 'any', inplace = True)

tempDT = trainDT.drop(['Sex', 'Pclass', 'Embarked', 'Survived', 'Cabin2',
                       'Age_Missing', 'Age_Children', 'Age_Adults', 'Age_Old'], axis = 1)

testDT = pd.read_csv('Data/test.csv')
testDT['Age_Missing'] = np.where(testDT['Age'].isnull(), 1, 0)
testDT['Age_Children'] = np.where(testDT['Age'] <= 18, 1, 0)
testDT['Age_Old'] = np.where(testDT['Age'] > 65, 1, 0)
testDT['Age_Adults'] = np.where((testDT['Age_Missing'] + testDT['Age_Children'] + testDT['Age_Old']) == 0, 1, 0)
testDT['Cabin2'] = testDT['Cabin'].str.slice(start = 0, stop = 1)

testDT.drop(['Name', 'Ticket', 'Cabin', 'Age'], axis = 1, inplace = True)
testDT1 = testDT.drop(['PassengerId', 'Sex', 'Pclass', 'Embarked', 'Cabin2',
                       'Age_Missing', 'Age_Children', 'Age_Adults', 'Age_Old'], axis = 1)

# Standardize the test dataset using the mean and standard deviation from training dataset
tempDT1 = (testDT1 - tempDT.mean()) / tempDT.std()
del tempDT

testDT3 = pd.concat([
    tempDT1,
    testDT[['Age_Missing', 'Age_Children', 'Age_Adults', 'Age_Old']],
    pd.get_dummies(testDT.Sex, prefix = 'Sex', drop_first = True),
    pd.get_dummies(testDT.Pclass, prefix = 'Pclass', drop_first = True),
    pd.get_dummies(testDT.Cabin2, prefix = 'Cabin', drop_first = True, dummy_na = True),
    pd.get_dummies(testDT.Embarked, prefix = 'Embarked', drop_first = True, dummy_na = True)], axis = 1)

testDT3['Cabin_T'] = 0

sm = nn.Softmax(dim = 1)

# Run predictions on the test dataset
pred = sm(net(torch.FloatTensor(testDT3.to_numpy())))

# Store results
testResult = pd.concat([
    testDT[['PassengerId']],
    pd.DataFrame(pred.data.numpy())], axis = 1)

testResult.columns = ['PassengerId', 'S0', 'S1']

testResult['Survived'] = np.where(testResult['S0'] > testResult['S1'], 0, 1)

# Export results to csv
testResult.to_csv('Data/TrainedModelonTestData.csv')
