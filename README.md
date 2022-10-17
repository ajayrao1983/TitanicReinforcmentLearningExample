# Solving Titanic Dataset using Reinforcement Learning

Link to GitHub repository - https://github.com/ajayrao1983/TitanicReinforcmentLearningExample

##Libraries Used:
Python:
- numpy
- pandas
- collections 
- torch
- os 


##Folder Structure and Files in Repository
.

|--Data

| |- train.csv # Training Set

| |- test.csv # Test Set

|- TitanicEnv.py # Defining the Titanic Environment Class

|- AgentTrainer.py # Defining and training Agent Class

|- usingTrainedAgent.py # Calling the trained agent and processing the test data set

|--README.md


##Project Description

This is an attempt at using Reinforcement Learning for unlabelled data classification tasks, assuming there is a way to:

* Model the agent
* Model the environment
* Evaluate or score the actions of the agent

The resulting model scored a 78% accuracy on the test dataset. 
