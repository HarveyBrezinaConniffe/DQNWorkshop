import gym
import torch
from torch import nn
from collections import deque, namedtuple
import numpy as np
import random

# TASK 1
# Each time step we are going to be storing a "Transition". This describes how we move from one state to another.
# It consists of a state, action, reward and nextState.
# In Java we might create a Transition class( Which we could do in Python ). But there is an easier way!
# We can create something called a namedtuple, Which works just like a class but can be defined in one line!
# EXAMPLE:
# 	>> Person = namedtuple("Person", ["name", "age"])
#	>> harvey = Person("Harvey Brezina Conniffe", 20)
#	>> print("Harvey's age is "+harvey.age)
#
# YOUR CODE: Create a Transition namedtuple that stores 4 values. state, action, reward and nextState.
Transition = namedtuple("X", ["X"])

# TASK 2
# We want to store the most recent Transitions we've seen( If we stored all of them forever we'd quickly run out of memory space! )
# We could add them to a list and just delete the earliest elements when this list gets too big, But this is actually very inefficient!
# Instead we will use something called a deque, This will automatically delete old elements and is also very efficient!
# EXAMPLE:
# 	>> Colleges = deque(maxlen=2)
#	>> Colleges.append("UCD")
#	>> Colleges.append("Trinity")
#	>> Colleges.append("DCU")
#	>> print(colleges)
#	['Trinity', 'DCU']
#
# YOUR CODE: Create a deque to hold MEMORY_SIZE transitions
MEMORY_SIZE = 1000
replayMemory = "TODO"

# TASK 3
# Now it's time to get to the brains of the agent( AI )! This is the Q Network!
# It takes in the current state and outputs an estimate of how much future reward we will get for each action.
# In a self driving car it might take in a picture from a camera and output the reward from turning left, right or straight.
# In this environment we will take in the cart's velocity and position and predict rewards for accelerating left, right and not accelerating.
# EXAMPLE - A neural network that takes in 10 numbers and outputs 2 numbers:
# 	>> self.neuralNet = nn.Sequential(
#		nn.Linear(10, 256),
#		nn.LeakyReLU(),
#		nn.Linear(64, 2)
#	)
# We've already set up the scaffolding so you only need to fill in the part that says "YOUR CODE HERE"
#
# YOUR CODE: Create a neural network with at least 3 layers that takes 2 inputs and gives 3 outputs.
class DQN(nn.Module):
	def __init__(self):
		super(DQN, self).__init__()
		self.neuralNet = nn.Sequential(
			### YOUR CODE HERE ###
		)

	def forward(self, x):
		return self.neuralNet(x)

# Let's initialize a new Q network!
QNetwork = DQN()

# TASK 4
# Let's put the Q network to good use! We're going to make a function that takes in a state and decides which action to take.
# Remember that each of the Q network's outputs represents how much future reward it thinks you'll get from that action.
# The steps that this function will take are:
# 	1. Feed the state into the Q network.
#	2. Convert the Q networks output from a pytorch tensor into a numpy array.
#	3. Choose the action with the highest predicted reward.
#	4. Return that action.
def chooseAction(state):
	# Run the Q network on the input state.
	# EXAMPLE -- Running a cat or dog classifier on an image would look like:
	# 	>> result = catClassifier(image)
	predictedRewards = #YOUR CODE HERE#
	# Convert the output from a pytorch tensor to a numpy arra.
	# EXAMPLE -- Converting the cat classifiers output would look like:
	# 	>> resultNumpy = result.detach().numpy()
	predictedRewardsNumpy = #YOUR CODE HERE#
	# Choose the action with the highest predicted reward.
	# You could do this with a loop but the np.argmax function provides a much more elegant way to do it.
	# EXAMPLE:
	#	>> array = [1, 5, 2]
	#	>> np.argmax(array)
	#	1
	bestAction = -1
	# Return the best action.
	return bestAction
