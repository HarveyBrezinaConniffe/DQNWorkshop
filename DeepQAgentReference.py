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
Transition = namedtuple("Transition", ["state", "action", "reward", "nextState"])

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
MEMORY_SIZE = 10000
replayMemory = deque(maxlen=MEMORY_SIZE)

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
			nn.Linear(2, 128),
			nn.LeakyReLU(),
			nn.Linear(128, 128),
			nn.LeakyReLU(),
			nn.Linear(128, 64),
			nn.LeakyReLU(),
			nn.Linear(64, 3),
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
	predictedRewards = QNetwork(state)
	# Convert the output from a pytorch tensor to a numpy arra.
	# EXAMPLE -- Converting the cat classifiers output would look like:
	# 	>> resultNumpy = result.detach().numpy()
	predictedRewardsNumpy = predictedRewards.detach().numpy() 
	# Choose the action with the highest predicted reward.
	# You could do this with a loop but the np.argmax function provides a much more elegant way to do it.
	# EXAMPLE:
	#	>> array = [1, 5, 2]
	#	>> np.argmax(array)
	#	1
	bestAction = np.argmax(predictedRewardsNumpy)
	# ** These are weird lines of code neccessary for the tester program. Don't worry about them.
	chooseAction.predictedRewards = predictedRewards
	chooseAction.predictedRewardsNumpy = predictedRewardsNumpy
	# ** End of debug lines.
	# Return the best action.
	return bestAction

# Initialize the environment.
env = gym.make('MountainCar-v0')

# TASK 5
# Now that we have a way to play the game let's collect some transitions to learn from!
# This function will play one game( Until we reach completion ) while recording the transitions.
def collectTransitions():
	# Record the first frame of the game.
	observation = env.reset()
	# Once the game is done we want to return.
	done = False
	while not done:
		# Convert the current state to a numpy array.
		currentState = torch.from_numpy(observation)
		# Choose an action to take.

		# YOUR CODE HERE, USE THE chooseAction FUNCTION YOU MADE IN TASK 4.
		action = chooseAction(currentState)
		# END OF YOUR CODE

		# Tell the environment what action we are taking.
		observation, _, done, info = env.step(action)
		# By default this environment only returns a single reward at the end of the game if we win.
		# This is called a "sparse" reward and is quite challenging for a neural network to learn from!
		# I've implemented a much simpler reward since we have a very small neural network.
		# It just gives us a higher reward the closer we are to the flag( Goal of the game ).
		reward = np.exp(observation[0]*10, dtype=np.float32)

		# Convert the new observation to a numpy arra
		newState = torch.from_numpy(observation)

		# Construct a Transition namedTuple containing the currentState, action, reward and newState
		# YOUR CODE HERE - Fill in the transition with the correct variables and add it to the replayMemory
		transition = Transition(currentState, action, reward, newState)
		replayMemory.append(transition)
		# END OF YOUR CODE

# TASK 6
# Now that we can collect transitions how about we try and learn something from them!
# This function takes in a loss function and optimizer using them alongside a batch of transitions to update the models parameters!
def trainStep(lossFunction, optimizer, batchSize):
	# Take a random sample of batchSize transitions from the replay memory.
	# EXAMPLE -- To select a random sample of 2 elements from an array we can run:
	#	>> fruits = ["Apple", "Bannana", "Orange", "Grape"]
	#	>> random.sample(fruits, 2)
	#	["Grape", "Bannana"]
	# YOUR CODE HERE - Take a random sample of batchSize elements from replayMemory
	batch = random.sample(replayMemory, batchSize)
	# END OF YOUR CODE
	
	# These next few lines process the batch into the correct format for training.
	# Currently batch is an array of tuples, So if you wanted to get all of the reward values you would have to
	# get the 3rd element of the 1st tuple, the 3rd element of the 2nd tuple, the 3rd element of the 3rd tuple and so on.
	# While storing them in this way is nice for working with them it's not the best for feedint them into pytorch.
	# What we would like to do is make an array with all the rewards, an array with all the states...
	# We could do this with loops but there is an easier way!
	# The "zip" function in python takes in multiple arrays and reformats them into a different array for each element.
	# EXAMPLE -- The zip function:
	#	>> zip(["Apple", 157.90], ["Google", 2653.78], ["Microsoft", 291.45])
	# 	["Apple", "Google", "Microsoft"], [157.9, 2653.78, 291.45]
	# This is nearly what we need, If we ran zip(batch[0], batch[1], batch[2]) we would get 4 arrays
	# state, action, reward, nextState each with 3 entries( From the first 3 elements of batch ).
	# But how do we feed in the entire batch without having to write each one by hand?
	# Luckily python has a way to do this. Inputting an array to a function with a '*' before it treats each entry of the
	# array as a separate argument.
	# So zip(*batch) is the same as running zip and inputting each element of batch individually!
	states, actions, rewards, nextStates = zip(*batch)
	# Next we just convert some of these lists into tensors
	states = torch.stack(states)
	rewards = torch.from_numpy(np.array(rewards))
	nextStates = torch.stack(nextStates)
	# Because of how a function you'll be using later on( torch.gather ) works we need to wrap each element in actions
	# E.G. [1, 2, 3] -> [[1], [2], [3]]
	actions = torch.unsqueeze(torch.Tensor(actions), 1).to(torch.int64)
	
	# Ok, Let's do some training now! 
	# The first thing we want to do is see what rewards the Q network predicts for every state in this batch.
	# We did this before when collecting the transitions but remember we are going to learn from these more than once!
	# So the Q network might have changed since last time!
	# YOUR CODE HERE - Run the Q network on all states in this batch! If you're stuck have a look at collectTransitions
	predictedRewards = QNetwork(states)
	# END OF YOUR CODE

	# So we've gotten the Q network to make some predictions! But how do we train it now?
	# In order to train a neural network we need to know what the "right" answer is.
	# But what even is the right answer in this case?
	# Remember what the Q network is predicting, For each action it predicts the total future reward if you take that
	# action and then play perfectly after that.
	# Writing that more concisely: Q(action) = Reward from taking action + Highest reward possible from this point on
	# We already have the first part of this equation! For each transition we took one action and recieved one reward.
	# But how do we know what the second part of the equation should be?
	# Well we know that after taking each action we ended up in "newState", We want to know what the highest reward
	# we can get in newState is. But how do we know this?
	# Turns out we already have a way to predict future rewards, The Q network!

	# Let's start by picking predicting the total future reward for each action for each state in newStates
	# YOUR CODE HERE - Run the Q network on all states in nextStates
	# Note: We want to use the Q network here to work out the correct values, We don't want to train it here though!
	# So we use torch.no_grad()
	with torch.no_grad():
		futurePredictions = QNetwork(nextStates)
	# END OF YOUR CODE

	# futurePredictions is a 2d array, For each state in newState it contains a prediction for each possible action 
	# But we only care about the highest reward possible from any action!
	# EXAMPLE - Getting the maximum value in pytorch.
	#	>> a = torch.Tensor([1, 0, 5],
	#			    [9, 8, 2],
	#			    [2, 10, 3])
	#	>> torch.max(a, 1)[0]
	#	[5, 9, 10]
	# YOUR CODE HERE -- Get the maximum predicted reward for each prediction in futurePredictions.
	maxFutureRewards = torch.max(futurePredictions, 1)[0]
	# END OF YOUR CODE

	# We fisrt need to multiply maxFutureRewards by some value ( E.g. 0.99 ) since we care less about older rewards.
	# We're nearly there! We've worked out the highest reward possible! All we need to do is add on the reward we got.
	# YOUR CODE HERE -- Multiply maxFutureRewards by 0.99 and add rewards to maxFutureRewards
	maxFutureRewards *= 0.99
	maxFutureRewards += rewards
	# END OF YOUR CODE

	# Now we have something to compare the Q networks predictions to! But there's one more thing we need to do first.
	# Remember that maxFutureRewards was a 2d array but we reduced it down to 1d. We need to do the same to predictedRewards.
	# For each transition in the batch we only took 1 action when collecting it. This action is the only one that we
	# collected data about. So for each element of predictedRewards we only care about the prediction relating to that action.
	# Luckily PyTorch has a function for this! Torch.gather
	# EXAMPLE -- torch.gather
	# 	>> t = torch.tensor([[1, 2], [3, 4]])
	#	>> torch.gather(t, 1, torch.tensor([[0], [1]]))
	#	tensor([[1],
        #		[4]])
	# YOUR CODE HERE -- Select the correct entries in predictedRewards using the values of actions
	predictedRewards = torch.gather(predictedRewards, 1, actions)
	# END OF YOUR CODE

	# predictedRewards looks like [[1], [4]], maxFutureRewards looks like [1, 4]. In order to make them the same we have
	# to "squeeze" a dimension out of predictedRewards
	predictedRewards = torch.squeeze(predictedRewards, 1)

	# Great work! Now we have the predictions from the Q network along with the target values. Let's train it!
	# First we have to clear the optimizer, This means that it will only learn from the data we are showing it right now.
	optimizer.zero_grad()
	# Next we need to find the loss( How wrong ) the neural network is.
	loss = lossFunction(predictedRewards, maxFutureRewards)
	# Now we use propagate this loss backwards to work out how to update the Q networks weights.
	loss.backward()
	# Here we "clamp" the gradient to prevent any one update being too big. This helps with stability.
	for param in QNetwork.parameters():
		param.grad.data.clamp_(-1, 1)	
	# Finally we give these gradients to the optimizer which will update the Q networks weights.
	optimizer.step()
	# Return the loss value for debugging and progress monitoring
	return loss

# That's it! You've written all the parts neccessary to train a Deep Q Agent. Let's put the parts together!

# First we'll define a small helper function that just makes it easier to see how the agent is progressing.
# Don't worry about it too much but there isn't anything new here. You should be able to understand it now!
# Function that plays numGames and finds the average reward gained by the agent. It can also render the games!
def evaluateAgent(numGames, renderGames=False):
	# Record average reward
	avgReward = 0.
	for i in range(numGames):
		totalReward = 0.
		# Record the first frame of the game.
		observation = env.reset()
		# Once the game is done we want to return.
		done = False
		while not done:
			# Convert the current state to a numpy array.
			currentState = torch.from_numpy(observation)
			# Choose an action
			action = chooseAction(currentState)
			# Tell the environment what action we are taking.
			observation, _, done, info = env.step(action)
			reward = np.exp(observation[0]*10, dtype=np.float32)
			# Add reward to total
			totalReward += reward
			# If rendering game show the frame
			if renderGames:
				env.render()
		avgReward += totalReward
	avgReward /= numGames
	return avgReward

# Define some constants for training.

BATCH_SIZE = 32
STEPS_PER_TRAINSTEP = 5
# How often to evaluate the agent.
EVALUATE_EVERY = 25
# How often to render games for us to see( Fun but slow )
RENDER_EVERY = 100

# Initialize the loss function and optimizer
lossFunc = torch.nn.SmoothL1Loss()
optimizer = torch.optim.RMSprop(QNetwork.parameters())

# This makes sure the tester won't run this
if __name__ == "__main__":
	i = 0
	while True:
		# Collect some transitions
		collectTransitions()
		# Run a train step
		for _ in range(STEPS_PER_TRAINSTEP):
			trainStep(lossFunc, optimizer, BATCH_SIZE)
		# Evaluate agent
		if i%EVALUATE_EVERY == 0:
			avgReward = evaluateAgent(3, bool(i%RENDER_EVERY == 0))
			print("Train Step {}: Average reward is {}".format(i, avgReward))
		i += 1
