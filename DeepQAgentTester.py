from collections import deque
import numpy as np

try:
	import torch
except Exception as e:
	print(e)
	print("Failed to import pytorch!")
	print("Hint: Have you installed pytorch yet?")
	print("If not see: https://pytorch.org/get-started/locally/")
	exit(0)

try:
	import gym
except Exception as e:
	print(e)
	print("Failed to import openAI gym!")
	print("Hint: Have you installed gym yet?")
	print("If not see: https://gym.openai.com/docs/")
	exit(0)

try:
	env = gym.make('MountainCar-v0')
except Exception as e:
	print(e)
	print("Failed to initialize an environment!")
	print("Hint: You may need to install the classic control package")
	print("Try running: pip install gym[classic_control]")
	exit(0)

import DeepQAgentReference as DeepQAgent

# Check Task 1
if DeepQAgent.Transition.__name__ != "Transition":
	print("Error in Task 1!")
	print("Your 'Transition' namedtuple is called {}. It should be called Transition!".format(DeepQAgent.Transition.__name__))
	exit(0)

if DeepQAgent.Transition._fields != ("state", "action", "reward", "nextState"):
	print("Error in Task 1!")
	print("Your 'Transition' namedtuple has the fields {}. Remember that we want it to store state, action, reward and nextState!".format(DeepQAgent.Transition._fields))
	exit(0)

print("Task 1 successful! Your Transition namedTuple is correctly defined!")
print()


# Check Task 2
if type(DeepQAgent.replayMemory) != deque:
	print("Error in Task 2!")
	print("Replay memory is supposed to be a deque but is actually {}".format(type(DeepQAgent.replayMemory).__name__))
	exit(0)

if DeepQAgent.replayMemory.maxlen != DeepQAgent.MEMORY_SIZE:
	print("Error in Task 2!")
	print("Replay memory should only store the last {} transitions. Make sure you've set the maxlen property!".format(DeepQAgent.MEMORY_SIZE))
	exit(0)

print("Task 2 successful! replayMemory is correctly setup!")
print()

# Check Task 3
try:
	dqn = DeepQAgent.DQN()
except Exception as e:
	print("Error in Task 3!")
	print(e)
	exit(0)

children = dqn.named_children()
hasNN = False
for child in children:
	# Check that the neuralNet is there.
	if child[0] == "neuralNet":
		hasNN = True
		layers = list(child[1].children())
		# Check that there are some layers in the neural network.
		if len(layers) == 0:
			print("Error in Task 3!")
			print("Add some layers to the neuralNet!")
			exit(0)
		# Check that the first layer is a linear layer.
		if type(layers[0]) != torch.nn.modules.linear.Linear:
			print("Error in Task 3!")
			print("The first layer of your neural network should be a Linear layer!")
			exit(0)
		# Check that the neural network takes the correct input shape.
		if layers[0].in_features != 2:
			print("Error in Task 3!")
			print("Your neural network takes in {} inputs. Remember that we take in 2 inputs, Velocity and position!".format(layers[0].in_features))
			exit(0)
		# Check that the last layer is a linear layer.
		if type(layers[-1]) != torch.nn.modules.linear.Linear:
			print("Error in Task 3!")
			print("The last layer of your neural network is predicting total future reward, So it should be able to output any value! This means that it should just be a linear layer with no activation function!")
			exit(0)
		# Check that the neural network gives the correct output shape.
		if layers[-1].out_features != 3:
			print("Error in Task 3!")
			print("Your neural network has {} outputs. Remember that we have 3 possible actions, Accelerate left, Right or stay still!".format(layers[-1].out_features))
			exit(0)
		# Make sure we don't have any linear layers without activations between them.
		lastLayer = None
		totalLayers = 0
		for layer in layers:
			layerType = type(layer)
			if layerType == torch.nn.modules.linear.Linear:
				totalLayers += 1
			if layerType == torch.nn.modules.linear.Linear and lastLayer == torch.nn.modules.linear.Linear:
				print("Error in Task 3!")
				print("You have two linear layers without an activation between them! Activation functions are neccessary for a neural network to learn, Try adding a leakyRelu layer between them!")
				exit(0)
			lastLayer = layerType
		# Make sure that each layer takes the correct input size.
		lastOutSize = 2
		currentLinearLayer = 0
		for layer in layers:
			layerType = type(layer)
			if layerType == torch.nn.modules.linear.Linear:
				inputSize = layer.in_features
				if inputSize != lastOutSize:
					print("Error in Task 3!")
					print("Linear layer {} outputs {} features. But linear layer {} takes in {} features!".format(currentLinearLayer-1, lastOutSize, currentLinearLayer, inputSize))
					print("Make sure that each layer takes in the same number of features that the one before it outputs!")
					exit(0)
				currentLinearLayer += 1
				lastOutSize = layer.out_features

		
		# Check that they have at least 3 layers.
		if totalLayers < 3:
			print("Error in Task 3!")
			print("Your neural network only has {} layers. I'd recommend at least 3 to get good performance!".format(totalLayers))
			exit(0)

if hasNN == False:
	print("Error in Task 3!")
	print("I couldn't seem to find \"neuralNet\", Did you rename it?")
	exit(0)

print("Task 3 successful! Your neural network is set up correctly!")
print()

# Check Task 4 -- Choose best action
testState = torch.zeros((2))
try:
	bestAction = DeepQAgent.chooseAction(testState)
	# Check that they are running the neural network correctly.
	predictedRewards = DeepQAgent.chooseAction.predictedRewards
	predictedRewardsNumpy = DeepQAgent.chooseAction.predictedRewardsNumpy
	# Check that the predicted rewards are not just the default lists.
	if type(predictedRewards).__name__ == "list":
		print("Error in Task 4!")
		print("Predicted rewards is currently a static array! It should come from the Q network!")
		exit(0)
	if type(predictedRewardsNumpy).__name__ == "list":
		print("Error in Task 4!")
		print("predictedRewardsNumpy is currently a static array! It should be the result of converting predictedRewards.")
		exit(0)
	# Check that predicted rewards are the correct type
	if type(predictedRewards).__name__ != "Tensor":
		print("Error in Task 4!")
		print("predictedRewards should be a tensor! Are you getting it from the neural network?")
		exit(0)
	if type(predictedRewardsNumpy).__name__ != "ndarray":
		print("Error in Task 4!")
		print("predictedRewardsNumpy should be a numpy array! Are you converting it from the Tensor?")
		exit(0)
	# Check that the values match as they should.
	if not np.array_equal(predictedRewardsNumpy, predictedRewards.detach().numpy()):
		print("Error in Task 4!")
		print("predictedRewardsNumpy doesn't seem to match with predictedRewards, Are you converting it right?")
		exit(0)
	if np.argmax(predictedRewardsNumpy) != bestAction:
		print("Error in Task 4!")
		print("You aren't selecting the action with the highest value from predictedRewardsNumpy!")
		exit(0)

except Exception as e:
	print("Error in Task 4!")
	print(e)
	exit(0)

print("Task 4 successful! You can now use the Q function to choose an action!")
print()

# Test task 5 - Collecting transitions
DeepQAgent.collectTransitions()
# Test that there are entries in the replayMemory
if len(DeepQAgent.replayMemory) == 0:
	print("Error in Task 5!")
	print("Your replay memory is empty! Is your collectTransitions function appending to replayMemory?")
	exit(0)
# Test that the Transitions have the right structure
testTransition = DeepQAgent.replayMemory[0]

if type(testTransition.state).__name__ != "Tensor":
	print("Your transitions should contain values of the type Tensor in the state field, Your's contain values of the type {} instead!".format(type(testTransition.state).__name__))
	exit(0)

if type(testTransition.action).__name__ != "int64":
	print("Your transitions should contain values of the type int64 in the action field, Your's contain values of the type {} instead!".format(type(testTransition.action).__name__))
	exit(0)

if type(testTransition.reward).__name__ != "float32":
	print("Your transitions should contain values of the type float32 in the reward field, Your's contain values of the type {} instead!".format(type(testTransition.reward).__name__))
	exit(0)

if type(testTransition.nextState).__name__ != "Tensor":
	print("Your transitions should contain values of the type Tensor in the nextState field, Your's contain values of the type {} instead!".format(type(testTransition.nextState).__name__))
	exit(0)

print("Task 5 successful! You can now collect transitions to use for training!")

mse = torch.nn.MSELoss()
optimizer = torch.optim.RMSprop(dqn.parameters())
batchSize = 16
DeepQAgent.collectTransitions()
DeepQAgent.trainStep(mse, optimizer, batchSize)

print("Task 6 successful! You can now run a training step on your agent!")
