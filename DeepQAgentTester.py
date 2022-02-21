from collections import deque

try:
	import torch
except Exception as e:
	print(e)
	print("Failed to import pytorch!")
	print("Hint: Have you installed pytorch yet?")
	print("If not see: https://pytorch.org/get-started/locally/")

try:
	import gym
except Exception as e:
	print(e)
	print("Failed to import openAI gym!")
	print("Hint: Have you installed gym yet?")
	print("If not see: https://gym.openai.com/docs/")

try:
	env = gym.make('CartPole-v1')
except Exception as e:
	print(e)
	print("Failed to initialize an environment!")
	print("Hint: You may need to install the classic control package")
	print("Try running: pip install gym[classic_control]")

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
