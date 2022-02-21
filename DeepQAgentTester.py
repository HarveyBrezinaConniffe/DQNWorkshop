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
	env = gym.make('CartPole-v0')
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
