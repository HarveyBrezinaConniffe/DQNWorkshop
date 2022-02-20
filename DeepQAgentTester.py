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
	print("Your 'Transition' namedtuple has the fields {}. Remember that we want it to store state, action, reward and nextState!")
	exit(0)

print("Task 1 successful! Your Transition namedTuple is correctly defined!")
