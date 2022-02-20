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
# 	1. Person = namedtuple("Person", ["name", "age"])
#	2. harvey = Person("Harvey Brezina Conniffe", 20)
#	3. print("Harvey's age is "+harvey.age)
#
# YOUR CODE: Create a Transition namedtuple that stores 4 values. state, action, reward and nextState.
Transition = namedtuple("X", ["X"])
