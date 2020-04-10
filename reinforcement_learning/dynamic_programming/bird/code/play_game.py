import random
import time
from reinforcement_learning.dynamic_programming.bird.code import bird_env


class Policy_Iter:
    def __init__(self, bird):
       self.states = bird.states
       self.actions = bird.actions


