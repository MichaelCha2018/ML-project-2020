import gym
import time

import random
import torch
import numpy as np
from dqn_agent import Agent
from collections import deque

env = gym.make('LunarLander-v2')

agent = Agent(state_size=8, action_size=4, seed=0)
agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth',map_location=torch.device('cpu')))


for i in range(3):
    state = env.reset()
    for j in range(1000):
        action = agent.act(state)
        env.render()
        time.sleep(0.0005)
        state, reward, done, _ = env.step(action)
        if done:
            break 
            
env.close()