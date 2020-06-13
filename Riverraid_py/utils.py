import random, math
from collections import deque
from IPython import display
import matplotlib.pyplot as plt
import pickle
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd

import gym
from gym import wrappers
import atari_wrapper


def plot_score(mean_episode_score, episode):
    plt.figure()
    plt.title('Average Score : ' + str(int(mean_episode_score[-1])) + '  / Episode: ' + str(episode))
    plt.xlabel('Episode (x 1)')
    plt.ylabel('Average Score ')
    # (over last 20 episodes)
    plt.plot(mean_episode_score)
    display.display(plt.gcf())
    display.clear_output(wait=True)
    plt.pause(1e-6)