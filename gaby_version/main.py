import gym
from utils import WarpFrame, FrameStack
from matplotlib import pyplot as plt
import cv2
import math
import numpy as np
from model import DQNAgent
import time
from tensorboardX import SummaryWriter

EPSILON_START = 1.0
EPSILON_FINAL = 0.1
EPSILON_DECAY = 250000
EPISODES = 5000

epsilon_by_frame = lambda step_idx: EPSILON_FINAL + (EPSILON_START - EPSILON_FINAL) * math.exp(-1. * step_idx / EPSILON_DECAY)
writer = SummaryWriter(comment='DQN')

num_frames = 0
env = gym.make('Riverraid-v0')
agent = DQNAgent(env)
is_render = False
for i_episode in range(EPISODES):
    score = 0
    observation = env.reset()
    observation = WarpFrame(observation)
    observation = np.stack([observation] * 4, axis=0)
    done = False
    #is_render = i_episode % 10 == 0
    t = time.time()
    loss = []
    while not done:
        if is_render:
            env.render()
        #print(observation.shape)
        """
        if t % 10 == 0:
            cv2.imshow('frame', observation)
            cv2.waitKey(0)
        """
        #action = env.action_space.sample()
        epsilon = epsilon_by_frame(num_frames)
        action = agent.choose_action(observation, epsilon)
        new_observation, reward, done, info = env.step(action)
        new_observation = FrameStack(new_observation, observation)
        agent.replay_buffer.push(observation, action, reward, new_observation, done)
        observation = new_observation
        loss_frame = agent.learn(num_frames)
        loss.append(loss_frame)
        num_frames += 1
        score += reward
    t = time.time() - t
    loss = np.mean(loss)
    writer.add_scalar('dqn/loss', loss, global_step=i_episode)
    writer.add_scalar('dqn/score', score, global_step=i_episode)
    print(f"Episode: {i_episode} \t score: {score} \t time used: {t}s \t actions taken: {num_frames} \t epsilon: {epsilon}")
env.close()