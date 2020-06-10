import gym
import random
import torch
import numpy as np
from collections import deque

seed = 0

env = gym.make('LunarLander-v2')
env.seed(seed)
np.random.seed(seed)   
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.manual_seed(seed)


print('State shape: ', env.observation_space.shape)
print('Number of actions: ', env.action_space.n)

print(env.observation_space, env.action_space)
print(env.observation_space.sample())

from dqn_agent import Agent
import time
agent = Agent(state_size=8, action_size=4, seed=0)

# watch an untrained agent running for 1 episode.
state = env.reset()
for j in range(200):
    action = agent.act(state)
    time.sleep(.0125)
    env.render()
    state, reward, done, _ = env.step(action)
    if done:
        break 
env.close()

def dqn(n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            #env.render()
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 
        # print(f"#{i_episode}: {score}")
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        # if np.mean(scores_window)>=200.0:
        #     print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            #torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth') 
            # break
    return scores

scores = dqn()