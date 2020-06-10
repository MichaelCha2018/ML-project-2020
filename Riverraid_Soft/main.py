import random, math
from collections import deque
import pickle
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
import gym
from gym import wrappers
import atari_wrappers

logger = logging.getLogger('dqn_spaceinvaders')
logger.setLevel(logging.INFO)
logger_handler = logging.FileHandler('./data/dqn_spaceinvaders.log')
logger.addHandler(logger_handler)

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")
print(use_cuda)
print(device)

class DQNModel(nn.Module):
    def __init__(self):
        super(DQNModel, self).__init__()
        
        self.conv = nn.Sequential(nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU(),
                                  nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
                                  nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(7 * 7 * 64, 512), nn.ReLU(), nn.Linear(512, n_actions))

    def forward(self, obs):
        obs = self.conv(obs)
        obs = obs.view(obs.shape[0], obs.shape[1] * obs.shape[2] * obs.shape[3])
        actions = self.fc(obs)
        return actions
    
    
env = atari_wrappers.make_atari('RiverraidNoFrameskip-v4')
env = atari_wrappers.wrap_deepmind(env, clip_rewards=False, frame_stack=True, pytorch_img=True)
action_space = [a for a in range(env.action_space.n)]
print(env.observation_space)
print(env.unwrapped.get_action_meanings())

# -------------------------------------------------------------------
n_actions = len(action_space)

lr = 0.00030
alpha = 0.95

policy_model = DQNModel().to(device)
target_model = DQNModel().to(device)
target_model.load_state_dict(policy_model.state_dict())

optimizer = torch.optim.RMSprop(policy_model.parameters(), lr=lr, alpha=alpha)
# -------------------------------------------------------------------
max_episodes = 100000
batch_size = 32
target_update = 5000
gamma = 0.99
rep_buf_size = 250000
rep_buf_ini = 25000
skip_frame = 4
epsilon_start = 1.0
epsilon_final = 0.1
epsilon_decay = 400000
# -------------------------------------------------------------------
epsilon_by_frame = lambda step_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * step_idx / epsilon_decay)
# -------------------------------------------------------------------
class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append([state, action, reward, next_state, done])
    
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

def huber_loss(input, target, beta=1, size_average=True):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    n = torch.abs(input - target)
    cond = n < beta
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    if size_average:
        return loss.mean()
    return loss.sum()
# -------------------------------------------------------------------
replay_buffer = ReplayBuffer(rep_buf_size)
while len(replay_buffer) < rep_buf_ini:
    
    observation = env.reset()
    done = False
    
    while not done:

        with torch.no_grad():
            t_observation = torch.from_numpy(observation).float().to(device)
            t_observation = t_observation.view(1, t_observation.shape[0], t_observation.shape[1], t_observation.shape[2])
            action = random.sample(range(len(action_space)), 1)[0]
        
        next_observation, reward, done, info = env.step(action_space[action])
            
        replay_buffer.push(observation, action, reward, next_observation, done)
        observation = next_observation
        
print('Experience Replay buffer initialized')
# -------------------------------------------------------------------
episode_score = []
mean_episode_score = []
episode_score = []
mean_episode_score = []
TAU = 2e-3
num_frames = 0
episode = 0
score = 0
# -------------------------------------------------------------------
episode_score = []
mean_episode_score = []
TAU = 2e-3

num_frames = 0
episode = 0
score = 0

while episode < max_episodes:
    
    observation = env.reset()
    done = False
    #import time
    #start=time.time()
    
    while not done:

        with torch.no_grad():

            t_observation = torch.from_numpy(observation).float().to(device) / 255
            t_observation = t_observation.view(1, t_observation.shape[0], t_observation.shape[1], t_observation.shape[2])
            epsilon = epsilon_by_frame(num_frames)
            if random.random() > epsilon:
                q_value = policy_model(t_observation)
                action = q_value.argmax(1).data.cpu().numpy().astype(int)[0]
            else:
                action = random.sample(range(len(action_space)), 1)[0]
        
        next_observation, reward, done, info = env.step(action_space[action])
        num_frames += 1
        score += reward
            
        replay_buffer.push(observation, action, reward, next_observation, done)
        observation = next_observation
        
        # Update policy
        if len(replay_buffer) > batch_size and num_frames % skip_frame == 0:
            observations, actions, rewards, next_observations, dones = replay_buffer.sample(batch_size)          

            observations = torch.from_numpy(np.array(observations) / 255).float().to(device)
            
            actions = torch.from_numpy(np.array(actions).astype(int)).float().to(device)
            actions = actions.view(actions.shape[0], 1)
            
            rewards = torch.from_numpy(np.array(rewards)).float().to(device)
            rewards = rewards.view(rewards.shape[0], 1)
            
            next_observations = torch.from_numpy(np.array(next_observations) / 255).float().to(device)
            
            dones = torch.from_numpy(np.array(dones).astype(int)).float().to(device)
            dones = dones.view(dones.shape[0], 1)
            
            q_values = policy_model(observations)
            next_q_values = target_model(next_observations)

            q_value = q_values.gather(1, actions.long())
            next_q_value = next_q_values.max(1)[0].unsqueeze(1)
            expected_q_value = rewards + gamma * next_q_value * (1 - dones)

            loss = huber_loss(q_value, expected_q_value)

            optimizer.zero_grad()
            loss.backward()
            
            optimizer.step()
            
            for target_param, policy_param in zip(target_model.parameters(),policy_model.parameters()):
                target_param.data.copy_(TAU*policy_param.data + (1-TAU)*target_param.data)
            
    episode += 1
    episode_score.append(score)
    print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(episode_score)), end="")
    if info['ale.lives'] == 0:
        score = 0
    if episode % 100 == 0:
        mean_score = np.mean(episode_score)
        mean_episode_score.append(mean_score)
        episode_score = []
        print("------------------------")
        logger.info('Frame: ' + str(num_frames) + ' / Episode: ' + str(episode) + ' / Average Score (over last 20 episodes): ' + str(int(mean_score)))
        torch.save(policy_model.state_dict(), './data/dqn_riverraid_model_state_dict.pt')