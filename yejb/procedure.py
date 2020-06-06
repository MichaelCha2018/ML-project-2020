import world
import torch
import random
import gym
import numpy as np
from time import time
from tqdm import tqdm
from world import ARGS
from utils import Schedule, TO, TENSOR
from torch import Tensor
from torch.nn import Module
from models import DQN
from wrapper import WrapIt
from Buffer import ReplayBuffer
from collections import namedtuple
from torch.optim.optimizer import Optimizer


def train_DQN(env          : WrapIt, 
              Q            : DQN, 
              Q_target     : DQN, 
              optimizer    : namedtuple, 
              replay_buffer: ReplayBuffer,
              exploration  : Schedule):
    """
    @parameters
        Q:
        Q_target:
        optimizer: torch.nn.optim.Optimizer with parameters
        buffer: store the frame
    @return
        None
    """
    assert type(env.observation_space) == gym.spaces.Box
    assert type(env.action_space)      == gym.spaces.Discrete
    
    optimizer = optimizer.constructor(Q.parameters(), **optimizer.kwargs)
    
    num_actions = env.action_space.n
    num_param_updates = 0
    mean_episode_reward = -float('nan')
    best_mean_episode_reward = -float('inf')
    LOG_EVERY_N_STEPS = 10000
    last_obs = env.reset(passit=True)
    
    # Q.getSummary()
    
    out_count = 0
    bar = tqdm(range(ARGS.timesteps))
    for t in bar:
        last_idx = replay_buffer.store_frame(last_obs)
        recent_observations = replay_buffer.encode_recent_observation()
        if t > ARGS.startepoch:
            value = select_epsilon_greedy_action(Q, 
                                                 recent_observations, 
                                                 exploration, 
                                                 t, 
                                                 num_actions)
            action = value[0,0]
        else:
            action = random.randrange(num_actions)
        obs, reward, done, _ = env.step(action)
        reward = max(-1.0, min(reward, 1.0))
        replay_buffer.store_effect(last_idx, action, reward, done)
        
        if done:
            obs = env.reset()  
        last_obs = obs
        # bar.set_description(f"{obs.shape} {obs.dtype}")
        
        if (t > ARGS.startepoch and 
            t % ARGS.dqn_freq == 0 and 
            replay_buffer.can_sample(ARGS.batchsize)):
            bar.set_description("backward")
            (obs_batch, 
             act_batch, 
             rew_batch, 
             next_obs_batch, 
             done_mask) = replay_buffer.sample(ARGS.batchsize)
            (obs_batch, 
             act_batch, 
             rew_batch, 
             next_obs_batch, 
             not_done_mask) = TENSOR(obs_batch, act_batch, rew_batch, next_obs_batch, 1-done_mask)
            (obs_batch, 
             act_batch) = TO(obs_batch, act_batch)
            
            values = Q(obs_batch)
            current_Q_values = values.gather(1, act_batch.unsqueeze(1).long()).squeeze()
            # Compute next Q value based on which action gives max Q values
            # Detach variable from the current graph since we don't want gradients for next Q to propagated
            next_max_q = Q_target(next_obs_batch).detach().max(1)[0]
            next_Q_values = not_done_mask * next_max_q
            # Compute the target of the current Q values
            Q_target_values = rew_batch + (ARGS.gamma * next_Q_values)
            # Compute Bellman error
            bellman_error = Q_target_values - current_Q_values
            # clip the bellman error between [-1 , 1]
            clipped_bellman_error = bellman_error.clamp(-1, 1)
            # Note: clipped_bellman_delta * -1 will be right gradient
            d_error = clipped_bellman_error * -1.0
            # Clear previous gradients before backward pass
            optimizer.zero_grad()
            # run backward pass
            # current_Q_values.backward(d_error.data.unsqueeze(1))
            current_Q_values.backward(d_error.data)

            # Perfom the update
            optimizer.step()
            num_param_updates += 1
            
            if num_param_updates % ARGS.dqn_updatefreq == 0:
                bar.set_description("update")
                Q_target.load_state_dict(Q.state_dict())
            
            
def select_epsilon_greedy_action(model      : DQN, 
                                 obs        : Tensor, 
                                 exploration: Schedule, 
                                 t          : int,
                                 num_actions: int):
    """
    @parameter:
        model: DQN model
        obs: returned by env. Image
        exploration: value() will return a float represent the possibility of explore
        t: steps num
        num_actions: total number of available actions in env
    @return:
        Tensor with one data representing action
    """
    sample = random.random()
    eps_threshold = exploration.value(t)
    if sample > eps_threshold:
        obs = torch.from_numpy(obs).unsqueeze(0).float().to(world.DEVICE)
        with torch.no_grad():
            values = model(obs)
        return values.data.max(1)[1].cpu().unsqueeze(dim=1)
    else:
        return torch.IntTensor([[random.randrange(num_actions)]])