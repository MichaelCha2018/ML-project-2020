import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
from model import DQNModel, ReplayBuffer
import random
import gym
from gym import wrappers
import atari_wrapper
import math
import pickle
import logging
from collections import deque

# Hyperparameters :
lr = 0.00025  # Learning rate
alpha = 0.95  # For RMSprop momentum
max_episodes = 300000  # About 5000000 frame
batch_size = 32
target_update = 10000
gamma = 0.99
rep_buf_size = 1000000
rep_buf_ini = 50000
skip_frame = 4
epsilon_start = 1.0
epsilon_final = 0.1
epsilon_decay = 300000
TAU = 2e-3  # For soft update


def epsilon_by_frame(step_idx):
    """Epsilon Decay: From 1 to 0.01 ：In 1M Frames"""
    epsilon_true = epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * step_idx / epsilon_decay)
    return epsilon_true


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


def main():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("use_cuda: ", use_cuda)
    print("Device: ", device)

    env = atari_wrapper.make_atari('RiverraidNoFrameskip-v4')
    env = atari_wrapper.wrap_deepmind(env, clip_rewards=False, frame_stack=True, pytorch_img=True)

    action_space = [a for a in range(env.action_space.n)]
    n_action = len(action_space)

    # DQN Model and optimizer:
    policy_model = DQNModel().to(device)
    target_model = DQNModel().to(device)
    target_model.load_state_dict(policy_model.state_dict())

    optimizer = torch.optim.RMSprop(policy_model.parameters(), lr=lr, alpha=alpha)

    # Initialize the Replay Buffer
    replay_buffer = ReplayBuffer(rep_buf_size)

    while len(replay_buffer) < rep_buf_ini:

        observation = env.reset()
        done = False

        while not done:
            with torch.no_grad():
                t_observation = torch.from_numpy(observation).float().to(device)
                t_observation = t_observation.view(1, t_observation.shape[0], t_observation.shape[1],
                                                   t_observation.shape[2])
                action = random.sample(range(len(action_space)), 1)[0]

            next_observation, reward, done, info = env.step(action_space[action])

            replay_buffer.push(observation, action, reward, next_observation, done)
            observation = next_observation

    print('Experience Replay buffer initialized')

    # Use log to record the performance
    logger = logging.getLogger('dqn_Riverraid')
    logger.setLevel(logging.INFO)
    logger_handler = logging.FileHandler('./dqn_Riverraid.log')
    logger.addHandler(logger_handler)




    # Training part
    env.reset()
    score = 0
    episode_score = []
    mean_episode_score = []
    episode_true = 0
    num_frames = 0
    episode = 0
    last_100episode_score = deque(maxlen=100)

    while episode < max_episodes:

        observation = env.reset()
        done = False
        # import time
        # start=time.time()

        while not done:

            with torch.no_grad():

                t_observation = torch.from_numpy(observation).float().to(device) / 255
                t_observation = t_observation.view(1, t_observation.shape[0], t_observation.shape[1],
                                                   t_observation.shape[2])
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

                for target_param, policy_param in zip(target_model.parameters(), policy_model.parameters()):
                    target_param.data.copy_(TAU * policy_param.data + (1 - TAU) * target_param.data)

        episode += 1
        # episode_score.append(score)
        # end=time.time()
        # print("Running time ( %i episode): %.3f Seconds "%(episode ,end-start))

        if info['ale.lives'] == 0:
            # episode_score.append(score)
            mean_score = score
            episode_true += 1
            score = 0

            # if episode % 20 == 0:
            # mean_score = np.mean(episode_score)
            mean_episode_score.append(mean_score)
            last_100episode_score.append(mean_score)
            # episode_score = []
            logger.info('Frame: ' + str(num_frames) + ' / Episode: ' + str(episode_true) + ' / Average Score : ' + str(
                int(mean_score))
                        + '   / epsilon: ' + str(float(epsilon)))
            #plot_score(mean_episode_score, episode_true)
            pickle.dump(mean_episode_score, open('./dqn_Riverraid_mean_scores.pickle', 'wb'))
            if episode_true % 50 == 1:
                logger.info(
                    'Frame: ' + str(num_frames) + ' / Episode: ' + str(episode_true) + ' / Average Score : ' + str(
                        int(mean_score))
                    + '   / epsilon: ' + str(float(epsilon)) + '   / last_100episode_score: ' + str(
                        float(np.mean(last_100episode_score))))

        if episode % 50 == 0:
            torch.save(target_model.state_dict(), './dqn_spaceinvaders_target_model_state_dict.pt')
            torch.save(policy_model.state_dict(), './dqn_spaceinvaders_model_state_dict.pt')

    pass


if __name__ == '__main__':
    main()
