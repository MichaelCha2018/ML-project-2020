import os
import gym
import cv2
import numpy as np

def preprocess(observation):
    """
    image preprocess
    :param observation:
    :return:
    """
    #observation = cv2.cvtColor(cv2.resize(observation, (84, 110)), cv2.COLOR_BGR2GRAY)
    #observation = observation[26:110,:]
    observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
    observation = cv2.resize(observation, (84, 84), interpolation=cv2.INTER_AREA)
    #observation = np.expand_dims(observation, -1)
    #ret, observation = cv2.threshold(observation,1,255,cv2.THRESH_BINARY)
    #x = np.reshape(observation,(84, 84, 1))
    #return x.transpose((2, 0, 1))
    return observation / 255.0

def update_state(state, next_state):
    obs_small = preprocess(next_state)
    state[0 : 3, :, :] = state[1 :, :, :]
    state[3, :, :] = obs_small
    return state


def evaluate(env,policy,num_evaluate_episodes,is_render):
    for j in range(num_evaluate_episodes):
        is_render = j % 10 == 0
        print(j)
        obs = env.reset()
        obs = env.reset()
        obs = preprocess(obs)
        obs = np.stack([obs] * 4, axis=0)
        done = False
        ep_ret = 0
        ep_len = 0
        while not(done):
            if is_render:
                env.render()
            # Take deterministic actions at test time 
            ac = policy.step(obs)
            next_obs, reward, done, _ = env.step(ac)
            next_obs = update_state(obs, next_obs)
            policy.agent.learn(obs, ac, reward, next_obs, done)
            obs = next_obs
            ep_ret += reward
            ep_len += 1
        policy.logkv_mean("TestEpRet", ep_ret)
        policy.logkv_mean("TestEpLen", ep_len)
    policy.dumpkvs()