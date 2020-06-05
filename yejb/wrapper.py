import world
from world import ARGS
import gym
from gym import Wrapper
import cv2 as cv
import numpy as np


class PreProcess(Wrapper):
    def __init__(self, env : gym.Env):
        assert isinstance(env, gym.Env)
        super(PreProcess, self).__init__(env)
        # self.env = env
        self.steps = 0
    
    @staticmethod
    def resize(obs):
        obs = cv.cvtColor(obs, cv.COLOR_BGR2GRAY)
        # obs = cv.resize(obs, dsize=(obs.shape[1]//2, obs.shape[0]//2))
        obs = cv.resize(obs, dsize=(ARGS.imgDIM, ARGS.imgDIM)) / 255.0
        obs = obs[..., np.newaxis]
        return obs
    
    def reset(self):
        obs = self.env.reset()
        obs = PreProcess.resize(obs)
        return obs
    
    def step(self, action):
        self.steps += 1
        obs, reward, done, info = self.env.step(action)    
        obs = PreProcess.resize(obs)
        return obs, reward, done, info

    def getTotalStep(self):
        return self.steps