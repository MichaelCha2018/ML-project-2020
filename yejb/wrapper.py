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
        self.step = 0
    
    @staticmethod
    def resize(obs):
        obs = cv.cvtColor(obs, cv.COLOR_BGR2GRAY)
        # obs = cv.resize(obs, dsize=(obs.shape[1]//2, obs.shape[0]//2))
        obs = cv.resize(obs, dsize=(ARGS.imgDIM, ARGS.imgDIM)) / 255.0
        obs = obs[..., np.newaxis]
        return obs
    
    def _reset(self):
        obs = self.env.reset()
        obs = PreProcess.resize(obs)
        return obs
    
    def _step(self, action):
        self.step += 1
        results = self.env.step(action)    
        results[0] = PreProcess.resize(results[0])
        return results

    def getTotalStep(self):
        return self.step