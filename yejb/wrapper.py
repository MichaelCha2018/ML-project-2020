from os import path
import time
import world
from world import ARGS
import gym
from gym import Wrapper
import cv2 as cv
import numpy as np
from tensorboardX import SummaryWriter

class WrapIt(Wrapper):
    def __init__(self, 
                 env      : gym.Env,
                 log      : bool = True,
                 save_dir : str = 'tmp'):
        assert isinstance(env, gym.Env)
        super(WrapIt, self).__init__(env)
        # self.env = env
        self.log = log
        self.strtime = time.strftime("%m_%d_%H-%M")
        self.prename = path.join(save_dir, self.strtime)
        self.writer = SummaryWriter(self.prename)
        self._total_steps = 0
        self._total_rewards = 0.
        self._episodes = 0
        self._episode_rewards = []
    
    @staticmethod
    def resize(obs):
        obs = cv.cvtColor(obs, cv.COLOR_BGR2GRAY).astype('float32')
        obs = cv.resize(obs, dsize=(ARGS.imgDIM, ARGS.imgDIM)) / 255.0
        obs = obs[..., np.newaxis]
        return obs.astype('float32')
    
    def reset(self, passit=False):
        if not passit:
            self._total_steps += 1
            self._episodes += 1
            self._episode_rewards.append(self._total_rewards)
            self._total_rewards = 0.
            if self.log:
                self.writer.add_scalar("Rewards", self._episode_rewards[-1], self._episodes)
                # with open(self.prename+'.txt', 'a') as f:
                #     f.write(f"#{self._episodes} \t:: rewards={self._episode_rewards[-1]} \t: best={np.max(self._episode_rewards)} \t: mean={np.mean(self._episode_rewards):.3f} \n")
        # ------------------------------------------------------
        obs = self.env.reset()
        obs = WrapIt.resize(obs)
        return obs
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)    
        obs = WrapIt.resize(obs)
        # ------------------------------------------------------
        self._total_steps += 1
        self._total_rewards += reward
        return obs, reward, done, info

    def get_episode_rewards(self):
        return self._episode_rewards

    def getTotalStep(self):
        return self._total_steps
    
    def getEpisode(self):
        return self._episodes
    
    def close(self):
        self.writer.close()