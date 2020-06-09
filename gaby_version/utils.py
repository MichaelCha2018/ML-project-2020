import cv2
import numpy as np
import torch

def WarpFrame(obs):
    """
    :param obs: The raw observation returned by env, it should be a (210 * 160 * 3) RGB frame
    :return: ans: A (84 * 84) compressed gray style frame normalized in [0, 1]
    """
    frame = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
    #return frame[:, :, None]
    return frame

def Uint2Float(obs):
    return obs / 255.0

def FrameStack(new_obs, obs):
    """
    :param new_obs: A raw observation returned by env, it should be a (210 * 160 * 3) RGB frame
    :param obs: The stack of past 4 (84 * 84) compressed gray style frames
    :return: A new stack of past 4 (84 * 84) compressed gray style frames
    """
    new_obs = WarpFrame(new_obs)
    obs[0 : 3, :, :] = obs[1 :, :, :]
    obs[3, :, :] = new_obs
    return obs
