'''
    major program to run dqn
'''
import gym
import world
import utils
from Buffer import ReplayBuffer
from models import DQN
from world import Print, ARGS
from wrapper import PreProcess
from procedure import train_DQN


# ------------------------------------------------
env = gym.make('RiverraidNoFrameskip-v4')
env = PreProcess(env)
Print('ENV action', env.unwrapped.get_action_meanings())
Print('ENV observation', f"Image: {ARGS.imgDIM} X {ARGS.imgDIM} X {1}") # we assert to use gray image
# ------------------------------------------------
Optimizer = utils.getOptimizer()
schedule = utils.LinearSchedule(1000000, 0.1)

Gama_buffer = ReplayBuffer(ARGS.buffersize, ARGS.framelen)

Q = utils.init_model(env, DQN)
Q_target = utils.init_model(env, DQN)
# ------------------------------------------------
train_DQN(env, 
          Q=Q, 
          Q_target=Q_target)
