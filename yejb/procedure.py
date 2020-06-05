import world
import torch
import random
from utils import Schedule
from torch import Tensor
from models import DQN
from Buffer import ReplayBuffer
from collections import namedtuple
from torch.optim.optimizer import Optimizer


def train_DQN(self, 
              Q         : DQN, 
              Q_target  : DQN, 
              optimizer : namedtuple, 
              buffer    : ReplayBuffer):
    """
    @parameters

        pass
        
    @return
    
        pass
    """
    pass


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
        obs = torch.from_numpy(obs).unsqueeze(0)
        with torch.no_grad():
            values = model(obs)
        return values.data.max(1)[1].cpu().unsqueeze(dim=1)
    else:
        return torch.IntTensor([[random.randrange(num_actions)]])