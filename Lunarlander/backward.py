"""
    specific design for QNetwork in model.py
"""

import torch
import torch.nn as nn

def backward(model : nn.Module, 
             loss  : torch.Tensor,
             test_model = None,
             test_loss = None):
    '''
        model:
        QNetwork(
            (fc1): Linear(in_features=8, out_features=256, bias=True)
            (fc2): Linear(in_features=256, out_features=128, bias=True)
            (fc3): Linear(in_features=128, out_features=64, bias=True)
            (out): Linear(in_features=64, out_features=4, bias=True)
        )
        out.bias
        out.weight
        fc3.bias
        fc3.weight
        fc2.bias
        fc2.weight
        fc1.bias
        fc1.weight
    '''
    if test_loss is not None:
        test_loss.backward()
    back_order = list(model.named_parameters())
    back_order.reverse()
    weights_dict = {name : para for name, para in back_order}
    # ----------------------------------------------------
    # out
    pass
    # weight_dict['out.bias'].grad = 
    # weight_dict['out.weight'].grad = 
    # ----------------------------------------------------
    # fc3
    pass
    # ----------------------------------------------------
    # fc2
    pass
    # ----------------------------------------------------
    # fc1
    pass
    
