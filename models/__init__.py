from tools import Struct
import tools.model.io as model_io

import math

import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

import models.simple as model





def add_arguments(parser):
    for name, (default, help) in model.parameters.items():
        parser.add_argument('--' + name, default=default, type=type(default), help=help)


def get_params(args):
    params = {}

    for name in model.parameters.keys():
        params[name] = getattr(args, name)

    return Struct(**params)


def create(params):
    return model.create(params)

def save(path, model, model_params, epoch):
    state = {
        'epoch': epoch,
        'params': model_params,
        'state': model.state_dict()
    }

    model_io.save(path, epoch, state)

def load(path):

    state = model_io.load(path)
    params = state['params']
    model = create(params)

    model.load_state_dict(state['state'])

    return model, params, state['epoch']
