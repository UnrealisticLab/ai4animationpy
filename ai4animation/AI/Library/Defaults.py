# Copyright (c) Meta Platforms, Inc. and affiliates.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ai4animation.AI import Manifolds
from einops import rearrange
from torch.nn.parameter import Parameter

Dropout = 0.1
Activation = F.elu
Normalize = True
