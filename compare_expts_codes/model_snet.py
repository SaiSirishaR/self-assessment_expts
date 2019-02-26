import numpy as np
import os, sys
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import numpy

FALCON_DIR = '/home/srallaba/development/repos/falkon/'
sys.path.append(FALCON_DIR)
import src.nn.layers as layers

import torch.nn.functional as F
import random
import math


# Input: World ccoeffs of dim 60 Shape (B,T,C)
# Output: Logits of Shape (B,3)
class baseline_model(nn.Module):

      def __init__(self):
          super(baseline_model, self).__init__()


class soundnet_model(baseline_model):

        def __init__(self):
          super(soundnet_model, self).__init__()

          self.encoder_fc = nn.Linear(512, 256)
          self.dropout = nn.Dropout(0.4)

          self.decoder_fc = nn.Linear(256, 3)
         
        def forward(self, x):
 
          x = self.encoder_fc(x)
          x = self.dropout(x)
          return self.decoder_fc(x)

