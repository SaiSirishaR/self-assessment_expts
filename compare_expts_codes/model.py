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

class baseline_lstm(baseline_model):

        def __init__(self):
          super(baseline_lstm, self).__init__()

          self.encoder_fc = layers.SequenceWise(nn.Linear(60, 32))
#          self.encoder_dropout = layers.SequenceWise(nn.Dropout(0.7))

          self.seq_model = nn.LSTM(32, 64, 1, bidirectional=True, batch_first=True)
          self.prefinal_fc = layers.SequenceWise(nn.Linear(128, 32))
          self.final_fc = nn.Linear(128, 3)

        def forward(self, c):

           x = self.encoder_fc(c)
#           x = self.encoder_dropout(x)

           x, (c,h) = self.seq_model(x, None)
#           print("x", numpy.shape(x), "c", numpy.shape(c), "h", numpy.shape(h))
#           print("h0 is:", numpy.shape(h[0,:,:]), "h1 is:", numpy.shape(h[1,:,:]))
           hidden_left , hidden_right = h[0,:,:], h[1,:,:]
           hidden = torch.cat((hidden_left, hidden_right),1)
#           print("aftr cat", numpy.shape(hidden))
           x = self.final_fc(hidden)
           return x

        def forward_eval(self, c):

           x = self.encoder_fc(c)

           x, (c,h) = self.seq_model(x, None)
           hidden_left , hidden_right = h[0,:,:], h[1,:,:]
           hidden = torch.cat((hidden_left, hidden_right),1)
           x = self.final_fc(hidden)
           return x


