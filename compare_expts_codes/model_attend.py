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



class baseline_model(nn.Module):

      def __init__(self):
          super(baseline_model, self).__init__()


class attentionlstm(baseline_model):

        def __init__(self):
          super(attentionlstm, self).__init__()


          self.encoder_fc = layers.SequenceWise(nn.Linear(60,32))
          self.encoder_dropout = layers.SequenceWise(nn.Dropout(0.3))

          self.seq_model = nn.LSTM(32, 64, 1, bidirectional=True, batch_first=True)
          self.final_fc = nn.Linear(128, 3)
          self.scale = 1. / math.sqrt(60)



        def attend(self, S,lstm_out, lstm_out1):

           lstm_out = lstm_out.transpose(0,0).transpose(2,1)
           S=S.unsqueeze(1)
           energy = torch.bmm(S,lstm_out)
           energy = F.softmax(energy.mul_(self.scale), dim=2)

           linear_combination = torch.bmm(energy, lstm_out1).squeeze(1)
        
           return linear_combination

        def forward(self, c):
     
           x = self.encoder_fc(c)

           x, (h,c) = self.seq_model(x, None)
           hidden_left , hidden_right = h[0,:,:], h[1,:,:]
           hidden = torch.cat((hidden_left, hidden_right),1)
           weighted_representation = self.attend(hidden, x,x)
           x = self.final_fc(weighted_representation)
           return x

        def forward_eval(self, c):

           x = self.encoder_fc(c)

           x, (h, c) = self.seq_model(x, None)
           hidden_left , hidden_right = h[0,:,:], h[1,:,:]
           hidden = torch.cat((hidden_left, hidden_right),1)

           weighted_representation = self.attend(hidden)
           x = self.final_fc(weighted_representation)

           return x

