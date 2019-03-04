import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
import time
from collections import defaultdict
import pickle
import torch.nn.functional as F
import numpy
from keras.layers import Dense
from keras.models import Model, Sequential
from keras_gradient_noise import add_gradient_noise
from keras.optimizers import Adam

NoisyAdam = add_gradient_noise(Adam)

label_dict = defaultdict(int, l=0, m=1, h=2)
int2label = {i:w for w,i in label_dict.items()}

write_intermediate_flag = 0

class arctic_dataset(Dataset):

    def __init__(self, tdd_file, feats_dir):

        self.tdd_file = tdd_file
        self.feats_dir = feats_dir
        self.labels_array = []
        self.feats_array = [] 
        f = open(self.tdd_file)
        for line in f:
          line = line.split('\n')[0]  # removes trailing '\n' in line
          fname = line.split()[0]
          feats_fname = feats_dir + '/' + fname + '.npz'
          feats = numpy.load(feats_fname, encoding='latin1')
          a = feats['arr_0']
          feats = np.mean(a[4],axis=0)
#          print("feats ahpe is:", numpy.shape(feats))
          self.feats_array.append(feats)
          label = line.split()[1]
          self.labels_array.append(label)

    def __getitem__(self, index):
#          print("feats arry", self.feats_array[index], "labels", self.labels_array[index])
          return self.feats_array[index], self.labels_array[index]

    def __len__(self):
           return len(self.labels_array)



def collate_fn_chopping(batch):
    '''
    Separates given batch into array of y values and array of truncated x's
    All given x values are truncated to have the same length as the x value
    with the minimum length
    Args:
        batch: raw batch of data; array of x,y pairs
    
    Return:
        a_batch: batch-length array of float-array x values
        b_batch: batch-length array of int y values
    '''
    input_lengths = [len(x[0]) for x in batch]
#    print("all length", input_lengths)
    min_input_len = np.min(input_lengths)
#    print("max input length is", min_input_len)
#    print("x is", [numpy.shape(x[0]) for x in batch],"lets c", [x[0][:min_input_len] for x in batch])
    a = np.array( [ x[0][:min_input_len]  for x in batch ], dtype=np.float)
    b = np.array( [ label_dict[x[1]]  for x in batch ], dtype=np.int)
    a_batch = torch.FloatTensor(a)
    b_batch = torch.LongTensor(b)
    return a_batch, b_batch



tdd_file = '/home/srallaba/projects/siri/classification_task_data/compare/train_fem.txt'
feats_dir = '/home/srallaba/projects/siri/classification_task_data/compare/soundnet_feats/Train/'
train_set = arctic_dataset(tdd_file, feats_dir)
train_loader = DataLoader(train_set,
                          batch_size=32,
                          shuffle=True,
                          num_workers=4,
                          collate_fn=collate_fn_chopping
                          )

tdd_file = '/home/srallaba/projects/siri/classification_task_data/compare/test_fem_small.txt'
feats_dir = '/home/srallaba/projects/siri/classification_task_data/compare/soundnet_feats/Test/'
val_set = arctic_dataset(tdd_file, feats_dir)
val_loader = DataLoader(val_set,
                          batch_size=32,
                          shuffle=True,
                          num_workers=4,
                          collate_fn=collate_fn_chopping
                          )



for i, (ccoeffs,labels) in enumerate(train_loader):
    print("coeefs are:", numpy.shape(ccoeffs.detach().numpy()))
    inputs = ccoeffs.detach().numpy()
    targets = labels.detach().numpy()



def test_noisy_optimizer_with_simple_model_training():
        model = Sequential()
        model.add(Dense(256, input_shape=(512,)))
        model.add(Dense(64,  activation='selu'))
        model.add(Dense(1,  activation='selu'))
        model.compile(optimizer=NoisyAdam(), loss='mse')
        model.fit(inputs, targets, epochs=4)




test_noisy_optimizer_with_simple_model_training()

