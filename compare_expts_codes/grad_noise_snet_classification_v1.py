import numpy as np
from keras.layers.core import Dropout
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
from keras.optimizers import Adam, SGD
from utils import *


NoisyAdam = add_gradient_noise(Adam)

label_dict = defaultdict(int, l=0, h=2)
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



tdd_file = '/home/srallaba/projects/siri/classification_task_data/compare/scripts_exp1/keras_gradient_noise/train_lh_full.txt'
feats_dir = '/home/srallaba/projects/siri/classification_task_data/compare/soundnet_feats/Train/'
train_set = arctic_dataset(tdd_file, feats_dir)
train_loader = DataLoader(train_set,
                          batch_size=16,
                          shuffle=True,
                          num_workers=4,
                          collate_fn=collate_fn_chopping
                          )

tdd_file = '/home/srallaba/projects/siri/classification_task_data/compare/scripts_exp1/keras_gradient_noise/test_lhfull.txt'
feats_dir = '/home/srallaba/projects/siri/classification_task_data/compare/soundnet_feats/Test/'
val_set = arctic_dataset(tdd_file, feats_dir)
val_loader = DataLoader(val_set,
                          batch_size=16,
                          shuffle=True,
                          num_workers=4,
                          collate_fn=collate_fn_chopping
                          )



for i, (ccoeffs,labels) in enumerate(train_loader):
    inputs = ccoeffs.detach().numpy()
    targets = labels.detach().numpy()




def test_noisy_optimizer_with_simple_model_training():
        global model
        model = Sequential()
        model.add(Dense(512,  activation='selu'))
        model.add(Dense(64,  activation='selu'))
        model.add(Dense(64,  activation='selu'))
        model.add(Dense(64,  activation='selu'))
        model.add(Dense(64,  activation='selu'))
        model.add(Dense(3,  activation='softmax'))
        model.compile(optimizer=NoisyAdam(), loss='sparse_categorical_crossentropy')
#        sgd = SGD(lr=0.01, momentum=0.2, decay=1e-6, nesterov=False)
#        model.compile(optimizer=sgd, loss='sparse_categorical_crossentropy')
        model.fit(inputs, targets,  epochs=30)




test_noisy_optimizer_with_simple_model_training()
for i, (ccoeffs,labels) in enumerate(val_loader):
    val_inputs = ccoeffs.detach().numpy()
    val_outputs = labels.detach().numpy()
    predictions = model.predict(val_inputs)

    predicteds = return_classes(torch.Tensor(predictions))

    print("prediction are:", predicteds, "actual", val_outputs)
    print(classification_report(val_outputs, predicteds))
##    predicteds = return_classes(logits).cpu().numpy() 
 ##   for (t,p) in list(zip(targets, predicteds)):  
  #       y_true.append(t.item())
  #       y_pred.append(p)

