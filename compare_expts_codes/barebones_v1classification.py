import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
from model import baseline_lstm
import time
from collections import defaultdict
from utils import *
import pickle
import torch.nn.functional as F
import numpy

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
          feats_fname = feats_dir + '/' + fname + '.mgc_ascii'
          feats = np.loadtxt(feats_fname)
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



tdd_file = '/home/srallaba/projects/siri/classification_task_data/compare/train.txt'
feats_dir = '/home/srallaba/projects/siri/classification_task_data/compare/mgc/Train/'
train_set = arctic_dataset(tdd_file, feats_dir)
train_loader = DataLoader(train_set,
                          batch_size=16,
                          shuffle=True,
                          num_workers=4,
                          collate_fn=collate_fn_chopping
                          )

tdd_file = '/home/srallaba/projects/siri/classification_task_data/compare/test.txt'
feats_dir = '/home/srallaba/projects/siri/classification_task_data/compare/mgc/Test/'
val_set = arctic_dataset(tdd_file, feats_dir)
val_loader = DataLoader(val_set,
                          batch_size=16,
                          shuffle=True,
                          num_workers=4,
                          collate_fn=collate_fn_chopping
                          )



## Model
model = baseline_lstm()
print(model)
if torch.cuda.is_available():
   model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer_adam = torch.optim.Adam(model.parameters(), lr=0.0001)
optimizer_sgd = torch.optim.SGD(model.parameters(), lr=0.001)
optimizer = optimizer_sgd
updates = 0
regularizer = nn.MSELoss()



def val():
  model.eval()
  test_loss = 0
  y_true = []
  y_pred = []

  for i, (ccoeffs,labels) in enumerate(val_loader):
 
    inputs = torch.FloatTensor(ccoeffs)
    targets = torch.LongTensor(labels)
    inputs, targets = Variable(inputs), Variable(targets)
    if torch.cuda.is_available():
        inputs = inputs.cuda()
        targets = targets.cuda()

    logits = model(inputs)
#    print("logits are:", logits, "targets are:", targets)
    loss = criterion(logits, targets)
    test_loss += loss.item() 

    predicteds = return_classes(logits).cpu().numpy() 
#    print("preds", predicteds)
    for (t,p) in list(zip(targets, predicteds)):  
         y_true.append(t.item())
         y_pred.append(p)
 #        print("actual", t, "got", p)
 # print("compaing", y_true, y_pred)
  print(classification_report(y_true, y_pred))
  return test_loss/(i+1)
 
def train():
  model.train()
  optimizer.zero_grad()
  start_time = time.time()
  total_loss = 0
  global updates
  for i, (ccoeffs,labels) in enumerate(train_loader):
    updates += 1
    #print(labels)
 
    inputs = torch.FloatTensor(ccoeffs)
    targets = torch.LongTensor(labels)
    inputs, targets = Variable(inputs), Variable(targets)
    if torch.cuda.is_available():
        inputs = inputs.cuda()
        targets = targets.cuda()

#    print(numpy.shape(inputs))
    logits = model(inputs)
    optimizer.zero_grad()
    loss = criterion(logits, targets)
    total_loss += loss.item() 

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
    optimizer.step()
 
    # This 100 cannot be hardcoded
    if i % 100 == 1 and write_intermediate_flag:
       g = open(logfile_name, 'a')
       g.write("  Train loss after " + str(updates) +  " batches: " + str(l/(i+1)) + " " + str(r/(i+1)) + ". It took  " + str(time.time() - start_time) + '\n')
       g.close()


  return total_loss/(i+1)  


def main():
   for epoch in range(10):
     train_loss = train()
     val_loss = val()
     print("Train loss after ", epoch, " epochs: " , train_loss, "Val loss: ", val_loss )
     

main()
