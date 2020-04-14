import pandas as pd
import numpy as np

import _pickle as pickle
from collections import Counter
from pprint import pprint
from datetime import datetime
import shutil
import time
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau


####################### Classifer for outputs from the Stacked GRU #######################
class ClassPredictor(nn.Module):
    def __init__(self, input_size, num_classes, drop_prob):
        super(ClassPredictor, self).__init__()
        
        self.input_dout = nn.Dropout(drop_prob) 
        
        hidden_1 = 120
        hidden_2 = 80        
        
        self.fc1 = nn.Linear(input_size, hidden_1)
        self.l_relu1 = nn.LeakyReLU()
        self.dout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.l_relu2 = nn.LeakyReLU()
        self.dout2 = nn.Dropout(0.1)

        self.out = nn.Linear(hidden_2, num_classes)
        
        nn.init.orthogonal_(self.fc1.weight).requires_grad_().cuda()
        nn.init.orthogonal_(self.fc2.weight).requires_grad_().cuda()
        nn.init.orthogonal_(self.out.weight).requires_grad_().cuda()


    def forward(self, x):
        ## x: (input_size)

        # Manually use dropout for the Segment BiGRU output
        x = self.input_dout(x)
        
        a1 = self.fc1(x)
        h1 = self.l_relu1(a1)
        dout1 = self.dout1(h1)

        a2 = self.fc2(dout1)
        h2 = self.l_relu2(a2)
        dout2 = self.dout2(h2)

        # y: (num_classes)
        y = self.out(dout2)

        return y


####################### Stacked Bidirectional GRU #######################
class StackedBiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, segment_hidden_size, num_layers, num_classes, drop_prob):
        super(BiGRU, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.bigru =nn.GRU(input_size, hidden_size, num_layers, bidirectional=True, 
                            batch_first=True, bias=True)
        
        self.segment_hidden_size = segment_hidden_size
        self.segment_bigru =nn.GRU(hidden_size * 2 * 2, segment_hidden_size, num_layers, bidirectional=True, 
                            batch_first=True, bias=True)
        
        ## DNN for class prediction
        self.fc = ClassPredictor(self.segment_hidden_size * 2, num_classes, drop_prob)
 

    def init_hidden_state(self, batch_size):
        h0 = torch.empty(self.num_layers * 2, batch_size, self.hidden_size).double()
        h0 = nn.init.orthogonal_(h0)
        h0 = h0.requires_grad_().cuda()
        
        return h0
 

    def init_segment_hidden_state(self, batch_size):
        sh0 = torch.empty(self.num_layers * 2, batch_size, self.segment_hidden_size).double()
        sh0 = nn.init.orthogonal_(sh0)
        sh0 = sh0.requires_grad_().cuda()
        
        return sh0
    
    
    def forward(self, x, segment_indices):
        ## x: (batch_size, seq_len, feature_len)
        ## segment_indices: (num_segments, 2)
        
        batch_size = x.size(0)

        ## Set initial states
        ## h0: (num_layers * num_directions, batch_size, hidden_size)
        h0 = self.init_hidden_state(batch_size)
        
        ## Forward propagate
        ## out: tensor of shape (batch_size, seq_length, hidden_size * 2)
        out, _ = self.bigru(x, h0)  
        
        ## Use hidden states of each segment to predict their labels
        concat_input = []
        for (start, end) in segment_indices:
            hidden_states = out[:, start:end, :]
            
            ## Compute the hidden state by doing temporal pooling over all time steps
            ## pool_out: (hidden_size * 2)
            max_pool_out = F.adaptive_max_pool1d(hidden_states.permute(0,2,1), 1).squeeze()
            avg_pool_out = torch.mean(hidden_states, dim=1).squeeze()

            ## concat_pool_out: (hidden_size * 2 * 2)
            concat_pool_out = torch.cat([max_pool_out, avg_pool_out])

            concat_input.append(concat_pool_out)

        ## concat_input: (1, num_segments, hidden_size * 2 * 2)
        concat_input = torch.stack(concat_input).unsqueeze(dim=0)
        
        
        
        
        ## sh0: (num_layers * num_directions, batch_size, segment_hidden_size)
        sh0 = self.init_segment_hidden_state(batch_size)
        
        ## s_out: tensor of shape (batch_size, num_segments, segment_hidden_size * 2)
        s_out, _ = self.segment_bigru(concat_input, sh0) 
    
    
    

        num_segments = s_out.shape[1]
       
        segment_outputs = []
        for i in range(num_segments):
            ## inp: (segment_hidden_size * 2)
            inp = s_out[:,i,:].squeeze()

            ## output: (num_classes)
            output = self.fc(inp)

            segment_outputs.append(output)
        
        ## segment_outputs: (num_segments, num_classes)
        segment_outputs = torch.stack(segment_outputs)
        
        return segment_outputs





####################### Hyperparameters used #######################

## Model Architecture
input_dim = 400  # dimension of an i3D video frame
hidden_dim = 160 # dimension of hidden state
segment_hidden_dim = 100 # dimension of the hidden state of the segment RNN
layer_dim = 1    # number of stacked layers
output_dim = 48  # number of sub-action labels

drop_prob = 0.2 # dropout prob of final RNN output

model = StackedBiGRU(input_dim, hidden_dim, segment_hidden_dim, layer_dim, output_dim, drop_prob)

model = model.double() # transform the model parameters to double precision


## Loss function
loss_criterion = nn.CrossEntropyLoss()


## Optimizer
learning_rate = 0.01
weight_decay = 0.00005
momentum = 0.9

optimizer = optim.Adagrad(model.parameters(), lr=learning_rate, weight_decay=weight_decay) 


## Learning Rate Scheduler
patience = 2
decrease_factor = 0.7
min_learning_rate = 0.00005
scheduler = ReduceLROnPlateau(optimizer, mode='min', 
                              patience=patience, min_lr=min_learning_rate, factor=decrease_factor,
                              verbose=True)


num_epochs = 15
should_shuffle = True # whether training data is shuffled

## Train for 15 epochs
# training_losses, validation_losses, training_accuracies, validation_accuracies = train_model(model, num_epochs, should_shuffle)



## Train for 5 more epochs with reduced learning rate (total 20 epochs)
new_learning_rate = 0.007
for param_group in optimizer.param_groups:
    param_group['lr'] = new_learning_rate

num_epochs = 5

# t_losses, v_losses, t_accuracies, v_accuracies = train_model(model, num_epochs, should_shuffle)



## Train for 5 more epochs with reduced learning rate (total 20 epochs)
new_learning_rate = 0.0009
for param_group in optimizer.param_groups:
    param_group['lr'] = new_learning_rate


num_epochs = 5

# t_losses, v_losses, t_accuracies, v_accuracies = train_model(model, num_epochs, should_shuffle)
