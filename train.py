import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

# Function gen_minibatch gets a minibatch of data.

def train_data(mini_batch, targets, word_attn_model, sent_attn_model, word_optimizer, sent_optimizer, criterion):
    state_word = word_attn_model.init_hidden().cuda()
    state_sent = sent_attn_model.init_hidden().cuda()
#     print state_sent.size()
    max_sents, batch_size, max_tokens = mini_batch.size()
    word_optimizer.zero_grad()
    sent_optimizer.zero_grad()
    s = None
    for i in xrange(max_sents):
        _s = word_attn_model(mini_batch[i,:,:].transpose(0,1), state_word)
        if(s is None):
            s = _s
        else:
            s = torch.cat((s,_s),0)
#     print s.size()
            
    y_pred = sent_attn_model(s, state_sent)
    loss = criterion(y_pred.cuda(), targets) 
    loss.backward()
    
    word_optimizer.step()
    sent_optimizer.step()
    
    return loss.data[0]


learning_rate = 1e-3
word_optmizer = torch.optim.Adam(word_attn.parameters(), lr=learning_rate)
sent_optimizer = torch.optim.SGD(sent_attn.parameters(), lr=learning_rate)
criterion = nn.NLLLoss()
word_attn.cuda()
sent_attn.cuda()

loss_full = []
loss_smooth = []
counter = 0
for i in xrange(50000):
    tokens, labels = gen_minibatch(train, 64)
    loss = train_data(tokens.cuda(), labels.cuda(), word_attn, sent_attn, word_optmizer, sent_optimizer, criterion)
    loss_full.append(loss)
    loss_smooth.append(loss)
    counter += 1
    if counter == 30:
        print 'Loss after %d minibatches is %f' % (i, np.mean(loss_smooth))
        loss_smooth = []
        counter = 0