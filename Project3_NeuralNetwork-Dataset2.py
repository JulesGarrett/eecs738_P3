#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[60]:


##data preprocessing
df=pd.read_csv('german.data', sep=' ',header=None)
#need to change char indicators into number indicators
unique = dict()
for column in df:
    unique[column] = list(set(df[column]))
temp = dict()
  
#replace letters with a normalized number for the column
for column in df:
    c_list = []
    for val in df[column].values:
        if column == 21:
            c_list.append(unique[column].index(val))
        else:
            c_list.append(unique[column].index(val)/len(unique[column]))
    temp[column] = c_list
n_df = pd.DataFrame(temp)
##split data into train, validate, test
# ~75% training, ~12.5% validate, ~12.5% test
train_x = (n_df.iloc[:750, :-1]).values
train_y = (n_df.iloc[:750, -1:]).values
validate_x = (n_df.iloc[750:875,:-1 ]).values
validate_y = (n_df.iloc[750:875, -1:]).values
test_x = (n_df.iloc[ 875:,:-1]).values
test_y = (n_df.iloc[875:, -1:]).values


# In[61]:


#sigmoid functions for activation function
def sigmoid(x):
    return 1/(1+np.exp(-x));

def sigmoid_back(x):
    sig = sigmoid(x);
    return sig * (1.0 - sig);


# In[62]:


#create a nerual network class that has forward and backward functions
class NN:
    def __init__(self, x, y, rate):
        self.input = x
        self.weights1 = np.random.rand(self.input.shape[1], 4)
        self.weights2 = np.random.rand(4,1)
        self.y = y
        self.output = np.zeros(self.y.shape)
        self.LR = rate
    def forward(self):
        self.layer = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer, self.weights2))
        
    def back(self):
        d_w2 = np.dot(self.layer.T, (2*(self.y - self.output)* (self.LR*sigmoid_back(self.output))))
        d_w1 = np.dot(self.input.T, (np.dot(2*(self.y -self.output)* (self.LR*sigmoid_back(self.output)), self.weights2.T)*sigmoid_back(self.layer)))
        
        self.weights1 += d_w1
        self.weights2 += d_w2


# In[ ]:





# In[76]:


##TRAINING
#create neural network for training data
nn = NN(train_x, train_y, 0.005)
for i in range(5000):
    nn.forward()
    nn.back()
#since output won't be exact allow >.7 to be 1
# and allow <.3 to be 0
factor_hi = .7
factor_lo = .3
count = 0
for i in range(len(nn.output)):
    if train_y[i][0] == 1 and nn.output[i][0] >= factor_hi:
        count += 1
    if train_y[i][0] == 0 and nn.output[i][0] < factor_lo:
        count += 1
print(count/len(nn.output))


# In[85]:


##VALIDATE
nn_v = NN(validate_x, validate_y, .005)
for i in range(5000):
    nn_v.forward()
    nn_v.back()
#since output won't be exact allow >.7 to be 1
# and allow <.3 to be 0
factor_hi = .7
factor_lo = .3
count = 0
for i in range(len(nn_v.output)):
    if train_y[i][0] == 1 and nn_v.output[i][0] >= factor_hi:
        count += 1
    if train_y[i][0] == 0 and nn_v.output[i][0] < factor_lo:
        count += 1
print(count/len(nn_v.output))


# In[86]:


##TEST
nn_t = NN(test_x, test_y, .005)
for i in range(5000):
    nn_t.forward()
    nn_t.back()
#since output won't be exact allow >.7 to be 1
# and allow <.3 to be 0
factor_hi = .7
factor_lo = .3
count = 0
for i in range(len(nn_t.output)):
    if train_y[i][0] == 1 and nn_t.output[i][0] >= factor_hi:
        count += 1
    if train_y[i][0] == 0 and nn_t.output[i][0] < factor_lo:
        count += 1
print(count/len(nn_t.output))


# In[ ]:




