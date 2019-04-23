#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[25]:


##data preprocessing
df=pd.read_csv('mushrooms.csv', sep=',',header=None)

#need to change char indicators into number indicators
unique = dict()
for column in df:
    unique[column] = list(set(df[column]))

temp = dict()
  
#replace letters with a normalized number for the column
for column in df:
    c_list = []
    for val in df[column].values:
        if column == 0:
            c_list.append(unique[column].index(val))
        else:
            c_list.append(unique[column].index(val)/len(unique[column]))
    temp[column] = c_list
n_df = pd.DataFrame(temp)

##split data into train, validate, test
## ~75% training, ~12.5% validate, ~12.5% test
train_x = (n_df.iloc[:6200, 1:]).values
train_y = (n_df.iloc[:6200, :1]).values
validate_x = (n_df.iloc[6200:7162,1: ]).values
validate_y = (n_df.iloc[6200:7162, :1]).values
test_x = (n_df.iloc[ 7162:,1:]).values
test_y = (n_df.iloc[7162:, :1]).values


# In[26]:


#sigmoid functions for activation function
def sigmoid(x):
    return 1/(1+np.exp(-x));

def sigmoid_back(x):
    sig = sigmoid(x);
    return sig * (1.0 - sig);


# In[27]:


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





# In[34]:


##TRAINING
#create neural network for training data
nn = NN(train_x, train_y, 0.0005)
for i in range(5000):
    nn.forward()
    nn.back()
#since output won't be exact allow >.9 to be 1
# and allow <.1 to be 0
factor_hi = .9
factor_lo = .1
count = 0
for i in range(len(nn.output)):
    if train_y[i][0] == 1 and nn.output[i][0] >= factor_hi:
        count += 1
    if train_y[i][0] == 0 and nn.output[i][0] < factor_lo:
        count += 1
print(count/len(nn.output))


# In[35]:


##VALIDATE
nn_v = NN(validate_x, validate_y, .0005)
for i in range(5000):
    nn_v.forward()
    nn_v.back()
#since output won't be exact allow >.9 to be 1
# and allow <.1 to be 0
factor_hi = .9
factor_lo = .1
count = 0
for i in range(len(nn.output)):
    if train_y[i][0] == 1 and nn.output[i][0] >= factor:
        count += 1
    if train_y[i][0] == 0 and nn.output[i][0] < factor:
        count += 1
print(count/len(nn.output))


# In[37]:


##TEST
nn_t = NN(test_x, test_y, .0005)
for i in range(5000):
    nn_t.forward()
    nn_t.back()
#since output won't be exact allow >.9 to be 1
# and allow <.1 to be 0
factor_hi = .9
factor_lo = .1
count = 0
for i in range(len(nn.output)):
    if train_y[i][0] == 1 and nn.output[i][0] >= factor:
        count += 1
    if train_y[i][0] == 0 and nn.output[i][0] < factor:
        count += 1
print(count/len(nn.output))


# In[ ]:




