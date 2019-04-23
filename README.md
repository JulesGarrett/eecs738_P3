# EECS 738 Project 3: Says One Neuron to Another
Python Implementation of Neural Network from Scratch using numpy

## Datasets

[Mushroom Dataset](https://archive.ics.uci.edu/ml/datasets/mushroom) : From Audobon Society Field Guide; mushrooms described in terms of physical characteristics; classification: poisonous or edible

[German Credit Data](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)) : This dataset classifies people described by a set of attributes as good or bad credit risks.

When importing both of these dataset I normalized the values in each column, excluding the classification column. Also, in this process if the data had letters instead of numbers I converted it to numbers using the same normalization process. 

When splitting the data into train, validate, and testing data set I tried to follow a 75% for training 12.5% for validation and 12.5% for testing. Of course at times these values were rounded up or down depending on what made the most sense. I chose this split because I felt like it allowed for most of the data to go to training while still leaving enough for validation and testing. 

## Design 
For this project I implemented a neural network class that featured forward and backward functions. Forward implementing forward propagation and backwards implementing backward propagation. For the neural network I chose the sigmoid function as my activation function. 

## Training and Output
When training and testing the data I allowed for 5000 interations through the neural network and a learning rate of 0.0005 or 0.005 (lesser for the second and smaller dataset). I also allowed for a certain amount of error while still considering the classification a success. I did this because the neural network will not output exactly 1 or 0 but values close to that. That being said, for the larger dataset I allowed 0.1 interval of leaniancy and for the smaller dataset I allowed 0.3 interval of leaniancy.

#### Output
|Dataset 1: Mushrooms|                    |
|-------------------|:------------------:|
| Training Accuracy | 0.9691935483870968 |
| Validation Accuracy | 0.9811290322580645|
| Testing Accuracy  | 0.9811290322580645 |

Overall these results look really good. 

|Dataset 2: Germain Credit|           |
|-------------------|:------------------:|
| Training Accuracy | 0.6973333333333334 |
| Validation Accuracy | 0.6|
| Testing Accuracy  | 0.688              |

These results are a little lower than I would like, but that is likely due to the small size of that dataset.



## References
[How to Build you Own Neural Network from Scratch in Python](https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6)


** *I uploaded just .py files in addition to my jupyter notebooks. If running the .py files you will need pandas and numpy and to run type python Project3_NeuralNetwork-Dataset1 or Project3_NeuralNetwork-Dataset2*
