import numpy as np
import pandas as pd
from losses import cross_entropy
import pickle as pkl

class shallow_neural_net(object):

    def __init__(self, 
                 layer_1_size, 
                 layer_2_size, 
                 x, 
                 y, 
                 w1=None, 
                 w2=None, 
                 b2=None, 
                 b1=None, 
                 weights=False):
        self.layer_0_size = x.shape[0]
        self.layer_1_size = layer_1_size
        self.layer_2_size = layer_2_size
        self.x = x
        self.y = y
        self.w1 = w1
        self.b1 = b1
        self.w2 = w2
        self.b2 = b2
        self.no_of_samples = x.shape[1]
        self.weights = weights

    def __call__(self, epoch):
        """
        initializes weights if the weight are not 
        provided and calls forward

        Parameters:
        epoch = number of iterations for training
        """

        if not self.weights:
            self.initialize_weights()
        
        self.train(epoch)
    
    def train(self, epoch):

        for i in range(epoch):
            print(f'{i}th iteration')
            activations = self.forward()
            print(cross_entropy(self.y, activations['a2']))
            gradients = self.backpropagate(**activations)
            self.optimize(**gradients)
        
        final_weights = {'w1': self.w1, 'w2': self.w2, 'b2': self.b2, 'b1': self.b1} 
        self.save_weights(final_weights)
    
    def initialize_weights(self):
        """
        initializes weights based on the number of neurons in 
        each layer
        """

        self.w1 = np.random.randn(self.layer_1_size, self.layer_0_size) * 0.01
        self.b1 = np.zeros((self.layer_1_size, 1))
        self.w2 = np.random.randn(self.layer_2_size, self.layer_1_size) * 0.01
        self.b2 = np.zeros((self.layer_2_size, 1))
    
    def forward(self):
        """
        performs forward propogation

        Parameters:
        epoch = number of iterations to be performed for training

        Returns:
        dict of activation value of each layer 
        """

        z1 = np.dot(self.w1, self.x) + self.b1 
        a1 = np.tanh(z1)
        z2 = np.dot(self.w2, a1) + self.b2
        a2 = 1 / (1 + np.exp(-z2))
        assert(a2.shape == (1, self.x.shape[1]))
        
        return {'a1': a1, 'a2': a2}

    def backpropagate(self,a1, a2):
        """
        calculates the gradient of the existing weights and bias

        Parameters:
        a1 = activations value in layer 1
        a2 = activations value in layer 2

        Returns:
        dict of Gradient value of weights and bias in all layer
        """

        dz2 = a2 - self.y
        dw2 = np.dot(dz2, a1.T) / self.no_of_samples
        db2 = np.sum(dz2, axis=1, keepdims=True) / self.no_of_samples
        dz1 = np.dot(self.w2.T, dz2) * (1 - np.square(a1))
        dw1 = np.dot(dz1, self.x.T) / self.no_of_samples
        db1 = np.sum(dz1, axis=1, keepdims=True) / self.no_of_samples

        gradients = {
            'dw2': dw2,
            'dw1': dw1,
            'db2': db2,
            'db1': db1
        }
        
        return gradients

    def optimize(self, dw2, db2, dw1, db1, learning_rate=0.5):
        """
        updates the weights based on the derivatives

        Parameters:
        dw2 = derivative of w2 (weights used to compute a2(layer_2))
        db2 = derivative of b2 (bias used to compute a2(layer_2))
        dw1 = derivative of w1 (weights used to compute a1(layer_1))
        db1 = derivative of b1 (bias used to compute a1(layer_1)) 
        """

        self.w2 = self.w2 - (learning_rate * dw2)
        self.b2 = self.b2 - (learning_rate * db2)
        self.w1 = self.w1 - (learning_rate * dw1)
        self.b1 = self.b1 - (learning_rate * db1)
    
    def save_weights(self, weights):
        """
        saves(serialize) the final weights of the model, so that 
        we can reuse(deserialize) it for later prediciton 

        Parameters:
        weights: final weights after the n epoc
        """

        pickle_obj = pkl.dumps(weights)

        with open('model', 'wb') as file:
            pkl.dump(pickle_obj, file)
