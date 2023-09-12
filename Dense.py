from Layer import Layer
import numpy as np 


"""
This file contains the Dense layer, along with the softmax acivation function and the cross entropy loss function..
"""

#the softmax activation function and its derivative
def softmax(z):
    shifted_z = z - np.max(z)
    exp_values = np.exp(shifted_z)

    sum_exp_values = np.sum(exp_values, axis=0)
    log_sum_exp = np.log(sum_exp_values)

    probabilities = exp_values / sum_exp_values

    return probabilities

def softmax_prime(s):
    return np.diagflat(s) - np.dot(s, s.T)


class Dense(Layer):
    def __init__(self, inp_size, out_size):
        self.weights = np.random.randn(out_size, inp_size)
        self.biases = np.random.randn(out_size, 1)
            
    def forward(self, inp): #computes the classic output-input equation Y = Wx + B
        self.inp = inp
        
        flat_inp = self.inp.flatten().reshape(1, -1)
        
        self.out = softmax(np.dot(self.weights, flat_inp.T) + self.biases)
        
        return self.out
    
    def backward(self, dL_dout, learning_rate):
        dL_dy = np.dot(softmax_prime(self.out), dL_dout)
        dL_dw = np.dot(dL_dy, self.inp.flatten().reshape(1, -1))

        dL_db = dL_dy

        dL_dinput = np.dot(self.weights.T, dL_dy)
        dL_dinput = dL_dinput.reshape(self.inp.shape)

        self.weights -= learning_rate * dL_dw
        self.biases -= learning_rate * dL_db

        return dL_dinput
    

#cross entropy 
def cross_entropy(predictions, targets):
    num_samples = 10

    epsilon = 1e-7
    predictions = np.clip(predictions, epsilon, 1 - epsilon)

    loss = -np.sum(targets * np.log(predictions)) / num_samples

    return loss

def cross_entropy_gradient(actual_labels, predicted_probs):
    num_samples = actual_labels.shape[0]
    
    gradient = -actual_labels / (predicted_probs + 1e-7) / num_samples

    return gradient