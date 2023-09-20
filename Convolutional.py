from Layer import Layer
import numpy as np 
from scipy.signal import correlate2d, convolve2d

"""
This file contains the convolutional layer for the CNN.
"""

#the Convolutional layer class, which is the core of the CNN
class Convolutional(Layer):
    
    def __init__(self, inp_shape: tuple, num_kernels: int, kernel_size: int): #kernel = filter
        inp_h, inp_w = inp_shape[0], inp_shape[1] #the dimensions of the input formulated as its height and width 
        self.kernel_size = kernel_size
        self.num_kernels = num_kernels
        self.inp_shape = inp_shape
        self.inp_h, self.inp_w = inp_h, inp_w
        
        self.kernel_shape = (num_kernels, kernel_size, kernel_size) # (3,3)
        self.out_shape = (num_kernels, inp_h - kernel_size + 1, inp_w - kernel_size + 1)
        
        #initializing random weights (kernel entries) and biases which are to be optimized by gradient descent 
        self.kernels = np.random.randn(*self.kernel_shape)
        self.biases = np.random.randn(*self.out_shape)
        
    def forward(self, inp):
        self.inp = inp
        self.output = np.zeros(self.out_shape)
    
        #correlating the input with the kernels 
        for i in range(self.num_kernels):
            self.output[i] = correlate2d(self.inp, self.kernels[i], mode='valid')
                
        self.out = np.maximum(self.output, 0)
            
        return self.out
    
    def backward(self, dL_dout, learning_rate): #dL/dout -- the derivative of the loss function with respect to the output
        dL_dinp = np.zeros_like(self.inp)
        dL_dkernels = np.zeros_like(self.kernels)
        
        for i in range(self.num_kernels):
            dL_dkernels[i] = correlate2d(self.inp, dL_dout[i], mode='valid') 
            dL_dinp += correlate2d(dL_dout[i], self.kernels[i], mode='full')
           
        self.kernels -= dL_dkernels * learning_rate
        self.biases -= dL_dinp * learning_rate
        
        return dL_dinp