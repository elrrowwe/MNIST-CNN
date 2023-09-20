import numpy as np 

"""
This file contains the MaxPool layer.
"""

#the MaxPool class
class MaxPool:
    def __init__(self, pool_size, stride=2):
        self.pool_size = pool_size
        self.stride = stride 
        
        
    def forward(self, inp):
        self.inp = inp 
                
        self.inp_d, self.inp_h, self.inp_w = inp.shape
        
        #size of the output -- sqrt of the number of pools
        self.out_shape = (self.inp_d, int(self.inp_h // self.pool_size), int(self.inp_w // self.pool_size))
        
        self.out_h, self.out_w = self.out_shape[1], self.out_shape[2]
        
        self.out = np.zeros(self.out_shape)
        for k in range(self.inp_d):
            for i in np.arange(self.out_h):
                for j in np.arange(self.out_w):
                    
                    #calculating the indices for the pool(mask)
                    start_i = i * self.pool_size
                    start_j = j * self.pool_size

                    end_i = start_i + self.pool_size
                    end_j = start_j + self.pool_size

                    mat = self.inp[k,start_i:end_i, start_j:end_j]
                    #the max pooling operation itself 
                    self.out[k,i,j] = np.max(mat)
                    
        return self.out
    
    def backward(self, dL_dout, learning_rate):
        #the backward pass omits the learning_rate argument, as there are no gradient computations involved
        dL_dinp = np.zeros_like(self.inp)
        for k in range(self.inp_d):
            for i in np.arange(self.out_h, step=self.stride):
                for j in np.arange(self.out_w, step=self.stride):

                    start_i = i * self.pool_size
                    start_j = j * self.pool_size

                    end_i = start_i + self.pool_size
                    end_j = start_j + self.pool_size

                    patch = self.inp[k,start_i:end_i, start_j:end_j]

                    if not patch.size:
                        mask = np.zeros_like(patch)
                    else:
                        mask = patch == np.max(patch)

                    dL_dinp[k,start_i:end_i, start_j:end_j] = dL_dout[k, i, j] * mask

        return dL_dinp