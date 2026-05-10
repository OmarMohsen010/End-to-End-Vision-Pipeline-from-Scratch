"""
08_cnn_layers.py
================
Phase 5.3: CNN Building Blocks (Pure NumPy)

Contains the object-oriented layer definitions for a Convolutional Neural Network.
Every layer adheres to the forward/backward contract.
Expects image data in the shape: (Batch, Height, Width, Channels)
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# ══════════════════════════════════════════════════════════════════════════
#  Base Layer Interface
# ══════════════════════════════════════════════════════════════════════════

class Layer:
    def __init__(self):
        self.params = {}      # Holds weights and biases (W, b)
        self.gradients = {}   # Holds gradients (dW, db)
        self.cache = None     # Holds inputs needed for the backward pass

    def forward(self, X):
        raise NotImplementedError

    def backward(self, d_out):
        raise NotImplementedError

# ══════════════════════════════════════════════════════════════════════════
#  1. Fully Connected (Dense) Layer
# ══════════════════════════════════════════════════════════════════════════

class Dense(Layer):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # He Initialization (standard for ReLU networks)
        self.params['W'] = np.random.randn(input_dim, output_dim) * np.sqrt(2. / input_dim)
        self.params['b'] = np.zeros(output_dim)

    def forward(self, X):
        self.cache = X
        return np.dot(X, self.params['W']) + self.params['b']

    def backward(self, d_out):
        X = self.cache
        N = X.shape[0]
        
        # Calculate gradients for this layer's weights
        self.gradients['W'] = np.dot(X.T, d_out) / N
        self.gradients['b'] = np.sum(d_out, axis=0) / N
        
        # Calculate gradient to pass back to the previous layer
        dX = np.dot(d_out, self.params['W'].T)
        return dX

# ══════════════════════════════════════════════════════════════════════════
#  2. Activation & Utility Layers
# ══════════════════════════════════════════════════════════════════════════

class ReLU(Layer):
    def forward(self, X):
        self.cache = X
        return np.maximum(0, X)

    def backward(self, d_out):
        X = self.cache
        # Gradient is 1 where X > 0, and 0 otherwise.
        dX = d_out * (X > 0)
        return dX

class Flatten(Layer):
    def forward(self, X):
        self.cache = X.shape
        N = X.shape[0]
        # Flattens (N, H, W, C) into (N, H * W * C)
        return X.reshape(N, -1)

    def backward(self, d_out):
        # Reshapes the 1D gradient back into the 3D spatial tensor
        return d_out.reshape(self.cache)

# ══════════════════════════════════════════════════════════════════════════
#  3. Convolutional Layer (Conv2D)
# ══════════════════════════════════════════════════════════════════════════

from numpy.lib.stride_tricks import sliding_window_view

class Conv2D(Layer):
    def __init__(self, filter_size, in_channels, out_channels, stride=1, padding=0):
        super().__init__()
        self.F = filter_size
        self.S = stride
        self.P = padding
        
        # Weights shape: (Filter_H, Filter_W, In_Channels, Out_Channels)
        self.params['W'] = np.random.randn(self.F, self.F, in_channels, out_channels) * np.sqrt(2. / (self.F * self.F * in_channels))
        self.params['b'] = np.zeros(out_channels)

    def forward(self, X):
        # 1. Pad the input
        X_padded = np.pad(X, ((0,0), (self.P, self.P), (self.P, self.P), (0,0)), mode='constant')
        self.cache = X_padded
        
        # 2. Generate a vectorized view of all sliding windows instantly
        # Shape becomes: (Batch, H_out, W_out, Channels, Filter_H, Filter_W)
        windows = sliding_window_view(X_padded, window_shape=(self.F, self.F), axis=(1, 2))
        
        # 3. Apply the Stride by slicing the matrix
        windows = windows[:, ::self.S, ::self.S, :, :, :]
        
        # 4. The Matrix Multiplication
        # We multiply and sum across the Filter_H (axis 4), Filter_W (axis 5), and Channels (axis 3)
        # against the weights' Filter_H (axis 0), Filter_W (axis 1), and In_Channels (axis 2)
        Out = np.tensordot(windows, self.params['W'], axes=([4, 5, 3], [0, 1, 2])) + self.params['b']
        
        return Out

    def backward(self, d_out):
        """
        Backward pass is kept partially looped to prevent massive RAM spikes, 
        but utilizes tensordot for the heavy lifting.
        """
        X_padded = self.cache
        N, H_pad, W_pad, C_in = X_padded.shape
        _, H_out, W_out, C_out = d_out.shape
        
        dX_padded = np.zeros_like(X_padded)
        dW = np.zeros_like(self.params['W'])
        db = np.sum(d_out, axis=(0, 1, 2)) / N
        
        for h in range(H_out):
            for w in range(W_out):
                h_start, w_start = h * self.S, w * self.S
                h_end, w_end = h_start + self.F, w_start + self.F
                
                X_slice = X_padded[:, h_start:h_end, w_start:w_end, :]
                d_out_slice = d_out[:, h:h+1, w:w+1, :]
                
                dW += np.tensordot(X_slice, d_out_slice, axes=([0], [0])).reshape(dW.shape)
                dX_padded[:, h_start:h_end, w_start:w_end, :] += np.tensordot(d_out_slice, self.params['W'], axes=([3], [3])).reshape(N, self.F, self.F, C_in)
        
        self.gradients['W'] = dW / N
        self.gradients['b'] = db
        
        if self.P != 0:
            return dX_padded[:, self.P:-self.P, self.P:-self.P, :]
        return dX_padded

# ══════════════════════════════════════════════════════════════════════════
#  4. Pooling Layer (MaxPool2D)
# ══════════════════════════════════════════════════════════════════════════

class MaxPool2D(Layer):
    def __init__(self, pool_size=2, stride=2):
        super().__init__()
        self.F = pool_size
        self.S = stride

    def forward(self, X):
        self.cache = X
        
        # 1. Create the sliding window view
        windows = sliding_window_view(X, window_shape=(self.F, self.F), axis=(1, 2))
        
        # 2. Apply stride
        windows = windows[:, ::self.S, ::self.S, :, :, :]
        
        # 3. Vectorized Max operation across the filter dimensions (axes 4 and 5)
        Out = np.max(windows, axis=(4, 5))
        
        return Out

    def backward(self, d_out):
        X = self.cache
        N, H, W, C = X.shape
        _, H_out, W_out, _ = d_out.shape
        
        dX = np.zeros_like(X)
        
        # Backward pooling is tricky to vectorize without exploding RAM, 
        # so we keep it looped, but it executes much faster than convolution.
        for h in range(H_out):
            for w in range(W_out):
                h_start, w_start = h * self.S, w * self.S
                h_end, w_end = h_start + self.F, w_start + self.F
                
                X_slice = X[:, h_start:h_end, w_start:w_end, :]
                
                # Mask of where the max values are
                max_vals = np.max(X_slice, axis=(1, 2), keepdims=True)
                mask = (X_slice == max_vals)
                
                # Distribute the gradients ONLY to the max pixels
                dX[:, h_start:h_end, w_start:w_end, :] += mask * d_out[:, h:h+1, w:w+1, :]
                        
        return dX