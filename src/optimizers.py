"""
06_optimizers.py
================
Phase 6: Custom Optimizers

Contains optimizer classes for Deep Learning models. 
Each optimizer maintains its own state (for Momentum/Adam) and 
updates parameters when given a parameter name, its current values, and its gradient.
"""

import numpy as np

class Optimizer:
    """Base class for all optimizers."""
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, param_name, param, grad):
        raise NotImplementedError("Subclasses must implement the update method.")

# ──────────────────────────────────────────────────────────────────────────

class SGD(Optimizer):
    """
    Standard Stochastic Gradient Descent.
    Math: w = w - lr * gradient
    """
    def __init__(self, lr=0.01):
        super().__init__(lr)

    def update(self, param_name, param, grad):
        return param - (self.lr * grad)

# ──────────────────────────────────────────────────────────────────────────

class Adam(Optimizer):
    """
    Adaptive Moment Estimation (Adam).
    Maintains running averages of gradients (m) and squared gradients (v).
    """
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        # Memory dictionaries to store state for each specific parameter (e.g., 'W1', 'b1')
        self.m = {} 
        self.v = {}
        self.t = {} # Time step per parameter

    def update(self, param_name, param, grad):
        # 1. Initialize state for this parameter if it's the first time seeing it
        if param_name not in self.m:
            self.m[param_name] = np.zeros_like(param)
            self.v[param_name] = np.zeros_like(param)
            self.t[param_name] = 0
            
        self.t[param_name] += 1
        t = self.t[param_name]
        
        # 2. Update biased first moment estimate (mean)
        self.m[param_name] = self.beta1 * self.m[param_name] + (1 - self.beta1) * grad
        
        # 3. Update biased second raw moment estimate (variance)
        self.v[param_name] = self.beta2 * self.v[param_name] + (1 - self.beta2) * (grad ** 2)
        
        # 4. Compute bias-corrected estimates (to prevent zero-bias early in training)
        m_hat = self.m[param_name] / (1 - self.beta1 ** t)
        v_hat = self.v[param_name] / (1 - self.beta2 ** t)
        
        # 5. Apply the final Adam weight update
        param_updated = param - (self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon))
        
        return param_updated