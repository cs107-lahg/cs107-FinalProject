#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 19:06:57 2021

@author: lingfeng
"""

import numpy as np

class Variable:
    
    #initialize a "Variabe" object
    def __init__(self, value, derivative_seed=1):
        self.val = value
        self.der = derivative_seed
    
    #method to get value
    def get_value(self):
        return self.val
    
    #method to get derivative
    def get_derivative(self):
        return self.der
    
    #sin methods
    def sin(self):
        value = np.sin(self.val)
        derivative = np.cos(self.val) * self.der
        return Variable(value, derivative)
        
    #cos method
    def cos(self):
        value = np.cos(self.val)
        derivative = - np.sin(self.val) * self.der
        return Variable(value,derivative)
        
    #tan method
    def tan(self):
        value = np.tan(self.val)
        derivative = (1 / np.cos(self.val)) ** 2 * self.der
        return Variable(value, derivative)
    
    #arcsin method
    def arcsin(self):
        if abs(self.val) >= 1:
            raise ValueError(f"arcsin doesn't exist at {self.val}")
        value = np.arcsin(self.val)
        derivative = 1 / np.sqrt(1 - self.val ** 2) * self.der
        return Variable(value, derivative)
    
    #arccos method
    def arccos(self):
        if abs(self.val) >= 1:
            raise ValueError(f"arccos doesn't exist at {self.val}")
        value = np.arccos(self.var)
        derivative = - 1 / np.sqrt(1 - self.val ** 2) * self.der
        return Variable(value, derivative)
        
    #arctan method
    def arctan(self):
        value = np.arctan(self.val)
        derivative = 1 / (1 + self.val ** 2) * self.der
        return Variable(value, derivative)
    
    #exponential
    def exp(self):
        value = np.exp(self.val)
        derivative = np.exp(self.val)
        return Variable(value, derivative)
    
    
    
def make_variables(var_list, der_list):
    if len(var_list) != len(der_list):
        raise ValueError("The value list and derivative list should be of the same length")
    
    variables = []
    for val, der in zip(var_list, der_list):
        variables.append(Variable(val, der))
        
    return variables

def make_variable(var,der):
    return Variable(var, der)

#demos
if __name__ == '__main__':
    x = make_variables([1,2], [0,1])
    print(x[1].get_value())
    print(x[0].get_derivative())
    y = make_variable(5, 1)
    print(y.get_value())

'''
usage:
    sin(x) --> x.sin()
    cos(x) --> x.cos()
    tan(x) --> x.tan(x) 
    exp(x) --> x.exp()
    arcsin(x) --> x.arcsin()
    arccos(x) --> x.arccos()
    arctan(x) --> x.arctan()
'''