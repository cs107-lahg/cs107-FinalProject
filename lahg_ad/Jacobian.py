#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from fd import Variable
import numpy as np

def Jacobian_der(func_list):
    """
    Function for computing the Jacobian Matrix for vector function

    INPUTS
    ------
    func_list : a list of function expression
        
    RETURNS
    -------
    numpy array
        a 2-D numpy array representing the Jacobian Matrix
        
    RAISES
    ------
    Exception
        if the list contains non-Variable object
        if input functions have different dimensions
        
    EXAMPLES
    --------
    >>> x = Variable(4, np.array([1, 0]))
    >>> y = Variable(3, np.array([0, 1]))
    >>> f = [x+y, x**3, x*y]
    >>> print(Jacobian_der(f))
    [[ 1  1]
     [48  0]
     [ 3  4]]
    
    >>> x = Variable(4, np.array([1, 0]))
    >>> f = [x+2, x**2]
    >>> print(Jacobian_der(f))
    [[1 0]
     [8 0]]
    
   """
    der = []
    expect_shape = len(func_list[0].der)
    for func in func_list:
        if not isinstance(func, Variable):
            raise Exception("The input must be a list of Variable objects")
        if len(func.der) != expect_shape:
            raise Exception("The input functions have different dimensions!")
        der.append(func.der)
    return np.array(der)


def Jacobian_val(func_list):
    """
    Function for computing the value of vector function

    INPUTS
    ------
    func_list : a list of function expression

    RAISES
    ------
    Exception
        if the list contains non-Variable object

    RETURENS
    -------
    TYPE
         a 1-D numpy array representing the value of the vector function
         
    EXAMPLES
    --------
    >>> x = Variable(4, np.array([1, 0]))
    >>> y = Variable(3, np.array([0, 1]))
    >>> f = [x+y, x**3, x*y]
    >>> print(Jacobian_val(f))
    [ 7 64 12]
    
    >>> x = Variable(4, np.array([1, 0]))
    >>> f = [x+2, x**2]
    >>> print(Jacobian_val(f))
    [ 6 16]
    
    """
    val = []
    for func in func_list:
        if not isinstance(func, Variable):
            raise Exception("The input must be a list of Variable objects")
        val.append(func.val)
    return np.array(val)



if __name__ == "__main__":
    import doctest
    doctest.testmod()


