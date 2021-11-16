#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file contains the Variable class for performing forward mode automatic differentiation.
It also contains function for making scalar/vector inputs.
"""

import numpy as np

class Variable:
    """
    This is the Variable class with basic methods and operation overloading. 
    It is used to perform forward mode automatic differentiation for scalar/vector input of scalar function.

    EXAMPLES
    ============

    """
   
    
    def __init__(self, value, derivative_seed=1):
        """
        Variable class constructor
        
        INPUTS
        ------
        value : int or float
            Give the value of the variable
        derivative_seed : int or float, optional
            Give the derivative seed of the variable. The default is 1.
        """
        if isinstance(value, (int, float)) and isinstance(derivative_seed, (int, float)):
            self.val = value
            self.der = derivative_seed
        else: 
            raise TypeError("Type of value and derivative seed must be int or float!")
    
    
    def __repr__(self):
        """
        INPUTS
        ------
        None

        RETURNS
        -------
        The attributes of the Variable object
        
        EXAMPLES
        --------
        >>> x = Variable(2, 1)
        >>> print(x)
        value = 2, derivative = 1
        """
        
        return f"value = {self.val}, derivative = {self.der}"
    
    
    def get_value(self):
        """
        Method for getting the value of a Variable object

        RETURNS
        -------
        int or float
            give the value of a Variable object.
            
        EXAMPlES
        --------
        >>> x = Variable(2, 1)
        >>> print(x.get_value())
        2
        """
        
        return self.val
    
    
    def get_derivative(self):
        """
        Method for getting the value of a Variable object

        RETURNS
        -------
        int or float
            give the derivative of a Variable object.
            
        EXAMPLES
        --------
        >>> x = Variable(2, 1)
        >>> print(x.get_derivative())
        1

        """
        return self.der
    
    
    def sin(self):
        """
        Value and derivative computation of sin function
        
        INPUTS
        ------
        None
        
        RETURNS
        -------
        A Variable object
            This Variable object has the value and derivative of the sin function.
            
        EXAMPLES
        --------
        >>> import numpy as np
        >>> x = Variable(5, 1)
        >>> print(np.sin(x).get_value())
        -0.9589242746631385
        >>> print(np.sin(x).get_derivative())
        0.2836621854632263
        """
        
        value = np.sin(self.val)
        derivative = np.cos(self.val) * self.der
        return Variable(value, derivative)
        
    
    def cos(self):
        """
        Value and derivative computation of cos function
        
        INPUTS
        ------
        None

        RETURNS
        -------
        A Variable object
            This Variable object has the value and derivative of the cos function.
            
        EXAMPLES
        --------
        >>> import numpy as np
        >>> x = Variable(5, 2)
        >>> print(np.cos(x).get_value())
        0.2836621854632263
        >>> print(np.cos(x).get_derivative())
        1.917848549326277
        """
        
        value = np.cos(self.val)
        derivative = - np.sin(self.val) * self.der
        return Variable(value,derivative)
        
    
    def tan(self):
        """
        Value and derivative computation of tan function

        INPUTS
        ------
        None
        
        RETURNS
        -------
        A Variable object
            This Variable object has the value and derivative of the tan function.
            
        NOTES
        -----
        The function tan doesn't exist at odd multiple of pi/2, however, due to machine precision, the input will 
        never be exactly equal to odd multiple of pi/2. Therefore no ValueError will be raise.
        
        EXAMPLES
        --------
        >>> import numpy as np
        >>> x = Variable(np.pi, 1)
        >>> print(np.tan(x).get_value())
        -1.2246467991473532e-16
        >>> print(np.tan(x).get_derivative())
        1.0
        """
        
        value = np.tan(self.val)
        derivative = (1 / (np.cos(self.val) ** 2))* self.der
        return Variable(value, derivative)
    
    
    def arcsin(self):
        """
        Value and derivative computation of arcsin function

        INPUTS
        ------
        None
        
        RAISES
        ------
        ValueError
            When the input value is larger than 1 or samller than -1.

        RETURNS
        -------
        A Variable object
            This Variable object has the value and derivative of the arcsin function.
            
        EXAMPLES
        --------
        >>> import numpy as np
        >>> x = Variable(0.5, 1)
        >>> print(np.arcsin(x).get_value())
        0.5235987755982988
        >>> print(np.arcsin(x).get_derivative())
        1.1547005383792517
        """
        
        if abs(self.val) >= 1:
            raise ValueError(f"arcsin doesn't exist at {self.val}")
        value = np.arcsin(self.val)
        derivative = 1 / np.sqrt(1 - self.val ** 2) * self.der
        return Variable(value, derivative)
    
    
    def arccos(self):
        """
        Value and derivative computation of arccos function
        
        INPUTS
        ------
        None

        RAISES
        ------
        ValueError
            When the input value is larger than 1 or samller than -1.

        RETURNS
        -------
        A Variable object
            This Variable object has the value and derivative of the arccos function.
            
        EXAMPLES
        --------
        >>> import numpy as np
        >>> x = Variable(0.5, 1)
        >>> print(np.arccos(x).get_value())
        1.0471975511965976
        >>> print(np.arccos(x).get_derivative())
        -1.1547005383792517
        """
        
        if abs(self.val) >= 1:
            raise ValueError(f"arccos doesn't exist at {self.val}")
        value = np.arccos(self.val)
        derivative = - 1 / np.sqrt(1 - self.val ** 2) * self.der
        return Variable(value, derivative)
        
    
    def arctan(self):
        """
        Value and derivative computation of arctan function
        
        INPUTS
        ------
        None
        
        RETURNS
        -------
        A Variable object
            This Variable object has the value and derivative of the arctan function.
            
        EXAMPLES
        --------
        >>> import numpy as np
        >>> x = Variable(2, 1)
        >>> print(np.arctan(x).get_value())
        1.1071487177940906
        >>> print(np.arctan(x).get_derivative())
        0.2
        """
        
        value = np.arctan(self.val)
        derivative = 1 / (1 + self.val ** 2) * self.der
        return Variable(value, derivative)
    
    
    def exp(self):
        """
        Value and derivative computation of exp function
        
        INPUTS
        ------
        None

        RETURNS
        -------
        A Variable object
            This Variable object has the value and derivative of the exp function.
            
        EXAMPLES
        --------
        >>> import numpy as np
        >>> x = Variable(5, 2)
        >>> print(np.exp(x).get_value())
        148.4131591025766
        >>> print(np.exp(x).get_derivative())
        296.8263182051532
        """
        
        value = np.exp(self.val)
        derivative = np.exp(self.val) * self.der
        return Variable(value, derivative)
    
    def __eq__(self, other):
        """
        method for checking if two Variable objects are equal
        
        INPUTS
        ------
        other : A Variable object

        RETURNS
        -------
        bool
            if the values and derivatives of them are equal, it returns True, otherwise it returns False.
            
        EXAMPLES
        --------
        >>> x = Variable(3, 5)
        >>> y = Variable(3, 5)
        >>> x == y
        True

        """
        try:
            if (self.val == other.val) and (self.der == other.der):
                return True
            else:
                return False
        except:
            return False
        
    
    def __ne__(self, other):
        """
        method for checking if two Variable objects are not equal

        INPUTS
        ------
        other : A Variable object

        RETURNS
        -------
        bool
            if the values and derivatives of them are equal, it returns False, otherwise it returns True.
            
        EXAMPLES:
        --------
        >>> x = Variable(3, 5)
        >>> y = Variable(3, 5)
        >>> x != y
        False
        """
        
        try:
            if (self.val == other.val) and (self.der ==other.der):
                return False
            else: 
                return True
        except:
            return True


    def __add__(self, other):
        """
        method for adding two Variable objects

        INPUTS
        ------
        other : A Variable object

        RETURNS
        -------
        A Variable object
            This Variable object has the sum of the value and derivative of the two Variable objects.
            
        EXAMPLES
        --------
        >>> import numpy as np
        >>> x = Variable(5, 2)
        >>> y = Variable(4, 4)
        >>> z = x + y
        >>> print(z.value())
        9
        >>> print(z.get_derivative())
        6
        """
        
        try:
            new_val = self.val + other.val
            new_der = self.der + other.der
            return Variable(new_val, new_der)
        except AttributeError:
            new_val = self.val + other
            new_der = self.der
            return Variable(new_val, new_der)

    def __sub__(self, other):
        """
        method for subtracting one Variable object from another

        INPUTS
        ------
        other : A Variable object

        RETURNS
        -------
        A Variable object
            This Variable object has the difference of the value and derivative of the two Variable objects.
            
        EXAMPLES
        --------
        >>> import numpy as np
        >>> x = Variable(5, 6)
        >>> y = Variable(4, 4)
        >>> z = x - y
        >>> print(z.value())
        1
        >>> print(z.get_derivative())
        2
        """

        try:
            new_val = self.val - other.val
            new_der = self.der - other.der
            return Variable(new_val, new_der)
        except AttributeError:
            new_val = self.val - other
            new_der = self.der
            return Variable(new_val, new_der)

    def __mul__(self, other):
        """
        method for multiplying two Variable objects

        INPUTS
        ------
        other : A Variable object

        RETURNS
        -------
        A Variable object
            This Variable object has the product of the value and derivative of the two Variable objects.
            
        EXAMPLES
        --------
        >>> import numpy as np
        >>> x = Variable(5, 6)
        >>> y = Variable(4, 4)
        >>> z = x * y
        >>> print(z.value())
        20
        >>> print(z.get_derivative())
        44
        """
        
        try:
            new_val = self.val * other.val
            new_der = self.val * other.der + other.val * self.der
            return Variable(new_val, new_der)
        except AttributeError:
            new_val = self.val * other
            new_der = self.der * other
            return Variable(new_val, new_der)

    def __truediv__(self, other):
        """
        method for dividing one Variable object from another

        INPUTS
        ------
        other : A Variable object

        RETURNS
        -------
        A Variable object
            This Variable object has the quotient of the value and derivative of the two Variable objects.
            
        EXAMPLES
        --------
        >>> import numpy as np
        >>> x = Variable(5, 6)
        >>> y = Variable(5, 4)
        >>> z = x / y
        >>> print(z.value())
        1
        >>> print(z.get_derivative())
        0.4
        """

        try:
            new_val = self.val / other.val
            new_der = (self.der * other.val - self.val * other.der)/(other.self * other.self)
            return Variable(new_val, new_der)
        except AttributeError:
            new_val = self.val / other
            new_der = self.der / other
            return Variable(new_val, new_der)
        except ZeroDivisionError:
            print("Cannot divide by 0.")

    __radd__ = __add__

    __rsub__ = __sub__

    __rmul__ = __mul__

    __rtruediv__ = __truediv__
            
    
def make_variables(var_list, der_list):
    """
    Function to create a list of Variable objects

    INPUTS
    ------
    var_list : list of int or float
        input values of these new Variable objects.
    der_list : list of int or float
        input derivative seeds of these new Variable objects.

    RAISES
    ------
    ValueError
        if the input values and input derivative seeds are of different lengths.

    RETURNS
    -------
    variables : list of new Variable objects created.
    """
    
    if len(var_list) != len(der_list):
        raise ValueError("The value list and derivative list should be of the same length")
    
    variables = []
    for val, der in zip(var_list, der_list):
        variables.append(Variable(val, der))
        
    return variables



def make_variable(var,der):
    """
    Function to create a Variable object

    INPUTS
    ------
    var : int or float
        input value of the Variable object.
    der : int or float
        input derivative seed of the Variable object.

    RETURNS
    -------
    A new Variable object
    
    EXAMPLES
    --------
    >>> x = make_variable(1, 1)
    >>> print(x.get_derivative())
    1
    >>> print(x.get_value())
    1
    """
    return Variable(var, der)



if __name__ == "__main__":
    import doctest
    doctest.testmod()
    
