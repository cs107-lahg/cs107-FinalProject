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
    ========
    # scalar input
    >>> x = Variable(6, 1)
    >>> f = 15 / (x * 3 + 10)
    >>> print(f)
    value = 0.5357142857142857, derivative = -0.05739795918367347
    
    # vector input
    >>> x = Variable(2, 1)
    >>> y = Variable(3, 0)
    >>> f = (x + y) * y
    >>> print(f)
    value = 15, derivative = 3
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
            
        EXAMPLES
        --------
        >>> x = Variable(2, 1)
        >>> print(x)
        value = 2, derivative = 1
        """
        
        if isinstance(value, (int, float)) and isinstance(derivative_seed, (int, float)):
            self.val = value
            self.der = derivative_seed
        else: 
            raise TypeError("Type of value and derivative seed must be int or float!")
    
    
    def __repr__(self):
        """
        Dunder method for printing output
        
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

        INPUTS
        ------
        None
        
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

        INPUTS
        ------
        None
        
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
      
    
    def __neg__(self):
        """
        Method for taking negative (-) of a Variable object
        
        INPUTS
        ------
        None

        RETURNS
        -------
        A Variable object
        
        EXAMPLES:
        --------
        >>> x = Variable(2, 7)
        >>> print(- x)
        value = -2, derivative = -7
        """
        
        value = - self.val
        derivative = - self.der
        return Variable(value, derivative)
    
    
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
        >>> f = x.sin()
        >>> print(f)
        value = -0.9589242746631385, derivative = 0.2836621854632263
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
        >>> x = Variable(5, 2)
        >>> f = x.cos()
        >>> print(f)
        value = 0.2836621854632263, derivative = 1.917848549326277
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
        never be exactly equal to odd multiple of pi/2. Therefore no ValueError will be raised.
        
        EXAMPLES
        --------
        >>> import numpy as np
        >>> x = Variable(np.pi, 1)
        >>> f = x.tan()
        >>> print(f)
        value = -1.2246467991473532e-16, derivative = 1.0
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
            When the input value is larger than 1 or smaller than -1.

        RETURNS
        -------
        A Variable object
            This Variable object has the value and derivative of the arcsin function.
            
        EXAMPLES
        --------
        >>> import numpy as np
        >>> x = Variable(0.5, 1)
        >>> f = x.arcsin()
        >>> print(f)
        value = 0.5235987755982988, derivative = 1.1547005383792517
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
        >>> f = x.arccos()
        >>> print(f)
        value = 1.0471975511965976, derivative = -1.1547005383792517
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
        >>> f = x.arctan()
        >>> print(f)
        value = 1.1071487177940906, derivative = 0.2
        """
        
        value = np.arctan(self.val)
        derivative = 1 / (1 + self.val ** 2) * self.der
        return Variable(value, derivative)
    
    
    def sinh(self):
        """
        Value and derivative computation of sinh function
        
        RETURNS
        -------
        A Variable object
        
        EXAMPLES
        --------
        >>> x = Variable(2, 5)
        >>> f = x.sinh()
        >>> print(f)
        value = 3.6268604078470186, derivative = 18.81097845541816
        """
        
        val = np.sinh(self.val)
        der = np.cosh(self.val) * self.der
        return Variable(val, der)
    
    
    def cosh(self):
        """
        Value and derivative computation of cosh function

        RETURNS
        -------
        A Variable object
        
        EXAMPLES
        --------
        >>> x = Variable(6, 1)
        >>> f = x.cosh()
        >>> print(f)
        value = 201.7156361224559, derivative = 201.71315737027922
        """
        
        val = np.cosh(self.val)
        der = np.sinh(self.val) * self.der
        return Variable(val, der)
    
    
    def tanh(self):
        """
        Value and derivative computation of tanh function

        RETURNS
        -------
        A Variable object
        
        EXAMPLES
        --------
        >>> x = Variable(2, 1)
        >>> f = x.tanh()
        >>> print(f)
        value = 0.9640275800758169, derivative = 0.07065082485316447
        """
        
        val = np.tanh(self.val)
        der = (1 / (np.cosh(self.val) ** 2)) * self.der
        return Variable(val, der)
    
    
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
        >>> f = x.exp()
        >>> print(f)
        value = 148.4131591025766, derivative = 296.8263182051532
        """
        
        value = np.exp(self.val)
        derivative = np.exp(self.val) * self.der
        return Variable(value, derivative)
    
    def __eq__(self, other):
        """
        Method for checking if two Variable objects are equal, overloads ==
        
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
        
        >>> x = Variable(3, 5)
        >>> y = Variable(3, 6)
        >>> x == y
        False
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
        Method for checking if two Variable objects are not equal, overloads !=

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
        Method for adding two quantities, overloads +

        INPUTS
        ------
        other : A Variable object or a real number

        RETURNS
        -------
        A Variable object
            This Variable object has the sum of the value and derivative of the two Variable objects.
            
        EXAMPLES
        --------
        # addition of two Variable objects
        >>> x = Variable(5, 2)
        >>> y = Variable(4, 4)
        >>> f = x + y
        >>> print(f)
        value = 9, derivative = 6
        
        # addition of one Variable object and a real number
        >>> x = Variable(5, 1)
        >>> y = 10
        >>> f = x + y
        >>> print(f)
        value = 15, derivative = 1
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
        Method for subtracting one quantity from another, overloads -

        INPUTS
        ------
        other : A Variable object or a real number

        RETURNS
        -------
        A Variable object
            This Variable object has the difference of the value and derivative of the two Variable objects.
            
        EXAMPLES
        --------
        # subtraction of two Variable objects
        >>> x = Variable(5, 6)
        >>> y = Variable(4, 4)
        >>> f = x - y
        >>> print(f)
        value = 1, derivative = 2
        
        # subtraction of one Variable object and a real number
        >>> x = Variable(5, 1)
        >>> y = 5
        >>> f = x - y
        >>> print(f)
        value = 0, derivative = 1
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
        Method for multiplying two quantities, overloads *

        INPUTS
        ------
        other : A Variable object or a real number

        RETURNS
        -------
        A Variable object
            This Variable object has the product of the value and derivative of the two Variable objects.
            
        EXAMPLES
        --------
        # multiplication of two Variable objects
        >>> x = Variable(5, 6)
        >>> y = Variable(4, 4)
        >>> f = x * y
        >>> print(f)
        value = 20, derivative = 44
        
        # multiplication of a Variable object and a real number
        >>> x = Variable(5, 6)
        >>> y = 10
        >>> f = x * y
        >>> print(f)
        value = 50, derivative = 60
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
        method for division of two quantities, overloads /

        INPUTS
        ------
        other : A Variable object or a real number

        RETURNS
        -------
        A Variable object
            This Variable object has the quotient of the value and derivative of the two Variable objects.
            
        EXAMPLES
        --------
        # test division of two Variable object
        >>> import numpy as np
        >>> x = Variable(5, 6)
        >>> y = Variable(5, 4)
        >>> f = x / y
        >>> print(f)
        value = 1.0, derivative = 0.4
        
        # test division of a Variable object and a real number
        >>> import numpy as np
        >>> x = Variable(5, 6)
        >>> y = 5
        >>> f = x / y
        >>> print(f)
        value = 1.0, derivative = 1.2
        """
        
        try:
            if other.val == 0:
                raise ZeroDivisionError("Cannot divide by zero!")
            new_val = self.val / other.val
            new_der = (self.der * other.val - self.val * other.der)/(other.val ** 2)
            return Variable(new_val, new_der)
        except AttributeError:
            if other == 0:
                raise ZeroDivisionError("Cannot divide by zero!")
            new_val = self.val / other
            new_der = self.der / other
            return Variable(new_val, new_der)
        
            
            
    def __pow__(self, other):
        
        """
        Method for raising one quantity to the power of another, overloads **

        INPUTS
        ------
        other : a real number
        
        RAISES
        ------
        TypeError
            When other is a non-real number input
        ValueError
            When trying to raise a negative number to a fraction of a power (ex. (-2)**.2)


        RETURNS
        -------
        A Variable object
            This Variable object has the value raised to the power specified and new derivative.
            
        EXAMPLES
        --------
        # Raising a Variable object to the power of a positive int
        >>> x = Variable(5, 1)
        >>> y = 2
        >>> f = x ** y
        >>> print(f)
        value = 25.0, derivative = 10.0
        
        # Raising a Variable object to the power of a negative float
        >>> x = Variable(5, 1)
        >>> y = -2.5
        >>> f = x ** y
        >>> print(f)
        value = 0.01788854381999832, derivative = -0.00894427190999916
        """
        
        if self.val < 0 and isinstance(other, float) and (other - int(other) != 0):
            raise ValueError("Cannot take the root of a negative number")
        
        if not(isinstance(other, (int, float)) or isinstance(other, Variable)):
            raise TypeError("Can only raise to the power of a real number or variable!")
            
        try:
            other1 = float(other.val)
            new_val = np.power(self.val, other1)
            new_der = np.log(self.val) * np.power(self.val, other1)*self.der
            return Variable(new_val, new_der)
        
        except AttributeError:
            other = float(other)
            new_val = np.power(self.val, other)
            new_der = other*np.power(self.val, other-1)*self.der
            return Variable(new_val.item(), new_der.item())
          
    def log (self):
        """
        Value and derivative computation of the log function (base e)
        
        INPUTS
        ------
        None
        
        RAISES
        ------
        ValueError
            When the input value is less than or equal to 0

        
        RETURNS
        -------
        A Variable object
            This Variable object has the value and derivative of the log function.
            
        EXAMPLES
        --------
        >>> import numpy as np
        >>> x = Variable(5, 1)
        >>> f = x.log()
        >>> print(f)
        value = 1.6094379124341003, derivative = 0.2
        """
        if self.val <= 0:
            raise ValueError("Cannot take the log of 0 or a negative number")
 
        value = np.log(self.val)
        derivative = (1/self.val)*self.der
        return Variable(value, derivative)
    
    def sqrt (self):
        """
        Value and derivative computation of the square root function
        
        INPUTS
        ------
        None
        
        RETURNS
        -------
        A Variable object
            This Variable object has the value and derivative of the square root function.
            
        EXAMPLES
        --------
        >>> import numpy as np
        >>> x = Variable(5, 1)
        >>> f = x.sqrt()
        >>> print(f)
        value = 2.23606797749979, derivative = 0.22360679774997896
        """
        errormsg = 'cannot calculate square root of negative numbers, or derivative where value is 0'\
        ' because division by zero error'\
        
        if self.val <= 0 :
            raise ValueError(errormsg)
        value = np.sqrt(self.val)
        derivative = .5*(1/np.sqrt(self.val))*self.der
        return Variable(value, derivative)
    
    def __radd__(self, other):
        """
        Method for performing right side addition 

        INPUTS
        ------
        other : A Variable object
            

        RETURNS
        -------
        A Variable object
        
        EXAMPLES
        --------
        # addition of a Variable object and a number
        >>> x = Variable(2, 5)
        >>> f = 3 + x
        >>> print(f)
        value = 5, derivative = 5
        
        # addition of two Variable objects
        >>> x = Variable(1, 2)
        >>> y = Variable(2 ,3)
        >>> f = x + y
        >>> print(f)
        value = 3, derivative = 5
        """
        
        return self.__add__(other)


    def __rmul__(self, other):
        """
        Method for performing right side multiplication 
        
        INPUTS
        ----------
        other : A Variable object

        RETURNS
        -------
        A Variable object
        
        EXAMPLES:
        --------
        # multiplication of two Variable objects
        >>> x = Variable(2, 3)
        >>> y = Variable(3, 4)
        >>> f = x * y
        >>> print(f)
        value = 6, derivative = 17
        
        # multiplication of a Variable object and a number 
        >>> x = Variable(5, 6)
        >>> y = 3
        >>> f = x * y
        >>> print(f)
        value = 15, derivative = 18
        """
        
        return self.__mul__(other)
    
    
    def __rsub__(self, other):
        """
        Method for performing right side subtraction 

        INPUTS
        ------
        other : A Variable object or a real number

        RETURNS
        -------
        A Variable object
        
        EXAMPLES
        --------
        # subtraction of a Variable object and a real number
        >>> x = Variable(0)
        >>> y = 10
        >>> f = y - x
        >>> print(f)
        value = 10, derivative = -1
        
        # subtraction of two Variable objects
        >>> x = Variable(3, 5)
        >>> y = Variable(2, 3)
        >>> f = y - x
        >>> print(f)
        value = -1, derivative = -2
        """
        
        return other + (-self)
        
        

    def __rtruediv__(self, other):
        """
        Method for performing right side division

        INPUTS
        ------
        other : A Variable object or a real number

        RETURNS
        -------
        A Variable object
        
        EXAMPLES
        --------
        # division of two Variable objects
        >>> x = Variable(2, 3)
        >>> y = Variable(1, 2)
        >>> f = y / x
        >>> print(f)
        value = 0.5, derivative = 0.25
        
        # division of a Variable object and a number
        >>> x = Variable(2, 3)
        >>> y = 10
        >>> f = y / x
        >>> print(f)
        value = 5.0, derivative = -7.5
        
        # 0 divides a Variable object
        >>> y = 0
        >>> x = Variable(1, 2)
        >>> f = 0 / x
        >>> print(f)
        value = 0.0, derivative = 0.0
        """
        
        if self.val == 0:
                raise ZeroDivisionError("Cannot divide by zero!")
        try:
            new_val = other.val / self.val
            new_der = (other.der * self.val - other.val * self.der)/(self.val ** 2)
            return Variable(new_val, new_der)
        except AttributeError:
            new_val = other / self.val
            new_der = - other/ (self.val ** 2) * self.der
            return Variable(new_val, new_der)

            
    
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
    
    EXAMPLES
    --------
    >>> x = make_variables([1, 2], [1, 0])
    >>> print(x[0])
    value = 1, derivative = 1
    >>> print(x[1])
    value = 2, derivative = 0
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
    >>> print(x)
    value = 1, derivative = 1
    """
    return Variable(var, der)



if __name__ == "__main__":
    import doctest
    doctest.testmod()