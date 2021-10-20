# Milestone 1 Documentation

## Introduction

Differentiation plays a key role in modern algorithms in various fields of computational analysis, such as statistics, physics, machine learning and deep learning which are hot topics in  recent years. Newton's method for finding roots of functions without closed-form solution, backward propagation in training neural networks, and optimization (e.g. gradient descent) in machine learning algorithms, all rely on differentiation as their cornerstone. Therefore, the ability to quickly, accurately and efficiently compute differentiation is crucial. 

There are many ways to compute derivatives in a machine, for example, numerical differentiation and symbolic differentiation. For symbolic differentiation, it computes the expression of the derivative symbolically. This method is convenient for simple expressions, but can be inefficient and difficult to compute when the expression goes complex to higher orders, especially for gradient descent in machine learning when the target function has a complex form.

Numerical differentiation relies on finite-differences <img src="https://render.githubusercontent.com/render/math?math=\frac{df}{dx}=\frac{f(x+\epsilon)}{\epsilon}"> as <img src="https://render.githubusercontent.com/render/math?math=\epsilon"> goes to 0 to approximate the value of derivatives at a specific point. However,  choosing the most appropriate <img src="https://render.githubusercontent.com/render/math?math=\epsilon"> value is not an easy job. When the <img src="https://render.githubusercontent.com/render/math?math=\epsilon"> is too small, it has the problem of round-off errors associated with machine precision, and the approximation can be inaccurate when the <img src="https://render.githubusercontent.com/render/math?math=\epsilon"> is too large.

"Automatic Differentiation (AD)", is a technique to  compute derivatives numerically at a specified point, and surpasses the above two methods e in both time and resource efficiency and computational accuracy. Taking the symbolic derivative of a function can be difficult for a computer to implement as the function becomes more complex. Specifically, vectorized functions are especially difficult to take derivatives of symbolically so AD is useful in allowing users to take the gradient of complex functions at particular numerical values using the function’s computational graph. Furthermore, AD completes the numerical derivatives without losing accuracy and also improves computational speed compared to traditional symbolic derivatives for complex functions.

In our lahg_package, we implement the AD for both forward and reverse mode (The extension we choose). In section 2, you can find the details and mathematical background of AD. In section 3, you will get a sense of how to use our package. Section 4 and 5 provide a detailed introduction about how our package is planned. Section 6 is the short motivation for our license choice.


## Background

## How to use <AAD>

Our automatic differentiation package can be installed using command line, as we are going to distribute our package in PyPI:
```
python -m pip install lagh_package
```
Then users can import the package and all modules included using the command:

```python
import lagh_package as AD
```

To make use of automatic differentiation function, users will need to initiate AD variables/objects with value at a specified point and pass the derivative seed, for example 

```python
x = AD.make_variable(2,1)
y = AD.make_variable(5,0)
f = (x * y + AD.sin(x) + 1 - AD.cos(y))**0.5
print(f"value = {f.getvalue()}; derivative = {f.getderivative()}")
```

For higer dimensional functions where the derivative output should be a Jacobian matrix, we envision users to get the result using Jacobian function of our package.

```python
x = AD.make_variable(2,1)
y = AD.make_variable(5,0)
J=AD.Jacobian([x+y, x-y, x*y])
print(f"value = {J["val"]}; derivative = {J["der"]}")
```



## Organization
### Directory structure and modules

```
<Project Name>
├── README.md               Main project README
├── requirements.txt        Package dependencies
├── setup.py                setup function for package
│
├── docs
│   ├── README.md           README for docs
│   └── milestone[i].md    Documentation for each milestone
│
├── src:                    Package source files
│   ├── AD.py               Main constructor
│   ├── AD_helper.py        Helper functions
│   ├── AD_forward.py       Forward mode
│   └── AD_reverse.py       Reverse mode
│
└── tests                   Package test scripts
    ├── run_tests.py       script that runs all tests
    └── test_[i].py        tests function [i]

```
### What modules do you plan on including? What is their basic functions.
The modules we will include are
* numpy: numpy supports computation for multi-dimensional arrays and matricies
* math: math supports common mathematical objects and functions required for automatic differentiation
* setuptools: this allows us to setup our pacakge and facilitates easy installation and  distribution

### Where will your test suite live? Will you use TravisCI? CodeCov?

* Our test suite will live in the `tests/` directory, which will contain scripts that tests each function in our class
* TravisCI and CodeCov will be used to test our package and check code coverage.

### How will you distribute your package (e.g. PyPI)?

* We will use both Github and PyPI to distribute our package, this allows the user to install our package using both `pip` or building straight from source.
* We will also consider distribution using Anaconda since that is a popular numerical programming package distribution channel.

### How will you package your software? Will you  use a framework? If so which one and why? If not, why not?

* We will not use a framework for our software because it is simple enough to not use a framework -- a framework will overcomplicate the design.
* We will use `setup_tools` library to help us with package development.

## Implementation

### What are the core data structures?

* The core data structures we will be using for our implementation are numpy arrays and python dictionaries to store our partial derivatives. 

### What classes will you implement?

* We will be creating one class called an AD class, which will handle our automatic differentiations. 
* Within the class, we will be defining some parameters within our function, including self.val and self.der, which are the value and derivatives of an intermediate.

### What method and name attributes will your classes have?

The method and name attributes our class will have include:
* __init__: This method will initialize our parameters in our class
* __add__: This method will be used to add two values together
* __sub__: This method will be used to subtract a value from another
* __mul__: This method will be used to multiply two values
* __truediv__: This method will be used to divide a value by another
* __sin__: This method will be used to find the sine of a value
* __cos__: This method will be used to find the cosine of a value
* self.getvalue: This method will return the value of an intermediate
* self.getderivative: This method will return the derivative of an intermediate
* self.make_variables: This method will create intermediates from values and seeds
* self.exp: This method will return e raised to the power of the intermediate
* self.power: This method will return the intermediate raised to a power
* self.log: This method will return the log of an intermediate
* self.sqrt: This method will return the square root of an intermediate
* self.cross_product: This method will return the cross product of two vectors
* __str__: This method will return a printed representation of a value and its derivative

### What external dependencies will you rely on?

* An external dependency we will rely on is numpy to create our numpy arrays.

### How will you deal with elementary functions like sin, sqrt, log, and exp (and all the others)?

* We will either be implementing dunder methods or creating functions for each elementary function. Whenever an elementary function needs to be applied to an intermediate value, we will pass the intermediate value into the method/function. For example, __add__, __sub__, __mul__, __truediv__, __sin__ and __cos__ will handle addition, subtraction, multiplication, division, sine and cosine functions, respectively. Values will be passed into the function and the operation will be applied to the values. For operations that don’t have default methods, like exp, power, log and sqrt, we will create functions to handle those operators. 
* To handle vectors as input, we will first check the length of our input to see if it is a vector or a scalar, Then based on the length, we will perform the specific operation. For vector functions of vectors, such as cross product, vectors will be passed into the function and the function will return a vector after an operation has been performed on them.

## License

We choose **MIT License** for our packgae, since it is simple and permissive. We want to enable others from the community to make use of our code and package, and we are also willing to accept contributions from the community such as forking. In our package we make use of existing open-source libraries such as NumPy and math, we would like to make our package open-source too. Currently we are not going to deal with patent issue, therefore the weak copyleft license works for us.
