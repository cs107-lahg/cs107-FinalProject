import pytest
import os
import sys

os.chdir(sys.path[0])
sys.path.append('../lahg_ad/')
import fd
import numpy as np

def test_Variable():
    assert fd.Variable(0) == (0, 1)
    
def test_variable_types():
    with pytest.raises(TypeError):
        assert fd.Variable('test')
    
def test_make_variables():
    with pytest.raises(ValueError):
        assert fd.make_variable(0,0) == (0, 0)
        
def test_make_variables():
    with pytest.raises(ValueError):
        assert fd.make_variable(0,0) == (0, 0)

def test_cos():
    x = fd.Variable(0)
    y = x.cos()
    assert y.val == 1.0
    assert y.der == np.sin(0)

def test_sin():
    x = fd.Variable(0)
    y = x.sin()
    assert y.val == 0
    assert y.der == np.cos(0)
    
def test_sin():
    x = fd.Variable(0)
    y = x.sin()
    assert y.val == 0
    assert y.der == np.cos(0)

if __name__ == '__main__':
    print(fd.Variable('test'))
    # print(fd.Variable(0))
    # print(fd.Variable(0,5).exp().der)