import pytest
import os
import sys

os.chdir(sys.path[0])
sys.path.append('../lahg_ad/')
import ad as ad
import numpy as np

def test_Variable():
    ad_var = ad.Variable(0)
    assert ad_var.val == 0
    assert ad_var.der == 1
    
def test_variable_types():
    with pytest.raises(TypeError):
        assert ad.Variable('test')
    
def test_make_variables():
    with pytest.raises(ValueError):
        assert ad.make_variable(0,0) == (0, 0)
        
def test_make_variables():
    with pytest.raises(ValueError):
        assert ad.make_variable(0,0) == (0, 0)

def test_cos():
    x = ad.Variable(0)
    y = x.cos()
    assert y.val == 1.0
    assert y.der == np.sin(0)

def test_sin():
    x = ad.Variable(0)
    y = x.sin()
    assert y.val == 0
    assert y.der == np.cos(0)
    
def test_sin():
    x = ad.Variable(0)
    y = x.sin()
    assert y.val == 0
    assert y.der == np.cos(0)

if __name__ == '__main__':
    print(ad.Variable('test'))
    # print(ad.Variable(0))
    # print(ad.Variable(0,5).exp().der)