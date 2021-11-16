import pytest
import os
import sys

# from lahg_ad.fd import Variable

os.chdir(sys.path[0])
sys.path.append('../')
import lahg_ad as ad
import numpy as np

def test_Variable():
    ad_var = ad.Variable(0)
    assert ad_var.val == 0
    assert ad_var.der == 1
    
def test_get_value():
    x = ad.Variable(1.03)
    assert x.get_value() == 1.03

def test_get_derivative():
    x = ad.Variable(1.03, 5.02)
    assert x.get_derivative() == 5.02
    
def test_variable_types():
    with pytest.raises(TypeError):
        assert ad.Variable('test')
        
def test_make_variable():
    # assert ad.make_variable(3, 5) == ad.Variable(3,5)
    assert ad.make_variable(3, 5).val == 3.0
    assert ad.make_variable(3, 5).der == 5.0
    
def test_make_variables():
    v = ad.make_variables([5,6,7], [1,2,3])
    assert v[0] == ad.Variable(5,1)
    assert v[1] == ad.Variable(6,2)
    assert v[2] == ad.Variable(7,3)
    
def test_cos():
    assert ad.Variable(0).cos().val == 1.0
    assert ad.Variable(0).cos().der == np.sin(0)

def test_sin():
    x = ad.Variable(0).sin()
    assert y.val == 0
    assert y.der == np.cos(0)
    
def test_sin():
    x = ad.Variable(0)
    y = x.sin()
    assert y.val == 0
    assert y.der == np.cos(0)
    
def test_arcsin_domain():
    with pytest.raises(ValueError):
        ad.Variable(1).arcsin()
    with pytest.raises(ValueError):
        ad.Variable(-1).arcsin()
        
def test_arccos_domain():
    with pytest.raises(ValueError):
        ad.Variable(1).arccos()
    with pytest.raises(ValueError):
        ad.Variable(-1).arccos()

def test_exp():
    assert ad.Variable(1.05).cos().val == np.exp(1.05)
    assert ad.Variable(1.05).cos().der == np.exp(1.05)
    
def test_exp():
    assert ad.Variable(1.05).cos().val == np.exp(1.05)
    assert ad.Variable(1.05).cos().der == np.exp(1.05)

if __name__ == '__main__':
    test_arcsin_domain()
    test_get_derivative()
    test_get_value()
    test_make_variable()
    test_make_variables()
