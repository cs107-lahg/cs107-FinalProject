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
    # assert v[0] == ad.Variable(5,1)
    # assert v[1] == ad.Variable(6,2)
    # assert v[2] == ad.Variable(7,3)
    
def test_cos():
    assert ad.Variable(0).cos().val == 1.0
    assert ad.Variable(0).cos().der == np.sin(0)

def test_sin():
    x = ad.Variable(0).sin()
    assert x.val == 0
    assert x.der == np.cos(0)
    
def test_arcsin_domain():
    with pytest.raises(ValueError):
        ad.Variable(1).arcsin()
    with pytest.raises(ValueError):
        ad.Variable(-1).arcsin()
        
def test_arcsin():
    value, deriv_seed = np.random.uniform(size=2)
    x = ad.Variable(value, deriv_seed).arcsin()
    assert x.val == np.arcsin(value)
    assert x.der == 1/(np.sqrt(1-value**2)) * deriv_seed 
    
def test_arccos():
    value, deriv_seed = np.random.uniform(size=2)
    x = ad.Variable(value, deriv_seed).arccos()
    assert x.val == np.arccos(value)
    assert x.der == -1/(np.sqrt(1-value**2)) * deriv_seed 
        
def test_arccos_domain():
    with pytest.raises(ValueError):
        ad.Variable(1).arccos()
    with pytest.raises(ValueError):
        ad.Variable(-1).arccos()

def test_exp():
    assert ad.Variable(1.05, 3.2).exp().val == np.exp(1.05)
    assert ad.Variable(1.05, 3.2).exp().der == np.exp(1.05)*3.2
    
def test_tan():
    assert ad.Variable(1.05, 1).tan().val == np.tan(1.05)
    assert ad.Variable(1.05, 1).tan().der == 1/(np.cos(1.05)**2)
    
def test_arctan():
    value, deriv_seed = np.random.uniform(size=2)
    x = ad.Variable(value, deriv_seed).arctan()
    assert x.val == np.arctan(value)
    assert x.der == 1/(1 + value**2) * deriv_seed 


if __name__ == '__main__':
    test_arcsin_domain()
    test_get_derivative()
    test_get_value()
    test_make_variable()
    test_make_variables()
    test_tan()
    test_arctan()
    test_arccos()
    test_arcsin()