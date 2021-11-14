import pytest
import os
import sys

os.chdir(sys.path[0])
sys.path.append('../')
import lahg_ad.fd as ad
import numpy as np

def test_Variable():
    ad_var = ad.Variable(0)
    assert ad_var.val == 0
    assert ad_var.der == 1
    
def test_variable_types():
    with pytest.raises(TypeError):
        assert ad.Variable('test')
        
def test_make_variable():
    assert ad.make_variable(3, 5).val == 3.0
    assert ad.make_variable(3, 5).der == 5.0
    
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

if __name__ == '__main__':
    print(ad.Variable('test'))
    test_Variable()
    # test_variable_types()
    # print(ad.Variable(0))
    # print(ad.Variable(0,5).exp().der)