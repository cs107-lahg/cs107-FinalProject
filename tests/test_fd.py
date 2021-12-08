import pytest
import os
import sys

os.chdir(sys.path[0])
sys.path.append('../')
import lahg_ad as ad
import numpy as np

def test_Variable():
    ad_var = ad.Variable(0)
    assert ad_var.val == 0
    assert ad_var.der == 1

    ad_var_2 = ad.Variable(np.array([2, 1]), np.array([3, 4]))
    assert np.array_equal(ad_var_2.val, np.array([2, 1]))
    assert np.array_equal(ad_var_2.der, np.array([3, 4]))

def test_repr():
    assert ad.Variable(1).__repr__() == "value = 1, derivative = 1"

def test_get_value():
    x = ad.Variable(1.03)
    assert x.get_value() == 1.03

    x = ad.Variable(np.array([2, 1]), np.array([3, 4]))
    assert np.array_equal(x.get_value(), np.array([2, 1]))

def test_get_derivative():
    x = ad.Variable(1.03, 5.02)
    assert x.get_derivative() == 5.02

    x = ad.Variable(np.array([2, 1]), np.array([3, 4]))
    assert np.array_equal(x.get_derivative(), np.array([3, 4]))

def test_neg():
    x = ad.Variable(2, 7)
    assert -x.val == -2
    assert -x.der == -7
    x = ad.Variable(np.array([2, 7]), np.array([8, 7]))
    assert np.array_equal(-x.val, np.array([-2, -7]))
    assert np.array_equal(-x.der, np.array([-8, -7]))

def test_sin():
    x = ad.Variable(0).sin()
    assert x.val == 0.0
    assert x.der == 1.0

    x = ad.Variable(5, 1).sin()
    assert x.val == -0.9589242746631385
    assert x.der == 0.28366218546322625

    x = ad.Variable(np.array([np.pi/2, np.pi/2]), np.array([0, 0])).sin()
    assert np.array_equal(x.val, np.array([1, 1]))
    assert np.array_equal(x.der, np.array([0, 0]))
    
def test_cos():
    x = ad.Variable(0).cos()
    assert x.val == 1.0
    assert x.der == 0.0

    x = ad.Variable(np.array([0, 0]), np.array([np.pi/2, np.pi/2])).cos()
    assert np.array_equal(x.val, np.array([1, 1]))
    assert np.array_equal(x.der, np.array([0, 0]))

def test_tan():
    assert ad.Variable(1.05, 1).tan().val == np.tan(1.05)
    assert ad.Variable(1.05, 1).tan().der == pytest.approx(1/(np.cos(1.05)**2))

    x = ad.Variable(np.array([np.pi/4, 0]), np.array([0, 1])).tan()
    assert np.array_equal(x.val, np.array([np.tan(np.pi/4), np.tan(0)]))
    assert np.array_equal(x.der, np.array([0, 1]))

def test_arcsin():
    value, deriv_seed = np.random.uniform(size=2)
    x = ad.Variable(value, deriv_seed).arcsin()
    assert x.val == np.arcsin(value)
    assert x.der == 1/(np.sqrt(1-value**2)) * deriv_seed 

    value1, value2, deriv_seed1, deriv_seed2 = np.random.uniform(size=4)
    x = ad.Variable(np.array([value1, value2]), np.array([deriv_seed1, deriv_seed2])).arcsin()
    assert np.array_equal(x.val, np.array([np.arcsin(value1), np.arcsin(value2)]))
    assert np.array_equal(x.der, np.array([1/(np.sqrt(1-value1**2)) * deriv_seed1, 1/(np.sqrt(1-value2**2)) * deriv_seed2]))
    
def test_arccos():
    value, deriv_seed = np.random.uniform(size=2)
    x = ad.Variable(value, deriv_seed).arccos()
    assert x.val == np.arccos(value)
    assert x.der == -1/(np.sqrt(1-value**2)) * deriv_seed 

    value1, value2, deriv_seed1, deriv_seed2 = np.random.uniform(size=4)
    x = ad.Variable(np.array([value1, value2]), np.array([deriv_seed1, deriv_seed2])).arccos()
    assert np.array_equal(x.val, np.array([np.arccos(value1), np.arccos(value2)]))
    assert np.array_equal(x.der, np.array([-1/(np.sqrt(1-value1**2)) * deriv_seed1 , -1/(np.sqrt(1-value2**2)) * deriv_seed2]))
    
def test_arctan():
    value, deriv_seed = np.random.uniform(size=2)
    x = ad.Variable(value, deriv_seed).arctan()
    assert x.val == np.arctan(value)
    assert x.der == 1/(1 + value**2) * deriv_seed 

    value1, value2, deriv_seed1, deriv_seed2 = np.random.uniform(size=4)
    x = ad.Variable(np.array([value1, value2]), np.array([deriv_seed1, deriv_seed2])).arctan()
    assert np.array_equal(x.val, np.array([np.arctan(value1), np.arctan(value2)]))
    assert np.array_equal(x.der, np.array([1/(1 + value1**2) * deriv_seed1, 1/(1 + value2**2) * deriv_seed2]))

def test_sinh():
    assert ad.Variable(1).sinh().val == 1.1752011936438014
    assert ad.Variable(1).sinh().der == 1.5430806348152437

    x = ad.Variable(np.array([1, 2]), np.array([3, 4])).sinh()
    assert np.array_equal(x.val, np.array([np.sinh(1), np.sinh(2)]))
    assert np.array_equal(x.der, np.array([np.cosh(1) * 3, np.cosh(2) * 4]))

def test_cosh():
    assert ad.Variable(1).cosh().val == 1.5430806348152437
    assert ad.Variable(1).cosh().der == 1.1752011936438014

    x = ad.Variable(np.array([1, 2]), np.array([3, 4])).cosh()
    assert np.array_equal(x.val, np.array([np.cosh(1), np.cosh(2)]))
    assert np.array_equal(x.der, np.array([np.sinh(1) * 3, np.sinh(2) * 4]))
    
def test_tanh():
    assert ad.Variable(0).tanh().val == 0.0
    assert ad.Variable(0).tanh().der == 1
    assert ad.Variable(1).tanh().val == np.tanh(1)
    assert ad.Variable(1).tanh().der == pytest.approx(1 - np.tanh(1)**2)

    x = ad.Variable(np.array([1, 2]), np.array([3, 4])).tanh()
    assert np.array_equal(x.val, np.array([np.tanh(1), np.tanh(2)]))
    assert np.array_equal(x.der, np.array([(1 / (np.cosh(1) ** 2)) * 3, (1 / (np.cosh(2) ** 2)) * 4]))

def test_exp():
    assert ad.Variable(1.05, 3.2).exp(base = 2).val == 2 ** 1.05
    assert ad.Variable(1.05, 3.2).exp(base = 2).der == 4.592582163796847
    
    x = ad.Variable(np.array([1, 2]), np.array([3, 4])).exp()
    assert np.array_equal(x.val, np.array([np.exp(1), np.exp(2)]))
    assert np.array_equal(x.der, np.array([np.exp(1)*3, np.exp(2)*4]))
    with pytest.raises(ValueError):
        ad.Variable(1.05, 3.2).exp(base = "Exponential")

def test_eq():
    assert ad.Variable(1, 2) == ad.make_variable(1, 2)
    assert (ad.Variable(1, 2) == 1) == False
    assert ad.Variable(np.array([2, 1]), 3) == ad.make_variable(np.array([2, 1]), 3)
    assert ad.Variable(3, np.array([4, 5])) == ad.make_variable(3, np.array([4, 5]))

def test_ne():
    assert ad.Variable(1, 2) != ad.Variable(2, 3)
    assert ad.Variable(1, 2) != 1

    assert np.not_equal(ad.Variable(np.array([1, 2]), np.array([1, 2])), ad.Variable(np.array([1, 2]), np.array([3, 4])))
    assert np.not_equal(ad.Variable(np.array([1, 2]), np.array([1, 2])), 1)
    
def test_add():
    assert (ad.Variable(1) + 1).val == 2
    assert (ad.Variable(1) + 1).der == 1
    assert (1 + ad.Variable(1)).val == 2
    assert (1 + ad.Variable(1)).der == 1
    assert (ad.Variable(1) + ad.Variable(1)).val == 2
    assert (ad.Variable(1) + ad.Variable(1)).der == 2
    assert np.array_equal((ad.Variable(np.array([1,2]), np.array([1,2])) + ad.Variable(np.array([1,2]), np.array([1,2]))).val, np.array([2, 4]))
    assert np.array_equal((ad.Variable(np.array([1,2]), np.array([1,2])) + ad.Variable(np.array([1,2]), np.array([1,2]))).der, np.array([2, 4]))

    assert np.array_equal((ad.Variable(np.array([1,2]), np.array([1,2])) + 1).val, np.array([2, 3]))
    assert np.array_equal((ad.Variable(np.array([1,2]), np.array([1,2])) + 2).der, np.array([1, 2]))
    assert np.array_equal((1 + ad.Variable(np.array([1,2]), np.array([1,2]))).val, np.array([2, 3]))
    assert np.array_equal((1 + ad.Variable(np.array([1,2]), np.array([1,2]))).der, np.array([1, 2]))

def test_sub():
    assert (ad.Variable(1) - 1).val == 0
    assert (ad.Variable(1) - 1).der == 1
    assert (ad.Variable(1) - ad.Variable(1)).val == 0
    assert (ad.Variable(1) - ad.Variable(1)).der == 0
    assert np.array_equal((ad.Variable(np.array([1,2]), np.array([3,4])) - ad.Variable(np.array([1,2]), np.array([1,2]))).val, np.array([0, 0]))
    assert np.array_equal((ad.Variable(np.array([1,2]), np.array([3,4])) - ad.Variable(np.array([1,2]), np.array([1,2]))).der, np.array([2, 2]))

    assert np.array_equal((ad.Variable(np.array([1,2]), np.array([1,2])) - 1).val, np.array([0, 1]))
    assert np.array_equal((ad.Variable(np.array([1,2]), np.array([1,2])) - 2).der, np.array([1, 2]))

def test_rsub():
    assert (1 - ad.Variable(1)).val == 0
    assert (1 - ad.Variable(1)).der == -1
    assert (ad.Variable(2) - ad.Variable(1)).val == 1
    assert (ad.Variable(2) - ad.Variable(1)).der == 0

    assert np.array_equal((1 - ad.Variable(np.array([1,2]), np.array([1,2]))).val, np.array([0, -1]))
    assert np.array_equal((1 - ad.Variable(np.array([1,2]), np.array([1,2]))).der, np.array([-1, -2]))

def test_mul():
    assert (ad.Variable(1) * 2).val == 2
    assert (ad.Variable(1) * 2).der == 2
    assert (2 * ad.Variable(1)).val == 2
    assert (2 * ad.Variable(1)).der == 2
    assert (ad.Variable(1) * ad.Variable(2)).val == 2
    assert (ad.Variable(1) * ad.Variable(2)).der == 3

    x = ad.Variable(np.array([1,2]), np.array([3,4])) * ad.Variable(np.array([2,2]), np.array([2,2]))
    assert np.array_equal(x.val, np.array([2, 4]))
    assert np.array_equal(x.der, np.array([8, 12]))

    x = ad.Variable(np.array([1,2]), np.array([6,8])) * 2
    assert np.array_equal(x.val, np.array([2, 4]))
    assert np.array_equal(x.der, np.array([12, 16]))

def test_rmul():
    assert (2 * ad.Variable(1)).val == 2
    assert (2 * ad.Variable(1)).der == 2
    assert (ad.Variable(2) * ad.Variable(1)).val == 2
    assert (ad.Variable(2) * ad.Variable(1)).der == 3

    x = ad.Variable(np.array([2,2]), np.array([2,2])) * ad.Variable(np.array([1,2]), np.array([3,4]))
    assert np.array_equal(x.val, np.array([2, 4]))
    assert np.array_equal(x.der, np.array([8, 12]))

    x = 2 * ad.Variable(np.array([1,2]), np.array([6,8]))
    assert np.array_equal(x.val, np.array([2, 4]))
    assert np.array_equal(x.der, np.array([12, 16]))

def test_truediv():
    x = ad.Variable(0)
    y = ad.Variable(2)
    with pytest.raises(ZeroDivisionError):
        y/x
    with pytest.raises(ZeroDivisionError):
        y/0
    with pytest.raises(ZeroDivisionError):
        1/x
        
    z1 = x/y
    assert z1.val == 0
    assert z1.der == (y.val*x.der - x.val*y.der)/(y.val**2)
    
    z2 = 0/y
    assert z2.val == 0
    assert z2.der == 0
    
    x = ad.Variable(1,5)
    y = ad.Variable(5,2)
        
    z1 = x/y
    assert z1.val == 1/5
    assert z1.der == (y.val*x.der - x.val*y.der)/(y.val**2)
    
    z2 = y/x
    assert z2.val == 5
    assert z2.der == pytest.approx((x.val*y.der - y.val*x.der)/(x.val**2))
    
    z3 = 3/x
    assert (3/x).val == 3.0
    assert (3/x).der == (-3*x.der)/(x.val**2)
    
    z4 = x/3 
    assert z4.val == 1/3
    assert z4.der == x.der/3

    x = ad.Variable(np.array([1,2]), np.array([3,4])) / ad.Variable(np.array([2,2]), np.array([5,6]))
    assert np.array_equal(x.val, np.array([0.5, 1]))
    assert np.array_equal(x.der, np.array([0.25, -1]))

    x = ad.Variable(np.array([1,2]), np.array([3,4])) / 2
    assert np.array_equal(x.val, np.array([0.5, 1]))
    assert np.array_equal(x.der, np.array([1.5, 2]))

    x = 2 / ad.Variable(np.array([1,2]), np.array([3,4]))
    assert np.array_equal(x.val, np.array([2, 1]))
    assert np.array_equal(x.der, np.array([-6, -2]))

    x = ad.Variable(np.array([0,2]), np.array([5,6]))
    with pytest.raises(ZeroDivisionError):
        1/x

def test_pow():
    assert (ad.Variable(1) ** 1).val == 1
    assert (ad.Variable(1) ** 1).der == 1
    with pytest.raises(ValueError):
        ad.Variable(-1)**.2
    with pytest.raises(TypeError):
        ad.Variable(1)**"abc"

    assert (ad.Variable(2, 1) ** 2).val == 2 ** 2
    assert (ad.Variable(2, 1) ** 2).der == 2 * 2 * 1
    with pytest.raises(ValueError):
        ad.Variable(-1) ** 0.2
    with pytest.raises(TypeError):
        ad.Variable(1)** "abc"
    assert (3 ** ad.Variable(2, 2)).val == 3 ** 2
    assert (3 ** ad.Variable(2, 2)).der == np.log(3) * 3 ** 2 * 2
    with pytest.raises(ValueError):
        ad.Variable(0) ** (-2)
    x = ad.Variable(2, 1)
    y = ad.Variable(3, 0)
    assert (x ** y).val == 2 ** 3
    assert (x ** y).der == 3 * 2 ** (3 - 1) * 1 + np.log(2) * 2 ** 3 * 0
    assert (y ** x).val == 3 ** 2
    assert (y ** x).der == 2 * 3 ** (2 - 1) * 0 + np.log(3) * 3 ** 2 * 1

def test_log():
    assert ad.Variable(1).log(base = np.e).val == 0
    assert ad.Variable(1).log(base = np.e).der == 1
    with pytest.raises(ValueError):
        ad.Variable(-1).log()
    with pytest.raises(ValueError):
        ad.Variable(0).log()

    x = ad.Variable(np.array([1,2]), np.array([3,4])).log()
    assert np.array_equal(x.val, np.array([np.log(1)/np.log(10), np.log(2)/np.log(10)]))
    assert np.array_equal(np.round(x.der,8), np.array([1.30288345, 0.86858896]))

    with pytest.raises(ValueError):
        ad.Variable(np.array([1,-1]), np.array([3,4])).log()
    with pytest.raises(ValueError):
        ad.Variable(np.array([0,2]), np.array([3,4])).log()
    with pytest.raises(ValueError):
        ad.Variable(np.array([1,2]), np.array([3,4])).log(base = -1)
    with pytest.raises(ValueError):
        ad.Variable(np.array([1,2]), np.array([3,4])).log(base = "Natural")

def test_sqrt():
    assert ad.Variable(1).sqrt().val == 1.0
    assert ad.Variable(1).sqrt().der == .5
    with pytest.raises(ValueError):
        ad.Variable(-1).sqrt()
    with pytest.raises(ValueError):
        ad.Variable(0).sqrt()

def test_variable_types():
    with pytest.raises(TypeError):
        assert ad.Variable('test')
    with pytest.raises(ValueError):
        assert ad.Variable(np.array(["test", "test"]), 2)
        
def test_make_variable():
    assert ad.make_variable(3, 5) == ad.Variable(3,5)
    assert ad.make_variable(3, 5).val == 3.0
    assert ad.make_variable(3, 5).der == 5.0
    assert ad.make_variable(np.array([1, 2]), 5) == ad.Variable(np.array([1, 2]), 5)
    assert ad.make_variable(np.array([1, 2]), np.array([3,3])) == ad.Variable(np.array([1, 2]), np.array([3,3]))
    
def test_make_variables():
    v = ad.make_variables([5,6,7], [1,2,3])
    with pytest.raises(ValueError):
        ad.make_variables([1, 2], [1, 0,1])
    assert v[0] == ad.Variable(5,1)
    assert v[1] == ad.Variable(6,2)
    assert v[2] == ad.Variable(7,3)

    v = ad.make_variables([np.array([3, 4]), np.array([1, 5])], [np.array([1, 2]), np.array([1, 5])])
    assert v[0] == ad.Variable(np.array([3, 4]), np.array([1, 2]))
    assert v[1] == ad.Variable(np.array([1, 5]), np.array([1, 5]))

def test_arcsin_domain():
    with pytest.raises(ValueError):
        ad.Variable(1).arcsin()
    with pytest.raises(ValueError):
        ad.Variable(-1).arcsin()
    with pytest.raises(ValueError):
        ad.Variable(np.array([2, 2]), np.array([2, 2])).arcsin()
    with pytest.raises(ValueError):
        ad.Variable(np.array([-2, 0.8]), np.array([-2, 0.7])).arcsin()
        
def test_arccos_domain():
    with pytest.raises(ValueError):
        ad.Variable(1).arccos()
    with pytest.raises(ValueError):
        ad.Variable(-1).arccos()
    with pytest.raises(ValueError):
        ad.Variable(np.array([2, 2]), np.array([2, 2])).arccos()
    with pytest.raises(ValueError):
        ad.Variable(np.array([-2, 0.8]), np.array([-2, 0.7])).arccos()
        
def test_logistic():
    x = ad.Variable(5, 1)
    f = x.logistic()
    assert f.val == 0.9933071490757153
    assert f.der == 0.006648056670790156
    
    x = ad.Variable(np.array([1, 2, 3]), np.array([1, 1, 1]))
    f = x.logistic()
    assert np.array_equal(np.round(f.val, 8), np.array([0.73105858, 0.88079708, 0.95257413]))
    assert np.array_equal(np.round(f.der, 8), np.array([0.19661193, 0.10499359, 0.04517666]))

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
    test_tanh()
    test_cosh()
    test_sinh()
    test_cos()
    test_sin()
    test_repr()
    test_logistic()