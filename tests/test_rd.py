import pytest
import os
import sys

os.chdir(sys.path[0])
sys.path.append('../')
import lahg_ad as ad
import numpy as np

def test_rdsin():
    x = ad.RD(np.array([2]))
    f = x.sin()
    assert (all(np.around(x.get_derivative(), 8) == [-0.41614684]))
    assert(all(np.around(f.get_value(), 8) == [0.90929743]))


def test_rdcos():
    x = ad.RD(np.array([2]))
    f = x.cos()
    assert (all(np.around(x.get_derivative(), 8) == [-0.90929743]))
    assert all(np.around(f.get_value(), 8) == [-0.41614684])

def test_rdtan():
    x = ad.RD(np.array([2]))
    f = x.tan()
    assert (all(np.around(x.get_derivative(), 8) == [5.7743992]))
    assert all(np.around(f.get_value(), 8) == [-2.18503986])

def test_rdsinh():
    x = ad.RD(np.array([2]))
    f = x.sinh()
    assert (all(np.around(x.get_derivative(), 8) == [3.76219569]))
    assert all(np.around(f.get_value(), 8) == [3.62686041])

def test_rdcosh():
    x = ad.RD(np.array([2]))
    f = x.cosh()
    assert (all(np.around(x.get_derivative(), 8) == [3.62686041]))
    assert all(np.around(f.get_value(), 8) == [3.76219569])

def test_rdtanh():
    x = ad.RD(np.array([2]))
    f = x.tanh()
    assert (all(np.around(x.get_derivative(), 8) == [0.07065082]))
    assert all(np.around(f.get_value(), 8) == [0.96402758])

def test_rdneg():
    x = ad.RD(np.array([2]))
    f = -x
    assert (all(x.get_derivative() == [-1.]))
    assert all(f.get_value() == [-2.])

def test_rdsqrt():
    x = ad.RD(np.array([2]))
    f = x.sqrt()
    assert(all(np.around(x.get_derivative(), 8) == [0.35355339]))
    assert all(np.around(f.get_value(), 8) == [1.41421356])

def test_rdreset():
    x = ad.RD(np.array([1,2,3]))
    f = x.sqrt()
    assert(all(np.around(x.get_derivative(), 4) == [.5000,0.3536 , 0.2887]))
    x.reset()
    assert(all(x.get_derivative() == [1., 1., 1.]))

def test_rdarcsin():
    x = ad.RD(np.array([.5]))
    f = x.arcsin()
    assert (all(np.around(x.get_derivative(), 4) == [1.1547]))
    assert(all(np.around(f.get_value(), 8) == [0.52359878]))

def test_rdarccos():
    x = ad.RD(np.array([.5]))
    f = x.arccos()
    assert (all(np.around(x.get_derivative(), 4) == [-1.1547]))
    assert(all(np.around(f.get_value(), 8) == [1.04719755]))

def test_rdarctan():
    x = ad.RD(np.array([.5]))
    f = x.arctan()
    assert (all(np.around(x.get_derivative(), 1) == [.8]))
    assert(all(np.around(f.get_value(), 8) == [0.46364761]))

def test_rdlog():
    x = ad.RD(np.array([2]))
    f = x.log()
    assert (all(np.around(x.get_derivative(), 8) == [0.21714724])) # assuming base 10 log
    assert(all(np.around(f.get_value(), 8) == [0.30103])) # assuming base 10 log

    x = ad.RD(np.array([1,-2,3]))
    with pytest.raises(Exception):
            x.log()

    x = ad.RD(np.array([-1]))
    with pytest.raises(Exception):
            x.log()

def test_rdexp():
    x = ad.RD(np.array([2]))
    f = x.exp()
    assert (all(np.around(x.get_derivative(), 8) == [7.3890561]))
    assert all(np.around(f.get_value(), 8) == [7.3890561])

def test_rdlogistic():
    x = ad.RD(np.array([2]))
    f = x.logistic()
    assert (all(np.around(x.get_derivative(), 8) == [0.10499359]))
    assert all(np.around(f.get_value(), 8) == [0.88079708])

def test_RD():
    x = ad.RD(np.array([2]))
    assert all(x.val == [2,1,4])
    assert all(x.grad == [1,1,1])
    assert x.children == []
    with pytest.raises(Exception):
            x = ad.RD(np.array(['a']))


def test_rdadd():
    x = ad.RD(np.array([2]))
    f = x + 1
    assert (all(np.around(x.get_derivative(), 1) == [1.]))
    assert all(np.around(f.get_value(), 1) == [3.])
    x = ad.RD(np.array([2]))
    f = 1 + x
    assert (all(np.around(x.get_derivative(), 1) == [1.]))
    assert all(np.around(f.get_value(), 1) == [3.])
    x = ad.RD(np.array([2]))
    y = ad.RD(np.array([1]))
    f = x + y
    assert (all(np.around(x.get_derivative(), 1) == [1.]))
    assert all(np.around(f.get_value(), 1) == [3.])
    x = ad.RD(np.array([2,1]))
    y = ad.RD(np.array([1,2]))
    f = x + y
    assert (all(np.around(x.get_derivative(), 1) == [1,1]))
    assert all(np.around(f.get_value(), 1) == [3,3])
    x = ad.RD(np.array([2]))
    y = ad.RD(np.array([1,2]))
    with pytest.raises(Exception):
                x + y

def test_rdsub():
    x = ad.RD(np.array([2]))
    f = x - 1
    assert (all(np.around(x.get_derivative(), 1) == [1.]))
    assert all(np.around(f.get_value(), 1) == [1.])
    x = ad.RD(np.array([2]))
    f = 1 - x
    assert (all(np.around(x.get_derivative(), 1) == [-1.]))
    assert all(np.around(f.get_value(), 1) == [-1.])
    x = ad.RD(np.array([2]))
    y = ad.RD(np.array([1]))
    f = x - y
    assert (all(np.around(x.get_derivative(), 1) == [1.]))
    assert all(np.around(f.get_value(), 1) == [1.])
    x = ad.RD(np.array([2,1]))
    y = ad.RD(np.array([1,2]))
    f = x - y
    assert (all(np.around(x.get_derivative(), 1) == [1,1]))
    assert all(np.around(f.get_value(), 1) == [1,-1])
    x = ad.RD(np.array([2]))
    y = ad.RD(np.array([1,2]))
    with pytest.raises(Exception):
        x-y

def test_rdmul():
    x = ad.RD(np.array([2]))
    f = x * 2
    assert (all(np.around(x.get_derivative(), 1) == [2.]))
    assert all(np.around(f.get_value(), 1) == [4.])
    x = ad.RD(np.array([2]))
    f = 2 * x
    assert (all(np.around(x.get_derivative(), 1) == [2.]))
    assert all(np.around(f.get_value(), 1) == [4.])
    x = ad.RD(np.array([2]))
    y = ad.RD(np.array([1]))
    f = x * y
    assert (all(np.around(x.get_derivative(), 1) == [1.]))
    assert all(np.around(f.get_value(), 1) == [2.])
    x = ad.RD(np.array([2,1]))
    y = ad.RD(np.array([1,2]))
    f = x * y
    assert (all(np.around(x.get_derivative(), 1) == [1,2]))
    assert all(np.around(f.get_value(), 1) == [2,2])
    x = ad.RD(np.array([2]))
    y = ad.RD(np.array([1,2]))
    with pytest.raises(Exception):
        x * y

def test_rddiv():
    x = ad.RD(np.array([2]))
    f = x / 2
    assert (all(np.around(x.get_derivative(), 1) == [.5]))
    assert all(np.around(f.get_value(), 1) == [1.])
    x = ad.RD(np.array([2]))
    f = 2 / x
    assert (all(np.around(x.get_derivative(), 1) == [-.5]))
    assert all(np.around(f.get_value(), 1) == [1.])
    x = ad.RD(np.array([2]))
    y = ad.RD(np.array([1]))
    f = x / y
    assert (all(np.around(x.get_derivative(), 1) == [1.]))
    assert all(np.around(f.get_value(), 1) == [2.])
    x = ad.RD(np.array([2,1]))
    y = ad.RD(np.array([1,2]))
    f = x / y
    assert (all(np.around(x.get_derivative(), 1) == [1,.5]))
    assert all(np.around(f.get_value(), 1) == [2,.5])
    x = ad.RD(np.array([2]))
    y = ad.RD(np.array([1,2]))
    with pytest.raises(Exception):
        x / y

def test_rdpow():
    x = ad.RD(np.array([2]))
    f = x ** 2
    assert (all(np.around(x.get_derivative(), 1) == [4]))
    assert all(np.around(f.get_value(), 1) == [4.])
    x = ad.RD(np.array([2]))
    f = 2 ** x
    assert (all(np.around(x.get_derivative(), 8) == [2.77258872]))
    assert all(np.around(f.get_value(), 1) == [4.])
    x = ad.RD(np.array([2]))
    y = ad.RD(np.array([1]))
    f = x ** y
    assert (all(np.around(x.get_derivative(), 1) == [1.]))
    assert all(np.around(f.get_value(), 1) == [2.])
    x = ad.RD(np.array([2,1]))
    y = ad.RD(np.array([1,2]))
    f = x ** y
    assert (all(np.around(x.get_derivative(), 1) == [1,2]))
    assert all(np.around(f.get_value(), 1) == [2,1])
    x = ad.RD(np.array([2]))
    y = ad.RD(np.array([1,2]))
    with pytest.raises(Exception):
        x ** y

def test_rdeq():
    x = ad.RD(np.array([1,2]))
    assert (x == 1) == False
    x = ad.RD(np.array([1,2]))
    y = ad.RD(np.array([1,2]))
    assert (x == y)

def test_rdne():
    x = ad.RD(np.array([1,2]))
    y = ad.RD(np.array([3,3]))
    assert (x != y)

if __name__ == '__main__':
    test_rdsin()
    test_rdcos()
    test_rdtan()
    test_rdsinh()
    test_rdcosh()
    test_rdtanh()
    test_rdneg()
    test_rdsqrt()
    test_rdreset()
    test_rdarcsin()
    test_rdarccos()
    test_rdarctan()
    test_rdlog()
    test_rdexp()
    test_rdlogistic()
    test_RD()
    test_rdadd()
    test_rdsub()
    test_rdmul()
    test_rddiv()
    test_rdpow()
    test_rdeq()
    test_rdne()






    






