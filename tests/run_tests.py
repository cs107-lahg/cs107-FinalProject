import os
import sys
os.chdir(sys.path[0])
sys.path.append('../lahg_ad/')

import fd
import numpy as np

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
    test_cos()
    test_sin()
    print(np.exp(fd.Variable(0,5)).der)
    print(fd.Variable(0,5).exp().der)