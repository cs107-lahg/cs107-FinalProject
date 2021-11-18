import pytest
import os
import sys

os.chdir(sys.path[0])
sys.path.append('../')
import lahg_ad as ad
import numpy as np

"""
This file tests complex composite functions
"""

def test_sincosmul_function():
    assert ad.Variable(1).cos().sin().val == 0.5143952585235492
    assert ad.Variable(1).cos().sin().der == -0.7216061490634433
    
def test_comp2():
    x = ad.Variable(3, 5)
    f = (x * 2) ** 3 + 5
    assert f.get_value() == 221
    assert f.get_derivative() == 1080
    
def test_comp3():
    x = ad.Variable(2, 1)
    f = (x / 3 + 2) * 5
    assert f.get_value() == (2 / 3 + 2) * 5
    assert f.get_derivative() == pytest.approx(5/3 * 1)

if __name__ == '__main__':
    test_comp2()
    test_comp3()
    test_sincosmul_function()