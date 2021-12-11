#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 23:12:54 2021

@author: lingfeng
"""
import pytest
import os
import sys

os.chdir(sys.path[0])
sys.path.append("../")
import lahg_ad as ad
import numpy as np


def test_jacobian():
    x = ad.Variable(1, np.array([0, 1]))
    y = ad.Variable(2, np.array([1, 0]))
    f = [x ** 2, x + y, y ** 2]
    assert np.array_equal(ad.Vector(f).jacobian, np.array([[0, 2], [1, 1], [4, 0]]))
    assert np.array_equal(ad.Vector(f).vals, np.array([1, 3, 4]))

    f = [x ** 2, 3, y ** 2]
    with pytest.raises(Exception):
        ad.Vector(f)

    z = ad.Variable(2, np.array([1, 0, 0]))
    f = [z ** 2, x + y, x * y]
    with pytest.raises(Exception):
        ad.Vector(f)
        
def test_jacobian_repr():
    x, y = ad.make_variables([2, 1], np.eye(2))
    f = ad.Vector([x+y, x*y])
    assert f.__repr__() == "values:\n[3 2]\njacobian:\n[[1. 1.]\n [1. 2.]]"


if __name__ == "__main__":
    test_jacobian()
    test_jacobian_repr()
