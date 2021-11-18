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
def test_sincosmul_function(self):
    assert ad.Variable(1).cos().sin().val == 0.5143952585235492
    assert ad.Variable(1).cos().sin().der == -0.7216061490634433
    

