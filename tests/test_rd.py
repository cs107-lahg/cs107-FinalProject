import pytest
import os
import sys

os.chdir(sys.path[0])
sys.path.append("../")
import lahg_ad as ad
import numpy as np


def test_rdsin():
    x = ad.RD(np.array([2]))
    f = x.sin()
    assert all(np.around(x.get_derivative(), 8) == [-0.41614684])
    assert all(np.around(f.get_value(), 8) == [0.90929743])

    x = ad.RD(np.array([1, 2, 3]))
    f = x.sin()
    assert np.array_equal(
        np.round(f.get_value(), 8), np.array([0.84147098, 0.90929743, 0.14112001])
    )
    assert np.array_equal(
        np.round(x.get_derivative(), 8), np.array([0.54030231, -0.41614684, -0.9899925])
    )


def test_rdcos():
    x = ad.RD(np.array([2]))
    f = x.cos()
    assert all(np.around(x.get_derivative(), 8) == [-0.90929743])
    assert all(np.around(f.get_value(), 8) == [-0.41614684])

    x = ad.RD(np.array([1, 2, 3]))
    f = x.cos()
    assert np.array_equal(
        np.round(f.get_value(), 8), np.array([0.54030231, -0.41614684, -0.9899925])
    )
    assert np.array_equal(
        np.round(x.get_derivative(), 8),
        np.array([-0.84147098, -0.90929743, -0.14112001]),
    )


def test_rdtan():
    x = ad.RD(np.array([2]))
    f = x.tan()
    assert all(np.around(x.get_derivative(), 8) == [5.7743992])
    assert all(np.around(f.get_value(), 8) == [-2.18503986])

    x = ad.RD(np.array([1, 2, 3]))
    f = x.tan()
    assert np.array_equal(
        np.round(f.get_value(), 8), np.array([1.55740772, -2.18503986, -0.14254654])
    )
    assert np.array_equal(
        np.round(x.get_derivative(), 8), np.array([3.42551882, 5.7743992, 1.02031952])
    )


def test_rdsinh():
    x = ad.RD(np.array([2]))
    f = x.sinh()
    assert all(np.around(x.get_derivative(), 8) == [3.76219569])
    assert all(np.around(f.get_value(), 8) == [3.62686041])

    x = ad.RD(np.array([1, 2, 3]))
    f = x.sinh()
    assert np.array_equal(
        np.round(f.get_value(), 8), np.array([1.17520119, 3.62686041, 10.01787493])
    )
    assert np.array_equal(
        np.round(x.get_derivative(), 8), np.array([1.54308063, 3.76219569, 10.067662])
    )


def test_rdcosh():
    x = ad.RD(np.array([2]))
    f = x.cosh()
    assert all(np.around(x.get_derivative(), 8) == [3.62686041])
    assert all(np.around(f.get_value(), 8) == [3.76219569])

    x = ad.RD(np.array([1, 2, 3]))
    f = x.cosh()
    assert np.array_equal(
        np.round(f.get_value(), 8), np.array([1.54308063, 3.76219569, 10.067662])
    )
    assert np.array_equal(
        np.round(x.get_derivative(), 8), np.array([1.17520119, 3.62686041, 10.01787493])
    )


def test_rdtanh():
    x = ad.RD(np.array([2]))
    f = x.tanh()
    assert all(np.around(x.get_derivative(), 8) == [0.07065082])
    assert all(np.around(f.get_value(), 8) == [0.96402758])

    x = ad.RD(np.array([1, 2, 3]))
    f = x.tanh()
    assert np.array_equal(
        np.round(f.get_value(), 8), np.array([0.76159416, 0.96402758, 0.99505475])
    )
    assert np.array_equal(
        np.round(x.get_derivative(), 8), np.array([0.41997434, 0.07065082, 0.00986604])
    )


def test_rdneg():
    x = ad.RD(np.array([2]))
    f = -x
    assert all(x.get_derivative() == [-1.0])
    assert all(f.get_value() == [-2.0])

    x = ad.RD(np.array([1, 2, 3]))
    f = -x
    assert np.array_equal(np.round(f.get_value(), 8), np.array([-1, -2, -3]))
    assert np.array_equal(np.round(x.get_derivative(), 8), np.array([-1, -1, -1]))


def test_rdsqrt():
    x = ad.RD(np.array([2]))
    f = x.sqrt()
    assert all(np.around(x.get_derivative(), 8) == [0.35355339])
    assert all(np.around(f.get_value(), 8) == [1.41421356])

    x = ad.RD(np.array([1, 2, 3]))
    f = x.sqrt()
    assert np.array_equal(
        np.round(f.get_value(), 8), np.array([1.0, 1.41421356, 1.73205081])
    )
    assert np.array_equal(
        np.round(x.get_derivative(), 8), np.array([0.5, 0.35355339, 0.28867513])
    )


def test_rdreset():
    x = ad.RD(np.array([1, 2, 3]))
    f = x.sqrt()
    assert all(np.around(x.get_derivative(), 4) == [0.5000, 0.3536, 0.2887])
    x.reset()
    assert all(x.get_derivative() == [1.0, 1.0, 1.0])


def test_rdarcsin():
    x = ad.RD(np.array([0.5]))
    f = x.arcsin()
    assert all(np.around(x.get_derivative(), 4) == [1.1547])
    assert all(np.around(f.get_value(), 8) == [0.52359878])

    x = ad.RD(np.array([0.1, 0.2, 0.3]))
    f = x.arcsin()
    assert np.array_equal(
        np.round(f.get_value(), 8), np.array([0.10016742, 0.20135792, 0.30469265])
    )
    assert np.array_equal(
        np.round(x.get_derivative(), 8), np.array([1.00503782, 1.02062073, 1.04828484])
    )


def test_rdarccos():
    x = ad.RD(np.array([0.5]))
    f = x.arccos()
    assert all(np.around(x.get_derivative(), 4) == [-1.1547])
    assert all(np.around(f.get_value(), 8) == [1.04719755])

    x = ad.RD(np.array([0.1, 0.2, 0.3]))
    f = x.arccos()
    assert np.array_equal(
        np.round(f.get_value(), 8), np.array([1.47062891, 1.36943841, 1.26610367])
    )
    assert np.array_equal(
        np.round(x.get_derivative(), 8),
        np.array([-1.00503782, -1.02062073, -1.04828484]),
    )


def test_rdarctan():
    x = ad.RD(np.array([0.5]))
    f = x.arctan()
    assert all(np.around(x.get_derivative(), 1) == [0.8])
    assert all(np.around(f.get_value(), 8) == [0.46364761])

    x = ad.RD(np.array([0.1, 0.2, 0.3]))
    f = x.arctan()
    assert np.array_equal(
        np.round(f.get_value(), 8), np.array([0.09966865, 0.19739556, 0.29145679])
    )
    assert np.array_equal(
        np.round(x.get_derivative(), 8), np.array([0.99009901, 0.96153846, 0.91743119])
    )


def test_rdlog():
    x = ad.RD(np.array([2]))
    f = x.log()
    assert all(np.around(x.get_derivative(), 8) == [0.21714724])  # assuming base 10 log
    assert all(np.around(f.get_value(), 8) == [0.30103])  # assuming base 10 log

    x = ad.RD(np.array([1, 2, 3]))
    f = x.log(base=np.e)
    assert np.array_equal(
        np.round(f.get_value(), 8), np.array([0.0, 0.69314718, 1.09861229])
    )
    assert np.array_equal(
        np.round(x.get_derivative(), 8), np.array([1.0, 0.5, 0.33333333])
    )

    x = ad.RD(np.array([1, 2, 3]))
    with pytest.raises(Exception):
        x.log(base=-1)

    with pytest.raises(Exception):
        x.log(base="Natural")

    x = ad.RD(np.array([1, -2, 3]))
    with pytest.raises(Exception):
        x.log()

    x = ad.RD(np.array([-1]))
    with pytest.raises(Exception):
        x.log()


def test_rdexp():
    x = ad.RD(np.array([2]))
    f = x.exp()
    assert all(np.around(x.get_derivative(), 8) == [7.3890561])
    assert all(np.around(f.get_value(), 8) == [7.3890561])

    x = ad.RD(np.array([1, 2, 3]))
    f = x.exp()
    assert np.array_equal(
        np.round(f.get_value(), 8), np.array([2.71828183, 7.3890561, 20.08553692])
    )
    assert np.array_equal(
        np.round(x.get_derivative(), 8), np.array([2.71828183, 7.3890561, 20.08553692])
    )
    with pytest.raises(Exception):
        x.exp(base="Natural")


def test_rdlogistic():
    x = ad.RD(np.array([2]))
    f = x.logistic()
    assert all(np.around(x.get_derivative(), 8) == [0.10499359])
    assert all(np.around(f.get_value(), 8) == [0.88079708])

    x = ad.RD(np.array([1, 2, 3]))
    f = x.logistic()
    assert np.array_equal(
        np.round(f.get_value(), 8), np.array([0.73105858, 0.88079708, 0.95257413])
    )
    assert np.array_equal(
        np.round(x.get_derivative(), 8), np.array([0.19661193, 0.10499359, 0.04517666])
    )


def test_RD():
    x = ad.RD(np.array([2, 1, 4]))
    assert all(x.val == [2, 1, 4])
    assert all(x.grad == [1, 1, 1])
    assert x.children == []
    with pytest.raises(Exception):
        x = ad.RD(np.array(["a"]))


def test_rdadd():
    x = ad.RD(np.array([2]))
    f = x + 1
    assert all(np.around(x.get_derivative(), 1) == [1.0])
    assert all(np.around(f.get_value(), 1) == [3.0])
    x = ad.RD(np.array([2]))
    f = 1 + x
    assert all(np.around(x.get_derivative(), 1) == [1.0])
    assert all(np.around(f.get_value(), 1) == [3.0])
    x = ad.RD(np.array([2]))
    y = ad.RD(np.array([1]))
    f = x + y
    assert all(np.around(x.get_derivative(), 1) == [1.0])
    assert all(np.around(f.get_value(), 1) == [3.0])
    x = ad.RD(np.array([2, 1]))
    y = ad.RD(np.array([1, 2]))
    f = x + y
    assert all(np.around(x.get_derivative(), 1) == [1, 1])
    assert all(np.around(y.get_derivative(), 1) == [1, 1])
    assert all(np.around(f.get_value(), 1) == [3, 3])
    x = ad.RD(np.array([2]))
    y = ad.RD(np.array([1, 2]))
    with pytest.raises(Exception):
        x + y


def test_rdsub():
    x = ad.RD(np.array([2]))
    f = x - 1
    assert all(np.around(x.get_derivative(), 1) == [1.0])
    assert all(np.around(f.get_value(), 1) == [1.0])
    x = ad.RD(np.array([2]))
    f = 1 - x
    assert all(np.around(x.get_derivative(), 1) == [-1.0])
    assert all(np.around(f.get_value(), 1) == [-1.0])
    x = ad.RD(np.array([2]))
    y = ad.RD(np.array([1]))
    f = x - y
    assert all(np.around(x.get_derivative(), 1) == [1.0])
    assert all(np.around(f.get_value(), 1) == [1.0])
    x = ad.RD(np.array([2, 1]))
    y = ad.RD(np.array([1, 2]))
    f = x - y
    assert all(np.around(x.get_derivative(), 1) == [1, 1])
    assert all(np.around(y.get_derivative(), 1) == [-1, -1])
    assert all(np.around(f.get_value(), 1) == [1, -1])
    x = ad.RD(np.array([2]))
    y = ad.RD(np.array([1, 2]))
    with pytest.raises(Exception):
        x - y


def test_rdmul():
    x = ad.RD(np.array([2]))
    f = x * 2
    assert all(np.around(x.get_derivative(), 1) == [2.0])
    assert all(np.around(f.get_value(), 1) == [4.0])
    x = ad.RD(np.array([2]))
    f = 2 * x
    assert all(np.around(x.get_derivative(), 1) == [2.0])
    assert all(np.around(f.get_value(), 1) == [4.0])
    x = ad.RD(np.array([2]))
    y = ad.RD(np.array([1]))
    f = x * y
    assert all(np.around(x.get_derivative(), 1) == [1.0])
    assert all(np.around(f.get_value(), 1) == [2.0])
    x = ad.RD(np.array([2, 1]))
    y = ad.RD(np.array([1, 2]))
    f = x * y
    assert all(np.around(x.get_derivative(), 1) == [1, 2])
    assert all(np.around(y.get_derivative(), 1) == [2, 1])
    assert all(np.around(f.get_value(), 1) == [2, 2])
    x = ad.RD(np.array([2]))
    y = ad.RD(np.array([1, 2]))
    with pytest.raises(Exception):
        x * y


def test_rddiv():
    x = ad.RD(np.array([2]))
    f = x / 2
    assert all(np.around(x.get_derivative(), 1) == [0.5])
    assert all(np.around(f.get_value(), 1) == [1.0])
    x = ad.RD(np.array([2]))
    f = 2 / x
    assert all(np.around(x.get_derivative(), 1) == [-0.5])
    assert all(np.around(f.get_value(), 1) == [1.0])
    x = ad.RD(np.array([2]))
    y = ad.RD(np.array([1]))
    f = x / y
    assert all(np.around(x.get_derivative(), 1) == [1.0])
    assert all(np.around(f.get_value(), 1) == [2.0])
    x = ad.RD(np.array([2, 1]))
    y = ad.RD(np.array([1, 2]))
    f = x / y
    assert all(np.around(x.get_derivative(), 1) == [1, 0.5])
    assert all(np.around(f.get_value(), 1) == [2, 0.5])
    x = ad.RD(np.array([2]))
    y = ad.RD(np.array([1, 2]))
    with pytest.raises(Exception):
        x / y


def test_rdpow():
    x = ad.RD(np.array([2]))
    f = x ** 2
    assert all(np.around(x.get_derivative(), 1) == [4])
    assert all(np.around(f.get_value(), 1) == [4.0])
    x = ad.RD(np.array([2]))
    f = 2 ** x
    assert all(np.around(x.get_derivative(), 8) == [2.77258872])
    assert all(np.around(f.get_value(), 1) == [4.0])
    x = ad.RD(np.array([2]))
    y = ad.RD(np.array([1]))
    f = x ** y
    assert all(np.around(x.get_derivative(), 1) == [1.0])
    assert all(np.around(f.get_value(), 1) == [2.0])
    x = ad.RD(np.array([2, 1]))
    y = ad.RD(np.array([1, 2]))
    f = x ** y
    assert all(np.around(x.get_derivative(), 1) == [1, 2])
    assert all(np.around(f.get_value(), 1) == [2, 1])
    x = ad.RD(np.array([2]))
    y = ad.RD(np.array([1, 2]))
    with pytest.raises(Exception):
        x ** y


def test_rdeq():
    x = ad.RD(np.array([1, 2]))
    assert (x == 1) == False
    x = ad.RD(np.array([1, 2]))
    y = ad.RD(np.array([1, 2]))
    assert x == y


def test_rdne():
    x = ad.RD(np.array([1, 2]))
    y = ad.RD(np.array([3, 3]))
    assert x != y


def test_rdrepr():
    x = ad.RD(np.array([2]))
    assert x.__repr__() == "value = [2], derivative = [1.]"


if __name__ == "__main__":
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
    test_rdrepr()
