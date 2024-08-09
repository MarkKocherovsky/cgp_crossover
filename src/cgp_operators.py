import numpy as np


def add(x, y):
    return x + y


def sub(x, y):
    return x - y


def mul(x, y):
    return x * y


def div(x, y):  #analytical quotient, Ni, Drieberg, Rockett, 2013
    return x / np.sqrt(1 + y ** 2)
