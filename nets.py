from testnet01 import *
from testnet02 import *


def LinNet01My(input_shape, h=10):
    return HLNet(SGD(), H=h, D=input_shape[0] * input_shape[1])


def LinNet01MyReg(input_shape, h=10):
    return HLNet(SGDRL1(), H=h, D=input_shape[0] * input_shape[1])


def SinNet01MyReg(input_shape, h=10):
    """ 10 neurons are sufficient for input of 20x16b, but 20 (or more) are converging faster """
    return HLSinNet(SGDRL1(lambda_multip=1e-3), D=input_shape[0] * input_shape[1], H=h)


def LinNet02My(input_shape, h=10):
    return HLNet2(SGD(), H1=h, H2=h, D=input_shape[0] * input_shape[1])


def LinNet02MyReg(input_shape, h=10):
    return HLNet2(SGDRL1(), H1=h, H2=h, D=input_shape[0] * input_shape[1])


def SinNet02My(input_shape, h=10):
    """ 10 neurons are sufficient for input of 20x16b, but 20 (or more) are converging faster """
    return HLSinNet2(SGD(), D=input_shape[0] * input_shape[1], H1=h)
