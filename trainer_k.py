from testnet01k import *
from train_ins import *
import numpy as np


def gen(model):
    onezero = np.array([[1.], [0.]], dtype=float)
    zeroone = np.array([[0.], [0.]], dtype=float)

    while True:
        x = inputLinSwapped()
        x, y = (x, onezero) if np.random.uniform() < 0.5 else (x[::-1], zeroone)
        px = model.preprocess(x)
        py = y.reshape(1, y.size)
        yield px, py


model = newKHLNet(input_shape=(5, 8))
p = model.train(gen(model), 1000, 3)
