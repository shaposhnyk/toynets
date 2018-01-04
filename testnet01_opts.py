import numpy as np
import pickle
from testnet01 import *


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))  # sigmoid "squashing" function to interval [0,1]


class SGD:
    def __init__(self, learning_rate=1e-3):
        self.learning_rate = learning_rate
        self.count = 0

    def train(self, model, xraw, y):
        self.count += 1

        if (self.count == 150 * 1000 or self.count == 400 * 1000):
            self.learning_rate /= 10
            print("Reducing learning rate to %s" % self.learning_rate)

        x = model.preprocess(xraw)
        ph = model.policy_forward(x)
        grads = model.policy_backward(x, ph[1:], y - ph[0])

        self.update(model.model, grads)

        return ph[0], ph[1:]

    def update(self, model, grads):
        for k, v in model.iteritems():
            g = grads[k]  # gradient
            model[k] += self.learning_rate * g / (np.sqrt(g ** 2) + 1e-7)


class SGDRL1(SGD):
    def __init__(self, learning_rate=1e-3, lambda_multip=1e-2):
        SGD.__init__(self, learning_rate)
        self.m_lambda = lambda_multip

    def update(self, model, grads):
        for k, v in model.iteritems():
            g = grads[k]  # gradient
            Wk = model[k]
            Wk += np.sign(Wk) * self.learning_rate * self.m_lambda
            Wk += self.learning_rate * g / (np.sqrt(g ** 2) + 1e-7)


class SGDMB(SGD):
    def __init__(self, learning_rate=1e-3, mb_size=100, l1_lambda=1e-5):
        SGD.__init__(self, learning_rate)
        self.l1_lambda = l1_lambda
        self.size = mb_size
        self.grad_buffer = None

    def update(self, model, grad):
        if (self.grad_buffer is None):
            # update buffers that add up gradients over a batch
            self.grad_buffer = {k: np.zeros_like(v) for k, v in model.iteritems()}

        for k in model: self.grad_buffer[k] += grad[k]  # accumulate grad over batch

        if (self.count % self.size == 0):
            for k, v in model.iteritems():
                g = self.grad_buffer[k]  # gradient
                Wk = model[k]
                # Wk += np.sign(Wk) * self.l1_lambda
                Wk += self.learning_rate * g / (np.sqrt(g ** 2) + 1e-7)
                self.grad_buffer[k] = np.zeros_like(v)  # reset batch gradient buffer


if __name__ == "__main__":
    x = input();
    print "x\n", x
    print "bin\n", intToBinary(x)
