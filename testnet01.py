import numpy as np
import pickle
from train_ins import *


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))  # sigmoid "squashing" function to interval [0,1]


class HLNet:
    def __init__(self, H=10, D=8 * 5, O=2, resume=False):
        self.learning_rate = 1e-3
        if resume:
            self.model = pickle.load(open('weights/TestNet01.p', 'rb'))
        else:
            model = {}
            model['W1'] = np.random.randn(H, D) / np.sqrt(D)  # "Xavier" initialization
            model['W2'] = np.random.randn(O, H) / np.sqrt(H)
            self.model = model

    def policy_forward(self, x):
        h = np.dot(self.model['W1'], x)
        h[h < 0] = 0  # ReLU nonlinearity
        logp = np.dot(self.model['W2'], h)
        p = sigmoid(logp)
        return p, h  # return probability of taking action 2, and hidden state

    def policy_backward(self, epx, eph, epdlogp):
        """ backward pass. (eph is array of intermediate hidden states) """
        dW2 = np.dot(epdlogp, eph.T)
        dh = np.dot(self.model['W2'].T, epdlogp)
        dh[eph <= 0] = 0  # backpro prelu
        dW1 = np.dot(dh, epx.T)
        return {'W1': dW1, 'W2': dW2}

    def predict(self, xraw):
        x = self.preprocess(xraw)
        p, h = self.policy_forward(x)
        return p

    def train(self, xraw, y):
        x = self.preprocess(xraw)
        p, h = self.policy_forward(x)
        grads = self.policy_backward(x, h, y - p)

        self.update(grads)

        return p, h

    def preprocess(self, x):
        binX = intToBinary(x)
        return binX.reshape((binX.size, 1))

    def update(self, grads):
        for k, v in self.model.iteritems():
            g = grads[k]  # gradient
            self.model[k] += self.learning_rate * g / (np.sqrt(g ** 2) + 1e-7)


class HLNetRegL1(HLNet):
    def __init__(self, H=10, D=8 * 5, O=2, resume=False):
        HLNet.__init__(self, H, D, O, resume)
        self.l1lambda = self.learning_rate / 100.0

    def update(self, grads):
        for k, v in self.model.iteritems():
            g = grads[k]  # gradient
            Wk = self.model[k]
            Wk += np.sign(Wk) * self.l1lambda
            Wk += self.learning_rate * g / (np.sqrt(g ** 2) + 1e-7)


class HLSinNet(HLNetRegL1):
    def __init__(self, H=10, D=8 * 5, O=2, resume=False):
        HLNetRegL1.__init__(self, H, D, O, resume)
        self.l1lambda = self.learning_rate * 1e-4

    def preprocess(self, x):
        binX = floatToBinary(x, digits=16)
        return binX.reshape((binX.size, 1))


def LinNet01My(input_shape, h=10):
    return HLNet(D=input_shape[0] * input_shape[1])


def LinNet01MyReg(input_shape, h=10):
    return HLNetRegL1(D=input_shape[0] * input_shape[1])


def SinNet01MyReg(input_shape, h=80):
    """ 40 neurons seem sufficient for input of 20x16b, but 60 (or more) are converging faster """
    return HLSinNet(D=input_shape[0] * input_shape[1])


def decode_predictions(p):
    return [p]


if __name__ == "__main__":
    x = input();
    print "x\n", x
    print "bin\n", intToBinary(x)
