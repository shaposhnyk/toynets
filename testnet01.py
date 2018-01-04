import numpy as np
import pickle
from train_ins import *
from testnet01_opts import *


class HLNet:
    def __init__(self, optimizer, H=10, D=8 * 5, O=2, resume=False):
        self.optimizer = optimizer
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

    def policy_backward(self, epx, aeph, epdlogp):
        """ backward pass. (eph is array of intermediate hidden states) """
        eph = aeph[0]
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
        return self.optimizer.train(self, xraw, y)

    def preprocess(self, x):
        binX = intToBinary(x)
        return binX.reshape((binX.size, 1))


class HLSinNet(HLNet):
    def __init__(self, optimizer, H=10, D=8 * 5, O=2, resume=False):
        HLNet.__init__(self, optimizer, H, D, O, resume)

    def preprocess(self, x):
        binX = floatToBinary(x, digits=16)
        return binX.reshape((binX.size, 1))


def decode_predictions(p):
    return [p]


if __name__ == "__main__":
    x = input();
    print "x\n", x
    print "bin\n", intToBinary(x)
