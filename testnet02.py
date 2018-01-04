import numpy as np
import pickle
from train_ins import *
from testnet01_opts import *


class HLNet2:
    def __init__(self, optimizer, H1=10, H2=8, D=8 * 5, O=2, resume=False):
        self.optimizer = optimizer
        if resume:
            self.model = pickle.load(open('weights/TestNet02.p', 'rb'))
        else:
            model = {}
            model['W1'] = np.random.randn(H1, D) / np.sqrt(D)  # "Xavier" initialization
            model['W2'] = np.random.randn(H2, H1) / np.sqrt(H1)
            model['W3'] = np.random.randn(O, H2) / np.sqrt(H2)
            self.model = model

    def policy_forward(self, x):
        h1 = np.dot(self.model['W1'], x)
        h1[h1 < 0] = 0  # ReLU nonlinearity
        h2 = np.dot(self.model['W2'], h1)
        h2[h2 < 0] = 0
        logp = np.dot(self.model['W3'], h2)
        p = sigmoid(logp)
        return p, h1, h2  # return probability of taking action 2, and hidden state

    def policy_backward(self, epx, aeph, epdlogp):
        """ backward pass. (eph is array of intermediate hidden states) """
        eph1, eph2 = aeph

        dW3 = np.dot(epdlogp, eph2.T)

        dh2 = np.dot(self.model['W3'].T, epdlogp)
        dh2[eph2 < 0] = 0  # backpro prelu
        dW2 = np.dot(dh2, eph1.T)

        dh1 = np.dot(self.model['W2'].T, dh2)
        dh1[eph1 < 0] = 0
        dW1 = np.dot(dh1, epx.T)

        return {'W1': dW1, 'W2': dW2, 'W3': dW3}

    def predict(self, xraw):
        x = self.preprocess(xraw)
        p, h1, h2 = self.policy_forward(x)
        return p

    def train(self, xraw, y):
        return self.optimizer.train(self, xraw, y)

    def preprocess(self, x):
        binX = intToBinary(x)
        return binX.reshape((binX.size, 1))


class HLSinNet2(HLNet2):
    def __init__(self, optimizer, H1=10, H2=8, D=8 * 5, O=2, resume=False):
        HLNet2.__init__(self, optimizer, H1, H2, D, O, resume)

    def preprocess(self, x):
        binX = floatToBinary(x, digits=16)
        return binX.reshape((binX.size, 1))


def decode_predictions(p):
    return [p]


if __name__ == "__main__":
    model = HLNet2(SGD(), H1=4, H2=3)
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([[1], [0]])
    p = model.predict(x)
    print("p" % p)

    p, h1, h2 = model.train(x, y)
    print("p" % p)
