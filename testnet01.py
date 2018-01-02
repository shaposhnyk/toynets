import numpy as np
import pickle


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
        return x.reshape((x.size, 1))

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


def TestNet01My(input_shape):
    return HLNet()


def TestNet01MyReg(input_shape):
    return HLNetRegL1()


def inputLin(lo=0, hi=255, size=5):
    """ raising sequence """
    x = np.random.randint(lo, hi, size)
    x.sort()
    return x


def inputLinSwapped(lo=0, hi=255, size=5):
    """ raising sequence with 2nd and 4th el swapped """
    x = np.random.randint(lo, hi, size)
    x.sort()
    x[1], x[3] = x[3], x[1]
    return x


def preprocess_input(x):
    binX = toBinary(x)
    return binX.reshape((1, binX.shape[0], binX.shape[1]))


def decode_predictions(p):
    return [p]


def toBinary(x, digits=8):
    """ Converts array x to an array of bits of x with a given number of digits """
    masks = np.array([1 << d for d in range(0, digits)])
    x = x.reshape((len(x), 1))
    masked = x.repeat(digits, 1) & masks
    masked[masked > 0] = 1.0
    return np.array(masked, dtype=float)


if __name__ == "__main__":
    x = input();
    print "x\n", x
    print "bin\n", toBinary(x)
