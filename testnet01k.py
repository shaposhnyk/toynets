import numpy as np
import pickle
from train_ins import *
from testnet01_opts import *
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Dropout, MaxPooling2D, Activation, Flatten
from keras import regularizers


class KHLNet:
    def __init__(self, optimizer='sgd', H=10, D=8 * 5, O=2, resume=False):
        model = Sequential()
        model.add(Dense(H, activation='relu', input_dim=D))
        model.add(Dense(H - 2, activation='relu'))
        model.add(Dense(H - 4, activation='relu'))
        model.add(Dense(O, activation='sigmoid'))

        model.compile(loss='mean_squared_error',
                      optimizer=optimizer,
                      metrics=['accuracy'])

        self.model = model

    def predict(self, xraw):
        x = self.preprocess(xraw)
        return self.model.predict(x, batch_size=1)

    def train(self, gen, steps=1024, epochs=100):
        return self.model.fit_generator(gen, steps, epochs)

    def preprocess(self, x):
        binX = intToBinary(x)
        return binX.reshape(1, binX.size)


class KHLSinNet(KHLNet):
    def __init__(self, optimizer='sgd', H=10, D=8 * 5, O=2, resume=False):
        KHLNet.__init__(self, optimizer, H, D, O, resume)

    def preprocess(self, x):
        binX = floatToBinary(x, digits=16)
        return binX.reshape(1, binX.size)


class KConNet:
    def __init__(self, optimizer, inputs=30, C=8, H=10, O=2, digits=16, resume=False):
        model = Sequential()

        reg = regularizers.l2(0.001)
        model.add(Conv2D(C, (3, 3), activation='relu',
                         input_shape=(digits, inputs, 1),
                         kernel_regularizer=reg))
        model.add(Conv2D(C, (3, 3), activation='relu',
                         kernel_regularizer=reg))

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(H, activation='relu', kernel_regularizer=reg))
        model.add(Dense(O, activation='sigmoid', kernel_regularizer=reg))

        model.compile(loss='mean_squared_error',
                      optimizer=optimizer,
                      metrics=['accuracy'])

        model.summary()

        self.model = model

    def predict(self, xraw):
        x = self.preprocess(xraw)
        return self.model.predict(x, batch_size=1)

    def train(self, gen, steps=1024, epochs=100):
        return self.model.fit_generator(gen, steps, epochs)

    def preprocess(self, x):
        binX = floatToBinary(x, digits=16)
        return binX.T.reshape(1, binX.shape[1], binX.shape[0], 1)


if __name__ == "__main__":
    x = input();
    print "x\n", x
    print "bin\n", intToBinary(x)
