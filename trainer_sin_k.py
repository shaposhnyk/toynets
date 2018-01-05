from testnet01k import *
from train_ins import *
import numpy as np
import keras.optimizers as kop


def gen(model, itm, dg):
    count = 0

    while True:
        count += 1
        # print("here %d" % count)
        x, change = inputSinMyFq(2 * math.pi * np.random.uniform(), size=itm, digits=dg)
        y = np.zeros((1, 2))
        y[0, 0] = 1. if change > 1.05 else 0
        y[0, 1] = 1. if change < 0.95 else 0
        px = model.preprocess(x)
        yield px, y


itm, dg = 600, 16

sgd = kop.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
# model = KHLSinNet(optimizer=sgd, D=itm * dg, H=20)
model = KConNet(sgd, inputs=itm)
p = model.train(gen(model, itm, dg), 3000, 10)

for i in range(0, 5):
    x, change = inputSinMyFq(2 * math.pi * np.random.uniform(), size=itm, digits=dg)

    print("change: %s" % change)
    out1 = model.predict(x)
    print("out1: %s" % fstr(out1[0]))
