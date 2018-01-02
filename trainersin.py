from testnet01 import *
from train_ins import *
import numpy as np
import sys

decay = 0.999
error = 1.0
itm, dg = 20, 16
yout = np.array([[0.], [0.]], dtype=float)

model = SinNet01MyReg(input_shape=(20, 16))  # 20 items 16 bit each

step = 0
while True:
    step += 1
    x, change = inputSinMy(2 * math.pi * np.random.uniform(), size=itm, digits=dg)
    y = np.zeros_like(yout)
    y[0, 0] = 1. if change > 1.05 else 0
    y[1, 0] = 1. if change < 0.95 else 0

    # print "training (%0.3f) on %s -> %s" % (change, x[:3], y.T)
    p, h = model.train(x, y)

    cerr = np.sum((y - p) ** 2)
    if (step % 256 == 0):
        print("[%04d] Running err: %f; Curr err: %f" % (step, error, cerr))

    if (error < 1e-7):
        w1, w2 = model.model["W1"], model.model["W2"]
        print("Finished.\nW1 is:\n%s\nW2 is:\n%s" % (w1, w2))
        print("sum(W) are: %s; %s" % (np.sum(w1 ** 2), np.sum(w2 ** 2)))
        break

    error = decay * error + (1.0 - decay) * cerr
