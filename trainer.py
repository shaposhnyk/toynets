from testnet01 import *
import numpy as np
import sys

decay = 0.99
error = 1.0
onezero = np.array([[1.], [0.]], dtype=float)
zeroone = np.array([[0.], [0.]], dtype=float)

model = TestNet01MyReg(input_shape=(5, 8))

step = 0
while True:
    step += 1
    x = inputLinSwapped()
    x, y = (x, onezero) if np.random.uniform() < 0.5 else (x[::-1], zeroone)

    # print "training on %s -> %s" % (x, y)
    x = preprocess_input(x)
    p, h = model.train(x, y)

    cerr = np.sum((y - p) ** 2)
    if (step % 10 == 0):
        print("[%04d] Running err: %f; Curr err: %f" % (step, error, cerr))

    if (error < 1e-7):
        w1, w2 = model.model["W1"], model.model["W2"]
        print("Finished.\nW1 is:\n%s\nW2 is:\n%s" % (w1, w2))
        print("sum(W) are: %s; %s" % (np.sum(w1 ** 2), np.sum(w2 ** 2)))
        break

    error = decay * error + (1.0 - decay) * cerr
