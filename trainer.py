from nets import *
from train_ins import *
import numpy as np
import sys
import datetime as dt

decay = 0.999
error, lerr = 1.0, 1.0
onezero = np.array([[1.], [0.]], dtype=float)
zeroone = np.array([[0.], [0.]], dtype=float)

model = LinNet02My(input_shape=(5, 8))

step, st = 0, dt.datetime.now()
while True:
    step += 1
    x = inputLinSwapped()
    x, y = (x, onezero) if np.random.uniform() < 0.5 else (x[::-1], zeroone)

    # print "training on %s -> %s" % (x, y)
    p, h = model.train(x, y)

    cerr = np.sum((y - p) ** 2)
    lerr = cerr if cerr > 1e-5 else lerr
    if (step % 512 == 0):
        print("[%04d] Running err: %f; CErr: %f; LErr: %f" % (step, error, cerr, lerr))

    if (error < 1e-7):
        wi = model.model.values()
        print "Finished in %s" % (dt.datetime.now() - st)
        # print("W1 is:\n%s\n" % (w1))
        print("W[-1] is:\n%s" % (model.model["W" + str(len(wi))]))
        print("sum(W) are: %s" % ([int(np.sum(w ** 2)) for w in wi]))
        break

    error = decay * error + (1.0 - decay) * cerr
