from nets import *
from train_ins import *
import numpy as np
import datetime as dt

decay = 0.999
error, lerr = 1.0, 1.0
itm, dg = 40, 16
yout = np.array([[0.], [0.]], dtype=float)

model = SinNet02My(input_shape=(itm, dg), h=20)  # 30 items 16 bit each

# SinNet01MyReg(input_shape=(30, 16), h)
# h = 10 - 3-4% after 500k iters
# h = 20 - 2-3% after 500k iters
# h = 30 - 1-1.5% after 500k iters
# h = 500 - 0.2-0.5% after 900k iters


step, st = 0, dt.datetime.now()
while True:
    step += 1
    x, change = inputSinMyFq(2 * math.pi * np.random.uniform(), size=itm, digits=dg)
    y = np.zeros_like(yout)
    y[0, 0] = 1. if change > 1.05 else 0
    y[1, 0] = 1. if change < 0.95 else 0

    # print "training (%0.3f) on %s -> %s" % (change, x[:3], y.T)
    p, h = model.train(x, y)

    cerr = np.sum((y - p) ** 2)
    lerr = cerr if cerr > 1e-5 else lerr
    if (step % 1024 == 0):
        print("[%04d] Running err: %f; CErr: %f; LErr: %f" % (step, error, cerr, lerr))

    if (error < 1e-7):
        wi = model.model.values()
        print "Finished in %s" % (dt.datetime.now() - st)
        # print("W1 is:\n%s\n" % (w1))
        print("W[-1] is:\n%s" % (model.model["W" + str(len(wi))]))
        print("sum(W) are: %s" % ([int(np.sum(w ** 2)) for w in wi]))
        break

    error = decay * error + (1.0 - decay) * cerr
