import numpy as np
import math


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


def inputSinMy(phi0, size=10, parts=16, shift=math.pi / 2, digits=16):
    phi = math.pi / parts
    high = 2 ** (digits - 2)
    x0 = inputSinRaw(phi, phi0=phi0, size=size)
    x1 = inputSinRaw(phi, phi0=phi0 + shift, size=size)
    x0sum, x1sum = np.sum(x0), np.sum(x1)
    # print ("x0", fstr(x0), x0sum)
    # print ("x1", fstr(x1), x1sum)
    return [x0 * high + 2 * high, (x1sum + 2 * size) / (x0sum + 2 * size)]


def inputSinRaw(interval, phi0=0.0, high=1, size=10):
    x = np.arange(phi0, interval + phi0 - 1e-7, interval / size)
    return np.sin(x) if high == 1 else np.sin(x) * high


def intToBinary(x, digits=8):
    """ Converts array x to an array of bits of x with a given number of digits """
    masks = np.array([1 << d for d in range(digits - 1, -1, -1)])
    x = x.reshape((len(x), 1))
    masked = x.repeat(digits, 1) & masks
    masked[masked > 0] = 1.0
    return np.array(masked, dtype=float)


def floatToBinary(x, digits=8):
    """ converts array of floats to binary. fractional part is discarded. with a given number of digits """
    return intToBinary(np.array(x, dtype=int), digits)


def fstr(x):
    return [("%0.5f" % xi) for xi in x]


if __name__ == "__main__":
    print(intToBinary(np.array([255, 128, 4])))
    print(intToBinary(np.array([-127, 127, 4])))

    interval = math.pi
    x = inputSinRaw(interval)
    print(fstr(x))
    y = inputSinRaw(interval, phi0=interval)
    print(fstr(y))

    dg = 8
    z = inputSinRaw(interval / 2, size=3, high=2 ** dg)
    print(fstr(z))
    print(floatToBinary(z, dg))

    z2 = inputSinRaw(interval / 2, phi0=interval, size=3, high=2 ** dg)
    print(fstr(z2))
    print(floatToBinary(z2, dg))

    print inputSinMy(math.pi * np.random.uniform())

    for i in range(0, 64):
        print i, inputSinMy(math.pi * i / 16)[1]
