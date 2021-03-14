import numpy as np


def pctChange(values):
    changes = values.pct_change()
    changes.iloc[0] = 0
    return changes


# Relative strength index
def rsiFunc(prices, n=14):
    deltas = np.diff(prices)
    seed = deltas[:n + 1]
    up = seed[seed >= 0].sum() / n
    down = -seed[seed < 0].sum() / n
    rs = up / down
    rsi = np.zeros_like(prices)
    rsi[:n] = 100. - 100. / (1. + rs)

    for i in range(n, len(prices)):
        delta = deltas[i - 1]  # cause the diff is 1 shorter

        if delta > 0:
            up_val = delta
            down_val = 0.
        else:
            up_val = 0.
            down_val = -delta

        up = (up * (n - 1) + up_val) / n
        down = (down * (n - 1) + down_val) / n

        rs = up / down
        rsi[i] = 100. - 100. / (1. + rs)

    return rsi


def movingAverage(values, window):
    weights = np.repeat(1.0, window) / window
    smas = np.convolve(values, weights, 'valid')
    return smas  # as a numpy array


def expMovingAverage(values, window):
    weights = np.exp(np.linspace(-1., 0., window))
    weights /= weights.sum()
    a = np.convolve(values, weights, mode='full')[:len(values)]
    a[:window] = a[window]
    return a


def computeMACD(x, slow=26, fast=12):
    if slow >= len(x):
        slow = len(x) - 1
        fast = abs(slow / 2) - 1

    if fast <= 5:
        return 0, 0, np.zeros(len(x))

    emaslow = expMovingAverage(x, slow)
    emafast = expMovingAverage(x, fast)
    return emaslow, emafast, emafast - emaslow
