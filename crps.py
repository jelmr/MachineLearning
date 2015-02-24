import numpy as np

def heavyside(thresholds, actual):
    return thresholds >= actual

def is_cdf_valid(case):
    if case[0] < 0 or case[0] > 1:
        return False
    for i in xrange(1, len(case)):
        if case[i] > 1 or case[i] < case[i-1]:
            return False
    return True

def calc_crps(thresholds, predictions, actuals):
    #some vector algebra for speed
    obscdf = np.array([heavyside(thresholds, i) for i in actuals])
    crps = np.mean(np.mean((predictions - obscdf) ** 2))
    return crps
