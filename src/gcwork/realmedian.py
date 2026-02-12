import os
import sys
import math
import numpy as np

def realMedian(arr, index = False):
    """Takes a 1D array and returns the median value. For the odd case
    will return the middle value, for the ven case will return the
    higher of the two values at the center. This function is different
    than the numpy.median function in that it will not average the two
    middle values.
    """

    # sort the array
    s = np.argsort(arr)
    ind = s[np.ceil((len(arr)-1)/2.0)]
    medValue = arr[ind]

    if index:
        return ind
    else:
        return medValue
