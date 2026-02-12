import numpy as np
import os
import math
import matplotlib.pyplot as plt
import pandas as pd
import scipy
from scipy import stats
from collections import OrderedDict

def plot_2D_contour(arr1,arr2,fac=1,CL=[0.68],color=None,ax=None,linestyles=['-','-.','--'],
                    nbins=70,withSave=False,weights=None,extent=None,**kwargs):
    '''

    Plot joint probability distribution as contours. Consecutive calls
    to this function will overplot the contours.

    Inputs:
    arr1  - input array1 
    arr2  - input array2

    
    Keywords:
    CL - list of confidence levels to plot for the contours
    extent - [x_min, x_max, y_min, y_max] - sometimes the points are over a 
             very large range but the region of interest is smaller. (Default: None)
    
    '''
    if type(CL) is not np.array:
        CL=np.array(CL)

    if type(linestyles) is not list:
        linestyles = [linestyles]*np.size(CL)
    if len(linestyles) < np.size(CL):
        linestyles = linestyles*int(math.ceil(np.size(CL)/(1.*len(linestyles))))

    ls = np.array(linestyles[0:np.size(CL)])
    #the confidence levels need to be sorted from larger to smaller for contour
    #the corresponding linestyles are sorted as well.
    idx = np.argsort(-CL)

    CL  = CL[idx]
    ls = ls[idx]

    if color is not None:
        if type(color) is not list:
            color = [color]*np.size(CL)
        if len(linestyles) < np.size(CL):
            color = color*int(math.ceil(np.size(CL)/(1.*len(color))))
        color =np.array(color)
        color=color[idx]


    if type(fac) is not list:
        fac=[fac,fac]
    elif len(fac)==1:
        fac=[fac[0]]*2

    if extent is not None:
        good = np.where((arr1 >= extent[0]) & (arr1 <= extent[1]) & (arr2 >= extent[2]) & (arr2 <= extent[3]) )[0]
        arr1_ = arr1[good]
        arr2_ = arr2[good]
        if weights is not None:
            weights_ = weights[good]
        else:
            weights_ = None
    else:
        arr1_ = arr1
        arr2_ = arr2
        if weights is not None:
            weights_ = weights
        else:
            weights_ = None

        
        
    (hist, obins, ibins) = np.histogram2d(arr2_*fac[1],arr1_*fac[0], bins = nbins, weights = weights_)

    levels = getContourLevels(hist,percents=CL)
    if ax is None:
        ax = plt.gca()
    
    if color is None:
        c = ax.contour(hist, levels,origin=None,extent = [ibins[0], ibins[-1], obins[0], obins[-1]],linestyles=ls,**kwargs)
    else:
        c = ax.contour(hist, levels,origin=None,extent = [ibins[0], ibins[-1], obins[0], obins[-1]],colors=color,linestyles=ls,**kwargs)

def return_2D_contour_levels(arr1, arr2, weights = None, extent = None, nbins = 70, fac = 1):
    '''
    Get hist and bins with which to plot contours for two parameters

    Inputs:
    arr1  - input array1 
    arr2  - input array2

    
    Keywords:
    weights - array of weights
    extent - [x_min, x_max, y_min, y_max] - sometimes the points are over a 
             very large range but the region of interest is smaller. (Default: None)
             Warning: if you set the extent and calculate dkl from this, it will affect calculation.
             In that case, better to just impose limits on the plot later rather than on the hist.
    nbins - number of bins
    fac -   optional factor by which to multiply the arrays (e.g. to convert units if desired). (Default: 1, If no conversion required). 
    
    '''
    
    if extent is not None:
        good = np.where((arr1 >= extent[0]) & (arr1 <= extent[1]) & (arr2 >= extent[2]) & (arr2 <= extent[3]) )[0]
        arr1_ = arr1[good]
        arr2_ = arr2[good]
        if weights is not None:
            weights_ = weights[good]
        else:
            weights_ = None
    else:
        arr1_ = arr1
        arr2_ = arr2
        if weights is not None:
            weights_ = weights
        else:
            weights_ = None

    if type(fac) is not list:
        fac=[fac,fac]
    elif len(fac)==1:
        fac=[fac[0]]*2
        
    (hist, bins1, bins2) = np.histogram2d(arr2_*fac[1],arr1_*fac[0], bins = nbins, weights = weights_)

    return hist,bins1,bins2
    

def calculate_sigmas(chains, sigma, weights = None):
    # return the central confidence interval for the chain given a sigma
    percents = np.array([stats.norm.cdf(-sigma),stats.norm.cdf(sigma)])

    return weighted_percentile(chains,percents,weights=weights)

def calculate_sigmas_table(tab, sigma, weights=None):
    # given a pandas table, calculate and return a dictionary of
    # sigmas for each of the columns of the table
    sigma_dict = []
    percents = np.array([stats.norm.cdf(-sigma),stats.norm.cdf(sigma)])
    
    for param_name in tab.columns:
        sig = weighted_percentile(tab[param_name],percents,weights=weights)

        sigma_dict.append((param_name,(sig[0],sig[1])))
    return OrderedDict(sigma_dict)

def calculate_median(chains, weights = None):
    # return the weighted median of the chain
    return weighted_percentile(chains,0.5,weights=weights)

def calculate_median_table(tab, weights = None):
    # return the weighted median of the chain
    sigma_dict = []
    for param_name in tab.columns:
        med = weighted_percentile(tab[param_name],0.5,weights=weights)
        sigma_dict.append((param_name,med))
    return OrderedDict(sigma_dict)

def weighted_percentile(data, percents, weights=None):
    ''' percents
        weights specifies the frequency (count) of data.
        '''
    if weights is None:
        return np.percentile(data, 100*percents)
    ind=np.argsort(data)
    d=data[ind]
    w=weights[ind]
    p=1.*w.cumsum()/w.sum()
    y=np.interp(percents, p, d)

    return y

def weighted_avg(values, weights):
    """
        Returns the weighted average and standard deviation.
        
        values, weights -- Numpy ndarrays with the same shape.
        """
    
    average = np.average(values, weights=weights)
    variance = np.dot(weights, (values-average)**2)/weights.sum()
    
    return average, math.sqrt(variance)

def getContourLevels(probDist,percents = np.array([0.6827, .95, .997])):
    """
        If we want to overlay countours, we need to figure out the
        appropriate levels. The algorithim is:
        1. Sort all pixels in the 2D histogram (largest to smallest)
        2. Make a cumulative distribution function
        3. Find the level at which 68% of trials are enclosed.
        """
    # Get indices for sorted pixel values (smallest to largest)
    sid0 = probDist.flatten().argsort()
    # Reverse indices, now largest to smallest
    sid = sid0[::-1]
    # Sort the actual pixel values
    pixSort = probDist.flatten()[sid]
    
    # Make a cumulative distribution function starting from the
    # highest pixel value. This way we can find the level above
    # which 68% of the trials will fall.
    cdf = np.cumsum(pixSort)
    cdf = cdf/max(cdf)

    levels = np.interp(percents,cdf,pixSort)
    # Determine point at which we reach 68% level
    
    #levels = np.zeros(len(percents), dtype=float)
    #for ii in range(len(levels)):
        # Get the index of the pixel at which the CDF
        # reaches this percentage (the first one found)
    #    idx = (np.where(cdf < percents[ii]))[0]
        
        # Now get the level of that pixel
    #    levels[ii] = pixSort[idx[-1]]
    return np.array(levels)
