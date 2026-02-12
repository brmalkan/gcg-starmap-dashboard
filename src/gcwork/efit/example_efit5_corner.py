import numpy as np
from gcwork.efit import efit5_results
import matplotlib
import pylab as plt
from corner import corner
import scipy
from astropy.time import Time
import os
from scipy.stats import chi2
from collections import OrderedDict
import pandas as pd

font = {        'size'   : 20}
matplotlib.rc('font', **font)

# In this example, we will use the efit5_results module to load the results of an efit run.

orbit_dir = '/u/shoko/align/2024_02_14/orbits/' # directory where the efit5 input files are located
plot_dir = './'   # to be the directory where you want to output the plots

orbit_run = os.path.join(orbit_dir,'orbit.S0-20')
savefile = './orbit_'+orbit_run.split('/')[-1]

k = efit5_results.efit5(orbit_run, withInt=False)

# print the mean of the posteriors
print(k.BH.mean)

## Triangle plot of global parameters

#check to see if redshift is a fit parameter
if k.BH.mean['Redshift'][1] > 0:
    arr = np.array([k.BH.chain_glob['GM']/1e6,k.BH.chain_glob['R0']/1e3,k.BH.chain_glob['x0']*1e3,
                    k.BH.chain_glob['y0']*1e3,k.BH.chain_glob['vx0']*1e3,
                    k.BH.chain_glob['vy0']*1e3,k.BH.chain_glob['vz0'],
                    k.BH.chain_glob['Redshift']])
    arr = arr.transpose()
    plt.figure(figsize=(12,12))
    crange = [0.999,0.999,0.999,0.999,0.999,0.999,0.999,0.999]
    corner(arr,labels=['GM','R0','x0','y0','vx0','vy0','vz0','Redshift'],quantiles=[0.025,0.5-0.34,0.5+0.34,0.95],
           weights=k.weights,range=crange,show_titles=True)
else:
    arr = np.array([k.BH.chain_glob['GM']/1e6,k.BH.chain_glob['R0']/1e3,k.BH.chain_glob['x0']*1e3,
                    k.BH.chain_glob['y0']*1e3,k.BH.chain_glob['vx0']*1e3,
                    k.BH.chain_glob['vy0']*1e3,k.BH.chain_glob['vz0']])
    arr = arr.transpose()
    plt.figure(figsize=(12,12))
    crange = [0.999,0.999,0.999,0.999,0.999,0.999,0.999]

    corner(arr,labels=['GM','R0','x0','y0','vx0','vy0','vz0'],quantiles=[0.025,0.5-0.34,0.5+0.34,0.95],range=crange,
           weights=k.weights,show_titles=True)
    
plt.savefig(savefile+'_bh_corner.png')

# plot the stellar orbital parameters

arr = np.array([k.stars[0].chain['O'],
                k.stars[0].chain['w'],
                k.stars[0].chain['i'],
                k.stars[0].chain['P'],
                k.stars[0].chain['e'],
                k.stars[0].chain['T0']])
arr = arr.transpose()
plt.figure(figsize=(12,12))
crange = [0.999,0.999,0.999,0.999,0.999,0.999]

corner(arr,labels=['O','w','i','P','e','T0'],quantiles=[0.025,0.5-0.34,0.5+0.34,0.95],range=crange,
        weights=k.weights,show_titles=True)

plt.savefig(savefile+'_star_corner.png')