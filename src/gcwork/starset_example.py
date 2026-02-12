import os
import shutil
import sys
import math
import numpy as np
import pylab as py
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from functools import reduce
from gcreduce import gcutil
from gcwork import starset
from gcwork import starTables
from gcwork import objects
from gcwork.polyfit import accel
from scipy import stats
#from gcutil import nmpfit_mos as nmpfit
from stsci.tools import nmpfit
from gcwork import realmedian
from scipy import stats
from scipy import spatial
import random
import pickle
import pdb
from astropy.table import Table
import itertools


def starset_example():
    # This is an example of how to use starset to load align runs and polyfit directories
    # Author: T. Do, 2021-04-02
    
    # additional examples can be found in accel_class.py

    # for aligns, we use a concept of a rootDir, align name, polyfit directory,
    # and points directory names. These should be defined when using starset.

    rootDir = '/u/ghezgroup/align/18_10_02/'
    align = 'align/align_d_rms_1000_abs_t'
    poly= 'polyfit_3_c/fit' # polyfit after confusion removal
    points = 'points_3_c/'

    # Load up positional information from align. This will take a long time
    s = starset.StarSet(rootDir + align)


    # starset is setup as a set of objects that are stars with
    # attributes such as their positions and magnitudes. This info is
    # read from the align files. At the top level of the starset
    # object is information about the names of the stars and
    # magnitudes. The index number of the names array can be used to
    # find the correct star in the starset object.

    # first get the names and magnitudes out for reference
    names = s.getArray('name') # returns a list of names

    # create a numpy array so we can search it
    names = np.array(names,dtype='str')
    
    mag = s.getArray('mag') * 1.0

    # if you're looking for a specific star, search for it in the names list
    s02_ind = np.where(names == 'S0-2')[0]

    # if the array is not empty then, that name exists, get the index from the list
    s02_ind = s02_ind[0]

    s02_x = s.stars[s02_ind].getArrayAllEpochs('x')
    s02_y = s.stars[s02_ind].getArrayAllEpochs('y')

    # statistical uncertainties in positional error
    s02_xerr_p = s.stars[s02_ind].getArrayAllEpochs('xerr_p')
    s02_yerr_p = s.stars[s02_ind].getArrayAllEpochs('yerr_p')

    # alignment errors in position
    s02_xerr_a = s.stars[s02_ind].getArrayAllEpochs('xerr_a')
    s02_yerr_a = s.stars[s02_ind].getArrayAllEpochs('yerr_a')
    
    # S0-2 magnitudes
    s02_mag = s.stars[s02_ind].getArrayAllEpochs('mag')
    
    # for some reason, the time is not a value that you get using
    # getArrayAllEpochs, but is an attribute that you can access
    # directly
    s02_years = np.array(s.stars[s02_ind].years)



    # if you want information about all the stars, you can loop through them
    # for example:
    # for i in range(len(names)):
    #     ind = names.index(names[i])

    #     x = self.starSet.stars[ind].getArrayAllEpochs('x')
    #     y = self.starSet.stars[ind].getArrayAllEpochs('y')
    #     xerr_p = self.starSet.stars[ind].getArrayAllEpochs('xerr_p')
    #     yerr_p = self.starSet.stars[ind].getArrayAllEpochs('yerr_p')

    #     xerr_a = self.starSet.stars[ind].getArrayAllEpochs('xerr_a')
    #     yerr_a = self.starSet.stars[ind].getArrayAllEpochs('yerr_a')

    #     xerr = np.sqrt(xerr_p**2 + xerr_a**2)
    #     yerr = np.sqrt(yerr_p**2 + yerr_a**2)
    

    # load polyfit information. This will take a long time
    s.loadPolyfit(rootDir + poly, arcsec=1, silent=True)
    s.loadPolyfit(rootDir + poly, arcsec=1, accel=1, silent=True)

    # load the points files corresponding to the polyfit. Note
    # that this points file might not be the same as the align
    # file if some trimming of confused epochs is done.
    s.loadPoints(rootDir + points)


    # read accel fit result from poly file in mas/yr^2 note that these
    # values come from the polyfit and associate points files.
    # polyfit results popluate the 'global' array so you can get info
    # directly on the velocity and accelerations of all the stars
    # instead of having to loop through each one
    x = s.getArray('fitXa.p')
    y = s.getArray('fitYa.p')
    xerr = s.getArray('fitXa.perr')
    yerr = s.getArray('fitYa.perr')
    vx = s.getArray('fitXa.v')
    vy = s.getArray('fitYa.v')
    vxe = s.getArray('fitXa.verr')
    vye = s.getArray('fitYa.verr')
    ax = s.getArray('fitXa.a')
    ay = s.getArray('fitYa.a')
    axe = s.getArray('fitXa.aerr')
    aye = s.getArray('fitYa.aerr')
    r2d = s.getArray('r2d')
    chi2x = s.getArray('fitXa.chi2')
    chi2y = s.getArray('fitYa.chi2')
    t0 = s.getArray('fitXa.t0')

    # read the linear fitting result
    x_v = s.getArray('fitXv.p')
    y_v = s.getArray('fitYv.p')
    xe_v = s.getArray('fitXv.perr')
    ye_v = s.getArray('fitYv.perr')
    vx_v = s.getArray('fitXv.v')
    vy_v = s.getArray('fitYv.v')
    vxe_v = s.getArray('fitXv.verr')
    vye_v = s.getArray('fitYv.verr')
    chi2xv = s.getArray('fitXv.chi2')
    chi2yv = s.getArray('fitYv.chi2')
    t0_v = s.getArray('fitXv.t0')
    
