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

class accelClass(starset.StarSet):
    """Object containing acceleration analysis. This class will
    inherit methods from the StarSet class

    CLASS METHODS:
        computeSpeckleAOVel(): compute the speckle and AO velocities individually and
        check the difference in the velocity vectors.

        epochDistance() - find the range in epochs when two stars are within some
        threshold distance. This function is helpful for trimming epochs when sources
        may be confused

        findMisMatch(starName = 'irs16NE', bootStrap=False) - check to
        see whether there are points that are mismatched in the star
        fit.

        findNearestNeighbors() - find the nearest neighbors

        findNearestStar() - find the minimum distance to every star

        findNonPhysical() - find non-physical accelerations

        fitAccel(time, x, xerr) - fit for acceleration and returns mpfit
        object.

        fitVelocity(time, x, xerr) - fit for a velocity and returns an mpfit
        object.

        plotchi2(removeConfused = False) - plot the chi2 distribution
        of the acceleration and velocity fits.

        removeConfusedEpochs(mkPointsFiles=False, runPolyfit=False) -
        make a copy of the points files into points_c/ and then trim
        out all epochs from each confused star. Can be set to produce
        new points files and run polyfit on those

        testChiSqFitXY - look what additive error is necessary to add to
        the x and y coordinates

        updateAccel() - update the acceleration calculations of radial
        and tangential acceleration


    HISTORY: 2009-03-11 - T. Do
             2010-04-05 - T. Do - added function to compute the additive
             error necessary to get the Chi2 to be as expected.
    """

    # Notes X positions: the positions from polyfit are 'correct'
    # in the sense of RA increasing to the east, however, the
    # positions from starSet are NOT in the right direction. Must
    # multiply any starSet positions by -1.0 to get into the right
    # orientation.  Accelerations computed by polyfit are also in the
    # correct direction.

    # initialize physical and non-physical acceleration variables
    # these will be updated with real names when nonPhysical() is run
    nonPhysical = ''  # names of stars with sig. non-physical accel
    physical = ''   # names of stars with sig. physical accelerations
    nearest = np.zeros(1)   # the nearest star to every star
    nearestNonPhysical = np.zeros(1)  # the nearest star to unphysical stars
    bad = np.zeros(1)  # indices of non-physical stars
    goodAccel = np.zeros(1) # indices of sig. physical acceleration
    chiThreshold = 0     # reduced chi-square threshold

    prefix = '' # default prefix for file names
    maxEpochInd =  [] # index of stars detected in all epochs
    speckInd = [] # indices of speckle epochs
    aoInd = [] # indicies of AO epochs
    images = None # images corresponding to each aligned epoch

    rmax = 4.0 # maximum radius to compute the chi-square values for additive error tests
    rmin = 0.0 # minium radius to compute the chi-square values for additive error tests
    magmax = 14.5  # set the maximum magnitude, faintest end
    magmin = 0.0   # set the minimum magnitude, brightest end

    addErrStarsInd = 0 # the stars that are used in the additive error calculations

    # a python dictionary type that will hold the closest stars. The
    # keys will be the star names and it will contain an array of
    # names of closest stars. In addition, there are keys with
    # starname_dist which contain the corresponding closest approach
    # of that neighboring star. starname_ind will contain the index
    neighbors = None
    #confusionThreshold = 0.06  # distance within which a star is said to be confused
    confusionThreshold = 0.1  # distance within which a star is said to be confused
    confusionMagThreshold = 5  # < delta magnitude required to be considered confused
    confusedSources = [] # names of confused sources
    confusedSourcesInd = [] # index corresponding to the confused sources

    # F test probabilities for justifying the use of acceleration for fits
    xFProb = []
    yFProb = []


    def __init__(self, rootDir='./', align='align/align_d_rms_1000_abs_t',
                 poly='polyfit_3_c/fit', points='points_3_c/', sigma=5,
                 namesFile = False, findAddErr = False, rmax=0.0,
                 verbose=True, epochsRequired = 0, run_high_order_poly=False):

        # Load the align files into the class so other methods can use the info.
        self.rootDir = rootDir
        self.align = align
        self.points = points
        self.poly = poly
        self.sigma = sigma
        self.rmax = rmax  # maximum radius for chi-sq in additive error
        self.run_high_order_poly = run_high_order_poly

        # prefix for plots
        #self.plotPrefix = 'plots/'+os.path.split(os.path.dirname(rootDir))[1]+'_'
        self.plotPrefix = rootDir + 'plots/'
        self.prefix = os.path.split(os.path.dirname(rootDir))[1]
        # Load GC constants
        cc = objects.Constants()
        self.cc = cc

        # Load up positional information from align.
        s = starset.StarSet(rootDir + align)
        s.loadPolyfit(rootDir + poly, arcsec=1, silent=True)
        s.loadPolyfit(rootDir + poly, arcsec=1, accel=1, silent=True)

        # load the points files corresponding to the polyfit. Note
        # that this points file might not be the same as the align
        # file if some trimming of confused epochs is done.
        s.loadPoints(rootDir + points)

        names = s.getArray('name')
        mag = s.getArray('mag') * 1.0

        # load an alternative align with names. This align is to
        # provide names for the larger align and should be aligned to
        # the reference epoch. DOESN'T WORK RIGHT NOW - trimmed list
        # different than original reference list.
        if namesFile:
            print( 'getting names from a different align: ', namesFile)
            nameAlign = starset.StarSet(namesFile)
            refNames = nameAlign.getArray('name')

            print( refNames[20:30])
            print( names[20:30])
            # take the names here and replace the names in the input
            # alignment
            names[0:len(refNames)-1] = refNames

        # read accel fit result from poly file in mas/yr^2
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

        # set the chi-sq threshold to be three times the DOF for velocity fits
        #self.nEpochs = s.getArray('velCnt')
        self.nEpochs = s.getArray('pointsCnt')
        self.chiThreshold = (np.max(self.nEpochs) - 3.0) * 3.0

        # index of stars with the maximum number of epochs detected
        self.maxEpochInd = np.where(self.nEpochs == np.max(self.nEpochs))[0]

        # T0 for each of the acceleration fits
        self.epoch = s.getArray('fitXa.t0')

        # All epochs sampled
        self.allEpochs = np.array(s.stars[0].years)

        # figure out which  epochs are speckle and AO
        epochFile = rootDir+'scripts/epochsInfo.txt'

        if os.path.isfile(epochFile):
            # Load up epochsInfo.txt
            epochTab = Table.read(self.rootDir + '/scripts/epochsInfo.txt', format='ascii')

            # Trim out all the non-aligned stuff... we don't do anything with it anyhow
            idx = np.where(epochTab['doAlign'] == 1)
            epochTab = epochTab[idx]

            aoFlag = epochTab['isAO']
            epochList = epochTab['epoch']
            epochDir = epochTab['directory']

            # separate the speckle points from the AO points
            speckInd = np.where((aoFlag == 0))[0]
            aoInd = np.where((aoFlag == 1))[0]

            if verbose == True:
                print( 'speckle epochs: ', self.allEpochs[speckInd])
                print( 'AO epochs: ', self.allEpochs[aoInd])
            self.speckInd = speckInd
            self.aoInd = aoInd

        else:
            print( 'epochsInfo file missing: '+epochFile)

        self.names = np.array(names)  # star names
        self.mag = mag  # magnitudes
        self.cc = cc    # constants
        self.x, self.y = x, y  # fited positions at T0
        self.xerr, self.yerr = xerr, yerr # positional errors
        self.vx, self.vy = vx, vy  # velocities
        self.vxe, self.vye = vxe, vye  # velocity errors
        self.ax, self.ay = ax, ay  # accelerations
        self.axe, self.aye = axe, aye  # acceleration errors
        self.t0 = t0 # acceleration t0

        self.x_v, self.y_v = x_v, y_v # linear fitting position
        self.xe_v, self.ye_v = xe_v, ye_v
        self.vx_v, self.vy_v = vx_v, vy_v  # linear fitting velocities
        self.vxe_v, self.vye_v = vxe_v, vye_v
        self.t0_v = t0_v # acceleration t0

        self.chi2x, self.chi2y = chi2x, chi2y   # acceleartion fit chi-square
        self.chi2xv, self.chi2yv = chi2xv, chi2yv # velocity fit chi-square

        self.r2d = r2d    # radial distance from Sgr A*
        self.starSet = s    # the original starset object

        self.nearestStarName = np.zeros(len(x), dtype='S13')
        self.nearestStarDist = np.zeros(len(x), dtype=float)

        # scale factor for the errors in computing additive factor
        self.errScaleFactor = 1.0

        # initialize the acceleration information
        self.updateAccel()
        self.computeFTest()  # compute the F test
#        self.computeJerk()  # compute the jerk -- not functional yet
#        self.computeFTestJerk()  # compute the F test 

        # run higher order polynomial fit and f test
        self.poly_orders = np.array([1,2,3,4,5])
        ##only need to run this step once and then poly files are made
        if run_high_order_poly:
            self.run_ftest_hpoly(make_plot=False, p_crit=0.8) # run f test to find the best polynomial order

        self.findNearestStar()
        self.findNonPhysical(verbose=verbose, epochsRequired=epochsRequired)
        self.findNearestNeighbors()

        # self.findMismatch(bootStrap=True)
        #if (len(self.speckInd) > 3):
        #    self.computeSpeckleAOVel(requireAllEpochs=True)
        #self.saveClass()
        #self.testChi2Fit()

        if len(self.speckInd) > 0:
            data = 'speckle'
        else:
            data = 'ao'
        if findAddErr:
            # testErr should be in arcseconds
            self.testChi2Fit(scaleFactor = self.errScaleFactor, data = data,
                             testErr = np.arange(0.00005,0.0005,0.00001))


        # write all the data to a table
        t_write = Table()
        t_write['name'] = self.names.copy()
        t_write['mag'] = self.mag.copy()
        t_write['nEpochs'] = self.nEpochs.copy()
        t_write['r'] = self.r2d.copy()

        t_write['name'].format = '13s'
        t_write['mag'].format = '.2f'
        t_write['nEpochs'].format = '3d'
        t_write['r'].format = '.2e'

        # mark non physical accelerating sources
        idx_nonphy = [np.where(t_write['name']==i)[0][0] for i in self.nonPhysical]
        t_write['nonPhyAcc'] = 0
        t_write['nonPhyAcc'][idx_nonphy] = 1

        # mark significant accelerating sources
        # goodAccel: (ar+sigma*are < 0) & (abs(ar/are) > sigma) & (mag < 16.5) & not nonphysical
        idx_acc = self.goodAccel
        t_write['phyAcc'] = 0
        t_write['phyAcc'][idx_acc] = 1

        # accel fit for ar, at: km/s/yr
        t_write['ar'] = self.ar.copy()
        t_write['are'] = self.are.copy()
        t_write['at'] =  self.at.copy()
        t_write['ate'] =  self.ate.copy()

        t_write['ar'].format = '.2f'
        t_write['are'].format = '.2f'
        t_write['at'].format = '.2f'
        t_write['ate'].format = '.2f'

        # accel fit: as, as/yr, as/yr2
        t_write['x0'] = self.x.copy()
        t_write['y0'] = self.y.copy()
        t_write['x0e'] = self.xerr.copy()
        t_write['y0e'] = self.yerr.copy()
        t_write['vx'] = self.vx.copy()
        t_write['vy'] = self.vy.copy()
        t_write['vxe'] = self.vxe.copy()
        t_write['vye'] = self.vye.copy()
        t_write['ax'] = self.ax.copy()
        t_write['ay'] = self.ay.copy()
        t_write['axe'] = self.axe.copy()
        t_write['aye'] = self.aye.copy()
        t_write['t0'] = self.t0.copy()
        t_write['chi2x'] = self.chi2x.copy()
        t_write['chi2y'] = self.chi2y.copy()

        t_write['x0'].format = '.5f'
        t_write['y0'].format = '.5f'
        t_write['x0e'].format = '.5f'
        t_write['y0e'].format = '.5f'
        t_write['vx'].format = '.5f'
        t_write['vy'].format = '.5f'
        t_write['vxe'].format = '.5f'
        t_write['vye'].format = '.5f'
        t_write['ax'].format = '.6f'
        t_write['ay'].format = '.6f'
        t_write['axe'].format = '.6f'
        t_write['aye'].format = '.6f'
        t_write['t0'].format = '.2f'
        t_write['chi2x'].format = '.1f'
        t_write['chi2y'].format = '.1f'
        t_write.sort('r')
        t_write.write(os.path.join(self.rootDir,self.poly[:-3]+'accel.txt'), 
                format='ascii.fixed_width', delimiter=None, overwrite=True)


        # linear fit: as, as/yr
        # write all the data to a table
        t_write = Table()
        t_write['name'] = self.names.copy()
        t_write['mag'] = self.mag.copy()
        t_write['nEpochs'] = self.nEpochs.copy()
        t_write['r'] = self.r2d.copy()
        t_write['nonPhyAcc'] = 0
        t_write['nonPhyAcc'][idx_nonphy] = 1
        t_write['phyAcc'] = 0
        t_write['phyAcc'][idx_acc] = 1

        t_write['name'].format = '13s'
        t_write['mag'].format = '.2f'
        t_write['nEpochs'].format = '3d'
        t_write['r'].format = '.2e'

        t_write['x0'] = self.x_v.copy()
        t_write['y0'] = self.y_v.copy()
        t_write['x0e'] = self.xe_v.copy()
        t_write['y0e'] = self.ye_v.copy()
        t_write['vx'] = self.vx_v.copy()
        t_write['vy'] = self.vy_v.copy()
        t_write['vxe'] = self.vxe_v.copy()
        t_write['vye'] = self.vye_v.copy()
        t_write['t0'] = self.t0_v.copy()
        t_write['chi2x'] = self.chi2xv.copy()
        t_write['chi2y'] = self.chi2yv.copy()

        t_write['x0'].format = '.5f'
        t_write['y0'].format = '.5f'
        t_write['x0e'].format = '.5f'
        t_write['y0e'].format = '.5f'
        t_write['vx'].format = '.5f'
        t_write['vy'].format = '.5f'
        t_write['vxe'].format = '.5f'
        t_write['vye'].format = '.5f'
        t_write['t0'].format = '.2f'
        t_write['chi2x'].format = '.1f'
        t_write['chi2y'].format = '.1f'
        t_write.sort('r')

        t_write.write(os.path.join(self.rootDir,self.poly[:-3]+'linear.txt'), 
                format='ascii.fixed_width', delimiter=None, overwrite=True)


    def updateAccel(self):
        """ Updates the radial and tangential acceleration arrays.
        """
        cc = self.cc
        x, y = self.x, self.y
        vx, vy = self.vx, self.vy
        ax, ay = self.ax, self.ay
        axe, aye = self.axe, self.aye

        # Lets do radial/tangential
        r = np.sqrt(x**2 + y**2)
        if ('Radial' in self.poly):
            at = ax
            ar = ay
            ate = axe
            are = aye
        else:
            ar = ((ax*x) + (ay*y)) / r
            at = ((ax*y) - (ay*x)) / r
            are = np.sqrt((axe*x)**2 + (aye*y)**2) / r
            ate = np.sqrt((axe*y)**2 + (aye*x)**2) / r

        # Total acceleration
        atot = py.hypot(ax, ay)
        atoterr = np.sqrt((ax*axe)**2 + (ay*aye)**2) / atot

        # Calculate the acceleration limit set by the projected radius
        # Convert into cm
        r2d = r * cc.dist * cc.cm_in_au
        rarc =np.arange(0.01,10.0,0.01)
        rsim = rarc * cc.dist * cc.cm_in_au

        # acc1 in cm/s^2
        a2d = -cc.G * cc.mass * cc.msun / r2d**2
        a2dsim = -cc.G * cc.mass * cc.msun / rsim**2
        # acc1 in km/s/yr
        a2d *= cc.sec_in_yr / 1.0e5
        # a2d *= 1000.0 / cc.asy_to_kms
        a2dsim *= cc.sec_in_yr / 1.0e5
        # a2dsim *= 1000.0 / cc.asy_to_kms

        # convert between arcsec/yr^2 to km/s/yr
        ar *= cc.asy_to_kms
        are *= cc.asy_to_kms
        at *= cc.asy_to_kms
        ate *= cc.asy_to_kms

        self.ar = ar      # radial acceleration
        self.are = are    # radial acceleration error
        self.at = at      # tangential acceleration
        self.ate = ate    # tangential acceleration error
        self.atot = atot  # total acceleration
        self.a2d = a2d    # maximum acceleration for each star at the x, y position
        self.rarc, self.a2dsim = rarc, a2dsim # radial distance vs. accel (z = 0)

    def minDistance(self, xfit1,yfit1,xfit2,yfit2,t1,t2,trange,pause=0):
        """
        From the coefficients of the fits to two stars, find the minimum
        distance between them by using the acceleration fits.
        xfit1 = [x0, vx, ax]
        yfit1 = [y0, vy, ay]

        HISTORY: 2010-01-13 - T. Do
        """
        time = np.arange(trange[0],trange[1]+1,0.01)
        time1 = time-t1
        time2 = time-t2
        xpos1 = self.poly2(xfit1, time1)
        ypos1 = self.poly2(yfit1, time1)
        xpos2 = self.poly2(xfit2, time2)
        ypos2 = self.poly2(yfit2, time2)

        distance = np.sqrt((xpos1-xpos2)**2 + (ypos1 - ypos2)**2)
        if pause:
            clf()
            plot(xpos1,ypos1)
            plot(xpos2,ypos2)
        #print( time)
        return np.amin(distance)

    def epochDistance(self, xfit1,yfit1,xfit2,yfit2,t1,t2,trange,dist,pause=False):
        """ Given two sets of acceleration fits, and a threshold distance,
        compute the range of epochs in which the two stars are within
        that distance. Returns None if the two stars never gets that close.
        """

        time = np.arange(trange[0],trange[1]+1,0.01)
        time1 = time-t1
        time2 = time-t2
        xpos1 = self.poly2(xfit1, time1)
        ypos1 = self.poly2(yfit1, time1)
        xpos2 = self.poly2(xfit2, time2)
        ypos2 = self.poly2(yfit2, time2)

        distance = np.sqrt((xpos1-xpos2)**2 + (ypos1 - ypos2)**2)

        good = np.where(distance <= dist)[0]
        if pause:
            plt.clf()
            plt.plot(xpos1,ypos1)
            plt.plot(xpos2,ypos2)
            plt.plot(xpos1[good],ypos1[good],'bo')
            plt.plot(xpos2[good],ypos2[good],'go')
        #print( time)

        if (len(good) > 0):
            return np.array([np.min(time[good]), np.max(time[good])])
        else:
            return None

    def epochDistanceTest(self, index = 0):
        """ test the epochDistance procedure. Should have run
        findNearestNeighbors() funcition.
        """
        star1 = self.confusedSources[index]
        star2 = (self.neighbors[star1])[0]
        print( star1, star2)
        names = list(self.names)
        s1 = names.index(star1)
        s2 = names.index(star2)

        x,y = self.x,self.y
        vx, vy = self.vx, self.vy
        ax, ay = self.ax, self.ay

        xfit1 = [x[s1],vx[s1],ax[s1]]
        xfit2 = [x[s2],vx[s2],ax[s2]]
        yfit1 = [y[s1],vy[s1],ay[s1]]
        yfit2 = [y[s2],vy[s2],ay[s2]]

        epoch1 = self.epoch[s1]
        epoch2 = self.epoch[s2]
        minEpoch = np.amin(self.allEpochs)
        maxEpoch = np.amax(self.allEpochs)
        epochRange = self.epochDistance(xfit1, yfit1, xfit2, yfit2, epoch1, epoch2,
                                        [minEpoch, maxEpoch],self.confusionThreshold,
                                        pause = True)
        text(x[s1],y[s1],star1)
        text(x[s2],y[s2],star2)
        print( epochRange)

    def fitLin(self, p, fjac=None, x=None, y=None, err=None):
        # linear fit
        fun = p[0] + p[1]*x

        # deviations from the model
        deviates = (y - fun)/err
        return [0, deviates]

    def line(self, p, x):
        return p[0]+p[1]*x

    def fitfunPoly2(self, p, fjac=None, x=None, y=None, err=None):
        # second order polynomial
        fun = p[0] + p[1]*x + 0.5*p[2]*x**2

        # deviations from the model
        deviates = (y - fun)/err
        return [0, deviates]

    def poly2(self, p, x):
        return p[0] + p[1]*x + 0.5*p[2]*x**2


    def fitfunPoly3(self, p, fjac=None, x=None, y=None, err=None):
        # third order polynomial
        fun = p[0] + p[1]*x + 0.5*p[2]*x**2 + (1./6.)*p[3]*x**3

        # deviations from the model
        deviates = (y - fun)/err
        return [0, deviates]

    def poly3(self, p, x):
        return p[0] + p[1]*x + 0.5*p[2]*x**2 + (1./6.)*p[3]*x**3


    def isPhysical(self, x, y, ax, axe, ay, aye, arcsec = True):
        """
        Return a True or False depending on whether the acceleration
        measurement is physical. Unphysical accelerations are defined as
        either: 1. Significant tangential acceleration 2. Significant
        positive radial acceleration 3. Negative acceleration greater than
        the maximum allowed at the 2D position.

        RETURN: status array where 1 is true [tangential, pos. radial, > max radial]
        """

        status = np.zeros(3)  # return array

        # Lets do radial/tangential
        r = np.sqrt(x**2 + y**2)
        ar = ((ax*x) + (ay*y)) / r
        at = ((ax*y) - (ay*x)) / r
        are = np.sqrt((axe*x)**2 + (aye*y)**2) / r
        ate = np.sqrt((axe*y)**2 + (aye*x)**2) / r

        # Total acceleration
        atot = py.hypot(ax, ay)
        atoterr = np.sqrt((ax*axe)**2 + (ay*aye)**2) / atot

        # Calculate the acceleration limit set by the projected radius

        if arcsec:
            #convert to mks
            cc = objects.Constants()

            # Convert into cm
            r2d = r * cc.dist * cc.cm_in_au

            rarc =np.arange(0.01,10.0,0.01)
            rsim = rarc * cc.dist * cc.cm_in_au

            # acc1 in cm/s^2
            a2d = -cc.G * cc.mass * cc.msun / r2d**2
            a2dsim = -cc.G * cc.mass * cc.msun / rsim**2

            # acc1 in km/s/yr
            a2d *= cc.sec_in_yr / 1.0e5
            #a2d *= 1000.0 / cc.asy_to_kms

            # convert between arcsec/yr^2 to km/s/yr
            ar *= cc.asy_to_kms
            are *= cc.asy_to_kms

            at *= cc.asy_to_kms
            ate *= cc.asy_to_kms

            a2dsim *= cc.sec_in_yr / 1.0e5
            #a2dsim *= 1000.0 / cc.asy_to_kms

        # tests to see if the accelerations are physical
        if (abs(at/ate) > self.sigma):
            #print( 'significant tangential acceleration')
            status[0] = 1

        #print( 'radial acceleration %f +- %f' % (ar, are))
        #print( 'tangential acceleration %f +- %f' % (at, ate))
        if ((ar - (self.sigma*are)) > 0):
            #print( 'positive radial acceleration')
            status[1] = 1

        if (ar + (self.sigma*are) < a2d):
            #print( 'too large radial acceleration')
            status[2] = 1

        # print( some diagnostic info)
        print( 'isPhsyical: ar, are, at, ate: ', ar, are, at, ate)
        return status


    def findNearestStar(self):
        """Use the acceleration fits to find the nearest stars that
        gets within a certain distance. Will store in the dictionary
        self.neighbors
        """

        # get some variables in to local scope
        names = np.array(self.names)
        sigma = self.sigma
        x, y = self.x, self.y
        vx, vy = self.vx, self.vy
        ax, ay = self.ax, self.ay
        mag = self.mag
        cc = self.cc
        chi2x, chi2y = self.chi2x, self.chi2y
        chi2xv, chi2yv = self.chi2xv, self.chi2yv
        r2d = self.r2d
        rarc, a2dsim = self.rarc, self.a2dsim
        epoch, allEpochs = self.epoch, self.allEpochs # self.epoch is fitXa.t0

        nearestStar = np.zeros(len(x))  # distance to nearest star
        nearestStarInd = np.zeros(len(x),'int32') # index of nearest star

        minEpoch = np.amin(allEpochs)
        maxEpoch = np.amax(allEpochs)

        # look for the minium distance between all stars according to the fit
        for ii in np.arange(0,len(x)):
            distances = np.sqrt((x[ii] - x)**2 + (y[ii] - y)**2)
            srt = distances.argsort()
            distances = distances[srt]
            #nearestStar[ii] = distances[1]

            #loop over the 5 closest sources to see which one is closest
            #print( names[ii])
            fitMin = np.zeros(5)

            for rr in np.arange(1,6):
                #print( names[srt[rr]])

                xfit1 = [x[ii],vx[ii],ax[ii]]
                xfit2 = [x[srt[rr]],vx[srt[rr]],ax[srt[rr]]]
                yfit1 = [y[ii],vy[ii],ay[ii]]
                yfit2 = [y[srt[rr]],vy[srt[rr]],ay[srt[rr]]]
                fitMin[rr-1] = self.minDistance(xfit1,yfit1,xfit2,yfit2,
                                                epoch[ii],epoch[srt[rr]],[minEpoch,maxEpoch])

            minInd = np.argmin(fitMin)
            nearestStar[ii] = fitMin[minInd]
            # add 1 to the index because the first star will be itself
            nearestStarInd[ii] = srt[minInd+1]

        self.nearestStarName = names[nearestStarInd]
        self.nearestStarDist = nearestStar

    def findNearestNeighbors(self):
        """Use the acceleration fits to find the nearest stars that
        gets within a certain distance. Will store in the dictionary
        self.neighbors
        """
        threshold = self.confusionThreshold
        magThreshold = self.confusionMagThreshold
        # get some variables in to local scope
        names = np.array(self.names)
        sigma = self.sigma
        x, y = self.x, self.y
        vx, vy = self.vx, self.vy
        ax, ay = self.ax, self.ay
        mag = self.mag
        cc = self.cc
        chi2x, chi2y = self.chi2x, self.chi2y
        chi2xv, chi2yv = self.chi2xv, self.chi2yv
        r2d = self.r2d
        rarc, a2dsim = self.rarc, self.a2dsim
        epoch, allEpochs = self.epoch, self.allEpochs #self.epoch is fitXa.t0

        # reset the confusion arrays
        self.neighbors = None
        self.confusedSources = []
        self.confusedSourcesInd = []

        nearestStar = np.zeros(len(x))  # distance to nearest star
        nearestStarInd = np.zeros(len(x),'int32') # index of nearest star

        minEpoch = np.amin(allEpochs)
        maxEpoch = np.amax(allEpochs)

        # look for the minium distance between all stars according to the fit
        for ii in np.arange(0,len(x)):
            distances = np.sqrt((x[ii] - x)**2 + (y[ii] - y)**2)
            srt = distances.argsort()
            distances = distances[srt]
            closeStarNames = names[srt[0:30]]
            closeStarInds = srt[0:30]
            # find the differences in magnitude between primary and neighbors
            magDiff = mag[closeStarInds] - mag[ii]
            #nearestStar[ii] = distances[1]

            #loop over the 10 closest sources to see which one is closest

            fitMin = np.zeros(len(closeStarNames))

            for rr in np.arange(1,len(closeStarNames)):
                #print( names[srt[rr]])
                xfit1 = [x[ii],vx[ii],ax[ii]]
                xfit2 = [x[srt[rr]],vx[srt[rr]],ax[srt[rr]]]
                yfit1 = [y[ii],vy[ii],ay[ii]]
                yfit2 = [y[srt[rr]],vy[srt[rr]],ay[srt[rr]]]

                fitMin[rr] = self.minDistance(xfit1,yfit1,xfit2,yfit2,
                                             epoch[ii],epoch[srt[rr]],[minEpoch,maxEpoch])


            # find all the stars that will come within the threshold
            # (and exclude the star itself)

            # include only stars that are brighter than the magnitude
            # threshold difference. Note that the magnitude filter is
            # non-symmetric. If magThreshold = 5, A 10th magnitude
            # star will not be considered confused next to a 15th
            # magnitude star, but the 15th magnitude star will be
            # deemd confused by the 10th magnitude star

            good = np.where((fitMin < threshold) & (fitMin > 0) & (magDiff < magThreshold))[0]
            if (len(good) > 0):
                # record the name of the potential confused source
                # those stars are crossing another stars within certain radius and mag difference
                # only if one of them are detected in those epochs, they will be classified as confusion
                # if both of them are detected, there is no confusion removal
                self.confusedSources = np.concatenate((self.confusedSources, [names[ii]]))
                self.confusedSourcesInd = np.concatenate((self.confusedSourcesInd, [ii]))

                minInd = np.argmin(fitMin)


                # fill in the arrays
                if self.neighbors is None:
                    self.neighbors = {names[ii]:closeStarNames[good],
                                      names[ii]+'_dist':fitMin[good],
                                      names[ii]+'_ind': closeStarInds[good]}
                else:
                    self.neighbors[names[ii]] = closeStarNames[good]
                    self.neighbors[names[ii]+'_dist'] = fitMin[good]
                    self.neighbors[names[ii]+'_ind'] = closeStarInds[good]
            else:
                if self.neighbors is None:
                    self.neighbors = {names[ii]:None,
                                      names[ii]+'_dist':None,
                                      names[ii]+'_ind':None}
                else:
                    self.neighbors[names[ii]] = None
                    self.neighbors[names[ii]+'_dist'] = None
                    self.neighbors[names[ii]+'_ind'] = None

    def removeConfusedEpochs(self, mkPointsFiles = True, Points='points_3_c/', Poly='polyfit_3_c/', 
            debug = False, runPolyfit=True, rlim=0.5):
        """ Take a look at the closest neighbors of all stars to
        figure out what epochs are possibly confused. The epochs that
        are found in one star but not the other are assumed to be
        confused and should be removed. This function requires that
        findNearestNeighbors() has been run.

        Will create new points files in points_c/ without the confused
        epochs. The file points_c/removedEpochs.txt will contain a
        list of stars, number of epochs removed, and the epochs
        removed

        Keywords: mkPointsFiles - create new points files that have
        the confused sources removed and put them in points_c/

        rlim - radius inside of which the confusion code will ignore
        because it doesn't deal with orbits at the moment (does not
        apply to manually indicated epochs in epochsfile)

        runPolyfit - run poly fit on the points_c directory and put
        polyfit results in polyfit_c/
        """
        names = list(self.names)
        x = self.x
        y = self.y
        vx, vy, ax, ay = self.vx, self.vy, self.ax, self.ay
        r2d = self.r2d
        mag = self.mag
        neighbors = self.neighbors
        epochsRemoved = np.zeros(len(names))
        oldDir = os.path.split(self.rootDir+self.points)[0]
        newDir = self.rootDir+Points
        epochLimits = np.array([np.min(self.allEpochs), np.max(self.allEpochs)])
        epochsRemoved = None
        nEpochsRemoved = np.zeros(len(names))
        culprits = None

        if mkPointsFiles:
            # make a new directory
            if os.path.isdir(newDir):
                print( 'removing exisiting points_c directory: '+newDir)
                shutil.rmtree(newDir,ignore_errors=True)

            # copy the files into the new directory
            print( 'copying files to :', newDir)
            cmdStr = 'cp -rf %s %s' % (oldDir, newDir)
            os.system(cmdStr)
            print( 'done copying files')
            workDir = newDir+'/'
        else:
            # if not making new files then use the old directory
            workDir = oldDir+'/'

        if debug:
            nRange = np.arange(10)
            checkind = np.where(np.array(names) == 'S0-2')[0]
            nRange = np.append(checkind,nRange)
            print( nRange)
        else:
            nRange = np.arange(len(names))


        # loop through stars and go over the ones that are greater
        # than 0.5" away because those stars could potentially have a
        # lot of problems and are likely not able to be fit by
        # acceleration alone

        # for those stars with possible fake sources
        # donnot remove their epochs based on those fake sources
        pri = np.load('polyfit_2_s/edge/stars_fake.npy')

        for i in nRange:
            if r2d[i] < rlim:
                continue

            # check to see if any points are within rlim, if so, do not run this star
            # open the points files
            ptFile = workDir + names[i]+'.points'
            phFile = workDir + names[i]+'.phot'

            # check if this file is empty
            # because this star could be a confusing source before and be removed
            if os.stat(ptFile).st_size == 0:
                continue
            tab1 = starTables.read_points(ptFile)
            phot1 = starTables.read_points(phFile)

            r2d_arr = np.sqrt(tab1['x']**2 + tab1['y']**2)
            if np.any(r2d_arr < rlim):
                continue
            
            
            if (neighbors[names[i]] is not None):
                star1 = names[i]
                closeStars = neighbors[names[i]]
                closeStarsInd = neighbors[names[i]+'_ind']

                # open the points files (commented out, already open above)
                # ptFile = workDir + names[i]+'.points'
                # phFile = workDir + names[i]+'.phot'

                # check if this file is empty
                # because this star could be a confusing source before and be removed
                if os.stat(ptFile).st_size == 0:
                    continue
                tab1 = starTables.read_points(ptFile)
                phot1 = starTables.read_points(phFile)

                if len(tab1) > 0:
                    epochs1 = tab1['epoch']
                    bad1_total = []

                    # loop through the stars close to this one
                    for j in range(len(closeStars)):
                        star2 = closeStars[j]
                        ptFile2 = workDir+closeStars[j]+'.points'
                        phFile2 = workDir+closeStars[j]+'.phot'

                        if os.stat(ptFile2).st_size == 0:
                            continue
                        tab2 = starTables.read_points(ptFile2)
                        phot2 = starTables.read_points(phFile2)

                        if len(tab2) > 0:
                            epochs2 = tab2['epoch']

                            # get the range of epochs when they were confused
                            s1 = i
                            s2 = names.index(star2)  # index of close neighbor

                            xfit1 = [x[s1],vx[s1],ax[s1]]
                            xfit2 = [x[s2],vx[s2],ax[s2]]
                            yfit1 = [y[s1],vy[s1],ay[s1]]
                            yfit2 = [y[s2],vy[s2],ay[s2]]
                            epochRange = self.epochDistance(xfit1, yfit1, xfit2, yfit2,
                                                            self.epoch[s1], self.epoch[s2],
                                                            epochLimits, self.confusionThreshold)


                            good1 = np.where((epochs1 >= epochRange[0]) & (epochs1 <= epochRange[1]))[0]
                            good2 = np.where((epochs2 >= epochRange[0]) & (epochs2 <= epochRange[1]))[0]


                            # find the epochs that are not overlapping (the
                            # star was detected in one epoch for one of the
                            # stars, but not the other)

                            # require both stars are detected in more than 10 epochs
                            nepochs_limit = 10
                            if len(epochs1)<nepochs_limit or len(epochs2)<nepochs_limit:
                                missingEpochs = np.array([])
                                missingEpochs1 = np.array([])
                                missingEpochs2 = np.array([])
                            else:
                                missingEpochs = np.setxor1d(epochs1[good1],epochs2[good2])
                                # record the epochs that will be removed in the star1
                                missingEpochs1 = np.intersect1d(missingEpochs, epochs1)
                                # record the epochs that will be removed in the star2
                                missingEpochs2 = np.intersect1d(missingEpochs, epochs2)

                                # don't remove if star2 is star1's fake sources
                                if np.in1d(star1, pri):
                                    fake_t = Table.read('polyfit_2_s/edge/star_' + star1 + '_fake.txt', format='ascii')
                                    fakes = fake_t['fake']
                                    if np.in1d(star2, fakes):
                                        missingEpochs = np.array([])
                                        missingEpochs1 = np.array([])
                                        missingEpochs2 = np.array([])
                                
                                # also don't remove if star1 is star2's fake sources
                                if np.in1d(star2, pri):
                                    fake_t = Table.read('polyfit_2_s/edge/star_' + star2 + '_fake.txt', format='ascii')
                                    fakes = fake_t['fake']
                                    if np.in1d(star1, fakes):
                                        missingEpochs = np.array([])
                                        missingEpochs1 = np.array([])
                                        missingEpochs2 = np.array([])

                            if epochsRemoved is None:
                                epochsRemoved = {star1:missingEpochs1}
                            else:
                                if star1 in epochsRemoved:
                                    epochsRemoved[star1] = \
                                                np.concatenate([epochsRemoved[star1], missingEpochs1])
                                    epochsRemoved[star1] = np.unique(epochsRemoved[star1])
                                else:
                                    epochsRemoved[star1] = missingEpochs1

                            if epochsRemoved is None:
                                epochsRemoved = {star1:missingEpochs1}
                            else:
                                if star2 in epochsRemoved:
                                    epochsRemoved[star2] = \
                                                np.concatenate([epochsRemoved[star2], missingEpochs2])
                                    epochsRemoved[star2] = np.unique(epochsRemoved[star2])
                                else:
                                    epochsRemoved[star2] = missingEpochs2


                            nEpochsRemoved[s1] = len(epochsRemoved[star1])
                            nEpochsRemoved[s2] = len(epochsRemoved[star2])

                            if debug:
                                print( star1, self.mag[s1], star2, self.mag[s2])
                                print( star1, epochs1)
                                print( star2, epochs2)
                                print( 'missing: ', missingEpochs)
                                print( 'all epochs that will be removed from '+star1,epochsRemoved[star1])

                            # Keep track of the culprit star to print( to text file)
                            if len(missingEpochs1) > 0:
                                if culprits is None:
                                    culprits = {star1:np.array([star2])}
                                else:
                                    if star1 in culprits:
                                        culprits[star1] = np.concatenate([culprits[star1], np.array([star2])])
                                        culprits[star1] = np.unique(culprits[star1])
                                    else:
                                        culprits[star1] = np.array([star2])

                            if len(missingEpochs2) > 0:
                                if culprits is None:
                                    culprits = {star2:np.array([star1])}
                                else:
                                    if star2 in culprits:
                                        culprits[star2] = np.concatenate([culprits[star2], np.array([star1])])
                                        culprits[star2] = np.unique(culprits[star2])
                                    else:
                                        culprits[star2] = np.array([star1])


                            # look for the epochs with bad points and remove them
                            bad2_total = []
                            for ee in missingEpochs:
                                bad1 = np.where(epochs1 == ee)[0]
                                bad2 = np.where(epochs2 == ee)[0]
                                if debug:
                                    print( 'bad1 '+star1, bad1)
                                    print( 'bad2 '+star2, bad2)
                                if len(bad1) > 0:
                                    bad1_total.append(bad1[0])
                                if len(bad2) > 0:
                                    bad2_total.append(bad2[0])

                            if len(bad2_total) != 0:
                                tab2.remove_rows(bad2_total)
                                phot2.remove_rows(bad2_total)

                            # write out the points files
                            starTables.write_points(tab2, ptFile2)
                            starTables.write_points(phot2, phFile2)

                    bad1_total = np.unique(np.array(bad1_total))
                    
                    if len(bad1_total) !=0:
                        tab1.remove_rows(bad1_total)
                        phot1.remove_rows(bad1_total)
                    starTables.write_points(tab1, ptFile)
                    starTables.write_points(phot1, phFile)

        # output a file with the number of epochs removed
        print( 'writing: '+ workDir+'epochsRemoved.txt')
        output = open(workDir+'epochsRemoved.txt','w')
        for rr in np.arange(len(nEpochsRemoved)):
            if nEpochsRemoved[rr] > 0:
                epochStr = ' '.join(np.array(epochsRemoved[names[rr]],dtype='str'))+'\n'
                output.write(names[rr] + ' ' + epochStr)
                #outStr = names[rr]+'\t'+str(int(nEpochsRemoved[rr]))+'\t'+epochStr
        output.close()

        # output a file with the stars that caused another star to be confused
        print( 'writing: '+ workDir+'confusingSources.txt')
        output = open(workDir+'confusingSources.txt','w')
        for star in sorted(culprits.keys()):
            epochStr = ' '.join(np.array(culprits[star],dtype='str'))+'\n'
            output.write(star + ' ' + epochStr)
        output.close()


        # run polyfit if needed
        if runPolyfit:
            newPoly = self.rootDir + Poly
            if os.path.isdir(newPoly):
                print('removing exisiting %s directory: ' %newPoly)
                shutil.rmtree(newPoly, ignore_errors=True)
            gcutil.mkdir(newPoly)

            print( 'now running polyfit in '+self.rootDir+Poly)
            cmd = 'polyfit -d 2 -linear -i '+self.rootDir+self.align
            cmd += ' -jackknife -points '+self.rootDir+ Points + ' -o '+self.rootDir+Poly+'fit'
            os.system(cmd)

    def plot_confused_sources(self, input_points, output_points, output_poly='polyfit_3_c/'):
        """
        plot the histogram of confused sources as a function of mag and radius
        plot each confusion sources"""

        if not os.path.exists(output_poly + 'plots_confusion'):
            os.mkdir(output_poly + 'plots_confusion')

        #######
        #plot the histogram of confused sources as a function of mag and radius
        #######
        con_ind = self.confusedSourcesInd
        con_ind = con_ind.astype(int)
        con_r = self.r2d[con_ind]
        con_mag = self.mag[con_ind]
        fig, axes = plt.subplots(1,2,figsize=(20,10))
        axes[0].hist(self.r2d, bins=np.linspace(0,9,20), color='g', label='all')
        axes[0].hist(con_r, bins=np.linspace(0,9,20), color='r', label='confusion')
        axes[0].set_xlabel('r2d(arcsec)')
        axes[0].set_title('confusion limit: %.2f mas, %d mag' %(self.confusionThreshold, self.confusionMagThreshold))
        axes[0].legend()
        axes[1].hist(self.mag, bins=np.linspace(8,20,20), color='g', label='all')
        axes[1].hist(con_mag, bins=np.linspace(8,20,20), color='r', label='confusion')
        axes[1].set_xlabel('mag')
        axes[1].legend()
        plt.savefig(output_poly + 'plots_confusion/confusion_dist.png', format='png')
        plt.close()

        #######
        #plot each confusion sources
        #######
        # read in the acceleration fit result
        x,y = self.x, self.y
        vx, vy, ax, ay = self.vx, self.vy, self.ax, self.ay
        t0 = self.epoch
        names = self.names

        with open(output_points+'/confusingSources.txt', 'r') as f:
            content = f.readlines()

        # define markers and colors
        years = self.allEpochs
        cnorm = colors.Normalize(years.min(), years.max()+1)
        cmap = cm.gist_ncar
        linecolors = []
        for ee in range(len(years)):
            linecolors.append(cmap(cnorm(years[ee])))

        markers = ['o', '^', 'v', 'd', 's', '*']

        for line in content:
            fig = plt.figure(figsize=(10,10))
            axes = plt.subplot(111)
            stars = line.split()

            # plot the polynomial fit
            star = stars[0]
            s = list(names).index(star)
            time = np.arange(years.min(),years.max()+1,0.01)
            dt = time-t0[s]
            xpos = self.poly2([x[s], vx[s], ax[s]], dt)
            ypos = self.poly2([y[s], vy[s], ay[s]], dt)
            axes.plot(xpos, ypos, 'k--')

            # plot the star that is confused by other star
            # plot before confusion
            star_track = np.genfromtxt(input_points+star+'.points')
            time = star_track[:,0]
            idx_color = [list(years).index(i) for i in time]
            x1 = star_track[:,1]*-1.
            y1 = star_track[:,2]
            xe1 = star_track[:,3]
            ye1 = star_track[:,4]
            for i in range(len(x1)):
                axes.errorbar(x1[i], y1[i], xerr=xe1[i], yerr=ye1[i], color=linecolors[idx_color[i]],
                        fmt=markers[0], ms=5, mfc='w', mec=linecolors[idx_color[i]], capsize=0)

            # plot after confusion
            star_track = np.genfromtxt(output_points+star+'.points')
            if os.stat(output_points+star+'.points').st_size == 0:
                x2 = []
            elif len(star_track.shape)==1:
                time = [star_track[0]]
                idx_color = [list(years).index(time)]
                x2 = np.array([star_track[1]])*-1.
                y2 = np.array([star_track[2]])
                xe2 = np.array([star_track[3]])
                ye2 = np.array([star_track[4]])
            else:
                time = star_track[:,0]
                idx_color = [list(years).index(i) for i in time]
                x2 = star_track[:,1]*-1.
                y2 = star_track[:,2]
                xe2 = star_track[:,3]
                ye2 = star_track[:,4]

            for i in range(len(x2)):
                plt.errorbar(x2[i], y2[i], xerr=xe2[i], yerr=ye2[i], color=linecolors[idx_color[i]],
                            fmt=markers[0], ms=5, mec='none',capsize=0)

            idx_star = np.where(self.names==star)[0]
            axes.annotate(star + ': K=%.1f' %(self.mag[idx_star])[0], (x1[x1.argmax()], y1[x1.argmax()]), fontweight='bold')

            # make year annotation
            space = 1./int(years.max()-years.min())
            previous_year = 0
            n_year = 0
            for i in range(len(years)):
                year = years[i]
                if int(previous_year) != int(year):
                    axes.annotate(str(int(year)), (1.03, 0 + n_year*space), color=linecolors[i],
                            xycoords='axes fraction', fontsize=15)
                    previous_year = year
                    n_year += 1
            plt.subplots_adjust(right=0.9)


            # plot confusing sources
            neighbors = self.neighbors[star]
            neighbors_dist = self.neighbors[star+'_dist']
            neighbors_ind = self.neighbors[star+'_ind']
            for i in range(len(stars)-1):
                star = stars[i+1]

                s = list(names).index(star)
                time = np.arange(years.min(),years.max()+1,0.01)
                dt = time-t0[s]
                xpos = self.poly2([x[s], vx[s], ax[s]], dt)
                ypos = self.poly2([y[s], vy[s], ay[s]], dt)
                axes.plot(xpos, ypos, 'k--')

                j = i+1
                while j >= len(markers):
                    j -= len(markers)
                marker = markers[j]

                # plot before confusion
                star_track = np.genfromtxt(input_points+star+'.points')
                time = star_track[:,0]
                idx_color = [list(years).index(i) for i in time]
                x1 = star_track[:,1]*-1.
                y1 = star_track[:,2]
                xe1 = star_track[:,3]
                ye1 = star_track[:,4]
                for i in range(len(x1)):
                    axes.errorbar(x1[i], y1[i], xerr=xe1[i], yerr=ye1[i], color=linecolors[idx_color[i]],
                            fmt=marker, ms=5, mec=linecolors[idx_color[i]], mfc='w', capsize=0)

                # plot after confusion
                star_track = np.genfromtxt(output_points+star+'.points')
                if os.stat(output_points+star+'.points').st_size == 0:
                    x2 = []
                elif len(star_track.shape)==1:
                    time = [star_track[0]]
                    idx_color = [list(years).index(time)]
                    x2 = np.array([star_track[1]])*-1.
                    y2 = np.array([star_track[2]])
                    xe2 = np.array([star_track[3]])
                    ye2 = np.array([star_track[4]])
                else:
                    time = star_track[:,0]
                    idx_color = [list(years).index(i) for i in time]
                    x2 = star_track[:,1]*-1.
                    y2 = star_track[:,2]
                    xe2 = star_track[:,3]
                    ye2 = star_track[:,4]

                for i in range(len(x2)):
                    axes.errorbar(x2[i], y2[i], xerr=xe2[i], yerr=ye2[i], color=linecolors[idx_color[i]],
                            fmt=marker, ms=5, mec='none',capsize=0)

                idx_star = np.where(self.names==star)[0] 
                if len(idx_star)>0:
                    idx_star = idx_star[:1]
                    plt.annotate(star+': K=%.1f' %(self.mag[idx_star]),(x1[x1.argmax()], y1[x1.argmax()]))
                else:
                    pdb.set_trace()

            axes.invert_xaxis()
            axes.set_xlabel('X offset from Sgr A* (arcsec)')
            axes.set_ylabel('Y offset from Sgr A* (arcsec)')
            plt.savefig(output_poly + 'plots_confusion/%s.png' %stars[0], format='png')
            plt.close()

   
    def run_ftest_hpoly(self, make_plot=False, p_crit=0.8):
        # run f test to find the best polynomial order
        # find the first order that makes 1-p smaller than 0.8
        poly_orders = self.poly_orders
        Fx = np.zeros((len(poly_orders)-1, len(self.nEpochs)))
        Fy = np.zeros((len(poly_orders)-1, len(self.nEpochs)))
        chix = np.zeros((len(poly_orders), len(self.nEpochs)))
        chiy = np.zeros((len(poly_orders), len(self.nEpochs)))

        # calculate the F test value for all orders
        for i in range(len(poly_orders)-1):
            Fx[i], Fy[i], chix[i], chiy[i], chix[i+1], chiy[i+1] = self.calculate_f_poly(poly_orders[i], poly_orders[i+1])
        self.Fx_poly = Fx
        self.Fy_poly = Fy
        self.chix_poly = chix
        self.chiy_poly = chiy

        ## find the best fit order
        ## based on the local  minimum
        #best_order = np.zeros(len(self.nEpochs))
        #for i in range(len(best_order)):
        #    best_order[i] = 1
        #    f = F[:,i]
        #    f0 = f[0]
        #    j = 1
        #    while j<poly_orders.max()-1 and f[j] < f0:
        #        f0 = f[j]
        #        best_order[i] += 1
        #        j += 1
        #self.best_poly_order = best_order

        # based on 1-p value
        best_order_pvalue = np.zeros(len(self.nEpochs))
        for i in range(len(self.nEpochs)):
            best_order_pvalue[i] = self.find_best_order(i, p_crit)
        self.best_poly_order_pvalue = best_order_pvalue

        # make plot and calculate best order
        if make_plot:
            for i in range(len(self.nEpochs)):
                self.plot_f_hpoly(i)

        # summarize the distribution of polynomial order
        idx_1order = np.where(self.best_poly_order_pvalue==1)[0]
        idx_2order = np.where(self.best_poly_order_pvalue==2)[0]
        idx_high = np.where(self.best_poly_order_pvalue>2)[0]
        print('among %d stars:' %(len(self.best_poly_order_pvalue)))
        print('%d can be fit by 1st order' %(len(idx_1order)))
        print('%d can be fit by 2nd order' %(len(idx_2order)))
        print('%d need be fit by higher order' %(len(idx_high)))


    def find_best_order(self, idx, p_crit):
        """find the best order of polynomial fit using F test"""
        # read in chi2 at different order
        orders = self.poly_orders
        chix = self.chix_poly[:, idx]
        chiy = self.chiy_poly[:, idx]
        n_epochs = self.nEpochs[idx]

        # calculate Number of parameters at different order 
        n_par = orders+1

        # loop through all the orders: from order X to X+1, calculate f=1-p from f test
        # if f > 0.5, meaning it's improving from X to X+1, so we go to X+1
        # repeat this step until f < 0.5, record X as the best fit order

        i = 0
        for i in range(len(orders)-1):
            if (self.p_value(chix[i], chix[i+1], n_epochs, n_par[i], n_par[i+1]) < p_crit) \
                    and (self.p_value(chiy[i], chiy[i+1], n_epochs, n_par[i], n_par[i+1]) < p_crit):
                return orders[i]
        return orders[i+1]
 
        # the other method
        # loop through all the orders: from order X to X+1, calculate f=1-p from f test
        # if f > 0.5, meaning it's improving from X to X+1, so we go to X+1
        # repeat this step until f < 0.5, record X as the best fit order
        #
        # However, it's not improving from X to X+1, doesn't gaurantee it's not improving from X to X+more
        # because maybe both X and X+1 are poor fit, you need to go higher order to get a good fit
        #
        # So, compare X with other higher orders
        # if f < 0.5 from X to any other higher order, then X is the best fit order
        # if f > 0.5 from X to Y, then record Y as the best fit order and repeat the previou step 
        #
        # until we find an order Z, for which f<0.5 from Z to any other higher order
        #i = 0
        #j = 1

        #while i+j < len(orders):
        #    if self.p_value(chi2[i], chi2[i+j], n_data, n_par[i], n_par[i+j]) > p_crit:
        #        i += j
        #        j = 1
        #    else:
        #        j += 1
        #return  orders[i]


    # use delta(chi2) to do f test
    #def p_value(self, chi1, chi2, n_data, n_par1, n_par2):
    #    f_ratio = (chi1 - chi2) / chi2 * (n_data - n_par2) / (n_par2 - n_par1)
    #    p = stats.f.sf(f_ratio, n_par2-n_par1, n_data-n_par2)
    #    return (1-p)

    # use chi2_1 and chi2_2 to do f test
    def p_value(self, chi1, chi2, n_data, n_par1, n_par2):
        dof1 = n_data - n_par1
        dof2 = n_data - n_par2
        f_ratio = (chi1/dof1) / (chi2/dof2)
        p = stats.f.sf(f_ratio, dof1, dof2)
        idx = np.where(p>1)[0]
        for i in idx:
            p[i] = 1/p[i]
        return (1-p)
        
    def calculate_f_poly(self, order1, order2):
        # read chi2
        poly1 = 'polyfit_5_high_order' + str(order1) +'/fit'
        poly2 = 'polyfit_5_high_order' + str(order2) +'/fit'
        t1 = Table.read(poly1 + '.accelFormal', format='ascii')
        t2 = Table.read(poly2 + '.accelFormal', format='ascii')

        if len(t1) != len(t2):
            sys.exit("not same number of stars for 2nd and 3rd polynomial fit")

        chix_1 =  t1['chiSqX']
        chiy_1 =  t1['chiSqY']
        chix_2 =  t2['chiSqX']
        chiy_2 =  t2['chiSqY']

        # calcualte degree of freedom
        nEpochs =  self.nEpochs
        npar_1 = order1+1
        npar_2 = order2+1

        # F test
        Fx = self.p_value(chix_1, chix_2, nEpochs, npar_1, npar_2)
        Fy = self.p_value(chiy_1, chiy_2, nEpochs, npar_1, npar_2)
        return Fx, Fy, chix_1, chiy_1, chix_2, chiy_2

    def plot_f_hpoly(self, idx):
        # plot the ftest result
        #best_order = self.best_poly_order[idx]
        best_order_pvalue = self.best_poly_order_pvalue[idx]
        poly_orders = self.poly_orders
        norders = len(poly_orders)
        name = self.names[idx]
        Fx_poly = self.Fx_poly[:,idx]
        Fy_poly = self.Fy_poly[:,idx]
        chix = self.chix_poly[:, idx]
        chiy = self.chiy_poly[:, idx]

        # plot how F test value changes
        plt.figure(figsize=(20,10))
        plt.subplot2grid((norders, 3), (0,0), rowspan=norders)
        plt.plot(poly_orders[:-1], Fx_poly, 'g-', label='x')
        plt.plot(poly_orders[:-1], Fy_poly, 'b-', label='y')
        #plt.plot(poly_orders[:-1][int(best_order-1)], F_poly[int(best_order-1)], 'bo', label='best fit from local minimum')
        plt.axvline(x=poly_orders[int(best_order_pvalue-1)], color='r', ls='--')
        #if int(best_order_pvalue) == poly_orders.max():
        #    #plt.plot(poly_orders[int(best_order_pvalue-1)], Fx_poly.min(), 'ro', label='best fit from 1-p value')
        #else:
        #    #plt.plot(poly_orders[int(best_order_pvalue-1)], Fx_poly[int(best_order_pvalue-1)], 'ro', label='best fit from 1-p value')
        #    #plt.plot(poly_orders[int(best_order_pvalue-1)], Fy_poly[int(best_order_pvalue-1)], 'ro', label='best fit from 1-p value')
        plt.xlabel('poly order')
        plt.ylabel('1-p')
        plt.legend()
        plt.title('%s' %name)
        plt.gca().ticklabel_format(useOffset=False)
        plt.xticks(poly_orders)
        plt.tight_layout()

        # plot how polynomial fit looks like
        t_points = Table.read('points_4_trim/' + name + '.points', format='ascii')
        time = t_points['col1']
        x = t_points['col2']
        y = t_points['col3']
        xe = t_points['col4']
        ye = t_points['col5']

        x_subplots = []
        y_subplots = []
        for i in range(len(poly_orders)):
            order = poly_orders[i]
            xfit, yfit = self.fit_poly(order, idx)


            axx = plt.subplot2grid((norders, 3), (i,1))
            plt.errorbar(time, x-xfit, yerr = xe, fmt='.', label='order=%d' %order)
            plt.hlines(0, xmin=time.min(), xmax=time.max())
            plt.legend()
            plt.annotate('chi2=%d' %chix[i], color='r', xy=(0.5, 0.5), xycoords='axes fraction')
            x_subplots.append(axx)

            axy = plt.subplot2grid((norders, 3), (i,2))
            plt.errorbar(time, y-yfit, fmt='.', yerr = ye)
            plt.hlines(0, xmin=time.min(), xmax=time.max())
            plt.annotate('chi=%d' %chiy[i], color='r', xy=(0.5, 0.5), xycoords='axes fraction')
            y_subplots.append(axy)

        axx.set_xlabel('year')
        x_subplots[0].set_title('x-xfit (arcsec)')
        axx.get_shared_x_axes().join(x_subplots[0], x_subplots[1], x_subplots[2], x_subplots[3], x_subplots[4])
        for x_subplot in x_subplots[:-1]:
            x_subplot.set_xticklabels([]) 

        axy.set_xlabel('year')
        y_subplots[0].set_title('y-yfit (arcsec)')
        axy.get_shared_x_axes().join(y_subplots[0], y_subplots[1], y_subplots[2], y_subplots[3], y_subplots[4])
        for y_subplot in y_subplots[:-1]:
            y_subplot.set_xticklabels([]) 

        plt.tight_layout()
        plt.subplots_adjust(hspace=0)
        plt.savefig('plots/poly/' + name + '.png', format='png')

        plt.close()


    def fit_poly(self, order, idx):
        poly = 'polyfit_5_high_order' + str(order) + '/fit'
        # read in t0 info
        t0 = Table.read(poly+'.t0', format='ascii')
        t0x = t0['t0X'][idx]
        t0y = t0['t0Y'][idx]

        # calculate dt 
        name = self.names[idx]
        t_points = Table.read('points_4_trim/' + name + '.points', format='ascii')
        years = t_points['col1']
        dtx = years - t0x
        dty = years - t0y

        # read in polynomial fit
        t = Table.read(poly+'.accelFormal', format='ascii')
        ndata = len(t)
        fitx = np.zeros(len(years))
        fity = np.zeros(len(years))

        # calculate fit result
        for i in range(order+1):
            fitx += t['a_x' + str(i)][idx] * (dtx**i) / math.factorial(i)
            fity += t['a_y' + str(i)][idx] * (dty**i) / math.factorial(i)

        return fitx, fity




    def findNonPhysical(self, plotDist = False, epochsRequired=0.0, verbose=True):
        """
        Print out a list of stars that have non-physical accelerations.

        Return: returns a list of star names that are unphysical
        """

        # get some variables in to local scope
        names = np.array(self.names)
        at, ate = self.at, self.ate
        ar, are = self.ar, self.are
        sigma = self.sigma
        x, y = -self.x, self.y
        vx, vy = self.vx, self.vy
        ax, ay = self.ax, self.ay
        epoch, allEpochs = self.epoch, self.allEpochs
        nEpochs = self.nEpochs  # number of epochs each star was detected
        a2d = self.a2d
        mag = self.mag
        cc = self.cc
        chi2x, chi2y = self.chi2x, self.chi2y
        chi2xv, chi2yv = self.chi2xv, self.chi2yv
        chi2x_red, chi2y_red = chi2x/(nEpochs-3), chi2y/(nEpochs-3)
        chi2xv_red, chi2yv_red = chi2xv/(nEpochs-2), self.chi2yv/(nEpochs-2)
        r2d = self.r2d
        rarc, a2dsim = self.rarc, self.a2dsim
        if self.run_high_order_poly:
            best_order = self.best_poly_order_pvalue
        else:
            best_order = np.zeros(len(names))
        
        ##########
        # Non-physical accelerations.
        ##########

        if not os.path.exists('plots'):
            os.mkdir('plots')
        f = open('plots/accel_sum.txt', 'w')

        # report total number of stars
        print('Found {0} Stars'.format(len(at)))
        print('Found {0} Orbital stars'.format(len(np.where(r2d<=0.5)[0])))
        f.write('Found {0} Stars\n'.format(len(at)))
        f.write('Found {0} Orbital stars\n'.format(len(np.where(r2d<=0.5)[0])))

        # first define all accelerating sources:
        idx_acc = np.where(((abs(at/ate)>5) | (abs(ar/are)>5)) & (r2d>0.5))[0]
        idx_radius = r2d[idx_acc].argsort()
        idx_acc = idx_acc[idx_radius]
        if verbose == True:
            print('Found {0} Significant Acc'.format(len(idx_acc)))
            print('( Significant Acc are: abs(ar/are) > 5 OR abs(at/ate)>5 AND r>0.5as)\n')
            f.write('Found {0} Significant Acc\n'.format(len(idx_acc)))
            f.write('( Significant Acc are: abs(ar/are) > 5 OR abs(at/ate)>5 AND r>0.5as)\n \n')

        # then physical acc
        idx_phy = np.where((ar/are<-5) & (abs(at/ate)<3) & ((a2d-ar)/are<3) & (r2d>0.5))[0]  
        idx_radius = r2d[idx_phy].argsort()
        idx_phy = idx_phy[idx_radius]
        if verbose == True:
            print('Found {0} Significant Physcial Acc stars'.format(len(idx_phy)))
            print('(Significant Physical Acc are: ar/are < -5 AND abs(at/ate)<3 AND (amax-ar)/are<3 AND r>0.5as)')
            print(names[idx_phy], '\n')
            f.write('Found {0} Significant Physical Acc stars\n'.format(len(idx_phy)))
            f.write('(Significant Physical Acc are: ar/are < -5 AND abs(at/ate)<3 AND (amax-ar)/are<3) AND r>0.5as\n')
            f.write(str(names[idx_phy]) + '\n\n')
       
        # the rest will be nonphysical acc
        idx_nonphy = np.setdiff1d(idx_acc, idx_phy)
        idx_radius = r2d[idx_nonphy].argsort()
        idx_nonphy = idx_nonphy[idx_radius]
        if verbose == True:
            print('Found {0} Significant Non-physical Acc stars'.format(len(idx_nonphy)))
            print(' (Significant Non-physical Acc are: Significant Acc - Significant Physical Acc)')
            print(names[idx_nonphy],'\n')
            f.write('Found {0} Significant Non-physical Acc stars\n'.format(len(idx_nonphy)))
            f.write(' (Significant Non-physical Acc are: Significant Acc - Significant Physical Acc)\n')
            f.write(str(names[idx_nonphy]) + '\n\n')
 
        # write good accel, nonphysical accel to txt table
        if verbose == True:
            # print out with increasing radius
            f.write('****** Significant Physical Acc ******\n')
            f.write('   name    mag  radius ar(km/s/yr) are(km/s/yr) at(km/s/yr) ate(km/s/yr) polyfit_order\n')
            for i in idx_phy:
                f.write('%10s %5.1f %4.2f %10.2f %10.2f %10.2f %10.2f %10d\n' 
                        %(names[i], mag[i], r2d[i], ar[i], are[i], at[i], ate[i], best_order[i]))

            f.write('****** Significant Non-physical Acc ******\n')
            f.write('   name    mag  radius ar(km/s/yr) are(km/s/yr) at(km/s/yr) ate(km/s/yr) polyfit_order\n')
            for i in idx_nonphy:
                f.write('%10s %5.1f %4.2f %10.2f %10.2f %10.2f %10.2f %10d\n' 
                        %(names[i], mag[i], r2d[i], ar[i], are[i], at[i], ate[i], best_order[i]))
        f.close()

        # make the plot as needed
        nearestStar = self.nearestStarDist
        if plotDist:
            # plot the histograms
            py.clf()
            py.subplot(221)
            nAll, allbins, patches = py.hist(nearestStar,bins=50)

            nBad, badbins, patches2 = py.hist(nearestStar[idx_nonphy],bins=10)
            py.setp(patches2, 'facecolor','r')
            py.xlabel('Nearest Neighbor (arcsec)')
            py.ylabel('N Stars')


            # do a ks test of the two distributions
            ksTest = stats.ks_2samp(nAll, nBad)
            print( 'K-S test of the distribution of nearest star in whole sample vs. in non-physical sources')
            print( ksTest)

            py.subplot(222)
            py.plot(rarc,a2dsim)
            py.plot([0,10],[0,0],hold=True)

            #plot(r2d[bad],ar[bad],are[bb],'ro')
            py.errorbar(r2d[idx_nonphy],ar[idx_nonphy],are[idx_nonphy],fmt='ro')
            py.xlim(0,5)
            py.ylim(np.max(ar[idx_nonphy]),np.min(ar[idx_nonphy]))
            py.xlabel('Distance from Sgr A* (arcsec)')
            py.ylabel('Acceleration (km/s/yr)')


            py.subplot(223)
            # look at the distribution of chi-squares for stars brighter than 16 magnitude
            good = np.where(mag < 16.0)

            # use mean degree of freedom:
            dof = np.floor(np.mean(self.nEpochs)-3)

            print( 'Mean degree of freedom %f' % dof)

            n, bins, patches1 = py.hist(chi2x[good], bins = np.arange(0, 20, 0.1),
                                     normed=1,label='x',alpha=0.6,color='blue')
            n, bins, patches2 = py.hist(chi2y[good], bins = bins, normed=1,
                                     label='y',alpha=0.6,color='green')
            py.xlabel('Reduced Chi^2')

            chiInd = np.arange(0.01,100,0.01)
            chi2Theory = stats.chi2.pdf(chiInd,dof)
            py.plot(chiInd, chi2Theory)

            py.legend()
            py.subplot(224)
            n, bins, patches1 = py.hist(chi2x[idx_nonphy], bins = np.arange(0,20,0.5), normed=1,label='x',alpha=0.6)
            n, bins, patches2 = py.hist(chi2y[idx_nonphy], bins = bins,normed=1,label='y',alpha=0.6)
            py.xlabel('Reduced Chi^2')
            py.plot(chiInd, chi2Theory)

            py.legend()

        # update list of physical stars
        self.goodAccel = idx_phy
        self.physical = names[idx_phy]
        # update list of non-physical stars
        self.bad = idx_nonphy  # indices of bad points
        returnNames = names[idx_nonphy]
        self.nonPhysical = returnNames
        self.nearestNonPhysical = nearestStar[idx_nonphy]
        return returnNames

    def analyze_acc(self):
        idx_1order = np.where(self.best_poly_order_pvalue==1)[0]
        idx_2order = np.where(self.best_poly_order_pvalue==2)[0]
        idx_high = np.where(self.best_poly_order_pvalue>2)[0]

        # in nonphysical accel
        idx_nonphy = self.bad
        idx_nonphy_1order = np.intersect1d(idx_nonphy, idx_1order)
        idx_nonphy_2order = np.intersect1d(idx_nonphy, idx_2order)
        idx_nonphy_high = np.intersect1d(idx_nonphy, idx_high)
        
        print('among %d nonphysical acceleration' %(len(idx_nonphy)))
        print('%d need 1st order poly fit' %(len(idx_nonphy_1order)))
        print(self.names[idx_nonphy_1order])

        print('%d need 2nd order poly fit' %(len(idx_nonphy_2order)))
        print(self.names[idx_nonphy_2order])

        print('%d need more then 2nd order poly fit' %(len(idx_nonphy_high)))
        print(self.names[idx_nonphy_high])
        print('\n')
        
        # in accel
        idx_acc = self.goodAccel
        idx_acc_1order = np.intersect1d(idx_acc, idx_1order)
        idx_acc_2order = np.intersect1d(idx_acc, idx_2order)
        idx_acc_high = np.intersect1d(idx_acc, idx_high)
        
        print('among %d  acceleration' %(len(idx_acc)))
        print('%d need 1st order poly fit' %(len(idx_acc_1order)))
        print(self.names[idx_acc_1order])

        print('%d need 2nd order poly fit' %(len(idx_acc_2order)))
        print(self.names[idx_acc_2order])

        print('%d need more then 2nd order poly fit' %(len(idx_acc_high)))
        print(self.names[idx_acc_high])
        print('\n')

        # in 2nd order poly
        print('among %d stars that should be fit by 2nd order polynomial fit' %(len(idx_2order)))
        print('%d are accel stars' %(len(idx_acc_2order)))
        print('%d are nonphy accel stars' %(len(idx_nonphy_2order)))
        print('the rest of them are:')
        idx_2order_other = np.setdiff1d(idx_2order, np.union1d(idx_acc, idx_nonphy))
        print(self.names[idx_2order_other])
        print('\n')
 
        # look at the a/ae for idx_2order_other
        from astropy.table import Table
        t = Table.read(os.path.join(self.rootDir,'polyfit_4_trim/accel.txt'), format='ascii')
        sigma = []
        for star in self.names[idx_2order_other]:
            idx = np.where(t['name']==star)[0]
            ar = t[idx]['ar']
            are = t[idx]['are']
            at = t[idx]['at']
            ate = t[idx]['ate']
            sigma.append(float(max(abs(ar/are), abs(at/ate))))
        

        # for all accel and nonphyscial accel stars, find the (1-p) value at best fit order
        p = []
        for i in idx_acc:
            best_order = int(self.best_poly_order_pvalue[i])
            px = self.Fx_poly[best_order-2, i]
            py = self.Fy_poly[best_order-2, i]
            p.append(max(px, py))

        for i in idx_nonphy:
            best_order = int(self.best_poly_order_pvalue[i])
            if best_order == 1:
                continue
            px = self.Fx_poly[best_order-2, i]
            py = self.Fy_poly[best_order-2, i]
            p.append(max(px, py))


    def findMismatch(self, starName = 'irs16NE', mkplots = True, bootStrap = False, iterate = False, useAlign = False):
        """Check the list of unphysical stars to find which of them
        are mismatches. This will be done by looking at fits with
        large chisquare values in the acceleration and removing points
        that are causing the deviation to see if the chi-square
        improves.

        KEYWORDS: useAlign - use the points from the align file (by
        default uses the points from the points directory)
        """

         # threshold in chi-square to check for mismatches
        chiThreshold = self.chiThreshold

        # find the star and get its position
        names = list(self.names)
        ind = names.index(starName)

        if useAlign:
            x = -self.starSet.stars[ind].getArrayAllEpochs('x')
            y = self.starSet.stars[ind].getArrayAllEpochs('y')
            # positional errors
            xerr_p = self.starSet.stars[ind].getArrayAllEpochs('xerr_p')
            yerr_p = self.starSet.stars[ind].getArrayAllEpochs('yerr_p')

            # alignment errors
            xerr_a = self.starSet.stars[ind].getArrayAllEpochs('xerr_a')
            yerr_a = self.starSet.stars[ind].getArrayAllEpochs('yerr_a')

            # add the two errors in quadrature
            xerr = np.sqrt(xerr_p**2 + xerr_a**2)
            yerr = np.sqrt(yerr_p**2 + yerr_a**2)
        else:
            x = -self.starSet.stars[ind].getArrayAllEpochs('pnt_x')
            y = self.starSet.stars[ind].getArrayAllEpochs('pnt_y')
            # positional+alignment errors
            xerr = self.starSet.stars[ind].getArrayAllEpochs('pnt_xe')
            yerr = self.starSet.stars[ind].getArrayAllEpochs('pnt_ye')


        t0 = self.starSet.stars[ind].fitXa.t0    # use the same T0 as polyfit
        years = np.array(self.starSet.stars[ind].years)

        good = (np.where(abs(x) < 500))[0]
        if (len(good) > 1):
            x = x[good]
            y = y[good]
            xerr = xerr[good]
            yerr = yerr[good]
            years = years[good]

        print( 'x ', x)
        print( 'xerr ', xerr)
        print( 'y ', y)
        print( 'y err', yerr)
        functargsX = {'x':years-t0, 'y':x, 'err': xerr}
        functargsY = {'x':years-t0, 'y':y, 'err': yerr}

        p0x = [self.x[ind],(np.amax(x) - np.amin(x))/(np.amax(years)-np.amin(years)), 0]
        p0y = [self.y[ind],(np.amax(y) - np.amin(y))/(np.amax(years)-np.amin(years)), 0]

        #clf()
        #plot(x,y,'o')
        #py.errorbar(x,y,xerr,yerr,fmt='o')
        #do the fitting
        xfit = nmpfit.mpfit(self.fitfunPoly2, p0x, functkw=functargsX,quiet=1)
        yfit = nmpfit.mpfit(self.fitfunPoly2, p0y, functkw=functargsY,quiet=1)
        dof = len(x) - len(xfit.params)
        xredChiSq = np.sqrt(xfit.fnorm/dof)
        yredChiSq = np.sqrt(yfit.fnorm/dof)

        print( xredChiSq)
        print( yredChiSq)
        print( xfit.fnorm)
        print( yfit.fnorm)

        simTime = np.arange(np.amin(years)-t0,np.amax(years)+0.1-t0,0.1)
        simX = self.poly2(xfit.params,simTime)
        simY = self.poly2(yfit.params,simTime)
        #plot(simX,simY)

        modelAccelX = self.poly2(xfit.params, years - t0)
        modelAccelY = self.poly2(yfit.params, years - t0)

        xfitChi2 = np.sum((modelAccelX - x)**2/xerr**2.0)
        yfitChi2 = np.sum((modelAccelY - y)**2/yerr**2.0)
        print( 'xfitchi2 %f' % xfitChi2)
        print( 'yfitchi2 %f' % yfitChi2)
        # fit a line
        guessX =  [self.x[ind],(np.amax(x) - np.amin(x))/(np.amax(years)-np.amin(years))]
        guessY =  [self.y[ind],(np.amax(y) - np.amin(y))/(np.amax(years)-np.amin(years))]
        xfitLin = nmpfit.mpfit(self.fitLin, guessX, functkw=functargsX,quiet=1)
        yfitLin = nmpfit.mpfit(self.fitLin, guessY, functkw=functargsY,quiet=1)


        simXLin = self.line(xfitLin.params,simTime)
        simYLin = self.line(yfitLin.params,simTime)
        #plot(simXLin,simYLin)

        # if the accleration fit has too large a chi-square, go
        # through and remove points that are bad. then refit until the
        # chi-square goes below the threshold

        # initialize the chi square values
        loopChiX = xfit.fnorm/dof
        loopChiY = yfit.fnorm/dof

        loopLinChiX = xfitLin.fnorm/(dof + 1.0)
        loopLinChiY = yfitLin.fnorm/(dof + 1.0)
        xPoints = x
        xPointsErr = xerr
        yPoints = y
        yPointsErr = yerr
        subTime = years - t0


        print( xfit.params)
        print( yfit.params)
        x0 = self.poly2(xfit.params, 0.0)
        y0 = self.poly2(yfit.params, 0.0)
        print( 'running isPhyiscal on polyfit results')
        print( self.isPhysical(self.x[ind],self.y[ind],self.ax[ind],self.axe[ind],self.ay[ind],self.aye[ind]))

        print( 'running isPhysical on fit: ')
        accelStatus = self.isPhysical(x0, y0,xfit.params[2],xfit.perror[2],
                                      yfit.params[2], yfit.perror[2], arcsec = True)
        print( accelStatus)
        print( 't0: ', t0)
        print( 'from poly fit')
        print( 'x0, vx, ax:')
        print( [self.x[ind],self.vx[ind],self.ax[ind]])
        print( [self.xerr[ind],self.vxe[ind],self.axe[ind]])
        print( 'y0, yx, yx:')
        print( [self.y[ind],self.vy[ind],self.ay[ind]])
        print( [self.yerr[ind],self.vye[ind],self.aye[ind]])
##         print( '[x, y, ax, ay]')
##         print( [self.x[ind],self.y[ind],self.ax[ind],self.ay[ind]])
##         print( '[xerr, yerr, axerr, ayerr]')
##         print( [self.xerr[ind],self.yerr[ind],self.axe[ind],self.aye[ind]])
        print( 'from current accel fit')
        print( 'xfit: ')
        print( xfit.params)
        print( xfit.perror)
        print( 'yfit: ')
        print( yfit.params)
        print( yfit.perror)

        print( [x0, y0, xfit.params[2], yfit.params[2]])
        #yfit.perror = yfit.perror*np.sqrt(yfit.fnorm/dof)
        #xfit.perror = xfit.perror*np.sqrt(xfit.fnorm/dof)
        print( [xfit.perror[0], yfit.perror[0], xfit.perror[2], yfit.perror[2]])
        print( np.sqrt(xfit.perror[2]**2+yfit.perror[2]**2))
        print( accelStatus)

        print( 'Start chi2x: %7.3f chi2y: %7.3f' % (loopChiX, loopChiY))
        print( 'Start linear chi2x: %7.3f chi2y: %7.3f' % (loopLinChiX, loopLinChiY))

        # loop until a good chi2 is obtained or if the number of points goes below five
        while (((loopChiX > chiThreshold) | (loopChiY > chiThreshold)) & (len(xPoints) > 4)):

            # look for the difference between model and data
            xModel = self.poly2(xfit.params, subTime)
            yModel = self.poly2(yfit.params, subTime)
            xDiff = (xModel - xPoints)/xPointsErr
            yDiff = (yModel - yPoints)/yPointsErr
            totalDiff = np.sqrt(xDiff**2 + yDiff**2)

            xLinModel = self.line(xfitLin.params, subTime)
            yLinModel = self.line(yfitLin.params, subTime)
            xLinDiff = (xModel - xPoints)/xPointsErr
            yLinDiff = (yModel - yPoints)/yPointsErr
            totalLinDiff = np.sqrt(xLinDiff**2 + yLinDiff**2)

            # find the maximum difference
            maxInd = np.argmax(totalDiff)
            print( 'Epoch %7.3f is off by %7.3f sigma' % (subTime[maxInd]+t0, totalDiff[maxInd]))

            # look at how the fit imporves if we remove either end point
            xPointsEnd = xPoints[0:-1]
            xPointsErrEnd = xPointsErr[0:-1]
            yPointsEnd = yPoints[0:-1]
            yPointsErrEnd = yPointsErr[0:-1]
            subTimeEnd = subTime[0:-1]

            functargsXEnd = {'x':subTimeEnd, 'y':xPointsEnd, 'err': xPointsErrEnd}
            functargsYEnd = {'x':subTimeEnd, 'y':yPointsEnd, 'err': yPointsErrEnd}

            p0xEnd = [xPointsEnd[0],(np.amax(xPointsEnd) - np.amin(xPointsEnd))/(np.amax(subTimeEnd)-np.amin(subTimeEnd)), 0]
            p0yEnd = [yPointsEnd[0],(np.amax(yPointsEnd) - np.amin(yPointsEnd))/(np.amax(subTimeEnd)-np.amin(subTimeEnd)), 0]

            # do the fitting
            xfitEnd = nmpfit.mpfit(self.fitfunPoly2, p0xEnd, functkw=functargsXEnd,quiet=1)
            yfitEnd = nmpfit.mpfit(self.fitfunPoly2, p0yEnd, functkw=functargsYEnd,quiet=1)
            dofEnd = len(xPointsEnd) - len(xfitEnd.params)
            loopChiXEnd = xfitEnd.fnorm/dofEnd
            loopChiYEnd = yfitEnd.fnorm/dofEnd


            # remove max difference and refit
            xPoints = np.delete(xPoints, maxInd)
            xPointsErr = np.delete(xPointsErr, maxInd)
            yPoints = np.delete(yPoints, maxInd)
            yPointsErr = np.delete(yPointsErr, maxInd)
            subTime = np.delete(subTime,maxInd)

            functargsX = {'x':subTime, 'y':xPoints, 'err': xPointsErr}
            functargsY = {'x':subTime, 'y':yPoints, 'err': yPointsErr}

            p0x = [xPoints[0],(np.amax(xPoints) - np.amin(xPoints))/(np.amax(subTime)-np.amin(subTime)), 0]
            p0y = [yPoints[0],(np.amax(yPoints) - np.amin(yPoints))/(np.amax(years)-np.amin(years)), 0]

            # do the fitting
            xfit = nmpfit.mpfit(self.fitfunPoly2, p0x, functkw=functargsX,quiet=1)
            yfit = nmpfit.mpfit(self.fitfunPoly2, p0y, functkw=functargsY,quiet=1)
            dof = len(xPoints) - len(xfit.params)
            loopChiX = xfit.fnorm/dof
            loopChiY = yfit.fnorm/dof

            # fit a line
            guessX =  [self.x[ind],(np.amax(x) - np.amin(x))/(np.amax(years)-np.amin(years))]
            guessY =  [self.y[ind],(np.amax(y) - np.amin(y))/(np.amax(years)-np.amin(years))]
            xfitLin = nmpfit.mpfit(self.fitLin, guessX, functkw=functargsX,quiet=1)
            yfitLin = nmpfit.mpfit(self.fitLin, guessY, functkw=functargsY,quiet=1)

            loopLinChiX = xfitLin.fnorm/(dof + 1.0)
            loopLinChiY = yfitLin.fnorm/(dof + 1.0)

            print( 'New chi2x: %7.3f chi2y: %7.3f' % (loopChiX, loopChiY))
            print( 'New linear chi2x: %7.3f chi2y: %7.3f' % (loopLinChiX, loopLinChiY))
            print( 'New chi2x without end point: %7.3f chi2y: %7.3f' % (loopChiXEnd, loopChiYEnd))



        x0 = self.poly2(xfit.params, 0.0)
        y0 = self.poly2(yfit.params, 0.0)
        accelStatus = self.isPhysical(x0, y0,xfit.params[2],xfit.perror[2],
                                      yfit.params[2], yfit.perror[2], arcsec = True)
        print( accelStatus)

        if mkplots:
            mTime = np.arange(np.min(subTime), np.max(subTime), 0.05)
            xModel = self.poly2(xfit.params, mTime)
            yModel = self.poly2(yfit.params, mTime)

            xLinModel = self.line(xfitLin.params, mTime)
            yLinModel = self.line(yfitLin.params, mTime)
            clf()
            py.errorbar(x, y, xerr, yerr, fmt = 'o',color='red')
            py.errorbar(xPoints, yPoints, xPointsErr, yPointsErr, fmt = 'o',color='blue')
            plot(xModel, yModel,color='red')
            plot(xLinModel, yLinModel, color = 'green')
            xlim(np.max(xPoints+xPointsErr), np.min(xPoints-xPointsErr))
            title(starName)

            # plot the original fit

            for ii in range(len(x)):
                text(x[ii],y[ii], years[ii])

            # find the median point and find the velocity between each point
            d1 = np.sqrt(x**2 + y**2)
            d1Err = np.sqrt((x**2*xerr**2 + y**2*yerr**2)/d1**2)
            medianInd = realmedian.realMedian(d1,index=True)
            plot(x[medianInd],y[medianInd],'co')

            # find distance from the median point
            d1 = np.sqrt((x-x[medianInd])**2+(y-y[medianInd])**2)

            d2 = np.delete(d1, medianInd)
            d2Err = np.delete(d1, medianInd)
            d2Year = np.delete(years, medianInd)

            d1 = d1[medianInd]
            d1Err = d1Err[medianInd]
            d1Year = years[medianInd]
            for i in range(len(d2)):
                t = array([d1Year, d2Year[i]]) - t0
                d3 = array([d1, d2[i]])
                d3err = array([d1Err, d2Err[i]])

                # sort the time so that all points will fit in one direction
                vFit = self.fitVelocity(t, d3, d3err)
                #print( d2Year[i])
                #print( np.abs(vFit.params[1]))

        if bootStrap:
            self.bootstrapVel(x, y, xerr, yerr, years, t0, iterate=iterate)

    def fitVelocity(self, x, y,  yerr):
        """ takes in an array of x and y position as well as time and
        fit for the velocity. Will return an mpfit object
        """
        guessY =  [y[0],(np.amax(y) - np.amin(y))/(np.amax(x)-np.amin(x))]
        functargsY = {'x':x, 'y':y, 'err': yerr}
        yfitLin = nmpfit.mpfit(self.fitLin, guessY, functkw=functargsY,quiet=1)

        return yfitLin

    def fitAccel(self, x, y,  yerr):
        """ takes in an array of x and y position as well as time and
        fit for the acceleration. Will return an mpfit object
        """
        guessY =  [y[0],(np.amax(y) - np.amin(y))/(np.amax(x)-np.amin(x)), 0]
        functargsY = {'x':x, 'y':y, 'err': yerr}
        yfitLin = nmpfit.mpfit(self.fitfunPoly2, guessY, functkw=functargsY,quiet=1)

        return yfitLin

    #def fitJerk(self, x, y,  yerr):
    #    """ takes in arrays for x and y position and time and
    #    fit for jerk. Will return an mpfit object
    #    """
    #    guessY =  [y[0],(np.amax(y) - np.amin(y))/(np.amax(x)-np.amin(x)), 0, 0]
    #    functargsY = {'x':x, 'y':y, 'err': yerr}
    #    yfitLin = nmpfit.mpfit(self.fitfunPoly3, guessY, functkw=functargsY,quiet=1)

    #    return yfitLin

    def bootstrapVel(self, xInput, yInput, xerrInput, yerrInput, yearsInput,
                     t0, n = 500, fraction = 0.05, iterate = False):
        """ Do a half-sample bootstrap of the velocity fits to
        determine if there are incorrect points.

        KEYWORDS: n = 500 - number of bootstraps to do
                  fraction = 0.05 - fraction of times a point is detected
                                    to be kept
        """
        xStack = np.zeros(n)
        yStack = np.zeros(n)
        velx = np.zeros(n)
        velxErr = np.zeros(n)
        velxChi2 = np.zeros(n)
        vely = np.zeros(n)
        velyErr = np.zeros(n)
        velyChi2 = np.zeros(n)
        drawStack = np.zeros((len(xInput)/2, n))

        # degree of freedom for each half sample boot strap
        bootStrapDOF = len(xInput)/2.0 - 2.0
        chiThreshold = bootStrapDOF*3.0
        print( 'boot strap chi2 threshold: ', chiThreshold)
        xMedianErr = realmedian.realMedian(xerrInput)
        yMedianErr = realmedian.realMedian(yerrInput)

        for i in range(n):
            # draw N/2 indices without replacement
            draw = random.sample(np.arange(len(xInput)), len(xInput)/2)
            drawStack[:,i] = draw
            x = xInput[draw]
            y = yInput[draw]

            # make all the errors the median error (won't really work
            # because this will make points that are badly measured
            # have low errors)

            # xerr = np.zeros(len(xInput)/2)+xMedianErr
            # yerr = np.zeros(len(xInput)/2)+yMedianErr

            xerr = xerrInput[draw]
            yerr = yerrInput[draw]

            years = yearsInput[draw]-t0
            #print( '%f xerr' % i)
            #print( x)
            #print( xerr)
            #print( years)
            xfit = self.fitVelocity(years, x, xerr)
            yfit = self.fitVelocity(years, y, yerr)

            dof = len(x) - len(xfit.params)
            xredChiSq = xfit.fnorm
            yredChiSq = yfit.fnorm

            #print( xfit.params)
            xStack[i] = xfit.params[0]
            yStack[i] = yfit.params[0]
            velx[i] = xfit.params[1]
            #velxErr[i] = xfit.perror[1]

            velxChi2[i] = xredChiSq
            vely[i] = yfit.params[1]
            #velyErr[i] = yfit.perror[1]
            velyChi2[i] = yredChiSq

        # compute the tangential and radial directions to the velocity
        xMed = realmedian.realMedian(xStack)
        yMed = realmedian.realMedian(yStack)
        r = np.sqrt(xMed**2 + yMed**2)
        vr = ((velx*xMed) + (vely*yMed))/r
        vt = ((velx*yMed) - (vely*xMed))/r
        clf()

        subplot(231)
        mVr = np.mean(vr)
        sVr = np.std(vr)
        mVt = np.mean(vt)
        sVt = np.std(vt)

        binWidth = (np.max(vr) - np.min(vr))/20.0  # make sure there are at least 10 bins

        nbins, bins, patches1 = hist(vr, bins = np.arange(np.min(vr), np.max(vr), binWidth))
        xlabel('Vr')
        ylabel('N stars')
        subplot(232)
        binWidth = (np.max(vt) - np.min(vt))/20.0
        #nbins, bins, patches1 = hist(vt, bins = np.arange(mVt - sVt*10.0, mVt+sVt*10.0, 0.0001))
        nbins, bins, patches1 = hist(vt, bins = np.arange(np.min(vt), np.max(vt), binWidth))
        xlabel('Vt')
        ylabel('N stars')
        subplot(233)
        print( 'min chi2 x: %6.3f, max chi2 x: %6.3f' % (np.min(velxChi2), np.max(velxChi2)))
        nbins, bins, patches1 = hist(velxChi2, bins = np.arange(0, self.chiThreshold, self.chiThreshold/30.0))
        xlabel('X Vel. Chi-Sq')
        ylabel('N Stars')
        subplot(234)
        nbins, bins, patches1 = hist(velyChi2, bins = np.arange(0, self.chiThreshold, self.chiThreshold/30.0))
        print( 'min chi2 y: %6.3f, max chi2 y: %6.3f' % (np.min(velyChi2), np.max(velyChi2)))
        xlabel('Y Vel. Chi-Sq')
        ylabel('N Stars')
        subplot(235)
        good = np.where((velxChi2 < chiThreshold) & (velyChi2 < chiThreshold))[0]
        subStack = drawStack[:,good]
        nbins, bins, patches = hist(subStack.flatten(),bins = np.arange(len(xInput)+1))
        xlabel('Epoch')
        ylabel('Times Used')
        print( bins)
        py.xticks(np.arange(len(xInput)), yearsInput, rotation = 45)
        #nbins, bins, patches1 = hist(velyChi2, bins = np.arange(0, 100, 0.5))
        nbins = np.array(nbins,dtype='float')
        nbins = nbins/np.sum(nbins)
        print( nbins)

        # select out all the bad points that aren't used often. Take
        # out the points that are found in fewer than a fraction of
        # the maximum bin. This is done so that if stars are found in
        # all epochs, there could potentially be a case where all the
        # points will be dropped.
        bad = np.where(nbins < fraction*np.max(nbins))[0]

        # recompute the fits without the bad points
        print( 'dropping ', yearsInput[bad])
        x = np.delete(xInput, bad)
        xerr = np.delete(xerrInput, bad)
        y = np.delete(yInput, bad)
        yerr = np.delete(yerrInput, bad)
        years = np.delete(yearsInput, bad)-t0
        print( years)
        print( x)
        xfit = self.fitVelocity(years, x, xerr)
        yfit = self.fitVelocity(years, y, yerr)
        subplot(236)
        py.errorbar(xInput, yInput, xerrInput, yerrInput, fmt='ro')
        py.errorbar(x,y,xerr,yerr,fmt='o')
        plot(self.line(xfit.params,years),self.line(yfit.params,years))
        xlabel('RA offset (arcsec)')
        ylabel('DEC offset (arcsec)')
        velChi2x = xfit.fnorm
        velChi2y = yfit.fnorm
        print( 'xfit', xfit.params)
        print( 'xfit error: ', xfit.perror)
        print( 'xfit chi2: ', xfit.fnorm)
        print( 'yfit', yfit.params)
        print( 'yfit error: ', yfit.perror)
        print( 'yfit chi2: ', yfit.fnorm)
        xfit = self.fitAccel(years, x, xerr)
        yfit = self.fitAccel(years, y, yerr)
        print( 'acceleration fits:')
        print( 'xfit', xfit.params)
        print( 'xfit error: ', xfit.perror)
        print( 'xfit chi2: ', xfit.fnorm)
        print( 'yfit', yfit.params)
        print( 'yfit error: ', yfit.perror)
        print( 'yfit chi2: ', yfit.fnorm)
        accelChi2x = xfit.fnorm
        accelChi2y = yfit.fnorm

        # the F-test for which model to prefer
        fValue = ((velChi2x - accelChi2x)/1.0)/(accelChi2x/(len(x)-3.0))
        print( 'F-test value X fit', fValue)
        print( 'F-test probability: ', stats.f.sf(fValue, 1, len(x) -3.0))
        fValue = ((velChi2y - accelChi2y)/1.0)/(accelChi2x/(len(x)-3.0))
        print( 'F-test value Y fit', fValue)
        print( 'F-test probability: ', stats.f.sf(fValue, 1, len(x) -3.0))

        if iterate:
            self.bootstrapVel(x, y, xerr, yerr, years+t0,
                         t0, n = 500, fraction = 0.05, iterate = False)



    def computeAccel(self):
        """Recomputes the velocity and acceleration of all sources and
        updates those fields as well as the chi2 values. This will use
        the internal fitter instead of the one from polyfit
        """
        # get variables into local scope
        cc = self.cc
        names = list(self.names)
        x, y = self.x, self.y
        xerr, yerr = self.xerr, self.yerr
        vx, vy = self.vx, self.vy

        # loop through the names and get the points for the acceleration fits
        for i in range(len(names)):
            ind = names.index(names[i])

            x = self.starSet.stars[ind].getArrayAllEpochs('x')
            y = self.starSet.stars[ind].getArrayAllEpochs('y')
            xerr_p = self.starSet.stars[ind].getArrayAllEpochs('xerr_p')
            yerr_p = self.starSet.stars[ind].getArrayAllEpochs('yerr_p')

            xerr_a = self.starSet.stars[ind].getArrayAllEpochs('xerr_a')
            yerr_a = self.starSet.stars[ind].getArrayAllEpochs('yerr_a')

            xerr = np.sqrt(xerr_p**2 + xerr_a**2)
            yerr = np.sqrt(yerr_p**2 + yerr_a**2)

            t0 = self.starSet.stars[ind].fitXa.t0    # use the same T0 as polyfit
            years = np.array(self.starSet.stars[ind].years)

            good = (np.where(abs(x) < 500))[0]

            if (len(good) > 3):
                x = x[good]
                y = y[good]
                xerr = xerr[good]
                yerr = yerr[good]
                years = years[good]

                #functargsX = {'x':years-t0, 'y':x, 'err': xerr}
                #functargsY = {'x':years-t0, 'y':y, 'err': yerr}

                #p0x = [self.x[ind],(np.amax(x) - np.amin(x))/(np.amax(years)-np.amin(years)), 0]
                #p0y = [self.y[ind],np.amax(y) - np.amin(y)/(np.amax(years)-np.amin(years)), 0]
                #xfit = nmpfit.mpfit(self.fitfunPoly2, p0x, functkw=functargsX,quiet=1)
                #yfit = nmpfit.mpfit(self.fitfunPoly2, p0y, functkw=functargsY,quiet=1)
                xfit = self.fitAccel(years - t0, x, xerr)
                yfit = self.fitAccel(years - t0, y, yerr)
                dof = len(x) - len(xfit.params)
                if names[i] in ['irs29N']:
                    print( 'yerr_a: ', yerr_a)
                    print( 'yerr_p: ', yerr_p)
                    print( 'yerr: ', yerr)
                # put the results in the correct star
                if names[i] in self.nonPhysical:
                    print( '%s    \t xAccel: %10.8f xPolyFit: %10.8f Diff: %10.8f  xErr: %10.8f polyErr: %10.8f ErrDiff: %10.8f' % (names[i], xfit.params[2], self.ax[i],
                                                                                               xfit.params[2] - self.ax[i], xfit.perror[2], self.axe[i],
                                                                                                                                self.vxe[i]/xfit.perror[1]))
                self.x[i] = xfit.params[0]
                self.ax[i] = xfit.params[2]
                self.axe[i] = xfit.perror[2]
                self.vx[i] = xfit.params[1]
                self.vxe[i] = xfit.perror[1]

                self.y[i] = yfit.params[0]
                self.ay[i] = yfit.params[2]
                self.aye[i] = yfit.perror[2]
                self.vy[i] = yfit.params[1]
                self.vye[i] = yfit.perror[1]
        self.updateAccel() # update the radial and tangential acceleration calculations


    def computeFTest(self):
        """ Go through the acceleration and velocity chi-squares to
        compute the F test values.
        """
        #print( 'computing F-Test for all valid stars ')
        chi2xAccel = self.chi2x
        chi2yAccel = self.chi2y
        chi2xVel = self.chi2xv
        chi2yVel = self.chi2yv
        nEpochs = self.nEpochs

        xFValue = np.zeros(len(chi2xAccel))
        yFValue = np.zeros(len(chi2xAccel))
        xFProb = np.zeros(len(chi2xAccel))
        yFProb = np.zeros(len(chi2xAccel))
        dof = np.zeros(len(chi2xAccel))

        good = np.where(chi2xAccel > 0)[0]
        dof[good] = nEpochs[good] - 3.0  # dof freedom for acceleration

        xFValue[good] = ((chi2xVel[good] - chi2xAccel[good])/1.0)/(chi2xAccel[good]/dof[good])
        yFValue[good] = ((chi2yVel[good] - chi2yAccel[good])/1.0)/(chi2yAccel[good]/dof[good])

        xFProb[good] = stats.f.sf(xFValue[good], 1, dof[good])
        yFProb[good] = stats.f.sf(yFValue[good], 1, dof[good])

        self.xFProb = xFProb
        self.yFProb = yFProb


    #def computeFTestJerk(self):
    #    """ Go through the jerk and acceleration chi-squares to
    #    compute the F test values.
    #    """
    #    print( 'computing F-Test for all valid stars ')
    #    chi2xAccel = self.chi2x
    #    chi2yAccel = self.chi2y
    #    chi2xJerk = self.chi2xj
    #    chi2yJerk = self.chi2yj
    #    nEpochs = self.nEpochs
    #    
    #    xFValue = np.zeros(len(chi2xAccel))
    #    yFValue = np.zeros(len(chi2xAccel))
    #    xFProb = np.zeros(len(chi2xAccel))
    #    yFProb = np.zeros(len(chi2xAccel))
    #    dof = np.zeros(len(chi2xAccel))
    #    
    #    good = np.where(chi2xJerk > 0)[0]
    #    dof[good] = nEpochs[good] - 4.0  # dof freedom for jerk 
    #    
    #    xFValue[good] = ((chi2xAccel[good] - chi2xJerk[good])/1.0)/(chi2xJerk[good]/dof[good])
    #    yFValue[good] = ((chi2yAccel[good] - chi2yJerk[good])/1.0)/(chi2yJerk[good]/dof[good])

    #    xFProb[good] = stats.f.sf(xFValue[good], 1, dof[good])
    #    yFProb[good] = stats.f.sf(yFValue[good], 1, dof[good])

    #    self.xFProbJrk = xFProb
    #    self.yFProbJrk = yFProb
        
    def computeSpeckleAOVel(self, requireAllEpochs = False, verbose=True):
        """ compute the speckle and AO velocities individually and
        check the difference in the velocity vectors.

        Keywords: requireAllEpochs - require the stars to be detected
                  in all speckle or all AO epochs to be computed
        """
        names = self.names
        starsX = self.x
        starsY = self.y

        # look up the epoch for speckle and AO
        epochFile = self.rootDir+'scripts/epochsInfo.txt'

        if os.path.isfile(epochFile):
            epochTab = Table.read(epochFile, format='ascii')
            aoFlag = epochTab['isAO']
            alignFlag = epochTab['doAlign']

            bad = np.where(alignFlag == 0)[0]
            aoFlag = np.delete(aoFlag,bad)

            # separate the speckle points from the AO points
            speckInd = np.where((aoFlag == 0))[0]
            aoInd = np.where((aoFlag == 1))[0]

            if verbose == True:
                print( 'speckle epochs: ', self.allEpochs[speckInd])
                print( 'AO epochs: ', self.allEpochs[aoInd])
        else:
            return False

        # define some new arrays for velocity calculations
        n = len(names)
        speckVx = np.zeros(n)
        speckVy = np.zeros(n)
        speckVxErr = np.zeros(n)
        speckVyErr = np.zeros(n)
        speckT0 = np.zeros(n)
        nSpeckle = np.zeros(n)  # number of speckle epochs
        speckChi2x = np.zeros(n) # chi2 for speckle velocities
        speckChi2y = np.zeros(n) # chi2 for speckle velocities

        aoVx = np.zeros(n)
        aoVy = np.zeros(n)
        aoVxErr = np.zeros(n)
        aoVyErr = np.zeros(n)
        aoT0 = np.zeros(n)
        aoChi2x = np.zeros(n) # chi2 for AO velocities
        aoChi2y = np.zeros(n) # chi2 for AO velocities
        nAO = np.zeros(n) # number of AO epochs

        velDiffX = np.zeros(n)
        velDiffY = np.zeros(n)

        # loop through names
        #for ind in range(n):
        for ind in range(n):

            x = -self.starSet.stars[ind].getArrayAllEpochs('x')
            y = self.starSet.stars[ind].getArrayAllEpochs('y')
            # positional errors
            xerr_p = self.starSet.stars[ind].getArrayAllEpochs('xerr_p')
            yerr_p = self.starSet.stars[ind].getArrayAllEpochs('yerr_p')

            # alignment errors
            xerr_a = self.starSet.stars[ind].getArrayAllEpochs('xerr_a')
            yerr_a = self.starSet.stars[ind].getArrayAllEpochs('yerr_a')

            # add the two errors in quadrature
            xerr = np.sqrt(xerr_p**2 + xerr_a**2)
            yerr = np.sqrt(yerr_p**2 + yerr_a**2)

            t0 = self.starSet.stars[ind].fitXa.t0    # use the same T0 as polyfit
            years = np.array(self.starSet.stars[ind].years)

            speckX = x[speckInd]
            speckY = y[speckInd]
            speckXErr = xerr[speckInd]
            speckYErr = yerr[speckInd]
            speckYears = years[speckInd]

            aoX = x[aoInd]
            aoY = y[aoInd]
            aoXErr = xerr[aoInd]
            aoYErr = yerr[aoInd]
            aoYears = years[aoInd]

            goodS = (np.where(abs(speckX) < 500))[0]
            nSpeckle[ind] = len(goodS)
            if (len(goodS) > 3):
                speckX = speckX[goodS]
                speckY = speckY[goodS]
                speckXErr = speckXErr[goodS]
                speckYErr = speckYErr[goodS]
                speckYears = speckYears[goodS]
                speckT0[ind] = np.sum(speckYears/speckXErr**2)/np.sum(1.0/speckXErr**2)
                speckYears = speckYears - speckT0[ind]

            goodAO = (np.where(abs(aoX) < 500))[0]
            nAO[ind] = len(goodAO)
            if (len(goodAO) > 3):
                aoX = aoX[goodAO]
                aoY = aoY[goodAO]
                aoXErr = aoXErr[goodAO]
                aoYErr = aoYErr[goodAO]
                aoYears = aoYears[goodAO]
                aoT0[ind] = np.sum(aoYears/aoXErr**2)/np.sum(1.0/aoXErr**2)
                aoYears = aoYears - aoT0[ind]

            if requireAllEpochs:
                # only do fitting if both speckle and AO are found in all epochs
                # print( 'requiring stars to be detected in all speckle and AO epochs')
                filterTest = (len(goodS) == len(speckInd)) & (len(goodAO) == len(aoInd))
            else:
                # do fitting if stars are found in more than three epochs
                filterTest = (len(goodAO) > 3) & (len(goodS) > 3)

            if filterTest:
                functargsX = {'x':speckYears, 'y':speckX, 'err': speckXErr}
                functargsY = {'x':speckYears, 'y':speckY, 'err': speckYErr}
                guessX =  [np.mean(speckX),(np.amax(speckX) - np.amin(speckX))/(np.amax(speckYears)-np.amin(speckYears))]
                guessY =  [np.mean(speckY),(np.amax(speckY) - np.amin(speckY))/(np.amax(speckYears)-np.amin(speckYears))]
                xfitLin = nmpfit.mpfit(self.fitLin, guessX, functkw=functargsX,quiet=1)
                yfitLin = nmpfit.mpfit(self.fitLin, guessY, functkw=functargsY,quiet=1)
                print( 'speckle fit x: ', xfitLin.params, xfitLin.perror)
                print( 'speckle fit y : ', yfitLin.params, yfitLin.perror)
                speckVx[ind] = xfitLin.params[1]
                if xfitLin.perror is None:
                    speckVxErr[ind] = 0
                else:
                    speckVxErr[ind] = xfitLin.perror[1]
                    speckChi2x[ind] = xfitLin.fnorm

                speckVy[ind] = yfitLin.params[1]
                if yfitLin.perror is None:
                    speckVyErr[ind] = 0
                else:
                    speckVyErr[ind] = yfitLin.perror[1]
                    speckChi2y[ind] = yfitLin.fnorm


                # fit the AO portion
                functargsX = {'x':aoYears, 'y':aoX, 'err': aoXErr}
                functargsY = {'x':aoYears, 'y':aoY, 'err': aoYErr}
                guessX =  [np.mean(aoX),(np.amax(aoX) - np.amin(aoX))/(np.amax(aoYears)-np.amin(aoYears))]
                guessY =  [np.mean(aoY),(np.amax(aoY) - np.amin(aoY))/(np.amax(aoYears)-np.amin(aoYears))]
                xfitLin = nmpfit.mpfit(self.fitLin, guessX, functkw=functargsX,quiet=1)
                yfitLin = nmpfit.mpfit(self.fitLin, guessY, functkw=functargsY,quiet=1)
                print( 'ao fit x: ', xfitLin.params, xfitLin.perror)
                print( 'ao fit y: ', yfitLin.params, yfitLin.perror)
                aoVx[ind] = xfitLin.params[1]
                if xfitLin.perror is None:
                    aoVxErr[ind] = 0
                else:
                    aoVxErr[ind] = xfitLin.perror[1]
                    aoChi2x[ind] = xfitLin.fnorm

                aoVy[ind] = yfitLin.params[1]
                if yfitLin.perror is None:
                    aoVyErr[ind] = 0
                else:
                    aoVyErr[ind] = yfitLin.perror[1]
                    aoChi2y[ind] = yfitLin.fnorm


                # take the difference between the two velocities
                velDiffX[ind] = aoVx[ind] - speckVx[ind]
                velDiffY[ind] = aoVy[ind] - speckVy[ind]

        self.speckVx = speckVx
        self.speckVy = speckVy
        self.speckVxErr = speckVxErr
        self.speckVyErr = speckVyErr
        self.nSpeckle = nSpeckle
        self.speckChi2x = speckChi2x
        self.speckChi2y = speckChi2y


        self.aoVx = aoVx
        self.aoVy = aoVy
        self.aoVxErr = aoVxErr
        self.aoVyErr = aoVyErr
        self.nAO = nAO
        self.aoChi2x = aoChi2x
        self.aoChi2y = aoChi2y
        self.velDiffX = velDiffX
        self.velDiffY = velDiffY
        self.plotSpeckleAOVel()
        return

    def chi2DistFit(self, p, direction = 'x', returnChi2 = False, maxChi2 = None,
                    scaleFactor = 1.0, data = 'both', removeConfused = True):
        """ function to fit the chi2 distribution of velocities to
        figure out the additive error that needs to be included to
        bring the chi2 distribution in line with the expected. Will only
        consider stars that are detected in all epochs.

        Input: p - additive error parameter
               x - bin locations
               y - theoretical chi2 curve

        Keywords: direction - 'x' (def), 'y', or 'both' to calculate the chi2 in
                  either direction or combine both.

                  returnChi2 - return the chi2
                  values instead of the difference between chi2 and
                  model

                  maxChi2 - the maximum chi2 value to create the
                  histogram. Defaults to 3 times the number of epochs

                  removeConfused - look at the self.confusedStars array and not
                  include them in the fit

                  self.rmax - the maximum radius to compute the additive
                  error. Default rmax = 0 for the entire range.
        """
        names = self.names
        maxEpochInd = self.maxEpochInd
        mag = self.mag

        # remove the points that are confused
        if removeConfused:
            # remove points from maxEpochInd that are in the confusedSourcesInd
            maxEpochInd2 = np.array(np.setdiff1d(maxEpochInd, self.confusedSourcesInd),dtype='int32')
            print( 'removing %.0f confused sources out of %.0f total sources' % \
              (len(maxEpochInd)-len(maxEpochInd2),len(maxEpochInd)))
            maxEpochInd = maxEpochInd2

        # limit the additive error within a certain radial range
        if (self.rmax > 0):
            goodR = np.where((self.r2d < self.rmax) & (self.r2d > self.rmin))[0]
            maxEpochInd = np.intersect1d(goodR, maxEpochInd)
            print( 'using only stars within %f arcsec, beyong %f arcsec, faintest magnitude: %f' \
                    % (self.rmax,self.rmin,np.amax(mag[maxEpochInd])))

        # limit the additive error within a certain mag
        if (self.magmax > 0):
            goodMag = np.where((mag <= self.magmax) & (mag > self.magmin))[0]
            maxEpochInd = np.intersect1d(goodMag, maxEpochInd)
            print( 'limiting magnitude to those brighter than %f, n stars: %f' % (self.magmax,len(maxEpochInd)))

        n = len(maxEpochInd)

        # keep track of which stars were used to compute the additive error
        self.addErrStarsInd = maxEpochInd

        chi2x = np.zeros(n)
        chi2y = np.zeros(n)

        # allow for the possibility of subtracting some error
        if p[0] < 0:
            sign = -1.0
        else:
            sign = 1.0

        # make sure if there are no speckle data that we set the flag to AO only
        if len(self.speckInd) == 0:
            data = 'ao'

        if (data == 'both'):
            epochInd = np.union1d(self.speckInd, self.aoInd)
        elif (data == 'speckle'):
            epochInd = self.speckInd
        elif (data == 'ao'):
            epochInd = self.aoInd

        print( 'using error scale factor: ', scaleFactor)

        # remove confused sources
        print( ','.join(names[maxEpochInd]))

        # go through good stars and compute the chi2 values
        for i in range(n):
            ind = maxEpochInd[i]

            # Grab the x and y points from align
            xpts = -self.starSet.stars[ind].getArrayAllEpochs('x')[epochInd]
            ypts = self.starSet.stars[ind].getArrayAllEpochs('y')[epochInd]

            # Grab the x and y positional errors from align
            # Scale the positional errors if specified.
            xerr_p = self.starSet.stars[ind].getArrayAllEpochs('xerr_p')[epochInd] / scaleFactor
            xerr_a = self.starSet.stars[ind].getArrayAllEpochs('xerr_a')[epochInd]

            yerr_p = self.starSet.stars[ind].getArrayAllEpochs('yerr_p')[epochInd] / scaleFactor
            yerr_a = self.starSet.stars[ind].getArrayAllEpochs('yerr_a')[epochInd]

            # add the two errors in quadrature and add the new error
            xerr = np.sqrt(xerr_p**2 + xerr_a**2 + sign*p[0]**2)
            yerr = np.sqrt(yerr_p**2 + yerr_a**2 + sign*p[0]**2)

            years = np.array(self.starSet.stars[ind].years)[epochInd]
            if (data == 'both'):
                t0 = self.starSet.stars[ind].fitXa.t0    # use the same T0 as polyfit
            else:
                # compute TO with weighted mean epoch by x positional error
                t0 = np.sum(years/xerr**2)/np.sum(1.0/xerr**2)


            # go through each star and compute the velocity as well as the errors
            if (direction == 'x') or (direction == 'both'):
                functargsX = {'x': years - t0, 'y': xpts, 'err': xerr}
                guessX = [np.mean(xpts), (np.amax(xpts) - np.amin(xpts)) / (np.amax(years) - np.amin(years))]
                xfitLin = nmpfit.mpfit(self.fitLin, guessX, functkw=functargsX, quiet=1)
                chi2x[i] = xfitLin.fnorm

            if (direction == 'y') or (direction == 'both'):
                functargsY = {'x':years - t0, 'y':ypts, 'err': yerr}
                guessY = [np.mean(xpts),(np.amax(xpts) - np.amin(xpts))/(np.amax(years) - np.amin(years))]
                yfitLin = nmpfit.mpfit(self.fitLin, guessY, functkw=functargsY,quiet=1)
                chi2y[i] = yfitLin.fnorm


        # put both together if doing both directions
        if (direction == 'both'):
            chi2_final = np.concatenate((chi2x, chi2y))

        if returnChi2:
            return chi2_final
        else:
            if maxChi2 is None:
                maxChi2 = len(years) * 3
            dof = len(years) - 2.0
            # use pdf difference
            width = maxChi2 / 30.0
            nStars, bins, patches = py.hist(chi2_final, bins=np.arange(0,maxChi2, width), color='b')
            chiInd = bins
            chi2Theory = stats.chi2.pdf(chiInd, dof) * np.sum(nStars) * width
            py.clf()
            py.plot(chiInd, chi2Theory)
            py.plot(bins[0:-1], nStars,'go')
            good = np.where(nStars != 0)[0]
            bad = np.where(nStars == 0)[0]
            err = np.sqrt(nStars)
            err[bad] = 1.0
            diff = np.zeros(len(nStars))
            diff = (nStars - chi2Theory[0:-1]) / err
            return diff



    def testChi2Fit(self, testErr = None, scaleFactor = 1.0, data = 'both'):
        """
        test the chi2DistFit function by going through a series
        ofadditive factors to minimize the chi2
        """
        if testErr is None:
            testErr = np.arange(0.00001,0.001,0.00001)
        chi2x = np.zeros(len(testErr))
        chi2y = np.zeros(len(testErr))

        # record the number of degree of freedom in the test
        if (data == 'speckle'):
            self.errAddDof = len(self.speckInd)-2.0
        elif data == ('ao'):
            self.errAddDof = len(self.aoInd)-2.0
        else:
            self.errAddDof = len(self.allEpochs)-2.0

        for ii in range(len(testErr)):
            fit = self.chi2DistFit(np.array([testErr[ii]]), direction = 'x',
                                   scaleFactor = scaleFactor, data = data)
            chi2x[ii] = np.sum(fit**2)
            print( 'additive error factor: %f chi2: %f' % (testErr[ii],chi2x[ii]))

        for ii in range(len(testErr)):
            fit = self.chi2DistFit(np.array([testErr[ii]]), direction = 'y',
                                   scaleFactor = scaleFactor, data = data)
            chi2y[ii] = np.sum(fit**2)
            print( 'additive error factor: %f chi2: %f' % (testErr[ii],chi2y[ii]))

        clf()
        plot(testErr, chi2x, 'bo', label ='x')
        plot(testErr, chi2y, 'go', label = 'y')
        xlabel('Additive Factor (arcsec)')
        ylabel('Chi Sq. Difference')
        legend()
        self.errAdd = testErr
        self.errAddChi2x = chi2x
        self.errAddChi2y = chi2y
        savefig(self.plotPrefix+'additive_error_test.png')


    def testChi2FitXY(self, testErr = None, scaleFactor = None, data = 'both'):
        """ test the chi2DistFit function by going through a series
        ofadditive factors to minimize the chi2 over BOTH directions
        """

        if scaleFactor is None:
            scaleFactor = self.errScaleFactor

        if testErr is None:
            testErr = np.arange(0.00001,0.0005,0.00001)

        chi2x = np.zeros(len(testErr))
        chi2y = np.zeros(len(testErr))

        # record the number of degree of freedom in the test
        if (data == 'speckle'):
            self.errAddDof = len(self.speckInd)-2.0
        elif data == ('ao'):
            self.errAddDof = len(self.aoInd)-2.0
        else:
            self.errAddDof = len(self.allEpochs)-2.0

        # use pdf chi2
        for ii in np.arange(len(testErr)):
            fit = self.chi2DistFit(np.array([testErr[ii]]), direction='both',
                                   scaleFactor=scaleFactor, data=data)
            chi2x[ii] = np.sum(fit**2)
            print( 'additive error factor: %f chi2: %f' % (testErr[ii],chi2x[ii]))

        py.clf()
        py.plot(testErr, chi2x, 'bo', label ='x')
        py.xlabel('Additive Factor (arcsec)')
        py.ylabel('Chi Sq. Difference')
        py.legend()
        self.errAdd = testErr
        self.errAddChi2xy = chi2x

        outfile = self.plotPrefix + 'additive_error_test_xy_' + data + '.png'
        py.savefig(outfile)

    #def computeJerk(self):
    #    """Computes the jerk for all sources. Uses the internal fitter for jerk.
    #          ***NOT FUNCTIONAL YET***
    #    """
    #    # get variables into local scope
    #    cc = self.cc
    #    names = list(self.names)
    #    x, y = self.x, self.y
    #    xerr, yerr = self.xerr, self.yerr
    #    vx, vy = self.vx, self.vy
    #    axe, aye = self.axe, self.aye

    #    #jx = np.zeros(len(names), dtype=float)
    #    #jxe = np.zeros(len(names), dtype=float)
    #    #jy = np.zeros(len(names), dtype=float)
    #    #jye = np.zeros(len(names), dtype=float)
    #    jx = []
    #    jxe = []
    #    jy = []
    #    jye = []

    #    # loop through the names and get the points for the jerk fits
    #    for i in range(len(names)):
    #        ind = names.index(names[i])

    #        x = self.starSet.stars[ind].getArrayAllEpochs('x')
    #        y = self.starSet.stars[ind].getArrayAllEpochs('y')
    #        xerr_p = self.starSet.stars[ind].getArrayAllEpochs('xerr_p')
    #        yerr_p = self.starSet.stars[ind].getArrayAllEpochs('yerr_p')

    #        xerr_a = self.starSet.stars[ind].getArrayAllEpochs('xerr_a')
    #        yerr_a = self.starSet.stars[ind].getArrayAllEpochs('yerr_a')

    #        xerr = np.sqrt(xerr_p**2 + xerr_a**2)
    #        yerr = np.sqrt(yerr_p**2 + yerr_a**2)

    #        try:
    #            t0 = self.starSet.stars[ind].fitXa.t0    # use the same T0 as polyfit
    #        except AttributeError:
    #            continue
    #        years = np.array(self.starSet.stars[ind].years)

    #        good = (np.where(abs(x) < 500))[0]

    #        if (len(good) > 4):
    #            x = x[good]
    #            y = y[good]
    #            xerr = xerr[good]
    #            yerr = yerr[good]
    #            years = years[good]

    #            xfit = self.fitJerk(years - t0, x, xerr)
    #            yfit = self.fitJerk(years - t0, y, yerr)
    #            dof = len(x) - len(xfit.params)
    #            # put the results in the correct star
    #            self.x[i] = xfit.params[0]
    #            jx = np.concatenate([jx, [xfit.params[3]]])
    #            jxe = np.concatenate([jxe, [xfit.perror[3]]])
    #            self.ax[i] = xfit.params[2]
    #            self.axe[i] = xfit.perror[2]
    #            self.vx[i] = xfit.params[1]
    #            self.vxe[i] = xfit.perror[1]

    #            self.y[i] = yfit.params[0]
    #            jy = np.concatenate([jy, [yfit.params[3]]])
    #            jye = np.concatenate([jye, [yfit.perror[3]]])
    #            self.ay[i] = yfit.params[2]
    #            self.aye[i] = yfit.perror[2]
    #            self.vy[i] = yfit.params[1]
    #            self.vye[i] = yfit.perror[1]

    #            #self.chi2xj[i] = xfit.fnorm
    #            #self.chi2yj[i] = yfit.fnorm

    #    self.jx = jx
    #    self.jxe = jxe
    #    self.jy = jy
    #    self.jye = jye


    def plotSpeckleAOVel(self):
        """ plot the AO - speckle velocities
        """
        # plot the differences in velocity
        velDiffX = self.velDiffX*1e3
        velDiffY = self.velDiffY*1e3
        starsX = self.x
        starsY = self.y

        good = np.where((velDiffX != 0) & (velDiffX < 5) & (velDiffY < 5))[0]
        py.figure(1,figsize=(8,5))
        py.clf()
        py.subplots_adjust(wspace=0.25, right=0.9, left=0.1, top=0.85, bottom=0.15)
        py.subplot(1,2,1)
        qvr = py.quiver(starsX[good],starsY[good],velDiffX[good],velDiffY[good],width=0.03,units='x',angles='xy',scale=1.0)
        py.quiverkey(qvr,5,3,1,'1 mas/yr',coordinates='data')
        py.xlim(6,-6)
        py.ylim(-4,4)
        py.subplot(1,2,2)
        binsIn = np.arange(-20, 20, 0.5)
        (nx, bx, ptx) = py.hist(velDiffX[good],bins=binsIn,color='r',histtype='step',linewidth=1,label='X')
        (ny, by, pty) = py.hist(velDiffY[good],bins=binsIn,color='b',histtype='step',linewidth=1,label='Y')
        py.plot([0,0],[0,120],'k--')
        py.xlabel('Velocity Differences (mas/yr)',fontsize=12)
        py.legend(numpoints=1)
        py.show()
        print('Median (AO - speckle) velocity (X,Y) = (%6.3f, %6.3f) mas/yr' % \
              (np.median(velDiffX[good]),np.median(velDiffY[good])))
        print('Std (AO - speckle) velocity (X,Y) = (%6.3f, %6.3f) mas/yr' % \
              (np.std(velDiffX[good]),np.std(velDiffY[good])))

    def plotAccelMap(self,save = False):
        """Plot the locations of the unphysical accelerations along
        with the acceleration vector
        """
        bad = self.bad
        goodChi = np.where((self.chi2x < self.chiThreshold*10.0) | (self.chi2y < self.chiThreshold*10))[0]
        bad = np.intersect1d(bad,goodChi)

        names = np.array(self.names)[bad]
        x = self.x[bad]
        y = self.y[bad]
        ax = self.ax[bad]*500.0
        ay = self.ay[bad]*500.0
        clf()
        plot(x,y,'o')
        xlim(6,-6)
        xlabel('RA Offset (arcsec)')
        ylabel('DEC Offset (arcsec)')
        quiver(x,y,ax,ay,angles='xy',scale=.7,width=.03,units='x')
        for ii in range(len(names)):
            text(x[ii],y[ii],names[ii])
        if save:
            savefig(self.plotPrefix+'accelMap.png')

    def plotchi2(self, removeConfused = False, goodGiven=None):
        """Plot the chi-square distribution of the velocity and acceleration fits

        KEYWORDS: removeConfused - remove the confused sources from the analysis
        """
        # get some variables in to local scope
        names = np.array(self.names)
        at, ate = self.at, self.ate
        ar, are = self.ar, self.are
        sigma = self.sigma
        x, y = self.x, self.y
        vx, vy = self.vx, self.vy
        ax, ay = self.ax, self.ay
        epoch, allEpochs = self.epoch, self.allEpochs
        nEpochs = self.nEpochs
        a2d = self.a2d
        mag = self.mag
        cc = self.cc
        chi2x, chi2y = self.chi2x, self.chi2y
        chi2xv, chi2yv = self.chi2xv, self.chi2yv
        r2d = self.r2d
        rarc, a2dsim = self.rarc, self.a2dsim
        bad, goodAccel = self.bad, self.goodAccel

        # plot some chi-square info
        plt.clf()
        plt.figure(figsize=(30,30))
        plt.subplot(231)
        # look at the distribution of chi-squares for stars brighter
        # than 16 magnitude and have maximum degree of freedom
        maxEpoch = np.amax(self.nEpochs)
        dof = maxEpoch - 3
        if goodGiven == None:
            good = np.where((mag < 16.0) & (nEpochs == maxEpoch))[0]
        else:
            goodstars = np.intersect1d(goodGiven, names)
            good = [list(names).index(i) for i in goodstars]

        # remove confused sources if specified
        if removeConfused:
            good = np.setdiff1d(good, self.confusedSourcesInd)

        # filter out the accelerations that do not have the max number of epochs measured
        #print( shape(good))
        #print( shape(bad))
        bad = np.intersect1d(good, bad)
        goodAccel = np.intersect1d(goodAccel, good)

        print( 'Degree of freedom %f' % dof)
        maxChi=dof*4.0
        #binWidth = (maxChi-0)/30.0
        binWidth = 5
        n, bins, patches1 = plt.hist(chi2x[good], bins = np.arange(0, maxChi, binWidth), normed=1, label='x', alpha=0.6, color='blue', histtype='step')
        n, bins, patches2 = plt.hist(chi2y[good], bins = bins, normed=1, label='y',alpha=0.6,color='green', histtype='step')
        plt.xlabel('Chi-Sq Acc')
        plt.title('All stars')
        chiInd = np.arange(0.01,100,0.01)
        chi2Theory = stats.chi2.pdf(chiInd,dof)
        plt.plot(chiInd, chi2Theory, color='r')

        plt.legend()
        plt.subplot(232)
        if len(bad)!= 0:
            n, bins, patches1 = plt.hist(chi2x[bad], bins = np.arange(0,maxChi,binWidth), normed=1,label='x',alpha=0.6, histtype='step')
            n, bins, patches2 = plt.hist(chi2y[bad], bins = bins,normed=1,label='y',alpha=0.6, histtype='step')
        plt.title('Non-physical Accelerations')
        plt.xlabel('Chi-Sq Acc')
        plt.plot(chiInd, chi2Theory,color='r')
        plt.legend()

        # chi-sq distribution of physical accelerations
        plt.subplot(233)
        if len(goodAccel)!= 0:
            n, bins, patches1 = plt.hist(chi2x[goodAccel], bins = np.arange(0,maxChi,binWidth), normed=1,label='x',alpha=0.6, histtype='step')
            n, bins, patches2 = plt.hist(chi2y[goodAccel], bins = bins,normed=1,label='y',alpha=0.6, histtype='step')
        plt.title('Physical Accelerations')
        plt.xlabel('Chi-Sq Acc')
        plt.plot(chiInd, chi2Theory,color='r')
        plt.legend()

        # plot the velocities
        velDof = dof + 1
        plt.subplot(234)
        n, bins, patches1 = plt.hist(chi2xv[good], bins = np.arange(0, maxChi, binWidth),normed=1,label='x',alpha=0.6,color='blue', histtype='step')
        n, bins, patches2 = plt.hist(chi2yv[good], bins = bins, normed=1, label='y',alpha=0.6,color='green', histtype='step')
        plt.xlabel('Chi-Sq Vel')
        plt.title('All stars')
        chiInd = np.arange(0.01,100,0.01)
        chi2Theory = stats.chi2.pdf(chiInd,velDof)
        plt.plot(chiInd, chi2Theory,color='r')
        plt.legend()

        plt.subplot(235)
        if len(bad)!=0:
            n, bins, patches1 = plt.hist(chi2xv[bad], bins = np.arange(0,maxChi,binWidth), normed=1,label='x',alpha=0.6, histtype='step')
            n, bins, patches2 = plt.hist(chi2yv[bad], bins = bins,normed=1,label='y',alpha=0.6, histtype='step')
        plt.title('Non-physical Accelerations')
        plt.xlabel('Chi-Sq Vel')
        plt.plot(chiInd, chi2Theory,color='r')
        plt.legend()

        plt.subplot(236)
        if len(goodAccel) != 0:
            n, bins, patches1 = plt.hist(chi2xv[goodAccel], bins = np.arange(0,maxChi,binWidth), normed=1,label='x',alpha=0.6, histtype='step')
            n, bins, patches2 = plt.hist(chi2yv[goodAccel], bins = bins,normed=1,label='y',alpha=0.6, histtype='step')
        plt.title('Physical Accelerations')
        plt.xlabel('Chi-Sq Vel')
        plt.plot(chiInd, chi2Theory,color='r')
        plt.legend()
        plt.savefig(self.plotPrefix+'Chi2_dist_accel_vel.png')
        plt.close()

        # plot chi2 comparison between vel and acc
        plt.clf()
        plt.figure(figsize=(30,8))
        plt.subplot(131)
        plt.loglog(chi2xv[good]/(dof+1.0),chi2x[good]/dof,'bo',label='X',alpha=0.6)
        plt.plot(chi2yv[good]/(dof+1.0),chi2y[good]/dof,'go',label='Y',alpha=0.6)
        plt.plot([0.001,100],[0.001,100])
        plt.xlim(0.01,maxChi)
        plt.ylim(0.01,maxChi)
        plt.xlabel('Vel. Fit Reduced Chi-Sq')
        plt.ylabel('Acc. Fit Reduced Chi-Sq')
        plt.title('All Stars')
        plt.legend(loc=2)

        plt.subplot(132)
        if len(bad) != 0:
            plt.loglog(chi2xv[bad]/(nEpochs[bad]-2.0),chi2x[bad]/(nEpochs[bad]-3.0),'bo',label='X',alpha=0.6)
            plt.plot(chi2yv[bad]/(nEpochs[bad]-2.0),chi2y[bad]/(nEpochs[bad]-3.0),'go',label='Y',alpha=0.6)
        plt.plot([0.001,100],[0.001,100])
        plt.xlim(0.01,maxChi)
        plt.ylim(0.01,maxChi)
        plt.xlabel('Vel. Fit Reduced Chi-Sq')
        plt.ylabel('Acc. Fit Reduced Chi-Sq')
        plt.title('Non-physical Accelerations')
        plt.legend(loc=2)

        plt.subplot(133)
        if len(goodAccel) != 0:
            plt.loglog(chi2xv[goodAccel]/(nEpochs[goodAccel]-2.0),chi2x[goodAccel]/(nEpochs[goodAccel]-3.0),'bo',label='X',alpha=0.6)
            plt.plot(chi2yv[goodAccel]/(nEpochs[goodAccel]-2.0),chi2y[goodAccel]/(nEpochs[goodAccel]-3.0),'go',label='Y',alpha=0.6)
        plt.plot([0.001,100],[0.001,100])
        plt.xlim(0.01,maxChi)
        plt.ylim(0.01,maxChi)
        plt.xlabel('Vel. Fit Reduced Chi-Sq')
        plt.ylabel('Acc. Fit Reduced Chi-Sq')
        plt.title('Physical Accelerations')
        plt.legend(loc=2)
        plt.savefig(self.plotPrefix+'Chi2_vel_vs_accel.png')
        plt.close()

        # plot chi-sq as a function of magnitude
        plt.clf()
        plt.figure(figsize=(30,20))
        plt.subplot(231)
        plt.semilogy(mag[good],chi2x[good]/(nEpochs[good]-3.0),'bo',label='X',alpha=0.6)
        plt.plot(mag[good],chi2y[good]/(nEpochs[good]-3.0),'go',label='Y',alpha=0.6)
        plt.ylim(0.01,maxChi)
        plt.xlabel('K mag')
        plt.ylabel('Accel. Fit Chi-Sq')
        plt.title('All Stars')
        plt.legend(loc=3)

        plt.subplot(232)
        if len(bad) != 0:
            plt.semilogy(mag[bad],chi2x[bad]/(nEpochs[bad]-3.0),'bo',label='X',alpha=0.6)
            plt.plot(mag[bad],chi2y[bad]/(nEpochs[bad]-3.0),'go',label='Y',alpha=0.6)
        plt.ylim(0.01,maxChi)
        plt.xlabel('K mag')
        plt.ylabel('Accel. Fit Chi-Sq')
        plt.title('Non-physical Accelerations')
        plt.legend(loc=3)

        plt.subplot(233)
        if len(goodAccel) != 0:
            plt.semilogy(mag[goodAccel],chi2x[goodAccel]/(nEpochs[goodAccel]-3.0),'bo',label='X',alpha=0.6)
            plt.plot(mag[goodAccel],chi2y[goodAccel]/(nEpochs[goodAccel]-3.0),'go',label='Y',alpha=0.6)
        plt.ylim(0.01,maxChi)
        plt.xlabel('K mag')
        plt.ylabel('Accel. Fit Chi-Sq')
        plt.title('Physical Accelerations')
        plt.legend(loc=3)

        plt.subplot(234)
        plt.semilogy(mag[good],chi2xv[good]/(nEpochs[good]-2.0),'bo',label='X',alpha=0.6)
        plt.plot(mag[good],chi2yv[good]/(nEpochs[good]-2.0),'go',label='Y',alpha=0.6)
        plt.ylim(0.01,maxChi)
        plt.xlabel('K mag')
        plt.ylabel('Vel. Fit Chi-Sq')
        plt.title('All Stars')
        plt.legend(loc=3)

        plt.subplot(235)
        if len(bad) != 0:
            plt.semilogy(mag[bad],chi2xv[bad]/(nEpochs[bad]-2.0),'bo',label='X',alpha=0.6)
            plt.plot(mag[bad],chi2yv[bad]/(nEpochs[bad]-2.0),'go',label='Y',alpha=0.6)
        plt.ylim(0.01,maxChi)
        plt.xlabel('K mag')
        plt.ylabel('Vel. Fit Chi-Sq')
        plt.title('Non-physical Accelerations')
        plt.legend(loc=3)

        plt.subplot(236)
        if len(goodAccel) != 0:
            plt.semilogy(mag[goodAccel],chi2xv[goodAccel]/(nEpochs[goodAccel]-2.0),'bo',label='X',alpha=0.6)
            plt.plot(mag[goodAccel],chi2yv[goodAccel]/(nEpochs[goodAccel]-2.0),'go',label='Y',alpha=0.6)
        plt.ylim(0.01,maxChi)
        plt.xlabel('K mag')
        plt.ylabel('Vel. Fit Chi-Sq')
        plt.title('Physical Acceleration')
        plt.legend(loc=3)
        plt.savefig(self.plotPrefix+'Chi2_vs_mag.png')
        plt.close()

    def plot_red_chi2(self, epochs_cut=30, mag_cut=100, r_min=0.4, maxChi_cdf=2, maxChi_pdf=4, 
            goodGiven=None, removeConfused = False, plot_vel_chi2=False):
        """Plot the reduced-chi-square distribution of the velocity and acceleration fits
        KEYWORDS: removeConfused - remove the confused sources from the analysis
        """
        # get some variables in to local scope
        names = np.array(self.names)
        at, ate = self.at, self.ate
        ar, are = self.ar, self.are
        sigma = self.sigma
        x, y = self.x, self.y
        vx, vy = self.vx, self.vy
        ax, ay = self.ax, self.ay
        epoch, allEpochs = self.epoch, self.allEpochs
        nEpochs = self.nEpochs
        a2d = self.a2d
        mag = self.mag
        cc = self.cc
        chi2x, chi2y = self.chi2x, self.chi2y
        chi2xv, chi2yv = self.chi2xv, self.chi2yv
        r2d = self.r2d
        rarc, a2dsim = self.rarc, self.a2dsim
        bad, goodAccel = self.bad, self.goodAccel

        # look at the distribution of reduced chi-squares for stars
        # detected in more than 30 epochs and r>0.4
        if goodGiven == None:
            if epochs_cut == 'max':
                epochs_cut = len(self.allEpochs)
            good = np.where((r2d>r_min) & (nEpochs>=epochs_cut) & (mag<mag_cut))[0]
        else:
            goodstars = np.intersect1d(np.load(goodGiven), names)
            good = [list(names).index(i) for i in goodstars]
            if epochs_cut == 'max':
                epochs_cut = len(self.allEpochs)
            good_epoch_cut = np.where(nEpochs>=epochs_cut)[0]
            good = np.intersect1d(good, good_epoch_cut)

        # remove confused sources if specified
        if removeConfused:
            good = np.setdiff1d(good, self.confusedSourcesInd)

        # filter out the accelerations that do not have the max number of epochs measured
        bad = np.intersect1d(good, bad)
        goodAccel = np.intersect1d(goodAccel, good)

        # plot the reduced chi2 cdf
        dof = nEpochs[good] - 3
        red_chi2_x = chi2x[good]/dof
        red_chi2_y = chi2y[good]/dof
        idx_tmp = np.where((red_chi2_x<maxChi_cdf) & (red_chi2_y<maxChi_cdf))[0]
        red_chi2_x_cut = red_chi2_x[idx_tmp]
        sorted_x = np.sort(red_chi2_x_cut)
        yvals_x = np.arange(len(sorted_x))/float(len(sorted_x))

        red_chi2_y_cut = red_chi2_y[idx_tmp]
        sorted_y = np.sort(red_chi2_y_cut)
        yvals_y = np.arange(len(sorted_y))/float(len(sorted_y))

        chiInd = np.arange(0.01,100,0.01)
        chi2Theory_max = stats.chi2.cdf(chiInd,dof.max())
        chi2Theory_min = stats.chi2.cdf(chiInd,dof.min())

        plt.figure(figsize=(10,10))
        plt.clf()
        plt.plot(sorted_x, yvals_x, 'b-', label=r'$\chi^2_{red}$ in x')
        plt.plot(sorted_y, yvals_y, 'g-', label=r'$\chi^2_{red}$ in y')
        legend1 = plt.legend(loc='upper left')

        l1, = plt.plot(chiInd/dof.max(), chi2Theory_max, 'r--', label='dof=%d' %dof.max())
        l2, = plt.plot(chiInd/dof.min(), chi2Theory_min, 'm--', label='dof=%d' %dof.min())
        ax = plt.gca().add_artist(legend1)
        plt.legend(handles=[l1, l2], loc='center right')
        plt.xlim(0,maxChi_cdf)
        plt.ylim(0,1)
        plt.annotate('N_star = %d' %(len(sorted_y)), xy=(0.5, 0.3), xycoords='axes fraction')
        plt.xlabel(r'$\chi^2_{red}$')
        plt.title(r'CDF of $\chi^2_{red}$')
        plt.savefig(self.plotPrefix+'Redchi2_cdf.png', format='png')
        
        # plot the reduced chi2 pdf
        #plot the ACC fit reduced chi2 pdf
        plt.figure(figsize=(30,10))
        binWidth = (maxChi_pdf-0)/30.0
        n_x, bins_x = np.histogram(chi2x[good]/dof, bins = np.arange(0, maxChi_pdf, binWidth))
        n_y, bins_y = np.histogram(chi2y[good]/dof, bins = np.arange(0, maxChi_pdf, binWidth))
        n, bins = np.histogram(np.concatenate((chi2y[good]/dof, chi2x[good]/dof)), bins = np.arange(0, maxChi_pdf, binWidth))
        n_x = list(n_x)
        n_x.append(0)
        n_y = list(n_y)
        n_y.append(0)
        n = list(n)
        n.append(0)
        
        n_x = np.array(n_x)
        n_y = np.array(n_y)
        n = np.array(n)

        ax1 = plt.subplot(131)
        plt.step(bins_x, n_x/float(np.max(n_x)), color='blue')
        chiInd = np.arange(0.01,100,0.01)
        chi2Theory = stats.chi2.pdf(chiInd,dof.min())
        l1, = plt.plot(chiInd/dof.min(), chi2Theory/chi2Theory.max(), color='r', label='dof=%d'%(dof.min()))
        chi2Theory = stats.chi2.pdf(chiInd,dof.max())
        l2, = plt.plot(chiInd/dof.max(), chi2Theory/chi2Theory.max(), color='m', label='dof=%d'%(dof.max()))
        plt.legend(handles=[l1, l2], loc='upper right')
        plt.annotate('N_star = %d' %(np.sum(np.array(n_x))), xy=(0.7, 0.5), xycoords='axes fraction')
        plt.xlabel(r'$\chi^2_{red}$')
        plt.title(r'Reduced $\chi^2$ distribution in X from Acc fit')
        plt.xlim(-0.5, maxChi_pdf+1)

        plt.subplot(132, sharey=ax1)
        plt.step(bins_y, n_y/float(np.max(n_y)), color='green', label='y')
        chi2Theory = stats.chi2.pdf(chiInd,dof.min())
        l1, = plt.plot(chiInd/dof.min(), chi2Theory/chi2Theory.max(), color='r', label='dof=%d'%(dof.min()))
        chi2Theory = stats.chi2.pdf(chiInd,dof.max())
        l2, = plt.plot(chiInd/dof.max(), chi2Theory/chi2Theory.max(), color='m', label='dof=%d'%(dof.max()))
        plt.legend(handles=[l1, l2], loc='upper right')
        plt.annotate('N_star = %d' %(np.sum(np.array(n_y))), xy=(0.7, 0.5), xycoords='axes fraction')
        plt.xlabel(r'$\chi^2_{red}$')
        plt.title(r'Reduced $\chi^2$ distribution in Y from Acc fit')
        plt.xlim(-0.5, maxChi_pdf+1)

        plt.subplot(133, sharey=ax1)
        plt.step(bins, n/float(np.max(n)), color='green', label='y')
        chi2Theory = stats.chi2.pdf(chiInd,dof.min())
        l1, = plt.plot(chiInd/dof.min(), chi2Theory/chi2Theory.max(), color='r', label='dof=%d'%(dof.min()))
        chi2Theory = stats.chi2.pdf(chiInd,dof.max())
        l2, = plt.plot(chiInd/dof.max(), chi2Theory/chi2Theory.max(), color='m', label='dof=%d'%(dof.max()))
        plt.legend(handles=[l1, l2], loc='upper right')
        plt.annotate('N_star*2 = %d' %(np.sum(np.array(n))), xy=(0.7, 0.5), xycoords='axes fraction')
        plt.xlabel(r'$\chi^2_{red}$')
        plt.title(r'Reduced $\chi^2$ distribution in X&Y from Acc fit')
        plt.xlim(-0.5, maxChi_pdf+1)


        plt.tight_layout()
        plt.savefig(self.plotPrefix+'Redchi2_pdf.png')
        plt.close()

        # plot the Vel fit reduced chi2 pdf
        if plot_vel_chi2:
            n_x, bins_x = np.histogram(chi2xv[good]/dof, bins = np.arange(0, maxChi_pdf, binWidth))
            n_y, bins_y = np.histogram(chi2yv[good]/dof, bins = np.arange(0, maxChi_pdf, binWidth))
            n_x = list(n_x)
            n_x.append(0)
            n_y = list(n_y)
            n_y.append(0)

            plt.figure(figsize=(20,10))
            dof = nEpochs[good] - 2
            ax1 = plt.subplot(121)
            plt.step(bins_x, n_x/float(np.max(n_x)), color='blue')
            chiInd = np.arange(0.01,100,0.01)
            chi2Theory = stats.chi2.pdf(chiInd,dof.min())
            l1, = plt.plot(chiInd/dof.min(), chi2Theory/chi2Theory.max(), color='r', label='dof=%d'%(dof.min()))
            chi2Theory = stats.chi2.pdf(chiInd,dof.max())
            l2, = plt.plot(chiInd/dof.max(), chi2Theory/chi2Theory.max(), color='m', label='dof=%d'%(dof.max()))
            plt.legend(handles=[l1, l2], loc='upper right')
            plt.annotate('N_star = %d' %(np.sum(np.array(n_x))), xy=(0.7, 0.5), xycoords='axes fraction')
            plt.xlabel(r'$\chi^2_{x}$')
            plt.title(r'Red $\chi^2$ distribution from Lin fit')

            plt.subplot(122, sharey=ax1)
            plt.step(bins_y, n_y/float(np.max(n_y)), color='green')
            chiInd = np.arange(0.01,100,0.01)
            chi2Theory = stats.chi2.pdf(chiInd,dof.min())
            l1, = plt.plot(chiInd/dof.min(), chi2Theory/chi2Theory.max(), color='r', label='dof=%d'%(dof.min()))
            chi2Theory = stats.chi2.pdf(chiInd,dof.max())
            l2, = plt.plot(chiInd/dof.max(), chi2Theory/chi2Theory.max(), color='m', label='dof=%d'%(dof.max()))
            plt.legend(handles=[l1, l2], loc='upper right')
            plt.annotate('N_star = %d' %(np.sum(np.array(n_y))), xy=(0.7, 0.5), xycoords='axes fraction')
            plt.xlabel(r'$\chi^2_{y}$')

            plt.tight_layout()
            plt.savefig(self.plotPrefix+'Redchi2_vel_pdf.png')
            plt.close()

        # plot chi2 comparison between vel and acc
        dof = nEpochs[good] - 3
        plt.clf()
        plt.figure(figsize=(10,10))
        plt.subplot(111)
        plt.plot(chi2xv[good]/(dof+1.0),chi2x[good]/dof,'bo',label='X',alpha=0.6)
        plt.plot(chi2yv[good]/(dof+1.0),chi2y[good]/dof,'go',label='Y',alpha=0.6)
        plt.plot([0.1,100],[0.1,100])
        plt.xlim(0.1,maxChi_pdf)
        plt.ylim(0.1,maxChi_pdf)
        plt.xlabel(r'Vel. Fit $\chi^2_{red}$')
        plt.ylabel(r'Acc. Fit $\chi^2_{red}$')
        plt.legend(loc=2)
        plt.tight_layout()
        plt.savefig(self.plotPrefix+'Redchi2_vel_vs_accel.png')
        plt.close()

        # plot chi-sq as a function of magnitude
        dof = nEpochs[good] - 3
        plt.clf()
        plt.figure(figsize=(10,10))
        plt.subplot(111)
        plt.plot(mag[good],chi2x[good]/dof,'bo',label='X',alpha=0.6)
        plt.plot(mag[good],chi2y[good]/dof,'go',label='Y',alpha=0.6)
        plt.ylim(0.1,maxChi_pdf)
        plt.xlabel('K mag')
        plt.ylabel(r'Accel. Fit $\chi^2_{red}$')
        plt.legend(loc=3)
        plt.savefig(self.plotPrefix+'Redchi2_vs_mag.png')
        plt.tight_layout()
        plt.close()

        return

    def plotChi2RadialDependence(self):
        """ Plot the chi2 distribution by splitting the stars into
        different distances from Sgr A* to see if there is a
        dependence in the additive error that we need
        """
        pass

    def saveClass(self, filename = False):
        """Save the class with a pickle file
        """
        if not(filename):
            filename = self.prefix+'_accelClass.pkl'
        # open a file
        f = open(filename,'wb')
        print( 'saving file: '+filename)
        cPickle.dump(self,f,-1)
        f.close()



    def GoodStars(self, removeConfused = True, data='both',
                 magMax=16, rMax=10, rMin=0.8, veMax=None,
                 goodGiven=None):
        """
        good stars which are defined:
            detected in all epochs
            not confused
            radius cut
            magnitute cut
            ve cut
        Input:
            accelClass object
        Output:
            good star indices
            good star names
            good star table saved in plots
        """
        # use stars that are detected in each epoch
        maxEpochInd = self.maxEpochInd

        # remove points from maxEpochInd that are in the confusedSourcesInd
        if removeConfused:
            maxEpochInd2 = np.array(np.setdiff1d(maxEpochInd, self.confusedSourcesInd),dtype='int32')
            print( 'total sources that are detected in all epochs: %.0f' %(len(maxEpochInd)))
            print( 'using %.0f stars after removing confused stars'  %(len(maxEpochInd2)))
            maxEpochInd = maxEpochInd2

        # write a table of goodstars: position and position error
        epochInd = np.union1d(self.speckInd, self.aoInd)
        X = np.zeros((len(maxEpochInd), len(epochInd)))
        Y = np.zeros((len(maxEpochInd), len(epochInd)))
        Xerr_p = np.zeros((len(maxEpochInd), len(epochInd)))
        Yerr_p = np.zeros((len(maxEpochInd), len(epochInd)))
        Xerr_a = np.zeros((len(maxEpochInd), len(epochInd)))
        Yerr_a = np.zeros((len(maxEpochInd), len(epochInd)))
        Xerr = np.zeros((len(maxEpochInd), len(epochInd)))
        Yerr = np.zeros((len(maxEpochInd), len(epochInd)))
        Years = np.zeros((len(maxEpochInd), len(epochInd)))
        T0 = np.zeros(len(maxEpochInd))

        for i in range(len(maxEpochInd)):
            ind = maxEpochInd[i]
            # Grab the x and y points from align
            xpts = -self.starSet.stars[ind].getArrayAllEpochs('x')[epochInd]
            ypts = self.starSet.stars[ind].getArrayAllEpochs('y')[epochInd]

            # Grab the x and y positional and align errors from align
            xerr_p = self.starSet.stars[ind].getArrayAllEpochs('xerr_p')[epochInd]
            xerr_a = self.starSet.stars[ind].getArrayAllEpochs('xerr_a')[epochInd]

            yerr_p = self.starSet.stars[ind].getArrayAllEpochs('yerr_p')[epochInd]
            yerr_a = self.starSet.stars[ind].getArrayAllEpochs('yerr_a')[epochInd]

            # add the two errors in quadrature and add the new error
            xerr = np.sqrt(xerr_p**2 + xerr_a**2)
            yerr = np.sqrt(yerr_p**2 + yerr_a**2)

            years = np.array(self.starSet.stars[ind].years)[epochInd]

            if (data == 'both'):
                t0 = self.starSet.stars[ind].fitXa.t0    # use the same T0 as polyfit
            else:
                # compute TO with weighted mean epoch by x positional error
                t0 = np.sum(years/xerr**2)/np.sum(1.0/xerr**2)

            X[i] = xpts
            Y[i] = ypts
            Xerr_p[i] = xerr_p
            Yerr_p[i] = yerr_p
            Xerr_a[i] = xerr_a
            Yerr_a[i] = yerr_a
            Xerr[i] = xerr
            Yerr[i] = yerr
            Years[i] = years
            T0[i] = t0

        t = Table()
        t['name'] = self.names[maxEpochInd]
        t['mag'] = self.mag[maxEpochInd]
        t['t0'] = T0
        # accel fitting
        t['x'] = self.x[maxEpochInd]
        t['y'] = self.y[maxEpochInd]
        t['xe'] = self.xerr[maxEpochInd]
        t['ye'] = self.yerr[maxEpochInd]
        t['vx'] = self.vx[maxEpochInd]
        t['vy'] = self.vy[maxEpochInd]
        t['vxe'] = self.vxe[maxEpochInd]
        t['vye'] = self.vye[maxEpochInd]
        t['ax'] = self.ax[maxEpochInd]
        t['ay'] = self.ay[maxEpochInd]
        t['axe'] = self.axe[maxEpochInd]
        t['aye'] = self.aye[maxEpochInd]
        t['chi2x'] = self.chi2x[maxEpochInd]
        t['chi2y'] = self.chi2y[maxEpochInd]
        # linear fitting position and velocity
        t['x_v'] = self.x_v[maxEpochInd]
        t['y_v'] = self.y_v[maxEpochInd]
        t['xe_v'] = self.xe_v[maxEpochInd]
        t['ye_v'] = self.ye_v[maxEpochInd]
        t['vx_v'] = self.vx_v[maxEpochInd]
        t['vy_v'] = self.vy_v[maxEpochInd]
        t['vxe_v'] = self.vxe_v[maxEpochInd]
        t['vye_v'] = self.vye_v[maxEpochInd]
        t['chi2xv'] = self.chi2xv[maxEpochInd]
        t['chi2yv'] = self.chi2yv[maxEpochInd]
        # data from each epoch
        t['years'] = Years
        t['X'] = X
        t['Y'] = Y
        t['Xerr_p'] = Xerr_p
        t['Yerr_p'] = Yerr_p
        t['Xerr_a'] = Xerr_a
        t['Yerr_a'] = Yerr_a
        t['Xerr'] = Xerr
        t['Yerr'] = Yerr

        # limit the stars within a certain radial range
        if goodGiven == None:
            r = np.hypot(t['x'], t['y'])
            ve = np.array(np.hypot(t['vxe'], t['vye']))
            mag = t['mag']
            if veMax==None:
                veMax= np.median(ve) + 3 * ve.std()
            goodR = np.where((r < rMax) & (r > rMin) & (mag<=magMax) & (ve<veMax))[0]
            print( 'using %.0f stars within %f arcsec, beyong %f arcsec \n \
                    with magnitute smaller than %f \n \
                    with velocity error smaller than %f mas/yr' \
                    % (len(goodR), rMax, rMin, magMax, veMax*1000.))
        else:
            names = t['name']
            names = np.array(names)
            names_common = np.intersect1d(names, goodGiven)
            print('among %d given good stars, %d are detected in all epochs with no confusion'
                %(len(goodGiven), len(names_common)))
            names = list(names)
            goodR = [names.index(i) for i in names_common]
        t = t[goodR]
        # currently not working because of hdf5 doesn support Unicode
        #t.write('plots/goodstars.hdf5', overwrite=True, path='x')
        return np.array(t['name'])



def KsChi2Test(alignErr, refStars = None,
              magBoundary = None, direction='both',
              maxChi2 = None, nbin_ks=60, nbin_chi2=10):
    """
    for each star, chi2 value could be calculated.
    Those chi2 values should follow a chi2 distribution with (N-2) degree of freedom.
    So we can test whether those chi2 values distribution follow a standard chi2 distribution.

    Input:
    ----------
        alignErr: the errScale from epochsInfo.txt in mas
                  only for output purpose.
        refStars: name of reference stars.if given, chi2 distribution of those reference stars.
                  if not given, no refStars chi2 will be made
        magBoundaray: Chi2 distribution for bright and faint stars,
                       if not given, use the median magnitude as the boundary.
        maxchi2: chi2 larger than this will not be used in chi2 distribution.
                if not given, use 3*degree of freedom
        direction: chi2 in x or y or both direction.

    Output:
    -------------
        chi2 distribution for all stars, bright and faint stars, reference stars.
        ks test for all stars, bright and faint stars, reference stars.
    """

    # use goodstars that defined in GoodStars function
    t = Table.read('plots/goodstars.hdf5', path='x')
    names = np.array(t['name'])
    mag = np.array(t['mag'])

    # keep track of which stars were used to compute the additive error
    print( 'stars that are used to calculate chi2:')
    print(names)
    print( '\n')

    #chi2x = t['chi2xv']
    #chi2y = t['chi2yv']
    chi2x = t['chi2xa']
    chi2y = t['chi2ya']
    years = t['years'][0]

    # set a maximum chi2, stars with chi2 larger than that will not be used
    if maxChi2 is None:
        maxChi2 = len(years) * 3

    if (direction == 'x'):
        good_ind = np.where(chi2x < maxChi2)[0]
        chi2xx = chi2x[good_ind]
        chi2yy = chi2y[good_ind]
        chi2_trim = chi2xx
    if (direction == 'y'):
        good_ind = np.where(chi2y < maxChi2)[0]
        chi2xx = chi2x[good_ind]
        chi2yy = chi2y[good_ind]
        chi2_trim = chi2yy
    if (direction == 'both'):
        good_ind = np.where((chi2x < maxChi2) & (chi2y < maxChi2))[0]
        chi2xx = chi2x[good_ind]
        chi2yy = chi2y[good_ind]
        chi2_trim = np.concatenate((chi2xx, chi2yy))

    namesP = names[good_ind]

    print( 'Using additive err: %f mas' %(alignErr) )
    print( 'using %d good stars with chi2 less than %d:' %(len(good_ind), maxChi2))
    print( namesP)

    if len(chi2_trim>10):
        #dof = len(years) - 2.0
        dof = len(years) - 3.0
        #----------------------------
        # KS test for all good stars
        #-----------------------------

        # KS test
        #--------
        py.clf()
        py.figure(figsize=(10,10))
        bins = np.linspace(0, maxChi2, nbin_ks)
        py.hist(chi2_trim, normed=1, bins=bins, cumulative=True, histtype='step',
                label='all good', color='b')
        py.hist(stats.chi2.rvs(dof,size=1000), normed=1, bins=bins, cumulative=True, histtype='step',
                color='k', label='standard chi2 (dof= %d)' %dof)
        ks_p = stats.kstest(chi2_trim, lambda x: stats.chi2.cdf(x,dof))
        py.title('chi2 distribution for all good stars' )
        py.annotate( r'P=%4.2f  D=%4.2f  $N_{star}$=%d' %(ks_p[1],ks_p[0],len(chi2_trim)/2), xy=(0.4,0.1), xycoords='axes fraction')
        py.annotate( 'additive error=%.3f mas' %(alignErr),  xy=(0.1,0.95), xycoords='axes fraction')
        py.legend(loc='upper right')
        py.ylim(0,1.3)
        py.xlim(0,150)
        py.savefig('plots/ksTest.png')
        py.close()
        print( 'KS Test:   P_value = %f  D_value = %f \n' %(ks_p[1], ks_p[0]))

        #chi2 test
        #----------
        py.figure(figsize=(10,10))
        py.clf()
        bins2 = np.linspace(0, maxChi2, nbin_chi2)
        n, bins2, patches = py.hist(chi2_trim, color = 'b', bins=bins2, label='all good')
        chi2_bin = (bins2[0:-1]+bins2[1:])/2
        chi2_stand = stats.chi2(dof).pdf(chi2_bin) * len(chi2_trim) * maxChi2/nbin_chi2
        py.plot(chi2_bin, chi2_stand, 'ko', label='standard chi2')
        # use 1 as uncertanity for 0 sample
        ne = n
        idx_temp = np.where(n==0)[0]
        ne[idx_temp] = 1
        chi2_all = np.sum((n - chi2_stand)**2 / ne) / len(chi2_bin)
        py.annotate( r'$\chi^2_{red} = %.2f$ ' %(chi2_all) , color='b',xy=(0.7,0.5), xycoords='axes fraction')
        py.annotate( r'$N_{star}$ = %d' %(len(chi2_trim)/2) , color='k',xy=(0.7,0.45), xycoords='axes fraction')
        py.annotate( 'additive error=%.3f mas' %(alignErr), xy=(0.1,0.95), xycoords='axes fraction')
        py.title( 'chi2 distribution for all good stars')
        py.legend(loc='upper right')
        py.savefig('plots/chi2Test_all' + '.png' )
        py.ylim(0,180)
        py.close()



        #----------------------------
        # KS test for bright and faint stars
        #-----------------------------
        magP = mag[good_ind]
        if magBoundary == None:
            magBoundary = [mag.min(), np.percentile(mag, 20), np.percentile(mag, 40),
                      np.percentile(mag, 60), np.percentile(mag, 80), mag.max()]
        magN = len(magBoundary)
        ks_bf = np.zeros((magN-1, 2))
        chi2_bf = np.zeros(magN-1)


        # chi2 test
        #--------------------------
        #colors = ['b', 'g', 'c', 'm', 'r']
        colors = cm.rainbow(np.linspace(0, 1, magN-1))
        for i in range(magN-1):
            py.figure(figsize=(10,10))
            py.clf()
            chi2bf_ind = np.where((magP > magBoundary[i]) & (magP < magBoundary[i+1]))[0]
            if (direction == 'x'):
                chi2bf = chi2xx[chi2bf_ind]
            if (direction == 'y'):
                chi2bf = chi2yy[chi2bf_ind]
            if (direction == 'both'):
                chi2bf = np.concatenate((chi2xx[chi2bf_ind], chi2yy[chi2bf_ind]))
            bins2 = np.linspace(0, maxChi2, nbin_chi2)
            n, bins2, patches = py.hist(chi2bf, color = colors[i], bins=bins2,
                    label='%.1f< mag <%.1f' %(magBoundary[i], magBoundary[i+1]))
            chi2_bin = (bins2[0:-1]+bins2[1:])/2
            chi2_stand = stats.chi2(dof).pdf(chi2_bin) * len(chi2bf) * maxChi2/nbin_chi2
            py.plot(chi2_bin, chi2_stand, 'ko', label='standard chi2')
            ne = n
            idx_temp = np.where(n==0)[0]
            ne[idx_temp] = 1
            chi2_bf[i] = np.sum((n - chi2_stand)**2 / ne) / len(chi2_bin)
            py.annotate( r'$\chi^2_{red} = %.2f$ ' %(chi2_bf[i]), color = colors[i], xy=(0.7,0.5), xycoords='axes fraction')
            py.annotate( r'$N_{star}$ = %d' %(len(chi2bf)/2), color='k',xy=(0.7,0.45), xycoords='axes fraction')
            py.annotate( 'additive error=%.3f mas' %(alignErr), xy=(0.1,0.95), xycoords='axes fraction')
            py.title( 'chi2 distribution for bright and faint stars')
            py.legend(loc='upper right')
            py.savefig('plots/chi2Test_bright_faint_%d' %i + '.png' )
            py.ylim(0,50)
            py.close()


        # ks test
        #-----------------------
        py.clf()
        fig = py.figure(figsize=(15,12))
        ax = plt.subplot(1,1,1)
        colors = cm.rainbow(np.linspace(0, 1, magN-1))
        #colors = ['b', 'g', 'c', 'm', 'r', 'k']
        for i in range(magN-1):
            chi2bf_ind = np.where((magP > magBoundary[i]) & (magP < magBoundary[i+1]))[0]
            if (direction == 'x'):
                chi2bf = chi2xx[chi2bf_ind]
            if (direction == 'y'):
                chi2bf = chi2yy[chi2bf_ind]
            if (direction == 'both'):
                chi2bf = np.concatenate((chi2xx[chi2bf_ind], chi2yy[chi2bf_ind]))
            ax.hist(chi2bf, normed=1, bins=bins, cumulative=True, histtype='step',
               color = colors[i], label='%.1f< mag <%.1f' %(magBoundary[i], magBoundary[i+1]))

            #chi2_bin_min = stats.chi2.ppf(0.0, dof)
            #chi2_bin_max = stats.chi2.ppf(0.95, dof)
            #chi2bf_trim = chi2bf[np.where((chi2_bin_min<chi2bf) & (chi2bf<chi2_bin_max))[0]]
            #chi2_bin = np.linspace(chi2_bin_min, chi2_bin_max, 10000)
            #chi2bf_stand = stats.chi2(dof).cdf(chi2_bin) * len(chi2bf)
            #ks_bf[i] = stats.ks_2samp(chi2bf_trim, chi2bf_stand)
            ks_bf[i] = stats.kstest(chi2bf, lambda x: stats.chi2.cdf(x,dof))

            ax.annotate( r'P=%4.2f  D=%4.2f  $N_{star}$=%d' %(ks_bf[i,1], ks_bf[i,0], len(chi2bf)/2) , color = colors[i] ,
                        xy=(0.6,0.5-i/20.), xycoords='axes fraction')
        ax.hist(stats.chi2.rvs(dof,size=1000), normed=1, bins=bins, cumulative=True, histtype='step',
                color='k', label='chi2 (dof= %d)' %dof)
        ax.annotate( 'additive error=%.3f mas' %(alignErr), xy=(0.1,0.95), xycoords='axes fraction')
        ax.set_title( 'chi2 distribution for bright and faint stars')
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        #ax.set_position([box.x0, box.y0 + box.height * 0.1,box.width, box.height * 0.9])
        #ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
        #ax.set_ylim(0,1.3)
        ax.set_xlim(0,150)
        #fig.tight_layout()
        fig.savefig('plots/ksTest_bright_faint'  + '.png')
        py.close()


        #---------------------------
        # KS test for reference stars
        #----------------------------
        if refStars != None:
            refGoodStars = set(refStars) & set(namesP)
            ref_ind = [(list(namesP)).index(refStar) for refStar in refGoodStars]

            if (direction == 'x'):
                ref_chi2 = chi2xx[ref_ind]
            if (direction == 'y'):
                ref_chi2 = chi2yy[ref_ind]
            if (direction == 'both'):
                ref_chi2 = np.concatenate((chi2xx[ref_ind], chi2yy[ref_ind]))

            #chi2 test
            #----------
            py.figure(figsize=(10,10))
            py.clf()
            bins2 = np.linspace(0, maxChi2, nbin_chi2)
            n, bins2, patches = py.hist(ref_chi2, color = 'b', label='ref', bins=bins2)
            chi2_bin = (bins2[0:-1]+bins2[1:])/2
            chi2_stand = stats.chi2(dof).pdf(chi2_bin) * len(ref_chi2) * maxChi2/nbin_chi2
            py.plot(chi2_bin, chi2_stand, 'ko', label='standard chi2')
            ne = n
            idx_temp = np.where(n==0)[0]
            ne[idx_temp] = 1
            chi2_ref = np.sum((n - chi2_stand)**2 / ne) / len(chi2_bin)
            py.annotate( r'$\chi^2_{red} = %.2f$ ' %(chi2_ref), color='b', xy=(0.7,0.5), xycoords='axes fraction')
            py.annotate( r'$N_{star}$ = %d' %(len(ref_chi2)/2), color='k',xy=(0.7,0.45), xycoords='axes fraction')
            py.annotate( 'additive error=%.3f mas' %(alignErr), xy=(0.1,0.95), xycoords='axes fraction')
            py.title( 'chi2 distribution for reference stars')
            py.legend(loc='upper right')
            py.savefig('plots/chi2Test_ref' + '.png' )
            py.ylim(0,120)
            py.close()

            #ks test
            #--------------
            py.clf()
            py.figure(figsize=(10,10))
            py.hist(ref_chi2, normed=1, bins=bins, cumulative=True, histtype='step', color='b', label='reference star')
            py.hist(stats.chi2.rvs(dof,size=1000), normed=1, bins=bins, cumulative=True, histtype='step', color='r', \
                    label='standard chi2 (dof= %d)' %dof)
            ks_ref_p = stats.kstest(ref_chi2, lambda x: stats.chi2.cdf(x,dof))
            py.title('chi2 distribution for reference stars')
            py.annotate(r'P=%4.2f, D=%4.2f $N_{star}=%d$' %(ks_ref_p[1], ks_ref_p[0], len(ref_chi2)/2), xy=(0.6,0.1), xycoords='axes fraction')
            py.annotate( 'additive error=%.3f mas' %(alignErr), xy=(0.1,0.95), xycoords='axes fraction')
            py.legend(loc='upper right')
            py.ylim(0,1.3)
            py.xlim(0,150)
            py.savefig('plots/ksTest_ref' + '.png')
            py.close()

    else:
        print( 'less than 10 stars are found, skip plotting chi2 distribution')
    py.close('all')

    if refStars != None:
        return ks_p, ks_bf, ks_ref_p, chi2_all, chi2_bf, chi2_ref
    else:
        return ks_p, ks_bf,chi2_all, chi2_bf

def chi2_pos():
    # use goodstars that defined in GoodStars function
    t = Table.read('plots/goodstars.hdf5', path='x')
    names = t['name']
    mag = t['mag']
    chi2x = t['chi2xa']
    chi2y = t['chi2ya']
    chi2 = t['chi2a']
    x = t['x']
    y = t['y']

    plt.clf()
    color_map = plt.cm.get_cmap('rainbow')
    sc = plt.scatter(x, y, c=chi2, marker='o', cmap=color_map, edgecolor=None, facecolors=None)
    plt.colorbar(sc)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('chi2 over position')
    plt.savefig('plots/chi2pos.png', format='png')
    return

def chi2VAplot():
    t = Table.read('plots/goodstars.hdf5', path='x')

    # plot chi2v vs velocity, velocity err
    plt.clf()
    v = np.hypot(t['vx'], t['vy'])
    plt.scatter(np.log10(t['chi2v']), v, c=t['mag'], s=50, edgecolor=None)
    plt.colorbar()
    plt.xlabel('log(chi2)')
    plt.ylabel('velocity(arcsec/yr)')
    plt.title('chi2 from linear fitting for all good stars')
    #plt.xlim(0,120)
    plt.savefig('plots/chi2_v.png', format='png')

    plt.clf()
    ve = np.hypot(t['vxe'], t['vye'])
    plt.scatter(np.log10(t['chi2v']), ve, c=t['mag'], s =50, edgecolor=None)
    plt.colorbar()
    plt.xlabel('log(chi2)')
    plt.ylabel('velocity error (arcsec/yr)')
    plt.title('chi2 from linear fitting for all good stars')
    #plt.xlim(0,120)
    plt.savefig('plots/chi2_ve.png', format='png')

    plt.clf()
    plt.scatter(np.log10(t['chi2xa']), t['ax']/t['axe'], label='ax', c=t['mag'], s=50, edgecolor=None)
    plt.scatter(np.log10(t['chi2ya']), t['ay']/t['aye'], label='ay', c=t['mag'], s=50, edgecolor=None)
    plt.colorbar()
    plt.xlabel('log(chi2)')
    plt.ylabel('accel/accel_err')
    plt.title('chi2 from acc fitting for all good stars')
    #plt.xlim(0,60)
    #plt.ylim(-10,10)
    plt.savefig('plots/chi2_aSigma.png', format='png')

    plt.clf()
    ae = np.hypot(t['axe'], t['aye'])
    plt.scatter(t['chi2a'], ae, c=t['mag'], s =50, edgecolor=None)
    cb = plt.colorbar()
    cb.set_label('Kp')
    plt.xlabel('chi2')
    plt.ylabel(r'acceleration error (arcsec/yr$^2$)')
    plt.title('chi2 from acc fitting for all good stars')
    plt.xlim(0,95)
    plt.savefig('plots/chi2_ae.png', format='png')

    plt.clf()
    plt.plot(t['mag'], ae*1000, '.')
    plt.xlabel('Kp')
    plt.ylabel(r'acceleration error (mas/yr$^2$)')
    plt.savefig('plots/mag_ae.png',format='png')

    return

def err_mag():
    t = Table.read('plots/goodstars.hdf5', path='x')
    xerr = t['Xerr']*1000
    yerr = t['Yerr']*1000
    xerr_a = t['Xerr_a']*1000
    yerr_a = t['Yerr_a']*1000
    xerr_p = t['Xerr_p']*1000
    yerr_p = t['Yerr_p']*1000
    mag = t['mag']

    plt.clf()
    plt.plot(mag, np.mean(xerr, axis=1), 'r.', label='perr')
    plt.plot(mag, np.mean(yerr, axis=1), 'r.')
    plt.plot(mag, np.mean(xerr_a, axis=1), 'm+', label='perr_a')
    plt.plot(mag, np.mean(yerr_a, axis=1), 'm+')
    plt.plot(mag, np.mean(xerr_p, axis=1), 'c+', label='perr_p')
    plt.plot(mag, np.mean(yerr_p, axis=1), 'c+')
    plt.legend()
    plt.xlabel('mag')
    plt.ylabel('average position error')
    plt.savefig('plots/err_mag.png', format='png')

    return

def compare_chi2(path1, path2, align1, align2, r_min=0.4, mag_cut=100, epochs_cut=30):
    """
    compare chi2 for a small number of samples that are not ref star
    Input:
    path1: path to the 1st align"
    path2: path to the 2nd align"
    align1: name for the 1st align
    align2: name for the 2nd align
    Output:
    Chi2_compare: chi2 compare"""

    # read in tables
    t1 = Table.read(path1+'/plots/all.txt', format='ascii.fixed_width')
    t2 = Table.read(path2+'/plots/all.txt', format='ascii.fixed_width')

    x1 = t1['x(arcsec)']
    y1 = t1['y(arcsec)']
    r1 = np.hypot(x1, y1)
    nEpoch1 = t1['nEpochs']
    mag1 = t1['mag']
    name1 = t1['name']
    chi2x1 = t1['chi2x']
    chi2y1 = t1['chi2y']

    x2 = t2['x(arcsec)']
    y2 = t2['y(arcsec)']
    r2 = np.hypot(x2, y2)
    nEpoch2 = t2['nEpochs']
    mag2 = t2['mag']
    name2 = t2['name']
    chi2x2 = t2['chi2x']
    chi2y2 = t2['chi2y']

    # cut sample
    idx1 = np.where((r1>r_min) & (mag1<mag_cut) & (nEpoch1>epochs_cut))[0]
    idx2 = np.where((r2>r_min) & (mag2<mag_cut) & (nEpoch2>epochs_cut))[0]
    name_chi2 = np.intersect1d(name1[idx1], name2[idx2])
    t_ref = Table.read(path2+'/source_list/label.dat', format='ascii')
    ref_star = t_ref['col1'][np.where((t_ref['col12']=='2') | (t_ref['col12']=='8') | (t_ref['col12']=='2,8'))[0]]
    non_ref = np.setdiff1d(name_chi2, ref_star)

    # plot chi2 comparsion
    idx1 = [list(name1).index(i) for i in non_ref]
    idx2 = [list(name2).index(i) for i in non_ref]
    plt.clf()
    plt.plot(chi2x1[idx1], chi2x2[idx2],'rx', label='chi2 in x')
    plt.plot(chi2y1[idx1], chi2y2[idx2],'g+', label='chi2 in y')
    plt.xlim(0.01,1000)
    plt.ylim(0.01,1000)
    plt.plot([0,100], [0,100],'m--')
    plt.xlabel('Red Accel Chi2 for ' + align1)
    plt.ylabel('Red Accel Chi2 for ' + align2)
    plt.gca().set_xscale('log')
    plt.gca().set_yscale('log')
    plt.legend(loc='best')
    plt.title('Reduced chi2 comparison for non-ref stars')
    plt.savefig('chi2_'+align1+'_'+align2+'.pdf', format='pdf')


def compare_ae_c_jackknife():
    """
    compare the acceleration uncertainty in polyfit_c and polyfit_jackknife
    """
    t_c = Table.read('polyfit_c/fit.accelFormal', format='ascii')
    t_jackknife = Table.read('polyfit_jackknife/fit.accelFormal', format='ascii')

    fig, axes = plt.subplots(1,2,figsize=(20,10))
    axes[0].hist(t_c['sig_a_x2']*1000, histtype='step',color='b',label='polyfit_c',bins=np.linspace(0,0.6,20))
    axes[0].hist(t_jackknife['sig_a_x2']*1000, histtype='step',color='r',label='polyfit_jackknife',bins=np.linspace(0,0.6,20))
    axes[0].set_xlabel('acceleratin uncertainty in x (mas/yr2)')
    axes[0].legend()
    axes[1].hist(t_c['sig_a_y2']*1000, histtype='step',color='b',label='polyfit_c',bins=np.linspace(0,0.6,20))
    axes[1].hist(t_jackknife['sig_a_y2']*1000, histtype='step',color='r',label='polyfit_jackknife',bins=np.linspace(0,0.6,20))
    axes[1].set_xlabel('acceleratin uncertainty in y (mas/yr2)')
    axes[1].legend()
    plt.savefig('plots/ae_c_jackknife.png', format='png')
    plt.close()
    return


def compare_a_c_jackknife():
    """
    compare the acceleration uncertainty in polyfit_c and polyfit_jackknife
    """
    t_c = Table.read('polyfit_c/fit.accelFormal', format='ascii')
    t_jackknife = Table.read('polyfit_jackknife/fit.accelFormal', format='ascii')

    fig, axes = plt.subplots(1,2,figsize=(20,10))
    axes[0].hist(t_c['a_x2']*1000, histtype='step',color='b',label='polyfit_c',bins=np.linspace(0,0.6,20))
    axes[0].hist(t_jackknife['a_x2']*1000, histtype='step',color='r',label='polyfit_jackknife',bins=np.linspace(0,0.6,20))
    axes[0].set_xlabel('acceleratin in x (mas/yr2)')
    axes[0].legend()
    axes[1].hist(t_c['a_y2']*1000, histtype='step',color='b',label='polyfit_c',bins=np.linspace(0,0.6,20))
    axes[1].hist(t_jackknife['a_y2']*1000, histtype='step',color='r',label='polyfit_jackknife',bins=np.linspace(0,0.6,20))
    axes[1].set_xlabel('acceleratin in y (mas/yr2)')
    axes[1].legend()
    plt.savefig('plots/a_c_jackknife.png', format='png')
    plt.close()
    return


def plot_pairs(rootDir='./', align='align/align_d_rms_1000_abs_t', points='points_2_s/', poly='polyfit_2_s/fit',
        dr_max=0.07, dv_max=0.003):
    """ plot dv_dm & dv_dr for all epochs"""
    t = Table.read('scripts/epochsInfo.txt', format='ascii')
    idx = np.where(t['doAlign']==1)[0]

    save_dic = {}
    pris = []
    fakes = []
    epochs = []

    for epoch in t['epoch'][idx]:
        pri, fake = plot_dm_dv_dr(rootDir=rootDir, align=align, points=points, poly=poly, epoch=epoch, dr_max=dr_max, dv_max=dv_max)
        save_dic[epoch+'_pri'] = pri
        save_dic[epoch+'_fake'] = fake

        for p in pri:
            pris.append(p)
        for f in fake:
            fakes.append(f)
        for e in range(len(pri)):
            epochs.append(epoch)

    # loop through the possible fakes sources in all epochs again
    # and for each pri sources, make a file with the possible fake sources and its epoch
    pris = np.array(pris)
    fakes = np.array(fakes)
    epochs = np.array(epochs)

    pris_uni = np.unique(pris)
    np.save(poly[:-3] + 'edge/stars_fake.npy', pris_uni)

    for p in pris_uni:
        idx = np.where(pris==p)[0]
        t = Table()
        t['epoch'] = epochs[idx]
        t['fake'] = fakes[idx]
        t.write(poly[:-3] + 'edge/star_' + p +'_fake.txt', format='ascii.fixed_width', delimiter=None, overwrite=True)

    return

def plot_dm_dv_dr(rootDir='./', align='align/align_d_rms_1000_abs_t', points='points_2_s/', poly='polyfit_2_s/fit',
        epoch='11julngs_kp', dr_max=0.07, dv_max=0.003, show_name=False, n_neighbor=5):
    """
    plot dr VS dm, with red points limited to dv < dv_max
    plot dv VS dm, with red points limited to dr < dr_max
    plot position and dr for pair stars with dv<dv_nmax, dr<dr_max
    dr_max, dv_max: dr and dv upper limit to find possible fake sources"""
    if not os.path.exists(poly[:-3] + 'edge'):
        os.mkdir(poly[:-3] + 'edge')

    #######
    # plot the dm vs dr
    #######
    # read in the linear fitting result
    s = starset.StarSet(rootDir + align)
    s.loadPolyfit(rootDir + poly, arcsec=1, silent=True)
    s.loadPolyfit(rootDir + poly, arcsec=1, accel=1, silent=True)
    vx = s.getArray('fitXv.v')
    vy = s.getArray('fitYv.v')
    vxe = s.getArray('fitXv.verr')
    vye = s.getArray('fitYv.verr')
    x0 = s.getArray('fitXv.p')
    y0 = s.getArray('fitYv.p')
    t0 = s.getArray('fitXv.t0')
    r = s.getArray('r2d')
    mag = s.getArray('mag')
    names = s.getArray('name')

    # read in position in certain epochs,
    # note: not all stars are detected
    t_label = Table.read(rootDir + 'scripts/epochsInfo.txt', format='ascii')
    idx_align = np.where(t_label['doAlign']==1)[0]
    idx_epoch = np.where(t_label[idx_align]['epoch']==epoch)[0][0]
    x = s.getArrayFromEpoch(idx_epoch, 'x')
    y = s.getArrayFromEpoch(idx_epoch, 'y')

    # find stars that are detected
    idx_det = np.where(x>-1e5)[0]
    vx = vx[idx_det]
    vy = vy[idx_det]
    vxe = vxe[idx_det]
    vye = vye[idx_det]
    x = x[idx_det]
    y = y[idx_det]
    mag = mag[idx_det]
    r = r[idx_det]
    names = np.array(names)[idx_det]
    N = len(mag)

    # make pairs between every two stars
    idx_pair = list(itertools.combinations(range(len(mag)), 2))
    idx_pair = np.array(idx_pair)

    dm = mag[idx_pair][:,0]-mag[idx_pair][:,1]
    dvx = vx[idx_pair][:,0]-vx[idx_pair][:,1]
    dvy = vy[idx_pair][:,0]-vy[idx_pair][:,1]
    dv = np.hypot(dvx, dvy)
    vxe_pair = np.hypot(vxe[idx_pair][:,0], vxe[idx_pair][:,1])
    vye_pair = np.hypot(vye[idx_pair][:,0], vye[idx_pair][:,1])
    ve_pair = np.hypot(vxe_pair, vye_pair)
    dvx_sig = dvx/vxe_pair
    dvy_sig = dvy/vye_pair
    dv_sig = dv/ve_pair
    dx = x[idx_pair][:,0]-x[idx_pair][:,1]
    dy = y[idx_pair][:,0]-y[idx_pair][:,1]
    dr = np.hypot(dx, dy)

    plt.figure(figsize=(10,10))
    idx_dr = np.where(dr<dr_max)[0]
    plt.plot(abs(dm), dv, 'b.')
    plt.plot(abs(dm[idx_dr]), dv[idx_dr], 'r.')
    plt.xlabel('dm')
    plt.ylabel('dv')
    plt.ylim(0, dv_max)
    plt.savefig(poly[:-3] + 'edge/epoch_' + epoch + '_dm_dv.png', format='png')
    plt.close()

    plt.figure(figsize=(15,15))
    idx_dv = np.where(dv<dv_max)[0]
    plt.plot(abs(dm), dr, 'b.')
    plt.plot(abs(dm[idx_dv]), dr[idx_dv], 'r.')
    plt.xlabel('dm')
    plt.ylabel('dr')
    plt.ylim(0, dr_max)
    plt.savefig(poly[:-3] + 'edge/epoch_' + epoch + '_dm_dr.png', format='png')
    plt.close()

    # print( out the stars that could possibly be fake stars)
    idx_q = np.where((abs(dvx)<dv_max) & (abs(dvy)<dv_max) & (dr<dr_max))[0]
    idx_pair_q = idx_pair[idx_q]

    # keep record of info of each pair for future plots
    names1_q = names[idx_pair_q[:,0]]
    names2_q = names[idx_pair_q[:,1]]
    x1_q = x[idx_pair_q[:,0]]
    y1_q = y[idx_pair_q[:,0]]
    x2_q = x[idx_pair_q[:,1]]
    y2_q = y[idx_pair_q[:,1]]
    vx1_q = vx[idx_pair_q[:,0]]
    vy1_q = vy[idx_pair_q[:,0]]
    vx2_q = vx[idx_pair_q[:,1]]
    vy2_q = vy[idx_pair_q[:,1]]
    vxe1_q = vxe[idx_pair_q[:,0]]
    vye1_q = vye[idx_pair_q[:,0]]
    ve1_q = np.hypot(vxe1_q, vye1_q)
    vxe2_q = vxe[idx_pair_q[:,1]]
    vye2_q = vye[idx_pair_q[:,1]]
    ve2_q = np.hypot(vxe2_q, vye2_q)
    m1_q = mag[idx_pair_q[:,0]]
    m2_q = mag[idx_pair_q[:,1]]

    dx_q = dx[idx_q]
    dy_q = dy[idx_q]
    dvx_q = dvx[idx_q]
    dvy_q = dvy[idx_q]
    dv_q = dv[idx_q]
    vxe_q = vxe_pair[idx_q]
    vye_q = vye_pair[idx_q]
    dvx_sig_q = dvx_sig[idx_q]
    dvy_sig_q = dvy_sig[idx_q]
    dv_sig_q = dv_sig[idx_q]
    dm_q = dm[idx_q]
    N_q = len(idx_q)

    # put bright star in the fist array and faint star in the second array
    for i in range(N_q):
        m1i = m1_q[i]
        m2i = m2_q[i]
        if m1i < m2i:
            pass
        else:
            names1_q[i], names2_q[i] = names2_q[i], names1_q[i]
            m1_q[i], m2_q[i] = m2_q[i], m1_q[i]
            x1_q[i], x2_q[i] = x2_q[i], x1_q[i]
            y1_q[i], y2_q[i] = y2_q[i], y1_q[i]
            vx1_q[i], vx2_q[i] = vx2_q[i], vx1_q[i]
            vy1_q[i], vy2_q[i] = vy2_q[i], vy1_q[i]
            vxe1_q[i], vxe2_q[i] = vxe2_q[i], vxe1_q[i]
            vye1_q[i], vye2_q[i] = vye2_q[i], vye1_q[i]

            dm_q[i] *= -1.
            dx_q[i] *= -1.
            dy_q[i] *= -1.
            dvx_q[i] *= -1.
            dvy_q[i] *= -1.


    for i in range(N_q):
        _output1 = "%s: m=%.1f, x=%.2f as, y=%.2f as, vx=%.1f mas/yr, vy=%.1f mas/yr, vxe=%.1f mas/yr, vye=%.1f mas/yr"
        _output2 = "dm=%.1f, dx=%d mas, dy=%d mas, dvx=%.1f mas/yr,  dvy=%.1f mas/yr, vxe=%.1f mas/yr, vye=%.1f mas/yr\n"
        print(_output1 %(names1_q[i], m1_q[i], x1_q[i], y1_q[i], vx1_q[i]*1000, vy1_q[i]*1000, vxe1_q[i]*1000, vye1_q[i]*1000))
        print(_output1 %(names2_q[i], m2_q[i], x2_q[i], y2_q[i], vx2_q[i]*1000, vy2_q[i]*1000, vxe2_q[i]*1000, vye2_q[i]*1000))
        print(_output2 %(dm_q[i], dx_q[i]*1000, dy_q[i]*1000, dvx_q[i]*1000, dvy_q[i]*1000, vxe_q[i]*1000, vye_q[i]*1000))

    # plot the position and quiver of those pair stars
    plt.figure()
    plt.clf()
    plt.plot(x, y, 'bx')
    plt.quiver(x1_q, y1_q, dx_q, -dy_q, lw=0.001)

    for i in range(N_q):
        plt.plot(x1_q[i], y1_q[i], 'ro')
        if show_name:
            plt.annotate(names1_q[i] + ':' + str(np.int(np.degrees(np.arctan(dy_q[i]/dx_q[i])))), (x1_q[i], y1_q[i]))
    plt.gca().invert_xaxis()

    # use 10 nearest neighbors to find fake sources
    pri = []
    fake = []
    fake_idx = []
    for i in range(N_q):
        name1i = names1_q[i]
        xi = x1_q[i]
        yi = y1_q[i]
        dxi = dx_q[i]
        dyi = dy_q[i]

        # calculate the position angle [-90,90],
        #the reason is some star has more than one fake source in opposite direction
        thetai = np.degrees(np.arctan(dyi/dxi))

        # find the nearest stars
        d = np.hypot(x1_q - xi, y1_q - yi)
        near_idx = d.argsort()[1:1+n_neighbor]

        data = np.degrees(np.arctan(dy_q[near_idx]/dx_q[near_idx]))
        data_std = np.std(data)
        #data_mean = np.mean(data)
        #data_cut_idx = np.where((data<data_mean+data_std) & (data>data_mean-data_std))[0]
        dtheta=20
        data_cut_idx = np.where((data<thetai+dtheta) & (data>thetai-dtheta))[0]
        data_cut = data[data_cut_idx]

        # alternative way: not work as well
        #use sigma_clipped stats -> mean and std without outlier
        #from astropy.stats import sigma_clipped_stats
        #near_theta, median, std_theta = sigma_clipped_stats (data, sigma=1, iters=1)
        #from scipy.stats import sigmaclip
        #data_cut = sigmaclip(data, low=3, high=3)


        if len(data_cut)<3:
            continue
        else:
            near_theta = np.mean(data_cut)
            std_theta = np.std(data_cut)

        # 10 and 8 degree are hard coded here
        if (abs(thetai-near_theta)<10) & (std_theta<8):
            # decide if this star is unique
            pri_idx = np.where(names1_q==name1i)[0]
            if len(pri_idx) == 1:
                fake_idx.append(i)
                fake.append(names2_q[i])
                pri.append(names1_q[i])
            else:
                for j in pri_idx:
                    name2j = names2_q[j]
                    pri.append(names1_q[i])
                    fake_idx.append(j)
                    fake.append(names2_q[j])

    for i in range(len(fake)):
        print('primary: %s , fake: %s' %(pri[i], fake[i]))

    plt.quiver(x1_q[fake_idx], y1_q[fake_idx], dx_q[fake_idx], -dy_q[fake_idx], color='r',lw=0.001)
    plt.savefig(poly[:-3] + 'edge/epoch_' + epoch + '_dr_dv_quiver.png', format='png')
    plt.close()

    # write to a table
    if len(pri) != 0:
        t = Table()
        t['pri'] = pri
        t['fake'] = fake
        t.write(poly[:-3] + 'edge/epoch_' + epoch + '_pri_fake.txt', format='ascii.fixed_width', delimiter=None, overwrite=True)

    return pri, fake

# run higher order polynomial fit
def run_hpoly(poly_orders):
    # run polyfit with higher order polynomial fit
    points = 'points_4_trim/'
    for i in poly_orders:
        poly = 'polyfit_5_high_order' + str(i)
        if not os.path.exists(poly):
            os.mkdir(poly)
        print('now running polyfit in ' + poly)
        cmd = 'polyfit -d ' + str(i) + ' -linear -i ./align/align_d_rms_1000_abs_t'
        cmd += ' -jackknife -points '+ points + ' -o '+ poly + '/fit'
        os.system(cmd)
 

