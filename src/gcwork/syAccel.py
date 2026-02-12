from gcwork import objects
from gcwork import starset
from gcwork import util
from gcwork import orbits
from gcwork import young
import sqlite3 as sqlite
import scipy
from scipy import stats
from gcwork import starTables
import pickle
#import nmpfit_sy
import asciidata, os, sys
from pylab import *
import numpy as np
import pylab as py
import math
import histNofill
#import matplotlib.axes3d as p3
import pdb

# Several functions were taken from:
# /ghezgroup/code/python/gcwork/polyfit/accel.py,
# velocity.py and residuals.py, written by JLu.

def usetexTrue():
    rc('text', usetex=True)
    rc('font', **{'family':'sans-serif', 'size':16})
    rc('axes', titlesize=16, labelsize=16)
    rc('xtick', labelsize=12)
    rc('ytick', labelsize=12)

def usetexFalse():
    rc('text', usetex=False)
    rc('font', family='sans-serif', size=14)
    rc('axes', titlesize=16, labelsize=16)
    rc('xtick', labelsize=14)
    rc('ytick', labelsize=14)

#root = '/u/syelda/research/gc/aligndir/'
root = '/g/ghez/align/'

pi = math.pi

# Mass and Ro from S0-2 - Ghez et al. 2008
mass = 4.07e6
dist = 7960.0
G = 6.6726e-8
msun = 1.99e33
GM = G * mass * msun
sec_in_yr = 3.1557e7
cm_in_au = 1.496e13
cm_in_pc = 3.086e18
km_in_pc = 3.086e13
au_in_pc = 206265.0
asy_to_kms = dist * cm_in_au / (1e5 * sec_in_yr)

def plotVelocityMap(alnDir='11_08_29/',
                    align='align/align_d_rms_1000_abs_t', poly='polyfit_c/fit'):

    outdir = root + alnDir + 'plots/'
    s = starset.StarSet(root + alnDir + align)
    s.loadPolyfit(root + alnDir + poly)

    name = s.getArray('name')
    x = s.getArray('fitXv.p')
    y = s.getArray('fitYv.p')
    vx = s.getArray('fitXv.v')
    vy = s.getArray('fitYv.v')
    vxe = s.getArray('fitXv.verr')
    vye = s.getArray('fitYv.verr')
    vx *= 1e3
    vy *= 1e3
    vxe *= 1e3
    vye *= 1e3

    # temp
    cnt = s.getArray('velCnt')
    idx = np.where(cnt > 20)[0]
    print 'Ratio of X to Y proper motion errors: %4.3f' % (vxe[idx] / vye[idx]).mean()

    #1 outlier:
    #foo = np.where(vy > 500)[0]
    #print name[foo],x[foo],y[foo],vx[foo],vy[foo]

    py.clf()
    arrScale = 1.0
    qvr = py.quiver([x], [y], [vx], [vy], headwidth=3, color='black')
    py.quiverkey(qvr, 4.0, 5.0, 15, '15 mas/yr', coordinates='data')
    py.savefig(outdir + 'velMap.png')
    #py.savefig(outdir + 'velMap.eps')

    # To get young stars:
    #yng = youngNames.loadYoungStars()
    #print [yng[ii] for ii in range(len(yng))]

    py.clf()
    py.figure(figsize=(12,5))
    py.subplots_adjust(wspace=0.25, hspace=0.25, left=0.08, right=0.95, top=0.95)
    py.subplot(1,3,1)
    py.semilogy(np.hypot(x,y), np.abs(vx), 'r.')
    py.xlabel('Projected Radius (arcsec)')
    py.ylabel('|X Velocity| (mas/yr)')
    py.subplot(1,3,2)
    py.semilogy(np.hypot(x,y), np.abs(vy), 'b.')
    py.xlabel('Projected Radius (arcsec)')
    py.ylabel('|Y Velocity| (mas/yr)')
    py.subplot(1,3,3)
    py.semilogy(np.hypot(x,y), np.hypot(vx,vy), 'k.')
    py.xlabel('Projected Radius (arcsec)')
    py.ylabel('Proper Motion (mas/yr)')
    py.savefig(outdir + 'pm_vs_r2d.png')

    

def loadPop(alnDir='11_08_29/',
            align = 'align/align_d_rms_1000_abs_t',
            poly='polyfit_c/fit',points='points_c/', starlist='yngstars'):
    """Load a specific population of stars.

    Loads align data, omitting stars (according to chosen starlist) 
    Set starlist='all' to keep young stars in list, while still
    making LGS cuts. Set starlist='oldstars' to only include old stars
    (actually, these are non-known-young stars). Set
    starlist='yngstars' to only include known young stars.
    """

    starlists = ['all','oldstars','yngstars']

    if not(starlists.__contains__(starlist)):
        print "Star list must be 'all', 'oldstars', or 'yngstars'"
        print "Using default starlist='oldstars'"
        starlist='yngstars'

    s = starset.StarSet(root + alnDir + align)
    s.loadPolyfit(root + alnDir + poly, accel=0, arcsec=1)
    s.loadPolyfit(root + alnDir + poly, accel=1, arcsec=1)
    s.loadPoints(root + alnDir + points)

    # Get the most up-to-date young stars from Tuan's sql data base.
    yng = young.youngStarNames()

    # Pull out old stars
    yngstars = []
    names = [star.name for star in s.stars]

    for name in yng:
	try:
	    idx = names.index(name)
	    star = s.stars[idx]
	    yngstars.append(star)
	except ValueError:
	    continue

    oldstars = []

    if (starlist=='all' or starlist=='oldstars'):
        # Include all stars
        for star in s.stars:
	    oldstars.append(star)

    if starlist=='oldstars':
        # Remove young stars
        for star in yngstars:
            try:
   	        oldstars.remove(star)
            except ValueError:
                continue

    if starlist=='yngstars':
        for star in yngstars:
            r = np.sqrt(star.x**2+star.y**2)
            # Don't include the central arcsecond young stars:
            #if r > 0.8:
	    #    oldstars.append(star)
	    oldstars.append(star)
                # This is called oldstars but it's now the young stars
         

    # Remove Sgr A from list
    try:
        idx = names.index('SgrA')
        star = s.stars[idx]
        oldstars.remove(star)
    except ValueError:
        pass

#    # Remove any stars with < 3 points, as accelerations
#    # cannot be fit to these anyway
#    for star in oldstars:
#        if star.pointsCnt < 3:
#            oldstars.remove(star)

    # Build oldstars 
    for star in oldstars:
	#star.x = star.fitXv.p
	#star.y = star.fitYv.p
	#star.vx = star.fitXv.v * asy_to_kms
	#star.vy = star.fitYv.v * asy_to_kms
	#star.xerr = star.fitXv.perr
	#star.yerr = star.fitYv.perr
	#star.vxerr = star.fitXv.verr * asy_to_kms
	#star.vyerr = star.fitYv.verr * asy_to_kms
	#star.xchi2 = star.fitXv.chi2
	#star.ychi2 = star.fitYv.chi2
	#star.xchi2r = star.fitXv.chi2red
	#star.ychi2r = star.fitYv.chi2red

        # CAUTION! A star may have a velocity fit but NOT
        # have an accel fit (in the case of having only 3 points)
        # Similarly, a star may be in the align files, but not in
        # polyfit files (fit.linearFormal or fit.accelFormal)
        try:
            # In arcsec, mas/yr, mas/yr^2
            star.x0 = star.fitXa.p
            star.y0 = star.fitYa.p
            star.x0e = star.fitXa.perr
            star.y0e = star.fitYa.perr
            star.vx = star.fitXa.v
            star.vy = star.fitYa.v
            star.vxe = star.fitXa.verr
            star.vye = star.fitYa.verr
            star.ax = star.fitXa.a
            star.ay = star.fitYa.a
            star.axe = star.fitXa.aerr
            star.aye = star.fitYa.aerr
            star.cnt = star.velCnt
	    star.xchi2 = star.fitXa.chi2
	    star.ychi2 = star.fitYa.chi2
	    star.xchi2r = star.fitXa.chi2red
	    star.ychi2r = star.fitYa.chi2red
            star.t0x = star.fitXa.t0
            star.t0y = star.fitYa.t0
        except AttributeError, e:
            print '%s does not have acceleration fit' % star.name
            oldstars.remove(star)
    
    s.stars = oldstars

    if starlist=='all':
        tag = 'stars'
    if starlist=='oldstars':
        tag = 'old stars'
    if starlist=='yngstars':
        tag = 'young stars'

    print 'Found %d %s' % (len(oldstars),tag)

    return s


def histAccel(alnDir='11_10_26/',
              align = 'align/align_d_rms_1000_abs_t',
              poly='polyfit_c/fit', points='points_c/', sigma=5.0,
              f_test=True, pvalue=4.0, 
              starlist='all', plotSigAcc=False, magCut=16, nEpochs=30):
    """
    Make a histogram of the accelerations in the radial/tangential,
    X/Y, and inline/perp. direction of motion. Also, the large outliers
    (> sigma) are printed out.

    Inputs:
    alnDir   = The root directory of an astrometry analysis
                (e.g. '08_02_16/' or './' if you are in the directory).
    align     = The align root file name (including the directory relative
                to root). Make sure that polyfit was run on this align
		output.
    poly      = The polyfit root file name (including the directory relative
                to root). This should be run on the same align as above.
    points    = The points directory.
    starlist  = Only plot specific subset of stars. Must be 'oldstars', 
                'yngstars', 'all'.
    plotSigAcc = Set to True to plot the fits for stars with significant radial
                 accelerations.
    nEpochs    = Number of epochs a star must have been detected in for this analysis 

    Output:
    plots/polyfit_hist_accel.eps (and png)
    -- Contains the histograms of the accelerations.

    plots/polyfit_hist_accel_nepochs.eps (and png)
    -- Contains a plot of number of stars vs. acceleration significance.
    """
    outdir = root + alnDir + 'plots/'
    
    s = loadPop(alnDir=alnDir,starlist=starlist,align=align,poly=poly,points=points)

    names = s.getArray('name')

    # In arcsec, mas/yr, mas/yr^2
    x0 = s.getArray('x0')
    y0 = s.getArray('y0')
    x0e = s.getArray('x0e')
    y0e = s.getArray('y0e')
    vx = s.getArray('vx')
    vy = s.getArray('vy')
    vxe = s.getArray('vxe')
    vye = s.getArray('vye')
    ax = s.getArray('ax')
    ay = s.getArray('ay')
    axe = s.getArray('axe')
    aye = s.getArray('aye')
    cnt = s.getArray('cnt')
    mag = s.getArray('mag')

    # Make an epochs cut
    idx = np.where((cnt > nEpochs) & (mag < magCut))[0]
    x0 = x0[idx]
    y0 = y0[idx]
    x0e = x0e[idx]
    y0e = y0e[idx]
    vx = vx[idx]
    vy = vy[idx]
    vxe = vxe[idx]
    vye = vye[idx]
    ax = ax[idx]
    ay = ay[idx]
    axe = axe[idx]
    aye = aye[idx]
    cnt = cnt[idx]
    mag = mag[idx]
    names = [names[nn] for nn in idx]

    # Get accelerations in the radial/tangential direction
    r = np.hypot(x0, y0)
    ar = ((ax*x0) + (ay*y0)) / r
    at = ((ax*y0) - (ay*x0)) / r
    are =  (axe*x0/r)**2 + (aye*y0/r)**2
    are += (y0*x0e*at/r**2)**2 + (x0*y0e*at/r**2)**2
    are =  np.sqrt(are)
    ate =  (axe*y0/r)**2 + (aye*x0/r)**2
    ate += (y0*x0e*ar/r**2)**2 + (x0*y0e*ar/r**2)**2
    ate =  np.sqrt(ate)

    # Lets also do parallel/perpendicular to velocity
    v = np.sqrt(vx**2 + vy**2)
    am = ((ax*vx) + (ay*vy)) / v
    an = ((ax*vy) - (ay*vx)) / v
    ame = np.sqrt((axe*vx)**2 + (aye*vy)**2) / v
    ane = np.sqrt((axe*vy)**2 + (aye*vx)**2) / v

    # Total acceleration
    atot = py.hypot(ax, ay)
    atoterr = np.sqrt((ax*axe)**2 + (ay*aye)**2) / atot

    # Print out some info
    foo = np.where((mag < 13.5) & (cnt > nEpochs))[0]
    print 'For stars brighter than K=13.5 (N=%s)' % str(len(foo))
    print 'Average radial acceleration error: %5.3f +- %5.3f mas/yr^2' % \
          (are[foo].mean()*1.e3, are[foo].std()*np.sqrt(len(foo)/(len(foo)-1.0))*1.e3)
    print 'Average tangential acceleration error: %5.3f +- %5.3f mas/yr^2' % \
          (ate[foo].mean()*1.e3, ate[foo].std()*np.sqrt(len(foo)/(len(foo)-1.0))*1.e3)
    print 'Average X velocity error: %5.3f +- %5.3f mas/yr' % \
          (vxe[foo].mean()*1.e3, vxe[foo].std()*np.sqrt(len(foo)/(len(foo)-1.0))*1.e3)
    print 'Average Y velocity error: %5.3f +- %5.3f mas/yr' % \
          (vye[foo].mean()*1.e3, vye[foo].std()*np.sqrt(len(foo)/(len(foo)-1.0))*1.e3)
    print 'Average X position error: %5.3f +- %5.3f mas' % \
          (x0e[foo].mean()*1.e3, x0e[foo].std()*np.sqrt(len(foo)/(len(foo)-1.0))*1.e3)
    print 'Average Y position error: %5.3f +- %5.3f mas' % \
          (y0e[foo].mean()*1.e3, y0e[foo].std()*np.sqrt(len(foo)/(len(foo)-1.0))*1.e3)
    print ''

    foo = np.where((mag < 15.5) & (cnt > nEpochs))[0]
    print 'For stars brighter than K=15.5 (N=%s)' % str(len(foo))
    print 'Median radial acceleration error: %5.3f +- %5.3f mas/yr^2' % \
          (np.median(are[foo])*1.e3, are[foo].std()*np.sqrt(len(foo)/(len(foo)-1.0))*1.e3)
    print 'Average radial acceleration error: %5.3f +- %5.3f mas/yr^2' % \
          (are[foo].mean()*1.e3, are[foo].std()*np.sqrt(len(foo)/(len(foo)-1.0))*1.e3)
    print 'Average tangential acceleration error: %5.3f +- %5.3f mas/yr^2' % \
          (ate[foo].mean()*1.e3, ate[foo].std()*np.sqrt(len(foo)/(len(foo)-1.0))*1.e3)
    print 'Average X velocity error: %5.3f +- %5.3f mas/yr' % \
          (vxe[foo].mean()*1.e3, vxe[foo].std()*np.sqrt(len(foo)/(len(foo)-1.0))*1.e3)
    print 'Average Y velocity error: %5.3f +- %5.3f mas/yr' % \
          (vye[foo].mean()*1.e3, vye[foo].std()*np.sqrt(len(foo)/(len(foo)-1.0))*1.e3)
    print 'Average X position error: %5.3f +- %5.3f mas' % \
          (x0e[foo].mean()*1.e3, x0e[foo].std()*np.sqrt(len(foo)/(len(foo)-1.0))*1.e3)
    print 'Average Y position error: %5.3f +- %5.3f mas' % \
          (y0e[foo].mean()*1.e3, y0e[foo].std()*np.sqrt(len(foo)/(len(foo)-1.0))*1.e3)

    ##########
    #
    # KS Test... Are the distributions Normal?
    #
    ##########
    sigmaR = ar/are
    sigmaT = at/ate
    sigmaAll = concatenate([sigmaR, sigmaT])
    (ksdR, kspR) = stats.stats.kstest(sigmaR, 'norm', N=len(sigmaR))
    (ksdT, kspT) = stats.stats.kstest(sigmaT, 'norm', N=len(sigmaT))
    (ksdA, kspA) = stats.stats.kstest(sigmaAll, 'norm', N=len(sigmaAll))
    #print 'KS Test for normality (prob that observed is gaussian):'
    #print '\tRadial Acc. KS Prob. = %5.3f' % (kspR)
    #print '\tTangen Acc. KS Prob. = %5.3f' % (kspT)
    #print '\tCombo Acc. KS Prob. = %5.3f' % (kspA)
    print ''

    ###########
    #
    # F Test -- Accel or Velocity fit?
    #
    ###########
    if f_test == True:
        # Pass in the star names and run the F test
        pass_f = run_f_test(names,pvalue,alnDir,align,poly,points,verbose=True)
    else:
        pass_f = np.ones(len(names)) # check everything

    def makePlots(val, acc, r, v, pass_f, pos, label):
        py.figure(1)
        py.subplot(3, 2, pos)
        py.hist(val, bins=range(-25, 25, 1), color='b')
        #py.axis([-8, 8, 0, (len(val)/3) + 2])
        py.title(label)

        fmt = '%15s  %5.2f   %5.3f"  %8.3f mas/yr^2  %8.2f sigma  %2d epochs  %3s'
        hdr = '%15s  %5s  %6s  %17s  %14s  %8s  %3s '
        if (label == 'Radial'):
            # Get significant radial accelerations (physical)
            idx = (np.where((val < -sigma)))[0]
            print '%s Significant %s (physical) Accelerations' % (len(idx), label)
            print hdr % ('Name','K','r (")', 'a_rad (mas/yr^2)','a_rad (sigma)', 'Nepochs', 'Fit')
            if len(idx) > 0:
                for i in idx:
                    if pass_f[i] == 1:
                        fit = 'Acc'
                    elif pass_f[i] == 0:
                        fit = 'Vel'
                    print fmt  % (names[i], mag[i], r[i], acc[i]*1e3, val[i], cnt[i], fit)
                    py.figure(2)
                    py.plot(r[i],acc[i]*1e3,'k.')

                    if plotSigAcc == True:
                        py.figure(2)
                        py.clf()
                        plotStar(names[i].strip(),align=align,poly=poly,points=points)

                py.xlabel('R (arcsec)')
                py.ylabel('Negative Radial Acceleration (mas/yr/yr)')
                py.savefig(outdir + 'physical_radial.png')
                py.close(2)    

            # Get significant unphysical accelerations (positive radial)
            idx = (np.where((val > sigma)))[0]
            print
            print '%s Significant Positive %s (unphysical) Accelerations' % \
                  (len(idx), label)
            print hdr % ('Name','K','r (")', 'a_rad (mas/yr^2)','a_rad (sigma)',
                         'Nepochs', 'Fit')
            if len(idx) > 0:
                for i in idx:
                    if pass_f[i] == 1:
                        fit = 'Acc'
                    elif pass_f[i] == 0:
                        fit = 'Vel'
                    print fmt % (names[i], mag[i], r[i], acc[i]*1e3, val[i], cnt[i], fit)
                    py.figure(3)
                    py.plot(r[i],acc[i]*1e3,'k.')

                    if plotSigAcc == True:
                        py.figure(2)
                        py.clf()
                        plotStar(names[i].strip(),align=align,poly=poly,
                                 points=points,radial=True,suffix='_unph')

                py.xlabel('R (arcsec)')
                py.ylabel('Positive (unphysical) Radial Acceleration (mas/yr/yr)')
                py.savefig(outdir + 'unphysical_radial.png')
                print ''
                print 'Median positive radial acceleration = %5.3f mas/yr/yr' % \
                      np.median(acc[idx]*1.e3)
                    
                py.close(3)    

        if (label == 'Tangential'):
            # Get significant unphysical accelerations (tangential)
            idx = (np.where((np.abs(val) > sigma)))[0]
            print
            print '%s Significant %s (unphysical) Accelerations' % (len(idx), label)
            print hdr % ('Name','K','r (")', 'a_tan (mas/yr^2)','a_tan (sigma)', 'Nepochs', 'Fit')
            if len(idx) > 0:
                for i in idx:
                    if pass_f[i] == 1:
                        fit = 'Acc'
                    elif pass_f[i] == 0:
                        fit = 'Vel'
                    print fmt % (names[i], mag[i], r[i], acc[i]*1e3, val[i], cnt[i], fit)
                    py.figure(4)
                    py.plot(r[i],acc[i]*1e3,'k.')

                    if plotSigAcc == True:
                        py.figure(2)
                        py.clf()
                        plotStar(names[i].strip(),align=align,poly=poly,points=points,
                                 radial=True,suffix='_unph')

                py.axis([0,4,-0.3,0.3])
                py.xlabel('R (arcsec)')
                py.ylabel('Tangential (unphysical) Acceleration (mas/yr/yr)')
                py.savefig(outdir + 'unphysical_tangential.png')
                print ''
                print 'Median tangential acceleration = %5.3f mas/yr/yr' % \
                      np.median(acc[idx]*1.e3)
                py.close(4)

    tag = ''
    if starlist=='all':
        tag = '_all'
    if starlist=='oldstars':
        tag = '_old'
    if starlist=='yngstars':
        tag = '_yng'

   # py.figure(3, figsize=(7.5,10))
   # py.clf()
   # makePlots(atot / atoterr, 1, 'Total')

    py.clf()

    for ii in range(4):
        py.figure(ii)
        py.figure(ii,figsize=(10,10))
        py.subplots_adjust(wspace=0.3, hspace=0.3, right=0.95, top=0.95)
        py.clf()

    # XY (in units of sigma)
    makePlots(ax / axe, ax, r, v, pass_f, 1, 'X')
    makePlots(ay / aye, ay, r, v, pass_f, 2, 'Y')
    
    # Radial/Tangential (in units of sigma)
    makePlots(ar / are, ar, r, v, pass_f, 3, 'Radial')
    makePlots(at / ate, at, r, v, pass_f, 4, 'Tangential')

    # In line of Motion and out (in units of sigma)
    makePlots(an / ane, an, r, v, pass_f, 5, 'Perpendicular to v')
    makePlots(am / ame, am, r, v, pass_f, 6, 'Parallel to v')

    #py.savefig(outdir + 'polyfit_hist_accel%s.eps' % tag)
    py.savefig(outdir + 'polyfit_hist_accel%s.png' % tag)

    # Analyze the non-central arcsec sources for Nepochs threshold
    idx = (np.where(r > 0.8))[0]
    #idx = (np.where(r > 0.0))[0]

    py.close(1)

    py.figure(5)
    py.clf()
    py.subplots_adjust(hspace=0.2, left=0.15, right=0.85,
                       top=0.9, bottom=0.1)
    py.subplot(2, 1, 1)
    py.plot(ar[idx] / are[idx], cnt[idx], 'k.')
    py.axis([-20, 20, 0, 45])
    py.xlabel('Radial Acc. Sig. (sigma)')
    py.ylabel('Number of Epochs Detected')

    py.subplot(2, 1, 2)
    py.plot(at[idx] / ate[idx], cnt[idx], 'k.')
    py.axis([-20, 20, 0, 45])
    py.xlabel('Tangent. Acc. Sig. (sigma)')
    py.ylabel('Number of Epochs Detected')

    py.savefig(outdir + 'polyfit_hist_accel_nepochs%s.png' % tag)
    py.close(5)

def run_f_test(stars, pvalue, alnDir, align, poly, points, returnAcc=False,
               verbose=True):
    """
    Send in list of names and check whether a velocity or accel fit is best based
    on the F test. Calls Tuan's accelclass code to do the F test.

    Input:
    stars -- List of names to run F test on
    pvalue -- Significance threshold for F test

    """
    outdir = root + alnDir + 'plots/'
    
    signif = scipy.special.erfc(pvalue/np.sqrt(2.0))
    print 'Significance value: %8.3e' % signif

    import accel_class as acc
    data = acc.accelClass(rootDir=root+alnDir,align=align,poly=poly,points=points,
                          verbose=verbose)
    data.computeFTest()
    #data.computeFTestJerk()

    xFprob = data.xFProb # F test p-value for accel vs. vel
    yFprob = data.yFProb

    # Round to nearest decimal
    xFprob = np.around(xFprob,decimals=4)
    yFprob = np.around(yFprob,decimals=4)

    # Other info from this align
    names = data.names
    nEpochs = data.nEpochs # original number of epochs -- but not necessarily correct
    mag = data.mag
    x = data.x
    y = data.y
    xe = data.xerr
    ye = data.yerr
    vx = data.vx
    vy = data.vy
    vxe = data.vxe
    vye = data.vye
    ax = data.ax
    ay = data.ay
    axe = data.axe
    aye = data.aye
    r2d = data.r2d
    cnt = data.nEpochs

    acc = np.zeros(len(stars))
    accnames = []
    accXFp = []
    accYFp = []

    for ss in range(len(stars)):
        idx = np.where(names == stars[ss])[0]
        xFp = xFprob[idx]
        yFp = yFprob[idx]

        #print stars[ss], xFp, yFp

        if ((xFp < signif) | (yFp < signif)):
            acc[ss] = 1 # Accel fit
            accnames = np.concatenate([accnames, [stars[ss]]])
            accXFp = np.concatenate([accXFp, xFp])
            accYFp = np.concatenate([accYFp, yFp])
        else:
            acc[ss] = 0 # Velocity fit
    
    # Pass back arrays specifying accelerating names and F probs
    if returnAcc == True:
        return accnames, accXFp, accYFp
    # Pass back an array specifying if accel or velocity fit is best
    else:
        return acc


def accelLimit(alnDir='11_10_26/',
              align = 'align/align_d_rms_1000_abs_t',
              poly='polyfit_c/fit', points='points_c/',
              starlist='all', nEpochs=30):
    """
    Plot plane-of-the-sky acceleration limits as a function of projected distance.

    Inputs:
    alnDir   = The root directory of an astrometry analysis
                (e.g. '08_02_16/' or './' if you are in the directory).
    align     = The align root file name (including the directory relative
                to root). Make sure that polyfit was run on this align
		output.
    poly      = The polyfit root file name (including the directory relative
                to root). This should be run on the same align as above.
    points    = The points directory.
    starlist  = Only plot specific subset of stars. Must be 'oldstars', 
                'yngstars', 'all'. 
    nEpochs   = Minimum number of epochs a star must be detected in (default=20).

    Output:

    """
    outdir = root + alnDir + 'plots/'

    if starlist=='all':
        tag = '_all'
        lbl = 'All'
    if starlist=='oldstars':
        tag = '_old'
        lbl = 'Old'
    if starlist=='yngstars':
        tag = '_yng'
        lbl = 'Young'

    # Load data
    s = loadPop(alnDir=alnDir,align=align,poly=poly,points=points,starlist=starlist)

    names = s.getArray('name')
    mag = s.getArray('mag')

    x0 = s.getArray('x0')
    y0 = s.getArray('y0')
    x0e = s.getArray('x0e')
    y0e = s.getArray('y0e')
    vx = s.getArray('vx')
    vy = s.getArray('vy')
    vxe = s.getArray('vxe')
    vye = s.getArray('vye')
    ax = s.getArray('ax')
    ay = s.getArray('ay')
    axe = s.getArray('axe')
    aye = s.getArray('aye')
    t0 = s.getArray('t0x')

    x0_pc = x0 * dist / au_in_pc
    y0_pc = y0 * dist / au_in_pc
    x0_km = x0_pc * km_in_pc
    y0_km = y0_pc * km_in_pc
    r2d = np.sqrt(x0**2 + y0**2) # arcsec
    r2d_pc = r2d * dist / au_in_pc
    rcgs = r2d * dist * cm_in_pc / au_in_pc

    # Put velocities in km/s and accelerations in km/s/yr
    vx_kms = vx * asy_to_kms
    vy_kms = vy * asy_to_kms
    vxe_kms = vxe * asy_to_kms
    vye_kms = vye * asy_to_kms
    ax *= asy_to_kms
    ay *= asy_to_kms
    axe *= asy_to_kms
    aye *= asy_to_kms

    # Determine # of data points per star
    pntcnt = np.zeros(len(names))
    for ii in range(len(names)):
        pntFileName = '%s%s%s%s.points' % (root, alnDir, points, names[ii])
        pntFile = open(pntFileName)
        data = pntFile.readlines()
        pntcnt[ii] = len(data)

    # Only plot acceleration upper limits for stars detected in Nepochs:
    idx = np.where(pntcnt >= nEpochs)[0]

    names = [names[ii] for ii in idx]
    x0 = x0[idx]
    y0 = y0[idx]
    x0_pc = x0_pc[idx]
    y0_pc = y0_pc[idx]
    x0e = x0e[idx]
    y0e = y0e[idx]
    vx = vx[idx]
    vy = vy[idx]
    vxe = vxe[idx]
    vye = vye[idx]
    vx_kms = vx_kms[idx]
    vy_kms = vy_kms[idx]
    vxe_kms = vxe_kms[idx]
    vye_kms = vye_kms[idx]
    ax = ax[idx]
    ay = ay[idx]
    axe = axe[idx]
    aye = aye[idx]
    r2d = r2d[idx]
    r2d_pc = r2d_pc[idx]
    rcgs = rcgs[idx]
    t0 = t0[idx]

    # Total acceleration
    atot = py.hypot(ax, ay)
    atoterr = np.sqrt((ax*axe)**2 + (ay*aye)**2) / atot

    # Construct an array of radii out to 4 arcsec in steps of 0.1''
    r = (arange(10 * 5) * 0.1) + 0.1
    r_au = r * dist
    r_pc = r_au / au_in_pc
    r_cm = r_au * cm_in_au

    # Determine the theoretical curve for a vs. r
    a_cm_s2 = -GM/ r_cm**2
    a_km_s_yr = a_cm_s2 * sec_in_yr / 1.0e5


    # What is the polyfit acceleration upper limit
    # along the radial component? Convert 2D cartesian to circular:
    (ar, at, are, ate) = util.xy2circErr(x0, y0, ax, ay,
                                         x0e, y0e, axe, aye)
    
    # What is the radial component limit (3 sigma)
    accLimRadial = ar - (3.0 * are) # km/s/yr
    alimcgs = accLimRadial * 1.0e5 / sec_in_yr
    arcgs = ar *1.0e5/sec_in_yr

    # also look at the 1 sigma value
    accLimRadial1 = ar - (1.0 * are) # km/s/yr
    alimcgs1 = accLimRadial1 * 1.0e5 / sec_in_yr

    # What is the z-lower limit as set by the accel. limits
    # for just those stars below the curve
    sig_id = []
    py.figure(1, figsize=(6,6))
    py.clf()
    _out = open(root + alnDir + 'tables/kinematics'+tag+'.txt','w')
    hdr = '#%10s  %7s  %12s  %12s  %12s  %7s  %7s  %15s   %9s\n'
    fmt = '%10s  %7.2f  %12.4e  %12.4e  %12.4e  %7.2f  %7.2f  %15.2f  %11.3f\n'
    _out.write(hdr % ('Name','t_ref','X (km)', 'Y (km)', 'Zmin (km)','Vx (km/s)',\
                      'Vy (km/s)','a_rad (km/s/yr)','Zmin (pc)'))

    print
    print 'Stars with (radial acc + 1 sigma) below curve:'
    print '%8s  %7s  %7s  %7s  %7s   %16s' % \
          ('Star','X (pc)', 'Y (pc)','R2D (pc)', 'Zmin (pc)', 'a_rad (km/s/yr)')
    print '-----------------------------------------------------------------'

    # Plot zmin vs. 2d radius
    # If stars fall below the curve in the above plot, then their 3D distance must
    # be greater than their projected distance.  Can therefore get z_min.
    min_cnt = 0
    for ii in range(len(rcgs)):
        # Upper limits to accelerations give lower limits to 3D radii
        # Figure out which stars are below the curve, use ar + 1 sigma to get sources (alimcgs)
        foo = (-GM * rcgs[ii] / alimcgs1[ii])**(2.0/3.0) # Gives (minimum r)-squared
        #foo = (-GM * rcgs[ii] / arcgs[ii])**(2.0/3.0) # Gives (minimum r)-squared
        if ((foo - rcgs[ii]**2) > 0): # Below the curve
            # Get the indices of these stars for later
            sig_id = concatenate([sig_id,[ii]])

            zmincgs = np.sqrt(foo - rcgs[ii]**2)
            r_p = rcgs / cm_in_pc
            zmin = zmincgs / (cm_in_pc) # pc
            zmin_km = zmincgs / 1.e5

            py.plot([r_p[ii]], [zmin], 'k.')
            py.xlabel('Projected Radius (pc)')
            py.ylabel('Minimum Z (pc)')

            print '%8s  %7.3f  %7.3f  %7.3f  %7.3f     %6.3f' % \
                  (names[ii], x0_pc[ii], y0_pc[ii], r_p[ii], zmin, ar[ii])

            # Write out the kinematic variables to a file
            _out.write(fmt % (names[ii], t0[ii],  x0_km[ii], y0_km[ii], zmin_km, \
                              vx_kms[ii], vy_kms[ii], ar[ii], zmin))

            min_cnt += 1
            
    _out.close()

    print
    print 'Number of stars below curve: %s' % min_cnt

    py.figure(1)
    py.savefig(outdir+'zmin_projR%s.png' % tag)
    py.close(1)

    # Also plot the significant radial acceleration detections,
    # with their 3 sigma error bars, for just those stars below the curve
    py.figure(2)
    py.figure(figsize=(6,6))
    py.clf()
    usetexTrue()
    dtct = []
    for ii in sig_id: # they're below the curve
        if (-ar[ii] - (3.*are[ii]) > 0.0): # and they're signif detections
            det = py.errorbar(r2d_pc[ii], -ar[ii], yerr=(3.*are[ii]), fmt='b.',ms=8)
            xt = r2d_pc[ii]
            yt = -ar[ii]
            nm = names[int(ii)]
            #text(xt,yt,nm,fontsize=9)
            dtct = np.concatenate([dtct,[ii]])
    #plot([0,0.1],[0,0],'k--')
    # Plot upper limits (from top of this program)
    for ii in range(len(r2d_pc)):
        if ii in dtct:
            continue
        else:
            uppr = py.plot([r2d_pc[ii]], [-accLimRadial[ii]], 'k_')
            py.arrow(r2d_pc[ii], -accLimRadial[ii], 0, -5, hold=True, color='black',\
                     width=0.0003, head_length=1, linewidth=0.001) #, \
            #          units='x',width=0.0005, color='black')
    py.plot(r_pc, -a_km_s_yr, '--k')
    lgd1 = 'Upper Limit'
    lgd2 = 'Detection'
    #lgd = py.legend((uppr[0], det[0]), (lgd1,lgd2), fancybox=True, numpoints=1)
    #lgdLines = lgd.get_lines()
    #lgdLines[0].set_marker('_')
    #lgdLines[1].set_marker('.')
    #lgdLines[0].set_mfc('k')
    #lgdLines[1].set_mfc('b')
    #lgdLines[0].set_ms(8)
    #lgdLines[1].set_ms(8)
    py.text(0.02, 80, r'{\bf $|$a$|_{max} = \frac{GM}{R^2}$}')
    py.xlabel(r'{\bf Projected Radius (pc)}')
    py.ylabel(r'{\bf $|$a$_R|$ (km/s/yr)}')
    py.title('Acceleration Detections \& Upper Limits')
    py.axis([0,0.1,-10,100])
    py.savefig(outdir+'r2d_accel_detectUpLim%s.png' % tag)
    py.savefig(outdir+'r2d_accel_detectUpLim%s.eps' % tag)
    py.close(2)

    # Find the significant accelerations above zero
    #print ''
    #for ii in sig_id:
    #    ii = int(ii)
    #    if ((ar[ii] + (3.*are[ii])) < 0.):
    #        print names[ii]

    #fig=figure(3)
    #clf()
    #ax = p3.Axes3D(fig)
    py.figure(4)
    py.clf()
    for ii in range(len(rcgs)):
        foo = (-GM * rcgs[ii] / alimcgs1[ii])**(2.0/3.0) # Gives (minimum r)-squared
        if ((foo - rcgs[ii]**2) > 0):
            if (-ar[ii] - (3.*are[ii]) > 0.0): # significant acceleration (above zero)
                zmincgs = np.sqrt(foo - rcgs[ii]**2)
                zmin = zmincgs / (cm_in_pc)

                #figure(3)
                #ax.scatter3D([x0_pc[ii]],[zmin],[y0_pc[ii]],facecolor='blue')
                #if tag == '_old':
                #    ax.plot3D([x0_pc[ii],x0_pc[ii]],[zmin,zmin],[y0_pc[ii],-0.06],\
                #              color='g',ls='--',lw='1')
                #else:
                #    ax.plot3D([x0_pc[ii],x0_pc[ii]],[zmin,zmin],[y0_pc[ii],-0.08],\
                #              color='g',ls='--',lw='1')
                rmin = np.sqrt(x0_pc[ii]**2+y0_pc[ii]**2+zmin**2)

                figure(4)
                plotVvsR(names[ii],rmin,vx_kms[ii],vy_kms[ii],vxe_kms[ii],vye_kms[ii])
                

    #figure(3)
    #ax.scatter3D([0.0],[0.0],[0.0],facecolor='orange')
    #if tag == '_old':
    #    ax.plot3D([0.0,0.0],[0.0,0.0],[0.0,-0.06],color='r',ls='--',lw='1')
    #else:
    #    ax.plot3D([0.0,0.0],[0.0,0.0],[0.0,-0.08],color='r',ls='--',lw='1')
    #ax.set_xlabel('X')
    #ax.set_ylabel('Z min')
    #ax.set_zlabel('Y')
    #savefig(outdir+'3d_xyzmin%s.png' % tag)

    py.figure(4)
    py.xlabel('3D Radius (pc)')
    py.ylabel('Proper Motion (km/s)')
    py.title('Proper Motion vs. Radius')
    py.savefig(outdir+'VvsR%s.png' % tag)

    usetexFalse()

def plotVvsR(name,rmin,vx,vy,vxe,vye):

    # Construct an array of radii out to 4 arcsec in steps of 0.1''
    r = (arange(10 * 5) * 0.1) + 0.1
    r_au = r * dist
    r_pc = r_au / au_in_pc
    r_cm = r_au * cm_in_au
    r_km = r_cm / 1.e5

    # Compute total proper motion and error for this star
    v_kms = np.sqrt(vx**2+vy**2)
    ve_kms = np.sqrt(vx**2+vy**2)

    # NOTE: still need to account for the error in the 3D radius
    #       and error in the escape velocity curve

    # Find fast objects
    v_esc_cms = np.sqrt((2. * G * mass * msun) / r_cm) # for curve on plot
    v_esc_kms = v_esc_cms / 1.e5

    # Plot velocities vs. r, w/ error bars
    plot([rmin],[v_kms],'b.')
    errorbar([rmin],[v_kms],yerr=ve_kms,fmt='b.')
    plot(r_pc, v_esc_kms,'r-')
    if name == 'S1-13':
        vtot_kms = np.sqrt(vx**2+vy**2+745.**2)
        plot([rmin],[vtot_kms],'r.')
        text(rmin + 0.002,vtot_kms,name,fontsize=9)


def plotLimits(alnDir='11_08_29/',
               align='align/align_d_rms_1000_abs_t',
               poly='polyfit_c/fit', points='points_c/',
               youngOnly=True, sigma=3):
    """
    Find the 3 sigma radial acceleration limits for each star.
    """
    # Load GC constants
    #cc = objects.Constants()

    # Load up positional information from align. 
    s = starset.StarSet(root + alnDir + align)
    s.loadPolyfit(root + alnDir + poly, arcsec=1)
    s.loadPolyfit(root + alnDir + poly, arcsec=1, accel=1)

    names = s.getArray('name')

    # In arcsec
    x = s.getArray('x')
    y = s.getArray('y')
    # In arcsec/yr
    vx = s.getArray('fitXa.v')
    vy = s.getArray('fitYa.v')
    # In mas/yr^2
    ax = s.getArray('fitXa.a') * 1000.0
    ay = s.getArray('fitYa.a') * 1000.0
    axe = s.getArray('fitXa.aerr') * 1000.0
    aye = s.getArray('fitYa.aerr') * 1000.0
    r2d = s.getArray('r2d')
    cnt = s.getArray('velCnt')

    tag = '_all'
    tag2 = 'All'

    if (youngOnly == True):
        yngNames = young.youngStarNames()

        idx = []

        for ii in range(len(names)):
            #if (r2d[ii] > 0.8 and
            if (names[ii] in yngNames and cnt[ii] >= 20):
                idx.append(ii)

        names = [names[i] for i in idx]
        x = x[idx]
        y = y[idx]
        vx = vx[idx]
        vy = vy[idx]
        ax = ax[idx]
        ay = ay[idx]
        axe = axe[idx]
        aye = aye[idx]
        r2d = r2d[idx]
        print 'Found %d young stars' % len(names)
        tag = '_yng'
        tag2 = 'Young'

    # Lets do radial/tangential
    r = np.sqrt(x**2 + y**2) # arcsec
    if ('Radial' in poly):
        at = ax
        ar = ay
        ate = axe
        are = aye
    else:
        # In mas/yr^2
        ar = ((ax*x) + (ay*y)) / r 
        at = ((ax*y) - (ay*x)) / r
        are = np.sqrt((axe*x)**2 + (aye*y)**2) / r
        ate = np.sqrt((axe*y)**2 + (aye*x)**2) / r

    # Total acceleration
    atot = py.hypot(ax, ay)
    atoterr = np.sqrt((ax*axe)**2 + (ay*aye)**2) / atot
    # Acceleration upper limit for each star
    accLim = ar - (sigma * are)

    # Calculate the acceleration limit set by the projected radius
    # Convert into cm
    r2d = r * dist * cm_in_au
    # acc1 in cm/s^2
    a2d = -G * mass * msun / r2d**2
    # acc1 in km/s/yr
    a2d *= sec_in_yr / 1.0e5
    # acc1 in mas/yr^2
    a2d *= 1000.0 / asy_to_kms
    accLimR2d = a2d

    _f = open(root + alnDir + 'tables/accel_limits' + tag + '.txt', 'w')

    _f.write('###   Radial Acceleration Limits for %s Stars   ###\n' % tag2)
    _f.write('%13s  %7s - (%d * %5s) =  %9s     %21s  Constrained?  Non-zero?\n' % \
             ('Name', 'a_rad', sigma, 'aerr', 'a_obs_lim', 'a_proj_lim'))
    
    fmt = '%13s  %7.3f - (%d * %5.3f) =    %7.3f  '
    fmt += 'vs %10.3f (mas/yr^2)  %8s  %8s'
    fmt2 = fmt + '\n'
    fmt3 = '%13s  %7.3f - (%d * %5.3f) =    %7.3f  vs %10.3f (mas/yr^2)  %5.3f\n'

    hiPri = []
    for i in range(len(ar)):
        constrained = ''
        ltzero = ''
        if (accLim[i] > accLimR2d[i]): # Below the curve (remember, negative values)
            constrained = '**'
            
            # Find those with significant accelerations below zero
            if ((ar[i] < 0.) & ((ar[i] + (sigma*are[i])) < 0.)):
                ltzero = '**'
                # Save these indices as high priority
                hiPri = np.concatenate([hiPri,[i]])

        # Print info for all young stars
        print fmt % (names[i], ar[i], sigma, are[i], accLim[i],
                     accLimR2d[i], constrained, ltzero)
        _f.write(fmt2 % (names[i], ar[i], sigma, are[i], accLim[i],
                         accLimR2d[i], constrained, ltzero))

    _f.close()

    # Make a high-priority list for stars w/ acc. detections
    _hipri = open(root + alnDir + 'tables/accel_limits' + tag + '_hiPriority.txt', 'w')
    _hipri.write('###   Radial Acceleration Limits for %s Stars   ###\n' % tag2)
    _hipri.write('%13s  %7s - (%d * %5s) =  %9s     %21s   %5s\n' % \
             ('Name', 'a_rad', sigma, 'aerr', 'a_obs_lim', 'a_proj_lim', 'r2d (")'))
    hiPri = [int(ii) for ii in hiPri]
    for h in hiPri:
        _hipri.write(fmt3 % (names[h], ar[h], sigma, are[h], accLim[h],
                         accLimR2d[h], r[h]))
    _hipri.close()

    # Stuff below confirms that these accelerations match those in accelLimit()
    # Convert accelerations to km/s/yr
    #ar = ar / 1000.0 * asy_to_kms 
    #are = are / 1000.0 * asy_to_kms 
    #accLim = ar - (sigma * are)
    #accLimR2d = accLimR2d / 1000.0 *asy_to_kms
    #clf()
    #plot(r2d/cm_in_pc,-accLim,'k.')
    #foo = argsort(r)
    #plot(r[foo]*0.04,-accLimR2d[foo],'--k')
    #axis('normal')
    #axis([0,0.18,0,40])
    #savefig(root + alnDir + 'plots/r2d_accel_limit_yng2.png')



def compareVelocity(alnDir='11_10_26/',
                    align = 'align/align_d_rms_1000_abs_t',
                    poly='polyfit_c/fit', points='points_c/'):
    """
    Compare velocities from the linear and acceleration fits.

    alnDir -- An align root directory such as '10_04_22/' 
    align -- Align root name 
    poly -- Polyfit root name 
    points -- Points directory
    """
    
    s = starset.StarSet(root + alnDir + align)
    s.loadPolyfit(root + alnDir + poly, accel=0, arcsec=1)
    s.loadPolyfit(root + alnDir + poly, accel=1, arcsec=1)

    names = s.getArray('name')
    x = s.getArray('x')
    y = s.getArray('y')
    r = hypot(x, y)

    vx_vel = s.getArray('fitXv.v')
    vy_vel = s.getArray('fitYv.v')
    vxe_vel = s.getArray('fitXv.verr')
    vye_vel = s.getArray('fitYv.verr')

    vx_acc = s.getArray('fitXa.v')
    vy_acc = s.getArray('fitYa.v')
    vxe_acc = s.getArray('fitXa.verr')
    vye_acc = s.getArray('fitYa.verr')
    
    # Calculate the residuals
    diffx = vx_vel - vx_acc
    diffxErr = np.sqrt(vxe_vel**2 + vxe_acc**2)
    
    diffy = vy_vel - vy_acc
    diffyErr = np.sqrt(vye_vel**2 + vye_acc**2)
    
    diff = py.hypot(diffx, diffy)
    diffErr = np.sqrt((diffx*diffxErr)**2 + (diffy*diffyErr)**2) / diff

    yngNames = young.youngStarNames()
    
    idx = (np.where((r > 0.8) & ((diff/diffErr) > 3.0)))[0]

    print '** Stars with large velocity discrepencies (N=%d): **' % len(idx)
    print '%15s  %5s  %5s  %5s  %3s' % ('Name', 'Sigma', 'X', 'Y', 'YNG?')
    for i in idx:
        if names[i] in yngNames:
            yngString = 'yng'
        else:
            yngString = ''
            
        print '%15s  %5.1f  %5.2f  %5.2f  %3s' % \
              (names[i], diff[i]/diffErr[i], x[i], y[i], yngString)

    # get the velocities for just the young stars
    vVx_yng = []
    vVy_yng = []
    aVx_yng = []
    aVy_yng = []
    vVxe_yng = []
    vVye_yng = []
    aVxe_yng = []
    aVye_yng = []
    for yy in range(len(yngNames)):
        #sidx = np.where(names == yngNames[yy])[0]
        if yngNames[yy] not in names:
            continue
        sidx = names.index(yngNames[yy])
        vVx_yng = np.concatenate([vVx_yng, [vx_vel[sidx]*1.e3]])
        vVy_yng = np.concatenate([vVy_yng, [vy_vel[sidx]*1.e3]])
        aVx_yng = np.concatenate([aVx_yng, [vx_acc[sidx]*1.e3]])
        aVy_yng = np.concatenate([aVy_yng, [vy_acc[sidx]*1.e3]])
        vVxe_yng = np.concatenate([vVxe_yng, [vxe_vel[sidx]*1.e3]])
        vVye_yng = np.concatenate([vVye_yng, [vye_vel[sidx]*1.e3]])
        aVxe_yng = np.concatenate([aVxe_yng, [vxe_acc[sidx]*1.e3]])
        aVye_yng = np.concatenate([aVye_yng, [vye_acc[sidx]*1.e3]])
        print 'Young star %6s: linear fit vx,vy = %6.2f, %6.2f mas/yr' % \
              (yngNames[yy], vx_vel[sidx]*1.e3, vy_vel[sidx]*1.e3)
        print '                   accel fit vx,vy = %6.2f, %6.2f mas/yr' % \
              (vx_acc[sidx]*1.e3, vy_acc[sidx]*1.e3)

    # Plot just young stars; X and Y seperately
    py.clf()
    py.subplot(2, 1, 1)
    py.errorbar(vVx_yng, aVx_yng, xerr=vVxe_yng, yerr=aVxe_yng, fmt='k.')
    rng = py.axis()
    py.plot(rng[0:2], rng[0:2], 'b--')
    py.ylabel('Accel Fit (mas/yr)')

    py.subplot(2, 1, 2)
    py.errorbar(vVy_yng, aVy_yng, xerr=vVye_yng, yerr=aVye_yng, fmt='k.')
    rng = py.axis()
    py.plot(rng[0:2], rng[0:2], 'b--')
    py.xlabel('Linear Fit (mas/yr)')
    py.ylabel('Accel Fit (mas/yr)')
    py.savefig(root + alnDir + 'plots/vel_linear_vs_accel_fit_yngstars.png')
    
    

    # Plot all stars; X and Y seperately
    py.clf()
    py.subplot(2, 1, 1)
    py.errorbar(vx_vel*1.e3, vx_acc*1.e3, xerr=vxe_vel*1.e3,
                yerr=vxe_acc*1.e3, fmt='k.')
    rng = py.axis()
    py.plot(rng[0:2], rng[0:2], 'b--')
    py.ylabel('Accel Fit (mas/yr)')

    py.subplot(2, 1, 2)
    py.errorbar(vy_vel*1.e3, vy_acc*1.e3, xerr=vye_vel*1.e3,
                yerr=vye_acc*1.e3, fmt='k.')
    rng = py.axis()
    py.plot(rng[0:2], rng[0:2], 'b--')
    py.xlabel('Linear Fit (mas/yr)')
    py.ylabel('Accel Fit (mas/yr)')
    py.savefig(root + alnDir + 'plots/vel_linear_vs_accel_fit_allstars.png')
    
    
def highSigSrcs(radiusCut, sigmaCut, alnDir='11_08_29/',
                poly='polyfit_c/fit',
                save=False, verbose=True):
    
    """
    Make a list of all sources with significant accelerations.
    Assumes a plate scale of 9.95 mas/pixel.

    Inputs:
    radiusCut - the largest radius to include (arcsec)
    sigmaCut - the lowest a/a_err to include
    save - Save to tables/accelHighSigSrcs.txt (def=False)
    verbose - Print to screen (def=True)

    Outputs:
    srcNames - returns a list of stars with significant radial
    accelerations based on the passed in criteria.
    """
    # Load up the accelPolar file
    fitFile = root + alnDir + poly + '.accelPolar'
    scale = 0.00995  # arcsec/pixel

    tab = asciidata.open(fitFile)

    name = tab[0]._data
    radius = tab[2].tonumarray() * scale
    acc = tab[10].tonumarray() * scale
    accErr = tab[12].tonumarray() * scale
    sigma = acc / accErr

    # Make cuts in radius and sigma
    idx = ( where((radius < radiusCut) & (sigma > sigmaCut)) )[0]

    # Sort
    rdx = sigma[idx].argsort()
    idx = idx[rdx[::-1]]

    # Print out to the screen
    if (verbose == True):
        print '** Found %d significantly accelerating sources **' % len(idx)
        print ''
        print '%-15s  %8s  %10s  %8s' % \
              ('Name', 'Radius', 'Accel', 'Signif.')
        print '%-15s  %8s  %10s  %8s' % \
              ('', '(arcsec)', '(mas/yr^2)', '(sigma)')
        print '%-15s  %8s  %10s  %8s' % \
              ('---------------', '--------', '----------', '--------')

        for ii in idx:
            print '%-15s  %8.3f  %10.5f  %8.1f' % \
                  (name[ii], radius[ii], acc[ii], sigma[ii])


    # Save data to an output file
    if (save == True):
        outFile = 'tables/accelHighSigSrcs.txt'
        _out = open(outFile, 'w')

        _out.write('# Python: syAccel.highSigSrcs')
        _out.write('(%5.2f, %5.2f)\n' % \
                   (radiusCut, sigmaCut))
        _out.write('#\n')
        _out.write('%-15s  %8s  %10s  %8s\n' % \
                   ('# Name', 'Radius', 'Accel', 'Signif.'))
        _out.write('%-15s  %8s  %10s  %8s\n' % \
                   ('#', '(arcsec)', '(mas/yr^2)', '(sigma)'))
        _out.write('%-15s  %8s  %10s  %8s\n' % \
                   ('#--------------', '--------', '----------', '--------'))

        for ii in idx:
            _out.write('%-15s  %8.3f  %10.5f  %8.1f\n' % \
                       (name[ii], radius[ii], acc[ii], sigma[ii]))

        _out.close()

    # Return the list of significantly accelerating sources.
    return [name[ii] for ii in idx]
    
    
def velVsAcc():
    """
    Plot v/v_circular vs. a/a_bound.
    NOT COMPLETE?????
    """
    # Load up the accelPolar file
    fitFile = root + alnDir + poly + '.accelPolar'
    scale = 0.00995  # arcsec/pixel

    tab = asciidata.open(fitFile)

    name = tab[0]._data
    radius = tab[2].tonumarray() * scale

    velPhi = tab[5].tonumarray() * scale
    velRad = tab[6].tonumarray() * scale
    velPhiErr = tab[7].tonumarray() * scale
    velRadErr = tab[8].tonumarray() * scale
    acc = tab[10].tonumarray() * scale
    accErr = tab[12].tonumarray() * scale

    # Need to get the line-of-sight velocity from Paumard et al.
    

    vel = np.sqrt(velPhi**2 + velRad**2)
    velErr = np.sqrt((velPhi*velPhiErr)**2 + (velRad*velRadErr)**2) / vel

    # Determine the circular velocity 



def plotStar(starName, alnDir='11_08_29/',
             align='align/align_d_rms_1000_abs_t',
             poly='polyfit_c/fit', points='points_c/', suffix='', radial=False):

    outdir = root + alnDir + 'plots/'

    s = starset.StarSet(root + alnDir + align)
    s.loadPolyfit(root + alnDir + poly, accel=0, arcsec=0)
    s.loadPolyfit(root + alnDir + poly, accel=1, arcsec=0)

    names = s.getArray('name')
    
    ii = names.index(starName)
    star = s.stars[ii]

    pointsTab = asciidata.open(root + alnDir + points + starName + '.points')

    time = pointsTab[0].tonumpy()
    x = pointsTab[1].tonumpy()
    y = pointsTab[2].tonumpy()
    xerr = pointsTab[3].tonumpy()
    yerr = pointsTab[4].tonumpy()

    fitx = star.fitXv
    fity = star.fitYv
    dt = time - fitx.t0
    fitLineX = fitx.p + (fitx.v * dt)
    fitSigX = np.sqrt( fitx.perr**2 + (dt * fitx.verr)**2 )

    fitLineY = fity.p + (fity.v * dt)
    fitSigY = np.sqrt( fity.perr**2 + (dt * fity.verr)**2 )

    if (radial == True):
        # Lets also do radial/tangential
        x0 = fitx.p
        y0 = fity.p
        vx = fitx.v
        vy = fity.v
        x0e = fitx.perr
        y0e = fity.perr
        vxe = fitx.verr
        vye = fity.verr
        
        r0 = np.sqrt(x0**2 + y0**2)

        vr = ((vx*x0) + (vy*y0)) / r0
        vt = ((vx*y0) - (vy*x0)) / r0
        vre =  (vxe*x0/r0)**2 + (vye*y0/r0)**2
        vre += (y0*x0e*vt/r0**2)**2 + (x0*y0e*vt/r0**2)**2
        vre =  np.sqrt(vre)
        vte =  (vxe*y0/r0)**2 + (vye*x0/r0)**2
        vte += (y0*x0e*vr/r0**2)**2 + (x0*y0e*vr/r0**2)**2
        vte =  np.sqrt(vte)

        r = ((x*x0) + (y*y0)) / r0
        t = ((x*y0) - (y*x0)) / r0
        rerr = (xerr*x0/r0)**2 + (yerr*y0/r0)**2
        rerr += (y0*x0e*t/r0**2)**2 + (x0*y0e*t/r0**2)**2
        rerr =  np.sqrt(rerr)
        terr =  (xerr*y0/r0)**2 + (yerr*x0/r0)**2
        terr += (y0*x0e*r/r0**2)**2 + (x0*y0e*r/r0**2)**2
        terr =  np.sqrt(terr)

        fitLineR = ((fitLineX*x0) + (fitLineY*y0)) / r0
        fitLineT = ((fitLineX*y0) - (fitLineY*x0)) / r0
        fitSigR = ((fitSigX*x0) + (fitSigY*y0)) / r0
        fitSigT = ((fitSigX*y0) - (fitSigY*x0)) / r0

        diffR = r - fitLineR
        diffT = t - fitLineT
        sigR = diffR / rerr
        sigT = diffT / terr

        idxR = np.where(abs(sigR) > 4)
        idxT = np.where(abs(sigT) > 4)
        

    diffX = x - fitLineX
    diffY = y - fitLineY
    #diff = np.hypot(diffX, diffY)
    diff = (diffX + diffY) / 2.0
    rerr = np.sqrt((diffX*xerr)**2 + (diffY*yerr)**2) / diff
    sigX = diffX / xerr
    sigY = diffY / yerr
    sig = diff / rerr

    
    # Determine if there are points that are more than 5 sigma off
    idxX = np.where(abs(sigX) > 4)
    idxY = np.where(abs(sigY) > 4)
    idx = np.where(abs(sig) > 4)

    print 'Star:        ', starName
    print '\tX Chi^2 = %5.2f (%6.2f for %2d dof)' % \
          (fitx.chi2red, fitx.chi2, fitx.chi2/fitx.chi2red)
    print '\tY Chi^2 = %5.2f (%6.2f for %2d dof)' % \
          (fity.chi2red, fity.chi2, fity.chi2/fity.chi2red)
    print 'X  Outliers: ', time[idxX]
    print 'Y  Outliers: ', time[idxY]
    if (radial):
        print 'R  Outliers: ', time[idxX]
        print 'T  Outliers: ', time[idxY]
    print 'XY Outliers: ', time[idx]
    print '*****************'
    print


    close(2)
    figure(2, figsize=(7, 8))
    clf()

    dateTicLoc = MultipleLocator(3)
    #dateTicRng = [2006, 2010]
    dateTicRng = [1995, 2010]

    maxErr = np.array([xerr, yerr]).max()
    resTicRng = [-3*maxErr, 3*maxErr]

    from matplotlib.ticker import FormatStrFormatter
    fmtX = FormatStrFormatter('%5i')
    fmtY = FormatStrFormatter('%6.2f')

    paxes = subplot(3, 2, 1)
    plot(time, fitLineX, 'b-')
    plot(time, fitLineX + fitSigX, 'b--')
    plot(time, fitLineX - fitSigX, 'b--')
    errorbar(time, x, yerr=xerr, fmt='k.')
    rng = axis()
    axis(dateTicRng + [rng[2], rng[3]])
    xlabel('Date (yrs)')
    ylabel('X (pix)')
    #paxes.get_xaxis().set_major_locator(dateTicLoc)
    paxes.xaxis.set_major_formatter(fmtX)
    paxes.get_xaxis().set_major_locator(dateTicLoc)
    paxes.yaxis.set_major_formatter(fmtY)
    
    paxes = subplot(3, 2, 2)
    plot(time, fitLineY, 'b-')
    plot(time, fitLineY + fitSigY, 'b--')
    plot(time, fitLineY - fitSigY, 'b--')
    errorbar(time, y, yerr=yerr, fmt='k.')
    rng = axis()
    axis(dateTicRng + [rng[2], rng[3]])
    xlabel('Date (yrs)')
    ylabel('Y (pix)')
    #paxes.get_xaxis().set_major_locator(dateTicLoc)
    paxes.xaxis.set_major_formatter(fmtX)
    paxes.get_xaxis().set_major_locator(dateTicLoc)
    paxes.yaxis.set_major_formatter(fmtY)
    
    paxes = subplot(3, 2, 3)
    plot(time, np.zeros(len(time)), 'b-')
    plot(time, fitSigX, 'b--')
    plot(time, -fitSigX, 'b--')
    errorbar(time, x - fitLineX, yerr=xerr, fmt='k.')
    axis(dateTicRng + resTicRng)
    xlabel('Date (yrs)')
    ylabel('X Residuals (pix)')
    paxes.get_xaxis().set_major_locator(dateTicLoc)

    paxes = subplot(3, 2, 4)
    plot(time, np.zeros(len(time)), 'b-')
    plot(time, fitSigY, 'b--')
    plot(time, -fitSigY, 'b--')
    errorbar(time, y - fitLineY, yerr=yerr, fmt='k.')
    axis(dateTicRng + resTicRng)
    xlabel('Date (yrs)')
    ylabel('Y Residuals (pix)')
    paxes.get_xaxis().set_major_locator(dateTicLoc)

    bins = np.arange(-7, 7, 1)
    subplot(3, 2, 5)
    (n, b, p) = hist(sigX, bins)
    setp(p, 'facecolor', 'k')
    axis([-5, 5, 0, 20])
    xlabel('X Residuals (sigma)')
    ylabel('Number of Epochs')

    subplot(3, 2, 6)
    (n, b, p) = hist(sigY, bins)
    axis([-5, 5, 0, 20])
    setp(p, 'facecolor', 'k')
    xlabel('Y Residuals (sigma)')
    ylabel('Number of Epochs')

    subplots_adjust(wspace=0.4, hspace=0.4, right=0.95, top=0.95)
    #savefig(outdir+'plotStar_' + starName + '.eps')
    savefig(outdir+'plotStar_' + starName + suffix + '.png')
    #show()

    ##########
    #
    # Also plot radial/tangential
    #
    ##########
    if (radial == True):
        clf()

        dateTicLoc = MultipleLocator(3)
        dateTicRng = [1995, 2008]
        
        maxErr = np.array([rerr, terr]).max()
        resTicRng = [-3*maxErr, 3*maxErr]
        
        from matplotlib.ticker import FormatStrFormatter
        fmtX = FormatStrFormatter('%5i')
        fmtY = FormatStrFormatter('%6.2f')
        
        paxes = subplot(3, 2, 1)
        plot(time, fitLineR, 'b-')
        plot(time, fitLineR + fitSigR, 'b--')
        plot(time, fitLineR - fitSigR, 'b--')
        errorbar(time, r, yerr=rerr, fmt='k.')
        rng = axis()
        axis(dateTicRng + [rng[2], rng[3]])
        xlabel('Date (yrs)')
        ylabel('R (pix)')
        paxes.xaxis.set_major_formatter(fmtX)
        paxes.get_xaxis().set_major_locator(dateTicLoc)
        paxes.yaxis.set_major_formatter(fmtY)
        
        paxes = subplot(3, 2, 2)
        plot(time, fitLineT, 'b-')
        plot(time, fitLineT + fitSigT, 'b--')
        plot(time, fitLineT - fitSigT, 'b--')
        errorbar(time, t, yerr=terr, fmt='k.')
        rng = axis()
        axis(dateTicRng + [rng[2], rng[3]])
        xlabel('Date (yrs)')
        ylabel('T (pix)')
        paxes.xaxis.set_major_formatter(fmtX)
        paxes.get_xaxis().set_major_locator(dateTicLoc)
        paxes.yaxis.set_major_formatter(fmtY)
        
        paxes = subplot(3, 2, 3)
        plot(time, np.zeros(len(time)), 'b-')
        plot(time, fitSigR, 'b--')
        plot(time, -fitSigR, 'b--')
        errorbar(time, r - fitLineR, yerr=rerr, fmt='k.')
        axis(dateTicRng + resTicRng)
        xlabel('Date (yrs)')
        ylabel('R Residuals (pix)')
        paxes.get_xaxis().set_major_locator(dateTicLoc)
        
        paxes = subplot(3, 2, 4)
        plot(time, np.zeros(len(time)), 'b-')
        plot(time, fitSigT, 'b--')
        plot(time, -fitSigT, 'b--')
        errorbar(time, t - fitLineT, yerr=terr, fmt='k.')
        axis(dateTicRng + resTicRng)
        xlabel('Date (yrs)')
        ylabel('T Residuals (pix)')
        paxes.get_xaxis().set_major_locator(dateTicLoc)
        
        bins = np.arange(-7, 7, 1)
        subplot(3, 2, 5)
        (n, b, p) = hist(sigR, bins)
        setp(p, 'facecolor', 'k')
        axis([-5, 5, 0, 20])
        xlabel('T Residuals (sigma)')
        ylabel('Number of Epochs')
        
        subplot(3, 2, 6)
        (n, b, p) = hist(sigT, bins)
        axis([-5, 5, 0, 20])
        setp(p, 'facecolor', 'k')
        xlabel('Y Residuals (sigma)')
        ylabel('Number of Epochs')
        
        subplots_adjust(wspace=0.4, hspace=0.4, right=0.95, top=0.95)
        #savefig(outdir+'plotStarRadial_' + starName + '.eps')
        savefig(outdir+'plotStarRadial_' + starName + suffix + '.png')


def getResiduals(alnDir='11_08_29/',
                align='align/align_d_rms_1000_abs_t',
                poly='polyfit_c/fit', points='points_c/',
                additiveErr=True, trimOutliers=False, trimSigma=4,
                useAccFits=False, radCut=4.0,suffix=''):
    """Analyze the distribution of points relative to their best
    fit velocities. Optionally trim the largest outliers in each
    stars *.points file.  Optionally make radius cut with radCut flag.
    Set additiveErr=True to remove the additive error from positional
    errors in points files, and to plot this as a separate line.
    """

    outdir = root + alnDir + 'plots/'

    s = starset.StarSet(root + alnDir + align)
    s.loadPolyfit(root + alnDir + poly, accel=0, arcsec=0)
    s.loadPolyfit(root + alnDir + poly, accel=1, arcsec=0)
    s.loadPoints(root + alnDir + points)
    names = s.getArray('name')
    x = s.getArray('x')
    y = s.getArray('y')

    addErr = 0
    if additiveErr == True:
        # Define the additive error:
        addErr = 0.1 # mas, AO data only

    # Make some empty arrays to hold all our results.
    sigmaX = np.arange(0, dtype=float)
    sigmaY = np.arange(0, dtype=float)
    sigma  = np.arange(0, dtype=float)
    diff_all = np.arange(0, dtype=float)
    diffX_all = np.arange(0, dtype=float)
    diffY_all = np.arange(0, dtype=float)
    xerr_all = np.arange(0, dtype=float)
    yerr_all = np.arange(0, dtype=float)
    mag_all = np.arange(0, dtype=float)
    xeom_all = np.arange(0, dtype=float)
    yeom_all = np.arange(0, dtype=float)
    xaln_all = np.arange(0, dtype=float)
    yaln_all = np.arange(0, dtype=float)
    starName_all = np.arange(0, dtype=float)

    nCnt = 0
    # Loop through all the stars and combine their residuals and their errors.
    for star in s.stars:
        starName = star.name
        cnt = star.pointsCnt        
        if cnt < 5:
            continue
        nCnt += 1

        ############
        # Points files - include err on mean, align err, and additive err
        #   -- only need these total errs when looking at residuals in units of 'sigma'
        ############

        t = np.array(star.years)
        errx_a_allEp = star.getArrayAllEpochs('xerr_a')*1e3 # mas
        erry_a_allEp = star.getArrayAllEpochs('yerr_a')*1e3
        errx_p_allEpTmp = star.getArrayAllEpochs('xerr_p')*1e3 # mas, includes additive
        erry_p_allEpTmp = star.getArrayAllEpochs('yerr_p')*1e3
        x_pts = star.getArrayAllEpochs('pnt_x')
        y_pts = star.getArrayAllEpochs('pnt_y')
        xe_pts = star.getArrayAllEpochs('pnt_xe') # include eom, align, additive err
        ye_pts = star.getArrayAllEpochs('pnt_ye')
        mag = star.getArrayAllEpochs('phot_mag')
        mag_e = star.getArrayAllEpochs('phot_mage')
        r2d = np.hypot(x_pts,y_pts)

        # Remove non-detections
        kp = np.where(x_pts > -999)[0]
        errx_a_allEp = errx_a_allEp[kp]
        erry_a_allEp = erry_a_allEp[kp]
        errx_p_allEpTmp = errx_p_allEpTmp[kp]
        erry_p_allEpTmp = erry_p_allEpTmp[kp]
        x_pts = x_pts[kp]
        y_pts = y_pts[kp]
        xe_pts = xe_pts[kp]
        ye_pts = ye_pts[kp]
        mag = mag[kp]
        mag_e = mag_e[kp]
        r2d = r2d[kp]
        t = t[kp]
        
        if ((radCut != None) and (r2d.mean() > radCut)):
            continue

        # Remove additive err
        if additiveErr == True:
            errx_p_allEp = np.sqrt(errx_p_allEpTmp**2 - addErr**2)  
            erry_p_allEp = np.sqrt(erry_p_allEpTmp**2 - addErr**2)  
        else:
            errx_p_allEp = errx_p_allEpTmp
            erry_p_allEp = erry_p_allEpTmp
            
        
        ############
        # Residuals - Best fit velocity model
        ############
        if (useAccFits == True):
            fitx = star.fitXa
            fity = star.fitYa
        else:
            fitx = star.fitXv
            fity = star.fitYv

        if np.isnan(fitx.t0) == True:
            pdb.set_trace()
        #    continue

        dt = t - fitx.t0
        fitLineX = fitx.p + (fitx.v * dt)
        fitSigX = np.sqrt( fitx.perr**2 + (dt * fitx.verr)**2 )

        fitLineY = fity.p + (fity.v * dt)
        fitSigY = np.sqrt( fity.perr**2 + (dt * fity.verr)**2 )

        if (useAccFits == True):
            fitLineX += (fitx.a * dt**2) / 2.0
            fitLineY += (fity.a * dt**2) / 2.0
            fitSigX = np.sqrt(fitSigX**2 + (dt**2 * fitx.aerr / 2.0)**2)
            fitSigY = np.sqrt(fitSigY**2 + (dt**2 * fity.aerr / 2.0)**2)

        # Residuals - 'sigma' includes err on mean, align err, and additive err
        diffX = np.abs(x_pts - fitLineX)
        diffY = np.abs(y_pts - fitLineY)
        #diff = np.hypot(diffX, diffY)
        diff = (diffX + diffY) / 2.0
        re_pts = np.sqrt((diffX*xe_pts)**2 + (diffY*ye_pts)**2) / diff
        sigX = diffX / xe_pts
        sigY = diffY / ye_pts
        sig = diff / re_pts
        #print nCnt, starName,mag.mean(), diff.mean()

        # Combine this stars information with all other stars.
        sigmaX = np.concatenate((sigmaX, sigX))
        sigmaY = np.concatenate((sigmaY, sigY))
        sigma = np.concatenate((sigma, sig))
        diffX_all = np.concatenate((diffX_all,diffX))
        diffY_all = np.concatenate((diffY_all,diffY))
        diff_all = np.concatenate((diff_all,diff))
        mag_all = np.concatenate((mag_all,mag))
        # error on mean from 3 submaps, does not include additive error:
        xeom_all = np.concatenate((xeom_all,errx_p_allEp))
        yeom_all = np.concatenate((yeom_all,erry_p_allEp))
        # align errors:
        xaln_all = np.concatenate((xaln_all,errx_a_allEp))
        yaln_all = np.concatenate((yaln_all,erry_a_allEp))

        # Concatenate the stars' names so we can group by star if needed
        for nn in range(len(diff)):
            starName_all = np.concatenate([starName_all,[starName]])
        

    # Get average of all the errors
    eom_all = (xeom_all + yeom_all) / 2.0
    aln_all = (xaln_all + yaln_all) / 2.0

    # Residuals should have a gaussian probability distribution
    # with a mean of 0 and a sigma of 1. Overplot this to be sure.
    ggx = np.arange(-7, 7, 0.25)
    ggy = normpdf(ggx, 0, 1)

    # Get average residual and average error per magnitude bin
    magStep = 1.0
    magBins = np.arange(10.0, 20.0, magStep)
    residMag = np.zeros(len(magBins), float)
    errMag = np.zeros(len(magBins), float)
    alnMag = np.zeros(len(magBins), float)
    print ''
    print '%4s  %8s  %9s  %9s  %11s' % \
          ('Mag', 'Residual', 'RMS Error', 'Aln Error','Total Error')

    for mm in range(len(magBins)):
        mMin = magBins[mm] - (magStep / 2.0)
        mMax = magBins[mm] + (magStep / 2.0)
        idx = (np.where((mag_all >= mMin) & (mag_all < mMax)))[0]

        # Use the average X and Y residuals
        if (len(idx) > 0):
            residMag[mm] = np.abs(np.median(diff_all[idx])*1e3)
            errMag[mm] = np.median(eom_all[idx])
            alnMag[mm] = np.median(aln_all[idx])
        if additiveErr==True:
            print '%4.1f  %8.3f  %9.3f  %9.3f  %11.3f ' % \
                  (magBins[mm], residMag[mm], errMag[mm],alnMag[mm],
                   np.sqrt(errMag[mm]**2+addErr**2+alnMag[mm]**2))
        else:
            print '%4.1f  %8.4f  %9.4f  %9.4f  %11.4f' % \
                  (magBins[mm], residMag[mm], errMag[mm],alnMag[mm],
                   np.sqrt(errMag[mm]**2+alnMag[mm]**2))


    # Sum all the error sources
    err_tot = np.sqrt(errMag**2 + addErr**2 + alnMag**2)

    # Get typical alignment error
    med_alnErr = np.median(aln_all)

    print ''
    print 'Median residual for K<14.5: %5.3f mas' % np.median(residMag[0:5])
    print 'Median RMS error for K<14.5: %5.3f mas' % np.median(errMag[0:5])
    print 'Median align error for K<14.5: %5.3f mas' % np.median(alnMag[0:5])
    if additiveErr==True:
        print 'Median quad sum of RMS err, align err, and additive err for K<14.5: %5.3f mas' % \
              np.median(np.sqrt(errMag**2+alnMag**2+addErr**2)[0:5])
        print ''
        print 'Median quad sum of RMS err and additive err for K<14.5: %5.3f mas' % \
              np.median(np.sqrt(errMag**2+addErr**2)[0:5])
    print ''


    # Also look at the errors in magnitude bins with equal number of stars,
    # instead of equally-spaced magnitude bins
    midx = mag_all.argsort()
    nMeasurements = 500
    magN = mag_all[midx]
    starN = starName_all[midx]
    eomN = eom_all[midx]
    alnN = aln_all[midx]
    diffN = diff_all[midx]

    Nbins = np.arange(0, len(midx)+nMeasurements, nMeasurements)
    residNmag = np.zeros(len(Nbins), float)
    errNmag = np.zeros(len(Nbins), float)
    alnNmag = np.zeros(len(Nbins), float)
    nMagBins = np.zeros(len(Nbins), float)
    numStars = np.zeros(len(Nbins), float)
    reqNmag = np.zeros(len(Nbins), float)
    print '%4s  %7s  %8s  %9s  %9s  %11s  %11s' % \
          ('#Mag', '# stars', 'Residual', 'RMS Error', 'Aln Error','Total Error','Resid-Total')
    for nn in range(1, len(Nbins)):
        nMin = Nbins[nn-1]
        nMax = Nbins[nn]
        residNmag[nn] = np.abs(np.median(diff_all[nMin:nMax])*1e3)
        errNmag[nn] = np.median(eom_all[nMin:nMax])
        alnNmag[nn] = np.median(aln_all[nMin:nMax])

        nMagBins[nn] = (magN[nMin:nMax]).mean()

        # How many stars on average go into these nMeasurements?
        numStars[nn] = len(np.unique(starN[nMin:nMax]))

        # What additive error is required to get quad sum of all errors
        # to total the residuals?
 #       if (residNmag[nn]**2 > (alnNmag[nn]**2 + errNmag[nn]**2)):
        reqNmag[nn] = np.sqrt(residNmag[nn]**2 - errNmag[nn]**2 - alnNmag[nn]**2)
 #       else:
 #           reqNmag[nn] = 0.0

        print '%4.1f  %7i  %8.4f  %9.4f  %9.4f  %11.4f  %11.4f' % \
              (nMagBins[nn], numStars[nn], residNmag[nn], errNmag[nn],alnNmag[nn],
               np.sqrt(errNmag[nn]**2+alnNmag[nn]**2), reqNmag[nn])
              
        
    residNmag = residNmag[np.nonzero(residNmag)[0]]
    errNmag = errNmag[np.nonzero(errNmag)[0]]
    alnNmag = alnNmag[np.nonzero(alnNmag)[0]]
    reqNmag = reqNmag[np.nonzero(reqNmag)[0]]
    nMagBins = nMagBins[np.nonzero(nMagBins)[0]]

    ##########
    # Plot
    ##########
    binsIn = np.arange(-7, 7, 0.5)
    py.clf()
    py.figure(figsize=(6,6))
    (nx, bx, ptx) = py.hist(sigmaX,bins=binsIn,color='k',histtype='step',linewidth=1)
    ggamp = ((sort(nx))[-2:]).sum() / (2.0 * ggy.max())
    plot(ggx, ggy*ggamp, 'k-', ms=5)
    py.xlabel('X Residuals (sigma)',fontsize=12)
    py.ylabel('N',fontsize=12)
    py.savefig(root+alnDir+'plots/residualsX.png')

    py.clf()
    (ny, by, pty) = py.hist(sigmaY,bins=binsIn,color='k',histtype='step',linewidth=1)
    ggamp = ((sort(ny))[-2:]).sum() / (2.0 * ggy.max())
    plot(ggx, ggy*ggamp, 'k-', ms=5)
    py.xlabel('Y Residuals (sigma)',fontsize=12)
    py.ylabel('N',fontsize=12)
    py.savefig(root+alnDir+'plots/residualsY.png')

    py.clf()
    sigmaA = []
    for ss in range(len(sigmaX)):
        sigmaA = np.concatenate([sigmaA,[sigmaX[ss]]])
        sigmaA = np.concatenate([sigmaA,[sigmaY[ss]]])
    (na, ba, pa) = py.hist(sigmaA, bins=binsIn,color='k',histtype='step',linewidth=1)
    ggamp = ((sort(na))[-2:]).sum() / (2.0 * ggy.max())
    py.plot(ggx, ggy*ggamp, 'k-')
    py.xlabel('Residuals (sigma)')
    py.ylabel('N',fontsize=12)
    py.savefig(outdir+'residualsXY.eps')
    py.savefig(outdir+'residualsXY.png')

    # Put them all in one plot
    py.clf()
    py.figure(figsize=(7,7))
    py.subplot(3, 1, 1)
    (nx, bx, px) = py.hist(sigmaX, bins=binsIn, color='k',histtype='step',linewidth=1)
    ggamp = ((sort(nx))[-2:]).sum() / (2.0 * ggy.max())
    py.plot(ggx, ggy*ggamp, 'k-')
    py.xlabel('X Residuals (sigma)')

    py.subplot(3, 1, 2)
    (ny, by, pty) = py.hist(sigmaY, bins=binsIn, color='k',histtype='step',linewidth=1)
    ggamp = ((sort(ny))[-2:]).sum() / (2.0 * ggy.max())
    py.plot(ggx, ggy*ggamp, 'k-')
    py.xlabel('Y Residuals (sigma)')

    py.subplot(3, 1, 3)
    (ny, by, pty) = py.hist(sigma, bins=np.arange(0, 7, 0.5),color='k',histtype='step',linewidth=1)
    py.xlabel('Total Residuals (sigma)')

    py.subplots_adjust(wspace=0.34, hspace=0.33, right=0.95, top=0.97)
    py.savefig(outdir+'residualsDistribution.png')

    # Also plot the various sources of error (averaged across epochs)
    # as a function of K mag
    # "Residuals" here is the offset in mas from the best fit
    import matplotlib.font_manager
    prop = matplotlib.font_manager.FontProperties(size=14)
    py.figure(figsize=(6,6))
    py.clf()
    py.plot(magBins, errMag, 'b-.', label=r'$\bf{\sigma_{rms}}$', lw=3)
    if additiveErr == True:
        py.plot([magBins.min(),magBins.max()], [addErr,addErr],'g--',lw=2,label=r'$\bf{\sigma_{add}}$')
    #py.plot([magBins.min(),magBins.max()], [med_alnErr,med_alnErr],'c--',lw=2,label=r'$\bf{\sigma_{aln}}$')
    py.plot(magBins, alnMag, 'c--', lw=2, label=r'$\bf{\sigma_{aln}}$')
    py.plot(magBins, err_tot, 'k-', label=r'$\bf{\sigma_{tot}}$')
    py.plot(magBins, residMag, 'r:', label='Residuals', lw=3)
    py.xlabel('K Magnitude', fontsize=16)
    py.ylabel('Positional Uncertainty (mas)', fontsize=16)
    py.legend(prop=prop,loc=2,markerscale=0.5)
    py.savefig(root + alnDir + 'plots/aveErrVsMag%s.png' % suffix)
    py.savefig(root + alnDir + 'plots/aveErrVsMag%s.eps' % suffix)

    # Plot residuals vs. K mag
    py.clf()
    py.semilogy(mag_all, diff_all*1e3,'k.',ms=0.5)
    py.xlabel('K Magnitude')
    py.ylabel('Residuals (mas)')
    py.savefig(outdir+'residVsKmag.png')


    # Also plot the various sources of error (averaged across epochs)
    # as a function of K mag, but binned by number of stars
    py.figure(figsize=(6,6))
    py.clf()
    py.plot(nMagBins, errNmag, 'b-.', label=r'$\bf{\sigma_{rms}}$', lw=3)
    #if additiveErr == True:
    # only plot the reqNmag where they are 'nan'
    foo = np.where(np.isnan(reqNmag) != True)[0]
    py.plot(nMagBins[foo], reqNmag[foo],'g--',lw=2,label=r'$\bf{\sigma_{add}}$')
    py.plot(nMagBins, alnNmag, 'c--', lw=2, label=r'$\bf{\sigma_{aln}}$')
    #py.plot(nMagBins, np.sqrt(errNmag**2+alnNmag**2), 'k-', label=r'$\bf{\sigma_{tot}}$')
    py.plot(nMagBins, residNmag, 'r:', label='Residuals', lw=3)
    py.xlabel('K Magnitude', fontsize=16)
    py.ylabel('Positional Uncertainty (mas)', fontsize=16)
    py.legend(prop=prop,loc=2,markerscale=0.5)
    py.savefig(root + alnDir + 'plots/aveErrVsMag_Nbins%s.png' % suffix)
    py.savefig(root + alnDir + 'plots/aveErrVsMag_Nbins%s.eps' % suffix)



def sigmaVsEpoch(alnDir='11_08_29/',
                 align='align/align_d_rms_1000_abs_t',
                 poly='polyfit_c/fit', useAccFits=False):
    """
    Plot the average offset (in sigma) from the best fit
    velocity as a function of epoch.
    """
    outdir = root + alnDir + 'plots/'

    s = starset.StarSet(root + alnDir + align, relErr=1)
    s.loadPolyfit(root + alnDir + poly, accel=0, arcsec=1)
    s.loadPolyfit(root + alnDir + poly, accel=1, arcsec=1)

    numEpochs = len(s.stars[0].years)
    
    # Use only stars detected in all epochs
    epochCnt = s.getArray('velCnt')
    idx = (np.where(epochCnt == epochCnt.max()))[0]
    newStars = []
    for ii in idx:
        newStars.append(s.stars[ii])
    s.stars = newStars
    
    print 'Using %d out of %d stars detected in %d epochs' % \
          (len(newStars), len(epochCnt), epochCnt.max())

    # Make some empty arrays to hold all our results.
    sigmaX = np.zeros(numEpochs, float)
    sigmaY = np.zeros(numEpochs, float)
    sigma  = np.zeros(numEpochs, float)
    diffEpX = np.zeros(numEpochs, float)
    diffEpY = np.zeros(numEpochs, float)
    diffEp  = np.zeros(numEpochs, float)

    # Fetch the fit parameters for all the stars
    if (useAccFits == True):
        fitVarX = 'fitXa'
        fitVarY = 'fitYa'
    else:
        fitVarX = 'fitXv'
        fitVarY = 'fitYv'

    t0 = s.getArray(fitVarX + '.t0')
    x0 = s.getArray(fitVarX + '.p')
    vx = s.getArray(fitVarX + '.v')
    y0 = s.getArray(fitVarY + '.p')
    vy = s.getArray(fitVarY + '.v')

    x0e = s.getArray(fitVarX + '.perr')
    y0e = s.getArray(fitVarY + '.perr')
    vxe = s.getArray(fitVarX + '.verr')
    vye = s.getArray(fitVarY + '.verr')

    if (useAccFits == True):
        ax = s.getArray(fitVarX + '.a')
        ay = s.getArray(fitVarY + '.a')
        axe = s.getArray(fitVarX + '.aerr')
        aye = s.getArray(fitVarY + '.aerr')

    # Loop through all the epochs and determine average residuals
    for ee in range(numEpochs):
        # Observed data
        x = s.getArrayFromEpoch(ee, 'x')
        y = s.getArrayFromEpoch(ee, 'y')
        xerr_p = s.getArrayFromEpoch(ee, 'xerr_p')
        yerr_p = s.getArrayFromEpoch(ee, 'yerr_p')
        xerr_a = s.getArrayFromEpoch(ee, 'xerr_a')
        yerr_a = s.getArrayFromEpoch(ee, 'yerr_a')
        xerr = hypot(xerr_p, xerr_a)
        yerr = hypot(yerr_p, yerr_a)
        t = s.stars[0].years[ee]

        dt = t - t0
        fitLineX = x0 + (vx * dt)
        fitSigX = np.sqrt( x0e**2 + (dt * vxe)**2 )

        fitLineY = y0 + (vy * dt)
        fitSigY = np.sqrt( y0e**2 + (dt * vye)**2 )

        if (useAccFits == True):
            fitLineX += (ax * dt**2) / 2.0
            fitLineY += (ay * dt**2) / 2.0
            fitSigX = np.sqrt(fitSigX**2 + (dt**2 * axe / 2.0)**2)
            fitSigY = np.sqrt(fitSigY**2 + (dt**2 * aye / 2.0)**2)

        # Residuals
        diffX = x - fitLineX
        diffY = y - fitLineY
        #diff = hypot(diffX, diffY)
        diff = (diffX + diffY) / 2.0
        rerr = np.sqrt((diffX*xerr)**2 + (diffY*yerr)**2) / diff
        sigX = diffX / xerr
        sigY = diffY / yerr
        sig = diff / rerr

        #print 'Epoch %d' % ee

        #print '%8.5f +/- %8.5f   %8.5f +/- %8.5f  %8.5f (%5.1f sigma)' % \
        #      (x[0], xerr[0], fitLineX[0], fitSigX[0], diffX[0], sigX[0])
        #print '%8.5f +/- %8.5f   %8.5f +/- %8.5f  %8.5f (%5.1f sigma)' % \
        #      (y[0], yerr[0], fitLineY[0], fitSigY[0], diffY[0], sigY[0])

        sigmaX[ee] = sigX.mean()
        sigmaY[ee] = sigY.mean()
        sigma[ee] = median(sig)

        diffEpX[ee] = diffX.mean()
        diffEpY[ee] = diffY.mean()
        diffEp[ee] = median(diff)

                        
    ##########
    # Plot
    ##########
    clf()
    years = s.stars[0].years
    plot(years, sigmaX, 'rx')
    plot(years, sigmaY, 'bx')
    plot(years, sigma, 'ko')
    xlabel('Epoch (years)')
    ylabel('Median Residual Error (sigma)')
    legend(('X', 'Y', 'Total'))

    #savefig(outdir+'residualsVsEpoch.eps')
    savefig(outdir+'residualsVsEpoch.png')

    clf()
    plot(years, diffEpX*1000.0, 'rx')
    plot(years, diffEpY*1000.0, 'bx')
    plot(years, diffEp*1000.0, 'ko')
    xlabel('Epoch (years)')
    ylabel('Median Residual Error (mas)')
    legend(('X', 'Y', 'Total'))
    #savefig(outdir+'residualsVsEpochMAS.eps')
    savefig(outdir+'residualsVsEpochMAS.png')


    # Print out epochs with higher than 3 sigma median residuals
    hdx = (np.where(sigma > 3))[0]
    print 'Epochs with median residuals > 3 sigma:'
    for hh in hdx:
        print '%8.3f  residual = %4.1f' % (s.stars[0].years[hh], sigma[hh])


def chi2distrib(alnDir='11_08_29/',
                align='align/align_d_rms_1000_abs_t', poly='polyfit_c/fit',
                points='points_c/',starlist='all',epochCut=10):

    outdir = root + alnDir + 'plots/'
    
    tag = ''
    if starlist=='all':
        tag = '_all'
    if starlist=='oldstars':
        tag = '_old'
    if starlist=='yngstars':
        tag = '_yng'

    s = loadPop(alnDir=alnDir,align=align,poly=poly,points=points,starlist=starlist)

    names = s.getArray('name')

    xchi2 = s.getArray('fitXv.chi2')
    ychi2 = s.getArray('fitYv.chi2')
    xchi2r = s.getArray('fitXv.chi2red')
    ychi2r = s.getArray('fitYv.chi2red')
    mag = s.getArray('mag')

    pntcnt = np.zeros(len(names))
    for ii in range(len(names)):
        pntFileName = '%s%s%s%s.points' % (root, alnDir, points, names[ii])
        pntFile = open(pntFileName)
        data = pntFile.readlines()
        pntcnt[ii] = len(data)
    # how many epochs are there?
    numEpochs = np.int(pntcnt[0]) # 16C should be in all epochs

    idx = np.where((pntcnt >= epochCut) & (mag < 16))[0]
    xchi2 = xchi2[idx]
    ychi2 = ychi2[idx]
    xchi2r = xchi2r[idx]
    ychi2r = ychi2r[idx]
    mag = mag[idx]
    pntcnt = pntcnt[idx]

    binsInX = np.arange(0.0, max(xchi2)+1, 2.0)
    binsInY = np.arange(0.0, max(ychi2)+1, 2.0)
    # Expected chi2 for n dof:
    dof = numEpochs - 2
    xpctdX = stats.chi2.pdf(binsInX,dof)

    # Only plot the stars detected in N=44 epochs
    all = np.where(pntcnt > 30)[0]
    print '%i stars with detections in >40 epochs' % len(all)
    py.clf()
    py.figure(figsize=(6,6))
    (nx,bx,p1x) = py.hist(xchi2[all],bins=binsInX,color='r',histtype='step',linewidth=2,label='X',normed=True)
    (ny,by,p1y) = py.hist(ychi2[all],bins=binsInY,color='b',histtype='step',linewidth=2,label='Y',normed=True)
    #print np.sum(nx*np.diff(bx))
    #print np.sum(ny*np.diff(by))
    py.plot(binsInX,xpctdX,'k--')
    py.xlabel('Total Chi Squared',fontsize=12)
    py.ylabel('N',fontsize=12)
    py.legend(numpoints=1)
    py.axis([0,100,0,0.03])
    py.savefig(root+alnDir+'plots/hist_chi2%s.png' % tag)


def fitfuncPL(p, fjac=None, e=None, cdf=None, err=None):
    """Find residuals of fit.

    For power-law, p should be list of form [A, alpha],
    while data should be list of form [e, cdf, err].
    """
    num = len(e)
    model = zeros(num, dtype=float)
    devs = zeros(num, dtype=float)

    # Set parameters
    A = p[0]
    alpha = p[1]

    model = A*(e**alpha)
    residuals = (cdf - model)/err
    status = 0

    return [status, residuals]

def fitPowerLawMP(p0=None,data=None,quiet=0):
    """Fits power law using mpfit.

    Inputs should be given as p0=[A, alpha] and
    data=[e,cdf,err]. Returns object of class
    mpfit.
    """

    print 'Initial Guess:'
    print '   A     = %6.2f' % p0[0]
    print '   alpha = %5.3f' % p0[1]

    # Set data to be passed to fit
    functargs = {'e':data[0],'cdf':data[1],'err':data[2]}

    # Set initial values and limits (no limits on parameters)
    pinfo = [{'value':0,'fixed':0,'limited':[0,0],
	      'limits':[0,0]}]*len(p0)
    for ii in range(len(p0)):
        pinfo[ii]['value'] = p0[ii]

    m = nmpfit_sy.mpfit(fitfuncPL, p0, functkw=functargs, parinfo=pinfo,
		    quiet=quiet)
    if (m.status <= 0):
        print 'Error message = ', m.errmsg

    p = m.params                # Best-fit parameters
    perr = m.perror             # Error in parameter fits
                                # from covariance matrix

    m.dof = len(data[0])-len(p) # Number of degrees of freedom
    Rchi2 = m.fnorm/m.dof       # Reduced Chi^2 statistic

    p = m.params          # Best-fit parameters
    perr = m.perror       # Error in parameter fits from covariance matrix
    #x   = (data[0])[nonzero(data[2])]  
    #m.dof = len(x)-len(p) # Number of degrees of freedom

    print 'Final Solution:'
    print '   A      = %6.2f +/- %5.2f' % (p[0],perr[0])
    print '   alpha  = %5.3f +/- %5.3f' % (p[1],perr[1])
    print '   chi^2  = %7.4f' % m.fnorm
    print '   Rchi^2 = %7.4f' % Rchi2

    eline = arange(.01,1.0,.1)
    cdfline = p[0]*eline**p[1]

    return m


def getOrbElements(alnDir='09_09_20/',
                   starType='all'):
    """
    For the stars with orbits from efit, read their .param output
    files and get their orbital elements
    """

    _list = asciidata.open(root + alnDir + 'efit/names.dat')
    names = _list[0]._data
    _type = _list[1].tonumpy()

    idx = []
    if starType == 'yng':
        for i in range(len(names)):
            if (_type[i] == 1):
                idx = np.concatenate([idx, [i]])
    elif starType == 'old':
        for i in range(len(names)):
            if (_type[i] == 0):
                idx = np.concatenate([idx, [i]])
    elif starType == 'all':
        idx = range(len(names))
    names = np.array([names[int(ii)] for ii in idx])
    _type = np.array([_type[int(ii)] for ii in idx])

    numStars = len(names)
    r0 = np.zeros(numStars)
    a = np.zeros(numStars)
    p = np.zeros(numStars)
    e = np.zeros(numStars)
    t0 = np.zeros(numStars)
    smallOmega = np.zeros(numStars)
    i = np.zeros(numStars)
    bigOmega = np.zeros(numStars)
    aUp = np.zeros(numStars)
    pUp = np.zeros(numStars)
    eUp = np.zeros(numStars)
    t0Up = np.zeros(numStars)
    smallOmegaUp = np.zeros(numStars)
    iUp = np.zeros(numStars)
    bigOmegaUp = np.zeros(numStars)
    aLo = np.zeros(numStars)
    pLo = np.zeros(numStars)
    eLo = np.zeros(numStars)
    t0Lo = np.zeros(numStars)
    smallOmegaLo = np.zeros(numStars)
    iLo = np.zeros(numStars)
    bigOmegaLo = np.zeros(numStars)

    search = 2

    outFile = root + alnDir + 'efit/new_orbits.dat'
    out = open(outFile, 'w')
    hdr = '%10s  %6s  %6s  %10s  %7s  %6s  %6s  %6s  %5s\n'
    fmt = '%10s  %6.2f  %6.1f  %10.4f  %7.4f  %6.1f  %6.1f  %6.1f  %5i\n'
    out.write(hdr % ('#Star','P','A','t0','e','i','Omega','omega','search'))
    out.write(hdr % ('#Name','(yrs)','(mas)','(yrs)','()','(deg)','(deg)','(deg)','(pix)'))

    outFile2 = root + alnDir + 'efit/cntrl_rva.dat'
    out2 = open(outFile2, 'w')
    out2.write('%15s    %5s     %12s      %14s     %12s  %8s\n' % \
          ('Name','r (")','v2d (mas/yr)','a_rad (mas/yr^2)','a_rad (sigma)','t0'))

    s = loadPop(alnDir=alnDir,starlist='all',align='align/align_p_rms_1000_abs_t',
                poly='polyfit_p/fit')
    allNames = s.getArray('name')

    # In mas, mas/yr, mas/yr^2
    x0 = s.getArray('x0')
    y0 = s.getArray('y0')
    x0e = s.getArray('x0e')
    y0e = s.getArray('y0e')
    vx = s.getArray('vx')
    vy = s.getArray('vy')
    v2d = np.sqrt(vx**2 + vy**2)
    vxe = s.getArray('vxe')
    vye = s.getArray('vye')
    ax = s.getArray('ax')
    ay = s.getArray('ay')
    a2d = np.sqrt(ax**2 + ay**2)
    axe = s.getArray('axe')
    aye = s.getArray('aye')
    cnt = s.getArray('cnt')
    t0 = s.getArray('t0x')
    mag = s.getArray('mag')

    # Get accelerations in the radial/tangential direction
    r = np.sqrt(x0**2 + y0**2)
    ar = ((ax*x0) + (ay*y0)) / r
    at = ((ax*y0) - (ay*x0)) / r
    are =  (axe*x0/r)**2 + (aye*y0/r)**2
    are += (y0*x0e*at/r**2)**2 + (x0*y0e*at/r**2)**2
    are =  np.sqrt(are)
    ate =  (axe*y0/r)**2 + (aye*x0/r)**2
    ate += (y0*x0e*ar/r**2)**2 + (x0*y0e*ar/r**2)**2
    ate =  np.sqrt(ate)

    r_stars = []
    orb_mags = []

    print '%15s   %5s    %5s     %12s      %14s     %12s  %8s' % \
          ('Name','KMag','r (")','v2d (mas/yr)','a_rad (mas/yr^2)','a_rad (sigma)','t0')
    for ss in range(numStars):
        _fileo = open(root + alnDir + 'efit/orbit.' + names[ss] + '.output.param1')
        _file = _fileo.readline().split()
        _elem = open(root + alnDir + 'efit/orbit.' + names[ss] + '.output.error')
        elem = _elem.readline().split()

        r0[ss] = float(_file[0])
        a[ss] = float(elem[0])
        aUp[ss] = float(elem[1])
        aLo[ss] = float(elem[2])
        p[ss] = float(elem[3])
        pUp[ss] = float(elem[4])
        pLo[ss] = float(elem[5])
        e[ss] = float(elem[6])
        eUp[ss] = float(elem[7])
        eLo[ss] = float(elem[8])
        t0[ss] = float(elem[9])
        t0Up[ss] = float(elem[10])
        t0Lo[ss] = float(elem[11])
        smallOmega[ss] = float(elem[12])
        smallOmegaUp[ss] = float(elem[13])
        smallOmegaLo[ss] = float(elem[14])
        i[ss] = float(elem[15])
        iUp[ss] = float(elem[16])
        iLo[ss] = float(elem[17])
        bigOmega[ss] = float(elem[18])
        bigOmegaUp[ss] = float(elem[19])
        bigOmegaLo[ss] = float(elem[20])

        # Write it out to an orbits.dat type file
        out.write(fmt % (names[ss], p[ss], a[ss], t0[ss], e[ss], i[ss],
                         bigOmega[ss], smallOmega[ss], search))

        # Get other information on these stars (r, v, a, etc)
        nn = allNames.index(names[ss])
        print '%15s   %5.2f   %5.3f"   %8.3f mas/yr   %9.3f mas/yr^2   %8.2f sigma    %8.3f' % \
              (names[ss], mag[nn], r[nn], v2d[nn]*1e3, ar[nn]*1e3, ar[nn]/are[nn], t0[nn])
        out2.write('%15s   %5.3f"   %8.3f mas/yr   %9.3f mas/yr^2   %8.2f sigma    %8.3f\n' % \
              (names[ss], r[nn], v2d[nn]*1e3, ar[nn]*1e3, ar[nn]/are[nn], t0[nn]))

        # Get the r2d's for these stars:
        r_stars = np.concatenate([r_stars,[r[nn]]])

        orb_mags = np.concatenate([orb_mags,[mag[nn]]])
        
    out.close()
    out2.close()

    # Convert semi-major axis from mas to pc
    a *= dist / (1.e3*206265.)
    aUp *= dist / (1.e3*206265.)
    aLo *= dist / (1.e3*206265.)

    # Make errors symmetric by taking average of upper and lower errors
    a_err = (aUp + aLo) / 2.0
    p_err = (pUp + pLo) / 2.0
    e_err = (eUp + eLo) / 2.0
    t0_err = (t0Up + t0Lo) / 2.0
    smallOmega_err = (smallOmegaUp + smallOmegaLo) / 2.0
    i_err = (iUp + iLo) / 2.0
    bigOmega_err = (bigOmegaUp + bigOmegaLo) / 2.0

    # Get peri- and apo- distances 
    amin = a * (1. - e)
    amax = a * (1. + e)

    inner = np.where(r_stars < 0.8)[0]
    outer = np.where(r_stars >= 0.8)[0]
    yng = np.where(_type == 1)[0]
    old = np.where(_type == 0)[0]
    unk = np.where(_type < 0)[0]

    iDisk = 115.
    iDiske = 3.
    WDisk = 100.
    WDiske = 3.
    iSig = [np.hypot(iDiske, ii) for ii in i_err]
    WSig = [np.hypot(WDiske, ii) for ii in bigOmega_err]
    onD = np.where((abs(i - iDisk) < iSig) | (abs(bigOmega - WDisk) < WSig))[0]

    fig = py.figure(1)
    fig.clear()
    fig.hold(True)
    ax = fig.add_subplot(111)
    #loga = np.log10(a)
    #logaEU = np.log10(a+a_err)-loga
    #logaEL = -np.log10(a-a_err)+loga
    #logaE = np.array((logaEL, logaEU))
    #logaE_old = np.array((logaEL[old], logaEU[old]))
    #logaE_yng = np.array((logaEL[yng], logaEU[yng]))
    #logaE_unk = np.array((logaEL[unk], logaEU[unk]))
    p1 = ax.errorbar(a[old],e[old],xerr=a_err[old],yerr=e_err[old],fmt='r.',ms=8)
    p2 = ax.errorbar(a[yng],e[yng],xerr=a_err[yng],yerr=e_err[yng],fmt='b.',ms=8)
    p3 = ax.errorbar(a[unk],e[unk],xerr=a_err[unk],yerr=e_err[unk],fmt='k.',ms=8)
    #p1 = py.errorbar(loga[old],e[old],xerr=logaE_old,yerr=e_err[old],fmt='r.',ms=8)
    #p2 = py.errorbar(loga[yng],e[yng],xerr=logaE_yng,yerr=e_err[yng],fmt='b.',ms=8)
    #p3 = py.errorbar(loga[unk],e[unk],xerr=logaE_unk,yerr=e_err[unk],fmt='k.',ms=8)
    vi = ax.xaxis.get_view_interval().astype(np.int)
    ax.set_xscale('log')
    py.xlabel('Semi-major Axis (pc)')
    py.ylabel('Eccentricity')
    lgd1 = 'Old'
    lgd2 = 'Young'
    lgd3 = 'Unknown'
    lgd = ax.legend((p1[0], p2[0], p3[0]), (lgd1,lgd2,lgd3), numpoints=1, fancybox=True, loc=2,
                    labelspacing=0.01,handletextpad=0.01,borderpad=0.1,handlelength=0.75)
    for t in lgd.get_texts():
        t.set_fontsize('x-small')
    py.savefig(root + alnDir + 'plots/cntrl_eccVsA_test.png')
    py.close(1)

    fig = py.figure(2)
    fig.clear()
    fig.hold(True)
    usetexTrue()
    ax = fig.add_subplot(111)
    pIn = ax.errorbar(i[inner],bigOmega[inner],xerr=i_err[inner],yerr=bigOmega_err[inner],fmt='r.')
    pOut = ax.errorbar(i[outer],bigOmega[outer],xerr=i_err[outer],yerr=bigOmega_err[outer],fmt='b.')
    # Also plot a point for the i,Omega of the disk (per JLu)
    ax.errorbar([115],[100],yerr=3,xerr=3,fmt='ko',ms=5)
    lgda = r'r $<$ 0.8"'
    lgdb = r'r $\geq$ 0.8"'
    lgd2 = ax.legend((pIn[0], pOut[0]), (lgda,lgdb), numpoints=1)
    lgdLines = lgd2.get_lines()
    py.xlabel('Inclination (deg)')
    py.ylabel(r'$\Omega$ (deg)')
    ax.axis([0,180,0,360])
    py.savefig(root + alnDir + 'plots/cntrl_OmegaInc.png')
    py.close(2)

    py.figure(3)
    py.clf()
    py.plot(amin,amax,'k.')
    py.xlabel('Periastron Distance (pc)')
    py.ylabel('Apoastron Distance (pc)')
    py.savefig(root + alnDir + 'plots/cntrl_AminVsAmax.png')
    py.close(3)

    py.figure(4)
    py.clf()
    binsIn = np.arange(0.0001, 1.0, 0.05)
    py.figure(figsize=(6,6))
    (bins,data)=histNofill.hist(binsIn,e)
    py.plot(bins,data)
    py.xlabel('Eccentricity',fontsize=12)
    py.ylabel('N',fontsize=12)
    py.axis([0,1,0,max(data)+1])
    py.savefig(root + alnDir + 'plots/hist_ecc.png')
    py.close(4)

    # Also plot up the cumulative distribution function for eccentricities
    e_sort = sort(e)
    cdf = range(1, len(e) + 1)
    cdf = [cdf[cc] / (len(e)*1.0) for cc in range(len(cdf))]
    e0 = [1.0, 2.0]
    ecc_err = array([1.0*len(e_sort)])
    eccfit = fitPowerLawMP(e0,[e_sort, cdf, ecc_err],1)
    eccParams = eccfit.params
    eline = np.arange(0.01, 1.0, 0.01)
    cdfline = eccParams[0]*eline**eccParams[1]
    py.figure(5)
    py.clf()
    py.plot(e_sort, cdf,'k.')
    py.plot(eline,cdfline,'k-')
    py.xlabel('Eccentricity',fontsize=12)
    py.ylabel('CDF',fontsize=12)
    py.text(0.05,0.7, r'cdf ${\sim} e^{%4.2f}$' % eccParams[1])
    py.axis([0,1,0,1])
    py.savefig(root + alnDir + 'plots/cdf_ecc.png')
    usetexFalse()
    py.close(5)


    py.figure(6)
    py.clf()
    p1 = py.errorbar(a[old],orb_mags[old],xerr=a_err[old],fmt='r.',ms=15)
    p2 = py.errorbar(a[yng],orb_mags[yng],xerr=a_err[yng],fmt='b.',ms=15)
    p3 = py.errorbar(a[unk],orb_mags[unk],xerr=a_err[unk],fmt='k.',ms=15)
    py.xlabel('Semi-major Axis (pc)')
    py.ylabel('K Magnitude')
    lgd1 = 'Old'
    lgd2 = 'Young'
    lgd3 = 'Unknown'
    lgd = py.legend((p1, p2, p3), (lgd1,lgd2,lgd3), numpoints=1)
    lgdLines = lgd.get_lines()
    lgdLines[0].set_marker('.')
    lgdLines[1].set_marker('.')
    lgdLines[2].set_marker('.')
    lgdLines[0].set_mfc('r')
    lgdLines[1].set_mfc('b')
    lgdLines[2].set_mfc('k')
    lgdLines[0].set_ms(12)
    lgdLines[1].set_ms(12)
    lgdLines[2].set_ms(12)
    py.savefig(root + alnDir + 'plots/mag_A.png')




#    Star   X (pc)   Y (pc)  Zmin (pc)    a_rad (km/s/yr)
#--------------------------------------------------------
#    S0-3    0.013    0.005    0.013     -34.670
#    S0-5    0.007   -0.014    0.007     -53.029
#  * S0-26    0.013    0.008    0.026     -8.939
#  *  S0-8   -0.009    0.006    0.007     -88.541
#    S0-4    0.016   -0.012    0.012     -27.715
#    S0-7    0.019    0.004    0.029     -7.195
#    S0-9    0.008   -0.023    0.093     -0.111
#   S0-30   -0.019   -0.015    0.024     -9.248
#  * S0-31    0.021    0.017    0.070     -0.167
#  * S0-14   -0.030   -0.011    0.051     -2.275
#  *  S1-3    0.014    0.034    0.037     -4.243
#  *  S1-1    0.040    0.001    0.056     -1.862
#  *  S1-2    0.002   -0.039    0.056     -1.823
#    S1-4    0.033   -0.026    0.023     -6.586
#  *  S1-8   -0.023   -0.035    0.069     -1.062
#  * irs16C    0.042    0.021    0.045     -2.726
#  * S1-33   -0.048   -0.001    0.041     -2.623
#  * S1-12   -0.030   -0.040    0.085     -0.570
#  * S1-14   -0.052   -0.014    0.025     -4.304
# irs16SW    0.042   -0.037    0.034     -3.211
#  * S1-21   -0.064    0.004    0.015     -3.668
#  * S1-22   -0.062   -0.020    0.019     -3.416
#    S2-4    0.058   -0.057    0.049     -1.338
#    S2-6    0.064   -0.052    0.009     -2.243
#  * irs29N   -0.061    0.054    0.022     -1.840
#   S2-17    0.051   -0.073    0.098     -0.310

