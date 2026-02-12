import asciidata, pyfits, pickle
import os, sys, math, time, mpfit
import nmpfit_sy
import __builtin__
from matplotlib import patches
import numpy as np
import pylab as py
import random
import shutil
import healpy
import matplotlib.nxutils as nx
from gcwork import starset
from gcwork import objects
from gcwork import util
from gcwork import young
from gcwork import orbits
from gcwork import analyticOrbits as aorb
#from gcwork import analyticOrbits2 as aorb
from gcwork import starTables
from gcwork import plot_disk_healpix as pdh
from gcwork.plotgc import plotStar
from gcreduce import gcutil
import starTables as tabs
import syYoung
import sy_open 
from pysqlite2 import dbapi2 as sqlite
from scipy import stats
from scipy import optimize
import scipy.integrate
import accel_class as acc
import ks2
import histNofill
import histogram2d as h2d
import pdb

root = '/u/syelda/research/gc/aligndir/'
alnDir = '11_10_26/'
poly='polyfit_c/fit'
points='points_c/'
plotdir = root + alnDir + 'plots/'
# Wide mosaic files
#mscDir=root + '11_09_12_dp_msc/'
#polyM='polyfit_1000/fit'
#pointsM='points_1000/'
mscDir=root + '12_01_30_dp_msc/'
polyM='polyfit_noDistErr/fit'
pointsM='points_noDistErr/'
#mscDir=root + '12_01_22_dp_msc/'
#polyM='polyfit_noDistErr/fit'
#pointsM='points_noDistErr/'
# Maser mosaic files
#mscDir='/u/syelda/research/gc/absolute/11_05_09/'
#polyM='polyfit_noDistErr/fit'
#pointsM='points_noDistErr/'
idisk = 130.2
odisk = 96.3
ierr = 2.0
oerr = 2.0
rad2deg = 180.0 / math.pi
angleCut = 15.2
#angleCut = 10.0
diskDensity = 0.024 # stars/deg^2
mass = 4.6e6 # latest values from 11_10_26/efit
massErr = 0.72e6 # latest values from 11_10_26/efit
dist = 8232.9 # latest values from 11_10_26/efit


areaOnSky = 4.0 * math.pi * (180.0 / math.pi)**2    # in deg^2


def go():
    # Plot acceleration vs. projected radius
    #accelLimit(ftest=True,pvalue=4.0,suffix='_4sigmaFtest')
    # updated version in:
    syelda_yngstars.accelLimit(alnDir='11_10_26/', suffix='_paper')

    ##################
    #
    # Orbit analysis
    #
    ##################
    orbitAnalysisAll(makePlot=True, pvalue=4.0)

    # Use Mass/Ro/focus measurements for BH potential from Ghez et al. 2008
    #orbitAnalysisPDF('aorb_acc/', 10**5)

    # Create a PDF of Mass/Ro/focus for BH potential based on latest efit
    # and create a PDF of rho0's from Schoedel+09 extended mass distribution
    from gcwork import bhPotential as bh
    pt = bh.BHprops()
    pt.generate(efitFile=root + alnDir + 'efit/MC/mc_total.log')
    pt.saveToFile(root + alnDir + 'tables/massRo_111026.dat')
    mr_new = pickle.load( open(root + alnDir + 'tables/massRo_111026.dat') )

    pt.generate_rho0()
    pt.saveToFile(root + alnDir + 'tables/rho0_schoedel09.dat')
    rho0data = pickle.load( open(root + alnDir + 'tables/rho0_schoedel09.dat') )

    # Use efit MC from our latest efit (in 11_10_26) and include extended mass
    orbitAnalysisPDF('aorb_thesis/', 99998, doMassRo=mr_new, mExtended=rho0data, mosaic=True)
    # Make density map for inner, middle and outer radial bins
    disk = aorb.Disk(root+alnDir, 'aorb_thesis/', mscDir=mscDir)
    disk.run(makeplot=True, do_all=True, do_radial_bins=False)
    # 3 radial bins, including mosaic
    disk.run(makeplot=True, do_all=False, do_radial_bins=True, do_r1=True)
    disk.run(makeplot=True, do_all=False, do_radial_bins=True, do_r2=True)
    disk.run(makeplot=True, do_all=False, do_radial_bins=True, do_r3=True)

    # Plot up the results from the disk analysis
    pdh.go('aorb_thesis/disk.neighbor.dat', 49152, 4, plottrial=2)
    pdh.go('aorb_thesis/inner_disk.neighbor.dat', 49152, 4, plottrial=2)
    pdh.go('aorb_thesis/middle_disk.neighbor.dat', 49152, 4, plottrial=2)
    pdh.go('aorb_thesis/outer_disk.neighbor.dat', 49152, 4, plottrial=2)

    # Disk Membership
    diskMembers(mosaic=True, suffix='_mosaic', singlePdf=True)
    diskMembers(mosaic=True, suffix='_inner_disk',radial_bin=1,
                file1='inner_disk.neighbor.dat',
                file2='inner_disk.neighborStd.dat', singlePdf=True)
    diskMembers(mosaic=True, suffix='_middle_disk',radial_bin=2,
                file1='middle_disk.neighbor.dat',
                file2='middle_disk.neighborStd.dat',
                singlePdf=True)
    diskMembers(mosaic=True, suffix='_outer_disk',radial_bin=3,
                file1='outer_disk.neighbor.dat',
                file2='outer_disk.neighborStd.dat',
                singlePdf=True)
    # Determine peak positions and uncertainties of disk
    peakPosErrPdf()

    histSolidAngles()
    membersVsRadius()

    # Eccentricity Analysis
    plotEccAnalytic()
    pdfEccentricity(pdfdir='aorb_thesis/', mosaic=True, simFlat=False,
                    suffix='_mosaic')

    # CCW disk?
    diskMembersCCW()
    checkCCWdisk(aperture=False)



def usetexTrue():
    py.rc('text', usetex=True)
    py.rc('font', **{'family':'sans-serif', 'size':16})
    #py.rc('axes', titlesize=20, labelsize=20)
    py.rc('xtick', labelsize=16)
    py.rc('ytick', labelsize=16)

def usetexFalse():
    py.rc('text', usetex=False)
    py.rc('font', family='sans-serif', size=14)
    #py.rc('axes', titlesize=16, labelsize=16)
    py.rc('xtick', labelsize=14)
    py.rc('ytick', labelsize=14)


def merge(ob1, ob2):
    """
    Merge two starset objects. Useful for merging the objects from
    the central 10 arcsec analysis with the deep mosaic analysis.

    ob1 should be central 10 arcsec data set
    ob2 should be wide mosaic data set
    """

    names = ob1.getArray('name')

    # Loop through the mosaic stars
    for ii in range(len(ob2.stars)):
        # If this mosaic star is already in central 10 asec, don't include it!
        if ob2.stars[ii].name in names:
            continue
        else:
            ob1.stars.append(ob2.stars[ii])

    return ob1



def accelLimit(align = 'align/align_d_rms_1000_abs_t',sig_acc=5,
              ftest=False,pvalue=None,suffix=''):
    """
    Plot plane-of-the-sky acceleration limits as a function of projected distance.

    Inputs:
    align     = The align root file name (including the directory relative
                to root). Make sure that polyfit was run on this align
		output.
    sig_acc   = Sigma value for what is considered 'significant' (def=4 sigma)
    ftest     = Set to True to compute F statistic to see if acceleration
     		fit is warranted over a velocity fit.
    pvalue    = Significance threshold for F test
    suffix    = Extention to filename for plots

    Output:
    r2d_accel_detectUpLim_yng.png

    """
    if ftest == True:
        # Compute F test to see if acceleration fits are warranted
        # pvalue = Significance threshold for F test
        # Stars that should have acceleration fits are returned:
        nameF, xFp, yFp = syYoung.velocity_vs_accel(alnDir=alnDir,pvalue=pvalue)
        
    outdir = root + alnDir + 'plots/'

    # Load up young stars
    # young.loadYoungStars calls youngStarNames but grabs
    # only young stars beyond r = 0.8 arcsec
    yng = young.loadYoungStars(root+alnDir,fit=poly,points=points,radiusCut=0.8,
                               withRVonly=True,skipStar=['S5-237']) 
        
    rv_ref = np.array(yng.getArray('rv_ref')) # just for knowing how many have RVs
    noRV = np.where(np.array([rr is None for rr in rv_ref]) == True)[0]
    hasRV = np.where(np.array([rr is None for rr in rv_ref]) == False)[0]

    cc = objects.Constants()
    #GM = cc.G * cc.mass * cc.msun
    GM = cc.G * mass * cc.msun

    # Construct an array of radii out to 7 arcsec in steps of 0.1''
    r = (np.arange(10 * 7) * 0.1) + 0.1
    r_au = r * dist
    r_pc = r_au / cc.au_in_pc
    r_cm = r_au * cc.cm_in_au

    # Determine the theoretical amount for a vs. r
    a_cm_s2 = -GM / r_cm**2
    # Switch to using mass + 1 sigma for BH mass
    #a_cm_s2 = -GM3sig / r_cm**2
    a_km_s_yr = a_cm_s2 * cc.sec_in_yr / 1.0e5

    # Error on the computed theoretical a_max curve
    sig_a_cgs = cc.G / r_cm**2 * massErr * cc.msun
    sig_a_mks = sig_a_cgs * cc.sec_in_yr / 1.0e5

    # Upper limit of theoretical a_max curve (3 sigma)
    GM3sig = cc.G * (mass + 3.0*massErr) * cc.msun
    #GM1sig = cc.G * (mass + 1.0*massErr) * cc.msun

    # Plot this curve.
    usetexTrue()
    py.figure(figsize=(6,6))
    py.clf()
    py.plot(r_pc, -a_km_s_yr, 'k--')
    py.plot(r_pc, -(a_km_s_yr+3.*sig_a_mks), 'k:')
    py.plot(r_pc, -(a_km_s_yr-3.*sig_a_mks), 'k:')
    py.plot([0,0.25],[0,0],'k--')

    # Set axis labels and font properties
    py.xlabel(r'{\bf Projected Radius (pc)}')
    py.ylabel(r'{\bf $|$a$_\rho|$ (km/s/yr)}')

    py.text(0.031, 33, r'{\bf $|$a$|_{max} = \frac{GM}{\rho^2}$}')

    # Loop through all the stars.
    names = yng.getArray('name')
    xchi2r = yng.getArray('fitpXa.chi2red')
    ychi2r = yng.getArray('fitpYa.chi2red')

    allAccLimRad = np.zeros(len(names))
    
    outFile = '%s%stables/young_accel%s.dat' % (root, alnDir, suffix)
    out = open(outFile, 'w')
    if ftest == True:
        hdr = '%2s  %15s  %4s  %4s  %6s vs. %6s      %5s   %6s %6s   %6s   %8s  %10s  %11s\n'
        print 'Accelerations in km/s/yr, radius in parsecs'
        print hdr % \
              ('# ', 'Name', 'K', 'Cnts', 'aLim', 'aMax', 'rLo', 'ar', 'are', 'azmax',
               'abound','detection?','Acc or Vel?')
        out.write('Accelerations in km/s/yr, radius in parsecs\n')
        out.write(hdr % ('# ', 'Name', 'K', 'Cnts', 'aLim', 'aMax', 'rLo', 'ar', 'are',
                         'azmax', 'abound', 'detection?','Acc or Vel?'))
    else:
        hdr = '%2s  %15s  %4s  %4s  %6s vs. %6s      %5s   %6s %6s   %6s   %8s  %10s\n'
        print 'Accelerations in km/s/yr, radius in parsecs'
        print hdr % \
              ('# ', 'Name', 'K', 'Cnts', 'aLim', 'aMax', 'rLo', 'ar', 'are',
               'azmax', 'abound', 'detection?')
        out.write('Accelerations in km/s/yr, radius in parsecs\n')
        out.write(hdr %
                  ('# ', 'Name', 'K', 'Cnts', 'aLim', 'aMax', 'rLo', 'ar', 'are',
                   'azmax', 'abound', 'detection?'))
        
    ii = 0

    detection = np.zeros((len(names)),dtype=int)

    #if ftest == True:
    #    _acc = open(root + alnDir + 'tables/accelerating_sources.dat','w')
    #_upper = open(root + alnDir + 'tables/accel_upperLimit_sources.dat','w')

    rLower_all = []
    ar_all = []
    at_all = []
    are_all = []
    ate_all = []
    abound_all = []
    cnt_all = []
    xe_all = []
    ye_all = []
    pm_all = []
    pme_all = []
    for name in names:
	i = names.index(name)

	star = yng.stars[i]

        cc.asy_to_kms = dist * cc.cm_in_au / (1.e5 * cc.sec_in_yr)

        mag = yng.stars[i].mag
        x = star.fitXa.p
        y = star.fitYa.p
        xe = star.fitXa.perr
        ye = star.fitYa.perr
        vx = star.fitXv.v * cc.asy_to_kms # km/s - NOTE: using velocity fit here!
        vy = star.fitYv.v * cc.asy_to_kms # km/s - NOTE: using velocity fit here!
        vxe = star.fitXv.verr * cc.asy_to_kms # km/s - NOTE: using velocity fit here!
        vye = star.fitYv.verr * cc.asy_to_kms # km/s - NOTE: using velocity fit here!
        ax = star.fitXa.a
        ay = star.fitYa.a
        axe = star.fitXa.aerr
        aye = star.fitYa.aerr
        t0 = star.fitXa.t0
        #cnt = star.velCnt

        pm = np.hypot(vx, vy)
        pm_err = np.sqrt(((vx*vxe)**2 + (vy*vye)**2) / pm**2) # km/s
        pm_all = np.concatenate([pm_all, [pm]])
        pme_all = np.concatenate([pme_all, [pm_err]])
        
        # How many epochs was each star detected in?
        pts = asciidata.open('%s%s%s%s.points' % (root, alnDir, points, name))
        ep = pts[0].tonumpy()
        cnt = len(ep)
        cnt_all = np.concatenate([cnt_all, [cnt]]) 

        r = np.sqrt(x**2 + y**2)
        rcgs = r * dist * cc.cm_in_pc / cc.au_in_pc
        
	# Lower allowed radius is set by 2D distance.
	rLower = r * dist / cc.au_in_pc

	ax *= cc.asy_to_kms
	ay *= cc.asy_to_kms
	axe *= cc.asy_to_kms
	aye *= cc.asy_to_kms
	
	# What is the polyfit acceleration upper limit
	# along the radial component, in km/s/yr
        (ar, at, are, ate) = util.xy2circErr(x, y, ax, ay,
                                             xe, ye, axe, aye)

        # Save off the projected radii
        rLower_all = np.concatenate([rLower_all, [rLower]]) # pc
        # Save off the accels
        ar_all = np.concatenate([ar_all, [ar]]) # km/s/yr
        at_all = np.concatenate([at_all, [at]]) # km/s/yr
        # Save off the accel errors for a histogram later
        are_all = np.concatenate([are_all, [are]]) # km/s/yr
        ate_all = np.concatenate([ate_all, [ate]]) # km/s/yr

        # Save off the positinoal errors for a histogram later
        xe_all = np.concatenate([xe_all, [xe*1.e3]]) # mas
        ye_all = np.concatenate([ye_all, [ye*1.e3]]) # mas 
	
	# What is the radial component upper limit
	accLimRadial = ar - (sig_acc * are) # km/s/yr
	#accLimRadial = ar - (3.0 * are) # km/s/yr
        alimcgs = accLimRadial * 1.0e5 / cc.sec_in_yr # cm/s/s

        # What is the z-lower limit as set by this accel. limit
        foo = (-GM * rcgs / alimcgs)**(2.0/3.0)# Gives (minimum r)-squared
        #foo = (-GM3sig * rcgs / alimcgs)**(2.0/3.0)# Gives (minimum r)-squared
        #if ((foo - rcgs**2) > 0): # Below the curve (at X sigma)
        if (((foo - rcgs**2) > 0) | (name == 'S0-15') | (name == 'S1-14')): # Below the curve (at X sigma)
            # NOTE: specifically selecting S0-15 and S1-14 b/c they are
            # below the curve when considering the curve+1 sigma, which is the
            # new criteria
            zmincgs = np.sqrt(foo - rcgs**2)
            # Is it also significantly different from zero?
            if (-ar - (sig_acc*are) > 0.0):
                # Constraint on z
                detection[ii] = 2
                det = py.errorbar(rLower, -ar, yerr=(sig_acc*are), fmt='b.',ms=8)
                if ftest == True:
                    if ((name in nameF) & (rv_ref[ii] != None) & (cnt > 30)):
                        py.plot(rLower, -ar, 'bx', ms=8)
                        # Write out this star name to a file listing accelerating sources
                        #_acc.write('%8s\n' % name)
            else:
                # Below the curve, but still consistent w/ zero. Still gives
                # strong constraints on zmin (rules out small z)
                detection[ii] = 1
                uppr = py.plot([rLower], [-accLimRadial], 'k_')
                py.arrow(rLower, -accLimRadial, 0, -2, hold=True, color='black',\
                         width=0.0005, head_length=1, linewidth=0.002) #, \
                #py.plot(rLower, -ar, 'gx')
                #_upper.write('%8s\n' % name)
                if ftest == True:
                    #if name in nameF:
                    if ((name in nameF) & (rv_ref[ii] != None) & (cnt > 30)):
                        py.plot(rLower, -ar, 'bx', ms=8)
        else:
            zmincgs = 0.0
            detection[ii] = 0
            uppr = py.plot([rLower], [-accLimRadial], 'k_')
            py.arrow(rLower, -accLimRadial, 0, -2, hold=True, color='black',\
                     width=0.0005, head_length=1, linewidth=0.002) #, \
            #py.plot(rLower, -ar, 'gx')
            if ftest == True:
                if name in nameF:
                    py.plot(rLower, -ar, 'bx', ms=8)

        zmin = zmincgs / (cc.cm_in_pc)
          

        # We need to add in the highest possible acceleration allowed
        # in the line-of-sight direction. There are two possibilities:
        #   1. highest acc is at smallest z allowed by acc in sky
        #   2. highest acc is at z = rho / sqrt(2) (from derivative)
        if (zmincgs > (rcgs / np.sqrt(2.0))):
            #print 'max az set by z-limit from acc. in sky'
            azmax = -GM * zmincgs / (rcgs**2 + zmincgs**2)**(3.0/2.0)
            #azmax = -GM3sig * zmincgs / (rcgs**2 + zmincgs**2)**(3.0/2.0)
        else:
            #print 'max az set by z = rho / np.sqrt(2)'
            azmax = -2.0 * GM / (np.sqrt(27.0) * rcgs**2)
            #azmax = -2.0 * GM3sig / (np.sqrt(27.0) * rcgs**2)
            
        azmax *= cc.sec_in_yr / 1.0e5
        # Compare to the maximum allowed acceleration at this distance
        accMaxRadial = cc.sec_in_yr * -GM / (rcgs**2 * 1.0e5)
        #accMaxRadial = cc.sec_in_yr * -GM3sig / (rcgs**2 * 1.0e5)

        # There is also a lower limit to the acceleration if we assume
        # the star is bound. It's velocity cannot exceed the escape velocity
        # This corresponds to the largest allowed line of sight distance
        vz = star.vz
        if (vz == None) | (star.vz == ''):
            vz = 0.0
            print 'No radial velocity for %s' % name
        vtotcgs = np.sqrt(vx**2 + vy**2 + vz**2) * 1.0e5
        abound_cgs = -rcgs * vtotcgs**6 / (8.0 * GM**2)
        abound = abound_cgs * 1000.0 * cc.sec_in_yr**2
        abound /= (cc.cm_in_au * dist)      # mas/yr^2
	abound = abound / 1000.0 * cc.asy_to_kms  # km/s/yr

        # Save off abound
        abound_all = np.concatenate([abound_all, [abound]])
        
        if ftest == True:
            if name in nameF:
                a_or_v = 'Acc'
            else:
                a_or_v = 'Vel'

            fmt = '%2i  %15s  %4.1f  %4i  %6.2f vs. %6.2f      %5.3f   '
            fmt += '%6.2f %6.2f   %6.2f  %8.2e  %5i  %11s\n'
            print fmt % \
                  (ii+1, name, mag, cnt, accLimRadial, accMaxRadial,
                   rLower, ar, are, azmax, abound, detection[ii], a_or_v)
            out.write(fmt % (ii+1, name, mag, cnt, accLimRadial, accMaxRadial,
                   rLower, ar, are, azmax, abound, detection[ii], a_or_v))
        else:
            fmt = '%2i  %15s  %4.1f  %4i  %6.2f vs. %6.2f      '
            fmt += '%5.3f   %6.2f %6.2f   %6.2f  %8.2e  %5i\n' 
            print fmt % \
                  (ii+1, name, mag, cnt, accLimRadial, accMaxRadial,
                   rLower, ar, are, azmax, abound, detection[ii])
            out.write(fmt %
                      (ii+1, name, mag, cnt, accLimRadial, accMaxRadial,
                       rLower, ar, are, azmax, abound, detection[ii]))

        ii += 1

    #if ftest == True:
    #    _acc.close()
    #_upper.close()

    lgd1 = 'Upper Limit'
    lgd2 = 'Detection'
    import matplotlib.font_manager
    prop = matplotlib.font_manager.FontProperties(size=12)
    lgd = py.legend((uppr[0], det[0]), (lgd1,lgd2), fancybox=True,
                    numpoints=1,prop=prop)
    py.title('Acceleration Detections \& Upper Limits (%i sigma)' % sig_acc)
    py.axis([0,0.25,-3,40])
    #py.axis([0,0.14,-10,30])
    #py.show()
    py.savefig(outdir+'r2d_accel_detectUpLim_yng%s.png' % suffix) 
    py.close()

    # Which are significantly different from zero?
    # (Without taking into account theoretical curve of amax for projected R)
    ar_sig = np.where(((ar_all + (sig_acc*are_all)) < 0.0) & (ar_all < 0.0))[0]
    py.clf()
    py.figure(figsize=(6,6))
    for aa in ar_sig:
        py.errorbar(rLower_all[aa], -ar_all[aa], yerr=(sig_acc*are_all[aa]), fmt='b.')
        py.plot(rLower_all[aa], -abound_all[aa], 'r.')
    py.plot(r_pc, -a_km_s_yr, 'k--')
    py.plot([0,0.25],[0,0],'k--')
    py.plot(r_pc, -(a_km_s_yr+sig_acc*sig_a_mks), 'k:')
    py.plot(r_pc, -(a_km_s_yr-sig_acc*sig_a_mks), 'k:')
    py.title('Acceleration Detections \& Lower Limits (Bound orbit; %i sigma)' % sig_acc,fontsize=14)
    py.axis([0.03,0.1,-1,15])
    py.xlabel(r'{\bf Projected Radius (pc)}')
    py.ylabel(r'{\bf $|$a$_\rho|$ (km/s/yr)}')
    py.savefig(outdir+'r2d_accel_detectBound_yng%s.png' % suffix) 
    py.close()

    # Make a similar plot but for all stars, showing abound
    py.clf()
    py.figure(figsize=(6,6))
    py.plot(r_pc, -a_km_s_yr, 'k--')
    py.errorbar(rLower_all, -ar_all, yerr=(sig_acc*are_all), fmt='b.')
    py.plot(rLower_all, -abound_all, 'r.')
    py.plot([0,0.25],[0,0],'k--')
    py.plot(r_pc, -(a_km_s_yr+sig_acc*sig_a_mks), 'k:')
    py.plot(r_pc, -(a_km_s_yr-sig_acc*sig_a_mks), 'k:')
    py.title('Acceleration Measurements \& Lower Limits (Bound orbit; %i sigma)' % sig_acc,fontsize=14)
    py.axis([0.00,0.25,-10,30])
    py.xlabel(r'{\bf Projected Radius (pc)}')
    py.ylabel(r'{\bf $|$a$_\rho|$ (km/s/yr)}')
    py.savefig(outdir+'r2d_accel_measuredBound_yng%s.png' % suffix) 
    py.close()
    usetexFalse()

    # Plot acceleration significance vs. number of epochs
    _cnt30 = np.where(cnt_all > 30)[0]
    cnt30 = len(_cnt30)
    _cnt30_rv = np.where((np.array([rr is not None for rr in rv_ref]) == True) & (cnt_all > 30))[0]
    cnt30_rv = len(_cnt30_rv)
    print 'Number of stars in at least 30 epochs: %i' % cnt30
    print '  Number of these with RVs: %i' % cnt30_rv
    print '  Stars in 30 epochs but without RVs:'
    for ii in np.setdiff1d(_cnt30, _cnt30_rv):
        print '    ' + names[ii]
    py.clf()
    py.figure(figsize=(6,6))
    py.plot(cnt_all, ar_all/are_all, 'r.', label='a_rad')
    py.plot(cnt_all, at_all/ate_all, 'b.', label='a_tan')
    py.plot([cnt_all.min(),cnt_all.max()], [sig_acc, sig_acc], 'k--')
    py.plot([cnt_all.min()-1,cnt_all.max()+1], [-sig_acc, -sig_acc], 'k--')
    py.plot([30,30],[(ar_all/are_all).min()-3,(ar_all/are_all).max()+1], 'k--')
    py.text(30.2,-10,'# Epochs Cut',fontsize=10)
    py.legend(numpoints=1,loc=3,fancybox=True,prop=prop)
    py.xlabel('Number of Epochs')
    py.ylabel('Acceleration Significance (sigma)')
    py.savefig(outdir + 'accel_signif_vs_numEpochs%s.png' % suffix)
    py.close()

    usetexTrue()
    # Plot acceleration vs. proper motion
    py.figure(figsize=(8,4))
    py.subplots_adjust(left=0.1,right=0.95,top=0.95,bottom=0.15,
                       wspace=0.3, hspace=0.3)
    py.clf()
    # identify significant vels and accs (5 sigma, and 3 sigma)
    asig5 = np.where(np.abs(ar_all / are_all) > 5.0)[0]    
    vsig5 = np.where(np.abs(pm_all / pme_all) > 5.0)[0]    
    asig3 = np.where(np.abs(ar_all / are_all) > 3.0)[0]    
    vsig3 = np.where(np.abs(pm_all / pme_all) > 3.0)[0]
    py.subplot(1,2,1)
    py.errorbar(pm_all, ar_all, fmt='k.', xerr=pme_all,
                yerr=are_all)
    py.errorbar(pm_all[asig3], ar_all[asig3], fmt='b.',
                xerr=pme_all[asig3], yerr=are_all[asig3],
                label=r'3$\sigma$ Accel')
    py.errorbar(pm_all[asig5], ar_all[asig5], fmt='r.',
                xerr=pme_all[asig5], yerr=are_all[asig5],
                label=r'5$\sigma$ Accel')
    py.plot([0.0, pm_all.max()+10], [0, 0], 'k--')
    py.legend(numpoints=1,loc=2,fancybox=True,prop=prop)
    py.xlabel('Proper Motions (km/s)')
    py.ylabel('Acceleration (km/s/yr)')
    # Plot acceleration in sigma vs. proper motion in km/s
    py.subplot(1,2,2)
    py.plot(pm_all, ar_all/are_all, 'k.')
    py.plot(pm_all[asig3], ar_all[asig3]/are_all[asig3], 'b.')
    py.plot(pm_all[asig5], ar_all[asig5]/are_all[asig5], 'r.')
    py.plot([0.0, pm_all.max()+10], [0, 0], 'k--')
    py.xlabel('Proper Motions (km/s)')
    py.ylabel('Acceleration (sigma)')
    py.savefig(outdir + 'accel_vs_proper_motion%s.png' % suffix)
    py.close()

    # put the accel errors in mas/yr^2 for plotting
    are_all = are_all / cc.asy_to_kms*1.e3
    ate_all = ate_all / cc.asy_to_kms*1.e3
    py.clf()
    py.figure(figsize=(6,6))
    binsIn = np.arange(0,1,0.01)
    py.hist(are_all, bins=binsIn, color='r',histtype='step',label='Radial')
    py.hist(ate_all, bins=binsIn, color='b',histtype='step',label='Tangential')
    py.legend(numpoints=1,fancybox=True)
    py.xlabel(r'Acceleration Error (mas/yr$^2$)')
    py.ylabel('N')
    py.axis([0, 1.0, 0, 20])
    py.savefig(outdir + 'hist_accel_error%s.png' % suffix)
    py.close()
    usetexFalse()
    print 'Average radial acceleration error: %4.3f mas/yr^2' % are_all.mean()
    print 'Median radial acceleration error: %4.3f mas/yr^2' % np.median(are_all)
    print 'Average tangential acceleration error: %4.3f mas/yr^2' % ate_all.mean()
    print 'Median tangential acceleration error: %4.3f mas/yr^2' % np.median(ate_all)

    # print out some info
    det = np.where(detection == 2)[0] # significant and below curve
    upLim = np.where(detection == 1)[0] # upper limit is below curve
    nondet = np.where(detection == 0)[0] # upper limit is above curve
    print ''
    print '%i acceleration detections' % len(det)
    print '%i acceleration %1s sigma upper limits' % (len(upLim), sig_acc)
    print '%i acceleration non-detections' % len(nondet)

    out.write('%i acceleration detections\n' % len(det))
    out.write('%i acceleration %1s sigma upper limits\n' % (len(upLim), sig_acc))
    out.write('%i acceleration non-detections\n' % len(nondet))
    out.close()

    print 'Average positional error (X, Y): %4.3f, %4.3f mas' % \
          (xe_all.mean(), ye_all.mean())
    print 'Median positional error (X, Y): %4.3f, %4.3f mas' % \
          (np.median(xe_all), np.median(ye_all))


def ftest_radial(pvalue=4.0):
    """
    Runs the F test for accelerations using the radial and tangential
    fits to the stars' motions.
    """
    
    # Load up young stars
    yng = young.loadYoungStars(root+alnDir,fit='polyfit_radial/fit',
                               points=points,radiusCut=0.8,
                               withRVonly=True,skipStar=['S5-237']) 

    # for the radial polyfit, X is the perpendicular coordinate
    # and Y is the parallel (radial) coordinate.
    chi2vPerp = yng.getArray('fitpXv.chi2')
    chi2vPar = yng.getArray('fitpYv.chi2')
    chi2aPerp = yng.getArray('fitpXa.chi2')
    chi2aPar = yng.getArray('fitpYa.chi2')
    names = yng.getArray('name')

    cnt_all = []
    for name in names:
        # How many epochs was each star detected in?
        pts = asciidata.open('%s%s%s%s.points' % (root, alnDir, points, name))
        ep = pts[0].tonumpy()
        cnt = len(ep)
        cnt_all = np.concatenate([cnt_all, [cnt]])

    dof = cnt_all - 3.0

    # Run the F test
    signif = scipy.special.erfc(pvalue/np.sqrt(2.0))

    perpFvalue = ((chi2vPerp - chi2aPerp)/1.0) / (chi2aPerp/dof)
    parFvalue = ((chi2vPar - chi2aPar)/1.0) / (chi2aPar/dof)

    perpFprob = stats.f.sf(perpFvalue, 1, dof)
    parFprob = stats.f.sf(parFvalue, 1, dof)

    # Round to nearest decimal
    perpFprob = np.around(perpFprob,decimals=4)
    parFprob = np.around(parFprob,decimals=4)

    acc = np.where((perpFprob < signif) | (parFprob < signif))[0]
    accPerp = np.where(perpFprob < signif)[0]
    accPar = np.where(parFprob < signif)[0]
    
    pdb.set_trace()

def orbitAnalysisAll(mosaic=False, makePlot=False, pvalue=4.0):
    # Load names of young stars 
    if mosaic == True:
        # Load up mosaic data as well; select only stars at r>4, since
        # we don't want to add any info from mosaics if we have it in
        # the central 10" already
        yng1 = young.loadYoungStars(root+alnDir,fit=poly,points=points,
                                    withRVonly=True,skipStar=['S5-237'],silent=True)
        yng2 = young.loadYoungStars(mscDir,fit=polyM,points=pointsM,
                                    mosaic=True, withRVonly=True,silent=True)
        cntrlNames = yng1.getArray('name')
        mscNames = yng2.getArray('name')
        # Merge this object with object from central 10" analysis
        yng = merge(yng1, yng2)
    else:
        yng = young.loadYoungStars(root+alnDir,fit=poly,points=points,
                                   withRVonly=True,silent=True)

    yngNames = yng.getArray('name')

    names = yngNames
    names.sort()

    # Compute F test to see if acceleration fits are warranted
    # pvalue = Significance threshold for F test
    # Stars that should have acceleration fits are returned:
    nameF, xFp, yFp = syYoung.velocity_vs_accel(alnDir=alnDir,pvalue=pvalue,
                                                verbose=False)

    for name in names:
        # Does this star have acceleration info we can use?
        acc = False
        #if name in nameF:
        if ((name in nameF) or (name == 'S0-14')): # S0-14 is RIGHT on border of passing F test
            acc = True

        mscStar = False
        if (name in mscNames) & (name not in cntrlNames):
            mscStar = True

        if (os.path.exists(root + alnDir + 'analyticOrbits/' + name + '.results.dat') == True):
            print 'Skipping ', name + '.results.dat'
            continue

        try:
            orbitAnalysis(name, acc=acc, makePlot=makePlot, mosaic=mosaic)
        except Exception, inst:
            print 'ORBIT_ANALYSIS: Failed to find bound solutions for: ', name
            print inst

def orbitAnalysis(star, acc=False, nodata=False, makePlot=False, mosaic=False):
    """
    NOT FUNCTIONAL WITH MOSAIC YET! (in analyticOrbits.py)

    Plot analytical orbit solutions for a full range of
    line of sight distances (z) for the star specified.
    Produces an output postscript file with the name
    <star>_orbit.png.

    @param star: The name of the star to plot
    @type star: string
    @param acc: Use acceleration info from polyfit for this star
    @type acc: boolean (default = False)
    @kwparam nodata: Set to True for orbit analysis of stars
    that aren't in our data (info taken from Paumard 2006 only).
    @type nodata: boolean
    @kwparam makePlot: Make a plot of all the orbital
    parameters as a function of line-of-sight Z distance. The
    plot will be stored in plots/<star>_orbit.png.
    @type makePlot: boolean (default = False)
    """
    cc = objects.Constants()
    #GM = cc.G * cc.msun * cc.mass
    GM = cc.G * cc.msun * mass

    print '*****'
    print 'ORBIT ANALYSIS: ', star
    if (nodata):
        print 'No data for %s' % star
        ## Loading ALL
        #print 'Loading ALL young stars'
        #yng = young.loadAllYoungStars(root+alnDir)
    else:
        # Load names of young stars 
        if mosaic == True:
            # Load up mosaic data as well; select only stars at r>4, since
            # we don't want to add any info from mosaics if we have it in
            # the central 10" already
            yng1 = young.loadYoungStars(root+alnDir,fit=poly,points=points,
                                        withRVonly=True,silent=True,skipStar=['S5-237']) 
            yng2 = young.loadYoungStars(mscDir,fit=polyM,points=pointsM,
                                        mosaic=True, withRVonly=True,silent=True)
            cntrlNames = yng1.getArray('name')
            mscNames = yng2.getArray('name')
            # Merge this object with object from central 10" analysis
            yng = merge(yng1, yng2)
        else:
            yng = young.loadYoungStars(root+alnDir,fit=poly,points=points,
                                       withRVonly=True,silent=True)

    name = yng.getArray('name')

    ii = name.index(star)
    theStar = yng.stars[ii]

    mscStar = False
    if (name in mscNames) & (name not in cntrlNames):
        mscStar = True

    if acc == True:
        theStar.accel = True
        print 'F test suggests polyfit acceleration fit should be used for %s' % star
    else:
        theStar.accel = False

    if theStar.vz != None:
        oo = aorb.StarOrbits(theStar, nodata=nodata, mscStar=mscStar)
        oo.calc()

        pickleFile = root + alnDir + 'analyticOrbits/' + star + '.results.dat'
        oo.saveToFile(pickleFile)


        ##########
        #
        # Plotting
        #
        ##########
        py.figure(2, figsize=(10,7.5))
    
        z_arc = oo.z# / (dist * cc.cm_in_au)
    
        xlab = 'z (arcsec)'
        
        py.clf()
        py.subplots_adjust(left=0.06, right=0.98, top=0.95,
                        wspace=0.34, hspace=0.3)
    
        # Store some font sizes for labels and ticks
        labelFont = {'fontsize': 12}
        tickSize = 8
    else:
        makePlot = False
        print 'No radial velocity for %s' % star

    # Define a temporary function to make filled areas
    # for the allowed range on a parameter from our orbit fitting.
    def shadeBox(ylo, yhi):
        # Makes box with certain Y range. Covers entire X range
        rng = py.axis()
        boxX = [rng[0], rng[1], rng[1], rng[0]]
        boxY = [ylo, ylo, yhi, yhi]
        
        py.fill(boxX, boxY, alpha=0.1)

    def axisLabel(xtext, ytext, yrange=None):
        # Rescales fonts for ticks and labels
        thePlot = py.gca()
        py.setp( thePlot.get_xticklabels(), fontsize=tickSize )
        py.setp( thePlot.get_yticklabels(), fontsize=tickSize )

        # Add axis labels
        py.xlabel(xtext, labelFont)
        py.ylabel(ytext, labelFont)

        # Optional re-scale axes
        if (yrange != None):
            rng = py.axis()
            py.axis([rng[0], rng[1], yrange[0], yrange[1]])

    if (makePlot == True):
        ##########
        # First plot orbital parameters
        ##########
        # Plot Eccentricity
        py.subplot(3, 4, 1)
        py.errorbar(z_arc, oo.e, yerr=oo.ee)
        py.title(star)
        #if not nodata: shadeBox(theStar.efit.ecc_lo, theStar.efit.ecc_hi)
        shadeBox(oo.e_lo, oo.e_hi)
        axisLabel(xlab, 'Eccentricity', yrange=[0, 1.0])

        # Plot Inclination
        py.subplot(3, 4, 2)
        py.errorbar(z_arc, oo.i, yerr=oo.ie)
        #if not nodata: shadeBox(theStar.efit.incl_lo, theStar.efit.incl_hi)
        shadeBox(oo.i_lo, oo.i_hi)
        axisLabel(xlab, 'Inclination (deg)', yrange=[0, 180.0])

        # Plot PA to Ascending Node
        py.subplot(3, 4, 5)
        py.errorbar(z_arc, oo.o, yerr=oo.oe)
        #if not nodata: shadeBox(theStar.efit.bigOmega_lo, theStar.efit.bigOmega_hi)
        shadeBox(oo.o_lo, oo.o_hi)
        axisLabel(xlab, 'PA to Asc. Node (deg)', yrange=[0, 360])

        # Plot omega
        py.subplot(3, 4, 6)
        py.errorbar(z_arc, oo.w, yerr=oo.we)
        #if not nodata: shadeBox(theStar.efit.omega_lo, theStar.efit.omega_hi)
        shadeBox(oo.w_lo, oo.w_hi)
        axisLabel(xlab, 'Angle to Periapse (deg)', yrange=[0, 360])

        # Plot period
        py.subplot(3, 4, 9)
        py.errorbar(z_arc, np.log10(oo.p), yerr=np.log10(oo.pe))
        #if not nodata: shadeBox(theStar.efit.period, theStar.efit.period)
        shadeBox(np.log10(oo.p_lo), np.log10(oo.p_hi))
        axisLabel(xlab, 'Log [Period (yrs)]', yrange=[-1, 8])

        # Plot t0
        py.subplot(3, 4, 10)
        py.errorbar(z_arc, oo.t0, yerr=oo.t0e)
        #if not nodata: shadeBox(theStar.efit.t0_lo, theStar.efit.t0_hi)
        shadeBox(oo.t0_lo, oo.t0_hi)
        axisLabel(xlab, 'Time of Periapse (yrs)', yrange=[0, 4000.0])

        ##########
        # Plot other stuff
        ##########

        # Plot acceleration
        py.subplot(3, 4, 3)
        py.plot(z_arc, oo.a2d_mag)
        rng = py.axis()
        if not nodata: 
            boxX = [rng[0], rng[1], rng[1], rng[0]]
            conv = 1.0e5 * cc.sec_in_yr * 1000.0 / (cc.cm_in_au * dist)
            #boxY = [0, 0, theStar.efit.ap_hi * conv, theStar.efit.ap_hi * conv]
            boxY = [0, 0, oo.alimR, oo.alimR]
            py.fill(boxX, boxY, alpha=0.1)
        axisLabel(xlab, 'Acceleration (mas/yr^2)')


        # Escape Velocity
        py.subplot(3, 4, 4)
        py.errorbar(z_arc, oo.v_mag/oo.vesc, yerr=oo.ve_mag/oo.vesc)
        axisLabel(xlab, 'V/v_esc')

        # h/rv
        py.subplot(3, 4, 7)
        py.errorbar(z_arc, oo.h_rv, yerr=oo.h_rve)
        axisLabel(xlab, '|h|/|r||v| = sin(theta)', yrange=[0, 1])

        # The next 3 figures only exist for stars that are in our dataset.
        if not nodata:
            # rms
            py.subplot(3, 4, 8)
            py.plot(z_arc, oo.rms * 1000.0)
            rng = py.axis()
            py.plot([rng[0], rng[1]],
                 [oo.avg_err*1000.0, oo.avg_err*1000.0], 'k--')
            axisLabel(xlab, 'RMS (mas)')

            # chi2
            py.subplot(3, 4, 11)
            py.plot(z_arc, oo.chi2)
            rng = py.axis()
            chi2min = oo.chi2.min()
            chi2max = chi2min + 3.0
            py.plot([rng[0], rng[1]], [chi2max, chi2max], 'k--')
            axisLabel(xlab, 'Chi^2')

            # Reduced
            py.subplot(3, 4, 12)
            py.plot(z_arc, oo.chi2red)
            axisLabel(xlab, 'Reduced Chi^2')
        else:
            # Plot phase
            py.subplot(3, 4, 8)
            py.errorbar(z_arc, oo.ph, yerr=oo.phe)
            axisLabel(xlab, 'Phase', yrange=[0, 1])

        py.savefig(root + alnDir + 'plots/' + star+'_orbit.png')
        py.close()
    

def orbitAnalysisPDF(pdfroot, pdftrials, doMassRo=None, mExtended=None,
                     pvalue=4.0, chi2Cut=None, mosaic=True, uniformAcc=False):
    """
    Set uniformAcc=True to NOT use any acceleration measurements, and just
    use a uniform acceleration prior for all stars. This will help us see the
    effects of the handful of stars with acceleration measurements.

    chi2Cut  -- Chi2 value above which mosaic stars will be cut. The
    		X chi2 values and Y chi2 values for velocity are checked
                separately. If either are > chi2Cut, the stars will be excluded
                from the analysis.

    Dependencies:
    		accelLimit() -- creates accelerating_sources.dat and
                	        accel_upperLimit_sources.dat, which we need
                                to read in to determine how to use accelerations
    """

    # Load names of young stars 
    if mosaic == True:
        # Load up mosaic data as well; select only stars at r>4, since
        # we don't want to add any info from mosaics if we have it in
        # the central 10" already
        yng1 = young.loadYoungStars(root+alnDir,fit=poly,points=points,
                                    withRVonly=True,silent=True,skipStar=['S5-237']) 
        yng2 = young.loadYoungStars(mscDir,fit=polyM,points=pointsM,
                                    mosaic=True, withRVonly=True,silent=True)
        #yng2 = young.loadYoungStars(mscDir,align='align/align_abs_t',fit=polyM,points=pointsM,
        #                            mosaic=True, withRVonly=True,silent=True)
        cntrlNames = yng1.getArray('name')
        mscNames = yng2.getArray('name')
        # Merge this object with object from central 10" analysis
        yng = merge(yng1, yng2)
    else:
        yng = young.loadYoungStars(root+alnDir,fit=poly,points=points,
                                   withRVonly=True,silent=True)

    names = yng.getArray('name') 
    names.sort()
    yngNames = yng.getArray('name')

    if chi2Cut != None:
        xchi2M = yng2.getArray('fitpXv.chi2') # mosaic only
        ychi2M = yng2.getArray('fitpYv.chi2') # mosaic only

        # Throw out the mosaic stars with chi2 values > chi2Cut
        bad = np.where((xchi2M > chi2Cut) | (ychi2M > chi2Cut))[0]
        bad = [np.int(bb) for bb in bad]
        badNames = [mscNames[bb] for bb in bad]

    pdfroot = root + alnDir + pdfroot

    # Compute F test to see if acceleration fits are warranted
    # pvalue = Significance threshold for F test
    # Stars that should have acceleration fits are returned:
    #nameF, xFp, yFp = syYoung.velocity_vs_accel(alnDir=alnDir,pvalue=pvalue,
    #                                            verbose=False)

    # Read in accelerating_sources.dat, which lists accelerating stars that
    # passed the F test. We will use the accel measurements for these stars
    _acc = asciidata.open(root + alnDir + 'tables/accelerating_sources.dat')
    accels = _acc[0].tonumpy()
    acc = [aa.strip() for aa in accels]

    # Read in accel_upperLimit_sources.dat, which lists the stars with
    # upper limits below the a_max curve. We will sample from uniform accel
    # prior, between a_min and the upper limit.
    _upp = asciidata.open(root + alnDir + 'tables/accel_upperLimit_sources.dat')
    upper = _upp[0].tonumpy()
    upp = [uu.strip() for uu in upper]

    # For all other stars, we'll sample from uniform accel prior, between
    # a_min and a_max

    py.clf()

    # temp
    #skip = ['S1-1', 'S0-14', 'S0-15', 'S1-12', 'S1-14', 'S1-18', 'S1-19', 'S1-2', 'S1-21','S1-22','S1-24']
    # end temp

    cnt_all = []
    for name in names:
        # temp
        #if (name == 'S1-1') | (name == 'S0-14') | (name == 'S0-15'):
        #if name in skip:
        #    continue
        # end temp
        mscStar = False
        
        # How many epochs was each star detected in?
        # First check if this star is from the mosaic or central 10:
        if mosaic == True:
            if (name in mscNames) & (name not in cntrlNames):
                pts = asciidata.open('%s%s%s.points' % (mscDir, pointsM, name))
                mscStar = True

                # Do we need to cut based on chi2 values?
                if chi2Cut != None:
                    if name in badNames:
                        print 'Excluding %s based on high chi2 values (> %3.1f)' % \
                              (name, chi2Cut)
                        continue
            else:
                pts = asciidata.open('%s%s%s%s.points' % (root, alnDir, points, name))
        else:
            pts = asciidata.open('%s%s%s%s.points' % (root, alnDir, points, name))

        ep = pts[0].tonumpy()
        cnt = len(ep)
        cnt_all = np.concatenate([cnt_all, [cnt]])

        # Get the star object
        idx = yngNames.index(name)
        yngStar = yng.stars[idx]

        if yngStar.vz != None:
            print
            print 'MC ON: %s, using RV from %s' % (name, yngStar.rv_ref)

            # Does this star have acceleration info we can use?
            # Also require a minimum of 30 epochs to use acceleration info
            #if ((cnt >= 30) and (name in nameF)):
            #if ((cnt >= 30) and (name in nameF) | (name == 'S0-14')):
            #    zfrom = 'acc'
            #else:
            #    zfrom = 'uni_acc'

            # Passes F test for accel, has significant accel above 0 and below a_max
            # must also be in > 30 epochs
            if (name in acc) and (cnt > 30):
                zfrom = 'acc'
                useAcc = True # Use accel fit for position, velocity, and accels
            elif (name in upp): # and (name != 'irs16SW-E'):
                zfrom = 'uni_acc_limit'
                useAcc = True # Use accel fit for position, velocity, and accels
            elif name == 'S1-8':
                zfrom = 'acc'
                useAcc = True 
            else:  # this will always include mosaic stars
                zfrom = 'uni_acc'
                useAcc = False # Use velocity fit for position, velocity, and accels

            print '%s: accel from %s' % (name, zfrom)

            # Force uniform accel prior? (for testing purposes)
            if uniformAcc == True:
                zfrom = 'uni_acc'

            #if (os.path.exists(pdfroot + yngStar.name + '.mc.dat') == True):
            #    print 'Skipping ', pdfroot + yngStar.name + '.mc.dat'
            #    continue

            # Run a Monte Carlo to determine the probability density function (PDF)
            # for this star. Save it off to a file
            mcfile = '%s%s.mc.dat' % (pdfroot, name)
            mc = aorb.StarOrbitsMC(yngStar, ntrials=pdftrials,
                                   outroot=pdfroot, mscStar=mscStar,
                                   useAcc=useAcc)
            mc.run(mcMassRo=doMassRo, zfrom=zfrom, mExtended=mExtended)
            mc.makePdfHealpix(makeplot=True)
            mc.saveToFile(mcfile)
    
            #plotOrbitPdf(name, pdfroot, save=True)
        else:
            print 'No radial velocity for %s. Skipping.' % name

    print 'Total number of positional measurements for all young stars: %5d' % cnt_all.sum()

def plotOrbitPdf(star, pdfdir, save=False):
    cc = objects.Constants()

    # File contains analytic orbit solutions without acceleration limits
    #pickleFile = root + alnDir + 'analyticOrbits/' + star + '.results.dat'
    #oo = pickle.load(open(pickleFile))

    # File contains analytic orbit solutions with acceleration limits (MC)
    pdffile = '%s%s.mc.dat' % (pdfdir, star)
    print os.access(pdffile, os.R_OK)
    while os.access(pdffile, os.R_OK) == False:
        print 'Cannot access file %s' % pdffile
    #if os.access(pdffile, os.R_OK) == True:
    #print 'Test 1'
    pdf = pickle.load(open(pdffile))

    #print 'Test 2'
    ##########
    #
    # Plotting
    #
    ##########
    py.figure(2, figsize=(10,8))
    #py.figure(2, figsize=(8,4.5))

    py.clf()
    py.subplots_adjust(left=0.12, right=0.98, top=0.92,
                    wspace=0.36, hspace=0.35)
    usetexTrue()

    # Store some font sizes for labels and ticks
    labelFont = {'fontsize': 14, 'fontweight': 'medium'}

    tickSize = 10

    def dispHist(xx, yy, xxtheory, yytheory):
        fmt = 'k.'
        fmt2 = 'r--'
        pntsize = 2
        #cmap = py.cm.YlGnBu
        #cmap = py.cm.summer_r
        cmap = py.cm.hot_r

        # Make 2D histogram
        (probDist, b1, b2) = h2d.histogram2d(xx, yy, bins=(50, 50))

        # Need to convert the 2d histogram into floats
        probDist = np.array(probDist, dtype=float)
        
        # Determine contour levels
        # Flatten and reverse sort our prob. distribution
        sid0 = probDist.flatten().argsort()
        sid = sid0[::-1]
        pixSort = probDist.flatten()[sid]
        
        # Make a cumulative distribution function starting from the
        # highest pixel value. This way we can find the level above
        # which 68% of the trials will fall.
        cdf = np.cumsum(pixSort)
        
        # Determine point at which we reach 68% level
        #percents = np.array([0.6827, 0.9545, 0.9973]) * len(xx)
        percents = np.array([0.6827, 0.9545]) * len(xx)
        levels = np.zeros(len(percents), dtype=float)
        for ii in range(len(levels)):
            # Get the index of the pixel at which the CDF
            # reaches this percentage (the first one found)
            idx = (np.where(cdf < percents[ii]))[0]

            # Now get the level of that pixel
            levels[ii] = pixSort[idx[-1]]
        #print levels
            
        # Mask out the parts where we don't have data.
        foo = np.where(probDist == 0)
        probDist = np.log10(probDist)
        probDist[foo] = 0.0
        levels = np.log10(levels)

        py.imshow(probDist, extent=[b1[0], b1[-1], b2[0], b2[-1]],
               cmap=cmap, origin='lower', aspect='auto',
               interpolation='bilinear')
        py.contour(probDist, levels, origin=None, colors='black',
                extent=[b1[0], b1[-1], b2[0], b2[-1]])

        #py.plot(xxtheory, yytheory, fmt2)
        #py.plot(xx, yy, fmt, markersize=pntsize)

    def axisLabel(xtext, ytext, yrange=None):
        # Rescales fonts for ticks and labels
        thePlot = py.gca()
        rng = py.axis()

        # Increment for ticks on the X axis
        tmp = np.abs(float(rng[1]) - float(rng[0])) / 5.0
        xinc = __builtin__.round(tmp, 1)

        if (xinc == 0):
            xinc = 0.05
            #xinc = 0.03
        
        thePlot.get_xaxis().set_major_locator(py.MultipleLocator(xinc))
        py.setp( thePlot.get_xticklabels(), fontsize=tickSize )
        py.setp( thePlot.get_yticklabels(), fontsize=tickSize )

        # Add axis labels
        py.xlabel(xtext, labelFont)
        py.ylabel(ytext, labelFont)

        # Optional re-scale axes
        if (yrange != None):
            py.axis([rng[0], rng[1], yrange[0], yrange[1]])

    #xlab = r'a$_\rho$ (mas/yr$^2$)'
    xlab = r'{\bf z (pc)}'
    xdat = pdf.z * dist / cc.au_in_pc
    #xdat2 = oo.z * dist / cc.au_in_pc
    xdat2 = None

    # Plot Eccentricity
    py.subplot(2, 3, 1)
    #dispHist(xdat, pdf.e, xdat2, oo.e)
    dispHist(xdat, pdf.e, xdat2, None)
    axisLabel(xlab, r'${\bf {\it e}}$', yrange=[0, 1.0])

    # Plot Inclination
    py.subplot(2, 3, 2)
    #dispHist(xdat, pdf.i, xdat2, oo.i)
    dispHist(xdat, pdf.i, xdat2, None)
    axisLabel(xlab, r'${\bf {\it i}}$ {\bf (deg)}', yrange=[0, 180.0])

    if ('irs' in star):
        starLabel = '{\\bf IRS %s}' % (star[3:])
    else:
        starLabel = '{\\bf %s}' % (star)
    py.title(starLabel, labelFont)

    # Plot PA to Ascending Node
    py.subplot(2, 3, 3)
    idx = (np.where(pdf.o < 0))[0]
    pdf.o[idx] += 360.0
    #dispHist(xdat, pdf.o, xdat2, oo.o)
    dispHist(xdat, pdf.o, xdat2, None)
    axisLabel(xlab, r'${\bf \Omega}$ {\bf (deg)}', yrange=[0, 360])

    # Plot omega
    py.subplot(2, 3, 4)
    #dispHist(xdat, pdf.w, xdat2, oo.w)
    dispHist(xdat, pdf.w, xdat2, None)
    axisLabel(xlab, r'${\bf \omega}$ {\bf (deg)}', yrange=[0, 360])

    # Plot period
    py.subplot(2, 3, 5)
    #dispHist(xdat, log10(pdf.p), xdat2, log10(oo.p))
    dispHist(xdat, np.log10(pdf.p), xdat2, None)
    axisLabel(xlab, r'${\bf \log{{\it P}}}$ {\bf (yrs)}', yrange=[-1, 8])

    # Plot t0
    py.subplot(2, 3, 6)
    #dispHist(xdat, pdf.t0/1000.0, xdat2, oo.t0/1000.0)
    dispHist(xdat, pdf.t0/1000.0, xdat2, None)
    axisLabel(xlab, r'{\bf ${\bf {\it t_0}}$ (${\bf \times}$10${\bf ^3}$ yrs)}',
              yrange=[0, 4.0])

    if (save):
        py.savefig(pdfdir + 'pdfparams/sythesis_pdf_params_%s.png' % star, dpi=150)
        py.close()
    else:
        py.show()

    usetexFalse()


def orbitAnalysisPDFNoAccel(pdfroot, pdftrials, doMassRo=None, zfrom='all', mosaic=False):
    cc = objects.Constants()
    # Load names of young stars 
    if mosaic == True:
        # Load up mosaic data as well; select only stars at r>4, since
        # we don't want to add any info from mosaics if we have it in
        # the central 10" already
        yng1 = young.loadYoungStars(root+alnDir,fit=poly,points=points,
                                    withRVonly=True,silent=True,skipStar=['S5-237']) 
        yng2 = young.loadYoungStars(mscDir,fit=polyM,points=pointsM,
                                    mosaic=True, withRVonly=True,silent=True)
        # Merge this object with object from central 10" analysis
        yng = merge(yng1, yng2)
    else:
        yng = young.loadYoungStars(root+alnDir,fit=poly,points=points,
                                   withRVonly=True,silent=True)

    
    names = yng.getArray('name')

    for ii in range(len(names)):
        star = yng.stars[ii]

        if star.vz != None:
            #if (os.path.exists(pdfroot + star.name + '.mc.dat') == True):
            #    print 'Skipping ', pdfroot + star.name + '.mc.dat'
            #    continue

            fitxv = star.getFitXv()
            fityv = star.getFitYv()
    
            # Need to convert positions from arcsec +x = East to +x = West
            x = -1.0 * fitxv.p
            y = fityv.p
            xerr = fitxv.perr
            yerr = fityv.perr
            vx = -1.0 * fitxv.v
            vy = fityv.v
            vxerr = fitxv.verr
            vyerr = fityv.verr
            
            star.setFitpXa(fitxv.t0, x, xerr, vx, vxerr, 0.0, 0.0)
            star.setFitpYa(fityv.t0, y, yerr, vy, vyerr, 0.0, 0.0)
            
            print 'MC ON: %s  zfrom = %s' % (star.name, zfrom)
    
            # Run a Monte Carlo to determine the probability density function (PDF)
            # for this star. Save it off to a file
            mcfile = '%s%s.mc.dat' % (pdfroot, star.name)
            mc = aorb.StarOrbitsMC(star, ntrials=pdftrials,
                                             outroot=pdfroot)
            mc.run(mcMassRo=doMassRo, zfrom=zfrom)
            mc.makePdfHealpix(makeplot=True)
            mc.saveToFile(mcfile)
    
            plotOrbitPdf(star.name, pdfroot, save=True)
        else:
            print 'No radial velocity for %s. Skipping.' % names[ii]


def plot_acc_z_hist(pdfroot='aorb_acc_mrPDF_MC_newMosaic/',mosaic=True):
    cc = objects.Constants()

    # Load names of young stars 
    if mosaic == True:
        # Load up mosaic data as well; select only stars at r>4, since
        # we don't want to add any info from mosaics if we have it in
        # the central 10" already
        yng1 = young.loadYoungStars(root+alnDir,fit=poly,points=points,
                                    withRVonly=True,silent=True,skipStar=['S5-237']) 
        yng2 = young.loadYoungStars(mscDir,fit=polyM,points=pointsM,
                                    mosaic=True, withRVonly=True,silent=True)
        cntrlNames = yng1.getArray('name')
        mscNames = yng2.getArray('name')
        # Merge this object with object from central 10" analysis
        yng = merge(yng1, yng2)
    else:
        yng = young.loadYoungStars(root+alnDir,fit=poly,points=points,
                                   withRVonly=True,silent=True)

    names = yng.getArray('name') 
    names.sort()
    yngNames = yng.getArray('name')

    pdfroot = root + alnDir + pdfroot

    # Read in accelerating_sources.dat, which lists accelerating stars that
    # passed the F test. We will use the accel measurements for these stars
    _acc = asciidata.open(root + alnDir + 'tables/accelerating_sources.dat')
    accels = _acc[0].tonumpy()
    acc = [aa.strip() for aa in accels]

    # Read in accel_upperLimit_sources.dat, which lists the stars with
    # upper limits below the a_max curve. We will sample from uniform accel
    # prior, between a_min and the upper limit.
    _upp = asciidata.open(root + alnDir + 'tables/accel_upperLimit_sources.dat')
    upper = _upp[0].tonumpy()
    upp = [uu.strip() for uu in upper]

    # For all other stars, we'll sample from uniform accel prior, between
    # a_min and a_max

    # For the stars w/out accel detections, we create random numbers:
    gen = aorb.create_generators(1,10**8)
    agen = gen[0]

    for ii in range(len(names)):
        star = yng.stars[ii]
        starName = star.name

        mscStar = False
        if (starName in mscNames) & (starName not in cntrlNames):
            mscStar = True

        if starName in acc: # use accel fit
            starXfit = star.fitpXa
            starYfit = star.fitpYa
            ax = star.fitpXa.a # (+x west)
            ay = star.fitpYa.a
            axerr = star.fitpXa.aerr
            ayerr = star.fitpYa.aerr
        else: # use velocity fit
            starXfit = star.fitpXv
            starYfit = star.fitpYv

        # arcsec
        x = starXfit.p # (+x west)
        y = starYfit.p
        xerr = starXfit.perr
        yerr = starYfit.perr
        r2d = star.r2d

        # arcsec/yr or km/s (radial)
        vx = starXfit.v # (+x west)
        vy = starYfit.v
        vz = star.vz
        vz_ref = star.rv_ref
        vxerr = starXfit.verr
        vyerr = starYfit.verr
        vzerr = star.vzerr

        # For stars w/ accel detections, plot acc +/- acc_err
        # For others, plot uniform hist between min and max acc allowed
        ax = np.zeros((10**5), dtype=float)
        ay = np.zeros((10**5), dtype=float)
        ar = np.zeros((10**5), dtype=float)
        #if starName in acc:
        #    for zz in range(10**5):
        #        ax[zz] = agen.gauss(ax, axerr)
        #        ay[zz] = agen.gauss(ay, ayerr)
        #        # Convert into mas/yr^2  
        #        (ax, ay) = util.vPix2Arc(ax, ay, self.trans)
        #        ax *= 1000.0
        #        ay *= 1000.0
        #        
        #        # Convert into radial and tangential
        #        (ar, at) = util.xy2circ(x, y, ax, ay)
        #else:
        #    # get min and max accels:
        #    (amin, amax) = getMinMaxAccel(x, y, vx, vy, vz)
        #    for zz in range(10**5):
        #        ar[zz] = agen.uniform(amin, amax)
        

        print 'Plotting star %s' % starName
        # File contains analytic orbit solutions with acceleration limits (MC)
        pdffile = '%s%s.mc.dat' % (pdfroot, starName)
        pdf = pickle.load(open(pdffile))
        z = pdf.z * dist / cc.au_in_pc
        x = pdf.x * dist / cc.au_in_pc
        y = pdf.y * dist / cc.au_in_pc
        r2d = np.hypot(x, y)

        #binsIn = np.arange(z.min(),z.max(),(z.max()-z.min())/1000.)
        py.clf()
        py.figure(1)
        py.figure(figsize=(10,5))
        py.subplots_adjust(hspace=0.3, wspace=0.3, top=0.9, bottom=0.1, left=0.1, right=0.95)
        py.subplot(1,2,1)
        py.hist(z, bins=100, histtype='step', color='k', normed=True)
        py.xlabel('z (pc)')
        py.ylabel('PDF')
        py.title(starName)
        py.subplot(1,2,2)
        py.plot(r2d, z, 'k.')
        thePlot = py.gca()
        # Increment for ticks on the X axis
        rng = py.axis()
        tmp = np.abs(float(rng[1]) - float(rng[0])) / 3.0
        xinc = __builtin__.round(tmp, 4)
        if (xinc == 0):
            xinc = 0.0001
        thePlot.xaxis.set_major_formatter(py.FormatStrFormatter('%6.4f'))
        thePlot.get_xaxis().set_major_locator(py.MultipleLocator(xinc))
        py.xlabel('Sampled Projected Radius (pc)')
        py.ylabel('Sampled z (pc)')
        py.savefig(pdfroot + 'pdfparams/hist_pdf_z_r2d_%s.png' % starName)
        py.close(1)
        #pdb.set_trace()


def plotIOcontours(align = 'align/align_d_rms_1000_abs_t', suffix=''):
    """
    Plot contours for the density of normal vectors in order to
    get a sense of the 2D inclination/bigOmega structure. Also integrate
    the IO probablity density to find the boundary of the disk.

    Inputs:  aorb_acc/disk.neighbor.dat[2]
    Outputs: plots/plotIOcontours.png
    """
    orbDir = 'aorb_thesis/'
    #orbDir = 'aorb_acc_mrPDF_MC_newMosaic/'

    # Disk solution
    numTrials = 10**5
    nside = 64
    npix = healpy.nside2npix(nside)
    pixIdx = np.arange(0, npix)

    (disk, diskSig) = loadDiskDensity(npix, orbDir=orbDir)

    (i, o) = healpy.pix2ang(nside, pixIdx)
    i *= 180.0 / math.pi
    o *= 180.0 / math.pi

    # Determine the background by averaging iteratively.
    its = 2
    for it in range(its):
        if (it == 0):
            idx = range(len(disk))
        else:
            # Reject the highest pixels
            idx = (np.where(disk < (avg+(3.0*std))))[0]

        avg = disk[idx].mean()
        std = disk[idx].std(ddof=1)

    
    mdx = disk.argmax()
    peak = disk[mdx]
    peakErr = diskSig[mdx] / np.sqrt(numTrials)
    print 'Max value is %8.2e +/- %8.2e' % (peak, peakErr)

    print 'Background avg = %8.2e  std = %8.2e' % (avg, std)
    print 'Significance = %4.1f' % ((peak - avg) / std)
    print 'Rejected %d out of %d pixels (%5.2f sr)' % \
          ((len(pixIdx) - len(idx)), len(pixIdx),
           (len(pixIdx) - len(idx)) * 4.0 * math.pi / len(pixIdx))

    ##########
    #
    # Integrate the peak
    #
    ##########
    idx = (np.where( disk > (avg+(3.0*std)) ))[0]
    pixDisk = disk[idx]
    iDisk = i[idx]
    oDisk = o[idx]

    # Sort the pixels
    sid0 = pixDisk.argsort()
    sid = sid0[::-1]  # reverse the array
    pixSort = pixDisk[sid]
    iPeak = iDisk[sid[0]]
    oPeak = oDisk[sid[0]]

    # Make a cumulative distribution function starting from the
    # highest pixel value. This way we can find the level above
    # which 68% of the trials will fall.
    cdf = np.cumsum(pixSort)
    idx = (np.where( (cdf / cdf[-1]) > 0.68 ))[0]
    edgeLevel = pixSort[idx[0]]
    print '1 sigma contour at %8.2e' % (edgeLevel)

    levels = np.array([avg - (3.*std),
                    avg + (3.*std),
                    edgeLevel,
                    peak / 2.0, 
                    peak-(5.*std)])
    colors = ['greenyellow','green', 'maroon', 'red', 'blue', 'cyan']
    #colors = ['y','g', 'm', 'r', 'b', 'c']
    legLab = ['Above (avg - 3*std)', 'Above (avg + 3*std)', 'Above 68% contour',
              'Above 50% of peak', 'Above (peak - 5*std)']

    # Plot the CDF
    deg2_per_pix = (4.0 * 180.0**2) / (math.pi * npix)
    pixArea = np.arange(len(cdf), dtype=float) * deg2_per_pix

    py.figure(1)
    py.clf()
    py.plot(pixArea, cdf / cdf[-1])
    py.xlabel("Square Degrees")
    py.ylabel("CDF -- Not Really")
    py.savefig('plots/plotIOcontours_cdf'+suffix+'.png')
    py.close()

    py.figure(2)
    py.clf()
    for ii in range(len(levels)):
        idx = (np.where(disk > levels[ii]))[0]
        print '%20s  =  %9.2e  i=[%3d-%3d]  o=[%3d-%3d]  (%5.2f sr, %2d stars)' % \
              (legLab[ii], levels[ii],
               i[idx].min(), i[idx].max(), o[idx].min(), o[idx].max(),
               len(idx) * 4.0 * math.pi / len(disk),
               (disk[idx]*deg2_per_pix).sum())

        py.plot(o[idx], i[idx], colors[ii], marker = '.')
        #py.plot(o[idx], i[idx], colors[ii] + '.')
        py.plot([20], [60 - (ii*8)], colors[ii], marker = 's')
        #py.plot([20], [60 - (ii*8)], colors[ii] + 's')
        py.text(30, 55 - (ii*8), legLab[ii])

    # Print out errors on mean for peak. Use 68% confidence range.
    level = levels[2]
    idx = (np.where(disk > level))[0]
    imin = i[idx].min()
    imax = i[idx].max()
    omin = o[idx].min()
    omax = o[idx].max()
    isig1 = abs(iPeak - imin) / np.sqrt(numTrials)
    isig2 = abs(iPeak - imax) / np.sqrt(numTrials)
    osig1 = abs(oPeak - omin) / np.sqrt(numTrials)
    osig2 = abs(oPeak - omax) / np.sqrt(numTrials)
    ierr = max(isig1, isig2)
    oerr = max(osig1, osig2)
    print ''
    print 'Disk Solution: i = %6.2f +/- %6.2f   o = %6.2f +/- %6.2f' % \
          (iPeak, ierr, oPeak, oerr)


    # Draw the highest point
    py.plot([o[mdx]], [i[mdx]], 'k.')

    py.axis([0, 360, 0, 180])

    py.title('Density Peak = %7.1e,  Bkg (%7.1e +/- %7.1e)' % (peak, avg, std))
    py.xlabel('PA to Asc. Node (deg)')
    py.ylabel('Inclination (deg)')

    py.savefig(root + alnDir + 'plots/plotIOcontour'+suffix+'.png')
    py.close()

def plotOrbAccelerators(mosaic=True, ntrials=99998):
    """
    Plot the combined PDF(i,O), and eccentricity distribution for just the
    disk stars for which we have acceleration measurements.

    Dependencies:
    accelLimit() - for accelerating sources that passed the F test

    Output:
    
    """

    # Load names of young stars 
    if mosaic == True:
        # Load up mosaic data as well; select only stars at r>4, since
        # we don't want to add any info from mosaics if we have it in
        # the central 10" already
        yng1 = young.loadYoungStars(root+alnDir,fit=poly,points=points,
                                    withRVonly=True,silent=True,skipStar=['S5-237']) 
        yng2 = young.loadYoungStars(mscDir,fit=polyM,points=pointsM,
                                    mosaic=True, withRVonly=True,silent=True)
        # Merge this object with object from central 10" analysis
        yng = merge(yng1, yng2)
    else:
        yng = young.loadYoungStars(root+alnDir,fit=poly,points=points,
                                   withRVonly=True,silent=True)

    orbDir = 'aorb_thesis/'
    #orbDir = 'aorb_acc_mrPDF_MC_newMosaic/'
    suffix = ''

    yngNames = yng.getArray('name')
    yngNames.sort()

    # Disk solution
    nside = 64
    npix = healpy.nside2npix(nside)
    pixIdx = np.arange(0, npix)
    (disk, diskStd) = loadDiskDensity(npix, orbDir=orbDir)

    # Read in file listing accelerating sources
    #_acc = asciidata.open('%s/%s/tables/outer_radial.dat' % (root, alnDir))
    _acc = asciidata.open('%s/%s/tables/accelerating_sources.dat' % (root, alnDir))
    acc = _acc[0].tonumpy()
    accels = [aa.strip() for aa in acc]

    # Read in file listing accelerating sources
    _upp = asciidata.open('%s/%s/tables/accel_upperLimit_sources.dat' % (root, alnDir))
    upp = _upp[0].tonumpy()
    upper = [uu.strip() for uu in upp]

    #acc_names = np.concatenate([accels, upper])
    acc_names = accels

    pdf_all = np.zeros((npix), dtype=float)
    #pdf_allStd = np.zeros((len(accels), self.npix), dtype=float)
    eccAll = []
    eccAllDisk = []
    eccAllNonDisk = []
    eccAcc = []
    eccAccDisk = []
    eccAccNonDisk = []
    accCnt = 0
    eccNonAcc = []
    eccNonAccDisk = []
    eccNonAccNonDisk = []
    ecc1 = []
    eccDisk1 = []
    eccNonDisk1 = []
    ecc2 = []
    eccDisk2 = []
    eccNonDisk2 = []
    ecc3 = []
    eccDisk3 = []
    eccNonDisk3 = []

    diskTab = asciidata.open(root+alnDir+'tables/disk_membership_prob_mosaic.dat')
    nameP = np.array([diskTab[0][ss].strip() for ss in range(diskTab.nrows)])
    diskP = diskTab[1].tonumpy()
    diskIdx = (np.where(diskP > 2.7e-3))[0]
    
    for ss in range(len(yngNames)):
        name = yngNames[ss]
        ioFile = ('%s/%s/%s/%s_mc_heal.dat' % (root, alnDir, orbDir, name))
        print 'Loading file for %s' % name

        nIdx = np.where(nameP == name)[0]

        # Get all disk candidates first
        orbFile = ('%s/%s/%s/%s.mc.dat' % (root, alnDir, orbDir, name))
        tmp = pickle.load(open(orbFile))

        # Separate out the disk from non-disk solutions
        idx = whereInDisk(tmp, angleCut=angleCut)
        eD = tmp.e[idx]
        nd = np.setdiff1d(np.arange(ntrials), idx)
        eND = tmp.e[nd]

        if diskP[nIdx] > 2.7e-3:
            # Get the eccentricities 
            eccAll = np.concatenate([eccAll, tmp.e])

            # Separate out just the disk solutions
            eccAllDisk = np.concatenate([eccAllDisk, eD]) 

            # Non-disk solutions
            eccAllNonDisk = np.concatenate([eccAllNonDisk, eND])

        # Get just accelerators that are on the disk
        #if (name not in accels) | (diskP[nIdx] < 2.7e-3):
        #    continue
        #if (name in accels) and (diskP[nIdx] > 2.7e-3):
        if (name in acc_names): # and (diskP[nIdx] > 2.7e-3):
            # First make the healpix maps
            print 'Adding %s to density map of accelerating sources (prob = %8.2e)' % \
                  (name, diskP[nIdx])
            pdf = np.fromfile(ioFile, dtype=float)
            pdf_all = pdf_all + pdf
    
            # Get the eccentricities
            eccAcc = np.concatenate([eccAcc, tmp.e])
    
            # Separate out just the disk solutions
            print '%-13s has %5d disk solutions' % (name, len(idx))
            eccAccDisk = np.concatenate([eccAccDisk, eD]) 
    
            # Non-disk solutions
            eccAccNonDisk = np.concatenate([eccAccNonDisk, eND])
    
            accCnt += 1

        # Get the non-accelerators that are on the disk (category 3)
        if (name not in acc_names) and (diskP[nIdx] > 2.7e-3):
            # Get the eccentricities
            eccNonAcc = np.concatenate([eccNonAcc, tmp.e])
    
            # Separate out just the disk solutions
            eccNonAccDisk = np.concatenate([eccNonAccDisk, eD]) 
    
            # Non-disk solutions
            eccNonAccNonDisk = np.concatenate([eccNonAccNonDisk, eND])
    
            #accCnt += 1

        # Get all stars that are on the disk (category 3)
        if (diskP[nIdx] > 2.7e-3):
            # Get the eccentricities
            ecc3 = np.concatenate([ecc3, tmp.e])
    
            # Separate out just the disk solutions
            eccDisk3 = np.concatenate([eccDisk3, eD]) 
    
            # Non-disk solutions
            eccNonDisk3 = np.concatenate([eccNonDisk3, eND])

        # Get all stars that are on the disk (category 2)
        if (diskP[nIdx] > 0.0455):
            # Get the eccentricities
            ecc2 = np.concatenate([ecc2, tmp.e])
    
            # Separate out just the disk solutions
            eccDisk2 = np.concatenate([eccDisk2, eD]) 
    
            # Non-disk solutions
            eccNonDisk2 = np.concatenate([eccNonDisk2, eND])

        # Get all stars that are on the disk (category 1)
        if (diskP[nIdx] > 0.3173):
            # Get the eccentricities
            ecc1 = np.concatenate([ecc1, tmp.e])
    
            # Separate out just the disk solutions
            eccDisk1 = np.concatenate([eccDisk1, eD]) 
    
            # Non-disk solutions
            eccNonDisk1 = np.concatenate([eccNonDisk1, eND])
    


    #outFile = '%s/%s/%s/outer_mc_heal.dat' % (root, alnDir, orbDir)
    outFile = '%s/%s/%s/accelerators_mc_heal.dat' % (root, alnDir, orbDir)

    pdf_all.tofile(outFile)

    # Plot the results on a healpix map
    pdh.go(outFile, npix, 1, 0)

    #pdb.set_trace()
    usetexTrue()
    # Print out some info
    ave_edisk = eccAccDisk.mean()
    std_edisk = eccAccDisk.std(ddof=1)
    med_edisk = np.median(eccAccDisk)
    rms_edisk = np.sqrt((eccAccDisk**2).sum() / len(eccAccDisk)) # RMS, not standard dev.
    print
    print 'Accelerating Stars - Disk Solutions:'
    print '   < e > = %4.2f +- %4.2f' % (ave_edisk, std_edisk)
    print '   median e = %4.2f' % med_edisk
    print '   RMS e = %4.2f' % rms_edisk
    print

    aveNA_edisk = eccNonAccDisk.mean()
    stdNA_edisk = eccNonAccDisk.std(ddof=1)
    medNA_edisk = np.median(eccNonAccDisk)
    rmsNA_edisk = np.sqrt((eccNonAccDisk**2).sum() / len(eccNonAccDisk)) # RMS, not standard dev.
    print
    print 'Non-accelerating Stars - Disk Solutions:'
    print '   < e > = %4.2f +- %4.2f' % (aveNA_edisk, stdNA_edisk)
    print '   median e = %4.2f' % medNA_edisk
    print '   RMS e = %4.2f' % rmsNA_edisk
    print

    # Determine the eccentricity probability density function
    # for JUST accelerating stars, separated by disk vs. non-disk solutions
    import matplotlib.font_manager
    prop = matplotlib.font_manager.FontProperties(size=14)
    eccStep = 0.05
    binsIn = np.arange(0, 1+eccStep, eccStep)
    py.clf()
    py.figure(1)
    py.figure(figsize=(7,6))
    py.subplots_adjust(left=0.15, right=0.95)
    ###py.hist(ecc, bins=binsIn, histtype='step', normed=True, label='All')
    py.hist(eccAccDisk, bins=binsIn, histtype='step', color='k', normed=True,
            ls='solid',lw=2,label='Disk Solutions')
    #py.hist(eccAccNonDisk, bins=binsIn, histtype='step', color='b', normed=True,
    #        ls='dashed',lw=2,label='Non-Disk Solutions')
    py.xlabel('Eccentricity', fontsize=16)
    #py.ylabel('N Solutions', fontsize=16)
    py.ylabel('Probability Density', fontsize=16)
    #py.legend(numpoints=1,fancybox=True,prop=prop)
    py.axis([0, 1.0, 0, 5.0])
    #py.title('Accelerating Sources on Disk (N = %i)' % accCnt, fontsize=16)
    py.savefig(root+alnDir+'plots/eccPDF_accelerators.png')
    py.savefig(root+alnDir+'plots/eps/eccPDF_accelerators.eps')
    py.close(1)

    # Ecc for accelerating stars, all solutions
    py.clf()
    py.figure()
    py.figure(figsize=(7,6))
    py.subplots_adjust(left=0.15, right=0.95)
    py.hist(eccAcc, bins=binsIn, histtype='step', color='k', normed=True,
            ls='solid',lw=2)
    py.xlabel('Eccentricity', fontsize=16)
    py.ylabel('Probability Density', fontsize=16)
    #py.axis([0, 1.0, 0, 5.0])
    py.title('Accelerating Sources on Disk (N = %i), All Soln' % accCnt, fontsize=16)
    py.savefig(root+alnDir+'plots/eccPDF_accelerators_allSoln.png')
    py.savefig(root+alnDir+'plots/eps/eccPDF_accelerators_allSoln.eps')
    py.close()

    # Determine the eccentricity probability density function
    # for ALL stars compared to accelerating, showing all solutions
    py.clf()
    py.figure(2)
    py.figure(figsize=(7,6))
    py.subplots_adjust(left=0.15, right=0.95)
    py.hist(eccAll, bins=binsIn, color='b', histtype='step', normed=True, label='All')
    nn,bb,pp = py.hist(eccAcc, bins=binsIn, color='r', histtype='step', normed=True, label='Accelerating')
    py.xlabel('Eccentricity', fontsize=16)
    py.ylabel('Probability Density', fontsize=16)
    py.legend(numpoints=1,fancybox=True,prop=prop)
    py.axis([0, 1.0, 0, nn.max()+0.2])
    py.title('All Disk Stars vs. Accelerating Disk Stars - All Solutions', fontsize=14)
    py.savefig(root+alnDir+'plots/eccPDF_accel_vs_all.png')
    py.close(2)

    # Determine the eccentricity probability density function
    # for ALL stars compared to ACCELERATING stars, showing only disk solutions
    py.clf()
    py.figure(3)
    py.figure(figsize=(7,6))
    py.subplots_adjust(left=0.15, right=0.95)
    py.hist(eccAllDisk, bins=binsIn, color='b', histtype='step', normed=True, label='All')
    nn,bb,pp = py.hist(eccAccDisk, bins=binsIn, color='r', histtype='step', normed=True, label='Accelerating')
    py.xlabel('Eccentricity', fontsize=16)
    py.ylabel('Probability Density', fontsize=16)
    py.legend(numpoints=1,fancybox=True,prop=prop)
    py.axis([0, 1.0, 0, nn.max()+0.2])
    py.title('All Disk Stars vs. Accelerating Disk Stars - Disk Solutions', fontsize=14)
    py.savefig(root+alnDir+'plots/eccPDF_accel_vs_all_diskSoln.png')
    py.savefig(root+alnDir+'plots/eps/eccPDF_accel_vs_all_diskSoln.eps')
    py.close(3)

    # Determine the eccentricity probability density function
    # for ACCELERATING vs NON_ACCELERATING stars, showing only disk solutions
    py.clf()
    py.figure(4)
    py.figure(figsize=(7,6))
    py.subplots_adjust(left=0.15, right=0.95)
    nn,bb,pp = py.hist(eccAccDisk, bins=binsIn, color='r', histtype='step', normed=True, label='Accelerating')
    py.hist(eccNonAccDisk, bins=binsIn, color='b', histtype='step', normed=True, label='Non-Accel')
    py.xlabel('Eccentricity', fontsize=16)
    py.ylabel('Probability Density', fontsize=16)
    py.legend(numpoints=1,fancybox=True)
    py.axis([0, 1.0, 0, nn.max()+0.2])
    #py.title('Accelerating vs. Non-Accelerating Disk Stars - Disk Solutions', fontsize=14)
    py.savefig(root+alnDir+'plots/eccPDF_accel_vs_nonAccel_diskSoln.png')
    py.savefig(root+alnDir+'plots/eps/eccPDF_accel_vs_nonAccel_diskSoln.eps')
    py.close(4)

    py.clf()
    py.figure(5)
    py.figure(figsize=(6,6))
    py.subplots_adjust(left=0.15, right=0.95, top=0.95)
    py.hist(eccDisk1, bins=binsIn, color='r', histtype='step', normed=True, label='Category 1')
    py.hist(eccDisk2, bins=binsIn, color='g', histtype='step', normed=True, label='Category 2')
    py.hist(eccDisk3, bins=binsIn, color='b', histtype='step', normed=True, label='Category 3')
    nn,bb,pp = py.hist(eccAccDisk, bins=binsIn, color='k', histtype='step', normed=True, label='Accelerating')
    py.xlabel('Eccentricity', fontsize=16)
    py.ylabel('Probability Density', fontsize=16)
    py.legend(numpoints=1,fancybox=True,prop=prop)
    py.axis([0, 1.0, 0, nn.max()+0.2])
    py.savefig(root+alnDir+'plots/eccPDF_category123_diskSoln.png')
    py.savefig(root+alnDir+'plots/eps/eccPDF_category123_diskSoln.eps')
    py.close(5)
    
    py.clf()
    py.figure(6)
    py.figure(figsize=(10,5))
    py.subplots_adjust(wspace=0.3, hspace=0.3, left=0.1, right=0.95)
    py.subplot(1,2,1)
    nn,bb,pp = py.hist(eccAccDisk, bins=binsIn, color='k', histtype='step', normed=True)
    py.xlabel('Eccentricity')#, fontsize=16)
    py.ylabel('Probability Density')#, fontsize=16)
    py.axis([0, 1.0, 0, nn.max()+0.2])
    py.subplot(1,2,2)
    nn,bb,pp = py.hist(eccNonAccDisk, bins=binsIn, color='k', histtype='step', normed=True)
    py.xlabel('Eccentricity')#, fontsize=16)
    py.ylabel('Probability Density')#, fontsize=16)
    py.axis([0, 1.0, 0, nn.max()+0.2])
    py.savefig(root+alnDir+'plots/eccPDF_accel_vs_nonAccel_diskSoln_2panel.png')
    py.savefig(root+alnDir+'plots/eps/eccPDF_accel_vs_nonAccel_diskSoln_2panel.eps')
    py.close(6)

    py.clf()
    py.figure(7)
    py.figure(figsize=(10,5))
    py.subplots_adjust(wspace=0.3, hspace=0.3, left=0.1, right=0.95)
    py.subplot(1,2,1)
    nn,bb,pp = py.hist(eccAllDisk, bins=binsIn, color='k', lw=1.5, histtype='step', normed=True)
    py.xlabel('Eccentricity')
    py.ylabel('Probability Density')
    py.axis([0, 1.0, 0, nn.max()+0.2])
    py.subplot(1,2,2)
    n1,b1,p1 = py.hist(eccAccDisk, bins=binsIn, color='k', lw=1.5, histtype='step',
                       normed=True, label='Accel.')
    n2,b2,p2 = py.hist(eccNonAccDisk, bins=binsIn, color='k', ls='dashed', lw=1.5,
                       histtype='step', normed=True, label='Non-accel.')
    py.xlabel('Eccentricity')
    py.ylabel('Probability Density')
    py.legend(numpoints=1,fancybox=True,prop=prop)
    py.axis([0, 1.0, 0, max(n1.max(),n2.max())+0.2])
    py.savefig(root+alnDir+'plots/eccPDF_all_AccAndNonAcc_diskSoln_2panel.png')
    py.savefig(root+alnDir+'plots/eccPDF_all_AccAndNonAcc_diskSoln_2panel.eps')
    py.close(7)
    usetexFalse()
    
    

def diskMembers(orbDir='aorb_thesis/', mosaic=True, suffix='',
                file1='disk.neighbor.dat', file2='disk.neighborStd.dat',
                radial_bin=None, singlePdf=True, returnProps=False, verbose=True,
                LHsigCut=3.0):
    """
    Determine which stars are members of the disk.

    radial_bin (int):  Set to 1, 2, or 3 if determining disk membership for a
    		       certain radial bin. 1=inner, 2=middle, 3=outer
                       NOTE: Bin intervals are hard-coded!
    returnProps (bool): Return properties of the disk (peak angle, radius,
    			density, number of members)
    LHsigCut (float):  Significance threshold for non-members;
            	       default is set to 3.0, meaning that stars with
                       likelihood of not being on the disk of >3 sigma
                       are considered non-members. The rest are candidates.

    Output:
    tables/disk_membership_prob.dat -- text table containing results
    plots/disk_membership_hist.png -- Histogram of disk membership
       probabilities.
    """
    # Load names of young stars 
    if mosaic == True:
        # Load up mosaic data as well; select only stars at r>4, since
        # we don't want to add any info from mosaics if we have it in
        # the central 10" already
        yng1 = young.loadYoungStars(root+alnDir,fit=poly,points=points,
                                    withRVonly=True,silent=True,skipStar=['S5-237']) 
        yng2 = young.loadYoungStars(mscDir,fit=polyM,points=pointsM,
                                    mosaic=True, withRVonly=True,silent=True)
        # Merge this object with object from central 10" analysis
        yng = merge(yng1, yng2)
    else:
        yng = young.loadYoungStars(root+alnDir,fit=poly,points=points,
                                   withRVonly=True,silent=True)

    # Sort by names, and sort the radii accordingly
    # radii will be needed if we are looking at particular radial_bin
    yngNames = yng.getArray('name')
    r2d = yng.getArray('r2d')
    nArg = np.argsort(yngNames)
    yngNames = np.array([yngNames[aa] for aa in nArg])
    r2d = np.array([r2d[aa] for aa in nArg])

    if radial_bin != None:
        # Define the radial bin intervals (this should match what was run
        # in analyticOrbits.Disk()
        r1 = 3.197
        r2 = 6.473

        if radial_bin == 1:
            ridx = np.where(r2d <= r1)[0]
        if radial_bin == 2:
            ridx = np.where((r2d > r1) & (r2d <= r2))[0]
        if radial_bin == 3:
            ridx = np.where(r2d > r2)[0]

        yngNames = yngNames[ridx]
        r2d = r2d[ridx]

    # Disk solution
    nside = 64
    npix = healpy.nside2npix(nside)
    pixIdx = np.arange(0, npix)
    (disk, diskStd) = loadDiskDensity(npix, orbDir=orbDir,
                                      file1=file1, file2=file2,
                                      singlePdf=singlePdf)

    (i, o) = healpy.pix2ang(nside, pixIdx)
    i *= 180.0 / math.pi
    o *= 180.0 / math.pi

    didx = disk.argmax()
    peak = disk[didx]
    idisk = i[didx]
    odisk = o[didx]
    print 'Peak at %.2e  stars/deg^2' % (peak)

    # Determine which pixels on the sky are "in the disk" by those
    # that have a density that is within 50% of the peak density value.
    #pidx = (np.where(disk > (0.5 * peak)))[0]
    # NOTE: The above assumes there is only one structure in the
    # HEALpix maps! If there are two significant structures, the pixels
    # from both will get merged
    pidx = (np.where(disk > (0.5 * peak)))[0]

    itmp = i[pidx]
    ilo = itmp.min()
    ihi = itmp.max()
    otmp = o[pidx]
    olo = otmp.min()
    ohi = otmp.max()

    print 'Disk at    i = %5.1f [%5.1f - %5.1f]   o = %5.1f [%5.1f - %5.1f]' % \
          (idisk, ilo, ihi, odisk, olo, ohi)


    # Determine the probability for each star to be in this pixel
    peakProb = np.zeros(len(yngNames), dtype=float)
    solidAngle = np.zeros(len(yngNames), dtype=float)
    offDiskLH = np.zeros(len(yngNames), dtype=float)

    _out = open(root + alnDir + 'tables/disk_membership_prob' + suffix + '.dat', 'w')

    # Determine the total solid angle for the "in the disk" region
    totalSolidAngle = 0.0
    areaPerPixel = areaOnSky / npix  # in deg^2
    degsq2str = (math.pi / 180.0)**2

    diskSolidAngle = len(pidx) * areaPerPixel
    diskRadius = radiusOfCone(diskSolidAngle)
    print '%6.1f - Disk Solid Angle (deg^2)' % diskSolidAngle
    print '%6.2f - Disk Radius (deg)' % diskRadius

    # Now determine the integrated probability of falling on the
    # disk.
    for ss in range(len(yngNames)):
        name = yngNames[ss]
        orbFile = root + alnDir + orbDir + name + '_mc_heal.dat'
        if os.path.exists(orbFile) == False:
            continue

        pdf = np.fromfile(orbFile, dtype=float)

        # Determine the 68.4% confidence region solid angle
        sid = (pdf.argsort())[::-1]  # reverse sort
        pdfSort = pdf[sid] # highest value first

        # Make a cumulative distribution function starting from the
        # highest pixel value. This way we can find the level above
        # which 68% of the trials will fall.
        cdf = np.cumsum(pdfSort)

        # Determine point at which we reach 68% level
        idx = (np.where(cdf > 0.6827))[0]
        level = pdfSort[idx[0]]
        solidAngle[ss] = (idx[0] + 1) * areaPerPixel * degsq2str

        # Keep a running tally for the average solid angle
        totalSolidAngle += solidAngle[ss]
        
        #nzid = (np.where(pdf > 0))[0]  # not used?
        nzDisk = (np.where((disk > (0.5 * peak)) & (pdf > 0)))[0]
        solidAngleInDisk = len(nzDisk) * areaPerPixel
        maxProb = pdfSort[0:len(pidx)].sum()

        peakProb[ss] = pdf[pidx].sum() / maxProb
        #offDiskLH[ss] =  (1.0 - peakProb[ss]) * areaOnSky / solidAngle[ss]
        #offDiskLH[ss] =  (1.0 - peakProb[ss]) / maxProb
        #offDiskLH[ss] = (1.0 - peakProb[ss]) * areaOnSky / solidAngle[ss]
        #offDiskLH[ss] = 1.0 / offDiskLH[ss]
        # Likelihood that the star is not in the disk plane:
        offDiskLH[ss] = 1.0 - peakProb[ss]

        # What is the threshold for disk/non-disk members?
        lhNotOnDisk_cut = scipy.special.erf(LHsigCut/np.sqrt(2.))
        probOnDisk_cut = 1.0 - lhNotOnDisk_cut

        if (peakProb[ss] != 0):
            onDisk = ''
            if (peakProb[ss] < probOnDisk_cut):
            #if (peakProb[ss] < 2.7e-3):
                onDisk = 'Not on disk'
            if verbose == True:
                print '%13s  %8.2e  %8.2e   %6.3f  %s ' % \
                      (name, peakProb[ss], offDiskLH[ss], solidAngle[ss], onDisk)

        _out.write('%13s  %8.2e  %6.3f\n' % 
		   (name, peakProb[ss], solidAngle[ss]))

    avgSolidAngle = totalSolidAngle / len(yngNames)

    print '%8.5f - Average Solid Angle (sr) Per Star (1 sigma)' % \
	(avgSolidAngle)
    print '%8.5f - Expected density for isotropic population' % \
	(len(yngNames) / areaOnSky)


    # What is the measured density outside the disk region:
    avgBkg = 1.0
    stdBkg = 1.0

    # Iteratively compute background
    for n in range(2):
        idx = (np.where(disk < (avgBkg + (3.0 * stdBkg))))[0]

        avgBkg = disk[idx].mean()
        stdBkg = disk[idx].std(ddof=1)
        print 'trial = %2d   avg = %e   std = %e, rejecting %d of %d' % \
              (n, avgBkg, stdBkg, (len(disk)-len(idx)), len(disk))

    print ''
    print 'Average Background   rho = %e' % avgBkg
    print 'Stddev of Background rho = %e' % stdBkg
    print 'Density Ratio for CW     = %5.2f' % ((peak - avgBkg) / stdBkg)

    #diskIdx = (np.where(peakProb > 2.7e-3))[0]
    diskIdx = (np.where(peakProb > probOnDisk_cut))[0]
    print ''
    if radial_bin == None:
        print 'Total number of stars: %2d' % (len(peakProb))
    else:
        print 'Total number of stars in radial bin %d: %2d' % \
              (radial_bin, len(peakProb))
        
    print 'On CW disk:       %2d' % (len(diskIdx))
    print 'Not on CW disk:   %2d' % (len(peakProb) - len(diskIdx))

    _out.close()

    print
    print 'Uncertainty on peak position:'
    print '   HWHM / sqrt(N_candidate_disk) = %4.2f deg' % \
          (diskRadius / np.sqrt(len(diskIdx))) 
    
    non0 = (np.where(peakProb != 0))[0]

    probGood = peakProb[non0]
    print '****'
    print 'Sum of probs = # disk members = %6.3f' % peakProb.sum()
    print '****'

    # Now plot a histogram of the values
    py.figure(figsize=(6,6))
    py.subplots_adjust(left=0.1, right=0.96, top=0.95)
    py.clf()
    aa,bb,cc = py.hist(np.log10(probGood), bins=np.arange(-5,0.1,0.1),
                       histtype='step', color='k', linewidth=2)
    py.plot([-2.57,-2.57],[0,aa.max()+2],'k--') # 3 sigma cut
    py.plot([-1.34,-1.34],[0,aa.max()+2],'k--') # 2 sigma cut
    py.plot([-0.5,-0.5],[0,aa.max()+2],'k--') # 1 sigma cut
    py.xlabel('Log Probability ', fontsize=18)
    py.ylabel('Number of Stars', fontsize=18)
    py.axis([-4.5, 0, 0, aa.max()+2])
    #py.title('Ranges: i (%3d - %3d), O (%3d - %3d)' % (ilo, ihi, olo, ohi))
    py.savefig(root + alnDir + 'plots/disk_membership_hist' + suffix + '.png')
    py.close()

    #print
    #print yngNames

    if returnProps == True:
        return (idisk,odisk,peak,diskRadius,len(diskIdx)) 


def histSolidAngles(mosaic=True, suffix='_mosaic'):
    # Load names of young stars 
    if mosaic == True:
        # Load up mosaic data as well; select only stars at r>4, since
        # we don't want to add any info from mosaics if we have it in
        # the central 10" already
        yng1 = young.loadYoungStars(root+alnDir,fit=poly,points=points,
                                    withRVonly=True,silent=True,skipStar=['S5-237']) 
        yng2 = young.loadYoungStars(mscDir,fit=polyM,points=pointsM,
                                    mosaic=True, withRVonly=True,silent=True)
        # Merge this object with object from central 10" analysis
        yng = merge(yng1, yng2)
    else:
        yng = young.loadYoungStars(root+alnDir,fit=poly,points=points,
                                   withRVonly=True,silent=True)

    names = yng.getArray('name')
    velCnt = yng.getArray('velCnt')

    nside = 64
    npix = healpy.nside2npix(nside)

    #areaPerPixel = areaOnSky / npix                   # deg^2
    areaOnSkyStr = 4.0 * math.pi         # steradians
    areaPerPixel = areaOnSkyStr / npix   # steradians

    solidAngle = np.zeros(len(names), float)
    for ss in range(len(names)):
        name = names[ss]
        orbFile = root + alnDir + 'aorb_thesis/' + name + '_mc_heal.dat'
        #orbFile = root + alnDir + 'aorb_acc_mrPDF_MC_newMosaic/' + name + '_mc_heal.dat'
        #orbFile = root + alnDir + 'aorb_acc/' + name + '_mc_heal.dat'

        pdf = np.fromfile(orbFile, dtype=float)

        # Determine the 68.4% confidence region solid angle
        sid = (pdf.argsort())[::-1]  # reverse sort
        pdfSort = pdf[sid]
        
        # Make a cumulative distribution function starting from the
        # highest pixel value. This way we can find the level above
        # which 68% of the trials will fall.
        cdf = np.cumsum(pdfSort)
        
        # Determine point at which we reach 68% level
        idx = (np.where(cdf > 0.6827))[0]
        level = pdfSort[idx[0]]
        solidAngle[ss] = (idx[0] + 1) * areaPerPixel

    py.clf()
    py.figure(figsize=(6,6))
    usetexTrue()

    #binsIn = np.arange(0, areaOnSkyStr, 0.151)
    binsIn = np.arange(0, areaOnSkyStr, 0.1)

    print 'Average Solid Angle: (sr)'

    #idx = (np.where(velCnt == 0))[0]
    #(bins, data) = histNofill.hist(binsIn, solidAngle[idx])
    #paum = plot(bins, data, color='#9b9b9b', linewidth=3)
    #print '  Secondary: %3.1f' % (solidAngle[idx].mean())
    
    #idx = (np.where(velCnt != 0))[0]
    ours = py.hist(solidAngle, bins=binsIn, histtype='step', color='k', linewidth=2)
    print '    Primary: %3.1f sr' % (solidAngle.mean())
    
    py.xlabel(r'Solid Angle of 1$\sigma_{\vec{n}}$ (sr)')
    py.ylabel(r'Number of Stars')

    #legend((ours, paum), ('Stars in this work', 'Other sources'))
    if mosaic == False:
        py.axis([0, 2, 0, 25])
    else:
        py.axis([0, 2, 0, 30])

    py.savefig(plotdir + 'hist_solidAngle' + suffix + '.png')
    py.close()
    usetexFalse()


def peakPosErrPdf(orbDir='aorb_thesis/', mosaic=True, radial_bins=True,
                  cntrl3bins=False,singlePdf=True):
    """
    Determine peak position and uncertainty of disk.
    Also determines peak of inner, middle, and outer disk solutions,
    as well as disk solution using Lu+09 sample.

    Dependencies:
    	diskMembers()
    	disk = aorb.Disk()
    	disk.run(do_all=True)
    	disk.run(do_radial_bins=True)
    	disk.run(lu09_sample=True)

    """
    # Load names of young stars 
    if mosaic == True:
        yng1 = young.loadYoungStars(root+alnDir,fit=poly,points=points,
                                    withRVonly=True,silent=True,skipStar=['S5-237']) 
        yng2 = young.loadYoungStars(mscDir,fit=polyM,points=pointsM,
                                    mosaic=True, withRVonly=True,silent=True)
        # Merge this object with object from central 10" analysis
        yng = merge(yng1, yng2)
    else:
        yng = young.loadYoungStars(root+alnDir,fit=poly,points=points,
                                   withRVonly=True,silent=True)

    yngNames = yng.getArray('name')
    r2d = yng.getArray('r2d')
    yngNames.sort()

    if mosaic == True:
        r1 = np.where(r2d <= 3.197)[0]
        r2 = np.where((r2d > 3.197) & (r2d <= 6.473))[0]
        r3 = np.where(r2d > 6.473)[0]
    elif cntrl3bins == True:
        r1 = np.where(r2d < 2.266)[0]
        r2 = np.where((r2d >= 2.266) & (r2d < 3.538))[0]
        r3 = np.where(r2d >= 3.538)[0]

    # Disk solution
    nside = 64
    npix = healpy.nside2npix(nside)
    pixIdx = np.arange(0, npix)

    (i, o) = healpy.pix2ang(nside, pixIdx)
    i *= 180.0 / math.pi
    o *= 180.0 / math.pi

    degsq2str = (math.pi / 180.0)**2

    if mosaic == True:
        if radial_bins == True: 
            sample = ['Full sample', 'Inner (r <= 3.197 asec)', 'Middle (3.197 < r <= 6.473 asec)',
                      'Outer (r > 6.473 asec)']
            f1 = ['disk.neighbor.dat', 'inner_disk.neighbor.dat',
                  'middle_disk.neighbor.dat','outer_disk.neighbor.dat']
            f2 = ['disk.neighborStd.dat', 'inner_disk.neighborStd.dat',
                  'middle_disk.neighborStd.dat','outer_disk.neighborStd.dat']
            memberFile = ['mosaic', 'inner_disk', 'middle_disk', 'outer_disk']

            nn_rad = []
            for ii in range(len(memberFile)):
                # Get disk membership 
                diskTab = asciidata.open(root+alnDir+'tables/disk_membership_prob_'+\
                                         memberFile[ii]+'.dat')
                name = [diskTab[0][ss].strip() for ss in range(diskTab.nrows)]
                diskP = diskTab[1].tonumpy()
                diskIdx = (np.where(diskP > 2.7e-3))[0]
            
                if ii == 0:
                    r_disk = np.where(diskP > 2.7e-3)[0]
                if ii == 1:
                    r_disk = np.where(diskP > 2.7e-3)[0]
                elif ii == 2:
                    r_disk = np.where(diskP > 2.7e-3)[0]
                elif ii == 3:
                    r_disk = np.where(diskP > 2.7e-3)[0]
    
                nn_rad = np.concatenate([nn_rad, [len(r_disk)]])
        #else:
        #    sample = ['Full sample']
        #    f1 = ['disk.neighbor.dat']
        #    f2 = ['disk.neighborStd.dat']
        #    memberFile = 'mosaic'
        #    # Get disk membership 
        #    diskTab = asciidata.open(root+alnDir+'tables/disk_membership_prob_'+\
        #                             memberFile+'.dat')
        #    name = [diskTab[0][ss].strip() for ss in range(diskTab.nrows)]
        #    diskP = diskTab[1].tonumpy()
        #    diskIdx = (np.where(diskP > 2.7e-3))[0]
        #    nn = [len(diskIdx)]

    elif cntrl3bins == True:
        sample = ['Full sample', 'Inner (r < 2.266 asec)',
                  'Middle (2.266 < r < 3.538 asec)', 'Outer (r >= 3.538 asec)']
        f1 = ['disk.neighbor.dat','inner_disk_cntrl.neighbor.dat',
              'middle_disk_cntrl.neighbor.dat', 'outer_disk_cntrl.neighbor.dat']
        f2 = ['disk.neighborStd.dat','inner_disk_cntrl.neighborStd.dat',
              'middle_disk_cntrl.neighborStd.dat','outer_disk_cntrl.neighborStd.dat']
        memberFile = ['inner_disk_cntrl', 'middle_disk_cntrl', 'outer_disk_cntrl']
        nn_rad = []
        for ii in range(len(memberFile)):
            # Get disk membership per radial bin
            diskTab = asciidata.open(root+alnDir+'tables/disk_membership_prob_'+\
                                     memberFile[ii]+'.dat')
            name = [diskTab[0][ss].strip() for ss in range(diskTab.nrows)]
            diskP = diskTab[1].tonumpy()
            diskIdx = (np.where(diskP > 2.7e-3))[0]
        
            if ii == 0:
                r_disk = np.where(diskP > 2.7e-3)[0]
            if ii == 1:
                r_disk = np.where((r2d < 2.266) & (diskP > 2.7e-3))[0]
            elif ii == 2:
                r_disk = np.where((r2d >= 2.266) & (r2d < 3.538) & (diskP > 2.7e-3))[0]
            elif ii == 3:
                r_disk = np.where((r2d >= 3.538) & (diskP > 2.7e-3))[0]

            nn_rad = np.concatenate([nn_rad, [len(r_disk)]])
    else:
        sample = ['Full sample']
        f1 = ['disk.neighbor.dat']
        f2 = ['disk.neighborStd.dat']
        diskTab = asciidata.open(root+alnDir+'tables/disk_membership_prob.dat')
        name = [diskTab[0][ss].strip() for ss in range(diskTab.nrows)]
        diskP = diskTab[1].tonumpy()
        diskIdx = (np.where(diskP > 2.7e-3))[0]
        print 'Number of disk candidates for full sample: %i' % len(diskIdx)
        nn = [len(diskIdx)]

    if radial_bins == True | cntrl3bins == True:
        nn = [nn_rad[0], nn_rad[1], nn_rad[2], nn_rad[3]]
        print 'Number of disk candidates for full sample:'
        print '  all: %i' % nn[0]
        print 'Number of disk candidates per radial bin:'
        print '  r1: %i' % nn[1]
        print '  r2: %i' % nn[2]
        print '  r3: %i' % nn[3]

    idisk_all = []
    odisk_all = []
    for ss in range(len(sample)):
        ##########
        # Nearest Neighbor
        ##########
        #if len(sample) == 1:  # ran 4 NN density maps on the full disk analysis (11_10_26)
        #    foo = False
        #else:
        #    foo = True # ran 1 NN density map on the full disk analysis (11_10_26)
        (disk, diskStd) = loadDiskDensity(npix, orbDir=orbDir,
                                          file1=f1[ss],
                                          file2=f2[ss],
                                          singlePdf=singlePdf)

        didx = disk.argmax()
        peak = disk[didx]
        idisk = i[didx]
        odisk = o[didx]

        # Determine which pixels on the sky are "in the disk" by those
        # that have a density that is within 50% of the peak density value.
        pidx = (np.where(disk > (0.5 * peak)))[0]

        itmp = i[pidx]
        ilo = itmp.min()
        ihi = itmp.max()
        otmp = o[pidx]
        olo = otmp.min()
        ohi = otmp.max()

        # Determine the total solid angle for the "in the disk" region
        areaPerPixel = areaOnSky / npix  # in deg^2
        diskSolidAngle = len(pidx) * areaPerPixel
        diskRadius = radiusOfCone(diskSolidAngle)

        # Uncertainty in peak position (HWHM / sqrt(N_disk_members)
        posErr = diskRadius / np.sqrt(nn[ss])

        print 'Disk solution (%s):' % sample[ss]
        print '  i = %5.1f +- %3.1f [%5.1f - %5.1f]' % (idisk, posErr, ilo, ihi)
        print '  o = %5.1f +- %3.1f [%5.1f - %5.1f]' % (odisk, posErr, olo, ohi)
    
        print '  Disk Solid Angle  = %6.1f deg^2 (%3.1f sr)' % \
              (diskSolidAngle, diskSolidAngle*degsq2str)
        print '  Disk Radius (HWHM) = %6.2f deg' % diskRadius
        print 'Uncertainty determined as HWHM / sqrt(%i)' % nn[ss]
        print

        # gather the solutions for each radius to return to caller
        idisk_all = np.concatenate([idisk_all, [idisk]])
        odisk_all = np.concatenate([odisk_all, [odisk]])

    return idisk_all, odisk_all


def yngDiskInclination(mosaic=True, suffix='_mosaic'):
    """
    Plot the angle between the disk and each star's normal vector.

    Outputs:
    plots/yng_angle_off_disk 
    """
    cc = objects.Constants()
    # Load up directory names
    orbDir = 'aorb_thesis/'
    
    # Load disk star names and probability of disk membership
    (names, diskP) = readDiskProb(suffix=suffix, diskOnly=False)

    # Load kinematic data for disk stars
    yng = loadYoungByName(names, mosaic=mosaic)
    x = yng.getArray('x')
    y = yng.getArray('y')
    yngRadius = np.sqrt(x**2 + y**2)

    # Disk solution
    irad = np.radians(idisk)
    orad = np.radians(odisk)
    sinip = np.sin(np.radians(idisk))
    cosip = np.cos(np.radians(idisk))

    #####
    # Get angles for all other young stars
    #####
    # Setup i/Omega for each pixel on sky
    nside = 64
    npix = healpy.nside2npix(nside)
    (iheal, oheal) = healpy.pix2ang(nside, np.arange(0, npix))
    iheal *= rad2deg
    oheal *= rad2deg
    sini = np.sin(iheal / rad2deg)
    cosi = np.cos(iheal / rad2deg)

    # Determine angular offset to disk for every other point on the sky
    cosodiff = np.cos( (oheal - odisk) / rad2deg )
    angOffCW = np.arccos( (sini * sinip * cosodiff) + (cosi * cosip) )
    angOffCW *= rad2deg

    # Store angles in array
    yngAngle = np.zeros(len(names), float)
    yngAngleErr = np.zeros(len(names), float)
    minAngle = np.zeros(len(names), float)
    maxAngle = np.zeros(len(names), float)

    # Loop through each star and find range of angles
    for ss in range(len(names)):
        name = names[ss]
        orbFile = root + alnDir + orbDir + name + '_mc_heal.dat'

        pdf = np.fromfile(orbFile, dtype=float)

        # Determine the 68.4% confidence region solid angle
        sid = (pdf.argsort())[::-1]  # reverse sort
        peakPix = sid[0]
        pdfSort = pdf[sid]
        
        # Make a cumulative distribution function starting from the
        # highest pixel value. This way we can find the level above
        # which 68% of the trials will fall.
        cdf = np.cumsum(pdfSort)

        # Determine point at which we reach 68% level
        idx = (np.where(cdf > 0.6827))[0]
        level = pdfSort[idx[0]] # stars/deg^2

        # Calculate angle of every pixel from peak of this star's PDF:
        sinipeak = np.sin(np.radians(iheal[peakPix]))
        cosipeak = np.cos(np.radians(iheal[peakPix]))
        cosodiff = np.cos( (oheal - oheal[peakPix]) / rad2deg )
        angle = np.arccos( (sini * sinipeak * cosodiff) + (cosi * cosipeak) )
        angle *= rad2deg
        
        # Select only the most probable of the two
        # degenerate solutions.
        idx = (np.where((pdf > level) & (angle < 45.0)))[0]
            
        # Find the range in angles... calc uncertainty from it.
        minAngle[ss] = angOffCW[idx].min()
        maxAngle[ss] = angOffCW[idx].max()
        yngAngleErr[ss] = (maxAngle[ss] - minAngle[ss]) / 2.0
        yngAngle[ss] = minAngle[ss] + yngAngleErr[ss]

    import matplotlib.font_manager
    prop = matplotlib.font_manager.FontProperties(size=14)

    usetexTrue()
    py.clf()
    py.figure(figsize=(6,6))
    py.subplots_adjust(left=0.2, right=0.96, top=0.95)
    onD = (np.where(diskP >= 2.7e-3))[0]
    offD = (np.where(diskP < 2.7e-3))[0]
    p1 = py.errorbar(yngRadius[onD], yngAngle[onD], yerr=yngAngleErr[onD],
                  fmt='r.', ecolor='r', mfc='r', mec='r', label='Disk')
    p2 = py.errorbar(yngRadius[offD], yngAngle[offD], yerr=yngAngleErr[offD],
			fmt='k.', ecolor='k', mfc='k', mec='k', label='Non-Disk')
    py.legend(numpoints=1,fancybox=True,loc=1,prop=prop)
    py.xlabel('Sky Projected Radius (arcsec)')
    py.ylabel('Inclination w.r.t. Disk (deg)')
    py.savefig(plotdir + 'yng_angle_off_disk.png')
    py.close()

    # Plot a histogram of the angular offset
    binsIn = np.arange(0, 185, 5)
    fig = py.figure(figsize=(6,6))
    fig.subplots_adjust(left=0.15, right=0.96, top=0.95)
    fig.clf()
    ax = fig.add_subplot(111)
    ax.hist(yngAngle[onD], bins=binsIn, histtype='step',color='r',
            linewidth=2,label='Disk')
    ax.hist(yngAngle[offD], bins=binsIn, histtype='step',color='k',
            linewidth=2,ls='dashed',label='Non-Disk')
    ax.set_xlabel('Inclination w.r.t. Disk (deg)')
    ax.set_ylabel('N')
    ax.legend(numpoints=1,prop=prop,fancybox=True)
    #ax.axis([0, 160, 0, 25])
    fig.savefig(plotdir + 'yng_angle_off_disk_hist.png')
    py.close()

    # Plot a histogram of the angular offset
    binsIn = np.arange(0, 185, 5)
    fig = py.figure(figsize=(6,6))
    fig.subplots_adjust(left=0.15, right=0.96, top=0.95)
    fig.clf()
    ax = fig.add_subplot(111)
    ax.hist(minAngle[onD], bins=binsIn, histtype='step',color='r',
            linewidth=2,label='Disk')
    ax.hist(minAngle[offD], bins=binsIn, histtype='step',color='k',
            linewidth=2,ls='dashed',label='Non-Disk')
    ax.set_xlabel('Minimum Angular Distance to Disk (deg)')
    ax.set_ylabel('N')
    ax.legend(numpoints=1,prop=prop,fancybox=True)
    ax.axis([0, 160, 0, 25])
    fig.savefig(plotdir + 'yng_minAngle_off_disk_hist.png')
    fig.savefig(plotdir + 'eps/yng_minAngle_off_disk_hist.eps')
    py.close()

    # Plot a histogram of the angular offset, separated by disk membership prob
    c1 = np.where(diskP >= 0.1)[0]
    c2 = np.where((diskP >= 2.7e-3) & (diskP < 0.1))[0]
    c3 = np.where(diskP < 2.7e-3)[0]
    binsIn = np.arange(0, 185, 5)
    fig = py.figure(figsize=(6,6))
    fig.subplots_adjust(left=0.15, right=0.96, top=0.95)
    fig.clf()
    ax = fig.add_subplot(111)
    ax.hist(minAngle[c1], bins=binsIn, histtype='step',color='r',
            linewidth=2,label='Disk')
    ax.hist(minAngle[c2], bins=binsIn, histtype='step',color='b',
            linewidth=2,ls='dashed',label='Candidate')
    ax.hist(minAngle[c3], bins=binsIn, histtype='step',color='k',
            linewidth=2,ls='dashdot',label='Non-Disk')
    ax.set_xlabel('Minimum Angular Distance to Disk (deg)')
    ax.set_ylabel('N')
    ax.legend(numpoints=1,prop=prop,fancybox=True)
    #ax.axis([0, 160, 0, 25])
    fig.savefig(plotdir + 'yng_minAngle_off_disk_hist_3categories.png')
    fig.savefig(plotdir + 'eps/yng_minAngle_off_disk_hist_3categories.eps')
    py.close()

    # temp
   # for cc in c1:
   #     print '%10s  %5.3f  %4.1f' % (names[cc],diskP[cc],minAngle[cc])
    # end temp

    # Plot the minimum angle as a function of radial vector
    theta = np.arctan2(y, -x) * 180.0 / np.pi
    # Need to convert this so that the angle is computed from North thru East
    theta_NE = []
    for tt in range(len(theta)):
        # First put all the data together
        if ((theta[tt] > -90.) or ((theta[tt] < 180) and (theta[tt] > 0))):
            theta0 = theta[tt] - 90.
        else:
            theta0 = theta[tt] + 270.
        theta_NE = np.concatenate([theta_NE, [theta0]])
    # Identify the stars near line of nodes that appear to contribute to warp
    nodeStars = np.array(['S3-3', 'S3-5', 'S3-10', 'S3-314', 'S3-190', 'S4-36',
                          'S4-169', 'S5-231', 'S6-81', 'S6-82', 'irs34W',
                          'S7-161','S10-32'])
    #nodeStars = np.array(['S3-3', 'S3-5', 'S3-10', 'S3-314', 'S3-190', 'S4-36',
    #                      'S4-169', 'S5-231', 'S5-237', 'S6-81', 'irs1W', 'irs34W'])
    nidx = []
    for nn in range(len(nodeStars)):
        foo = np.where(np.array(names) == nodeStars[nn])[0]
        nidx = np.concatenate([nidx, foo])
    nidx = [np.int(nn) for nn in nidx]
    fig = py.figure(figsize=(6,6))
    fig.subplots_adjust(left=0.15, right=0.96, top=0.95)
    fig.clf()
    ax = fig.add_subplot(111)
    ax.plot(theta_NE[onD], minAngle[onD], 'r.', label='Disk')
    ax.plot(theta_NE[offD], minAngle[offD], 'k.', label='Non-Disk')
    ax.plot(theta_NE[nidx], minAngle[nidx], 'go', mfc='none',mec='g',ms=7, label='Nodes')
    ax.plot([odisk,odisk],[0,160],'k--')
    ax.plot([odisk-180,odisk-180],[0,160],'k--')
    ax.set_xlabel('Radial Vector E of N (deg)')
    ax.set_ylabel('Minimum Angular Distance to Disk (deg)')
    ax.legend(numpoints=1,prop=prop,fancybox=True)
    fig.savefig(plotdir + 'yng_minAngle_vs_PA.png')
    py.close()

    # Plot the minimum angle vs. log probability
    fig.clf()
    ax = fig.add_subplot(111)
    ax.semilogx(diskP[onD],minAngle[onD],'r.',label='Disk')
    ax.semilogx(diskP[offD],minAngle[offD],'k.',label='Non-Disk')
    ax.set_xlabel('Log Probability ', fontsize=14)
    ax.set_ylabel('Minimum Angular Distance to Disk (deg)', fontsize=14)
    ax.legend(numpoints=1,fancybox=True,prop=prop,loc=2)
    ax.axis([9e-6, 1.1, 0, 180])
    fig.savefig(plotdir + 'yng_minAngle_vs_diskProb.png')
    py.close()
    usetexFalse()



def lineOfNodesStars():
    """
    Plot inclination and Omega vs. z for the sources sitting near
    the line of nodes.
    """
    orbdir = 'aorb_thesis/'
    #orbdir = 'aorb_acc_mrPDF_MC_newMosaic/'
    pdfdir = root + alnDir + orbdir
    cc = objects.Constants()

    # Stars that appear to be on the line of nodes, and thereby
    # contributing to the "warp" feature (in the middle radial bin only)
    # Selected as clockwise moving stars that have no acceleration constraints
    # and are within 1.5" from line of nodes, and in the middle radial bin
    #nodeStars = np.array(['S3-3', 'S3-5', 'S3-10', 'S3-314', 'S3-190', 'S4-36',
    #                      'S4-169', 'S5-231', 'S6-81', 'S6-82', 'irs34W',
    #                      'S7-161','S10-32'])
    nodeStars1 = np.array(['S3-3', 'S3-5', 'S3-10', 'S3-190', 'S3-314', 'S4-36'])
    nodeStars2 = np.array(['irs34W', 'S4-169','S5-231', 'S6-81', 'S6-82', 'S7-161',
                           'S10-32'])

    def dispHist(xx, yy, xxtheory, yytheory):
        fmt = 'k.'
        fmt2 = 'r--'
        pntsize = 2
        #cmap = py.cm.Blues
        #cmap = py.cm.binary
        cmap = py.cm.hot_r

        # Make 2D histogram
        (probDist, b1, b2) = h2d.histogram2d(xx, yy, bins=(50, 50))

        # Need to convert the 2d histogram into floats
        probDist = np.array(probDist, dtype=float)
        
        # Determine contour levels
        # Flatten and reverse sort our prob. distribution
        sid0 = probDist.flatten().argsort()
        sid = sid0[::-1]
        pixSort = probDist.flatten()[sid]
        
        # Make a cumulative distribution function starting from the
        # highest pixel value. This way we can find the level above
        # which 68% of the trials will fall.
        cdf = np.cumsum(pixSort)
        
        # Determine point at which we reach 68% level
        #percents = np.array([0.6827, 0.9545, 0.9973]) * len(xx)
        percents = np.array([0.6827, 0.9545]) * len(xx)
        levels = np.zeros(len(percents), dtype=float)
        for ii in range(len(levels)):
            # Get the index of the pixel at which the CDF
            # reaches this percentage (the first one found)
            idx = (np.where(cdf < percents[ii]))[0]

            # Now get the level of that pixel
            levels[ii] = pixSort[idx[-1]]
            
        # Mask out the parts where we don't have data.
        foo = np.where(probDist == 0)
        probDist = np.log10(probDist)
        probDist[foo] = 0.0
        levels = np.log10(levels)

        py.imshow(probDist, extent=[b1[0], b1[-1], b2[0], b2[-1]],
               cmap=cmap, origin='lower', aspect='auto',
               interpolation='bicubic')
        py.contour(probDist, levels, origin=None, colors='black',
                extent=[b1[0], b1[-1], b2[0], b2[-1]])


    def axisLabel(xtext, ytext, yinc=40, yrange=None, last=False, Om=False):
        # Rescales fonts for ticks and labels
        thePlot = py.gca()
        rng = py.axis()

        # Increment for ticks on the X axis
        tmp = np.abs(float(rng[1]) - float(rng[0])) / 5.0
        xinc = __builtin__.round(tmp, 1)
        xinc = 0.5
        #yinc = 30

        #if (xinc == 0):
        #    xinc = 0.05
        #    #xinc = 0.03
        
        if last == True:
            vis = True
        else:
            vis = False
        thePlot.get_xaxis().set_major_locator(py.MultipleLocator(xinc))
        py.setp( thePlot.get_xticklabels(), fontsize=tickSize, visible=vis)
        #py.setp( thePlot.get_yticklabels(), fontsize=tickSize )

        thePlot.get_yaxis().set_major_locator(py.MultipleLocator(yinc))
        if Om == True:
            thePlot.yaxis.tick_right()
        py.setp( thePlot.get_yticklabels(), fontsize=tickSize, visible=True)

        # Add axis labels
        #py.xlabel(xtext, labelFont)
        #py.ylabel(ytext, labelFont)

        # Optional re-scale axes
        if (yrange != None):
            py.axis([-0.8, 0.8, yrange[0], yrange[1]])
            #py.axis([rng[0], rng[1], yrange[0], yrange[1]])

    ### Plotting
    usetexTrue()
    labelFont1 = {'fontsize': 10, 'fontweight': 'medium'}
    labelFont2 = {'fontsize': 14, 'fontweight': 'medium'}
    labelFont3 = {'fontsize': 16, 'fontweight': 'medium'}
    tickSize = 10
    xlab = r'{\bf z (pc)}'
    ylab = r'{\bf (degrees)}'

    for ii in range(2):
        fig = py.figure(ii+1)
        py.figure(figsize=(6,8))
        py.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.07,
                           wspace=0.001, hspace=0.02)
        py.clf()
        if ii == 0:
            nodeStars = nodeStars1
            lastN = 5
        elif ii == 1:
            nodeStars = nodeStars2
            lastN = 6
            
        for nn in range(len(nodeStars)):
            star = nodeStars[nn]
            # File contains analytic orbit solutions with acceleration limits (MC)
            pdffile = '%s%s.mc.dat' % (pdfdir, star)
            pdf = pickle.load(open(pdffile))
    
            xdat = pdf.z * dist / cc.au_in_pc
            xdat2 = None
    
            ##########
            #
            # Plotting
            #
            ##########
            last = False
            if (nn == lastN):
                last = True
            # Plot Inclination
            py.subplot(len(nodeStars), 2, 2*nn+1)
            dispHist(xdat, pdf.i, xdat2, None)
    
            # Plot a line at the disk solution
            #py.plot([xdat.min(),xdat.max()],[idisk,idisk],'k--')
            py.plot([-0.8,0.8],[idisk,idisk],'k--')
            if nn == 0:
                py.title(r'${\bf {\it i}}$',labelFont3)
            axisLabel(xlab, r'${\bf {\it i}}$ {\bf (deg)}', yrange=[90, 180],
                      last=last, yinc=40)
    
            if ('irs' in star):
                starLabel = '{\\bf IRS %s}' % (star[3:])
            else:
                starLabel = '{\\bf %s}' % (star)
            #py.title(starLabel, labelFont)
            xt = -0.7
            yt = 150
            py.text(xt, yt, starLabel, labelFont1)
            if nn == lastN:
                py.xlabel(xlab, labelFont2)
            if nn == 3:
                py.ylabel(ylab, labelFont2)
    
            # Plot PA to Ascending Node
            py.subplot(len(nodeStars), 2, 2*nn+2)
            idx = (np.where(pdf.o < 0))[0]
            pdf.o[idx] += 360.0
            dispHist(xdat, pdf.o, xdat2, None)
    
            # Plot a line at the disk solution
            #py.plot([xdat.min(),xdat.max()],[odisk,odisk],'k--')
            py.plot([-0.8,0.8],[odisk,odisk],'k--')
            if nn == 0:
                py.title(r'${\bf \Omega}$',labelFont3)
            axisLabel(xlab, r'${\bf \Omega}$ {\bf (deg)}', yrange=[0, 360],
                      last=last, yinc=100, Om=True)
    
            if nn == lastN:
                py.xlabel(xlab, labelFont2)
    
        py.savefig('%spdfparams/nodeStars%d_iO_z.png' % (pdfdir, ii+1))
        py.savefig('%spdfparams/nodeStars%d_iO_z.eps' % (pdfdir, ii+1))
        py.close(ii+1)   

def combine_disk_IO_sampleWR(samples=10, radialBins=True):
    """
    Combines density maps created from sampling with replacement
    (in aorb.Disk()).  Calculates mean and standard deviation at
    each pixel from the trials that were run.

    Input:
    	samples = number of trials in the sample w/ replacement
    """
    orbDir = 'aorb_acc_mrPDF_MC/sample_wr/'

    rbins = ['r1', 'r2', 'r3']
    #rbins = ['r1', 'r2', 'r3', 'all']
    #rbins = ['all']
    nside = 64
    npix = healpy.nside2npix(nside)
    pixIdx = np.arange(0, npix)
    # Determine the significance value at the location of the
    # disk solution (given by rbinDisk)
    (i, o) = healpy.pix2ang(nside, pixIdx)
    i *= 180.0 / np.pi
    o *= 180.0 / np.pi

    disk_all = np.zeros((len(rbins), npix), dtype=float)
    disk_std = np.zeros((len(rbins), npix), dtype=float)
    peak = np.zeros((len(rbins),samples), dtype=float)
    idisk = np.zeros((len(rbins),samples), dtype=float)
    odisk = np.zeros((len(rbins),samples), dtype=float)
    diskRadius = np.zeros((len(rbins),samples), dtype=float)

    areaPerPixel = areaOnSky / npix  # in deg^2

    for rr in range(len(rbins)):
        for ss in range(samples):
            strtt = str(ss).zfill(4)   # string version

            f1 = 'disk.neighbor_%s_%s.dat' % (rbins[rr], strtt)
            f2 = 'disk.neighborStd_%s_%s.dat' % (rbins[rr], strtt)

            (disk, diskStd) = loadDiskDensity(npix, orbDir=orbDir,
                                              file1=f1, file2=f2,
                                              singlePdf=True)

            # Add all the maps together so we can take an average
            # and standard deviation in the end
            disk_all[rr,:] += disk
            disk_std[rr,:] += disk**2

            # For each sample, locate the peak and its value
            didx = disk.argmax()
            peak[rr,ss] = disk[didx]
            idisk[rr,ss] = i[didx]
            odisk[rr,ss] = o[didx]

            # Determine which pixels on the sky are "in the disk" by those
            # that have a density that is within 50% of the peak density value.
            pidx = (np.where(disk > (0.5 * disk[didx])))[0]

            # Determine the total solid angle for the "in the disk" region
            diskSolidAngle = len(pidx) * areaPerPixel
            diskRadius[rr,ss] = radiusOfCone(diskSolidAngle)

            
    disk_all /= samples
    disk_std = np.sqrt((disk_std / samples) - disk_all**2)

    # Create the Healpix maps for each radial bin, with one
    # map indicating the average at each pixel over all samples,
    # another showing the RMS error
    for rr in range(len(rbins)):
        aveFile = '%s%s%sdisk.neighbor_%s_swr_avg.dat' % \
                        (root, alnDir, orbDir, rbins[rr])
        stdFile = '%s%s%sdisk.neighbor_%s_swr_std.dat' % \
                        (root, alnDir, orbDir, rbins[rr])
        disk_all[rr].tofile(aveFile)
        disk_std[rr].tofile(stdFile)

        pdh.go(aveFile, npix, 1, plottrial=0)
        pdh.go(stdFile, npix, 1, plottrial=0)


    hdr = '%2s  %14s  %14s  %18s  %15s'
    fmt = '%2s  %6.2f +- %6.2f  %6.2f +- %6.2f  %8.3e +- %8.3e  %5.2f +- %5.2f'
    print 
    print hdr % ('rbin','Ave Incl','Ave Omega','Ave Density','Ave Disk Radius')

    for rr in range(len(rbins)):
        # Print out the average and RMS of disk location, and peak density
        print fmt % (rbins[rr], idisk[rr].mean(), idisk[rr].std(ddof=1),
                     odisk[rr].mean(), odisk[rr].std(ddof=1),
                     peak[rr].mean(), peak[rr].std(ddof=1),
                     diskRadius[rr].mean(), diskRadius[rr].std(ddof=1))



def warp_significance():
    """
    Determines the significance of the peak location of
    the disk within each radial bin.  Divides the disk
    solution for each radial bin by the error map determined
    from sampling with replacement (and calculated above in
    combine_disk_IO_sampleWR().
    """
    
    orbDir = 'aorb_acc_mrPDF_MC_newMosaic/'
    outdir = root + alnDir + orbDir 

    rbins = ['all','r1', 'r2', 'r3']
    #rbins = ['all']
    nside = 64
    npix = healpy.nside2npix(nside)
    pixIdx = np.arange(0, npix)

    rFiles = ['disk', 'inner_disk_cntrl', 'middle_disk_cntrl', 'outer_disk_cntrl']
    #rFiles = ['disk']

    hdr = '%s  %6s  %6s  %5s  %8s'
    fmt = '%s  %6.2f  %6.2f  %5.2f  %8.2e stars/deg^2'
    for rr in range(len(rbins)):
        rstd = 'sample_wr/disk.neighbor_%s_swr_std.dat' % rbins[rr]
        rdisk = rFiles[rr] + '.neighbor.dat'

        # NOTE: loadDiskDensity() requires two files be sent in, so
        # we will load this radial bin's density map and the error map
        # from sampling w/ replacement. We will then divide the former
        # by the latter to get the significance.
        (rbinDisk, dummy) = loadDiskDensity(npix, orbDir=orbDir,
                                          file1=rdisk, file2=rdisk,
                                          singlePdf=False)

        (rbinDiskStd, dummy) = loadDiskDensity(npix, orbDir=orbDir,
                                          file1=rstd, file2=rstd,
                                          singlePdf=True)

        # For signal to noise maps, we only care about the areas
        # that have signal. Otherwise, the low-signal and very-low
        # noise regions drown out everything else.  So require
        # a certain signal in order to be plotted in SNR map
        #minSignal = rbinDisk.max() / 2.0
        #bad = np.where(rbinDisk < minSignal)[0] # set these to 0
        #rbinDisk[bad] = 0.0

        # Calculate the signal to noise and create the map
        rbinSig = rbinDisk / rbinDiskStd
        rbinSigFile = outdir + rFiles[rr] + '_sig.neighbor.dat'
        rbinSig.tofile(rbinSigFile)
        pdh.go(rbinSigFile, npix, 1, plottrial=0)

        # Determine the significance value at the location of the
        # disk solution (given by rbinDisk)
        (i, o) = healpy.pix2ang(nside, pixIdx)
        i *= 180.0 / np.pi
        o *= 180.0 / np.pi

        didx = rbinDisk.argmax()
        peak = rbinDisk[didx]
        idisk = i[didx]
        odisk = o[didx]
        sigdisk = rbinSig[didx]

        print 
        print hdr % ('Bin', 'Incl', 'Omega', 'Sigma', 'Density')
        print fmt % (rbins[rr], idisk, odisk, sigdisk, peak)
        #pdb.set_trace()


def make_healpix_sampleWR(samples=10):
    """
    Creates healpix maps for all trials in the sampling
    with replacement analysis (in aorb.Disk()).
    """
    sampleDir = root + alnDir + 'aorb_acc_mrPDF_MC/sample_wr/'

    rbins = ['r1', 'r2', 'r3']
    nside = 64
    npix = healpy.nside2npix(nside)
    
    for rr in range(len(rbins)):
        for ss in range(samples):
            strtt = str(ss).zfill(4)   # string version

            f1 = '%sdisk.neighbor_%s_%s.dat' % (sampleDir, rbins[rr], strtt)

            pdh.go(f1, npix, 1, plottrial=0)


def membersVsRadius(mosaic=True,suffix='_mosaic'):
    """
    Analyze the fractional disk membership as a function of
    radius.
    """
    pdfdir = 'aorb_thesis/'
    #pdfdir = 'aorb_acc_mrPDF_MC_newMosaic/'

    # Load disk star names and probability of disk membership
    (names, diskP) = readDiskProb(diskOnly=False,suffix=suffix)

    # Load kinematic data for disk stars
    yng = loadYoungByName(names, mosaic=mosaic)

    mag = yng.getArray('mag')
    x = yng.getArray('x')
    y = yng.getArray('y')

    r = np.sqrt(x**2 + y**2)

    onIdx = (np.where(diskP >= 2.7e-3))[0]
    offIdx = (np.where(diskP < 2.7e-3))[0]

    # Print out the radius at which half the sample is enclosed
    rdx = r.argsort()
    rSorted = r[rdx]
    rAtHalf = rSorted[len(rdx)/2]
    print 'Half-Sample Radius = %5.2f' % (rAtHalf)
    print "*** Making Cut at r = 3'' ***"

    #rBinsEdges = array([0, rAtHalf, 14.0])
    #rBinsEdges = np.array([0.1, 0.8, 3, 14.0])
    #rBinsEdges = np.array([0.1, 0.8, 3.0, 7.0])
    rBinsEdges = np.array([0.8, 2.266, 3.538, 7.0])

    rBinsAvg = np.zeros(len(rBinsEdges)-1, float)
    onData = np.zeros(len(rBinsAvg), float)
    offData = np.zeros(len(rBinsAvg), float)
    fract = np.zeros(len(rBinsAvg), float)
    fractErr = np.zeros(len(rBinsAvg), float)

    histBins = np.zeros(2*len(rBinsEdges), float)
    histOn = np.zeros(2*len(rBinsEdges), float)
    histOff = np.zeros(2*len(rBinsEdges), float)

    for rr in range(len(rBinsAvg)):
        rdxAll = (np.where((r >= rBinsEdges[rr]) &
                        (r < rBinsEdges[rr+1])))[0]
        rdxOnn = (np.where((r[onIdx] >= rBinsEdges[rr]) &
                        (r[onIdx] < rBinsEdges[rr+1])))[0]
        rdxOff = (np.where((r[offIdx] >= rBinsEdges[rr]) &
                        (r[offIdx] < rBinsEdges[rr+1])))[0]


        if (len(rdxAll) > 0):
            rBinsAvg[rr] = r[rdxAll].mean()
        else:
            rBinsAvg[rr] = 0.5 * (rBinsEdges[rr] + rBinsEdges[rr+1])

        onData[rr] = len(rdxOnn)
        offData[rr] = len(rdxOff)
	# Do something special for the innermost (central arcsec)
	if (rr == 0):
	    onData[rr] = 1.0
	    offData[rr] = 11.0

	inBin = (onData[rr] + offData[rr])
        fract[rr] = onData[rr] / inBin
        fractErr[rr] = np.sqrt(onData[rr]*offData[rr] / inBin**3)
	
        # Now make histogram
        hh = (2*rr) + 1

        histBins[hh] = rBinsEdges[rr]
        histBins[hh+1] = rBinsEdges[rr+1]
        histOn[hh] = onData[rr]
        histOn[hh+1] = onData[rr]
        histOff[hh] = offData[rr]
        histOff[hh+1] = offData[rr]

        print 'Between r = [%4.1f - %4.1f]:' % \
              (rBinsEdges[rr], rBinsEdges[rr+1]),
        print ' %4.2f +/- %4.2f on disk (%3d on, %3d off, %3d total)' % \
              (fract[rr], fractErr[rr], onData[rr], offData[rr], len(rdxAll))

    histBins[0] = rBinsEdges[0]
    histBins[-1] = rBinsEdges[-1]
        
    # Calculate fractions
    fractH = histOn / (histOff + histOn)
    fractErrH = np.sqrt(histOn * histOff) / (histOn + histOff)**(3/2.0)

    fractH[0] = 0
    fractErrH[0] = 0
    fractH[-1] = 0
    fractErrH[-1] = 0

    py.clf()
    py.figure(figsize=(6,6))
    on = py.plot(histBins, histOn, 'r-', linewidth=2)
    off = py.plot(histBins, histOff, 'b-', linewidth=2)
    py.xlabel('Projected Radius (arcsec)')
    py.ylabel('Number of Stars')
    py.legend(('On Disk', 'Off Disk'))
    py.savefig(plotdir + 'membersVsRadiusOnOff.png')
    py.close()

    py.clf()
    py.figure(figsize=(6,6))
    py.semilogx(histBins, fractH, 'k-')
    py.errorbar(rBinsAvg, fract, yerr=fractErr, fmt='k.')
    py.ylim(0, 1)
    py.xlim(0.1, 15)
    py.xlabel('Projected Radius (arcsec)')
    py.ylabel('Fraction On Disk')
    py.savefig(plotdir + 'membersVsRadiusFract.png')
    py.close()

    py.clf()
    py.figure(figsize=(6,6))
    py.semilogx(histBins[1:], fractH[1:], 'k-')
    py.errorbar(rBinsAvg[1:], fract[1:], yerr=fractErr[1:], fmt='k.')
    py.ylim(0, 1)
    py.xlim(0.8, 15)
    py.xlabel('Projected Radius (arcsec)')
    py.ylabel('Fraction On Disk')
    py.savefig(plotdir + 'membersVsRadiusFract_nocent.png')
    py.close()

    py.clf()
    py.figure(figsize=(6,6))
    py.subplots_adjust(left=0.15, right=0.96, top=0.95,
                       wspace=0.25, hspace=0.25)
    py.semilogy(r,diskP,'k.')
    py.semilogy([0,14],[2.7e-3,2.7e-3],'k--') # 3 sigma cut
    py.semilogy([0,14],[0.0455,0.0455],'k--') # 2 sigma cut
    py.semilogy([0,14],[0.3173,0.3173],'k--') # 1 sigma cut
    py.xlabel('Projected Radius (arcsec)')
    py.ylabel('Disk Membership Probability')
    py.savefig(plotdir + 'membersVsR2d.png')
    py.close()


def eccVector(mosaic=True,suffix='_mosaic'):
    """
    Determine the eccentricity vector (mean, covariance, significance) for
    each star. Do this for all solutions and just disk solutions.
    """
    cc = objects.Constants()

    pdfdir = 'aorb_thesis/'
    #pdfdir = 'aorb_acc_mrPDF_MC_newMosaic/'

    # Load star names and probability of disk membership
    (names, diskP) = readDiskProb(diskOnly=False,suffix=suffix)

    starCnt = len(name)
    numTrials = 99998

    # Setup variables
    evec_avg = np.zeros((starCnt, 3), float)
    evec_covar = np.zeros((starCnt, 3, 3), float)

    evec_avg_disk = np.zeros((starCnt, 3), float)
    evec_covar_disk = np.zeros((starCnt, 3, 3), float)

    evecX = np.zeros((starCnt, numTrials), float)
    evecY = np.zeros((starCnt, numTrials), float)
    evecZ = np.zeros((starCnt, numTrials), float)
    angle = np.zeros((starCnt, numTrials), float)

    sinip = np.sin(np.radians(idisk))
    cosip = np.cos(np.radians(idisk))

    # Setup vars for the distributions
    eccStep = 0.05
    binsIn = np.arange(-1, 1, eccStep)

    _eccDat = open(root + alnDir + 'tables/eccVector'+suffix+'.dat', 'w')
    _eccDat.write('%-13s  ' % '#Name')
    _eccDat.write('%-6s ' % 'e_all')
    _eccDat.write('%13s  %13s  %13s  ' % ('(1 sigma)','(2 sigma)','(3 sigma)'))
    _eccDat.write('%-6s ' % 'e_disk')
    _eccDat.write('%13s  %13s  %13s  ' % ('(1 sigma)','(2 sigma)','(3 sigma)'))
    _eccDat.write('\n')

    # Loop through
    for ii in range(len(name)):
        # File contains analytic orbit solutions with acceleration limits (MC)
        pdffile = '%s%s%s%s.mc.dat' % (root, alnDir, pdfdir, name[ii])
        pdf = pickle.load(open(pdffile))

        # Determine angular offset to disk for each solution
        sini = np.sin(pdf.i * np.pi / 180.0)
        cosi = np.cos(pdf.i * np.pi / 180.0)
        cosodiff = np.cos( (pdf.o - odisk) * np.pi / 180.0 )
        angle[ii,:] = np.arccos( (sini * sinip * cosodiff) + (cosi * cosip) )
        angle[ii,:] *= 180.0 / np.pi

        evecX[ii,:] = pdf.evec[:,0]
        evecY[ii,:] = pdf.evec[:,1]
        evecZ[ii,:] = pdf.evec[:,2]

        evec_avg[ii] = np.average(pdf.evec)
        evec_covar[ii] = np.cov(np.transpose(pdf.evec))

        # Find all solutions within XX degrees of the disk.
        inD = (np.where(angle[ii,:] < angleCut))[0]

	if (len(inD) > 0):
	    evec_avg_disk[ii] = np.average(pdf.evec[inD])
	    evec_covar_disk[ii] = np.cov(np.transpose(pdf.evec[inD]))

	# Do a 3D histogram of the eccentricity vector to determine the
	# 1 sigma and 3 sigma lower limits.
	estep = 0.025
	ebins = np.arange(-1, 1, estep, dtype=float)
	e3D = np.zeros((len(ebins), len(ebins), len(ebins)), float)
	e3Ddisk = np.zeros((len(ebins), len(ebins), len(ebins)), float)
	eMag = np.zeros((len(ebins), len(ebins), len(ebins)), float)
	
	for xx in range(len(ebins)):
	    for yy in range(len(ebins)):
		for zz in range(len(ebins)):
		    eMag[xx, yy, zz] = np.sqrt((ebins[xx]+(estep/2.0))**2 + 
					    (ebins[yy]+(estep/2.0))**2 + 
					    (ebins[zz]+(estep/2.0))**2)

	for xx in range(len(ebins)):
	    xdx = (np.where((evecX[ii,:] > ebins[xx]) & 
			 (evecX[ii,:] < (ebins[xx] + estep))))[0]

	    if (len(xdx) == 0):
		continue
	    else:
		evecYtmp = evecY[ii,xdx]
		evecZtmp = evecZ[ii,xdx]
		angletmp = angle[ii,xdx]

	    for yy in range(len(ebins)):

		ydx = (np.where((evecYtmp > ebins[yy]) & 
			     (evecYtmp < (ebins[yy] + estep))))[0]
		
		if (len(ydx) == 0):
		    continue
		else:
		    evecZtmp2 = evecZtmp[ydx]
		    angletmp2 = angletmp[ydx]

		for zz in range(len(ebins)):
		    zdx = (np.where((evecZtmp2 > ebins[zz]) & 
				 (evecZtmp2 < (ebins[zz] + estep))))[0]

		    if (len(zdx) == 0):
			continue
		    else:
			angletmp3 = angletmp2[zdx]

		    ddx = (np.where(angletmp3 < angleCut))[0]

		    e3D[xx, yy, zz] = len(zdx)

		    if (len(ddx > 0)):
			e3Ddisk[xx, yy, zz] = len(ddx)

	print 'Totals: all orbits = %d   on disk %d' % \
              (e3D.sum(), e3Ddisk.sum())
		    
	# Find the 1, 2, 3 sigma contours
        # Determine contour levels
        # Flatten and reverse sort our prob. distribution
        sid0 = e3D.flatten().argsort()
        sid = sid0[::-1]
        eSort = e3D.flatten()[sid] # gives number of solutions per bin, starting with highest

        sid0d = e3Ddisk.flatten().argsort()
        sidd = sid0d[::-1]
        eSortDisk = e3Ddisk.flatten()[sidd]

        cdf = np.cumsum(eSort)
        cdfD = np.cumsum(eSortDisk)
        
        # Determine point at which we reach 68% level
        percents = np.array([0.6827, 0.9545, 0.9973])
        levels = np.zeros(len(percents), float)
        levelsD = np.zeros(len(percents), float)
        for jj in range(len(levels)):
            # Get the index of the pixel at which the CDF
            # reaches this percentage (the first one found)
            jdx1 = (np.where(cdf < percents[jj]*cdf[-1]))[0]

            # Now get the level of that pixel
            levels[jj] = eSort[jdx1[-1]]

	    # Same for the disks
	    jdx2 = (np.where(cdfD < percents[jj]*cdfD[-1]))[0]
	    if (len(jdx2) > 0):
		levelsD[jj] = eSortDisk[jdx2[-1]]

	# Now we have the levels. Lets print everything out:
	idx1 = (np.where(e3D.flatten() >= levels[0]))[0]
	idx2 = (np.where(e3D.flatten() >= levels[1]))[0]
	idx3 = (np.where(e3D.flatten() >= levels[2]))[0]

	ddx1 = (np.where(e3Ddisk.flatten() >= levelsD[0]))[0]
	ddx2 = (np.where(e3Ddisk.flatten() >= levelsD[1]))[0]
	ddx3 = (np.where(e3Ddisk.flatten() >= levelsD[2]))[0]	

	peakId = (np.where(e3D.flatten() == e3D.flatten().max()))[0]
	peakIdDisk = (np.where(e3Ddisk.flatten() == e3Ddisk.flatten().max()))[0]
        peakId = peakId[0]
        peakIdDisk = peakIdDisk[0]

	eflat = eMag.flatten()

	epeak = eflat[peakId]
	elo1 = eflat[idx1].min()
	ehi1 = eflat[idx1].max()
	elo2 = eflat[idx2].min()
	ehi2 = eflat[idx2].max()
	elo3 = eflat[idx3].min()
	ehi3 = eflat[idx3].max()

	if (elo1 < 0.03):
	    elo1 = 0.0
	if (elo2 < 0.03):
	    elo2 = 0.0
	if (elo3 < 0.03):
	    elo3 = 0.0
	if (ehi1 > 0.97):
	    ehi1 = 1.0
	if (ehi2 > 0.97):
	    ehi2 = 1.0
	if (ehi3 > 0.97):
	    ehi3 = 1.0
	    
	print '##### %-13s' % (name[ii])
	print '  Eccentricity Lower Limits for All  Orbital Solutions:'
	print '  Peak Eccentricity = %5.2f' % (epeak)
	print '  1 sigma = %5.2f - %5.2f' % (elo1, ehi1)
	print '  2 sigma = %5.2f - %5.2f' % (elo2, ehi2)
	print '  3 sigma = %5.2f - %5.2f' % (elo3, ehi3)

	if (len(inD) > 0):
	    epeakD = eflat[peakIdDisk]
	    elo1D = eflat[ddx1].min()
	    ehi1D = eflat[ddx1].max()
	    elo2D = eflat[ddx2].min()
	    ehi2D = eflat[ddx2].max()
	    elo3D = eflat[ddx3].min()
	    ehi3D = eflat[ddx3].max()

	    if (elo1D < 0.03):
		elo1D = 0.0
	    if (elo2D < 0.03):
		elo2D = 0.0
	    if (elo3D < 0.03):
		elo3D = 0.0
	    if (ehi1D > 0.97):
		ehi1D = 1.0
	    if (ehi2D > 0.97):
		ehi2D = 1.0
	    if (ehi3D > 0.97):
		ehi3D = 1.0

	    print '  Eccentricity Lower Limits for Disk Orbital Solutions:'
	    print '  Peak Eccentricity = %5.2f' % (epeakD)
	    print '  1 sigma = %5.2f - %5.2f' % (elo1D, ehi1D)
	    print '  2 sigma = %5.2f - %5.2f' % (elo2D, ehi2D)
	    print '  3 sigma = %5.2f - %5.2f' % (elo3D, ehi3D)
	else:
	    print '  Not in disk'

	_eccDat.write('%-13s  ' % name[ii])
	_eccDat.write('%5.2f  ' % (epeak))
	_eccDat.write('%5.2f - %5.2f  %5.2f - %5.2f  %5.2f - %5.2f  ' % \
		      (elo1, ehi1, elo2, ehi2, elo3, ehi3))

	if (len(inD) > 0):
	    _eccDat.write('%5.2f  ' % (epeakD))
	    _eccDat.write('%5.2f - %5.2f  %5.2f - %5.2f  %5.2f - %5.2f  ' % \
			  (elo1D, ehi1D, elo2D, ehi2D, elo3D, ehi3D))

	else:
	    _eccDat.write('%5.2f  ' % (-1.0))
	    _eccDat.write('%5.2f - %5.2f  %5.2f - %5.2f  %5.2f - %5.2f  ' % \
			  (-1.0, -1.0, -1.0, -1.0, -1.0, -1.0))
	_eccDat.write('\n')

    _eccDat.close()



def diskEccVector(mosaic=True, suffix='_mosaic'):
    """
    Determine the eccentricity vector (mean, covariance, significance) for
    each star. Do this for all solutions and just disk solutions.
    """
    pdfdir = 'aorb_thesis/'
    #pdfdir = 'aorb_acc_mrPDF_MC_newMosaic/'

    # Load disk star names and probability of disk membership
    (names, diskP) = readDiskProb(diskOnly=False,suffix=suffix)

    idx = np.where(diskP > 2.7e-3)[0] # candidate disk members
    names = [names[ii] for ii in idx]
    diskP = diskP[idx]
    
    # Load kinematic data for disk stars
    yng = loadYoungByName(names, mosaic=mosaic)
    
    starCnt = len(yng.stars)
    numTrials = 99998

    # Setup variables
    evec_avg = np.zeros((starCnt, 3), float)
    evec_covar = np.zeros((starCnt, 3, 3), float)

    evec_avg_disk = np.zeros((starCnt, 3), float)
    evec_covar_disk = np.zeros((starCnt, 3, 3), float)

    evecX = np.zeros((starCnt, numTrials), float)
    evecY = np.zeros((starCnt, numTrials), float)
    evecZ = np.zeros((starCnt, numTrials), float)
    angle = np.zeros((starCnt, numTrials), float)

    sinip = np.sin(np.radians(idisk))
    cosip = np.cos(np.radians(idisk))

    _eccDat = open(root + alnDir + 'tables/diskEccVector'+suffix+'.dat', 'w')
    _eccDat.write('%-13s  ' % 'Name')
    _eccDat.write('%-6s ' % 'e_all')
    _eccDat.write('%13s  %13s  %13s  ' % ('(1 sigma)','(2 sigma)','(3 sigma)'))
    _eccDat.write('%-6s ' % 'e_disk')
    _eccDat.write('%13s  %13s  %13s  ' % ('(1 sigma)','(2 sigma)','(3 sigma)'))
    _eccDat.write('\n')

    r2d_all = []
    # Loop through and trim down to only disk stars
    for ii in range(len(idx)):
        # File contains analytic orbit solutions with acceleration limits (MC)
        starName = yng.stars[ii].name
        r2d = yng.stars[ii].r2d
        r2d_all = np.concatenate([r2d_all, [r2d]])

        print "Loading eccentricities for ", starName

        pdffile = '%s%s%s%s.mc.dat' % (root, alnDir, pdfdir, starName)
        pdf = pickle.load(open(pdffile))

        # Determine angular offset to disk for each solution
        sini = np.sin(pdf.i * np.pi / 180.0)
        cosi = np.cos(pdf.i * np.pi / 180.0)
        cosodiff = np.cos( (pdf.o - odisk) * np.pi / 180.0 )
        angle[ii,:] = np.arccos( (sini * sinip * cosodiff) + (cosi * cosip) )
        angle[ii,:] *= 180.0 / np.pi

        evecX[ii,:] = pdf.evec[:,0]
        evecY[ii,:] = pdf.evec[:,1]
        evecZ[ii,:] = pdf.evec[:,2]

        evec_avg[ii] = np.average(pdf.evec)
        evec_covar[ii] = np.cov(np.transpose(pdf.evec))

        # Find all solutions within XX degrees of the disk.
        inD = np.where(angle[ii,:] < angleCut)[0]
        evec_avg_disk[ii] = np.average(pdf.evec[inD])
        evec_covar_disk[ii] = np.cov(np.transpose(pdf.evec[inD]))

	# Do a 3D histogram of the eccentricity vector to determine the
	# 1 sigma and 3 sigma lower limits.
	estep = 0.025
	ebins = np.arange(-1, 1, estep, dtype=float)
	e3D = np.zeros((len(ebins), len(ebins), len(ebins)), float)
	e3Ddisk = np.zeros((len(ebins), len(ebins), len(ebins)), float)
	eMag = np.zeros((len(ebins), len(ebins), len(ebins)), float)
	
	for xx in range(len(ebins)):
	    for yy in range(len(ebins)):
		for zz in range(len(ebins)):
		    eMag[xx, yy, zz] = np.sqrt((ebins[xx]+(estep/2.0))**2 + 
					    (ebins[yy]+(estep/2.0))**2 + 
					    (ebins[zz]+(estep/2.0))**2)

	for xx in range(len(ebins)):
	    xdx = (np.where((evecX[ii,:] > ebins[xx]) & 
			 (evecX[ii,:] < (ebins[xx] + estep))))[0]

	    if (len(xdx) == 0):
		continue
	    else:
		evecYtmp = evecY[ii,xdx]
		evecZtmp = evecZ[ii,xdx]
		angletmp = angle[ii,xdx]

	    for yy in range(len(ebins)):

		ydx = (np.where((evecYtmp > ebins[yy]) & 
			     (evecYtmp < (ebins[yy] + estep))))[0]
		
		if (len(ydx) == 0):
		    continue
		else:
		    evecZtmp2 = evecZtmp[ydx]
		    angletmp2 = angletmp[ydx]

		for zz in range(len(ebins)):
		    zdx = (np.where((evecZtmp2 > ebins[zz]) & 
				 (evecZtmp2 < (ebins[zz] + estep))))[0]

		    if (len(zdx) == 0):
			continue
		    else:
			angletmp3 = angletmp2[zdx]

		    ddx = (np.where(angletmp3 < angleCut))[0]

		    e3D[xx, yy, zz] = len(zdx)
		    e3Ddisk[xx, yy, zz] = len(ddx)

	print 'Totals: ', e3D.sum(), e3Ddisk.sum()
		    
	# Find the 1, 2, 3 sigma contours
        # Determine contour levels
        # Flatten and reverse sort our prob. distribution
        sid0 = e3D.flatten().argsort()
        sid = sid0[::-1]
        eSort = e3D.flatten()[sid]

        sid0d = e3Ddisk.flatten().argsort()
        sidd = sid0d[::-1]
        eSortDisk = e3Ddisk.flatten()[sidd]

        cdf = np.cumsum(eSort)
        cdfD = np.cumsum(eSortDisk)
        
        # Determine point at which we reach 68% level
        percents = np.array([0.6827, 0.9545, 0.9973])
        levels = np.zeros(len(percents), float)
        levelsD = np.zeros(len(percents), float)
        for jj in range(len(levels)):
            # Get the index of the pixel at which the CDF
            # reaches this percentage (the first one found)
            jdx1 = (np.where(cdf < percents[jj]*cdf[-1]))[0]
            jdx2 = (np.where(cdfD < percents[jj]*cdfD[-1]))[0]

            # Now get the level of that pixel
            levels[jj] = eSort[jdx1[-1]]
            levelsD[jj] = eSortDisk[jdx2[-1]]


	# Now we have the levels. Lets print everything out:
	idx1 = (np.where(e3D.flatten() >= levels[0]))[0]
	idx2 = (np.where(e3D.flatten() >= levels[1]))[0]
	idx3 = (np.where(e3D.flatten() >= levels[2]))[0]

	ddx1 = (np.where(e3Ddisk.flatten() >= levelsD[0]))[0]
	ddx2 = (np.where(e3Ddisk.flatten() >= levelsD[1]))[0]
	ddx3 = (np.where(e3Ddisk.flatten() >= levelsD[2]))[0]	

	peakId = (np.where(e3D.flatten() == e3D.flatten().max()))[0]
	peakIdDisk = (np.where(e3Ddisk.flatten() == e3Ddisk.flatten().max()))[0]
	#print peakId
	#print peakIdDisk

        peakId = peakId[0]
        peakIdDisk = peakIdDisk[0]

	eflat = eMag.flatten()
	print '##### %-13s' % (yng.stars[ii].name)
	print '  Eccentricity Lower Limits for All  Orbital Solutions:'
	print '  Peak Eccentricity = %5.2f' % \
	    (eflat[peakId])
	print '  1 sigma = %5.2f - %5.2f' % \
	    (eflat[idx1].min(), eflat[idx1].max())
	print '  2 sigma = %5.2f - %5.2f' % \
	    (eflat[idx2].min(), eflat[idx2].max())
	print '  3 sigma = %5.2f - %5.2f' % \
	    (eflat[idx3].min(), eflat[idx3].max())

	if (len(inD) > 0):
	    print '  Eccentricity Lower Limits for Disk Orbital Solutions:'
	    print '  Peak Eccentricity = %5.2f' % \
		(eflat[peakIdDisk])
	    print '  1 sigma = %5.2f - %5.2f' % \
		(eflat[ddx1].min(), eflat[ddx1].max())
	    print '  2 sigma = %5.2f - %5.2f' % \
		(eflat[ddx2].min(), eflat[ddx2].max())
	    print '  3 sigma = %5.2f - %5.2f' % \
		(eflat[ddx3].min(), eflat[ddx3].max())
	else:
	    print '  Not in disk'

	_eccDat.write('%-13s  ' % yng.stars[ii].name)
	_eccDat.write('%5.2f  ' % eflat[peakId])
	_eccDat.write('%5.2f - %5.2f  %5.2f - %5.2f  %5.2f - %5.2f  ' % \
		      (eflat[idx1].min(), eflat[idx1].max(),
		       eflat[idx2].min(), eflat[idx2].max(),
		       eflat[idx3].min(), eflat[idx3].max()))
	if (len(inD) > 0):
	    _eccDat.write('%5.2f  ' % eflat[peakIdDisk])
	    _eccDat.write('%5.2f - %5.2f  %5.2f - %5.2f  %5.2f - %5.2f  ' % \
			  (eflat[ddx1].min(), eflat[ddx1].max(),
			   eflat[ddx2].min(), eflat[ddx2].max(),
			   eflat[ddx3].min(), eflat[ddx3].max()))
	else:
	    _eccDat.write('%5.2f  ' % (-1.0))
	    _eccDat.write('%5.2f - %5.2f  %5.2f - %5.2f  %5.2f - %5.2f  ' % \
			  (-1.0, -1.0, -1.0, -1.0, -1.0, -1.0))
	_eccDat.write('\n')

    _eccDat.close()

    # Now calculate the average and standard deviation
    # For each trial calculate the:
    #   - avg
    #   - std
    #   - rms
    avg_trialX = np.sum(evecX, axis=0) / starCnt
    avg_trialY = np.sum(evecY, axis=0) / starCnt
    avg_trialZ = np.sum(evecZ, axis=0) / starCnt
    rms_trialX = np.sqrt(np.sum(evecX**2, axis=0) / starCnt)
    rms_trialY = np.sqrt(np.sum(evecY**2, axis=0) / starCnt)
    rms_trialZ = np.sqrt(np.sum(evecZ**2, axis=0) / starCnt)
    std_trialX = np.sqrt(np.sum((evecX - avg_trialX)**2, axis=0) / (starCnt - 1))
    std_trialY = np.sqrt(np.sum((evecY - avg_trialY)**2, axis=0) / (starCnt - 1))
    std_trialZ = np.sqrt(np.sum((evecZ - avg_trialZ)**2, axis=0) / (starCnt - 1))

    print evecX.shape, evecY.shape, evecZ.shape
    print rms_trialX.shape, rms_trialY.shape, rms_trialZ.shape

    # For all trials calculate the average, rms, and std for each
    avg_avgX = avg_trialX.mean()
    avg_avgY = avg_trialY.mean()
    avg_avgZ = avg_trialZ.mean()
    avg_rmsX = np.sqrt( np.sum(avg_trialX**2) / numTrials )
    avg_rmsY = np.sqrt( np.sum(avg_trialY**2) / numTrials )
    avg_rmsZ = np.sqrt( np.sum(avg_trialZ**2) / numTrials )
    avg_stdX = np.sqrt( np.sum((avg_trialX - avg_avgX)**2) / (numTrials - 1) )
    avg_stdY = np.sqrt( np.sum((avg_trialY - avg_avgY)**2) / (numTrials - 1) )
    avg_stdZ = np.sqrt( np.sum((avg_trialZ - avg_avgZ)**2) / (numTrials - 1) )

    rms_avgX = rms_trialX.mean()
    rms_avgY = rms_trialY.mean()
    rms_avgZ = rms_trialZ.mean()
    rms_rmsX = np.sqrt( np.sum(rms_trialX**2) / numTrials )
    rms_rmsY = np.sqrt( np.sum(rms_trialY**2) / numTrials )
    rms_rmsZ = np.sqrt( np.sum(rms_trialZ**2) / numTrials )
    rms_stdX = np.sqrt( np.sum((rms_trialX - rms_avgX)**2) / (numTrials - 1) )
    rms_stdY = np.sqrt( np.sum((rms_trialY - rms_avgY)**2) / (numTrials - 1) )
    rms_stdZ = np.sqrt( np.sum((rms_trialZ - rms_avgZ)**2) / (numTrials - 1) )

    std_avgX = std_trialX.mean()
    std_avgY = std_trialY.mean()
    std_avgZ = std_trialZ.mean()
    std_rmsX = np.sqrt( np.sum(std_trialX**2) / numTrials )
    std_rmsY = np.sqrt( np.sum(std_trialY**2) / numTrials )
    std_rmsZ = np.sqrt( np.sum(std_trialZ**2) / numTrials )
    std_stdX = np.sqrt( np.sum((std_trialX - std_avgX)**2) / (numTrials - 1) )
    std_stdY = np.sqrt( np.sum((std_trialY - std_avgY)**2) / (numTrials - 1) )
    std_stdZ = np.sqrt( np.sum((std_trialZ - std_avgZ)**2) / (numTrials - 1) )

    print 'All Disk Stars, All Solutions:'
    print '  Ecc X:'
    print '  Avg = %5.2f +- %5.2f' % (avg_avgX, avg_stdX)
    print '  RMS = %5.2f +- %5.2f' % (rms_avgX, rms_stdX)
    print '  Std = %5.2f +- %5.2f' % (std_avgX, std_stdX)
    print '  Ecc Y:'
    print '  Avg = %5.2f +- %5.2f' % (avg_avgY, avg_stdY)
    print '  RMS = %5.2f +- %5.2f' % (rms_avgY, rms_stdY)
    print '  Std = %5.2f +- %5.2f' % (std_avgY, std_stdY)
    print '  Ecc Z:'
    print '  Avg = %5.2f +- %5.2f' % (avg_avgZ, avg_stdZ)
    print '  RMS = %5.2f +- %5.2f' % (rms_avgZ, rms_stdZ)
    print '  Std = %5.2f +- %5.2f' % (std_avgZ, std_stdZ)
    print ' Means: %5.2f  %5.2f  %5.2f' % \
          (evecX.mean(), evecY.mean(), evecZ.mean())

    ##########
    #
    # Now only use disk solutions
    #   -- assumes disk solutions are within X degrees
    #   -- weights by probability of being in the disk
    #
    ##########
    # Loop through and trim down to only disk stars
    evecXdisk = np.arange(0, dtype=float)
    evecYdisk = np.arange(0, dtype=float)
    evecZdisk = np.arange(0, dtype=float)
    evecXdiskStars = []
    evecYdiskStars = []
    evecZdiskStars = []
    numOrbitsOnDisk = np.zeros(starCnt)

    avg_trial_diskX = np.zeros(numTrials, float)
    avg_trial_diskY = np.zeros(numTrials, float)
    avg_trial_diskZ = np.zeros(numTrials, float)
    rms_trial_diskX = np.zeros(numTrials, float)
    rms_trial_diskY = np.zeros(numTrials, float)
    rms_trial_diskZ = np.zeros(numTrials, float)
    std_trial_diskX = np.zeros(numTrials, float)
    std_trial_diskY = np.zeros(numTrials, float)
    std_trial_diskZ = np.zeros(numTrials, float)

    oldNumTrials = numTrials

    for nn in range(numTrials):
        adx = (np.where(angle[:,nn] < angleCut))[0]
        trialX = evecX[adx,nn]
        trialY = evecY[adx,nn]
        trialZ = evecZ[adx,nn]

        if (len(adx) < 2):
            numTrials -= 1.0
            continue

        avg_trial_diskX[nn] = trialX.mean()
        avg_trial_diskY[nn] = trialY.mean()
        avg_trial_diskZ[nn] = trialZ.mean()
        rms_trial_diskX[nn] = np.sqrt( np.sum(trialX**2) / len(trialX) )
        rms_trial_diskY[nn] = np.sqrt( np.sum(trialY**2) / len(trialY) )
        rms_trial_diskZ[nn] = np.sqrt( np.sum(trialZ**2) / len(trialZ) )
        std_trial_diskX[nn] = trialX.std()
        std_trial_diskY[nn] = trialY.std()
        std_trial_diskZ[nn] = trialZ.std()

    print 'Found %d trials with less than 2 stars in the disk' % \
          (oldNumTrials - numTrials)

    for ii in range(starCnt):
        idx = (np.where(angle[ii,:] < angleCut))[0]
        evecXdisk = np.concatenate((evecXdisk, evecX[ii,idx]))
        evecYdisk = np.concatenate((evecYdisk, evecY[ii,idx]))
        evecZdisk = np.concatenate((evecZdisk, evecZ[ii,idx]))

        evecXdiskStars.append(evecX[ii,idx])
        evecYdiskStars.append(evecY[ii,idx])
        evecZdiskStars.append(evecZ[ii,idx])

        numOrbitsOnDisk[ii] = len(idx)

    print numTrials, len(evecXdisk)

    # For all trials calculate the average, rms, and std for each
    avg_avgX = np.sum( avg_trial_diskX ) / numTrials
    avg_avgY = np.sum( avg_trial_diskY ) / numTrials
    avg_avgZ = np.sum( avg_trial_diskZ ) / numTrials
    avg_rmsX = np.sqrt( np.sum(avg_trial_diskX**2) / numTrials )
    avg_rmsY = np.sqrt( np.sum(avg_trial_diskY**2) / numTrials )
    avg_rmsZ = np.sqrt( np.sum(avg_trial_diskZ**2) / numTrials )
    avg_stdX = np.sqrt( np.sum((avg_trial_diskX - avg_avgX)**2) / (numTrials - 1) )
    avg_stdY = np.sqrt( np.sum((avg_trial_diskY - avg_avgY)**2) / (numTrials - 1) )
    avg_stdZ = np.sqrt( np.sum((avg_trial_diskZ - avg_avgZ)**2) / (numTrials - 1) )

    rms_avgX = np.sum( rms_trial_diskX ) / numTrials
    rms_avgY = np.sum( rms_trial_diskY ) / numTrials
    rms_avgZ = np.sum( rms_trial_diskZ ) / numTrials
    rms_rmsX = np.sqrt( np.sum(rms_trial_diskX**2) / numTrials )
    rms_rmsY = np.sqrt( np.sum(rms_trial_diskY**2) / numTrials )
    rms_rmsZ = np.sqrt( np.sum(rms_trial_diskZ**2) / numTrials )
    rms_stdX = np.sqrt( np.sum((rms_trial_diskX - rms_avgX)**2) / (numTrials - 1) )
    rms_stdY = np.sqrt( np.sum((rms_trial_diskY - rms_avgY)**2) / (numTrials - 1) )
    rms_stdZ = np.sqrt( np.sum((rms_trial_diskZ - rms_avgZ)**2) / (numTrials - 1) )

    std_avgX = np.sum( std_trial_diskX ) / numTrials
    std_avgY = np.sum( std_trial_diskY ) / numTrials
    std_avgZ = np.sum( std_trial_diskZ ) / numTrials
    std_rmsX = np.sqrt( np.sum(std_trial_diskX**2) / numTrials )
    std_rmsY = np.sqrt( np.sum(std_trial_diskY**2) / numTrials )
    std_rmsZ = np.sqrt( np.sum(std_trial_diskZ**2) / numTrials )
    std_stdX = np.sqrt( np.sum((std_trial_diskX - std_avgX)**2) / (numTrials - 1) )
    std_stdY = np.sqrt( np.sum((std_trial_diskY - std_avgY)**2) / (numTrials - 1) )
    std_stdZ = np.sqrt( np.sum((std_trial_diskZ - std_avgZ)**2) / (numTrials - 1) )
    print 'All Disk Stars, Disk Solutions, Weighted by Disk Prob:'
    print '  Ecc X:'
    print '  Avg = %5.2f +- %5.2f' % (avg_avgX, avg_stdX)
    print '  RMS = %5.2f +- %5.2f' % (rms_avgX, rms_stdX)
    print '  Std = %5.2f +- %5.2f' % (std_avgX, std_stdX)
    print '  Ecc Y:'
    print '  Avg = %5.2f +- %5.2f' % (avg_avgY, avg_stdY)
    print '  RMS = %5.2f +- %5.2f' % (rms_avgY, rms_stdY)
    print '  Std = %5.2f +- %5.2f' % (std_avgY, std_stdY)
    print '  Ecc Z:'
    print '  Avg = %5.2f +- %5.2f' % (avg_avgZ, avg_stdZ)
    print '  RMS = %5.2f +- %5.2f' % (rms_avgZ, rms_stdZ)
    print '  Std = %5.2f +- %5.2f' % (std_avgZ, std_stdZ)
    print ' Means: %5.2f  %5.2f  %5.2f' % \
          (evecXdisk.mean(), evecYdisk.mean(), evecZdisk.mean())

    # Save into file to be plotted later with plotDiskEccVector()
    _file = open(root + alnDir + 'tables/diskEccVector'+suffix+'.pickle', 'w')
    pickle.dump(starCnt, _file)
    pickle.dump(names, _file)
    pickle.dump(r2d_all, _file)
    pickle.dump(diskP, _file)
    pickle.dump(numOrbitsOnDisk, _file)
    pickle.dump(evecX, _file)
    pickle.dump(evecY, _file)
    pickle.dump(evecZ, _file)
    pickle.dump(evecXdisk, _file)
    pickle.dump(evecYdisk, _file)
    pickle.dump(evecZdisk, _file)
    pickle.dump(evecXdiskStars, _file)
    pickle.dump(evecYdiskStars, _file)
    pickle.dump(evecZdiskStars, _file)
    _file.close()


def plotDiskEccVector(suffix='_mosaic'):
    pdfdir = 'aorb_thesis/'
    #pdfdir = 'aorb_acc_mrPDF_MC_newMosaic/'

    _file = open(root + alnDir + 'tables/diskEccVector'+suffix+'.pickle', 'r')

    starCnt = pickle.load(_file)
    names = pickle.load(_file)
    r2d = pickle.load(_file)
    diskP = pickle.load(_file)
    numOrbitsOnDisk = pickle.load(_file)
    evecX = pickle.load(_file)
    evecY = pickle.load(_file)
    evecZ = pickle.load(_file)
    evecXdisk = pickle.load(_file)
    evecYdisk = pickle.load(_file)
    evecZdisk = pickle.load(_file)
    evecXdiskStars = pickle.load(_file)
    evecYdiskStars = pickle.load(_file)
    evecZdiskStars = pickle.load(_file)
    _file.close()

    # Get the accelerating sources, for plotting later
    _acc = asciidata.open(root + alnDir + 'tables/accelerating_sources.dat')
    accels = _acc[0].tonumpy()
    accNames = [aa.strip() for aa in accels] 

    # Get the stars w/ acceleration upper limits, for plotting later
    _upp = asciidata.open(root + alnDir + 'tables/accel_upperLimit_sources.dat')
    upper = _upp[0].tonumpy()
    uppNames = [uu.strip() for uu in upper] 

    #acc_names = np.concatenate([accNames, uppNames])
    acc_names = accNames

    # Setup vars for the distributions
    eccStep = 0.05
    binsIn = np.arange(-1, 1+eccStep, eccStep)

    #
    #  Make 1D histograms of each of the eccentricity vector components
    #  with 3 different variations:
    #     1) each star has even weight when combining
    #     2) weight each stars contribution by the likelihood of
    #        disk membership, rather than probability of
    #     3) weight each stars contribution by the probability
    #        that it is on the disk (not normalized by star's solid angle)
    #
    histXnone = None
    histYnone = None
    histZnone = None
    histXprob = None
    histYprob = None
    histZprob = None
    histXlike = None
    histYlike = None
    histZlike = None

    histXnoneAcc = None
    histYnoneAcc = None
    histZnoneAcc = None
    histXprobAcc = None
    histYprobAcc = None
    histZprobAcc = None
    histXlikeAcc = None
    histYlikeAcc = None
    histZlikeAcc = None

    numAcc = 0
    for ii in range(starCnt):
            
        eccXForStar = evecXdiskStars[ii]
        eccYForStar = evecYdiskStars[ii]
        eccZForStar = evecZdiskStars[ii]
        
        (bbx, nnX) = histNofill.hist(binsIn, eccXForStar, normed=True)
        (bby, nnY) = histNofill.hist(binsIn, eccYForStar, normed=True)
        (bbz, nnZ) = histNofill.hist(binsIn, eccZForStar, normed=True)

        #if names[ii] in accNames:
        if names[ii] in acc_names:
            (bbxAcc, nnXAcc) = histNofill.hist(binsIn, eccXForStar, normed=True)
            (bbyAcc, nnYAcc) = histNofill.hist(binsIn, eccYForStar, normed=True)
            (bbzAcc, nnZAcc) = histNofill.hist(binsIn, eccZForStar, normed=True)
            numAcc += 1

        if (histXnone == None):
            # No weights
            histXnone = nnX
            histYnone = nnY
            histZnone = nnZ

            # Weight by likelihood
            histXlike = nnX * diskP[ii]
            histYlike = nnY * diskP[ii]
            histZlike = nnZ * diskP[ii]

            # Weight by probability on disk
            histXprob = nnX * numOrbitsOnDisk[ii]
            histYprob = nnY * numOrbitsOnDisk[ii]
            histZprob = nnZ * numOrbitsOnDisk[ii]

            #if (names[ii] in accNames) and (diskP[ii] > 2.7e-3):
            if (names[ii] in acc_names) and (diskP[ii] > 2.7e-3):
                print 'Accelerating star: %s, disk prob: %6.1e' % (names[ii], diskP[ii])
                # Accelerating sources only:
                # No weights
                histXnoneAcc = nnXAcc
                histYnoneAcc = nnYAcc
                histZnoneAcc = nnZAcc

                # Weight by likelihood
                histXlikeAcc = nnXAcc * diskP[ii]
                histYlikeAcc = nnYAcc * diskP[ii]
                histZlikeAcc = nnZAcc * diskP[ii]
                diskPAcc = diskP[ii]
    
                # Weight by probability on disk
                histXprobAcc = nnXAcc * numOrbitsOnDisk[ii]
                histYprobAcc = nnYAcc * numOrbitsOnDisk[ii]
                histZprobAcc = nnZAcc * numOrbitsOnDisk[ii]
                numOrbitsOnDiskAcc = numOrbitsOnDisk[ii]

        else:
            # No weights
            histXnone += nnX
            histYnone += nnY
            histZnone += nnZ
            
            # Weight by likelihood
            histXlike += nnX * diskP[ii]
            histYlike += nnY * diskP[ii]
            histZlike += nnZ * diskP[ii]
            
            # Weight by probability on disk
            histXprob += nnX * numOrbitsOnDisk[ii]
            histYprob += nnY * numOrbitsOnDisk[ii]
            histZprob += nnZ * numOrbitsOnDisk[ii]
            
            #if (names[ii] in accNames) and (diskP[ii] > 2.7e-3):
            if (names[ii] in acc_names) and (diskP[ii] > 2.7e-3):
                # Accelerating sources only:
                # No weights
                histXnoneAcc += nnXAcc
                histYnoneAcc += nnYAcc
                histZnoneAcc += nnZAcc
                
                # Weight by likelihood
                histXlikeAcc += nnXAcc * diskP[ii]
                histYlikeAcc += nnYAcc * diskP[ii]
                histZlikeAcc += nnZAcc * diskP[ii]
                diskPAcc += diskP[ii]
                
                # Weight by probability on disk
                histXprobAcc += nnXAcc * numOrbitsOnDisk[ii]
                histYprobAcc += nnYAcc * numOrbitsOnDisk[ii]
                histZprobAcc += nnZAcc * numOrbitsOnDisk[ii]
                numOrbitsOnDiskAcc += numOrbitsOnDisk[ii]

    histXnone /= starCnt
    histYnone /= starCnt
    histZnone /= starCnt

    histXlike /= diskP.sum()
    histYlike /= diskP.sum()
    histZlike /= diskP.sum()

    # Jessica had the following 3 lines, but this is wrong
    #histXlike /= numOrbitsOnDisk.sum()
    #histYlike /= numOrbitsOnDisk.sum()
    #histZlike /= numOrbitsOnDisk.sum()

    # Should be histXprob instead
    histXprob /= numOrbitsOnDisk.sum()
    histYprob /= numOrbitsOnDisk.sum()
    histZprob /= numOrbitsOnDisk.sum()

    # Accelerating sources only:
    histXnoneAcc /= numAcc
    histYnoneAcc /= numAcc
    histZnoneAcc /= numAcc

    histXlikeAcc /= diskPAcc.sum()
    histYlikeAcc /= diskPAcc.sum()
    histZlikeAcc /= diskPAcc.sum()

    histXprobAcc /= numOrbitsOnDiskAcc.sum()
    histYprobAcc /= numOrbitsOnDiskAcc.sum()
    histZprobAcc /= numOrbitsOnDiskAcc.sum()

    # 
    # Plot all possible solutions (no weights)
    #
    py.clf()
    py.figure(figsize=(6,6))
    py.subplots_adjust(left=0.2, right=0.96, top=0.9)
    (bins, endisp1x) = histNofill.hist(binsIn, evecX, normed=True)
    (fooo, endisp1y) = histNofill.hist(binsIn, evecY, normed=True)
    (fooo, endisp1z) = histNofill.hist(binsIn, evecZ, normed=True)

    xx = py.plot(bins, endisp1x, color='black')
    yy = py.plot(bins, endisp1y, color='red')
    zz = py.plot(bins, endisp1z, color='blue')
    py.xlabel('Eccentricity Vector')
    py.ylabel('PDF')
    py.legend((xx, yy, zz), ('X', 'Y', 'Z'))
    py.xlim(-1, 1)
    py.savefig(plotdir + 'evec_all%s.png' % suffix)
    py.savefig(plotdir + 'eps/evec_all%s.eps' % suffix)
    py.close()

    #
    # Plot only disk solutions (no weights)
    #
    py.clf()
    py.figure(figsize=(6,6))
    py.subplots_adjust(left=0.2, right=0.96, top=0.9)
    xx = py.plot(bins, histXnone, color='black')
    yy = py.plot(bins, histYnone, color='red')
    zz = py.plot(bins, histZnone, color='blue')
    py.xlabel('Eccentricity Vector')
    py.ylabel('PDF')
    py.title('Unweighted')
    py.legend((xx, yy, zz), ('X', 'Y', 'Z'))
    py.xlim(-1, 1)
    py.savefig(plotdir + 'evec_disk_none%s.png' % suffix)
    py.savefig(plotdir + 'eps/evec_disk_none%s.eps' % suffix)
    py.close()

    #
    # Plot only disk solutions (probability on disk)
    #
    py.clf()
    py.figure(figsize=(6,6))
    py.subplots_adjust(left=0.2, right=0.96, top=0.9)
    xx = py.plot(bins, histXprob, color='black')
    yy = py.plot(bins, histYprob, color='red')
    zz = py.plot(bins, histZprob, color='blue')
    py.xlabel('Eccentricity Vector')
    py.ylabel('PDF')
    py.title('Weighted by P(on disk)')
    py.legend((xx, yy, zz), ('X', 'Y', 'Z'))
    py.xlim(-1, 1)
    py.savefig(plotdir + 'evec_disk_prob%s.png' % suffix)
    py.savefig(plotdir + 'eps/evec_disk_prob%s.eps' % suffix)
    py.close()

    #
    # Plot only disk solutions (likelihood on disk)
    #
    py.clf()
    py.figure(figsize=(6,6))
    py.subplots_adjust(left=0.2, right=0.96, top=0.9)
    xx = py.plot(bins, histXlike, color='black')
    yy = py.plot(bins, histYlike, color='red')
    zz = py.plot(bins, histZlike, color='blue')
    py.xlabel('Eccentricity Vector')
    py.ylabel('PDF')
    py.title('Weighted by L(on disk)')
    py.legend((xx, yy, zz), ('X', 'Y', 'Z'))
    py.xlim(-1, 1)
    py.savefig(plotdir + 'evec_disk_like%s.png' % suffix)
    py.savefig(plotdir + 'eps/evec_disk_like%s.eps' % suffix)
    py.close()

    #####
    # Accelerating sources only
    #####
    #
    # Plot only disk solutions (no weights)
    #
    py.clf()
    py.figure(figsize=(6,6))
    py.subplots_adjust(left=0.2, right=0.96, top=0.9)
    xx = py.plot(bins, histXnoneAcc, color='black')
    yy = py.plot(bins, histYnoneAcc, color='red')
    zz = py.plot(bins, histZnoneAcc, color='blue')
    py.xlabel('Eccentricity Vector')
    py.ylabel('PDF')
    py.title('Unweighted')
    py.legend((xx, yy, zz), ('X', 'Y', 'Z'))
    py.xlim(-1, 1)
    py.savefig(plotdir + 'evec_accel_disk_none%s.png' % suffix)
    py.savefig(plotdir + 'eps/evec_accel_disk_none%s.eps' % suffix)
    py.close()

    #
    # Plot only disk solutions (probability on disk)
    #
    py.clf()
    py.figure(figsize=(6,6))
    py.subplots_adjust(left=0.2, right=0.96, top=0.9)
    xx = py.plot(bins, histXprobAcc, color='black')
    yy = py.plot(bins, histYprobAcc, color='red')
    zz = py.plot(bins, histZprobAcc, color='blue')
    py.xlabel('Eccentricity Vector')
    py.ylabel('PDF')
    py.title('Weighted by P(on disk)')
    py.legend((xx, yy, zz), ('X', 'Y', 'Z'))
    py.xlim(-1, 1)
    py.savefig(plotdir + 'evec_accel_disk_prob%s.png' % suffix)
    py.savefig(plotdir + 'eps/evec_accel_disk_prob%s.eps' % suffix)
    py.close()

    #
    # Plot only disk solutions (likelihood on disk)
    #
    py.clf()
    py.figure(figsize=(6,6))
    py.subplots_adjust(left=0.2, right=0.96, top=0.9)
    xx = py.plot(bins, histXlikeAcc, color='black')
    yy = py.plot(bins, histYlikeAcc, color='red')
    zz = py.plot(bins, histZlikeAcc, color='blue')
    py.xlabel('Eccentricity Vector')
    py.ylabel('PDF')
    py.title('Weighted by L(on disk)')
    py.legend((xx, yy, zz), ('X', 'Y', 'Z'))
    py.xlim(-1, 1)
    py.savefig(plotdir + 'evec_accel_disk_like%s.png' % suffix)
    py.savefig(plotdir + 'eps/evec_accel_disk_like%s.eps' % suffix)
    py.close()


    ##########
    #
    # Convert to Disk Plane
    #
    ##########
    # Switch everything into disk coordinate system. Coordinates
    # are p, q, n where
    #   n is tangential to the disk plane.
    #   p goes along the ascending node in the disk plane.
    #   q is perpindicular to p in the disk plane.
    # Get the directional vectors
    (eP, eQ, eN) = diskProject(evecX, evecY, evecZ, idisk=idisk, odisk=odisk)
    (ePdisk, eQdisk, eNdisk) = diskProject(evecXdisk, evecYdisk, evecZdisk,
                                           idisk=idisk, odisk=odisk)

    ePdiskStars = []
    eQdiskStars = []
    eNdiskStars = []

    for ss in range(starCnt):
        (ePtmp, eQtmp, eNtmp) = diskProject(evecXdiskStars[ss],
                                            evecYdiskStars[ss],
                                            evecZdiskStars[ss],
                                            idisk=idisk, odisk=odisk)
        ePdiskStars.append(ePtmp)
        eQdiskStars.append(eQtmp)
        eNdiskStars.append(eNtmp)

    ##########
    #
    # Now make 1D histograms in each direction
    #
    ##########
    histPnone = None
    histQnone = None
    histNnone = None
    histPprob = None
    histQprob = None
    histNprob = None
    histPlike = None
    histQlike = None
    histNlike = None

    histPnoneAcc = None
    histQnoneAcc = None
    histNnoneAcc = None
    histPprobAcc = None
    histQprobAcc = None
    histNprobAcc = None
    histPlikeAcc = None
    histQlikeAcc = None
    histNlikeAcc = None

    numAcc = 0
    for ii in range(starCnt):
        eccPForStar = ePdiskStars[ii]
        eccQForStar = eQdiskStars[ii]
        eccNForStar = eNdiskStars[ii]
        
        (bbp, nnP) = histNofill.hist(binsIn, eccPForStar, normed=True)
        (bbq, nnQ) = histNofill.hist(binsIn, eccQForStar, normed=True)
        (bbn, nnN) = histNofill.hist(binsIn, eccNForStar, normed=True)

        #if names[ii] in accNames:
        if names[ii] in acc_names:
            (bbpAcc, nnPAcc) = histNofill.hist(binsIn, eccPForStar, normed=True)
            (bbqAcc, nnQAcc) = histNofill.hist(binsIn, eccQForStar, normed=True)
            (bbnAcc, nnNAcc) = histNofill.hist(binsIn, eccNForStar, normed=True)
            numAcc += 1

        if (histPnone == None):
            # No weights
            histPnone = nnP
            histQnone = nnQ
            histNnone = nnN
            
            # Weight by likelihood
            histPlike = nnP * diskP[ii]
            histQlike = nnQ * diskP[ii]
            histNlike = nnN * diskP[ii]
            
            # Weight by probability on disk
            histPprob = nnP * numOrbitsOnDisk[ii]
            histQprob = nnQ * numOrbitsOnDisk[ii]
            histNprob = nnN * numOrbitsOnDisk[ii]

            #if (names[ii] in accNames) and (diskP[ii] > 2.7e-3):
            if (names[ii] in acc_names) and (diskP[ii] > 2.7e-3):
                # Accelerating sources only:
                # No weights
                histPnoneAcc = nnPAcc
                histQnoneAcc = nnQAcc
                histNnoneAcc = nnNAcc

                # Weight by likelihood
                histPlikeAcc = nnPAcc * diskP[ii]
                histQlikeAcc = nnQAcc * diskP[ii]
                histNlikeAcc = nnNAcc * diskP[ii]
                diskPAcc = diskP[ii]
    
                # Weight by probability on disk
                histPprobAcc = nnPAcc * numOrbitsOnDisk[ii]
                histQprobAcc = nnQAcc * numOrbitsOnDisk[ii]
                histNprobAcc = nnNAcc * numOrbitsOnDisk[ii]
                numOrbitsOnDiskAcc = numOrbitsOnDisk[ii]

        else:
            # No weights
            histPnone += nnP
            histQnone += nnQ
            histNnone += nnN
            
            # Weight by likelihood
            histPlike += nnP * diskP[ii]
            histQlike += nnQ * diskP[ii]
            histNlike += nnN * diskP[ii]
            
            # Weight by probability on disk
            histPprob += nnP * numOrbitsOnDisk[ii]
            histQprob += nnQ * numOrbitsOnDisk[ii]
            histNprob += nnN * numOrbitsOnDisk[ii]

            #if (names[ii] in accNames) and (diskP[ii] > 2.7e-3):
            if (names[ii] in acc_names) and (diskP[ii] > 2.7e-3):
                # Accelerating sources only:
                # No weights
                histPnoneAcc += nnPAcc
                histQnoneAcc += nnQAcc
                histNnoneAcc += nnNAcc
                
                # Weight by likelihood
                histPlikeAcc += nnPAcc * diskP[ii]
                histQlikeAcc += nnQAcc * diskP[ii]
                histNlikeAcc += nnNAcc * diskP[ii]
                diskPAcc += diskP[ii]
                
                # Weight by probability on disk
                histPprobAcc += nnPAcc * numOrbitsOnDisk[ii]
                histQprobAcc += nnQAcc * numOrbitsOnDisk[ii]
                histNprobAcc += nnNAcc * numOrbitsOnDisk[ii]
                numOrbitsOnDiskAcc += numOrbitsOnDisk[ii]

    histPnone /= starCnt
    histQnone /= starCnt
    histNnone /= starCnt

    histPlike /= diskP.sum()
    histQlike /= diskP.sum()
    histNlike /= diskP.sum()

    # Jessica had the following 3 lines, but this is wrong
    #histPlike /= numOrbitsOnDisk.sum()
    #histQlike /= numOrbitsOnDisk.sum()
    #histNlike /= numOrbitsOnDisk.sum()

    # Should be histPprob instead
    histPprob /= numOrbitsOnDisk.sum()
    histQprob /= numOrbitsOnDisk.sum()
    histNprob /= numOrbitsOnDisk.sum()

    # Accelerating sources only:
    histPnoneAcc /= numAcc
    histQnoneAcc /= numAcc
    histNnoneAcc /= numAcc

    histPlikeAcc /= diskPAcc.sum()
    histQlikeAcc /= diskPAcc.sum()
    histNlikeAcc /= diskPAcc.sum()

    histPprobAcc /= numOrbitsOnDiskAcc.sum()
    histQprobAcc /= numOrbitsOnDiskAcc.sum()
    histNprobAcc /= numOrbitsOnDiskAcc.sum()


    #
    # Plot all solutions.
    #
    py.clf()
    py.figure(figsize=(6,6))
    py.subplots_adjust(left=0.2, right=0.96, top=0.9)
    (bins, dataP) = histNofill.hist(binsIn, eP, normed=True)
    (bins, dataQ) = histNofill.hist(binsIn, eQ, normed=True)
    (bins, dataN) = histNofill.hist(binsIn, eN, normed=True)
    pp = py.plot(bins, dataP, color='black')
    qq = py.plot(bins, dataQ, color='red')
    nn = py.plot(bins, dataN, color='blue')
    py.xlabel('Eccentricity Vector')
    py.ylabel('PDF')
    py.legend((pp, qq, nn), ('P', 'Q', 'N'))
    py.xlim(-1, 1)
    py.savefig(plotdir + 'evec_all_plane%s.png' % suffix)
    py.savefig(plotdir + 'eps/evec_all_plane%s.eps' % suffix)
    py.close()

    #
    # Plot only disk solutions (no weights)
    #
    py.clf()
    py.figure(figsize=(6,6))
    py.subplots_adjust(left=0.2, right=0.96, top=0.9)
    pp = py.plot(bins, histPnone, color='black')
    qq = py.plot(bins, histQnone, color='red')
    nn = py.plot(bins, histNnone, color='blue')
    py.xlabel('Eccentricity Vector')
    py.ylabel('PDF')
    py.title('Unweighted')
    py.legend((pp, qq, nn), ('P', 'Q', 'N'))
    py.xlim(-1, 1)
    py.savefig(plotdir + 'evec_disk_plane_none%s.png' % suffix)
    py.savefig(plotdir + 'eps/evec_disk_plane_none%s.eps' % suffix)
    py.close()

    #
    # Plot only disk solutions (probability on disk)
    #
    py.clf()
    py.figure(figsize=(6,6))
    py.subplots_adjust(left=0.2, right=0.96, top=0.9)
    pp = py.plot(bins, histPprob, color='black')
    qq = py.plot(bins, histQprob, color='red')
    nn = py.plot(bins, histNprob, color='blue')
    py.xlabel('Eccentricity Vector')
    py.ylabel('PDF')
    py.title('Weighted by P(on disk)')
    py.legend((pp, qq, nn), ('P', 'Q', 'N'))
    py.xlim(-1, 1)
    py.savefig(plotdir + 'evec_disk_plane_prob%s.png' % suffix)
    py.savefig(plotdir + 'eps/evec_disk_plane_prob%s.eps' % suffix)
    py.close()

    #
    # Plot only disk solutions (likelihood on disk)
    #
    py.clf()
    py.figure(figsize=(6,6))
    py.subplots_adjust(left=0.2, right=0.96, top=0.9)
    pp = py.plot(bins, histPlike, color='black')
    qq = py.plot(bins, histQlike, color='red')
    nn = py.plot(bins, histNlike, color='blue')
    py.xlabel('Eccentricity Vector')
    py.ylabel('PDF')
    py.title('Weighted by L(on disk)')
    py.legend((pp, qq, nn), ('P', 'Q', 'N'))
    py.xlim(-1, 1)
    py.savefig(plotdir + 'evec_disk_plane_like%s.png' % suffix)
    py.savefig(plotdir + 'eps/evec_disk_plane_like%s.eps' % suffix)
    py.close()

    #####
    # Accelerating sources only
    #####
    #
    # Plot only disk solutions (no weights)
    #
    py.clf()
    py.figure(figsize=(6,6))
    py.subplots_adjust(left=0.2, right=0.96, top=0.9)
    pp = py.plot(bins, histPnoneAcc, color='black')
    qq = py.plot(bins, histQnoneAcc, color='red')
    nn = py.plot(bins, histNnoneAcc, color='blue')
    py.xlabel('Eccentricity Vector')
    py.ylabel('PDF')
    py.title('Unweighted')
    py.legend((pp, qq, nn), ('P', 'Q', 'N'))
    py.xlim(-1, 1)
    py.savefig(plotdir + 'evec_accel_disk_plane_none%s.png' % suffix)
    py.savefig(plotdir + 'eps/evec_accel_disk_plane_none%s.eps' % suffix)
    py.close()

    #
    # Plot only disk solutions (probability on disk)
    #
    py.clf()
    py.figure(figsize=(6,6))
    py.subplots_adjust(left=0.2, right=0.96, top=0.9)
    pp = py.plot(bins, histPprobAcc, color='black')
    qq = py.plot(bins, histQprobAcc, color='red')
    nn = py.plot(bins, histNprobAcc, color='blue')
    py.xlabel('Eccentricity Vector')
    py.ylabel('PDF')
    py.title('Weighted by P(on disk)')
    py.legend((pp, qq, nn), ('P', 'Q', 'N'))
    py.xlim(-1, 1)
    py.savefig(plotdir + 'evec_accel_disk_plane_prob%s.png' % suffix)
    py.savefig(plotdir + 'eps/evec_accel_disk_plane_prob%s.eps' % suffix)
    py.close()

    #
    # Plot only disk solutions (likelihood on disk)
    #
    py.clf()
    py.figure(figsize=(6,6))
    py.subplots_adjust(left=0.2, right=0.96, top=0.9)
    pp = py.plot(bins, histPlikeAcc, color='black')
    qq = py.plot(bins, histQlikeAcc, color='red')
    nn = py.plot(bins, histNlikeAcc, color='blue')
    py.xlabel('Eccentricity Vector')
    py.ylabel('PDF')
    py.title('Weighted by L(on disk)')
    py.legend((pp, qq, nn), ('P', 'Q', 'N'))
    py.xlim(-1, 1)
    py.savefig(plotdir + 'evec_accel_disk_plane_like%s.png' % suffix)
    py.savefig(plotdir + 'eps/evec_accel_disk_plane_like%s.eps' % suffix)
    py.close()

    ##########
    #
    # Plotting of 2D histograms. Used for plotting 
    # eccentricity vector in the disk plane.
    #
    ##########
    def dispHist(hist2d, binsX, binsY,
                 contourOnly=False,
                 contourColor='black',
                 contourLevel=None):
        fmt = 'k.'
        fmt2 = 'r--'
        pntsize = 2
        cmap = py.cm.hot_r
        #cmap = py.cm.gray_r

        probDist = hist2d.copy()
        
        # Determine contour levels
        # Flatten and reverse sort our prob. distribution
        sid0 = probDist.flatten().argsort()
        sid = sid0[::-1]
        pixSort = probDist.flatten()[sid]
        
        # Make a cumulative distribution function starting from the
        # highest pixel value. This way we can find the level above
        # which 68% of the trials will fall.
        cdf = np.cumsum(pixSort)
        
        # Determine point at which we reach 68% level
        #percents = array([0.6827, 0.9545, 0.9973]) * len(xx)
        sigmas = [0.6827, 0.9545, 0.9973]
        if (contourLevel == None):
            percents = np.array([0.6827, 0.9545]) * cdf[-1]
        else:
            percents = np.array([sigmas[contourLevel-1]]) * cdf[-1]
            
        levels = np.zeros(len(percents), float)
        for ii in range(len(levels)):
            # Get the index of the pixel at which the CDF
            # reaches this percentage (the first one found)
            cdx = (np.where(cdf < percents[ii]))[0]

            if (len(cdx) > 0):
                # Now get the level of that pixel
                levels[ii] = pixSort[cdx[-1]]
            else:
                print 'dispHist: No level found for %5.3f' % (sigmas[ii])
            
        # Mask out the parts where we don't have data.
        #probDist = ma.masked_where(probDist == 0, log10(probDist))
        #levels = log10(levels)
        foo = np.where(probDist == 0)
        probDist = np.sqrt(probDist)
        probDist[foo] = 0.0
        levels = np.sqrt(levels)

        if (contourOnly == False):
            py.imshow(probDist, extent=[b1[0], b1[-1], b2[0], b2[-1]],
                   cmap=cmap, origin='lower', aspect='auto')
            cbar = py.colorbar(shrink=0.78)

        py.contour(probDist, levels, origin=None, colors=contourColor,
                extent=[b1[0], b1[-1], b2[0], b2[-1]])


    ##########
    #
    # Make 2D histograms (weight = none, prob, like)
    #   of just the candidate disk stars
    #
    ##########
    binsIn = np.arange(-1, 1.01, 0.1)
    dist2none = None
    dist2prob = None
    dist2like = None
    dist2prob1 = None
    dist2prob2 = None
    dist2prob3 = None
    dist2prob4 = None
    dist2prob5 = None
    dist2prob6 = None
    dist2prob7 = None
    dist2prob8 = None
    dist2prob9 = None
    numOrbitsOnDiskSum1 = 0
    numOrbitsOnDiskSum2 = 0
    numOrbitsOnDiskSum3 = 0
    numOrbitsOnDiskSum4 = 0
    numOrbitsOnDiskSum5 = 0
    numOrbitsOnDiskSum6 = 0
    numOrbitsOnDiskSum7 = 0
    numOrbitsOnDiskSum8 = 0
    numOrbitsOnDiskSum9 = 0
    r1 = np.where(r2d < 2.0)[0]
    r2 = np.where((r2d >= 2.0) & (r2d < 3.0))[0]
    r3 = np.where((r2d >= 3.0) & (r2d < 4.0))[0]
    r4 = np.where((r2d >= 4.0) & (r2d < 5.0))[0]
    r5 = np.where((r2d >= 5.0) & (r2d < 6.0))[0]
    r6 = np.where((r2d >= 6.0) & (r2d < 7.0))[0]
    r7 = np.where((r2d >= 7.0) & (r2d < 8.0))[0]
    r8 = np.where((r2d >= 8.0) & (r2d < 9.0))[0]
    r9 = np.where(r2d >= 9.0)[0]
    rbin = [r1,r2,r3,r4,r5,r6,r7,r8,r9] 
    rbinN = [len(r1),len(r2),len(r3),len(r4),len(r5),len(r6),len(r7),len(r8),len(r9)] 
    for ii in range(starCnt):
        xx = ePdiskStars[ii]
        yy = eQdiskStars[ii]
        
        (tmpDist, b1, b2) = h2d.histogram2d(xx, yy,
                                            bins=[binsIn, binsIn],
                                            normed=True)
        probDist = np.array(tmpDist, float)
        
        if (dist2none == None):
            dist2none = probDist
            dist2like = probDist * diskP[ii]
            dist2prob = probDist * numOrbitsOnDisk[ii]

            # Also group things by radius
            if (r2d[ii] < 2.0):
                dist2prob1 = probDist * numOrbitsOnDisk[ii]
                numOrbitsOnDiskSum1 += numOrbitsOnDisk[ii]
            elif (r2d[ii] >= 2.0) & (r2d[ii] < 3.0):
                dist2prob2 = probDist * numOrbitsOnDisk[ii]
                numOrbitsOnDiskSum2 += numOrbitsOnDisk[ii]
            elif (r2d[ii] >= 3.0) & (r2d[ii] < 4.0):
                dist2prob3 = probDist * numOrbitsOnDisk[ii]
                numOrbitsOnDiskSum3 += numOrbitsOnDisk[ii]
            elif (r2d[ii] >= 4.0) & (r2d[ii] < 5.0):
                dist2prob4 = probDist * numOrbitsOnDisk[ii]
                numOrbitsOnDiskSum4 += numOrbitsOnDisk[ii]
            elif (r2d[ii] >= 5.0) & (r2d[ii] < 6.0):
                dist2prob5 = probDist * numOrbitsOnDisk[ii]
                numOrbitsOnDiskSum5 += numOrbitsOnDisk[ii]
            elif (r2d[ii] >= 6.0) & (r2d[ii] < 7.0):
                dist2prob6 = probDist * numOrbitsOnDisk[ii]
                numOrbitsOnDiskSum6 += numOrbitsOnDisk[ii]
            elif (r2d[ii] >= 7.0) & (r2d[ii] < 8.0):
                dist2prob7 = probDist * numOrbitsOnDisk[ii]
                numOrbitsOnDiskSum7 += numOrbitsOnDisk[ii]
            elif (r2d[ii] >= 8.0) & (r2d[ii] < 9.0):
                dist2prob8 = probDist * numOrbitsOnDisk[ii]
                numOrbitsOnDiskSum8 += numOrbitsOnDisk[ii]
            elif (r2d[ii] >= 9.0):
                dist2prob9 = probDist * numOrbitsOnDisk[ii]
                numOrbitsOnDiskSum9 += numOrbitsOnDisk[ii]

            print 'Radius of first star: %5.2f' % r2d[ii]
        else:
            dist2none += probDist
            dist2like += probDist * diskP[ii]
            dist2prob += probDist * numOrbitsOnDisk[ii]

            # Also group things by radius
            if (r2d[ii] < 2.0):
                if dist2prob1 != None:
                    dist2prob1 += probDist * numOrbitsOnDisk[ii]
                else:
                    dist2prob1 = probDist * numOrbitsOnDisk[ii]
                numOrbitsOnDiskSum1 += numOrbitsOnDisk[ii]
            elif (r2d[ii] >= 2.0) & (r2d[ii] < 3.0):
                if dist2prob2 != None:
                    dist2prob2 += probDist * numOrbitsOnDisk[ii] 
                else:
                    dist2prob2 = probDist * numOrbitsOnDisk[ii] 
                numOrbitsOnDiskSum2 += numOrbitsOnDisk[ii]
            elif (r2d[ii] >= 3.0) & (r2d[ii] < 4.0):
                if dist2prob3 != None:
                    dist2prob3 += probDist * numOrbitsOnDisk[ii]
                else:
                    dist2prob3 = probDist * numOrbitsOnDisk[ii]
                numOrbitsOnDiskSum3 += numOrbitsOnDisk[ii]
            elif (r2d[ii] >= 4.0) & (r2d[ii] < 5.0):
                if dist2prob4 != None:
                    dist2prob4 += probDist * numOrbitsOnDisk[ii]
                else:
                    dist2prob4 = probDist * numOrbitsOnDisk[ii]
                numOrbitsOnDiskSum4 += numOrbitsOnDisk[ii]
            elif (r2d[ii] >= 5.0) & (r2d[ii] < 6.0):
                if dist2prob5 != None:
                    dist2prob5 += probDist * numOrbitsOnDisk[ii]
                else:
                    dist2prob5 = probDist * numOrbitsOnDisk[ii]
                numOrbitsOnDiskSum5 += numOrbitsOnDisk[ii]
            elif (r2d[ii] >= 6.0) & (r2d[ii] < 7.0):
                if dist2prob6 != None:
                    dist2prob6 += probDist * numOrbitsOnDisk[ii]
                else:
                    dist2prob6 = probDist * numOrbitsOnDisk[ii]
                numOrbitsOnDiskSum6 += numOrbitsOnDisk[ii]
            elif (r2d[ii] >= 7.0) & (r2d[ii] < 8.0):
                if dist2prob7 != None:
                    dist2prob7 += probDist * numOrbitsOnDisk[ii]
                else:
                    dist2prob7 = probDist * numOrbitsOnDisk[ii]
                numOrbitsOnDiskSum7 += numOrbitsOnDisk[ii]
            elif (r2d[ii] >= 8.0) & (r2d[ii] < 9.0):
                if dist2prob8 != None:
                    dist2prob8 += probDist * numOrbitsOnDisk[ii]
                else:
                    dist2prob8 = probDist * numOrbitsOnDisk[ii]
                numOrbitsOnDiskSum8 += numOrbitsOnDisk[ii]
            elif (r2d[ii] >= 9.0):
                if dist2prob9 != None:
                    dist2prob9 += probDist * numOrbitsOnDisk[ii]
                else:
                    dist2prob9 = probDist * numOrbitsOnDisk[ii]
                numOrbitsOnDiskSum9 += numOrbitsOnDisk[ii]

    dist2none /= starCnt
    dist2like /= diskP.sum()
    dist2prob /= numOrbitsOnDisk.sum()

    # 2D histograms binned by radius
    if dist2prob1 != None:
        dist2prob1 /= numOrbitsOnDiskSum1
    if dist2prob2 != None:
        dist2prob2 /= numOrbitsOnDiskSum2
    if dist2prob3 != None:
        dist2prob3 /= numOrbitsOnDiskSum3
    if dist2prob4 != None:
        dist2prob4 /= numOrbitsOnDiskSum4
    if dist2prob5 != None:
        dist2prob5 /= numOrbitsOnDiskSum5
    if dist2prob6 != None:
        dist2prob6 /= numOrbitsOnDiskSum6
    if dist2prob7 != None:
        dist2prob7 /= numOrbitsOnDiskSum7
    if dist2prob8 != None:
        dist2prob8 /= numOrbitsOnDiskSum8
    if dist2prob9 != None:
        dist2prob9 /= numOrbitsOnDiskSum9
    dist2probRad = [dist2prob1,dist2prob2,dist2prob3,dist2prob4,
                    dist2prob5,dist2prob6,dist2prob7,dist2prob8,dist2prob9]
    radLbl = ['r < 2"', '2" < r < 3"', '3" < r < 4"', '4" < r < 5"','5" < r < 6"',
              '6" < r < 7"', '7" < r < 8"', '8" < r < 9"', 'r > 9"']

    #
    # Plot 2D histogram of ecc. vector on disk plane (weight=none)
    #
    py.clf()
    py.figure(figsize=(6,6))
    py.subplots_adjust(left=0.2, right=0.96, top=0.9)
    dispHist(dist2none, b1, b2)
    py.xlabel('Ecc. Vector: p-Component')
    py.ylabel('Ecc. Vector: q-Component')
    py.axis([1, -1, -1, 1])
    py.savefig(plotdir + 'evec_disk_pdf_none%s.png' % suffix)
    py.savefig(plotdir + 'eps/evec_disk_pdf_none%s.eps' % suffix)
    py.close()

    # Find the peak of the probability distribution:
    pdx = np.where(dist2none == dist2none.max())
    ePpeak = b1[pdx[1][0]]
    eQpeak = b2[pdx[0][0]]
    print 'Peak of evec_disk_pdf is at P = %5.2f  Q = %5.2f (weight=none)' % \
          (ePpeak, eQpeak)

    #
    # Plot 2D histogram of ecc. vector on disk plane (weight=like)
    #
    py.clf()
    py.figure(figsize=(6,6))
    py.subplots_adjust(left=0.2, right=0.96, top=0.9)
    dispHist(dist2like, b1, b2)
    py.xlabel('Ecc. Vector: p-Component')
    py.ylabel('Ecc. Vector: q-Component')
    py.axis([1, -1, -1, 1])
    py.savefig(plotdir + 'evec_disk_pdf_like%s.png' % suffix)
    py.savefig(plotdir + 'eps/evec_disk_pdf_like%s.eps' % suffix)
    py.close()

    # Find the peak of the probability distribution:
    pdx = np.where(dist2like == dist2like.max())
    ePpeak = b1[pdx[1][0]]
    eQpeak = b2[pdx[0][0]]
    print 'Peak of evec_disk_pdf is at P = %5.2f  Q = %5.2f (weight=like)' % \
          (ePpeak, eQpeak)

    #
    # Plot 2D histogram of ecc. vector on disk plane (weight=prob)
    #
    py.clf()
    py.figure(figsize=(6,6))
    py.subplots_adjust(left=0.2, right=0.96, top=0.9)
    dispHist(dist2prob, b1, b2)
    py.xlabel('Ecc. Vector: p-Component')
    py.ylabel('Ecc. Vector: q-Component')
    py.axis([1, -1, -1, 1])
    py.savefig(plotdir + 'evec_disk_pdf_prob%s.png' % suffix)
    py.savefig(plotdir + 'eps/evec_disk_pdf_prob%s.eps' % suffix)
    py.close()

    # Find the peak of the probability distribution:
    pdx = np.where(dist2prob == dist2prob.max())
    ePpeak = b1[pdx[1][0]]
    eQpeak = b2[pdx[0][0]]
    print 'Peak of evec_disk_pdf is at P = %5.2f  Q = %5.2f (weight=prob)' % \
          (ePpeak, eQpeak)

    # Radial binning of the 
    # 2D histogram of ecc. vector on disk plane (weight=prob)
    #
    py.figure(figsize=(10,10))
    py.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1,
                       wspace=0.3, hspace=0.3)
    py.clf()
    for ii in range(9):
        py.subplot(3,3,ii+1)
        print
        if dist2probRad[ii] != None:
            dispHist(dist2probRad[ii], b1, b2)
            print 'Stars in bin %i:' % (ii+1)
            for rr in range(rbinN[ii]):
                print names[rbin[ii][rr]]
        else:
            continue
        py.title('%s (N = %i)' % (radLbl[ii], rbinN[ii]))
        py.plot([0],[0],'rx',ms=5)
        if ii in np.array([0,3,6]):
            py.ylabel('Ecc. Vector: q-Component')
        if ii > 5:
            py.xlabel('Ecc. Vector: p-Component')
        py.axis([1, -1, -1, 1])
    py.savefig(plotdir + 'evec_disk_pdf_prob_radial%s.png' % suffix)
    py.savefig(plotdir + 'eps/evec_disk_pdf_prob_radial%s.eps' % suffix)
    py.close()

    ##########
    #
    # Make 2D histograms (weight = none, prob, like)
    #   of the accelerating stars that are disk candidates
    #
    ##########
    binsIn = np.arange(-1, 1.01, 0.1)
    dist2none = None
    dist2prob = None
    dist2like = None
    print ''
    print 'Accelerating disk members:'
    for ii in range(starCnt):
        #if names[ii] not in accNames:
        if names[ii] not in acc_names:
            continue
        xx = ePdiskStars[ii]
        yy = eQdiskStars[ii]
        
        (tmpDist, b1, b2) = h2d.histogram2d(xx, yy,
                                            bins=[binsIn, binsIn],
                                            normed=True)
        probDist = np.array(tmpDist, float)
        
        if (dist2none == None):
            dist2none = probDist
            dist2like = probDist * diskP[ii]
            dist2prob = probDist * numOrbitsOnDisk[ii]
        else:
            dist2none += probDist
            dist2like += probDist * diskP[ii]
            dist2prob += probDist * numOrbitsOnDisk[ii]
        print names[ii]

    print 
    dist2none /= starCnt
    dist2like /= diskP.sum()
    dist2prob /= numOrbitsOnDisk.sum()

    #
    # Plot 2D histogram of accelerating stars' ecc. vector on disk plane (weight=none)
    #
    py.clf()
    py.figure(figsize=(6,6))
    py.subplots_adjust(left=0.2, right=0.96, top=0.9)
    dispHist(dist2none, b1, b2)
    py.xlabel('Ecc. Vector: p-Component')
    py.ylabel('Ecc. Vector: q-Component')
    py.axis([1, -1, -1, 1])
    py.savefig(plotdir + 'evec_accel_disk_pdf_none%s.png' % suffix)
    py.savefig(plotdir + 'eps/evec_accel_disk_pdf_none%s.eps' % suffix)
    py.close()

    # Find the peak of the probability distribution:
    pdx = np.where(dist2none == dist2none.max())
    ePpeak = b1[pdx[1][0]]
    eQpeak = b2[pdx[0][0]]
    print 'Peak of evec_accel_disk_pdf is at P = %5.2f  Q = %5.2f (weight=none)' % \
          (ePpeak, eQpeak)

    #
    # Plot 2D histogram of accelerating stars' ecc. vector on disk plane (weight=like)
    #
    py.clf()
    py.figure(figsize=(6,6))
    py.subplots_adjust(left=0.2, right=0.96, top=0.9)
    dispHist(dist2like, b1, b2)
    py.xlabel('Ecc. Vector: p-Component')
    py.ylabel('Ecc. Vector: q-Component')
    py.axis([1, -1, -1, 1])
    py.savefig(plotdir + 'evec_accel_disk_pdf_like%s.png' % suffix)
    py.savefig(plotdir + 'eps/evec_accel_disk_pdf_like%s.eps' % suffix)
    py.close()

    # Find the peak of the probability distribution:
    pdx = np.where(dist2like == dist2like.max())
    ePpeak = b1[pdx[1][0]]
    eQpeak = b2[pdx[0][0]]
    print 'Peak of evec_accel_disk_pdf is at P = %5.2f  Q = %5.2f (weight=like)' % \
          (ePpeak, eQpeak)

    #
    # Plot 2D histogram of accelerating stars' ecc. vector on disk plane (weight=prob)
    #
    py.clf()
    py.figure(figsize=(6,6))
    py.subplots_adjust(left=0.2, right=0.96, top=0.9)
    dispHist(dist2prob, b1, b2)
    py.xlabel('Ecc. Vector: p-Component')
    py.ylabel('Ecc. Vector: q-Component')
    py.axis([1, -1, -1, 1])
    py.savefig(plotdir + 'evec_accel_disk_pdf_prob%s.png' % suffix)
    py.savefig(plotdir + 'eps/evec_accel_disk_pdf_prob%s.eps' % suffix)
    py.close()

    # Find the peak of the probability distribution:
    pdx = np.where(dist2prob == dist2prob.max())
    ePpeak = b1[pdx[1][0]]
    eQpeak = b2[pdx[0][0]]
    print 'Peak of evec_accel_disk_pdf is at P = %5.2f  Q = %5.2f (weight=prob)' % \
          (ePpeak, eQpeak)

    # Make plots of individual stars (in disk).
    # Show the 1 and 3 sigma contours for the eccentricity vector
    # projected into the disk plane.
    for ii in range(starCnt):
        py.clf()
        py.figure(figsize=(6,6))
        py.subplots_adjust(left=0.2, right=0.96, top=0.9)
        (tmpDist, b1, b2) = h2d.histogram2d(ePdiskStars[ii], eQdiskStars[ii],
                                            bins=(20, 20))
        probDist = np.array(tmpDist, float)
        dispHist(probDist, b1, b2)
        py.plot([0], [0], 'k+')
        py.xlabel('eccP')
        py.ylabel('eccQ')
        py.axis([1, -1, -1, 1])
        py.title(names[ii])
        py.savefig(plotdir + 'eccVector/evec_disk_%s.png' % (names[ii]))

    ########################################
    # Make individual contours for each disk star's eccentricity
    # 1 sigma. Only do this for those stars we claim have 3 sigma
    # eccentricity limits above 0.2.
    ########################################
    colors = ['orange', 'red', 'brown', 'turquoise', 'tomato', 'cyan',
              'purple', 'blue', 'navy', 'magenta', 'green', 'purple',
              'mediumorchid', 'deeppink', 'olive', 'steelblue',
              'salmon', 'rosybrown', 'sienna', 'plum']

    eccTab = asciidata.open(root + alnDir + 'tables/eccVector.dat')
    eccName = [eccTab[0][ss].strip() for ss in range(eccTab.nrows)]
    eccLo = eccTab[18].tonumpy()

    edx = (np.where((eccLo > 0.2) & (eccLo > 0)))[0]

    # Index into the stars array
    highEccIdx = []
    for ee in edx:
        try:
            ii = names.index(eccName[ee])
            highEccIdx.append(ii)
            print 'Plotting %s' % eccName[ee]
        except ValueError:
            print 'Not a disk star: %s' % eccName[ee]
    
    
    py.clf()
    py.figure(figsize=(6,6))
    py.subplots_adjust(left=0.2, right=0.96, top=0.9)
    for ii in highEccIdx:
        color = colors[ii % len(colors)]
        (tmpDist, b1, b2) = h2d.histogram2d(eP[ii], eQ[ii], bins=(20, 20))
        probDist = np.array(tmpDist, float)
        dispHist(probDist, b1, b2, contourOnly=True,
                 contourColor=color, contourLevel=1)

    py.plot([0], [0], 'k+')
    py.xlabel('eccP')
    py.ylabel('eccQ')
    py.axis([1, -1, -1, 1])
    py.savefig(plotdir + 'evec_all_stars_pdf1%s.png' % suffix)
    py.savefig(plotdir + 'eps/evec_all_stars_pdf1%s.eps' % suffix)
    py.close()

    # 3 sigma
    py.clf()
    py.figure(figsize=(6,6))
    py.subplots_adjust(left=0.2, right=0.96, top=0.9)
    for ii in highEccIdx:
        color = colors[ii % len(colors)]
        (tmpDist, b1, b2) = h2d.histogram2d(eP[ii], eQ[ii], bins=(20, 20))
        probDist = np.array(tmpDist, float)
        dispHist(probDist, b1, b2, contourOnly=True,
                 contourColor=color, contourLevel=3)

    py.plot([0], [0], 'k+')
    py.xlabel('eccP')
    py.ylabel('eccQ')
    py.axis([1, -1, -1, 1])
    py.savefig(plotdir + 'evec_all_stars_pdf3%s.png' % suffix)
    py.savefig(plotdir + 'eps/evec_all_stars_pdf3%s.eps' % suffix)
    py.close()


    # Do the same but for only disk solutions.
    # 1 sigma
    py.clf()
    py.figure(figsize=(6,6))
    py.subplots_adjust(left=0.2, right=0.96, top=0.9)
    for ii in highEccIdx:
        color = colors[ii % len(colors)]
        (tmpDist, b1, b2) = h2d.histogram2d(ePdiskStars[ii], eQdiskStars[ii],
                                            bins=(20, 20))
        probDist = np.array(tmpDist, float)
        dispHist(probDist, b1, b2, contourOnly=True,
                 contourColor=color, contourLevel=1)

    py.plot([0], [0], 'k+')
    py.xlabel('eccP')
    py.ylabel('eccQ')
    py.axis([1, -1, -1, 1])
    py.savefig(plotdir + 'evec_disk_stars_pdf1%s.png' % suffix)
    py.savefig(plotdir + 'eps/evec_disk_stars_pdf1%s.eps' % suffix)
    py.close()

    # 3 sigma
    py.clf()
    py.figure(figsize=(6,6))
    py.subplots_adjust(left=0.2, right=0.96, top=0.9)
    for ii in highEccIdx:
        color = colors[ii % len(colors)]
        (tmpDist, b1, b2) = h2d.histogram2d(ePdiskStars[ii], eQdiskStars[ii],
                                            bins=(20, 20))
        probDist = np.array(tmpDist, float)
        dispHist(probDist, b1, b2, contourOnly=True,
                 contourColor=color, contourLevel=3)

    py.plot([0], [0], 'k+')
    py.xlabel('eccP')
    py.ylabel('eccQ')
    py.axis([1, -1, -1, 1])
    py.savefig(plotdir + 'evec_disk_stars_pdf3%s.png' % suffix)
    py.savefig(plotdir + 'eps/evec_disk_stars_pdf3%s.eps' % suffix)
    py.close()

def plotOrbitOmegaPdf(pdfdir='aorb_thesis/', mosaic=True,
                      suffix=''):
    cc = objects.Constants()
    # Load names of young stars 
    if mosaic == True:
        # Load up mosaic data as well; select only stars at r>4, since
        # we don't want to add any info from mosaics if we have it in
        # the central 10" already
        yng1 = young.loadYoungStars(root+alnDir,fit=poly,points=points,
                                    withRVonly=True,silent=True,skipStar=['S5-237']) 
        yng2 = young.loadYoungStars(mscDir,fit=polyM,points=pointsM,
                                    mosaic=True, withRVonly=True,silent=True)
        cntrlNames = yng1.getArray('name')
        mscNames = yng2.getArray('name')
        # Merge this object with object from central 10" analysis
        yng = merge(yng1, yng2)
    else:
        yng = young.loadYoungStars(root+alnDir,fit=poly,points=points,
                                   withRVonly=True,silent=True)

    ##########
    #
    # Plotting
    #
    ##########
    py.clf()
    usetexTrue()

    # Store some font sizes for labels and ticks
    labelFont = {'fontsize': 14, 'fontweight': 'bold'}

    tickSize = 14

    def dispHist(xx, yy):
        fmt = 'k.'
        fmt2 = 'r--'
        pntsize = 2
        cmap = py.cm.hot_r

        # Make 2D histogram
        (probDist, b1, b2) = h2d.histogram2d(xx, yy, bins=(50, 50))

        # Need to convert the 2d histogram into floats
        probDist = np.array(probDist, float)
        
        # Determine contour levels
        # Flatten and reverse sort our prob. distribution
        sid0 = probDist.flatten().argsort()
        sid = sid0[::-1]
        pixSort = probDist.flatten()[sid]
        
        # Make a cumulative distribution function starting from the
        # highest pixel value. This way we can find the level above
        # which 68% of the trials will fall.
        cdf = np.cumsum(pixSort)
        
        # Determine point at which we reach 68% level
        #percents = array([0.6827, 0.9545, 0.9973]) * len(xx)
        percents = np.array([0.6827, 0.9545]) * len(xx)
        levels = np.zeros(len(percents), float)
        for ii in range(len(levels)):
            # Get the index of the pixel at which the CDF
            # reaches this percentage (the first one found)
            idx = (np.where(cdf < percents[ii]))[0]

            # Now get the level of that pixel
            levels[ii] = pixSort[idx[-1]]
            
        # Mask out the parts where we don't have data.
        foo = np.where(probDist == 0)
        probDist = np.log10(probDist)
        probDist[foo] = 0.0
        levels = np.log10(levels)

        py.imshow(probDist, extent=[b1[0], b1[-1], b2[0], b2[-1]],
                 cmap=cmap, origin='lower', aspect='auto',
                 interpolation='bilinear')

        py.contour(probDist, levels, origin=None, colors='black',
                extent=[b1[0], b1[-1], b2[0], b2[-1]])


    def axisLabel(xtext, ytext, yrange=None):
        # Rescales fonts for ticks and labels
        thePlot = py.gca()
        rng = py.axis()

        # Incrememnt for ticks on the X axis
        tmp = np.abs(float(rng[1]) - float(rng[0])) / 5.0
        xinc = __builtin__.round(tmp, 1)

        if (xinc == 0):
            xinc = 0.05
        
        thePlot.get_xaxis().set_major_locator(py.MultipleLocator(xinc))
        py.setp( thePlot.get_xticklabels(), fontsize=tickSize )
        py.setp( thePlot.get_yticklabels(), fontsize=tickSize )

        # Add axis labels
        py.xlabel(xtext, labelFont)
        py.ylabel(ytext, labelFont)

        # Optional re-scale axes
        if (yrange != None):
            py.axis([rng[0], rng[1], yrange[0], yrange[1]])

    pdf_O_all = []

    for star in yng.stars:
        mscStar = False
        mStr = ''
        
        if mosaic == True:
            if (star.name in mscNames) & (star.name not in cntrlNames):
                mscStar = True
                mStr = '(mosaic star)'

        # File contains analytic orbit solutions with acceleration limits (MC)
        print 'Working on %s %s' % (star.name, mStr)
        
        pdffile = '%s%s%s%s.mc.dat' % (root, alnDir, pdfdir, star.name)
        if os.path.exists(pdffile) == False:
            continue

        py.clf()
        py.figure(figsize=(7,10))
        py.subplots_adjust(left=0.1, right=0.96, top=0.95, bottom=0.05,
                           wspace=0.25, hspace=0.25)
        pdf = pickle.load(open(pdffile))

        xdat = pdf.z * dist / cc.au_in_pc

        # Plot Omega
        py.subplot(3, 1, 1)
        dispHist(xdat, pdf.o)
        axisLabel(r'{\bf z (pc)}', r'{\bf $\Omega$}', yrange=[0, 360.0])

        if ('irs' in star.name):
            starLabel = '{\\bf IRS %s}' % (star.name[3:])
        else:
            starLabel = '{\\bf %s}' % (star.name)

        py.title(starLabel, labelFont)

        py.subplot(3, 1, 2)
        binsIn = np.arange(0, 360.0, 1.0)
        py.hist(pdf.o,bins=binsIn,histtype='step',normed=True,color='k')
        axisLabel(r'{\bf $\Omega$}', r'{\bf PDF}')

        py.subplot(3, 1, 3)
        py.plot(pdf.m/1.e6, pdf.o, 'k.')
        axisLabel(r'{\bf MC Mass ($M_{BH} + M_{ext}$)', r'{\bf $\Omega$}')

        py.savefig(root + alnDir + pdfdir + 'pdfparams/pdf_Omega_%s.png' % star.name, dpi=75)
        py.close()

        # Save all PDFs to plot the sum later
        pdf_O_all =  np.concatenate([pdf_O_all, pdf.o])

    binsIn = np.arange(0, 360.0, 1.0)
    (bb1, nn1) = histNofill.hist(binsIn, pdf_O_all, normed=True)
    py.clf()
    py.figure(figsize=(7,7))
    py.plot(bb1, nn1, color='black', linewidth=2)
    py.xlabel(r'$\Omega$ (deg)')
    py.ylabel('Probability Density')
    py.savefig(root + alnDir + pdfdir + 'pdfparams/pdf_Omega_allCombined%s.png' % suffix)
    py.close()

    usetexFalse()

def plotOrbitInclPdf(pdfdir='aorb_thesis/', mosaic=True,
                      suffix=''):
    cc = objects.Constants()
    # Load names of young stars 
    if mosaic == True:
        # Load up mosaic data as well; select only stars at r>4, since
        # we don't want to add any info from mosaics if we have it in
        # the central 10" already
        yng1 = young.loadYoungStars(root+alnDir,fit=poly,points=points,
                                    withRVonly=True,silent=True,skipStar=['S5-237']) 
        yng2 = young.loadYoungStars(mscDir,fit=polyM,points=pointsM,
                                    mosaic=True, withRVonly=True,silent=True)
        cntrlNames = yng1.getArray('name')
        mscNames = yng2.getArray('name')
        # Merge this object with object from central 10" analysis
        yng = merge(yng1, yng2)
    else:
        yng = young.loadYoungStars(root+alnDir,fit=poly,points=points,
                                   withRVonly=True,silent=True)

    ##########
    #
    # Plotting
    #
    ##########
    py.clf()
    usetexTrue()

    # Store some font sizes for labels and ticks
    labelFont = {'fontsize': 14, 'fontweight': 'bold'}

    tickSize = 14

    def dispHist(xx, yy):
        fmt = 'k.'
        fmt2 = 'r--'
        pntsize = 2
        cmap = py.cm.hot_r

        # Make 2D histogram
        (probDist, b1, b2) = h2d.histogram2d(xx, yy, bins=(50, 50))

        # Need to convert the 2d histogram into floats
        probDist = np.array(probDist, float)
        
        # Determine contour levels
        # Flatten and reverse sort our prob. distribution
        sid0 = probDist.flatten().argsort()
        sid = sid0[::-1]
        pixSort = probDist.flatten()[sid]
        
        # Make a cumulative distribution function starting from the
        # highest pixel value. This way we can find the level above
        # which 68% of the trials will fall.
        cdf = np.cumsum(pixSort)
        
        # Determine point at which we reach 68% level
        #percents = array([0.6827, 0.9545, 0.9973]) * len(xx)
        percents = np.array([0.6827, 0.9545]) * len(xx)
        levels = np.zeros(len(percents), float)
        for ii in range(len(levels)):
            # Get the index of the pixel at which the CDF
            # reaches this percentage (the first one found)
            idx = (np.where(cdf < percents[ii]))[0]

            # Now get the level of that pixel
            levels[ii] = pixSort[idx[-1]]
            
        # Mask out the parts where we don't have data.
        foo = np.where(probDist == 0)
        probDist = np.log10(probDist)
        probDist[foo] = 0.0
        levels = np.log10(levels)

        py.imshow(probDist, extent=[b1[0], b1[-1], b2[0], b2[-1]],
                 cmap=cmap, origin='lower', aspect='auto',
                 interpolation='bilinear')

        py.contour(probDist, levels, origin=None, colors='black',
                extent=[b1[0], b1[-1], b2[0], b2[-1]])


    def axisLabel(xtext, ytext, yrange=None):
        # Rescales fonts for ticks and labels
        thePlot = py.gca()
        rng = py.axis()

        # Incrememnt for ticks on the X axis
        tmp = np.abs(float(rng[1]) - float(rng[0])) / 5.0
        xinc = __builtin__.round(tmp, 1)

        if (xinc == 0):
            xinc = 0.05
        
        thePlot.get_xaxis().set_major_locator(py.MultipleLocator(xinc))
        py.setp( thePlot.get_xticklabels(), fontsize=tickSize )
        py.setp( thePlot.get_yticklabels(), fontsize=tickSize )

        # Add axis labels
        py.xlabel(xtext, labelFont)
        py.ylabel(ytext, labelFont)

        # Optional re-scale axes
        if (yrange != None):
            py.axis([rng[0], rng[1], yrange[0], yrange[1]])

    pdf_i_all = []

    for star in yng.stars:
        mscStar = False
        mStr = ''
        
        if mosaic == True:
            if (star.name in mscNames) & (star.name not in cntrlNames):
                mscStar = True
                mStr = '(mosaic star)'

        # File contains analytic orbit solutions with acceleration limits (MC)
        print 'Working on %s %s' % (star.name, mStr)
        
        pdffile = '%s%s%s%s.mc.dat' % (root, alnDir, pdfdir, star.name)
        if os.path.exists(pdffile) == False:
            continue

        py.clf()
        py.figure(figsize=(7,10))
        py.subplots_adjust(left=0.1, right=0.96, top=0.95, bottom=0.05,
                           wspace=0.25, hspace=0.25)
        pdf = pickle.load(open(pdffile))

        xdat = pdf.z * dist / cc.au_in_pc

        # Plot Incl
        py.subplot(3, 1, 1)
        dispHist(xdat, pdf.i)
        axisLabel(r'{\bf z (pc)}', r'{\bf Inclination}', yrange=[0, 180.0])

        if ('irs' in star.name):
            starLabel = '{\\bf IRS %s}' % (star.name[3:])
        else:
            starLabel = '{\\bf %s}' % (star.name)

        py.title(starLabel, labelFont)

        py.subplot(3, 1, 2)
        binsIn = np.arange(0, 180.0, 1.0)
        py.hist(pdf.i,bins=binsIn,histtype='step',normed=True,color='k')
        axisLabel(r'{\bf Inclination}', r'{\bf PDF}')

        py.subplot(3, 1, 3)
        py.plot(pdf.m/1.e6, pdf.i, 'k.')
        axisLabel(r'{\bf MC Mass ($M_{BH} + M_{ext}$)', r'{\bf Inclination}')

        py.savefig(root + alnDir + pdfdir + 'pdfparams/pdf_Incl_%s.png' % star.name, dpi=75)
        py.close()

        # Save all PDFs to plot the sum later
        pdf_i_all =  np.concatenate([pdf_i_all, pdf.i])

    binsIn = np.arange(0, 180.0, 1.0)
    (bb1, nn1) = histNofill.hist(binsIn, pdf_i_all, normed=True)
    py.clf()
    py.figure(figsize=(7,7))
    py.plot(bb1, nn1, color='black', linewidth=2)
    py.xlabel(r'Inclination (deg)')
    py.ylabel('Probability Density')
    py.savefig(root + alnDir + pdfdir + 'pdfparams/pdf_Incl_allCombined%s.png' % suffix)
    py.close()

    usetexFalse()




def plotOrbitPhasePdf(pdfdir='aorb_thesis/', mosaic=True,
                      examineOutliers=False, extendedMass=False, suffix=''):
    cc = objects.Constants()
    # Load names of young stars 
    if mosaic == True:
        # Load up mosaic data as well; select only stars at r>4, since
        # we don't want to add any info from mosaics if we have it in
        # the central 10" already
        yng1 = young.loadYoungStars(root+alnDir,fit=poly,points=points,
                                    withRVonly=True,silent=True,skipStar=['S5-237']) 
        yng2 = young.loadYoungStars(mscDir,fit=polyM,points=pointsM,
                                    mosaic=True, withRVonly=True,silent=True)
        cntrlNames = yng1.getArray('name')
        mscNames = yng2.getArray('name')
        # Merge this object with object from central 10" analysis
        yng = merge(yng1, yng2)
    else:
        yng = young.loadYoungStars(root+alnDir,fit=poly,points=points,
                                   withRVonly=True,silent=True)

    ##########
    #
    # Plotting
    #
    ##########
    py.clf()
    usetexTrue()

    # Store some font sizes for labels and ticks
    labelFont = {'fontsize': 14, 'fontweight': 'bold'}

    tickSize = 14

    def dispHist(xx, yy):
        fmt = 'k.'
        fmt2 = 'r--'
        pntsize = 2
        cmap = py.cm.hot_r

        # Make 2D histogram
        (probDist, b1, b2) = h2d.histogram2d(xx, yy, bins=(50, 50))

        # Need to convert the 2d histogram into floats
        probDist = np.array(probDist, float)
        
        # Determine contour levels
        # Flatten and reverse sort our prob. distribution
        sid0 = probDist.flatten().argsort()
        sid = sid0[::-1]
        pixSort = probDist.flatten()[sid]
        
        # Make a cumulative distribution function starting from the
        # highest pixel value. This way we can find the level above
        # which 68% of the trials will fall.
        cdf = np.cumsum(pixSort)
        
        # Determine point at which we reach 68% level
        #percents = array([0.6827, 0.9545, 0.9973]) * len(xx)
        percents = np.array([0.6827, 0.9545]) * len(xx)
        levels = np.zeros(len(percents), float)
        for ii in range(len(levels)):
            # Get the index of the pixel at which the CDF
            # reaches this percentage (the first one found)
            idx = (np.where(cdf < percents[ii]))[0]

            # Now get the level of that pixel
            levels[ii] = pixSort[idx[-1]]
            
        # Mask out the parts where we don't have data.
        foo = np.where(probDist == 0)
        probDist = np.log10(probDist)
        probDist[foo] = 0.0
        levels = np.log10(levels)

        py.imshow(probDist, extent=[b1[0], b1[-1], b2[0], b2[-1]],
                 cmap=cmap, origin='lower', aspect='auto',
                 interpolation='bilinear')

        py.contour(probDist, levels, origin=None, colors='black',
                extent=[b1[0], b1[-1], b2[0], b2[-1]])


    def axisLabel(xtext, ytext, yrange=None):
        # Rescales fonts for ticks and labels
        thePlot = py.gca()
        rng = py.axis()

        # Incrememnt for ticks on the X axis
        tmp = np.abs(float(rng[1]) - float(rng[0])) / 5.0
        xinc = __builtin__.round(tmp, 1)

        if (xinc == 0):
            xinc = 0.05
        
        thePlot.get_xaxis().set_major_locator(py.MultipleLocator(xinc))
        py.setp( thePlot.get_xticklabels(), fontsize=tickSize )
        py.setp( thePlot.get_yticklabels(), fontsize=tickSize )

        # Add axis labels
        py.xlabel(xtext, labelFont)
        py.ylabel(ytext, labelFont)

        # Optional re-scale axes
        if (yrange != None):
            py.axis([rng[0], rng[1], yrange[0], yrange[1]])

    pdf_ph_all = []

    out = open(root + alnDir + 'tables/pdf_phase_outliers%s.txt' % suffix, 'w')
    for star in yng.stars:
        mscStar = False
        mStr = ''
        
        if mosaic == True:
            if (star.name in mscNames) & (star.name not in cntrlNames):
                mscStar = True
                mStr = '(mosaic star)'

        # File contains analytic orbit solutions with acceleration limits (MC)
        print 'Working on %s %s' % (star.name, mStr)
        
        pdffile = '%s%s%s%s.mc.dat' % (root, alnDir, pdfdir, star.name)
        if os.path.exists(pdffile) == False:
            continue

        py.clf()
        py.figure(figsize=(7,10))
        py.subplots_adjust(left=0.1, right=0.96, top=0.95, bottom=0.05,
                           wspace=0.25, hspace=0.25)
        pdf = pickle.load(open(pdffile))

        xdat = pdf.z * dist / cc.au_in_pc

        # Plot Orbital Phase
        py.subplot(3, 1, 1)
        dispHist(xdat, pdf.ph)
        axisLabel(r'{\bf z (pc)}', r'{\bf Orbital Phase}', yrange=[0, 1.0])

        if ('irs' in star.name):
            starLabel = '{\\bf IRS %s}' % (star.name[3:])
        else:
            starLabel = '{\\bf %s}' % (star.name)

        py.title(starLabel, labelFont)

        py.subplot(3, 1, 2)
        (bins, data) = histNofill.hist(np.arange(0, 1+0.05, 0.05), pdf.ph)
        #(bins, data) = histNofill.hist(np.arange(0, 1, 0.01), pdf.ph)
        py.plot(bins, data / len(pdf.ph))
        axisLabel(r'{\bf Orbital Phase}', r'{\bf PDF}')
        rng = py.axis()
        py.axis([0, 1, rng[2], rng[3]])

        py.subplot(3, 1, 3)
        py.plot(pdf.m/1.e6, pdf.ph, 'k.')
        axisLabel(r'{\bf MC Mass ($M_{BH} + M_{ext}$)', r'{\bf Orbital Phase}')

        py.savefig(root + alnDir + pdfdir + 'pdfparams/pdf_phase_%s.png' % star.name, dpi=75)
        py.close()

        # Find the outliers that are contributing to the peak near phase=0
        if pdf.ph.mean() < 0.05:
            print '**** %s: <phase> = %5.3f +- %5.3f' % \
                  (star.name, pdf.ph.mean(), pdf.ph.std())
        if data.argmax() < 2:
            print '**** %s: <phase> = %5.3f +- %5.3f' % \
                  (star.name, pdf.ph.mean(), pdf.ph.std())
            print '         Peak of PDF = %5.3f, phase = [%4.2f - %4.2f]' % \
                  (data.max(), bins[data.argmax()], bins[data.argmax()+2])
            out.write('%s\n' % star.name)

        # Save all PDFs to plot the sum later
        pdf_ph_all =  np.concatenate([pdf_ph_all, pdf.ph])

    out.close()

    # Call function to examine the phase outliers a little more carefully
    if examineOutliers == True:
        phaseOutliers(pdfdir,mosaic=mosaic,extendedMass=False,suffix=suffix)

    binsIn = np.arange(0, 1.05, 0.05)
    (bb1, nn1) = histNofill.hist(binsIn, pdf_ph_all, normed=True)
    py.clf()
    py.figure(figsize=(7,7))
    py.plot(bb1, nn1, color='black', linewidth=2)
    py.xlabel('Orbital Phase')
    py.ylabel('Probability Density')
    py.xlim(0, 1.0)
    py.savefig(root + alnDir + pdfdir + 'pdfparams/pdf_phase_allCombined%s.png' % suffix)
    py.close()

    usetexFalse()


def phaseOutliers(pdfdir, mosaic=False, extendedMass=False, suffix=''):
    """
    Examine the kinematics of the outliers in orbital phase
    """
    cc = objects.Constants()

    # Read in file containing the stars with very low phase
    infile = asciidata.open(root + alnDir + 'tables/pdf_phase_outliers%s.txt' % suffix)
    ol = infile[0].tonumpy()

    # Get all sorts of info for these stars
    # Load names of young stars 
    if mosaic == True:
        # Load up mosaic data as well; select only stars at r>4, since
        # we don't want to add any info from mosaics if we have it in
        # the central 10" already
        yng1 = young.loadYoungStars(root+alnDir,fit=poly,points=points,
                                    withRVonly=True,silent=True,skipStar=['S5-237']) 
        yng2 = young.loadYoungStars(mscDir,fit=polyM,points=pointsM,
                                    mosaic=True, withRVonly=True,silent=True)
        cntrlNames = yng1.getArray('name')
        mscNames = yng2.getArray('name')
        # Merge this object with object from central 10" analysis
        yng = merge(yng1, yng2)
    else:
        yng = young.loadYoungStars(root+alnDir,fit=poly,points=points,
                                   withRVonly=True,silent=True)
    yngNames = yng.getArray('name')

    py.clf()
    py.figure(figsize=(10,10))
    py.subplots_adjust(left=0.1, right=0.96, top=0.95,
                       wspace=0.25, hspace=0.25)
    
    for name in yngNames:
        # Get the star object
        idx = yngNames.index(name)
        star = yng.stars[idx]
        mag = star.mag
        r2d = star.r2d # arcsec

        mscStar = False
        mStr = ''
        
        if mosaic == True:
            if (name in mscNames) & (name not in cntrlNames):
                mscStar = True
                alnTmp = mscDir
                ptsTmp = pointsM
                xfit = star.fitXv
                yfit = star.fitYv
            else:
                alnTmp = root + alnDir
                ptsTmp = points
                xfit = star.fitXa
                yfit = star.fitYa
        else:
            alnTmp = root + alnDir
            ptsTmp = points
            xfit = star.fitXa
            yfit = star.fitYa

        # How many epochs was each star detected in?
        pts = asciidata.open('%s%s%s.points' % (alnTmp, ptsTmp, name))
        ep = pts[0].tonumpy()
        cnt = len(ep)


        # Polyfit results
        x = xfit.p
        y = yfit.p
        xe = xfit.perr
        ye = yfit.perr
        vx = xfit.v # km/s
        vy = yfit.v # km/s
        vz = star.vz # km/s
        vxe = xfit.verr # km/s
        vye = yfit.verr # km/s
        vze = star.vzerr # km/s
        if mscStar == False:
            ax = xfit.a
            ay = yfit.a
            axe = xfit.aerr
            aye = yfit.aerr
        t0 = xfit.t0

        pm = np.hypot(vx, vy)
        pm_err = np.sqrt(((vx*vxe)**2 + (vy*vye)**2) / pm**2) # km/s
        vel_ratio = pm / vz

        # Angle between r2d and proper motion (dot product)
        #rdotv = x*vx + y*vy
        rdotv = np.vdot([x,y],[vx,vy])
        rdotverr = (x*vxe)**2 + (y*vye)**2

        v3d = np.sqrt(vx**2 + vy**2 + vz**2)
        v3d_err = np.sqrt(((vx*vxe)**2 + (vy*vye)**2 + (vz*vze)**2) / v3d**2) # km/s

        r_au = r2d * dist
        r_pc = r_au / cc.au_in_pc
        r_cm = r_au * cc.cm_in_au

        if extendedMass == True:
            # Include extended mass distribution from Trippe et al. (2008)
            Mbh = mass 	# solar masses
            rho0 = 2.1e6  	# solar masses/pc^3
            Rb_as = 8.9	# break radius; arcsec
            r2d_pc = r_pc
            Rb = Rb_as * dist / 206265. # pc
            const = 4.0 * np.pi * rho0
            Mext = scipy.integrate.quad(lambda r: r**2 / (1 + (r / Rb)**2), 0, r2d_pc)
            Mext = const * Mext[0]
                
            mTot = (Mbh + Mext) * cc.msun
            GM = cc.G * mTot
        else:
            GM = cc.G * mass * cc.msun

        # Also determine theoretical escape velocity for this r2d
        vesc_cm_s = np.sqrt(2. * GM / r_cm)
        vesc_km_s = vesc_cm_s / 1.0e5

        # Error on the computed theoretical v_esc 
        sig_vesc_cgs = massErr * cc.msun * np.sqrt(cc.G / (2.*mass*cc.msun*r_cm))
        sig_vesc_mks = sig_vesc_cgs / 1.0e5

        vesc_ratio = v3d / vesc_km_s
        # Properly propagate the error on the vesc_ratio
        v3d_cms = v3d * 1.e5
        v3d_err_cms = v3d_err * 1.e5
        vesc_ratio_err = np.sqrt((v3d_err_cms**2 / vesc_cm_s**2) +
                                 (v3d_cms**2 * sig_vesc_cgs**2 / (4 * (vesc_cm_s**2)**3)))

        angle = np.arccos(rdotv / (r2d*pm)) * 180./np.pi

        #(ar, at, are, ate) = util.xy2circErr(x, y, ax, ay,
        #                                     xe, ye, axe, aye)

        if name in ol:
            fmt = 'r.'
            mfc = 'r'
        else:
            fmt = 'k.'
            mfc = 'None'
            
        # Plot PM error vs. RV error
        py.subplot(2,2,1)
        py.plot(pm_err, vze, fmt, mfc=mfc)

        # Plot angle between r2d and v2d vs. number of epochs
        py.subplot(2,2,2)
        py.plot(cnt, angle, fmt, mfc=mfc)

        # Plot angle between r2d and v2d vs. r2d
        py.subplot(2,2,3)
        py.plot(r2d, angle, fmt, mfc=mfc)

        # Plot angle between r2d and v2d vs. ratio of v3d/v_esc 
        py.subplot(2,2,4)
        py.errorbar(vesc_ratio, angle, xerr=vesc_ratio_err, fmt=fmt, mfc=mfc)


    py.subplot(2,2,1)
    py.xlabel('Proper Motion Error (km/s)')
    py.ylabel('Radial Velocity Error (km/s)')
    py.plot([24], [123], 'r.')
    py.text(25, 120, 'Phase Outliers', color='r')

    py.subplot(2,2,2)
    py.plot([0,45],[90,90],'k--')
    py.xlabel('Number of Epochs')
    py.ylabel('Angle Between R2D and PM (deg)')
    
    py.subplot(2,2,3)
    py.plot([0,7],[90,90],'k--')
    py.xlabel('Projected Radius (arcsec)')
    py.ylabel('Angle Between R2D and PM (deg)')

    py.subplot(2,2,4)
    #py.plot([-50,250],[90,90],'k--')
    py.xlabel(r'$V_{3D} / V_{escape}$')
    py.ylabel('Angle Between R2D and PM (deg)')

    if extendedMass == True:
        py.savefig(root + alnDir + pdfdir + 'pdfparams/phase_outliers_Mext%s.png' % suffix)
        py.close()
    else:
        py.savefig(root + alnDir + pdfdir + 'pdfparams/phase_outliers%s.png' % suffix)
        py.close()


def pdfOrbitalPhase(pdfdir='aorb_thesis/', mosaic=True,
                    simFlat=False, suffix='_mosaic'):
    """
    Plot of the distribution of orbital phases to
    see if there is any coherence for stars in the disk
    """
    #(phase, phaseDisk) = paramStats('ph3',pdfdir=pdfdir)
    #(phase, phaseDisk) = paramStats('ph2',pdfdir=pdfdir)
    (phase, phaseDisk) = paramStats('ph', pdfdir=pdfdir, mosaic=mosaic, simFlat=simFlat)
    
    # Determine the eomega probability density function
    step = 0.05
    #binsIn = np.arange(-0.5, 0.5, step)
    binsIn = np.arange(0, 1.0+step, step)

    (bbdisp1, nndisp1) = histNofill.hist(binsIn, phase, normed=True)
    (bbdisp2, nndisp2) = histNofill.hist(binsIn, phaseDisk, normed=True)

    py.clf()
    py.figure(figsize=(7,7))
    py.subplots_adjust(left=0.15, right=0.95)
    py.plot(bbdisp1, nndisp1, color='black', linewidth=2)
    py.xlabel('Orbital Phase')
    py.ylabel('Probability Density')
    py.xlim(0, 1.0)
    #py.xlim(-0.5, 0.5)
    py.savefig(plotdir + 'pdf_phase_all%s.png' % suffix)
    py.close()

    py.clf()
    py.subplots_adjust(left=0.15, right=0.95)
    py.plot(bbdisp2, nndisp2, color='black', linewidth=2)
    py.xlabel('Orbital Phase')
    py.ylabel('Probability Density')
    py.xlim(0, 1.0)
    #py.xlim(-0.5, 0.5)
    py.savefig(plotdir + 'pdf_phase_disk%s.png' % suffix)
    py.close()

    py.clf()
    py.subplots_adjust(left=0.15, right=0.95)
    p1 = py.plot(bbdisp1, nndisp1, color='gray', linewidth=2)
    p2 = py.plot(bbdisp2, nndisp2, color='black', linewidth=2)
    py.legend((p1, p2), ('All Solutions', 'Disk Solutions'))
    py.xlabel('Orbital Phase')
    py.ylabel('Probability Density')
    py.xlim(0, 1.0)
    #py.xlim(-0.5, 0.5)
    py.savefig(plotdir + 'pdf_phase_both%s.png' % suffix)
    py.close()


def calcDiskDensity(weighted=False, mosaic=True, suffix='_mosaic'):
    cc = objects.Constants()

    pdfdir = 'aorb_thesis/'
    #pdfdir = 'aorb_acc_mrPDF_MC_newMosaic/'

    # Load disk star names and probability of disk membership
    (names, diskP) = readDiskProb(suffix=suffix, diskOnly=False)

    # Load disk star weights
    #if (weighted == True):
    #    weight = readDiskWeight(extended=extended)

    # Load kinematic data for disk stars
    yng = loadYoungByName(names, mosaic=mosaic)

    yngNames = yng.getArray('name')
    x = yng.getArray('x') * dist / cc.au_in_pc
    y = yng.getArray('y') * dist / cc.au_in_pc
    z = np.array([aorb.plane2z(x[i], y[i]) for i in range(len(x))])

    # Switch everything into disk coordinate system. Coordinates
    # are p, q, n where
    #   n is tangential to the disk plane.
    #   p goes along the ascending node in the disk plane.
    #   q is perpindicular to p in the disk plane.
    # Get the directional vectors
    (p, q, n) = diskProject(x, y, z, idisk=idisk, odisk=odisk)

    rInDisk = np.sqrt(p**2 + q**2)

    ##########
    #
    # Calculate Density
    #
    ##########
    # Divide sky up into grid of 0.02 x 0.02 pc (0.5")
    coordP = np.arange(-0.4, 0.401, 0.02)
    coordQ = np.arange(-0.4, 0.401, 0.02)
    grid = np.zeros((len(coordQ), len(coordP)), dtype=float)

    for pp in range(len(coordP)):
        pdist = p - coordP[pp]

        for qq in range(len(coordQ)):
            qdist = q - coordQ[qq]

            # Calculate the distance to this point (in mpc)
            rdist = np.sqrt(pdist**2 + qdist**2)

            #-- Nearest Neighbor Analysis
            # Sort the distances and take the 5 nearest neighbors
            sdx = rdist.argsort()            

            starCount = 4    # neighbor
            area = math.pi * rdist[sdx[starCount-1]]**2

            if (weighted == True):
                #-- Weighted Method 2 --#
                cumsumWeight = np.cumsum(weight[sdx])
                ii = (where(cumsumWeight < (starCount * weightFactor)))[0]
                starCount = cumsumWeight[ii[-1]]
                #starCount = ii[-1]
                area = math.pi * rdist[sdx[ii[-1]]]**2

                #-- Weighted Method 1 --#
                #starCount = weight[sdx[0:starCount-1]].sum()

            #-- Aperture Analysis
            #radius = 0.06
            #rdx = (where(rdist <= radius))[0]
            #starCount = len(rdx)
            #if (weighted == True):
            #   starCount = weight[rdx].sum()
            #area = math.pi * radius**2

            grid[qq, pp] = float(starCount) / area

    # Find the peak in the surface density
    idx = np.where(grid == grid.max())
    peakP = coordP[idx[1][0]]
    peakQ = coordQ[idx[0][0]]
    print 'Peak = %3d stars/pc^2' % (grid.max())
    print '    at (P = %5.2f,  Q = %5.2f)' % (peakP, peakQ)

    return (coordP, coordQ, grid)

def calcDiskDensityMC(recalcPQ=True, weighted=False, mosaic=True, suffix='_mosaic'):
    """
    Calculate the surface density in the disk plane by using the results
    of the MC simulations and weight by the likelihood of disk membership.
    From the simulations, for each pixel, pick out the most probable
    surface density and the 68% high and low limits.

    Parameters:
    recalcPQ -- (def=True) Reload all star's coordinates from the Monte Carlo
                simulation results (e.g. aorb_efit/S0-15.mc.dat). If false,
                then pull these values from a pickled file from a previous
                call to this function. This is mainly just a time saver.
    weighted -- (def=False) Use weighting scheme if True (broken).

    Results:
    tables/calcDiskDensityMC_pq.dat (or _ext.dat)
    -- this file contains the disk coordinates for every MC trial for every
       star. Created with recalcPQ=True, reloaded with recalcPQ=False.

    tables/calcDiskDensityMC_results_0.04.dat (or _ext.dat)
    -- contains 2D gridded disk plane coordinates and surface density values.
    -- 1st tuple (coordP, coordQ, coordStep)
    -- 2nd tuple (gridMax, gridLo, gridHi)

    tables/calcDiskDensityMC_densityPDFs_0.04.dat (or _ext.dat)
    -- contains 3D grid of disk plane (P x Q x MCtrial) for all trials.
    -- 1st tuple (coordP, coordQ, coordStep)
    -- 2nd tuple (densityPDFs)
    """
    cc = objects.Constants()

    pdfdir = 'aorb_thesis/'
    #pdfdir = 'aorb_acc_mrPDF_MC_newMosaic/'

    # Load disk star names and probability of disk membership
    (names, diskP) = readDiskProb(suffix=suffix,diskOnly=True)

    # Load disk star weights
    #if (weighted == True):
    #    weight = readDiskWeight(extended=extended)

    starCnt = len(names)
    numTrials = 100000

    #----------
    #
    # Load up Prob(p, q, n) for each star on disk
    #   -- take all solutions within 10 deg of disk
    #   -- construct 100,000 trials of this prob distribution
    #   -- do this by repeating the results of the MC simulation
    #      until you have enough.
    #
    #----------
    if (recalcPQ == True):
        pAll = np.zeros((starCnt, numTrials), dtype=float)
        qAll = np.zeros((starCnt, numTrials), dtype=float)

        # Load up all the PDF files.
        print 'Loading PDFs'
        py.clf()
        for ss in range(starCnt):
            name = names[ss]

            pdffile = '%s%s.mc.dat' % (pdfdir, name)
            pdf = pickle.load(open(pdffile))
            
            idx = whereInDisk(pdf, angleCut=angleCut)
            print '%-13s initially has %5d disk solutions' % (name, len(idx))

            x = pdf.x[idx] * dist / cc.au_in_pc # pc
            y = pdf.y[idx] * dist / cc.au_in_pc
            z = pdf.z[idx] * dist / cc.au_in_pc
                        
            (p, q, n) = diskProject(x, y, z, idisk=idisk, odisk=odisk)
            
            # Repeat until we have 100,000 of this prob dist.
            pTmp = p
            while (len(pTmp) < numTrials):
                pTmp = np.concatenate((pTmp, p))

            pAll[ss,:] = pTmp[0:numTrials]

            qTmp = q
            while (len(qTmp) < numTrials):
                qTmp = np.concatenate((qTmp, q))

            qAll[ss,:] = qTmp[0:numTrials]

        pickFile = open(root+alnDir+'tables/calcDiskDensityMC_pq'+suffix+'.dat', 'w')
        pickle.dump((pAll, qAll), pickFile)
        pickFile.close()

    else:
        print 'Loading P, Q from File'
        pickFile = open(root+alnDir+'tables/calcDiskDensityMC_pq'+suffix+'.dat', 'r')
        pAll, qAll =  pickle.load(pickFile)
        pickFile.close()
        

    ##########
    #
    # Calculate Density
    #
    ##########
    print 'Calculating density for each trial'

    # Divide sky up into grid of 0.04 x 0.04 pc (1.0")
    coordStep = 0.04
    #coordP = np.arange(-0.4, 0.401, coordStep)
    #coordQ = np.arange(-0.4, 0.401, coordStep)
    coordP = np.arange(-1.5, 1.501, coordStep)
    coordQ = np.arange(-1.5, 1.501, coordStep)

    densityStep = 20
    densityBins = np.arange(0, 3000+(2*densityStep), densityStep, dtype=float)
    radius = 2.0 * coordStep
    area = math.pi * radius**2

    # Calculate the average and median (or peak and 68% confidence)
    # for each grid pixel.
    gridMax = np.zeros((len(coordQ), len(coordP)), dtype=float)
    gridLo = np.zeros((len(coordQ), len(coordP)), dtype=float)
    gridHi = np.zeros((len(coordQ), len(coordP)), dtype=float)

    densityPDFs = np.zeros((len(coordQ), len(coordP), len(densityBins)-1),
                        dtype=float)
            
    # Array for all densities for a given pp, qq pair
    # This gets re-used each loop.
    densities = np.zeros(numTrials, dtype=float)

    for pp in range(len(coordP)):
        pdist = pAll - coordP[pp]
        pdist2 = pdist**2

        for qq in range(len(coordQ)):
            qdist = qAll - coordQ[qq]
            qdist2 = qdist**2

            # Calculate the distance from this point
            # for all stars, all trials (stars, trials)
            distAll = np.sqrt(pdist2 + qdist2)
            distAllSort = distAll.copy()
            distAllSort.sort(axis=0)

            for nn in range(numTrials):
                # Get the distances for all stars, this trial
                rdist = distAll[:,nn]
                distSort = distAllSort[:,nn]

                ### Nearest neighbor###
                # Sort the distances and take the 6th nearest neighbor
                starCount = 6    # neighbor

                area = math.pi * distSort[starCount-1]**2

                if (weighted == True):
                    #-- Weighted Method 2 --#
                    sdx = rdist.argsort()
                    cumsumWeight = np.cumsum(weight[sdx])
                    ii = (np.where(cumsumWeight < (starCount * weightFactor)))[0]
                    
                    starCount = cumsumWeight[ii[-1]]
                    #starCount = ii[-1]
                    area = math.pi * rdist[sdx[ii[-1]]]**2

                    #-- Weighted Method 1 --#
                    #starCount = weight[sdx[0:starCount-1]].sum()

                #-- Aperture Analysis
                #radius = 0.06
                #rdx = (np.where(rdist <= radius))[0]
                #starCount = len(rdx)
                #if (weighted == True):
                #   starCount = weight[rdx].sum()
                #area = math.pi * radius**2

                ### Save Results ###
                densities[nn] = starCount / area

            #----------
            #
            # Combining all Trials
            #
            #----------

            #*** Average/Standard Deviation Method ***#
            #gridMax[qq, pp] = densities.mean()
            #gridLo[qq, pp] = densities.std()
            #gridHi[qq, pp] = gridLo[qq, pp]

            #(probDist, bins) = matplotlib.mlab.hist(densities,
            #                                        bins=densityBins)
            (probDist, bins, ppp) = py.hist(densities, bins=densityBins)

            densityPDFs[qq, pp] = probDist
            
            idx = np.where(probDist[1:] != 0)
            if (len(idx) == 0):
                continue

            probDist = np.array(probDist, dtype=float)
            probDist /= float(numTrials)

            #*** 68% confidence, symmetric method ***#
            #----------
            # Determine 68% contour around the peak
            #----------
#             # Find peak in prob dist.
#             peakBin = probDist.argmax()

#             # Walk out from the peak (assumes single peaked)
#             diffBins = abs(densityBins - densityBins[peakBin])
#             sid = diffBins.argsort()            

#             # Sum up probability until we get out to 1 sigma (68%)
#             cdf = cumsum(probDist[sid])
#             idx = (np.where(cdf >= 0.6827))[0]
#             hiBins = sid[0:idx[0]]
#             if (len(hiBins) == 0):
#                 hiBins = array([0])

#             # Find how far out we had to go to get there.
#             minBin = hiBins.min()
#             maxBin = hiBins.max()

#             peakValue = densityBins[peakBin]
#             minValue = densityBins[minBin]
#             maxValue = densityBins[maxBin]
            
#             gridMax[qq, pp] = peakValue
#             gridLo[qq, pp] = abs(minValue - peakValue)
#             gridHi[qq, pp] = abs(maxValue - peakValue)

#             if (gridHi[qq, pp] == 0):
#                 gridHi[qq, pp] = densityStep #/ 2.0
            
            #*** 68% confidence, probability level method ***#
            # Determine 68% contour level
            sid0 = probDist.argsort()
            sid = sid0[::-1] # reverse
            probSorted = probDist[sid]
            cdf = np.cumsum(probSorted)
            idx = (np.where(cdf >= 0.6827))[0]
            
            # Contours
            if (len(idx) > 0):
                level68 = probSorted[idx[0]]

                # Peak
                gridMax[qq, pp] = densityBins[sid[0]]
            
                # Now pull out all bins with probabilities higher
                # than the 68% level.
                idx = (np.where(probDist >= level68))[0]

                # Now save the peak, upper, and lower errorbars
                gridLo[qq, pp] = np.abs(densityBins[idx[0]] - gridMax[qq, pp])
                gridHi[qq, pp] = np.abs(densityBins[idx[-1]] - gridMax[qq, pp])

                if (gridHi[qq, pp] == 0):
                    gridHi[qq, pp] = densityStep #/ 2.0


            #clf()
            #plot(densityBins, probDist)
            #xlabel('Density (stars/arcsec^2)')
            #ylabel('Probability Distribution')
            #title('P = %5.2f, Q = %5.2f' % (coordP[pp], coordQ[qq]))
            print 'P = %5.2f, Q = %5.2f: max = %4d (%4d - %4d)' % \
                  (coordP[pp], coordQ[qq], gridMax[qq,pp],
                   gridMax[qq,pp] - gridLo[qq,pp],
                   gridMax[qq,pp] + gridHi[qq,pp]), time.ctime()

    pickFile = open(root+alnDir+'tables/calcDiskDensityMC_results'+suffix+'.dat', 'w')
    pickle.dump((coordP, coordQ, coordStep), pickFile)
    pickle.dump((gridMax, gridLo, gridHi), pickFile)
    pickFile.close()

    pickFile = open(root+alnDir+'tables/calcDiskDensityMC_densityPDFs'+suffix+'.dat', 'w')
    pickle.dump((coordP, coordQ, coordStep), pickFile)
    pickle.dump((densityPDFs), pickFile)
    pickFile.close()


def plotDiskDensityMC(mosaic=True, suffix='_mosaic'):
    cc = objects.Constants()

    #(tmp, suffix) = lu.loadFilenameVars(extended=extended)

    pickFile = open(root+alnDir+'tables/calcDiskDensityMC_results'+suffix+'.dat', 'r')
    (coordP, coordQ, coordStep) = pickle.load(pickFile)
    (gridMax, gridLo, gridHi) = pickle.load(pickFile)
    pickFile.close()

    #--------------------------------------------------
    #
    # Plotting
    #
    #--------------------------------------------------
    
    # Need to determine the corners of our field of view
    # so that we know what our selection effects are.
    (cornersP, cornersQ, cornersN) = getDiskCornersPQN()
    (innerP, innerQ, innerN) = getDiskInnerEdgePQN()

    #----------
    # Observed Density
    #----------
    py.clf()
    py.subplots_adjust(left=0.08, right=0.9, top=0.8, bottom=0.1)
    py.imshow(gridMax, origin='lower', cmap=py.cm.hot_r,
              extent=[coordP[0], coordP[-1],
                      coordQ[0], coordQ[-1]])
    py.plot([0], [0], 'kx')
    py.plot(cornersP, cornersQ, 'k--')
    py.plot(innerP, innerQ, 'k-.')
    #py.plot([-0.038], [0.064], 'kd')
    #py.axis([0.4, -0.4, -0.4, 0.4])
    py.axis([1.5, -1.5, -1.5, 1.5])

    # Colorbar Wonderful... gotta work to make it pretty
    cbar = py.colorbar(orientation='horizontal')

    # Swap the colorbar's axis to the top
    cbarX = cbar.ax.get_xaxis()
    cbarX.set_ticks_position('top')
    cbarX.set_label_position('top')

    # Swap positions for colorbar and plot itself
    #mainLeft, mainLower, mainWidth, mainHeight = py.gca().get_position().bounds
    #cbarLeft, cbarLower, cbarWidth, cbarHeight = cbar.ax.get_position().bounds

    py.gca().set_position([0.15, 0.1, 0.8, 0.8])
    cbar.ax.set_position([0.15, 0.8, 0.8, 0.1])

    cbarX.get_label().set_text('Surface Density (stars/pc^2)')

    py.xlabel('Disk Plane p-Direction (pc)')
    py.ylabel('Disk Plane q-Direction (pc)')
    py.savefig(plotdir+'diskDensityMC'+suffix+'.png')
    py.close()

    #----------
    # Observed Density
    #----------
    py.clf()
    py.imshow(gridHi, origin='lower', cmap=py.cm.hot_r,
              extent=[coordP[0], coordP[-1], coordQ[0], coordQ[-1]])
    py.plot([0], [0], 'kx')
    py.plot(cornersP, cornersQ, 'k--')
    py.plot(innerP, innerQ, 'k-.')
    #py.plot([-0.038], [0.064], 'kd')
    #py.axis([0.4, -0.4, -0.4, 0.4])
    py.axis([1.5, -1.5, -1.5, 1.5])

    # Colorbar Wonderful... gotta work to make it pretty
    cbar = py.colorbar(orientation='horizontal')

    # Swap the colorbar's axis to the top
    cbarX = cbar.ax.get_xaxis()
    cbarX.set_ticks_position('top')
    cbarX.set_label_position('top')

    # Swap positions for colorbar and plot itself
    #mainLeft, mainLower, mainWidth, mainHeight = py.gca().get_position()
    #cbarLeft, cbarLower, cbarWidth, cbarHeight = cbar.ax.get_position()

    py.gca().set_position([0.15, 0.1, 0.8, 0.8])
    cbar.ax.set_position([0.15, 0.8, 0.8, 0.1])

    cbarX.get_label().set_text('Surface Density (stars/pc^2)')

    py.xlabel('Disk Plane p-Direction (pc)')
    py.ylabel('Disk Plane q-Direction (pc)')
    py.savefig(plotdir+'diskDensityMC_hi'+suffix+'.png')
    py.close()

    #----------
    # Observed Density
    #----------
    py.clf()
    py.imshow(gridLo, origin='lower', cmap=py.cm.hot_r,
              extent=[coordP[0], coordP[-1], coordQ[0], coordQ[-1]])
    py.plot([0], [0], 'kx')
    py.plot(cornersP, cornersQ, 'k--')
    py.plot(innerP, innerQ, 'k-.')
    #py.plot([-0.038], [0.064], 'kd')
    #py.axis([0.4, -0.4, -0.4, 0.4])
    py.axis([1.5, -1.5, -1.5, 1.5])

    # Colorbar Wonderful... gotta work to make it pretty
    cbar = py.colorbar(orientation='horizontal')

    # Swap the colorbar's axis to the top
    cbarX = cbar.ax.get_xaxis()
    cbarX.set_ticks_position('top')
    cbarX.set_label_position('top')

    # Swap positions for colorbar and plot itself
    #mainLeft, mainLower, mainWidth, mainHeight = py.gca().get_position()
    #cbarLeft, cbarLower, cbarWidth, cbarHeight = cbar.ax.get_position()

    py.gca().set_position([0.15, 0.1, 0.8, 0.8])
    cbar.ax.set_position([0.15, 0.8, 0.8, 0.1])

    cbarX.get_label().set_text('Surface Density (stars/pc^2)')

    py.xlabel('Disk Plane p-Direction (pc)')
    py.ylabel('Disk Plane q-Direction (pc)')
    py.savefig(plotdir+'diskDensityMC_lo'+suffix+'.png')
    py.close()

#def syGetRadialDist(numTrials=10000, suffix='_mosaic'):
#    
#    cc = objects.Constants()
#
#    pdfdir = 'aorb_acc_mrPDF_MC_newMosaic/'
#
#    # Load disk star names and probability of disk membership
#    (names, diskP) = readDiskProb(suffix=suffix,diskOnly=True)
#    starCnt = len(names)
#
#    # Define radial bins in pc
#    rstep = 0.02
#    rbins = np.arange(0.032, 0.5+rstep, rstep)
#
#    rAll = np.zeros((starCnt, numTrials), dtype=float)
#
#    # Loop through and trim down to only disk solutions
#    for ii in range(starCnt):
#        name = names[ii]
#
#        # File contains analytic orbit solutions with acceleration limits (MC)
#        pdffile = '%s%s%s%s.mc.dat' % (root, alnDir, pdfdir, name)
#        pdf = pickle.load(open(pdffile))
#
#        # Get radius PDF for disk solutions
#        adx = whereInDisk(pdf, angleCut=angleCut)
#
#        r = np.sqrt(pdf.x[adx]**2 + pdf.y[adx]**2 + pdf.z[adx]**2)
#        r *= dist / cc.au_in_pc     # pc
#
#        # Why was this done? (stacking the position PDF until there were 100000 positions)??
#        # Repeat until we have 100,000 of this prob dist.
#        rTmp = r
#        while (len(rTmp) < numTrials):
#            rTmp = np.concatenate((rTmp, r))
#
#        rAll[ii,:] = rTmp[0:numTrials]
#
#        print '%-13s  r = %6.3f +/- %6.3f pc' % (name, r.mean(), r.std(ddof=1))
#


def calcDiskRadialDist(weighted=False, mosaic=True, top20=False, suffix='_mosaic'):
    """
    Calculate the density within the disk plane. Also run a Monte Carlo
    using the measured radial distribution and random azimuthal
    distribution to see if the disk is really azimuthally asymmetric.

    Method: Reproduce observed orbital solutions until you get to
    100,000 trials. Then bin up into 2D histogram of P(n, r). 

    Parameters:
    top20 -- select only the top 20% of candidate disk members
    weighted -- use weighting algorithm (doesn't work; def = False)
    """
    cc = objects.Constants()

    pdfdir = 'aorb_thesis/'
    #pdfdir = 'aorb_acc_mrPDF_MC_newMosaic/'

    # Load disk star names and probability of disk membership
    (names, diskP) = readDiskProb(suffix=suffix,diskOnly=True,top20=top20)

    # Load disk star weights
    weight = None
    if (weighted == True):
        weight = readDiskWeight()

    starCnt = len(names)
    numTrials = 99998
    #numTrials = 100000

    # Info for determining the radial distribution
    rStep = 0.025
    #rBins = arange(0.025, 1.0, rStep)
    #rLo = array([0.8, 2, 3, 4, 6, 8, 10]) * 0.04 # pc
    #rHi = array([2, 3, 4, 6, 8, 10, 15]) * 0.04  # pc
    #rLo = np.array([0.032, 0.090, 0.115, 0.140, 0.165, 0.215, 0.280, 0.330, 0.410, 0.550, 0.700]) # pc
    #rHi = np.array([0.090, 0.115, 0.140, 0.165, 0.215, 0.280, 0.330, 0.410, 0.550, 0.700, 1.000]) # pc
    #rLo = np.array([0.032, 0.090, 0.115, 0.140, 0.165, 0.190, 0.270, 0.360, 0.450, 0.550, 0.700]) # pc
    #rHi = np.array([0.090, 0.115, 0.140, 0.165, 0.190, 0.270, 0.360, 0.450, 0.550, 0.700, 1.000]) # pc
    rLo = np.array([0.032, 0.045, 0.060, 0.090, 0.115, 0.140, 0.165, 0.190, 0.270, 0.360, 0.450]) # pc
    rHi = np.array([0.045, 0.060, 0.090, 0.115, 0.140, 0.165, 0.190, 0.270, 0.360, 0.450, 0.550]) # pc
    #rLo = np.arange(0.025, 1.0, rStep)
    #rHi = np.arange(0.025+rStep, 1.0+rStep, rStep)
    rBins = rLo + ((rHi - rLo) / 2.0)
    rDist = np.zeros((starCnt, len(rBins)), dtype=float)

    rAll = np.zeros((starCnt, numTrials), dtype=float)

    # Loop through and trim down to only disk solutions
    for ii in range(starCnt):
        name = names[ii]

        # File contains analytic orbit solutions with acceleration limits (MC)
        pdffile = '%s%s%s%s.mc.dat' % (root, alnDir, pdfdir, name)
        pdf = pickle.load(open(pdffile))

        # Get radius PDF for disk solutions
        adx = whereInDisk(pdf, angleCut=angleCut)

        r = np.sqrt(pdf.x[adx]**2 + pdf.y[adx]**2 + pdf.z[adx]**2)
        r *= dist / cc.au_in_pc     # pc

        # Repeat until we have 100,000 of this prob dist.
        rTmp = r
        while (len(rTmp) < numTrials):
            rTmp = np.concatenate((rTmp, r))

        rAll[ii,:] = rTmp[0:numTrials]

        print '%-13s  r = %5.2f +/- %5.2f' % (name, r.mean(), r.std(ddof=1))


    ##########
    #
    # Starting probability calculation
    #
    ##########
    prob = np.zeros((starCnt+1, len(rBins)), dtype=float)


    ##########
    # For each possible radius and number of stars at that radius,
    # figure out the probability of that occurrence.
    ##########
    nbins = np.arange(0, starCnt+1)
    # Loop for each radius
    for rr in range(len(rBins)):

        print time.ctime(), '  Start: r = %5.3f' % rBins[rr]
        #rlo = rBins[rr]
        #rhi = rBins[rr] + rStep
        rlo = rLo[rr]
        rhi = rHi[rr]

        # Loop for each trial
        # Make the probability distribution function of N.
        for tt in range(numTrials):
            
            rdx = (np.where((rAll[:,tt] > rlo) & (rAll[:,tt] < rhi)))[0]
            numStars = len(rdx)

            if (weighted == True):
                numStars = np.round(weight[rdx].sum())

            prob[numStars, rr] += 1.0

        prob[:, rr] /= numTrials

    print time.ctime(), '  Finish'

    # Save this P(n, r) to a pickle file for reloading at a later
    # date because it is freaking expensive to keep calculating.
    _pic = open(root+alnDir+'tables/diskRadialDistProb2'+suffix+'.dat', 'w')
    pickle.dump((rBins, rLo, rHi), _pic)
    pickle.dump(prob, _pic)
    _pic.close()

def plotDiskRadialDist(mosaic=True, suffix='_mosaic'):
    """
    Load up the P(NumStars, radius) calculated using calcDiskRadialDist()
    and plot and analyze the results.

    Parameters:
    """
    cc = objects.Constants()

    pdfdir = 'aorb_thesis/'
    #pdfdir = 'aorb_acc_mrPDF_MC_newMosaic/'

    # Load up results from calc'ing the 1D radial distribution
    # Results are actually in the form Prob(N, r)
    _pic = open(root+alnDir+'tables/diskRadialDistProb2'+suffix+'.dat', 'r')
    (rBins, rLo, rHi) = pickle.load(_pic)
    prob = pickle.load(_pic)
    _pic.close()

    sBins = np.arange(0, prob.shape[0])

    ### Plotting ###
    #
    # P(N, r)
    #
    py.clf()

    # Mask out where there is zero probability to make it easier to see.
    probmask = np.ma.masked_where(prob == 0, prob)
    py.imshow(probmask, extent=[rBins[0], rBins[-1], sBins[0], sBins[-1]],
	      aspect='auto')
    py.xlabel('Radius in Disk (pc)')
    py.ylabel('Number of Stars')
    py.title('Probability P(N, r)')
    py.colorbar()
    py.savefig(plotdir+'diskRadialDistProb2d'+suffix+'.png')
    py.close()

    ##########
    #
    # Figure out the correction factor for FOV effects.
    # Do this brute force by gridding and integrating.
    # Only applicable when using the primary sample.
    #
    ##########
    #if (extended == False):
    #    compFactor = diskCorrectFOV(rBins)
    #else:
    #    compFactor = 1.0

    compFactor = diskCorrectFOV(rBins)
    #compFactor = 1.0

    ##########
    #
    # Compute from P(N, r) the radial distribution with errors.
    # Do this by choosing peak as value and 68% contour as error bar.
    #
    ##########
    rDist = np.zeros(len(rBins), dtype=float)
    rDistErrHi = np.zeros(len(rBins), dtype=float)
    rDistErrLo = np.zeros(len(rBins), dtype=float)
    for rr in range(len(rBins)):
        # Get P(N given r). The total probability should add to 1.0 (not PDF).
        # Recall that N maps directly to the indices of the P(N) array.
        probAtR = prob[:,rr]

        # Determine 68% contour level
        sid0 = probAtR.argsort()
        sid = sid0[::-1] # reverse
        probAtRsorted = probAtR[sid]
        cdf = np.cumsum(probAtRsorted)
        idx = (np.where(cdf >= 0.6827))[0]

        # Peak
        rDist[rr] = sid[0]

        # Contours
        if (len(idx) > 0):
            level68 = probAtRsorted[idx[0]]

            # Now pull out all bins with probabilities higher
            # than the 68% level.
            idx = (np.where(probAtR >= level68))[0]

            # Now save the peak, upper, and lower errorbars
            rDistErrLo[rr] = np.abs(idx[0] - rDist[rr])
            rDistErrHi[rr] = np.abs(idx[-1] - rDist[rr])

        # We shouldn't have errors smaller than our bin size (1)
        if (rDistErrLo[rr] < 1):
            rDistErrLo[rr] = 0.0
        if (rDistErrHi[rr] < 1):
            rDistErrHi[rr] = 1.0

        print 'At r = %4.2f  N = %4.1f  -%4.1f  +%4.1f' % \
              (rBins[rr], rDist[rr], rDistErrLo[rr], rDistErrHi[rr])

    ##########
    #
    # Compute azimuthally symmetric density at each radius.
    #
    ##########
    areaInAnnulus = math.pi * (rHi**2 - rLo**2)
    azsymDensity = rDist * compFactor / areaInAnnulus
    azsymDensityLo = (rDist - rDistErrLo) * compFactor / areaInAnnulus
    azsymDensityLo =np. abs(azsymDensity - azsymDensityLo)
    azsymDensityHi = (rDist + rDistErrHi) * compFactor / areaInAnnulus
    azsymDensityHi = np.abs(azsymDensity - azsymDensityHi)

    for rr in range(len(rBins)):
        print 'At r = %4.2f    %5.3f  %5.3f  %5.3f    %4.1f  %4.1f  %4.1f' % \
              (rBins[rr], rDist[rr], rDistErrLo[rr], rDistErrHi[rr],
               azsymDensity[rr], azsymDensityLo[rr], azsymDensityHi[rr])

    fileDist1d = open(root+alnDir+'tables/diskRadialDist1D'+suffix+'.dat', 'w')
    pickle.dump((rBins, rLo, rHi), fileDist1d)
    pickle.dump(compFactor, fileDist1d)
    pickle.dump((rDist, rDistErrLo, rDistErrHi), fileDist1d)
    pickle.dump((azsymDensity, azsymDensityLo, azsymDensityHi), fileDist1d)
    fileDist1d.close()

    ### Plotting ###
    #
    # Radial Distribution (1D)
    #
    py.clf()
    p2 = py.errorbar(rBins, rDist*compFactor,
                     yerr=[rDistErrLo*compFactor, rDistErrHi*compFactor],
                     fmt='k-')
    #if (extended == False):
    #    # Show both completion corrected and directly observed
    #    # radial profiles.
    #    p1 = plot(rBins, rDist, 'k--')
    #    legend((p1, p2), ('Observed', 'FOV Corrected'), 'upper left')

    #py.axis([0, 0.4, 0, 15])
    py.xlabel('Radius in Disk (pc)')
    py.ylabel('Number of Stars')
    py.title('Total Disk Stars %d' % (len(sBins)-1))
    #py.xlim(0, 0.4)
    py.savefig(plotdir+'diskRadialDist'+suffix+'.png')
    py.close()


    ### Plotting ###
    #
    # Azimuthally Symmetric Density
    #
    py.clf()
    p1 = py.plot(rBins, azsymDensity, 'k.')
    p1 = py.errorbar(rBins, azsymDensity, yerr=[azsymDensityLo, azsymDensityHi])
    py.xlabel('Radius in Disk (pc)')
    py.ylabel('Surface Density (stars/pc^2)')
    #py.axis([0, 0.5, 0, 600])
    py.savefig(plotdir+'diskDensityAzSym'+suffix+'.png')
    py.close()


    ##########
    #
    # Compare observed density map to azimuthally symmetric case.
    #
    ##########
    # Load up the observed 2D surface density
    dens2dFile = open(root+alnDir+'tables/calcDiskDensityMC_results'+suffix+'.dat', 'r')
    (coordP, coordQ, coordStep) = pickle.load(dens2dFile)
    (obsDensity, obsDensityLo, obsDensityHi) = pickle.load(dens2dFile)
    dens2dFile.close()

    diskAzsym = np.zeros((len(coordQ), len(coordP)), dtype=float)
    diskAzsymLo = np.zeros((len(coordQ), len(coordP)), dtype=float)
    diskAzsymHi = np.zeros((len(coordQ), len(coordP)), dtype=float)
    diskExcess = np.zeros((len(coordQ), len(coordP)), dtype=float)
    
    for qq in range(len(coordQ)):
        for pp in range(len(coordP)):
            radius = np.hypot(coordQ[qq], coordP[pp])

            # Find closest radial bin
            #rdx = abs(rBins - radius).argmin()
            rdx = (np.where((radius >= rLo) & (radius < rHi)))[0]
            if (len(rdx) == 0):
                continue
            rdx = rdx[0]

            diskAzsym[qq, pp] = azsymDensity[rdx]
            diskAzsymLo[qq, pp] = azsymDensityLo[rdx]
            diskAzsymHi[qq, pp] = azsymDensityHi[rdx]

            error = np.sqrt(diskAzsymHi[qq, pp]**2 + obsDensityLo[qq, pp]**2)
            #error = diskAzsymHi[qq, pp]
            #error = obsDensityLo[qq, pp]

            if (error > 0):
                diskExcess[qq, pp] = (obsDensity[qq, pp] - diskAzsym[qq, pp])
                diskExcess[qq, pp] /= error

            # Set to nothing for inner FOV limits
            (xx, yy, zz) = diskDeproject(coordP[pp], coordQ[qq], 0.0)
            if (np.hypot(xx, yy) < (0.8 * 0.04)):
                diskExcess[qq, pp] = 0.0

            print '%5.2f  %5.2f   (%7.1f - %7.1f) / %7.1f =  %5.2f' % \
                  (coordQ[qq], coordP[pp], obsDensity[qq,pp], 
                   azsymDensity[rdx], error, diskExcess[qq,pp])

    ### Plot ###
    #
    # 2D spatial disk of the density excess
    #
    py.clf()

    (cornersP, cornersQ, cornersN) = getDiskCornersPQN()
    (innerP, innerQ, innerN) = getDiskInnerEdgePQN()

    #diskExcess = ma.masked_where(obsDensity < 25, diskExcess)
    foo = coordStep / 2.0
    py.imshow(diskExcess, origin='lower', cmap=py.cm.gray_r,
              extent=[coordP[0]-foo, coordP[-1]+foo, coordQ[0]-foo, coordQ[-1]+foo])
    py.plot([0], [0], 'kx')
    py.plot(cornersP, cornersQ, 'k--')
    py.plot(innerP, innerQ, 'k-.')
    #py.axis([0.4, -0.4, -0.4, 0.4])

    # Colorbar Wonderful... gotta work to make it pretty
    cbar = py.colorbar(orientation='horizontal')

    # Swap the colorbar's axis to the top
    cbarX = cbar.ax.get_xaxis()
    cbarX.set_ticks_position('top')
    cbarX.set_label_position('top')

    # Swap positions for colorbar and plot itself
    #mainLeft, mainLower, mainWidth, mainHeight = gca().get_position()
    #cbarLeft, cbarLower, cbarWidth, cbarHeight = cbar.ax.get_position()

    py.gca().set_position([0.15, 0.1, 0.8, 0.8])
    cbar.ax.set_position([0.15, 0.8, 0.8, 0.1])

    cbarX.get_label().set_text('Surface Density Excess (sigma)')

    py.xlabel('Disk Plane p-Direction (pc)')
    py.ylabel('Disk Plane q-Direction (pc)')
    py.savefig(plotdir+'diskDensityExcess'+suffix+'.png')
    py.close()

    # Find the peak in the surface density
    idx = np.where(diskExcess == diskExcess.max())
    peakP = coordP[idx[1][0]]
    peakQ = coordQ[idx[0][0]]

    print 'Peak Density Excess:'
    print '    at P = %4.2f, Q = %4.2f' % (peakP, peakQ)
    print '    %4.1f sigma above azimuthally symmetric density' % \
          (diskExcess.max())
    
    ### Plot ###
    #
    # 1D distribution of surface densities
    #
    py.clf()
    py.hist(diskExcess)

def plotRadialDist(mosaic=True, suffix='_mosaic'):
    orbDir = 'aorb_thesis/'
    #orbDir = 'aorb_acc_mrPDF_MC_newMosaic/'

    ##########
    # Load up radial distributions (already binned)
    ##########
    fileDist1d = open(root+alnDir+'tables/diskRadialDist1D'+suffix+'.dat', 'r')

    # Radial Bins
    (rBins, rLo, rHi) = pickle.load(fileDist1d)

    # Completeness factor (multiply distObs * compFactor = dist)
    compFactor = pickle.load(fileDist1d)

    # Observed distribution (before corrected for completeness)
    (distObs, distLoObs, distHiObs) =  pickle.load(fileDist1d)

    # Completeness corrected radial distribution
    (rdist, distLo, distHi) = pickle.load(fileDist1d)

    fileDist1d.close()

    ##########
    #
    # Plot projected 1D radial distribution in disk
    #
    ##########
    #
    # Radial Distribution (1D)
    #
    py.clf()
    p2 = py.errorbar(rBins, distObs*compFactor,
                     yerr=[distLoObs*compFactor, distHiObs*compFactor],
                     fmt='k-')
    #if (extended == False):
    #    # Show both completion corrected and directly observed
    #    # radial profiles.
    #    p1 = plot(rBins, distObs, 'k--')
    #    legend((p1, p2), ('Observed', 'FOV Corrected'), 'upper left')

    #py.axis([0, 1.0, 0, 15])
    py.xlabel('Radius in Disk (pc)')
    py.ylabel('Number of Stars')
    py.savefig(plotdir+'diskRadialDist2'+suffix+'.png')
    py.close()


    ### Plotting ###
    #
    # Azimuthally Symmetric Density
    #
    # Hack the stuff to be plotted on a log scale
    idx = (np.where(rdist == 0))[0]
    rdist[idx] = 0.001

    idx = (np.where(rBins == 0))[0]
    rBins[idx] = 0.001

    idx = (np.where(rdist <= distLo))[0]
    if (len(idx) > 0):
        distLo[idx] = distLo[idx] - 0.001

    ##########
    #
    # Find functional form of radial distribution
    #
    ##########
    # Fit a line in log-log to get the index of the radial distribution
    # BUT only fit within 0.04 - 0.6 pc
    def lineResiduals(params, x, y):
        slope = params[0]
        inter = params[1]

        model = inter + (slope*x)
        return (y - model)

    #if (extended == True):
    #    rdx = (where((rBins > 0.07) & (rBins <= 0.5)))[0]
    #else:
    rdx = (np.where((rBins > 0.07) & (rBins <= 0.6)))[0]
        
    logRadiusAll = np.log10(rBins[rdx])
    logDensityAll = np.log10(rdist[rdx])
    out = optimize.leastsq(lineResiduals, [0,0],
                           args=(logRadiusAll, logDensityAll), full_output=1)
    p = out[0]
    pcov = out[1]

    slopeMC = np.zeros(1000, dtype=float)
    for ii in range(1000):
        randDen = np.random.randn(len(rdx))
        pos = (np.where(randDen > 0))[0]
        neg = (np.where(randDen <= 0))[0]

        newDensity = rdist[rdx]
        newDensity[pos] += randDen[pos] * (distHi[rdx])[pos]
        newDensity[neg] += randDen[neg] * (distLo[rdx])[neg]

        for nn in range(len(newDensity)):
            if (newDensity[nn] < 0):
                newDensity[nn] = 0.00001

        newDensity = np.log10(newDensity)

        tmpout = optimize.leastsq(lineResiduals, [0, 0],
                                  args=(logRadiusAll, newDensity))

        slopeMC[ii] = tmpout[0][0]

    print np.sqrt(1.0 / pcov[0][0])
    #p = polyfit(logRadiusAll, logDensityAll, 1)
    print 'Best-fit power law of slope = %5.2f +/- %5.2f ' % \
          (p[0], slopeMC.std(ddof=1))
    print 'Amplitude = %5.2f stars/pc^2' % (10.0**p[1])

    usetexTrue()
    py.clf()
    py.figure(figsize=(6,6))
    py.subplots_adjust(left=0.15, right=0.95, top=0.95)
    p1 = py.loglog(rBins, rdist, 'k.')
    p1 = py.errorbar(rBins, rdist, xerr=[rBins-rLo, rHi-rBins],
                     yerr=[distLo, distHi], fmt='k.')
    py.xlabel('Radius in Disk (pc)')
    py.ylabel(r'Surface Density (stars/pc$^2$)')

    plawX = np.array([0.001, 1])
    plawY = 10**(p[1] + np.log10(plawX)*p[0])
    py.plot(plawX, plawY, 'k--')
    py.text(0.08, 600, r'$R^{%5.2f \pm %4.2f}$' % (p[0], slopeMC.std()),
            horizontalalignment='left')

    py.axis([0.03, 1.0, 0.1, 1100])
    py.savefig(plotdir+'diskDensityAzSym2'+suffix+'.png')
    py.close()
    usetexFalse()


def diskProject(x, y, z, xerr=None, yerr=None, zerr=None,
                idisk=idisk, odisk=odisk, ierr=ierr, oerr=oerr):
    """
    Switch everything into disk coordinate system. Coordinates
    are p, q, n where
      n is tangential to the disk plane.
      p goes along the ascending node in the disk plane.
      q is perpindicular to p in the disk plane.
    Get the directional vectors
    """
    irad = np.radians(idisk)
    orad = np.radians(odisk)
    ierad = np.radians(ierr)
    oerad = np.radians(oerr)
    
    nhat = np.array([ np.sin(irad) * np.cos(orad),
                   -np.sin(irad) * np.sin(orad),
                   -np.cos(irad)], float)    
    phat = util.cross_product(nhat, np.array([0.0, 0.0, -1.0]))
    qhat = util.cross_product(nhat, phat)

    nherr = np.array([
        np.sqrt((ierad*np.cos(irad)*np.cos(orad))**2 + (oerad*np.sin(irad)*np.sin(orad))**2),
        np.sqrt((ierad*np.cos(irad)*np.sin(orad))**2 + (oerad*np.sin(irad)*np.cos(orad))**2),
        np.sqrt((ierad*np.sin(irad))**2)], float)
    pherr = np.array([nherr[1], nherr[0], 0.0])
    qherr = np.array([
        np.sqrt((ierad*np.cos(orad))**2 + (oerad*np.sin(irad)*np.cos(irad)*np.sin(orad))**2),
        np.sqrt((ierad*np.sin(orad))**2 + (oerad*np.sin(irad)*np.cos(irad)*np.cos(orad))**2),
        np.sqrt((2.0*ierad*np.sin(irad)*np.cos(irad))**2)], float)
        
    #print 'n = [%5.2f, %5.2f, %5.2f] +/-' % (nhat[0], nhat[1], nhat[2])
    #print '    [%5.2f, %5.2f, %5.2f]' % (nherr[0], nherr[1], nherr[2])
    #print 'p = [%5.2f, %5.2f, %5.2f] +/-' % (phat[0], phat[1], phat[2])
    #print '    [%5.2f, %5.2f, %5.2f]' % (pherr[0], pherr[1], pherr[2])
    #print 'q = [%5.2f, %5.2f, %5.2f] +/-' % (qhat[0], qhat[1], qhat[2])
    #print '    [%5.2f, %5.2f, %5.2f]' % (qherr[0], qherr[1], qherr[2])

    n = (x*nhat[0]) + (y*nhat[1]) + (z*nhat[2])
    p = (x*phat[0]) + (y*phat[1]) + (z*phat[2])
    q = (x*qhat[0]) + (y*qhat[1]) + (z*qhat[2])

    if ((xerr != None) and (yerr != None) and (zerr != None)):
        nerr = (xerr*nhat[0])**2 + (yerr*nhat[1])**2 + (zerr*nhat[2])**2
        nerr += (x*nherr[0])**2 + (y*nherr[1])**2 + (z*nherr[2])**2
        nerr = np.sqrt(nerr)

        perr = (xerr*phat[0])**2 + (yerr*phat[1])**2 + (zerr*phat[2])**2
        perr += (x*pherr[0])**2 + (y*pherr[1])**2 + (z*pherr[2])**2
        perr = np.sqrt(perr)
        
        qerr = (xerr*qhat[0])**2 + (yerr*qhat[1])**2 + (zerr*qhat[2])**2
        qerr += (x*qherr[0])**2 + (y*qherr[1])**2 + (z*qherr[2])**2
        qerr = np.sqrt(qerr)

        return (p, q, n, perr, qerr, nerr)

    return (p, q, n)

def diskThickness_weighted(idisk=idisk, odisk=odisk, mosaic=True, suffix='_mosaic'): 
    """
    Calculate the disk thickness and opening angle as estimated by the velocity
    dispersion, where the dispersion is weighted by disk membership probability.

    Depends: diskMembers()
    Outputs: plots/diskThickness.png
    """
    cc = objects.Constants()

    pdfdir = 'aorb_thesis/'
    #pdfdir = 'aorb_acc_mrPDF_MC_newMosaic/'

    # Load disk star names and probability of disk membership
    (names, diskP) = readDiskProb(suffix=suffix,diskOnly=False)

    # Load kinematic data for disk stars
    yng = loadYoungByName(names, mosaic=mosaic)

    yngNames = yng.getArray('name')
    x = yng.getArray('x')
    y = yng.getArray('y')
    z = np.array([aorb.plane2z(x[i], y[i]) for i in range(len(x))])
    vx = yng.getArray('vx')
    vy = yng.getArray('vy')
    vz = yng.getArray('vz')
    vtot = np.sqrt(vx**2 + vy**2 + vz**2)
    vxerr = yng.getArray('vxerr')
    vyerr = yng.getArray('vyerr')
    vzerr = yng.getArray('vzerr')
    jz = (x * vy - y * vx) / (np.hypot(x, y) * np.hypot(vx, vy))

    # Switch everything into disk coordinate system. Coordinates
    # are p, q, n where
    #   n is tangential to the disk plane.
    #   p goes along the ascending node in the disk plane.
    #   q is perpindicular to p in the disk plane.
    # Get the directional vectors
    (p, q, n) = diskProject(x, y, z, idisk=idisk, odisk=odisk)
    (vp, vq, vn, vperr, vqerr, vnerr) = \
         diskProject(vx, vy, vz,
                     xerr=vxerr, yerr=vyerr, zerr=vzerr,
                     idisk=idisk, odisk=odisk)

    diskR = np.sqrt(p**2 + q**2)

    # should we keep the probs at 0? -- doesn't make a difference here
    #idx = (np.where(diskP == 0))[0]
    #diskP[idx] = 1.0e-5
    idx = (np.where(diskP == 1.0e-5))[0]
    diskP[idx] = 0

    # Define the weight for each star
    wt = diskP

    print
    print 'Disk at i = %5.1f  o = %5.1f' % (idisk, odisk)
    diskR_wtmean = (diskR * wt).sum() / wt.sum()
    print 'Weighted Average Radius in Disk = %5.3f arcsec' % diskR_wtmean
    print

    print 'Half Angle -- based on velocity dispersion out of the disk plane'
    print
    print '%6s   %4s    %4s  (%3s, %3s, %2s)   %13s  %11s' % \
          ('radius', 'vdis','err', 'mea', 'bia', 'N',
           'h/r=vndisp/v', '*Half Angle')

    nstars = len(diskP)
    nz = np.where(diskP > 0.0)[0] # must use number of non-zero weights
    nstarsNZ = len(nz)
    
    # Calc the unbiased velocity dispersion for the full sample,
    # weighted by disk membership probability
    vndisp_meas = np.sqrt( nstarsNZ * ((vn**2 * wt).sum() / wt.sum()) / (nstarsNZ - 1) )
    vndisp_bias = np.sqrt( nstarsNZ * ((vnerr**2 * wt).sum() / wt.sum()) / (nstarsNZ - 1) )
    vndisp = np.sqrt(vndisp_meas**2 - vndisp_bias**2)
    #vndisperr = vndisp / np.sqrt(2.*len(vn) - 1.)
    vndisperr = np.sqrt(wt.sum())

    vdisk = np.sqrt(vp**2 + vq**2)

    vtot_wtmean = (vtot * wt).sum() / wt.sum()
    vtoterr_wtmean = np.sqrt(1. / wt.sum())

    scaleHeight = vndisp / vtot_wtmean
    scaleHeightErr = scaleHeight / np.sqrt(nstarsNZ)

    halfAngle = math.degrees((np.sqrt(2.0) * scaleHeight))
    halfAngleErr = np.sqrt(2.0) * math.degrees(scaleHeightErr)
    #print vtot_wtmean

    fmt = ' %5s  %3d +/- %3d  (%3d, %3d, %2d)  %4.2f +/- %4.2f %4.1f +/- %3.1f'
    print fmt % ('ALL', vndisp, vndisperr, vndisp_meas, vndisp_bias, nstars,
                 scaleHeight, scaleHeightErr, halfAngle, halfAngleErr)

    # Calculate the unbiased vel dispersion as a function of radius in disk plane
    radCuts = np.array([1.5, 3.0, 4.5, 6.0, 7.5, 9.0], dtype=float)
    vndispR = np.zeros(len(radCuts), dtype=float)
    vndispRerr = np.zeros(len(radCuts), dtype=float)
    for rr in range(len(radCuts)):
        idx = np.where(diskR <= radCuts[rr])[0]
        nstarsR = len(idx)
        diskPR = diskP[idx]
        wtR = diskPR
        
        vndisp_measR = np.sqrt( nstarsR * ((vn[idx]**2 * wtR).sum() / wtR.sum()) / (nstarsR - 1) )
        vndisp_biasR = np.sqrt( nstarsR * ((vnerr[idx]**2 * wtR).sum() / wtR.sum()) / (nstarsR - 1) )
        vndispR[rr] = np.sqrt(vndisp_measR**2 - vndisp_biasR**2)

        vndispRerr[rr] = vndispR[rr] / np.sqrt(2.*len(idx) - 1.)
    
        vtotR_wtmean = (vtot[idx] * wtR).sum() / wtR.sum()
    
        scaleHeightR = vndispR[rr] / vtotR_wtmean
        scaleHeightErrR = scaleHeightR / np.sqrt(nstarsR)
    
        halfAngleR = math.degrees((np.sqrt(2.0) * scaleHeightR))
        halfAngleErrR = np.sqrt(2.0) * math.degrees(scaleHeightErrR)
        #print vtotR_wtmean

        print fmt % (radCuts[rr], vndispR[rr], vndispRerr[rr], vndisp_measR, vndisp_biasR, nstarsR,
                     scaleHeightR, scaleHeightErrR, halfAngleR, halfAngleErrR)


    # Now just split it up into two radial bins, inner and outer
    # Make the split at the radius where the sum of the weights are equal
    rsrt = diskR.argsort()
    namesRsrt = [names[aa] for aa in rsrt]
    cdf = np.cumsum(diskP[rsrt])
    cidx = (np.where(cdf >= diskP.sum()/2))[0][0]
    radCut = diskR[cidx]
    bin = ['bin1', 'bin2']
    print
    fmt = ' %5s  %3d +/- %3d (%3d, %3d, %2d)  %4.2f +/- %4.2f %4.1f +/- %3.1f %6.3f'
    print 'Split into 2 radial bins, where sum of weights are equal in each bin:'
    print 'radius = %5.3f arcsec' % radCut
    print '%6s   %4s    %4s  (%3s, %3s, %2s)   %13s  %11s  %14s' % \
          ('radius', 'vdis', 'err', 'mea', 'bia', 'N',
           'h/r=vndisp/v', '*Half Angle','Sum of weights')
    
    vndispR2 = np.zeros(2, dtype=float)
    vndispR2err = np.zeros(2, dtype=float)
    for rr in range(2):
        if rr == 0:
            idx = np.where(diskR < radCut)[0]
        else:
            idx = np.where(diskR >= radCut)[0]
        nstarsR = len(idx)
        diskPR = diskP[idx]
        wtR = diskPR
            
        vndisp_measR = np.sqrt( nstarsR * ((vn[idx]**2 * wtR).sum() / wtR.sum()) / (nstarsR - 1) )
        vndisp_biasR = np.sqrt( nstarsR * ((vnerr[idx]**2 * wtR).sum() / wtR.sum()) / (nstarsR - 1) )
        vndispR2[rr] = np.sqrt(vndisp_measR**2 - vndisp_biasR**2)
        vndispR2err[rr] = vndispR2[rr] / np.sqrt(2.*len(idx) - 1.)
    
        vtotR_wtmean = (vtot[idx] * wtR).sum() / wtR.sum()
    
        scaleHeightR = vndispR2[rr] / vtotR_wtmean
        scaleHeightErrR = scaleHeightR / np.sqrt(nstarsR)
    
        halfAngleR = math.degrees((np.sqrt(2.0) * scaleHeightR))
        halfAngleErrR = np.sqrt(2.0) * math.degrees(scaleHeightErrR)
    
        #print vtotR_wtmean
        print fmt % (bin[rr], vndispR2[rr], vndispR2err[rr], vndisp_measR, vndisp_biasR, nstarsR,
                     scaleHeightR, scaleHeightErrR, halfAngleR, halfAngleErrR, wtR.sum())

    ##########
    #
    # Plotting
    #
    ##########
    #
    # Plot v_perp vs. disk probability
    #
    usetexTrue()
    py.figure(1)
    py.figure(figsize=(6,6))
    py.clf()
    py.subplots_adjust(left=0.15, right=0.95)
    py.errorbar(diskP, vn, yerr=vnerr, fmt='k.')

    py.plot([1e-5, 1], [0, 0], 'k--')
    py.plot([1e-5, 1], [vndisp,vndisp], 'b-')
    py.plot([1e-5, 1], [-vndisp,-vndisp], 'b-')
    py.semilogx()
    py.axis([1e-3,1,-400,400])

    py.xlabel('Probability of Disk Membership')
    py.ylabel('Velocity Out-of-the-Plane (km/s)')
    py.title('Disk Thickness')
    py.savefig(plotdir + 'diskThickness_weighted'+suffix+'.png')
    py.close(1)

    # Plot v_perp vs. radius in disk
    py.figure(2)
    py.figure(figsize=(6,6))
    py.clf()
    py.subplots_adjust(left=0.15, right=0.95)
    py.errorbar(diskR, vn, yerr=vnerr, fmt='k.')

    py.plot([0, 14], [0, 0], 'k--')
    py.plot(radCuts, vndispR, 'b-')
    py.plot(radCuts, -vndispR, 'b-')
    py.axis([0,14,-400,400])

    py.xlabel('Radius in Disk Plane (arcsec)')
    py.ylabel('Velocity Out-of-the-Plane (km/s)')
    py.title('Disk Thickness')
    py.savefig(plotdir + 'diskThickness_diskPlaneRad'+suffix+'.png')
    py.close(2)

    usetexFalse()


def diskThickness(idisk=idisk, odisk=odisk, mosaic=True, suffix='_mosaic'): 
    """
    Plot the vertical (out of the plane) velocity vs. probability of
    disk membership. Calculate the disk thickness and opening angle
    as estimated by the velocity dispersion.

    Depends: diskMembers()
    Outputs: plots/diskThickness.png
    """
    cc = objects.Constants()

    pdfdir = 'aorb_thesis/'
    #pdfdir = 'aorb_acc_mrPDF_MC_newMosaic/'

    # Load disk star names and probability of disk membership
    (names, diskP) = readDiskProb(suffix=suffix,diskOnly=False)

    # Load kinematic data for disk stars
    yng = loadYoungByName(names, mosaic=mosaic)

    yngNames = yng.getArray('name')
    x = yng.getArray('x')
    y = yng.getArray('y')
    z = np.array([aorb.plane2z(x[i], y[i]) for i in range(len(x))])
    vx = yng.getArray('vx')
    vy = yng.getArray('vy')
    vz = yng.getArray('vz')
    vtot = np.sqrt(vx**2 + vy**2 + vz**2)
    vxerr = yng.getArray('vxerr')
    vyerr = yng.getArray('vyerr')
    vzerr = yng.getArray('vzerr')
    jz = (x * vy - y * vx) / (np.hypot(x, y) * np.hypot(vx, vy))

    # Switch everything into disk coordinate system. Coordinates
    # are p, q, n where
    #   n is tangential to the disk plane.
    #   p goes along the ascending node in the disk plane.
    #   q is perpindicular to p in the disk plane.
    # Get the directional vectors
    (p, q, n) = diskProject(x, y, z, idisk=idisk, odisk=odisk)
    (vp, vq, vn, vperr, vqerr, vnerr) = \
         diskProject(vx, vy, vz,
                     xerr=vxerr, yerr=vyerr, zerr=vzerr,
                     idisk=idisk, odisk=odisk)

    diskR = np.sqrt(p**2 + q**2)

    idx = (np.where(diskP == 0))[0]
    diskP[idx] = 1.0e-5

    print 'Disk at i = %5.1f  o = %5.1f' % (idisk, odisk)
    print 'Average Radius in Disk = %f' % diskR.mean()
    print ''

    # Velocity dispersion as a function of where we cut in probability
    probCuts = np.array([3.4e-1, 1e-1, 4.55e-2, 1e-2, 5.0e-3, 2.7e-3,
                         1e-4, 1e-5], dtype=float)
    vndisp = np.zeros(len(probCuts), dtype=float)

    print 'Half Angle 1 -- based on velocity dispersion out of the disk plane'
    print 'Half Angle 2 -- based on velocity vector angle relative to disk plane'
    print '%11s %4s (%3s, %3s, %2s)   %13s  %13s  %11s  %11s' % \
          ('Probability', 'vdis', 'mea', 'bia', 'N',
           'h/r=vndisp/v', '<Y^2>^0.5',
           '*Half Angle 1', 'Half Angle 2')

    angle = np.arcsin(vn / vtot) * 180.0 / math.pi
    tmp = np.sqrt(1.0 - (vn/vtot)**2)
    angleErr = (((1.0/vtot) - (vn**2/vtot**3)) * vnerr / tmp)**2
    angleErr += ((vn*vp/vtot**3) * vperr / tmp)**2
    angleErr += ((vn*vq/vtot**3) * vqerr / tmp)**2
    angleErr = np.sqrt(angleErr) * 180.0 / math.pi
        
    for ii in range(len(probCuts)):
        idx = (np.where(diskP > probCuts[ii]))[0]
        nstars = len(idx)
        
        # Calc the unbiased velocity dispersion
        vndisp_meas = np.sqrt( (vn[idx]**2).sum() / (len(idx) - 1) )
        vndisp_bias = np.sqrt( (vnerr[idx]**2).sum() / (len(idx) - 1) )
        vndisp[ii] = np.sqrt(vndisp_meas**2 - vndisp_bias**2)

        angMean = np.abs(angle[idx]).mean() # * 2.0
        angle_meas = np.sqrt( (angle[idx]**2).sum() / len(idx) )
        angle_bias = np.sqrt( (angleErr[idx]**2).sum() / len(idx) )
        angStdv = math.radians( np.sqrt(angle_meas**2 - angle_bias**2) )
        angStdvErr = angStdv / np.sqrt(len(idx))

        vdisk = np.sqrt(vp[idx]**2 + vq[idx]**2)

        scaleHeight = vndisp[ii] / vtot[idx].mean()
        scaleHeightErr = scaleHeight / np.sqrt(len(idx))

        halfAngle1 = math.degrees((np.sqrt(2.0) *angStdv))
        #halfAngleErr1 = (angStdvErr)
        #halfAngleErr1 /= np.sqrt(1.0 - (np.sqrt(2.0) * angStdvErr)**2)
        #halfAngleErr1 = math.degrees( halfAngleErr1 )
        halfAngleErr1 = np.sqrt(2.0) * math.degrees(angStdvErr)

        halfAngle2 = math.degrees((np.sqrt(2.0) * scaleHeight))
        #halfAngleErr2 = (scaleHeightErr)
        #halfAngleErr2 /= np.sqrt(1.0 - (np.sqrt(2.0) * scaleHeight)**2)
        #halfAngleErr2 = math.degrees( halfAngleErr2 )
        halfAngleErr2 = np.sqrt(2.0) * math.degrees(scaleHeightErr)

        print ' > %7.1e  %3d  (%3d, %3d, %2d)  ' % \
              (probCuts[ii], vndisp[ii], vndisp_meas, vndisp_bias, nstars),
        print '%4.2f +/- %4.2f  %4.2f +/- %4.2f  ' % \
               (scaleHeight, scaleHeightErr, angStdv, angStdvErr),
        print '%4.1f +/- %3.1f  %4.1f +/- %3.1f' % \
               (halfAngle2, halfAngleErr2, halfAngle1, halfAngleErr1)

    ##########
    #
    # Plotting
    #
    ##########
    #
    # Plot v_perp vs. disk probability
    #
    py.clf()
    py.figure(figsize=(6,6))
    py.subplots_adjust(left=0.15, right=0.95)
    py.semilogx(diskP, vn, 'k.')
    py.errorbar(diskP, vn, yerr=vnerr, fmt='k.')

    py.plot([1e-5, 1], [0, 0], 'k--')
    py.plot(probCuts, vndisp, 'b-')
    py.plot(probCuts, -vndisp, 'b-')

    py.xlabel('Probability of Disk Membership')
    py.ylabel('Velocity Out-of-the-Plane (km/s)')
    py.title('Disk Thickness')

    py.savefig(plotdir + 'diskThickness'+suffix+'.png')
    py.close()

    import matplotlib.font_manager
    prop = matplotlib.font_manager.FontProperties(size=12)

    def plotVelPerp(i_disk, o_disk, suffix2, ylab):
        (p, q, n) = diskProject(x, y, z, idisk=i_disk, odisk=o_disk)
        (vp, vq, vn, vperr, vqerr, vnerr) = \
             diskProject(vx, vy, vz,
                         xerr=vxerr, yerr=vyerr, zerr=vzerr,
                         idisk=idisk, odisk=odisk)

        # Identify disk members
        idx = (np.where(diskP > 2.7e-3))[0]

        # Identify non-disk members with clockwise motions
        cwid = (np.where((jz > 0) & (diskP <= 2.7e-3)))[0]
        ## Throw out IRS 33N
        #cwid = cwid[:-1]
        
        ylims = 310
        xlims = 4

        theta = np.arctan2(x, y) * 180.0 / math.pi
        thetaDisk = (np.arctan2(p, q) * 180.0 / math.pi)
        rDisk = np.sqrt(p**2 + q**2)

        print 'Adding Stars: '
        for ff in cwid:
            print '%15s  %4d' % (yngNames[ff], vn[ff])
        
        py.clf()
        py.figure(figsize=(6,6))
        py.subplots_adjust(left=0.15, right=0.95)
        py.errorbar(theta[cwid], vn[cwid], vnerr[cwid],
                 fmt='o', mfc='gray', mec='gray', ecolor='gray',label='CW Non-disk')
        py.errorbar(theta[idx], vn[idx], vnerr[idx],
                 fmt='o', mfc='k', mec='k', ecolor='k', label='CW Disk')
        py.xlabel('Position Angle from North (deg)')
        py.ylabel(ylab)
        py.legend(numpoints=1,fancybox=True,prop=prop)
        
        #for i in idx:
        #    text(theta[i], vn[i], yngNames[i])

        py.plot([-1000,1000], [0,0], 'k--')
        #py.plot([-1000,1000], [-40,-40], 'k--')
        #py.plot([-1000,1000], [40,40], 'k--')
        py.xlim(-200, 200)
        py.ylim(-ylims, ylims)
        
        py.savefig(plotdir+'diskThickness_vperp_vs_pa_' + suffix2 + '.png')
        py.close()


    ##########
    #
    # Plot v_perp vs. PA in disk for OUR ORBITS DISK
    #
    ##########
    plotVelPerp(idisk, odisk, 'ours',
                'Vel. Residuals for Our Disk (km/s)')

    ##########
    #
    # Plot v_perp vs. PA in disk for PAUMARDS CHI2 DISK
    #
    ##########
    idisk = 127.0
    odisk = 99.0
    plotVelPerp(idisk, odisk, 'paum',
                'Vel. Residuals for Old Disk (km/s)')

    ##########
    #
    # Plot v_perp vs. PA in disk for OUR CHI2 DISK
    #
    ##########
    #idisk = 113.5
    #odisk = 102.6
    #plotVelPerp(idisk, odisk, 'chi2',
    #            'Vel. Residuals for Chi^2 Disk (km/s)')


def velDispAnalysis(idisk=idisk, odisk=odisk, mosaic=True, suffix='_mosaic'): 
    """
    Plot the vertical (out of the plane) velocity vs. radius and
    various other parameters for those stars in the disk.

    Outputs:
    plots/vn_vs_prob    -- v-perp vs. probability of disk membership
    plots/vn_vs_radius  -- v-perp vs. radius in disk
    plots/vn_vs_vtot    -- v-perp vs. total velocity

    Depends: diskMembers()
    """
    cc = objects.Constants()

    # Loaded directory and suffix for extended or not.    
    pdfdir = 'aorb_thesis/'
    #pdfdir = 'aorb_acc_mrPDF_MC_newMosaic/'

    # Load disk star names and probability of disk membership
    (names, diskP) = readDiskProb(suffix=suffix,diskOnly=False)

    # Load kinematic data for disk stars
    yng = loadYoungByName(names, mosaic=mosaic)

    yngNames = yng.getArray('name')
    x = yng.getArray('x') * dist / cc.au_in_pc
    y = yng.getArray('y') * dist / cc.au_in_pc
    z = np.array([aorb.plane2z(x[i], y[i]) for i in range(len(x))])
    vx = yng.getArray('vx')
    vy = yng.getArray('vy')
    vz = yng.getArray('vz')
    vtot = np.sqrt(vx**2 + vy**2 + vz**2)
    vxerr = yng.getArray('vxerr')
    vyerr = yng.getArray('vyerr')
    vzerr = yng.getArray('vzerr')
    vtoterr = np.sqrt((vx*vxerr)**2 + (vy*vyerr)**2 + (vz*vzerr)**2) / vtot

    # Switch everything into disk coordinate system. Coordinates
    # are p, q, n where
    #   n is tangential to the disk plane.
    #   p goes along the ascending node in the disk plane.
    #   q is perpindicular to p in the disk plane.
    # Get the directional vectors
    (p, q, n) = diskProject(x, y, z, idisk=idisk, odisk=odisk)
    (vp, vq, vn, vperr, vqerr, vnerr) = \
         diskProject(vx, vy, vz,
                     xerr=vxerr, yerr=vyerr, zerr=vzerr,
                     idisk=idisk, odisk=odisk)

    rInDisk = np.sqrt(p**2 + q**2)
    r = np.sqrt(p**2 + q**2 + n**2)

    vcirc = np.sqrt(cc.G * mass * cc.msun / (r * cc.cm_in_pc))
    vcirc /= 1.0e5
    vInDisk = np.sqrt(vp**2 + vq**2)
    vInDiskErr = np.sqrt((vp*vperr)**2 + (vq*vqerr)**2) / vInDisk

    angle = np.arctan2(vn, np.hypot(vp, vq)) * 180.0 / math.pi
    tmp = np.sqrt(1.0 - (vn/vtot)**2)
    angleErr = (((1.0/vtot) - (vn**2/vtot**3)) * vnerr / tmp)**2
    angleErr += ((vn*vp/vtot**3) * vperr / tmp)**2
    angleErr += ((vn*vq/vtot**3) * vqerr / tmp)**2
    angleErr = np.sqrt(angleErr) * 180.0 / math.pi

    # Identify Disk Members
    idx = (np.where(diskP > 2.7e-3))[0]

    ##########
    #
    # Plot v-perp vs. disk probability
    #
    ##########
    py.clf()
    py.subplots_adjust(left=0.15, right=0.95)
    py.semilogx(diskP, vn, 'k.')
    py.errorbar(diskP, vn, yerr=vnerr, fmt='k.')

    py.plot([1e-6, 1], [0, 0], 'k--')
    #py.plot([2.7e-3, 2.7e-3], [-400, 400], 'k--')
    py.axis([3e-6, 1, -400, 400])

    py.xlabel('Probability of Disk Membership')
    py.ylabel('Velocity Out-of-the-Plane (km/s)')

    py.savefig(plotdir + 'vn_vs_prob'+suffix+'.png')
    py.close()


    ##########
    #
    # Plot vdisp_vert vs. vdisp_radial
    #
    ##########
    vndispV_meas = np.sqrt( (vn[idx]**2).sum() / (len(idx) - 1) )
    vndispV_bias = np.sqrt( (vnerr[idx]**2).sum() / (len(idx) - 1) )
    vndispV = np.sqrt(vndispV_meas**2 - vndispV_bias**2)

    vdiffRad = vInDisk-vcirc
    vndispR_meas = np.sqrt( (vdiffRad[idx]**2).sum() / (len(idx) - 1) )
    vndispR_bias = np.sqrt( (vInDiskErr[idx]**2).sum() / (len(idx) - 1) )
    vndispR = np.sqrt(vndispR_meas**2 - vndispR_bias**2)
    print 'Vertical Velocity Dispersion Mean: %3d' % (vn[idx].mean())
    print 'Radial Velocity Dispersion Mean:   %3d' % (vdiffRad[idx].mean())
    print 'Vertical Velocity Dispersion: %3d +/- %3d   %2d' % \
          (vndispV, vndispV/np.sqrt(len(idx)), len(idx))
    print 'Radial Velocity Dispersion:   %3d +/- %3d   %2d' % \
          (vndispR, vndispR/np.sqrt(len(idx)), len(idx))

    py.clf()
    py.subplots_adjust(left=0.15, right=0.95)
    py.errorbar(vdiffRad[idx], vn[idx],
             xerr=vInDiskErr[idx], yerr=vnerr[idx], fmt='k.')
    py.plot([0, 0], [-250, 250], 'k--')
    py.plot([-250, 250], [0, 0], 'k--')
    #py.axis([-250, 250, -250, 250])
    #for ii in idx:
    #    py.text(vInDisk[ii]-vcirc[ii], vn[ii], yngNames[ii])

    py.xlabel('Radial Velocity Dispersion (km/s)')
    py.ylabel('Vertical Velocity Dispersion (km/s)')
    
    py.savefig(plotdir + 'vn_vs_vradial'+suffix+'.png')
    py.close()

    py.clf()
    py.figure(figsize=(6,6))
    py.subplots_adjust(left=0.15, right=0.95)
    py.plot(vcirc[idx], vInDisk[idx],'k.')
    py.plot([0,900],[0,900],'k--')
    py.xlabel('Circular Velocity (km/s)')
    py.ylabel('Velocity in Disk Plane (km/s)')
    py.savefig(plotdir + 'vInDisk_vs_vcirc'+suffix+'.png')
    py.close()


    ##########
    #
    # Plot vdisp vs. radius
    #
    ##########
    py.clf()
    py.subplots_adjust(left=0.15, right=0.95)
    py.errorbar(rInDisk[idx], vn[idx], yerr=vnerr[idx], fmt='k.')

    py.plot([0, 0.5], [0, 0], 'k--')
    py.axis([0, 0.5, -300, 300])

    py.xlabel('Radius in Disk Plane (pc)')
    py.ylabel('Velocity Out-of-the-Plane (km/s)')
    
    py.savefig(plotdir + 'vn_vs_radius'+suffix+'.png')
    py.close()

    ##########
    #
    # Plot radially averaged vdisp vs. radius
    #
    ##########
    py.clf()
    py.subplots_adjust(left=0.15, right=0.95)

    #if (extended == True):
    #    rlims = array([0, 0.164, 0.4, 10.0])
    #else:

    rlims = np.array([0, 0.1, 0.18, 10.0]) # pc

    vndispR = np.zeros(len(rlims)-1, dtype=float)
    vndispRerr = np.zeros(len(rlims)-1, dtype=float)
    avgR = np.zeros(len(rlims)-1, dtype=float)

    for rr in range(len(rlims)-1):
        rdx = (np.where((diskP > 2.7e-3) &
                     (rInDisk > rlims[rr]) &
                     (rInDisk < rlims[rr+1])))[0]

        vndisp_meas = np.sqrt( (vn[rdx]**2).sum() / (len(rdx) - 1) )
        vndisp_bias = np.sqrt( (vnerr[rdx]**2).sum() / (len(rdx) - 1) )

        vndispR[rr] = np.sqrt(vndisp_meas**2 - vndisp_bias**2)
        vndispRerr[rr] = vndispR[rr] / np.sqrt(2.0 * len(rdx))

        avgR[rr] = rInDisk[rdx].mean()

        print 'r = %5.2f - %5.2f    %7d +/- %7d   %2d' % \
              (rlims[rr], rlims[rr+1], vndispR[rr], vndispRerr[rr], len(rdx))

    
    py.errorbar(avgR, vndispR, yerr=vndispRerr, fmt='k.')

    py.plot([0, 0.4], [0, 0], 'k--')
    #py.axis([0, 0.4, -150, 150])
    
    py.xlabel('Radius in Disk Plane (pc)')
    py.ylabel('Velocity Out-of-the-Plane (km/s)')
    
    py.savefig(plotdir + 'vdisp_vs_radius'+suffix+'.png')
    py.close()
    
    ##########
    #
    # Plot vdisp vs. vtot
    #
    ##########
    py.clf()
    py.subplots_adjust(left=0.15, right=0.95)
    py.errorbar(vtot[idx], vn[idx], yerr=vnerr[idx], fmt='k.')
    py.plot([0, 1000], [0, 0], 'k--')

    py.axis([0, 800, -300, 300])

    py.xlabel('Total Velocity (km/s)')
    py.ylabel('Velocity Out-of-the-Plane (km/s)')
    
    py.savefig(plotdir + 'vn_vs_vtot'+suffix+'.png')
    py.close()


    ##########
    #
    # Plot inclination vs. disk probability
    #
    ##########
    py.clf()
    py.subplots_adjust(left=0.15, right=0.95)
    py.semilogx(diskP, angle, 'w,')

    idx = (np.where(diskP < 2.7e-3))[0]
    offD = py.errorbar(diskP[idx], angle[idx], yerr=angleErr[idx],
                    fmt='b^', ecolor='b', mfc='b', mec='b')

    idx = (np.where(diskP >= 2.7e-3))[0]
    onD = py.errorbar(diskP[idx], angle[idx], yerr=angleErr[idx],
                   fmt='ro', ecolor='r', mfc='r', mec='r')

    py.plot([1e-6, 1], [0, 0], 'k--')
    py.axis([3e-6, 1.05, -90, 90])

    #lgd = py.legend((offD, onD),
    #             ('Non-Disk Members', 'Candidate Disk Members'),
    #             numpoints=1)
    #lgdLines = lgd.get_lines()
    #lgdLines[0].set_marker('^')
    #lgdLines[1].set_marker('o')
    #lgdLines[0].set_mfc('b')
    #lgdLines[1].set_mfc('r')
    #lgdLines[0].set_mec('b')
    #lgdLines[1].set_mec('r')

    py.xlabel('1 - L(not on disk)')
    py.ylabel('Velocity Vector Angle w.r.t. Disk (deg)')

    py.savefig(plotdir + 'i_vs_prob'+suffix+'.png')
    py.close()


    ##########
    #
    # Plot inclination vs. radius
    #
    ##########
    py.clf()
    py.subplots_adjust(left=0.15, right=0.95)
    py.errorbar(rInDisk[idx], angle[idx], yerr=angleErr[idx], fmt='k.')

    py.plot([0, 0.5], [0, 0], 'k--')
    py.axis([0, 0.4, -100, 100])

    py.xlabel('Radius in Disk Plane (pc)')
    #py.ylabel('Inclination w.r.t. Disk (deg)')
    py.ylabel('Velocity Vector Angle w.r.t. Disk (deg)')
    py.title('Disk Candidates')
    
    py.savefig(plotdir + 'i_vs_radius'+suffix+'.png')
    py.close()


    ##########
    #
    # Plot vdisp vs. vtot
    #
    ##########
    py.clf()
    py.subplots_adjust(left=0.15, right=0.95)
    py.errorbar(vtot[idx], angle[idx], yerr=angleErr[idx], fmt='k.')
    py.plot([0, 1000], [0, 0], 'k--')

    py.axis([0, 800, -100, 100])

    py.xlabel('Total Velocity (km/s)')
    #py.ylabel('Inclination w.r.t. Disk (deg)')
    py.ylabel('Velocity Vector Angle w.r.t. Disk (deg)')
    py.title('Disk Candidates')
    
    py.savefig(plotdir+'i_vs_vtot'+suffix+'.png')
    py.close()


def plotEccAnalytic(label=False, plotVsAng=False, mosaic=True,
                    diskOnly=False, suffix='_mosaic'):
    """
    Plots eccentricity measurements for stars w/ accelerations.
    Plot eccentricity 3 sigma lower limits vs. projected radius from
    Sgr A*. 

    Dependencies:
    diskMembers() - for disk membership probabilities
    accelLimit() - for accelerating sources that passed the F test

    Output:
    plots/sythesis_ecc_limits.png
    plots/eccVsSemimajorAxis_accelerators.png
    plots/hist_eccentricity_accelerators.png
    tables/sythesis_ecc_limits.dat
    """
    cc = objects.Constants()

    pdfdir = root + alnDir + 'aorb_thesis/'
    #pdfdir = root + alnDir + 'aorb_acc_mrPDF_MC_newMosaic/'

    # Load disk star names and probability of disk membership
    (names, diskP) = readDiskProb(diskOnly=False, suffix=suffix)

    # Load kinematic data for disk stars
    yng = loadYoungByName(names, mosaic=mosaic)
    yngNames = yng.getArray('name')
    yngMags = yng.getArray('mag')

    starCnt = len(yngNames)
    #starCnt = 5
    r = np.zeros(starCnt, float)
    elo = np.zeros(starCnt, float)
    ehi = np.zeros(starCnt, float)
    eloPos = np.zeros(starCnt, float)
    ehiPos = np.zeros(starCnt, float)
    eloNeg = np.zeros(starCnt, float)
    ehiNeg = np.zeros(starCnt, float)
    eloDisk = np.zeros(starCnt, float)
    ehiDisk = np.zeros(starCnt, float)
    eloCdf = np.zeros(starCnt, float)
    e_detect = np.zeros(starCnt, float)
    e_sigma = np.zeros(starCnt, float)
    direction = ['' for ss in yngNames]
    indisk = np.zeros(starCnt, float)

    ebins = np.arange(0, 1, 0.01)
    ehist = np.zeros((starCnt, len(ebins)), dtype=float)

    # Disk properties
    idiskP = 130.0
    odiskP = 96.0
    sinip = math.sin(math.radians(idiskP))
    cosip = math.cos(math.radians(idiskP))
    
    #idisk = [91, 127]
    #odisk = [84, 124]

    # Identify Disk Members
    idxDisk = (np.where(diskP > 2.7e-3))[0]
    diskCnt = len(idxDisk)

    # Keep track of the E(e^2)^1/2 values    
    eccAvg = np.zeros(starCnt, dtype=float)
    eccStd = np.zeros(starCnt, dtype=float)
    eccRms = np.zeros(starCnt, dtype=float)
    eccAvgDisk = np.zeros(starCnt, dtype=float)
    eccStdDisk = np.zeros(starCnt, dtype=float)
    eccRmsDisk = np.zeros(starCnt, dtype=float)

    # Keep track of semi-major axes averages
    smaAvg = np.zeros(starCnt, dtype=float)
    smaStd = np.zeros(starCnt, dtype=float)
    smaAvgDisk = np.zeros(starCnt, dtype=float)
    smaStdDisk = np.zeros(starCnt, dtype=float)

    _out = open(root + alnDir + 'tables/sythesis_ecc_limits' + suffix + '.dat', 'w')
    _out.write('#%14s  %9s   %9s   %9s   %10s %5s  %s\n' % \
               ('Name', 'ecc (all)', 'ecc (z>0)', 'ecc (z<0)',
                'ecc (disk)', 'eccLo', 'Direction'))

    # Read in file listing accelerating sources
    _acc = asciidata.open(root + alnDir + 'tables/accelerating_sources.dat')
    acc = _acc[0].tonumpy()
    accel = [aa.strip() for aa in acc]

    for ss in range(starCnt):
        name = yngNames[ss]
        star = yng.stars[ss]
        r[ss] = star.r2d # arcsec
        
        # File contains analytic orbit solutions with acceleration limits (MC)
        pdffile = '%s%s.mc.dat' % (pdfdir, name)
        pdf = pickle.load(open(pdffile))

        # Determine if this is a CW or CCW star
        if (pdf.i[0] > 90):
            direction[ss] = 'CW'
        else:
            direction[ss] = 'CCW'

        # Determine if this star has a significant Prob of being in the disk
        if (diskP[ss] > 2.7e-3):
            indisk[ss] = 1

        #py.clf()
        #(en, eb, pp) = py.hist(pdf.e, bins=ebins,color='k',histtype='step')
        #pdb.set_trace()
        #ehist[ss] = en

        # Determine angular offset to disk for each solution
        sini = np.sin(pdf.i * np.pi / 180.0)
        cosi = np.cos(pdf.i * np.pi / 180.0)
        cosodiff = np.cos( (pdf.o - odiskP) * np.pi / 180.0 )
        angOff = np.arccos( (sini * sinip * cosodiff) + (cosi * cosip) )
        angOff *= 180.0 / np.pi

        # Marginalize PDF over eccentricity
        elo[ss] = pdf.e.min()
        ehi[ss] = pdf.e.max()

        # Find range with only z > 0 solutions
        pid = (np.where(pdf.z > 0))[0]
        ePos = pdf.e[pid]
        eloPos[ss] = ePos.min()
        ehiPos[ss] = ePos.max()

        # Find range with only z < 0 solutions
        nid = (np.where(pdf.z <= 0))[0]
        eNeg = pdf.e[nid]
        eloNeg[ss] = eNeg.min()
        ehiNeg[ss] = eNeg.max()

        # Find ecc lower limit from CDF
        esort = np.sort(pdf.e)

        # Now, take the last 997 points which should correspond to a
        # three sigma lower limit.
        lolimit = int(0.997 * len(esort))
        #lolimit = int(0.683 * len(esort))
        eloCdf[ss] = esort[-lolimit:].min()

        # Keep track of average eccentricity
        eccAvg[ss] = pdf.e.mean()
        eccStd[ss] = pdf.e.std(ddof=1)
        eccRms[ss] = np.sqrt( (pdf.e**2).sum() / len(pdf.e) )

        # Keep track of average semi-major axis
        smaAvg[ss] = ((pdf.p**2 * pdf.m)**(1.0/3.0)).mean()
        smaStd[ss] = ((pdf.p**2 * pdf.m)**(1.0/3.0)).std(ddof=1)
        

        # Find range with disk solutions
        #did = (where((pdf.i > idisk[0]) & (pdf.i < idisk[1]) &
        #             (pdf.o > odisk[0]) & (pdf.o < odisk[1])))[0]
        did = (np.where(angOff < angleCut))[0]
        if ((len(did) > 0) and (indisk[ss] == 1)):
            eDisk = pdf.e[did]
            eloDisk[ss] = eDisk.min()
            ehiDisk[ss] = eDisk.max()

            # Average eccentricity in disk
            eccAvgDisk[ss] = pdf.e[did].mean()
            eccStdDisk[ss] = pdf.e[did].std(ddof=1)
            eccRmsDisk[ss] = np.sqrt( (pdf.e[did]**2).sum() / len(did) )
            #eccRmsDisk[ss] = np.sqrt( (pdf.e**2).sum() / len(pdf.e) )

            # Average semi-major axis in disk
            smaAvgDisk[ss] = ((pdf.p[did]**2 * pdf.m[did])**(1.0/3.0)).mean()
            smaStdDisk[ss] = ((pdf.p[did]**2 * pdf.m[did])**(1.0/3.0)).std(ddof=1)

            if (plotVsAng):
                py.clf()
                py.plot(angOff, pdf.e, 'k.', markersize=4)
                py.axis([0, 180, 0, 1])
                py.title(name)
                py.xlabel('Angular Offset from Disk (deg)')
                py.ylabel('Eccentricity')
                py.show()
                foo = raw_input('Continue? ')
                if (foo == 'q'):
                    return

        # Write to the output file
        fmt = '%15s  %4.2f %4.2f   %4.2f %4.2f   %4.2f %4.2f   %4.2f %4.2f   '
        fmt += '%4.2f   %s\n'
        _out.write(fmt % \
                   (name, elo[ss], ehi[ss],
                    eloPos[ss], ehiPos[ss], eloNeg[ss], ehiNeg[ss],
                    eloDisk[ss], ehiDisk[ss], eloCdf[ss], direction[ss]))
        _out.flush()

        #clf()
        #hist(pdf.e, 20)
        #show()

    _out.close()

    # Determine the RMS average eccentricity
    eRms = np.sqrt((eccRms**2).sum() / len(eccRms))
    eRmsDisk = np.sqrt((eccRmsDisk[idxDisk]**2).sum() / len(idxDisk))

    print ' ALL:  RMS = %4.2f  AVG = %4.2f  STD = %4.2f  AVGSTD = %4.2f' % \
          (eRms, eccAvg.mean(), eccAvg.std(ddof=1), eccStd.mean())
    print 'DISK:  RMS = %4.2f  AVG = %4.2f  STD = %4.2f  AVGSTD = %4.2f' % \
          (eRmsDisk, eccAvgDisk.mean(), eccAvgDisk.std(ddof=1), eccStdDisk.mean())

    direction = np.array(direction)
    ncw = len(np.where(direction == 'CW')[0])
    ncw1 = len(np.where((direction == 'CW') & (r <= 3.2))[0])
    ncw2 = len(np.where((direction == 'CW') & (r > 3.2) & (r <= 6.473))[0])
    ncw3 = len(np.where((direction == 'CW') & (r > 6.473))[0])
    nccw = len(np.where(direction == 'CCW')[0])
    nccw1 = len(np.where((direction == 'CCW') & (r <= 3.2))[0])
    nccw2 = len(np.where((direction == 'CCW') & (r > 3.2) & (r <= 6.473))[0])
    nccw3 = len(np.where((direction == 'CCW') & (r > 6.473))[0])
    print
    print 'Total number of stars with inclination > 90 (CW direction): %3i' % ncw
    print '  Numbers per radial bin: %i, %i, %i' % (ncw1,ncw2,ncw3)
    print 'Total number of stars with inclination < 90 (CCW direction): %3i' % nccw
    print '  Numbers per radial bin: %i, %i, %i' % (nccw1,nccw2,nccw3)
    
    ##########
    #
    # Plotting
    #
    ##########
    py.clf()
    py.figure(figsize=(10,5))
    py.subplots_adjust(left=0.1, right=0.95)
    py.subplot(1,2,1)
    #py.plot(r, eloCdf, 'k.')
    xtitle = 'Projected Distance from Sgr A* (arcsec)'
    ytitle = 'Eccentricity'
    py.xlabel(xtitle)
    py.ylabel(ytitle)
    py.title('All Solutions and 3 sigma Lower Limits', fontsize=16)
    py.subplot(1,2,2)
    xtitle = 'Projected Distance from Sgr A* (arcsec)'
    ytitle = 'Eccentricity'
    py.xlabel(xtitle)
    py.ylabel(ytitle)
    py.title('Disk Solutions', fontsize=16)

    detect = [] # keep track of which disk stars have accel measurements
    nondiskCW = 0 # keep track of non-disk stars w/ CW motions
    diskCW = 0 # keep track of disk stars w/ CW motions
    nondiskCCW = 0 # keep track of disk stars w/ CCW motions
    #for i in range(5):
    for i in range(len(yngNames)):
    	print '%15s  %4.2f  %4s  %1i' % (yngNames[i], elo[i], direction[i], indisk[i])
    
    	if (diskP[i] >= 2.7e-3):
            # Disk candidate
    	    color = 'r'
	    marker = 'ro'
            mec = 'k'
            mfc = 'r'
    	else:
            # Non-disk candidate
    	    color = 'b'
	    marker = 'bs'
            mec = 'k'
            mfc = 'b'

            if (direction[i] == 'CCW'):
                mfc = 'w'

	    if (diskOnly == True):
		continue

        # Keep track of how many non-disk stars have CW directions
        if mfc == 'b':
            nondiskCW += 1
        if mfc == 'r':
            diskCW += 1
        if mfc == 'w':
            nondiskCCW += 1

        # If we have accelerations, then we have eccentricities.
        # Without acceleration detections, we have lower limits on eccentricities.
        # FIRST plot all solutions
        py.subplot(1,2,1)
        if ((yngNames[i] in accel) and (indisk[i] == 1)):
	    py.errorbar([r[i]], [eccAvg[i]], yerr=eccStd[i], fmt='r.') # this shows average over
            							       # pos and neg z solutions
        elif ((yngNames[i] in accel) and (indisk[i] == 0)):
	    py.errorbar([r[i]], [eccAvg[i]], yerr=eccStd[i], fmt='b.') # this shows average over
            							       # pos and neg z solutions
        else:
            # Plot 3 sigma lower limits
	    py.plot([r[i]], [eloCdf[i]], marker, mec=mec, mfc=mfc)
    	    arrow = py.Arrow(r[i], eloCdf[i], 0, 0.05, width=0.1, 
    		           facecolor=color, edgecolor=color)
    	    fig = py.gca()
    	    fig.add_patch(arrow)

        # Add star names 
        if (label):
            py.text(r[i], eloCdf[i], yngNames[i])

        # If we have accelerations, then we have eccentricities.
        # Without acceleration detections, we have lower limits on eccentricities.
        # But plot the disk solutions for all the stars
        py.subplot(1,2,2)
        if (yngNames[i] in accel): # Identify the accelerating sources
            if (indisk[i] == 1): # On disk
                detect = np.concatenate([detect, [i]])
	        py.errorbar([r[i]], [eccAvgDisk[i]], yerr=eccStdDisk[i], fmt='rx')
        else: # Stars without acceleration detections
            if (indisk[i] == 1): # On disk
	        py.errorbar([r[i]], [eccAvgDisk[i]], yerr=eccStdDisk[i], fmt='r.')

        # Add star names 
        if (label):
            py.text(r[i], eloCdf[i], yngNames[i])
            
    print
    print 'Number of disk stars with CW orbits: %d' % diskCW
    print 'Number of non-disk stars with CW orbits: %d' % nondiskCW
    print 'Number of non-disk stars with CCW orbits: %d' % nondiskCCW

    py.subplot(1,2,1)
    py.plot([4.25], [0.94], 'ro')
    py.plot([4.25], [0.89], 'bs')
    py.text(4.4, 0.92, 'Disk Candidate', {'color' : 'r', 'fontsize' : 12})
    py.text(4.4, 0.87, 'Non-Disk', {'color' : 'b', 'fontsize' : 12})
    py.axis([0, 7, 0, 1.0])

    py.subplot(1,2,2)
    #py.axis([0, 2.0, 0, 600.0])
    #py.plot([4.25], [1.15], 'ro')
    #py.plot([4.25], [1.1], 'bs')
    #py.text(4.4, 1.13, 'Disk Candidate', {'color' : 'r', 'fontsize' : 12})
    #py.text(4.4, 1.08, 'Non-Disk', {'color' : 'b', 'fontsize' : 12})
    py.axis([0, 7, 0, 1.0])

    if (diskOnly == True):
	py.savefig(plotdir + 'sythesis_ecc_limits_disk' + suffix + '.png')
    else:
	py.savefig(plotdir + 'sythesis_ecc_limits' + suffix + '.png')
    py.close()

    # Plot the eccentricity (measurements or lower limits) vs. Magnitude
    py.clf()
    py.figure(figsize=(7,7))
    for i in range(len(yngNames)):
        if (yngNames[i] in accel):
            detect = np.concatenate([detect, [i]])
	    py.errorbar([yngMags[i]], [eccAvg[i]], yerr=eccStd[i], fmt='b.')
        else:
            # Plot 3 sigma lower limits
	    py.plot([yngMags[i]], [eloCdf[i]], 'k.')
    	    arrow = py.Arrow(yngMags[i], eloCdf[i], 0, 0.05, width=0.07, 
    		             facecolor='k', edgecolor='k')
    	    fig = py.gca()
    	    fig.add_patch(arrow)
    py.xlabel('K Magnitude')
    py.ylabel('Eccentricity')
    py.savefig(plotdir + 'sythesis_ecc_vs_Kmag.png')
    py.close()

    # Plot some stuff of just the disk members with accelerations
    ecc_disk = np.zeros(len(detect), dtype=float)
    eccE_disk = np.zeros(len(detect), dtype=float)
    sma_disk = np.zeros(len(detect), dtype=float)
    smaE_disk = np.zeros(len(detect), dtype=float)
    cnt = 0
    detect = [int(dd) for dd in detect]
    for dd in detect:
        ecc_disk[cnt] = eccAvgDisk[dd]
        eccE_disk[cnt] = eccStdDisk[dd]
        sma_disk[cnt] = smaAvgDisk[dd]
        smaE_disk[cnt] = smaStdDisk[dd]
        cnt += 1

    # Plot eccentricity vs. semi-major axis
    py.clf()
    py.figure(figsize=(7,5))
    py.subplots_adjust(left=0.15, right=0.95)
    xtitle = 'Semi-major Axis (pc)'
    ytitle = 'Eccentricity'
    py.xlabel(xtitle)
    py.ylabel(ytitle)
    py.title('e Vs. a for Accelerating Stars on CW Disk', fontsize=16)
    py.errorbar(sma_disk/cc.au_in_pc, ecc_disk,
                xerr=smaE_disk/cc.au_in_pc, yerr=eccE_disk, fmt='k.')
    py.axis([0,0.1,0,0.5])
    py.savefig(plotdir + 'eccVsSemimajorAxis_accelerators.png')
    py.close()

    # Plot a histogram of eccentricity measurements on the disk
    binsIn = np.arange(0, 1.05, 0.05)
    py.clf()
    py.figure(figsize=(6,6))
    py.hist(ecc_disk, bins=binsIn, color='k', histtype='step')
    py.xlabel('Average Eccentricity')
    py.ylabel('N')
    py.title('Eccentricity of Accelerating Stars on CW Disk', fontsize=16)
    py.savefig(plotdir + 'hist_eccentricity_accelerators.png')
    py.close()



            

    # Find the most probable eccentricity for each star
    #emaxes = np.zeros(len(yngNames), dtype=float)

    #for i in range(len(yngNames)):
    #    emaxes[i] = ebins[ehist[i].argmax()]

    #eidx = emaxes.argsort()
    #eidx = eidx[::-1]

#     clf()
#     axis([0, 1, 0, 33])

#     emaxAll = ehist.max()    
#     for i in range(len(eidx)):
#         eid = eidx[i]
#         xtmp = zeros(len(ebins)) + i
#         points = zip(ebins, xtmp)
#         segments = zip(points[:-1], points[1:])

#     	if (direction[i] == 'CW'):
#             colors = cm.Reds(ehist[eid] / ehist[i].max())
#             #colors = cm.Reds(ehist[eid] / emaxAll)
# 	    marker = 'ro'
#             marker2 = 'r>'
#         else:
#             colors = cm.Blues(ehist[eid] / ehist[i].max())
#             #colors = cm.Blues(ehist[eid] / emaxAll)
# 	    marker = 'bs'
#             marker2 = 'b>'

#         line = matplotlib.collections.LineCollection(segments,
#                                                      colors=colors)
#         line.set_linewidth(2)
#         gca().add_collection(line)
#         plot(array([emaxes[eid]]), array([i]), marker)
#         #plot(array([eloCdf[i]]), array([i]), marker2)


#     xlabel('Eccentricity')
#     axis([0, 1, 0, 33])
#     show()
#     draw()


def ecc_bias_simulation(nstars=100, ntrials=10**4, vfrac=None, egrid=False,
                        diskFraction=True, getEcc=False,
                        mockdir='sim_vkick_fracCircVel/',
                        mockfile='circularFlatMC_mockdata.pickle',
                        errorfile='pos_errorRange_vs_radius.dat',
                        sigma=4.0, plotsOnly=False):
    """
    Creates mock data assuming an orbit with a given velocity kick specified
    by the fraction of the local orbital velocity.  A range of initial
    eccentrities are used to create the mock data before the kick is applied,
    which results in a spread in the eccentricities and inclination and Omega.
    
    Runs orbital analysis on this mock data and determines
    the eccentricity bias resulting from noise in our measurements and the
    distribution that looks most similar to the observed.

    Inputs:
      nstars  = The number of stars to create mock data for
      ntrials = The number of trials to use in the orbital simulations.
      vfrac   = Fraction of the orbital velocity to be applied as the velocity kick
      		Use this to run a single simulation. --NOT FUNCTIONAL YET.
      egrid    = Set to True to run a grid of simulations of different
      	        inital eccentricities.
      diskFraction = Set to True to run a grid of simulations of differing disk fractions
      getEcc 	= Set to True to get eccentricity results for the diskFraction
      		  simulations (so diskFraction must be set to True also).
      plotsOnly = Set to True to just run ecc_bias_simulation_results(), which
      		  plots the results for the specified simulation. If set to True,
                  the simulation is not run, only plots are created.
    """

    import sythesis_sim as sim
    usetexTrue()
    import matplotlib.font_manager
    prop = matplotlib.font_manager.FontProperties(size=12)

    if ((egrid == False) & (diskFraction == False)):
        if plotsOnly != True:
            gcutil.mkdir(root + alnDir + mockdir)
        
            # Create the mock data for circular orbits
            sim.mock_data(nstars=nstars, e=0)
    
            # Run orbit simulation
            mc = sim.simulate_orbits(ntrials=ntrials, mockdir=mockdir, mockfile=mockfile,
                                     errorfile=errorfile, sigma=sigma)
            mc.run()

        elif plotsOnly == True:
            # Call function to make plots and examine the results
            # the function will return the chi2 statistic, which compares the
            # observed eccentricity distribution to the simulation's for
            # accelerating stars only
            print 'Making plots of the results for simulation e_initial = 0'
            chi2 = ecc_bias_simulation_results(nstars=nstars, ntrials=ntrials,
            #chi2 = compare_ecc_simVsObs(nstars=nstars, ntrials=ntrials,chi2=True,
                                        pdfdir=mockdir, mockfile=mockfile,
                                        makeStarPlots=False, suffix=suffix,
                                        vel_kick=False,sigma=sigma)
    elif egrid == True:
        # Run a grid of simulations
        #e_grid = np.arange(0.0, 1.0, 0.1) # Mean eccentricities
        #eWidth_grid = np.arange(0.0, 0.5, 0.05) # Intrinsic spread of eccentricities

        #v_grid = np.array([0.0, 0.07, 0.08, 0.09, 0.1])
        v_grid = np.array([0.0])
        clr = ['k.', 'r.', 'b.', 'g.', 'c.']
        e_grid = np.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.27, 0.3, 0.32, 0.35, 0.4, 0.45, 0.5])
        #e_grid = np.array([0.0, 0.1, 0.2, 0.25, 0.27, 0.3, 0.32, 0.35, 0.4, 0.45, 0.5])#, 0.55, 0.6, 0.65])

        # The chi2s that get returned from ecc_bias_simulation_results() are
        # for comparison of observed vs. simulated accelerating stars (chi2A) 
        # and non-accelerating (chi2NA) stars 
        chi2A = np.zeros((len(v_grid),len(e_grid)), dtype=float)
        chi2NA = np.zeros((len(v_grid),len(e_grid)), dtype=float)

        for vv in range(len(v_grid)):
            # Loop thru each eccentricity and spread in the grid
            for ee in range(len(e_grid)):
                ecc = e_grid[ee]
                
                #for ss in range(len(eWidth_grid)):
                #    ecc_std = eWidth_grid[ss]
    
                print '**** Eccentricity Grid Simulation ****'
                print '   e = %4.2f' % ecc
                print '   velocity fraction = %4.2f' % v_grid[vv]
                print ''

                if v_grid[vv] == 0.0:
                    gdir = 'vkick_%sfrac/sim_ecc_%s/' % (str(v_grid[vv]), str(ecc))
                else:
                    gdir = 'vkick_%sfrac/sim_vkick_%s/' % (str(v_grid[vv]), str(ecc))
                gcutil.mkdir(root + alnDir + mockdir)
                gcutil.mkdir(root + alnDir + mockdir + gdir)
                gcutil.mkdir(root + alnDir + mockdir + gdir + 'plots/')
                gcutil.mkdir(root + alnDir + mockdir + gdir + 'plots/eps/')
    
                if plotsOnly != True:
                    # Create the mock data
                    #sim.mock_data(nstars=nstars, e=ecc, eSTD=ecc_std,
                    #		   mockdir=mockdir+gdir,
                    #              outfile='ecc_%s_%s_mockdata.pickle' % \
                    #			    (str(ecc), str(ecc_std)))
        
                    sim.mock_data(nstars=nstars, e=ecc, frac=v_grid[vv],
                                  mockdir=mockdir+gdir,
                                  outfile='ecc_%s_vkick_mockdata.pickle' % str(ecc),
                                  vel_kick=False) ## temporary! while running e_grid sims with no kick
                                  #vel_kick=True)
        
                    # Run orbit simulation
                    mc = sim.simulate_orbits(ntrials=ntrials, mockdir=mockdir+gdir,
                                             mockfile='ecc_%s_vkick_mockdata.pickle' % \
                                             str(ecc), 
                                             errorfile=errorfile, sigma=sigma)
                    mc.run()
        
        
                elif plotsOnly == True:
                    # Call function to make plots and examine the results
                    # the function will return the chi2 statistic, which compares the
                    # observed eccentricity distribution to the simulation's for
                    # accelerating stars only
                    print 'Plotting results for vkick fraction = %s, e_initial = %s' %\
                          (str(v_grid[vv]), str(ecc))
                    #if (vv == 0) & (ee == 0):
                    #    eccPickle = False
                    #else:
                    #    eccPickle = True # load the pickle file for the observed ecc's

                    # Let's just use the current pickle file of the observed
                    # eccentricities for now. (if any orbits are re-run, we'll
                    # have to re-make this pickle file
                    eccPickle = True
                    #chi2A[vv,ee], chi2NA[vv,ee] = compare_ecc_simVsObs(nstars=nstars,
                    chi2A[vv,ee], chi2NA[vv,ee] = ecc_bias_simulation_results(nstars=nstars,
                                                sameCoverage=True,
                                                ntrials=ntrials, pdfdir=mockdir+gdir,
                                                mockfile='ecc_%s_vkick_mockdata.pickle' % \
                                                                       str(ecc),
                                                makeStarPlots=False,sigma=sigma,
                                                suffix='ecc_%s' % str(ecc), chi2=True,
                                                vel_kick=True)#,eccPickleLoad=eccPickle)

            usetexTrue()
            plotdir = '%s/%s/%s/vkick_%sfrac/plots/' % \
                      (root, alnDir, mockdir, str(v_grid[vv])) 
            gcutil.mkdir(plotdir+'eps')
            py.clf()
            py.figure(figsize=(6,6))
            py.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.1)
            py.plot(e_grid, chi2A[vv,:], 'k.',ms=8)#,label='Accel.')
            #py.plot(e_grid, chi2NA[vv,:], 'bs',ms=5,label='Non-accel.')
            py.xlabel(r'{\bf \huge{$e_0$}}')
            py.ylabel(r'{\bf \huge{$\chi^{2}$}}')
            #py.legend(numpoints=1,fancybox=True,loc=2,prop=prop)
            py.axis([-0.01, 0.55, 0, 110])
            py.savefig(plotdir + 'chi2_vs_eInitial.png')
            py.savefig(plotdir + 'eps/chi2_vs_eInitial.eps')
            py.close()
            #py.figure()
            #py.figure(figsize=(5,10))
            #py.subplots_adjust(left=0.15, right=0.96, wspace=0.25, hspace=0.25, top=0.9, bottom=0.1)
            #py.clf()
            #py.subplot(2,1,1)
            #py.plot(e_grid, chi2A[vv,:], 'k.')
            #py.xlabel('Initial Eccentricity')
            #py.ylabel('Chi2')
            #py.title('Accelerating Stars')
            #chi2min = chi2A[vv,:].min()
            #chi2max = chi2A[vv,:].max()
            #py.axis([-0.01, 0.7, chi2min-2, chi2max+2])
            #py.subplot(2,1,2)
            #py.plot(e_grid, chi2NA[vv,:], 'k.')
            #py.xlabel('Initial Eccentricity')
            #py.ylabel('Chi2')
            #py.title('Non-accelerating Stars')
            #chi2min = chi2NA[vv,:].min()
            #chi2max = chi2NA[vv,:].max()
            #py.axis([-0.01, 0.7, chi2min-2, chi2max+2])
            #py.savefig(plotdir + 'chi2_vs_eInitial.png')
            #py.savefig(plotdir + 'eps/chi2_vs_eInitial.eps')
            #py.close()
            pdb.set_trace()

    elif diskFraction == True:
        # Only make the plots...the simulations were run manually

        # Simulations run with 5%-55% disk stars
        frac = np.arange(0.05, 0.6, 0.05)
        ntot = 120

        # The chi2s that get returned from ecc_bias_simulation_results() are
        # for comparison of observed vs. simulated accelerating stars (chi2A) 
        # and non-accelerating (chi2NA) stars 
        chi2A = np.zeros(len(frac), dtype=float)
        chi2NA = np.zeros(len(frac), dtype=float)

        for ii in range(len(frac)):
            ff = np.around(frac[ii], decimals=2)   # have to do this b/c sometimes a
            				 # dumb rounding error is made
            ndisk = ff * ntot
            ff2 = np.around(1-ff,decimals=2) 
            niso = ff2 * ntot
    
            simDir = '%s/disk%s/' % (mockDir, int(ff*100))
            mockfile = 'nDisk%s_nIso%s_mockdata.pickle' % (int(ndisk), int(niso))
        
            print 'Plotting results for disk fraction = %4.2f' % frac[ii]

            # Let's just use the current pickle file of the observed
            # eccentricities for now. (if any orbits are re-run, we'll
            # have to re-make this pickle file
            eccPickle = False
            #chi2A[ii], chi2NA[ii] = compare_ecc_simVsObs(nstars=nstars,
            #                        ntrials=ntrials, pdfdir=simDir,
            #                        mockfile=mockfile, makeStarPlots=True,
            #                        vel_kick=True, eccPickleLoad=eccPickle)
            if getEcc == True:
                ecc_bias_simulation_results(nstars=nstars, sameCoverage=False,
                                            ntrials=ntrials, pdfdir=simDir,
                                            mockfile=mockfile, makeStarPlots=False,
                                            vel_kick=True, suffix='disk%s' % int(ff*100),
                                            sigma=sigma)

        #plotdir = '%s/plots/' % (root+alnDir+'sim_diskFraction/plots/')
        #py.figure(2)
        #py.clf()
        #py.figure(2, figsize=(10,4))
        #py.subplots_adjust(left=0.1, right=0.96, wspace=0.25, hspace=0.25, bottom=0.15)
        #py.subplot(1,2,1)
        #py.plot(frac, chi2A, 'k.')
        #py.xlabel('Disk Fraction')
        #py.ylabel('Chi2')
        #py.title('Accelerating Stars')
        #chi2min = chi2A.min()
        #chi2max = chi2A.max()
        #py.axis([-0.01, 0.7, chi2min-2, chi2max+2])
        #py.subplot(1,2,2)
        #py.plot(frac, chi2NA, 'k.')
        #py.xlabel('Disk Fraction')
        #py.ylabel('Chi2')
        #py.title('Non-accelerating Stars')
        #chi2min = chi2NA.min()
        #chi2max = chi2NA.max()
        #py.axis([-0.01, 0.7, chi2min-2, chi2max+2])
        #py.savefig(plotdir + 'chi2ecc_vs_diskFraction.png')
        #py.close(2)

            
    # Plot all on one figure
    py.figure()
    py.figure(figsize=(5,10))
    py.clf()
    py.subplots_adjust(left=0.15, right=0.96, wspace=0.3, hspace=0.3, top=0.94, bottom=0.1)
    py.subplot(2,1,1)
    for vv in range(len(v_grid)):
        py.plot(e_grid, chi2A[vv,:], clr[vv], label=str(v_grid[vv]))
    py.xlabel('Initial Eccentricity')
    py.ylabel(r'$\chi^{2}$')
    py.title('Accelerating Stars')
    chi2min = chi2A.min()
    chi2max = chi2A.max()
    py.axis([-0.01, 0.7, chi2min-2, chi2max+2])
    py.legend(loc=2,numpoints=1,fancybox=True,prop=prop)
    py.subplot(2,1,2)
    for vv in range(len(v_grid)):
        py.plot(e_grid, chi2NA[vv,:], clr[vv], label=str(v_grid[vv]))
    py.xlabel('Initial Eccentricity')
    py.ylabel(r'$\chi^{2}$')
    py.title('Non-accelerating Stars')
    chi2min = chi2NA.min()
    chi2max = chi2NA.max()
    py.axis([-0.01, 0.7, chi2min-2, chi2max+2])
    py.savefig('%s/%s/%s/plots/chi2_vs_eInitial_all.png' % (root, alnDir, mockdir))
    py.savefig('%s/%s/%s/plots/eps/chi2_vs_eInitial_all.eps' % (root, alnDir, mockdir))
    py.close()
    usetexFalse()

    if egrid == True:
        print ''
        print 'All simulations -- Accelerating stars: '
        print ('%14s  %7s  %6s') % ('Kick Fraction', 'Chi2min', 'e_init')
        for vv in range(len(v_grid)):
            chi2Amin = chi2A[vv,:].min()
            midx = chi2A[vv,:].argmin()
            print '%14d  %7.2f  %6.2f' % (v_grid[vv]*100., chi2Amin, e_grid[midx])
        print ''
        print 'All simulations -- Non-accelerating stars: '
        print ('%14s  %7s  %6s') % ('Kick Fraction', 'Chi2min', 'e_init')
        for vv in range(len(v_grid)):
            chi2NAmin = chi2NA[vv,:].min()
            midx = chi2NA[vv,:].argmin()
            print '%14d  %7.2f  %6.2f' % (v_grid[vv]*100., chi2NAmin, e_grid[midx])


def compare_ecc_simVsObs(nstars=100, ntrials=10**4, pdfdir='sim_vkick_fracCircVel/',
                         mockfile='circularFlatMC_mockdata.pickle',
                         sigma=4.0, makeStarPlots=True, suffix='circular',
                         eccPickleLoad=True, vel_kick=True, sameCoverage=True):
    # Read in the mock data
    # Astrometric units are: arcsec, mas/yr, mas/yr^2
    mockdata = open(root + alnDir + pdfdir + mockfile)
    mbh = pickle.load(mockdata)
    dist = pickle.load(mockdata)
    orb_all = pickle.load(mockdata)
    sma_all = pickle.load(mockdata) # AU
    xM = pickle.load(mockdata) # (+x to east)
    yM = pickle.load(mockdata)
    zM = pickle.load(mockdata)
    vxM = pickle.load(mockdata) # (+x to east)
    vyM = pickle.load(mockdata)
    vzM = pickle.load(mockdata) # mas/yr
    axM = pickle.load(mockdata) # (+x to east)
    ayM = pickle.load(mockdata)
    azM = pickle.load(mockdata)
    t0M = pickle.load(mockdata) # periapse passage
    t_obs = pickle.load(mockdata) # observational time for mock data point
    mockdata.close()

    cc = objects.Constants()
    asy_to_kms = dist * cc.cm_in_au / (1.e5 * cc.sec_in_yr)

    # Do we want to cover the same field as our observations?
    if sameCoverage == True:
        kp = np.where((xM > -10.) & (xM < 15.) & (yM > -8.) & (yM < 14.))[0]
        orb_all = orb_all[kp]
        sma_all = sma_all[kp]
        xM = xM[kp]
        yM = yM[kp]
        zM = zM[kp]
        vxM = vxM[kp]
        vyM = vyM[kp]
        vzM = vzM[kp]
        axM = axM[kp]
        ayM = ayM[kp]
        azM = azM[kp]
        t0M = t0M[kp]
        nstars = len(kp)
    else:
        kp = np.arange(nstars)

    # Define some variables
    ecc = np.zeros((nstars, ntrials), dtype=float)
    incl = np.zeros((nstars, ntrials), dtype=float)
    Omega = np.zeros((nstars, ntrials), dtype=float)
    w = np.zeros((nstars, ntrials), dtype=float)
    prd = np.zeros((nstars, ntrials), dtype=float)
    phase = np.zeros((nstars, ntrials), dtype=float)
    t0 = np.zeros((nstars, ntrials), dtype=float)
    ar = np.zeros((nstars, ntrials), dtype=float)
    asig = np.zeros((nstars), dtype=float)
    evecX = np.zeros((nstars, ntrials), float)
    evecY = np.zeros((nstars, ntrials), float)
    evecZ = np.zeros((nstars, ntrials), float)
    evecP = np.zeros((nstars, ntrials), float)
    evecQ = np.zeros((nstars, ntrials), float)
    evecN = np.zeros((nstars, ntrials), float)

    # Simulated stars - separate the accel vs. non-accel stars' disk solutions
    simEccAccDiskTmp = []
    simEccNonAccDiskTmp = []
    simEccMembersTmp = []
    evecP_AccDiskTmp = []
    evecQ_AccDiskTmp = []
    evecN_AccDiskTmp = []
    evecP_NonAccDiskTmp = []
    evecQ_NonAccDiskTmp = []
    evecN_NonAccDiskTmp = []
    evecP_allDiskTmp = []
    evecQ_allDiskTmp = []
    evecN_allDiskTmp = []

    # Read in the disk membership probabilities
    pfile = '%s%s%s/plots/HEALpixMaps/simdisk_membership_prob.dat' % (root, alnDir, pdfdir)
    diskTab = asciidata.open(pfile)
    name = [diskTab[0][ss].strip() for ss in range(diskTab.nrows)]
    diskP = diskTab[1].tonumpy()
    diskIdx = (np.where(diskP > 0.3173))[0]
    ndIdx = (np.where(diskP <= 0.3173))[0]

    for ss in range(nstars):
        sidx = kp[ss]
        pdffile = '%s/%s/%s/star%s.mc.dat' % (root, alnDir, pdfdir, sidx)
        pdf = pickle.load(open(pdffile))

        # Read in the orbital parameters from the simulation
        ecc[ss,:] = pdf.e
        evecX[ss,:] = pdf.evec[:,0]
        evecY[ss,:] = pdf.evec[:,1]
        evecZ[ss,:] = pdf.evec[:,2]
        incl[ss,:] = pdf.i
        Omega[ss,:] = pdf.o # big omega
        w[ss,:] = pdf.w     # little omega
        prd[ss,:] = pdf.p
        phase[ss,:] = pdf.ph
        t0[ss,:] = pdf.t0
        ar[ss,:] = pdf.ar  # mas/yr^2
        asig[ss] = pdf.asig # significance of accel (if < -4, accel measurement used)
        mbh = pdf.m[0]
        R0 = pdf.r0[0]
        x = pdf.x 
        y = pdf.y
        r2d = np.sqrt(x**2 + y**2) # arcsec
        r2d_cm = r2d * dist * cc.cm_in_au

        # Make sure the acceleration is greater than the theoretical minimum
        # at this 2D radius (remember, accel is negative)
        GM = cc.G * mbh * cc.msun # cgs
        amin_cgs = -GM / r2d_cm**2 # cm/s^2
        amin = (amin_cgs * 1.e3 * cc.sec_in_yr**2)
        amin /= (cc.cm_in_au * dist) # mas/yr^2

        unphys = np.where(ar[ss,:] < amin)[0]
        if len(unphys) > 0:
            print 'Found %d accelerations below the theoretical minimum' % len(unphys)
        
        # Project eccentricities onto disk plane
        (eP, eQ, eN) = diskProject(evecX[ss,:], evecY[ss,:], evecZ[ss,:],
                                   idisk=idisk, odisk=odisk)
        evecP[ss,:] = eP
        evecQ[ss,:] = eQ
        evecN[ss,:] = eN

        sIdx = simWhereInDisk(incl[ss,:], Omega[ss,:]) # disk solutions
        #if ss in sig: # sig accel
        if asig[ss] < -4.0:
            #simEccAccDisk = np.concatenate([simEccAccDisk, ecc[ss,sIdx]])
            #evecP_AccDisk = np.concatenate([evecP_AccDisk, evecP[ss,sIdx]])
            #evecQ_AccDisk = np.concatenate([evecQ_AccDisk, evecQ[ss,sIdx]])
            #evecN_AccDisk = np.concatenate([evecN_AccDisk, evecN[ss,sIdx]])
            simEccAccDiskTmp.append(ecc[ss,sIdx])
            evecP_AccDiskTmp.append(evecP[ss,sIdx])
            evecQ_AccDiskTmp.append(evecQ[ss,sIdx])
            evecN_AccDiskTmp.append(evecN[ss,sIdx])
        #if ss in nonsig: # uniform prior (non-sig accel)
        else:
            #simEccNonAccDisk = np.concatenate([simEccNonAccDisk, ecc[ss,sIdx]])
            #evecP_NonAccDisk = np.concatenate([evecP_NonAccDisk, evecP[ss,sIdx]])
            #evecQ_NonAccDisk = np.concatenate([evecQ_NonAccDisk, evecQ[ss,sIdx]])
            #evecN_NonAccDisk = np.concatenate([evecN_NonAccDisk, evecN[ss,sIdx]])
            simEccNonAccDiskTmp.append(ecc[ss,sIdx])
            evecP_NonAccDiskTmp.append(evecP[ss,sIdx])
            evecQ_NonAccDiskTmp.append(evecQ[ss,sIdx])
            evecN_NonAccDiskTmp.append(evecN[ss,sIdx])

        # Get ecc for candidate disk members
        if ss in diskIdx:
            simEccMembersTmp.append(ecc[ss,sIdx])

        # get evec disk solutions for all the stars as well (accel and non-accel)
        #evecP_allDisk = np.concatenate([evecP_allDisk, evecP[ss,sIdx]])
        #evecQ_allDisk = np.concatenate([evecQ_allDisk, evecQ[ss,sIdx]])
        #evecN_allDisk = np.concatenate([evecN_allDisk, evecN[ss,sIdx]])
        evecP_allDiskTmp.append(evecP[ss,sIdx])
        evecQ_allDiskTmp.append(evecQ[ss,sIdx])
        evecN_allDiskTmp.append(evecN[ss,sIdx])

    # What we have from above are lists of arrays. We can merge everything
    # inside the list so it's just one big list
    simEccMembers = [ee for ii in simEccMembersTmp for ee in ii]
    simEccAccDisk = [ee for ii in simEccAccDiskTmp for ee in ii]
    evecP_AccDisk = [ee for ii in evecP_AccDiskTmp for ee in ii]
    evecQ_AccDisk = [ee for ii in evecQ_AccDiskTmp for ee in ii]
    evecN_AccDisk = [ee for ii in evecN_AccDiskTmp for ee in ii]
    simEccNonAccDisk = [ee for ii in simEccNonAccDiskTmp for ee in ii]
    evecP_NonAccDisk = [ee for ii in evecP_NonAccDiskTmp for ee in ii]
    evecQ_NonAccDisk = [ee for ii in evecQ_NonAccDiskTmp for ee in ii]
    evecN_NonAccDisk = [ee for ii in evecN_NonAccDiskTmp for ee in ii]
    evecP_allDisk = [ee for ii in evecP_allDiskTmp for ee in ii]
    evecQ_allDisk = [ee for ii in evecQ_allDiskTmp for ee in ii]
    evecN_allDisk = [ee for ii in evecN_allDiskTmp for ee in ii]

    print eccPickleLoad
    if eccPickleLoad != True:
        # Observations
        # Read in file listing accelerating sources
        _acc = asciidata.open('%s/%s/tables/accelerating_sources.dat' % (root, alnDir))
        acc = _acc[0].tonumpy()
        accels = [aa.strip() for aa in acc]
        orbDir = 'aorb_thesis/'
        #orbDir = 'aorb_acc_mrPDF_MC_newMosaic/'
        eccAccTmp = []
        eccAccDiskTmp = []
        eccAccNonDiskTmp = []
        eccNonAccTmp = []
        eccNonAccDiskTmp = []
        eccNonAccNonDiskTmp = []
    
        # Disk candidates
        diskTab = asciidata.open(root+alnDir+'tables/disk_membership_prob_mosaic.dat')
        name = [diskTab[0][jj].strip() for jj in range(diskTab.nrows)]
        diskP = diskTab[1].tonumpy()
        diskIdx = (np.where(diskP > 2.7e-3))[0]
        diskCand = [name[nn] for nn in diskIdx]
    
        # Loop over the disk candidates in the real data
        for ii in diskIdx:
            starName = name[ii]
            
            orbFile = ('%s/%s/%s/%s.mc.dat' % (root, alnDir, orbDir, starName))
            if os.path.exists(orbFile) == False:
                continue
            tmp = pickle.load(open(orbFile))
            print 'Candidate disk member: %s' % starName
    
            # Separate out the disk from non-disk solutions
            idx = whereInDisk(tmp, angleCut=angleCut)
            eD = tmp.e[idx]
            nd = np.setdiff1d(np.arange(ntrials), idx)
            eND = tmp.e[nd]
    
            if starName in accels:
                # Accelerating stars:
                # Get the eccentricities 
                #eccAcc = np.concatenate([eccAcc, tmp.e])
                eccAccTmp.append(tmp.e)
        
                # Separate out just the disk solutions
                #eccAccDisk = np.concatenate([eccAccDisk, eD]) 
                eccAccDiskTmp.append(eD) 
        
                # Non-disk solutions
                #eccAccNonDisk = np.concatenate([eccAccNonDisk, eND])
                eccAccNonDiskTmp.append(eND)
            else:
                # Non-accelerating stars:
                # Get the eccentricities 
                #eccNonAcc = np.concatenate([eccNonAcc, tmp.e])
                eccNonAccTmp.append(tmp.e)
        
                # Separate out just the disk solutions
                #eccNonAccDisk = np.concatenate([eccNonAccDisk, eD]) 
                eccNonAccDiskTmp.append(eD) 
        
                # Non-disk solutions
                #eccNonAccNonDisk = np.concatenate([eccNonAccNonDisk, eND])
                eccNonAccNonDiskTmp.append(eND)
    
        # What we have from above are lists of arrays. We can merge everything
        # inside the list so it's just one big list
        eccAcc = [ee for ii in eccAccTmp for ee in ii]
        eccAccDisk = [ee for ii in eccAccDiskTmp for ee in ii]
        eccAccNonDisk = [ee for ii in eccAccNonDiskTmp for ee in ii]
        eccNonAcc = [ee for ii in eccNonAccTmp for ee in ii]
        eccNonAccDisk = [ee for ii in eccNonAccDiskTmp for ee in ii]
        eccNonAccNonDisk = [ee for ii in eccNonAccNonDiskTmp for ee in ii]
    
        #if pickleDump == True:
        _file = open(root + alnDir + 'tables/observed_eccentricities.pickle', 'w')
        pickle.dump(eccAcc, _file)
        pickle.dump(eccAccDisk, _file)
        pickle.dump(eccAccNonDisk, _file)
        pickle.dump(eccNonAcc, _file)
        pickle.dump(eccNonAccDisk, _file)
        pickle.dump(eccNonAccNonDisk, _file)
        _file.close()

    elif eccPickleLoad == True:
        print
        print 'Loading observed eccentricities from pickle file'
        eccFile = sy_open.go(root + alnDir + 'tables/observed_eccentricities.pickle')
        #eccFile = open(root + alnDir + 'tables/observed_eccentricities.pickle')
        eccAcc = pickle.load(eccFile)
        eccAccDisk = pickle.load(eccFile)
        eccAccNonDisk = pickle.load(eccFile)
        eccNonAcc = pickle.load(eccFile)
        eccNonAccDisk = pickle.load(eccFile)
        eccNonAccNonDisk = pickle.load(eccFile)
        #eccFile.close()
        
    usetexTrue()
    # Plot ecc for candidate disk members in the simulation
    eccStep = 0.05
    binsIn = np.arange(0, 1+eccStep, eccStep)
    py.figure(figsize=(7,6))
    py.subplots_adjust(left=0.1, right=0.95, top=0.9)
    py.clf()
    aa, bb, cc = py.hist(simEccMembers, bins=binsIn, color='k', histtype='step',
                              normed=True)
    py.xlabel('Eccentricity', fontsize=16)
    py.ylabel('Probability Density', fontsize=16)
    py.title('Candidate Disk Members (Disk Solutions)', fontsize=16)
    py.axis([0, 1, 0, aa.max()+0.2])
    py.savefig(root+alnDir+pdfdir+'plots/eccPDF_candidate_members.png')
    py.close()
    usetexFalse()

    # How many fall into each eccentricity bin? for statistics purposes later...
    nObsE, nbb, ncc = py.hist(eccAccDisk, bins=binsIn, color='b', histtype='step',
                              normed=False)
    nSimE, nbb, ncc = py.hist(simEccAccDisk, bins=binsIn, color='r', histtype='step',
                              normed=False)
    # This plot is also made in ecc_bias_simulation_results(), but this function is faster
    py.clf()
    py.figure(figsize=(7,6))
    pdfObsE, bb, cc = py.hist(eccAccDisk, bins=binsIn, color='b', histtype='step',
                              normed=True, label='Observed')
    pdfSimE, bb, cc = py.hist(simEccAccDisk, bins=binsIn, color='r', histtype='step',
                              normed=True, label='Simulated')
    py.xlabel('Eccentricity', fontsize=16)
    py.ylabel('Probability Density', fontsize=16)
    py.title('Accelerating Disk Members')
    py.axis([0, 1, 0, max(max(pdfObsE), max(pdfSimE))+0.2])
    py.legend(numpoints=1,fancybox=True)
    py.savefig(root+alnDir+pdfdir+'plots/eccPDF_compare_to_obs.png')
    py.close()

    py.clf()
    py.figure(figsize=(7,6))
    pdfObsEna, bb, cc = py.hist(eccNonAccDisk, bins=binsIn, color='b', histtype='step',
                              normed=True, label='Observed')
    pdfSimEna, bb, cc = py.hist(simEccNonAccDisk, bins=binsIn, color='r', histtype='step',
                              normed=True, label='Simulated')
    py.xlabel('Eccentricity', fontsize=16)
    py.ylabel('Probability Density', fontsize=16)
    py.title('Non-accelerating Disk Members')
    py.axis([0, 1, 0, max(max(pdfObsEna), max(pdfSimEna))+0.2])
    py.legend(numpoints=1,fancybox=True)
    py.savefig(root+alnDir+pdfdir+'plots/eccPDF_compare_to_obs_nonAccel.png')
    py.close()

    # Compare accelerating stars (sim vs. obs)
    errTot2 = np.ones(len(pdfObsE)) # assuming errors = 1, for now
    echi2_acc = ((pdfObsE - pdfSimE)**2. / (errTot2) ).sum() 

    # Compare non-accelerating stars (sim vs. obs)
    errTot2 = np.ones(len(pdfObsEna)) # assuming errors = 1, for now
    echi2_nonAcc = ((pdfObsEna - pdfSimEna)**2. / (errTot2) ).sum() 
                                                              
    print 'Accelerating stars comparison:'
    print '  Chi2 statistic: %8.2f' % echi2_acc
    print
    print 'Non-accelerating stars comparison:'
    print '  Chi2 statistic: %8.2f' % echi2_nonAcc
    print

    return echi2_acc, echi2_nonAcc

def ecc_bias_simulation_results(nstars=100, ntrials=10**4, e0=0.32,
                                pdfdir='sim_vkick_fracCircVel/',
                                mockfile='circularFlatMC_mockdata.pickle',
                                sigma=4.0, makeStarPlots=True, suffix='circular',
                                ks=False, chi2=True, vel_kick=True, sameCoverage=True):

    # Read in the mock data
    # Astrometric units are: arcsec, mas/yr, mas/yr^2
    #mockdata = sy_open.go(root + alnDir + pdfdir + mockfile)
    mockdata = open(root + alnDir + pdfdir + mockfile)
    mbh = pickle.load(mockdata)
    dist = pickle.load(mockdata)
    orb_all = pickle.load(mockdata)
    sma_all = pickle.load(mockdata) # AU
    xM = pickle.load(mockdata) # (+x to east)
    yM = pickle.load(mockdata)
    zM = pickle.load(mockdata)
    vxM = pickle.load(mockdata) # (+x to east)
    vyM = pickle.load(mockdata)
    vzM = pickle.load(mockdata) # mas/yr
    axM = pickle.load(mockdata) # (+x to east)
    ayM = pickle.load(mockdata)
    azM = pickle.load(mockdata)
    t0M = pickle.load(mockdata) # periapse passage
    t_obs = pickle.load(mockdata) # observational time for mock data point
    mockdata.close()

    # Do we want to cover the same field as our observations?
    if sameCoverage == True:
        nstarsOrig = nstars
        kp = np.where((xM > -10.) & (xM < 15.) & (yM > -8.) & (yM < 14.))[0]
        orb_all = orb_all[kp]
        sma_all = sma_all[kp]
        xM = xM[kp]
        yM = yM[kp]
        zM = zM[kp]
        vxM = vxM[kp]
        vyM = vyM[kp]
        vzM = vzM[kp]
        axM = axM[kp]
        ayM = ayM[kp]
        azM = azM[kp]
        t0M = t0M[kp]
        nstars = len(kp)
    else:
        kp = np.arange(nstars)

    new_orb = []

    p_tmp = np.array([orb.p for orb in orb_all])
    phM = ((t_obs - t0M[:]) % p_tmp) / p_tmp

    r2dM = np.sqrt(xM**2 + yM**2)

    # Define some variables
    ecc = np.zeros((nstars, ntrials), dtype=float)
    incl = np.zeros((nstars, ntrials), dtype=float)
    Omega = np.zeros((nstars, ntrials), dtype=float)
    w = np.zeros((nstars, ntrials), dtype=float)
    prd = np.zeros((nstars, ntrials), dtype=float)
    phase = np.zeros((nstars, ntrials), dtype=float)
    t0 = np.zeros((nstars, ntrials), dtype=float)
    vx = np.zeros((nstars, ntrials), dtype=float)
    vy = np.zeros((nstars, ntrials), dtype=float)
    vz = np.zeros((nstars, ntrials), dtype=float)
    ar = np.zeros((nstars, ntrials), dtype=float)
    sma_sim = np.zeros((nstars, ntrials), dtype=float)
    angOffCW = np.zeros((nstars, ntrials), dtype=float)
    asig = np.zeros((nstars), dtype=float)
    zS_ave = np.zeros((nstars), dtype=float)
    zS_rms = np.zeros((nstars), dtype=float)
    e_rms = np.zeros((nstars), dtype=float)
    e_med = np.zeros((nstars), dtype=float)
    i_ave = np.zeros((nstars), dtype=float)
    i_std = np.zeros((nstars), dtype=float)
    i_med = np.zeros((nstars), dtype=float)
    o_ave = np.zeros((nstars), dtype=float)
    o_std = np.zeros((nstars), dtype=float)
    o_med = np.zeros((nstars), dtype=float)
    evecX = np.zeros((nstars, ntrials), float)
    evecY = np.zeros((nstars, ntrials), float)
    evecZ = np.zeros((nstars, ntrials), float)
    evecP = np.zeros((nstars, ntrials), float)
    evecQ = np.zeros((nstars, ntrials), float)
    evecN = np.zeros((nstars, ntrials), float)
    e_input = np.zeros(nstars, float)
    p_input = np.zeros(nstars, float)
    i_input = np.zeros(nstars, float)
    O_input = np.zeros(nstars, float)
    w_input = np.zeros(nstars, float)
    t0_input = np.zeros(nstars, float)

    cc = objects.Constants()

    sinip = np.sin(np.radians(idisk))
    cosip = np.cos(np.radians(idisk))

    asy_to_kms = dist * cc.cm_in_au / (1.e5 * cc.sec_in_yr)

    unb = []
    
    outroot = root + alnDir + pdfdir + 'plots/'
    for ss in range(nstars):
        sidx = kp[ss]
        pdffile = '%s/%s/%s/star%s.mc.dat' % (root, alnDir, pdfdir, sidx)
        pdf = pickle.load(open(pdffile))
        
        # Make a HEALpix map
        #hp = makePdfHealpix('star'+str(sidx), pdf, outroot,
        #                    ntrials=ntrials, nside=64,makeplot=True)

        # Read in the orbital parameters from the simulation
        ecc[ss,:] = pdf.e
        evecX[ss,:] = pdf.evec[:,0]
        evecY[ss,:] = pdf.evec[:,1]
        evecZ[ss,:] = pdf.evec[:,2]
        incl[ss,:] = pdf.i
        Omega[ss,:] = pdf.o # big omega
        w[ss,:] = pdf.w     # little omega
        prd[ss,:] = pdf.p
        phase[ss,:] = pdf.ph
        t0[ss,:] = pdf.t0
        vx[ss,:] = pdf.vx # km/s
        vy[ss,:] = pdf.vy # km/s
        vz[ss,:] = pdf.vz # km/s
        ar[ss,:] = pdf.ar # mas/yr^2
        mbh = pdf.m[0]
        R0 = pdf.r0[0]

        # Project eccentricities onto disk plane
        (eP, eQ, eN) = diskProject(evecX[ss,:], evecY[ss,:], evecZ[ss,:],
                                   idisk=idisk, odisk=odisk)
        evecP[ss,:] = eP
        evecQ[ss,:] = eQ
        evecN[ss,:] = eN

        # calculate the semi-major axis
        sma_sim[ss,:] = (mbh * prd[ss,:]**2)**(1./3.) / dist # arcsec
        
        asig[ss] = pdf.asig # significance of accel (if < -4, accel measurement used)
        
        v3d = np.sqrt(vx[ss,:]**2 + vy[ss,:]**2 + vz[ss,:]**2)
  
        # Position
        x = pdf.x 
        y = pdf.y
        z = pdf.z
        r2d = np.sqrt(x**2 + y**2)
        r3d = np.sqrt(x**2 + y**2 + z**2) # arcsec

        # Save the simulated z's for plotting later
        zS_ave[ss] = z.mean()
        zS_rms[ss] = z.std(ddof=1)
        #pdb.set_trace()

        r3d_au = r3d * R0
        r3d_cm = r3d_au * cc.cm_in_au

        # Escape velocity
        GM = cc.G * mbh * cc.msun # cgs
        vr_esc = np.sqrt(2.0 * GM / r3d_cm) / 1.e5 # escape speed at star's position
        
        rr = (np.arange(10 * 25) * 0.1) + 0.1
        rr_au = rr * R0
        rr_pc = rr_au / cc.au_in_pc
        rr_cm = rr_au * cc.cm_in_au
        v_esc = np.sqrt(2.0 * GM / rr_cm) / 1.e5 # v_esc for range of radii

        vrat = v3d / vr_esc

        # Get the angular distance from the disk for all solutions
        # Calculate angle from input i,O of disk:
        sini = np.sin(incl[ss,:] / rad2deg)
        cosi = np.cos(incl[ss,:] / rad2deg)
        cosodiff = np.cos( (Omega[ss,:] - odisk) / rad2deg )
        angOffCW[ss,:] = (np.arccos( (sini * sinip * cosodiff) + (cosi * cosip) )) * rad2deg

        # Compare input to expectation value of distributions
        e_rms[ss] = np.sqrt((ecc[ss,:]**2).sum() / ntrials) # RMS, not standard dev.
        e_med[ss] = np.median(ecc[ss,:])
        i_ave[ss] = incl[ss,:].mean()
        i_std[ss] = incl[ss,:].std(ddof=1)
        i_med[ss] = np.median(incl[ss,:])
        o_ave[ss] = Omega[ss,:].mean()
        o_std[ss] = Omega[ss,:].std(ddof=1)
        o_med[ss] = np.median(Omega[ss,:])

        xLo = r3d.min() - 1.
        xHi = r3d.max() + 1.
        yLo = vrat.min() 
        yHi = vrat.max()

        if vel_kick == True:
            # What was the input orbit? This should only be different
            # than the assumed orbit if we gave the stars an extra
            # velocity kick. This will not be in orb_all b/c orb_all
            # has the orbit that produced the mock data before a velocity
            # kick was added
            rvec = np.array([xM[ss], yM[ss], zM[ss]])
            vvec = np.array([vxM[ss]/1.e3*asy_to_kms, vyM[ss]/1.e3*asy_to_kms, vzM[ss]/1.e3*asy_to_kms])
            revec = np.zeros(3, dtype=float)
            vevec = np.zeros(3, dtype=float)
            try:
                newOrb = orbits.Orbit()
                newOrb.xyz2kep(rvec, vvec, revec, vevec, t_obs[0], mass=mass, dist=dist)
            except ValueError:
                newOrb = None
                # Recall sidx gives the star ID, which may not be the same as ss
                # if we cut stars to account for the observed field coverage
                print 'Star %i -- velocity kick produces unbound orbit!' % sidx
                unb = np.concatenate([unb, [ss]])
            new_orb = np.concatenate([new_orb, [newOrb]])

            # What were the input orbital parameters (after the velocity kick added)
            if ss not in unb:
                e_input[ss] = newOrb.e
                p_input[ss] = newOrb.p
                i_input[ss] = newOrb.i
                O_input[ss] = newOrb.o
                w_input[ss] = newOrb.w
                t0_input[ss] = newOrb.t0
            elif ss in unb:
                e_input[ss] = -1
                p_input[ss] = -1
                i_input[ss] = -1
                O_input[ss] = -1
                w_input[ss] = -1
                t0_input[ss] = -1

        if makeStarPlots == True:
            # Plot up some distributions for each star

            if vel_kick == True:
                e_in = e_input[ss]
                i_in = i_input[ss]
                O_in = O_input[ss]
                w_in = w_input[ss]
                t0_in = t0_input[ss]
                # get the phase:
                p_tmp = p_input[ss]
                ph_in = ((t_obs[0] - t0_in) % p_tmp) / p_tmp
            else:
                e_in = orb_all[ss].e
                ph_in = phM[ss]
                i_in = orb_all[ss].i
                O_in = orb_all[ss].o
                w_in = orb_all[ss].w
                t0_in = orb_all[ss].t0

            py.clf()
            py.figure()
            py.figure(figsize=(10,10))
            py.subplots_adjust(left=0.1, right=0.95, wspace=0.3, hspace=0.3,
                               top=0.94, bottom=0.1)
            py.subplot(3,3,1)
            nn, bb, pp = py.hist(ecc[ss,:].flatten(), bins=np.arange(0, 1.1, 0.1),
                    histtype='step',normed=True)
            py.text(0.6, nn.max()-1.0, 'e_in = %4.2f' % e_in, fontsize=10)
            py.xlabel('Eccentricity')
            py.title('Star %i (R2D=%6.2f arcsec)' % \
                     (sidx, r2d.mean()), fontsize=11)
    
            py.subplot(3,3,2)
            py.hist(phase[ss,:].flatten(), bins=np.arange(0, 1.01, 0.01),
                    histtype='step',normed=True)
            py.xlabel('Phase')
            py.title('Mock phase = %4.2f' % ph_in, fontsize=12)
    
            py.subplot(3,3,3)
            nn, bb, pp = py.hist(incl[ss,:].flatten(), bins=np.arange(0, 181, 1.0),
                         histtype='step',normed=True)
            py.axis([90, 180, 0, nn.max()+0.01])
            py.xlabel('Inclination (deg)')
            py.title('Mock Incl = %5.1f deg' % i_in, fontsize=12)
    
            py.subplot(3,3,4)
            nn, bb, pp = py.hist(Omega[ss,:].flatten(), bins=np.arange(0, 370, 10.0),
                         histtype='step',normed=True)
            py.axis([0, 200, 0, nn.max()+0.01])
            py.xlabel('PA to the Ascending Node (deg)')
            py.title('Mock Omega = %5.1f deg' % O_in, fontsize=12)
    
            py.subplot(3,3,5)
            py.hist(w[ss,:].flatten(), bins=np.arange(0, 365, 5.0),
                    histtype='step',normed=True)
            py.xlabel('Argument of Periapse (deg)')
            py.title('Mock w = %6.2f deg' % w_in, fontsize=12)
    
            py.subplot(3,3,6)
            py.hist(t0[ss,:].flatten(), bins=np.arange(0, 5100, 100.0),
                    histtype='step',normed=True)
            py.xlabel('Periapse Passage (year)')
            py.title('Mock t0 = %7.2f' % t0_in, fontsize=12)

            py.subplot(3,3,7)
            #pmin = prd[ss,:].min()
            #pmax = prd[ss,:].max()
            #xmax = np.median(prd[ss,:])*5.0 # for plotting purposes
            #pInc = (pmax - pmin) / 100.0
            #nn, bb, pp = py.hist(prd[ss,:].flatten(), bins=np.arange(pmin, pmax, pInc),
            #        histtype='step',normed=True)
            #py.xlabel('Period (years)')
            #py.axis([pmin,np.median(prd[ss,:])*2.0,0,nn.max()+(nn.max()/100)])
            #py.title('Mock Period = %7.2f yrs' % orb_all[ss].p, fontsize=12)
            minAng = angOffCW[ss,:].min()
            maxAng = angOffCW[ss,:].max()
            angInc = (maxAng - minAng) / 100.0
            py.hist(angOffCW[ss,:].flatten(), bins=np.arange(minAng,maxAng,angInc),
                    histtype='step', normed=True)
            py.xlabel('Angular Distance from Disk (deg)')
            py.title('accel sigma = %6.2f' % asig[ss], fontsize=12)
    
            py.subplot(3,3,8)
            sma_ave = sma_sim[ss,:].mean()
            sma_std = sma_sim[ss,:].std(ddof=1)
            sma_all_asec = sma_all[ss] / dist
            axMin = max(0, sma_ave-3.*sma_std)
            axMax = min(100, sma_ave+3.*sma_std)
            nn, bb, pp = py.hist(sma_sim[ss,:].flatten(), bins=100,
                         range=((sma_ave-5.*sma_std), (sma_ave+5.*sma_std)),
                         histtype='step', normed=True)
            py.xlabel('Semi-major Axis (arcsec)')
            py.title('Mock SMA = %7.3f"' % sma_all_asec, fontsize=12)
            py.axis([axMin, axMax, 0, nn.max()])
    
            py.subplot(3,3,9)
            zmin = z.min()
            zmax = z.max()
            zInc = (zmax-zmin) / 100.
            nn, bb, pp = py.hist(z, bins=np.arange(zmin,zmax,zInc), histtype='step', normed=True)
            py.title('Mock z = %6.3f"' % zM[ss], fontsize=12)
            py.xlabel('z (arcsec)')
            #py.plot(z, phase[ss,:], 'k.')
            #py.xlabel('z (arcsec)')
            #py.ylabel('Phase')
            py.savefig(root+alnDir+pdfdir+'plots/sim_orbPDF_star'+str(sidx)+'.png')
            py.close()

    sig = np.where(asig <= -sigma)[0]
    nonsig = np.where(asig > -sigma)[0]
    print 'Number of stars with significant accelerations (%i sigma): %i' % \
          (int(sigma), len(sig))

    # Print out some info for stars whose accelerations were used
    ave_e_sig = ecc[sig,:].mean()
    std_e_sig = ecc[sig,:].std(ddof=1)
    med_e_sig = np.median(ecc[sig,:])
    print
    print 'Significant accelerations:'
    print '   < e > = %4.2f +- %4.2f' % (ave_e_sig, std_e_sig)
    print '   median e = %4.2f' % med_e_sig
    ave_e_nsig = ecc[nonsig,:].mean()
    std_e_nsig = ecc[nonsig,:].std(ddof=1)
    med_e_nsig = np.median(ecc[nonsig,:])
    print
    print 'Uniform acceleration prior:'
    print '   < e > = %4.2f +- %4.2f' % (ave_e_nsig, std_e_nsig)
    print '   median e = %4.2f' % med_e_nsig

    # Separate out the disk and non-disk solutions
    #simEccDisk = []
    #simEccNonDisk = []
    #dIdx = simWhereInDisk(incl[ss,:], Omega[ss,:])
    #simEccDisk = np.concatenate([simEccDisk, ecc[ss,dIdx]])
    #ndIdx = np.setdiff1d(np.arange(ntrials), dIdx)
    #simEccNonDisk = np.concatenate([simEccNonDisk, ecc[ss,ndIdx]])

    if vel_kick == True:
        gcutil.mkdir(root+alnDir+pdfdir+'plots/eps')
        # Plot up the original eccentricities from the mock data
        eccStep = 0.05
        binsIn = np.arange(0, 1+eccStep, eccStep)
        py.clf()
        py.figure(figsize=(6,6))
        ii = np.where(e_input >= 0.0)[0]
        aa,bb,cc = py.hist(e_input[ii], bins=binsIn, histtype='step', normed=True,
                           color='black')
        print 'Input ecc = %4.2f +- %4.2f' % (e_input[ii].mean(), (e_input[ii]).std(ddof=1))
        py.plot([e0, e0],[0,aa.max()+(0.01*aa.max())],'k--')
        py.xlabel('Eccentricity', fontsize=16)
        #py.ylabel('Probability Density', fontsize=16)
        py.axis([0.0, 1.0, 0.0, aa.max()+(0.01*aa.max())])
        #py.title('Mock Data Input Eccentricities')
        py.savefig(root+alnDir+pdfdir+'plots/mockdata_inputEcc_%s.png' % suffix)
        py.savefig(root+alnDir+pdfdir+'plots/eps/mockdata_inputEcc_%s.eps' % suffix)
        py.close()
        # temp
        if sameCoverage == False:
            ndf = suffix[-2:]
            if ndf == 'k5':
                ndf = 5.
            nd = int(ndf)/100. * nstars
            foo = np.where(e_input >= 0.0)[0]
            fidx = np.where(foo < nd)[0] # disk stars for 5% fraction
            aveEdisk = e_input[foo[fidx]].mean()
            stdEdisk = e_input[foo[fidx]].std(ddof=1)
            aveIdisk = i_input[foo[fidx]].mean()
            stdIdisk = i_input[foo[fidx]].std(ddof=1)
            aveOdisk = O_input[foo[fidx]].mean()
            stdOdisk = O_input[foo[fidx]].std(ddof=1)
            print
            print '****Average input eccentricity for %d disk stars: %5.3f +- %5.3f' % \
                  (nd,aveEdisk,stdEdisk)
            print '****Average input inclination for %d disk stars: %6.2f +- %6.2f' % \
                  (nd,aveIdisk,stdIdisk)
            print '****Average input OMega for %d disk stars: %6.2f +- %6.2f' % \
                  (nd,aveOdisk,stdOdisk)
            py.clf()
            py.figure(figsize=(6,6))
            aa,bb,cc = py.hist(e_input[foo[fidx]], bins=binsIn, histtype='step', normed=True,
                               color='black')
            py.plot([e0, e0],[0,aa.max()+(0.01*aa.max())],'k--')
            py.xlabel('Eccentricity of Disk Stars', fontsize=16)
            #py.ylabel('Probability Density', fontsize=16)
            py.axis([0.0, 1.0, 0.0, aa.max()+(0.01*aa.max())])
            #py.title('Mock Data Input Eccentricities')
            py.savefig(root+alnDir+pdfdir+'plots/mockdata_inputEccDiskStars_%s.png' % suffix)
            #py.savefig(root+alnDir+pdfdir+'plots/eps/mockdata_inputEccDiskStars_%s.eps' % suffix)
            py.close()
        # end temp
  
        # Plot up the original inclinations from the mock data
        binsIn = np.arange(0, 181, 1)
        py.clf()
        py.figure(figsize=(6,6))
        aa,bb,cc = py.hist(i_input[ii], bins=binsIn, histtype='step', normed=True,
                           color='black')
        py.plot([131, 131],[0,aa.max()+(0.01*aa.max())],'k--')
        #py.plot([idisk, idisk],[0,aa.max()+(0.01*aa.max())],'k--')
        py.xlabel('Inclination (deg)', fontsize=16)
        #py.ylabel('Probability Density', fontsize=16)
        py.axis([110, 150, 0.0, aa.max()+(0.01*aa.max())])
        py.savefig(root+alnDir+pdfdir+'plots/mockdata_inputIncl_%s.png' % suffix)
        py.savefig(root+alnDir+pdfdir+'plots/eps/mockdata_inputIncl_%s.eps' % suffix)
        py.close()
  
        # Plot up the original Omegas from the mock data
        binsIn = np.arange(0, 361, 2)
        py.clf()
        py.figure(figsize=(6,6))
        aa,bb,cc = py.hist(O_input[ii], bins=binsIn, histtype='step', normed=True,
                           color='black')
        py.plot([97, 97],[0,aa.max()+(0.01*aa.max())],'k--')
        #py.plot([odisk, odisk],[0,aa.max()+(0.01*aa.max())],'k--')
        py.xlabel('Angle to the Ascending Node (deg)', fontsize=16)
        #py.ylabel('Probability Density', fontsize=16)
        py.axis([60, 130, 0.0, aa.max()+(0.01*aa.max())])
        py.savefig(root+alnDir+pdfdir+'plots/mockdata_inputOmega_%s.png' % suffix)
        py.savefig(root+alnDir+pdfdir+'plots/eps/mockdata_inputOmega_%s.eps' % suffix)
        py.close()
  

    #################
    # Plot distributions of all stars combined
    #################
    
    # Determine the eccentricity probability density function
    eccStep = 0.05
    binsIn = np.arange(0, 1+eccStep, eccStep)
    py.clf()
    py.figure(1)
    py.figure(figsize=(7,6))
    py.subplots_adjust(left=0.15, right=0.95)
    #py.hist(ecc.flatten(), bins=binsIn, histtype='step', normed=True)
    aa,bb,cc = py.hist(ecc[sig,:].flatten(), bins=binsIn, color='r', histtype='step',
                       normed=True, label='Accelerating')
    py.hist(ecc[nonsig,:].flatten(), bins=binsIn, color='b', histtype='step',
            normed=True, label='Non-Accel')
    print np.sum(aa * np.diff(bb)) # trapezoidal integration of the pdf, this should be 1.0
    py.xlabel('Eccentricity', fontsize=16)
    py.ylabel('Probability Density', fontsize=16)
    py.axis([0.0, 1.0, 0.0, aa.max()+0.2])
    #py.title('Mock Data Orbit Simulation')
    py.legend(numpoints=1,fancybox=True)
    py.savefig(root+alnDir+pdfdir+'plots/eccPDF_%s.png' % suffix)
    py.savefig(root+alnDir+pdfdir+'plots/eps/eccPDF_%s.eps' % suffix)
    py.close(1)

    # Simulated stars - separate the accel vs. non-accel stars' disk solutions
    simEccAccDisk = []
    simEccNonAccDisk = []
    evecP_AccDisk = []
    evecQ_AccDisk = []
    evecN_AccDisk = []
    evecP_NonAccDisk = []
    evecQ_NonAccDisk = []
    evecN_NonAccDisk = []
    evecP_allDisk = []
    evecQ_allDisk = []
    evecN_allDisk = []
    for ss in range(nstars):
        sIdx = simWhereInDisk(incl[ss,:], Omega[ss,:]) # disk solutions
        if ss in sig: # sig accel
            simEccAccDisk = np.concatenate([simEccAccDisk, ecc[ss,sIdx]])
            evecP_AccDisk = np.concatenate([evecP_AccDisk, evecP[ss,sIdx]])
            evecQ_AccDisk = np.concatenate([evecQ_AccDisk, evecQ[ss,sIdx]])
            evecN_AccDisk = np.concatenate([evecN_AccDisk, evecN[ss,sIdx]])
        if ss in nonsig: # uniform prior (non-sig accel)
            simEccNonAccDisk = np.concatenate([simEccNonAccDisk, ecc[ss,sIdx]])
            evecP_NonAccDisk = np.concatenate([evecP_NonAccDisk, evecP[ss,sIdx]])
            evecQ_NonAccDisk = np.concatenate([evecQ_NonAccDisk, evecQ[ss,sIdx]])
            evecN_NonAccDisk = np.concatenate([evecN_NonAccDisk, evecN[ss,sIdx]])

        # get evec disk solutions for all the stars as well (accel and non-accel)
        evecP_allDisk = np.concatenate([evecP_allDisk, evecP[ss,sIdx]])
        evecQ_allDisk = np.concatenate([evecQ_allDisk, evecQ[ss,sIdx]])
        evecN_allDisk = np.concatenate([evecN_allDisk, evecN[ss,sIdx]])

    if ((ks == True) | (chi2 == True)):
        # Do a KS test on the accelerating stars ecc distribution and
        # the observed distribution for accelerating disk stars

        #def ks_2samp(data1, data2, n1, n2):
        #    """
        #    Perform a 2 sample KS test. Does not assume that len(data1) = n1,
        #    as scipy's version does. n1 and n2 should be the number of
        #    independent trials, but data1 and data2 may not necessarily all
        #    be independent.
    
        #    Adapted from scipy.stats.ks_2samp()
        #    """
        #    data1 = np.sort(data1)
        #    data2 = np.sort(data2)
        #    data_all = np.concatenate([data1, data2])
        #    cdf1 = np.searchsorted(data1, data_all, side='right')/(1.0*n1)
        #    cdf2 = (np.searchsorted(data2, data_all, side='right'))/(1.0*n2)
        #    d = np.max(np.absolute(cdf1 - cdf2))
        #    # Note: d is absolute value, not signed difference
        #    en = np.sqrt(n1*n2/float(n1+n2))
        #    pdb.set_trace()
        #    try:
        #        prob = scipy.stats.ksprob((en+0.12+0.11/en)*d)
        #    except:
        #        prob = 1.0
        #    return d, prob

        # Read in file listing accelerating sources
        _acc = asciidata.open('%s/%s/tables/accelerating_sources.dat' % (root, alnDir))
        acc = _acc[0].tonumpy()
        accels = [aa.strip() for aa in acc]
        orbDir = 'aorb_thesis/'
        #orbDir = 'aorb_acc_mrPDF_MC_newMosaic/'
        eccAcc = []
        eccAccDisk = []
        eccAccNonDisk = []
        eccNonAcc = []
        eccNonAccDisk = []
        eccNonAccNonDisk = []

        # Disk candidates
        diskTab = asciidata.open(root+alnDir+'tables/disk_membership_prob_mosaic.dat')
        name = [diskTab[0][jj].strip() for jj in range(diskTab.nrows)]
        diskP = diskTab[1].tonumpy()
        diskIdx = (np.where(diskP > 2.7e-3))[0]
        diskCand = [name[nn] for nn in diskIdx]

        print
        #print 'Performing KS test on simulation vs. observed accelerating disk stars:'
        #for ii in range(len(accels)):
        #    name = accels[ii]
        #    orbFile = ('%s/%s/%s/%s.mc.dat' % (root, alnDir, orbDir, name))
        #    if os.path.exists(orbFile) == False:
        #        continue
        #    tmp = pickle.load(open(orbFile))
        #    print 'Accelerating disk star: %s' % name
    #
    #        # Separate out the disk from non-disk solutions
    #        idx = whereInDisk(tmp, angleCut=angleCut)
    #        eD = tmp.e[idx]
    #        nd = np.setdiff1d(np.arange(ntrials), idx)
    #        eND = tmp.e[nd]
    #
    #        # Get the eccentricities 
    #        eccAcc = np.concatenate([eccAcc, tmp.e])
    #
    #        # Separate out just the disk solutions
    #        eccAccDisk = np.concatenate([eccAccDisk, eD]) 
    #
    #        # Non-disk solutions
    #        eccAccNonDisk = np.concatenate([eccAccNonDisk, eND])
#
        for ii in range(len(diskCand)):
            starName = name[ii]
            
            orbFile = ('%s/%s/%s/%s.mc.dat' % (root, alnDir, orbDir, starName))
            if os.path.exists(orbFile) == False:
                continue
            tmp = pickle.load(open(orbFile))
            #print 'Candidate disk member: %s' % starName
    
            # Separate out the disk from non-disk solutions
            idx = whereInDisk(tmp, angleCut=angleCut)
            eD = tmp.e[idx]
            nd = np.setdiff1d(np.arange(ntrials), idx)
            eND = tmp.e[nd]
    
            if starName in accels:
                # Accelerating stars:
                # Get the eccentricities 
                eccAcc = np.concatenate([eccAcc, tmp.e])
        
                # Separate out just the disk solutions
                eccAccDisk = np.concatenate([eccAccDisk, eD]) 
        
                # Non-disk solutions
                eccAccNonDisk = np.concatenate([eccAccNonDisk, eND])
            else:
                # Non-accelerating stars:
                # Get the eccentricities 
                ecNoncAcc = np.concatenate([eccNonAcc, tmp.e])
        
                # Separate out just the disk solutions
                eccNonAccDisk = np.concatenate([eccNonAccDisk, eD]) 
        
                # Non-disk solutions
                eccNonAccNonDisk = np.concatenate([eccNonAccNonDisk, eND])

        # How many fall into each eccentricity bin? for statistics purposes later...
        nObsE, nbb, ncc = py.hist(eccAccDisk, bins=binsIn, color='b', histtype='step',
                                  normed=False)
        nSimE, nbb, ncc = py.hist(simEccAccDisk, bins=binsIn, color='r', histtype='step',
                                  normed=False)
        py.clf()
        py.figure(figsize=(7,6))
        pdfObsE, bb, cc = py.hist(eccAccDisk, bins=binsIn, color='b', histtype='step',
                                  normed=True, label='Observed')
        pdfSimE, bb, cc = py.hist(simEccAccDisk, bins=binsIn, color='r', histtype='step',
                                  normed=True, label='Simulated')
        py.xlabel('Eccentricity', fontsize=16)
        py.ylabel('Probability Density', fontsize=16)
        py.title('Accelerating Disk Members')
        py.axis([0, 1, 0, max(pdfObsE.max(), pdfSimE.max())+0.2])
        py.legend(numpoints=1,fancybox=True)
        py.savefig(root+alnDir+pdfdir+'plots/eccPDF_compare_to_obs.png')
        py.close()

        py.clf()
        py.figure(figsize=(7,6))
        pdfObsEna, bb, cc = py.hist(eccNonAccDisk, bins=binsIn, color='b', histtype='step',
                                  normed=True, label='Observed')
        pdfSimEna, bb, cc = py.hist(simEccNonAccDisk, bins=binsIn, color='r', histtype='step',
                                  normed=True, label='Simulated')
        py.xlabel('Eccentricity', fontsize=16)
        py.ylabel('Probability Density', fontsize=16)
        py.title('Non-accelerating Disk Members')
        py.axis([0, 1, 0, max(pdfObsEna.max(), pdfSimEna.max())+0.2])
        py.legend(numpoints=1,fancybox=True)
        py.savefig(root+alnDir+pdfdir+'plots/eccPDF_compare_to_obs_nonAccel.png')
        py.close()

        # compare the distributions (accelerating stars in simulation vs. accelerating
        # disk stars in real data)
        eccAccDiskSrt = np.sort(eccAccDisk)
        simEccAccDiskSrt = np.sort(simEccAccDisk)
        if ks == True:
            foo = scipy.stats.ks_2samp(eccAccDiskSrt, simEccAccDiskSrt)
            print
            print 'KS Test: D = %5.2f  P = %8.5f' % foo
            print
            #print 'KS Test -- Normalized by number of # stars in each distribution:'
            #n1 = len(eccAccDiskSrt)*1.0
            #n2 = len(simEccAccDiskSrt)*1.0
            #nstars1 = len(accels)*1.0
            #nstars2 = len(sig)*1.0
            ##(dstat, prob) = ks_2samp(eccAccDiskSrt, simEccAccDiskSrt, nstars1, nstars2)
            #oldFactor = np.sqrt((n1*n2) / (n1+n2))
            #newFactor = np.sqrt((nstars1 * nstars2) / (nstars1 + nstars2))
            #dnorm = foo[0] / oldFactor * newFactor
            ##print '		D = %5.2f  P = %8.5f' % (dstat,prob)
            ## Determine the critical value using the approximation for large
            ## sample sizes
            #alpha = 0.005 # 99.5% 
            #calpha = 1.73 # for alpha = 0.005 (99.5% confidence)
            #dcrit = calpha * np.sqrt((nstars1 + nstars2)/(nstars1*nstars2))
            #pdb.set_trace()
            ## compare our D to this critical D
            #if foo[0] > dcrit:
            #    print 'Eccentricity distributions are the same at the alpha=%5.3f level' % alpha
            #else:
            #    print 'Eccentricity distributions are different at the alpha=%5.3f level' % alpha
        elif chi2 == True:
            
            errTot2 = np.ones(len(pdfObsE)) # assuming errors = 1, for now
            echi2_acc = ((pdfObsE - pdfSimE)**2. / (errTot2) ).sum() 

            errTot2 = np.ones(len(pdfObsEna)) # assuming errors = 1, for now
            echi2_nonAcc = ((pdfObsEna - pdfSimEna)**2. / (errTot2) ).sum() 
                                                                      
            print 'Accelerating stars comparison:'
            print '  Chi2 statistic: %8.2f' % echi2_acc
            print
            print 'Non-accelerating stars comparison:'
            print '  Chi2 statistic: %8.2f' % echi2_nonAcc
            #pdb.set_trace()

    print

    usetexTrue()
    # Determine the inclination probability density function
    step = 1.0
    binsIn = np.arange(0, 180+step, step)
    py.clf()
    py.figure(2)
    py.figure(figsize=(7,6))
    py.subplots_adjust(left=0.15, right=0.95)
    py.hist(incl.flatten(), bins=binsIn, histtype='step', normed=True, color='black')
    #py.hist(incl[sig,:].flatten(), bins=binsIn, histtype='step', normed=True)
    py.xlabel('Inclination (deg)', fontsize=16)
    py.ylabel('Probability Density', fontsize=16)
    #py.axis([90, 180, 0, 0.06])
    #py.title('Mock Data Orbit Simulation')
    py.savefig(root+alnDir+pdfdir+'plots/inclPDF_%s.png' % suffix)
    #py.savefig(root+alnDir+pdfdir+'plots/eps/inclPDF_%s.eps' % suffix)
    py.close(2)

    # Determine the Omega probability density function
    step = 1.0
    binsIn = np.arange(0, 360+step, step)
    py.clf()
    py.figure(3)
    py.figure(figsize=(7,6))
    py.subplots_adjust(left=0.15, right=0.95)
    py.hist(Omega.flatten(), bins=binsIn, histtype='step', normed=True, color='black')
    #py.hist(Omega[sig,:].flatten(), bins=binsIn, histtype='step', normed=True)
    py.xlabel('PA to the Ascending Node (deg)', fontsize=16)
    py.ylabel('Probability Density', fontsize=16)
    #py.axis([0, 180, 0, 0.04])
    #py.title('Mock Data Orbit Simulation')
    py.savefig(root+alnDir+pdfdir+'plots/OmegaPDF_%s.png' % suffix)
    #py.savefig(root+alnDir+pdfdir+'plots/eps/OmegaPDF_%s.eps' % suffix)
    py.close(3)

    # Determine the little omega probability density function
    step = 5.0
    binsIn = np.arange(0, 360+step, step)
    py.clf()
    py.figure(4)
    py.figure(figsize=(7,6))
    py.subplots_adjust(left=0.15, right=0.95)
    py.hist(w.flatten(), bins=binsIn, histtype='step', normed=True)
    py.xlabel('Argument of Periapse (deg)')
    py.ylabel('Probability Density')
    py.title('Mock Data Orbit Simulation')
    py.savefig(root+alnDir+pdfdir+'plots/argPeriPDF_%s.png' % suffix)
    py.close(4)

    # Determine the period probability density function
    step = 10.0
    binsIn = np.arange(0, 10**5+step, step)
    py.clf()
    py.figure(5)
    py.figure(figsize=(7,6))
    py.subplots_adjust(left=0.15, right=0.95)
    py.hist(prd.flatten(), bins=binsIn, histtype='step', normed=True)
    py.xlabel('Period (years)')
    py.ylabel('Probability Density')
    py.title('Mock Data Orbit Simulation')
    py.savefig(root+alnDir+pdfdir+'plots/periodPDF_%s.png' % suffix)
    py.close(5)

    # Determine the phase probability density function
    step = 0.01
    binsIn = np.arange(0, 1.0+step, step)
    py.clf()
    py.figure(6)
    py.figure(figsize=(7,6))
    py.subplots_adjust(left=0.15, right=0.95)
    py.hist(phase.flatten(), bins=binsIn, histtype='step', normed=True)
    py.xlabel('Phase')
    py.ylabel('Probability Density')
    py.title('Mock Data Orbit Simulation')
    py.savefig(root+alnDir+pdfdir+'plots/phasePDF_%s.png' % suffix)
    py.close(6)

    # Determine the t0 probability density function
    step = 100.0
    binsIn = np.arange(0, 5000.+step, step)
    py.clf()
    py.figure(7)
    py.figure(figsize=(7,6))
    py.subplots_adjust(left=0.15, right=0.95)
    py.hist(t0.flatten(), bins=binsIn, histtype='step', normed=True)
    py.xlabel('t0')
    py.ylabel('Probability Density')
    py.title('Mock Data Orbit Simulation')
    py.savefig(root+alnDir+pdfdir+'plots/t0PDF_%s.png' % suffix)
    py.close(7)

    # Compare input vs. output values
    py.clf()
    py.figure(8)
    py.figure(figsize=(12,4))
    py.subplots_adjust(left=0.1, right=0.95, wspace=0.3, hspace=0.3,
                       top=0.95, bottom=0.1)
    py.subplot(1,3,1)
    py.hist(e_rms, bins=np.arange(0,1,0.05), histtype='step', normed=True)
    py.xlabel('RMS Eccentricity')
    py.subplot(1,3,2)
    nn, bb, pp = py.hist(i_ave, bins=np.arange(120,150,1), histtype='step', normed=True)
    py.plot([orb_all[0].i,orb_all[0].i], [0,nn.max()+0.05], 'k--')
    py.xlabel('Average Inclination (deg)')
    py.subplot(1,3,3)
    nn, bb, pp = py.hist(o_ave, bins=np.arange(50,150,1), histtype='step', normed=True)
    py.plot([orb_all[0].o,orb_all[0].o], [0,nn.max()+0.05], 'k--')
    py.xlabel('Average Omega (deg)')
    py.savefig(root+alnDir+pdfdir+'plots/mock_vs_average.png')
    py.close(8)
    print 'Median of the accelerating stars RMS eccentricity: %4.2f' % np.median(e_rms[sig])
    print 'Median of the non-accelerating stars RMS eccentricity: %4.2f' % np.median(e_rms[nonsig])

    py.clf()
    py.figure(9)
    py.figure(figsize=(7,6))
    py.errorbar(r2dM, i_ave, yerr=i_std, fmt='r.', label='Incl')
    py.errorbar(r2dM, o_ave, yerr=o_std, fmt='b.', label='Omega')
    py.plot([0,r2dM.max()+0.5], [orb_all[0].i, orb_all[0].i], 'k--')
    py.plot([0,r2dM.max()+0.5], [orb_all[0].o, orb_all[0].o], 'k--')
    py.xlabel('Mock 2D Radius (arcsec)')
    py.ylabel('Average Angle (deg)')
    py.legend(numpoints=1, fancybox=True)
    py.savefig(root+alnDir+pdfdir+'plots/mock_r2d_ave_inclOmega.png')
    py.close(9)

    py.clf()
    py.figure(10)
    py.figure(figsize=(10,5))
    py.subplots_adjust(left=0.1, right=0.95, wspace=0.3, hspace=0.3,
                       top=0.95, bottom=0.1)
    py.subplot(1,2,1)
    py.plot(r2dM, e_med, 'k.')
    py.plot(r2dM[sig], e_med[sig], 'r.')
    py.xlabel('Mock 2D Radius (arcsec)')
    py.ylabel('Median Eccentricity')
    py.subplot(1,2,2)
    py.plot(r2dM, e_rms, 'k.')
    py.plot(r2dM[sig], e_rms[sig], 'r.')
    py.xlabel('Mock 2D Radius (arcsec)')
    py.ylabel('RMS Eccentricity')
    py.savefig(root+alnDir+pdfdir+'plots/mock_r2d_eccentricity.png')
    py.close(10)

    # Determine the period probability density function
    binsIn = np.arange(0.0, 90.0, 1)
    py.clf()
    py.figure(11)
    py.figure(figsize=(7,6))
    py.subplots_adjust(left=0.15, right=0.95)
    py.hist(angOffCW.flatten(), bins=binsIn, histtype='step', normed=True)
    py.xlabel('Angular Distance from Disk (deg)')
    py.ylabel('Probability Density')
    py.savefig(root+alnDir+pdfdir+'plots/angDistance_%s.png' % suffix)
    py.close(11)

    # Eccentricity vector 1D histograms
    py.clf()
    py.figure(12)
    py.figure(figsize=(10,5))
    py.subplots_adjust(left=0.1, right=0.95, wspace=0.3, hspace=0.3,
                       top=0.92, bottom=0.1)
    eccStep = 0.05
    binsIn = np.arange(-1, 1, eccStep)
    py.subplot(1,2,1)
    n1,b1,p1 = py.hist(evecX.flatten(), bins=binsIn, color='k', histtype='step',
                       normed=True, label='X')
    n2,b2,p2 = py.hist(evecY.flatten(), bins=binsIn, color='r', histtype='step',
                       normed=True, label='Y')
    n3,b3,p3 = py.hist(evecZ.flatten(), bins=binsIn, color='b', histtype='step',
                       normed=True, label='Z')
    py.xlabel('Eccentricity Vector', fontsize=16)
    py.ylabel('PDF', fontsize=16)
    py.title('All Sources')
    py.legend(numpoints=1,fancybox=True)
    py.axis([-1.0, 1.0, 0.0, max(n1.max(), n2.max(), n3.max())+0.5])
    py.subplot(1,2,2)
    n1,b1,p1 = py.hist(evecX[sig].flatten(), bins=binsIn, color='k', histtype='step',
                       normed=True)
    n2,b2,p2 = py.hist(evecY[sig].flatten(), bins=binsIn, color='r', histtype='step',
                       normed=True)
    n3,b3,p3 = py.hist(evecZ[sig].flatten(), bins=binsIn, color='b', histtype='step',
                       normed=True)
    py.xlabel('Eccentricity Vector', fontsize=16)
    py.ylabel('PDF', fontsize=16)
    py.title('Accelerating Sources')
    py.axis([-1.0, 1.0, 0.0, max(n1.max(), n2.max(), n3.max())+0.5])
    py.savefig(root+alnDir+pdfdir+'plots/evecXYZ_%s.png' % suffix)
    py.close(12)

    # Eccentricity vector 1D histograms - all solutions
    py.clf()
    py.figure(13)
    py.figure(figsize=(10,5))
    py.subplots_adjust(left=0.1, right=0.95, wspace=0.3, hspace=0.3,
                       top=0.92, bottom=0.1)
    eccStep = 0.05
    binsIn = np.arange(-1, 1, eccStep)
    py.subplot(1,2,1)
    n1,b1,p1 = py.hist(evecP.flatten(), bins=binsIn, color='k', histtype='step',
                       normed=True, label='P')
    n2,b2,p2 = py.hist(evecQ.flatten(), bins=binsIn, color='r', histtype='step',
                       normed=True, label='Q')
    n3,b3,p3 = py.hist(evecN.flatten(), bins=binsIn, color='b', histtype='step',
                       normed=True, label='N')
    py.xlabel('Eccentricity Vector', fontsize=16)
    py.ylabel('PDF', fontsize=16)
    py.title('All Sources')
    py.legend(numpoints=1,fancybox=True)
    py.axis([-1.0, 1.0, 0.0, max(n1.max(), n2.max(), n3.max())+0.5])
    py.subplot(1,2,2)
    n1,b1,p1 = py.hist(evecP[sig].flatten(), bins=binsIn, color='k', histtype='step',
                       normed=True)
    n2,b2,p2 = py.hist(evecQ[sig].flatten(), bins=binsIn, color='r', histtype='step',
                       normed=True)
    n3,b3,p3 = py.hist(evecN[sig].flatten(), bins=binsIn, color='b', histtype='step',
                       normed=True)
    py.xlabel('Eccentricity Vector', fontsize=16)
    py.ylabel('PDF', fontsize=16)
    py.title('Accelerating Sources')
    py.axis([-1.0, 1.0, 0.0, max(n1.max(), n2.max(), n3.max())+0.5])
    py.savefig(root+alnDir+pdfdir+'plots/evecPQN_%s.png' % suffix)
    py.close(13)

    # Eccentricity vector 1D histograms - disk solutions only
    py.clf()
    py.figure(14)
    py.figure(figsize=(10,5))
    py.subplots_adjust(left=0.1, right=0.95, wspace=0.3, hspace=0.3,
                       top=0.92, bottom=0.1)
    eccStep = 0.05
    binsIn = np.arange(-1, 1, eccStep)
    py.subplot(1,2,1)
    n1,b1,p1 = py.hist(evecP_allDisk, bins=binsIn, color='k', histtype='step',
                       normed=True, label='P')
    n2,b2,p2 = py.hist(evecQ_allDisk, bins=binsIn, color='r', histtype='step',
                       normed=True, label='Q')
    n3,b3,p3 = py.hist(evecN_allDisk, bins=binsIn, color='b', histtype='step',
                       normed=True, label='N')
    py.xlabel('Eccentricity Vector', fontsize=16)
    py.ylabel('PDF', fontsize=16)
    py.title('All Sources - Disk Solutions')
    py.legend(numpoints=1,fancybox=True)
    py.axis([-1.0, 1.0, 0.0, max(n1.max(), n2.max(), n3.max())+0.5])
    py.subplot(1,2,2)
    n1,b1,p1 = py.hist(evecP_AccDisk, bins=binsIn, color='k', histtype='step',
                       normed=True)
    n2,b2,p2 = py.hist(evecQ_AccDisk, bins=binsIn, color='r', histtype='step',
                       normed=True)
    n3,b3,p3 = py.hist(evecN_AccDisk, bins=binsIn, color='b', histtype='step',
                       normed=True)
    py.xlabel('Eccentricity Vector', fontsize=16)
    py.ylabel('PDF', fontsize=16)
    py.title('Accelerating Sources - Disk Solutions')
    py.axis([-1.0, 1.0, 0.0, max(n1.max(), n2.max(), n3.max())+0.5])
    py.savefig(root+alnDir+pdfdir+'plots/evecPQN_diskSoln_%s.png' % suffix)
    py.close(14)

    # Determine the eccentricity probability density function
    eccStep = 0.05
    binsIn = np.arange(0, 1+eccStep, eccStep)
    py.clf()
    py.figure(15)
    py.figure(figsize=(7,6))
    py.subplots_adjust(left=0.15, right=0.95)
    aa,bb,cc = py.hist(simEccAccDisk, bins=binsIn, color='k', lw=1.5, histtype='step',
                       normed=True, label='Accel.')
    py.hist(simEccNonAccDisk, bins=binsIn, color='k', ls='dashed', lw=1.5, histtype='step',
            normed=True, label='Non-accel.')
    print np.sum(aa * np.diff(bb)) # trapezoidal integration of the pdf, this should be 1.0
    py.xlabel('Eccentricity', fontsize=16)
    py.ylabel('Probability Density', fontsize=16)
    py.axis([0.0, 1.0, 0.0, aa.max()+0.2])
    #py.title('Mock Data Sim - Disk Solutions')
    py.legend(numpoints=1,fancybox=True)
    py.savefig(root+alnDir+pdfdir+'plots/eccPDF_diskSoln_%s.png' % suffix)
    py.savefig(root+alnDir+pdfdir+'plots/eps/eccPDF_diskSoln_%s.eps' % suffix)
    py.close(15)
    usetexFalse()

    if chi2 == True:
        return echi2_acc, echi2_nonAcc

def best_toy_model(nstars=100, ntrials=10**4,
                   pdfdir='sim_vkick_fracCircVel/vkick_0.07frac/sim_vkick_0.32/',
                   mockfile='ecc_0.32_vkick_mockdata.pickle',
                   sigma=4.0, makeStarPlots=True, suffix='ecc_0.32',
                   sameCoverage=True):

    # Read in the mock data
    # Astrometric units are: arcsec, mas/yr, mas/yr^2
    mockdata = open(root + alnDir + pdfdir + mockfile)
    mbh = pickle.load(mockdata)
    dist = pickle.load(mockdata)
    orb_all = pickle.load(mockdata)
    sma_all = pickle.load(mockdata) # AU
    xM = pickle.load(mockdata) # (+x to east)
    yM = pickle.load(mockdata)
    zM = pickle.load(mockdata)
    vxM = pickle.load(mockdata) # (+x to east)
    vyM = pickle.load(mockdata)
    vzM = pickle.load(mockdata) # mas/yr
    axM = pickle.load(mockdata) # (+x to east)
    ayM = pickle.load(mockdata)
    azM = pickle.load(mockdata)
    t0M = pickle.load(mockdata) # periapse passage
    t_obs = pickle.load(mockdata) # observational time for mock data point
    mockdata.close()

    # Do we want to cover the same field as our observations?
    if sameCoverage == True:
        kp = np.where((xM > -10.) & (xM < 15.) & (yM > -8.) & (yM < 14.))[0]
        orb_all = orb_all[kp]
        sma_all = sma_all[kp]
        xM = xM[kp]
        yM = yM[kp]
        zM = zM[kp]
        vxM = vxM[kp]
        vyM = vyM[kp]
        vzM = vzM[kp]
        axM = axM[kp]
        ayM = ayM[kp]
        azM = azM[kp]
        t0M = t0M[kp]
        nstars = len(kp)
    else:
        kp = np.arange(nstars)

    # Define some variables
    ecc = np.zeros((nstars, ntrials), dtype=float)
    incl = np.zeros((nstars, ntrials), dtype=float)
    Omega = np.zeros((nstars, ntrials), dtype=float)

    p_tmp = np.array([orb.p for orb in orb_all])
    phM = ((t_obs - t0M[:]) % p_tmp) / p_tmp

    r2dM = np.sqrt(xM**2 + yM**2)

    acc = []
    eccSets1 = []
    eccSets2 = []
    eccSets3 = []
    eccSets4 = []
    ecc_all = []
    cnt = 0

    # We know there are 7 accelerating stars in the observed data set; and 32 in
    # this simulation.
    # So create 32 % 7 = 4 subsets for the ecc distribution for error estimate
    arr = np.arange(32)
    np.random.shuffle(arr)
    s1 = arr[0:7]
    s2 = arr[7:14]
    s3 = arr[14:21]
    s4 = arr[21:28]
    
    # Find just the accelerating stars and get their eccentricities
    for ss in range(nstars):
        sidx = kp[ss]
        pdffile = '%s/%s/%s/star%s.mc.dat' % (root, alnDir, pdfdir, sidx)
        pdf = pickle.load(open(pdffile))

        # Read in the orbital parameters from the simulation
        ecc = pdf.e
        asig = pdf.asig # significance of accel (if < -4, accel measurement used)
        # We only want accelerators:
        if asig > -4:
            continue
        
        incl = pdf.i
        Omega = pdf.o # big omega

        sIdx = simWhereInDisk(incl, Omega) # disk solutions
        #simEccAccDisk = []
        #simEccAccDisk = np.concatenate([simEccAccDisk, ecc[sIdx]])

        # For every 7 stars, combine the ecc's into an array
        if ss in s1:
            eccSets1 = np.concatenate([eccSets1, ecc[sIdx]])
        if ss in s2:
            eccSets2 = np.concatenate([eccSets2, ecc[sIdx]])
        if ss in s3:
            eccSets3 = np.concatenate([eccSets3, ecc[sIdx]])
        if ss in s4:
            eccSets4 = np.concatenate([eccSets4, ecc[sIdx]])
        ecc_all = np.concatenate([ecc_all, ecc[sIdx]])

        cnt += 1

    if (len(eccSets1) == 0) | (len(eccSets2) == 0) | (len(eccSets3) == 0) | (len(eccSets4) == 0):
        print '0-sized array'
        pdb.set_trace()

    # Now get the bins in each of the 4 distributions, and take their
    # standard deviation. Use this as the error on each bin in our
    # observed ecc distribution
    eccStep = 0.05
    binsIn = np.arange(0, 1+eccStep, eccStep)
    py.clf()
    py.figure(figsize=(7,6))
    e1, bb, ncc = py.hist(eccSets1, bins=binsIn, color='r', histtype='step',
                              normed=False)
    e2, bb, ncc = py.hist(eccSets2, bins=binsIn, color='g', histtype='step',
                              normed=False)
    e3, bb, ncc = py.hist(eccSets3, bins=binsIn, color='b', histtype='step',
                              normed=False)
    e4, bb, ncc = py.hist(eccSets4, bins=binsIn, color='c', histtype='step',
                              normed=False)
    py.xlabel('Eccentricity', fontsize=16)
    py.title('Best Toy Model - 4 sets', fontsize=16)
    py.axis([0, 1, 0, max(e1.max(),e2.max(),e3.max(),e4.max())+100])
    py.savefig(root+alnDir+pdfdir+'plots/eccPDF_4separate_sets.png')
    py.close()

    bstdv = np.zeros(len(e1), dtype=float)
    for ee in range(len(e1)):
        # get the standard deviation for each ecc bin
        bsig = np.array(np.concatenate([[e1[ee]],[e2[ee]],[e3[ee]],[e4[ee]]]))
        bstdv[ee] = bsig.std()

    pdb.set_trace()

    #numAcc = len(acc)
    ## We need to estimate the errors in our observed and simulated ecc distributions
    ## We have 7 accelerating disk stars in our real data. Create (numAcc/7) sets of eccentricity
    ## distributions so we can estimate the Poisson errors in each bin.
    ## We'll assign these errors to the observed ecc's
    #nSets = numAcc % 7 # We'll lose a few stars if it's not evenly divisible by 7, but that's ok
    ## Randomly select 7 accelerating stars for each nSet
    #np.random.shuffle(acc)
    #acc = np.array([int(aa) for aa in acc])
    #aSets = acc[0:(7*nSets)].reshape(nSets,7)
    #eccSets = np.zeros((nSets, 7), dtype=float)

    #for aa in range(len(aSets)):
    #    idx = aSets[aa]
    #    cnt = 0
    #    for bb in idx:
    #        # Pull the eccentricities for just these stars
    #        pdffile = '%s/%s/%s/star%s.mc.dat' % (root, alnDir, pdfdir, bb)
    #        pdf = pickle.load(open(pdffile))
#
#            # Read in the orbital parameters from the simulation
#            eccSets[aa,:] = pdf.e
#
#            cnt += 1
#
        
        

def simulate_pa0_stars(nstars=500, ntrials=10**5, errorfile='pos_error_vs_radius.dat',
                       sigma=4.0):
    """
    Run a simulation that produces stars with isotropic velocities and a radial
    profile similar to what we observe.  Pulls stars due north of Sgr A*, and runs
    the orbit analysis on these stars alone, as we usually do. Assigns errors
    based on radius (as done in sythesis_sim.py).
    """

    import sythesis_sim as sim

    mockdir = 'sim_isotropic_pa0_stars/'
    outfile = 'isotropic_mockdata.pickle'
    
    # Create the mock data for circular orbits
    #sim.mock_data(nstars=nstars, e=-1, mockdir=mockdir,
    #              outfile=outfile, vel_kick=False)
   
    # Run orbit simulation
    #mc = sim.simulate_orbits(ntrials=ntrials, mockdir=mockdir, mockfile=outfile,
    #                         errorfile=errorfile, sigma=sigma)
    #mc.run(pa0_simulation=True)

    # Read in the mock data
    # Astrometric units are: arcsec, mas/yr, mas/yr^2
    mockdata = open(root + alnDir + mockdir + outfile)
    mbh = pickle.load(mockdata)
    dist = pickle.load(mockdata)
    orb_all = pickle.load(mockdata)
    sma_all = pickle.load(mockdata) # AU
    xM = pickle.load(mockdata) # (+x to east)
    yM = pickle.load(mockdata)
    zM = pickle.load(mockdata)
    vxM = pickle.load(mockdata) # (+x to east)
    vyM = pickle.load(mockdata)
    vzM = pickle.load(mockdata) # mas/yr
    axM = pickle.load(mockdata) # (+x to east)
    ayM = pickle.load(mockdata)
    azM = pickle.load(mockdata)
    t0M = pickle.load(mockdata) # periapse passage
    t_obs = pickle.load(mockdata) # observational time for mock data point
    mockdata.close()

    # Only ran simulation for stars in the patch of sky due north of Sgr A*
    pa0 = np.where((xM > -2.5) & (xM < 2.5) & (yM > 7.0))[0]
    # Redefine the mock data arrays
    orb_all = orb_all[pa0]
    sma_all = sma_all[pa0]
    xM = xM[pa0]
    yM = yM[pa0]
    zM = zM[pa0]
    vxM = vxM[pa0]
    vyM = vyM[pa0]
    vzM = vzM[pa0]
    axM = axM[pa0]
    ayM = ayM[pa0]
    azM = azM[pa0]
    t0M = t0M[pa0]
    r2d = np.sqrt(xM**2 + yM**2)

    nstars = len(pa0)
    print 'Total of %d stars' % nstars

    incl = np.zeros((nstars, ntrials), dtype=float)
    Omega = np.zeros((nstars, ntrials), dtype=float)

    # Plot up the results
    for ss in range(nstars):
        pdffile = '%s/%s/%s/star%s.mc.dat' % (root, alnDir, mockdir, str(ss))
        pdf = pickle.load(open(pdffile))

        # Read in the orbital parameters from the simulation
        incl[ss,:] = pdf.i
        Omega[ss,:] = pdf.o # big omega

        i_in = orb_all[ss].i
        O_in = orb_all[ss].o

        py.clf()
        py.figure(figsize=(10,5))
        py.subplot(1,2,1)
        nn, bb, pp = py.hist(incl[ss,:].flatten(), bins=np.arange(0, 181, 1.0),
                     histtype='step',normed=True)
        py.axis([0, 180, 0, nn.max()+0.01])
        py.xlabel('Inclination (deg)')
        py.title('Mock Incl = %5.1f deg' % i_in, fontsize=12)
        py.subplot(1,2,2)
        nn, bb, pp = py.hist(Omega[ss,:].flatten(), bins=np.arange(0, 370, 10.0),
                     histtype='step',normed=True)
        py.axis([0, 360, 0, nn.max()+0.01])
        py.xlabel('PA to the Ascending Node (deg)')
        py.title('Mock Omega = %5.1f deg' % O_in, fontsize=12)
        py.savefig('%s/%s/%s/plots/star%s_inclOmega.png' % (root,alnDir,mockdir,str(ss)))
        py.close()

    # This plot takes a while, so commenting it out for now:
    #py.clf()
    #py.figure(figsize=(10,5))
    #py.subplot(1,2,1)
    #nn, bb, pp = py.hist(incl.flatten(), bins=np.arange(0, 181, 1.0),
    #             histtype='step',normed=True)
    #py.axis([0, 180, 0, nn.max()+0.01])
    #py.xlabel('Inclination (deg)')
    #py.subplot(1,2,2)
    #nn, bb, pp = py.hist(Omega.flatten(), bins=np.arange(0, 370, 10.0),
    #             histtype='step',normed=True)
    #py.axis([0, 360, 0, nn.max()+0.01])
    #py.xlabel('PA to the Ascending Node (deg)')
    #py.savefig('%s/%s/%s/plots/allStars_inclOmega.png' % (root,alnDir,mockdir))
    #py.close()
    
    py.clf()
    py.figure(figsize=(6,6))
    py.quiver([xM], [yM], [vxM], [vyM], headwidth=1.5, minshaft=1.5,
              color='black', units='y', angles='xy', scale=5)
    py.plot([0],[0],'rx')
    py.axis([15.0, -15.0, -15.0, 15.0])
    py.xlabel('X (arcsec)',fontsize=16)
    py.ylabel('Y (arcsec)',fontsize=16)
    py.title('Mock Data Velocities (PA~0 stars)',fontsize=16)
    py.savefig('%s/%s/%s/plots/velVector_starsUsed_mockdata.png' % (root,alnDir,mockdir))
    py.close()


def run_disk_frac_sims(frac=0.05,numtrials=10,start=1,
                       simRoot='sim_diskFraction4/'):
    import sythesis_sim as sim

    ntot = 120
    for ii in range(numtrials):

        tt = start + ii
        
        ff = np.around(frac, decimals=2) # have to do this b/c sometimes a
        			         # dumb rounding error is made
        ndiskSim = np.int(ff * ntot)
        ff2 = np.around(1-ff,decimals=2) 
        niso = np.int(ff2 * ntot)

        if tt == 1:
            simDir = '%s/disk%s/' % (simRoot, int(ff*100))
        else:
            simDir = '%s/disk%s_%s/' % (simRoot, int(ff*100), int(tt))
        mockFile = 'nDisk%s_nIso%s_mockdata.pickle' % (int(ndiskSim), int(niso))

        gcutil.mkdir(root + alnDir + simDir)
        gcutil.mkdir(root + alnDir + simDir + 'plots/')
        gcutil.mkdir(root + alnDir + simDir + 'plots/HEALpixMaps/')
        gcutil.mkdir(root + alnDir + simDir + 'plots/eps/')

        print 'Running disk fraction simulation: %4.2f, trial #%i' % (ff, tt)
        sim.mock_data(nstars=ndiskSim,mockdir=simDir,outfile=mockFile,isoPop=niso)
        mc = sim.simulate_orbits(mockdir=simDir,mockfile=mockFile)
        mc.run()
        

def disk_fraction_results(simRoot='sim_diskFraction4/',getIOresults=False,
                          getDiskProperties=True,
                          getDensity=False,do_all=False,non_disk=False,
                          do_radial=False,do_r1=False,do_r2=False,do_r3=False,
                          i_in=130.2,O_in=96.3,getAcc=False,signifAcc=5.0,
                          multiSims=False,fraction=None):
    """
    Plot up the results from the disk simulations run with various fractions
    of disk stars (in sim_diskFraction/disk##).
    Input:
    getIOresults: 	Calls simulate_disk() to plot up incl and Omega results
    getDensity: 	Calls sythesis_sim.simDisk() and runs nearest neighbor
    			analysis, creates HEALpixMaps for entire sample if
                        do_all=True, or radial bins if do_r#=True.
    getDiskProperties: 	Calls sythesis_sim.diskMembership() to pull peak angles,
    		     	density, disk radius, and candidate disk members. The
                        density maps must have been created for this to work,
                        so use getDensity=True must have been done before.
    getAcc:		Get the acceleration significance for each simulated star.
    			This takes significantly longer to run (up to 1 hour).
    signifAcc:		Definition of acceleration significance (def = 4 sigma)
    multiSims:		If True, run the code on the 10 simulations in which
    			ff% of the 120 stars were on the disk. This was done to
                        build up a noise map for each disk fraction case.
                        Must specify which disk fraction to run using 'fraction' flag.
    fraction:		Specific disk fraction simulation to use. Only specify if
    			multiSims flag is also set.
    """
    import sythesis_sim as sim
    usetexTrue()

    # Simulations run with 5%-55% disk stars
    if multiSims == True:
        frac = np.ones(10)*fraction
    else:
        frac = np.arange(0.05, 0.59, 0.05)

    ntot = 120

    def getMockData(ndisk, mockFile, returnPos=False):
        # Read in the mock data
        # Astrometric units are: arcsec, mas/yr, mas/yr^2
        mockdata = open(root + alnDir + mockFile)
        mbh = pickle.load(mockdata)
        dist = pickle.load(mockdata)
        orb_all = pickle.load(mockdata)
        sma_all = pickle.load(mockdata) # AU
        xM = pickle.load(mockdata) # (+x to east)
        yM = pickle.load(mockdata)
        zM = pickle.load(mockdata)
        vxM = pickle.load(mockdata) # (+x to east)
        vyM = pickle.load(mockdata)
        vzM = pickle.load(mockdata) # mas/yr
        axM = pickle.load(mockdata) # (+x to east)
        ayM = pickle.load(mockdata)
        azM = pickle.load(mockdata)
        t0M = pickle.load(mockdata) # periapse passage
        t_obs = pickle.load(mockdata) # observational time for mock data point
        mockdata.close()
        r2dM = np.sqrt(xM**2 + yM**2)

        # Get surface density profile
        #rbins = np.arange(0,14,1.0)
        #sd = np.zeros((len(rbins)-1), dtype=float)
        #for rr in range(len(rbins) - 1):
        #    nn = len(np.where((r2dM > rbins[rr]) & (r2dM < rbins[rr+1]))[0])
        #    area = np.pi * (rbins[rr+1]**2 - rbins[rr]**2)
        #    sd[rr] = nn / area

        # Return the number of stars in each radial bin, and the number of
        # disk stars per bin
        rad1 = np.where(r2dM <= 3.197)[0]
        rad2 = np.where((r2dM > 3.197) & (r2dM <= 6.473))[0]
        rad3 = np.where(r2dM > 6.473)[0]

        # Figure out how many disk stars per radial bin
        # based on the total # of disk stars
        d1 = len(np.where(rad1 < ndisk)[0])
        d2 = len(np.where(rad2 < ndisk)[0])
        d3 = len(np.where(rad3 < ndisk)[0])

        #py.close('all')
        #py.clf()
        #py.loglog(rbins[:-1]+0.5, sd, 'k.')
        #py.show()

        if returnPos == True:
            return xM, yM, zM
        else:
            return (d1,d2,d3) # number of disk stars per radial bin

    # real data:
    idiskR = np.zeros(3, dtype=float)
    odiskR = np.zeros(3, dtype=float)
    peakDensityR = np.zeros(3, dtype=float)
    diskRadiusR = np.zeros(3, dtype=float)
    numMembersR = np.zeros(3, dtype=float)
    p1 = np.zeros(3, dtype=float)

#    if getDiskProperties == True:
    # simulated data:
    ndiskSim = np.zeros(len(frac), dtype=float)
    niso = np.zeros(len(frac), dtype=float)
    idiskSim = np.zeros(len(frac), dtype=float) 
    odiskSim = np.zeros(len(frac), dtype=float)
    peakDensitySim = np.zeros(len(frac), dtype=float)
    diskRadiusSim = np.zeros(len(frac), dtype=float)
    posErrSim = np.zeros(len(frac), dtype=float)
    # for numMembersSim, 2D array: 11 sims and 3 membership sigma cuts 
    numMembersSim = np.zeros((len(frac), 3), dtype=float)
    # for radial bin arrays, 2D array: 11 sims and 3 radial bins
    idiskSimR = np.zeros((len(frac),3), dtype=float) 
    odiskSimR = np.zeros((len(frac),3), dtype=float)
    peakDensitySimR = np.zeros((len(frac),3), dtype=float)
    diskRadiusSimR = np.zeros((len(frac),3), dtype=float)
    numMembersSimR = np.zeros((len(frac),3), dtype=float)
    posErrSimR = np.zeros((len(frac),3), dtype=float)
    ndiskSimR = np.zeros((len(frac),3), dtype=float)
    sigAccels = np.zeros(len(frac), dtype=float) # number of sig accels in sim
    miss3sig = np.zeros(len(frac), dtype=float) 
    miss2sig = np.zeros(len(frac), dtype=float) 
    miss1sig = np.zeros(len(frac), dtype=float) 
    fp3sig = np.zeros(len(frac), dtype=float) 
    fp2sig = np.zeros(len(frac), dtype=float) 
    fp1sig = np.zeros(len(frac), dtype=float) 

    # Setup i/Omega for each pixel on sky
    nside = 64
    npix = healpy.nside2npix(nside)
    (iheal, oheal) = healpy.pix2ang(nside, np.arange(0, npix))
    iheal *= rad2deg
    oheal *= rad2deg
    sini = np.sin(iheal / rad2deg)
    cosi = np.cos(iheal / rad2deg)
    sinip = np.sin(np.radians(i_in))
    cosip = np.cos(np.radians(i_in))

    # Determine angular offset to disk for every other point on the sky
    cosodiff = np.cos( (oheal - O_in) / rad2deg )
    angOffCW = np.arccos( (sini * sinip * cosodiff) + (cosi * cosip) )
    angOffCW *= rad2deg

    rFiles = ['inner_','middle_','outer_']
    # Get the observed disk properties for comparison
    (idisk,odisk,peakDensity,diskRadius,numMembers) = \
              diskMembers(file1='disk.neighbor.dat', file2='disk.neighborStd.dat',
                         verbose=False, returnProps=True)
    # Get the observed disk properties in each radial bin 
    for rr in range(3):
        (idiskR[rr],odiskR[rr],peakDensityR[rr],
           diskRadiusR[rr],numMembersR[rr]) = \
                  diskMembers(file1=rFiles[rr]+'disk.neighbor.dat',
                             file2=rFiles[rr]+'disk.neighborStd.dat',
                             verbose=False, returnProps=True)
    
        #ndiskR[rr] = ndR[rr]

    if getAcc == True:
        numAccDiskCand = []
        numAccNonDisk = []
        AccDiskTrue = []
        AccNonDiskTrue = []
        numAccDiskTrue = []
        numAccNonDiskTrue = []

    if multiSims == True:
        d_node_angles = []
        nd_node_angles = []
        d_probs = []
        nd_probs = []
        acc_sig_all_sims = []
        disk_mem_all_sims = []
        rvec_angle_nodes_all_sims = []
        r3d_all_sims = []

    for ii in range(len(frac)):
        print
        ff = np.around(frac[ii], decimals=2) # have to do this b/c sometimes a
        				     # dumb rounding error is made
        ndiskSim[ii] = ff * ntot
        ff2 = np.around(1-ff,decimals=2) 
        niso[ii] = ff2 * ntot

        if multiSims == True:
            if ii == 0:
                simDir = '%s/disk%s/' % (simRoot, int(ff*100))
            else:
                simDir = '%s/disk%s_%d/' % (simRoot, int(ff*100), int(ii+1))
        else:
            simDir = '%s/disk%s/' % (simRoot, int(ff*100))
        mockFile = 'nDisk%s_nIso%s_mockdata.pickle' % (int(ndiskSim[ii]), int(niso[ii]))
        
        plotdir = '%s/%s/%s/plots/' % (root, alnDir, simDir)
        print
        print '*****Working on results in directory: %s' % simDir
        print
        
        if getIOresults == True:
            print 'Getting IO results for disk fraction simulation %4.2f' % ff
            simulate_disk(nDiskStars=ndiskSim[ii], nIsoStars=niso[ii], mockdir=simDir,
                          outfile=mockFile, errorfile='pos_errorRange_vs_radius.dat',
                          vel_kick=True, incl_in=130.2, O_in=96.3, run_sim=False,
                          makeHealPix=True,dumpPickle=True,loadPickle=False)
                          #makeHealPix=False,dumpPickle=True,loadPickle=False)

        if getDensity == True:
            print 'Running nearest-neighbor density analysis for f = %4.2f' % ff
            disk = sim.simDisk(root+alnDir, simDir, mockFile, ntot)
            disk.run(do_all=do_all,non_disk=non_disk, do_radial_bins=do_radial,
                     do_r1=do_r1, do_r2=do_r2, do_r3=do_r3)
            # Recently added:
            # Needs to be run after the above steps are done:
            if do_all == True:
                simFile = 'simdisk.neighbor.dat'
            if non_disk == True:
                simFile = 'non_disk_fov.neighbor.dat'
            if do_r1 == True:
                simFile = 'inner_simdisk.neighbor.dat'
            if do_r2 == True:
                simFile = 'middle_simdisk.neighbor.dat'
            if do_r3 == True:
                simFile = 'outer_simdisk.neighbor.dat'
            pdh.go(root+alnDir+simDir+simFile, 49152, 1)


        if getDiskProperties == True:
            import matplotlib.font_manager
            prop = matplotlib.font_manager.FontProperties(size=12)
            print
            print '****Getting disk properties for disk fraction %4.2f****' % ff
            dM = sim.simDisk(root+alnDir, simDir, mockFile, ntot)
            lhsig = [3.0, 2.0, 1.0]
            for jj in range(3): # loop thru each sigma cut we want
                (idiskSim[ii],odiskSim[ii],peakDensitySim[ii],
                 diskRadiusSim[ii],numMembersSim[ii,jj]) = \
                                   dM.diskMembership(file1='simdisk.neighbor.dat',
                                        file2='simdisk.neighborStd.dat',verbose=False,
                                        LHsigCut=lhsig[jj])

                # Assume the error on the position is the
                # disk radius / sqrt(actual number of members)
                posErrSim[ii] = diskRadiusSim[ii] / ndiskSim[ii]

            xM, yM, zM = getMockData(ndiskSim[ii], simDir+mockFile, returnPos=True)

#            # Get properties for each radial bin
#            # Need to figure out how many DISK stars per radial bin:
#            (d1,d2,d3) = getMockData(ndiskSim[ii], simDir+mockFile)
#            ndR = np.array([d1,d2,d3])
#            for rr in range(3):
#                ndiskSimR[ii,rr] = ndR[rr]
#                dM = sim.simDisk(root+alnDir, simDir, mockFile, ntot)
#                (idiskSimR[ii,rr],odiskSimR[ii,rr],peakDensitySimR[ii,rr],
#                 diskRadiusSimR[ii,rr],numMembersSimR[ii,rr]) = \
#                          dM.diskMembership(file1=rFiles[rr]+'simdisk.neighbor.dat',
#                                            file2=rFiles[rr]+'simdisk.neighborStd.dat',
#                                            verbose=False, radialBin=rFiles[rr],
#                                            LHsigCut=3.0)
#            # Assume the error on the position is the
#            # disk radius / sqrt(actual number of members)
#            posErrSimR[ii,rr] = diskRadiusSimR[ii,rr] / ndiskSimR[ii,rr]

            # Read in the disk membership probabilities
            pfile = '%s/HEALpixMaps/simdisk_membership_prob.dat' % plotdir
            diskTab = asciidata.open(pfile)
            name = [diskTab[0][ss].strip() for ss in range(diskTab.nrows)]
            diskP = diskTab[1].tonumpy()
            diskIdx = (np.where(diskP > 2.7e-3))[0]
            ndIdx = (np.where(diskP <= 2.7e-3))[0]
            #diskIdx = (np.where(diskP > 0.3173))[0]
            #ndIdx = (np.where(diskP <= 0.3173))[0]
            zeroP = (np.where(diskP == 0))[0]
            diskP[zeroP] = 1.e-5 
            # How many actual disk members fall into non-disk category for
            # the 1, 2, and 3 sigma cuts?
            miss3sig[ii] = len(np.where(diskP[0:ndiskSim[ii]] < 0.0027)[0])
            miss2sig[ii] = len(np.where(diskP[0:ndiskSim[ii]] < 0.0455)[0])
            miss1sig[ii] = len(np.where(diskP[0:ndiskSim[ii]] < 0.3173)[0])

            # How many false positive disk members do we get? (non-disk falling
            # into disk category)
            fp3sig[ii] = len(np.where(diskP[ndiskSim[ii]:] > 0.0027)[0])
            fp2sig[ii] = len(np.where(diskP[ndiskSim[ii]:] > 0.0455)[0])
            fp1sig[ii] = len(np.where(diskP[ndiskSim[ii]:] > 0.3173)[0])

            # Now plot a histogram of the values
            usetexTrue()
            py.figure(figsize=(6,6))
            py.subplots_adjust(left=0.13, right=0.96, top=0.95)
            py.clf()
            a1,b1,c1 = py.hist(np.log10(diskP[0:ndiskSim[ii]]), bins=np.arange(-5,0.1,0.1),
                    histtype='step', color='r', linewidth=2, label='Disk')
            a2,b2,c2 = py.hist(np.log10(diskP[ndiskSim[ii]:]), bins=np.arange(-5,0.1,0.1),
                    histtype='step', color='k', ls='dashed', linewidth=2, label='Non-Disk')
            py.plot([-2.57,-2.57],[0,max(a1.max(),a2.max())+2],'k--') # 3 sigma cut (see my notes)
            py.plot([-1.34,-1.34],[0,max(a1.max(),a2.max())+2],'k--') # 2 sigma cut
            py.plot([-0.5,-0.5],[0,max(a1.max(),a2.max())+2],'k--') # 1 sigma cut
            py.xlabel('Log Probability ')
            py.ylabel('Number of Stars')
            py.legend(numpoints=1,loc=1,fancybox=True,prop=prop)
            py.axis([-5.5, 0, 0, max(a1.max(),a2.max())+2])
            py.savefig('%s/disk_membership_hist.png' % (plotdir))
            py.savefig('%s/eps/disk_membership_hist.eps' % (plotdir))
            py.close()
            usetexFalse()

            numTrials = 1e4
            angle = np.zeros((ntot, numTrials), float)
            minAngle = np.zeros(ntot, dtype=float)
            maxAngle = np.zeros(ntot, dtype=float)
            asig = np.zeros(ntot, dtype=float)
            eccDisk = []
            rvec_angle_nodes_all = []
            disk_mem_all = []

            # Loop through each star and find range of angles
            for ss in range(ntot):
                name = 'star%s' % str(ss)
                # File contains analytic orbit solutions with acceleration limits (MC)
                orbFile ='%s/%s/%s/plots/HEALpixMaps/%s_disk_mc_heal.dat' % \
                       (root, alnDir, simDir, name)
        
                pdf = np.fromfile(orbFile, dtype=float)
        
                # Determine the 68.4% confidence region solid angle
                sid = (pdf.argsort())[::-1]  # reverse sort
                peakPix = sid[0]
                pdfSort = pdf[sid]
                
                # Make a cumulative distribution function starting from the
                # highest pixel value. This way we can find the level above
                # which 68% of the trials will fall.
                cdf = np.cumsum(pdfSort)
        
                # Determine point at which we reach 68% level
                idx = (np.where(cdf > 0.6827))[0]
                level = pdfSort[idx[0]] # stars/deg^2
        
                # Calculate angle of every pixel from peak of this star's PDF:
                sinipeak = np.sin(np.radians(iheal[peakPix]))
                cosipeak = np.cos(np.radians(iheal[peakPix]))
                cosodiff = np.cos( (oheal - oheal[peakPix]) / rad2deg )
                angle = np.arccos( (sini * sinipeak * cosodiff) + (cosi * cosipeak) )
                angle *= rad2deg
                
                # Select only the most probable of the two
                # degenerate solutions.
                idx = (np.where((pdf > level) & (angle < 45.0)))[0]
                #print ss
                #print orbFile
                #if ss == 87:
                #    pdb.set_trace()
                if len(idx) == 0:
                    continue
                    
                # Find the range in angles... calc uncertainty from it.
                minAngle[ss] = angOffCW[idx].min()
                maxAngle[ss] = angOffCW[idx].max()
                #yngAngleErr[ss] = (maxAngle[ss] - minAngle[ss]) / 2.0
                #yngAngle[ss] = minAngle[ss] + yngAngleErr[ss]

                # Calculate position angle relative to CW disk's line of nodes
                rvec_angle = np.arctan2(yM[ss],-xM[ss])*(180./np.pi) # angle from (x,y)=(1,0)
                rvec_angle_nodes = rvec_angle + (90. - O_in) # angle relative to Omega
                if np.abs(rvec_angle_nodes) < 90.:
                    rvec_angle_nodes = np.abs(rvec_angle_nodes)
                elif np.abs(rvec_angle_nodes) >= 90.:
                    rvec_angle_nodes = 180.0 - np.abs(rvec_angle_nodes)
                rvec_angle_nodes_all = np.concatenate([rvec_angle_nodes_all, [rvec_angle_nodes]])
                rvec_angle_nodes_all_sims = np.concatenate([rvec_angle_nodes_all_sims, [rvec_angle_nodes]])

                disk_mem_all = np.concatenate([disk_mem_all, [diskP[ss]]])
                disk_mem_all_sims = np.concatenate([disk_mem_all_sims, [diskP[ss]]])

                r3d = np.sqrt(xM[ss]**2 + yM[ss]**2 + zM[ss]**2)
                r3d_all_sims = np.concatenate([r3d_all_sims, [r3d]])

                if getAcc == True:
                    # Also open MC file to see how many stars had significant accels
                    pdffile = '%s/%s/%s/%s.mc.dat' % (root, alnDir, simDir, name)
                    pdf = pickle.load(open(pdffile))
                    asig[ss] = pdf.asig # significance of accel (if < -4, accel measurement used)
                    ecc = pdf.e

                    if ss in diskIdx: # disk member
                        sIdx = simWhereInDisk(pdf.i, pdf.o) # disk solutions
                        if sIdx != None:
                            eccDisk = np.concatenate([eccDisk, ecc[sIdx]])
                            
                    acc_sig_all_sims = np.concatenate([acc_sig_all_sims, [asig[ss]]])

            if getAcc == True:
                #sigAccels[ii] = len(np.where(asig < -signifAcc)[0])
                # Plot acceleration significance
                py.figure(figsize=(6,6))
                py.subplots_adjust(left=0.1, right=0.96, top=0.9)
                py.clf()
                a1,b1,c1 = py.hist(np.abs(asig[diskIdx]), bins=np.arange(0,100,2),
                        histtype='step', color='r', linewidth=2, label='Disk-Candidates')
                a2,b2,c2 = py.hist(np.abs(asig[ndIdx]), bins=np.arange(0,100,2),
                        histtype='step', color='k', ls='dashed', linewidth=2,
                        label='Non-Candidates')
                py.plot([signifAcc,signifAcc],[0,max(a1.max(),a2.max())+2],'k--')
                py.xlabel('Acceleration Significance', fontsize=14)
                py.ylabel('Number of Stars', fontsize=14)
                py.title('Disk Membership Estimated')
                py.legend(numpoints=1,loc=1,fancybox=True,prop=prop)
                py.axis([-0.2, 30, 0, max(a1.max(),a2.max())+2])
                py.savefig('%s/accel_signif_histByMemberPrb.png' % (plotdir))
                py.close()
    
                py.figure(figsize=(6,6))
                py.subplots_adjust(left=0.1, right=0.96, top=0.9)
                py.clf()
                a1,b1,c1 = py.hist(np.abs(asig[0:ndiskSim[ii]]), bins=np.arange(0,100,2),
                        histtype='step', color='r', linewidth=2, label='Disk')
                a2,b2,c2 = py.hist(np.abs(asig[ndiskSim[ii]:]), bins=np.arange(0,100,2),
                        histtype='step', color='k', ls='dashed', linewidth=2,
                        label='Non-Disk')
                py.plot([signifAcc,signifAcc],[0,max(a1.max(),a2.max())+2],'k--')
                py.xlabel('Acceleration Significance', fontsize=14)
                py.ylabel('Number of Stars', fontsize=14)
                py.title('True Disk Status')
                py.legend(numpoints=1,loc=1,fancybox=True,prop=prop)
                py.axis([-0.2, 30, 0, max(a1.max(),a2.max())+2])
                py.savefig('%s/accel_signif_hist.png' % (plotdir))
                py.close()

                # Plot eccentricities for candidate disk members
                usetexTrue()
                py.figure(figsize=(7,6))
                py.subplots_adjust(left=0.15, right=0.95, top=0.9)
                py.clf()
                aa,bb,cc = py.hist(eccDisk.flatten(), bins=np.arange(0,1.05,0.05),
                        histtype='step', color='k', linewidth=1, normed=True)
                py.xlabel('Eccentricity', fontsize=16)
                py.ylabel('Probability Density', fontsize=16)
                py.title('Candidate Disk Members (Disk Solutions)', fontsize=16)
                py.savefig('%s/eccPDF_candidate_members.png' % (plotdir))
                py.close()
                usetexFalse()

                # Keep track of the # of significant accels for candidate disk members
                foo = len(np.where(np.abs(asig[diskIdx]) > signifAcc)[0])
                fooND = len(np.where(np.abs(asig[ndIdx]) > signifAcc)[0])
                numAccDiskCand = np.concatenate([numAccDiskCand, [foo]])
                numAccNonDisk = np.concatenate([numAccNonDisk, [fooND]])
                
                #trueDsig = np.where(np.abs(asig[0:ndiskSim[ii]]) > signifAcc)[0]
                #trueNDsig = np.where(np.abs(asig[ndiskSim[ii]:]) > signifAcc)[0]
                trueDsig = np.where(asig[0:ndiskSim[ii]] < -signifAcc)[0]
                trueNDsig = np.where(asig[ndiskSim[ii]:] < -signifAcc)[0]
                AccDiskTrue = np.concatenate([AccDiskTrue, asig[0:ndiskSim[ii]]])
                AccNonDiskTrue = np.concatenate([AccNonDiskTrue, asig[ndiskSim[ii]:]])
                numAccDiskTrue = np.concatenate([numAccDiskTrue, [len(trueDsig)]])
                numAccNonDiskTrue = np.concatenate([numAccNonDiskTrue, [len(trueNDsig)]])
    
                sigAccels[ii] = len(trueDsig)

            # Plot a histogram of the angular offset
            binsIn = np.arange(0, 185, 5)
            fig = py.figure(figsize=(6,6))
            fig.subplots_adjust(left=0.11, right=0.96, top=0.95)
            fig.clf()
            ax = fig.add_subplot(111)
            a1,b1,c1 = ax.hist(minAngle[0:ndiskSim[ii]], bins=binsIn,
                               histtype='step',color='r', linewidth=2,label='Disk')
            a2,b2,c2 = ax.hist(minAngle[ndiskSim[ii]:], bins=binsIn,
                               histtype='step',color='k', linewidth=2, ls='dashed',
                               label='Non-Disk')
            ax.set_xlabel('Minimum Angular Distance to Disk (deg)')
            ax.set_ylabel('N')
            ax.legend(numpoints=1,prop=prop,fancybox=True)
            ax.axis([0, 180, 0, max(a1.max(),a2.max())+2])
            fig.savefig('%s/yng_minAngle_off_disk_hist.png' % plotdir)
            py.close()

            # Plot angular offset vs. probability
            py.figure(figsize=(6,6))
            py.subplots_adjust(left=0.14, right=0.96, top=0.95)
            py.clf()
            py.semilogx(diskP[0:ndiskSim[ii]],minAngle[0:ndiskSim[ii]],'r.',label='Disk')
            py.semilogx(diskP[ndiskSim[ii]:],minAngle[ndiskSim[ii]:],'k.',label='Non-Disk')
            py.xlabel('Log Probability ', fontsize=14)
            py.ylabel('Minimum Angular Distance to Disk (deg)', fontsize=14)
            py.legend(numpoints=1,fancybox=True,prop=prop,loc=2)
            py.axis([9e-6, 1.1, 0, 180])
            py.savefig('%s/minAngle_vs_diskProb.png' % plotdir)
            py.close()

            # Plot angular offset of radius vector from Line of Nodes vs. probability
            py.figure(figsize=(6,6))
            py.subplots_adjust(left=0.14, right=0.96, top=0.95)
            py.clf()
            py.semilogy(rvec_angle_nodes_all[0:ndiskSim[ii]],disk_mem_all[0:ndiskSim[ii]],'r.',label='Disk')
            py.semilogy(rvec_angle_nodes_all[ndiskSim[ii]:],disk_mem_all[ndiskSim[ii]:],'k.',label='Non-Disk')
            py.xlabel('Position Angle offset from Line of Nodes (deg)', fontsize=14)
            py.ylabel('Log Probability ', fontsize=14)
            py.legend(numpoints=1,fancybox=True,prop=prop,loc=4)
            py.axis([0, 90, 9e-6, 1.1])
            py.savefig('%s/posAngleOffNodes_vs_diskProb.png' % plotdir)
            py.close()

            # save some variables for later plotting, separate for the true disk and non-disk:
            d_node_angles = np.concatenate([d_node_angles, rvec_angle_nodes_all[0:ndiskSim[ii]]])
            nd_node_angles = np.concatenate([nd_node_angles, rvec_angle_nodes_all[ndiskSim[ii]:]])
            d_probs = np.concatenate([d_probs, disk_mem_all[0:ndiskSim[ii]]])
            nd_probs = np.concatenate([nd_probs, disk_mem_all[ndiskSim[ii]:]])

                
    # The typical ranges for the input i,O (after applying velocity kick) are:
    irange = 3.0 # deg
    Orange = 3.9 # deg

    import matplotlib.font_manager
    prop = matplotlib.font_manager.FontProperties(size=12)

    if ((getDiskProperties == True) & (multiSims == False)):
        py.figure(1)
        py.figure(figsize=(8,8))
        py.clf()
        py.subplots_adjust(left=0.15, right=0.96, wspace=0.25,
                           hspace=0.25, top=0.95, bottom=0.07)
        py.subplot(2,2,1)
        py.errorbar(frac, idiskSim, yerr=posErrSim, fmt='k.')
        py.plot([0,0.6], [idisk,idisk], 'k--')
        py.plot([0,0.6], [idisk+irange,idisk+irange], 'k-.')
        py.plot([0,0.6], [idisk-irange,idisk-irange], 'k-.')
        py.xlabel('Fraction of Disk Stars')
        py.ylabel('Peak Inclination (deg)')
        py.subplot(2,2,2)
        py.errorbar(frac, odiskSim, yerr=posErrSim, fmt='k.')
        py.plot([0,0.6], [odisk,odisk], 'k--')
        py.plot([0,0.6], [odisk+Orange,odisk+Orange], 'k-.')
        py.plot([0,0.6], [odisk-Orange,odisk-Orange], 'k-.')
        py.axis([0,0.61,90,110]) # the 5% fraction simulation has a peak Omega ~300 deg
        py.xlabel('Fraction of Disk Stars')
        py.ylabel('Peak Omega (deg)')
        py.subplot(2,2,3)
        py.plot(frac, peakDensitySim, 'k.')
        py.plot([0,0.6], [diskDensity,diskDensity], 'k--')
        py.xlabel('Fraction of Disk Stars')
        py.ylabel(r'Peak Density (stars/deg$^{2}$)')
        py.subplot(2,2,4)
        py.plot(frac, diskRadiusSim, 'k.')
        py.plot([0,0.6], [angleCut,angleCut], 'k--')
        py.xlabel('Fraction of Disk Stars')
        py.ylabel('Disk Radius (deg)')
        py.savefig(root+alnDir+simRoot+'/plots/disk_properties_vs_fraction.png')
        py.savefig(root+alnDir+simRoot+'/plots/eps/disk_properties_vs_fraction.eps')
        py.close(1)

        py.figure(2)
        py.figure(figsize=(5,5))
        #py.figure(figsize=(8,4))
        py.clf()
        py.subplots_adjust(left=0.14, right=0.96, wspace=0.25,
                           hspace=0.25, top=0.95, bottom=0.11)
        #py.subplot(1,2,1)
        #py.plot(ndiskSim, numMembersSim, 'k.')
        #py.plot([40,70],[40,70],'k--')
        #py.xlabel('Input \# of Disk Stars')
        #py.ylabel('Estimated \# Disk Stars')
        #py.subplot(1,2,2)
        fmt = ['ko','r^','gs']
        lbl = [r'3$\sigma$', r'2$\sigma$', r'1$\sigma$']
        #for jj in range(3):
        for jj in range(1):
            py.plot(frac, numMembersSim[:,jj]/ndiskSim, fmt[jj], label=lbl[jj]) 
        py.plot([0,0.6],[1.0,1.0],'k--')
        #py.axis([0,0.61,0,10])
        #py.legend(numpoints=1,fancybox=True,prop=prop)
        py.xlabel('Disk Fraction Simulation')
        py.ylabel('Overestimation of Disk Membership')
        py.savefig(root+alnDir+simRoot+'/plots/disk_membership_vs_fraction_3sig.png')
        py.savefig(root+alnDir+simRoot+'/plots/eps/disk_membership_vs_fraction_3sig.eps')
        py.close(2)

        # Plot the false pos. and false neg. for each fraction
        py.figure(3)
        py.figure(figsize=(10,5))
        py.clf()
        py.subplots_adjust(left=0.08, right=0.96, wspace=0.35,
                           hspace=0.35, top=0.95, bottom=0.15)
        py.subplot(1,2,1)
        py.plot(frac, fp3sig, fmt[0], label=lbl[0]) # false pos (nondisk stars that fell in disk category)
        py.plot(frac, fp2sig, fmt[1], label=lbl[1]) 
        py.plot(frac, fp1sig, fmt[2], label=lbl[2]) 
        py.axis([0,0.61,-1,max(fp3sig.max(),fp2sig.max(),fp1sig.max())+5])
        py.xlabel('Disk Fraction Simulation')
        py.ylabel('N Contaminants')
        py.legend(numpoints=1,fancybox=True,prop=prop)
        py.subplot(1,2,2)
        py.plot(frac, miss3sig/ndiskSim, fmt[0]) # false neg (disk star fell in non-disk category)
        py.plot(frac, miss2sig/ndiskSim, fmt[1])
        py.plot(frac, miss1sig/ndiskSim, fmt[2])
        py.axis([0,0.61,-0.02,0.3])
        py.xlabel('Disk Fraction Simulation')
        py.ylabel('Fraction of Disk Stars Missed')
        py.savefig(root+alnDir+simRoot+'/plots/falsePosNeg_vs_fraction.png')
        py.savefig(root+alnDir+simRoot+'/plots/eps/falsePosNeg_vs_fraction.eps')
        py.close(3)
        

        print
        if getAcc == True:
            fmt = '%4.2f  %6.2f +- %6.2f  %6.2f  %6.2f +- %6.2f  %6.2f  %6i'
            hdr = '%4s  %6s +- %6s  %6s  %6s +- %6s  %6s  %6s'
            print hdr % \
                  ('frac', 'idisk', 'iErr', 'sigOff', 'odisk', 'oErr', 'sigOff', 'accels')

            # also plot the number of significant accels for candidate disk members
            # and non-members for each simulation
            py.clf()
            py.figure(figsize=(6,6))
            py.subplots_adjust(left=0.13, right=0.96, wspace=0.25,
                               hspace=0.25, top=0.95, bottom=0.1)
            py.plot(frac, numAccDiskCand, 'r.', label='Disk-Candidates')
            py.plot(frac, numAccNonDisk, 'k.', label='Non-Members')
            py.axis([0,0.61,0,max(numAccDiskCand.max(), numAccNonDisk.max())+5])
            py.legend(numpoints=1,prop=prop,fancybox=True)
            py.xlabel('Disk Fraction Simulation')
            py.ylabel(r'Number of Significant Accelerations (%d$\sigma$)' % signifAcc)
            py.savefig(root+alnDir+simRoot+'/plots/numAccel_diskMembership_vs_frac.png')

            # plot the number of significant accels for true disk members
            # and true non disk members for each simulation
            py.clf()
            py.figure(figsize=(6,6))
            py.subplots_adjust(left=0.13, right=0.96, wspace=0.25,
                               hspace=0.25, top=0.95, bottom=0.1)
            py.plot(frac, numAccDiskTrue, 'r.', label='Disk')
            py.plot(frac, numAccNonDiskTrue, 'k.', label='Iso')
            py.axis([0,0.61,0,max(numAccDiskCand.max(), numAccNonDisk.max())+5])
            py.legend(numpoints=1,prop=prop,fancybox=True)
            py.title('True Disk Status')
            py.xlabel('Disk Fraction Simulation')
            py.ylabel(r'Number of Significant Accelerations (%d$\sigma$)' % signifAcc)
            py.savefig(root+alnDir+simRoot+'/plots/numAccel_trueDiskStatus_vs_frac.png')
        else:
            fmt = '%4.2f  %6.2f +- %6.2f  %6.2f  %6.2f +- %6.2f  %6.2f'
            hdr = '%4s  %6s +- %6s  %6s  %6s +- %6s  %6s'
            print hdr % \
                  ('frac', 'idisk', 'iErr', 'sigOff', 'odisk', 'oErr', 'sigOff')
        # how many sigma off from the input angle?
        isig = np.abs(idiskSim - idisk) / posErrSim
        osig = np.abs(odiskSim - odisk) / posErrSim
        for ff in range(len(frac)):
            if getAcc == True:
                print fmt % \
                      (frac[ff], idiskSim[ff], posErrSim[ff], isig[ff],
                       odiskSim[ff], posErrSim[ff], osig[ff], sigAccels[ff])
            else:
                print fmt % \
                      (frac[ff], idiskSim[ff], posErrSim[ff], isig[ff],
                       odiskSim[ff], posErrSim[ff], osig[ff])

        # Make the same plots for the 3 radial bins
        py.figure(4)
        py.figure(figsize=(10,10))
        py.subplots_adjust(left=0.1, right=0.96, wspace=0.25,
                           hspace=0.25, top=0.95, bottom=0.07)
        fmt3 = ['r.','g.','b.']
        fmt3l = ['r--','g--','b--']
        # dummy plots for legend
        py.subplot(3,2,1)
        p1 = py.errorbar(frac, frac, fmt=fmt3[0])
        p2 = py.errorbar(frac, frac, fmt=fmt3[1])
        p3 = py.errorbar(frac, frac, fmt=fmt3[2])
        py.clf()
  #      for rr in range(3):
  #          py.subplot(3,2,1)
  #          py.errorbar(frac, idiskSimR[:,rr], yerr=posErrSimR[:,rr], fmt=fmt3[rr])
  #          py.plot([0,0.6], [idiskR[rr],idiskR[rr]], fmt3l[rr])
  #          py.xlabel('Fraction of Disk Stars')
  #          py.ylabel('Peak Inclination (deg)')
  #          py.subplot(3,2,2)
  #          py.errorbar(frac, odiskSimR[:,rr], yerr=posErrSimR[:,rr], fmt=fmt3[rr])
  #          py.plot([0,0.6], [odiskR[rr],odiskR[rr]], fmt3l[rr])
  #          #py.axis([0,0.61,90,110]) # the 5% fraction simulation has a peak Omega ~300 deg
  #          py.xlabel('Fraction of Disk Stars')
  #          py.ylabel('Peak Omega (deg)')
  #          py.subplot(3,2,3)
  #          py.plot(frac, peakDensitySimR[:,rr], fmt3[rr])
  #          py.plot([0,0.6], [peakDensityR[rr],peakDensityR[rr]], fmt3l[rr])
  #          py.xlabel('Fraction of Disk Stars')
  #          py.ylabel(r'Peak Density (stars/deg$^{2}$)')
  #          py.subplot(3,2,4)
  #          py.plot(frac, diskRadiusSimR[:,rr], fmt3[rr])
  #          py.plot([0,0.6], [diskRadiusR[rr],diskRadiusR[rr]], fmt3l[rr])
  #          py.xlabel('Fraction of Disk Stars')
  #          py.ylabel('Disk Radius (deg)')
  #          #py.subplot(3,2,5)
  #          #py.plot(ndiskSimR[:,rr], numMembersSimR[:,rr], fmt3[rr])
  #          ##py.plot([40,70],[40,70],'k--')
  #          #py.xlabel('Input \# of Disk Stars')
  #          #py.ylabel('Candidate Disk Stars')
  #          #py.subplot(3,2,6)
  #          #py.plot(frac, numMembersSimR[:,rr]/ndiskSimR[:,rr], fmt3[rr])
  #          #py.plot([0,0.6],[1.0,1.0],'k--')
  #          #py.axis([0,0.61,0,10])
  #          #py.xlabel('Fraction of Disk Stars')
  #          #py.ylabel('Overestimation of Disk Stars')
  #      py.subplot(3,2,1)
  #      py.legend((p1, p2, p3), ('Inner','Middle','Outer'), numpoints=1,
  #                fancybox=True, loc=4, prop=prop)
  #      py.savefig(root+alnDir+simRoot+\
  #                 '/plots/diskRadial_properties_vs_fraction.png')
  #      py.close(4)

    elif ((getDiskProperties == True) & (multiSims == True)):
        suffix = '_%s_multiSims' % str(fraction)

        py.figure(1)
        py.figure(figsize=(8,8))
        py.clf()
        py.subplots_adjust(left=0.15, right=0.96, wspace=0.25,
                           hspace=0.25, top=0.95, bottom=0.07)
        py.subplot(2,2,1)
        py.hist(idiskSim, bins=np.arange(120,140,1),histtype='step',color='k')
        py.plot([idisk,idisk],[0,10], 'k--')
        py.plot([idisk+irange,idisk+irange], [0,10], 'k-.')
        py.plot([idisk-irange,idisk-irange], [0,10], 'k-.')
        py.axis([125,135,0,3])
        py.xlabel('Peak Inclination (deg)')
        py.ylabel('N')
        py.subplot(2,2,2)
        py.hist(odiskSim, bins=np.arange(90,100,1),histtype='step',color='k')
        py.plot([odisk,odisk], [0,10], 'k--')
        py.plot([odisk+Orange,odisk+Orange], [0,10], 'k-.')
        py.plot([odisk-Orange,odisk-Orange], [0,10], 'k-.')
        py.axis([90,105,0,3.5])
        py.xlabel('Peak Omega (deg)')
        py.ylabel('N')
        py.subplot(2,2,3)
        py.hist(peakDensitySim*1.e3, bins=10,histtype='step',color='k')
        #py.hist(peakDensitySim*1.e3, bins=np.arange(15,55,5),histtype='step',color='k')
        py.plot([diskDensity*1.e3,diskDensity*1.e3],[0,10], 'k--')
        #py.axis([0,55,0,3.5])
        py.xlabel(r'Peak Density ($\times$10$^{-3}$ stars/deg$^{2}$)')
        py.ylabel('N')
        py.subplot(2,2,4)
        py.hist(diskRadiusSim,bins=10,histtype='step',color='k')
        #py.hist(diskRadiusSim,bins=np.arange(7,18,1),histtype='step',color='k')
        py.plot([angleCut,angleCut],[0,10], 'k--')
        py.axis([6,18,0,2.5])
        py.xlabel('Disk Radius (deg)')
        py.ylabel('N')
        py.savefig(root+alnDir+simRoot+'/plots/disk_properties_vs_fraction%s.png' % suffix)
        py.savefig(root+alnDir+simRoot+'/plots/eps/disk_properties_vs_fraction%s.eps' % suffix)
        py.close(1)

        py.figure(2)
        py.figure(figsize=(5,5))
        py.clf()
        py.subplots_adjust(left=0.14, right=0.96, wspace=0.25,
                           hspace=0.25, top=0.95, bottom=0.11)
        clr = ['k','r','g']
        lbl = [r'3$\sigma$', r'2$\sigma$', r'1$\sigma$']
        for jj in range(3):
            #py.hist(numMembersSim[:,jj]/ndiskSim, bins=np.arange(1,3.5,0.1),
            py.hist(numMembersSim[:,jj]/ndiskSim, bins=10,
                    histtype='step',color=clr[jj], label=lbl[jj]) 
        py.axis([1,3.5,0,7])
        py.legend(numpoints=1,fancybox=True,prop=prop)
        py.xlabel('Overestimation of Disk Membership')
        py.ylabel('N')
        py.savefig(root+alnDir+simRoot+'/plots/disk_membership_vs_fraction%s.png' % suffix)
        py.savefig(root+alnDir+simRoot+'/plots/eps/disk_membership_vs_fraction%s.eps' % suffix)
        py.close(2)

        # Plot the false pos. and false neg. for each fraction
        py.figure(3)
        py.figure(figsize=(10,5))
        py.clf()
        py.subplots_adjust(left=0.08, right=0.96, wspace=0.35,
                           hspace=0.35, top=0.95, bottom=0.15)
        py.subplot(1,2,1)
        #py.hist(fp3sig,bins=np.arange(15,50,1),histtype='step',color=clr[0],label=lbl[0]) # false pos (nondisk stars that fell in disk category)
        #py.hist(fp2sig,bins=np.arange(5,35,1),histtype='step',color=clr[1],label=lbl[1])
        #py.hist(fp1sig,bins=np.arange(0,15,1),histtype='step',color=clr[2],label=lbl[2])
        py.hist(fp3sig,bins=10,histtype='step',color=clr[0],label=lbl[0]) # false pos (nondisk stars that fell in disk category)
        py.hist(fp2sig,bins=10,histtype='step',color=clr[1],label=lbl[1])
        py.hist(fp1sig,bins=10,histtype='step',color=clr[2],label=lbl[2])
        py.axis([0,14,0,5])
        py.xlabel('N Contaminants')
        py.ylabel('N')
        py.legend(numpoints=1,fancybox=True,prop=prop)
        py.subplot(1,2,2)
        py.hist(miss3sig/ndiskSim,bins=10,histtype='step',color=clr[0]) # false neg (disk star fell in non-disk category)
        py.hist(miss2sig/ndiskSim,bins=10,histtype='step',color=clr[1])
        py.hist(miss1sig/ndiskSim,bins=10,histtype='step',color=clr[2])
        py.axis([0,0.3,0,15])
        py.xlabel('Fraction of Disk Stars Missed')
        py.ylabel('N')
        py.savefig(root+alnDir+simRoot+'/plots/falsePosNeg_vs_fraction%s.png' % suffix)
        py.savefig(root+alnDir+simRoot+'/plots/eps/falsePosNeg_vs_fraction%s.eps' % suffix)
        py.close(3)
        
        # From the fit to the sum_of_squared_diffs data, the best fit disk fraction is 21%
        # If we restrict candidates to just the top 20%, how many contaminants are there?
        all_probs = np.concatenate([d_probs, nd_probs])
        sidx = all_probs.argsort()[::-1][0:np.int(fraction*120.*10.)] # reverse sort
        diskPsrt = all_probs[sidx]
        # in this new array, the non-disk stars start at:
        nd_idx = len(d_probs)
        contam = np.where(sidx > nd_idx)[0]
        print '**********************************'
        print 'Selecting the top 20% of candidates:'
        if len(contam) > 0:
            print '20th percentile disk membership probability: %6.3f' % diskPsrt[-1]
            print ' %s true disk members correctly identified (should be 240)' % str(240-len(contam))
            print ' %i non-disk members incorrectly identified as disk members (should be 0)' % len(contam)

        zeroprob = np.where(nd_probs == 1.e-5)[0]
        print
        print 'Number of simulated stars on disk: %i' % len(d_probs)
        print 'Number of simulated stars not on disk: %i' % len(nd_probs) 
        print 'Number of non-disk stars with zero disk membership probability: %i' % len(zeroprob)
        someprob = np.where(np.log(nd_probs) > -2.57)[0]
        print 'Number of non-disk stars mistakenly IDed as disk stars: %i' % len(someprob)
            
        # Plot combined histogram of disk membership probability
        py.figure(figsize=(6,6))
        py.subplots_adjust(left=0.13, right=0.96, top=0.95)
        py.clf()
        a1,b1,c1 = py.hist(np.log10(d_probs), bins=np.arange(-5,0.1,0.1),
                histtype='step', color='r', lw=2, label='Disk')
        a2,b2,c2 = py.hist(np.log10(nd_probs), bins=np.arange(-5,0.1,0.1),
                histtype='step', color='k', ls='dashed', linewidth=2, label='Non-Disk')
        py.plot([-2.57,-2.57],[0,max(a1.max(),a2.max())+2],'b:',lw=2) # 3 sigma cut (see my notes)
        #py.plot([np.log10(diskPsrt[-1]),np.log10(diskPsrt[-1])],
        #        [0,max(a1.max(),a2.max())+2],'g-.', lw=2) # 20th percentile
        #py.text(np.log10(diskPsrt[-1])-0.55,110,r'Top 20\%',color='g',fontsize=12)
        #py.arrow(np.log10(diskPsrt[-1])-0.4,108,0.2,0.0,head_width=1.25,
        #         head_length=0.15,color='g')
        py.xlabel('Log Disk Membership Probability ')
        py.ylabel('N Simulated Stars')
        py.legend(numpoints=1,loc=2,fancybox=True,prop=prop)
        py.axis([-4.2, 0, 0, a1.max()+30])
        #py.axis([-5.5, 0, 0, max(a1.max(),a2.max())+2])
        py.savefig('%s/%s/%s/plots/disk_membership_hist_%smultiSims.png' % \
                   (root,alnDir,simRoot,str(fraction)))
        py.savefig('%s/%s/%s/plots/eps/disk_membership_hist_%smultiSims.eps' % \
                   (root,alnDir,simRoot,str(fraction)))
        py.close()

        # For a given bin in disk membership probability, how many true and fake
        # disk members are there?
        pbins = b1
        # True disk members per bin:
        ntrue = a1
        # Non-disk members per bin:
        nfake = a2
        # Write these out to a file
        hdr = '%8s  %6s  %6s\n'
        fmt = '%8.1f  %6i  %6i\n'
        pout = open('%s/%s/%s/plots/disk_membership_hist_%smultiSims.txt' % \
                    (root,alnDir,simRoot,str(fraction)),'w')
        pout.write(hdr % ('Prob Bin', 'N Disk', 'N Iso'))
        for ii in range(len(pbins)-1):
            pout.write(fmt % (pbins[ii],ntrue[ii],nfake[ii]))
        pout.close()

        # Plot average angular offset of radius vector from Line of Nodes in
        # 10-degree bins vs. probability
        d_node_bin = []
        d_probs_bin = []
        nbin = np.arange(0,90,10)
        prob_bin_med = np.zeros(len(nbin),dtype=float)
        prob_bin_ave = np.zeros(len(nbin),dtype=float)
        prob_bin_rms = np.zeros(len(nbin),dtype=float)
        for ss in range(len(AccDiskTrue)):
            if AccDiskTrue[ss] < -5.0:
                continue
            else:
                d_node_bin = np.concatenate([d_node_bin,[d_node_angles[ss]]])
                d_probs_bin = np.concatenate([d_probs_bin,[d_probs[ss]]])
        for bb in range(len(nbin)):
            bidx = np.where((d_node_bin > nbin[bb]) & (d_node_bin < nbin[bb]+10.))[0]
            prob_bin_med[bb] = np.median(d_probs_bin[bidx])
            prob_bin_ave[bb] = d_probs_bin[bidx].mean()
            prob_bin_rms[bb] = d_probs_bin[bidx].std(ddof=1)
        py.figure(figsize=(6,6))
        py.subplots_adjust(left=0.14, right=0.96, top=0.95)
        py.clf()
        #py.errorbar(nbin+5.0, prob_bin_med, yerr=prob_bin_rms, fmt='k.')
        py.errorbar(nbin+5.0, prob_bin_ave, yerr=prob_bin_rms, fmt='k.')
        py.xlabel('Position Angle offset from Line of Nodes (deg)', fontsize=14)
        py.ylabel('Average Disk Membership Probability ', fontsize=14)
        py.axis([0, 90, 9e-6, 1.1])
        py.savefig('%s/%s/%s/plots/ave_posAngleOffNodes_vs_diskProb_%smultiSims_linear.png' % \
                   (root,alnDir,simRoot,str(fraction)))
        py.savefig('%s/%s/%s/plots/eps/ave_posAngleOffNodes_vs_diskProb_%smultiSims_linear.eps' % \
                   (root,alnDir,simRoot,str(fraction)))
        py.close()
        # Write the averages to a file
        outfile = '%s/%s/%s/plots/med_posAngleOffNodes_vs_diskProb_%smultiSims.txt' % \
                   (root,alnDir,simRoot,str(fraction))
        foo = open(outfile,'w')
        hdr = '%3s  %4s  %4s\n'
        foo.write(hdr % ('#Bin','Med','RMS'))
        fmt = '%3i  %4.2f  %4.2f\n'
        for ff in range(len(nbin)):
            foo.write(fmt % (int(nbin[ff]+5), prob_bin_med[ff], prob_bin_rms[ff]))
        foo.close()

        print
        # What sigma level gives equal number of contaminants and missed disk stars?
        for kk in np.arange(1.0,3.75,0.25):
            nmiss = np.zeros(len(nbin),dtype=int) 
            ncont = np.zeros(len(nbin),dtype=int)
            for bb in range(len(nbin)):
                didx = np.where((d_node_angles > nbin[bb]) & (d_node_angles < nbin[bb]+10.))[0]
                nmiss[bb] = len(np.where(d_probs[didx] < (prob_bin_med[bb]-kk*prob_bin_rms[bb]))[0])
                ndidx = np.where((nd_node_angles > nbin[bb]) & (nd_node_angles < nbin[bb]+10.))[0]
                ncont[bb] = len(np.where(nd_probs[ndidx] > (prob_bin_med[bb]-kk*prob_bin_rms[bb]))[0])
            print 'For %4.2f sigma: %i disk stars missed, %i contaminants included' % \
                  (kk, nmiss.sum(), ncont.sum())
        print
        usetexTrue()
        # Plot angular offset of radius vector from Line of Nodes vs. probability
        py.figure(figsize=(6,6))
        py.subplots_adjust(left=0.14, right=0.96, top=0.9)
        py.clf()
        py.plot(d_node_angles, d_probs, 'ro', marker='o', mfc='None', mec='r',
                    ms=8, mew=1.5,label='Disk')
        py.plot(nd_node_angles, nd_probs, 'ks', marker='s', mfc='None', mec='k',
                    mew=1.5, label='Non-Disk')
        for ss in range(len(acc_sig_all_sims)):
            if acc_sig_all_sims[ss] < -5.0:
                #print 'testing %8.3f, %8.3f, %8.3f' % \
                #      (acc_sig_all_sims[ss],rvec_angle_nodes_all_sims[ss],disk_mem_all_sims[ss])
                py.plot(rvec_angle_nodes_all_sims[ss], disk_mem_all_sims[ss], 'kx', mew=1.5)
        #py.plot(nbin+5.0, prob_bin_med-1.*prob_bin_rms, 'g-',lw=2)
        #py.plot(nbin+5.0, prob_bin_med-2.*prob_bin_rms, 'g-',lw=2)
        py.plot(nbin+5.0, prob_bin_med, 'g-',lw=2)
        py.plot(nbin+5.0, prob_bin_med-3*prob_bin_rms, 'g--',lw=2)
        py.title('Mock Data')
        py.xlabel('Position Angle from Line of Nodes (deg)')
        py.ylabel('Disk Membership Probability ')
        py.legend(numpoints=1,fancybox=True,prop=prop,loc=1)
        #py.axis([0, 90, 0.1, 1.0])
        py.axis([0, 90, 9e-6, 1.1])
        py.savefig('%s/%s/%s/plots/posAngleOffNodes_vs_diskProb_%smultiSims_linear.png' % \
                   (root,alnDir,simRoot,str(fraction)))
        py.savefig('%s/%s/%s/plots/eps/posAngleOffNodes_vs_diskProb_%smultiSims_linear.eps' % \
                   (root,alnDir,simRoot,str(fraction)))
        py.close()

        py.figure(figsize=(6,6))
        py.subplots_adjust(left=0.14, right=0.96, top=0.96)
        py.clf()
        for ss in range(len(acc_sig_all_sims)):
            py.plot(r3d_all_sims[ss],np.abs(acc_sig_all_sims[ss]),'k.',label='Non-Disk')
        py.plot([0,25],[0,0],'k--')
        py.axis([0,25,-2,45])
        #py.legend(numpoints=1,fancybox=True,loc=1,prop=prop)
        py.xlabel('3D Radius (arcsec)')
        py.ylabel(r'$|$Acceleration$|$ ($\sigma$)')
        py.savefig('%s/%s/%s/plots/r3d_vs_accel_%smultiSims.png' % \
                   (root,alnDir,simRoot,str(fraction)))
        py.close()

        print
        if getAcc == True:
            fmt = '%4.2f  %6.2f +- %6.2f  %6.2f  %6.2f +- %6.2f  %6.2f  %6i'
            hdr = '%4s  %6s +- %6s  %6s  %6s +- %6s  %6s  %6s'
            print hdr % \
                  ('frac', 'idisk', 'iErr', 'sigOff', 'odisk', 'oErr', 'sigOff', 'accels')

            # also plot the number of significant accels for candidate disk members
            # and non-members for each simulation
            py.clf()
            py.figure(figsize=(6,6))
            py.subplots_adjust(left=0.13, right=0.96, wspace=0.25,
                               hspace=0.25, top=0.95, bottom=0.1)
            py.hist(numAccDiskCand, bins=np.arange(0,12,1), histtype='step',
                    color='r', label='Disk-Candidates')
            py.hist(numAccNonDisk, bins=np.arange(0,12,1), histtype='step',
                     color='k', label='Non-Members')
            py.axis([0,12,0,5])
            py.legend(numpoints=1,loc=2,prop=prop,fancybox=True)
            py.xlabel(r'Number of Significant Accelerations (%d$\sigma$)' % signifAcc)
            py.ylabel('N')
            py.savefig(root+alnDir+simRoot+'/plots/numAccel_diskMembership_vs_frac%s.png' % suffix)

            # plot the number of significant accels for true disk members
            # and true non disk members for each simulation
            py.clf()
            py.figure(figsize=(6,6))
            py.subplots_adjust(left=0.13, right=0.96, wspace=0.25,
                               hspace=0.25, top=0.93, bottom=0.1)
            py.hist(numAccDiskTrue, bins=np.arange(0,14,1), histtype='step',
                     color='r', label='Disk')
            py.hist(numAccNonDiskTrue, bins=np.arange(0,14,1), histtype='step',
                     color= 'k', label='Iso')
            py.axis([0,14,0,4])
            py.legend(numpoints=1,prop=prop,fancybox=True)
            py.title('True Disk Status')
            py.xlabel(r'Number of Significant Accelerations (%d$\sigma$)' % signifAcc)
            py.ylabel('N')
            py.savefig(root+alnDir+simRoot+'/plots/numAccel_trueDiskStatus_vs_frac%s.png' % suffix)
            py.close()
        else:
            fmt = '%4.2f  %6.2f +- %6.2f  %6.2f  %6.2f +- %6.2f  %6.2f'
            hdr = '%4s  %6s +- %6s  %6s  %6s +- %6s  %6s'
            print hdr % \
                  ('frac', 'idisk', 'iErr', 'sigOff', 'odisk', 'oErr', 'sigOff')
        # how many sigma off from the input angle?
        isig = np.abs(idiskSim - idisk) / posErrSim
        osig = np.abs(odiskSim - odisk) / posErrSim
        for ff in range(len(frac)):
            if getAcc == True:
                print fmt % \
                      (frac[ff], idiskSim[ff], posErrSim[ff], isig[ff],
                       odiskSim[ff], posErrSim[ff], osig[ff], sigAccels[ff])
            else:
                print fmt % \
                      (frac[ff], idiskSim[ff], posErrSim[ff], isig[ff],
                       odiskSim[ff], posErrSim[ff], osig[ff])
        # Make the same plots for the 3 radial bins
        py.figure(4)
        py.figure(figsize=(8,8))
        py.subplots_adjust(left=0.1, right=0.96, wspace=0.25,
                           hspace=0.25, top=0.95, bottom=0.07)
#        py.clf()
#        clr3 = ['r','g','b']
#        fmt3l = ['r--','g--','b--']
#        # dummy plots for legend
#        py.subplot(2,2,1)
#        p1 = py.errorbar(frac, frac, fmt='r-')
#        p2 = py.errorbar(frac, frac, fmt='g-')
#        p3 = py.errorbar(frac, frac, fmt='b-')
#        py.clf()
#        for rr in range(3):
#            py.subplot(2,2,1)
#            py.hist(idiskSimR[:,rr],bins=np.arange(0,180,3),histtype='step',color=clr3[rr])
#            py.plot([idiskR[rr],idiskR[rr]], [0,10], fmt3l[rr])
#            py.axis([0,180,0,5])
#            py.xlabel('Peak Inclination (deg)')
#            py.ylabel('N')
#            py.subplot(2,2,2)
#            py.hist(odiskSimR[:,rr], bins=np.arange(50,300,5),histtype='step', color=clr3[rr])
#            py.plot([odiskR[rr],odiskR[rr]], [0,10], fmt3l[rr])
#            py.axis([50,360,0,7])
#            py.xlabel('Peak Omega (deg)')
#            py.ylabel('N')
#            py.subplot(2,2,3)
#            py.hist(peakDensitySimR[:,rr]*1.e3, bins=np.arange(0,25,1),
#                    histtype='step', color=clr3[rr])
#            py.plot([peakDensityR[rr]*1.e3,peakDensityR[rr]*1.e3],[0,10], fmt3l[rr])
#            py.axis([0,25,0,5])
#            py.xlabel(r'Peak Density ($\times$10$^{-3}$ stars/deg$^{2}$)')
#            py.ylabel('N')
#            py.subplot(2,2,4)
#            py.hist(diskRadiusSimR[:,rr],bins=np.arange(0,100,5),histtype='step',color=clr3[rr])
#            py.plot([diskRadiusR[rr],diskRadiusR[rr]], [0,10], fmt3l[rr])
#            py.axis([0,100,0,5])
#            py.xlabel('Disk Radius (deg)')
#            py.ylabel('N')
#        py.subplot(2,2,1)
#        py.legend((p1, p2, p3), ('Inner','Middle','Outer'), numpoints=1,
#                  fancybox=True, loc=2, prop=prop)
#        py.savefig(root+alnDir+simRoot+\
#                   '/plots/diskRadial_properties_vs_fraction%s.png' % suffix)
#        py.close(4)
#        
        usetexFalse()


def disk_fraction_sumSquaredDiff(mockdir='sim_diskFraction4/',ntrials=10):
    """
    For each disk fraction model, compute the sum of the squared difference between
    the density maps from the model and the observations, where:
    	SS_diff = Sum_pix(model - observed)**2
          Sum_pix = sum of the squared differences at each pixel in density map
        
    Input:
       ntrials (int):  Number of times each disk fraction case was run.
    """
    import sythesis_sim as sim

    frac = np.arange(0.05, 0.59, 0.05)
    #frac = np.arange(0.15, 0.29, 0.05)

    ntot = 120
    niso = np.zeros(len(frac), dtype=float)
    ndiskSim = np.zeros(len(frac), dtype=float)
    numMembersSim = np.zeros((len(frac), 10), dtype=float)
    idiskSim = np.zeros((len(frac), 10), dtype=float) 
    odiskSim = np.zeros((len(frac), 10), dtype=float)
    peakDensitySim = np.zeros((len(frac), 10), dtype=float)
    diskRadiusSim = np.zeros((len(frac), 10), dtype=float)

    simDir = root + alnDir + mockdir

    # Get the observed density map for comparison to simulations
    orbDir = 'aorb_thesis/'
    nside = 64
    npix = healpy.nside2npix(nside)
    print npix
    pixIdx = np.arange(0, npix)
    (disk, diskStd) = loadDiskDensity(npix, orbDir=orbDir,
                                      file1='disk.neighbor.dat',
                                      file2='disk.neighborStd.dat')
    (i, o) = healpy.pix2ang(nside, pixIdx)
    i *= 180.0 / math.pi
    o *= 180.0 / math.pi

    didx = disk.argmax()
    peak = disk[didx]
    idisk = i[didx]
    odisk = o[didx]
    print 'Peak found at (i,O) = (%5.1f, %5.1f) with %.2e  stars/deg^2' % \
          (idisk,odisk,peak)

    sini = np.sin(np.radians(idisk))
    cosi = np.cos(np.radians(idisk))
    sino = np.sin(np.radians(odisk))
    coso = np.cos(np.radians(odisk))

    angle = np.zeros(len(i), float)

    sum_diff_sq = np.zeros((len(frac), ntrials), dtype=float)

    for ii in range(len(frac)):
        print
        ff = np.around(frac[ii], decimals=2) # have to do this b/c sometimes a
        				     # dumb rounding error is made
        file1 = 'simdisk.neighbor.dat' 
        file2 = 'simdisk.neighborStd.dat'

        ndiskSim[ii] = ff * ntot
        ff2 = np.around(1-ff,decimals=2) 
        niso[ii] = ff2 * ntot
    
        for nn in range(ntrials):
            if nn == 0:
                fracdir = 'disk%s/' % int(ff*100)
                orbDir = '%s/%s/' % (simDir, fracdir)
            else:
                fracdir = 'disk%s_%s/' % (int(ff*100), str(nn+1))
                orbDir = '%s/%s/' % (simDir, fracdir)
            # Load density map for this disk fraction model
            (diskS, diskStdS) = loadDiskDensity(npix, orbDir=orbDir,
                                                file1=file1, file2=file2,
                                                singlePdf=True, simdisk=True,
                                                silent=True)
        
            # Calculate angle of every pixel from peak of the observed PDF
            for k in range(len(i)):
                inc = np.radians(i[k])
                ome = np.radians(o[k])
          
                angle[k] = np.arccos(sini * coso * np.sin(inc) * np.cos(ome) + \
                                  sini * sino * np.sin(inc) * np.sin(ome) + \
                                  cosi * np.cos(inc))
                angle[k] = np.degrees(angle[k])
         
            aid = (np.where(angle < 30.0))[0]
        
            sum_diff_sq[ii,nn] = ((diskS[aid] - disk[aid])**2).sum()
            print 'Trial %i Disk Fraction %s: SS_diff = %5.3f' % \
                  (nn+1, frac[ii], sum_diff_sq[ii,nn])

            # Let's also get the over-estimation of disk membership
            mockFile = 'nDisk%s_nIso%s_mockdata.pickle' % \
                       (int(ndiskSim[ii]), int(niso[ii]))
            dM = sim.simDisk(root+alnDir, mockdir+fracdir, mockFile, ntot)
            (idiskSim[ii,nn],odiskSim[ii,nn],peakDensitySim[ii,nn],
             diskRadiusSim[ii,nn],numMembersSim[ii,nn]) = \
                               dM.diskMembership(file1=file1, file2=file2,
                                                 verbose=False)

    print
    print 'Sum of Squared Diffs per trial:'
    hdr = '%5s  %5s  %5s'
    fmt = '%5i  %5.3f  %5.2f'
    print hdr % ('Trial','Min.','Frac.')
    clr = ['black', 'red','green','blue', 'orange', 'purple', 'mediumblue',
           'steelblue', 'teal', 'maroon', 'lightsalmon','darkslateblue',
           'greenyellow', 'gold', 'darkorange','orangered']
    usetexTrue()
    py.figure(1)
    py.figure(figsize=(6,6))
    py.clf()
    py.subplots_adjust(left=0.17, right=0.96, wspace=0.25,
                       hspace=0.25, top=0.95, bottom=0.11)
    for nn in range(ntrials):
        midx = sum_diff_sq[:,nn].argmin()
        for ff in range(len(frac)):
            py.semilogy(frac[ff], sum_diff_sq[ff,nn], color=clr[nn], marker='o',
                        alpha=(1.-nn*0.09))
        print fmt % (nn+1, sum_diff_sq[:,nn].min(), frac[midx])
    #py.axis([0,0.602,0.08,1])
    py.xlabel(r'{\bf \huge{$f_{disk}$}}')
    py.ylabel(r'{\bf \huge{$\xi$}}')
    py.savefig('%s/plots/rms_diff_vs_frac_%strials.png' % (simDir,str(ntrials)))
    py.savefig('%s/plots/eps/rms_diff_vs_frac_%strials.eps' % (simDir,str(ntrials)))
    py.close(1)

    # Fit a polynomial to the sum of squared diff data
    xdata = frac
    ydata = [sum_diff_sq[ff,:].mean() for ff in range(len(xdata))]
    yerrdata = [sum_diff_sq[ff,:].std(ddof=1) for ff in range(len(xdata))]
    #zz = np.polyfit(xdata, ydata, 2) # works, but doesn't account for errors.
    #pp = np.poly1d(zz)

    # p0 should contain parameters for a 2nd degree polynomial: [A, B, C] where
    # A*x**2 + B*x + C
    p0 = [5.3, -2.1, 0.2] # these initial guesses were determined from np.polyfit above
    pfit = fitParabola(p0,[xdata, ydata, yerrdata],1)
    fparams = pfit.params
    fparamserr = pfit.perror
    fitx = np.arange(0.01,0.6,0.01)
    #fitx = np.arange(xdata.min(),xdata.max()+0.01,0.01)
    fity = fparams[0]*fitx**2 + fparams[1]*fitx + fparams[2]
    print 
    print 'Minimum sum of squared diffs = %5.3f, at disk fraction = %4.2f' % \
          (fity.min(),fitx[fity.argmin()])
    print
    fiterr = fparamserr * np.sqrt(pfit.fnorm / pfit.dof)
    pdb.set_trace()
    fity_upSig = (fparams[0]+fparamserr[0])*fitx**2 + (fparams[1]+fparamserr[1])*fitx + (fparams[2] + fparamserr[2])
    fity_loSig = (fparams[0]-fparamserr[0])*fitx**2 + (fparams[1]-fparamserr[1])*fitx + (fparams[2] - fparamserr[2])
    print 'Minimum sum of squared diffs (+ 1 sigma) = %5.3f, at disk fraction = %4.2f' % \
          (fity_upSig.min(),fitx[fity_upSig.argmin()])
    print 'Minimum sum of squared diffs (- 1 sigma) = %5.3f, at disk fraction = %4.2f' % \
          (fity_loSig.min(),fitx[fity_loSig.argmin()])

    py.figure(2)
    py.figure(figsize=(6,6))
    py.clf()
    py.subplots_adjust(left=0.17, right=0.96, wspace=0.25,
                       hspace=0.25, top=0.95, bottom=0.11)
    # Plot the best fit
    py.plot(fitx,fity,'k--')
    py.plot(fitx,fity_upSig,'k-.') 
    py.plot(fitx,fity_loSig,'k-.')
    for ff in range(len(frac)):
        py.errorbar(frac[ff], sum_diff_sq[ff,:].mean(),
                    yerr=sum_diff_sq[ff,:].std(ddof=1), fmt='k.')
        py.semilogy()
    py.axis([0,0.602,0.004,1])
    py.xlabel(r'{\bf \huge{$f_{disk}$}}')
    py.ylabel(r'{\bf \huge{$\xi$}}')
    py.savefig('%s/plots/ave_rms_diff_vs_frac_%strials.png' % (simDir,str(ntrials)))
    py.savefig('%s/plots/eps/ave_rms_diff_vs_frac_%strials.eps' % (simDir,str(ntrials)))
    py.close(2)

    print
    print 'Sum of Squared Diffs per disk fraction:'
    print hdr % ('Frac.','Mean','STDEV')
    fmt = '%5.2f  %5.3f  %5.3f'
    for ff in range(len(frac)):
        print fmt % (frac[ff], sum_diff_sq[ff,:].mean(), sum_diff_sq[ff,:].std(ddof=1))

    # Overestimation of disk members:
    over = np.zeros((len(frac), 10), dtype=float)
    py.figure(3)
    py.figure(figsize=(6,6))
    py.clf()
    py.subplots_adjust(left=0.14, right=0.96, wspace=0.25,
                       hspace=0.25, top=0.95, bottom=0.11)
    for nn in range(ntrials):
        for ff in range(len(frac)):
            over[ff,nn] = numMembersSim[ff,nn]/ndiskSim[ff]
            py.plot(frac[ff], numMembersSim[ff,nn]/ndiskSim[ff], marker='o',
                   color=clr[nn],  alpha=(1.-nn*0.09))
    py.plot([0,0.6],[1.0,1.0],'k--')
    py.axis([0,0.6,0,22])
    py.xlabel(r'{\bf \huge{$f_{disk}$}}')
    py.ylabel('Overestimation of Disk Membership')
    py.savefig('%s/plots/disk_membership_vs_fraction_%strials.png' % (simDir,str(ntrials)))
    py.savefig('%s/plots/eps/disk_membership_vs_fraction_%strials.eps' % (simDir,str(ntrials)))
    py.close(3)

    # Fit a polynomial to the overestimation data
    ydata = [over[ff,:].mean() for ff in range(len(xdata))]
    yerrdata = [over[ff,:].std(ddof=1) for ff in range(len(xdata))]

    p0 = [1, -0.5] 
    pfit = fitPowerLawMP(p0,[xdata, ydata, yerrdata],1)
    fparams = pfit.params
    #fitx2 = np.arange(xdata.min(),xdata.max()+0.01,0.01)
    fitx2 = np.arange(0.01,0.6,0.01)
    fity2 = fparams[0]*fitx2**fparams[1]
    fidx = np.where(fitx2 == fitx[fity.argmin()])[0]
    print 
    print 'Overestimation of disk membership for f_disk = %5.3f is %4.2f' % \
          (fitx2[fidx],(fity2[fidx]))
    print 

    py.figure(4)
    py.figure(figsize=(6,6))
    py.clf()
    py.subplots_adjust(left=0.14, right=0.96, wspace=0.25,
                       hspace=0.25, top=0.95, bottom=0.11)
    # Plot the best fit
    py.plot(fitx2,fity2,'k--')
    for ff in range(len(frac)):
        py.errorbar(frac[ff], (over[ff,:]).mean(), yerr=(over[ff,:]).std(ddof=1), fmt='k.')
        #py.semilogy()
    py.plot([0,0.6],[1.0,1.0],'k--')
    py.axis([0,0.6,0.0,10.0])
    #py.axis([0,0.6,0.0,30.0])
    py.xlabel(r'{\bf \huge{$f_{disk}$}}')
    py.ylabel('Overestimation of Disk Membership')
    py.savefig('%s/plots/ave_disk_membership_vs_fraction_%strials.png' % (simDir,str(ntrials)))
    py.savefig('%s/plots/eps/ave_disk_membership_vs_fraction_%strials.eps' % (simDir,str(ntrials)))
    py.close(4)

    usetexFalse()

    print
    print 'Overestimation of disk membership per disk fraction:'
    print hdr % ('Frac.','Mean','STDEV')
    fmt = '%5.2f  %5.3f  %5.3f'
    for ff in range(len(frac)):
        print fmt % (frac[ff], over[ff,:].mean(), over[ff,:].std(ddof=1))
        



def modelGaussian(x, A, mu, sigma):
    model = A/(sigma*np.sqrt(2*np.pi))*np.exp(-.5*((x-mu)/sigma)**2)
    return model

def fitfuncGauss(p, fjac=None, x=None, occ=None, err=None):
    """Find residuals of Gauss fit.

    For Gaussian, p should be list of form [A, mu, sigma],
    while data should be list of form [x, occ, err], where 
    x is the deviation in units of sigma and occ is the
    occupation of the histogram bin.
    """

    num = len(x)
    model = np.zeros(num, dtype=float)
    devs = np.zeros(num, dtype=float)

    # Set parameters
    # Include normalization constant A as the data
    # is assumed not to be normalized
    A = p[0]
    mu = p[1]
    sigma = p[2]

    model = modelGaussian(x, A, mu, sigma)
    residuals = (occ - model)/err
    status = 0

    return [status, residuals]

def fitGaussianMP(p0=None,data=None,quiet=0):
    """Fits Gaussian using mpfit.

    Inputs should be given as p0=[A, mu, sigma] and
    data=[x, occ, err], where  x is the deviation in
    units of sigma and occ is the occupation of the
    histogram bin.    Returns object of class mpfit.
    """

    print 'Initial Guess:'
    print '   A     = %6.2f' % p0[0]
    print '   mu    = %5.3f' % p0[1]
    print '   sigma = %5.3f' % p0[2]

    # Remove data with zero error (residuals go as 1/err)
    x   = (data[0])[np.nonzero(data[2])]
    occ = (data[1])[np.nonzero(data[2])]
    err = (data[2])[np.nonzero(data[2])]

    # Set data to be passed to fit
    functargs = {'x':x,'occ':occ,'err':err}

    # Set initial values and limits (no limits on parameters)
    pinfo = [{'value':0,'fixed':0,'limited':[0,0],
	      'limits':[0,0]}]*len(p0)
    for ii in range(len(p0)):
        pinfo[ii]['value'] = p0[ii]

    # Use mpfit fitting algorithm to fit parameters
    m = nmpfit_sy.mpfit(fitfuncGauss, p0, functkw=functargs, parinfo=pinfo,
		    quiet=quiet)
    if (m.status <= 0):
        print 'Error message = ', m.errmsg

    p = m.params          # Best-fit parameters
    perr = m.perror       # Error in parameter fits from covariance matrix
    m.dof = len(x)-len(p) # Number of degrees of freedom
    Rchi2 = m.fnorm/m.dof # Reduced Chi^2 statistic

    print 'Final Solution:'
    print '   A      = %6.2f +/- ' % p[0]
    print '   mu     = %5.3f +/- ' % p[1]
    print '   sigma  = %5.3f +/- ' % p[2]
    #print '   A      = %6.2f +/- %5.2f' % (p[0],perr[0])
    #print '   mu     = %5.3f +/- %5.3f' % (p[1],perr[1])
    #print '   sigma  = %5.3f +/- %5.3f' % (p[2],perr[2])
    print '   chi^2  = %5.2f' % m.fnorm
    print '   dof    = %2d' % m.dof
    print '   Rchi^2 = %5.2f' % Rchi2

    return m


def make_isotropic_healpix(nstars=200):

    # bad stars in simulations in /sim_isotropic_big/ :
    #badStars = [18, 1324, 1775, 1893, 1913, 1989] 

    ntrials = 10**4
    #mockdir = '/sim_isotropic_big/'
    mockdir = 'sim_disk_lineOfNodes/'
    #mockdir = 'sim_isotropic2/'
    #outroot = mockdir + 'plots/'
    outroot = root + alnDir + mockdir + 'plots/'
    for ss in range(nstars):
        #if ss in badStars:
        #    continue
        #pdffile = '%s/star%s.mc.dat' % (mockdir, str(ss))
        pdffile = '%s/%s/%s/star%s.mc.dat' % (root, alnDir, mockdir, str(ss))
        pdf = pickle.load(open(pdffile))

        hp = makePdfHealpix('star'+str(ss), pdf, outroot+'HEALpixMaps/',
                            ntrials=ntrials, nside=64,makeplot=True)


def makePdfHealpix_simstars(start,end):
    badStars = [112,207,523,643,726,1046,1076,1129,1783,1971,2402,2434,2569,2680,
                3045,3312,3470,3917,4465,4662,4800,4978]

    mockfile='isotropic_5000stars_mockdata.pickle'
    rootDir = root + alnDir
    mockdir = 'sim_true_isotropic/'
    outroot = rootDir + mockdir + 'plots/'

    for ss in range(start,end+1):
        if ss in badStars:
            continue
        pdffile = '%s/%s/%s/star%s.mc.dat' % (root, alnDir, mockdir, str(ss))
        pdf = pickle.load(open(pdffile))

        hp = makePdfHealpix('star'+str(ss), pdf, outroot+'HEALpixMaps/',
                            ntrials=10**4, nside=64,makeplot=True)


def run_isotropic(all=False,r1=False,r2=False,r3=False,nIso=5000,
                  start=0,end=999):
    """
    Creates mock data for 5000 randomly distributed stars. Randomly selects 40 within our
    observed field of view (OSIRIS and SINFONI) and in either the inner, middle, or outer radial bin and
    runs density analysis on them. Does the random sampling of N=40 stars a total of 50 times.

    start and end numbers refer to the simulation trial to run the nearest neighbor
    density map (for all, r1, r2, or r3). Run a total of 1000 simulated density maps.
    """

    import sythesis_sim as sim
    #mockdir = 'sim_isotropic_big/'
    #rootDir = '/'
    mockfile='isotropic_5000stars_mockdata.pickle'
    rootDir = root + alnDir
    mockdir = 'sim_true_isotropic/'
    outroot = rootDir + mockdir + 'plots/'

    #badStars = [18, 1324, 1775, 1893, 1913, 1989] # bad stars in simulations in /sim_isotropic_big/
    badStars = [112,207,523,643,726,1046,1076,1129,1783,1971,2402,2434,2569,2680,
                3045,3312,3470,3917,4465,4662,4800,4978]

    #sim.mock_data(nstars=0,e=-1,vel_kick=False,isoPop=nIso,
    #              mockdir=mockdir, outfile=mockfile)

#    mc = sim.simulate_orbits(mockdir=mockdir,mockfile=mockfile)
#    mc.run()

#    # made function above makePdfHealpix_simstars() to run the next few lines faster:
#    # Make HEALpix maps for individual stars
#    #for ss in range(1500, 1560):
#    for ss in range(nIso):
#        if ss in badStars:
#            continue
#        pdffile = '%s/%s/%s/star%s.mc.dat' % (root, alnDir, mockdir, str(ss))
#        pdf = pickle.load(open(pdffile))
#
#        hp = makePdfHealpix('star'+str(ss), pdf, outroot+'HEALpixMaps/',
#                            ntrials=10**4, nside=64,makeplot=True)

    nside = 64
    npix = healpy.nside2npix(nside)
    pixIdx = np.arange(npix, dtype=int)
    (i, o) = healpy.pix2ang(nside, pixIdx)
    i *= 180.0 / math.pi
    o *= 180.0 / math.pi

    def loadStars():
        # Load up the mock data
        # Astrometric units are: arcsec, mas/yr, mas/yr^2
        mockdata = open(rootDir + mockdir + mockfile)
        mbh = pickle.load(mockdata)
        dist = pickle.load(mockdata)
        orb_all = pickle.load(mockdata)
        sma_all = pickle.load(mockdata) # AU
        xM = pickle.load(mockdata) # (+x to east)
        yM = pickle.load(mockdata)
        zM = pickle.load(mockdata)
        vxM = pickle.load(mockdata) # (+x to east)
        vyM = pickle.load(mockdata)
        vzM = pickle.load(mockdata) # mas/yr
        axM = pickle.load(mockdata) # (+x to east)
        ayM = pickle.load(mockdata)
        azM = pickle.load(mockdata)
        t0M = pickle.load(mockdata) # periapse passage
        t_obs = pickle.load(mockdata) # observational time for mock data point
        mockdata.close()

        r2d = np.sqrt(xM**2 + yM**2)
        x = xM
        y = yM

        return x, y, r2d

    def densityPDF(iAll, oAll, neighbors=[6], npix=npix, nside=nside,
                   pdftrials=10**4):
        """
        Map out the PDF for the density of normal vectors
        """
        nstars = iAll.shape[0]
        print nstars, type(nstars)
        trials = 10000
        npdfs = len(neighbors)

        pixIdx = np.arange(npix, dtype=int)
        (ipix, opix) = healpy.pix2ang(nside, pixIdx)
        sinip = np.sin(ipix)
        cosip = np.cos(ipix)

        siniAll = np.sin(iAll)
        cosiAll = np.cos(iAll)

        onesNpix = np.ones(npix, dtype=float)
        onesNstars = np.ones(nstars, dtype=float)
        factor = 2.0 * math.pi * (180.0 / math.pi)**2

        if (trials > pdftrials):
            print 'Must have more PDF trials than disk trials'
        
        # Compute the PDF for the density at each pixel along
        # with the weighted average density at each pixel.
        neighborMap = np.zeros((npdfs, npix), dtype=float)
        neighborMapStd = np.zeros((npdfs, npix), dtype=float)

        # Keep track of the peak density and position for each trial
        peakDensity = np.zeros((npdfs, trials), dtype=float)
        peakIncli = np.zeros((npdfs, trials), dtype=float)
        peakOmega = np.zeros((npdfs, trials), dtype=float)

        _out1 = open('%s/%s/disk_nn_results.txt' % \
                     (rootDir, mockdir), 'w')

        print 'Running MC to obtain density map.'

        # temp
        import Numeric
        # end temp
        
        for ii in range(trials):
            if ((ii % 100) == 0):
                print 'Trial %d' % ii, time.ctime(time.time())
            
            # Randomly select an (i,o) pair out of each star's
            # marginalized PDF.
            incl = iAll[:, ii]
            omeg = oAll[:, ii]
            sini = siniAll[:, ii]
            cosi = cosiAll[:, ii]
            
            # Check for bad things
            idx = (np.where((incl == float('nan')) |
                         (omeg == float('nan'))))[0]
            if (len(idx) > 0):
                print ii, idx
                    
            # Find densities
            omegSq = np.outer(omeg, onesNpix)
            opixSq = np.outer(onesNstars, opix)
            cosodiff = np.cos(opixSq - omegSq)

            sinSq = np.outer(sini, sinip)
            cosSq = np.outer(cosi, cosip)

            # Angular offset from each pixel for all stars (radians)
            angOff = np.arccos( (sinSq * cosodiff) + cosSq )
            angOff.sort(axis=0)

            # Density per square degree from nearest neighbor
            # Solid angle is 2 * pi * (1 - cos theta)
            for nn in range(npdfs):
                # Nearest neighbor algorithm
                nth = neighbors[nn]
                densityMap = nth / (factor*(1.0 - np.cos(angOff[nth-1,:])))
                maxPix = densityMap.argmax()

                neighborMap[nn,:] += densityMap
                neighborMapStd[nn,:] += densityMap**2
                peakDensity[nn,ii] = densityMap[maxPix]
                peakOmega[nn,ii] = opix[maxPix]
                peakIncli[nn,ii] = ipix[maxPix]

                # Save to an output file
                if (nn == 2):
                    fubar = array(densityMap * 10**5, dtype=int64)
                    fubar.tofile(_out1)
                    fubar = None

        neighborMap /= trials
        neighborMapStd = np.sqrt( (neighborMapStd / trials) - neighborMap**2 )

        return (neighborMap, neighborMapStd, peakDensity, peakIncli, peakOmega)

    fmt = '%3i  %5.1f  %5.1f  %.2e  %.2e  %.2e  %5.2f\n'

    nUse = 40
    if all == True:
        strbin = 'full_'
        nUse = 116 
    elif r1 == True:
        strbin = 'inner_'
    elif r2 == True:
        strbin = 'middle_'
    elif r3 == True:
        strbin = 'outer_'

    # Running 10 simulations at a time to speed this up.
    # Be sure to write each out to a different file:
    suffix = str(start)
    out = open('%s/%s/%s%sstars_nnDensity_%s.dat' % \
               (rootDir, mockdir, strbin, str(nUse), suffix), 'w')
    for tt in range(start,end+1): 
        # Run density analysis on a randomly selected group of 40 stars
        # within our field of view constraints and for the specified radial bin

        # Load the mock data
        x, y, r2d = loadStars()

        # Get all the stars in the appropriate FOV
        if all == True:
            stars = np.where(r2d < 15.0)[0] # The FOV mask below will constrain this further    
        if r1 == True:
            stars = np.where(r2d <= 3.197)[0]    
        if r2 == True:
            stars = np.where((r2d > 3.197) & (r2d <=6.473))[0]    
        if r3 == True:
            stars = np.where(r2d > 6.473)[0]    
        xys = np.column_stack((x, y))
        fields = asciidata.open('/sim_isotropic_big/osiris_sinfoni_fields.txt')
        #fields = asciidata.open('/u/syelda/research/gc/aligndir/11_10_26/tables/osiris_sinfoni_fields.txt')
        xvrt0 = fields[0].tonumpy()
        yvrt0 = fields[1].tonumpy()
        xvrt = np.array([np.float(xx) for xx in xvrt0])
        yvrt = np.array([np.float(yy) for yy in yvrt0])
        verts = np.column_stack((xvrt, yvrt))
        mask = nx.points_inside_poly(xys, verts)
            
        idx = []
        for rr in stars:
            # Check that this star is in the field
            if ((mask[rr] == False) | (rr in badStars)):
                print 'Skipping star %s - outside field (x,y) = (%6.2f, %6.2f)' % \
                      (rr, x[rr], y[rr])
                continue
            #elif rr < 2337:  # There were 2336 stars in the isotropic simulation total (that I ran orbits on)
            else:
                print 'Keeping star: %s' % rr
                idx = np.concatenate([idx, [rr]])
        idx = [int(ii) for ii in idx]
        print ''
        print 'Number of stars available within our FOV constraints: %i' % len(idx) 

        # Now randomly select nUse stars from these; do this by shuffling the idx array
        # then taking the first nUse stars in the list
        np.random.shuffle(idx)
        samp = idx[0:nUse]

        # Now run the nearest neighbor density analysis on these stars
        iomap = np.zeros((1, npix), dtype=float)
        ioCntMap = np.zeros((1, npix), dtype=float)
        
        iAll = None
        oAll = None
        cnt = 0
        for ss in samp:

            name = 'star%s' % str(ss)

            # Read in the 6D Probability distribution of orbital
            # parameters for this star.
            _f = open('%s/%s/%s.mc.dat' % (rootDir, mockdir, name), 'r')
            mc = pickle.load(_f)
            _f.close()
            print 'Adding %15s (%d) to disk' % (name, len(mc.i))

            # Marginalize the PDF for this star onto just
            # incl (i) and PA to ascending node (o).
            hpfile = '%s/%s/plots/HEALpixMaps/%s_disk_mc_heal.dat' %\
                      (rootDir, mockdir, name)
            pdf = np.fromfile(hpfile, dtype=float)

            # Store the monte carlo results as (i, o) pairs.
            if (iAll == None):
                pdftrials = len(mc.i)
                iAll = np.zeros((nUse, pdftrials), dtype=float)
                oAll = np.zeros((nUse, pdftrials), dtype=float)
                 
            # This assumes that the Mass/r0/x0/y0 values are the
            # same for a single trial across all stars.
            iAll[cnt,:] = mc.i
            oAll[cnt,:] = mc.o

            iomap[0] += pdf
            ioCntMap[0] += (pdf / pdf.max())
            cnt += 1

        # Get the (i,o) values for each pixel in the sky
        iAll *= math.pi / 180.0
        oAll *= math.pi / 180.0

        # Save the marginalized PDF to a file.
        iomap.tofile('%s/%s/%ssimdisk_%s.heal.dat' % \
                          (rootDir, mockdir, strbin, str(tt)))
        ioCntMap.tofile('%s/%s/%ssimdisk_%s.iocnt.dat' % \
                          (rootDir, mockdir, strbin, str(tt)))

        # Map out the PDF for the density of normal vectors
        (neigh, neighStd, peakD, peakI, peakO) = densityPDF(iAll, oAll, neighbors=[6],
                                                            npix=npix,nside=nside)
        
        # Save the i/o density maps
        print 'Making density map'
        neigh.tofile('%s/%s/%ssimdisk_%s.neighbor.dat' % \
                     (rootDir, mockdir, strbin, str(tt)))
        neighStd.tofile('%s/%s/%ssimdisk_%s.neighborStd.dat' % \
                     (rootDir, mockdir, strbin, str(tt)))
        peakD.tofile('%s/%s/%ssimdisk_%s.peakDensity.dat' % \
                     (rootDir, mockdir, strbin, str(tt)))
        peakI.tofile('%s/%s/%ssimdisk_%s.peakIncli.dat' % \
                     (rootDir, mockdir, strbin, str(tt)))
        peakO.tofile('%s/%s/%ssimdisk_%s.peakOmega.dat' % \
                     (rootDir, mockdir, strbin, str(tt)))

        # Make the HEALpix map
        pdh.go('%s/%s/%ssimdisk_%s.neighbor.dat' % (rootDir,mockdir,strbin,str(tt)), 49152, 1)

        # Now check the significance of any peak
        f1 = '%ssimdisk_%s.neighbor.dat' % (strbin, str(tt))
        f2 = '%ssimdisk_%s.neighborStd.dat' % (strbin, str(tt))
        (disk, diskStd) = loadDiskDensity(npix, orbDir=rootDir+mockdir, aperture=False,
                                          file1=f1, file2=f2, simdisk=True)
        didx = disk.argmax()
        peak = disk[didx]
        idisk = i[didx]
        odisk = o[didx]
        print 'Peak found at (i,O) = (%5.1f, %5.1f) with %.2e  stars/deg^2' % \
              (idisk,odisk,peak)
        print
        print '%8.5f - Expected density for isotropic population with %i stars' % \
              ((nUse / areaOnSky), nUse)

        # What is the measured density outside the disk region:
        avgBkg = 1.0
        stdBkg = 1.0

        # Iteratively compute background
        for n in range(2):
            idx = (np.where(disk < (avgBkg + (3.0 * stdBkg))))[0]

            avgBkg = disk[idx].mean()
            stdBkg = disk[idx].std(ddof=1)
            print 'trial = %2d   avg = %e   std = %e, rejecting %d of %d' % \
                  (n, avgBkg, stdBkg, (len(disk)-len(idx)), len(disk))

        signif = (peak - avgBkg) / stdBkg

        print ''
        print 'Average Background   rho = %e' % avgBkg
        print 'Stddev of Background rho = %e' % stdBkg
        print 'Density Ratio for CW disk = %5.2f' % signif
        #print 'Density Ratio for CW disk = %5.2f' % ((peak - avgBkg) / stdBkg)

        # Write to file
        out.write(fmt % (tt, idisk, odisk, peak, avgBkg, stdBkg, signif))


    out.close()



def run_isotropic_results(all=False,r1=False,r2=False,r3=False,numSims=1000):
    """
    Plot results from the isotropic simulations above in run_isotropic().
    The results were saved in a file called <radial bin>_40stars_nnDensity.dat.
    The stars were randomly selected in each simulation such that they fell within
    observed field of view (OSIRIS and SINFONI) and in one of the radial bins, or the
    full radial extent was used (within the observed field, of course).  Then
    runs density analysis on them. 
    """

    mockfile='isotropic_5000stars_mockdata.pickle'
    rootDir = root + alnDir
    mockdir = 'sim_true_isotropic/'
    outdir = rootDir + mockdir + 'plots/'

    nUse = 40
    if all == True:
        strbin = 'full_'
        nUse = 116
        rbin = 0
    elif r1 == True:
        strbin = 'inner_'
        rbin = 1
    elif r2 == True:
        strbin = 'middle_'
        rbin = 2
    elif r3 == True:
        strbin = 'outer_'
        rbin = 3

    nside = 64
    npix = healpy.nside2npix(nside)
    pixIdx = np.arange(npix, dtype=int)
    (i, o) = healpy.pix2ang(nside, pixIdx)
    i *= 180.0 / math.pi
    o *= 180.0 / math.pi

    print 'Getting results for isotropic simulation of %s bin' % strbin
    print

    out = open('%s/%s/%s%sstars_nnDensity.dat' % \
               (rootDir, mockdir, strbin, str(nUse)), 'w')
    hpmaps = asciidata.open('/%s%s/%ssimdisks_run_so_far.txt' % \
                            (rootDir, mockdir, strbin))
    hpmap = hpmaps[0].tonumpy()
    trials = len(hpmap)
    fmt = '%4s  %5.1f  %5.1f  %.2e  %.2e  %.2e  %5.2f\n'

    for tt in range(trials):
        # Now check the significance of any peak
        #f1 = '%ssimdisk_%s.neighbor.dat' % (strbin, str(tt))
        #f2 = '%ssimdisk_%s.neighborStd.dat' % (strbin, str(tt))

        trial = hpmap[tt].split('.')[0][13:]
        f1 = '%s' % hpmap[tt]
        f2 = str(hpmap[tt]).replace('neighbor', 'neighborStd')
        (disk, diskStd) = loadDiskDensity(npix, orbDir=rootDir+mockdir, aperture=False,
                                          file1=f1, file2=f2, simdisk=True)
        didx = disk.argmax()
        peak = disk[didx]
        idisk = i[didx]
        odisk = o[didx]
        print 'Peak found at (i,O) = (%5.1f, %5.1f) with %.2e  stars/deg^2' % \
              (idisk,odisk,peak)
        print
        print '%8.5f - Expected density for isotropic population with %i stars' % \
              ((nUse / areaOnSky), nUse)

        # What is the measured density outside the disk region:
        avgBkg = 1.0
        stdBkg = 1.0

        # Iteratively compute background
        for n in range(2):
            idx = (np.where(disk < (avgBkg + (3.0 * stdBkg))))[0]

            avgBkg = disk[idx].mean()
            stdBkg = disk[idx].std(ddof=1)
            print 'trial = %2d   avg = %e   std = %e, rejecting %d of %d' % \
                  (n, avgBkg, stdBkg, (len(disk)-len(idx)), len(disk))

        signif = (peak - avgBkg) / stdBkg

        print ''
        print 'Average Background   rho = %e' % avgBkg
        print 'Stddev of Background rho = %e' % stdBkg
        print 'Density Ratio for CW disk = %5.2f' % signif
        #print 'Density Ratio for CW disk = %5.2f' % ((peak - avgBkg) / stdBkg)

        # Write to file
        out.write(fmt % (trial, idisk, odisk, peak, avgBkg, stdBkg, signif))

    out.close()

    results = asciidata.open('/%s%s/%s%sstars_nnDensity.dat' % (rootDir, mockdir, strbin, str(nUse)))
    incl = results[1].tonumpy()
    Omega = results[2].tonumpy()
    pkDen = results[3].tonumpy() *1.e3
    avDen = results[4].tonumpy()
    sdDen = results[5].tonumpy()
    sig = results[6].tonumpy()

    usetexTrue()
    py.clf()
    py.figure(figsize=(8,8))
    py.subplots_adjust(left=0.12, right=0.95, top=0.94, bottom=0.1,
                       wspace=0.3, hspace=0.3)
    py.subplot(2,2,1)
    aa,bb,cc = py.hist(incl, bins=np.arange(0, 181, 10.0), histtype='step',color='k')
    py.axis([0,180,0,aa.max()+2])
    py.xlabel('Peak Inclination (deg)')
    py.ylabel('N')
    py.subplot(2,2,2)
    aa,bb,cc = py.hist(Omega, bins=np.arange(0, 361, 10.0), histtype='step',color='k')
    py.axis([0,360,0,aa.max()+2])
    py.xlabel('Peak Omega (deg)')
    py.ylabel('N')
    py.subplot(2,2,3)
    aa,bb,cc = py.hist(pkDen, bins=np.arange(0, pkDen.max()+1, 0.1), histtype='step',
                       color='k')
    py.axis([pkDen.min()-1,pkDen.max()+1,0,aa.max()+2])
    py.xlabel(r'Peak Density ($\times$10$^{-3}$ stars deg$^{-2}$)')
    py.ylabel('N')
    py.subplot(2,2,4)
    aa,bb,cc = py.hist(sig, bins=np.arange(0, sig.max()+1, 0.2), histtype='step',
                       color='k')
    py.axis([0,sig.max()+1.,0,aa.max()+2])
    py.xlabel('Overdensity')
    py.ylabel('N')
    py.savefig('/%s/%sisotropic_significance.png' % (outdir, strbin))
    py.savefig('/%s/%sisotropic_significance.eps' % (outdir, strbin))
    py.close()

    py.clf()
    py.figure(figsize=(6,6))
    (nx, bx, ptx) = py.hist(pkDen, bins=np.arange(0, pkDen.max()+1, 0.1), histtype='step',
                            normed=True,color='k')
    # Fit a Gaussian to the peak density distribution
    mu_guess = pkDen.mean()
    std_guess = pkDen.std(ddof=1)
    p0 = [1.0, mu_guess, std_guess] # remember, the peak density is plotted in units of 1e-3
    #p0 = [1.0, 7.0, 0.5] # remember, the peak density is plotted in units of 1e-3
    gfit = fitGaussianMP(p0, [bx, nx, np.sqrt(nx)],1)
    rparams = gfit.params
    xfit = np.arange(1,15,0.01)
    yfit = modelGaussian(xfit, rparams[0], rparams[1], rparams[2])
    print 'Multiply the mean and sigma above by 1.e-3 to get in stars/deg^2'
    py.plot(xfit, yfit, 'k-', lw=1.5)
    py.axis([pkDen.min()-1,pkDen.max()+1,0,nx.max()+0.2])
    py.xlabel(r'Peak Density ($\times$10$^{-3}$ stars deg$^{-2}$)')
    py.ylabel('N')
    py.savefig('/%s/%sisotropic_peak_densities.png' % (outdir, strbin))
    py.savefig('/%s/%sisotropic_peak_densities.eps' % (outdir, strbin))
    py.close()

    # Peak density observed in the full, inner, middle, and outer radial bins:
    pk_obs_all = [0.024, 0.014, 0.0025, 0.0044] 
    pk_obs_incl = [130.2, 128.7, 127.2, 117.3]
    pk_obs_Omega = [96.3, 97.7, 103.4, 192.0]

    pk_obs = pk_obs_all[rbin] 
    pk_obs_i = pk_obs_incl[rbin] 
    pk_obs_O = pk_obs_Omega[rbin] 

    # Gaussian fit:
    ave_dist = rparams[1] * 1.e-3
    std_dist = rparams[2] * 1.e-3

    # Significance of observed peak compared to the Gaussian fit (over all inclinations)
    obs_sig = (pk_obs - ave_dist) / std_dist    

    #print
    #print 'Observed peak density for the %s bin = %6.4f stars/deg^2' % (strbin, pk_obs)
    #print 'Compared to Gaussian fit to distribution of simulated peak densities (across entire sky):'
    #print '  observed feature is %5.2f sigma from the mean' % obs_sig

    py.clf()
    py.figure(figsize=(6,6))
    aa,bb,cc = py.hist(sig, bins=np.arange(0, sig.max()+1, 0.2), histtype='step',
                       normed=True,color='k')
    py.axis([0,sig.max()+1,0,aa.max()+0.2])
    py.xlabel('Overdensity')
    py.ylabel('PDF')
    py.savefig('/%s/%sisotropic_sig_peak.png' % (outdir, strbin))
    py.savefig('/%s/%sisotropic_sig_peak.eps' % (outdir, strbin))
    py.close()

    # How many times do we get a >X sigma peak? (where X is what we observed)
    # Use the observed sigma over the background (which we know is an overestimate)
    obs_sig_bkg = [11.5, 15.7, 3.2, 5.0]
    bs = np.where(sig > obs_sig_bkg[rbin])[0]
    print
    sig_5sig = len(bs)*100. / numSims
    print 'Number of >%4.1f sigma results in an isotropic sim: %2i out of %2i (%4.1f percent)' % \
          (obs_sig_bkg[rbin], len(bs), numSims, sig_5sig)

    # Loop over the 1000 simulations and get the average and rms
    # density within a strip across the HEALpix map of width
    # inclination = 30 deg, across all Omegas. This gives
    # an ave and rms for 6 inclination bins.
    nside = 64
    npix = healpy.nside2npix(nside)
    pixIdx = np.arange(npix, dtype=int)
    (i, o) = healpy.pix2ang(nside, pixIdx)
    i *= 180.0 / math.pi
    o *= 180.0 / math.pi
    disk1 = []
    disk2 = []
    disk3 = []
    disk4 = []
    disk5 = []
    disk6 = []
    disk7 = []
    disk8 = []
    disk9 = []
    disk1pk = []
    disk2pk = []
    disk3pk = []
    disk4pk = []
    disk5pk = []
    disk6pk = []
    disk7pk = []
    disk8pk = []
    disk9pk = []
    diskPk_all_i = []

    hpmaps = asciidata.open('/%s%s/%ssimdisks_run_so_far.txt' % (rootDir, mockdir, strbin))
    hpmap = hpmaps[0].tonumpy()

    density_ave = np.zeros((npix), dtype=float)
    density_std = np.zeros((npix), dtype=float)
    for ii in range(len(hpmap)):
        #hpfile = '%s/%s' % (rootDir, hpmap[ii])     

        # Now check the significance of any peak
        f1 = '%s' % hpmap[ii]
        f2 = str(hpmap[ii]).replace('neighbor', 'neighborStd')
        (disk, diskStd) = loadDiskDensity(npix, orbDir=rootDir+mockdir, aperture=False,
                                          file1=f1, file2=f2, simdisk=True, silent=True)

        # Get the average of the entire map
        density_ave[:] += disk
        density_std[:] += disk**2

        # Inclination bins
        idx1 = np.where(i <= 20.0)[0]   
        idx2 = np.where((i > 20.0) & (i <= 40.0))[0]   
        idx3 = np.where((i > 40.0) & (i <= 60.0))[0]   
        idx4 = np.where((i > 60.0) & (i <= 80.0))[0]   
        idx5 = np.where((i > 80.0) & (i <= 100.0))[0]   
        idx6 = np.where((i > 100.0) & (i <= 120.0))[0]   
        idx7 = np.where((i > 120.0) & (i <= 140.0))[0]   
        idx8 = np.where((i > 140.0) & (i <= 160.0))[0]   
        idx9 = np.where((i > 160.0) & (i <= 180.0))[0]   

        # Get the densities within each incl range
        disk1 = np.concatenate([disk1,disk[idx1]])
        disk2 = np.concatenate([disk2,disk[idx2]])
        disk3 = np.concatenate([disk3,disk[idx3]])
        disk4 = np.concatenate([disk4,disk[idx4]])
        disk5 = np.concatenate([disk5,disk[idx5]])
        disk6 = np.concatenate([disk6,disk[idx6]])
        disk7 = np.concatenate([disk7,disk[idx7]])
        disk8 = np.concatenate([disk8,disk[idx8]])
        disk9 = np.concatenate([disk9,disk[idx9]])

        # Get the PEAK densities within each incl range
        disk1pk = np.concatenate([disk1pk,[(disk[idx1]).max()]])
        disk2pk = np.concatenate([disk2pk,[(disk[idx2]).max()]])
        disk3pk = np.concatenate([disk3pk,[(disk[idx3]).max()]])
        disk4pk = np.concatenate([disk4pk,[(disk[idx4]).max()]])
        disk5pk = np.concatenate([disk5pk,[(disk[idx5]).max()]])
        disk6pk = np.concatenate([disk6pk,[(disk[idx6]).max()]])
        disk7pk = np.concatenate([disk7pk,[(disk[idx7]).max()]])
        disk8pk = np.concatenate([disk8pk,[(disk[idx8]).max()]])
        disk9pk = np.concatenate([disk9pk,[(disk[idx9]).max()]])

        diskPk_all_i = np.concatenate([diskPk_all_i,[disk.max()]])


    density_ave /= len(hpmap)
    density_std = np.sqrt((density_std / len(hpmap)) - density_ave**2)
    avefile = '/%s%s/%s%s_neighbor_aveAllTrials.dat' % (rootDir,mockdir,strbin,str(nUse))
    stdfile = '/%s%s/%s%s_neighbor_rmsAllTrials.dat' % (rootDir,mockdir,strbin,str(nUse))
    density_ave.tofile(avefile)
    density_std.tofile(stdfile)
    pdh.go(avefile, npix, 1)
    pdh.go(stdfile, npix, 1)

    # Get the average and rms density within each strip of incl:
    ave_dens1, std_dens1 = (disk1.mean(), disk1.std(ddof=1))
    ave_dens2, std_dens2 = (disk2.mean(), disk2.std(ddof=1))
    ave_dens3, std_dens3 = (disk3.mean(), disk3.std(ddof=1))
    ave_dens4, std_dens4 = (disk4.mean(), disk4.std(ddof=1))
    ave_dens5, std_dens5 = (disk5.mean(), disk5.std(ddof=1))
    ave_dens6, std_dens6 = (disk6.mean(), disk6.std(ddof=1))
    ave_dens7, std_dens7 = (disk7.mean(), disk7.std(ddof=1))
    ave_dens8, std_dens8 = (disk8.mean(), disk8.std(ddof=1))
    ave_dens9, std_dens9 = (disk9.mean(), disk9.std(ddof=1))

    # Get the average and rms PEAK density within each strip of incl:
    ave_dens1pk, std_dens1pk = (disk1pk.mean(), disk1pk.std(ddof=1))
    ave_dens2pk, std_dens2pk = (disk2pk.mean(), disk2pk.std(ddof=1))
    ave_dens3pk, std_dens3pk = (disk3pk.mean(), disk3pk.std(ddof=1))
    ave_dens4pk, std_dens4pk = (disk4pk.mean(), disk4pk.std(ddof=1))
    ave_dens5pk, std_dens5pk = (disk5pk.mean(), disk5pk.std(ddof=1))
    ave_dens6pk, std_dens6pk = (disk6pk.mean(), disk6pk.std(ddof=1))
    ave_dens7pk, std_dens7pk = (disk7pk.mean(), disk7pk.std(ddof=1))
    ave_dens8pk, std_dens8pk = (disk8pk.mean(), disk8pk.std(ddof=1))
    ave_dens9pk, std_dens9pk = (disk9pk.mean(), disk9pk.std(ddof=1))

    ave_densPk_all_i, std_densPk_all_i = (diskPk_all_i.mean(), diskPk_all_i.std(ddof=1))

    print
    print 'Average Density per inclination bin:'
    print 'Average and STD of density for i =   0 -  20 deg: %e +- %e' % (ave_dens1, std_dens1) 
    print 'Average and STD of density for i =  20 -  40 deg: %e +- %e' % (ave_dens2, std_dens2) 
    print 'Average and STD of density for i =  40 -  60 deg: %e +- %e' % (ave_dens3, std_dens3) 
    print 'Average and STD of density for i =  60 -  80 deg: %e +- %e' % (ave_dens4, std_dens4) 
    print 'Average and STD of density for i =  80 - 100 deg: %e +- %e' % (ave_dens5, std_dens5) 
    print 'Average and STD of density for i = 100 - 120 deg: %e +- %e' % (ave_dens6, std_dens6) 
    print 'Average and STD of density for i = 120 - 140 deg: %e +- %e' % (ave_dens7, std_dens7) 
    print 'Average and STD of density for i = 140 - 160 deg: %e +- %e' % (ave_dens8, std_dens8) 
    print 'Average and STD of density for i = 160 - 180 deg: %e +- %e' % (ave_dens9, std_dens9) 

    print
    print 'Average and STD of density for all inclinations: %e +- %e' % \
          (density_ave.mean(), density_std.mean()) 
    
    print
    print 'Average PEAK DENSITY per inclination bin:'
    print 'Average and STD of peak density for i =   0 -  20 deg: %e +- %e' % (ave_dens1pk, std_dens1pk) 
    print 'Average and STD of peak density for i =  20 -  40 deg: %e +- %e' % (ave_dens2pk, std_dens2pk) 
    print 'Average and STD of peak density for i =  40 -  60 deg: %e +- %e' % (ave_dens3pk, std_dens3pk) 
    print 'Average and STD of peak density for i =  60 -  80 deg: %e +- %e' % (ave_dens4pk, std_dens4pk) 
    print 'Average and STD of peak density for i =  80 - 100 deg: %e +- %e' % (ave_dens5pk, std_dens5pk) 
    print 'Average and STD of peak density for i = 100 - 120 deg: %e +- %e' % (ave_dens6pk, std_dens6pk) 
    print 'Average and STD of peak density for i = 120 - 140 deg: %e +- %e' % (ave_dens7pk, std_dens7pk) 
    print 'Average and STD of peak density for i = 140 - 160 deg: %e +- %e' % (ave_dens8pk, std_dens8pk) 
    print 'Average and STD of peak density for i = 160 - 180 deg: %e +- %e' % (ave_dens9pk, std_dens9pk) 

    print
    print 'Average and STD of peak density for all inclinations: %e +- %e' % \
          (ave_densPk_all_i, std_densPk_all_i) 

    # which inclination bin is the observed peak density in? pk_obs_i
    i_bins = np.arange(0,180,20)
    ave_dens_all = [ave_dens1, ave_dens2, ave_dens3, ave_dens4, ave_dens5, ave_dens6,
                    ave_dens7, ave_dens8, ave_dens9]
    std_dens_all = [std_dens1, std_dens2, std_dens3, std_dens4, std_dens5, std_dens6,
                    std_dens7, std_dens8, std_dens9]
    ave_denspk_all = [ave_dens1pk, ave_dens2pk, ave_dens3pk, ave_dens4pk, ave_dens5pk, ave_dens6pk,
                    ave_dens7pk, ave_dens8pk, ave_dens9pk]
    std_denspk_all = [std_dens1pk, std_dens2pk, std_dens3pk, std_dens4pk, std_dens5pk, std_dens6pk,
                    std_dens7pk, std_dens8pk, std_dens9pk]
    irng1 = pk_obs_i - (pk_obs_i % 20)
    irng2 = irng1 + 20
    ib = np.where(irng1 == i_bins)[0] 

    sigPk_incl = (pk_obs - ave_denspk_all[ib]) / std_denspk_all[ib]
    sig_incl = (pk_obs - ave_dens_all[ib]) / std_dens_all[ib]

    print
    print 'Signif. of peak as compared to simulated densities between i = (%5.1f - %5.1f) deg = %8.3f' % \
          (irng1, irng2, sig_incl)
    print
    print
    print 'Signif. of peak as compared to simulated PEAK DENSITIES between i = (%5.1f - %5.1f) deg = %8.3f' % \
          (irng1, irng2, sigPk_incl)
    print
    sigPk_incl_all_i = (pk_obs - ave_densPk_all_i) / std_densPk_all_i
    print 'Signif. of peak as compared to simulated PEAK DENSITIES over all inclinations = %8.3f' % \
          sigPk_incl_all_i
    print

    usetexTrue()
    py.clf()
    py.figure(figsize=(6,6))
    py.subplots_adjust(left=0.15,right=0.95,top=0.95,bottom=0.12)
    py.errorbar(i_bins,ave_dens_all,yerr=std_dens_all,fmt='k.')
    py.xlabel(r'Inclination bin (deg)')
    py.ylabel(r'Average Density per Inclination Bin (stars deg$^{-2}$)')
    py.axis([-20,180,0.0,max(ave_dens_all+std_dens_all)+0.001])
    py.savefig('/%s/%save_density_per_inclination_bin.png' % (outdir, strbin))
    py.close()

    # Plot histograms of the peak distributions for each inclination bin
    denspk_all = [disk1pk, disk2pk, disk3pk, disk4pk, disk5pk, disk6pk, disk7pk, disk8pk, disk9pk] 
    py.clf()
    py.figure(figsize=(10,10))
    py.subplots_adjust(wspace=0.3,hspace=0.3,left=0.1,right=0.9,top=0.9,bottom=0.1)
    py.rc('xtick', labelsize=10)
    py.rc('ytick', labelsize=10)
    for ii in range(len(denspk_all)):
        py.subplot(3,3,ii+1)
        nn,bb,pp = py.hist(denspk_all[ii],bins=50,histtype='step',color='k')
        xinc = ( denspk_all[ii].max() - denspk_all[ii].min() ) / 2.
        thePlot = py.gca()
        thePlot.xaxis.set_major_formatter(py.FormatStrFormatter('%5.3e'))
        thePlot.get_xaxis().set_major_locator(py.MultipleLocator(xinc))
        py.title('i = %3i - %3i' % (i_bins[ii],i_bins[ii]+20),fontsize=12)
        if ii == 3:
            py.ylabel('N')
        if ii == 7:
            py.xlabel('Peak Density (stars/deg^2)')
    py.suptitle('Isotropic Simulations - Peak Density by Inclination')
    py.savefig('/%s/%sisotropic_peak_hist.png' % (outdir, strbin))
    py.close()
    usetexFalse()
    
    #pdb.set_trace()


def run_isotropic_results_impatient():
    """
    Similar to run_isotropic_results() above, but just reads in all
    the HEALpix maps and prints the results to a file. The above function
    requires that I wait for all the maps to already be made, but this function
    just looks at the ones that have finished so far.
    """
    mockdir = 'sim_isotropic_big/'
    mockfile='isotropic_5000stars_mockdata.pickle'
    rootDir = '/'
    outroot = rootDir + mockdir + 'plots/'
    hpmaps = asciidata.open('/sim_isotropic_big/full_simdisks_run_so_far.txt')
    hpmap = hpmaps[0].tonumpy()
    nside = 64
    npix = healpy.nside2npix(nside)
    pixIdx = np.arange(npix, dtype=int)
    (i, o) = healpy.pix2ang(nside, pixIdx)
    i *= 180.0 / math.pi
    o *= 180.0 / math.pi
    
    out = open(rootDir + mockdir + 'full_116stars_nnDensity_impatient.dat', 'w')
    fmt = '%3i  %5.1f  %5.1f  %.2e  %.2e  %.2e  %5.2f\n'

    for ii in range(len(hpmap)):
        #hpfile = '%s/%s' % (rootDir, hpmap[ii])     

        # Now check the significance of any peak
        f1 = '%s' % hpmap[ii]
        f2 = str(hpmap[ii]).replace('neighbor', 'neighborStd')
        (disk, diskStd) = loadDiskDensity(npix, orbDir=rootDir+mockdir, aperture=False,
                                          file1=f1, file2=f2, simdisk=True)
        didx = disk.argmax()
        peak = disk[didx]
        idisk = i[didx]
        odisk = o[didx]
        print 'Peak found at (i,O) = (%5.1f, %5.1f) with %.2e  stars/deg^2' % \
              (idisk,odisk,peak)
        print
        print '%8.5f - Expected density for isotropic population with 116 stars' % \
              (116. / areaOnSky)

        # What is the measured density outside the disk region:
        avgBkg = 1.0
        stdBkg = 1.0

        # Iteratively compute background
        for n in range(2):
            idx = (np.where(disk < (avgBkg + (3.0 * stdBkg))))[0]

            avgBkg = disk[idx].mean()
            stdBkg = disk[idx].std(ddof=1)
            print 'trial = %2d   avg = %e   std = %e, rejecting %d of %d' % \
                  (n, avgBkg, stdBkg, (len(disk)-len(idx)), len(disk))

        signif = (peak - avgBkg) / stdBkg

        print ''
        print 'Average Background   rho = %e' % avgBkg
        print 'Stddev of Background rho = %e' % stdBkg
        print 'Density Ratio for CW disk = %5.2f' % signif

        # Write to file
        out.write(fmt % (ii, idisk, odisk, peak, avgBkg, stdBkg, signif))

    out.close()

    results = asciidata.open('/%s/full_116stars_nnDensity_impatient.dat' % mockdir)
    incl = results[1].tonumpy()
    Omega = results[2].tonumpy()
    pkDen = results[3].tonumpy() *1.e3
    avDen = results[4].tonumpy()
    sdDen = results[5].tonumpy()
    sig = results[6].tonumpy()

    py.clf()
    py.figure(figsize=(6,6))
    aa,bb,cc = py.hist(sig, bins=np.arange(0, sig.max()+1, 0.2), histtype='step',
                       normed=True,color='k')
    py.axis([0,sig.max()+1,0,aa.max()+0.2])
    py.xlabel('Overdensity')
    py.ylabel('PDF')
    py.savefig('/%s/full_isotropic_sig_peak_impatient.png' % mockdir)
    py.savefig('/%s/full_isotropic_sig_peak_impatient.eps' % mockdir)
    py.close()

    # How many times do we get a >X sigma peak? (where X is what we observed)
    # Use the observed sigma over the background (which we know is an overestimate)
    obs_sig_bkg = 11.5
    bs = np.where(sig > obs_sig_bkg)[0]
    print
    prob_sig = len(bs)*100. / len(hpmap)
    print 'Number of >%4.1f sigma results in an isotropic sim: %2i out of %2i (%4.1f percent)' % \
          (obs_sig_bkg, len(bs), len(hpmap), prob_sig)

def plot_vel_vectors():
    """
    This will over-write the file velVector_mockdata.png! Use with caution.
    Made this function to re-make this plot with black arrows instead of red.
    """
    mockdir = 'ecc_bias_circ/'
    outfile = 'ecc_bias_circ_mockdata.pickle'
    
    # Read in the mock data
    # Astrometric units are: arcsec, mas/yr, mas/yr^2
    mockdata = open(root + alnDir + mockdir + outfile)
    mbh = pickle.load(mockdata)
    dist = pickle.load(mockdata)
    orb_all = pickle.load(mockdata)
    sma_all = pickle.load(mockdata) # AU
    xM = pickle.load(mockdata) # (+x to east)
    yM = pickle.load(mockdata)
    zM = pickle.load(mockdata)
    vxM = pickle.load(mockdata) # (+x to east)
    vyM = pickle.load(mockdata)
    vzM = pickle.load(mockdata) # mas/yr
    axM = pickle.load(mockdata) # (+x to east)
    ayM = pickle.load(mockdata)
    azM = pickle.load(mockdata)
    t0M = pickle.load(mockdata) # periapse passage
    t_obs = pickle.load(mockdata) # observational time for mock data point
    mockdata.close()

    py.clf()
    py.figure(figsize=(6,6))
    py.quiver([xM], [yM], [vxM], [vyM], headwidth=1.5, minshaft=1.5,
              color='black', units='y', angles='xy', scale=5)
    py.plot([0],[0],'rx')
    py.text(10,13,'i = 130 deg', fontsize=10)
    py.text(10,12,'O =  96 deg', fontsize=10)
    py.text(10,11,'e = 0.0', fontsize=10)
    py.axis([15.0, -15.0, -15.0, 15.0])
    py.xlabel('X (arcsec)',fontsize=16)
    py.ylabel('Y (arcsec)',fontsize=16)
    py.title('Mock Data Velocities',fontsize=16)
    py.savefig('%s/%s/%s/plots/velVector_mockdata.png' % (root,alnDir,mockdir))
    py.close()
    
        
def sim_disk_features(non_disk_fov=False):
    """
    Find features in HEALpix maps from simulations and calculate their significance
    over the background.
    """
    # Disk solution
    nside = 64
    npix = healpy.nside2npix(nside)
    pixIdx = np.arange(0, npix)

    (i, o) = healpy.pix2ang(nside, pixIdx)
    i *= 180.0 / math.pi
    o *= 180.0 / math.pi

    if non_disk_fov == True:
        # Look at the density maps from the disk fraction sims (5-55%)
        # that were made using only non-disk members within observed FOV
        suffix = ['10','15','20','25','30','35','40','45','50','55']
        sims = 'disk' 
        bins = ['non_disk_fov']
        nstars = [108,102,96,90,84,78,72,66,60,54] # not on disk
    else:
        # Look at the density maps from the 20% disk simulations (11 of them were run)
        bins = ['middle_simdisk']
        #bins = ['inner_simdisk','middle_simdisk','outer_simdisk']
        sims = 'disk20_'
        suffix = np.arange(0,11)
        nstars = [40, 40, 40] # per radial bin

    for ii in range(len(suffix)): 
        if ((non_disk_fov == False) & (ii == 0)):
            orbDir = 'sim_diskFraction3/disk20/'
        else:
            orbDir = 'sim_diskFraction3/'+ sims + str(suffix[ii]) + '/'
        #orbDir = 'sim_isotropic2/'
        #bins = ['outer_simdiskfov_observed']
        #nstars = [46]
        print '*****Simulation from %s: *****' % orbDir
        for rr in range(len(bins)):
            print '    %s radial bin' % bins[rr]
            f1 = bins[rr] + '.neighbor.dat'
            f2 = bins[rr] + '.neighborStd.dat'
            (disk, diskStd) = loadDiskDensity(npix, orbDir=orbDir, aperture=False,
                                              file1=f1, file2=f2, simdisk=True)

            didx = disk.argmax()
            peak = disk[didx]
            idisk = i[didx]
            odisk = o[didx]
            print 'Peak found at (i,O) = (%5.1f, %5.1f) with %.2e  stars/deg^2' % \
                  (idisk,odisk,peak)
            print
            print '%8.5f - Expected density for isotropic population with %i stars' % \
                  ((nstars[rr] / areaOnSky), nstars[rr])

            # What is the measured density outside the disk region:
            avgBkg = 1.0
            stdBkg = 1.0

            # Iteratively compute background
            for n in range(2):
                idx = (np.where(disk < (avgBkg + (3.0 * stdBkg))))[0]

                avgBkg = disk[idx].mean()
                stdBkg = disk[idx].std(ddof=1)
                print 'trial = %2d   avg = %e   std = %e, rejecting %d of %d' % \
                      (n, avgBkg, stdBkg, (len(disk)-len(idx)), len(disk))

            print ''
            print 'Average Background   rho = %e' % avgBkg
            print 'Stddev of Background rho = %e' % stdBkg
            print 'Density Ratio for CW disk = %5.2f' % ((peak - avgBkg) / stdBkg)
            print
            print 'Searching for other features...'
            
            ipeak = 40. # input inclination to start search
            opeak = 135. # input Omega to start search

            sini = np.sin(np.radians(ipeak))
            cosi = np.cos(np.radians(ipeak))
            sino = np.sin(np.radians(opeak))
            coso = np.cos(np.radians(opeak))

            angle = np.zeros(len(i), float)
            for k in range(len(i)):
                inc = np.radians(i[k])
                ome = np.radians(o[k])
        
                angle[k] = np.arccos(sini * coso * np.sin(inc) * np.cos(ome) + \
                                  sini * sino * np.sin(inc) * np.sin(ome) + \
                                  cosi * np.cos(inc))
                angle[k] = np.degrees(angle[k])
        
            aid = (np.where(angle < 20.0))[0]
            did = (np.where(angle < 5.0))[0]
            mid = disk[aid].argmax()
            maxDensity = (disk[aid])[mid]
            imax = (i[aid])[mid]
            omax = (o[aid])[mid]
        
            print 'At the location i = %5.3f and o = %5.3f' % (ipeak, opeak)
            print '   density found is rho = %e  stars/deg^2' % disk[did[0]]
            print 'Within 20 deg radius of this direction:'
            print '   Maximum Density at   i = %5.3f and o = %5.3f' % (imax, omax)
            print '   Maximum Density is   rho = %e  stars/deg^2' % maxDensity
        
            # What is this limit in terms of number of stars within XX deg
            radiusForDisk = 20.0 # disk thickness proposed by Paumard+06 for CCW disk
            numStars = maxDensity * solidAngleOfCone(radiusForDisk)
            print 'Number of Stars within %2d deg:   %4.1f' % (radiusForDisk, numStars)
        
            # What is the measured density outside this region:
            nid = (np.where(angle > radiusForDisk))[0]
            avgBkg = 1.0
            stdBkg = 1.0

            # Iteratively compute background
            for n in range(2):
                bkg = disk[nid]
                idx = (np.where(bkg < (avgBkg + (3.0 * stdBkg))))[0]
        
                avgBkg = bkg[idx].mean()
                stdBkg = bkg[idx].std()
                #print 'trial = %2d   avg = %e   std = %e' % (n, avgBkg, stdBkg)
                print 'trial = %2d   avg = %e   std = %e, rejecting %d of %d' % \
                      (n, avgBkg, stdBkg, (len(disk)-len(idx)), len(disk))
        
            print ''
            print 'Average Background   rho = %e  stars/deg^2' % avgBkg
            print 'Stddev of Background rho = %e  stars/deg^2' % stdBkg
            print 'Density Ratio for other feature    = %5.2f' % ((maxDensity - avgBkg) / stdBkg)
            print ''
        
            # Save the density ratio (i.e., the significance of the peak)

    print ''


def simulate_disk(nDiskStars=100,nIsoStars=None,mockdir='',
                  outfile='',ntrials=10**4,bothZ=True,
                  vel_kick=False, errorfile='pos_errorRange_vs_radius.dat',
                  sigma=5.0,incl_in=130.2, O_in=96.3,run_sim=False,
                  makeHealPix=False,dumpPickle=False,loadPickle=True,
                  mass=4.6e6,dist=8232.9):
    """
    Run a simulation that produces stars with orbits consistent with the CW disk.
    Pull the mock data that corresponds to the time near the star's apoapse, which
    is when radial velocity is the smallest.
    Run the orbit analysis on these stars alone, as we usually do. Assigns errors
    based on radius (as done in sythesis_sim.py).

    O_in and incl_in should be the nominal Omega and inclination values
    (before any velocity kick).  If a velocity kick was given, the true
    inclination and Omega will be determined in the code.

    Also plots up all the results from the simulation. Can choose to just
    make plots and not run simulation by setting run_sim = False (def).
    """

    import sythesis_sim as sim
    cc = objects.Constants()

    #mockdir = 'sim_disk_zSign/'
    #outfile = 'pos_neg_z_mockdata.pickle'
    #mockdir = 'sim_apoapse_stars/'
    #outfile = 'apoapse_mockdata.pickle'
    #mockdir = 'sim_disk_isoPop3/'
    #outfile = 'ecc_0.32_isoPop_mockdata.pickle'
    #mockdir = 'sim_disk_Omega35/'
    #outfile = 'ecc_0.32_Omega35_mockdata.pickle'
    #mockdir = 'ecc_bias_sim_meanEcc/'
    #outfile = 'meanObservedEccDisk_mockdata.pickle'
    #mockdir = 'ecc_bias_sim_0.3_0.09err/'
    #outfile = 'ecc_err_FLATMC_mockdata.pickle'
    #mockdir = 'sim_vkick_fracCircVel/vkick_0.07frac/sim_vkick_0.32/'
    #outfile = 'ecc_0.32_vkick_mockdata.pickle'
    
    asy_to_kms = dist * cc.cm_in_au / (1.e5 * cc.sec_in_yr)

    if run_sim == True:
        # Create the mock data for circular orbits
        #sim.mock_data(nstars=nstars, e=0.3, tObs='apoapse', mockdir=mockdir,
        #             outfile=outfile, vel_kick=False)
        sim.mock_data(nstars=nDiskStars, e=0.3, mockdir=mockdir,
                     outfile=outfile, vel_kick=False)
       
        # Run orbit simulation
        mc = sim.simulate_orbits(ntrials=ntrials, mockdir=mockdir, mockfile=outfile,
                                 errorfile=errorfile, sigma=sigma)
        mc.run()
    
    # Read in the mock data
    # Astrometric units are: arcsec, mas/yr, mas/yr^2
    mockdata = open(root + alnDir + mockdir + outfile)
    mbh = pickle.load(mockdata)
    dist = pickle.load(mockdata)
    orb_all = pickle.load(mockdata)
    sma_all = pickle.load(mockdata) # AU
    xM = pickle.load(mockdata) # (+x to east)
    yM = pickle.load(mockdata)
    zM = pickle.load(mockdata)
    vxM = pickle.load(mockdata) # (+x to east)
    vyM = pickle.load(mockdata)
    vzM = pickle.load(mockdata) # mas/yr
    axM = pickle.load(mockdata) # (+x to east)
    ayM = pickle.load(mockdata)
    azM = pickle.load(mockdata)
    t0M = pickle.load(mockdata) # periapse passage
    t_obs = pickle.load(mockdata) # observational time for mock data point
    mockdata.close()

    r2d = np.sqrt(xM**2 + yM**2)

    print 'Total of %d stars' % len(xM)

    # Get total number of stars (in case of disk+isotropic simulations)
    # Also set the index of where disk stars begin and end, and
    # where isotropic stars begin and end
    if nIsoStars != None:
        nstars = nDiskStars + nIsoStars
    else:
        nstars = nDiskStars

    zOut = np.zeros((nstars, ntrials), dtype=float)
    eOut = np.zeros((nstars, ntrials), dtype=float)
    incl = np.zeros((nstars, ntrials), dtype=float)
    Omega = np.zeros((nstars, ntrials), dtype=float)
    ipeak = np.zeros(nstars, dtype=float)
    Opeak = np.zeros(nstars, dtype=float)
    med_incl = np.zeros(nstars, dtype=float)
    med_Omega = np.zeros(nstars, dtype=float)
    ave_incl = np.zeros(nstars, dtype=float)
    ave_Omega = np.zeros(nstars, dtype=float)
    std_incl = np.zeros(nstars, dtype=float)
    std_Omega = np.zeros(nstars, dtype=float)
    pIncAve = np.zeros(nstars, dtype=float)
    nIncAve = np.zeros(nstars, dtype=float)
    pOmAve = np.zeros(nstars, dtype=float)
    nOmAve = np.zeros(nstars, dtype=float)
    pIncStd = np.zeros(nstars, dtype=float)
    nIncStd = np.zeros(nstars, dtype=float)
    pOmStd = np.zeros(nstars, dtype=float)
    nOmStd = np.zeros(nstars, dtype=float)
    I_true = np.zeros(nstars, dtype=float)
    O_true = np.zeros(nstars, dtype=float)
    e_true = np.zeros(nstars, dtype=float)

    # Setup i/Omega for each pixel on sky
    nside = 64
    npix = healpy.nside2npix(nside)
    (iheal, oheal) = healpy.pix2ang(nside, np.arange(0, npix))
    iheal *= rad2deg
    oheal *= rad2deg

    outroot = root + alnDir + mockdir + 'plots/'

    if vel_kick == True:
        unb = []
        new_orb = []

    if loadPickle == False:
        # Plot up the results
        for ss in range(nstars):
            if vel_kick == True:
                # What was the input orbit? This should only be different
                # than the assumed orbit if we gave the stars an extra
                # velocity kick. This will not be in orb_all b/c orb_all
                # has the orbit that produced the mock data before a velocity
                # kick was added
                rvec = np.array([xM[ss], yM[ss], zM[ss]])
                vvec = np.array([vxM[ss]/1.e3*asy_to_kms, vyM[ss]/1.e3*asy_to_kms, vzM[ss]/1.e3*asy_to_kms])
                revec = np.zeros(3, dtype=float)
                vevec = np.zeros(3, dtype=float)
                try:
                    newOrb = orbits.Orbit()
                    newOrb.xyz2kep(rvec, vvec, revec, vevec, t_obs[0], mass=mass, dist=dist)
                except ValueError:
                    newOrb = None
                    print 'Star %i -- velocity kick produces unbound orbit!' % ss
                    unb = np.concatenate([unb, [ss]])
                new_orb = np.concatenate([new_orb, [newOrb]])


            pdffile = '%s/%s/%s/star%s.mc.dat' % (root, alnDir, mockdir, str(ss))
            # temp
            #hpfile = '%s/%s/%s/plots/HEALpixMaps/star%s_disk_mc_heal.dat.png' %\
            #          (root, alnDir, mockdir, str(ss))
            #if os.path.exists(hpfile) == False:
            #    print ss
            #    pdb.set_trace()
            #else:
            #    continue
            # end temp
    
            pdf = pickle.load(open(pdffile))
    
            # Read in the orbital parameters from the simulation
            incl[ss,:] = pdf.i
            Omega[ss,:] = pdf.o # big omega
            zOut[ss,:] = pdf.z
            eOut[ss,:] = pdf.e
    
            if makeHealPix == True:
                # temporary, to fix bug:
                #if 'disk15' in mockdir:
                #    if ss > 44:
                #    # end temporary, to fix bug:
                # Make a HEALpix map
                hp = makePdfHealpix('star'+str(ss), pdf, outroot+'HEALpixMaps/',
                                    ntrials=ntrials, nside=64,makeplot=True)
    
            # Next get orbital info
            hpfile = '%s/%s/%s/plots/HEALpixMaps/star%s_disk_mc_heal.dat' %\
                      (root, alnDir, mockdir, str(ss))
            hp = np.fromfile(hpfile, dtype=float)
    
            # Find the peak of the PDF
            sid = (hp.argsort())[::-1]  # reverse sort
            peakPix = sid[0]
            ipeak[ss] = iheal[peakPix]
            Opeak[ss] = oheal[peakPix]
    
            #i_inS = orb_all[ss].i
            #O_inS = orb_all[ss].o

            if vel_kick == True:
                if ss in unb:
                    print 'Star %i UNBOUND due to velocity kick'
                    I_true[ss] = -1
                    O_true[ss] = -1
                else:
                    I_true[ss] = newOrb.i
                    O_true[ss] = newOrb.o
                    e_true[ss] = newOrb.e
            else:
                I_true[ss] = orb_all[ss].i
                O_true[ss] = orb_all[ss].o
                e_true[ss] = orb_all[ss].e


            # Save off the average inclinations and Omegas for +z, -z
            pz = np.where(zOut[ss,:] >= 0.0)[0] 
            nz = np.where(zOut[ss,:] < 0.0)[0]
            pIncAve[ss] = incl[ss,pz].mean()
            pOmAve[ss] = Omega[ss,pz].mean()
            nIncAve[ss] = incl[ss,nz].mean()
            nOmAve[ss] = Omega[ss,nz].mean()
            pIncStd[ss] = incl[ss,pz].std(ddof=1)
            pOmStd[ss] = Omega[ss,pz].std(ddof=1)
            nIncStd[ss] = incl[ss,nz].std(ddof=1)
            nOmStd[ss] = Omega[ss,nz].std(ddof=1)

            # Save off the average inclinations and Omegas for all z
            med_incl[ss] = np.median(incl[ss,:])
            med_Omega[ss] = np.median(Omega[ss,:])
            ave_incl[ss] = incl[ss,:].mean()
            std_incl[ss] = incl[ss,:].std(ddof=1)
            ave_Omega[ss] = Omega[ss,:].mean()
            std_Omega[ss] = Omega[ss,:].std(ddof=1)

            py.clf()
            py.figure(figsize=(8,12))
            py.subplots_adjust(left=0.1, right=0.95, top=0.94, bottom=0.07,
                               wspace=0.3, hspace=0.3)
            py.subplot(3,2,1)
            py.quiver([xM], [yM], [vxM], [vyM], headwidth=1.5, minshaft=1.5,
              color='black', units='y', angles='xy', scale=5)
            py.quiver([xM[ss]], [yM[ss]], [vxM[ss]], [vyM[ss]], headwidth=1.5,
                      minshaft=1.5, color='red', units='y', angles='xy', scale=5)
            #if ((str(ss) == '15') | (str(ss) == '22')): # for disk fraction 20%
            if (str(ss) == '27') | (str(ss) == '9'): # for disk fraction 40%
                py.plot(xM[ss],yM[ss],'ro',mfc='None',mec='r',mew=1.5,ms=14)
            # Plot the line of nodes:
            xp1 = 15.0*np.cos(np.radians(90.-O_true[ss]))
            yp1 = 15.0*np.sin(np.radians(90.-O_true[ss]))
            xp2 = -15.0*np.cos(np.radians(90.-O_true[ss]))
            yp2 = -15.0*np.sin(np.radians(90.-O_true[ss]))
            py.plot([xp1,xp2],[yp1,yp2],'k--',lw=2)
            py.xlabel('X (arcsec)',fontsize=16)
            py.ylabel('Y (arcsec)',fontsize=16)
            py.axis([15.0, -15.0, -15.0, 15.0])
            py.subplot(3,2,3)
            nn, bb, pp = py.hist(incl[ss,:].flatten(), bins=np.arange(0, 181, 1.0),
                         histtype='step',color='k',normed=True)
            #py.plot([incl_in,incl_in],[0,nn.max()],'k--')
            if I_true[ss] < 0:
                py.text(100,nn.max()-0.03, 'Unbound star', fontsize=10)
            else:
                py.plot([I_true[ss],I_true[ss]],[0,nn.max()],'k--')
            py.axis([0, 180, 0, nn.max()+0.01])
            py.xlabel('Inclination (deg)')
            py.title('Mock Incl = %5.1f deg' % I_true[ss], fontsize=12)
            py.subplot(3,2,4)
            nn, bb, pp = py.hist(Omega[ss,:].flatten(), bins=np.arange(0, 370, 10.0),
                         histtype='step',color='k',normed=True)
            #py.plot([O_in,O_in],[0,nn.max()],'k--')
            py.plot([O_true[ss],O_true[ss]],[0,nn.max()],'k--')
            py.axis([0, 360, 0, nn.max()+0.01])
            py.xlabel('PA to the Ascending Node (deg)')
            py.title('Mock Omega = %5.1f deg' % O_true[ss], fontsize=12)
            # Plot incl & Omega against the output z values
            py.subplot(3,2,5)
            py.plot(zOut[ss,:],incl[ss,:],'k.',mfc='None',mec='k',ms=4)
            #py.plot(zOut[ss,:],incl[ss,:],'k.')
            if I_true[ss] >= 0:
                py.plot([zOut[ss,:].min()-0.1,zOut[ss,:].max()+0.1],[I_true[ss],I_true[ss]],'k--')
                py.plot(zM[ss],I_true[ss],'rD',ms=7)
                #py.plot([0],I_true[ss],'rx',ms=7)
            py.xlabel('Z output (arcsec)')
            py.ylabel('Inclination (deg)')
            py.subplot(3,2,6)
            py.plot(zOut[ss,:],Omega[ss,:],'k.',mfc='None',mec='k',ms=4)
            #py.plot(zOut[ss,:],Omega[ss,:],'k.')
            py.plot([zOut[ss,:].min()-0.1,zOut[ss,:].max()+0.1],[O_true[ss],O_true[ss]],'k--')
            py.plot(zM[ss],O_true[ss],'rD',ms=7)
            #py.plot([0],O_true[ss],'rx',ms=7)
            py.xlabel('Z output (arcsec)')
            py.ylabel('PA to the Ascending Node (deg)')
            py.savefig('%s/star%s_inclOmega.png' % (outroot, str(ss)))
            #if (str(ss) == '15') | (str(ss) == '22'): # for disk fraction 20%
            if (str(ss) == '27') | (str(ss) == '9'): # for disk fraction 40%
                print 'star %i: z = %5.2f arcsec' % (ss, zM[ss])
                py.savefig('%s/eps/star%s_inclOmega.eps' % (outroot, str(ss)))
            py.close()

            # Plot eccentricity vs. z
            py.clf()
            py.plot(zOut[ss,:],eOut[ss,:],'k.')
            py.xlabel('Z output (arcsec)')
            py.ylabel('Eccentricity')
            py.title('Input e = %4.2f' % e_true[ss])
            py.axis([zOut[ss,:].min(),zOut[ss,:].max(),0,1])
            py.savefig('%s/star%s_eccVsZ.png' % (outroot, str(ss)))
            py.close()

        if dumpPickle == True:
            pickFile = open('%s/simulation_results.pickle' % outroot, 'w')
            pickle.dump(incl, pickFile)
            pickle.dump(Omega, pickFile)
            pickle.dump(ave_Omega, pickFile)
            pickle.dump(std_Omega, pickFile)
            pickle.dump(ave_incl, pickFile)
            pickle.dump(std_incl, pickFile)
            pickle.dump(ipeak, pickFile)
            pickle.dump(Opeak, pickFile)
            pickle.dump(zOut, pickFile)
            pickle.dump(pIncAve, pickFile)
            pickle.dump(nIncAve, pickFile)
            pickle.dump(pOmAve, pickFile)
            pickle.dump(nOmAve, pickFile)
            pickle.dump(pIncStd, pickFile)
            pickle.dump(nIncStd, pickFile)
            pickle.dump(pOmStd, pickFile)
            pickle.dump(nOmStd, pickFile)
            pickle.dump(I_true, pickFile)
            pickle.dump(O_true, pickFile)
            pickFile.close()

    elif loadPickle == True:
        pickFile = open('%s/simulation_results.pickle' % outroot)
        incl = pickle.load(pickFile)
        Omega = pickle.load(pickFile)
        ave_Omega = pickle.load(pickFile) # only really makes sense if one
        				  # sign for z was used.
        std_Omega = pickle.load(pickFile)
        ave_incl = pickle.load(pickFile)
        std_incl = pickle.load(pickFile)
        ipeak = pickle.load(pickFile)
        Opeak = pickle.load(pickFile)
        zOut = pickle.load(pickFile)
        pIncAve = pickle.load(pickFile)
        nIncAve = pickle.load(pickFile)
        pOmAve = pickle.load(pickFile)
        nOmAve = pickle.load(pickFile)
        pIncStd = pickle.load(pickFile)
        nIncStd = pickle.load(pickFile)
        pOmStd = pickle.load(pickFile)
        nOmStd = pickle.load(pickFile)
        I_true = pickle.load(pickFile)
        O_true = pickle.load(pickFile)
        pickFile.close()
    
    # Identify the biased sources assuming Gaussian errors first:
    # How many sigma away from input Omega is it?
    # The way we determine the bias depends on whether the simulation
    # used both positive and negative z's, or just one or the other:
    if bothZ == False:
        # Identify the sources with biased Omega
        # How many sigma away from input Omega is it?
        # Have to identify the unbound cases that have I_true or O_true = -1
        # These are unbound b/c of the velocity kick, very few cases of this
        OsigG = (ave_Omega - O_true) / std_Omega
        bOmG = np.where((np.abs(OsigG) > 1.0) & (O_true >= 0))[0]
        # Identify the sources with biased inclination
        # How many sigma away from input inclination is it?
        IsigG = (ave_incl - I_true) / std_incl
        bIncG = np.where((np.abs(IsigG) > 1.0) & (I_true >= 0))[0]
        # Identify which stars are biased in both i and Omega
        bothG = np.where((np.abs(IsigG) > 1.0) & (np.abs(OsigG) > 1.0))[0]

        # Now define biased sources as those that are > 15 deg from input values
        bOmP = np.where((np.abs(Opeak-O_true) > 15.) & (O_true >= 0))[0]
        bIncP = np.where((np.abs(ipeak-I_true) > 15.) & (I_true >= 0))[0]
        bothP = np.intersect1d(bOmP, bIncP)
    else:
        # Look at both the +z and -z solutions
        pIsig = (pIncAve - I_true) / pIncStd
        nIsig = (nIncAve - I_true) / nIncStd

        pOsig = (pOmAve - O_true) / pOmStd
        nOsig = (nOmAve - O_true) / nOmStd

        bIncG = np.where((np.abs(pIsig) > 1.0) & (np.abs(nIsig) > 1.0) & (I_true >= 0))[0]
        # For most stars, the inclination isn't double peaked, so look at all soln:
        #bIncG = np.where(np.abs(med_incl - I_true) > 30.0)[0]
        bOmG = np.where((np.abs(pOsig) > 1.0) & (np.abs(nOsig) > 1.0) & (I_true >= 0))[0]

        # Which are biased in both?
        bothG = np.intersect1d(bIncG,bOmG)

    # Compute the quantity used to calculate Omega:
    Onum = (vxM/1.e3 * zM) - (xM * vzM/1.e3)
    Oden = (vyM/1.e3 * zM) - (yM * vzM/1.e3)
    Oquant = Onum / Oden

    import matplotlib.font_manager
    prop = matplotlib.font_manager.FontProperties(size=12)

    # Plot this quantity against the average Omega
    py.clf()
    py.figure(figsize=(12,4))
    py.subplots_adjust(left=0.07, right=0.95, top=0.9, bottom=0.13,
                       wspace=0.3, hspace=0.3)
    py.subplot(1,3,1)
    if nIsoStars != None:
        py.semilogy(ave_Omega[nDiskStars:], np.abs(Onum[nDiskStars:]), 'b.',label='Iso')
        py.semilogy(ave_Omega[0:nDiskStars], np.abs(Onum[0:nDiskStars]), 'r.',label='Disk')
    else:
        py.semilogy(ave_Omega, np.abs(Onum), 'k.')
    py.plot([O_in,O_in],[np.abs(Onum).min(),np.abs(Onum).max()],'k--')
    if nIsoStars != None:
        py.legend(numpoints=1,prop=prop,fancybox=True,loc=4)
    py.xlabel('Ave Omega from Simulation (deg)')
    py.ylabel('Omega Numerator')
    #py.axis([50,130,-10,10])
    py.subplot(1,3,2)
    if nIsoStars != None:
        py.semilogy(ave_Omega[nDiskStars:], np.abs(Oden[nDiskStars:]), 'b.')
        py.semilogy(ave_Omega[0:nDiskStars], np.abs(Oden[0:nDiskStars]), 'r.')
    else:
        py.semilogy(ave_Omega, np.abs(Oden), 'k.')
    py.plot([O_in,O_in],[np.abs(Oden).min(),np.abs(Oden).max()],'k--')
    py.xlabel('Ave Omega from Simulation (deg)')
    py.ylabel('Omega Denominator')
    #py.axis([50,130,-10,10])
    py.subplot(1,3,3)
    if nIsoStars != None:
        py.semilogy(ave_Omega[nDiskStars:], np.abs(Oquant[nDiskStars:]), 'b.')
        py.semilogy(ave_Omega[0:nDiskStars], np.abs(Oquant[0:nDiskStars]), 'r.')
    else:
        py.semilogy(ave_Omega, np.abs(Oquant), 'k.')
    py.plot([O_in,O_in],[np.abs(Oquant).min(),np.abs(Oquant).max()],'k--')
    py.xlabel('Ave Omega from Simulation (deg)')
    py.ylabel('Omega Quantity')
    py.savefig('%s/OmegaQuantity_vs_aveOmega.png' % outroot)
    py.close()

    # Plot the average incl & Omega against vz
    py.clf()
    py.figure(figsize=(10,5))
    py.subplots_adjust(left=0.07, right=0.95, top=0.9, bottom=0.13,
                       wspace=0.3, hspace=0.3)
    py.subplot(1,2,1)
    if nIsoStars != None:
        py.errorbar(vzM[nDiskStars:], ave_incl[nDiskStars:], fmt='b.',
                    yerr=std_incl[nDiskStars:], label='Iso')
        py.errorbar(vzM[0:nDiskStars], ave_incl[0:nDiskStars], fmt='r.',
                    yerr=std_incl[0:nDiskStars],label='Disk')
    else:   
        py.errorbar(vzM, ave_incl, fmt='k.',yerr=std_incl)
    py.plot([-30,30],[incl_in,incl_in],'g--')
    if nIsoStars != None:
        py.legend(numpoints=1,prop=prop,fancybox=True,loc=1)
    py.xlabel('RV (mas/yr)')
    py.ylabel('Average Inclination (deg)')
    py.title('Input Inclination = %3i deg' % incl_in)
    py.subplot(1,2,2)
    if nIsoStars != None:
        py.errorbar(vzM[nDiskStars:], ave_Omega[nDiskStars:], fmt='b.',
                    yerr=std_Omega[nDiskStars:], label='Iso')
        py.errorbar(vzM[0:nDiskStars], ave_Omega[0:nDiskStars], fmt='r.',
                    yerr=std_Omega[0:nDiskStars],label='Disk')
    else:   
        py.errorbar(vzM, ave_Omega, fmt='k.',yerr=std_Omega)
    py.plot([-30,30],[O_in,O_in],'g--')
    py.xlabel('RV (mas/yr)')
    py.ylabel('Average Omega (deg)')
    py.title('Input Omega = %3i deg' % O_in)
    py.savefig('%s/ave_IO_vz.png' % outroot)
    py.close()

    # Plot the average incl & Omega against vx
    py.clf()
    py.figure(figsize=(10,5))
    py.subplots_adjust(left=0.07, right=0.95, top=0.9, bottom=0.13,
                       wspace=0.3, hspace=0.3)
    py.subplot(1,2,1)
    if nIsoStars != None:
        py.errorbar(vxM[nDiskStars:], ave_incl[nDiskStars:], fmt='b.',
                    yerr=std_incl[nDiskStars:], label='Iso')
        py.errorbar(vxM[0:nDiskStars], ave_incl[0:nDiskStars], fmt='r.',
                    yerr=std_incl[0:nDiskStars], label='Disk')
    else:
        py.errorbar(vxM, ave_incl, fmt='k.',yerr=std_incl)
    py.plot([-30,30],[incl_in,incl_in],'g--')
    if nIsoStars != None:
        py.legend(numpoints=1,prop=prop,fancybox=True,loc=1)
    py.xlabel('X Velocity (mas/yr)')
    py.ylabel('Average Inclination (deg)')
    py.title('Input Inclination = %3i deg' % incl_in)
    py.subplot(1,2,2)
    if nIsoStars != None:
        py.errorbar(vxM[nDiskStars:], ave_Omega[nDiskStars:], fmt='b.',
                    yerr=std_Omega[nDiskStars:], label='Iso')
        py.errorbar(vxM[0:nDiskStars], ave_Omega[0:nDiskStars], fmt='r.',
                    yerr=std_Omega[0:nDiskStars],label='Disk')
    else:   
        py.errorbar(vxM, ave_Omega, fmt='k.',yerr=std_Omega)
    py.plot([-30,30],[O_in,O_in],'g--')
    py.xlabel('X Velocity (mas/yr)')
    py.ylabel('Average Omega (deg)')
    py.title('Input Omega = %3i deg' % O_in)
    py.savefig('%s/ave_IO_vx.png' % outroot)
    py.close()

    # Plot the average incl & Omega against vy
    py.clf()
    py.figure(figsize=(10,5))
    py.subplots_adjust(left=0.07, right=0.95, top=0.9, bottom=0.13,
                       wspace=0.3, hspace=0.3)
    py.subplot(1,2,1)
    if nIsoStars != None:
        py.errorbar(vyM[nDiskStars:], ave_incl[nDiskStars:], fmt='b.',
                    yerr=std_incl[nDiskStars:], label='Iso')
        py.errorbar(vyM[0:nDiskStars], ave_incl[0:nDiskStars], fmt='r.',
                    yerr=std_incl[0:nDiskStars], label='Disk')
    else:
        py.errorbar(vyM, ave_incl, fmt='k.',yerr=std_incl)
    py.plot([-30,30],[incl_in,incl_in],'g--')
    if nIsoStars != None:
        py.legend(numpoints=1,prop=prop,fancybox=True,loc=1)
    py.xlabel('Y Velocity (mas/yr)')
    py.ylabel('Average Inclination (deg)')
    py.title('Input Inclination = %3i deg' % incl_in)
    py.subplot(1,2,2)
    if nIsoStars != None:
        py.errorbar(vyM[nDiskStars:], ave_Omega[nDiskStars:], fmt='b.',
                    yerr=std_Omega[nDiskStars:], label='Iso')
        py.errorbar(vyM[0:nDiskStars], ave_Omega[0:nDiskStars], fmt='r.',
                    yerr=std_Omega[0:nDiskStars],label='Disk')
    else:   
        py.errorbar(vyM, ave_Omega, fmt='k.',yerr=std_Omega)
    py.plot([-30,30],[O_in,O_in],'g--')
    py.xlabel('Y Velocity (mas/yr)')
    py.ylabel('Average Omega (deg)')
    py.title('Input Omega = %3i deg' % O_in)
    py.savefig('%s/ave_IO_vy.png' % outroot)
    py.close()

    # Plot the peak incl & Omega against vz
    py.clf()
    py.figure(figsize=(10,5))
    py.subplots_adjust(left=0.07, right=0.95, top=0.9, bottom=0.13,
                       wspace=0.3, hspace=0.3)
    py.subplot(1,2,1)
    if nIsoStars != None:
        py.plot(vzM[nDiskStars:], ipeak[nDiskStars:], 'b.', label='Iso')
        py.plot(vzM[0:nDiskStars], ipeak[0:nDiskStars], 'r.', label='Disk')
    else:   
        py.plot(vzM, ipeak, 'k.')
    py.plot([-30,30],[incl_in,incl_in],'g--')
    if nIsoStars != None:
        py.legend(numpoints=1,prop=prop,fancybox=True,loc=1)
    py.axis([-30,30,0,180])
    py.xlabel('RV (mas/yr)')
    py.ylabel('Peak Inclination (deg)')
    py.title('Input Inclination = %3i deg' % incl_in)
    py.subplot(1,2,2)
    if nIsoStars != None:
        py.plot(vzM[nDiskStars:], Opeak[nDiskStars:], 'b.', label='Iso')
        py.plot(vzM[0:nDiskStars], Opeak[0:nDiskStars], 'r.', label='Disk')
    else:   
        py.plot(vzM, Opeak, 'k.')
    py.plot([-30,30],[O_in,O_in],'g--')
    py.axis([-30,30,0,360])
    py.xlabel('RV (mas/yr)')
    py.ylabel('Peak Omega (deg)')
    py.title('Input Omega = %3i deg' % O_in)
    py.savefig('%s/peak_IO_vz.png' % outroot)
    py.close()

    # Plot peak incl & Omega against R2D
    py.clf()
    py.figure(figsize=(10,5))
    py.subplots_adjust(left=0.07, right=0.95, top=0.9, bottom=0.13,
                       wspace=0.3, hspace=0.3)
    py.subplot(1,2,1)
    if nIsoStars != None:
        py.plot(r2d[nDiskStars:], ipeak[nDiskStars:], 'b.', label='Iso')
        py.plot(r2d[0:nDiskStars], ipeak[0:nDiskStars], 'r.', label='Disk')
    else:   
        py.plot(r2d, ipeak, 'k.')
    py.plot([0,25],[incl_in,incl_in],'g--')
    if nIsoStars != None:
        py.legend(numpoints=1,prop=prop,fancybox=True,loc=1)
    #py.axis([-30,30,0,180])
    py.xlabel('Projected Radius (arcsec)')
    py.ylabel('Peak Inclination (deg)')
    py.title('Input Inclination = %3i deg' % incl_in)
    py.subplot(1,2,2)
    if nIsoStars != None:
        py.plot(r2d[nDiskStars:], Opeak[nDiskStars:], 'b.', label='Iso')
        py.plot(r2d[0:nDiskStars], Opeak[0:nDiskStars], 'r.', label='Disk')
    else:   
        py.plot(r2d, Opeak, 'k.')
    py.plot([0,25],[O_in,O_in],'g--')
    #py.axis([-30,30,0,360])
    py.xlabel('Projected Radius (arcsec)')
    py.ylabel('Peak Omega (deg)')
    py.title('Input Omega = %3i deg' % O_in)
    py.savefig('%s/peak_IO_r2d.png' % outroot)
    py.close()

    # Plot the peak incl & Omega against vx
    py.clf()
    py.figure(figsize=(10,5))
    py.subplots_adjust(left=0.07, right=0.95, top=0.9, bottom=0.13,
                       wspace=0.3, hspace=0.3)
    py.subplot(1,2,1)
    if nIsoStars != None:
        py.plot(vxM[nDiskStars:], ipeak[nDiskStars:], 'b.', label='Iso')
        py.plot(vxM[0:nDiskStars], ipeak[0:nDiskStars], 'r.', label='Disk')
    else:
        py.plot(vxM, ipeak, 'k.')
    py.plot([-30,30],[incl_in,incl_in],'g--')
    if nIsoStars != None:
        py.legend(numpoints=1,prop=prop,fancybox=True,loc=1)
    py.axis([-30,30,0,180])
    py.xlabel('X Velocity (mas/yr)')
    py.ylabel('Peak Inclination (deg)')
    py.title('Input Inclination = %3i deg' % incl_in)
    py.subplot(1,2,2)
    if nIsoStars != None:
        py.plot(vxM[nDiskStars:], Opeak[nDiskStars:], 'b.', label='Iso')
        py.plot(vxM[0:nDiskStars], Opeak[0:nDiskStars], 'r.', label='Disk')
    else:   
        py.plot(vxM, Opeak, 'k.')
    py.plot([-30,30],[O_in,O_in],'g--')
    py.axis([-30,30,0,360])
    py.xlabel('X Velocity (mas/yr)')
    py.ylabel('Peak Omega (deg)')
    py.title('Input Omega = %3i deg' % O_in)
    py.savefig('%s/peak_IO_vx.png' % outroot)
    py.close()

    # Plot the peak incl & Omega against vy
    py.clf()
    py.figure(figsize=(10,5))
    py.subplots_adjust(left=0.07, right=0.95, top=0.9, bottom=0.13,
                       wspace=0.3, hspace=0.3)
    py.subplot(1,2,1)
    if nIsoStars != None:
        py.plot(vyM[nDiskStars:], ipeak[nDiskStars:], 'b.', label='Iso')
        py.plot(vyM[0:nDiskStars], ipeak[0:nDiskStars], 'r.', label='Disk')
    else:
        py.plot(vyM, ipeak, 'k.')
    py.plot([-30,30],[incl_in,incl_in],'g--')
    py.axis([-30,30,0,180])
    if nIsoStars != None:
        py.legend(numpoints=1,prop=prop,fancybox=True,loc=1)
    py.xlabel('Y Velocity (mas/yr)')
    py.ylabel('Average Inclination (deg)')
    py.title('Input Inclination = %3i deg' % incl_in)
    py.subplot(1,2,2)
    if nIsoStars != None:
        py.plot(vyM[nDiskStars:], Opeak[nDiskStars:], 'b.', label='Iso')
        py.plot(vyM[0:nDiskStars], Opeak[0:nDiskStars], 'r.',label='Disk')
    else:   
        py.plot(vyM, Opeak, 'k.')
    py.plot([-30,30],[O_in,O_in],'g--')
    py.axis([-30,30,0,360])
    py.xlabel('Y Velocity (mas/yr)')
    py.ylabel('Average Omega (deg)')
    py.title('Input Omega = %3i deg' % O_in)
    py.savefig('%s/peak_IO_vy.png' % outroot)
    py.close()


    py.clf()
    py.figure(figsize=(10,5))
    py.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.13,
                       wspace=0.3, hspace=0.3)
    py.subplot(1,2,1)
    if nIsoStars != None:
        nn, bb, pp = py.hist(incl[0:nDiskStars].flatten(), bins=np.arange(0, 182, 2.0),
                    histtype='step',color='r',normed=False)
        nn, bb, pp = py.hist(incl[nDiskStars:].flatten(), bins=np.arange(0, 182, 2.0),
                    histtype='step',color='k',normed=False)
    else:
        nn, bb, pp = py.hist(incl.flatten(), bins=np.arange(0, 182, 2.0),
                    histtype='step',color='k',normed=False)
    py.xlim(0, 180)
    py.xlabel('Inclination (deg)')
    py.subplot(1,2,2)
    if nIsoStars != None:
        nn, bb, pp = py.hist(Omega[0:nDiskStars].flatten(), bins=np.arange(0, 362, 2.0),
                     histtype='step',color='r',normed=False)
        nn, bb, pp = py.hist(Omega[nDiskStars:].flatten(), bins=np.arange(0, 362, 2.0),
                     histtype='step',color='k',normed=False)
    else:
        nn, bb, pp = py.hist(Omega.flatten(), bins=np.arange(0, 362, 2.0),
                     histtype='step',color='k',normed=False)
    py.xlim(0, 360)
    py.xlabel('PA to the Ascending Node (deg)')
    py.savefig('%s/allStars_inclOmega_notNormed.png' % outroot)
    py.savefig('%s/eps/allStars_inclOmega_notNormed.eps' % outroot)
    py.close()
    #pdb.set_trace()

    py.clf()
    py.figure(figsize=(10,5))
    py.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.13,
                       wspace=0.3, hspace=0.3)
    py.subplot(1,2,1)
    nn, bb, pp = py.hist(incl.flatten(), bins=np.arange(0, 182, 2.0),
                histtype='step',color='k',normed=True)
    #py.axis([0, 180, 0, 0.035])
    py.xlim(0, 180)
    py.xlabel('Inclination (deg)')
    py.subplot(1,2,2)
    nn, bb, pp = py.hist(Omega.flatten(), bins=np.arange(0, 362, 2.0),
                 histtype='step',color='k',normed=True)
    #py.axis([0, 360, 0, 0.014])
    py.xlim(0, 360)
    py.xlabel('PA to the Ascending Node (deg)')
    py.savefig('%s/allStars_inclOmega.png' % outroot)
    py.savefig('%s/eps/allStars_inclOmega.eps' % outroot)
    py.close()

    py.clf()
    py.figure(figsize=(10,5))
    py.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.13,
                       wspace=0.3, hspace=0.3)
    py.subplot(1,2,1)
    if nIsoStars != None:
        nn, bb, pp = py.hist(ipeak[0:nDiskStars], bins=np.arange(0, 185, 5.0),
                     histtype='step',color='r',normed=False)
    nn, bb, pp = py.hist(ipeak, bins=np.arange(0, 185, 5.0),
                histtype='step',color='k',normed=False)
    py.ylabel('N Stars')
    py.axis([0, 180, 0, nn.max()+2])
    py.xlabel('Peak Inclination (deg)')
    py.subplot(1,2,2)
    if nIsoStars != None:
        nn, bb, pp = py.hist(Opeak[0:nDiskStars], bins=np.arange(0, 365, 5.0),
                     histtype='step',color='r',normed=False)
    nn, bb, pp = py.hist(Opeak, bins=np.arange(0, 365, 5.0),
                 histtype='step',color='k',normed=False)
    py.axis([0, 360, 0, nn.max()+2])
    py.xlabel('Peak PA to the Ascending Node (deg)')
    py.savefig('%s/allStars_peakInclOmega.png' % outroot)
    py.savefig('%s/eps/allStars_peakInclOmega.eps' % outroot)
    py.close()

    # Plot average Inclination vs. Omega
    py.clf()
    py.figure(figsize=(6,6))
    if nIsoStars != None:
        py.plot(ave_incl[nDiskStars:], ave_Omega[nDiskStars:], 'b.', label='Iso')
        py.plot(ave_incl[0:nDiskStars], ave_Omega[0:nDiskStars], 'r.', label='Disk')
    else:
        py.plot(ave_incl, ave_Omega, 'k.')
    py.plot([0,180],[O_in,O_in],'k--')
    py.plot([incl_in,incl_in],[0,400],'k--')
    py.xlabel('Average Inclination (deg)')
    py.ylabel('Average Omega (deg)')
    if nIsoStars != None:
        py.legend(numpoints=1,prop=prop,fancybox=True,loc=2)
    py.savefig('%s/ave_incl_vs_Omega.png' % outroot)
    py.close()

    # Plot ave incl & Omega against R2D
    py.clf()
    py.figure(figsize=(10,5))
    py.subplots_adjust(left=0.07, right=0.95, top=0.9, bottom=0.13,
                       wspace=0.3, hspace=0.3)
    py.subplot(1,2,1)
    if nIsoStars != None:
        py.plot(r2d[nDiskStars:], ave_incl[nDiskStars:], 'b.', label='Iso')
        py.plot(r2d[0:nDiskStars], ave_incl[0:nDiskStars], 'r.', label='Disk')
    else:   
        py.plot(r2d, ave_incl, 'k.')
    py.plot([0,25],[incl_in,incl_in],'g--')
    if nIsoStars != None:
        py.legend(numpoints=1,prop=prop,fancybox=True,loc=1)
    #py.axis([-30,30,0,180])
    py.xlabel('Projected Radius (arcsec)')
    py.ylabel('Average Peak Inclination (deg)')
    py.title('Input Inclination = %3i deg' % incl_in)
    py.subplot(1,2,2)
    if nIsoStars != None:
        py.plot(r2d[nDiskStars:], ave_Omega[nDiskStars:], 'b.', label='Iso')
        py.plot(r2d[0:nDiskStars], ave_Omega[0:nDiskStars], 'r.', label='Disk')
    else:   
        py.plot(r2d, ave_Omega, 'k.')
    py.plot([0,25],[O_in,O_in],'g--')
    #py.axis([-30,30,0,360])
    py.xlabel('Projected Radius (arcsec)')
    py.ylabel('Average Omega (deg)')
    py.title('Input Omega = %3i deg' % O_in)
    py.savefig('%s/ave_IO_r2d.png' % outroot)
    py.close()

    # Plot the peak of each star's PDF, Inclination vs. Omega 
    py.clf()
    py.figure(figsize=(6,6))
    if nIsoStars != None:
        py.plot(ipeak[nDiskStars:], Opeak[nDiskStars:], 'b.', label='Iso')
        py.plot(ipeak[0:nDiskStars], Opeak[0:nDiskStars], 'r.', label='Disk')
    else:
        py.plot(ipeak, Opeak, 'k.')
    py.plot([0,180],[O_in,O_in],'k--')
    py.plot([incl_in,incl_in],[0,400],'k--')
    py.xlabel('Peak Inclination (deg)')
    py.ylabel('Peak Omega (deg)')
    if nIsoStars != None:
        py.legend(numpoints=1,prop=prop,fancybox=True,loc=2)
    py.savefig('%s/peak_incl_vs_Omega.png' % outroot)
    py.close()

    # Plot all Inclination vs. Omega 
    py.clf()
    py.figure(figsize=(6,6))
    if nIsoStars != None:
        py.plot(incl[nDiskStars:,:], Omega[nDiskStars:,:], 'b.', label='Iso')
        py.plot(incl[0:nDiskStars,:], Omega[0:nDiskStars,:], 'r.', label='Disk')
    else:
        py.plot(incl, Omega, 'k.')
    py.plot([0,180],[O_in,O_in],'k--')
    py.plot([incl_in,incl_in],[0,400],'k--')
    py.xlabel('Inclination (deg)')
    py.ylabel('Omega (deg)')
    py.savefig('%s/all_incl_vs_Omega.png' % outroot)
    py.close()

    # Mark the biased stars on a velocity vector map
    # Bias defined using average and STD of i,O distributions
    py.clf()
    # dummy plot for the legend
    bias2 = py.arrow(0,0,1,1,color='red')
    py.clf()
    py.figure(figsize=(6,6))
    py.quiver([xM[nDiskStars:]], [yM[nDiskStars:]], [vxM[nDiskStars:]],
              [vyM[nDiskStars:]], headwidth=1.5, minshaft=1.5,
              color='black', units='y', angles='xy', scale=5)
    py.quiver([xM[0:nDiskStars]], [yM[0:nDiskStars]], [vxM[0:nDiskStars]],
              [vyM[0:nDiskStars]], headwidth=1.5, minshaft=1.5,
              color='red', units='y', angles='xy', scale=5)
    if nIsoStars != None:
        foo = np.where(bothG < nDiskStars)[0] # disk stars
        bothG = bothG[foo]
        foo = np.where(bOmG < nDiskStars)[0] # disk stars
        bOmG = bOmG[foo]
        foo = np.where(bIncG < nDiskStars)[0] # disk stars
        bIncG = bIncG[foo]
        # Mark the biased (in both i and O) vectors as red
        py.quiver([xM[bothG]], [yM[bothG]], [vxM[bothG]], [vyM[bothG]], headwidth=1.5,
                  minshaft=1.5, color='green', units='y', angles='xy', scale=5)
        # Mark the Omega-biased stars w/ a circle
        biasO = py.plot(xM[bOmG], yM[bOmG], 'go', ms=7)
        # Mark the inclination-biased stars w/ a smaller circle
        biasI = py.plot(xM[bIncG], yM[bIncG], 'mo', mfc='None', mec='m', ms=7)
    else:
        # Mark the biased (in both i and O) vectors as red
        py.quiver([xM[bothG]], [yM[bothG]], [vxM[bothG]], [vyM[bothG]], headwidth=1.5,
                  minshaft=1.5, color='green', units='y', angles='xy', scale=5)
        # Mark the Omega-biased stars w/ a circle
        biasO = py.plot(xM[bOmG], yM[bOmG], 'go', ms=7)
        # Mark the inclination-biased stars w/ a smaller circle
        biasI = py.plot(xM[bIncG], yM[bIncG], 'mo',mfc='None', mec='m', ms=7)
    py.plot([0],[0],'rx')
    py.axis([15.0, -15.0, -15.0, 15.0])
    #py.legend((biasO, biasI, bias2), ('Omega-bias' , 'Incl-bias', 'Both bias'),
    #          numpoints=1,loc=1,prop=prop,fancybox=True)
    py.xlabel('X (arcsec)',fontsize=16)
    py.ylabel('Y (arcsec)',fontsize=16)
    py.title('Mock Data Velocities',fontsize=16)
    py.savefig('%s/velVector_mockdata_biasedOrbit_aves.png' % outroot)
    py.close()

    if bothZ == False:
        # Mark the biased stars on a velocity vector map
        # Bias defined using the peak of i,O distributions
        py.clf()
        # dummy plot for the legend
        bias2 = py.arrow(0,0,1,1,color='red')
        py.clf()
        py.figure(figsize=(6,6))
        py.quiver([xM], [yM], [vxM], [vyM], headwidth=1.5, minshaft=1.5,
                  color='black', units='y', angles='xy', scale=5)
        if nIsoStars != None:
            foo = np.where(bothP < nDiskStars)[0] # disk stars
            bothP = bothP[foo]
            foo = np.where(bOmP < nDiskStars)[0] # disk stars
            bOmP = bOmP[foo]
            foo = np.where(bIncP < nDiskStars)[0] # disk stars
            bIncP = bIncP[foo]
            # Mark the biased (in both i and O) vectors as red
            py.quiver([xM[bothP]], [yM[bothP]], [vxM[bothP]], [vyM[bothP]], headwidth=1.5,
                      minshaft=1.5, color='green', units='y', angles='xy', scale=5)
            # Mark the Omega-biased stars w/ a circle
            biasO = py.plot(xM[bOmP], yM[bOmP], 'go', ms=5)
            # Mark the inclination-biased stars w/ a smaller circle
            biasI = py.plot(xM[bIncP], yM[bIncP], 'mo', mfc='None', mec='m', ms=7)
        else:
            # Mark the biased (in both i and O) vectors as green
            py.quiver([xM[bothP]], [yM[bothP]], [vxM[bothP]], [vyM[bothP]], headwidth=1.5,
                      minshaft=1.5, color='green', units='y', angles='xy', scale=5)
            # Mark the Omega-biased stars w/ a circle
            biasO = py.plot(xM[bOmP], yM[bOmP], 'go', ms=5)
            # Mark the inclination-biased stars w/ a smaller circle
            biasI = py.plot(xM[bIncP], yM[bIncP], 'mo', mfc='None', mec='m', ms=7)
        py.plot([0],[0],'rx')
        py.axis([15.0, -15.0, -15.0, 15.0])
        py.legend((biasO, biasI, bias2), ('Omega-bias' , 'Incl-bias', 'Both bias'),
                  numpoints=1,loc=1,prop=prop,fancybox=True)
        py.xlabel('X (arcsec)',fontsize=16)
        py.ylabel('Y (arcsec)',fontsize=16)
        py.title('Mock Data Velocities',fontsize=16)
        py.savefig('%s/velVector_mockdata_biasedOrbit_peaks.png' % outroot)
        py.close()

    py.clf()
    py.figure(figsize=(6,6))
    py.quiver([xM], [yM], [vxM], [vyM], headwidth=1.5, minshaft=1.5,
              color='black', units='y', angles='xy', scale=5)
    if nIsoStars != None:
        py.quiver([xM[0:nDiskStars]], [yM[0:nDiskStars]],
                  [vxM[0:nDiskStars]], [vyM[0:nDiskStars]], headwidth=1.5, minshaft=1.5,
                  color='red', units='y', angles='xy', scale=5)
    py.plot([0],[0],'rx')
    py.axis([15.0, -15.0, -15.0, 15.0])
    py.xlabel('X (arcsec)',fontsize=16)
    py.ylabel('Y (arcsec)',fontsize=16)
    #py.title('Mock Data Velocities (PA~0 stars)',fontsize=16)
    py.savefig('%s/velVector_mockdata_notext.png' % outroot)
    py.savefig('%s/eps/velVector_mockdata_notext.eps' % outroot)
    py.close()
    
def get_biased_orbits(simRoot='sim_diskFraction3/'):
    """
    NOT COMPLETE (as of 2012 May 8)

    Read in the pickle file containing the simulation results
    from the disk_fraction_results(). Finds the stars that show
    the line of nodes bias in Omega, and determines the maximal
    distance from the line of nodes that this bias occurs.
    """

    frac = np.arange(0.05, 0.59, 0.05)
    ntot = 120

    for ii in range(len(frac)):
        ff = np.around(frac[ii], decimals=2) # have to do this b/c sometimes a
        				     # dumb rounding error is made
        ndiskSim[ii] = ff * ntot
        ff2 = np.around(1-ff,decimals=2) 
        niso[ii] = ff2 * ntot

        simDir = '%s/disk%s/' % (simRoot, int(ff*100))
        mockFile = 'nDisk%s_nIso%s_mockdata.pickle' % (int(ndiskSim[ii]), int(niso[ii]))
        
        plotdir = '%s/%s/%s/plots/' % (root, alnDir, simDir)

        # First read in the mock data
        # Astrometric units are: arcsec, mas/yr, mas/yr^2
        mockdata = open(root + alnDir + mockdir + outfile)
        mbh = pickle.load(mockdata)
        dist = pickle.load(mockdata)
        orb_all = pickle.load(mockdata)
        sma_all = pickle.load(mockdata) # AU
        xM = pickle.load(mockdata) # (+x to east)
        yM = pickle.load(mockdata)
        zM = pickle.load(mockdata)
        vxM = pickle.load(mockdata) # (+x to east)
        vyM = pickle.load(mockdata)
        vzM = pickle.load(mockdata) # mas/yr
        axM = pickle.load(mockdata) # (+x to east)
        ayM = pickle.load(mockdata)
        azM = pickle.load(mockdata)
        t0M = pickle.load(mockdata) # periapse passage
        t_obs = pickle.load(mockdata) # observational time for mock data point
        mockdata.close()

        r2d = np.sqrt(xM**2 + yM**2)

        # Read in the simulation results
        pickFile = open('%s/plots/simulation_results.pickle' % simDir)
        incl = pickle.load(pickFile)
        Omega = pickle.load(pickFile)
        ave_Omega = pickle.load(pickFile) # only really makes sense if one
        				  # sign for z was used.
        std_Omega = pickle.load(pickFile)
        ave_incl = pickle.load(pickFile)
        std_incl = pickle.load(pickFile)
        ipeak = pickle.load(pickFile)
        Opeak = pickle.load(pickFile)
        zOut = pickle.load(pickFile)
        pIncAve = pickle.load(pickFile)
        nIncAve = pickle.load(pickFile)
        pOmAve = pickle.load(pickFile)
        nOmAve = pickle.load(pickFile)
        pIncStd = pickle.load(pickFile)
        nIncStd = pickle.load(pickFile)
        pOmStd = pickle.load(pickFile)
        nOmStd = pickle.load(pickFile)
        I_true = pickle.load(pickFile)
        O_true = pickle.load(pickFile)
        pickFile.close()

        # Look at both the +z and -z solutions
        pIsig = (pIncAve - I_true) / pIncStd
        nIsig = (nIncAve - I_true) / nIncStd

        pOsig = (pOmAve - O_true) / pOmStd
        nOsig = (nOmAve - O_true) / nOmStd

        bIncG = np.where((np.abs(pIsig) > 1.0) & (np.abs(nIsig) > 1.0) & (I_true >= 0))[0]
        # For most stars, the inclination isn't double peaked, so look at all soln:
        #bIncG = np.where(np.abs(med_incl - I_true) > 30.0)[0]
        bOmG = np.where((np.abs(pOsig) > 1.0) & (np.abs(nOsig) > 1.0) & (I_true >= 0))[0]

        # Which are biased in both?
        bothG = np.intersect1d(bIncG,bOmG)

        # For stars that are biased, how far are they in projection
        # from the line of nodes for their particular Omega?
        #xp1 = 15.0*np.cos(np.radians(90.-Omdisk))
        #yp1 = 15.0*np.sin(np.radians(90.-Omdisk))
        #xp2 = -15.0*np.cos(np.radians(90.-Omdisk))
        #yp2 = -15.0*np.sin(np.radians(90.-Omdisk))

def sim_surface_density(simRoot='sim_diskFraction3/'):
    """
    Calculate and plot the surface density of the simulated stars.
    For disk stars, plot as a function of 3D radius.
    For isotropic population, plot as a function of 2D radius projected on sky.
    """
    # Simulations run with 5%-55% disk stars
    frac = np.arange(0.05, 0.59, 0.05)
    ntot = 120

    ndiskSim = np.zeros(len(frac), dtype=float)
    niso = np.zeros(len(frac), dtype=float)

    def getMockData(ndisk, mockFile):
        # Read in the mock data
        # Astrometric units are: arcsec, mas/yr, mas/yr^2
        mockdata = open(root + alnDir + mockFile)
        mbh = pickle.load(mockdata)
        dist = pickle.load(mockdata)
        orb_all = pickle.load(mockdata)
        sma_all = pickle.load(mockdata) # AU
        xM = pickle.load(mockdata) # (+x to east)
        yM = pickle.load(mockdata)
        zM = pickle.load(mockdata)
        vxM = pickle.load(mockdata) # (+x to east)
        vyM = pickle.load(mockdata)
        vzM = pickle.load(mockdata) # mas/yr
        axM = pickle.load(mockdata) # (+x to east)
        ayM = pickle.load(mockdata)
        azM = pickle.load(mockdata)
        t0M = pickle.load(mockdata) # periapse passage
        t_obs = pickle.load(mockdata) # observational time for mock data point
        mockdata.close()
        r2dM = np.sqrt(xM**2 + yM**2)
        r3dM = np.sqrt(xM**2 + yM**2 + zM**2)

        return (xM, yM, zM, r2dM, r3dM)

    import matplotlib.font_manager
    prop = matplotlib.font_manager.FontProperties(size=8)
    usetexTrue()

    fig1 = py.figure(figsize=(10,8))
    fig1.clf()
    fig1.subplots_adjust(left=0.07, right=0.98, top=0.95, bottom=0.07,
                         wspace=0.35, hspace=0.35)

    py.clf()
    fig2 = py.figure(figsize=(10,8))
    fig2.clf()
    fig2.subplots_adjust(left=0.07, right=0.98, top=0.95, bottom=0.07,
                         wspace=0.35, hspace=0.35)
    fmt = '%5.3f  %4i  %7.3f  %4i  %7.3f'
    hdr = '%5s  %4s  %7s  %4s  %7s'
    labely = [1,5,9]
    all_disk_slopes = []
    for ii in range(len(frac)):

        def fitSD(rBins, sd, pop):
            logRadiusAll = np.log10(rBins)
            logDensityAll = np.log10(sd)

            def lineResiduals(params, x, y):
                slope = params[0]
                inter = params[1]
        
                model = inter + (slope*x)
                return (y - model)

            out = optimize.leastsq(lineResiduals, [0,0],
                                   args=(logRadiusAll, logDensityAll),
                                   full_output=1)
            p = out[0]
            pcov = out[1]
            #print 'Power Law Fit:'
            print 'Best-fit power law of slope = %5.2f (%s) ' % (p[0], pop)
                  #(p[0], slopeMC.std(ddof=1))
            print 'Amplitude = %5.2f stars/pc^2' % (10.0**p[1])

            return (p[0], p[1])

        print
        ff = np.around(frac[ii], decimals=2) # have to do this b/c sometimes a
        				     # dumb rounding error is made
        print 'Getting surface density for disk fraction %4.2f' % ff
        ndiskSim[ii] = ff * ntot
        ff2 = np.around(1-ff,decimals=2) 
        niso[ii] = ff2 * ntot

        simDir = '%s/disk%s/' % (simRoot, int(ff*100))
        mockFile = 'nDisk%s_nIso%s_mockdata.pickle' % (int(ndiskSim[ii]), int(niso[ii]))
        
        plotdir = '%s/%s/%s/plots/' % (root, alnDir, simDir)

        x, y, z, r2d, r3d = getMockData(ndiskSim[ii], simDir+mockFile)

        # Get positions for the disk stars
        xD = x[0:ndiskSim[ii]] * 0.039 # pc
        yD = y[0:ndiskSim[ii]] * 0.039 # pc
        zD = z[0:ndiskSim[ii]] * 0.039 # pc

        # 2d radius for disk stars
        r2dD = r2d[0:ndiskSim[ii]] * 0.039 # pc
        # 3D radius for disk stars
        r3dD = r3d[0:ndiskSim[ii]] * 0.039 # pc

        # 2d radius for non-disk stars
        r2dND = r2d[ndiskSim[ii]:] * 0.039 # pc
        # 3d radius for non-disk stars
        r3dND = r3d[ndiskSim[ii]:] * 0.039 # pc

        # Switch the disk star positions into disk coordinate system. Coordinates
        # are p, q, n where
        #   n is tangential to the disk plane.
        #   p goes along the ascending node in the disk plane.
        #   q is perpindicular to p in the disk plane.
        # Get the directional vectors
        (p, q, n) = diskProject(xD, yD, zD, idisk=idisk, odisk=odisk)
        rInDisk = np.sqrt(p**2 + q**2)

        # Get surface density profile
        #rbins = np.arange(0,14,1.0)
        # define radial bins in pc
        rLo = np.array([0.032,0.045,0.060,0.090,0.115,0.140,0.165,0.190,0.270,0.360,0.450]) 
        rHi = np.array([0.045,0.060,0.090,0.115,0.140,0.165,0.190,0.270,0.360,0.450,0.550])
        rbins = rLo + ((rHi - rLo) / 2.0)
        sdDsky = np.zeros((len(rbins)), dtype=float)
        sdDplane = np.zeros((len(rbins)), dtype=float)
        sdNDsky = np.zeros((len(rbins)), dtype=float)
        print hdr % ('Rbin', 'Dsk', 'SD', 'NonD', 'SD')
        for rr in range(len(rbins)):
            areaInAnnulus = np.pi * (rHi[rr]**2 - rLo[rr]**2)

            # Just disk stars surface density in the disk plane
            # first get surface density projected on sky
            nnDplane = len(np.where((rInDisk > rLo[rr]) & (rInDisk < rHi[rr]))[0])
            sdDplane[rr] = nnDplane / areaInAnnulus

            # now get surface density projected onto sky
            nnDsky = len(np.where((r2dD > rLo[rr]) & (r2dD < rHi[rr]))[0])
            sdDsky[rr] = nnDsky / areaInAnnulus

            # Non-disk stars projected onto sky
            # Really this should be r3dND if we want to check that this matches
            # what we sampled from for the isotropic case, since we sampled a
            # 3D radius that followed a profile like r^-1.14; if this is done,
            # it should be plotted against r3d
            nnNDsky = len(np.where((r2dND > rLo[rr]) & (r2dND < rHi[rr]))[0])
            sdNDsky[rr] = nnNDsky / areaInAnnulus

            print fmt % (rbins[rr], nnDsky, sdDsky[rr], nnNDsky, sdNDsky[rr])

        # Fit the SD profiles for bins where we have stars
        nzdx1 = np.where(sdDplane != 0.0)[0]
        print 'Fitting radial profile to SD for f = %4.2f' % ff
        slopeSDplane, ampSDplane = fitSD(rbins[nzdx1], sdDplane[nzdx1], 'disk stars SD in disk plane')
        # save all the slopes
        all_disk_slopes = np.concatenate([all_disk_slopes, [slopeSDplane]])

        nzdx2 = np.where(sdDsky != 0.0)[0]
        slopeSDsky, ampSDsky = fitSD(rbins[nzdx2], sdDsky[nzdx2], 'disk stars SD on sky')

        nzdx3 = np.where(sdNDsky != 0.0)[0]
        slopeSDndsky, ampSDndsky = fitSD(rbins[nzdx3], sdNDsky[nzdx3], 'non-disk stars SD on sky')

        # Plot SD of disk stars projected on sky
        ax1 = fig1.add_subplot(3, 4, ii+1)
        ax1.loglog(rbins,sdDsky,'r.', label='Disk', ms=8)
        plawX = np.array([0.032, 0.6])
        plawY = 10**(ampSDsky + np.log10(plawX)*slopeSDsky)
        ax1.plot(plawX, plawY, 'r--')
        #ax1.text(0.05,sdD[nzdx1[0]]-100,'slope = %5.2f' % slopeSDd,color='r',fontsize=10)

        # Plot SD of non-disk stars projected on sky
        ax1.loglog(rbins,sdNDsky,'k.', label='Iso')
        plawY = 10**(ampSDndsky + np.log10(plawX)*slopeSDndsky)
        ax1.plot(plawX, plawY, 'k--')
        #ax1.text(0.3,sdND[nzdx3[-1]]+500,'slope = %5.2f' % slopeSDnd,color='k',fontsize=10)
        ax1.set_xlabel('R2D (pc)', fontsize=12)
        ax1.set_title(r'N$_{disk}$ = %4.1f' % ndiskSim[ii],fontsize=12)
        if ii == 0:
            ax1.legend(numpoints=1,fancybox=True,prop=prop)
        if (ii+1) in labely:
            ax1.set_ylabel(r'Surface Density (stars/pc$^2$)', fontsize=12)

        # Plot SD of disk stars in the disk plane
        ax2 = fig2.add_subplot(3, 4, ii+1)
        ax2.loglog(rbins,sdDplane,'r.', ms=8)
        plawY = 10**(ampSDplane + np.log10(plawX)*slopeSDplane)
        ax2.plot(plawX, plawY, 'k--')
        ax2.text(0.1,1000,'slope = %5.2f' % slopeSDplane,fontsize=10)
        ax2.axis([0,0.6,1,5e3])
        ax2.set_xlabel('Radius in Disk (pc)', fontsize=12)
        ax2.set_title(r'N$_{disk}$ = %4.1f' % ndiskSim[ii],fontsize=12)
        if ii == 0:
            ax2.legend(numpoints=1,fancybox=True,prop=prop)
        if (ii+1) in labely:
            ax2.set_ylabel(r'Surface Density (stars/pc$^2$)', fontsize=12)

    fig1.savefig('%s/%s/%s/plots/surface_densities_r2d_pc.png' % (root, alnDir, simRoot))
    py.close(1)

    fig2.savefig('%s/%s/%s/plots/surface_densities_diskPlane_pc.png' % (root, alnDir, simRoot))
    py.close(2)

    usetexFalse()

    print
    print 'Average SD slope in the disk plane = %5.2f +- %5.2f' % \
          (all_disk_slopes.mean(), all_disk_slopes.std(ddof=1)) 


def get_z_mockdata(pdfdir='ecc_bias_uniformAll/', mockfile='uniform_mockdata.pickle'):
    """
    Fit the z distribution that came out of the mock data created using
    uniform orbital parameters (i, O, w, e, t0) and a power law radial distribution.
    """

    # Read in the mock data
    # Astrometric units are: arcsec, mas/yr, mas/yr^2
    mockdata = open(root + alnDir + pdfdir + mockfile)
    mbh = pickle.load(mockdata)
    dist = pickle.load(mockdata)
    orb_all = pickle.load(mockdata)
    sma_all = pickle.load(mockdata) # AU
    xM = pickle.load(mockdata) # (+x to east)
    yM = pickle.load(mockdata)
    zM = pickle.load(mockdata)
    vxM = pickle.load(mockdata) # (+x to east)
    vyM = pickle.load(mockdata)
    vzM = pickle.load(mockdata) # mas/yr
    axM = pickle.load(mockdata) # (+x to east)
    ayM = pickle.load(mockdata)
    azM = pickle.load(mockdata)
    t0M = pickle.load(mockdata) # periapse passage
    t_obs = pickle.load(mockdata) # observational time for mock data point
    mockdata.close()

    zMmag = np.abs(zM)

    #def fitPlaw(params, x, y):
    #    A = params[0]
    #    alpha = params[1]
    #    const = params[2]

    #    model = A * (x**alpha) + const
    #    return (y - model)

    # Histogram of the mock line of sight distances
    py.clf()
    py.figure()
    py.figure(figsize=(6,6))
    nn, bb, pp = py.hist(zMmag, bins=np.arange(0, 25, 0.1), histtype='step')
    py.xlabel('Mock z (arcsec)')
    py.ylabel('Number of Stars')
    py.savefig(root + alnDir + pdfdir + 'plots/mock_z_distribution.png')
    py.close()

    # Fit a power law to the z distribution
    zz = np.array([(bb[ii]+bb[ii+1]) / 2.0 for ii in range(len(bb)-1)])
    ndata = np.array(nn)
    p0 = [1.0, -1.0, 0.0]
    out = optimize.leastsq(fitPlaw, p0,
                           args=(zz, ndata), full_output=1)
    p = out[0]
    pcov = out[1]
    print 'Power Law Fit:'
    print '  Amplitude: %5.3f' % p[0]
    print '  Alpha: %5.3f' % p[1]
    print '  Constant: %5.3f' % p[2]
    print ''
    print '  %5.3f * (z^%5.3f) + %5.3f' % (p[0], p[1], p[2])

    # Plot the power law fit over the data for z
    plawX = np.arange(0.0, 25.0, 0.001)
    plawY = p[0] * (zz**p[1]) + p[2]
    py.clf()
    py.figure(figsize=(6,6))
    py.plot(zz, plawY, 'k--')
    py.plot(zz, ndata, 'r.')
    py.xlabel('Mock z (arcsec)')
    py.ylabel('Number of Stars')
    py.axis([0, 25, 0, 900])
    py.savefig(root + alnDir + pdfdir + 'plots/mock_z_plaw.png')

    # Plot the semimajor axis
    sma = np.sqrt(xM**2 + yM**2 + zM**2)
    nn, bb, pp = py.hist(sma, bins=np.arange(0, 25, 0.1), histtype='step')
    sma = np.array([(bb[ii]+bb[ii+1]) / 2.0 for ii in range(len(bb)-1)])
    ndata = np.array(nn)
    py.clf()
    py.figure(figsize=(6,6))
    py.semilogx(sma, ndata, 'r.')
    py.xlabel('Mock Semi-major Axis (arcsec)')
    py.ylabel('Number of Stars')
    py.axis([0.8, 25.0, 0, 250])
    py.savefig(root + alnDir + pdfdir + 'plots/mock_semimajor.png')


def plot_mock_data(simdir='sim_true_isotropic/',numStars=5000):
    """
    Plot input orbital elements that produced the mock data
    in a given simulation
    """
    wdir = root + alnDir + simdir
    infile = asciidata.open('%s/orbital_elements.dat' % wdir)

    p = infile[1].tonumpy()
    a = infile[2].tonumpy()
    t0 = infile[3].tonumpy()
    e = infile[4].tonumpy()
    i = infile[5].tonumpy()
    O = infile[6].tonumpy()
    w = infile[7].tonumpy()

    usetexTrue()
    elems = [a/1.e3,t0,e,i,O,w]
    #elems = [a/1.e3,t0,e,np.cos(i/rad2deg),O,w]
    xlbl = [r'$a$ (arcsec)', r'T$_0$ (year)', r'$e$',
            r'$i$', r'$\Omega$ ($\deg$)', r'$\omega$ ($\deg$)']

    py.figure(figsize=(8,8))
    py.clf()
    py.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1,
                       wspace=0.3, hspace=0.3)
    for jj in range(len(elems)):
        py.subplot(3,2,jj+1)
        py.hist(elems[jj],bins=20,histtype='step')
        py.xlabel(xlbl[jj],fontsize=12)
        py.ylabel('N',fontsize=12)

    py.suptitle('Input Isotropic Orbits (N=%i)' % numStars)
    py.savefig('%s/plots/input_orbital_elements.png' % wdir)
    py.close()
    usetexFalse()

    # Put the data in a format for making a HEALpix density map
    #nside = 8
    nside = 64
    npix = healpy.nside2npix(nside)
    pixIdx = np.arange(0, npix)
    pdf = np.zeros((npix), dtype=float)

    # Determine which pixel in the map each of the
    # points goes (2D histogram)
    incl = i * np.pi / 180.0
    omeg = O * np.pi / 180.0
    hidx = healpy.ang2pix(nside, incl, omeg)
    for hh in hidx:
        pdf[hh] += 1.0
    pdf /= numStars
    pdfFile = '%s/iO_healpix.dat' % wdir
    pdf.tofile(pdfFile)

    # Plot the results on a healpix map
    pdh.go(pdfFile, npix, 1)
    
    

def pdfEccentricity(pdfdir='aorb_thesis/', mosaic=True, simFlat=False,
                    suffix='_mosaic', LHsigCut=3.0, nonDisk=False, radBin=False):
    """
    Monte Carlo the eccentricities and calculate the average and RMS
    for each trial. Then keep the resulting distribution.

    LHsigCut (float):  Significance threshold for non-members;
            	       default is set to 3.0, meaning that stars with
                       likelihood of not being on the disk of >3 sigma
                       are considered non-members. The rest are candidates.
    nonDisk (bool):    Set to True to return non-disk members (ruled out at
    		       the sigma level specified by LHsigCut).
    radBin (bool):     Set to True to return the parameter values for
    		       stars in a separate radial bins.
    """
    outdir = root + alnDir + 'plots/'

    py.close('all')
    def plotRadial(input,ii,label):
        eccStep = 0.05
        binsIn = np.arange(0, 1+eccStep, eccStep)
        py.subplot(3,3,ii)
        aa,bb,cc = py.hist(input.flatten(),bins=binsIn,histtype='step',normed=True,color='k')
        if ii in np.array([1,4,7]):
            py.ylabel('Probability Density')
        if ii > 6:
            py.xlabel('Eccentricity')
        py.title('%s' % label)

        print np.sum(aa * np.diff(bb)) # trapezoidal integration of the pdf, this should be 1.0

    # Get the global set of eccentricities for ALL solutions and for
    # DISK solutions of the candidate disk members (or non-disk members if nonDisk=True).
    # This also prints out useful stats
    py.clf()
    if nonDisk == False:
        if radBin == True:
            radLbl = ['r < 2"', '2" < r < 3"', '3" < r < 4"', '4" < r < 5"','5" < r < 6"',
                      '6" < r < 7"', '7" < r < 8"', '8" < r < 9"', 'r > 9"']
            py.figure(figsize=(10,10))
            py.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.07,
                               wspace=0.3, hspace=0.3)
            py.clf()
            rbin = np.arange(0,10)
            for rr in range(1,10):
                #print '*********' + str(rbin[rr])
                r1 = rbin[rr] *1.0
                if rbin[rr] == 1.0: # Set first interval to be r = 0.8-2"
                    r1 = 0.8
                if r1 == 9.0:
                    r2 = 20 # Set the last bin to be r > 9."
                else:
                    r2 = rbin[rr+1] *1.0
                (ecc, eccDisk) = paramStats('e',pdfdir=pdfdir,mosaic=mosaic,simFlat=simFlat,
                                            LHsigCut=LHsigCut, nonDisk=nonDisk,
                                            radBin=[r1,r2])
                plotRadial(input=eccDisk, ii=rr, label=radLbl[rr-1]) # for disk stars, interested in disk solutions 
            suffix = '_disk_radial' + suffix
            py.savefig(outdir + 'pdf_ecc%s.png' % suffix)
            py.savefig(outdir + 'eps/pdf_ecc%s.eps' % suffix)
            py.close()
        else:
            (ecc, eccDisk) = paramStats('e',pdfdir=pdfdir,mosaic=mosaic,simFlat=simFlat,
                                        LHsigCut=LHsigCut, nonDisk=nonDisk)
            suffix = '_disk' + suffix
    else: # non-disk stars
        if radBin == True:
            radLbl = ['r < 2"', '2" < r < 3"', '3" < r < 4"', '4" < r < 5"','5" < r < 6"',
                      '6" < r < 7"', '7" < r < 8"', '8" < r < 9"', 'r > 9"']
            py.figure(figsize=(10,10))
            py.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.07,
                               wspace=0.3, hspace=0.3)
            py.clf()
            rbin = np.arange(0,10)
            for rr in range(1,10):
                #print '*********' + str(rbin[rr])
                r1 = rbin[rr] *1.0
                if rbin[rr] == 1.0: # Set first interval to be r = 0.8-2"
                    r1 = 0.8
                if r1 == 9.0:
                    r2 = 20 # Set the last bin to be r > 9."
                else:
                    r2 = rbin[rr+1] *1.0
                (ecc, eccDisk) = paramStats('e',pdfdir=pdfdir,mosaic=mosaic,simFlat=simFlat,
                                            LHsigCut=LHsigCut, nonDisk=nonDisk,
                                            radBin=[r1,r2])
                plotRadial(input=ecc, ii=rr, label=radLbl[rr-1]) # for non-disk stars, interested in all solutions 
            suffix = '_disk_radial' + suffix
            py.savefig(outdir + 'pdf_ecc%s.png' % suffix)
            py.savefig(outdir + 'eps/pdf_ecc%s.eps' % suffix)
            py.close()
        else:
            ecc = paramStats('e',pdfdir=pdfdir,mosaic=mosaic,simFlat=simFlat,
                                        LHsigCut=LHsigCut, nonDisk=nonDisk, radBin=radBin)
            suffix = '_nonDisk' + suffix

    # Determine the eccentricity probability density function
    eccStep = 0.05
    binsIn = np.arange(0, 1+eccStep, eccStep)

    import matplotlib.font_manager
    prop = matplotlib.font_manager.FontProperties(size=12)

    if radBin == False:
        if nonDisk == False: # plot disk members
            py.clf()
            py.figure(figsize=(6,6))
            py.subplots_adjust(left=0.15, right=0.95)
            aa,bb,cc = py.hist(eccDisk.flatten(),bins=binsIn,histtype='step',normed=True,color='k')
            py.xlabel('Eccentricity')
            py.ylabel('Probability Density')
            py.savefig(outdir + 'pdf_ecc%s.png' % suffix)
            py.savefig(outdir + 'eps/pdf_ecc%s.eps' % suffix)
            py.close()
    
            print np.sum(aa * np.diff(bb)) # trapezoidal integration of the pdf, this should be 1.0
    
            py.clf()
            py.figure(figsize=(7,6))
            py.subplots_adjust(left=0.15, right=0.95)
            py.hist(ecc.flatten(), bins=binsIn, histtype='step', normed=True,
                    color='gray', linewidth=2, label='All Solutions')
            py.hist(eccDisk.flatten(), bins=binsIn, histtype='step', normed=True,
                    color='black', linewidth=2, label='Disk Solutions')
            #py.legend((p1, p2), ('Unweighted', 'Weighted'))
            py.legend(numpoints=1,fancybox=True,prop=prop)
            py.xlabel('Eccentricity')
            py.ylabel('Probability Density')
            py.title('Disk Candidates')
            py.xlim(0, 1)
            py.savefig(outdir + 'pdf_ecc_both%s.png' % suffix)
            py.savefig(outdir + 'eps/pdf_ecc_both%s.eps' % suffix)
            py.close()
        else:
            py.clf()
            py.figure(figsize=(7,6))
            py.subplots_adjust(left=0.15, right=0.95)
            py.hist(ecc.flatten(), bins=binsIn, histtype='step', normed=True, color='k')
            py.xlabel('Eccentricity')
            py.ylabel('Probability Density')
            py.savefig(outdir + 'pdf_ecc_allSoln%s.png' % suffix)
            py.close()



def pdf_IO(pdfdir='aorb_thesis/', mosaic=True, 
           suffix='_mosaic'):
    """
    Make 1D histograms of the inclination and Omega.
    """
    outdir = root + alnDir + 'plots/'

    import matplotlib.font_manager
    prop = matplotlib.font_manager.FontProperties(size=12)

    # Get the global set of incl & Omega for ALL solutions and for
    # DISK solutions. This also prints out useful stats
    (inclAll, inclDisk) = paramStats('i',pdfdir=pdfdir,mosaic=mosaic,allStars=True,
                                      simFlat=False)
    (OmAll, OmDisk) = paramStats('o',pdfdir=pdfdir,mosaic=mosaic,allStars=True,
                                  simFlat=False)

    # Read in membership probability
    diskTab = asciidata.open(root + alnDir + 'tables/disk_membership_prob_mosaic.dat')
    diskP = diskTab[1].tonumpy()
    didx = (np.where(diskP > 2.7e-3))[0] # disk
    ndidx = (np.where(diskP < 2.7e-3))[0] # non-disk
     
    # First get disk stars; paramStats returns (all, disk) solutions
    #(inclAllD, inclDiskD) = paramStats('i',pdfdir=pdfdir,mosaic=mosaic,simFlat=False)
    #(OmAllD, OmDiskD) = paramStats('o',pdfdir=pdfdir,mosaic=mosaic,simFlat=False)

    # Now get non-disk stars
    #(inclAllND, inclDiskND) = paramStats('i',pdfdir=pdfdir,mosaic=mosaic,nonDisk=True,
    #                                  simFlat=False)
    #(OmAllND, OmDiskND) = paramStats('o',pdfdir=pdfdir,mosaic=mosaic,nonDisk=True,
    #                              simFlat=False)

    # Plot all solutions, separately for disk stars and non-disk stars 
    py.clf()
    py.figure(figsize=(10,5))
    py.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.13,
                       wspace=0.3, hspace=0.3)
    py.subplot(1,2,1)
    nn, bb, pp = py.hist(inclAll[ndidx,:].flatten(), bins=np.arange(0, 182, 2.0),
                histtype='step',color='k',normed=False,label='Non-Disk Stars')
    nn, bb, pp = py.hist(inclAll[didx,:].flatten(), bins=np.arange(0, 182, 2.0),
                histtype='step',color='r',normed=False,label='Disk Stars')
    py.title('All Solutions')
    py.ylabel('Probability Density')
    py.legend(numpoints=1,loc=2,fancybox=True,prop=prop)
    py.axis([0, 180, 0, nn.max()+0.01])
    py.xlabel('Inclination (deg)')
    py.subplot(1,2,2)
    nn, bb, pp = py.hist(OmAll[ndidx,:].flatten(), bins=np.arange(0, 362, 2.0),
                histtype='step',color='k',normed=False)
    nn, bb, pp = py.hist(OmAll[didx,:].flatten(), bins=np.arange(0, 362, 2.0),
                histtype='step',color='r',normed=False)
    py.title('All Solutions')
    py.axis([0, 360, 0, nn.max()+0.01])
    py.xlabel('PA to the Ascending Node (deg)')
    py.savefig(outdir + 'pdf_IO_allSoln_diskVsNonDisk%s.png' % suffix)

    # Plot all solutions, separately for disk stars and all stars 
    py.clf()
    py.figure(figsize=(10,5))
    py.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.13,
                       wspace=0.3, hspace=0.3)
    py.subplot(1,2,1)
    nn, bb, pp = py.hist(inclAll.flatten(), bins=np.arange(0, 182, 2.0),
                histtype='step',color='k',normed=False,label='All Stars')
    nn, bb, pp = py.hist(inclAll[didx,:].flatten(), bins=np.arange(0, 182, 2.0),
                histtype='step',color='r',normed=False,label='Disk Stars')
    py.title('All Solutions')
    py.ylabel('Probability Density')
    py.legend(numpoints=1,loc=2,fancybox=True,prop=prop)
    py.axis([0, 180, 0, nn.max()+0.01])
    py.xlabel('Inclination (deg)')
    py.subplot(1,2,2)
    nn, bb, pp = py.hist(OmAll.flatten(), bins=np.arange(0, 362, 2.0),
                histtype='step',color='k',normed=False)
    nn, bb, pp = py.hist(OmAll[didx,:].flatten(), bins=np.arange(0, 362, 2.0),
                histtype='step',color='r',normed=False)
    py.title('All Solutions')
    py.axis([0, 360, 0, nn.max()+0.01])
    py.xlabel('PA to the Ascending Node (deg)')
    py.savefig(outdir + 'pdf_IO_allSoln_diskVsAll%s.png' % suffix)



def nodes_bias_observed():
    """
    Plot the disk membership probability vs. the position angle offset from the line of
    nodes for the observations.
    """
    cc = objects.Constants()
    # Load names of young stars 
    # Load up mosaic data as well; select only stars at r>4, since
    # we don't want to add any info from mosaics if we have it in
    # the central 10" already
    yng1 = young.loadYoungStars(root+alnDir,fit=poly,points=points,
                                withRVonly=True,silent=True,skipStar=['S5-237']) 
    yng2 = young.loadYoungStars(mscDir,fit=polyM,points=pointsM,
                                mosaic=True, withRVonly=True,silent=True)
    cntrlNames = yng1.getArray('name')
    mscNames = yng2.getArray('name')
    # Merge this object with object from central 10" analysis
    yng = merge(yng1, yng2)
    yngNames = yng.getArray('name')

    # Read in the table of disk membership probabilities
    diskTab = asciidata.open(root + alnDir + 'tables/disk_membership_prob_mosaic.dat')
    nameP = np.array([diskTab[0][ss].strip() for ss in range(diskTab.nrows)])
    diskP = diskTab[1].tonumpy()

    # Read in file with disk membership prob. vs. position angle
    # offset from line of nodes, from disk fraction simulations
    #simfile = '%s/%s/sim_diskFraction4/plots/ave_posAngleOffNodes_vs_diskProb_0.2multiSims.txt' % \
    simfile = '%s/%s/sim_diskFraction4/plots/med_posAngleOffNodes_vs_diskProb_0.2multiSims.txt' % \
               (root,alnDir)
    sim = asciidata.open(simfile)
    bin = sim[0].tonumpy()
    aveProb = sim[1].tonumpy()
    rmsProb = sim[2].tonumpy()

    # Read in file listing accelerating sources
    _acc = asciidata.open('%s/%s/tables/accelerating_sources.dat' % (root, alnDir))
    acc = _acc[0].tonumpy()
    accels = [aa.strip() for aa in acc]

    cc = objects.Constants()
    GM = cc.G * mass * cc.msun
    asy_to_kms = dist * cc.cm_in_au / (1.e5 * cc.sec_in_yr)

    x_all = []
    y_all = []
    vx_all = []
    vy_all = []
    diskP_all = []
    rvec_angle_nodes_all = []
    names_all = []
    for ii in range(len(yngNames)):
        name = yngNames[ii]
	i = yngNames.index(name)

	star = yng.stars[i]
        mag = yng.stars[i].mag

        mscStar = False

        # get the disk membership prob for this star
        idx = np.where(nameP == name)[0] 
        diskP_all = np.concatenate([diskP_all, diskP[idx]])

        if (name in mscNames) & (name not in cntrlNames):
            # For some reason, mosaic velocities from vel fit come
            # out in km/s; whereas central 10'' data come out in asec/yr
            # for the velocity fits, and km/s for acceleration fits.
            mscStar = True
            x = star.fitXv.p # arcsec
            y = star.fitYv.p # arcsec
            r2d = np.hypot(x, y) # arcsec
            xe = star.fitXv.perr # arcsec
            ye = star.fitYv.perr # arcsec
            vx = star.fitXv.v  # km/s
            vy = star.fitYv.v  # km/s
            vxe = star.fitXv.verr # km/s
            vye = star.fitYv.verr # km/s
        else:
            if name in accels:
                x = star.fitXa.p # arcsec 
                y = star.fitYa.p # arcsec
                r2d = np.hypot(x, y) # arcsec
                xe = star.fitXa.perr # arcsec
                ye = star.fitYa.perr # arcsec
                vx = star.fitXa.v # km/s
                vy = star.fitYa.v # km/s
                vxe = star.fitXa.verr # km/s
                vye = star.fitYa.verr # km/s
            else:
                x = star.fitXv.p # arcsec
                y = star.fitYv.p # arcsec
                r2d = np.hypot(x, y) # arcsec
                xe = star.fitXv.perr # arcsec
                ye = star.fitYv.perr # arcsec
                vx = star.fitXv.v * asy_to_kms # km/s
                vy = star.fitYv.v * asy_to_kms # km/s
                vxe = star.fitXv.verr * asy_to_kms# km/s
                vye = star.fitYv.verr * asy_to_kms # km/s

        x_all = np.concatenate([x_all, [x*1.e3]]) # mas
        y_all = np.concatenate([y_all, [y*1.e3]]) # mas 
        vx_all = np.concatenate([vx_all, [vx]]) # km/s
        vy_all = np.concatenate([vy_all, [vy]]) # km/s

        # Calculate position angle relative to CW disk's line of nodes
        rvec_angle = np.arctan2(y,-x)*(180./np.pi) # angle from (x,y)=(1,0)
        rvec_angle_nodes = rvec_angle + (90. - odisk) # angle relative to Omega
        if np.abs(rvec_angle_nodes) < 90.:
            rvec_angle_nodes = np.abs(rvec_angle_nodes)
        elif np.abs(rvec_angle_nodes) >= 90.:
            rvec_angle_nodes = 180.0 - np.abs(rvec_angle_nodes)
        if rvec_angle_nodes < 0.0:
            rvec_angle_nodes = np.abs(rvec_angle_nodes)
        rvec_angle_nodes_all = np.concatenate([rvec_angle_nodes_all, [rvec_angle_nodes]])
        names_all = np.concatenate([names_all, [name]])

    usetexTrue()
    outdir = root + alnDir + 'plots/'
    
    # Plot angular offset of radius vector from Line of Nodes vs. probability
    py.figure(1)
    py.figure(figsize=(6,6))
    py.subplots_adjust(left=0.14, right=0.96, top=0.9)
    py.clf()
    py.plot(rvec_angle_nodes_all, diskP_all, 'ko', marker='o', mfc='None', mec='k',
                ms=8, mew=1.5)
    for ss in range(len(diskP_all)):
        if yngNames[ss] in accels:
            py.plot(rvec_angle_nodes_all[ss], diskP_all[ss], 'kx', mew=1.5)
    #py.plot(bin,aveProb-1.*rmsProb,'g--',lw=2)
    #py.plot(bin,aveProb-2.*rmsProb,'g--',lw=2)
    py.plot(bin,aveProb-3*rmsProb,'g--',lw=2)
    py.title('Observations')
    py.xlabel('Position Angle from Line of Nodes (deg)')
    py.ylabel('Disk Membership Probability')
    py.axis([0, 90, 9e-6, 1.1])
    py.savefig('%s/posAngleOffNodes_vs_diskProb_observed.png' % outdir)
    py.savefig('%s/eps/posAngleOffNodes_vs_diskProb_observed.eps' % outdir)
    py.close(1)

    # Get disk membership and plot velocity vectors on a mosaic image
    imgFile = '/u/ghezgroup/data/gc/08maylgs2/combo/mag08maylgs2_dp_msc_kp.fits'
    scale = 0.00993
    sgra = [1398.4, 1500.4]
    img = pyfits.getdata(imgFile)
    imgsize = (img.shape)[0]
    # Make axes for images in arcsec
    pixL = np.arange(0,imgsize)
    xL = [-1*(xpos - sgra[0])*scale for xpos in pixL]
    yL = [(ypos - sgra[1])*scale for ypos in pixL]

    # normalized angular momentum
    vtot = np.hypot(vx_all*1.e5,vy_all*1.e5)

    x_cgs = x_all * dist * cc.cm_in_au
    y_cgs = y_all * dist * cc.cm_in_au
    r2d_cgs = np.hypot(x_cgs, y_cgs)
    j = (x_cgs*vy_all*1.e5 - y_cgs*vx_all*1.e5) / (r2d_cgs*vtot)

    py.figure(2)
    py.figure(figsize=(7,6))
    py.clf()
    py.subplots_adjust(left=0.12, right=0.98, top=0.95, bottom=0.1)
    py.imshow(np.log10(img+1), aspect='equal', interpolation='bicubic',
              extent=[max(xL), min(xL), min(yL), max(yL)],vmin=2.2,vmax=5,
              origin='lowerleft', cmap=py.cm.gray_r)
    cmap = py.cm.spectral
    norm = py.normalize(min(diskP_all),max(diskP_all))
    for ss in range(len(diskP_all)):
        #print diskP_all[ss],cmap(norm(diskP_all[ss]))
        # Determine which pos angle bin this star falls in:
        bidx = np.abs(rvec_angle_nodes_all[ss] - bin).argmin()
        # For this bin, is the star's disk membership probability above
        # the curve?
        #if ((diskP_all[ss] >= (aveProb[bidx]-3.0*rmsProb[bidx])) & (j[ss] > 0.)): # disk candidates
        #if (diskP_all[ss] >= (aveProb[bidx]-3.0*rmsProb[bidx])): # disk candidates
        #    qvr = py.quiver([x_all[ss]/1.e3], [y_all[ss]/1.e3], [vx_all[ss]], [vy_all[ss]],
        #                    headwidth=1.5, minshaft=1.5, color='red', units='y', angles='xy',
        #                    scale=200)
        #else: # everything else (not on disk)
        qvr = py.quiver([x_all[ss]/1.e3], [y_all[ss]/1.e3], [vx_all[ss]], [vy_all[ss]],
                  headwidth=1.5, minshaft=1.5, color=cmap(norm(diskP_all[ss])), units='y', angles='xy',
                  scale=200)
    mappable = py.cm.ScalarMappable(norm,cmap)
    mappable.set_array(diskP_all)
    cb = py.colorbar(mappable, shrink=0.9)
    cb.set_label('Disk Membership Probability')
    py.quiverkey(qvr,  9, 10, 300, '300 km/s', coordinates='data', color='black')
    py.plot([0],[0],'k+',ms=7,mew=2)
    an = np.linspace(0,2*np.pi,100)
    py.plot(0.8*np.cos(an),0.8*np.sin(an),'k--', lw=1.5) # marks radial bins used
    py.plot(3.2*np.cos(an),3.2*np.sin(an),'k--', lw=1.5) # marks radial bins used
    py.plot(6.5*np.cos(an),6.5*np.sin(an),'k--', lw=1.5) # marks radial bins used
    #py.plot(10.0*np.cos(an),10.0*np.sin(an),'k--', lw=1) # marks radial bins used
    py.axis([12.5, -14.0, -15.0, 12.5])
    py.xlabel('X (arcsec)',fontsize=16)
    py.ylabel('Y (arcsec)',fontsize=16)
    py.savefig('%s/velVectors_posAngle_diskMmbrs_observed.png' % outdir)
    py.savefig('%s/eps/velVectors_posAngle_diskMmbrs_observed.eps' % outdir)
    py.close(2)

    # Make a latex table with the star name, angle off the line of nodes,
    # and what sigma level the star falls above (1, 2, or 3)
    outFile = '%s/%s/tables/nodes_angle_prob_app.tex' % (root, alnDir)
    outFile2 = '%s/%s/tables/nodes_angle_prob_app.dat' % (root, alnDir)
    if os.access(outFile, os.F_OK): os.remove(outFile)
    if os.access(outFile2, os.F_OK): os.remove(outFile2)
    out = open(outFile, 'w')
    out2 = open(outFile2, 'w')
    out2.write('# Star    sigma level\n')
    fmt = '%9s & %4.1f & %1i$\sigma$ \\\\ \n'
    fmt2 = '%9s & %4.1f & %9s \\\\ \n'
    out.write('\\begin{deluxetable}{lrr}\n')
    out.write('\\tabletypesize{\\scriptsize}\n')
    out.write('\\tablewidth{0pt}\n')
    out.write('\\tablecaption{Disk Membership Sample')
    out.write('\\label{tab:nodes_app_table}}\n')
    out.write('\\tablehead{\n')
    out.write('  \\colhead{Name} &\n')
    out.write('  \\colhead{$PA_{nodes}$\\tablenotemark{a}} &\n')
    out.write('  \\colhead{Sample\\tablenotemark{b}} \\\\ \n')
    out.write('%\n')
    out.write('  \\colhead{} &\n')
    out.write('  \\colhead{(deg)} &\n')
    out.write('  \\colhead{} \n')
    out.write('}\n')
    out.write('\\startdata\n')
    sig_level = np.zeros(len(diskP_all),dtype=int)
    for ss in range(len(diskP_all)):
        bidx = np.abs(rvec_angle_nodes_all[ss] - bin).argmin()
        if (diskP_all[ss] >= (aveProb[bidx]-1.0*rmsProb[bidx])):
            sig_level[ss] = 1
        elif (diskP_all[ss] >= (aveProb[bidx]-2.0*rmsProb[bidx])):
            sig_level[ss] = 2
        elif (diskP_all[ss] >= (aveProb[bidx]-3.0*rmsProb[bidx])):
            sig_level[ss] = 3
        else:
            sig_level[ss] = 10 # dummy for the 'other' category
    # Sort the arrays so that most significant are printed out first
    sidx = sig_level.argsort()
    for ss in sidx:
        if sig_level[ss] != 10:
            out.write(fmt % (names_all[ss], rvec_angle_nodes_all[ss],
                             sig_level[ss]))
            out2.write('%10s  %1i\n' % (names_all[ss], sig_level[ss]))
        else:
            out.write(fmt2 % (names_all[ss], rvec_angle_nodes_all[ss],
                             'other'))
                    
    out.write('\\enddata \n')
    out.write('\\tablenotetext{a}{Position angle offset from the line of nodes\n')
    out.write('of the clockwise disk ($\Omega$=96.3$\deg$) with a range of 0$\deg$-90$\deg$.}\n')
    out.write('\\tablenotetext{b}{The level above which the star\'s disk membership probability \n')
    out.write('falls for its respective angular offset bin (see Figure \\ref{fig:probNodes}).}\n')
    out.write('\\end{deluxetable}\n')
    out.close()
    out2.close()

    usetexFalse()

    
def pdfPeriod(pdfdir='aorb_thesis/', mosaic=True, suffix='_mosaic'):
    """
    Monte Carlo the periods and calculate the average and RMS
    for each trial. Then keep the resulting distribution.
    """
    outdir = root + alnDir + 'plots/'

    # Get the global set of periods for ALL solutions and for
    # DISK solutions. This also prints out useful stats
    (prd, prdDisk) = paramStats('p',pdfdir=pdfdir,mosaic=mosaic)
    
    # Determine the period probability density function
    prdStep = 100
    binsIn = np.arange(0, 1e6+prdStep, prdStep)

    print 'Median period (all solns): %8.3f years' % np.median(prd)
    print 'Median period (disk solns): %8.3f years' % np.median(prdDisk)

    (pbdisp1, pndisp1) = histNofill.hist(binsIn, prd, normed=True)
    (pbdisp2, pndisp2) = histNofill.hist(binsIn, prdDisk, normed=True)

    py.clf()
    py.figure(figsize=(7,6))
    py.subplots_adjust(left=0.15, right=0.95)
    py.plot(pbdisp1, pndisp1, color='black')
    py.xlabel('Period (years)')
    py.ylabel('Probability Density')
    py.axis([0,1e4,0,0.0015])
    py.savefig(outdir + 'pdf_period_all%s.png' % suffix)

    py.clf()
    py.figure(figsize=(7,6))
    py.subplots_adjust(left=0.15, right=0.95)
    py.plot(pbdisp2, pndisp2, color='black')
    py.xlabel('Period (years)')
    py.ylabel('Probability Density')
    py.axis([0,1e4,0,0.0015])
    py.savefig(outdir + 'pdf_period_all_disk%s.png' % suffix)

    py.clf()
    py.figure(figsize=(7,6))
    py.subplots_adjust(left=0.15, right=0.95)
    p1 = py.plot(pbdisp1, pndisp1, color='gray', linewidth=2)
    p2 = py.plot(pbdisp2, pndisp2, color='black', linewidth=2)
    #py.legend((p1, p2), ('Unweighted', 'Weighted'))
    py.legend((p1, p2), ('All Solutions', 'Disk Solutions'))
    py.xlabel('Period (years)')
    py.ylabel('Probability Density')
    py.title('Disk Candidates')
    py.axis([0,1e4,0,0.0015])
    #py.xlim(0, 1)
    py.savefig(outdir + 'pdf_period_both%s.png' % suffix)


def paramStats(variable, pdfdir='aorb_thesis/',mosaic=True,simFlat=False,
               accelOnly=False,nonDisk=False,allStars=False,LHsigCut=3.0,radBin=None):
    """
    LHsigCut (float):  Significance threshold for non-members;
            	       default is set to 3.0, meaning that stars with
                       likelihood of not being on the disk of >3 sigma
                       are considered non-members. The rest are candidates.
    radBin (float):    Specify the radial interval to return the parameter values for
    		       stars in a specified radial bin. For example, radBin=[0.8,2.0].
    """
    cc = objects.Constants()
    # Load names of young stars 
    if mosaic == True:
        # Load up mosaic data as well; select only stars at r>4, since
        # we don't want to add any info from mosaics if we have it in
        # the central 10" already
        yng1 = young.loadYoungStars(root+alnDir,fit=poly,points=points,
                                    withRVonly=True,silent=True,skipStar=['S5-237']) 
        yng2 = young.loadYoungStars(mscDir,fit=polyM,points=pointsM,
                                    mosaic=True, withRVonly=True,silent=True)
        # Merge this object with object from central 10" analysis
        yng = merge(yng1, yng2)
    else:
        yng = young.loadYoungStars(root+alnDir,fit=poly,points=points,
                                   withRVonly=True,silent=True)
    yngNames = yng.getArray('name')
    sidx = np.array(yngNames).argsort()
    yng.stars = [yng.stars[ii] for ii in sidx]

    # Read in the table of disk membership probabilities
    if mosaic == True:
        diskTab = asciidata.open(root + alnDir + 'tables/disk_membership_prob_mosaic.dat')
    else:
       diskTab = asciidata.open(root + alnDir + 'tables/disk_membership_prob.dat')
    name = [diskTab[0][ss].strip() for ss in range(diskTab.nrows)]
    diskP = diskTab[1].tonumpy()

    # What is the threshold for disk/non-disk members?
    lhNotOnDisk_cut = scipy.special.erf(LHsigCut/np.sqrt(2.))
    probOnDisk_cut = 1.0 - lhNotOnDisk_cut
    print probOnDisk_cut

    if allStars == False:
        if nonDisk == False:
            # get disk stars
            if probOnDisk_cut == 0.0:
                idx = (np.where(diskP >= probOnDisk_cut))[0]
            else:
                idx = (np.where(diskP > probOnDisk_cut))[0]
        elif nonDisk == True:
            # get non-disk stars
            idx = (np.where(diskP < probOnDisk_cut))[0]
    elif allStars == True:
        idx = np.arange(len(yng.stars))
        
    yng.stars = [yng.stars[ii] for ii in idx]

    # Do we want disk candidates in a particular radial bin?
    if radBin != None:
        r1 = radBin[0]   
        r2 = radBin[1]
        r2d = np.array([yng.stars[rr].r2d for rr in range(len(yng.stars))])
        kp = np.where((r2d >= r1) & (r2d < r2))[0]
        yng.stars = [yng.stars[ii] for ii in kp]
        
    starCnt = len(yng.stars)
    numTrials = 99998
    print 'Number of disk candidates: %i' % starCnt

    param = np.zeros((starCnt, numTrials), float)
    angle = np.zeros((starCnt, numTrials), float)

    sinip = np.sin(np.radians(idisk))
    cosip = np.cos(np.radians(idisk))

    # Loop through and trim down to only disk stars
    for ii in range(starCnt):
        yngStar = yng.stars[ii]
        if yngStar.vz == None:
            continue

        # File contains analytic orbit solutions with acceleration limits (MC)
        if simFlat == True:
            pdffile = '%s%s%s%s.disk_mc.dat' % (root, alnDir, pdfdir, yng.stars[ii].name)
        else:
            pdffile = '%s%s%s%s.mc.dat' % (root, alnDir, pdfdir, yng.stars[ii].name)
        pdf = pickle.load(open(pdffile))

        print 'Adding disk star %s' % yng.stars[ii].name

        # Determine angular offset to disk for each solution
        sini = np.sin(pdf.i * np.pi / 180.0)
        cosi = np.cos(pdf.i * np.pi / 180.0)
        cosodiff = np.cos( (pdf.o - odisk) * np.pi / 180.0 )
        angle[ii,:] = np.arccos( (sini * sinip * cosodiff) + (cosi * cosip) )
        angle[ii,:] *= 180.0 / np.pi

        if (variable == 'e'):
            param[ii,:] = pdf.e
        elif (variable == 'ph'):
            param[ii,:] = pdf.ph
        elif (variable == 'ph2'):
            tmp = (yng.stars[ii].fitXa.t0 - pdf.t0) / pdf.p
            neg = (np.where(tmp < 0))[0]
            tmp[neg] += 1.0
            param[ii,:] = tmp
        elif (variable == 'ph3'): # Convention used in Beloborodov & Levin (2004)
            refTime = yng.stars[ii].fitXa.t0
            tidx = np.where(pdf.t0 > refTime)[0]
            tmp = pdf.t0
            tmp[tidx] = pdf.t0[tidx] - pdf.p[tidx]
            tperi_last = tmp # time of last periapse passage
            t_apo = tperi_last + (pdf.p / 2.0) # time of apoapse passage
            aidx = np.where(t_apo > refTime)[0] # orbits where polyfit's time is before apoapse
            bidx = np.where(t_apo <= refTime)[0] # orbits where polyfit's time is after apoapse
            param[ii,aidx] = 2.0 * (refTime - tperi_last[aidx]) / pdf.p[aidx]
            param[ii,bidx] = 2.0 * (pdf.p[bidx] - (refTime - tperi_last[bidx])) / pdf.p[bidx]
            # NOTE: I tested the above choices for phase calculation and
            # the cases ph and ph3 are identical.
        elif (variable == 'w'):
            param[ii,:] = pdf.w
        elif (variable == 'i'):
            param[ii,:] = pdf.i
        elif (variable == 'o'):
            param[ii,:] = pdf.o
        elif (variable == 'p'):
            param[ii,:] = pdf.p
	elif (variable == 't0'):
	    param[ii,:] = pdf.t0
	elif (variable == 'r'):
	    x = pdf.x
	    y = pdf.y
	    z = pdf.z
	    param[ii,:] = sqrt(x**2 + y**2 + z**2)
	elif (variable == 'x'):
	    param[ii,:] = pdf.x
	elif (variable == 'y'):
	    param[ii,:] = pdf.y
	elif (variable == 'z'):
	    param[ii,:] = pdf.z
	elif (variable == 'm'):
	    param[ii,:] = pdf.m
        else:
            print 'Incorrect variable name: ', variable

    # Now calculate the average and standard deviation
    # For each trial calculate the:
    #   - avg
    #   - std
    #   - rms
    avg_trial = np.sum(param, axis=0) / starCnt
    rms_trial = np.sqrt(np.sum(param**2, axis=0) / starCnt)
    std_trial = np.sqrt(np.sum((param - avg_trial)**2, axis=0) / (starCnt - 1))

    # For all trials calculate the average, rms, and std for each
    avg_avg = avg_trial.mean()
    avg_rms = np.sqrt( np.sum(avg_trial**2) / numTrials )
    avg_std = np.sqrt( np.sum((avg_trial - avg_avg)**2) / (numTrials - 1) )
    rms_avg = rms_trial.mean()
    rms_rms = np.sqrt( np.sum(rms_trial**2) / numTrials )
    rms_std = np.sqrt( np.sum((rms_trial - rms_avg)**2) / (numTrials - 1) )
    std_avg = std_trial.mean()
    std_rms = np.sqrt( np.sum(std_trial**2) / numTrials )
    std_std = np.sqrt( np.sum((std_trial - std_avg)**2) / (numTrials - 1) )
    print 'All Disk Stars, All Solutions:'
    print '  Avg = %4.2f +- %4.2f' % (avg_avg, avg_std)
    print '  RMS = %4.2f +- %4.2f' % (rms_avg, rms_std)
    print '  Std = %4.2f +- %4.2f' % (std_avg, std_std)

    ##########
    #
    # Now only use disk solutions
    #   -- assumes disk solutions are within 15 degrees
    #   -- weights by probability of being in the disk
    #
    ##########
    if nonDisk == False:
        # Loop through and trim down to only disk stars
        paramDisk = np.arange(0, dtype=float)
        avg_trial_disk = np.zeros(numTrials, float)
        rms_trial_disk = np.zeros(numTrials, float)
        std_trial_disk = np.zeros(numTrials, float)
    
        oldNumTrials = numTrials
    
        for nn in range(numTrials):
            adx = (np.where(angle[:,nn] < angleCut))[0]
            trial = param[adx,nn]
    
            if (len(adx) < 2):
                numTrials -= 1.0
                continue
    
            avg_trial_disk[nn] = trial.mean()
            rms_trial_disk[nn] = np.sqrt( np.sum(trial**2) / len(trial) )
            std_trial_disk[nn] = trial.std(ddof=1)
    
        print 'Found %d trials with less than 2 stars in the disk' % \
              (oldNumTrials - numTrials)
    
        for ii in range(starCnt):
            idx = (np.where(angle[ii,:] < angleCut))[0]
            paramDisk = np.concatenate((paramDisk, param[ii,idx]))
    
        # For all trials calculate the average, rms, and std for each
        avg_avg = np.sum( avg_trial_disk ) / numTrials
        avg_rms = np.sqrt( np.sum(avg_trial_disk**2) / numTrials )
        avg_std = np.sqrt( np.sum((avg_trial_disk - avg_avg)**2) / (numTrials - 1) )
        rms_avg = np.sum( rms_trial_disk ) / numTrials
        rms_rms = np.sqrt( np.sum(rms_trial_disk**2) / numTrials )
        rms_std = np.sqrt( np.sum((rms_trial_disk - rms_avg)**2) / (numTrials - 1) )
        std_avg = np.sum( std_trial_disk ) / numTrials
        std_rms = np.sqrt( np.sum(std_trial_disk**2) / numTrials )
        std_std = np.sqrt( np.sum((std_trial_disk - std_avg)**2) / (numTrials - 1) )
        print 'All Disk Stars, Disk Solutions, Weighted by Disk Prob:'
        print '  Avg = %4.2f +- %4.2f' % (avg_avg, avg_std)
        print '  RMS = %4.2f +- %4.2f' % (rms_avg, rms_std)
        print '  Std = %4.2f +- %4.2f' % (std_avg, std_std)
    
        _out = open(root + alnDir + 'tables/disk_stars_param_%s.dat' % variable, 'w')
        print 'Individual Orbital Elements (%s):' % variable
        for ii in range(starCnt):
            idx = (np.where(angle[ii,:] < angleCut))[0]
            print '  %12s  %5.2f +- %5.2f (all)    %5.2f +- %5.2f (disk)' % \
                  (yng.stars[ii].name,
                   param[ii,:].mean(), param[ii,:].std(ddof=1),
                   param[ii,idx].mean(), param[ii,idx].std(ddof=1))
            _out.write('%-12s  %5.2f +- %5.2f (all)    %5.2f +- %5.2f (disk)\n' % \
                       (yng.stars[ii].name,
                        param[ii,:].mean(), param[ii,:].std(ddof=1),
                        param[ii,idx].mean(), param[ii,idx].std(ddof=1)))
    
    
        _out.close()
        
    if nonDisk == False:
        return (param, paramDisk)
    else:
        return param

def simulate_flat_disk(pdftrials, doMassRo=None, mExtended=None,
                      mosaic=False):
    """
    Calls analyticOrbits.py Class simulate_flat_disk(), which uses the
    line of sight distance that will put the star on the CW disk. The
    (i,O) solution for the disk must already have been determined.
    """

    # Load names of young stars 
    if mosaic == True:
        # Load up mosaic data as well; select only stars at r>4, since
        # we don't want to add any info from mosaics if we have it in
        # the central 10" already
        yng1 = young.loadYoungStars(root+alnDir,fit=poly,points=points,
                                    withRVonly=True,silent=True,skipStar=['S5-237']) 
        yng2 = young.loadYoungStars(mscDir,fit=polyM,points=pointsM,
                                    mosaic=True, withRVonly=True,silent=True)
        cntrlNames = yng1.getArray('name')
        mscNames = yng2.getArray('name')
        # Merge this object with object from central 10" analysis
        yng = merge(yng1, yng2)
    else:
        yng = young.loadYoungStars(root+alnDir,fit=poly,points=points,
                                   withRVonly=True,silent=True)

    names = yng.getArray('name') 
    names.sort()
    yngNames = yng.getArray('name')

    pdfdir = 'aorb_acc_mrPDF_MC_newMosaic/'
    flatdir = root + alnDir + 'flatdisk/'

    starCnt = len(yngNames)
    numTrials = 99998
    angle = np.zeros((starCnt, numTrials), float)
    sinip = np.sin(np.radians(idisk))
    cosip = np.cos(np.radians(idisk))

    # Get disk membership
    diskTab = asciidata.open(root+alnDir+'tables/disk_membership_prob_mosaic.dat')
    name = [diskTab[0][ss].strip() for ss in range(diskTab.nrows)]
    diskP = diskTab[1].tonumpy()
    diskIdx = (np.where(diskP > 2.7e-3))[0]

    py.clf()

    print
    print 'Running flat disk simulation using candidate disk members only'
    disk_cnt = 0
    edisk = []
    for ii in range(len(names)):
        mscStar = False
        
        if mosaic == True:
            if (names[ii] in mscNames) & (names[ii] not in cntrlNames):
                mscStar = True

        # Get the star object
        idx = yngNames.index(names[ii])
        yngStar = yng.stars[idx]

        # Get the membership probability -- we only want disk candidates
        # first make sure the ordering of stars is the same
        if name[ii] != names[ii]:
            'WARNING: Disk membership file is not in the same order as align files!!'
            diskP[ii] = 1.0 # arbitrarily setting to 1.0 and using this star

        if diskP[ii] < 2.7e-3: # not a disk candidate
            continue

        disk_cnt += 1
        # File contains analytic orbit solutions with acceleration limits (MC)
        pdffile = '%s%s%s%s.mc.dat' % (root, alnDir, pdfdir, names[ii])
        pdf = pickle.load(open(pdffile))

        # Determine angular offset to disk for each solution
        sini = np.sin(pdf.i * np.pi / 180.0)
        cosi = np.cos(pdf.i * np.pi / 180.0)
        cosodiff = np.cos( (pdf.o - odisk) * np.pi / 180.0 )
        angle[ii,:] = np.arccos( (sini * sinip * cosodiff) + (cosi * cosip) )
        angle[ii,:] *= 180.0 / np.pi

        # Find solution that is closest to the disk.
        #inD = (np.where(angle[ii,:] == angle[ii,:].min()))[0]
        inD = angle[ii,:].argmin()
        min_angle = angle[ii,inD]
        # For this minimum angle from the disk, what is the z value?
        zdisk = pdf.z[inD]

        # What is the eccentricity for this z? Can compare to resulting ecc distribution
        edisk = np.concatenate([edisk, [pdf.e[inD]]])

        print 'MC ON: %s' % (names[ii])
        print '  assuming z = %8.3f arcsec on disk' % zdisk

        # Run a Monte Carlo to determine the probability density function (PDF)
        # for this star, assuming it's on the disk. Save it off to a file
        mcfile = '%s%s.disk_mc.dat' % (flatdir, names[ii])
        mc = aorb.simulate_flat_disk(yngStar, ntrials=pdftrials,
                                     outroot=flatdir, zDisk=zdisk, mscStar=mscStar)
        mc.runmc(mcMassRo=doMassRo, mExtended=mExtended)
        mc.makePdfHealpix(makeplot=True)
        mc.saveToFile(mcfile)

    print
    print 'Flat disk simulation complete. Used %i candidate disk stars.' % disk_cnt

    # Plot up the eccentricity distribution for the solution that puts the star on the disk
    py.clf()
    py.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.1)
    binsIn = np.arange(0, 1.1, 0.1)
    py.hist(edisk, bins=binsIn, color='k',histtype='step')
    py.xlabel('Eccentricity on Disk')
    py.ylabel('N')
    py.axis([0, 1, 0, 10])
    py.savefig(plotdir + 'simulate_flat_ecc_on_disk.png')


def plot_velocity(mosaic=True, suffix=''):
    """'
    Plots velocities and compares to what is expected if stars are bound.
    """

    # Load up young stars
    # young.loadYoungStars calls youngStarNames but grabs
    # only young stars beyond r = 0.8 arcsec
    # Load names of young stars 
    if mosaic == True:
        # Load up mosaic data as well; select only stars at r>4, since
        # we don't want to add any info from mosaics if we have it in
        # the central 10" already
        yng1 = young.loadYoungStars(root+alnDir,fit=poly,points=points,radiusCut=0.8,
                                    withRVonly=True,skipStar=['S5-237'])
        yng2 = young.loadYoungStars(mscDir,fit=polyM,points=pointsM,
                                    mosaic=True, withRVonly=True,silent=True)
        cntrlNames = yng1.getArray('name')
        mscNames = yng2.getArray('name')
        # Merge this object with object from central 10" analysis
        yng = merge(yng1, yng2)
    else:
        yng = young.loadYoungStars(root+alnDir,fit=poly,points=points,radiusCut=0.8,
                                   withRVonly=True)
        mscNames = ''
        cntrlNames = ''

    cc = objects.Constants()
    GM = cc.G * mass * cc.msun
    asy_to_kms = dist * cc.cm_in_au / (1.e5 * cc.sec_in_yr)

    # Construct an array of radii out to 15 arcsec in steps of 0.1''
    r = (np.arange(10 * 15) * 0.1) + 0.1
    r_au = r * dist
    r_pc = r_au / cc.au_in_pc
    r_cm = r_au * cc.cm_in_au

    # Also determine theoretical escape velocity curve
    v_esc_cm_s = np.sqrt(2. * GM / r_cm)
    v_esc_km_s = v_esc_cm_s / 1.0e5

    # Error on the computed theoretical v_esc curve
    sig_vesc_cgs = massErr * cc.msun * \
                   np.sqrt(cc.G / (2. * mass * cc.msun * r_cm))
    sig_vesc_mks = sig_vesc_cgs / 1.0e5

    # Loop through all the stars.
    names = yng.getArray('name')
    rv_ref = np.array(yng.getArray('rv_ref'))

    # Setup i/Omega for each pixel on sky
    nside = 64
    npix = healpy.nside2npix(nside)
    (iheal, oheal) = healpy.pix2ang(nside, np.arange(0, npix))
    iheal *= rad2deg
    oheal *= rad2deg

    idisk = np.zeros((len(names)),dtype=float)
    odisk = np.zeros((len(names)),dtype=float)

    # Read in file listing accelerating sources
    _acc = asciidata.open('%s/%s/tables/accelerating_sources.dat' % (root, alnDir))
    acc = _acc[0].tonumpy()
    accels = [aa.strip() for aa in acc]


    rLower_all = []
    cnt_all = []
    x_all = []
    y_all = []
    r2d_all = []
    xe_all = []
    ye_all = []
    vx_all = []
    vy_all = []
    vz_all = []
    vxe_all = []
    vye_all = []
    vze_all = []
    vtot_all = []
    vtoterr_all = []
    pmErr_all = []
    vesc_ratio_all = []
    msc = []
    xchi2_all = []
    ychi2_all = []
    names_all = []
    mag_all = []
    for ii in range(len(names)):
        name = names[ii]
	i = names.index(name)

	star = yng.stars[i]
        mag = yng.stars[i].mag

        mscStar = False

        if (name in mscNames) & (name not in cntrlNames):
            # For some reason, mosaic velocities from vel fit come
            # out in km/s; whereas central 10'' data come out in asec/yr
            # for the velocity fits, and km/s for acceleration fits.
            mscStar = True
            x = star.fitXv.p # arcsec
            y = star.fitYv.p # arcsec
            r2d = np.hypot(x, y) # arcsec
            xe = star.fitXv.perr # arcsec
            ye = star.fitYv.perr # arcsec
            vx = star.fitXv.v  # km/s
            vy = star.fitYv.v  # km/s
            vxe = star.fitXv.verr # km/s
            vye = star.fitYv.verr # km/s
            xchi2 = star.fitXv.chi2
            ychi2 = star.fitYv.chi2
        else:
            if name in accels:
                x = star.fitXa.p # arcsec 
                y = star.fitYa.p # arcsec
                r2d = np.hypot(x, y) # arcsec
                xe = star.fitXa.perr # arcsec
                ye = star.fitYa.perr # arcsec
                vx = star.fitXa.v # km/s
                vy = star.fitYa.v # km/s
                vxe = star.fitXa.verr # km/s
                vye = star.fitYa.verr # km/s
                xchi2 = star.fitXa.chi2
                ychi2 = star.fitYa.chi2
            else:
                x = star.fitXv.p # arcsec
                y = star.fitYv.p # arcsec
                r2d = np.hypot(x, y) # arcsec
                xe = star.fitXv.perr # arcsec
                ye = star.fitYv.perr # arcsec
                vx = star.fitXv.v * asy_to_kms # km/s
                vy = star.fitYv.v * asy_to_kms # km/s
                vxe = star.fitXv.verr * asy_to_kms# km/s
                vye = star.fitYv.verr * asy_to_kms # km/s
                xchi2 = star.fitXv.chi2
                ychi2 = star.fitYv.chi2
    
        vz = star.vz # km/s
        vze = star.vzerr # km/s
        t0 = star.fitXv.t0

        pm = np.sqrt(vx**2 + vy**2) # km/s
        pm_err = (np.sqrt(((vx*vxe)**2 + (vy*vye)**2) / pm**2)) # km/s

        vtot = np.sqrt(vx**2 + vy**2 + vz**2) # km/s
        vtot_err = (np.sqrt(((vx*vxe)**2 + (vy*vye)**2 + (vz*vze)**2) / vtot**2)) # km/s

        vtot_all = np.concatenate([vtot_all, [vtot]]) # km/s
        vtoterr_all = np.concatenate([vtoterr_all, [vtot_err]]) # km/s

        pmErr_all = np.concatenate([pmErr_all, [pm_err]]) # mas/yr

        xchi2_all = np.concatenate([xchi2_all, [xchi2]])
        ychi2_all = np.concatenate([ychi2_all, [ychi2]])

        # How many epochs was each star detected in?
        if mscStar == False:
            pts = asciidata.open('%s%s%s%s.points' % (root, alnDir, points, name))
        else:
            pts = asciidata.open('%s%s%s.points' % (mscDir, pointsM, name))
        ep = pts[0].tonumpy()
        cnt = len(ep)
        cnt_all = np.concatenate([cnt_all, [cnt]]) 

        r = np.sqrt(x**2 + y**2)
        rcgs = r * dist * cc.cm_in_pc / cc.au_in_pc
        
	# Lower allowed radius is set by 2D distance.
	rLower = r * dist / cc.au_in_pc # pc

        # Get the ratio of velocity to escape velocity at this r2d
        v_esc_at_r = np.sqrt(2. * GM / rcgs)
        v_esc_at_r /= 1.e5
        vesc_ratio = vtot / v_esc_at_r

        # Save off some info
        rLower_all = np.concatenate([rLower_all, [rLower]]) # pc
        vesc_ratio_all = np.concatenate([vesc_ratio_all, [vesc_ratio]]) # km/s

        # Save off some things for plotting later
        mag_all = np.concatenate([mag_all, [mag]])
        x_all = np.concatenate([x_all, [x*1.e3]]) # mas
        y_all = np.concatenate([y_all, [y*1.e3]]) # mas 
        r2d_all = np.concatenate([r2d_all, [r2d*1.e3]]) # mas 
        xe_all = np.concatenate([xe_all, [xe*1.e3]]) # mas
        ye_all = np.concatenate([ye_all, [ye*1.e3]]) # mas 
        vx_all = np.concatenate([vx_all, [vx]]) # km/s
        vy_all = np.concatenate([vy_all, [vy]]) # km/s
        vz_all = np.concatenate([vz_all, [vz]]) # km/s
        vxe_all = np.concatenate([vxe_all, [vxe]]) # km/s
        vye_all = np.concatenate([vye_all, [vye]]) # km/s
        vze_all = np.concatenate([vze_all, [vze]]) # km/s
        names_all = np.concatenate([names_all, [name]]) # km/s

        # Save off which were mosaic stars
        msc = np.concatenate([msc, [mscStar]])

        # Next get orbital info
        orbFile = root + alnDir + '/aorb_thesis/' + name + '_mc_heal.dat'
        #orbFile = root + alnDir + '/aorb_acc_mrPDF_MC_newMosaic/' + name + '_mc_heal.dat'
        pdf = np.fromfile(orbFile, dtype=float)

        # Find the peak of the PDF
        sid = (pdf.argsort())[::-1]  # reverse sort
        peakPix = sid[0]
        idisk[ii] = iheal[peakPix]
        odisk[ii] = oheal[peakPix]
        
    #temp = np.array(['S7-16', 'S8-15', 'S9-23', 'S10-4', 'S10-5'])
    temp = ['S6-100','S7-16','S8-15','S9-23','S10-4','S10-5']
    usetexTrue()
    py.clf()
    py.figure(1)
    py.figure(figsize=(6,6))
    py.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.1,
                       wspace=0.3, hspace=0.3)
    # plot theoretical escape velocity curve
    py.plot(r_pc, v_esc_km_s, 'k--')
    py.plot(r_pc, (v_esc_km_s+3.*sig_vesc_mks), 'k:')
    py.plot(r_pc, (v_esc_km_s-3.*sig_vesc_mks), 'k:')
    # Plot velocity measurements and 1 sigma errors
    for ii in range(len(vtot_all)):
        # mark the stars with Omega ~ 180 degrees
        #if np.abs(odisk[ii] - 180.) < 10.:
        #    py.plot(rLower_all[ii], vtot_all[ii], 'kx', ms=8, mew=1.5)
        if names[ii] in temp:
            py.plot(rLower_all[ii], vtot_all[ii], 'kx', ms=8, mew=1.5)
        if msc[ii] == True:
            py.errorbar(rLower_all[ii], vtot_all[ii], yerr=vtoterr_all[ii], fmt='g.')
        else:
            py.errorbar(rLower_all[ii], vtot_all[ii], yerr=vtoterr_all[ii], fmt='k.')

    py.text(0.1, 700, r'{\bf $v_{esc} = \sqrt{\frac{2GM}{\rho}} (+3\sigma$)',color='g')
    py.xlabel(r'{\bf Projected Radius (pc)}')
    py.ylabel(r'{\bf v$_{tot}$ (km/s)}')
    py.axis([0,0.55,0,1000])
    if mosaic == True:
        py.savefig(plotdir + 'v3d_vEsc_r2d_mosaic.png')
    else:
        py.savefig(plotdir + 'v3d_vEsc_r2d.png')
    py.close(1)

    py.clf()
    py.figure(2)
    py.figure(figsize=(10,6))
    py.subplots_adjust(left=0.08, right=0.98, top=0.9, bottom=0.1,
                       wspace=0.3, hspace=0.3)
    py.subplot(1,2,1)
    binsIn = np.arange(0, 100, 1)
    py.hist(vxe_all, bins=binsIn, color='r',histtype='step', label='x')
    py.hist(vye_all, bins=binsIn, color='b',histtype='step', label='y')
    py.hist(vze_all, bins=binsIn, color='g',histtype='step', label='z')
    py.legend(numpoints=1, fancybox=True)
    py.xlabel('Velocity Error (km/s)')
    py.ylabel('N')
    py.subplot(1,2,2)
    py.hist(vtoterr_all, bins=binsIn, color='k',histtype='step')
    py.xlabel('Total Velocity Error (km/s)')
    py.ylabel('N')
    if mosaic == True:
        py.savefig(plotdir + 'hist_vel_error_mosaic.png')
    else:
        py.savefig(plotdir + 'hist_vel_error.png')
    usetexFalse()
    py.close(2)

    c1 = 0
    c2 = 0
    c3 = 0

    import matplotlib.font_manager
    prop = matplotlib.font_manager.FontProperties(size=12)

    usetexTrue()
    # Get disk membership and plot velocity vectors on a mosaic image
    imgFile = '/u/ghezgroup/data/gc/08maylgs2/combo/mag08maylgs2_dp_msc_kp.fits'
    scale = 0.00993
    sgra = [1398.4, 1500.4]
    img = pyfits.getdata(imgFile)
    imgsize = (img.shape)[0]
    # Make axes for images in arcsec
    pixL = np.arange(0,imgsize)
    xL = [-1*(xpos - sgra[0])*scale for xpos in pixL]
    yL = [(ypos - sgra[1])*scale for ypos in pixL]
    diskTab = asciidata.open(root+alnDir+'tables/disk_membership_prob_mosaic.dat')
    nameDM = [diskTab[0][ss].strip() for ss in range(diskTab.nrows)]
    diskP = diskTab[1].tonumpy()

    diskIdx = (np.where(diskP > 2.7e-3))[0]
    py.figure(3)
    py.figure(figsize=(7,7))
    py.subplots_adjust(left=0.1, right=0.98, top=0.95, bottom=0.1)
    py.clf()
    py.imshow(np.log10(img+1), aspect='equal', interpolation='bicubic',
              extent=[max(xL), min(xL), min(yL), max(yL)],vmin=2.2,vmax=5,
              origin='lowerleft', cmap=py.cm.gray_r)
    for ii in range(len(nameDM)):
        ss = np.where(np.array(names) == nameDM[ii])[0]
        if diskP[ii] >= 2.7e-3: # candidates after excluding stars at 3 sigma level
            qvr = py.quiver([x_all[ss]/1.e3], [y_all[ss]/1.e3], [vx_all[ss]], [vy_all[ss]], headwidth=1.5,
                      minshaft=1.5, color='red', units='y', angles='xy', scale=200)
        else: # everything else (not on disk)
            py.quiver([x_all[ss]/1.e3], [y_all[ss]/1.e3], [vx_all[ss]], [vy_all[ss]], headwidth=1.5,
                      minshaft=1.5, color='black', units='y', angles='xy', scale=200)
        ##if diskP[ii] >= 0.1:
        #if diskP[ii] >= 0.3173: # stars not on disk at 1 sigma level, the rest are candidates
        #    py.quiver([x_all[ss]/1.e3], [y_all[ss]/1.e3], [vx_all[ss]], [vy_all[ss]], headwidth=1.5,
        #              minshaft=1.5, color='red', units='y', angles='xy', scale=200)
        #    print nameDM[ii], '1'
        #    c1 += 1 # membership category
        ##elif (diskP[ii] >= 2.7e-3) and (diskP[ii] < 0.1):
        #elif (diskP[ii] >= 0.0455): # stars not on disk at 2 sigma level 
        #    py.quiver([x_all[ss]/1.e3], [y_all[ss]/1.e3], [vx_all[ss]], [vy_all[ss]], headwidth=1.5,
        #              minshaft=1.5, color='green', units='y', angles='xy', scale=200)
        #    print nameDM[ii], '2'
        #    c2 += 1
        #elif (diskP[ii] >= 2.7e-3): # stars not on disk at 3 sigma level 
        #    py.quiver([x_all[ss]/1.e3], [y_all[ss]/1.e3], [vx_all[ss]], [vy_all[ss]], headwidth=1.5,
        #              minshaft=1.5, color='blue', units='y', angles='xy', scale=200)
        #    print nameDM[ii], '2'
        #    c3 += 1
        #else: # everything else, which have disk prob < 2.7e-3 (not on disk)
        #    py.quiver([x_all[ss]/1.e3], [y_all[ss]/1.e3], [vx_all[ss]], [vy_all[ss]], headwidth=1.5,
        #              minshaft=1.5, color='black', units='y', angles='xy', scale=200)
        #    print nameDM[ii], '3'
        #    #print '%8s  %5.2f  %5.2f  %5.2f  %5.2f  %4.1e' % \
        #    #      (nameDM[ii], x_all[ss]/1.e3, y_all[ss]/1.e3, vx_all[ss], vy_all[ss], diskP[ii])

    #print
    #print 'Disk Membership:'
    #print '  Category 1: N = %d' % c1
    #print '  Category 2: N = %d' % c2
    #print '  Category 3: N = %d' % c3
    py.quiverkey(qvr,  10, 10, 300, '300 km/s', coordinates='data', color='black')
    py.plot([0],[0],'k+',ms=7,mew=2)
    an = np.linspace(0,2*np.pi,100)
    py.plot(0.8*np.cos(an),0.8*np.sin(an),'k--', lw=1.5) # marks radial bins used
    py.plot(3.2*np.cos(an),3.2*np.sin(an),'k--', lw=1.5) # marks radial bins used
    py.plot(6.5*np.cos(an),6.5*np.sin(an),'k--', lw=1.5) # marks radial bins used
    #py.plot(10.0*np.cos(an),10.0*np.sin(an),'k--', lw=1) # marks radial bins used
    # Plot the line of nodes using the best (i,O) solution:
    py.axis([12.5, -14.0, -15.0, 12.5])
    py.xlabel('X (arcsec)',fontsize=16)
    py.ylabel('Y (arcsec)',fontsize=16)
    #py.title('Young Star Velocities',fontsize=16)
    py.savefig(plotdir + 'velVector_yngstars_diskMmbr' + suffix + '.png')
    py.savefig(plotdir + 'eps/velVector_yngstars_diskMmbr' + suffix + '.eps')
    py.close(3)

    py.figure(7)
    py.figure(figsize=(8,8))
    py.subplots_adjust(left=0.1, right=0.98, top=0.95, bottom=0.1)
    py.clf()
    py.imshow(np.log10(img+1), aspect='equal', interpolation='bicubic',
              extent=[max(xL), min(xL), min(yL), max(yL)],vmin=2.2,vmax=5,
              origin='lowerleft', cmap=py.cm.gray_r)
    py.plot([0],[0],'rx', ms=7, mew=1.5)
    for ss in range(len(names_all)):
        py.plot(x_all[ss]/1.e3,y_all[ss]/1.e3,'rx',ms=5)
        py.text(x_all[ss]/1.e3,y_all[ss]/1.e3+0.2,('%5s' % names[ss]),fontsize=8,color='k')
    an = np.linspace(0,2*np.pi,100)
    py.plot(0.8*np.cos(an),0.8*np.sin(an),'k--')
    py.plot(3.2*np.cos(an),3.2*np.sin(an),'k--')
    py.plot(6.473*np.cos(an),6.473*np.sin(an),'k--')
    py.plot([0,0],[-15,15],'k--')
    py.plot([15,-15],[0,0],'k--')
    py.axis([15,-15,-15,13])
    py.xlabel('X (arcsec)',fontsize=16)
    py.ylabel('Y (arcsec)',fontsize=16)
    py.savefig(plotdir + 'plot_names.png')
    py.close(7)

    py.figure(8)
    py.figure(figsize=(8,8))
    py.subplots_adjust(left=0.1, right=0.98, top=0.95, bottom=0.1)
    py.clf()
    py.imshow(np.log10(img+1), aspect='equal', interpolation='bicubic',
              extent=[max(xL), min(xL), min(yL), max(yL)],vmin=2.2,vmax=5,
              origin='lowerleft', cmap=py.cm.gray_r)
    py.plot([0],[0],'rx', ms=7, mew=1.5)
    for ss in range(len(names_all)):
        if np.sqrt((x_all[ss]/1.e3)**2 + (y_all[ss]/1.e3)**2) > 4.:
            continue
        py.plot(x_all[ss]/1.e3,y_all[ss]/1.e3,'rx',ms=7)
        py.text(x_all[ss]/1.e3,y_all[ss]/1.e3+0.1,('%5s' % names[ss]),fontsize=12,color='k')
    py.axis([3.5,-3.5,-3.5,3.5])
    py.xlabel('X (arcsec)',fontsize=16)
    py.ylabel('Y (arcsec)',fontsize=16)
    py.savefig(plotdir + 'plot_names_central.png')
    py.close(8)

    py.close('all')
    # Plot the velocity vectors with the line of nodes, and highlight
    # the stars within 1" from the line
    nodeStars = np.array([''])
    #nodeStars = np.array(['S3-3', 'S3-5', 'S3-10', 'S3-314', 'S3-190', 'S4-36',
    #                      'S4-169', 'S5-231', 'S6-81', 'S6-82', 'irs34W',
    #                      'S7-161','S10-32'])
    #nodeStars = np.array(['S3-3', 'S3-5', 'S3-314', 'S4-169', 'S5-231',
     #                     'S5-237', 'S6-81', 'irs1W', 'S7-161', 'S10-32'])
    Omdisk = 96.3
    # Get disk membership and plot velocity vectors on a mosaic image
    imgFile = '/u/ghezgroup/data/gc/08maylgs2/combo/mag08maylgs2_dp_msc_kp.fits'
    scale = 0.00993
    sgra = [1398.4, 1500.4]
    img = pyfits.getdata(imgFile)
    imgsize = (img.shape)[0]
    # Make axes for images in arcsec
    pixL = np.arange(0,imgsize)
    xL = [-1*(xpos - sgra[0])*scale for xpos in pixL]
    yL = [(ypos - sgra[1])*scale for ypos in pixL]
    py.figure(4)
    py.subplots_adjust(left=0.1, right=0.98, top=0.95, bottom=0.1)
    py.clf()
    py.figure(figsize=(8,8))
    py.imshow(np.log10(img+1), aspect='equal', interpolation='bicubic',
              extent=[max(xL), min(xL), min(yL), max(yL)],vmin=2.2,vmax=5,
              origin='lowerleft', cmap=py.cm.gray_r)
    for ii in range(len(names)):
        if names[ii] in nodeStars:
            #py.text(x_all[ii]/1.e3, y_all[ii]/1.e3, names[ii], fontsize=10)
            py.quiver([x_all[ii]/1.e3], [y_all[ii]/1.e3], [vx_all[ii]], [vy_all[ii]],
                      headwidth=1.5, minshaft=1.5, color='red', units='y', angles='xy',
                      scale=200)
        else:
            py.quiver([x_all[ii]/1.e3], [y_all[ii]/1.e3], [vx_all[ii]], [vy_all[ii]],
                      headwidth=1.5, minshaft=1.5, color='blue', units='y', angles='xy',
                      scale=200)

    py.plot([0],[0],'rx')
    an = np.linspace(0,2*np.pi,100)
    xp1 = 15.0*np.cos(np.radians(90.-Omdisk))
    yp1 = 15.0*np.sin(np.radians(90.-Omdisk))
    xp2 = -15.0*np.cos(np.radians(90.-Omdisk))
    yp2 = -15.0*np.sin(np.radians(90.-Omdisk))
    py.plot([xp1,xp2],[yp1,yp2],'k--',lw=1)
    py.plot([xp1,xp2],[yp1+1,yp2+1],'k--',lw=1)
    py.plot([xp1,xp2],[yp1-1,yp2-1],'k--',lw=1)
    py.axis([12.5, -14.0, -15.0, 12.5])
    py.xlabel('X (arcsec)',fontsize=16)
    py.ylabel('Y (arcsec)',fontsize=16)
    py.title('Young Star Velocities',fontsize=16)
    py.savefig(plotdir + 'velVector_yngstars_nodes_biased' + suffix + '.png')
    #py.savefig(plotdir + 'eps/velVector_yngstars_nodes_biased' + suffix + '.eps')
    #py.show()
    py.close(4)
    
    print ''
    print 'Average (Median) Velocity Error:'
    print '   x: %5.3f +- %5.3f (%5.3f) km/s' % \
          (vxe_all.mean(), vxe_all.std(ddof=1), np.median(vxe_all))
    print '   y: %5.3f +- %5.3f (%5.3f) km/s' % \
          (vye_all.mean(), vye_all.std(ddof=1), np.median(vye_all))
    nz = np.where(vze_all > 0.0)[0]
    print '   z: %5.3f +- %5.3f (%5.3f) km/s' % \
          (vze_all[nz].mean(), vze_all[nz].std(ddof=1), np.median(vze_all[nz]))
    print '   Total: %5.3f +- %5.3f (%5.3f) km/s' % \
          (vtoterr_all.mean(), vtoterr_all.std(ddof=1), np.median(vtoterr_all))

    # Plot velocities in X and Z directions; find stars within 1 sigma of
    # zero in both coordinates. These slow moving stars will lead to a bias
    # in Omega
    vxsig = np.abs(vx_all / vxe_all)
    vzsig = np.abs(vz_all / vze_all)
    idx = np.where((vxsig < 3.) & (vzsig < 3.))[0]
    print ''
    print 'Stars with Vx and Vz consistent w/ zero within 2 sigma:'
    for ii in idx:
        print names_all[ii]
    py.clf()
    py.figure(5)
    py.errorbar(vx_all, vz_all, fmt='k.', xerr=vxe_all, yerr=vze_all)
    py.errorbar(vx_all[idx], vz_all[idx], fmt='r.',
                xerr=vxe_all[idx], yerr=vze_all[idx])
    py.plot([-50,50],[0,0],'k--')
    py.plot([0,0],[-50,50],'k--')
    py.xlabel('X Velocity (km/s)')
    py.ylabel('Z Velocity (km/s)')
    # Zoom in to very low velocities
    py.axis([-50,50,-50,50])
    py.savefig(plotdir + 'vx_vs_vz.png')
    py.close(5)

    ucla = np.where(rv_ref == 'UCLA')[0]
    vlt = np.where((rv_ref == 'Bartko+2009') | (rv_ref == 'Paumard+2006'))[0]

    prop = matplotlib.font_manager.FontProperties(size=16)
    py.clf()
    py.figure(6)
    py.figure(figsize=(6,6))
    py.subplots_adjust(left=0.14, right=0.95, top=0.95)
    py.plot(mag_all[ucla], vze_all[ucla], 'rs',mfc='None',mec='r',
            mew=1.5,ms=6,label='OSIRIS')
    py.plot(mag_all[vlt], vze_all[vlt], 'k.',ms=8,mfc='None',mec='k',
            mew=1.5,label='SINFONI')
    py.xlabel('K mag')
    py.ylabel(r'Radial Velocity Error (km s$^{-1}$)')
    py.legend(numpoints=1,fancybox=True,prop=prop,loc=2)
    py.savefig(plotdir + 'vzErr_mag.png')
    py.savefig(plotdir + 'eps/vzErr_mag.eps')
    print 'Number of RVs from UCLA: %i' % len(ucla)
    print 'Number of RVs from VLT: %i' % len(vlt)

    py.figure(9)
    py.figure(figsize=(8,8))
    py.subplots_adjust(left=0.1, right=0.98, top=0.95, bottom=0.1)
    py.clf()
    py.imshow(np.log10(img+1), aspect='equal', interpolation='bicubic',
              extent=[max(xL), min(xL), min(yL), max(yL)],vmin=2.2,vmax=5,
              origin='lowerleft', cmap=py.cm.gray_r)
    py.plot([0],[0],'rx', ms=7, mew=1.5)
    for ss in range(len(names_all)):
        if mag_all[ss] < 14.:
            py.plot(x_all[ss]/1.e3,y_all[ss]/1.e3,'ro',ms=7, mfc='None',mec='r',mew=1.5)
        else:
            py.plot(x_all[ss]/1.e3,y_all[ss]/1.e3,'bo',ms=7, mfc='None',mec='b',mew=1.5)
    py.axis([15,-15,-15,13])
    py.xlabel('X (arcsec)',fontsize=16)
    py.ylabel('Y (arcsec)',fontsize=16)
    py.savefig(plotdir + 'plot_mags_on_image.png')
    py.close(9)
    
    usetexFalse()


def plot_IO_on_mosaic():
    cc = objects.Constants()

    yng1 = young.loadYoungStars(root+alnDir,fit=poly,points=points,radiusCut=0.8,
                                    withRVonly=True,skipStar=['S5-237'])
    yng2 = young.loadYoungStars(mscDir,fit=polyM,points=pointsM,
                                mosaic=True, withRVonly=True,silent=True)

    cc.asy_to_kms = dist * cc.cm_in_au / (1.e5 * cc.sec_in_yr)

    # Merge this object with object from central 10" analysis
    yng = merge(yng1, yng2)
    yngNames = yng.getArray('name')
    x = yng.getArray('x')
    y = yng.getArray('y')
    xerr = yng.getArray('fitpXv.perr')
    yerr = yng.getArray('fitpYv.perr')
    vx = yng.getArray('fitpXv.v') * cc.asy_to_kms
    vxerr = yng.getArray('fitpXv.verr') * cc.asy_to_kms
    vy = yng.getArray('fitpYv.v') * cc.asy_to_kms
    vyerr = yng.getArray('fitpYv.verr') * cc.asy_to_kms
    vz = yng.getArray('vz')
    vzerr = yng.getArray('vzerr')
    rv_ref = np.array(yng.getArray('rv_ref'))
    vz_sig = vz / vzerr

    r2d = np.hypot(x,y)

    # Image settings
    imgFile = '/u/ghezgroup/data/gc/08maylgs2/combo/mag08maylgs2_dp_msc_kp.fits'
    scale = 0.00993
    sgra = [1398.4, 1500.4]
    img = pyfits.getdata(imgFile)
    imgsize = (img.shape)[0]
    # Make axes for images in arcsec
    pixL = np.arange(0,imgsize)
    xL = [-1*(xpos - sgra[0])*scale for xpos in pixL]
    yL = [(ypos - sgra[1])*scale for ypos in pixL]

    # Setup i/Omega for each pixel on sky
    nside = 64
    npix = healpy.nside2npix(nside)
    (iheal, oheal) = healpy.pix2ang(nside, np.arange(0, npix))
    iheal *= rad2deg
    oheal *= rad2deg

    idisk = np.zeros((len(yngNames)),dtype=float)
    odisk = np.zeros((len(yngNames)),dtype=float)
    # Loop through each star and find the peak (i, O) from its PDF
    for ss in range(len(yngNames)):
        # Next get orbital info
        name = yngNames[ss]
        orbFile = root + alnDir + '/aorb_thesis/' + name + '_mc_heal.dat'
        #orbFile = root + alnDir + '/aorb_acc_mrPDF_MC_newMosaic/' + name + '_mc_heal.dat'

        pdf = np.fromfile(orbFile, dtype=float)

        # Find the peak of the PDF
        sid = (pdf.argsort())[::-1]  # reverse sort
        peakPix = sid[0]
        idisk[ss] = iheal[peakPix]
        odisk[ss] = oheal[peakPix]
        
    py.figure(1)
    py.figure(figsize=(8,8))
    py.clf()
    py.subplots_adjust(left=0.1, right=0.98, top=0.95, bottom=0.1)
    py.clf()
    py.imshow(np.log10(img+1), aspect='equal', interpolation='bicubic',
              extent=[max(xL), min(xL), min(yL), max(yL)],vmin=2.2,vmax=5,
              origin='lowerleft', cmap=py.cm.gray_r)
    py.plot([0],[0],'rx', ms=7, mew=1.5)
    py.plot(x,y,'kx',ms=5)
    for ss in range(len(yngNames)):
        if np.abs(idisk[ss] - 90.) < 10.:
            py.plot(x[ss],y[ss],'rx',ms=5)
            py.text(x[ss],y[ss]+0.2,('%5.1f' % idisk[ss]),fontsize=10,color='r')
        else:
            py.plot(x[ss],y[ss],'bx',ms=5)
            py.text(x[ss],y[ss]+0.2,('%5.1f' % idisk[ss]),fontsize=10,color='b')
    an = np.linspace(0,2*np.pi,100)
    py.plot(3.5*np.cos(an),3.5*np.sin(an),'k--')
    py.plot(7.*np.cos(an),7.*np.sin(an),'k--')
    py.plot([0,0],[-15,15],'k--')
    py.plot([15,-15],[0,0],'k--')
    py.xlabel('X (arcsec)',fontsize=16)
    py.ylabel('Y (arcsec)',fontsize=16)
    py.title('Inclination (deg)')
    py.savefig(plotdir + 'plot_incl_on_mosaic.png')
    py.close(1)
    print
    print 'Total number of young stars: %i' % len(x)
    print

    py.figure(2)
    py.figure(figsize=(8,8))
    py.clf()
    py.subplots_adjust(left=0.1, right=0.98, top=0.95, bottom=0.1)
    py.clf()
    py.imshow(np.log10(img+1), aspect='equal', interpolation='bicubic',
              extent=[max(xL), min(xL), min(yL), max(yL)],vmin=2.2,vmax=5,
              origin='lowerleft', cmap=py.cm.gray_r)
    py.plot([0],[0],'rx', ms=7, mew=1.5)
    for ss in range(len(yngNames)):
        if np.abs(odisk[ss] - 180.) < 10.:
            py.plot(x[ss],y[ss],'rx',ms=5)
            py.text(x[ss],y[ss]+0.2,('%5.1f' % odisk[ss]),fontsize=10,color='r')
            print yngNames[ss]
        else:
            py.plot(x[ss],y[ss],'bx',ms=5)
            py.text(x[ss],y[ss]+0.2,('%5.1f' % odisk[ss]),fontsize=10,color='b')
    an = np.linspace(0,2*np.pi,100)
    py.plot(3.5*np.cos(an),3.5*np.sin(an),'k--')
    py.plot(7.*np.cos(an),7.*np.sin(an),'k--')
    py.plot([0,0],[-15,15],'k--')
    py.plot([15,-15],[0,0],'k--')
    py.xlabel('X (arcsec)',fontsize=16)
    py.ylabel('Y (arcsec)',fontsize=16)
    py.title('Omega (deg)')
    py.savefig(plotdir + 'plot_Omega_on_mosaic.png')
    py.close(2)

    py.figure(3)
    py.figure(figsize=(8,8))
    py.clf()
    py.subplots_adjust(left=0.1, right=0.98, top=0.95, bottom=0.1)
    py.clf()
    py.imshow(np.log10(img+1), aspect='equal', interpolation='bicubic',
              extent=[max(xL), min(xL), min(yL), max(yL)],vmin=2.2,vmax=5,
              origin='lowerleft', cmap=py.cm.gray_r)
    py.plot([0],[0],'rx', ms=7, mew=1.5)
    for ss in range(len(yngNames)):
        if np.abs(vz_sig[ss]) < 1.0:
            py.plot(x[ss],y[ss],'rx',ms=5)
            py.text(x[ss],y[ss]+0.2,('%6.1f+-%6.1f' % (vz[ss],vzerr[ss])),fontsize=10,color='r')
        else:
            py.plot(x[ss],y[ss],'bx',ms=5)
            py.text(x[ss],y[ss]+0.2,('%6.1f+-%6.1f' % (vz[ss],vzerr[ss])),fontsize=10,color='b')
    py.plot([0,0],[-15,15],'k--')
    py.plot([15,-15],[0,0],'k--')
    py.xlabel('X (arcsec)',fontsize=16)
    py.ylabel('Y (arcsec)',fontsize=16)
    py.title('Radial Velocity (km/s)')
    py.savefig(plotdir + 'plot_rv_on_mosaic.png')
    py.close(3)

    usetexTrue()
    py.figure(4)
    py.clf()
    py.figure(figsize=(10,10))
    py.subplots_adjust(left=0.1, right=0.98, top=0.95, bottom=0.1,
                       wspace=0.3, hspace=0.3)
    x_km = x * dist * cc.cm_in_au / 1.e5
    xe_km = xerr * dist * cc.cm_in_au / 1.e5
    x_vz = x_km * vz
    idx = np.where(np.abs(odisk - 180.) < 10.)[0]
    py.subplot(2,2,1)
    py.errorbar(r2d, vz, yerr=vzerr, fmt='b.')
    py.errorbar(r2d[idx], vz[idx], yerr=vzerr[idx], fmt='r.')
    py.plot([0,14],[0,0],'k--')
    py.text(6,450,r'{$\Omega=180\pm10 deg$}',color='r',fontsize=12)
    py.xlabel('Projected Radius (arcsec)')
    py.ylabel('Vz (km/s)')
    
    py.subplot(2,2,2)
    py.errorbar(x_km, vz, yerr=vzerr, fmt='b.')
    py.errorbar(x_km[idx], vz[idx], yerr=vzerr[idx], fmt='r.')
    py.plot([-1.5e13,1.5e13],[0,0],'k--')
    py.plot([0,0],[-800,600],'k--')
    py.xlabel('X (km)')
    py.ylabel('Vz (km/s)')
    
    py.subplot(2,2,3)
    xvz_err = np.zeros((len(yngNames)),dtype=float)
    for ss in range(len(yngNames)):
        xvz_err[ss] = np.sqrt(vz[ss]**2*xe_km[ss]**2 + x_km[ss]**2*vzerr[ss]**2)
    py.errorbar(r2d, x_vz, yerr=xvz_err, fmt='b.')
    py.errorbar(r2d[idx], x_vz[idx], yerr=xvz_err[idx], fmt='r.')
    py.plot([0,14],[0,0],'k--')
    py.xlabel('Projected Radius (arcsec)')
    py.ylabel('X * Vz')

    py.subplot(2,2,4)
    py.errorbar(r2d, vx, yerr=vxerr, fmt='b.')
    py.errorbar(r2d[idx], vx[idx], yerr=vxerr[idx], fmt='r.')
    py.plot([0,14],[0,0],'k--')
    py.xlabel('Projected Radius (arcsec)')
    py.ylabel('Vx (km/s)')
    py.savefig(plotdir + 'x_vz_r2d.png')
    py.close(4)
    usetexFalse()

    py.clf()
    py.figure(5)
    py.figure(figsize=(6,6))
    py.plot(odisk, x_vz, 'k.')
    py.plot([180,180],[-3e15,2e15],'k--')
    py.xlabel('Omega (deg)')
    py.ylabel('X * Vz')
    py.savefig(plotdir + 'x_vz_Omega.png')
    py.close(5)

    py.figure(6)
    py.figure(figsize=(8,8))
    py.clf()
    py.subplots_adjust(left=0.1, right=0.98, top=0.95, bottom=0.1)
    py.clf()
    py.imshow(np.log10(img+1), aspect='equal', interpolation='bicubic',
              extent=[max(xL), min(xL), min(yL), max(yL)],vmin=2.2,vmax=5,
              origin='lowerleft', cmap=py.cm.gray_r)
    py.plot([0],[0],'rx', ms=7, mew=1.5)
    for ss in range(len(yngNames)):
        if np.abs(odisk[ss] - 180.) < 10.:
            py.plot(x[ss],y[ss],'rx',ms=5)
            py.text(x[ss],y[ss]+0.2,('%6.1f+-%6.1f' % (vx[ss],vxerr[ss])),fontsize=10,color='r')
    #    else:
    #        py.plot(x[ss],y[ss],'bx',ms=5)
    #        py.text(x[ss],y[ss]+0.2,('%6.1f+-%6.1f' % (vx[ss],vxerr[ss])),fontsize=10,color='b')
    an = np.linspace(0,2*np.pi,100)
    py.plot(7.*np.cos(an),7*np.sin(an),'k--')
    py.plot([0,0],[-15,15],'k--')
    py.plot([15,-15],[0,0],'k--')
    py.xlabel('X (arcsec)',fontsize=16)
    py.ylabel('Y (arcsec)',fontsize=16)
    py.title('X Velocity (km/s)')
    py.savefig(plotdir + 'plot_vx_on_mosaic.png')
    py.close(6)

    py.figure(7)
    py.figure(figsize=(8,8))
    py.clf()
    py.subplots_adjust(left=0.1, right=0.98, top=0.95, bottom=0.1)
    py.clf()
    py.imshow(np.log10(img+1), aspect='equal', interpolation='bicubic',
              extent=[max(xL), min(xL), min(yL), max(yL)],vmin=2.2,vmax=5,
              origin='lowerleft', cmap=py.cm.gray_r)
    py.plot([0],[0],'rx', ms=7, mew=1.5)
    for ss in range(len(yngNames)):
        if (odisk[ss] > 160.) & (odisk[ss] < 200.) & (idisk[ss] > 100.) & (idisk[ss] < 140.):
            py.plot(x[ss],y[ss],'rx',ms=5)
            py.text(x[ss],y[ss]+0.2,('%5s' % yngNames[ss]),fontsize=10,color='r')
            #py.quiver([x[ss]], [y[ss]], [vx[ss]], [vy[ss]], headwidth=2,
            #  color='black', units='y', angles='xy')#, scale=200)
    an = np.linspace(0,2*np.pi,100)
    py.plot(6.473*np.cos(an),6.473*np.sin(an),'k--')
    py.plot([0,0],[-15,15],'k--')
    py.plot([15,-15],[0,0],'k--')
    py.xlabel('X (arcsec)',fontsize=16)
    py.ylabel('Y (arcsec)',fontsize=16)
    py.title('Stars with Omega ~ 180 deg')
    py.savefig(plotdir + 'plot_names_Om180_on_mosaic.png')
    py.close(7)

    py.figure(8)
    py.figure(figsize=(10,5))
    py.subplots_adjust(left=0.1, right=0.98, top=0.95, bottom=0.1,
                       wspace=0.3, hspace=0.3)
    py.clf()
    py.subplot(1,2,1)
    py.errorbar(np.abs(vz), np.abs(vx), yerr=vxerr, fmt='b.')
    py.errorbar(np.abs(vz[idx]), np.abs(vx[idx]), yerr=vxerr[idx], fmt='r.')
    py.xlabel('Radial Velocity (km/s)')
    py.ylabel('X Velocity (km/s)')
    py.subplot(1,2,2)
    py.errorbar(np.abs(vz), np.abs(vy), yerr=vyerr, fmt='b.')
    py.errorbar(np.abs(vz[idx]), np.abs(vy[idx]), yerr=vyerr[idx], fmt='r.')
    py.xlabel('Radial Velocity (km/s)')
    py.ylabel('Y Velocity (km/s)')
    py.savefig(plotdir + 'vx_vy_vz.png')
    py.close(8)

    py.figure(9)
    py.figure(figsize=(12,5))
    py.subplots_adjust(left=0.1, right=0.98, top=0.95, bottom=0.1,
                       wspace=0.3, hspace=0.3)
    py.clf()
    py.subplot(1,3,1)
    py.plot(r2d, (np.abs(vz/vx)), 'k.')
    py.plot(r2d[idx], (np.abs(vz[idx]/vx[idx])), 'rx')
    py.axis([0,14,0,20])
    py.xlabel('Projected Radius (arcsec)')
    py.ylabel('Velocity Ratio (Vz/Vx)')
    py.subplot(1,3,2)
    py.plot(r2d, (np.abs(vz/vy)), 'k.')
    py.plot(r2d[idx], (np.abs(vz[idx]/vy[idx])), 'rx')
    py.axis([0,14,0,20])
    py.xlabel('Projected Radius (arcsec)')
    py.ylabel('Velocity Ratio (Vz/Vy)')
    py.subplot(1,3,3)
    py.plot(r2d, (np.abs(vx/vy)), 'k.')
    py.plot(r2d[idx], (np.abs(vx[idx]/vy[idx])), 'rx')
    py.axis([0,14,0,20])
    py.xlabel('Projected Radius (arcsec)')
    py.ylabel('Velocity Ratio (Vx/Vy)')
    py.savefig(plotdir + 'vel_ratios_vs_r2d.png')
    py.close(9)
               
   


def check_unbound(mosaic=False):
    """
    Plots accelerations and velocities and compares to what is
    expected if stars are bound.
    """

    # Load up young stars
    # young.loadYoungStars calls youngStarNames but grabs
    # only young stars beyond r = 0.8 arcsec
    # Load names of young stars 
    if mosaic == True:
        # Load up mosaic data as well; select only stars at r>4, since
        # we don't want to add any info from mosaics if we have it in
        # the central 10" already
        yng1 = young.loadYoungStars(root+alnDir,fit=poly,points=points,radiusCut=0.8,
                                    skipStar=['S5-237'])
        yng2 = young.loadYoungStars(mscDir,fit=polyM,points=pointsM,
                                    mosaic=True, withRVonly=True,silent=True)
        cntrlNames = yng1.getArray('name')
        mscNames = yng2.getArray('name')
        # Merge this object with object from central 10" analysis
        yng = merge(yng1, yng2)
    else:
        yng = young.loadYoungStars(root+alnDir,fit=poly,points=points,radiusCut=0.8) 

    rv_ref = np.array(yng.getArray('rv_ref')) # just for knowing how many have RVs

    cc = objects.Constants()
    GM = cc.G * mass * cc.msun

    # Construct an array of radii out to 7 arcsec in steps of 0.1''
    #r = (np.arange(10 * 7) * 0.1) + 0.1
    r = (np.arange(10 * 14) * 0.1) + 0.1
    r_au = r * dist
    r_pc = r_au / cc.au_in_pc
    r_cm = r_au * cc.cm_in_au

    # Determine the theoretical amount for a vs. r (set by r2d)
    a_cm_s2 = -GM / r_cm**2
    a_km_s_yr = a_cm_s2 * cc.sec_in_yr / 1.0e5

    # Also determine theoretical escape velocity curve
    v_esc_cm_s = np.sqrt(2. * GM / r_cm)
    v_esc_km_s = v_esc_cm_s / 1.0e5

    # Determine minimum |acceleration| assuming a bound orbit
    a_min_cm_s2 = -(r_cm * v_esc_cm_s**6) / (8.0 * GM**2)
    a_min_km_s_yr = a_min_cm_s2 * cc.sec_in_yr / 1.0e5 

    # Error on the computed theoretical a_max curve
    sig_a_cgs = cc.G / r_cm**2 * massErr * cc.msun
    sig_a_mks = sig_a_cgs * cc.sec_in_yr / 1.0e5

    # Error on the computed theoretical v_esc curve
    sig_vesc_cgs = massErr * cc.msun * np.sqrt(cc.G / (2.*mass*cc.msun*r_cm))
    sig_vesc_mks = sig_vesc_cgs / 1.0e5

    # Loop through all the stars.
    names = yng.getArray('name')

    allAccLimRad = np.zeros(len(names))
    
    ii = 0

    rLower_all = []
    raLower_all = []
    ar_all = []
    at_all = []
    are_all = []
    ate_all = []
    cnt_all = []
    xe_all = []
    ye_all = []
    vxe_all = []
    vye_all = []
    vze_all = []
    vtot_all = []
    vtoterr_all = []
    have_vz = []
    vesc_ratio_all = []
    vesc_atR_all = []
    vescA_ratio_all = []
    acc_ratio_all = []
    amin_all = []
    for name in names:
	i = names.index(name)

	star = yng.stars[i]
        mag = yng.stars[i].mag
        mscStar = False

        if (name in mscNames) & (name not in cntrlNames):
            mscStar = True
            x = star.fitXv.p
            y = star.fitYv.p
            r2d = np.hypot(x, y)
            xe = star.fitXv.perr
            ye = star.fitYv.perr
            vx = star.fitXv.v
            vy = star.fitYv.v
            vxe = star.fitXv.verr
            vye = star.fitYv.verr
            xchi2 = star.fitXv.chi2
            ychi2 = star.fitYv.chi2
        else:
            x = star.fitXa.p
            y = star.fitYa.p
            r2d = np.hypot(x, y)
            xe = star.fitXa.perr
            ye = star.fitYa.perr
            vx = star.fitXa.v
            vy = star.fitYa.v
            vxe = star.fitXa.verr
            vye = star.fitYa.verr
            ax = star.fitXa.a # arcsec/yr^2
            ay = star.fitYa.a # arcsec/yr^2
            axe = star.fitXa.aerr # arcsec/yr^2
            aye = star.fitYa.aerr # arcsec/yr^2
            xchi2 = star.fitXa.chi2
            ychi2 = star.fitYa.chi2

        vz = star.vz # km/s
        vze = star.vzerr # km/s

        #x = star.fitXa.p
        #y = star.fitYa.p
        #xe = star.fitXa.perr
        #ye = star.fitYa.perr
        #vx = star.fitXa.v # km/s
        #vy = star.fitYa.v # km/s
        #vxe = star.fitXa.verr # km/s
        #vye = star.fitYa.verr # km/s
        #t0 = star.fitXa.t0

        if vz == None:
            vz = 0.0
            vze = 0.0
            print 'No radial velocity for %s; assuming vz = 0.0!' % name
            has_vz = 0
        else:
            has_vz = 1
        vtot = np.sqrt(vx**2 + vy**2 + vz**2) # km/s
        vtot_err = (np.sqrt(((vx*vxe)**2 + (vy*vye)**2 + (vz*vze)**2) / vtot**2)) # km/s

        vtot_all = np.concatenate([vtot_all, [vtot]])
        vtoterr_all = np.concatenate([vtoterr_all, [vtot_err]])
        # Keep track of which stars have vz measurements and which do not
        have_vz = np.concatenate([have_vz, [has_vz]])

        # How many epochs was each star detected in?
        if mscStar == False:
            pts = asciidata.open('%s%s%s%s.points' % (root, alnDir, points, name))
        else:
            pts = asciidata.open('%s%s%s.points' % (mscDir, pointsM, name))
        ep = pts[0].tonumpy()
        cnt = len(ep)
        cnt_all = np.concatenate([cnt_all, [cnt]]) 

        r = np.sqrt(x**2 + y**2)
        rcgs = r * dist * cc.cm_in_pc / cc.au_in_pc
        
	# Lower allowed radius is set by 2D distance.
	rLower = r * dist / cc.au_in_pc # pc

        cc.asy_to_kms = dist * cc.cm_in_au / (1.e5 * cc.sec_in_yr)

        # Get the ratio of velocity to escape velocity at this r2d
        v_esc_at_r = np.sqrt(2. * GM / rcgs)
        v_esc_at_r /= 1.e5
        vesc_ratio = vtot / v_esc_at_r

        if mscStar == False:
    	    ax *= cc.asy_to_kms
	    ay *= cc.asy_to_kms
	    axe *= cc.asy_to_kms
	    aye *= cc.asy_to_kms
	
	    # What is the polyfit acceleration radial component, in km/s/yr
            (ar, at, are, ate) = util.xy2circErr(x, y, ax, ay,
                                                 xe, ye, axe, aye)

            # Get the ratio of accelerations to max accel at this r2d
            _ar_at_r = -GM / rcgs**2 # cm/s/s
            ar_at_r = _ar_at_r * cc.sec_in_yr / 1.0e5
            acc_ratio = ar / ar_at_r

            # Assuming a bound orbit, there is a minimum |acceleration| at this r2d
            _a_min_at_r = -(rcgs * vtot**6) / (8.0 * GM**2)
            a_min_at_r = _a_min_at_r * cc.sec_in_yr / 1.0e5

            ar_all = np.concatenate([ar_all, [ar]]) # km/s/yr
            at_all = np.concatenate([at_all, [at]]) # km/s/yr
            are_all = np.concatenate([are_all, [are]]) # km/s/yr
            ate_all = np.concatenate([ate_all, [ate]]) # km/s/yr
            acc_ratio_all = np.concatenate([acc_ratio_all, [acc_ratio]]) # km/s/yr
            amin_all = np.concatenate([amin_all, [a_min_at_r]]) # km/s/yr
            raLower_all = np.concatenate([raLower_all, [rLower]]) # pc (for accel stars)
            vescA_ratio_all = np.concatenate([vescA_ratio_all, [vesc_ratio]]) # km/s (for accel stars)
    
        # Save off some info
        vesc_ratio_all = np.concatenate([vesc_ratio_all, [vesc_ratio]]) # km/s
        vesc_atR_all = np.concatenate([vesc_atR_all, [v_esc_at_r]]) # km/s
        rLower_all = np.concatenate([rLower_all, [rLower]]) # pc
        xe_all = np.concatenate([xe_all, [xe*1.e3]]) # mas
        ye_all = np.concatenate([ye_all, [ye*1.e3]]) # mas 
        vxe_all = np.concatenate([vxe_all, [vxe]]) # km/s
        vye_all = np.concatenate([vye_all, [vye]]) # km/s
        vze_all = np.concatenate([vze_all, [vze]]) # km/s

    # Identify S0-15
    s015 = names.index('S0-15')
    # possibly unbound stars
    #unb = ['S0-15', 'S2-21', 'S3-2', 'S3-10']
    #unb = ['S0-15', 'S3-2', 'S2-21', 'irs16NE', 'irs9W'] # these are close-to-unbound?? Having trouble finding solutions for these when using extended mass distribution
    unb = ['']
    #pa180 = ['S6-100', 'S7-16', 'S8-4', 'S10-4', 'S10-5', 'S11-5']
    pa180 = ['S6-100','S7-16','S8-15','S9-23','S10-4','S10-5']

    print
    hdr = '%8s  %5s pc  %8s +- %8s km/s  %8s'
    fmt = '%8s  %5.3f pc  %8.2f +- %8.2f km/s  %8.2f'
    print hdr % ('Star','R2D','Vtot','Vtot Err','Vesc at R2D')
    usetexTrue()
    py.clf()
    py.figure(1)
    py.figure(figsize=(12,5))
    py.subplots_adjust(left=0.08, right=0.98, top=0.9, bottom=0.1,
                       wspace=0.3, hspace=0.3)
    py.subplot(1,3,1)
    # plot theoretical escape velocity curve
    py.plot(r_pc, v_esc_km_s, 'k--')
    py.plot(r_pc, (v_esc_km_s+3.*sig_vesc_mks), 'k:')
    py.plot(r_pc, (v_esc_km_s-3.*sig_vesc_mks), 'k:')
    py.plot(r_pc, (v_esc_km_s-2.*sig_vesc_mks), 'r:')
    py.plot(r_pc, (v_esc_km_s-1.*sig_vesc_mks), 'b:')
    # Plot velocity measurements and 1 sigma errors
    for ii in range(len(vtot_all)):
        if have_vz[ii] == 1: # This star has an RV
            py.errorbar(rLower_all[ii], vtot_all[ii], yerr=vtoterr_all[ii], fmt='k.')
        else: # no RV for this star
            py.errorbar(rLower_all[ii], vtot_all[ii], yerr=vtoterr_all[ii], fmt='g.')
        if names[ii] in pa180:
            py.plot(rLower_all[ii], vtot_all[ii], 'rx', ms=8)
            print fmt % (names[ii],rLower_all[ii], vtot_all[ii], vtoterr_all[ii],
                         vesc_atR_all[ii])
    py.plot([0.17],[20],'g.')
    py.text(0.175,10,'No RV',color='g',fontsize=12)
    py.text(0.1, 700, r'{\bf $v_{esc} = \sqrt{\frac{2GM}{\rho}} (+3\sigma$)',color='g')
    py.axis([0,0.6,0,1000])
    py.xlabel(r'{\bf Projected Radius (pc)}')
    py.ylabel(r'{\bf v$_{tot}$ (km/s)}')

    py.subplot(1,3,2)
    # plot theoretical max acceleration curve
    py.plot(r_pc, -a_km_s_yr, 'k--')
    py.plot(r_pc, -(a_km_s_yr+3.*sig_a_mks), 'k:')
    py.plot(r_pc, -(a_km_s_yr-3.*sig_a_mks), 'k:')
    py.text(0.033, 25, r'{\bf $|$a$|_{max} = \frac{GM}{\rho^2}$} ($+3\sigma$)',color='g')
    # plot theoretical min acceleration curve
    #py.plot(r_pc, -a_min_km_s_yr, 'k--') # no constraints from this info
    #py.plot(raLower_all, -amin_all, 'r.') # no constraints from this info
    # Plot acceleration measurements and 1 sigma errors
    py.errorbar(raLower_all, -ar_all, yerr=are_all, fmt='k.')
    py.plot([0,0.25],[0,0],'k--') # horizontal line at 0 
    for ii in range(len(ar_all)):
        if names[ii] in pa180:
            py.plot(raLower_all[ii], -ar_all[ii], 'rx', ms=8)
    py.axis([0,0.3,-10,30])
    py.xlabel(r'{\bf Projected Radius (pc)}')
    py.ylabel(r'{\bf $|$a$_\rho|$ (km/s/yr)}')

    
    py.subplot(1,3,3)
    py.plot([0,2],[1,1],'k--')
    py.plot([1,1],[-5,50],'k--')
    for ii in range(len(ar_all)):
        if rLower_all[ii] > 0.15:
            continue
        py.plot(vescA_ratio_all[ii], acc_ratio_all[ii], 'k.')
        if names[ii] in pa180:
            py.plot(vescA_ratio_all[ii], acc_ratio_all[ii], 'rx', ms=8)
    #for ii in range(len(vesc_ratio_all)):
    #    py.text(vesc_ratio_all[ii]+0.05, acc_ratio_all[ii],names[ii],fontsize=10)
    #py.axis([0, 1.1, -5, 50])
    py.axis([0, 1.1, -15, 10])
    py.xlabel(r'{\bf $v_{tot}/v_{esc}$}')
    py.ylabel(r'{\bf $a_{\rho}/a_{max}$}')
    py.title('Young Stars within 0.15 pc')
    py.savefig(plotdir + 'check_unbound_cases.png')
    py.close(1)

    py.clf()
    py.figure(2)
    py.figure(figsize=(10,6))
    py.subplots_adjust(left=0.08, right=0.98, top=0.9, bottom=0.1,
                       wspace=0.3, hspace=0.3)
    py.subplot(1,2,1)
    binsIn = np.arange(0, 100, 1)
    py.hist(vxe_all, bins=binsIn, color='r',histtype='step', label='x')
    py.hist(vye_all, bins=binsIn, color='b',histtype='step', label='y')
    py.hist(vze_all, bins=binsIn, color='g',histtype='step', label='z')
    py.legend(numpoints=1, fancybox=True)
    py.xlabel('Velocity Error (km/s)')
    py.ylabel('N')
    py.subplot(1,2,2)
    py.hist(vtoterr_all, bins=binsIn, color='k',histtype='step')
    py.xlabel('Total Velocity Error (km/s)')
    py.ylabel('N')
    py.savefig(plotdir + 'hist_vel_error.png')
    usetexFalse()
    
    print ''
    print 'Average (Median) Velocity Error:'
    print '   x: %5.3f +- %5.3f (%5.3f) km/s' % \
          (vxe_all.mean(), vxe_all.std(ddof=1), np.median(vxe_all))
    print '   y: %5.3f +- %5.3f (%5.3f) km/s' % \
          (vye_all.mean(), vye_all.std(ddof=1), np.median(vye_all))
    nz = np.where(vze_all > 0.0)[0]
    print '   z: %5.3f +- %5.3f (%5.3f) km/s' % \
          (vze_all[nz].mean(), vze_all[nz].std(ddof=1), np.median(vze_all[nz]))
    print '   Total: %5.3f +- %5.3f (%5.3f) km/s' % \
          (vtoterr_all.mean(), vtoterr_all.std(ddof=1), np.median(vtoterr_all))


def klf_disk_on_off(pdfdir='aorb_thesis/', mosaic=True, suffix='_mosaic',
                    diskCut=3.0, top20pct=False, probNodes=False):
    """
    Plot the K-band luminosity function separately for
    disk candidates and non-candidates

    Set top20pct = True to plot only the top 20% of candidate disk members.
    Set probNodes = True to plot only the most likely (>3 sigma) disk members
    		    according to the probability vs. PA from line of nodes relation.
    """


    # Load disk star names and probability of disk membership
    (names, diskP) = readDiskProb(diskOnly=False, suffix=suffix, diskCut=diskCut)

    lhNotOnDisk_cut = scipy.special.erf(diskCut/np.sqrt(2.))
    probOnDisk_cut = 1.0 - lhNotOnDisk_cut

    # Load kinematic data for disk stars
    yng = loadYoungByName(names, mosaic=mosaic)

    mag = yng.getArray('mag')
    x = yng.getArray('x')
    y = yng.getArray('y')

    r = np.sqrt(x**2 + y**2)

    names = np.array(names)

    #----------
    #
    # For the full sample of young stars, compare the KLF of those
    # stars on the disk and those stars off the disk.
    #
    #----------
    print '** All Young Stars: On Disk vs. Off Disk **'
    onIdx = (np.where(diskP >= probOnDisk_cut))[0] # 3 sigma non-members
    offIdx = (np.where(diskP < probOnDisk_cut))[0]

    if top20pct == True:
        num = np.floor(len(diskP) * 0.2)
        sidx = diskP.argsort()[::-1][0:int(num)] # reverse sort
        oidx = diskP.argsort()[::-1][int(num):] # everything else
        diskPsrt = diskP[sidx]

    if probNodes == True:
        # Get the stars above 3 sigma curve in the probability vs.
        # PA from line of nodes plot
        nodes = asciidata.open(root + alnDir + 'tables/nodes_angle_prob_app.dat')
        nstar = nodes[0].tonumpy()
        nstar = np.array([ss.strip() for ss in nstar])

        sidx = np.zeros(len(nstar))
        for nn in range(len(nstar)):
            sidx[nn] = np.where(names == nstar[nn])[0]

        sidx = [np.int(ss) for ss in sidx]
        diskPsrt = diskP[sidx]

        oidx = np.setdiff1d(np.arange(len(names)), sidx)

    mstep = 0.5
    magBins = np.arange(8.5, 15.5+mstep, mstep)
    (bins1, data1) = histNofill.hist(magBins, mag[onIdx])
    (bins2, data2) = histNofill.hist(magBins, mag[offIdx])

    # KS test: null hypothesis is that the two distributions
    # are the same. If p value is less than 1%, can reject
    # null hypothesis that they are the same. If it's high,
    # cannot reject null hypothesis, and therefore same KLF.
    foo = stats.stats.ks_2samp(mag[onIdx], mag[offIdx])
    print
    print 'Old Disk Membership Criteria (L > 3 sigma):'
    print 'On vs. Off disk:'
    print 'KS Test: D = %5.2f  P = %5.3f' % foo

    # Also pull the spectral type from Paumard+06 table for those
    # that have them.
    dbfile = '/u/ghezgroup/data/gc/database/stars.sqlite'
    # Create a connection to the database file
    connection = sqlite.connect(dbfile)
    # Create a cursor object
    cur = connection.cursor()
    cur.execute('SELECT * FROM paumard2006')
    mtch = []
    spT = []
    qual = []
    for yy in range(len(names)):
        # Identify this yng star in the Paumard table
        yngName = str(names[yy])
        cur.execute('SELECT ucla,name,type,quality FROM paumard2006 WHERE ucla=?', [yngName])
        for row in cur:
            #print 'Found %s' % yngName
            mtch = np.concatenate([mtch, [row[0]]])
            spT = np.concatenate([spT, [row[2]]])
            qual = np.concatenate([qual, [row[3]]])

    usetexTrue()
    import matplotlib.font_manager
    prop = matplotlib.font_manager.FontProperties(size=12)
    py.clf()
    py.figure(figsize=(6,6))
    on = py.plot(bins1, data1, 'k-', linewidth=2)
    off = py.plot(bins2, data2, 'k--', linewidth=2)

    py.legend((on, off), ('On Disk' , 'Off Disk'), numpoints=1,loc=2,prop=prop)
    py.xlabel('K Magnitude')
    py.ylabel('Number of Stars')
    py.savefig(plotdir + 'klf_disk_on_off%s_%isigMmbr.png' % (suffix,int(diskCut)))
    py.close()

    if top20pct == True:
        py.clf()
        py.figure(figsize=(6,6))
        py.hist(mag[sidx], bins=np.arange(9,16,1), ls='solid', lw=2,
                color='r', histtype='step', label=r'Top 20\% Disk')
        nn,bb,pp = py.hist(mag[oidx], bins=np.arange(9,16,1), ls='dashed', lw=2,
                   color='k', histtype='step', label='Non-Disk')
        py.xlabel('Observed K Magnitude')
        py.ylabel('Number of Stars')
        py.legend(numpoints=1,prop=prop,loc=2,fancybox=True)
        py.axis([8,16,0,nn.max()+2])
        py.savefig(plotdir + 'klf_top_20percent_disk.png')
        py.savefig(plotdir + 'eps/klf_top_20percent_disk.eps')
        py.close()

        # KS test: null hypothesis is that the two distributions
        # are the same. If p value is less than 1%, can reject
        # null hypothesis that they are the same. If it's high,
        # cannot reject null hypothesis, and therefore same KLF.
        foo = stats.stats.ks_2samp(mag[sidx], mag[oidx])
        print
        print 'Top 20% on disk vs. off disk:'
        print 'KS Test: D = %5.2f  P = %5.3f' % foo
        print
        print 'Disk membership probability range for top 20 pct = %6.3f - %6.3f' % \
              (diskPsrt.min(),diskPsrt.max())

    if probNodes == True:
        py.clf()
        py.figure(figsize=(6,6))
        py.hist(mag[sidx], bins=np.arange(9,16,1), ls='solid', lw=2,
                color='r', histtype='step', label=r'Disk Candidates')
        nn,bb,pp = py.hist(mag[oidx], bins=np.arange(9,16,1), ls='dashed', lw=2,
                   color='k', histtype='step', label='Non-Disk Candidates')
        py.xlabel('Observed K Magnitude')
        py.ylabel('Number of Stars')
        py.legend(numpoints=1,prop=prop,loc=2,fancybox=True)
        py.axis([8,16,0,nn.max()+2])
        py.savefig(plotdir + 'klf_top_members_nodes_disk.png')
        py.savefig(plotdir + 'eps/klf_top_members_nodes_disk.eps')
        py.close()

        py.clf()
        py.figure(figsize=(6,6))
        py.hist(mag[sidx], bins=np.arange(9,16,1), ls='solid', lw=2,
                color='r', histtype='step', normed=True, label=r'Disk Candidates')
        nn,bb,pp = py.hist(mag[oidx], bins=np.arange(9,16,1), ls='dashed', lw=2,
                   color='k', histtype='step',normed=True, label='Non-Disk Candidates')
        py.xlabel('Observed K Magnitude')
        py.ylabel('PDF')
        py.legend(numpoints=1,prop=prop,loc=2,fancybox=True)
        thePlot = py.gca()
        thePlot.get_yaxis().set_major_locator(py.MultipleLocator(0.1))
        py.axis([8,16,0,0.4])
        py.savefig(plotdir + 'klf_top_members_nodes_disk_normed.png')
        py.savefig(plotdir + 'eps/klf_top_members_nodes_disk_normed.eps')
        py.close()

        # KS test: null hypothesis is that the two distributions
        # are the same. If p value is less than 1%, can reject
        # null hypothesis that they are the same. If it's high,
        # cannot reject null hypothesis, and therefore same KLF.
        foo = stats.stats.ks_2samp(mag[sidx], mag[oidx])
        print
        print 'Top members based on PA from line of nodes: disk (N=%i) vs. off disk (N=%i):' %\
              (len(sidx), len(oidx))
        print 'KS Test: D = %5.2f  P = %5.3f' % foo
        print
        print 'Disk membership probability range = %6.3f - %6.3f' % \
              (diskPsrt.min(),diskPsrt.max())




    py.clf()
    py.figure(figsize=(6,6))
    py.subplots_adjust(left=0.15, right=0.96, top=0.95, bottom=0.1,
                       wspace=0.3, hspace=0.3)
    py.plot()
    py.semilogy(mag, diskP, 'k.')
    py.semilogy([9,16], [2.7e-3, 2.7e-3],'k--')
    py.semilogy([9,16], [4.55e-2, 4.55e-2],'k--')
    py.semilogy([9,16], [0.3173, 0.3173],'k--')
    #py.text(9.5,3.5e-3,'Candidate Disk Members',fontsize=12)
    py.text(9.3,3.45e-3,r'${\bf 3\sigma}$')
    py.text(9.3,5.0e-2,r'${\bf 2\sigma}$')
    py.text(9.3,0.5,r'${\bf 1\sigma}$')
    py.axis([9,16,0.98e-5,1.1])
    py.xlabel('K Magnitude')
    py.ylabel('Disk Membership Probability')
    py.savefig(plotdir + 'membership_vs_mag%s.png' % suffix)
    py.savefig(plotdir + 'eps/membership_vs_mag%s.eps' % suffix)
    py.close()

    py.clf()
    py.figure(figsize=(6,6))
    py.plot()
    py.semilogy(r, diskP, 'k.')
    if mosaic == True:
        py.semilogy([0,14], [2.7e-3,2.7e-3],'k--')
    else:
        py.semilogy([0,7], [2.7e-3,2.7e-3],'k--')
    py.text(0.5,3.5e-3,'Candidate Disk Members',fontsize=12)
    py.xlabel('Projected Radius (arcsec)')
    py.ylabel('Disk Membership Probability')
    py.savefig(plotdir + 'membership_vs_r2d%s.png' % suffix)
    py.close()
    nr1 = np.where((r <= 3.2) & (diskP > 2.7e-3))[0]
    nr2 = np.where((r > 3.2) & (r <= 6.473) & (diskP > 2.7e-3))[0]
    nr3 = np.where((r > 6.473) & (diskP > 2.7e-3))[0]
    print
    print 'Number of disk stars per radial bin:'
    print ' 0.8 - 3.2 arcsec: %i' % len(nr1)
    print ' 3.2 - 6.5 arcsec: %i' % len(nr2)
    print '     > 6.5 arcsec: %i' % len(nr3)

    py.clf()
    py.figure(figsize=(6,6))
    py.plot()
    py.plot(r, mag, 'k.')
    py.xlabel('Projected Radius (arcsec)')
    py.ylabel('K Magnitude')
    py.savefig(plotdir + 'mag_vs_r2d%s.png' % suffix)
    py.close()

    usetexTrue()
   
    

def diskMembersCCW(mosaic=True):
    """
    Determine which stars are members of the CCW disk.

    Output:
    tables/diskCCW_membership_prob.dat -- text table containing results
    plots/diskCCW_membership_hist.png -- Histogram of disk membership
       probabilities.
    """
    # Load names of young stars 
    if mosaic == True:
        # Load up mosaic data as well; select only stars at r>4, since
        # we don't want to add any info from mosaics if we have it in
        # the central 10" already
        yng1 = young.loadYoungStars(root+alnDir,fit=poly,points=points,
                                    withRVonly=True,silent=True,skipStar=['S5-237']) 
        yng2 = young.loadYoungStars(mscDir,fit=polyM,points=pointsM,
                                    mosaic=True, withRVonly=True,silent=True)
        # Merge this object with object from central 10" analysis
        yng = merge(yng1, yng2)
    else:
        yng = young.loadYoungStars(root+alnDir,fit=poly,points=points,
                                   silent=True,withRVonly=True) 

    yngNames = yng.getArray('name')

    orbDir = 'aorb_thesis/'
    #orbDir = 'aorb_acc_mrPDF_MC_newMosaic/'
    #orbDir = 'aorb_acc/'

    # Disk solution
    nside = 64
    npix = healpy.nside2npix(nside)
    pixIdx = np.arange(0, npix)

    (disk, diskStd) = loadDiskDensity(npix, orbDir=orbDir,
                                      file1='disk.neighbor.dat',
                                      file2='disk.neighborStd.dat')

    (i, o) = healpy.pix2ang(nside, pixIdx)
    i *= 180.0 / math.pi
    o *= 180.0 / math.pi

    # CW disk properties
    didx = disk.argmax()
    peak = disk[didx]
    idisk = i[didx]
    odisk = o[didx]
    cwid = (np.where(disk > (0.5 * peak)))[0]
    print 'Peak at %.3e  stars/deg^2' % (peak)

    # CCW disk properties (from Paumard et al. 2006)
    #ipeak = 24.0
    #opeak = 167.0
    # CCW disk properties (from Bartko et al. 2009)
    phi = 200.0
    theta = 142.0
    ipeak = 180.0 - theta
    opeak = 360.0 - phi
    #pdb.set_trace()

    # temp
    #ipeak = 65.
    #opeak = 260.
    # end temp

    sini = np.sin(np.radians(ipeak))
    cosi = np.cos(np.radians(ipeak))
    sino = np.sin(np.radians(opeak))
    coso = np.cos(np.radians(opeak))

    angle = np.zeros(len(i), float)
    for k in range(len(i)):
        inc = np.radians(i[k])
        ome = np.radians(o[k])

        angle[k] = np.arccos(sini * coso * np.sin(inc) * np.cos(ome) + \
                          sini * sino * np.sin(inc) * np.sin(ome) + \
                          cosi * np.cos(inc))
        angle[k] = np.degrees(angle[k])

    # Find the points in the sky that are within 10 deg from the
    # proposed CCW disk
    ccwid = (np.where(angle < 10.0))[0]

    # Determine the probability for each star to be in this pixel
    cwProb = np.zeros(len(yngNames), float)
    ccwProb = np.zeros(len(yngNames), float)
    solidAngle = np.zeros(len(yngNames), float)

    _out = open('tables/diskCCW_membership_prob.dat', 'w')

    # Determine the total solid angle for the "in the disk" region
    areaPerPixel = areaOnSky / npix  # deg^2
    degsq2str = (math.pi / 180.0)**2

    # Now determine the integrated probability of falling on the disk.
    for ss in range(len(yngNames)):
        name = yngNames[ss]
        orbFile = root + alnDir + orbDir + name + '_mc_heal.dat'

        pdf = np.fromfile(orbFile, dtype=float)

        # Determine the 68.4% confidence region solid angle
        sid = (pdf.argsort())[::-1]  # reverse sort
        pdfSort = pdf[sid]
        
        # Make a cumulative distribution function starting from the
        # highest pixel value. This way we can find the level above
        # which 68% of the trials will fall.
        cdf = np.cumsum(pdfSort)
        
        # Determine point at which we reach 68% level
        idx = (np.where(cdf > 0.6827))[0]
        level = pdfSort[idx[0]]
        solidAngle[ss] = (idx[0] + 1) * areaPerPixel * degsq2str # str

        # Integrate the peak of the stars' PDF over the
        # same area as the disk.

        # CW disk
        maxProbCW = pdfSort[0:len(cwid)].sum()
        cwProb[ss] = pdf[cwid].sum() / maxProbCW
        
        # CCW "disk"
        maxProbCCW = pdfSort[0:len(ccwid)].sum()
        ccwProb[ss] = pdf[ccwid].sum() / maxProbCCW

        notOn = 'Not on '
        if (cwProb[ss] < 2.7e-3):
            notOn += 'CW,'
        if (ccwProb[ss] < 2.7e-3):
            notOn += 'CCW'
        print '%13s  %8.2e  %8.2e  %s' % \
              (name, cwProb[ss], ccwProb[ss], notOn)

        _out.write('%13s  %8.2e  %8.2e\n' % (name, cwProb[ss], ccwProb[ss]))

    _out.close()

    cwstars = (np.where(cwProb > 2.7e-3))[0]
    ccwstars = (np.where(ccwProb > 2.7e-3))[0]
    both = (np.where((cwProb > 2.7e-3) & (ccwProb > 2.7e-3)))[0]
    neither = (np.where((cwProb < 2.7e-3) & (ccwProb < 2.7e-3)))[0]

    print 
    print 'Total number of stars: %2d   avg SA = %4.2f sr' % \
          (len(cwProb), solidAngle.mean())
    print 'Solid Angle of CW disk:         SA = %5.3f sr' % \
          (len(cwid) * areaPerPixel * degsq2str)
          #(len(cwid) * areaPerPixel / areaOnSky)
    print 'Solid Angle of CCW disk:        SA = %5.3f sr' % \
          (len(ccwid) * areaPerPixel * degsq2str)
          #(len(ccwid) * areaPerPixel / areaOnSky)
    print 'Not on CW:    %2d' % (len(cwProb) - len(cwstars))
    print 'Not on CCW:   %2d' % (len(cwProb) - len(ccwstars))
    print 'Consistent with CW:    %2d   avg SA = %4.2f sr' % \
          (len(cwstars), solidAngle[cwstars].mean())
    print 'Consistent with CCW:   %2d   avg SA = %4.2f sr' % \
          (len(ccwstars), solidAngle[ccwstars].mean())
    print 'Consistent with both:  %2d   avg SA = %4.2f sr' % \
          (len(both), solidAngle[both].mean())
    print 'Not on either disk:    %2d   avg SA = %4.2f sr' % \
          (len(neither), solidAngle[neither].mean())

    idx = (np.where(cwProb == 0))[0]
    cwProb[idx] = 1.0e-5
    idx = (np.where(ccwProb == 0))[0]
    ccwProb[idx] = 1.0e-5

    py.clf()
    py.loglog(cwProb, ccwProb, 'k.')
    py.axis([1e-6, 1, 1e-6, 1])
    py.savefig(plotdir + 'diskCCW_membership.png')


def checkCCWdisk(orbDir='aorb_thesis/', mosaic=True, suffix='_middle_disk',
                 aperture=False):
    """
    Calculate the probability of a CCW disk (by default, using the
    middle radial bin stars).
    """
    # Disk solution
    nside = 64
    npix = healpy.nside2npix(nside)
    pixIdx = np.arange(0, npix)

    (disk, diskStd) = loadDiskDensity(npix, orbDir=orbDir, aperture=aperture,
                                      file1='middle_disk.neighbor.dat',
                                      file2='middle_disk.neighborStd.dat')

    (i, o) = healpy.pix2ang(nside, pixIdx)
    i *= 180.0 / math.pi
    o *= 180.0 / math.pi

    # Find the maximum density within a r=10 degree circle of the
    # previously proposed CCW disk
    # CCW disk properties (from Paumard et al. 2006)
    #ipeak = 24.0
    #opeak = 167.0
    # CCW disk properties (from Bartko et al. 2009)
    #phi = 200.0
    #theta = 142.0
    #ipeak = 180.0 - theta
    #opeak = 360.0 - phi
    # temp
    ipeak = 135. # peak in middle radial bin
    opeak = 145. # peak in middle radial bin
    #ipeak = 120. # peak in outer radial bin
    #opeak = 190. # peak in outer radial bin
    #ipeak = 81. # peak in non-disk density map
    #opeak = 254. # peak in non-disk density map
    # end temp

    sini = np.sin(np.radians(ipeak))
    cosi = np.cos(np.radians(ipeak))
    sino = np.sin(np.radians(opeak))
    coso = np.cos(np.radians(opeak))

    angle = np.zeros(len(i), float)
    for k in range(len(i)):
        inc = np.radians(i[k])
        ome = np.radians(o[k])

        angle[k] = np.arccos(sini * coso * np.sin(inc) * np.cos(ome) + \
                          sini * sino * np.sin(inc) * np.sin(ome) + \
                          cosi * np.cos(inc))
        angle[k] = np.degrees(angle[k])

    aid = (np.where(angle < 5.0))[0]
    did = (np.where(angle < 1.0))[0]

    mid = disk[aid].argmax()
    maxDensity = (disk[aid])[mid]
    imax = (i[aid])[mid]
    omax = (o[aid])[mid]

    print 'Proposed CCW Disk at i = %5.3f and o = %5.3f' % (ipeak, opeak)
    print 'Proposed CCW Disk    rho = %e  stars/deg^2' % disk[did[0]]
    print 'Within 30 deg radius of proposed CCW disk:'
    print '   Maximum Density at   i = %5.3f and o = %5.3f' % (imax, omax)
    print '   Maximum Density is   rho = %e  stars/deg^2' % maxDensity

    # What is this limit in terms of number of stars within XX deg
    #radiusForDisk = 19.0 # disk thickness proposed by Paumard+06
    radiusForDisk = 10.0 
    numStars = maxDensity * solidAngleOfCone(radiusForDisk)
    print 'Number of Stars within %2d deg:   %4.1f' % (radiusForDisk, numStars)

    # What is the measured density outside this region:
    nid = (np.where(angle > radiusForDisk))[0]
    avgBkg = 1.0
    stdBkg = 1.0

    # Iteratively compute background
    for n in range(2):
        bkg = disk[nid]
        idx = (np.where(bkg < (avgBkg + (3.0 * stdBkg))))[0]

        avgBkg = bkg[idx].mean()
        stdBkg = bkg[idx].std()
        #print 'trial = %2d   avg = %e   std = %e' % (n, avgBkg, stdBkg)
        print 'trial = %2d   avg = %e   std = %e, rejecting %d of %d' % \
              (n, avgBkg, stdBkg, (len(disk)-len(idx)), len(disk))

    print ''
    print 'Average Background   rho = %e  stars/deg^2' % avgBkg
    print 'Stddev of Background rho = %e  stars/deg^2' % stdBkg
    print 'Density Ratio for CCW    = %5.2f' % ((maxDensity - avgBkg) / stdBkg)
    print ''

    # Are there any other peaks (> average bkg + 3 sigma)?
    #hi = np.where(disk > (avgBkg + 3.0*stdBkg))[0]
    #py.clf()
    #py.plot(i[hi],o[hi],'k.')
    #py.show()
    

    # What is the expected isotropic density:
    # Load names of young stars 
    if mosaic == True:
        # Load up mosaic data as well; select only stars at r>4, since
        # we don't want to add any info from mosaics if we have it in
        # the central 10" already
        yng1 = young.loadYoungStars(root+alnDir,fit=poly,points=points,
                                    withRVonly=True,silent=True,skipStar=['S5-237']) 
        yng2 = young.loadYoungStars(mscDir,fit=polyM,points=pointsM,
                                    mosaic=True, withRVonly=True,silent=True)
        # Merge this object with object from central 10" analysis
        yng = merge(yng1, yng2)
    else:
        yng = young.loadYoungStars(root+alnDir,fit=poly,points=points,
                                   silent=True,withRVonly=True) 
    #starCnt = len(yng.stars)

    #x = yng.getArray('x')
    #y = yng.getArray('y')
    #vx = yng.getArray('vx')
    #vy = yng.getArray('vy')

    #jz = (x*vy - y*vx) / (np.hypot(x,y) * np.hypot(vx,vy))
    #ccw = (np.where(jz < 0))[0]

    # Load disk star names and probability of disk membership.
    # Find the number of stars that are NOT on the CW disk.
    (names, diskP) = readDiskProb(diskOnly=False, suffix=suffix)
    idx = (np.where(diskP < 2.7e-3))[0]
    starCnt = len(names)
    starCntOffCW = len(idx)

    isoDensity = starCnt / areaOnSky
    isoDensityCCW = starCntOffCW / areaOnSky
    print ''
    print 'Isotropic Density (%2d stars)           rho = %e  stars/deg^2' % \
          (starCnt, isoDensity)
    print 'Isotropic Density (%2d stars not on CW) rho = %e  stars/deg^2' % \
          (starCntOffCW, isoDensityCCW)

    # Ratio of measured density to isotropic density
    print 'Density Ratio for all = %5.2f' % (maxDensity / isoDensity)
    print 'Density Ratio for CCW = %5.2f' % (maxDensity / isoDensityCCW)

    # Limits
    limitDensity = avgBkg + (3.0*stdBkg)
    #limitDensityCCW = disk[did[0]] + (3.0*stdBkg)
    limitDensityCCW = maxDensity + (3.0*stdBkg)

    #limitRadius = 19.0 # disk thickness proposed by Paumard+06
    limitRadius = 10.0 # disk thickness 
    limitArea = solidAngleOfCone(limitRadius)
    limitStars = limitDensity * limitArea
    limitStarsCCW = limitDensityCCW * limitArea
    print ''
    print 'Limit in whole sky:    %4.1f stars within r=%2d deg cone' % \
          (limitStars, limitRadius)
    print 'Limit in CCW region:   %4.1f stars within r=%2d deg cone' % \
          (limitStarsCCW, limitRadius)


def plot_extended_mass():
    """
    Plots extended mass distribution according to Trippe et al. (2008)
    and Schoedel et al. (2009) as a function of r2d.
    """
    usetexTrue()
    cc = objects.Constants()
  
    # Extended mass distribution from Trippe et al. (2008)
    Mbh_t08 = 4.0e6 	# solar masses
    rho0_t08 = 2.1e6  	# solar masses/pc^3
    Rb_as = 8.9	# break radius; arcsec
    Rb_t08 = Rb_as * dist / 206265. # pc
    const_t08 = 4.0 * math.pi * rho0_t08
    
    # Extended mass distribution from Schoedel et al. (2009)
    Mbh_s09 = 4.1e6 	# solar masses
    rho0_s09 = 3.2e4  	# solar masses/pc^3
    R_s09 = 5.0 	# pc
    const_s09 = 4.0 * math.pi * rho0_s09
    gamma = 1.0
    
    # Construct an array of radii out to 15 arcsec in steps of 0.1''
    r = (np.arange(10 * 15) * 0.1) + 0.1
    r_au = r * dist
    r_pc = r_au / cc.au_in_pc

    # Compute the enclosed mass at each radius
    mass_t08 = np.zeros((len(r_pc)), dtype=float)
    mass_s09 = np.zeros((len(r_pc)), dtype=float)
    for rr in range(len(r_pc)):
        # Trippe et al. (2008) extended mass 
        Mext_t08 = scipy.integrate.quad(lambda r: r**2 / (1 + (r / Rb_t08)**2), 0, r_pc[rr])
        Mext_t08 = const_t08 * Mext_t08[0]
        #mass_t08[rr] = (Mbh_t08 + Mext_t08) / 1.e6
        mass_t08[rr] = Mext_t08

        # Shoedel et al. (2009) extended mass 
        Mext_s09 = scipy.integrate.quad(lambda r: (r / R_s09)**(-gamma) * r**2, 0, r_pc[rr])
        Mext_s09 = const_s09 * Mext_s09[0]
        #mass_s09[rr] = (Mbh_s09 + Mext_s09) / 1.e6
        mass_s09[rr] = Mext_s09

    import matplotlib.font_manager
    prop = matplotlib.font_manager.FontProperties(size=12)

    py.clf()
    py.figure(figsize=(7,7))
    py.subplots_adjust(left=0.15, right=0.96, top=0.9, bottom=0.1,
                       wspace=0.3, hspace=0.3)
    py.semilogy(r, mass_t08, 'r-', label='Trippe+08')
    py.semilogy(r, mass_s09, 'b-', label='Schoedel+09')
    thePlot = py.gca()
    #thePlot.yaxis.set_major_formatter(py.FormatStrFormatter('%3.1f'))
    #py.axis([0,15,4.0,4.8])
    py.legend(numpoints=1,fancybox=True,loc=2,prop=prop)
    py.xlabel('Projected Radius (arcsec)')
    py.ylabel(r'$M_{ext}(r)$')
    py.savefig(plotdir + 'extended_mass_radius.png')
    usetexFalse()


def min_mass_2b_bound(mosaic=False):
    """
    
    """
    # Compute F test to see if acceleration fits are warranted
    # pvalue = Significance threshold for F test
    # Stars that should have acceleration fits are returned:
    #nameF, xFp, yFp = syYoung.velocity_vs_accel(alnDir=alnDir,pvalue=4.0,
    #                                            verbose=False)
    
    cc = objects.Constants()
    GM = cc.G * mass * cc.msun

    # Load our data
    # Load names of young stars 
    if mosaic == True:
        # Load up mosaic data as well; select only stars at r>4, since
        # we don't want to add any info from mosaics if we have it in
        # the central 10" already
        yng1 = young.loadYoungStars(root+alnDir,fit=poly,points=points,
                                    withRVonly=True,silent=True,skipStar=['S5-237']) 
        yng2 = young.loadYoungStars(mscDir,fit=polyM,points=pointsM,
                                    mosaic=True, withRVonly=True,silent=True)
        # Merge this object with object from central 10" analysis
        yng = merge(yng1, yng2)
    else:
        yng = young.loadYoungStars(root+alnDir,fit=poly,points=points,
                                   silent=True,withRVonly=True) 

    names = yng.getArray('name')

    # We have two equations, and two unknowns
    # Eq. 1: Max |acceleration| at projected radius r2d
    # Eq. 2 -- BOUND ASSUMPTION! v3d**2 / 2 <= GM/r3d
    # From these two, we can get lower limits for
    # the two unknowns, r3d and mass (BH + extended)

    #py.clf()
    #py.figure(figsize=(7,7))
    fig = py.figure(1)
    fig.clear()
    fig.hold(True)
    thePlot = fig.add_subplot(111)
    usetexTrue()

    for name in names:
	i = names.index(name)

	star = yng.stars[i]

        mag = yng.stars[i].mag
        x = star.fitXa.p
        y = star.fitYa.p
        xe = star.fitXa.perr
        ye = star.fitYa.perr
        vx = star.fitXa.v # km/s
        vy = star.fitYa.v # km/s
        vz = star.vz # km/s
        vxe = star.fitXa.verr # km/s
        vye = star.fitYa.verr # km/s
        vze = star.vzerr # km/s
        ax = star.fitXa.a
        ay = star.fitYa.a
        axe = star.fitXa.aerr
        aye = star.fitYa.aerr
        t0 = star.fitXa.t0

        # How many epochs was each star detected in?
        pts = asciidata.open('%s%s%s%s.points' % (root, alnDir, points, name))
        ep = pts[0].tonumpy()
        cnt = len(ep)

        r2d = np.sqrt(x**2 + y**2) # arcsec
        r2d_err = (np.sqrt(((x*xe)**2 + (y*ye)**2) / r2d**2)) # arcsec
        rcgs = r2d * dist * cc.cm_in_pc / cc.au_in_pc # cm
        rerr_cgs = r2d_err * dist * cc.cm_in_pc / cc.au_in_pc # cm
        
	# Lower allowed radius is set by 2D distance.
	rLower = r2d * dist / cc.au_in_pc # pc

        cc.asy_to_kms = dist * cc.cm_in_au / (1.e5 * cc.sec_in_yr)

	ax *= cc.asy_to_kms
	ay *= cc.asy_to_kms
	axe *= cc.asy_to_kms
	aye *= cc.asy_to_kms
	
	# What is the polyfit acceleration upper limit
	# along the radial component, in km/s/yr
        (ar, at, are, ate) = util.xy2circErr(x, y, ax, ay,
                                             xe, ye, axe, aye)

        # This plot will only work for stars w/ significant accelerations (detections)

	# What is the radial component upper limit
	accLimRadial = ar - (3.0 * are) # km/s/yr
        alimcgs = accLimRadial * 1.0e5 / cc.sec_in_yr # cm/s/s

        #foo = (-GM * rcgs / alimcgs)**(2.0/3.0)# Gives (minimum r)-squared
        #if ((foo - rcgs**2) > 0) and (-ar - (3.0*are) > 0.0): # Below the curve (at X sigma)
        if ((-ar - (3.0*are) > 0.0) and (cnt >= 30)):  # if it's significantly different
            					      # from zero
            detection = True
        else:
            continue

        #if (name not in nameF) or (cnt < 30):
        #    continue
        
        ar_cgs = ar * 1.0e5 / cc.sec_in_yr
        are_cgs = are * 1.0e5 / cc.sec_in_yr

        vtot = np.sqrt(vx**2 + vy**2 + vz**2) # km/s
        vtot_err = (np.sqrt(((vx*vxe)**2 + (vy*vye)**2 + (vz*vze)**2) / vtot**2)) # km/s
    
        vtot_cgs = vtot * 1.e5
        vtot_err_cgs = vtot_err * 1.e5

        # Lower limit on r3d:
        r3d_lim = (vtot_cgs / np.sqrt(2.0)) * (np.sqrt(rcgs / np.abs(ar_cgs)))
        r3d_lim_pc = r3d_lim / cc.cm_in_pc 
    
        # Lower limit on enclosed mass:
        mass_lim = (vtot_cgs**3 / (2.0*cc.G)) * (np.sqrt(rcgs / np.abs(ar_cgs)))
        mass_lim_msun = mass_lim / cc.msun

        # Propagated error on mass limit
        t1 = 9.0 * rcgs * vtot_err_cgs**2
        t2 = vtot_cgs**2 / (4.0 * rcgs) * rerr_cgs**2
        t3 = vtot_cgs**2 / (4.0 * ar_cgs**2) * (rcgs * are_cgs**2)
        ml_sig = np.sqrt(vtot_cgs**4 / (8.0*cc.G**2*np.abs(ar_cgs)) * (t1 + t2 + t3))
        ml_sig /= cc.msun

        rad_ratio = r3d_lim_pc / rLower

        p1 = thePlot.errorbar(rad_ratio, mass_lim_msun, yerr=ml_sig, fmt='k.')
        thePlot.text(rad_ratio+0.05,mass_lim_msun, name, fontsize=10)

    thePlot.semilogy([0,1.5],[4.0e6,4.0e6],'k-')
    thePlot.semilogy([0,1.5],[4.6e6,4.6e6],'k--')
    thePlot.semilogy([0,1.5],[3.4e6,3.4e6],'k--')
    thePlot.semilogy([1.0,1.0],[8e3,1e7],'k--')
    thePlot.set_yscale('log')
    thePlot.axis([0,1.4,6e3,1e7])
    py.xlabel(r'r$_{min} / r_{2d}$')
    py.ylabel(r'Minimum Mass to be Bound ($M_{sun}$)')
    py.savefig(plotdir + 'min_mass_2b_bound.png')
    usetexFalse()

    

def plotAllYngStars(align = 'align/align_d_rms_1000_abs_t',mosaic=True,
                    makeHealPix=False,aorbDir='aorb_thesis/',suffix=''):
    """
    Wrapper that calls plotgc/plotStar.go() for all young stars known
    """

    # Load up young stars
    # young.loadYoungStars calls youngStarNames but grabs
    # only young stars beyond r = 0.8 arcsec
    # Load names of young stars 
    if mosaic == True:
        # Load up mosaic data as well; select only stars at r>4, since
        # we don't want to add any info from mosaics if we have it in
        # the central 10" already
        yng1 = young.loadYoungStars(root+alnDir,fit=poly,points=points,
                                    silent=True, withRVonly=True,skipStar=['S5-237']) 
        yng2 = young.loadYoungStars(mscDir,align=align, fit=polyM,points=pointsM,
                                    mosaic=True, silent=True, withRVonly=True)
        cntrlNames = yng1.getArray('name')
        mscNames = yng2.getArray('name')
        # Merge this object with object from central 10" analysis
        yng = merge(yng1, yng2)
    else:
        yng = young.loadYoungStars(root+alnDir,fit=poly,points=points,silent=True) 
    names = yng.getArray('name')

    py.figure(figsize=(6,6))
    py.clf()
    py.subplots_adjust(left=0.1,bottom=0.1,right=0.98,top=0.95,
                       wspace=0.3,hspace=0.2)
    for yy in names:
        if (mosaic == True):
            if (yy in mscNames) & (yy not in cntrlNames):
                plotStar.go(yy, mscDir, align=align, poly=polyM, points=pointsM,
                            fit='linear', suffix=suffix, LGSonly=False)
        else:
            plotStar.go(yy, root+alnDir, align=align, poly=poly, points=points,
                        fit='accel', suffix=suffix, LGSonly=False)

        if makeHealPix == True:
            outroot = root+alnDir+aorbDir
            pdffile = '%s/%s.mc.dat' % (outroot, yy)
            pdf = pickle.load(open(pdffile))
            # Make a HEALpix map
            hp = makePdfHealpix(yy, pdf, outroot, sim=False,
                                ntrials=10**5, nside=64,makeplot=True)

def mosaic_vel_chi2(withRVonly=True, compare_jlu=False,
                    align='align/align_d_rms_1000_abs_t', suffix=''):
    """
    Plots histogram of chi2 values from velocity fit in wide mosaic and
    velocity error vs. magnitude.
    Also compares velocities to Bartko results to check for possible
    mismatches in our data set; and compares to JLu's alignment
    """
    cc = objects.Constants()

    cntrl = young.loadYoungStars(root+alnDir,fit=poly,points=points,
                                silent=True,skipStar=['S5-237']) 

    s = young.loadYoungStars(mscDir,align=align,fit=polyM,points=pointsM, withRVonly=withRVonly,
                             mosaic=True, silent=True)
    cntrlNames = cntrl.getArray('name')
    mscNames = s.getArray('name')

    mscOnly = np.setdiff1d(mscNames,cntrlNames)
    msc = np.zeros(len(mscOnly))
    for mm in range(len(mscOnly)):
        _msc = np.where(np.array(mscNames) == mscOnly[mm])[0]
        msc[mm] = _msc

    # msc contains indices of stars that are outside central 10 asec images
    msc = [int(mm) for mm in msc]

    mag = s.getArray('mag')

    # Polyfit 
    t0 = s.getArray('fitpXv.t0')
    velCnt = s.getArray('velCnt')

    x = s.getArray('fitpXv.p') * -1.0
    xerr = s.getArray('fitpXv.perr')
    vx = s.getArray('fitpXv.v') * 10**3 * -1.0
    vxerr = s.getArray('fitpXv.verr') * 10**3
    xchi2 = s.getArray('fitpXv.chi2')
    xchi2r = s.getArray('fitpXv.chi2red')

    y = s.getArray('fitpYv.p')
    yerr = s.getArray('fitpYv.perr')
    vy = s.getArray('fitpYv.v') * 10**3
    vyerr = s.getArray('fitpYv.verr') * 10**3
    ychi2 = s.getArray('fitpYv.chi2')
    ychi2r = s.getArray('fitpYv.chi2red')

    r2d = np.sqrt(x**2 + y**2)

    vtot = np.sqrt(vx**2 + vy**2)
    vtoterr = np.sqrt((vx*vxerr)**2 + (vy*vyerr)**2) / vtot

    all_chi2 = np.array([xchi2,ychi2]).flatten()

    # Average chi2
    ave_chi2 = (xchi2 + ychi2) / 2.0
    ave_chi2r = (xchi2r + ychi2r) / 2.0

    # Add the residual distortion error
    xerr_dist = np.sqrt((xerr*1e3)**2 + 1.0**2)
    yerr_dist = np.sqrt((yerr*1e3)**2 + 1.0**2)

    # Average pos error
    ave_posErr = (xerr + yerr) / 2.0 * 10**3

    # Average vel error
    ave_velErr = (vxerr + vyerr) / 2.0

    # Get radial velocities
    vz = s.getArray('vz')
    vzerr = s.getArray('vzerr')
    rv_ref = np.array(s.getArray('rv_ref'))
    noRV = np.where(np.array([rr is None for rr in rv_ref]) == True)[0]
    hasRV = np.where(np.array([rr is None for rr in rv_ref]) == False)[0]

    # Expected chi2 for n dof:
    binsIn = py.arange(0, 15, 0.4)
    numEpochs = 3
    dof = numEpochs - 2
    xpctd3 = stats.chi2.pdf(binsIn,dof) # for stars in all 3 epochs

    fmt = '%12s  %5.2f  %6.3f  %6.3f  %6.3f  %6.3f  %6.3f  %6.3f'
    hdr = '%12s  %5s  %6s  %6s  %6s  %6s  %6s  %6s'
    print
    print 'Stars outside central 10 arcsec FOV:'
    print hdr % ('Name', 'K', 'Vx', 'Vy', 'Vxerr', 'Vyerr', 'VxChi2r', 'VyChi2r')
    for mm in range(len(x)):
        if mm not in msc:
            continue
        print fmt % \
              (mscNames[mm], mag[mm], vx[mm], vy[mm], vxerr[mm],
               vyerr[mm], xchi2r[mm], ychi2r[mm])

    # Find stars with Xchi2 > 8.0 -- we will trim these from our analysis
    print ''
    #print 'Stars outside the central 10 asec with poor velocity fits (xchi2 or ychi2 > 8.0):'
    print 'Stars outside the central 10 asec with poor velocity fits (xchi2 > 4.0):'
    hdr = '%12s  %8s  %8s'
    fmt = '%12s  %8.2f  %8.2f'
    print hdr % ('Name', 'Xchi2', 'Ychi2')
    bad = np.where((xchi2 >= 8.0) | (ychi2 >= 8.0))[0]
    bad = np.where(xchi2 >= 4.0)[0]
    badCnt = 0
    for bb in range(len(bad)):
        if mscNames[bad[bb]] in cntrlNames:
            continue
        else:
            print fmt % (mscNames[bad[bb]], xchi2[bad[bb]], ychi2[bad[bb]])
            badCnt += 1
    print ''
    print 'Number of stars with xchi2 or ychi2 > 8.0: %i' % badCnt
    #print 'Number of stars with xchi2 > 4.0: %i' % badCnt
    print ''

    import matplotlib.font_manager
    prop = matplotlib.font_manager.FontProperties(size=12)

    # Pull Bartko+09 data from database and compare to our values
    bart = tabs.Bartko2009()
    bartNames = bart.ourName
    bmag = bart.Kmag
    bx = bart.x
    by = bart.y
    br2d = bart.r2d
    bvx = bart.vx # km/s
    bvy = bart.vy # km/s
    bvxe = bart.vxerr # km/s
    bvye = bart.vyerr # km/s
    py.figure()
    py.figure(1,figsize=(12,5))
    py.clf()
    py.subplots_adjust(left=0.1,bottom=0.1,right=0.95,top=0.9,
                       hspace=0.3,wspace=0.3)
    bcnt = 0
    cc.asy_to_kms = dist * cc.cm_in_au / (1.e5 * cc.sec_in_yr)
    for bb in range(len(bartNames)):
        try:
            idx = mscNames.index(bartNames[bb])
        except ValueError:
            print 'Cannot find %s in our data' % bartNames[bb]
            continue
        vx_kms = vx[idx]/1.e3 * cc.asy_to_kms
        vy_kms = vy[idx]/1.e3 * cc.asy_to_kms
        vxe_kms = vxerr[idx]/1.e3 * cc.asy_to_kms
        vye_kms = vyerr[idx]/1.e3 * cc.asy_to_kms

        # compare PMs
        py.subplot(1,3,1)
        py.plot(vx_kms,bvx[bb],'r.')
        py.plot(vy_kms,bvy[bb],'b.')

        # Compare magnitudes
        py.subplot(1,3,2)
        py.plot(mag[idx],bmag[bb],'k.')

        # Plot delta vels vs. radius
        #dm = np.abs(mag[idx] - bmag[bb])
        vtot_uc = np.hypot(vx_kms, vy_kms)
        vtote_uc = np.sqrt((vx_kms*vxe_kms)**2 + (vy_kms*vye_kms)**2) / vtot_uc
        vtot_vlt = np.hypot(bvx[bb], bvy[bb])
        vtote_vlt = np.sqrt((bvx[bb]*bvxe[bb])**2 + (bvy[bb]*bvye[bb])**2) / vtot_vlt
        vtotErr = np.hypot(vtote_uc, vtote_vlt)
        dv = np.abs(vtot_uc - vtot_vlt)
        py.subplot(1,3,3)
        py.errorbar(r2d[idx], dv, yerr=vtotErr, fmt='k.')
        #py.plot(r2d[idx], dv, 'k.')

        # Print out stars with significantly different proper motions
        # between UCLA and VLT
        if (dv / vtotErr) > 3.0:
            print 'Discrepant Proper motions (UCLA vs. VLT): %8s %6.1f vs. %6.1f km/s' % \
                  (mscNames[idx], vtot_uc, vtot_vlt)
        
        bcnt += 1

    # Which stars do not have significant proper motion measurements?
    # First check proper motions
    vidx = np.where((vtot - (3.0*vtoterr)) <= 0.0)[0]
    print ''
    print 'Stars with proper motions consistent with zero:'
    print '%12s  %5s  %5s  %5s' % ('Name', 'Vtot', 'Verr', 'Vsig')
    for vv in vidx:
        print '%12s  %5.3f  %5.3f  %5.3f' % \
              (mscNames[vv], vtot[vv], vtoterr[vv], vtot[vv]/vtoterr[vv])
    # Now check radial velocity measurements
    print ''
    print 'Stars with radial velocities consistent with zero:'
    for rr in hasRV:
        vzsig = np.abs(vz[rr]) - 3.0*vzerr[rr]
        if vzsig <= 0.0:
            print  '%12s  %8.3f  %8.3f  %5.3f' % \
              (mscNames[rr], vz[rr], vzerr[rr], np.abs(vz[rr])/vzerr[rr])

    print
    print 'Found %i stars in common with Bartko' % bcnt
    print

    # Finish plotting the bartko comparison
    py.subplot(1,3,1)
    py.plot([-550,550],[-550,550],'k--')
    py.axis([-550,550,-550,550])
    py.xlabel('UCLA Velocity (km/s)')
    py.ylabel('VLT Velocity (km/s)')
    py.title('Velocity Comparison')
    py.subplot(1,3,2)
    py.plot([9,15],[9,15],'k--')
    py.axis([9,15,9,15])
    py.xlabel('UCLA Magnitude')
    py.ylabel('VLT Magnitude')
    py.title('Magnitude Comparison')
    py.subplot(1,3,3)
    py.xlabel('Radius (arcsec)')
    py.ylabel('Delta Velocity (km/s)')
    py.savefig(mscDir + 'plots/compare_mosaic_velsMags_bartko%s.png' % suffix)
    py.close(1)

    if compare_jlu == True:
        # Compare with Jessica's mosaic alignment
        jlDir = '/g/uni/ghez/jlu/gc/dp_msc/2011_05_29/'
        jl = young.loadYoungStars(jlDir,fit='polyfit_d/fit',points='points_d/', withRVonly=withRVonly,
                                 mosaic=True, silent=True)
        jname = jl.getArray('name')
        jmag = jl.getArray('mag')
    
        # Polyfit 
        jt0 = jl.getArray('fitpXv.t0')
        jvelCnt = jl.getArray('velCnt')
    
        jx = jl.getArray('fitpXv.p') * -1.0
        jxerr = jl.getArray('fitpXv.perr')
        jvx = jl.getArray('fitpXv.v') * 10**3 * -1.0
        jvxerr = jl.getArray('fitpXv.verr') * 10**3
        jxchi2 = jl.getArray('fitpXv.chi2')
        jxchi2r = jl.getArray('fitpXv.chi2red')
    
        jy = jl.getArray('fitpYv.p')
        jyerr = jl.getArray('fitpYv.perr')
        jvy = jl.getArray('fitpYv.v') * 10**3
        jvyerr = jl.getArray('fitpYv.verr') * 10**3
        jychi2 = jl.getArray('fitpYv.chi2')
        jychi2r = jl.getArray('fitpYv.chi2red')
    
        jr2d = np.sqrt(jx**2 + jy**2)
    
        jvtot = np.sqrt(jvx**2 + jvy**2)
        jvtoterr = np.sqrt((jvx*jvxerr)**2 + (jvy*jvyerr)**2) / jvtot
    
        # plot Jessica's - chi2 histogram for all young stars in the mosaic
        py.clf()
        py.figure(figsize=(6,6))
        py.hist(jxchi2,binsIn,color='r',lw=2,histtype='step',normed=True,label='X')
        py.hist(jychi2,binsIn,color='b',lw=2,histtype='step',normed=True,label='Y')
        py.plot(binsIn,xpctd3,'k-',label='Expected',lw=1)
        py.title('JLu Align - All YSOs in Deep Mosaic (N=%i)' % len(xchi2),fontsize=14)
        py.xlabel(r'$\chi^2$')
        py.ylabel('N')
        py.legend(numpoints=1,fancybox=True,prop=prop)
        py.axis([0,16,0,0.7])
        py.savefig(mscDir + 'plots/jlu_chi2vel_deep_mosaic%s.png' % suffix)
        
        py.figure(2,figsize=(8,8))
        py.clf()
        py.subplots_adjust(left=0.12,bottom=0.1,right=0.95,top=0.95,
                           hspace=0.3,wspace=0.3)
        dvx_all = []
        dvy_all = []
        jcnt = 0
        for jj in range(len(jname)):
            # Find the matching young star in my align:
            try:
                idx = mscNames.index(jname[jj])
            except ValueError:
                continue
            vx_kms = vx[idx]/1.e3 * cc.asy_to_kms
            vy_kms = vy[idx]/1.e3 * cc.asy_to_kms
    
            jvx_kms = jvx[jj]/1.e3 * cc.asy_to_kms
            jvy_kms = jvy[jj]/1.e3 * cc.asy_to_kms
    
            # Velocity differences
            dvx = vx_kms - jvx_kms
            dvy = vy_kms - jvy_kms
            dvx_all = np.concatenate([dvx_all, [dvx]])
            dvy_all = np.concatenate([dvy_all, [dvy]])
    
            # compare velocities
            py.subplot(2,2,1)
            py.plot(vx_kms,jvx_kms,'r.')
            py.plot(vy_kms,jvy_kms,'b.')
    
            # Compare magnitudes
            py.subplot(2,2,3)
            py.plot(mag[idx],jmag[jj],'k.')
    
            # Plot delta vels vs. delta mag
            dm = np.abs(mag[idx] - jmag[jj])
            vtot_uc = np.hypot(vx_kms, vy_kms)
            vtot_jl = np.hypot(jvx_kms, jvy_kms)
            dv = np.abs(vtot_uc - vtot_jl)
            py.subplot(2,2,4)
            py.plot(dm, dv, 'k.')
            
            jcnt += 1
    
        print
        print 'Found %i young stars in common with JLu' % jcnt
    
        # Finish plotting the jlu comparison
        py.subplot(2,2,1)
        py.plot([-550,550],[-550,550],'k--')
        py.axis([-550,550,-550,550])
        py.xlabel('My Velocity (km/s)')
        py.ylabel('JLu Velocity (km/s)')
        py.title('Velocity Comparison')
        py.subplot(2,2,2)
        py.hist(dvx_all, bins=50, color='r', histtype='step', label='Vx')
        py.hist(dvy_all, bins=50, color='b', histtype='step', label='Vy')
        py.xlabel('Velocity Difference (km/s)')
        py.ylabel('N')
        py.legend(numpoints=1, fancybox=True, loc=2)
        py.subplot(2,2,3)
        py.plot([9,15],[9,15],'k--')
        py.axis([9,15,9,15])
        py.xlabel('My Magnitude')
        py.ylabel('JLu Magnitude')
        py.title('Magnitude Comparison')
        py.subplot(2,2,4)
        py.xlabel('Delta Magnitude')
        py.ylabel('Delta Velocity (km/s)')
        py.savefig(mscDir + 'plots/compare_mosaic_velsMags_jlu%s.png' % suffix)
        py.close(2)

    #py.close('all')
    # Plots
    usetexTrue()
    py.figure(1,figsize=(10,5))
    py.clf()
    py.subplots_adjust(left=0.1,bottom=0.15,right=0.98,top=0.9,
                       wspace=0.3,hspace=0.2)
    # plot 1 - chi2 histogram for all stars in the mosaic
    py.subplot(1,2,1)
    py.hist(xchi2,binsIn,color='r',lw=2,histtype='step',normed=True,label='X')
    py.hist(ychi2,binsIn,color='b',lw=2,histtype='step',normed=True,label='Y')
    py.plot(binsIn,xpctd3,'k-',label='Expected',lw=1)
    py.title('All YSOs in Deep Mosaic (N=%i)' % len(xchi2),fontsize=14)
    py.xlabel(r'$\chi^2$')
    py.ylabel('N')
    py.legend(numpoints=1,fancybox=True,prop=prop)
    py.axis([0,16,0,0.7])
    py.subplot(1,2,2)
    # Just plot those not in the central 10 arcsec images
    py.hist(xchi2[msc],binsIn,color='r',lw=2,histtype='step',normed=True)
    py.hist(ychi2[msc],binsIn,color='b',lw=2,histtype='step',normed=True)
    py.plot(binsIn,xpctd3,'k-',label='Expected',lw=1)
    py.title('YSOs Outside Central 10 arcsec (N=%i)' % len(msc),fontsize=14)
    py.xlabel(r'$\chi^2$')
    py.ylabel('N')
    py.axis([0,16,0,0.7])
    py.savefig(mscDir + 'plots/chi2vel_deep_mosaic%s.png' % suffix)
    py.close(1)
    usetexFalse()
    print 'Number of young stars in full mosaic: %i' % len(xchi2)
    print '<vxerr> = %5.3f +- %5.3f mas/yr' % (vxerr.mean(), vxerr.std(ddof=1))
    print '<vyerr> = %5.3f +- %5.3f mas/yr' % (vyerr.mean(), vyerr.std(ddof=1))
    print 'Median Velocity Error (X, Y): (%5.3f, %5.3f) mas/yr' % \
          (np.median(vxerr), np.median(vyerr))
    print
    print 'Number of young stars outside central 10 arcsec: %i' % len(msc)
    print '<vxerr> = %5.3f +- %5.3f mas/yr' % (vxerr[msc].mean(), vxerr[msc].std(ddof=1))
    print '<vyerr> = %5.3f +- %5.3f mas/yr' % (vyerr[msc].mean(), vyerr[msc].std(ddof=1))
    print 'Median Velocity Error (X, Y): (%5.3f, %5.3f) mas/yr' % \
          (np.median(vxerr[msc]), np.median(vyerr[msc]))
    print
    
    # Velocity error vs. K mag
    py.figure(2,figsize=(12,5))
    py.clf()
    py.subplots_adjust(left=0.08,bottom=0.15,right=0.98,top=0.9,
                       wspace=0.3,hspace=0.2)
    py.subplot(1,3,1)
    py.errorbar(r2d, vx, yerr=vxerr, fmt='r.', label='X')
    py.errorbar(r2d, vy, yerr=vyerr, fmt='b.', label='Y')
    py.plot([0,14],[0,0],'k--')
    py.legend(loc=1,numpoints=1,fancybox=True)
    py.title('All YSOs in Deep Mosaic (N=%i)' % len(xchi2),fontsize=12)
    py.xlabel('Projected Radius (arcsec)')
    py.ylabel('Velocity (mas/yr)')
    py.subplot(1,3,2)
    py.plot(mag, vxerr, 'r.')
    py.plot(mag, vyerr, 'b.')
    #py.legend(('X', 'Y'),loc=1,numpoints=1)
    py.title('All YSOs in Deep Mosaic (N=%i)' % len(xchi2),fontsize=12)
    py.xlabel('K Magnitude')
    py.ylabel('Velocity Error (mas/yr)')
    py.subplot(1,3,3)
    py.plot(mag[msc], vxerr[msc], 'r.')
    py.plot(mag[msc], vyerr[msc], 'b.')
    py.title('YSOs Outside Central 10 arcsec (N=%i)' % len(msc),fontsize=12)
    py.xlabel('K Magnitude')
    py.ylabel('Velocity Error (mas/yr)')
    py.savefig(mscDir + 'plots/velKmag_deep_mosaic%s.png' % suffix)
    py.close(2)

    #bad = np.where((vxerr > 1.0) | (vyerr > 1.0))[0]
    #fmt = '%12s  %5.3f  %7.3f  %5.3f  %5.3f'
    #for bb in bad:
    #    print fmt % (mscNames[bb], mag[bb], r2d[bb], vxerr[bb], vyerr[bb])

    # These are the stars that show inclination=90 degrees from analyticOrbits
    # analysis. It is likely b/c of their low velocities for their r2d since
    # i goes like arccos( r x v )
    #i90 = ['S1-1', 'S1-24', 'S2-66', 'S3-2', 'S5-34', 'S5-187', 'S5-235',
    #       'S5-236', 'S7-180', 'S8-4', 'S8-196', 'S9-9', 'S9-114', 'S9-283',
    #       'irs13E1', 'irs13E3b', 'irs29N']
    py.figure(3,figsize=(12,5))
    py.clf()
    py.subplots_adjust(left=0.08,bottom=0.15,right=0.98,top=0.9,
                       hspace=0.3,wspace=0.3)
    py.subplot(1,3,1)
    py.errorbar(r2d, vx, yerr=vxerr, fmt='r.')
    #for ii in range(len(r2d)):
    #    if mscNames[ii] in i90:
    #        py.plot(r2d[ii], vx[ii], 'kx', ms=7, mew=2)
    py.plot([0,14],[0,0],'k--')
    py.title('All YSOs in Deep Mosaic (N=%i)' % len(xchi2),fontsize=14)
    py.xlabel('Projected Radius (arcsec)')
    py.ylabel('X Velocity (mas/yr)')
    py.subplot(1,3,2)
    py.errorbar(r2d, vy, yerr=vyerr, fmt='b.')
    #for ii in range(len(r2d)):
    #    if mscNames[ii] in i90:
    #        py.plot(r2d[ii], vy[ii], 'kx', ms=7, mew=2)
    py.plot([0,14],[0,0],'k--')
    py.xlabel('Projected Radius (arcsec)')
    py.ylabel('Y Velocity (mas/yr)')
    py.subplot(1,3,3)
    py.errorbar(r2d, vtot, yerr=vtoterr, fmt='k.')
    #for ii in range(len(r2d)):
    #    if mscNames[ii] in i90:
    #        py.plot(r2d[ii], vtot[ii], 'kx', ms=7, mew=2)
    py.plot([0,14],[0,0],'k--')
    py.xlabel('Projected Radius (arcsec)')
    py.ylabel('Total Proper Motion (mas/yr)')
    py.savefig(mscDir + 'plots/vel_r2d_mosaic%s.png' % suffix)
    py.close(3)

    # Identify stars in outer bin that contribute to apparent structure
    #outer = ['S6-100','S7-16','S8-4','S8-15','S9-23','S10-4','S10-5','S11-5','S11-21']
    outer = ['S6-100','S7-16','S8-15','S9-23','S10-4','S10-5']
    py.figure(4)
    py.figure(figsize=(6,6))
    py.clf()
    py.subplots_adjust(left=0.15,bottom=0.15,right=0.95,top=0.9)
    for rr in hasRV:
        py.errorbar(r2d[rr], np.abs(vz[rr]), yerr=vzerr[rr], fmt='k.')
        if mscNames[rr] in outer:
            #py.plot(r2d[rr], np.abs(vz[rr]), 'rx', ms=7, mew=2)
            py.errorbar(r2d[rr], np.abs(vz[rr]), yerr=vzerr[rr], fmt='r.')
    py.plot([0,14],[0,0],'k--')
    py.title('All YSOs in Deep Mosaic (N=%i)' % len(xchi2),fontsize=14)
    py.xlabel('Projected Radius (arcsec)')
    py.ylabel('Radial Velocity (km/s)')
    py.savefig(mscDir + 'plots/rv_r2d_mosaic%s.png' % suffix)
    py.close(4)

    usetexTrue()
    py.figure(5)
    py.figure(figsize=(6,6))
    py.clf()
    py.subplots_adjust(left=0.15,bottom=0.15,right=0.95,top=0.9)
    py.semilogy(r2d, xchi2, 'r.', ms=7, mew=2)
    py.semilogy(r2d, ychi2, 'b.', ms=7, mew=2)
    #for rr in range(len(r2d)):
        #if mscNames[rr] in i90:
        #    py.semilogy(r2d[rr], xchi2[rr], 'rx', ms=7, mew=2)
        #    py.semilogy(r2d[rr], ychi2[rr], 'bx', ms=7, mew=2)
    py.title('All YSOs in Deep Mosaic (N=%i)' % len(xchi2),fontsize=14)
    py.xlabel('Projected Radius (arcsec)')
    py.ylabel(r'Velocity ${\chi^2}$')
    py.savefig(mscDir + 'plots/chi2_r2d_mosaic%s.png' % suffix)
    py.close(5)
    usetexFalse()

    py.figure(6)
    py.clf()
    py.figure(figsize=(6,6))
    x_km = x[hasRV] * dist * cc.cm_in_au / 1.e5
    xe_km = xerr[hasRV] * dist * cc.cm_in_au / 1.e5
    vzt = vz[hasRV]
    vzterr = vzerr[hasRV]
    x_vz = x_km * vzt
    for ii in range(len(hasRV)):
        xvz_err = np.sqrt(vzt[ii]**2*xe_km[ii]**2 + x_km[ii]**2*vzterr[ii]**2)
        py.errorbar(r2d[hasRV[ii]], x_vz[ii], yerr=xvz_err, fmt='k.')
    py.plot([0,14],[0,0],'k--')
    py.xlabel('Projected Radius (arcsec)')
    py.ylabel('X * Vz')
    py.savefig(mscDir + 'plots/x_vz_r2d%s.png' % suffix)
    py.close(6)

    usetexTrue()
    py.figure(7)
    py.figure(figsize=(6,6))
    py.clf()
    py.subplots_adjust(left=0.15,bottom=0.15,right=0.95,top=0.9)
    py.semilogy(r2d[msc], xchi2r[msc], 'r.', ms=7, mew=2)
    py.semilogy(r2d[msc], ychi2r[msc], 'b.', ms=7, mew=2)
    py.title('YSOs Outside Central 10 arcsec (N=%i)' % len(msc),fontsize=14)
    py.xlabel('Projected Radius (arcsec)')
    py.ylabel(r'Velocity ${\chi^2}$')
    py.savefig(mscDir + 'plots/chi2red_r2d_mosaic%s.png' % suffix)
    py.close(7)
    usetexFalse()
    

def listYngWithRV(align='align/align_d_rms_1000_abs_t', mosaic=False):
    """
    Lists young stars with RVs (from OSIRIS or other lit)
    """
    
    # Load names of young stars in our align
    if mosaic == True:
        # Load up mosaic data as well; select only stars at r>4, since
        # we don't want to add any info from mosaics if we have it in
        # the central 10" already
        yng1 = young.loadYoungStars(root+alnDir,fit=poly,points=points,verbose=True,
                                    radiusCut=0.8, silent=True,skipStar=['S5-237']) 
        yng2 = young.loadYoungStars(mscDir,fit=polyM,points=pointsM,verbose=True,
                                    mosaic=True, silent=True)
        # Merge this object with object from central 10" analysis
        yng = merge(yng1, yng2)
    else:
        yng = young.loadYoungStars(root+alnDir,fit=poly,points=points,silent=True,
                                   radiusCut=0.8, verbose=True)

    yngNames = yng.getArray('name')
    rv_ref = np.array(yng.getArray('rv_ref'))

    # Load up young stars listed in the SQL database
    allyng = young.youngStarNames()

    # Of the young stars in our align, how many have RVs?
    ucla = len(np.where(rv_ref == 'UCLA')[0])
    bart = len(np.where(rv_ref == 'Bartko+2009')[0])
    paum = len(np.where(rv_ref == 'Paumard+2006')[0])
    noRV = len(np.where(np.array([rr is None for rr in rv_ref]) == True)[0])
    print
    print 'Total number of young stars in align: %i' % len(yngNames)
    print 'Stars with RVs from OSIRIS: %i' % ucla
    print 'Stars with RVs from Bartko+2009: %i' % bart
    print 'Stars with RVs from Paumard+2006: %i' % paum
    print 'Stars with no RVs: %i' % noRV

    # Which ones are not in our align?
    missing = np.setdiff1d(allyng, yngNames)

    print 
    print '%i young stars not found in our align:' % len(missing)
    for mm in missing:
        print mm


def compare_RVs_lit():
    """
    Compare radial velocities from OSIRIS and SINFONI
    """
    rootDir = root + alnDir 
    align = 'align/align_d_rms_1000_abs_t'

    # Load up the absolute position of Sgr A* based on S0-2's orbit
    t = objects.Transform()
    t.loadAbsolute()

    # Load up position/velocity information from the central 10'' align,
    # since this is mainly where UCLA and VLT mainly have overlapping data
    s = starset.StarSet(rootDir + align, relErr=1, trans=t)
    s.loadPolyfit(rootDir + poly, arcsec=1, silent=False)

    yng = young.youngStarNames()
    # Tables in database w/ RV information
    ucla = tabs.UCLAstars()
    bart = tabs.Bartko2009()
    paum = tabs.Paumard2006()

    cc = objects.Constants()

    # Compare OSIRIS and SINFONI RVs that are in common:
    # First pull Bartko data, get the ucla name, then find ucla velocity
    ucN = bart.ourName
    bx = bart.x
    by = bart.y
    bvz = bart.vz
    bvze = bart.vzerr
    bvzt0 = bart.t0_spectra

    # Pull out from the set of stars those that are young
    # stars.
    stars = []
    names = [star.name for star in s.stars]
    radiusCut = 0.8

    ucvz_mtch = []
    ucvze_mtch = []
    bvz_mtch = []
    bvze_mtch = []
    name_mtch = []
    uct0_mtch = []
    ucx_mtch = []
    ucy_mtch = []
    bt0_mtch = []

    for name in yng:
	# Find the star in our star lists
	try:
	    idx = names.index(name)
	    star = s.stars[idx]

	    if (star.r2d >= radiusCut):
		stars.append(star)
            else:
                continue

	    # Find the radial velocity for each star
            # First look in OSIRIS data, then Bartko, then Paumard
            idx = np.where(ucla.ourName == name)[0]
            bidx = np.where(bart.ourName == name)[0]
            if (len(idx) > 0) and (len(bidx) > 0):
                if ucla.vz[idx][0] == None:
                    continue
	        star.ucx = ucla.x[idx][0]
	        star.ucy = ucla.y[idx][0]
	        star.ucvz = ucla.vz[idx][0]
                if star.ucvz == '':
                    continue
	        star.ucvzerr = ucla.vzerr[idx][0]
	        star.ucvzt0 = ucla.t0_spectra[idx][0]
                # Now go to other RV tables
                star.bvz = bart.vz[bidx][0] 
                star.bvzerr = bart.vzerr[bidx][0]
                star.bvzt0 = bart.t0_spectra[bidx][0]

                ucvz_mtch = np.concatenate([ucvz_mtch, [star.ucvz]])
                ucvze_mtch = np.concatenate([ucvze_mtch, [star.ucvzerr]])
                bvz_mtch = np.concatenate([bvz_mtch, [star.bvz]])
                bvze_mtch = np.concatenate([bvze_mtch, [star.bvzerr]])
                name_mtch = np.concatenate([name_mtch, [name]])
                uct0_mtch = np.concatenate([uct0_mtch, [star.ucvzt0]])
                bt0_mtch = np.concatenate([bt0_mtch, [star.bvzt0]])
                ucx_mtch = np.concatenate([ucx_mtch, [star.ucx]])
                ucy_mtch = np.concatenate([ucy_mtch, [star.ucy]])
                
        except ValueError,e:
            continue

    hdr = '%10s  %15s  %15s'
    print hdr % ('Name', 'UCLA', 'VLT')
    fmt = '%10s  %5i +- %3i (%6.1f)  %5i +- %3i  (%6.1f)'
    for nn in range(len(ucvz_mtch)):
        print fmt % (name_mtch[nn],
                     ucvz_mtch[nn], ucvze_mtch[nn], uct0_mtch[nn],
                     bvz_mtch[nn], bvze_mtch[nn], bt0_mtch[nn])

    print
    print 'Found %d young stars in common' % len(ucvz_mtch)
    print

    # Which RVs are significantly different?
    dvz = np.abs(ucvz_mtch - bvz_mtch)
    dvze = np.sqrt(ucvze_mtch**2 + bvze_mtch**2)
    outl = np.where((dvz / dvze) > 3.0)[0]
    print 'Found %d outliers:' % len(outl)
    fmt = '%10s  %5.1f  %5.1f'
    for oo in outl:
        print fmt % (name_mtch[oo], dvz[oo], dvze[oo])

    # What is the typical difference in the RV significance?
    dsig = np.median((ucvz_mtch/ucvze_mtch)**2 - (bvz_mtch/bvze_mtch)**2)

    # What is the median delta RV?
    print
    print 'Median difference in RV: %5.3f km/s' % np.median(dvz)
    print
    print 'Median difference in sigma: %5.3f' % dsig

    usetexTrue()
    py.figure(figsize=(6,6))
    py.clf()
    py.subplots_adjust(left=0.15, right=0.96, top=0.95, bottom=0.1)
    py.errorbar(ucvz_mtch, bvz_mtch, xerr=ucvze_mtch, yerr=bvze_mtch, fmt='k.')
    py.plot([-600,300],[-600,300],'k--')
    py.axis([-600,300,-600,300])
    py.xlabel('Keck/OSIRIS RV (km/s)')
    py.ylabel('VLT/SINFONI RV (km/s)')
    py.savefig(plotdir + 'osiris_sinfoni_RVs.png')
    py.close()

    # Plot the differences in RV against the OSIRIS observation date
    py.figure(figsize=(6,6))
    py.clf()
    py.subplots_adjust(left=0.15, right=0.96, top=0.95, bottom=0.1)
    py.plot((uct0_mtch-2000.0), (ucvz_mtch-bvz_mtch), 'k.')
    py.plot([7.5, 10.5], [0, 0], 'k--')
    #py.axis([-600,300,-600,300])
    py.xlabel('OSIRIS Observation Date (yr-2000)')
    py.ylabel('Keck - VLT RV (km/s)')
    py.savefig(plotdir + 'osiris_sinfoni_RVs_obsDate.png')
    py.close()
    
    # Plot the stars in common on top of an AO image
    imgFile = '/u/ghezgroup/data/gc/08maylgs2/combo/mag08maylgs2_dp_msc_kp.fits'
    scale = 0.00993
    sgra = [1398.4, 1500.4]
    img = pyfits.getdata(imgFile)
    imgsize = (img.shape)[0]
    # Make axes for images in arcsec
    pixL = np.arange(0,imgsize)
    xL = [-1*(xpos - sgra[0])*scale for xpos in pixL]
    yL = [(ypos - sgra[1])*scale for ypos in pixL]
    py.figure(figsize=(7,7))
    py.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
    py.clf()
    py.imshow(np.log10(img+1), aspect='equal', interpolation='bicubic',
              extent=[max(xL), min(xL), min(yL), max(yL)],vmin=2.2,vmax=5,
              origin='lowerleft', cmap=py.cm.gray_r)
    py.plot(ucx_mtch, ucy_mtch, 'bo', mfc='None', mec='r', mew=1.5)
    py.plot([0],[0],'kx', ms=7, mew=1.5)
    an = np.linspace(0,2*np.pi,100)
    # mark the border between the inner and middle radial bin:
    py.plot(3.2*np.cos(an), 3.2*np.sin(an),'k--', lw=2) 
    py.axis([12.5, -14.0, -15.0, 12.5])
    py.xlabel('X (arcsec)',fontsize=16)
    py.ylabel('Y (arcsec)',fontsize=16)
    py.savefig(plotdir + 'osiris_sinfoni_inCommon.png')
    py.close()

    usetexFalse()

def yngRVaccel():
    """
    Plots all OSIRIS RV measurements vs. time for young stars.
    """

    yngNames = young.youngStarNames()
    
    dbfile = '/u/ghezgroup/data/gc/database/stars.sqlite'

    # Create a connection to the database file
    connection = sqlite.connect(dbfile)

    # Create a cursor object
    cur = connection.cursor()

    usetexTrue()
    py.clf()
    for yy in range(len(yngNames)):
        date = []
        vz = []
        vze = []

        yngName = str(yngNames[yy])
        cur.execute('SELECT name,ddate,vlsr,vz_err FROM spectra WHERE name=?', [yngName])

        for row in cur:
            date = np.concatenate([date, [row[1]]])
            vz = np.concatenate([vz, [row[2]]])
            vze = np.concatenate([vze, [row[3]]])

        good = np.where((np.array([zz is None for zz in vz]) == False) &
                        (np.array([ee is None for ee in vze]) == False) &
                        (vz != '') & (vze != ''))[0]
        rvCnt = len(good)

        # Some of the data pulled from the database need to be converted to
        # floats in order to plot properly
        date = np.array([date[gg] for gg in good], float)
        vz = np.array([vz[gg] for gg in good], float)
        vze = np.array([vze[gg] for gg in good], float)

        if rvCnt > 0:
            # Define axis limits for plot
            xmin = date.min() - 2.0
            xmax = date.max() + 2.0
            vzmin = vz.argmin()
            vzmax = vz.argmax()
            ymin = vz[vzmin] - 2.0*vze[vzmin]
            ymax = vz[vzmax] + 2.0*vze[vzmax]
        
            py.clf()
            py.figure(figsize=(7,7))
            py.subplots_adjust(left=0.15, right=0.96, top=0.9, bottom=0.1,
                               wspace=0.3, hspace=0.3)
            py.errorbar(date, vz, yerr=vze, fmt='k.')
            thePlot = py.gca()
            thePlot.xaxis.set_major_formatter(py.FormatStrFormatter('%6i'))
            thePlot.get_xaxis().set_major_locator(py.MultipleLocator(2))
            py.xlabel('Date (year)')
            py.ylabel(r'Radial Velocity (km s$^{-1}$)')
            py.title(yngName,fontsize=18)
            py.axis([xmin, xmax, ymin, ymax])
            py.savefig(plotdir + 'rv_vs_time_%s.png' % yngNames[yy])
            py.close()

    usetexFalse()


def plotPosVsTime(mosaic=False):
    """
    Plot positions as a function of time for all the young stars.
    This produces a two-panel plot and is saved to a file called
    plots/posTime_<starname>.png.
    """
    cc = objects.Constants()
    # Load names of young stars in our align
    if mosaic == True:
        # Load up mosaic data as well; select only stars at r>4, since
        # we don't want to add any info from mosaics if we have it in
        # the central 10" already
        yng1 = young.loadYoungStars(root+alnDir,fit=poly,points=points,
                                    silent=True,skipStar=['S5-237']) 
        yng2 = young.loadYoungStars(mscDir,fit=polyM,points=pointsM,
                                    mosaic=True, silent=True)
        # Merge this object with object from central 10" analysis
        yng = merge(yng1, yng2)
    else:
        yng = young.loadYoungStars(root+alnDir,fit=poly,points=points,silent=True)


    names = yng.getArray('name')

    cc.asy_to_kms = dist * cc.cm_in_au / (1.e5 * cc.sec_in_yr)

    # Need to figure out the star with the largest velocity
    # then determine the change in position over 10 years.
    # This will be used as our Y axis range.
    vx = np.abs(yng.getArray('vx') / cc.asy_to_kms)
    vy = np.abs(yng.getArray('vy') / cc.asy_to_kms)
    
    vmaxX = np.max(vx)
    vmaxY = np.max(vy)
    vmax = np.max([vmaxX, vmaxY])
    deltaPos = vmax * (2005.0 - 1996.0)
    extentHalf = deltaPos / 2.0
    # Now round up to the nearest 0.01
    extentHalf = (np.int(extentHalf / 0.01) + 2.0) * 0.01
    print 'Using Axis Range of ', extentHalf * 2.0

    for ii in range(len(names)):
        name = names[ii]

        star = yng.stars[ii]

        # Polyfit results
        t0 = star.fitXa.t0
        x0 = star.fitXa.p
        y0 = star.fitYa.p
        x0e = star.fitXa.perr
        y0e = star.fitYa.perr
        vx = star.fitXa.v / cc.asy_to_kms
        vy = star.fitYa.v / cc.asy_to_kms
        vxe = star.fitXa.verr / cc.asy_to_kms
        vye = star.fitYa.verr / cc.asy_to_kms
        ax = star.fitXa.a
        ay = star.fitYa.a
        axe = star.fitXa.aerr
        aye = star.fitYa.aerr

        print '%2d  %15s  vx = %6.3f  vy = %6.3f' % (ii, name, vx, vy)
        
        # Raw data
        tp = star.years
        xp = star.getArrayAllEpochs('x_pnt')
        yp = star.getArrayAllEpochs('y_pnt')
        xep = star.getArrayAllEpochs('xe_pnt')
        yep = star.getArrayAllEpochs('ye_pnt')

        idx = (np.where(xp != -1000))[0]
        if (len(idx) > 0):
            tp = np.array([tp[ii] for ii in idx], float)
            xp = np.array([xp[ii] for ii in idx], float)
            yp = np.array([yp[ii] for ii in idx], float)
            xep = np.array([xep[ii] for ii in idx], float)
            yep = np.array([yep[ii] for ii in idx], float)
        
        # Convert into absolute coordinate system
        for tt in range(len(tp)):
            (xp[tt], yp[tt], xep[tt], yep[tt]) = \
                     util.rerrPix2Arc(xp[tt], yp[tt], xep[tt], yep[tt],
                                      yng.t, absolute=1, relErr=1)

        # Derive model value at each observed point
        dt = tp - t0
        x = x0 + (vx * dt) + (ax * dt**2 / 2.0)
        y = y0 + (vy * dt) + (ay * dt**2 / 2.0)
        xe = np.sqrt(x0e**2 + (vxe * dt)**2 + (axe * dt**2 / 2.0)**2)
        ye = np.sqrt(y0e**2 + (vye * dt)**2 + (aye * dt**2 / 2.0)**2)

        # We want to plot only the acceleration component. 
        # Subtract off the (x0 + v0 * t)
        xpa = xp - x0 - (vx * dt)
        ypa = yp - y0 - (vy * dt)
        xa = x - x0 - (vx * dt)
        ya = y - y0 - (vy * dt)
        xa_err = np.sqrt((axe * dt**2 / 2.0)**2)
        ya_err = np.sqrt((aye * dt**2 / 2.0)**2)

        # Points
        xpa *= 1000.0
        ypa *= 1000.0
        xpa_err = xep * 1000.0
        ypa_err = yep * 1000.0
        # Model
        xa *= 1000.0
        ya *= 1000.0
        xa_err *= 1000.0
        ya_err *= 1000.0
    
        # Need to determine the limits for plotting
        xcen = np.min(x) + ((np.max(x) - np.min(x)) / 2.0)
        ycen = np.min(y) + ((np.max(y) - np.min(y)) / 2.0)

        xlo = xcen - extentHalf
        xhi = xcen + extentHalf
        ylo = ycen - extentHalf
        yhi = ycen + extentHalf


        py.figure(1)
        py.figure(figsize=(10, 12))

        py.clf()
        py.subplots_adjust(left=0.15, right=0.96, top=0.95, bottom=0.05,
                        wspace=0.3, hspace=0.3)

        ##  X Plot ##
        py.subplot(3, 2, 1)
        py.errorbar(tp, xp, yerr=xep, fmt='k.')
        py.plot(tp, x, 'g-')
        py.plot(tp, x + xe, 'g--')
        py.plot(tp, x - xe, 'g--')
        py.axis([1994, 2011, xlo, xhi])

        # year tick locator
        thePlot = py.gca()
        thePlot.get_xaxis().set_major_locator(py.MultipleLocator(3))

        py.xlabel('Time (year)')
        py.ylabel('Position (arcsec)')
        py.title('X Direction')

        ##  Y Plot ##
        py.subplot(3, 2, 2)
        py.errorbar(tp, yp, yerr=yep, fmt='k.')
        py.plot(tp, y, 'g-')
        py.plot(tp, y + ye, 'g--')
        py.plot(tp, y - ye, 'g--')
        py.axis([1994, 2011, ylo, yhi])

        # year tick locator
        thePlot = py.gca()
        thePlot.get_xaxis().set_major_locator(py.MultipleLocator(3))

        py.xlabel('Time (year)')
        py.ylabel('Position (arcsec)')
        py.title('Y Direction')

        dispName = name.replace('irs', 'IRS ')
        py.text(1994.0 + ((2011. - 1994.)*0.95), ylo + ((yhi - ylo) * 0.93),
             dispName, horizontalalignment='right')

        ##########
        # Residuals after velocity fit
        ##########
        ## X
        py.subplot(3, 2, 3)
        py.errorbar(tp, xpa, yerr=xpa_err, fmt='k.')
        py.plot(tp, xa, 'g-')
        py.plot(tp, xa + xa_err, 'g--')
        py.plot(tp, xa - xa_err, 'g--')
        py.axis([1994, 2011, -8, 8])
    
        # year tick locator
        thePlot = py.gca()
        thePlot.get_xaxis().set_major_locator(py.MultipleLocator(3))
    
        py.xlabel('Time (year)')
        py.ylabel('Residuals after Vel. Fit (mas)')

        ## Y
        py.subplot(3, 2, 4)
        py.errorbar(tp, ypa, yerr=ypa_err, fmt='k.')
        py.plot(tp, ya, 'g-')
        py.plot(tp, ya + ya_err, 'g--')
        py.plot(tp, ya - ya_err, 'g--')
        py.axis([1994, 2011, -8, 8])
    
        # year tick locator
        thePlot = py.gca()
        thePlot.get_xaxis().set_major_locator(py.MultipleLocator(3))
    
        py.xlabel('Time (year)')
        py.ylabel('Residuals after Vel. Fit (mas)')


	##########
	# Residuals after acceleration fit
	##########
        ##  X Plot ##
        py.subplot(3, 2, 5)
        py.errorbar(tp, (xp-x)*1000.0, yerr=xep*1000., fmt='k.')
        py.plot(tp, x-x, 'g-')
        py.plot(tp, xe*1000.0, 'g--')
        py.plot(tp, -xe*1000.0, 'g--')
        py.axis([1994, 2011, -8, 8])

        # year tick locator
        thePlot = py.gca()
        thePlot.get_xaxis().set_major_locator(py.MultipleLocator(3))

        py.xlabel('Time (year)')
        py.ylabel('Residuals after Acc. Fit (mas)')

        ##  Y Plot ##
        py.subplot(3, 2, 6)
        py.errorbar(tp, (yp-y)*1000.0, yerr=yep*1000.0, fmt='k.')
        py.plot(tp, y-y, 'g-')
        py.plot(tp, ye*1000.0, 'g--')
        py.plot(tp, -ye*1000.0, 'g--')
        py.axis([1994, 2011, -8, 8])

        # year tick locator
        thePlot = py.gca()
        thePlot.get_xaxis().set_major_locator(py.MultipleLocator(3))

        py.xlabel('Time (year)')
        py.ylabel('Residuals after Acc. Fit (mas)')

        py.savefig(root + alnDir + 'plots/posTime_%s.png' % name)


def plotPosError():
    """
    Make figure of positional error as a function of
    magnitude.
    """
    alignRoot = 'align/align_d_rms_1000_abs_t'

    s = loadStarsetAbs(root + alignRoot, relErr=1)

    epochCnt = len(s.stars[0].years)

    magStep = 1
    magBins = arange(9, 21, magStep, dtype=float)
    errPerMagA = zeros([epochCnt, len(magBins)], float)
    errPerMagP = zeros([epochCnt, len(magBins)], float)

    radStep = 0.3
    radBins = arange(0, 3, radStep)
    errPerRadA = zeros([epochCnt, len(radBins)], float)
    errPerRadP = zeros([epochCnt, len(radBins)], float)

    x = s.getArray('x')
    y = s.getArray('y')
    r = hypot(x, y)

    # Average errors for ALL epochs
    cntTot = 0.0
    errAvgP = 0.0
    errAvgA = 0.0

    # Loop through epochs
    for ee in range(epochCnt):
        xerr_p = s.getArrayFromEpoch(ee, 'xerr_p')
        yerr_p = s.getArrayFromEpoch(ee, 'yerr_p')
        xerr_a = s.getArrayFromEpoch(ee, 'xerr_a')
        yerr_a = s.getArrayFromEpoch(ee, 'yerr_a')
        mag = s.getArrayFromEpoch(ee, 'mag')

        # Trim out stars without points in this epoch
        idx = (np.where(xerr_p > 0))[0]
        xpos = x[idx]
        ypos = y[idx]
        rad = r[idx]
        xerr_p = xerr_p[idx]
        yerr_p = yerr_p[idx]
        xerr_a = xerr_a[idx]
        yerr_a = yerr_a[idx]
        mag = mag[idx]

        idx = (np.where(rad < 2.0))[0]
        errAvgP += ((xerr_p + yerr_p)[idx]).sum()
        errAvgA += ((xerr_a + yerr_a)[idx]).sum()
        cntTot += 2.0 * len(xerr_p)

        for mm in range(len(magBins)):
            idx = (np.where((mag > magBins[mm]) &
                         (mag <= magBins[mm]+magStep)))[0]

            if (len(idx) > 0):
                xp = xerr_p[idx]
                yp = yerr_p[idx]
                xa = xerr_a[idx]
                ya = yerr_a[idx]

                avgp = (xp + yp) / 2.0
                avga = (xa + ya) / 2.0
                
                errPerMagP[ee, mm] = median(avgp)
                errPerMagA[ee, mm] = median(avga)

        for rr in range(len(radBins)):
            idx = (np.where((rad > radBins[rr]) &
                         (rad <= radBins[rr]+magStep) &
                         (mag < 16)))[0]

            if (len(idx) > 0):
                xp = xerr_p[idx]
                yp = yerr_p[idx]
                xa = xerr_a[idx]
                ya = yerr_a[idx]

                avgp = (xp + yp) / 2.0
                avga = (xa + ya) / 2.0
                
                errPerRadP[ee, rr] = median(avgp)
                errPerRadA[ee, rr] = median(avga)

    errAvgP *= 1000.0 / cntTot
    errAvgA *= 1000.0 / cntTot
    print 'Average Centroid  Error (all epochs): %5.2f mas' % errAvgP
    print 'Average Alignment Error (all epochs): %5.2f mas' % errAvgA

    # Lets get the young star sample, so that we know what their
    # magnitudes are.

    clf()
    subplots_adjust(top=0.95, right=0.95)

    plotEpochs = np.array([1, 9, 28])
    styles = ['k.-', 'bx-', 'rs-']
    lines = []

    magBins += (magStep / 2.0)
    radBins += (radStep / 2.0)
    
    subplot(211)
    for eid in range(len(plotEpochs)):
        ee = plotEpochs[eid]
        foo1 = semilogy(magBins, errPerMagP[ee]*1000.0, styles[eid])
        foo2 = semilogy(magBins, errPerMagA[ee]*1000.0, styles[eid] + '-')
        lines.append(foo1)

    ylim(0.2, 10)
    xlabel('K Magnitude')
    ylabel('Pos. Error (mas)')

    leg = legend(lines, ['Worst Speckle', 'Best Speckle',
                         'LGSAO 2005'],
                 loc='upper left', numpoints=2)
    setp(leg.get_texts(), fontsize='small')

    subplot(212)
    for eid in range(len(plotEpochs)):
        ee = plotEpochs[eid]
        semilogy(radBins, errPerRadP[ee]*1000.0, styles[eid])
        semilogy(radBins, errPerRadA[ee]*1000.0, styles[eid] + '-')
    xlabel('Radius (arcsec)')
    ylabel('Pos. Error (mas) for K < 13')

    ylim(0.2, 10)
    savefig(root + alnDir + 'plots/plotPosError.png')


def plotVelError(mosaic=False):
    """
    Make figure of velocity error as a function of
    magnitude.
    """
    # Load names of young stars in our align
    if mosaic == True:
        # Load up mosaic data as well; select only stars at r>4, since
        # we don't want to add any info from mosaics if we have it in
        # the central 10" already
        yng1 = young.loadYoungStars(root+alnDir,fit=poly,points=points,skipStar=['S5-237'])
        yng2 = young.loadYoungStars(mscDir,fit=polyM,points=pointsM,
                                    mosaic=True)
        # Merge this object with object from central 10" analysis
        yng = merge(yng1, yng2)
    else:
        yng = young.loadYoungStars(root+alnDir,fit=poly,points=points) 

    epochCnt = len(yng.stars[0].years)
    magStep = 1
    magBins = np.arange(9, 21, magStep, dtype=float)
    errPerMag = np.zeros(len(magBins), float)

    radStep = 0.3
    radBins = np.arange(0, 4, radStep)
    errPerRad = np.zeros(len(radBins), float)

    name = yng.getArray('name')
    mag = yng.getArray('mag')
    x = yng.getArray('x')
    y = yng.getArray('y')
    r = np.hypot(x, y)
    vx = yng.getArray('vx')
    vy = yng.getArray('vy')
    vxe = yng.getArray('vxerr')
    vye = yng.getArray('vyerr')

    for mm in range(len(magBins)):
        idx = (np.where((mag > magBins[mm]) &
                     (mag <= magBins[mm]+magStep)))[0]

        if (len(idx) > 0):
            vxem = vxe[idx]
            vyem = vye[idx]

            avg_vem = (vxem + vyem) / 2.0
            
            errPerMag[mm] = np.median(avg_vem)

    for rr in range(len(radBins)):
        idx = (np.where((r > radBins[rr]) &
                     (r <= radBins[rr]+magStep) &
                     (mag < 17)))[0]

        if (len(idx) > 0):
            vxer = vxe[idx]
            vyer = vye[idx]

            avg_ver = (vxer + vyer) / 2.0
            
            errPerRad[rr] = np.median(avg_ver)

    ave_ve = (vxe + vye) / 2.0

    hi = np.where(ave_ve > 5.0)[0]

    print 
    print 'Velocity errors for young stars'
    print 'Range: %5.2f - %5.2f km/s' % (ave_ve.min(), ave_ve.max())
    print 'Average velocity error: %5.2f +- %5.2f km/s' % \
          (ave_ve.mean(), ave_ve.std(ddof=1))
    print 'Median velocity error: %5.2f km/s' % np.median(ave_ve)

    py.clf()
    py.subplots_adjust(top=0.95, right=0.95)

    magBins += (magStep / 2.0)
    radBins += (radStep / 2.0)
    py.plot(magBins, errPerMag, 'k-')
    py.plot(mag, vxe, 'r.',label='X')
    py.plot(mag, vye, 'b.',label='Y')
    for hh in hi:
        py.text(mag[hh]+0.1, ave_ve[hh], name[hh], fontsize=8)
    py.legend(numpoints=1,fancybox=True)
    py.xlabel('K Magnitude')
    py.ylabel('Velocity Error (km/s)')
    py.axis([9,16,0.0,5])
    py.show()
    #py.savefig(root + alnDir + 'plots/plotVelError.png')


def inclOmegVsRad(mosaic=False):
    """
    Plot inclination and omega vs. 2D radius
    """
    cc = objects.Constants()

    pdfdir = root + alnDir + 'aorb_thesis/'
    #pdfdir = root + alnDir + 'aorb_acc_mrPDF_MC_newMosaic/'

    # Load names of young stars in our align
    if mosaic == True:
        # Load up mosaic data as well; select only stars at r>4, since
        # we don't want to add any info from mosaics if we have it in
        # the central 10" already
        yng1 = young.loadYoungStars(root+alnDir,fit=poly,points=points,
                                    withRVonly=True,silent=True,skipStar=['S5-237']) 
        yng2 = young.loadYoungStars(mscDir,fit=polyM,points=pointsM,
                                    mosaic=True, withRVonly=True,silent=True)
        cntrlNames = yng1.getArray('name')
        mscNames = yng2.getArray('name')
        # Merge this object with object from central 10" analysis
        yng = merge(yng1, yng2)
    else:
        yng = young.loadYoungStars(root+alnDir,fit=poly,points=points,
                                   withRVonly=True, silent=True) 

    names = yng.getArray('name')
    r2d = yng.getArray('r2d')
    names.sort()

    starCnt = len(names)
    r = np.zeros(starCnt, float)

    #numTrials = 62000 
    numTrials = 99998

    # TEMPORARY
    # Do not include these problematic sources:
    i90stars = ['S1-1', 'S1-24', 'S2-66', 'S3-2', 'S5-34', 'S5-187', 'S5-235',
                'S5-236', 'S7-180', 'S8-4', 'S8-196', 'S9-9', 'S9-114', 'S9-283',
                'irs13E1', 'irs13E3b', 'irs29N']
    # END TEMPORARY

    r1 = np.where(r2d < 3.5)[0]

    x_all = []
    y_all = []
    vx_all = []
    vy_all = []
    r_all = []
    i_all_pz = []
    o_all_pz = []
    i_all_nz = []
    o_all_nz = []
    ie_all_pz = []
    oe_all_pz = []
    ie_all_nz = []
    oe_all_nz = []
    for ii in range(starCnt):
        star = yng.stars[ii]
        name = star.name
        mscStar = False

        #if name in i90stars:
        #    continue
        
        if (name in mscNames) & (name not in cntrlNames):
            mscStar = True
            x = star.fitXv.p
            y = star.fitYv.p
            r2d = np.hypot(x, y)
            xe = star.fitXv.perr
            ye = star.fitYv.perr
            vx = star.fitXv.v
            vy = star.fitYv.v
            vxe = star.fitXv.verr
            vye = star.fitYv.verr
        else:
            x = star.fitXa.p
            y = star.fitYa.p
            r2d = np.hypot(x, y)
            xe = star.fitXa.perr
            ye = star.fitYa.perr
            vx = star.fitXa.v
            vy = star.fitYa.v
            vxe = star.fitXa.verr
            vye = star.fitYa.verr

        # File contains analytic orbit solutions with acceleration limits (MC)
        pdffile = '%s%s.mc.dat' % (pdfdir, name)
        if os.path.exists(pdffile):
            pdf = pickle.load(open(pdffile))
            print name

            #x_all = np.concatenate([x_all, [np.abs(x)]])
            #y_all = np.concatenate([y_all, [np.abs(y)]])
            x_all = np.concatenate([x_all, [x]])
            y_all = np.concatenate([y_all, [y]])
            r_all = np.concatenate([r_all, [r2d]])
            vx_all = np.concatenate([vx_all, [vx]])
            vy_all = np.concatenate([vy_all, [vy]])

            # The first half of the MC trials are for positive z
            pz = numTrials / 2
            # Mean of PDFs
            i_all_pz = np.concatenate([i_all_pz, [pdf.i[0:pz].mean()]])
            i_all_nz = np.concatenate([i_all_nz, [pdf.i[pz:].mean()]])
            o_all_pz = np.concatenate([o_all_pz, [pdf.o[0:pz].mean()]]) 
            o_all_nz = np.concatenate([o_all_nz, [pdf.o[pz:].mean()]])
            # Standard deviation
            ie_all_pz = np.concatenate([ie_all_pz, [pdf.i[0:pz].std(ddof=1)]])
            ie_all_nz = np.concatenate([ie_all_nz, [pdf.i[pz:].std(ddof=1)]])
            oe_all_pz = np.concatenate([oe_all_pz, [pdf.o[0:pz].std(ddof=1)]]) 
            oe_all_nz = np.concatenate([oe_all_nz, [pdf.o[pz:].std(ddof=1)]])


    usetexTrue()
    py.figure(1)
    py.figure(figsize=(10,10))
    py.clf()
    py.subplots_adjust(left=0.1, right=0.95, top=0.95,
                        wspace=0.3, hspace=0.3)
    py.subplot(2,2,1)
    py.errorbar(x_all, i_all_nz, fmt='r.', yerr=ie_all_nz)
    py.errorbar(x_all, i_all_pz, fmt='b.', yerr=ie_all_pz)
    #py.axis([0,7,0,180])
    py.xlabel('Delta RA (arcsec)', fontsize=14)
    py.ylabel('Average Inclination (degrees)', fontsize=14)
    py.subplot(2,2,2)
    py.errorbar(y_all, i_all_nz, fmt='r.', yerr=ie_all_nz)
    py.errorbar(y_all, i_all_pz, fmt='b.', yerr=ie_all_pz)
    #py.axis([0,7,0,180])
    py.xlabel('Delta Dec (arcsec)', fontsize=14)
    py.ylabel('Average Inclination (degrees)', fontsize=14)
    py.subplot(2,2,3)
    py.errorbar(x_all, o_all_nz, fmt='r.', yerr=oe_all_nz)
    py.errorbar(x_all, o_all_pz, fmt='b.', yerr=oe_all_pz)
    #py.axis([0,7,0,180])
    py.xlabel('Delta RA (arcsec)', fontsize=14)
    py.ylabel(r'Average $\Omega$ (degrees)', fontsize=14)
    py.subplot(2,2,4)
    py.errorbar(y_all, o_all_nz, fmt='r.', yerr=oe_all_nz, label='-z')
    py.errorbar(y_all, o_all_pz, fmt='b.', yerr=oe_all_pz, label='+z')
    #py.axis([0,7,0,180])
    #py.legend(numpoints=1,fancybox=True)
    py.xlabel('Delta Dec (arcsec)', fontsize=14)
    py.ylabel(r'Average $\Omega$ (degrees)', fontsize=14)
    py.savefig(plotdir + 'avg_inclOmeg_vs_XY.png')
    py.close(1)

    py.figure(2)
    py.figure(figsize=(8,5))
    py.clf()
    py.subplots_adjust(left=0.1, right=0.95, top=0.95,
                        wspace=0.3, hspace=0.3)
    py.subplot(1,2,1)
    py.errorbar(r_all, i_all_nz, fmt='r.', yerr=ie_all_nz)
    py.errorbar(r_all, i_all_pz, fmt='b.', yerr=ie_all_pz)
    #py.axis([0,7,0,180])
    py.xlabel('Projected Radius (arcsec)', fontsize=14)
    py.ylabel('Average Inclination (degrees)', fontsize=14)
    py.subplot(1,2,2)
    py.errorbar(r_all, o_all_nz, fmt='r.', yerr=oe_all_nz)
    py.errorbar(r_all, o_all_pz, fmt='b.', yerr=oe_all_pz)
    #py.axis([0,7,0,180])
    py.xlabel('Projected Radius (arcsec)', fontsize=14)
    py.ylabel(r'Average $\Omega$ (degrees)', fontsize=14)
    py.savefig(plotdir + 'avg_inclOmeg_vs_r2d.png')
    py.close(2)

    import matplotlib.font_manager
    prop = matplotlib.font_manager.FontProperties(size=10)
    py.figure(3)
    py.figure(figsize=(10,10))
    py.clf()
    py.subplots_adjust(left=0.1, right=0.95, top=0.95,
                        wspace=0.3, hspace=0.3)
    py.subplot(2,2,1)
    py.plot(vx_all, i_all_nz, 'r.')
    py.plot(vx_all, i_all_pz, 'b.')
    #py.axis([0,7,0,180])
    py.xlabel('Vx (mas/yr)', fontsize=14)
    py.ylabel('Average Inclination (degrees)', fontsize=14)
    py.subplot(2,2,2)
    py.plot(vy_all, i_all_nz, 'r.')
    py.plot(vy_all, i_all_pz, 'b.')
    #py.axis([0,7,0,180])
    py.xlabel('Vy (mas/yr)', fontsize=14)
    py.ylabel('Average Inclination (degrees)', fontsize=14)
    py.subplot(2,2,3)
    py.plot(vx_all, o_all_nz, 'r.')
    py.plot(vx_all, o_all_pz, 'b.')
    #py.axis([0,7,0,180])
    py.xlabel('Vx (mas/yr)', fontsize=14)
    py.ylabel(r'Average $\Omega$ (degrees)', fontsize=14)
    py.subplot(2,2,4)
    py.plot(vy_all, o_all_nz, 'r.', label='-z')
    py.plot(vy_all, o_all_pz, 'b.', label='+z')
    #py.axis([0,7,0,180])
    py.legend(numpoints=1,fancybox=True,loc=4,prop=prop)
    py.xlabel('Vy (mas/yr)', fontsize=14)
    py.ylabel(r'Average $\Omega$ (degrees)', fontsize=14)
    py.savefig(plotdir + 'avg_inclOmeg_vs_Vxy.png')
    py.close(3)

    py.figure(4)
    py.figure(figsize=(10,5))
    py.clf()
    py.subplots_adjust(left=0.1, right=0.95, top=0.9,
                        wspace=0.3, hspace=0.3)
    py.subplot(1,2,1)
    py.errorbar(o_all_nz, i_all_nz, fmt='r.', xerr=oe_all_nz, yerr=ie_all_nz)
    py.xlabel(r'Average $\Omega$ (degrees)', fontsize=14)
    py.ylabel('Average Inclination (degrees)', fontsize=14)
    py.title('Negative z')
    py.subplot(1,2,2)
    py.errorbar(o_all_pz, i_all_pz, fmt='b.', xerr=oe_all_pz, yerr=ie_all_pz)
    #py.axis([0,7,0,180])
    py.xlabel(r'Average $\Omega$ (degrees)', fontsize=14)
    py.ylabel('Average Inclination (degrees)', fontsize=14)
    py.title('Positive z')
    py.savefig(plotdir + 'avg_incl_vs_Omega.png')
    py.close(4)
    usetexFalse()

def plotInclToDisk(mosaic=True,suffix='_mosaic'):
    """
    DO NOT USE. Use yngDiskInclination() function above instead.

    
    Plot the inclination with respect to the disk for all stars and all
    solutions. Also plots the minimum angular distance to the disk.
    """
    cc = objects.Constants()
    # Load up directory names
    orbDir = 'aorb_thesis/'
    #orbDir = 'aorb_acc_mrPDF_MC_newMosaic/'
    
    # Load disk star names and probability of disk membership
    (names, diskP) = readDiskProb(suffix=suffix, diskOnly=False)

    # Load kinematic data for disk stars
    yng = loadYoungByName(names, mosaic=mosaic)
    x = yng.getArray('x')
    y = yng.getArray('y')
    yngRadius = np.sqrt(x**2 + y**2)

    # Disk solution
    irad = np.radians(idisk)
    orad = np.radians(odisk)
    sinip = np.sin(np.radians(idisk))
    cosip = np.cos(np.radians(idisk))

    #####
    # Get angles for all other young stars
    #####
    # Setup i/Omega for each pixel on sky
    nside = 64
    npix = healpy.nside2npix(nside)
    (iheal, oheal) = healpy.pix2ang(nside, np.arange(0, npix))
    iheal *= rad2deg
    oheal *= rad2deg
    sini = np.sin(iheal / rad2deg)
    cosi = np.cos(iheal / rad2deg)

    # Determine angular offset to disk for every other point on the sky
    cosodiff = np.cos( (oheal - odisk) / rad2deg )
    angOffCW = np.arccos( (sini * sinip * cosodiff) + (cosi * cosip) )
    angOffCW *= rad2deg

    # Store angles in array
    yngAngle = np.zeros(len(names), float)
    yngAngleErr = np.zeros(len(names), float)

    numTrials = 99998
    starCnt = len(names)
    angle = np.zeros((starCnt, numTrials), float)
    minAngle = np.zeros(starCnt, float)
    maxAngle = np.zeros(starCnt, float)

    # Loop through
    for ii in range(len(names)):
        # File contains analytic orbit solutions with acceleration limits (MC)
        pdffile = '%s%s%s%s.mc.dat' % (root, alnDir, orbDir, names[ii])
        pdf = pickle.load(open(pdffile))

        # Determine angular offset to disk for each solution
        sini = np.sin(pdf.i * np.pi / 180.0)
        cosi = np.cos(pdf.i * np.pi / 180.0)
        cosodiff = np.cos( (pdf.o - odisk) * np.pi / 180.0 )
        angle[ii,:] = np.arccos( (sini * sinip * cosodiff) + (cosi * cosip) )
        angle[ii,:] *= 180.0 / np.pi

        # Select only the most probable of the two
        # degenerate solutions.
        #idx = (np.where((pdf > level) & (angle < 30.0)))[0]
        #idx = (np.where((pdf > level) & (angle < 45.0)))[0]
            
        # Find the range in angles... calc uncertainty from it.
        minAngle[ii] = angle[ii].min()
        maxAngle[ii] = angle[ii].max()

        #pdb.set_trace()


    print 'Plotting angles w.r.t. disk direction'
    onD = (np.where(diskP >= 2.7e-3))[0]
    offD = (np.where(diskP < 2.7e-3))[0]

    #pdb.set_trace()
    import matplotlib.font_manager
    prop = matplotlib.font_manager.FontProperties(size=14)

    usetexTrue()
    # Plot a histogram of the minimum angular offset
    binsIn = np.arange(0, 185, 5)
    fig = py.figure(figsize=(6,6))
    fig.subplots_adjust(left=0.15, right=0.96, top=0.95)
    fig.clf()
    ax = fig.add_subplot(111)
    ax.hist(minAngle[onD], bins=binsIn, histtype='step',color='r',
            linewidth=2,label='Disk')
    ax.hist(minAngle[offD], bins=binsIn, histtype='step',color='k',
            linewidth=2,ls='dashed',label='Non-Disk')
    ax.set_xlabel('Minimum Angular Distance to Disk (deg)')
    ax.set_ylabel('N')
    ax.legend(numpoints=1,prop=prop,fancybox=True)
    ax.axis([0, 160, 0, 25])
    fig.savefig(plotdir + 'yng_minAngle_off_disk_hist.png')
    fig.savefig(plotdir + 'eps/yng_minAngle_off_disk_hist.eps')
    py.close()

    # Plot a histogram of all angular offsets
    binsIn = np.arange(0, 181, 1)
    fig = py.figure(figsize=(6,6))
    fig.subplots_adjust(left=0.15, right=0.96, top=0.95)
    fig.clf()
    ax = fig.add_subplot(111)
    ax.hist(angle.flatten(), bins=binsIn, histtype='step',color='k',
            linewidth=2,normed=True,label='Disk')
    #ax.hist(minAngle[offD], bins=binsIn, histtype='step',color='k',
    #        linewidth=2,ls='dashed',label='Non-Disk')
    ax.set_xlabel('Angular Distance to Disk (deg)')
    ax.set_ylabel('N')
    #ax.legend(numpoints=1,prop=prop,fancybox=True)
    #ax.axis([0, 160, 0, 25])
    fig.savefig(plotdir + 'yng_allAngles_off_disk_hist.png')
    fig.savefig(plotdir + 'eps/yng_allAngles_off_disk_hist.eps')
    py.close()
    usetexFalse()


def plot_osiris_fields(suffix=''):
    """
    Plot OSIRIS fields observed on top of a deep mosaic (30 arcsec) map.
    Mark the location of known young stars
    """
    # Load up mosaic data as well; select only stars at r>4, since
    # we don't want to add any info from mosaics if we have it in
    # the central 10" already
    yng1 = young.loadYoungStars(root+alnDir,fit=poly,points=points,
                                radiusCut=0.8, silent=True,
                                withRVonly=True,skipStar=['S5-237']) 
    yng2 = young.loadYoungStars(mscDir,fit=polyM,points=pointsM,
                                mosaic=True, silent=True, radiusCut=0.8,
                                withRVonly=True)
    # Merge this object with object from central 10" analysis
    yng = merge(yng1, yng2)
#    yng = young.loadYoungStars(root+alnDir,fit=poly,points=points,
#                               radiusCut=0.0, silent=True)

    rv_ref = np.array(yng.getArray('rv_ref'))
    rvt0 = np.array(yng.getArray('vzt0'))
    names = yng.getArray('name')
    x = yng.getArray('x')
    y = yng.getArray('y')
    vx = yng.getArray('vx')
    vy = yng.getArray('vy')
    r2d = np.hypot(x,y)

    imgFile = '/u/ghezgroup/data/gc/08maylgs2/combo/mag08maylgs2_dp_msc_kp.fits'
    scale = 0.00993
    sgra = [1398.4, 1500.4]
    #sgra = [627.,730.]
    img = pyfits.getdata(imgFile)
    imgsize = (img.shape)[0]
    # Make axes for images in arcsec
    pixL = np.arange(0,imgsize)
    xL = [-1*(xpos - sgra[0])*scale for xpos in pixL]
    yL = [(ypos - sgra[1])*scale for ypos in pixL]

    # Read in vertices of the OSIRIS outlines
    dbfile = '/u/ghezgroup/data/gc/database/stars.sqlite'
    # Create a connection to the database file
    connection = sqlite.connect(dbfile)

    # Create a cursor object
    cur = connection.cursor()

    cur.execute('SELECT * FROM fields')
    fx_all = []
    fy_all = []
    for row in cur:
        fld = row[0]
        if ('Imaging' in fld) | ('Verification' in fld):
            continue
        fx = row[12].split(',')
        fy = row[13].split(',')
        fx = np.concatenate([fx[0:4],[fx[0]]]) # repeat the first vertex to complete the box
        fy = np.concatenate([fy[0:4],[fy[0]]]) # repeat the first vertex to complete the box
        fx = [np.float(ii) for ii in fx]
        fy = [np.float(ii) for ii in fy]
        fx_all = np.concatenate([fx_all,fx])
        fy_all = np.concatenate([fy_all,fy])

    # Also plot up the fields covered by Bartko that include the RVs I use
    #b09 = asciidata.open('/u/ghezgroup/data/gc/source_list/bartko_sinfoni_outline.dat')
    b09 = asciidata.open(root+alnDir+'tables/bartko09_sinfoni_outline.dat')
    bx = b09[0].tonumpy()
    by = b09[1].tonumpy()

    import matplotlib.font_manager
    prop = matplotlib.font_manager.FontProperties(size=14)

    uc = np.where(rv_ref == 'UCLA')[0]
    bart = np.where((rv_ref == 'Bartko+2009') | (rv_ref == 'Bartko+2009 Paumard+2006'))[0]
    paum = np.where(rv_ref == 'Paumard+2006')[0]
    no = np.where((np.array([rr is None for rr in rv_ref]) == True))[0]
    py.figure(1)
    py.figure(figsize=(7,7))
    py.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
    py.clf()
    py.imshow(np.log10(img+1), aspect='equal', interpolation='bicubic',
              extent=[max(xL), min(xL), min(yL), max(yL)],vmin=2.2,vmax=5,
              origin='lowerleft', cmap=py.cm.gray_r)
    py.plot(x[uc], y[uc], 'bo', mfc='None', mec='r', mew=1.5, label='OSIRIS')
    py.plot(x[bart], y[bart], 'ro', mfc='None', mec='b', mew=1.5, label='Bartko+09')
    py.plot(x[paum], y[paum], 'go', mfc='None', mec='g', mew=1.5, label='Paumard+06')
    #py.plot(x[no], y[no], 'ko', mfc='None', mec='k', mew=1.5, label='No RV')
    py.plot([0],[0],'kx', ms=7, mew=1.5)
    for ii in range(0,len(fx_all/5),5):
        py.plot(fx_all[ii:ii+5],fy_all[ii:ii+5],'k-')
    py.plot([5,-5,-5,5,5], [-4,-4,6,6,-4], 'k--', lw=2)
    py.axis([12.5, -14.0, -15.0, 12.5])
    py.legend(numpoints=1, fancybox=True, loc=4, prop=prop)
    py.xlabel('X (arcsec)',fontsize=16)
    py.ylabel('Y (arcsec)',fontsize=16)
    #py.title('OSIRIS and Deep Mosaic Obs w/ Young Stars',fontsize=16)
    py.savefig(plotdir + 'osiris_fields_yngstars' + suffix + '.png')
    py.savefig(plotdir + 'eps/osiris_fields_yngstars' + suffix + '.eps')
    py.close(1)

    # Plot up the velocity vectors on a mosaic map
    ave_vtot = (np.hypot(vx, vy)).mean()

    # Exclude central arcsecond stars
    ridx = np.where(r2d > 0.8)[0]
    py.figure(2)
    py.subplots_adjust(left=0.1, right=0.98, top=0.95, bottom=0.1)
    py.clf()
    py.figure(figsize=(8,8))
    py.imshow(np.log10(img+1), aspect='equal', interpolation='bicubic',
              extent=[max(xL), min(xL), min(yL), max(yL)],vmin=2.2,vmax=5,
              origin='lowerleft', cmap=py.cm.gray_r)
    py.quiver([x[ridx]], [y[ridx]], [vx[ridx]], [vy[ridx]], headwidth=2,
              color='black', units='y', angles='xy', scale=200)
    py.plot([0],[0],'rx')
    py.axis([12.5, -14.0, -15.0, 12.5])
    py.xlabel('X (arcsec)',fontsize=16)
    py.ylabel('Y (arcsec)',fontsize=16)
    py.title('Young Star Velocities',fontsize=16)
    py.savefig(plotdir + 'velVector_yngstars' + suffix + '.png')
    py.close(2)

    py.figure(3)
    py.figure(figsize=(7,7))
    py.subplots_adjust(left=0.13, right=0.95, top=0.95, bottom=0.1)
    py.clf()
    py.imshow(np.log10(img+1), aspect='equal', interpolation='bicubic',
              extent=[max(xL), min(xL), min(yL), max(yL)],vmin=2.2,vmax=5,
              origin='lowerleft', cmap=py.cm.gray_r)
    py.plot(x[uc], y[uc], 'bo', mfc='None', mec='r', mew=1.5, label='OSIRIS')
    py.plot(x[bart], y[bart], 'ro', mfc='None', mec='b', mew=1.5, label='Bartko+09')
    py.plot(x[paum], y[paum], 'go', mfc='None', mec='g', mew=1.5, label='Paumard+06')
    py.plot([0],[0],'kx', ms=7, mew=1.5)
    for ii in range(0,len(fx_all/5),5):
        py.plot(fx_all[ii:ii+5],fy_all[ii:ii+5],'k-')
    py.plot([5,-5,-5,5,5], [-4,-4,6,6,-4], 'k--', lw=2)
    py.plot(bx,by,'b-')
    an = np.linspace(0,2*np.pi,100)
    py.plot(3.2*np.cos(an),3.2*np.sin(an),'k--', lw=2) 
    py.plot(6.5*np.cos(an),6.5*np.sin(an),'k--', lw=2) 
    #py.plot(0.8*np.cos(an),0.8*np.sin(an),'k--', lw=2) # Bartko's circles
    #py.plot(3.5*np.cos(an),3.5*np.sin(an),'k--', lw=2) 
    #py.plot(7.0*np.cos(an),7.0*np.sin(an),'k--', lw=2) 
    #py.plot(12.0*np.cos(an),12.0*np.sin(an),'k--', lw=2) 
    py.axis([12.5, -14.0, -15.0, 12.5])
    py.legend(numpoints=1, fancybox=True, loc=4, prop=prop)
    py.xlabel('X (arcsec)',fontsize=16)
    py.ylabel('Y (arcsec)',fontsize=16)
    py.savefig(plotdir + 'sinfoni_fields_yngstars' + suffix + '.png')
    py.close(3)

    # Figure out what field each of these sources are in for OSIRIS data:
    cur.execute('SELECT * FROM spectra')
    stars_all = []
    fld_all = []
    full_date = []
    dec_date = []
    for row in cur:
        star = row[0]
        date = row[1]
        ddate = row[2]
        fld = row[17]
        stars_all = np.concatenate([stars_all, [star]])
        fld_all = np.concatenate([fld_all, [fld]])
        dec_date = np.concatenate([dec_date, [ddate]])
        full_date = np.concatenate([full_date, [date]])

    # Print out some info 
    print 'UC RVs:'
    for ii in uc:
        idx = np.where(np.array(stars_all) == names[ii])[0]
        infld = fld_all[idx[0]]
        # Figure out the actual date for this rvt0
        foo = np.where(np.array(dec_date) == rvt0[ii])[0]
        date = full_date[foo[0]]
        print '%10s  %4s  %12s' % (names[ii], infld, date)
    print
    print 'B09 RVs:'
    for ii in bart:
        idx = np.where(np.array(stars_all) == names[ii])[0]
        if len(idx) > 0:
            infld = fld_all[idx[0]]
            print '%10s  %4s' % (names[ii], infld)
        else:
            print '%10s    --' % (names[ii])
    print
    print 'P06 RVs:'
    for ii in paum:
        idx = np.where(np.array(stars_all) == names[ii])[0]
        if len(idx) > 0:
            infld = fld_all[idx[0]]
            print '%10s  %4s' % (names[ii], infld)
        else:
            print '%10s    --' % (names[ii])
    


def plot_velocity_r2d(align='align/align_d_rms_1000_abs_t'):

    outdir = root + alnDir + 'plots/'
    yng = young.loadYoungStars(root+alnDir,fit=poly,points=points,radiusCut=0.8,
                               skipStar=['S5-237']) 

    name = yng.getArray('name')
    mag = yng.getArray('mag')
    x = yng.getArray('fitXv.p')
    y = yng.getArray('fitYv.p')
    vx = yng.getArray('fitXv.v')
    vy = yng.getArray('fitYv.v')
    vxe = yng.getArray('fitXv.verr')
    vye = yng.getArray('fitYv.verr')
    vx *= 1e3
    vy *= 1e3
    vxe *= 1e3
    vye *= 1e3
    vtote = np.sqrt(((vx*vxe)**2+(vy*vye)**2)/(vx**2+vy**2))

    # temp
    cnt = yng.getArray('velCnt')
    idx = np.where(cnt > 20)[0]
    print 'Ratio of X to Y proper motion errors: %4.3f' % (vxe[idx] / vye[idx]).mean()

    py.clf()
    py.figure(figsize=(12,5))
    py.subplots_adjust(wspace=0.25, hspace=0.25, left=0.08, right=0.95, top=0.95)
    py.subplot(1,3,1)
    py.errorbar(np.hypot(x,y), vx, yerr=vxe, fmt='r.')
    py.plot([0,7],[0,0],'k--')
    py.xlabel('Projected Radius (arcsec)')
    py.ylabel('X Velocity (mas/yr)')
    py.subplot(1,3,2)
    py.errorbar(np.hypot(x,y), vy, yerr=vye, fmt='b.')
    py.plot([0,7],[0,0],'k--')
    py.xlabel('Projected Radius (arcsec)')
    py.ylabel('Y Velocity (mas/yr)')
    py.subplot(1,3,3)
    py.errorbar(np.hypot(x,y), np.hypot(vx,vy), yerr=vtote, fmt='k.')
    py.xlabel('Projected Radius (arcsec)')
    py.ylabel('Proper Motion (mas/yr)')
    py.savefig(outdir + 'pm_vs_r2d_yngstars.png')


def compare_vels_VelVsAccelFit(align='align/align_d_rms_1000_abs_t'):
    """
    Compares velocities from polyfit's velocity fit and acceleration fit.
    """

    rootDir = root + alnDir
    
    s = starset.StarSet(rootDir + align)
    s.loadPolyfit(rootDir + poly, accel=0, arcsec=1)
    s.loadPolyfit(rootDir + poly, accel=1, arcsec=1)

    names = np.array(s.getArray('name'))
    x = s.getArray('x')
    y = s.getArray('y')
    r = np.hypot(x, y)

    vx_vel = s.getArray('fitXv.v')
    vy_vel = s.getArray('fitYv.v')
    vxe_vel = s.getArray('fitXv.verr')
    vye_vel = s.getArray('fitYv.verr')
    t0_vel = s.getArray('fitXv.t0')

    vx_acc = s.getArray('fitXa.v')
    vy_acc = s.getArray('fitYa.v')
    vxe_acc = s.getArray('fitXa.verr')
    vye_acc = s.getArray('fitYa.verr')
    t0_acc = s.getArray('fitXa.t0')
    
    # Calculate the residuals
    diffx = vx_vel - vx_acc
    diffxErr = np.sqrt(vxe_vel**2 + vxe_acc**2)
    
    diffy = vy_vel - vy_acc
    diffyErr = np.sqrt(vye_vel**2 + vye_acc**2)
    
    diff = np.hypot(diffx, diffy)
    diffErr = np.sqrt((diffx*diffxErr)**2 + (diffy*diffyErr)**2) / diff

    yngNames = young.youngStarNames()
    
    yng = []
    print '** Young stars with large velocity discrepancies: **'
    print '%15s  %5s  %5s  %5s  ' % ('Name', 'Sigma', 'X', 'Y')
    for ii in range(len(yngNames)):
        yngName = str(yngNames[ii])
        yy = np.where(names == yngName)[0]

        if len(yy) > 0:
            yng.append(yy)

            if ((diff[yy]/diffErr[yy]) > 2.0):   
                print '%15s  %5.1f  %5.2f  %5.2f ' % \
                      (names[yy], diff[yy]/diffErr[yy], x[yy], y[yy])

    # Plot X and Y seperatly
    py.clf()
    py.figure(figsize=(10,5))
    py.subplots_adjust(wspace=0.25, hspace=0.25, left=0.1, right=0.95, top=0.95)
    py.subplot(1, 2, 1)
    py.errorbar(vx_vel*1.e3, vx_acc*1.e3, xerr=vxe_vel*1.e3, yerr=vxe_acc*1.e3, fmt='k.')
    rng = py.axis()
    py.plot(rng[0:2], rng[0:2], 'b--')
    py.xlabel('Vx Velocity Fit (mas/yr)')
    py.ylabel('Vx Acceleration Fit (mas/yr)')

    py.subplot(1, 2, 2)
    py.errorbar(vy_vel*1.e3, vy_acc*1.e3, xerr=vye_vel*1.e3, yerr=vye_acc*1.e3, fmt='k.')
    rng = py.axis()
    py.plot(rng[0:2], rng[0:2], 'b--')
    py.xlabel('Vy Velocity Fit (mas/yr)')
    py.ylabel('Vy Acceleration Fit (mas/yr)')
    py.savefig(plotdir + 'compare_velocity_fits.png')

    # Plot X and Y seperatly for young stars only
    py.clf()
    py.figure(figsize=(10,10))
    py.subplots_adjust(wspace=0.25, hspace=0.25, left=0.1, right=0.95, top=0.95)
    py.subplot(2, 2, 1)
    for yy in yng:
        fmt = 'k.'
        if r[yy] < 0.8:
            fmt = 'r.'
        py.errorbar(vx_vel[yy]*1.e3, vx_acc[yy]*1.e3, \
                    xerr=vxe_vel[yy]*1.e3, yerr=vxe_acc[yy]*1.e3, fmt=fmt)
    rng = py.axis()
    py.plot(rng[0:2], rng[0:2], 'b--')
    py.xlabel('Vx Velocity Fit (mas/yr)')
    py.ylabel('Vx Acceleration Fit (mas/yr)')

    py.subplot(2, 2, 2)
    for yy in yng:
        fmt = 'k.'
        if r[yy] < 0.8:
            fmt = 'r.'
        py.errorbar(vy_vel[yy]*1.e3, vy_acc[yy]*1.e3, \
                    xerr=vye_vel[yy]*1.e3, yerr=vye_acc[yy]*1.e3, fmt=fmt)
    rng = py.axis()
    py.plot(rng[0:2], rng[0:2], 'b--')
    py.xlabel('Vy Velocity Fit (mas/yr)')
    py.ylabel('Vy Acceleration Fit (mas/yr)')

    py.subplot(2, 2, 3)
    for yy in yng:
        py.plot(r[yy], diff[yy]/diffErr[yy], 'k.')
    py.xlabel('Projected Radius (arcsec)')
    py.ylabel('Velocity Difference (sigma)')

    py.savefig(plotdir + 'compare_velocity_fits_yng.png')


def plot_h_statistic():
    """
    Plots the h statistic from Madigan & Levin (2011).
    """

    cc = objects.Constants()

    outdir = root + alnDir + 'plots/'
    yng1 = young.loadYoungStars(root+alnDir,fit=poly,points=points,
                                radiusCut=0.0, silent=True,skipStar=['S5-237']) 
    yng2 = young.loadYoungStars(mscDir,fit=polyM,points=pointsM,
                                mosaic=True, silent=True, radiusCut=0.0)
    # Merge this object with object from central 10" analysis
    yng = merge(yng1, yng2)

    name = yng.getArray('name')
    x = yng.getArray('fitXv.p') # arcsec
    y = yng.getArray('fitYv.p') # arcsec
    r2d = np.hypot(x,y) # arcsec
    vx = yng.getArray('vx') # km/s
    vy = yng.getArray('vy') # km/s
    vxe = yng.getArray('vxerr') # km/s
    vye = yng.getArray('vyerr') # km/s

    vx *= 1.e5 # cm/s
    vy *= 1.e5 # cm/s
    vxe *= 1.e5 # cm/s
    vye *= 1.e5 # cm/s
    vtot = np.hypot(vx, vy)
    vtote = np.sqrt(((vx*vxe)**2+(vy*vye)**2)/(vx**2+vy**2))

    x_cgs = x * dist * cc.cm_in_au
    y_cgs = y * dist * cc.cm_in_au
    r2d_cgs = r2d * dist * cc.cm_in_au

    GM = cc.G * mass * cc.msun # cgs
    # h statistic (Madigan & Levin 2011)
    hs = (x_cgs*vy - y_cgs*vx) / np.sqrt(GM*r2d_cgs)

    # normalized angular momentum
    j = (x_cgs*vy - y_cgs*vx) / (r2d_cgs*vtot)

    # CDF of the h statistic
    hidx = np.abs(hs).argsort()
    hs_sort = np.abs(hs)[hidx]
    cdf = np.cumsum(hs_sort)

    # Cumulative fraction of h statistic from Ann-Marie
    amFile = asciidata.open(root + alnDir + 'tables/madigan2011_cum_h_stat.dat')
    h_am = amFile[0].tonumpy()
    cf_am = amFile[1].tonumpy()

    # Which stars have acceleration detections?
    _acc = asciidata.open(root + alnDir + 'tables/accelerating_sources.dat')
    acc = _acc[0].tonumpy()
    acc = [aa.strip() for aa in acc]

    print 'Number of stars included: %i' % len(j)
    print
    s014 = np.where(np.array(name) == 'S0-14')[0]
    print 'S0-14, h = %6.3f' % hs[s014]

    # Find stars with h near zero
    hz = np.where(np.abs(hs) < 0.05)[0]
    print
    print 'Stars with |h| < 0.05:'
    for hh in hz:
        print '%10s, h = %6.3f' % (name[hh], hs[hh])

    # Find stars with |h| > 1.0
    hhi = np.where(np.abs(hs) > 1.0)[0]
    print
    print 'Stars with |h| > 1.0:'
    if len(hhi) > 0:
        for hh in hhi:
            print '%10s, h = %6.3f' % (name[hh], hs[hh])
    else:
        print 'None'

    # Plot h statistic
    py.clf()
    py.figure(figsize=(6,6))
    py.subplots_adjust(wspace=0.25, hspace=0.25, left=0.15, right=0.95, top=0.95)
    for ii in range(len(hs)):
        if name[ii] in acc:
            py.semilogx(r2d[ii], hs[ii], 'r.') # accelerating sources
        else:
            py.semilogx(r2d[ii], hs[ii], 'k.')
    py.semilogx([1e-5,15],[1e-5,1e-5],'k--')
    py.semilogx([0.8,0.8],[-1.5,1.5],'k--')
    py.axis([1e-1,15,-1.5,1.5])
    py.xlabel('Projected Radius (arcsec)')
    py.ylabel('h statistic')
    py.savefig(outdir + 'h_statistic_r2d.png')
    py.close()

    # Plot h vs. e for the accelerating sources 
    # NOT READY - need to run disk membership and peakPos functions to get new disk soln
    #py.clf()
    #py.figure(figsize=(6,6))
    #py.subplots_adjust(wspace=0.25, hspace=0.25, left=0.15, right=0.95, top=0.95)
    #py.plot(ecc, hs_acc, 'k.')
    #py.xlabel('Eccenctricity')
    #py.ylabel('h statistic')
    #py.savefig(outdir + 'h_vs_e_accelerators.png')
    #py.close()

    # Plot normalized angular momentum vector
    py.clf()
    py.figure(figsize=(6,6))
    py.subplots_adjust(wspace=0.25, hspace=0.25, left=0.15, right=0.95, top=0.95)
    py.semilogx(r2d, j, 'k.')
    py.semilogx([1e-5,15],[1e-5,1e-5],'k--')
    py.axis([1e-1,15,-1.5,1.5])
    py.xlabel('Projected Radius (arcsec)')
    py.ylabel('j')
    py.savefig(outdir + 'ang_momentum_r2d.png')
    py.close()

    # Plot the CDF of the h statistic
    py.clf()
    py.figure(figsize=(6,6))
    py.subplots_adjust(wspace=0.25, hspace=0.25, left=0.15, right=0.95, top=0.95)
    py.plot(hs_sort, cdf/hs_sort.sum(), 'k-')
    py.plot(h_am, cf_am, 'k--')
    py.xlabel('|h|')
    py.ylabel('N (<|h|)')
    py.savefig(outdir + 'h_statistic_cdf.png')
    py.close()
    
    


def compare_to_lu2009(mosaic=True, suffix='_mosaic'):
    """
    Compare our latest astrometry to those in Lu et al. (2009).
    """
    cc = objects.Constants()

    # Load names of young stars in our align
    if mosaic == True:
        # Load up mosaic data as well; select only stars at r>4, since
        # we don't want to add any info from mosaics if we have it in
        # the central 10" already
        yng1 = young.loadYoungStars(root+alnDir,fit=poly,points=points,
                                    radiusCut=0.8, silent=True,skipStar=['S5-237']) 
        yng2 = young.loadYoungStars(mscDir,fit=polyM,points=pointsM,
                                    mosaic=True, withRVonly=True,silent=True)
        # Merge this object with object from central 10" analysis
        yng = merge(yng1, yng2)
    else:
        yng = young.loadYoungStars(root+alnDir,fit=poly,points=points,
                                   radiusCut=0.8, silent=True)

    names = yng.getArray('name')

    cc.asy_to_kms = dist * cc.cm_in_au / (1.e5 * cc.sec_in_yr)

    # Get disk membership
    diskTab = asciidata.open(root+alnDir+'tables/disk_membership_prob%s.dat' % suffix)
    nameDM = [diskTab[0][ss].strip() for ss in range(diskTab.nrows)]
    diskP = diskTab[1].tonumpy()
    diskIdx = (np.where(diskP > 2.7e-3))[0]

    # Load Lu09 astrometry data (Table 2 in paper)
    lu09 = asciidata.open('/u/ghezgroup/data/gc/source_list/lu2009_astrometry.dat')
    luNames = lu09[0].tonumpy()
    luMag = lu09[1].tonumpy()
    luN = lu09[2].tonumpy()
    lut0 = lu09[3].tonumpy()
    luR = lu09[4].tonumpy()
    luX = lu09[5].tonumpy()
    luY = lu09[6].tonumpy()
    luVx = lu09[7].tonumpy()
    luVxe = lu09[8].tonumpy()
    luVy = lu09[9].tonumpy()
    luVye = lu09[10].tonumpy()
    luVz = lu09[11].tonumpy()
    luVze = lu09[12].tonumpy()
    luAr = lu09[13].tonumpy()
    luAre = lu09[14].tonumpy()
    luAt = lu09[15].tonumpy()
    luAte = lu09[16].tonumpy()
    altName = lu09[17].tonumpy()

    # JLu's 3 sigma upper limits
    luAr3Up = luAr + (3.0 * luAre)
    luAt3Up = luAt + (3.0 * luAte)
    # JLu's 3 sigma lower limits
    luAr3Lo = luAr - (3.0 * luAre)
    luAt3Lo = luAt - (3.0 * luAte)

    print 'Lu et al. (2009) median acceleration errors:'
    print '  radial:     %5.2f mas/yr^2' % np.median(luAre)
    print '  tangential: %5.2f mas/yr^2' % np.median(luAte)

    # Load Lu09 disk membership data (Table 3 in paper)
    lu09M = asciidata.open('/u/ghezgroup/data/gc/source_list/lu2009_disk_membership.dat')
    luNamesM = lu09M[0].tonumpy() # not in same order as luNames above
    luSA = lu09M[1].tonumpy()
    luDisk = lu09M[2].tonumpy()
    lu_eAll = lu09M[3].tonumpy()
    lu_eAll_rng = lu09M[4].tonumpy()
    lu_eDisk = lu09M[5].tonumpy()
    lu_eDisk_rng = lu09M[6].tonumpy()
    lu_drxn = lu09M[7].tonumpy()

    hdr = '%12s  %6s  %15s  %6s  %15s'
    fmt = '%12s  %6.3e  %15s  %6.3e  %15s'
    print hdr % ('Name', 'SyProb', 'On disk?', 'LuProb', 'On disk?')

    syX = []
    syY = []
    syVx = []
    syVy = []
    syAr = []
    syAt = []
    syProb = []
    luProb = []
    # Loop thru Lu stars and compare to our astrometry
    for ll in range(len(luNames)):
	i = np.where(np.array(names) == luNames[ll])[0]

	star = yng.stars[i]
        mag = yng.stars[i].mag
        x = star.fitXa.p
        y = star.fitYa.p
        xe = star.fitXa.perr
        ye = star.fitYa.perr
        vx = star.fitXa.v / cc.asy_to_kms * 1.e3 # mas/yr
        vy = star.fitYa.v / cc.asy_to_kms * 1.e3 # mas/yr
        vxe = star.fitXa.verr / cc.asy_to_kms * 1.e3 # mas/yr
        vye = star.fitYa.verr / cc.asy_to_kms * 1.e3 # mas/yr
        ax = star.fitXa.a
        ay = star.fitYa.a
        axe = star.fitXa.aerr
        aye = star.fitYa.aerr
        t0 = star.fitXa.t0

        # How many epochs was each star detected in?
        pts = asciidata.open('%s%s%s%s.points' % (root, alnDir, points, names[i]))
        ep = pts[0].tonumpy()
        cnt = len(ep)

        r = np.sqrt(x**2 + y**2)
        rcgs = r * dist * cc.cm_in_pc / cc.au_in_pc
        
	# Lower allowed radius is set by 2D distance.
	rLower = r * dist / cc.au_in_pc

	#axe *= cc.asy_to_kms
	#aye *= cc.asy_to_kms
	
	# What is the polyfit acceleration upper limit
	# along the radial component, in km/s/yr
        (ar, at, are, ate) = util.xy2circErr(x, y, ax, ay,
                                             xe, ye, axe, aye)

        # What is this star's probability of being on the disk?
	didx = np.where(np.array(nameDM) == luNames[ll])[0]
        syProb = np.concatenate([syProb, diskP[didx]]) # If > 2.7e-3, then on disk

        # What is this star's probability of being on the disk based on Lu et al?
	lidx = np.where(np.array(luNamesM) == luNames[ll])[0]
        luProb = np.concatenate([luProb, luDisk[lidx]])

        syOn = ' '
        luOn = ' '
        if diskP[didx] < 2.7e-3:
            syOn = 'SY: Not On Disk'
        if luDisk[lidx] < 2.7e-3:
            luOn = 'Lu: Not On Disk'

        print fmt % (names[i], diskP[didx], syOn, luDisk[lidx], luOn)

        # Build the arrays for plotting
        syX = np.concatenate([syX, [x]])
        syY = np.concatenate([syY, [y]])
        syVx = np.concatenate([syVx, [vx]])
        syVy = np.concatenate([syVy, [vy]])
        syAr = np.concatenate([syAr, [ar*1.e3]])
        syAt = np.concatenate([syAt, [at*1.e3]])

    print
    print '%i stars in common with Lu et al. (2009)' % len(syVx)

    import matplotlib.font_manager
    prop = matplotlib.font_manager.FontProperties(size=12)
    usetexTrue()
    py.figure(1)
    py.figure(figsize=(10, 10))
    py.clf()
    py.subplots_adjust(left=0.1, right=0.98, top=0.95,
                       bottom=0.1, wspace=0.25, hspace=0.25)
    py.subplot(2,2,1)
    py.plot(luAr3Lo, syAr, 'r.', label='Radial')
    py.plot(luAt3Lo, syAt, 'b.', label='Tangential')
    py.plot([-0.4,0.4],[-0.4,0.4],'k--')
    py.axis([-0.4,0.4,-0.4,0.4])
    py.xlabel(r'Lu et al. 3$\sigma$ Accel Lower Limit (mas/yr$^2$)')
    py.ylabel('Yelda et al. Acceleration (mas/yr$^2$)')
    py.legend(numpoints=1,fancybox=True,loc=2,prop=prop)
    py.subplot(2,2,2)
    py.plot(luAr3Up, syAr, 'r.', label='Radial')
    py.plot(luAt3Up, syAt, 'b.', label='Tangential')
    py.plot([-0.4,0.4],[-0.4,0.4],'k--')
    py.axis([-0.4,0.4,-0.4,0.4])
    py.xlabel(r'Lu et al. 3$\sigma$ Accel Upper Limit (mas/yr$^2$)')
    py.ylabel('Yelda et al. Acceleration (mas/yr$^2$)')
    py.subplot(2,2,3)
    py.plot(luAr, syAr, 'r.', label='Radial')
    py.plot(luAt, syAt, 'b.', label='Tangential')
    py.plot([-0.4,0.4],[-0.4,0.4],'k--')
    py.xlabel(r'Lu et al. Acceleration (mas/yr$^2$)')
    py.ylabel('Yelda et al. Acceleration (mas/yr$^2$)')
    py.axis([-0.4,0.4,-0.4,0.4])
    py.subplot(2,2,4)
    py.errorbar(luAr, np.arange(1,len(luAr)+1), xerr=3.0*luAre, fmt='r.')
    py.errorbar(luAt, np.arange(1.3,len(luAt)+1), xerr=3.0*luAte, fmt='b.', label='Lu')
    py.plot(syAr, np.arange(1, len(luAr)+1), 'rx')
    py.plot(syAt, np.arange(1.3, len(luAt)+1), 'bx', label='Yelda')
    py.xlabel(r'Acceleration Comparison (mas/yr$^2$)')
    py.ylabel('N')
    py.axis([-0.8, 0.8, 0, 33])
    py.legend(numpoints=1,fancybox=True,loc=3,prop=prop)
    py.savefig(plotdir + 'compare_accel_lu2009.png')
    py.close(1)
    usetexFalse()

    py.figure(2)
    py.figure(figsize=(6,6))
    py.clf()
    py.subplots_adjust(left=0.15)
    py.plot(luVx, syVx, 'r.', label='X')
    py.plot(luVy, syVy, 'b.', label='Y')
    py.plot([-15, 15], [-15, 15], 'k--')
    py.xlabel('Lu et al. Velocity (mas/yr)')
    py.ylabel('Yelda et al. Velocity (mas/yr)')
    py.axis([-15,15,-15,15])
    py.legend(numpoints=1,fancybox=True,loc=2,prop=prop)
    py.savefig(plotdir + 'compare_vel_lu2009.png')
    py.close(2)

    
    # Difference between Lu+09 and me
    import matplotlib.font_manager
    prop = matplotlib.font_manager.FontProperties(size=14)
    dvx = syVx - luVx
    dvy = syVy - luVy
    step = 0.25
    binsIn = np.arange(min(dvx.min(), dvy.min()), max(dvx.max(), dvy.max())+step, step)
    py.figure(3)
    py.figure(figsize=(10,5))
    py.clf()
    py.subplots_adjust(left=0.1, right=0.98, top=0.9,
                       bottom=0.1, wspace=0.25, hspace=0.25)
    py.subplot(1,2,1)
    qvr = py.quiver([syX], [syY], [dvx], [dvy], angles='xy')
    py.axis([4, -4, -4, 4])
    py.xlabel('RA Offset (arcsec)')
    py.ylabel('Dec Offset (arcsec)')
    py.title('Yelda - Lu Velocity Difference')
    py.quiverkey(qvr, -2.5, 3, 1.0, '1 mas/yr', coordinates='data', color='red')
    py.subplot(1,2,2)
    py.hist(dvx, bins=binsIn, color='r', histtype='step', label='X')
    py.hist(dvy, bins=binsIn, color='b', histtype='step', label='Y')
    py.legend(numpoints=1, fancybox=True, loc=2, prop=prop)
    py.xlabel('Yelda - Lu Velocity Difference (mas/yr)')
    py.ylabel('N')
    py.axis([-3.1, 3.1, 0, 8])
    py.savefig(plotdir + 'compare_vel_vector_lu2009.png')
    py.close(3)

    py.figure(4)
    py.figure(figsize=(6,6))
    py.clf()
    py.plot(luProb, syProb, 'k.')
    py.plot([0,1], [0,1], 'k--')
    for ll in range(len(luNames)):
        py.text(luProb[ll]+0.04, syProb[ll], luNames[ll], fontsize=8)
    py.xlabel('Lu et al. Probability')
    py.ylabel('Yelda et al. Probability')
    py.title('CW Disk Membership Comparison')
    py.savefig(plotdir + 'compare_diskMember_lu2009.png')
    py.close(4)

def plot_IO_vs_bartko():
    """
    Plots the inclination and Omega for the inner and outer
    radial bins against those reported in Bartko et al. (2009)
    """

    # The radial bins are separated slightly differently from Bartko
    # I've used bins such that equal number of stars fall in each
    r1 = np.array([0.8, 3.2])
    r1_mid = r1.mean()
    r2 = np.array([3.2, 6.5])
    r2_mid = r2.mean()
    r3 = np.array([6.5, 14.0])
    r3_mid = r3.mean()

    rad = np.array([r1_mid, r2_mid, r3_mid])

    # Values are for three radial bins incorporating the wide mosaic
    i_keck = [128.68, 127.17, 117.28]
    O_keck = [99.14, 103.36, 191.95]
    ie_keck = [2.64, 12.07, 7.18]
    Oe_keck = [2.64, 12.07, 7.18]
    nK = [40.0, 39.0, 39.0] # number of stars per radial bin

    # Bartko et al. (2009)
    # They have a typo in Table 3, for the 3rd radial bin's i, and Omega.  (phi, theta) are
    # correct. So let's use those values and do the math for them correctly.
    phi = [256.0, 215.0, 181.0]
    theta = [54.0, 28.0, 62.0]
    # convert from phi,theta to i,Omega (Paumard et al. 2006, Appendix A):
    itmp = [(180.0 - t) for t in theta]
    Otmp = [360.0 + (-1.0*p) for p in phi] 
    #i_vlt = [126.0, 152.0, 118.0]  # this is what it should be
    #O_vlt = [104.0, 145.0, 179.0]  # this is what it should be
    				  
    ie_vlt = [3.2, -999, 3.8]
    Oe_vlt = [3.2, -999, 3.8]
    nV = [32.0, 30.0, 28.0]

    # VLT doesn't have errors for their middle radial bin, so
    # we will estimate it for them:
    e_vlt = ie_keck[1] * np.sqrt(nK[1]/nV[1]) * (ie_vlt[0]/ie_keck[0]) * np.sqrt(nV[0]/nK[0])

    ie_vlt[1] = e_vlt
    Oe_vlt[1] = e_vlt

    import matplotlib.font_manager
    prop = matplotlib.font_manager.FontProperties(size=12)

    py.clf()
    py.figure(figsize=(8,5))
    py.subplots_adjust(left=0.1,right=0.98,top=0.95,bottom=0.1,wspace=0.35)
    py.subplot(1,2,1)
    # Plot Keck points
    py.errorbar(rad, i_keck, yerr=ie_keck, fmt='r.', label='Keck')
    # Plot VLT points; shift the points slightly over on the X axis to make clear
    py.errorbar(rad+0.2, i_vlt, yerr=ie_vlt, fmt='b.', label='VLT')
    py.legend(numpoints=1, fancybox=True, loc=1, prop=prop)
    py.xlabel('Projected Radius (arcsec)')
    py.ylabel('Inclination (deg)')
    py.axis([0,12.0,90,210])

    py.subplot(1,2,2)
    # Plot Keck points
    py.errorbar(rad, O_keck, yerr=Oe_keck, fmt='r.')
    # Plot VLT points; shift the points slightly over on the X axis to make clear
    py.errorbar(rad+0.2, O_vlt, yerr=Oe_vlt, fmt='b.')
    py.xlabel('Projected Radius (arcsec)')
    py.ylabel('Longitude of the Ascending Node (deg)')
    py.axis([0,12.0,90,210])
    py.savefig(plotdir + 'diskAngle_keck_vlt.png')
    

def plot_IO_central_3bins():
    """
    Plots the inclination and Omega for three radial bins within
    our central 10 arcsec imaging (note: not the mosaic!).
    Bins were separated at 2.266 and 3.538 arcsec.
    """

    r1 = np.array([0.8, 2.266])
    r1_mid = r1.mean()
    r2 = np.array([2.266, 3.538])
    r2_mid = r2.mean()
    r3 = np.array([3.538, 7.0])
    r3_mid = r3.mean()

    rad = np.array([r1_mid, r2_mid, r3_mid])

    # Values are for three radial bins within the central 10 arcsec data set
    # These need to be updated:
    i = [130.2, 132.6, 152.7]
    O = [99.1, 109.3, 113.1]
    ie = [2.9, 9.2, 12.9]
    Oe = [2.9, 9.2, 12.9]
    nn = [20.0, 23.0, 23.0] # number of candidate disk members per radial bin

    # Disk radius for each of the 3 radial bins (HWHM)
    hwhm = [12.87, 43.98, 61.91]
    hwhm_rad = [hh*(np.pi/180.) for hh in hwhm]

    usetexTrue()
    import matplotlib.font_manager
    prop = matplotlib.font_manager.FontProperties(size=12)

    py.figure(1)
    py.clf()
    py.figure(figsize=(8,4))
    py.subplots_adjust(left=0.1,right=0.98,top=0.9,bottom=0.15,wspace=0.35)
    py.subplot(1,2,1)
    # Plot Keck points
    py.errorbar(rad, i, yerr=ie, fmt='k.')
    py.xlabel('Projected Radius (arcsec)')
    py.ylabel(r'${\bf {\it i}}$ (deg)')
    py.axis([0,6.0,80,180])

    py.subplot(1,2,2)
    # Plot Keck points
    py.errorbar(rad, O, yerr=Oe, fmt='k.')
    py.xlabel('Projected Radius (arcsec)')
    py.ylabel(r'${\bf \Omega}$ (deg)')
    py.axis([0,6.0,80,180])
    py.savefig(plotdir + 'diskAngle_keck_central_3bins.png')
    py.close(1)

    py.figure(2)
    py.clf()
    py.figure(figsize=(6,6))
    py.plot(rad, np.tan(hwhm_rad), 'k.')
    py.xlabel('Projected Radius (arcsec)')
    py.ylabel('Disk Scale Height')
    py.axis([0,6.0,0,2])
    py.savefig(plotdir + 'diskRadius_keck_central_3bins.png')
    usetexFalse()


def plot_IO_3bins_mosaic():
    """
    Plots the inclination and Omega for three radial bins including
    our full data set (including mosaic).
    Bins were separated at 3.2 and 6.48 arcsec.
    """

    r1 = np.array([0.8, 3.2])
    r1_mid = r1.mean()
    r2 = np.array([3.2, 6.48])
    r2_mid = r2.mean()
    r3 = np.array([6.48, 14.0])
    r3_mid = r3.mean()

    rad = np.array([r1_mid, r2_mid, r3_mid])

    # Values are for three radial bins within the central 10 arcsec data set
    # These need to be updated:
    i = [128.68, 127.17, 117.28]
    O = [99.14, 103.36, 191.95]
    ie = [2.64, 12.07, 7.18]
    Oe = [2.64, 12.07, 7.18]
    nn = [26.0, 39.0, 22.0] # number of candidate disk members per radial bin

    # Disk radius for each of the 3 radial bins (HWHM)
    hwhm = [13.45, 75.38, 33.68]
    hwhm_rad = [hh*(np.pi/180.) for hh in hwhm]

    usetexTrue()
    import matplotlib.font_manager
    prop = matplotlib.font_manager.FontProperties(size=12)

    py.figure(1)
    py.clf()
    py.figure(figsize=(8,4))
    py.subplots_adjust(left=0.1,right=0.98,top=0.9,bottom=0.15,wspace=0.35)
    py.subplot(1,2,1)
    # Plot Keck points
    py.errorbar(rad, i, yerr=ie, fmt='k.')
    py.xlabel('Projected Radius (arcsec)')
    py.ylabel(r'${\bf {\it i}}$ (deg)')
    py.axis([0,14.0,80,180])

    py.subplot(1,2,2)
    # Plot Keck points
    py.errorbar(rad, O, yerr=Oe, fmt='k.')
    py.xlabel('Projected Radius (arcsec)')
    py.ylabel(r'${\bf \Omega}$ (deg)')
    py.axis([0,14.0,80,200])
    py.savefig(plotdir + 'diskAngle_keck_mosaic_3bins.png')
    py.close(1)

    py.figure(2)
    py.clf()
    py.figure(figsize=(6,6))
    py.plot(rad, np.tan(hwhm_rad), 'k.')
    py.xlabel('Projected Radius (arcsec)')
    py.ylabel('Disk Scale Height')
    py.axis([0,14.0,0,3.0])
    py.savefig(plotdir + 'diskRadius_keck_mosaic_3bins.png')
    usetexFalse()



def find_minMass_from_S015():
    """
    Determines the minimum black hole mass based on S0-15
    and the assumption that it is bound.
    Determines escape velocity at S0-15's projected radius,
    and what the minimum mass has to be in order for the star
    to stay bound.  Assumes the total velocity of S0-15 is
    it's 5 sigma upper bound.

    This will be used to determine what solutions we can
    throw out when running analyticOrbitsMC().
    """
    cc = objects.Constants()

    # Load our data
    yng = young.loadYoungStars(root+alnDir,fit=poly,points=points,
                               radiusCut=0.8, silent=True)

    names = yng.getArray('name')
    s015 = np.where(np.array(names) == 'S0-15')[0]
    star = yng.stars[s015]

    mag = yng.stars[s015].mag
    x = star.fitXa.p *dist * cc.cm_in_au # cm
    y = star.fitYa.p * dist * cc.cm_in_au # cm
    r2d = np.hypot(x, y) # cm
    xe = star.fitXa.perr * dist * cc.cm_in_au # cm
    ye = star.fitYa.perr * dist * cc.cm_in_au # cm
    vx = star.fitXa.v # km/s
    vy = star.fitYa.v # km/s
    vz = star.vz # km/s
    vxe = star.fitXa.verr # km/s
    vye = star.fitYa.verr # km/s
    vze = star.vzerr # km/s
    t0 = star.fitXa.t0

    # upper limit on the 2D position
    r2d_err = (np.sqrt(((x*xe)**2 + (y*ye)**2) / r2d**2)) # cm
    r2d_10up = (r2d + 10.0*r2d_err)

    vtot = np.sqrt(vx**2 + vy**2 + vz**2) # km/s
    vtot_err = (np.sqrt(((vx*vxe)**2 + (vy*vye)**2 + (vz*vze)**2) / vtot**2)) # km/s
    vtot_10up = vtot + 10.0*vtot_err # km/s

    vtot_cgs = vtot * 1.e5
    vtot_err_cgs = vtot_err * 1.e5
    vtot_10up_cgs = vtot_10up * 1.e5

    # Given S0-15's velocity and projected radius,
    # what is the minimum mass that keeps the star bound?
    #minMass = (vtot_cgs**2 * r2d) / (2 * cc.G) # grams
    minMass = (vtot_10up_cgs**2 * r2d_10up) / (2 * cc.G) # grams
    minMass_msun = minMass / cc.msun # Solar masses

    print 'S0-15 total velocity: %6.2f +- %6.2f km/s' % (vtot, vtot_err)
    print 'S0-15 projected radius: %6.3f +- %6.3f pc' % (r2d/cc.cm_in_pc, r2d_err/cc.cm_in_pc)
    print 
    print 'Minimum black hole mass assuming bound orbit for S0-15 and 10 sigma velocity upper limit:'
    print '  %8.2e solar masses' % minMass_msun



def table_compare_to_lit(align = 'align/align_d_rms_1000_abs_t', mosaic=False):
    """
    Creates table comparing young star disk membership from
    Lu et al. & Bartko et al.
    """

    outdir = root + alnDir + 'tables/'

    # Open Jessica's astrometry table (not the final paper version,
    # but we just need the star names)
    lu09 = asciidata.open(outdir + 'lu06yng_tab_astrom.dat')
    luNames = lu09[0].tonumpy()
    luNames = [ll.strip() for ll in luNames]

    # Get the star names in Bartko's table
    bart = tabs.Bartko2009()
    bartNames = bart.ourName

    # Load up young stars
    # young.loadYoungStars calls youngStarNames but grabs
    # only young stars beyond r = 0.8 arcsec
    # Load names of young stars in our align
    if mosaic == True:
        # Load up mosaic data as well; select only stars at r>4, since
        # we don't want to add any info from mosaics if we have it in
        # the central 10" already
        yng1 = young.loadYoungStars(root+alnDir,fit=poly,points=points,
                                    silent=True,skipStar=['S5-237']) 
        yng2 = young.loadYoungStars(mscDir,fit=polyM,points=pointsM,
                                    mosaic=True, withRVonly=True,silent=True)
        cntrlNames = yng1.getArray('name')
        mscNames = yng2.getArray('name')
        # Merge this object with object from central 10" analysis
        yng = merge(yng1, yng2)
    else:
        yng = young.loadYoungStars(root+alnDir,fit=poly,points=points,silent=True) 

    if mosaic == False:
        outFile = outdir + 'compare2lit.tex'
    else:
        outFile = outdir + 'compare2lit_mosaic.tex'
    if os.access(outFile, os.F_OK): os.remove(outFile)

    _out = open(outFile, 'w')
    _out.write('\\begin{deluxetable}{lrrrrrrr}\n')
    _out.write('\\tabletypesize{\\scriptsize}\n')
    _out.write('\\tablewidth{0pt}\n')
    _out.write('\\tablecaption{Literature}\n')
    _out.write('\\tablehead{\n')
    _out.write('  \\colhead{Name} &\n')
    _out.write('  \\colhead{K} &\n')
    _out.write('  \\colhead{Radius} &\n')
    _out.write('  \\colhead{L09} &\n')
    _out.write('  \\colhead{B09} \\\\\n')
    _out.write('%\n')
    _out.write('  \\colhead{} &\n')
    _out.write('  \\colhead{(mag)} &\n')
    _out.write('  \\colhead{(arcsec)} &\n')
    _out.write('  \\colhead{} &\n')
    _out.write('  \\colhead{}\n')
    _out.write('}\n')
    _out.write('\\startdata\n')

    xAll = yng.getArray('fitXa.p')
    yAll = yng.getArray('fitYa.p')
    rAll = np.hypot(xAll, yAll)

    ridx = rAll.argsort()

    fmt = '%12s & %4.1f & %4.2f & %3s & %3s \\\\ \n'

    for ii in range(len(ridx)):
        star = yng.stars[ridx[ii]]
        name = star.name
        mag = star.mag
        if (name in mscNames) & (name not in cntrlNames):
            x = star.fitXv.p
            y = star.fitYv.p
            r = np.sqrt(x**2 + y**2)
        else:
            x = star.fitXa.p
            y = star.fitYa.p
            r = np.sqrt(x**2 + y**2)

        # Check if this star was in Lu+09 or Bartko+09
        lu = ''
        bart = ''
        if name in luNames:
            lu = '*'
        if name in bartNames:
            bart = '*'

        _out.write(fmt % (name, mag, r, lu, bart))

    _out.write('\\enddata\n')
    _out.write('\\end{deluxetable}')
    _out.close()
        



def table_yng_astrometry(align = 'align/align_d_rms_1000_abs_t', mosaic=False):
    """
    Creates table of young star astrometry. 
    """

    outdir = root + alnDir + 'tables/'

    # Load up young stars
    # young.loadYoungStars calls youngStarNames but grabs
    # only young stars beyond r = 0.8 arcsec
    # Load names of young stars in our align
    if mosaic == True:
        # Load up mosaic data as well; select only stars at r>4, since
        # we don't want to add any info from mosaics if we have it in
        # the central 10" already
        yng1 = young.loadYoungStars(root+alnDir,fit=poly,points=points,
                                    silent=True,skipStar=['S5-237']) 
        yng2 = young.loadYoungStars(mscDir,fit=polyM,points=pointsM,
                                    mosaic=True, silent=True)
        cntrlNames = yng1.getArray('name')
        mscNames = yng2.getArray('name')
        # Merge this object with object from central 10" analysis
        yng = merge(yng1, yng2)
    else:
        yng = young.loadYoungStars(root+alnDir,fit=poly,points=points,
                                   silent=True) 

    cc = objects.Constants()

    outFile = outdir + 'sythesis_tab_astrom.tex'
    outFile2 = outdir + 'sythesis_tab_astrom.dat'
    if os.access(outFile, os.F_OK): os.remove(outFile)
    if os.access(outFile2, os.F_OK): os.remove(outFile2)

    _out2 = open(outFile2, 'w')

    _out = open(outFile, 'w')
    _out.write('\\begin{landscape}\n')
    _out.write('\\begin{deluxetable}{lrrrrrrrrrrrrr}\n')
    _out.write('\\tabletypesize{\\scriptsize}\n')
    _out.write('\\tablewidth{0pt}\n')
    _out.write('\\tablecaption{Astrometry of Galactic Center Young Stars')
    _out.write('\\label{tab:yng_pm_table}}\n')
    _out.write('\\tablehead{\n')
    _out.write('  \\colhead{} &\n')
    _out.write('  \\colhead{Name} &\n')
    _out.write('  \\colhead{K} &\n')
    _out.write('  \\colhead{N$_{epochs}$} &\n')
    _out.write('  \\colhead{Epoch} &\n')
    _out.write('  \\colhead{Radius} &\n')
    _out.write('  \\colhead{$\\Delta RA$ \\tablenotemark{a}} &\n')
    _out.write('  \\colhead{$\\Delta DEC$ \\tablenotemark{a}} &\n')
    _out.write('  \\colhead{$v_{ra}$} &\n')
    _out.write('  \\colhead{$v_{dec}$} &\n')
    _out.write('  \\colhead{$v_z$} &\n')
    _out.write('  \\colhead{$a_\\rho$} &\n')
    _out.write('  \\colhead{$a_{tan}$} &\n')
    _out.write('  \\colhead{Lit} \\\\\n')
    _out.write('%\n')
    _out.write('  \\colhead{} &\n')
    _out.write('  \\colhead{} &\n')
    _out.write('  \\colhead{(mag)} &\n')
    _out.write('  \\colhead{} &\n')
    _out.write('  \\colhead{(year)} &\n')
    _out.write('  \\colhead{(arcsec)} &\n')
    _out.write('  \\colhead{(arcsec)} &\n')
    _out.write('  \\colhead{(arcsec)} &\n')
    _out.write('  \\colhead{(mas/yr)} &\n')
    _out.write('  \\colhead{(mas/yr)} &\n')
    _out.write('  \\colhead{(km/s)} &\n')
    _out.write('  \\colhead{(mas/yr$^2$)} &\n')
    _out.write('  \\colhead{(mas/yr$^2$)} &\n')
    _out.write('  \\colhead{}\n')
    _out.write('}\n')
    _out.write('\\startdata\n')

    hdr2 = '%3s  %12s   %4s   %3s   %8s   '
    # Radius X Y
    hdr2 += '%6s  %15s  %15s  '
    # Velocity: vx, vy, vz (with errors)
    hdr2 += '%14s   %14s   %9s   '
    # Acceleration: a_rho (with errors)
    hdr2 += '%10s  %10s\n'
    _out2.write(hdr2 % ('#', 'Name', 'Mag', 'Cnt', 'Epoch',
                        'Radius', 'X Pos', 'Y Pos',
                        'X Vel', 'Y Vel', 'Z Vel',
                        'X Acc', 'Y Acc'))
    _out2.write(hdr2 % ('', '', '', '', '(year)',
                        '(")', '(")', '(")',
                        '(mas/yr)', '(mas/yr)', '(km/s)', 
                        '(mas/yr^2)', '(mas/yr^2)'))


    xAll = yng.getArray('fitXv.p')
    yAll = yng.getArray('fitYv.p')
    rAll = np.hypot(xAll, yAll)

    ridx = rAll.argsort()

    refEpoch = 2006.470

    # Keep a tally of the average X and Y positional errors
    xerrAll = np.zeros(len(ridx), float)
    yerrAll = np.zeros(len(ridx), float)
    
    # Open Jessica's astrometry table (not the final paper version,
    # but we just need the star names)
    lu09 = asciidata.open(outdir + 'lu06yng_tab_astrom.dat')
    luNames = lu09[0].tonumpy()
    luNames = [ll.strip() for ll in luNames]

    # Get the star names in Bartko's table
    bart = tabs.Bartko2009()
    bartNames = bart.ourName

    cc.asy_to_kms = dist * cc.cm_in_au / (1.e5 * cc.sec_in_yr)

    for ii in range(len(ridx)):
        # Name and Magnitude
        fmt = '%3s & %12s & %4.1f & %3d & %8.3f & '
        # Radius X Y
        fmt += '%5.2f & '
        fmt += '%7.3f & '
        fmt += '%7.3f & '
        # Velocity: vx, vy, vz (with errors)
        fmt += '%7.2f $\\pm$ %6.2f & %7.2f $\\pm$ %6.2f & %5d $\\pm$ %3d & '
        # Acceleration: a_rho (with errors) and a_tan (with errors)
        fmt += '%5.2f $\\pm$ %4.2f & %5.2f $\\pm$ %4.2f &  %7s \\\\ \n'
        # Alternate Name from Paumard.
        #fmt += '%3s \\\\\n'

        ### Format for data table ###
        fmt2 = '%12s   %4.1f   %3d   %8.3f   '
        # Radius X Y
        fmt2 += '%5.2f   %8.4f %6.4f  %8.4f %6.4f  '
        # Velocity: vx, vy, vz (with errors)
        fmt2 += '%7.2f %6.2f   %7.2f %6.2f   %5d %3d   '
        # Acceleration: a_rho (with errors)
        fmt2 += '%5.2f %4.2f  %5.2f %4.2f\n'

        star = yng.stars[ridx[ii]]
        name = star.name

        mscStar = False

        if (name in mscNames) & (name not in cntrlNames):
            mscStar = True
            
            dtX = refEpoch - star.fitXv.t0
            dtY = refEpoch - star.fitYv.t0

            # Convert into arcsec/yr
            vx = star.fitXv.v / cc.asy_to_kms
            vy = star.fitYv.v / cc.asy_to_kms
            vxerr = star.fitXv.verr / cc.asy_to_kms
            vyerr = star.fitYv.verr / cc.asy_to_kms

            x = star.fitXv.p
            y = star.fitYv.p
            xerr = star.fitXv.perr
            yerr = star.fitYv.perr

            t0 = star.fitXv.t0

            ax = -999
            ay = -999
            axerr = -999
            ayerr = -999
        else:
            dtX = refEpoch - star.fitXa.t0
            dtY = refEpoch - star.fitYa.t0

            # Convert into arcsec/yr
            vx = star.fitXa.v / cc.asy_to_kms
            vy = star.fitYa.v / cc.asy_to_kms
            vxerr = star.fitXa.verr / cc.asy_to_kms
            vyerr = star.fitYa.verr / cc.asy_to_kms
            x = star.fitXa.p
            y = star.fitYa.p
            xerr = star.fitXa.perr
            yerr = star.fitYa.perr

            ax = star.fitXa.a
            ay = star.fitYa.a
            axerr = star.fitXa.aerr
            ayerr = star.fitYa.aerr
            t0 = star.fitXa.t0
	    # What is the polyfit acceleration
	    # along the radial component
            (ar, at, arerr, aterr) = util.xy2circErr(x, y, ax, ay,
                                                     xerr, yerr, axerr, ayerr)

            
        r = np.sqrt(x**2 + y**2)

        # Radial velocity in km/s
        vz = star.vz
        vzerr = star.vzerr

        if vz == None:
            vz = ''
            vzerr = ''
            # Name and Magnitude
            fmt = '%3s & %12s & %4.1f & %3d & %8.3f & '
            # Radius X Y
            fmt += '%5.2f & '
            fmt += '%7.3f & '
            fmt += '%7.3f & '
            # Velocity: vx, vy, vz (with errors)
            fmt += '%7.2f $\\pm$ %6.2f & %7.2f $\\pm$ %6.2f & %5s        %3s & '
            # Acceleration: a_rho (with errors) and a_tan (with errors)
            fmt += '%5.2f $\\pm$ %4.2f & %5.2f $\\pm$ %4.2f &  %7s \\\\ \n'

            ### Format for data table ###
            fmt2 = '%12s   %4.1f   %3d   %8.3f   '
            # Radius X Y
            fmt2 += '%5.2f   %8.4f %6.4f  %8.4f %6.4f  '
            # Velocity: vx, vy, vz (with errors)
            fmt2 += '%7.2f %6.2f   %7.2f %6.2f   %5s %3s   '
            # Acceleration: a_rho (with errors)
            fmt2 += '%5.2f %4.2f  %5.2f %4.2f\n'

        xerrAll[ii] = xerr
        yerrAll[ii] = yerr
        
        # Do some unit conversion.
        #    Leave positions in arcsec
        #    Change velocities to km/s
        #    Change accelerations to km/s/yr
        #vx *= cc.asy_to_kms
        #vy *= cc.asy_to_kms
        #vxerr *= cc.asy_to_kms
        #vyerr *= cc.asy_to_kms
        #ar *= cc.asy_to_kms
        #arerr *= cc.asy_to_kms

        vx *= 1000.0
        vy *= 1000.0
        vxerr *= 1000.0
        vyerr *= 1000.0
        ar *= 1000.0
        arerr *= 1000.0
        at *= 1000.0
        aterr *= 1000.0
        ax *= 1000.0
        ay *= 1000.0
        axerr *= 1000.0
        ayerr *= 1000.0


        # Check if this star was in Lu+09 or Bartko+09
        lit = '-'
        if ((name in luNames) and (name in bartNames)):
            lit = 'L09,B09'
        elif (name in luNames):
            lit = 'L09'
        elif (name in bartNames):
            lit = 'B09'

        if ('irs' in name):
            name = 'IRS %s' % (name[3:])

        if mscStar == True:
            ar = -99
            at = -99
            arerr = 99
            aterr = 99

        _out.write(fmt % (str(ii+1), name, star.mag, star.velCnt,
                          t0, r, x, y, 
                          vx, vxerr, vy, vyerr, vz, vzerr,
                          ar, arerr, at, aterr, lit))

        _out2.write(fmt2 % (name, star.mag, star.velCnt,
                            t0, r, x, xerr, y, yerr,
                            vx, vxerr, vy, vyerr, vz, vzerr,
                            ar, arerr, at, aterr))

    xyerrAvg = np.concatenate((xerrAll, yerrAll)).mean() * 1000.0

    print 'Average Positional Error is %5.3f mas ' % xyerrAvg

    _out.write('\\enddata\n')
    _out.write('\\tablecomments{All uncertainties are 1$\sigma$ relative\n')
    _out.write('errors and do not include errors in the plate scale, \n')
    _out.write('location of Sgr A*, or position angle.}\n')
    _out.write('\\tablenotetext{a}{Positions as determined from polynomial\n')
    _out.write('fitting have relative errors of \n')
    _out.write('$\sim$%3.1f mas.}\n' % (xyerrAvg))
    #_out.write('\\tablenotetext{b}{Alternate names\n')
    #_out.write('obtained from Bartko et al. (2009).}\n')
    _out.write('\\end{deluxetable}\n')
    _out.write('\\end{landscape}')
    _out.close()

    _out2.close()


def pdf_radial_bins(mosaic=False):
    """
    Plot PDF(i,O) for separate radial bins:
    	r < 3.5 arcsec
     	3.5 < r < 7 arcsec
        and if mosaic == True:
     	r > 7 arcsec 

    NOTE: This just overplots all the PDFs based on the star's radius.
    	  To get a better sense of the peak in each radial bin, should run
          nearest neighbor density mapping:
             disk = aorb.Disk('./','aorb_acc/')
             disk.run(do_radial_bins=True)
    """

    # Load names of young stars in our align
    if mosaic == True:
        # Load up mosaic data as well; select only stars at r>4, since
        # we don't want to add any info from mosaics if we have it in
        # the central 10" already
        yng1 = young.loadYoungStars(root+alnDir,fit=poly,points=points,
                                    withRVonly=True,silent=True,skipStar=['S5-237']) 
        yng2 = young.loadYoungStars(mscDir,fit=polyM,points=pointsM,
                                    mosaic=True, withRVonly=True,silent=True)
        # Merge this object with object from central 10" analysis
        yng = merge(yng1, yng2)
    else:
        yng = young.loadYoungStars(root+alnDir,fit=poly,points=points,
                                   withRVonly=True,silent=True) 
    orbDir = 'aorb_thesis/'
    #orbDir = 'aorb_acc_mrPDF_MC_newMosaic/'
    suffix = ''

    yngNames = yng.getArray('name')
    r2d = yng.getArray('r2d')

    # Disk solution
    nside = 64
    npix = healpy.nside2npix(nside)
    pixIdx = np.arange(0, npix)
    (disk, diskStd) = loadDiskDensity(npix, orbDir=orbDir)

    pdf_bin1 = np.zeros((npix), dtype=float)
    pdf_bin2 = np.zeros((npix), dtype=float)

    inner = []
    middle = []
    if mosaic == True:
        pdf_bin3 = np.zeros((npix), dtype=float)
        outer = []

    for ss in range(len(yngNames)):
        name = yngNames[ss]
        r = r2d[ss]

        orbFile = ('%s/%s/%s/%s_mc_heal.dat' % (root, alnDir, orbDir, name))
        if os.path.exists(orbFile) == False:
            continue
        #if (name == 'S1-24') or (name == 'S3-2'):
        #    continue
         

        if r < 3.5:
            #print 'Adding %s to inner radial bin (r < 3.5 arcsec)' % name

            pdf = np.fromfile(orbFile, dtype=float)
            pdf_bin1 = pdf_bin1 + pdf
            #inner += 1
            #inner = np.concatenate([inner, [name]])
            inner.append(name)

        elif ((r > 3.5) & (r < 7.0)):
            #print 'Adding %s to outer radial bin (r >= 3.5 arcsec)' % name

            pdf = np.fromfile(orbFile, dtype=float)
            pdf_bin2 = pdf_bin2 + pdf
            middle.append(name)

        elif ((mosaic == True) & (r > 7.0)):
            pdf = np.fromfile(orbFile, dtype=float)
            pdf_bin3 = pdf_bin3 + pdf
            outer.append(name)


    bin1 = '%s/%s/%s/inner_radial_bin_mc_heal.dat' % (root, alnDir, orbDir)
    bin2 = '%s/%s/%s/mid_radial_bin_mc_heal.dat' % (root, alnDir, orbDir)

    pdf_bin1.tofile(bin1)
    pdf_bin2.tofile(bin2)

    # Plot the results on a healpix map
    pdh.go(bin1, npix, 1, 0)
    pdh.go(bin2, npix, 1, 0)

    if mosaic == True:
        bin3 = '%s/%s/%s/outer_radial_bin_mc_heal.dat' % (root, alnDir, orbDir)
        pdh.go(bin3, npix, 1, 0)

    print '%i stars at r < 3.5 arcsec:' % len(inner)
    for ii in inner:
        print ii
    print 
    print '%i stars at 3.5 < r < 7 arcsec:' % len(middle)
    for mm in middle:
        print mm
    print '%i stars at r > 7 arcsec:' % len(outer)
    for oo in outer:
        print oo
    print 


def cdf_radius(mosaic=True, chi2Cut=None, suffix=''):
    """
    Plots cumulative radial distribution of young stars.
    Only includes stars at r>0.8 arcsec and with RVs.
    """
    # Load up young stars
    if mosaic == True:
        # Load up mosaic data as well; select only stars at r>4, since
        # we don't want to add any info from mosaics if we have it in
        # the central 10" already
        yng1 = young.loadYoungStars(root+alnDir,fit=poly,points=points,skipStar=['S5-237'],
                                    withRVonly=True,silent=True,radiusCut=0.8) 
        yng2 = young.loadYoungStars(mscDir,fit=polyM,points=pointsM,
                                    mosaic=True, withRVonly=True,silent=True,
                                    radiusCut=0.8)
        cntrlNames = yng1.getArray('name')
        mscNames = yng2.getArray('name')
        # Merge this object with object from central 10" analysis
        yng = merge(yng1, yng2)
    else:
        yng = young.loadYoungStars(root+alnDir,fit=poly,points=points,
                                   withRVonly=True,silent=True,radiusCut=0.8) 

    yngNames = yng.getArray('name')
    r2d = yng.getArray('r2d')

    if chi2Cut != None:
        xchi2 = yng.getArray('fitpXv.chi2') 
        ychi2 = yng.getArray('fitpYv.chi2')

        bad = []
        # Throw out the mosaic stars with chi2 values > chi2Cut
        hiChi = np.where((xchi2 > chi2Cut) | (ychi2 > chi2Cut))[0]
        hiChi = [np.int(jj) for jj in hiChi]
        hiChiNames = [yngNames[jj] for jj in hiChi]
        # Keep all central 10" stars, throw out mosaic stars with high chi2
        for ii in range(len(hiChi)):
            if hiChiNames[ii] in cntrlNames:
                continue
            else:
                print 'Removing mosaic star %s from Disk analysis' % hiChiNames[ii]
                bb = np.where(np.array(yngNames) == hiChiNames[ii])[0]
                bad = np.concatenate([bad, bb])
        good = np.setdiff1d(np.arange(len(yngNames)), bad)
        print

        yngNames = [yngNames[nn] for nn in good]
        r2d = r2d[good]

    # Get number of stars in each radial bin
    step = 0.1
    rbin = np.arange(0,r2d.max()+step,step)
    yngBin = 0
    numYng = []
    for rr in range(len(rbin)):
        idx = np.where((r2d > rbin[rr]) & (r2d < rbin[rr]+step))[0]
        yngBin += len(idx)
        numYng = np.concatenate([numYng,[yngBin]])

    # Find what radii correspond to roughly 1/3 of the sample
    r2d.sort()
    third = np.floor(len(yngNames) / 3.0)
    r1 = r2d[third]
    r2 = r2d[2*third]

    print
    print 'Total number of young stars: %i' % len(r2d)
    print 'Max radius = %5.2f arcsec' % r2d.max()
    print

    rmid = rbin + step/2.0

    py.figure(1,figsize=(6,6))
    py.clf()
    py.subplots_adjust(left=0.15,right=0.9,top=0.9)
    py.plot(rmid,numYng,'k-',lw=1)
    py.plot([r1,r1],[0,numYng.max()+20],'k--')
    py.plot([r2,r2],[0,numYng.max()+20],'k--')
    py.text(r1+0.1,10,'1/3 of YSOs w/in R=%5.3f"' % r1, fontsize=10)
    py.text(r2+0.1,50,'2/3 of YSOs w/in R=%5.3f"' % r2, fontsize=10)
    py.xlabel('Projected Radius (arcsec)')
    py.ylabel('Cumulative Number of Young Stars')
    py.savefig(plotdir + 'numYngStars_radius%s.png' % suffix)

    ## For a 1/r^2 surface density profile, where does the sample split into thirds?
    #nTot = len(yngNames)
    #rad = np.arange()
    #sd = 1 / rad**2




#--------------------------------------------------
#
# Helper functions
#
#--------------------------------------------------
def loadStarsetAbs(alignRoot, relErr=1):
    t = objects.Transform()
    t.loadAbsolute()

    s = starset.StarSet(alignRoot, trans=t, relErr=relErr)

    return s

def fitPlaw(params, x, y):
    A = params[0]
    alpha = params[1]
    const = params[2]

    model = A * (x**alpha) + const
    return (y - model)


def whereInDisk(pdf, angleCut=12.5):
    # Determine angular offset to disk for each solution
    tmp = pdf.i / rad2deg

    sini = np.sin(tmp)
    cosi = np.cos(tmp)

    sinip = np.sin(np.radians(idisk))
    cosip = np.cos(np.radians(idisk))

    cosodiff = np.cos( (pdf.o - odisk) / rad2deg )

    angle = np.arccos( (sini * sinip * cosodiff) + (cosi * cosip) )
    angle *= rad2deg
            
    idx = (np.where(angle < angleCut))[0]

    return idx
            
def diskDeproject(p, q, n, idisk=idisk, odisk=odisk):
    """
    Switch everything into the plane-of-the-sky coordinate system. Coordinates
    are p, q, n where
      n is tangential to the disk plane.
      p goes along the ascending node in the disk plane.
      q is perpindicular to p in the disk plane.
    Get the directional vectors
    """
    irad = np.radians(idisk)
    orad = np.radians(odisk)
    nhat = np.array([ np.sin(irad) * np.cos(orad),
                   -np.sin(irad) * np.sin(orad),
                   -np.cos(irad)], float)
    phat = util.cross_product(nhat, np.array([0.0, 0.0, -1.0]))
    qhat = util.cross_product(nhat, phat)

    # Derotate
    tmp = nhat[0]**2 + nhat[1]**2
    xhat = (1.0/tmp) * np.array([-nhat[1], -nhat[0]*nhat[2], nhat[0]*tmp])
    yhat = (1.0/tmp) * np.array([ nhat[0], -nhat[1]*nhat[2], nhat[1]*tmp])
    zhat = np.array([0, 1, nhat[2]])

    x = (p*xhat[0]) + (q*xhat[1]) + (n*xhat[2])
    y = (p*yhat[0]) + (q*yhat[1]) + (n*yhat[2])
    z = (p*zhat[0]) + (q*zhat[1]) + (n*zhat[2])

    return (x, y, z)



def loadDiskDensity(npix, orbDir='aorb_acc/', aperture=False,
                    file1='disk.neighbor.dat', file2='disk.neighborStd.dat',
                    singlePdf=True, simdisk=False, silent=False):
    """
    Load up the maps for the average disk density and the
    standard deviation of the disk density from the MC simulations.
    """
    
    ##########
    #
    # Nearest Neighbor
    #
    ##########
    if simdisk == False:
        f1 = root + alnDir + orbDir + file1
        f2 = root + alnDir + orbDir + file2
    else:
        f1 = orbDir + file1
        f2 = orbDir + file2
    allMaps = np.fromfile(f1, dtype=float)
    allStds = np.fromfile(f2, dtype=float)

    if singlePdf != True:
        allMaps = allMaps.reshape((4, npix))
        allStds = allStds.reshape((4, npix))

        # Read the N=6 nearest neighbor analysis results
        disk = allMaps[2]
        diskStd = allStds[2]
    else:
        allMaps = allMaps.reshape((1, npix))
        allStds = allStds.reshape((1, npix))

        # Read the N=6 nearest neighbor analysis results
        disk = allMaps[0]
        diskStd = allStds[0]

    ##########
    #
    # Aperture 
    #
    ##########
    if (aperture == True):
        f1 = root + alnDir + orbDir + 'disk.aperture.dat'
        f2 = root + alnDir + orbDir + 'disk.apertureStd.dat'
        allMaps = np.fromfile(f1, dtype=float)
        allStds = np.fromfile(f2, dtype=float)
        
        allMaps = allMaps.reshape((1, npix))
        allStds = allStds.reshape((1, npix))
        
        disk = allMaps[0]
        diskStd = allStds[0]
        
    if silent != True:
        print 'loadDiskDensity: from file %s' % (f1)

    return (disk, diskStd)


def solidAngleOfCone(radius, radians=False):
    """
    Calculate the solid angle for a cone with the specified radius. Return
    in either deg^2 or in steradians. Input radius in degrees.
    """
    cosRad = math.cos(math.radians(radius))

    solidAngleRad = 2.0 * math.pi * (1.0 - cosRad)  # in steradians
    solidAngleDeg = solidAngleRad * rad2deg**2      # in deg^2

    if (radians == True):
        return solidAngleRad
    else:
        return solidAngleDeg


def radiusOfCone(solidAngleDeg, radians=False):
    """
    Calculate the radius of a cone with the specified solid angle. Return
    in either degrees or radians. Input solid angle in square-degrees.
    """

    solidAngleRad = solidAngleDeg / rad2deg**2

    radiusRad = math.acos(1.0 - (solidAngleRad / (2.0 * math.pi)))
    radiusDeg = radiusRad * rad2deg

    if (radians == True):
        return radiusRad
    else:
        return radiusDeg

def diskCorrectFOV(rBins):
    """
    Calculate for each radius the correction factor for our
    limited field-of-view. Mutliply any densities by the correction
    factor at this radius to see what a complete azimuthally
    averaged density should be.

    Input:
    rBins -- array of radial bins.
    """
    ##########
    #
    # Figure out the correction factor for FOV effects.
    # Do this brute force by gridding and integrating.
    #
    ##########
    # First divide up the disk plane into small segments for integrating.
    coordP = np.arange(-0.6, 0.6, 0.001) # pc
    coordQ = np.arange(-0.6, 0.6, 0.001)

    (gridP, gridQ) = np.meshgrid(coordP, coordQ)
    (gridX, gridY, gridZ) = diskDeproject(gridP, gridQ, gridP*0.0)

    # Get FOV constraints
    (cornersX, cornersY, cornersZ) = getDiskCornersXYZ()    
    (innerP, innerQ, innerN) = getDiskInnerEdgePQN()
    innerRadius = getDiskInnerRadius()

    # Now check which grid points are in the FOV
    diskRadius = np.sqrt(gridP**2 + gridQ**2)
    rStep = rBins[1] - rBins[0]

    compFactor = np.zeros(len(rBins), dtype=float)
    for rr in range(len(rBins)):
	rlo = rBins[rr]
	rhi = rBins[rr] + rStep

	rdx = (np.where((diskRadius.flat >= rlo) & (diskRadius.flat < rhi)))[0]
	
	totalCnt = len(rdx)
	
	x = gridX.flat[rdx]
	y = gridY.flat[rdx]
	r = np.sqrt(x**2 + y**2)

	inFOVidx = (np.where((x > cornersX[0]) & (x < cornersX[2]) & 
		       (y > cornersY[0]) & (y < cornersY[1]) & 
		       (r > innerRadius)))[0]

	inFOVcnt = len(inFOVidx)

	if (inFOVcnt == 0):
	    print 'At r = %4.2f, nothing in the FOV.' % (rlo)
	else:
	    compFactor[rr] = float(totalCnt) / inFOVcnt

	print 'At r = [%4.2f - %4.2f], %5d out of %5d pixels (factor = %f)' % \
	    (rlo, rhi, inFOVcnt, totalCnt, compFactor[rr])

    return compFactor

def getDiskInnerRadius():
    """
    Get the inner radius in the plane-of-the-sky of our field-of-view
    in units of parsecs.
    """
    cc = objects.Constants()

    innerRadius = 0.8 * dist / cc.au_in_pc

    return innerRadius

def getFOVinXYZ():
    # Get the FOV coverage from spectroscopy, combining both
    # OSIRIS and SINFONI data. Use SINFONI field coverage from
    # Bartko et al. (2009)

    # the FOV is an oddly-shaped polygon, so we'll create 3
    # separate rectangle grids and then combine them
    x1 = np.arange(-0.31, 0.50, 0.001) # pc
    y1 = np.arange(-0.23, 0.29, 0.001) # pc
    (gx1, gy1) = np.meshgrid(x1, y1)

    x2 = np.arange(-0.39, 0.27, 0.001) # pc
    y2 = np.arange(-0.31, -0.08, 0.001) # pc
    (gx2, gy2) = np.meshgrid(x2, y2)

    x3 = np.arange(-0.31, 0.23, 0.001) # pc
    y3 = np.arange(0.29, 0.50, 0.001) # pc
    (gx3, gy3) = np.meshgrid(x3, y3)

    py.figure(figsize=(6,6))
    py.clf()
    py.plot(gx1,gy1,'r-')
    py.plot(gx2,gy2,'g-')
    py.plot(gx3,gy3,'b-')
    py.axis([0.51, -0.4, -0.32, 0.51])
    py.savefig(plotdir + 'spec_fov_rough.png')
    pdb.set_trace()
    

def getDiskCornersXYZ():
    cc = objects.Constants()

    #cornersX = array([-2.90, -2.90, 3.92, 3.92, -2.90])
    #cornersY = array([-3.22, 3.05, 3.05, -3.22, -3.22])
    #cornersX = np.array([-13.51, -13.51, 13.67, 13.67, -13.51])
    #cornersY = np.array([-14.51, 12.25, 12.25, -14.51, -14.51])
    cornersX = np.array([-10.0, -10.0, 13.67, 13.67, -10.0])
    cornersY = np.array([-10.0, 13.0, 13.0, -10.0, -10.0])

    cornersZ = np.array([aorb.plane2z(cornersX[i], cornersY[i])
                         for i in range(len(cornersX))])

    # Convert positions into parsecs
    cornersX *= dist / cc.au_in_pc
    cornersY *= dist / cc.au_in_pc
    cornersZ *= dist / cc.au_in_pc

    return (cornersX, cornersY, cornersZ)


def getDiskCornersPQN(idisk=idisk, odisk=odisk):
    """
    Get corners of the FOV covered by this survey. Returns coordinates
    for the corners in the disk plane in parsecs.
    """
    cc = objects.Constants()

    #cornersX = array([-2.90, -2.90, 3.92, 3.92, -2.90])
    #cornersY = array([-3.22, 3.05, 3.05, -3.22, -3.22])
    #cornersX = np.array([-13.51, -13.51, 13.67, 13.67, -13.51])
    #cornersY = np.array([-14.51, 12.25, 12.25, -14.51, -14.51])
    cornersX = np.array([-10.0, -10.0, 13.67, 13.67, -10.0])
    cornersY = np.array([-10.0, 13.0, 13.0, -10.0, -10.0])

    cornersZ = np.array([aorb.plane2z(cornersX[i], cornersY[i])
                         for i in range(len(cornersX))])

    (cornersP, cornersQ, cornersN) = \
               diskProject(cornersX, cornersY, cornersZ,
                           idisk=idisk, odisk=odisk)

    # Convert positions into parsecs
    cornersP *= dist / cc.au_in_pc
    cornersQ *= dist / cc.au_in_pc
    cornersN *= dist / cc.au_in_pc

    return (cornersP, cornersQ, cornersN)

def getDiskInnerEdgePQN(idisk=idisk, odisk=odisk):
    """
    Return an array that can be drawn on the disk plane to
    represent the inner edge of our FOV. Returns in parsecs.
    """
    cc = objects.Constants()

    # Need to determine the inner annulus
    innerRadius = 0.8 # arcsec    
    x_tmp = np.arange(-innerRadius, innerRadius, 0.05)
    y_tmp = np.sqrt(innerRadius**2 - x_tmp**2)

    innerX = np.concatenate((x_tmp, -x_tmp))
    innerY = np.concatenate((y_tmp, -y_tmp))

    innerZ = np.array([aorb.plane2z(innerX[i], innerY[i])
                       for i in range(len(innerX))])

    (innerP, innerQ, innerN) = diskProject(innerX, innerY, innerZ,
                                           idisk=idisk, odisk=odisk)

    # Convert positions into parsecs
    innerP *= dist / cc.au_in_pc
    innerQ *= dist / cc.au_in_pc
    innerN *= dist / cc.au_in_pc

    return (innerP, innerQ, innerN)


def readDiskProb(diskOnly=False,suffix='_mosaic',diskCut=3.0,top20=False):
    """
    diskCut = 1, 2, or 3 sigma likelihood that the star is not on the disk
    top20 -- select only the top 20% of candidate disk members; if this is set to
     	     True, then diskCut parameter is ignored.
    """

    lhNotOnDisk_cut = scipy.special.erf(diskCut/np.sqrt(2.))
    probOnDisk_cut = 1.0 - lhNotOnDisk_cut

    # Read in the table of disk membership probabilities
    mmbrFile = root + alnDir + 'tables/disk_membership_prob'+suffix+'.dat'
    diskTab = asciidata.open(mmbrFile)
    print 'Reading disk membership probability from %s' % mmbrFile
    names = [diskTab[0][ss].strip() for ss in range(diskTab.nrows)]
    diskP = diskTab[1].tonumpy()

    idx = (np.where(diskP == 0))[0]
    diskP[idx] = 1.0e-5

    print probOnDisk_cut
    # Trim down to disk stars.
    if (diskOnly == True):
        if top20 == False:
            idx = (np.where(diskP > probOnDisk_cut))[0]
        else:
            num = np.floor(len(diskP) * 0.2)
            idx = diskP.argsort()[::-1][0:int(num)] # reverse sort

        names = [names[ii] for ii in idx]
        diskP = diskP[idx]

    return (names, diskP)

def simWhereInDisk(incl, Omega, angleCut=15.2):
    # For the simulations:
    # Determine angular offset to disk for each solution
    tmp = incl / rad2deg

    sini = np.sin(tmp)
    cosi = np.cos(tmp)

    sinip = np.sin(np.radians(idisk))
    cosip = np.cos(np.radians(idisk))

    cosodiff = np.cos( (Omega - odisk) / rad2deg )

    angle = np.arccos( (sini * sinip * cosodiff) + (cosi * cosip) )
    angle *= rad2deg

    idx = None
    try:
        idx = (np.where(angle < angleCut))[0]
    except ValueError:
        print 'No disk solutions'

    return idx


def loadYoungByName(names, mosaic=False):
    # Load names of young stars 
    if mosaic == True:
        # Load up mosaic data as well; select only stars at r>4, since
        # we don't want to add any info from mosaics if we have it in
        # the central 10" already
        yng1 = young.loadYoungStars(root+alnDir,fit=poly,points=points,
                                    withRVonly=True,skipStar=['S5-237']) 
        yng2 = young.loadYoungStars(mscDir,fit=polyM,points=pointsM,
                                    mosaic=True, withRVonly=True)
        # Merge this object with object from central 10" analysis
        yng = merge(yng1, yng2)
    else:
        yng = young.loadYoungStars(root+alnDir,fit=poly,points=points,withRVonly=True) 

    # Trim down to just disk stars
    stars = []
    yngNames = yng.getArray('name')
    
    for dd in range(len(names)):
        try:
            idx = yngNames.index(names[dd])
            stars.append(yng.stars[idx])
        except ValueError:
            print 'loadYoungByName: Did not find %s' % (names[dd])
            continue

    yng.stars = stars

    return yng


def makePdfHealpix(name, pdf, outroot, ntrials, nside=64, sim=True,
                   makeplot=False):
    """
    Make a 2D histogram of the inclination and PA to the 
    ascending node. The points of the monte carlo are distributed
    on a HEALPix map of the sky.
    """
    npix = healpy.nside2npix(nside)

    # Determine which pixel in the map each of the
    # points goes (2D histogram)
    incl = pdf.i * math.pi / 180.0
    omeg = pdf.o * math.pi / 180.0

    hidx = healpy.ang2pix(nside, incl, omeg)

    # Star's PDF
    pdf = np.zeros(npix, dtype=float)
    for hh in hidx:
        pdf[hh] += 1.0  
    pdf /= ntrials

    py.clf()
    if (makeplot):
        if sim == True:
            mcFile = '%s%s_disk_mc_heal.dat' % (outroot, name)
        else:
            mcFile = '%s%s_mc_heal.dat' % (outroot, name)
        pdf.tofile(mcFile)

        pdh.go(mcFile, npix, 1)
            
    return pdf


# To open all the plots for maser mosaic stars not in central arcsecond:
#open S10-32_mc_heal.dat.png S10-34_mc_heal.dat.png S10-48_mc_heal.dat.png S10-4_mc_heal.dat.png S10-50_mc_heal.dat.png S10-5_mc_heal.dat.png S10-7_mc_heal.dat.png S11-21_mc_heal.dat.png S11-5_mc_heal.dat.png S13-3_mc_heal.dat.png S4-364_mc_heal.dat.png S5-231_mc_heal.dat.png S5-235_mc_heal.dat.png S5-236_mc_heal.dat.png S6-100_mc_heal.dat.png S6-81_mc_heal.dat.png S6-82_mc_heal.dat.png S6-90_mc_heal.dat.png S6-93_mc_heal.dat.png S6-95_mc_heal.dat.png S6-96_mc_heal.dat.png S7-10_mc_heal.dat.png S7-16_mc_heal.dat.png S7-19_mc_heal.dat.png S7-20_mc_heal.dat.png S7-30_mc_heal.dat.png S7-36_mc_heal.dat.png S8-15_mc_heal.dat.png S8-4_mc_heal.dat.png S8-7_mc_heal.dat.png S9-114_mc_heal.dat.png S9-13_mc_heal.dat.png S9-1_mc_heal.dat.png S9-20_mc_heal.dat.png S9-23_mc_heal.dat.png S9-9_mc_heal.dat.png

def confused_epochs():
    """
    Counts how many epochs were removed for confusion
    """
    _cfile = root + alnDir + 'points_c/epochsRemoved.txt'

    cfile = open(_cfile)

    yng1 = young.loadYoungStars(root+alnDir,fit=poly,points=points,
                                withRVonly=True,silent=True,skipStar=['S5-237']) 
    yng2 = young.loadYoungStars(mscDir,fit=polyM,points=pointsM,
                                mosaic=True, withRVonly=True,silent=True)
    cntrlNames = yng1.getArray('name')
    mscNames = yng2.getArray('name')
    # Merge this object with object from central 10" analysis
    yng = merge(yng1, yng2)
    yngNames = yng.getArray('name')

    numStars = 0
    numEps = 0
    numYng = 0
    numYngEps = 0
    for cc in range(2138):
        conf = cfile.readline().split()
        star = conf[0]
        cEps = int(conf[1])

        if cEps == 0:
            continue

        numStars += 1
        numEps += cEps

        if star in yngNames:
            print star, cEps
            numYng += 1
            numYngEps += cEps
    
    print 'Number of stars affected by confusion: %d' % numStars
    print 'Total number of confusion events: %d' % numEps
    print
    print 'Number of YOUNG stars affected by confusion: %d' % numYng
    print 'Total number of confusion events for YOUNG stars: %d' % numYngEps


def get_local_dist():
    """
    Compares the errors in the *rms.lis and *rms_ld.lis to figure
    out what the local distortion error added was for the 3
    non-2006 epochs.
    """
    eps = ['04jullgs', '05junlgs', '05jullgs']

    import matplotlib.font_manager
    prop = matplotlib.font_manager.FontProperties(size=12)
    usetexTrue()
    py.figure(figsize=(12,4))
    py.subplots_adjust(left=0.08,right=0.97,top=0.9,bottom=0.15,
                       wspace=0.3, hspace=0.3)
    py.clf()
    binsIn = np.arange(0, 1, 0.05)
    yt1 = [80, 150, 400]
    yt2 = [70, 130, 350]
    for ee in range(len(eps)):
        rms = asciidata.open('%s%slis/mag%s_kp_rms.lis' % \
                             (root, alnDir, eps[ee]))
        rmsld = asciidata.open('%s%slis/mag%s_kp_rms_ld.lis' % \
                               (root, alnDir, eps[ee]))

        xe1 = rms[5].tonumpy()
        ye1 = rms[6].tonumpy()
        xe2 = rmsld[5].tonumpy()
        ye2 = rmsld[6].tonumpy()

        ldxe = np.sqrt(xe2**2 - xe1**2)
        ldye = np.sqrt(ye2**2 - ye1**2)

        py.subplot(1, 3, ee+1)
        py.hist(ldxe, bins=binsIn, color='r', histtype='step', label='X')
        py.hist(ldye, bins=binsIn, color='b', histtype='step', label='Y')
        py.text(0.5, yt1[ee], 'Med xe = %5.3f' % np.median(ldxe), fontsize=12)
        py.text(0.5, yt2[ee], 'Med ye = %5.3f' % np.median(ldye), fontsize=12)
        py.xlabel('Local Distortion Error (pix)')
        py.ylabel('N')
        py.title('%s (N = %i)' % (eps[ee], len(xe1)), fontsize=14)
        if ee == 0:
            py.legend(numpoints=1,fancybox=True,prop=prop)

    py.savefig(plotdir + 'local_dist_errors.png')

    py.close()
    usetexFalse()


def speckle_frame_percentage():
    """
    Determines the fraction of speckle frames a star was detected in.
    Will help determine at what point the edge of the speckle FOV is
    problematic.

    """
    rootDir = root + alnDir
    table = asciidata.open(rootDir + '/scripts/epochsInfo.txt')

    # List of columns in the table. Make an array for each one.
    epoch    = [table[0][ss].strip() for ss in range(table.nrows)]
    numEpochs = len(epoch)

    s = starset.StarSet(rootDir + 'align/align_d_rms_1000_abs_t')
    s.loadPolyfit(rootDir + 'polyfit_c/fit', accel=0, arcsec=0)
    s.loadPolyfit(rootDir + 'polyfit_c/fit', accel=1, arcsec=0)

    names = s.getArray('name')
    r2d = s.getArray('r2d')
    mag = s.getArray('mag')
    years = s.getArray('years')[0]

    # Read the align*param file to figure out fraction of speckle
    # frames a star was detected in for a given epoch
    f_par = open(rootDir + 'align/align_d_rms_1000_abs_t.param', 'r')
    #tab = asciidata.open(rootDir + '/scripts/epochsInfo.txt')

    # The first line is for 16C, which is in every speckle frame
    # Every 4th column gives the number of frames for that epoch
    par = f_par.readline()
    n16c = [par.split()[cc] for cc in range(3,len(par.split()),4)]
    n16c = np.array([float(nn) for nn in n16c])

    nameF = ['S0-15', 'S1-12', 'S1-14', 'S1-3', 'irs16SW', 'irs16C',
             'S1-2', 'S1-24', 'S1-8', 'S2-16', 'S3-10', 'irs16CC']
    problem = ['','','','','','',
               ' (confused)',' (edge)',' (confused)',' (edge)',' (edge)',' (confused)']
    nEps = [43,36,40,45,45,45,
            30,30,41,23,19,19]
    # Start the loop at index 1 b/c we've read in first line already
    py.figure()
    py.figure(figsize=(8,14))
    py.subplots_adjust(bottom=0.05,right=0.96,left=0.12,top=0.95,
                       hspace=0.1,wspace=0.1)
    py.clf()
    cntr = 0
    cntr2 = 0
    for i in range(1,len(names)):
        star = names[i]
        par = f_par.readline()

        nframes = [par.split()[cc] for cc in range(3,len(par.split()),4)]
        nframes = np.array([float(nn) for nn in nframes])
        frac = nframes / n16c

        if star in nameF:
            nn = nameF.index(star)
            py.subplot(len(nameF),1,nn+1)
            py.plot(np.arange(len(years)), frac, 'k.')
            py.text(35,0.5,star+problem[nn],fontsize=12,color='k')
            py.text(35,0.2,'N='+str(nEps[nn]),fontsize=12)
            yinc=0.25
            thePlot = py.gca()
            thePlot.get_yaxis().set_major_locator(py.MultipleLocator(yinc))
            vis = False
            if star == 'irs16CC':
                vis = True
            py.setp( thePlot.get_xticklabels(), fontsize=12, visible=vis)
            py.setp( thePlot.get_yticklabels(), fontsize=12)
            py.axis([0,45,0,1.05])
            cntr +=1.5
            cntr2 +=1

            if nn == len(nameF)-1:
                py.xlabel('Epoch')

    # Also plot irs16C, which is the reference
    py.subplot(len(nameF),1,6)
    py.plot(np.arange(len(years)), n16c/n16c, 'k.')
    py.axis([0,45,0,1.05])
    thePlot = py.gca()
    py.setp( thePlot.get_xticklabels(), fontsize=12, visible=False)
    py.text(35,0.5,'irs16C',fontsize=12,color='k')
    py.text(35,0.2,'N=45',fontsize=12,color='k')

    py.suptitle('Fraction of Frames')
    py.savefig(root+alnDir+'plots/frame_fraction_epoch.png')
    py.close()
                

def speckle_edge():
    """
    Loops thru young stars points files and determines how many star detections
    were affected by the edge of the FOV and subsequently thrown out.
    """
    yng = young.loadYoungStars(root+alnDir,fit=poly,points=points,
                                withRVonly=True,silent=True,skipStar=['S5-237']) 
    names = yng.getArray('name')
    
    p1 = root + alnDir + 'points_1000/'
    p2 = root + alnDir + 'points_s/'

    stars_affected = 0
    det_dropped = 0
    print 'Stars affected by edge of speckle:'
    print
    print '%10s  %5s  %7s' % ('Star', 'N_det', 'N_final')
    for yy in range(len(names)):
        pts1 = asciidata.open(p1 + names[yy] + '.points')
        pts2 = asciidata.open(p2 + names[yy] + '.points')

        num1 = len(pts1[0].tonumpy())
        num2 = len(pts2[0].tonumpy())

        if ((num1 == num2) | ((num1 - num2) < 14)):
            continue
        else:
            print '%10s  %5i  %7i' % (names[yy], num1, num2)
            stars_affected += 1
            det_dropped += (num1-num2)

    print
    print 'Total number of star affected: %i' % stars_affected
    print 'Total number of points dropped: %i' % det_dropped


def compare_accel_holography():
    """
    Compare accelerations and their errors in my main align to those
    in the holography align (../test_schoedel/12_05_22/).
    """

    
    main = asciidata.open(root + alnDir + 'ftest_accel.txt')
    name1 = main[0].tonumpy()
    frm1 = main[1].tonumpy()
    xfp1 = main[2].tonumpy()
    yfp1 = main[3].tonumpy()
    ax1 = main[4].tonumpy()
    ay1 = main[5].tonumpy()
    ar1 = main[6].tonumpy()
    are1 = main[7].tonumpy()
    at1 = main[9].tonumpy()
    ate1 = main[10].tonumpy()

    holo = asciidata.open(root + alnDir + 'ftest_accel_holography.txt')
    name2 = holo[0].tonumpy()
    frm2 = holo[1].tonumpy()
    xfp2 = holo[2].tonumpy()
    yfp2 = holo[3].tonumpy()
    ax2 = holo[4].tonumpy()
    ay2 = holo[5].tonumpy()
    ar2 = holo[6].tonumpy()
    are2 = holo[7].tonumpy()
    at2 = holo[9].tonumpy()
    ate2 = holo[10].tonumpy()

    py.figure(figsize=(10,6))
    py.subplots_adjust(left=0.1, right=0.95, top=0.9,hspace=0.3,wspace=0.3)
    py.clf()
    py.subplot(1,2,1)
    py.errorbar(ar1,ar2,xerr=are1,yerr=are2,fmt='k.')
    py.plot([-0.35,0.0],[-0.35,0.0],'k--')
    py.axis([-0.35,0.0,-0.35,0.0])
    py.xlabel('Standard Align')
    py.ylabel('Align w/ Holography')
    py.title('Radial Accelerations (mas/yr/yr)')
    py.subplot(1,2,2)
    py.errorbar(at1,at2,xerr=ate1,yerr=ate2,fmt='k.')
    py.plot([-0.15,0.05],[-0.15,0.05],'k--')
    py.axis([-0.15,0.05,-0.15,0.05])
    py.xlabel('Standard Align')
    py.ylabel('Align w/ Holography')
    py.title('Tangential Accelerations (mas/yr/yr)')
    py.savefig(root + alnDir + 'plots/compare_accel_holography.png')


def plot_disk_orbits(mosaic=True,prdCut=1000.,accelOnly=False,suffix=''):
    """
    Plots orbits of stars that are most likely to be on the disk,
    (those with probability > 0.1). Assumes the solution that is closest
    to the disk solution.

    This is just for visualization purposes and possibly for defense talk.
    """
    orbDir = 'aorb_thesis/'

    # Load names of young stars 
    if mosaic == True:
        yng1 = young.loadYoungStars(root+alnDir,fit=poly,points=points,
                                    withRVonly=True,silent=True,skipStar=['S5-237']) 
        yng2 = young.loadYoungStars(mscDir,fit=polyM,points=pointsM,
                                    mosaic=True, withRVonly=True,silent=True)
        # Merge this object with object from central 10" analysis
        yng = merge(yng1, yng2)
    else:
        yng = young.loadYoungStars(root+alnDir,fit=poly,points=points,
                                   withRVonly=True,silent=True)

    yngNames = yng.getArray('name')
    r2d = yng.getArray('r2d')

    diskTab = asciidata.open(root+alnDir+'tables/disk_membership_prob_mosaic.dat')
    name = [diskTab[0][ss].strip() for ss in range(diskTab.nrows)]
    diskP = diskTab[1].tonumpy()
    diskIdx = (np.where(diskP > 0.1))[0]
    print 'Number of disk candidates: %i' % len(diskIdx)

    # Setup i/Omega for each pixel on sky
    nside = 64
    npix = healpy.nside2npix(nside)
    (iheal, oheal) = healpy.pix2ang(nside, np.arange(0, npix))
    iheal *= rad2deg
    oheal *= rad2deg

    #starCnt = len(diskIdx)
    numTrials = 99998

    #param = np.zeros((starCnt, numTrials), float)
    #angle = np.zeros((starCnt, numTrials), float)

    sinip = np.sin(np.radians(idisk))
    cosip = np.cos(np.radians(idisk))

    # Array of time steps (0.1 yr steps)
    years = prdCut
    t = np.arange(1995.5, 1995.5+years, 0.1, dtype=float)

    skipStar = ['S1-22', 'S1-24', 'S5-34']
    
    _acc = asciidata.open(root + alnDir + 'tables/accelerating_sources.dat')
    accels = _acc[0].tonumpy()
    acc = [aa.strip() for aa in accels]

    usetexTrue()
    py.close('all')
    py.clf()
    py.figure(figsize=(7,5))
    py.subplots_adjust(left=0.12,right=0.95,top=0.95,bottom=0.12)
    py.plot([0], [0], 'rx', ms=5, mew=2)

    colors = ['orange', 'red', 'turquoise', 'steelblue', 'plum', 'navy',
              'green', 'blue', 'cyan', 'magenta', 'purple', 'mediumorchid',
              'deeppink', 'tomato', 'salmon', 'lightgreen', 'sienna']*4
    fmt = '%12s  %8.2f  %8.2f  %8.2f  %5.3f  %6.2f  %12.2f  %12.2f  %8.4f'
    hdr = '%12s  %8s  %8s  %8s  %5s  %6s  %12s  %12s  %8s'
    print
    print hdr % ('Name', 'P (yrs)', 'a (mas)', 't0', 'e', 'i (deg)',
                 'Omega (deg)', 'omega (deg)', 'L_disk')
              
    cc = 0
    # Loop through and trim down to only disk stars
    for ii in range(len(yngNames)):

        if accelOnly == False:
            if (ii not in diskIdx) | (yngNames[ii] in skipStar):
                continue
        elif accelOnly == True:
            if yngNames[ii] not in acc:
                continue

        yngStar = yng.stars[ii]

        # File contains analytic orbit solutions with acceleration limits (MC)
        pdffile = '%s%s%s%s.mc.dat' % (root, alnDir, orbDir, yng.stars[ii].name)
        pdf = pickle.load(open(pdffile))

        # Determine angular offset to disk for each solution
        sini = np.sin(pdf.i * np.pi / 180.0)
        cosi = np.cos(pdf.i * np.pi / 180.0)
        cosodiff = np.cos( (pdf.o - odisk) * np.pi / 180.0 )
        angle = np.arccos( (sini * sinip * cosodiff) + (cosi * cosip) )
        angle *= 180.0 / np.pi

        # Find minimum angle to disk solution
        idx = np.where(angle == angle.min())[0]

        # Create an orbit object for this star. We will
        # use this to make the model orbits later on
        yngStar.orbit = orbits.Orbit()

        # Now assume this is the true orbit
        yngStar.orbit.i = pdf.i[idx]
        yngStar.orbit.o = pdf.o[idx]
        yngStar.orbit.w = pdf.w[idx]
        yngStar.orbit.e = pdf.e[idx]
        yngStar.orbit.p = pdf.p[idx]
        yngStar.orbit.t0 = pdf.t0[idx]

        # Just get short period orbits for plotting
        if (yngStar.orbit.p > prdCut) & (accelOnly == False):
            continue

        #print 'Plotting star %s with color %s' % (yngStar.name, colors[cc])

        # Get the x,y positions for plotting
        (r, v, a) = yngStar.orbit.kep2xyz(t, mass=mass, dist=dist)

        xpos = r[:,0].copy()
        ypos = r[:,1].copy()

        # semi-major axis
        sma = (yngStar.orbit.p**2 * mass)**(1.0/3.0) # AU
        sma = sma / dist * 1.e3 # mas 
        
        print fmt % (yng.stars[ii].name, yngStar.orbit.p, sma, yngStar.orbit.t0,
                     yngStar.orbit.e, yngStar.orbit.i, yngStar.orbit.o, yngStar.orbit.w,
                     diskP[ii])

        # Plot this orbit
        py.plot(xpos, ypos, color=colors[cc])
        cc += 1

    print
    print 'Plotted %i stars' % cc
    if accelOnly == True:
        py.axis([3, -3, -2, 2])
    else:
        py.axis([10, -10, -4.5, 4.5])
    py.xlabel('RA Offset (arcsec)')
    py.ylabel('Dec Offset (arcsec)')
    py.savefig('%sdisk_star_best_orbits%s.png' % (plotdir, suffix), dpi=300)
    py.axis('off')
    py.savefig('%sdisk_star_best_orbits_trans%s.png' % (plotdir, suffix), transparent=True, dpi=300)
    usetexFalse()


def plot_rv_from_orb(suffix=''):
    
    orbDir = 'aorb_thesis/'

    # Load names of young stars 
    yng = young.loadYoungStars(root+alnDir,fit=poly,points=points,
                               withRVonly=True,silent=True)

    yngNames = yng.getArray('name')
    r2d = yng.getArray('r2d')

    diskTab = asciidata.open(root+alnDir+'tables/disk_membership_prob_mosaic.dat')
    name = [diskTab[0][ss].strip() for ss in range(diskTab.nrows)]
    diskP = diskTab[1].tonumpy()
    diskIdx = (np.where(diskP > 0.1))[0]
    print 'Number of disk candidates: %i' % len(diskIdx)

    # Setup i/Omega for each pixel on sky
    nside = 64
    npix = healpy.nside2npix(nside)
    (iheal, oheal) = healpy.pix2ang(nside, np.arange(0, npix))
    iheal *= rad2deg
    oheal *= rad2deg

    numTrials = 99998

    sinip = np.sin(np.radians(idisk))
    cosip = np.cos(np.radians(idisk))

    angleDiff = 0.1

    # Array of time steps (0.1 yr steps)
    years = 50.
    t = np.arange(1995.5, 1995.5+years, 0.1, dtype=float)

    _acc = asciidata.open(root + alnDir + 'tables/accelerating_sources.dat')
    accels = _acc[0].tonumpy()
    acc = [aa.strip() for aa in accels]

    cons = objects.Constants()
    asy_to_kms = dist * cons.cm_in_au / (1.e5 * cons.sec_in_yr)

    usetexTrue()
    py.clf()
    py.figure(1)
    py.figure(figsize=(6,6))
    py.subplots_adjust(left=0.15,right=0.95,top=0.95,bottom=0.12)

    colors = ['orange', 'red', 'turquoise', 'steelblue', 'plum', 'navy',
              'green', 'blue', 'cyan', 'magenta', 'purple', 'mediumorchid',
              'deeppink', 'tomato', 'salmon', 'lightgreen', 'sienna']*4
              
    cc = 0
    # Loop through and trim down to only disk stars
    for ii in range(len(yngNames)):

        if yngNames[ii] not in acc:
            continue

        yngStar = yng.stars[ii]

        # File contains analytic orbit solutions with acceleration limits (MC)
        pdffile = '%s%s%s%s.mc.dat' % (root, alnDir, orbDir, yng.stars[ii].name)
        pdf = pickle.load(open(pdffile))

        # Determine angular offset to disk for each solution
        sini = np.sin(pdf.i * np.pi / 180.0)
        cosi = np.cos(pdf.i * np.pi / 180.0)
        cosodiff = np.cos( (pdf.o - odisk) * np.pi / 180.0 )
        angle = np.arccos( (sini * sinip * cosodiff) + (cosi * cosip) )
        angle *= 180.0 / np.pi

        # Find minimum angle to disk solution
        idx = np.where(angle == angle.min())[0]

        # Create an orbit object for this star. We will
        # use this to make the model orbits later on
        yngStar.orbit = orbits.Orbit()

        # Now assume this is the true orbit
        yngStar.orbit.i = pdf.i[idx]
        yngStar.orbit.o = pdf.o[idx]
        yngStar.orbit.w = pdf.w[idx]
        yngStar.orbit.e = pdf.e[idx]
        yngStar.orbit.p = pdf.p[idx]
        yngStar.orbit.t0 = pdf.t0[idx]

        print 'Plotting star %s with color %s' % (yngStar.name, colors[cc])

        (r, v, a) = yngStar.orbit.kep2xyz(t, mass=mass, dist=dist)
        #xpos = r[:,0].copy()
        #ypos = r[:,1].copy()
        vz = v[:,2].copy() / 1.e3 * asy_to_kms

        # Plot this star's RV
        py.plot(t, vz, color=colors[cc])
        cc += 1

    thePlot = py.gca()
    thePlot.get_xaxis().set_major_locator(py.MultipleLocator(1))
    py.gca().xaxis.set_major_formatter(py.FormatStrFormatter('%4i'))
    py.axis([2008,2014,-800,800])
    py.xlabel('Date (years)')
    py.ylabel('Radial Velocity (km/s)')
    py.savefig('%sdisk_star_rv%s.png' % (plotdir, suffix))
    py.close(1)
    usetexFalse()


def test_uniform(amin=0.0,amax=1.0):

    gen = aorb.create_generators(1,10**8)
    agen = gen[0]

    aa = np.zeros((10**6), dtype=float)

    for ii in range(10**6):
        aa[ii] = agen.uniform(amin, amax)
        
    binsIn = np.arange(0,1.0,0.001)
    py.close('all')
    py.clf()
    py.hist(aa,bins=binsIn,histtype='step',normed=True)
    py.show()
    


def read_disk_mem(simRoot='sim_diskFraction3/'):

    frac = np.arange(0.05, 0.59, 0.05)

    for ii in range(len(frac)):
        print
        ff = np.around(frac[ii], decimals=2) # have to do this b/c sometimes a
        				     # dumb rounding error is made

        simDir = '%s%s%s/disk%s/' % (root, alnDir, simRoot, int(ff*100))
        dmFile = 'plots/HEALpixMaps/simdisk_membership_prob.dat'

        dm = asciidata.open(simDir + dmFile)
        diskP = dm[1].tonumpy()
 
        print 'Disk fraction = %2i' % int(ff*100)
        print 'Sum of probs = # disk members = %6.3f stars' % diskP.sum()
        print 
        


def g2_angle_offset(i,O,ierr,Oerr,
                 idisk=130.2,Odisk=96.3,iderr=2.0,Oderr=2.0):
    """
    For a given i and O, determine the angular offset to
    the disk solution
    """
    # Disk solution:
    sinip = np.sin(np.radians(idisk))
    cosip = np.cos(np.radians(idisk))

    # Input angle:
    sini = np.sin(i * np.pi / 180.0)
    cosi = np.cos(i * np.pi / 180.0)

    # Difference:
    cosodiff = np.cos( (O - Odisk) * np.pi / 180.0 )
    angle = np.arccos( (sini * sinip * cosodiff) + (cosi * cosip) )
    angle *= 180.0 / np.pi

    print 'Angular offset b/w disk and (i,O) = (%5.1f, %5.1f): %4.1f deg' % \
          (i, O, angle)


def fitfuncParabola(p, fjac=None, xdata=None, ydata=None, ydataerr=None):
    """Find residuals of fit.

    For 2nd degree polynomial, p should be list of form [A, B, C],
    while data should be list of form [xdata, ydata, ydataerr].
    """
    num = len(xdata)
    model = np.zeros(num, dtype=float)
    devs = np.zeros(num, dtype=float)

    # Set parameters
    A = p[0]
    B = p[1]
    C = p[2]

    model = A*(xdata**2) + B*xdata + C
    residuals = (ydata - model)/ydataerr
    status = 0

    return [status, residuals]


def fitParabola(p0=None,data=None,quiet=0):
    """Fits quadratic using mpfit.

    Inputs should be given as p0=[A, B, C] and
    data=[xdata,ydata,ydataerr]. Returns object of class
    mpfit.
    """

    print 'Initial Guess (A*x**2 + B*x + C):'
    print '   A     = %6.2f' % p0[0]
    print '   B     = %5.3f' % p0[1]
    print '   C     = %5.3f' % p0[2]

    # Set data to be passed to fit
    functargs = {'xdata':data[0],'ydata':data[1],'ydataerr':data[2]}

    # Set initial values and limits (no limits on parameters)
    pinfo = [{'value':0,'fixed':0,'limited':[0,0],
	      'limits':[0,0]}]*len(p0)
    for ii in range(len(p0)):
        pinfo[ii]['value'] = p0[ii]

    m = nmpfit_sy.mpfit(fitfuncParabola, p0, functkw=functargs, parinfo=pinfo,
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
    x   = (data[0])[np.nonzero(data[2])]  
    m.dof = len(x)-len(p) # Number of degrees of freedom

    print 'Final Solution (A*x**2 + B*x + C):'
    print '   A      = %6.2f +/- %5.2f' % (p[0],perr[0])
    print '   B      = %5.3f +/- %5.3f' % (p[1],perr[1])
    print '   C      = %5.3f +/- %5.3f' % (p[2],perr[2])
    print '   chi^2  = %5.2f' % m.fnorm
    print '   Rchi^2 = %5.2f' % Rchi2

    return m

def fitfuncPL(p, fjac=None, xdata=None, ydata=None, ydataerr=None):
    """Find residuals of fit.

    For power-law, p should be list of form [A, alpha],
    while data should be list of form [xdata, ydata, ydataerr].
    """
    num = len(xdata)
    model = np.zeros(num, dtype=float)
    devs = np.zeros(num, dtype=float)

    # Set parameters
    A = p[0]
    alpha = p[1]

    model = A*(xdata**alpha)
    residuals = (ydata - model)/ydataerr
    status = 0

    return [status, residuals]


def fitPowerLawMP(p0=None,data=None,quiet=0):
    """Fits power law using mpfit.

    Inputs should be given as p0=[A, alpha] and
    data=[rbins,vdisp,err]. Returns object of class
    mpfit.
    """

    print 'Initial Guess:'
    print '   A     = %6.2f' % p0[0]
    print '   alpha = %5.3f' % p0[1]

    # Set data to be passed to fit
    functargs = {'xdata':data[0],'ydata':data[1],'ydataerr':data[2]}

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
    x   = (data[0])[np.nonzero(data[2])]  
    m.dof = len(x)-len(p) # Number of degrees of freedom

    print 'Final Solution:'
    print '   A      = %6.2f +/- %5.2f' % (p[0],perr[0])
    print '   alpha  = %5.3f +/- %5.3f' % (p[1],perr[1])
    print '   chi^2  = %5.2f' % m.fnorm
    print '   Rchi^2 = %5.2f' % Rchi2

    return m


def fitSqrtFunc(p, fjac=None, xdata=None, ydata=None, ydataerr=None):
    """Find residuals of fit.

    For power-law, p should be list of form [A, alpha],
    while data should be list of form [xdata, ydata, ydataerr].
    """
    num = len(xdata)
    model = np.zeros(num, dtype=float)
    devs = np.zeros(num, dtype=float)

    # Set parameters
    A = p[0]

    model = A*(xdata**0.5)
    residuals = (ydata - model)/ydataerr
    status = 0

    return [status, residuals]


def fitSqrt(p0=None,data=None,quiet=0):
    """Fits for the scaling, forcing sqrt relation, using mpfit.

    Inputs should be given as p0=[A] and
    data=[rbins,vdisp,err]. Returns object of class
    mpfit.
    """

    print 'Initial Guess:'
    print '   A     = %6.2f' % p0[0]
    print '   alpha = 0.5'

    # Set data to be passed to fit
    functargs = {'xdata':data[0],'ydata':data[1],'ydataerr':data[2]}

    # Set initial values and limits (no limits on parameters)
    pinfo = [{'value':0,'fixed':0,'limited':[0,0],
	      'limits':[0,0]}]*len(p0)
    for ii in range(len(p0)):
        pinfo[ii]['value'] = p0[ii]

    m = nmpfit_sy.mpfit(fitSqrtFunc, p0, functkw=functargs, parinfo=pinfo,
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
    x   = (data[0])[np.nonzero(data[2])]  
    m.dof = len(x)-len(p) # Number of degrees of freedom

    print 'Final Solution:'
    print '   A      = %6.2f +/- %5.2f' % (p[0],perr[0])
    print '   chi^2  = %5.2f' % m.fnorm
    print '   Rchi^2 = %5.2f' % Rchi2

    return m
