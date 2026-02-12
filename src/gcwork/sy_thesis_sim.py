import asciidata, pyfits, pickle
import os, sys, math, time, mpfit
import __builtin__
from matplotlib import patches
import matplotlib.nxutils as nx
import numpy as np
import pylab as py
import random
import shutil
import healpy
from gcwork import starset
from gcwork import objects
from gcwork import util
from gcwork import young
from gcwork import orbits
from gcwork import analyticOrbits as aorb
from gcwork import starTables
from gcwork import plot_disk_healpix as pdh
from gcwork.plotgc import plotStar
import starTables as tabs
import syYoung 
import sythesis
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
#root = '/'
#alnDir = ''
poly='polyfit_c/fit'
points='points_c/'
plotdir = root + alnDir + 'plots/'
mscDir=root + '12_01_30_dp_msc/'
polyM='polyfit_noDistErr/fit'
pointsM='points_noDistErr/'
idisk=130.2
odisk=96.3
idiskerr=2.0
odiskerr=2.0
mbh = 4.6e6 # latest values from 11_10_26/efit
massErr = 0.72e6 # latest values from 11_10_26/efit
dist = 8232.9 # latest values from 11_10_26/efit
x0 = -0.001
y0 = -0.005
areaOnSky = 4.0 * math.pi * (180.0 / math.pi)**2    # in deg^2
rad2deg = 180.0 / math.pi


def mock_data(nstars=120, e=0.32, eSTD=0, mockdir='', outfile='',
              vel_kick=True, frac=0.07, tObs=None, isoPop=None,
              outerRadial=False,suffix=''):
    """
    Create mock data for nstars (def=100)  on orbits with a specified
    eccentricity (def = circular).

    Input:
       nstar (int):     Number of DISK stars to simulate. Measurement errors will be
       		        pulled for each star based on the star's projected radius.
       e (float [0,1]): Eccentricity of orbits; set to -1 to sample uniformly
       			between 0-1.
       eSTD (float): 	Standard deviation of eccentricity distribution, if not
       			all the same values.
       vel_kick (bool): If set to True, a velocity perturbation is added to the
       			mock data.
       frac (float):	Fraction of the local orbital velocity to be used as the velocity kick.
       tObs (string):   Set to 'apoapse' to get the mock data at apoapse
       			passage.
       isoPop (int):    To add an isotropic population on top of the disk,
       			set isoPop to a value corresponding to the number
                        of stars to include.
       outerRadial (bool): Set to True to allow only stars with r > 6.5 arcsec
    """

    ndisk = nstars # remember how many stars on disk

    # If simulating a disk plus an isotropic population, combine the nstars
    if isoPop != None:
        nstars += isoPop # make nstars the total of all stars

    # Assume a power law for surface density profile in disk
    # from sythesis.plotRadialDist()
    rstep = 0.01
    rad = np.arange(0.001, 1, rstep)
    dslope = 1.9 # slope for disk stars from my latest analysis, includes only disk candidates
    ndslope = 1.14 # slope for the non-disk stars (Do et al. 2013) 

    # Make all our random number generators. We will need
    # 6 all together (Omega, inclination, t0, omega, radius, ecc)
    #gens = create_generators(5, nstars*1000)
    if vel_kick == True:
        cg = 9
    else:
        cg = 6
    
    gens = create_generators(cg, nstars*1000)
    Ogen = gens[0]
    igen = gens[1]
    t0gen = gens[2]
    wgen = gens[3]
    plgen = gens[4]
    egen = gens[5]

    if vel_kick == True:
        dvxgen = gens[6]
        dvygen = gens[7]
        dvzgen = gens[8]

    cc = objects.Constants()
    asy_to_kms = dist * cc.cm_in_au / (1.e5*cc.sec_in_yr)
    
    epochs = np.array([1995.439, 1996.485, 1997.367, 1998.251, 1998.366, 1998.505,
              1998.590, 1998.771, 1999.333, 1999.559, 2000.381, 2000.548,
              2000.797, 2001.351, 2001.572, 2002.309, 2002.391, 2002.547,
              2003.303, 2003.554, 2003.682, 2004.327, 2004.564, 2004.567,
              2004.660, 2005.312, 2005.495, 2005.566, 2005.580, 2006.336,
              2006.470, 2006.541, 2007.374, 2007.612, 2008.371, 2008.562,
              2009.340, 2009.561, 2009.689, 2010.342, 2010.511, 2010.620,
              2011.401, 2011.543, 2011.642]) 

    # choose a reference epoch 
    # We normally use the weighted average of our epochs, weighted by
    # positional errors. Since we are assuming the same error for all
    # epochs, based on R2d, the weights are the same and we take the average:
    tref = [epochs.mean()]

    orb_all = []
    sma_all = []
    orb_all = []
    sma_all = []
    t0_all = []

    x = np.zeros((nstars), dtype=float)
    y = np.zeros((nstars), dtype=float)
    z = np.zeros((nstars), dtype=float)
    vx = np.zeros((nstars), dtype=float)
    vy = np.zeros((nstars), dtype=float)
    vz = np.zeros((nstars), dtype=float)
    ax = np.zeros((nstars), dtype=float)
    ay = np.zeros((nstars), dtype=float)
    az = np.zeros((nstars), dtype=float)
    ar = np.zeros((nstars), dtype=float)
    at = np.zeros((nstars), dtype=float)
    are = np.zeros((nstars), dtype=float)
    ate = np.zeros((nstars), dtype=float)
    #refTime = np.zeros((nstars), dtype=float)

    fmt = '%12s  %12.5e  %12.5e  %12.5e  %5.3f  %6.2f  %12.2f  %12.2f\n'
    hdr = '%12s  %12s  %12s  %12s  %5s  %6s  %12s  %12s\n'
    orbfile = root + alnDir + mockdir + 'orbital_elements'+suffix+'.dat'
    orb_elem = open(orbfile, 'w')
    orb_elem.write(hdr % ('#Name', 'P (yrs)', 'a (mas)', 't0', 'e', 'i (deg)',
                          'Omega (deg)', 'omega (deg)'))

    ecc_all = []
    nn = 0
    print nstars
    while nn < nstars:
        if nn < ndisk: 

            # Randomly select semi-major axis (radius for e=0) from uniform
            # distribution of surface density profile
    
            # Inverse transform of power law, for randomly sampling the radius
            u = plgen.uniform(0.0, 1.0)
            yy = u**(1./(-dslope+2))
            rScale = 14.0 # arcsec, scaling factor

            sma = rScale * yy * dist # AU

            # Assumed orbit:
            orb = orbits.Orbit()
            if (eSTD == 0) & (e >= 0):
                orb.e = e
            else:
                orb.e = -1.0 # quick and dirty way to make sure ecc > 0.0
                while (orb.e < 0.0) | (orb.e > 1.0):
                    if e >= 0:
                        orb.e = egen.gauss(e, eSTD)
                    elif e == -1:
                        orb.e = egen.uniform(0.0, 1.0)
            #print orb.e
            orb.w = wgen.uniform(0.0, 360.0)    
            orb.o = odisk # start with flat disk
            orb.i = idisk
            orb.p = np.sqrt((sma / mbh) * sma**2) # years
            orb.t0 = t0gen.uniform(1995.0, 1995.0+orb.p) # periapse passage

        else: # This is the isotropic population, if it's included

            # Inverse transform of power law, for randomly sampling the radius
            u = plgen.uniform(0.0, 1.0)
            yy = u**(1./(-ndslope+2))
            rScale = 14.0 # arcsec, scaling factor 

            sma = rScale * yy * dist # AU

            # Assumed orbit:
            orb = orbits.Orbit()

            # Inverse transform sampling for ecc 
            q = egen.uniform(0.0,1.0)
            yy = np.sqrt(q)
            orb.e = yy

            orb.w = wgen.uniform(0.0, 360.0)    
            orb.o = Ogen.uniform(0.0, 360.0)
            inc = igen.uniform(-1.0,1.0) # sample in cos(i)
            orb.i = np.arccos(inc)*rad2deg
            orb.p = np.sqrt((sma / mbh) * sma**2) # years
            orb.t0 = t0gen.uniform(1995.0, 1995.0+orb.p) # periapse passage

        # Convert Keplerian to Cartesian
        # units returned are arcsec, mas/yr, mas/yr^2
        # This is the mock data
        # choose new time for the mock data point (our observational point)
        if tObs == 'apoapse':
            t_obs = [orb.t0 + (orb.p / 2.0)]
        else:
            t_obs = tref
        (r, v, a) = orb.kep2xyz(t_obs, mass=mbh, dist=dist)
        x[nn] = r[0,0] # arcsec (+x to east)
        y[nn] = r[0,1]
        z[nn] = r[0,2]
        vx[nn] = v[0,0] # mas/yr (+x to east)
        vy[nn] = v[0,1]
        vz[nn] = v[0,2]
        ax[nn] = a[0,0] # mas/yr^2 (+x to east)
        ay[nn] = a[0,1]
        az[nn] = a[0,2]

        # Make sure this star is outside r = 0.8"
        r2d = np.sqrt(x[nn]**2 + y[nn]**2)
        if ((r2d < 0.8) | (r2d > 14.0)): 
            continue
        if outerRadial == True:
            if (r2d < 6.5):
                continue
        # Use the following to get line of nodes stars in middle radial interval
        #if  ((r2d < 3.2) | (r2d > 6.5) | (np.abs(z[nn]) > 1.0) | (np.abs(y[nn]) > 1.5)):
        #    continue

        print 'Star %i, radius = %6.3f' % (nn, r2d)
        if r2d < 0.8:
            pdb.set_trace()

        # semi-major axis in mas
        sma = sma / dist * 1.e3 # mas 
        # Write the orbital elements to a file
        orb_elem.write(fmt % (str(nn), orb.p, sma, orb.t0,
                              orb.e, orb.i, orb.o, orb.w))


        # Save off all the orbital parameters and cartesian coordinates
        orb_all = np.concatenate([orb_all, [orb]])
        sma_all = np.concatenate([sma_all, [sma]])

        ecc_all = np.concatenate([ecc_all, [orb.e]])

        t0_all = np.concatenate([t0_all, [orb.t0]])

        r3d = np.sqrt(x[nn]**2 + y[nn]**2 + z[nn]**2) * dist / cc.au_in_pc
        if vel_kick == True:
            # Add a velocity kick to each velocity component
            # Compute the local orbital velocity for this star
            vcirc = np.sqrt(cc.G * mbh * cc.msun / (r3d * cc.cm_in_pc)) / 1.e5 # km/s
            vcirc_masyr = vcirc / asy_to_kms * 1.e3
            vkick = frac * vcirc_masyr
            dvx = dvxgen.gauss(0.0, vkick)
            dvy = dvxgen.gauss(0.0, vkick)
            dvz = dvxgen.gauss(0.0, vkick)

            vx[nn] += dvx # mas/yr (+x to east)
            vy[nn] += dvy
            vz[nn] += dvz

        nn += 1

        # START TEMP: TEST
        #rvec = np.array([x[nn], y[nn], z[nn]])
        #vvec = np.array([vx[nn]/1.e3*asy_to_kms, vy[nn]/1.e3*asy_to_kms, vz[nn]/1.e3*asy_to_kms])
        #revec = np.zeros(3, dtype=float)
        #vevec = np.zeros(3, dtype=float)
        #newOrb = orbits.Orbit()
        #newOrb.xyz2kep(rvec, vvec, revec, vevec, t_obs,
        #            mass=mbh, dist=dist)
        #pdb.set_trace()
        # END TEMP: TEST

    orb_elem.close()

    py.clf()
    py.figure(1)
    py.figure(figsize=(6,6))
    py.subplots_adjust(left=0.18,right=0.92,top=0.9,bottom=0.1)
    if isoPop != None:
        py.quiver([x[ndisk:]], [y[ndisk:]], [vx[ndisk:]], [vy[ndisk:]],
                  headwidth=1.5, minshaft=1.5,
                  color='black', units='y', angles='xy', scale=5)
    py.quiver([x[0:ndisk]], [y[0:ndisk]], [vx[0:ndisk]], [vy[0:ndisk]],
              headwidth=1.5, minshaft=1.5,
              color='red', units='y', angles='xy', scale=5)
    py.plot([0],[0],'rx')
    py.axis([15.0, -15.0, -15.0, 15.0])
    py.text(10,13,'i = %3d deg' % idisk, fontsize=10)
    py.text(10,12,'O = %3d deg' % odisk, fontsize=10)
    if eSTD == None:
        py.text(10,11,'e = %4.2f' % e, fontsize=10)
    else:
        py.text(10,11,'e = %4.2f +- %4.2f' % (e, eSTD), fontsize=10)
    py.xlabel('X (arcsec)',fontsize=16)
    py.ylabel('Y (arcsec)',fontsize=16)
    py.title('Mock Data Velocities',fontsize=16)
    py.savefig(root+alnDir+mockdir+'/plots/velVector_mockdata'+suffix+'.png')
    py.close(1)

    py.close('all')
    py.clf()
    py.figure(2)
    py.figure(figsize=(6,6))
    py.subplots_adjust(left=0.18,right=0.92,top=0.9,bottom=0.1)
    sid = (np.array(ecc_all).argsort())
    py.hist(ecc_all,bins=100,histtype='step',normed=True,cumulative=True)
    py.axis([0,1,0,1])
    py.xlabel('e')
    py.ylabel('CDF')
    py.savefig(root+alnDir+mockdir+'/plots/ecc_dist_mockdata'+suffix+'.png')
    py.close(2)
    
    # Save the mock data to a pickle file to be used later by analyticOrbits.py
    pickFile = open(root+alnDir+mockdir+outfile, 'w')
    pickle.dump(mbh, pickFile)
    pickle.dump(dist, pickFile)
    pickle.dump(orb_all, pickFile)
    pickle.dump(sma_all, pickFile)
    pickle.dump(x, pickFile) # (+x to east)
    pickle.dump(y, pickFile)
    pickle.dump(z, pickFile)
    pickle.dump(vx, pickFile) # (+x to east)
    pickle.dump(vy, pickFile)
    pickle.dump(vz, pickFile)
    pickle.dump(ax, pickFile) # (+x to east)
    pickle.dump(ay, pickFile)
    pickle.dump(az, pickFile)
    pickle.dump(t0_all, pickFile) # periapse passage
    pickle.dump(t_obs, pickFile)
    pickFile.close()

class simulate_orbits():
    def __init__(self, ntrials=10000, mockdir='sim_vkick_fracCircVel/sim_vkick_0.0/',
                 mockfile='ecc_0.0_vkick_mockdata.pickle',
                 errorfile='pos_errorRange_vs_radius.dat', errRange=True,
                 sigma=5.0):
        """
        Run orbital analysis on the mock data produced above in mock_data()
    
        Input:
       	ntrials (int):	  Number of trials to run for each star in the mock data
            		  (def=10**5).
        mockdir (str):    Location where results are saved.
        mockfile (str):   Pickle file holding the mock data
        errorfile (str):  Text file containing astrometric errors vs. radius; file
        		  is expected to be in root+alndir+'tables/'.
        errRange (bool):  Type of error file -- If errRange == True, a range of
        		  errors is given in the file, between min and max of
                          observed errors. Errors will then be sampled from a
                          uniform distribution between min and max values given.
                          If errRange == False, the median error is assumed.
        sigma (float):    Significance value for acceleration measurements.
        pa0_simulation (bool): Set to True to pull stars due north of Sgr A*, and
        			run orbit analysis on only these.
        
        Dependencies:
        posErr_vs_radius() -- produces text file with median astrometric
            		      errors as a function of radius.
        """

        self.ntrials = ntrials
        self.mockfile = mockfile
        self.mockdir = mockdir
        self.sigma = sigma
        self.efile = errorfile
        self.errRange = errRange
    
    def run(self, pa0_simulation=False, zprior=False):
        # Mock data set (kinematic information for nstars):
        # Astrometric units are: arcsec, mas/yr, mas/yr^2
        mockdata = open(root + alnDir + self.mockdir + self.mockfile)
        mbh = pickle.load(mockdata)
        dist = pickle.load(mockdata)
        orb_all = pickle.load(mockdata)
        sma_all = pickle.load(mockdata)
        xM = pickle.load(mockdata) # (+x to east)
        yM = pickle.load(mockdata)
        zM = pickle.load(mockdata)
        vxM = pickle.load(mockdata) # (+x to east)
        vyM = pickle.load(mockdata)
        vzM = pickle.load(mockdata) # also in mas/yr!
        axM = pickle.load(mockdata) # (+x to east)
        ayM = pickle.load(mockdata)
        azM = pickle.load(mockdata)
        t0M = pickle.load(mockdata) # periapse passage
        t_obs = pickle.load(mockdata) # observational time for mock data point
        mockdata.close()

        # Convert x to be +x to west
        xM *= -1.0

        # Convert proper motions to asec/yr, with +x to west
        vxM /= -1.e3
        vyM /= 1.e3
        vzM /= 1.e3

        # Convert accelerations to asec/yr^2, with +x to west
        axM /= -1.e3
        ayM /= 1.e3

        self.refTime = t_obs
        
        nstars = len(orb_all)
    
        cc = objects.Constants()
        asy_to_kms = dist * cc.cm_in_au / (1e5*cc.sec_in_yr)
    
        r2dM = np.sqrt(xM**2 + yM**2) # arcsec

        if pa0_simulation == True:
            # we just want the stars in the patch of sky due north of Sgr A*
            pa0 = np.where((xM > -2.5) & (xM < 2.5) & (yM > 7.0))[0]
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
            nstars = len(pa0)

        # Pull the errors as a function of radius
        efile = asciidata.open(root+alnDir+'tables/'+self.efile)
        if self.errRange == False:
            rLo = efile[0].tonumpy()
            rHi = efile[1].tonumpy()
            xeR = efile[2].tonumpy()
            yeR = efile[3].tonumpy()
            vxeR = efile[4].tonumpy()
            vyeR = efile[5].tonumpy()
            vzeR = efile[6].tonumpy()
            axeR = efile[7].tonumpy()
            ayeR = efile[8].tonumpy()
        else:
            rLo = efile[0].tonumpy()
            rHi = efile[1].tonumpy()
            nObs = efile[2].tonumpy()
            xeRlo = efile[3].tonumpy()
            xeRhi = efile[4].tonumpy()
            yeRlo = efile[5].tonumpy()
            yeRhi = efile[6].tonumpy()
            vxeRlo = efile[7].tonumpy()
            vxeRhi = efile[8].tonumpy()
            vyeRlo = efile[9].tonumpy()
            vyeRhi = efile[10].tonumpy()
            vzeRlo = efile[11].tonumpy()
            vzeRhi = efile[12].tonumpy()
            axeRlo = efile[13].tonumpy()
            axeRhi = efile[14].tonumpy()
            ayeRlo = efile[15].tonumpy()
            ayeRhi = efile[16].tonumpy()
    
        GM = cc.G * mbh * cc.msun        # cgs

        for ss in range(nstars):
            # Make all our random number generators. We will need
            # some for the astrometry (x, y, vx, vy, vz, ax, ay)
            # and for the errors, if that flag is set (self.errRange)
            if self.errRange == False:
                gens = create_generators(8, self.ntrials*1000)
                xgen = gens[0]
                ygen = gens[1]
                vxgen = gens[2]
                vygen = gens[3]
                vzgen = gens[4]
                axgen = gens[5]
                aygen = gens[6]
                zgen = gens[7]
            else:
                gens = create_generators(15, self.ntrials*1000)
                xgen = gens[0]
                ygen = gens[1]
                vxgen = gens[2]
                vygen = gens[3]
                vzgen = gens[4]
                axgen = gens[5]
                aygen = gens[6]
                zgen = gens[7]
                xegen = gens[8]
                yegen = gens[9]
                vxegen = gens[10]
                vyegen = gens[11]
                vzegen = gens[12]
                axegen = gens[13]
                ayegen = gens[14]

            self._initVariables()
            self._initTransform()
        
            print 'Orbit simulation on mock data: star %i' % ss

            for nn in range(int(self.ntrials/2.0)):
                
                if ((nn % 1000) == 0):
                    print 'Trial %d' % nn, time.ctime(time.time())

                # Set temp values for our while loop
                amin = -1.0
                amax = -2.0
                aminLoopCount = 0
           
                while(amin >= amax):
                    aminLoopCount += 1

                    def sample_gaussian(self, x, y, ax, ay, axe, aye):
                        # Sample from Gaussian centered on our acceleration measurement
                        ax = axgen.gauss(ax, axe)  # arcsec/yr^2
                        ay = aygen.gauss(ay, aye)  # arcsec/yr^2

                        # Convert into mas/yr^2  
                        (ax, ay) = util.vPix2Arc(ax, ay, self.trans)
                        ax *= 1000.0
                        ay *= 1000.0
                        # Convert into radial and tangential
                        (ar, at) = util.xy2circ(x, y, ax, ay)
    
                        return ar # mas/yr^2
    
                    # Pull errors based on projected positions
                    ridx = np.where((rLo < r2dM[ss]) & (rHi > r2dM[ss]))[0]
                    # sometimes we get something slightly outside our
                    # observed field, catch these cases and assign them
                    # the error in the last radial bin
                    if (len(ridx) == 0) & (r2dM[ss] >= rHi[-1]):
                        ridx = [int(len(rHi) - 1)]
                    if self.errRange == False: # use the median value given in file
                        xe = xeR[ridx] / 1.e3 # arcsec
                        ye = yeR[ridx] / 1.e3 # arcsec
                        vxe = vxeR[ridx] / 1.e3 # asec/yr
                        vye = vyeR[ridx] / 1.e3 # asec/yr
                        vze = vzeR[ridx] / asy_to_kms # asec/yr
                        axe = axeR[ridx] / 1.e3 # asec/yr^2
                        aye = ayeR[ridx] / 1.e3 # asec/yr^2
                    else: # randomly sample from uniform error distribution
                        xe = xegen.uniform(xeRlo[ridx], xeRhi[ridx]) / 1.e3
                        ye = yegen.uniform(yeRlo[ridx], yeRhi[ridx]) / 1.e3
                        vxe = vxegen.uniform(vxeRlo[ridx], vxeRhi[ridx]) / 1.e3
                        vye = vyegen.uniform(vyeRlo[ridx], vyeRhi[ridx]) / 1.e3
                        vze = vzegen.uniform(vzeRlo[ridx], vzeRhi[ridx]) / asy_to_kms
                        axe = axegen.uniform(axeRlo[ridx], axeRhi[ridx]) / 1.e3
                        aye = ayegen.uniform(ayeRlo[ridx], ayeRhi[ridx]) / 1.e3
                    #print r2dM[ss], xe, ye, vxe, vye, vze, axe, aye

                    # convert the mock data to radial/tangential accels, to see if the
                    # accel is significant. if it is, we'll re-sample from the
                    # accel in the X and Y direction and re-compute ar and at.
                    # This is faster than doing an F test, as we do w/ real data, but
                    # stars that pass the F test all had significant ar, and vice versa
                    (arM, atM, are, ate) = \
                             util.xy2circErr(xM[ss], yM[ss], axM[ss], ayM[ss],
                                             xe, ye, axe, aye)
                    # Sample our monte carlo variables
                    x = xgen.gauss(xM[ss], xe)     # asec (+x west)
                    y = ygen.gauss(yM[ss], ye)     # asec (+y north)
                    vx = vxgen.gauss(vxM[ss], vxe) # asec/yr
                    vy = vygen.gauss(vyM[ss], vye) # asec/yr
                    vz = vzgen.gauss(vzM[ss], vze) # asec/yr
                
                    # Convert into positions relative to Sgr A*
                    (x, y) = util.rPix2Arc(x, y, self.trans)     # asec (+x east)
                    (vx, vy) = util.vPix2Arc(vx, vy, self.trans) # asec/yr                

                    # Convert velocities in km/s
                    vx *= asy_to_kms
                    vy *= asy_to_kms
                    vz *= asy_to_kms

                    # At this point:
                    #   all positions are in arcsec and
                    #   all velocities are in km/s
    
                    # Check for unbound cases
                    r2d = np.sqrt(x**2 + y**2)
                    r2dcgs = r2d * dist * cc.cm_in_au
                    vtot = np.sqrt(vx**2 + vy**2 + vz**2) # km/s
                    vtotcgs = vtot * 1e5
                    if (vtotcgs**2 > (2.0 * GM / r2dcgs)):
                        amin = -1.0
                        amax = -2.0
                        if (aminLoopCount % 1000) == 2:
                            print 'UNBOUND %5.3f vs. %5.3f km/s' % \
                                  ((vtotcgs/1.e5), (np.sqrt(2.*GM/r2dcgs)/1.e5))
                        continue # unbound....go back and resample
    
                    # Determine the maximum allowed acceleration (where a < 0)
                    # set by assuming a bound orbit.
                    amax = self._calcMaxAcc(x, y, vx, vy, vz, GM, dist, cc) # mas/yr^2
        
                    # Determine the minimum allowed acceleration (where a < 0)
                    # set by the minimum radius = 2D projected radius.
                    amin = self._calcMinAcc(x, y, GM, dist, cc) # mas/yr^2

                    zmax = acc2z(x, y, amax, dist, mbh)

                arGood = False
                arGoodCount = 0
                while (arGood == False):
                    arGoodCount += 1

                    # Are the computed accelerations significant given the estimated errors?
                    if r2dM[ss] >= 5.0: # no accels at large radii
                        asig = 0.0
                    else:
                        asig = arM /are
                    arM_lim = arM - (self.sigma*are) # lower limit
                    amin_theory =  self._calcMinAcc(xM[ss], yM[ss], GM, dist, cc) # mas/yr^2
                    if zprior == False:

                        if (arGoodCount % 1000) == 2:
                            try:
                                print self.mockdir
                                print 'Acceleration not between amin and amax'
                                print 'ar = %5.3f  [%5.3f -. %5.3f] count: %5i' % \
                                      (ar, amin, amax, arGoodCount)
                                if arGoodCount > 1e5:
                                    print 'Quitting simulation! Cannot find valid acceleration.'
                                    print 'Recreate mock data'
                                    pdb.set_trace()
                            except:
                                pdb.set_trace()

                        # If significant acceleration, sample from it
                        # Otherwise, uniform sampling from amin-amax
                        if (asig < -self.sigma):
                            ax = axM[ss]  # asec/yr^2
                            ay = ayM[ss]  # asec/yr^2
                            ar = sample_gaussian(self, x, y, ax, ay, axe, aye) # mas/yr^2
                        else:
                            ar = axgen.uniform(amin, amax)

                        self.asig = asig # significance of accel; will tell us if accel
                	   	         # measurement was used, or uniform prior

                        if (np.isnan(ar) == 1): # This is very bad
                            pdb.set_trace()

                        # Now convert our acceleration into a z-value. We will 
                        # have to do both the positive and negative solutions.
                        z = acc2z(x, y, ar, dist, mbh)

                        if (ar > amin and ar < amax):
                            arGood = True


                    elif zprior == True:
                        if (asig < -self.sigma):
                            ax = axM[ss]  # asec/yr^2
                            ay = ayM[ss]  # asec/yr^2
                            ar = sample_gaussian(self, x, y, ax, ay, axe, aye) # mas/yr^2
                            # Now convert our acceleration into a z-value. We will 
                            # have to do both the positive and negative solutions.
                            z = acc2z(x, y, ar, dist, mbh)
                        else:
                            # Sample from a power law in z
                            # with power law = -0.375
                            slope = -0.375
                            u = zgen.uniform(1e-5, 1.0)
                            yy = u**(1/(slope+2.0)) 
                            rScale = 20.0

                            z = rScale * yy # arcsec
                            ar = z2acc(x, y, z, dist, mbh)

                        self.asig = asig
                        if (ar > amin and ar < amax):
                            arGood = True

                        
                # Run the orbital analysis just as we do for our real data
                zidx = nn # positive z value run
                try:
                    self._runOrbit(zidx, x[0], y[0], z, vx[0], vy[0], vz[0],
                                   ar[0], mbh, dist, x0, y0)
                except ValueError, ee:
                    print 'Problem calculating orbits for POSITIVE %d' % nn
                    print ee
                    continue
                zidx = nn + (self.ntrials/2) # negative z value run
                try:
                    self._runOrbit(zidx, x[0], y[0], -z, vx[0], vy[0], vz[0],
                                   ar[0], mbh, dist, x0, y0)
                except ValueError, ee:
                    print 'Problem calculating orbits for POSITIVE %d' % nn
                    print ee
                    continue

                # END TEMP

            # Save the MC for this star
            mcfile = '%s/%s/%s/star%s.mc.dat' % (root, alnDir, self.mockdir, ss) 
            self.saveToFile(mcfile)


    def _initVariables(self):
        # These are all the variables that are spit out by
        # our monte carlo.
        self.i = np.zeros(self.ntrials, dtype=float)
        self.e = np.zeros(self.ntrials, dtype=float)
        self.evec = np.zeros((self.ntrials, 3), dtype=float)
        self.w = np.zeros(self.ntrials, dtype=float)
        self.o = np.zeros(self.ntrials, dtype=float)
        self.p = np.zeros(self.ntrials, dtype=float)
        self.t0 = np.zeros(self.ntrials, dtype=float)
        self.ph = np.zeros(self.ntrials, dtype=float)
        self.x = np.zeros(self.ntrials, dtype=float)
        self.y = np.zeros(self.ntrials, dtype=float)
        self.z = np.zeros(self.ntrials, dtype=float)
        self.vx = np.zeros(self.ntrials, dtype=float)
        self.vy = np.zeros(self.ntrials, dtype=float)
        self.vz = np.zeros(self.ntrials, dtype=float)
        self.ar = np.zeros(self.ntrials, dtype=float)
        self.m = np.zeros(self.ntrials, dtype=float)
        self.r0 = np.zeros(self.ntrials, dtype=float)
        self.x0 = np.zeros(self.ntrials, dtype=float)
        self.y0 = np.zeros(self.ntrials, dtype=float)

    def _initTransform(self):
        self.trans = objects.Transform()
        self.trans.scale = 1.0
        self.trans.scaleErr = 0.0
        self.trans.sgra = [-0.001, -0.005]
        self.trans.sgraErr = [0.0, 0.0]
        self.trans.angle = 0.0
        self.trans.angleErr = 0.0

    def _calcMaxAcc(self, x, y, vx, vy, vz, GM, dist, cc):
        """
        Maximum acceleration (where v = escape velocity).
        This corresponds to the largest allowed line of sight
        distance. 
        """
        r = np.sqrt(x**2 + y**2)             # arcsec
        rcgs = r * dist * cc.cm_in_au
        vcgs = np.sqrt(vx**2 + vy**2 + vz**2) * 1.0e5
        amax_cgs = -rcgs * vcgs**6 / (8.0 * GM**2)
        amax = amax_cgs * 1000.0 * cc.sec_in_yr**2
        amax /= (cc.cm_in_au * dist)      # mas/yr^2

        return amax

    def _calcMinAcc(self, x, y, GM, dist, cc):
        r = np.sqrt(x**2 + y**2)             # arcsec
        rcgs = r * dist * cc.cm_in_au
        amin_cgs = -GM / rcgs**2
        amin = amin_cgs * 1000.0 * cc.sec_in_yr**2
        amin /= (cc.cm_in_au * dist)      # mas/yr^2

        return amin


    def _runOrbit(self, nn, x, y, z, vx, vy, vz, ar, mass, dist, x0, y0):
        rvec = np.array([x, y, z])
        vvec = np.array([vx, vy, vz])
        revec = np.zeros(3, dtype=float)
        vevec = np.zeros(3, dtype=float)

        orb = orbits.Orbit()
        orb.xyz2kep(rvec, vvec, revec, vevec, self.refTime,
                    mass=mass, dist=dist)
        
        self.i[nn] = orb.i
        self.e[nn] = orb.e
        self.evec[nn] = orb.evec
        self.w[nn] = orb.w
        self.o[nn] = orb.o
        self.p[nn] = orb.p
        self.t0[nn] = orb.t0
        self.ph[nn] = orb.ph
        self.x[nn] = x
        self.y[nn] = y
        self.z[nn] = z
        self.vx[nn] = vx
        self.vy[nn] = vy
        self.vz[nn] = vz
        self.ar[nn] = ar
        self.m[nn] = mass
        self.r0[nn] = dist
        self.x0[nn] = x0
        self.y0[nn] = y0


    def saveToFile(self, savefile):
        _f = open(savefile, 'w')
        pickle.dump(self, _f)
        _f.close()

class simDisk(object):
    def __init__(self, root, mcdir, mockfile, nstars, nside=64):
        self.nside = nside
        self.npix = healpy.nside2npix(self.nside)
        self.mcdir = mcdir
        self.mockfile = mockfile
        self.root = root
        self.nstars = nstars

    def run(self, makeplot=False, do_all=True, non_disk=False, do_radial_bins=False,
            do_r1=False, do_r2=False, do_r3=False, fov_obs=False):

        # Load up i, omega and projected radius for all stars, all trials
        #  -- iomap = sum of all PDFs
        #  -- ioCntMap = number of trials at each position
        iAll, oAll, r2d, x, y = self.loadStars(return_r2d=True)

        # Save the marginalized PDF to a file.
        self.iomap.tofile('%s/%s/simdisk.heal.dat' % \
                          (self.root, self.mcdir))
        self.ioCntMap.tofile('%s/%s/simdisk.iocnt.dat' % \
                          (self.root, self.mcdir))

        # Optional plotting
        if (makeplot):
            mcFile = '%s/%s/simdisk.heal.dat' % (self.root, self.mcdir)
            pdh.go(mcFile, self.npix, 1)

        if do_all == True:
            # Map out the PDF for the density of normal vectors
            (neigh, neighStd, peakD, peakI, peakO) = self.densityPDF(iAll, oAll,
                                                                     neighbors=[6])
            
            # Save the i/o density maps
            print 'Making density map'
            neigh.tofile('%s/%s/simdisk.neighbor.dat' % \
                         (self.root, self.mcdir))
            neighStd.tofile('%s/%s/simdisk.neighborStd.dat' % \
                         (self.root, self.mcdir))
            peakD.tofile('%s/%s/simdisk.peakDensity.dat' % \
                         (self.root, self.mcdir))
            peakI.tofile('%s/%s/simdisk.peakIncli.dat' % \
                         (self.root, self.mcdir))
            peakO.tofile('%s/%s/simdisk.peakOmega.dat' % \
                         (self.root, self.mcdir))

        if non_disk == True:
            # Non-disk stars (must have determined which stars are candidate disk
            # members).
            # Read in disk_membership probability file and get the non-disk members
            diskTab = asciidata.open('%s/%s/plots/HEALpixMaps/simdisk_membership_prob.dat' % (self.root, self.mcdir))
            nameM = [diskTab[0][jj].strip() for jj in range(diskTab.nrows)]
            diskP = diskTab[1].tonumpy()
            nondiskIdx = (np.where(diskP <= 2.7e-3))[0]
            nondisk = [nameM[nn] for nn in nondiskIdx]

            # Also only select stars in our observed field of view:
            xys = np.column_stack((x, y))
            fields = asciidata.open('/u/syelda/research/gc/aligndir/11_10_26/tables/osiris_sinfoni_fields.txt')
            xvrt0 = fields[0].tonumpy()
            yvrt0 = fields[1].tonumpy()
            xvrt = np.array([np.float(xx) for xx in xvrt0])
            yvrt = np.array([np.float(yy) for yy in yvrt0])
            verts = np.column_stack((xvrt, yvrt))
            mask = nx.points_inside_poly(xys, verts)
            
            idx = []
            for nn in nondiskIdx:
                # Check that this star is in the field
                if mask[nn] == False:
                    print 'Skipping star %s - outside field (x,y) = (%6.2f, %6.2f)' % \
                          (nn, x[nn], y[nn])
                    continue
                print 'Keeping Non-disk star: %s' % nn
                idx = np.concatenate([idx, [nn]])
            idx = [int(ii) for ii in idx]

            i_nd = iAll[idx,:]
            o_nd = oAll[idx,:]

            # Map out the PDF for the density of normal vectors
            (neigh, neighStd, peakD, peakI, peakO) = self.densityPDF(i_nd, o_nd,
                                                          neighbors=[6],aperture=False)
            
            # Save the i/o density maps
            print 'Making density map for stars not on the disk and within observed FOV (N=%i)' %\
                  len(idx)
            neigh.tofile('%s/%s/non_disk_fov.neighbor.dat' % \
                         (self.root, self.mcdir))
            neighStd.tofile('%s/%s/non_disk_fov.neighborStd.dat' % \
                            (self.root, self.mcdir))
            peakD.tofile('%s/%s/non_disk_fov.peakDensity.dat' % \
                         (self.root, self.mcdir))
            peakI.tofile('%s/%s/non_disk_fov.peakIncli.dat' % \
                         (self.root, self.mcdir))
            peakO.tofile('%s/%s/non_disk_fov.peakOmega.dat' % \
                         (self.root, self.mcdir))

        if do_radial_bins == True:
            rad1 = np.where(r2d <= 3.197)[0]
            i_inner = iAll[rad1,:]
            o_inner = oAll[rad1,:]

            rad2 = np.where((r2d > 3.197) & (r2d <= 6.473))[0]
            i_middle = iAll[rad2,:]
            o_middle = oAll[rad2,:]

            rad3 = np.where(r2d > 6.473)[0]
            i_outer = iAll[rad3,:]
            o_outer = oAll[rad3,:]

            if do_r1 == True:
                # Radial bin 1
                # Map out the PDF for the density of normal vectors
                (neigh, neighStd, peakD, peakI, peakO) = self.densityPDF(i_inner, o_inner,
                                                                         neighbors=[6])
            
                # Save the i/o density maps
                print 'Making density map for stars at r < 3.197 arcsec'
                neigh.tofile('%s/%s/inner_simdisk.neighbor.dat' % \
                         (self.root, self.mcdir))
                neighStd.tofile('%s/%s/inner_simdisk.neighborStd.dat' % \
                         (self.root, self.mcdir))
                peakD.tofile('%s/%s/inner_simdisk.peakDensity.dat' % \
                         (self.root, self.mcdir))
                peakI.tofile('%s/%s/inner_simdisk.peakIncli.dat' % \
                         (self.root, self.mcdir))
                peakO.tofile('%s/%s/inner_simdisk.peakOmega.dat' % \
                         (self.root, self.mcdir))

            if do_r2 == True:
                # Radial bin 2
                # Map out the PDF for the density of normal vectors
                (neigh, neighStd, peakD, peakI, peakO) = self.densityPDF(i_middle, o_middle,
                                                                         neighbors=[6])
                
                ## Save the i/o density maps
                print 'Making density map for stars at 3.197 <= r < 6.473 arcsec'
                neigh.tofile('%s/%s/middle_simdisk.neighbor.dat' % \
                         (self.root, self.mcdir))
                neighStd.tofile('%s/%s/middle_simdisk.neighborStd.dat' % \
                         (self.root, self.mcdir))
                peakD.tofile('%s/%s/middle_simdisk.peakDensity.dat' % \
                         (self.root, self.mcdir))
                peakI.tofile('%s/%s/middle_simdisk.peakIncli.dat' % \
                         (self.root, self.mcdir))
                peakO.tofile('%s/%s/middle_simdisk.peakOmega.dat' % \
                         (self.root, self.mcdir))
    
            if do_r3 == True:
                suffix = ''
                if fov_obs == True:
                    # Also only select stars in our observed field of view. This should only
                    # matter for the outer radial bin (r > 6.5")
                    xys = np.column_stack((x, y))
                    fields = asciidata.open('/u/syelda/research/gc/aligndir/11_10_26/tables/osiris_sinfoni_fields.txt')
                    xvrt0 = fields[0].tonumpy()
                    yvrt0 = fields[1].tonumpy()
                    xvrt = np.array([np.float(xx) for xx in xvrt0])
                    yvrt = np.array([np.float(yy) for yy in yvrt0])
                    verts = np.column_stack((xvrt, yvrt))
                    mask = nx.points_inside_poly(xys, verts)
                
                    idx = []
                    for rr in rad3:
                        # Check that this star is in the field
                        if mask[rr] == False:
                            print 'Skipping star %s - outside field (x,y) = (%6.2f, %6.2f)' % \
                                  (rr, x[rr], y[rr])
                            continue
                        print 'Keeping star: %s' % rr
                        idx = np.concatenate([idx, [rr]])
                    idx = [int(ii) for ii in idx]
        
                    i_outer = iAll[idx,:]
                    o_outer = oAll[idx,:]
                    suffix = 'fov_observed'

                # Radial bin 3
                # Map out the PDF for the density of normal vectors
                (neigh, neighStd, peakD, peakI, peakO) = self.densityPDF(i_outer, o_outer,
                                                                         neighbors=[6])
                
                # Save the i/o density maps
                print 'Making density map for stars at r >= 6.473 arcsec'
                neigh.tofile('%s/%s/outer_simdisk%s.neighbor.dat' % \
                         (self.root, self.mcdir, suffix))
                neighStd.tofile('%s/%s/outer_simdisk%s.neighborStd.dat' % \
                         (self.root, self.mcdir, suffix))
                peakD.tofile('%s/%s/outer_simdisk%s.peakDensity.dat' % \
                         (self.root, self.mcdir, suffix))
                peakI.tofile('%s/%s/outer_simdisk%s.peakIncli.dat' % \
                         (self.root, self.mcdir, suffix))
                peakO.tofile('%s/%s/outer_simdisk%s.peakOmega.dat' % \
                         (self.root, self.mcdir, suffix))

    def densityPDF(self, iAll, oAll, neighbors=[6], aperture=False):
        """
        Map out the PDF for the density of normal vectors
        """
        nstars = iAll.shape[0]
        print nstars, type(nstars)
        trials = 10000
        npdfs = len(neighbors)

        pixIdx = np.arange(self.npix, dtype=int)
        (ipix, opix) = healpy.pix2ang(self.nside, pixIdx)
        sinip = np.sin(ipix)
        cosip = np.cos(ipix)

        siniAll = np.sin(iAll)
        cosiAll = np.cos(iAll)

        onesNpix = np.ones(self.npix, dtype=float)
        onesNstars = np.ones(nstars, dtype=float)
        factor = 2.0 * math.pi * (180.0 / math.pi)**2

        if (aperture == True):
            # We will be using nearest neighbor and aperture densities.
            # Pre-calc some stuff for the aperture densities.
            #angCut = 0.1745  # radians (10 deg )
            angCut = 0.1047  # radians (6 deg )
            angCutArea = factor * (1 - np.cos(angCut)) # area in deg^2
            
        if (trials > self.pdftrials):
            print 'Must have more PDF trials than disk trials'
        
        # Compute the PDF for the density at each pixel along
        # with the weighted average density at each pixel.
        neighborMap = np.zeros((npdfs, self.npix), dtype=float)
        neighborMapStd = np.zeros((npdfs, self.npix), dtype=float)

        # Keep track of the peak density and position for each trial
        peakDensity = np.zeros((npdfs, trials), dtype=float)
        peakIncli = np.zeros((npdfs, trials), dtype=float)
        peakOmega = np.zeros((npdfs, trials), dtype=float)

        if (aperture == True):
            # Also record maps from aperture density calculation
            apertureMap = np.zeros(self.npix, dtype=float)
            apertureMapStd = np.zeros(self.npix, dtype=float)
            
            # Keep track of the peak density and position for each trial
            peakDensityAp = np.zeros(trials, dtype=float)
            peakIncliAp = np.zeros(trials, dtype=float)
            peakOmegaAp = np.zeros(trials, dtype=float)

        _out1 = open('%s/%s/disk_nn_results.txt' % \
                     (self.root, self.mcdir), 'w')
        if aperture == True:
            _out2 = open('%s/%s/disk_ap_results.txt' % \
                     (self.root, self.mcdir), 'w')

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

                #pickle.dump(densityMap, nnFiles[nn])

            if (aperture == True):
                # Aperture density:
                densityMapAp = np.zeros(self.npix, dtype=float32)
                pidx = (np.where(angOff < angCut))[1]
                ifreq = (stats.itemfreq(pidx)).astype('int')
                for pix, cnt in ifreq:
                    densityMapAp[pix] = cnt / angCutArea

                maxPix = densityMapAp.argmax()

                apertureMap += densityMapAp
                apertureMapStd += densityMapAp**2
                peakDensityAp[ii] = apertureMap[maxPix]
                peakOmegaAp[ii] = opix[maxPix]
                peakIncliAp[ii] = ipix[maxPix]
                
                # Save to an output file
                fubar = array(densityMapAp * 10**5, dtype=int64)
                fubar.tofile(_out2)
                fubar = None

        neighborMap /= trials
        neighborMapStd = np.sqrt( (neighborMapStd / trials) - neighborMap**2 )

        if (aperture == True):
            apertureMap /= trials
            apertureMapStd = np.sqrt( (apertureMapStd / trials) - apertureMap**2 )

            # Save the i/o density maps
            print 'Making density map'
            apertureMap.tofile('%s/%s/disk.aperture.dat' % \
                               (self.root, self.mcdir))
            apertureMapStd.tofile('%s/%s/disk.apertureStd.dat' % \
                               (self.root, self.mcdir))
            peakDensityAp.tofile('%s/%s/disk.peakDensityAp.dat' % \
                               (self.root, self.mcdir))
            peakIncliAp.tofile('%s/%s/disk.peakIncliAp.dat' % \
                               (self.root, self.mcdir))
            peakOmegaAp.tofile('%s/%s/disk.peakOmegaAp.dat' % \
                               (self.root, self.mcdir))

        return (neighborMap, neighborMapStd, peakDensity, peakIncli, peakOmega)

        
    def loadStars(self, return_r2d=False):
        self.iomap = np.zeros((1, self.npix), dtype=float)
        self.ioCntMap = np.zeros((1, self.npix), dtype=float)
        
        iAll = None
        oAll = None

        # Load up the mock data
        # Astrometric units are: arcsec, mas/yr, mas/yr^2
        mockdata = open(self.root + self.mcdir + self.mockfile)
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

        self.r2d = np.sqrt(xM**2 + yM**2)
        self.x = xM
        self.y = yM

        for ss in range(self.nstars):

            name = 'star%s' % str(ss)

            # Read in the 6D Probability distribution of orbital
            # parameters for this star.
            _f = open('%s/%s/%s.mc.dat' % (self.root, self.mcdir, name), 'r')
            mc = pickle.load(_f)
            _f.close()
            print 'Adding %15s (%d) to disk' % (name, len(mc.i))

            # Marginalize the PDF for this star onto just
            # incl (i) and PA to ascending node (o).
            hpfile = '%s/%s/plots/HEALpixMaps/%s_disk_mc_heal.dat' %\
                      (self.root, self.mcdir, name)
            pdf = np.fromfile(hpfile, dtype=float)

            # Store the monte carlo results as (i, o) pairs.
            if (iAll == None):
                self.pdftrials = len(mc.i)
                iAll = np.zeros((self.nstars, self.pdftrials), dtype=float)
                oAll = np.zeros((self.nstars, self.pdftrials), dtype=float)
                 
            # This assumes that the Mass/r0/x0/y0 values are the
            # same for a single trial across all stars.
            iAll[ss,:] = mc.i
            oAll[ss,:] = mc.o

            self.iomap[0] += pdf
            self.ioCntMap[0] += (pdf / pdf.max())

        # Get the (i,o) values for each pixel in the sky
        iAll *= math.pi / 180.0
        oAll *= math.pi / 180.0

        if return_r2d:
            return iAll, oAll, self.r2d, self.x, self.y
        else:
            return iAll, oAll

    def diskMembership(self, file1='simdisk.neighbor.dat',
                       file2='simdisk.neighborStd.dat',verbose=True,
                       radialBin=None,LHsigCut=3.0):
        """
        Determine which stars are members of the disk
        Input:
        radialBin (str): If running membership on radial bin, specify
        		 the bin using 'inner_', 'middle_', 'outer_'.
        LHsigCut (float): Significance threshold for non-members;
        		  default is set to 3.0, meaning that stars with
                          likelihood of not being on the disk of >3 sigma
                          are considered non-members. The rest are candidates.

        Output:
        disk_membership_prob.dat -- text table containing results
        plots/disk_membership_hist.png -- Histogram of disk membership
        probabilities.
        """

        # Disk solution
        # Make sure to define this using the primary sample disk solution,
        # even if we are computing disk membership for the extended sample.
        nside = 64
        npix = healpy.nside2npix(nside)
        pixIdx = np.arange(0, npix)
        (disk, diskStd) = sythesis.loadDiskDensity(npix, orbDir=self.root+self.mcdir,
                                                   file1=file1, file2=file2,
                                                   singlePdf=True, simdisk=True)
    
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
        peakProb = np.zeros(self.nstars, dtype=float)
        solidAngle = np.zeros(self.nstars, dtype=float)
        offDiskLH = np.zeros(self.nstars, dtype=float)
    
        if radialBin != None:
            _out = open(self.root + self.mcdir + \
                    'plots/HEALpixMaps/'+ radialBin +'simdisk_membership_prob.dat', 'w')
        else:
            _out = open(self.root + self.mcdir + \
                    'plots/HEALpixMaps/simdisk_membership_prob.dat', 'w')
    
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
        for ss in range(self.nstars):
            name = 'star' + str(ss)
            orbFile = self.root + self.mcdir + 'plots/HEALpixMaps/' + name + '_disk_mc_heal.dat'
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
            
            nzDisk = (np.where((disk > (0.5 * peak)) & (pdf > 0)))[0]
            solidAngleInDisk = len(nzDisk) * areaPerPixel
            maxProb = pdfSort[0:len(pidx)].sum()
    
            peakProb[ss] = pdf[pidx].sum() / maxProb
            # Likelihood that the star is not in the disk plane:
            offDiskLH[ss] = 1.0 - peakProb[ss]

            # What is the threshold for disk/non-disk members?
            lhNotOnDisk_cut = scipy.special.erf(LHsigCut/np.sqrt(2.))
            probOnDisk_cut = 1.0 - lhNotOnDisk_cut

            if (peakProb[ss] != 0):
                onDisk = ''
                #if (peakProb[ss] < 2.7e-3):
                if (peakProb[ss] < probOnDisk_cut):
                    onDisk = 'Not on disk'
                if verbose == True:
                    print '%13s  %8.2e  %8.2e   %6.3f  %s ' % \
                          (name, peakProb[ss], offDiskLH[ss], solidAngle[ss], onDisk)

            _out.write('%13s  %8.2e  %6.3f\n' % 
    		   (name, peakProb[ss], solidAngle[ss]))

        avgSolidAngle = totalSolidAngle / self.nstars
    
        print '%8.5f - Average Solid Angle (sr) Per Star (1 sigma)' % \
    	(avgSolidAngle)
        print '%8.5f - Expected density for isotropic population' % \
    	(self.nstars / areaOnSky)

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
    
        diskIdx = (np.where(peakProb > probOnDisk_cut))[0]
        print ''
        print 'Total number of stars: %2d' % (len(peakProb))
            
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
        py.figure(figsize=(7,7))
        py.subplots_adjust(left=0.1, right=0.96, top=0.95)
        py.clf()
        py.hist(np.log10(probGood), 20, histtype='step', color='k', linewidth=2)
        py.plot([-2.57,-2.57],[0,16],'k--')
        py.xlabel('Log Probability ', fontsize=18)
        py.ylabel('Number of Stars', fontsize=18)
        py.axis([-4.5, 0, 0, 16])
        #py.title('Ranges: i (%3d - %3d), O (%3d - %3d)' % (ilo, ihi, olo, ohi))
        py.savefig(self.root + self.mcdir + 'plots/HEALpixMaps/disk_membership_hist_nonzero.png')
        py.close()

        return (idisk,odisk,peak,diskRadius,len(diskIdx)) 


def ave_rms_density_maps(all=True,inner=False,middle=False,outer=False,trials=10):
    """
    For our favorite disk fraction simulation (20%), combine all of the nearest
    neighbor density maps to get an average and RMS density at each pixel.
    This can be used to test the significance of features in our observations.
    """
    trials = trials*1.0

    # A total of 10 simulations were run, where in each 120 stars were simulated
    # and 20% of these stars were on the CW disk, 80% had isotropic orbits.
    suffix = ''
    if inner == True:
        suffix = 'inner_'
    elif middle == True:
        suffix = 'middle_'
    elif outer == True:
        suffix = 'outer_'

    nside = 64
    npix = healpy.nside2npix(nside)
    disk_sum = np.zeros(npix, dtype=float)
    disk_std = np.zeros(npix, dtype=float)

    for ii in range(trials):
        simDir = '%s%s/sim_diskFraction3/disk20_%s/' % (root, alnDir, str(ii+1))
        file1 = '%ssimdisk.neighbor.dat' % suffix
        file2 = '%ssimdisk.neighborStd.dat' % suffix

        # Disk solution
        # Make sure to define this using the primary sample disk solution,
        # even if we are computing disk membership for the extended sample.
        pixIdx = np.arange(0, npix)
        (disk, diskStd) = sythesis.loadDiskDensity(npix, orbDir=simDir,
                                                   file1=file1, file2=file2,
                                                   singlePdf=True, simdisk=True)
    
        (i, o) = healpy.pix2ang(nside, pixIdx)
        i *= 180.0 / math.pi
        o *= 180.0 / math.pi
    
        didx = disk.argmax()
        peak = disk[didx]
        idisk = i[didx]
        odisk = o[didx]
        print 'Peak at %.2e  stars/deg^2' % (peak)

        # Add up all the density maps
        disk_sum += disk
        disk_std += disk**2
        
    disk_ave = disk_sum / trials
    disk_rms = np.sqrt( (disk_std / trials) - disk_ave**2 )

    avefile = '%s%s/sim_diskFraction3/plots/%ssimdisk_f20_trials10_ave.dat' % \
            (root, alnDir, suffix)
    rmsfile = '%s%s/sim_diskFraction3/plots/%ssimdisk_f20_trials10_rms.dat' % \
            (root, alnDir, suffix)
    disk_ave.tofile(avefile)
    disk_rms.tofile(rmsfile)

    # Make the HEALpix map
    pdh.go(avefile, 49152, 1)
    pdh.go(rmsfile, 49152, 1)


########################
#
# Helper Functions
#
########################

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

def acc2z(x, y, ar, dist, mass):
    """
    Change acceleration (in the plane of the sky) from mas/yr^2 to
    line of sight distance in arcsec.

    Input:
    x - x position in arcsec
    y - y position in arcsec
    ar - plane of the sky acc in mas/yr^2
    dist - Ro in pc
    mass - in solar masses
    """
    cc = objects.Constants()
    GM = mass * cc.msun * cc.G

    # Convert acceleration into CGS
    arcgs = ar * dist * cc.cm_in_au / (cc.sec_in_yr**2 * 1000.0)

    # Convert into z-distance
    r = np.sqrt(x**2 + y**2)             # arcsec
    rcgs = r * dist * cc.cm_in_au
    tmp1 = (GM * rcgs / -arcgs)**(2.0/3.0)
    zcgs = np.sqrt(tmp1 - rcgs**2)
    z = zcgs / (cc.cm_in_au * dist)

    return z

def z2acc(x, y, z, dist, mass):
    """
    Change line of sight distance in arcsec to
    acceleration (in the plane of the sky) in mas/yr^2.

    Input:
    x - x position in arcsec
    y - y position in arcsec
    z - z position in arcsec
    dist - Ro in pc
    mass - in solar masses
    """
    cc = objects.Constants()
    GM = mass * cc.msun * cc.G

    # Convert distance into CGS
    r = np.sqrt(x**2 + y**2)             # arcsec
    zcgs = z * dist * cc.cm_in_au
    rcgs = r * dist * cc.cm_in_au

    arcgs = -GM * rcgs / (rcgs**2 + zcgs**2)**(3.0/2.0)
    ar = arcgs * cc.sec_in_yr**2 * 1000.0 / (dist * cc.cm_in_au)

    return ar

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

def posErr_vs_radius(errorfile='pos_errorRange_vs_radius.dat',median=False,
                     errRange=True):
    """
    Determines positional errors as a function of radius.
    Input:
    	errorfile (string): Name of file to write astrometric errors to.
        median (bool):	    Compute median astrometric error within radial bin.
 	errRange (bool):    Determine the range of astrometric errors within
        		    each radial bin.

    Output:
    	tables/<errorfile> (needed for creation of mock data)
    """
    cc = objects.Constants()
    asy_to_kms = dist * cc.cm_in_au / (1e5*cc.sec_in_yr)
    
    # Load up mosaic data as well; select only stars at r>4, since
    # we don't want to add any info from mosaics if we have it in
    # the central 10" already
    yng1 = young.loadYoungStars(root+alnDir,fit=poly,points=points,
                                withRVonly=True,silent=True) 
    yng2 = young.loadYoungStars(mscDir,fit=polyM,points=pointsM,
                                mosaic=True, withRVonly=True,silent=True)
    cntrlNames = yng1.getArray('name')
    mscNames = yng2.getArray('name')
    # Merge this object with object from central 10" analysis
    yng = merge(yng1, yng2)
    yngNames = yng.getArray('name')

    radStep = 1.0
    radBins = np.arange(0.5, 15.5, radStep)

    msc = []
    r2d_all = []
    xe_all = []
    ye_all = []
    vxe_all = []
    vye_all = []
    vze_all = []
    axe_all = []
    aye_all = []
    out = open(root+alnDir+'tables/'+errorfile, 'w')
    for name in yngNames:
        # Get the star object
        idx = yngNames.index(name)
        star = yng.stars[idx]
        mag = star.mag

        mscStar = False

        if (name in mscNames) & (name not in cntrlNames):
            mscStar = True
            # Polyfit results
            x = star.fitXv.p
            y = star.fitYv.p
            r2d = np.hypot(x, y)
            xe = star.fitXv.perr * 1.e3 # mas
            ye = star.fitYv.perr * 1.e3 # mas
            vxe = star.fitXv.verr / asy_to_kms * 1e3 # mas/yr
            vye = star.fitYv.verr / asy_to_kms * 1e3 # mas/yr
            vze = star.vzerr # km/s
            axe = 999
            aye = 999
        else:
            # Polyfit results
            x = star.fitXa.p
            y = star.fitYa.p
            r2d = np.hypot(x, y)
            xe = star.fitXa.perr * 1.e3 # mas
            ye = star.fitYa.perr * 1.e3 # mas
            vxe = star.fitXa.verr / asy_to_kms * 1e3 # mas/yr
            vye = star.fitYa.verr / asy_to_kms * 1e3 # mas/yr
            vze = star.vzerr # km/s
            axe = star.fitXa.aerr * 1.e3 # mas/yr^2
            aye = star.fitYa.aerr * 1.e3 # mas/yr^2
            

        r2d_all = np.concatenate([r2d_all, [r2d]])
        xe_all = np.concatenate([xe_all, [xe]])
        ye_all = np.concatenate([ye_all, [ye]])
        vxe_all = np.concatenate([vxe_all, [vxe]])
        vye_all = np.concatenate([vye_all, [vye]])
        vze_all = np.concatenate([vze_all, [vze]])
        axe_all = np.concatenate([axe_all, [axe]])
        aye_all = np.concatenate([aye_all, [aye]])
        msc = np.concatenate([msc, [mscStar]])

    if median == True:
        xe_rad = np.zeros(len(radBins), dtype=float)
        ye_rad = np.zeros(len(radBins), dtype=float)
        vxe_rad = np.zeros(len(radBins), dtype=float)
        vye_rad = np.zeros(len(radBins), dtype=float)
        vze_rad = np.zeros(len(radBins), dtype=float)
        axe_rad = np.zeros(len(radBins), dtype=float)
        aye_rad = np.zeros(len(radBins), dtype=float)
        hdr = '#%5s  %5s  %5s  %5s  %5s  %5s  %5s  %5s  %5s\n'
        out.write(hdr % ('r1', 'r2', 'xe', 'ye', 'vxe', 'vye', 'vze', 'axe', 'aye'))
        # Units of errors in output table are mas, mas/yr, mas/yr^2
        # (except RV error in km/s)
        fmt = '%5.2f  %5.2f  %5.3f  %5.3f  %5.3f  %5.3f  %5.3f  %5.3f  %5.3f\n'
    elif errRange == True:
        xe_min = np.zeros(len(radBins), dtype=float)
        ye_min = np.zeros(len(radBins), dtype=float)
        vxe_min = np.zeros(len(radBins), dtype=float)
        vye_min = np.zeros(len(radBins), dtype=float)
        vze_min = np.zeros(len(radBins), dtype=float)
        axe_min = np.zeros(len(radBins), dtype=float)
        aye_min = np.zeros(len(radBins), dtype=float)
        xe_max = np.zeros(len(radBins), dtype=float)
        ye_max = np.zeros(len(radBins), dtype=float)
        vxe_max = np.zeros(len(radBins), dtype=float)
        vye_max = np.zeros(len(radBins), dtype=float)
        vze_max = np.zeros(len(radBins), dtype=float)
        axe_max = np.zeros(len(radBins), dtype=float)
        aye_max = np.zeros(len(radBins), dtype=float)
        hdr = '#%6s  %6s  %2s  %6s  %6s  %6s  %6s  %6s  %6s  %6s  %6s  %6s  '
        hdr += '%6s  %6s %6s  %6s  %6s\n'
        out.write(hdr % ('r1', 'r2', 'N', 'xemin', 'xemax', 'yemin', 'yemax',
                         'vxemin', 'vxemax', 'vyemin', 'vyemax', 'vzemin', 'vzemax',
                         'axemin', 'axemax', 'ayemin', 'ayemax'))
        # Units of errors in output table are mas, mas/yr, mas/yr^2
        # (except RV error in km/s)
        fmt = '%6.2f  %6.2f  %2i  %6.3f  %6.3f  %6.3f  %6.3f  %6.3f  %6.3f  %6.3f  '
        fmt += '%6.3f  %6.3f  %6.3f %6.3f  %6.3f  %6.3f  %6.3f\n'

    ##########
    # Compute errors in radius bins
    ########## 
    for rr in range(len(radBins)):
        rMin = radBins[rr] - (radStep / 2.0)
        rMax = radBins[rr] + (radStep / 2.0)
        idx = (np.where((r2d_all >= rMin) & (r2d_all < rMax)))[0]
        nn = len(idx)
        print rMin, rMax

        if median == True:
            if (nn > 0):
                xe_rad[rr] = np.median(xe_all[idx])
                ye_rad[rr] = np.median(ye_all[idx])
                vxe_rad[rr] = np.median(vxe_all[idx])
                vye_rad[rr] = np.median(vye_all[idx])
                vze_rad[rr] = np.median(vze_all[idx])
                if rMin >= 5.0:
                    axe_rad[rr] = 999
                    aye_rad[rr] = 999
                else:
                    axe_rad[rr] = np.median(axe_all[idx])
                    aye_rad[rr] = np.median(aye_all[idx])
    
                out.write(fmt % (rMin, rMax, xe_rad[rr], ye_rad[rr], vxe_rad[rr],
                                 vye_rad[rr], vze_rad[rr], axe_rad[rr], aye_rad[rr]))
            else:
                # If there weren't any stars in this radial bin, use the
                # errors in the previous radial bin
                out.write(fmt % (rMin, rMax, xe_rad[rr-1], ye_rad[rr-1], vxe_rad[rr-1],
                                 vye_rad[rr-1], vze_rad[rr-1], axe_rad[rr-1],
                                 aye_rad[rr-1]))
        elif errRange == True:
            if (nn > 0): # only two radial bins don't have any stars (12-13" and 14-15")
                xe_min[rr] = (xe_all[idx]).min()
                ye_min[rr] = (ye_all[idx]).min()
                vxe_min[rr] = (vxe_all[idx]).min()
                vye_min[rr] = (vye_all[idx]).min()
                vze_min[rr] = (vze_all[idx]).min()
                xe_max[rr] = (xe_all[idx]).max()
                ye_max[rr] = (ye_all[idx]).max()
                vxe_max[rr] = (vxe_all[idx]).max()
                vye_max[rr] = (vye_all[idx]).max()
                vze_max[rr] = (vze_all[idx]).max()
                if rMin >= 5.0:
                    axe_min[rr] = 999
                    aye_min[rr] = 999
                    axe_max[rr] = 999
                    aye_max[rr] = 999
                else:
                    axe_min[rr] = (axe_all[idx]).min()
                    aye_min[rr] = (aye_all[idx]).min()
                    axe_max[rr] = (axe_all[idx]).max()
                    aye_max[rr] = (aye_all[idx]).max()
                # Assume all stars beyond 10" have the same error as those
                # in the 10-11" radial bin, which has 8 stars
                # (there are only 3 stars beyond 11")
                # If we don't do this, the errors are actually smaller for these
                # 3 stars than for the errors in the 10-11" bin.
                r10 = 10
                if rMin >= 11.0:
                    rr = r10
    
                out.write(fmt % (rMin, rMax, nn, xe_min[rr], xe_max[rr],
                                 ye_min[rr], ye_max[rr], vxe_min[rr], vxe_max[rr],
                                 vye_min[rr], vye_max[rr], vze_min[rr], vze_max[rr],
                                 axe_min[rr], axe_max[rr], aye_min[rr], aye_max[rr]))
            else:
                rr = 10
                out.write(fmt % (rMin, rMax, nn, xe_min[rr], xe_max[rr],
                                 ye_min[rr], ye_max[rr], vxe_min[rr], vxe_max[rr],
                                 vye_min[rr], vye_max[rr], vze_min[rr], vze_max[rr],
                                 axe_min[rr], axe_max[rr], aye_min[rr], aye_max[rr]))

    out.close()

    import matplotlib.font_manager
    prop = matplotlib.font_manager.FontProperties(size=12)
    sythesis.usetexTrue()
    mosaic = np.where(msc == True)[0]
    cntrl = np.where(msc == False)[0]

    xye = (xe_all + ye_all) / 2.0
    vxye = (vxe_all + vye_all) / 2.0
    axye = (axe_all + aye_all) / 2.0

    py.clf()
    py.figure(figsize=(12,4))
    py.subplots_adjust(wspace=0.32, hspace=0.32, left=0.08, right=0.95, top=0.95,
                       bottom=0.13)
    py.subplot(1,3,1)
    py.semilogy(r2d_all[cntrl],xye[cntrl],'k.',ms=9,label='Central')
    py.semilogy(r2d_all[mosaic],xye[mosaic],'ks',mfc='None',mec='k',
                mew=1.5,ms=5,label='Mosaic')
    #py.semilogy(r2d_all,xe_all,'rs',ms=4,label='X')
    #py.semilogy(r2d_all,ye_all,'b.',ms=6,label='Y')
    py.legend(numpoints=1,fancybox=True,prop=prop,loc=4)
    py.axis([0,14,0.03,1.1])
    py.xlabel('Projected Radius (arcsec)')
    py.ylabel(r'{\bf \huge{$\sigma_{pos}$}} (mas)')
    #py.ylabel('Positional Error (mas)')
    py.subplot(1,3,2)
    py.semilogy(r2d_all[cntrl],vxye[cntrl],'k.',ms=9)
    py.semilogy(r2d_all[mosaic],vxye[mosaic],'ks',mfc='None',mec='k',
                mew=1.5,ms=5,label='Mosaic')
    #py.semilogy(r2d_all,vxe_all,'rs',ms=4)
    #py.semilogy(r2d_all,vye_all,'b.',ms=6)
    py.axis([0,14,0.005,1.1])
    py.xlabel('Projected Radius (arcsec)')
    py.ylabel(r'{\bf \huge{$\sigma_{vel}$}} (mas yr$^{-1}$)')
    #py.ylabel(r'Velocity Error (mas yr$^{-1}$)')
    py.subplot(1,3,3)
    py.semilogy(r2d_all[cntrl],axye[cntrl],'k.',ms=9)
    #py.semilogy(r2d_all[mosaic],axye[mosaic],'ks',mfc='None',mec='k',
    #            mew=1.5,ms=5,label='Mosaic')
    #py.semilogy(r2d_all,axe_all,'rs',ms=4)
    #py.semilogy(r2d_all,aye_all,'b.',ms=6)
    py.axis([0,6.5,0.005,0.5])
    py.xlabel('Projected Radius (arcsec)')
    #py.ylabel(r'Acceleration Error (mas yr$^{-2}$)')
    py.ylabel(r'{\bf \huge{$\sigma_{acc}$}} ($\mu$as yr$^{-2}$)')
    py.savefig(root+alnDir+'plots/astrometric_error_for_simulations.png')
    py.savefig(root+alnDir+'plots/eps/astrometric_error_for_simulations.eps')
    sythesis.usetexFalse()


def get_errors(r2d, epochs):
    """
    Returns errors in r, v, a based on the projected radius.
    Must have a file that contains positional errors for various
    radial bins from 0-14 arcsec.
    """
    efile = asciidata.open(root+alnDir+'tables/pos_error_vs_radius.dat')
    rLo = efile[0].tonumpy()
    rHi = efile[1].tonumpy()
    xe = efile[2].tonumpy()
    ye = efile[3].tonumpy()
    vxe = efile[4].tonumpy()
    vye = efile[5].tonumpy()
    vze = efile[6].tonumpy()
    axe = efile[7].tonumpy()
    aye = efile[8].tonumpy()

    

    
def create_generators(num, delta, firstseed=None):
    """Return list of num distinct generators.
    Each generator has its own unique segment of delta elements
    from Random.random()'s full period.
    Seed the first generator with optional arg firstseed (default
    is None, to seed from current time).
    """

    g = random.Random(firstseed)
    result = [g]
    for i in range(num - 1):
        laststate = g.getstate()
        g = random.Random()
        g.setstate(laststate)
        g.jumpahead(delta)
        result.append(g)
    return result


def test_inversion_problem(mockdir='ecc_bias_sim/'):
    """
    Assuming some orbit, create mock data using
    orbits.kep2xyz(), then input this mock data
    into orbits.xyz2kep() to see if we get the
    same orbit out.
    """

    # Read in the mock data already created
    # Astrometric units are: arcsec, mas/yr, mas/yr^2
    mockdata = open(root + alnDir + mockdir + '/circularFlatMC_results.pickle')
    mbh = pickle.load(mockdata)
    dist = pickle.load(mockdata)
    orb_all = pickle.load(mockdata)
    sma_all = pickle.load(mockdata)
    xM = pickle.load(mockdata) # (+x to east)
    yM = pickle.load(mockdata)
    zM = pickle.load(mockdata)
    vxM = pickle.load(mockdata) # (+x to east)
    vyM = pickle.load(mockdata)
    vzM = pickle.load(mockdata) # also in mas/yr!
    axM = pickle.load(mockdata) # (+x to east)
    ayM = pickle.load(mockdata)
    azM = pickle.load(mockdata)
    t0M = pickle.load(mockdata) # periapse passage
    t_obs = pickle.load(mockdata) # observational time for mock data point
    mockdata.close()

    cc = objects.Constants()
    asy_to_kms = dist * cc.cm_in_au / (1.e5*cc.sec_in_yr)

    # Put velocities in km/s
    vxM = vxM / 1.e3 * asy_to_kms
    vyM = vyM / 1.e3 * asy_to_kms
    vzM = vzM / 1.e3 * asy_to_kms

    # cgs units
    rM = np.sqrt(xM**2 + yM**2 + zM**2) * dist * cc.cm_in_au
    vM = np.sqrt(vxM**2 + vyM**2 + vzM**2) * 1.0e5

    # Assumed orbit for mock data
    e_in = 0.0
    incl_in = orb_all[0].i 
    Omega_in = orb_all[0].o 
    t0_in = orb_all[0].t0 
    prd_in = orb_all[0].p
    w_in = orb_all[0].w

    nstars = len(xM)
    i = np.zeros(nstars, dtype=float)
    e = np.zeros(nstars, dtype=float)
    evec = np.zeros((nstars, 3), dtype=float)
    w = np.zeros(nstars, dtype=float)
    o = np.zeros(nstars, dtype=float)
    p = np.zeros(nstars, dtype=float)
    t0 = np.zeros(nstars, dtype=float)
    ph = np.zeros(nstars, dtype=float)
    h = np.zeros((nstars, 3), dtype=float)
    h_mag = np.zeros(nstars, dtype=float)
    theta = np.zeros(nstars, dtype=float)

    for nn in range(len(xM)):
        rvec = np.array([xM[nn], yM[nn], zM[nn]])
        vvec = np.array([vxM[nn], vyM[nn], vzM[nn]])
        revec = np.zeros(3, dtype=float)
        vevec = np.zeros(3, dtype=float)

        orb = orbits.Orbit()
        orb.xyz2kep(rvec, vvec, revec, vevec, t_obs,
                    mass=mbh, dist=dist)
        # rvec comes back in cgs units after calling orb.xyz2kep
        # vvec comes back in mks units after calling orb.xyz2kep

        i[nn] = orb.i
        e[nn] = orb.e
        evec[nn] = orb.evec
        w[nn] = orb.w
        o[nn] = orb.o
        p[nn] = orb.p
        t0[nn] = orb.t0
        ph[nn] = orb.ph
        h[nn] = orb.hvec


    pdb.set_trace()

    py.clf()
    py.figure(figsize=(10,5))
    py.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1,
                       wspace=0.3, hspace=0.3)
    py.subplot(1,3,1)
    py.hist(e, bins=np.arange(e.min(),e.max(),0.05),histtype='step')
    py.xlabel('Eccentricity')
    py.subplot(1,3,2)
    py.hist(i, bins=np.arange(i.min(),i.max(),1.0),histtype='step')
    py.xlabel('Inclination (deg)')
    py.subplot(1,3,3)
    py.hist(o, bins=np.arange(o.min(),o.max(),1.0),histtype='step')
    py.xlabel('Omega (deg)')
    py.savefig(root+alnDir+mockdir+'plots/test_inversion_hist_params.png')
    py.close()

    vtot = np.sqrt(vxM**2 + vyM**2 + vzM**2)
    py.clf()
    py.figure(figsize=(8,8))
    py.subplot(2,2,1)
    py.plot(theta, e, 'k.')
    py.xlabel('Theta (r cross v))')
    py.ylabel('Eccentricity')
    py.subplot(2,2,2)
    py.plot(theta, i, 'k.')
    py.xlabel('Theta (r cross v))')
    py.ylabel('Inclination (deg)')
    py.subplot(2,2,3)
    py.plot(theta, o, 'k.')
    py.xlabel('Theta (r cross v))')
    py.ylabel('Omega (deg)')
    py.show()



# Test what a uniform distribution sampling looks like:
def test_uni():
    gens = create_generators(2, 1000000)
    zgen1 = gens[0]
    zgen2 = gens[1]

    z1 = np.zeros(1e6,dtype=float)
    z2 = np.zeros(2e6,dtype=float)

    for ii in range(0,1e6,2):
        # randomly sample
        z1[ii] = zgen1.uniform(0.0, 1.0)
        z1[ii+1] = -z1[ii]
        
    for ii in range(2e6):
        z2[ii] = zgen2.uniform(-1.0, 1.0)

    py.close('all')
    py.clf()
    py.subplot(2,1,1)
    py.hist(z1,bins=np.arange(-1,1,0.001),histtype='step')
    py.subplot(2,1,2)
    py.hist(z2,bins=np.arange(-1,1,0.001),histtype='step')
    py.show()

