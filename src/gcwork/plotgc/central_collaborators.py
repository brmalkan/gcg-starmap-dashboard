import math, os, shutil
import pylab, pyfits
import asciidata
import matplotlib as mlib
import matplotlib.image as mpimg
from matplotlib.colors import colorConverter
from gcwork import starTables
from gcwork import starset
from gcwork import orbits
from gcwork import objects
from gcwork.plotgc import gccolors 
from numarray import *
from numpy import *
import numpy as np
import matplotlib.colors
import pdb

def usetexTrue():
    pylab.rc('text', usetex=True)
    pylab.rc('font', size=16)
    pylab.rc('axes', titlesize=20, labelsize=20)
    pylab.rc('xtick', labelsize=16)
    pylab.rc('ytick', labelsize=16)

def usetexFalse():
    pylab.rc('text', usetex=False)
    pylab.rc('font', size=14)
    pylab.rc('axes', titlesize=16, labelsize=16)
    pylab.rc('xtick', labelsize=14)
    pylab.rc('ytick', labelsize=14)


def orbitsAnimate(years=None, rad=0.25, alnDir='/u/syelda/research/gc/aligndir/11_10_26/',
                  align='align/align_d_rms_1000_abs_t',
                  poly='polyfit_c/fit'):
    """
    Create frames for the central arcsecond orbits animation, highlighting
    only S0-2. Pictures of the collaborators who have contributed
    to our knowledge of S0-2's orbit over the years are added in Photoshop.
    
    Set rad to indicate the radial extent of orbits animation.
    Default is rad=0.25, which will show the central arcsecond orbits.

    orbitFile (str):    File with all the orbital elements. Must include
    			full path to the file.
    """

    ##########
    #
    # START - Modify stuff in here only
    #
    ##########
    # Today's date
    today = 2012.4
    root = '/u/syelda/research/forAndrea/orbitsAnim_120420_crafoord/v3_collabo/'
    orbitFile = root + 'orbits_movie_crafoord.dat'
    collaboFile = root + 'collabo.dat'
    outdir = root
    
    
    # Load up a starset of just those stars in orbits_movie.dat
    s = getOrbitStars(orbitFile=orbitFile, align=alnDir+align, poly=alnDir+poly)
    tab = asciidata.open(orbitFile)

    ##########
    #
    # STOP - Modify stuff in here only
    #
    ##########

    name = s.getArray('name')
    mag = s.getArray('mag')

    # Get plotting properties from the orbits.dat file
    discovered = tab[9].tonumpy()  # Discovery date
    xshift1 = tab[10].tonumpy()    # Shifts for labels (in first frame)
    yshift1 = tab[11].tonumpy()
    xshift2 = tab[12].tonumpy()    # Shifts for labels (in last frame)
    yshift2 = tab[13].tonumpy()
    colors = [tab[14][ss].strip() for ss in range(tab.nrows)]

    # Determine the mass assuming a distance of 8.0 kpc
    star0orb = s.stars[0].orbit
    dist = 8000.0 # in parsec
    axis = (star0orb.a / 1000.0) * dist # in au
    mass = (axis)**3 / star0orb.p**2

    # Set the duration of the animation from the years keyword
    if (years == None):
        idx = name.index('S0-2')

        # Use S0-2's orbital period, rounded up to the nearest year
        years = math.ceil(s.stars[idx].orbit.p)

    # Array of time steps (0.05 yr steps so that S0-2's orbit doesn't look segmented)
    # But in Photoshop, we'll only import and use frames every 0.1 years.
    t = np.arange(1995.5, 1995.5+years, 0.05, dtype=float)

    # Do a flux scaling so that all the stars look good in our image.
    flux = 10.0**(mag/-4.0) # Had to change to get 19th mag star to show up!
    flux /= flux.max()

    # Loop through all the stars and make an array of the X and Y positions
    # as a function of time. Store this on the star object as
    #   star.xanim -- array of X positions at each time step in t
    #   star.yanim -- array of Y positions at each time step in t
    for star in s.stars:
        (r, v, a) = star.orbit.kep2xyz(t, mass=mass, dist=dist)

        star.xanim = r[:,0].copy()
        star.yanim = r[:,1].copy()

    # Make an image 500x500 pixels (1" x 1")
    imgSize = 500 # pixels
    scale = (2.0*rad) / imgSize
    xaxis = (np.arange(imgSize, dtype=float) - (imgSize/2.0)) # xextent
    xaxis *= -scale
    yaxis = (np.arange(imgSize, dtype=float) - (imgSize/2.0)) # yextent
    yaxis *= scale

    # Make grids of X/Y value at each pixel
    xx, yy = pylab.meshgrid(xaxis, yaxis)

    ##########
    #
    # Create image with gaussian PSF for each star
    #
    ##########
    fwhm = 0.020   # Make 20 mas instead of 55 mas

    for tt in range(len(t)):
        time = t[tt]
        img = np.zeros((imgSize, imgSize), dtype=float)
        xorb = []
        yorb = []
        
        for ss in range(len(s.stars)):
            star = s.stars[ss]

            xpos = star.xanim[tt]
            ypos = star.yanim[tt]

            # Make a 2D gaussian for this star
            psf = np.exp(-((xx - xpos)**2 + (yy - ypos)**2) / fwhm**2)

            img += flux[ss] * psf

        pylab.close(2)
        # For higher resolution, just increase figsize slightly
        pylab.figure(2, figsize=(5,5))
        #pylab.subplots_adjust(left=0,right=1.0,top=1.0,bottom=0)
        pylab.clf()
        pylab.axes([0.0, 0.0, 1.0, 1.0])
        pylab.axis('off')
        # Mark Sgr A*
        star5 = gccolors.Star(0,0,0.02)
        sgraColor = 'white'
        pylab.fill(star5[0], star5[1], fill=False,edgecolor=sgraColor,
                   linewidth=1.5, hold=True)

        def plotTrails():
            # Plot the trails for S0-2
            s02 = name.index('S0-2')

            star = s.stars[s02]

            before = where((t < time) & (t < discovered[s02]))[0]
            during = where((t < time) & (t >= discovered[s02]) & (t <= today))[0]
            future = where((t < time) & (t > today))[0]

            # Dashed before discovery and in the future
            if (len(before) > 0):
                pylab.plot(star.xanim[before], star.yanim[before], '--',
                           color=colors[s02], linewidth=2, hold=True)
            if (len(during) > 0):    
                pylab.plot(star.xanim[during], star.yanim[during], '-',
                           color=colors[s02], linewidth=2, hold=True)
            if (len(future) > 0):    
                pylab.plot(star.xanim[future], star.yanim[future], '--',
                           color=colors[s02], linewidth=2, hold=True)

        def addText():
            # Draw an outline box
            bx = rad-(0.02*rad)
            pylab.plot([bx, -bx, -bx, bx, bx], [-bx, -bx, bx, bx, -bx],
                       color='white', linewidth=2)#, hold=True)
    
            #pylab.text(rad-(0.1*rad), rad-(0.15*rad), str(t[tt]), color='white',
            #pylab.text(rad-(0.1*rad), -rad+(0.15*rad), str(t[tt]), color='white',
            pylab.text(0.04, 0.22, str(t[tt]), color='white',
                       fontsize=16, fontweight='bold',
                       horizontalalignment='left', verticalalignment='bottom')

        plotTrails()
        addText()

        # Add the background image of the stars
        cmap = gccolors.idl_rainbow()
        pylab.imshow(sqrt(img), origin='lowerleft', cmap=cmap,
                     extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]],
                     vmin=sqrt(0.01), vmax=sqrt(1.0))

        if (tt % 2 == 0):
            strtt = str(t[tt]) + '0'
        else:
            strtt = str(t[tt])

        pylab.axis([rad, -1.*rad, -1*rad, rad]) # Including this takes FOREVER!
        pylab.savefig('%s/img_%s.png' % (outdir, strtt), dpi=100)

        
def getOrbitStars(orbitFile, align='align/align_d_rms_t',
                  poly='polyfit_d/fit',
                  flatten=False):

    orbs = starTables.Orbits(orbitFile=orbitFile)

    # Load up observed data
    s = starset.StarSet(align, relErr=True)
    s.loadPolyfit(poly, arcsec=True)
    s.loadPolyfit(poly, arcsec=True, accel=True)

    # Need to repair a couple of names
    for ii in range(len(s.stars)):
        star = s.stars[ii]

        if (star.name == '29star_810'):
            star.name = 'S0-103'
        if (star.name == '36star_1491'):
            star.name = 'S0-104'

    # Trim out only the stars that are in orbits.dat
    stars = []
    allNames = [star.name for star in s.stars]
    for ii in range(len(orbs.name)):
        # Name in the orbits file
        name = orbs.name[ii]
        
        # Find this star in the align data
	try:
	    idx = allNames.index(name)

	    star = s.stars[idx]

            # Create an orbit object for this star. We will
            # use this to make the model orbits later on
            star.orbit = orbits.Orbit()
            star.orbit.a = orbs.a[ii]
            star.orbit.w = orbs.w[ii]
            star.orbit.o = orbs.o[ii]
            if (flatten):
                if (orbs.i[ii] > 90.0):
                    star.orbit.i = 180.0
                else:
                    star.orbit.i = 0.0
            else:
                star.orbit.i = orbs.i[ii]
            star.orbit.e = orbs.e[ii]
            star.orbit.p = orbs.p[ii]
            star.orbit.t0 = orbs.t0[ii]

            stars.append(star)

	except ValueError, e:
	    # Couldn't find the star in the align data
            print 'Could not find %s' % (name)
	    continue

    # Reset the starset's star list. Should only contain orbit stars now.
    s.stars = stars

    return s
