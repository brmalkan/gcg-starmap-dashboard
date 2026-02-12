import math
import nmpfit_sy
import numpy as np
import pylab as py
import pyfits
import histogram2d as h2d
import asciidata, shutil
import matplotlib.colors
import random
import sqlite3 as sqlite
from scipy import stats
from gcwork import starset
from gcwork import starTables
from gcwork import objects
from gcwork import util
from gcwork import young
from gcwork import oldNames 
import syAccel
import pdb

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

def plot_dither_on_image(absDir='13_08_21/',absFile='source_list/absolute_refs_noAccel.dat',
                         plotNewOsiris=False,suffix=''):

    root = '/g/ghez/align/%s' % absDir

    imgFile = '/u/ghezgroup/data/gc/06maylgs1/combo/mag06maylgs1_msr_kp.fits'
    scale = 0.00993
    sgra = [1596.,1010.]
    img = pyfits.getdata(imgFile)
    imgsize = (img.shape)[0]

    # Make axes for images in arcsec
    pixL = np.arange(0,imgsize)
    xL = [-1*(xpos - sgra[0])*scale for xpos in pixL]
    yL = [(ypos - sgra[1])*scale for ypos in pixL]

    # Find the maser positions
    data = asciidata.open(root+absFile)
    name = np.array([data[0][ss].strip() for ss in range(data.nrows)])
    mag = data[1].tonumpy()
    x = data[2].tonumpy()
    y = data[3].tonumpy()
    
    maserNames = ['irs9', 'irs7', 'irs12N', 'irs28',
                  'irs10EE', 'irs15NE', 'irs17']
    midx = [np.where(name == mm)[0][0] for mm in maserNames]
    xmsr = x[midx]
    ymsr = y[midx]

    usetexTrue()
    py.figure(1)
    py.figure(figsize=(8,8))
    py.subplots_adjust(left=0.1,right=0.98,top=0.95,
                    wspace=0.3,hspace=0.5)
    py.clf()
    py.imshow(np.log10(img+1), aspect='equal', interpolation='bicubic',
              extent=[max(xL), min(xL), min(yL), max(yL)],
              vmin=2.2,vmax=5, origin='lowerleft', cmap=py.cm.gray_r)

    def plot_dither_pattern():
        # Create arrays for overplotting the maser mosaic dither pattern
        msrCx = np.array([-0.5, 9.7, 9.7, -0.5, -0.5]) # 10.2 arcsecond width
        msrCy = np.array([-3.8, -3.8, 6.4, 6.4, -3.8]) # 10.2 arcsecond height

        py.plot(msrCx, msrCy, 'k--') # Central dither
        py.plot(msrCx+6.0,msrCy+6.0,'k--')
        py.plot(msrCx-6.0,msrCy+6.0,'k--')
        py.plot(msrCx+6.0,msrCy-6.0,'k--')
        py.plot(msrCx-6.0,msrCy-6.0,'k--')
        py.plot(msrCx+6.0,msrCy,'k--')
        py.plot(msrCx-6.0,msrCy,'k--')
        py.plot(msrCx,msrCy+6.0,'k--')
        py.plot(msrCx,msrCy-6.0,'k--')

    # get the maser positions separately for this plot
    plot_dither_pattern()
    py.plot(xmsr, ymsr, 'ko', mfc='none',mec='k',mew=1.5,ms=12)
    py.plot([0],[0],'r+',ms=8,mew=1.5)
    if plotNewOsiris == True:
        py.plot([14.5,-5.5,-5.5,14.5,14.5],[-8,-8,12,12,-8],'r-',lw=2)
    py.xlabel('RA Offset (arcsec)')
    py.ylabel('Dec Offset (arcsec)')
    py.axis([15.5, -6.5, -9.6, 12.4])
    #py.axis([15.5, -6.5, -9, 12])
    py.savefig(root+'plots/mosaic_dither%s.png' % suffix)
    py.close(1)
    usetexFalse()


def plotVels(absDir='10_06_22/',
             absFile='lis/absolute_refs.dat',minRad=5.0,maxRad=10.):
    """
    Plot velocity vectors over a maser mosaic image
    and plots PM vector angle.
    """
    root = '/u/syelda/research/gc/absolute/' + absDir

    maserNames = ['irs9', 'irs7', 'irs12N', 'irs28',
                  'irs10EE', 'irs15NE', 'irs17']

    data = asciidata.open(root+absFile)
    #name = data[0]._data
    name = np.array([data[0][ss].strip() for ss in range(data.nrows)])
    mag = data[1].tonumpy()
    x = data[2].tonumpy()
    y = data[3].tonumpy()
    xerr = data[4].tonumpy()
    yerr = data[5].tonumpy()
    vx = data[6].tonumpy()
    vy = data[7].tonumpy()
    vxerr = data[8].tonumpy()
    vyerr = data[9].tonumpy()
    r = np.hypot(x,y)
    use = data[11].tonumpy()
    print 'Maximum radius: %4.2f' % r.max()

    # Now grab everything outside of radius minRad
    ridx = np.where((r > minRad) & (r < maxRad))[0]
    x = x[ridx] #* -1.
    y = y[ridx]
    xerr = xerr[ridx]
    yerr = yerr[ridx]
    vx = vx[ridx] * -1.  # vx originally increases toward east, undo this for plotting
    vy = vy[ridx]
    vxerr = vxerr[ridx]
    vyerr = vyerr[ridx]
    use = use[ridx]
    r = r[ridx]

    # Do not include young stars in alignment b/c of known net rotation
    yng = young.youngStarNames()

    # Find the ones that are used as reference stars
    #ref = np.where((use == 8) | (use == 0))[0]
    #ref = np.where(use == 1)[0]
    #print 'Number of restricted stars in this radial bin: %3i' % len(x[ref])

    for ii in range(len(x)):
        # Exclude young stars
        if name[ii] in yng:
            use[ii] = 0

    idYng = np.where(use == 0)[0]
    nYng = len(idYng)
    print 'Number of young stars to be excluded: %s' % nYng

    idLate = np.where(use == 1)[0]
    print 'Number of non-known-young stars: %s' % len(idLate)

    imgFile = '/u/ghezgroup/data/gc/06maylgs1/combo/mag06maylgs1_msr_kp.fits'
    scale = 0.00993
    sgra = [1596.,1006.]
    img = pyfits.getdata(imgFile)
    imgsize = (img.shape)[0]
    #x_new = sgra[0]-x/scale
    #y_new = sgra[1]+y/scale
    #x_new = (sgra[0]-x/scale)*scale
    #y_new = (sgra[1]+y/scale)*scale

    # Make axes for images in arcsec
    pixL = np.arange(0,imgsize)
    xL = [-1*(xpos - sgra[0])*scale for xpos in pixL]
    yL = [(ypos - sgra[1])*scale for ypos in pixL]

    # For the net rotation, do not use young stars
    # Get everything but the young stars
    foo = np.where(use != 0)[0]
    pmAngle = np.arctan2(vy[foo],vx[foo])*(180./np.pi) # already multiplied vx by -1 to get increase towards west
    radAngle = np.arctan2(y[foo],-x[foo])*(180./np.pi) # need to multiply x by -1 here to get increase towards west

    pmAngle_NE = []
    radAngle_NE = []
    # numpy computes arctan2 angle from line segment (0,0)-(1,0) to (0,0)-(vx,vy)
    # Need to convert this so that the angle is computed from North thru East
    for aa in range(len(pmAngle)):
        # First put all the data together
        if ((pmAngle[aa] > -90.) or ((pmAngle[aa] < 180) and (pmAngle[aa] > 0))):
            pmAngle0 = pmAngle[aa] - 90.
        else:
            pmAngle0 = pmAngle[aa] + 270.
        pmAngle_NE = np.concatenate([pmAngle_NE, [pmAngle0]])
        # First put all the data together
        if ((radAngle[aa] > -90.) or ((radAngle[aa] < 180) and (radAngle[aa] > 0))):
            radAngle0 = radAngle[aa] - 90.
        else:
            radAngle0 = radAngle[aa] + 270.
        radAngle_NE = np.concatenate([radAngle_NE, [radAngle0]])


    # Bin up the proper motion vectors and see how many stars are in each bin
    nPerBin = []
    inc = 20
    bins = np.arange(-180, 180, inc)
    for ii in bins:
        idx = np.where((pmAngle_NE > ii) & (pmAngle_NE < ii+inc))[0]
        nPerBin = np.concatenate([nPerBin, [len(idx)]])
    # Get the non-zero bins and shift the bins to the middle of the interval (inc/2)
    kp = np.where(nPerBin != 0)[0]
    bins, nPerBin = bins[kp]+inc/2., nPerBin[kp]
    binsr = [math.radians(rr) for rr in bins]

    # Fit a cosine to each of the histograms
    c0 = [(nPerBin.max()-nPerBin.min())/2., math.radians(31.), nPerBin.mean()]
    cos0 = fitCos(c0,[binsr,nPerBin,np.sqrt(nPerBin)])
    cosPar = cos0.params

    # Create the cosine curve
    cosX = np.arange(-math.pi, math.pi, 0.1)
    cosY = cosPar[0]*np.cos(2.0*(cosX-cosPar[1]))+cosPar[2]

    py.rc('text', usetex=True)
    # Plot the distribution of angle of the proper motion vector
    # for the non-known-young stars
    py.figure(1)
    py.clf()
    pyxerr = np.repeat(inc,len(bins))
    py.errorbar(bins, nPerBin, xerr=np.repeat(inc/2.,len(bins)), yerr=np.sqrt(nPerBin),\
                fmt='k.', label='all')
    py.plot(cosX*180./math.pi,cosY,'k-')
    # Plot the angle of the Galactic plane (31.1 deg) and 180 deg opposite
    py.plot([31,31],[0,max(nPerBin)+10],'k--')
    py.plot([-149,-149],[0,max(nPerBin)+10],'k--')
    py.axis('tight')
    py.xlabel('Proper Motion Direction E of N (deg)',fontsize=18)
    py.ylabel('N',fontsize=18)
    py.savefig(root + 'plots/absolute_net_rotation.png')
    py.savefig(root + 'plots/absolute_net_rotation.eps')
    py.close(1)

    # Find the masers
    midx = []
    for i in range(len(maserNames)):
        ms = np.where(name == maserNames[i])[0]
        if len(ms) > 0:
            midx = np.concatenate([midx, ms])
        else:
            print '%s not found in reference list!' % maserNames[i]

    midx = [np.int(aa) for aa in midx]
    print 'Only %i masers in this reference list' % len(midx)

    py.clf()
    fig = py.figure(2, figsize=(8,8))
    py.gray()
    py.subplots_adjust(left=0.1,right=0.98,top=0.95,
                    wspace=0.3,hspace=0.5)
    py.imshow(np.log10(img+1), aspect='equal', interpolation='bicubic',
              extent=[max(xL), min(xL), min(yL), max(yL)],vmin=2.2,vmax=5,
              origin='lowerleft', cmap=py.cm.gray_r)

    xmsr = x[midx]
    ymsr = y[midx]
    # Make the young stars, and masers a different color
    qvrMsr = py.quiver([x[0:7]],[y[0:7]],[-vx[0:7]],[vy[0:7]],color='red',angles='xy',units='y',scale=8,\
                    width=0.1,headwidth=4,headlength=4,headaxislength=2)
    qvr = py.quiver([x[7:]],[y[7:]],[-vx[7:]],[vy[7:]],color='black',angles='xy',units='y',scale=8, \
                    width=0.05,headwidth=4,headlength=4,headaxislength=2)
    py.plot([sgra[0]],[sgra[1]],'r+',ms=8)
    py.plot(x[7:], y[7:], 'ko', mec='k',ms=2)
    py.plot(xmsr, ymsr, 'ro', mec='r',ms=2)
    py.plot([5,-5,-5,5,5],[-5,-5,5,5,-5],'b-',lw=2)
    py.plot([3.5,-1.5,-1.5,3.5,3.5],[-2.5,-2.5,2.5,2.5,-2.5],'b-',lw=2)
    py.quiverkey(qvr,12,-10.5,-5,'5 mas/yr',coordinates='data',labelpos = 'W', 
              fontproperties={'size': 'smaller'})
    py.xlabel('X (arcsec)')
    py.ylabel('Y (arcsec)')
    #py.title('Absolute Velocities of Reference Stars')
    py.axis([15.5, -6.5, -9, 12])
    py.savefig(root+'plots/absolute_velVectors.png')
    py.close(2)
 
    # Now just plot the positions
    py.figure(3,figsize=(7,7))
    py.subplots_adjust(left=0.1,right=0.98,top=0.95,
                    wspace=0.3,hspace=0.5)
    py.clf()
    py.imshow(np.log10(img+1), aspect='equal', interpolation='bicubic',
              extent=[max(xL), min(xL), min(yL), max(yL)],
              vmin=2.2,vmax=5, origin='lowerleft', cmap=py.cm.gray_r)
    # get the maser positions separately for this plot
    py.plot(x[7:], y[7:], 'ko', mfc='none',mec='k',mew=0.75,ms=6)
    py.plot(xmsr, ymsr, 'ro', mfc='none',mec='r',mew=0.75,ms=8)
    py.plot([sgra[0]],[sgra[1]],'r+',ms=8)
    py.plot([5,-5,-5,5,5],[-5,-5,5,5,-5],'b--',lw=2)
    py.plot([3.5,-1.5,-1.5,3.5,3.5],[-2.5,-2.5,2.5,2.5,-2.5],'b--',lw=2)
    py.xlabel('X (pix)')
    py.ylabel('Y (pix)')
    py.title('Positions of Reference Stars')
    py.axis([15.5, -6.5, -9, 12])
    py.savefig(root+'plots/absolute_positions.png')
    py.close(3)

    py.figure(4,figsize=(7,7))
    py.clf()
    py.plot(radAngle_NE,pmAngle_NE,'k.')
    py.xlabel('Radial Vector E of N (deg)')
    py.ylabel('Proper Motion Direction E of N (deg)')
    py.savefig(root+'plots/rad_pm_angle.png')
    py.close(4)

    # position and error of sgr a from Yelda et al. 2010 (table 5)
    sgr_x = -0.31 # mas
    sgr_y = 1.44
    sgr_xe = 0.55 # mas
    sgr_ye = 0.59

    # Convert velocities to polar coordinates
    # and fit a Gaussian to one of the histograms
    # use non-known-young stars only
    vr, vt, vre, vte = rec2polar_sgra(-x[idLate],y[idLate],vx[idLate],vy[idLate],vxerr[idLate],vyerr[idLate],sgr_xe,sgr_ye)
    binsIn = py.arange(-11, 11, 0.5)
    prop = matplotlib.font_manager.FontProperties(size=12)
    py.figure(5,figsize=(6,6))
    py.clf()
    (nx,bx,ptx) = py.hist(vr,binsIn,color='r',histtype='step',label=r'$v_r$',lw=1.5)
    (ny,by,pty) = py.hist(vt,binsIn,color='b',histtype='step',label=r'$v_t$',lw=1.5)
    p0 = [80, 0, 3]
    print 'V_rad Gaussian fit:'
    rfit = fitGaussianMP(p0,[bx, nx, np.sqrt(nx)],1)
    rparams = rfit.params
    rline_r = np.arange(-11,11,.1)
    rline_v = modelGaussian(rline_r, rparams[0], rparams[1], rparams[2])
    py.plot(rline_r,rline_v,'k-',lw=1.5)
    py.xlabel('Component of Velocity (mas/yr)')
    py.ylabel('N')
    py.legend(prop=prop,numpoints=1)
    py.savefig(root + 'plots/polarVels_hist.png')
    py.close(5)

    # Just plot tangential vels (excluding young stars)
    py.figure(6,figsize=(6,6))
    py.clf()
    (nt,bt,pt) = py.hist(vt,binsIn,color='k',histtype='step',lw=1.5)
    maxN = nt.max()
    py.plot([0,0],[0,maxN+5],'k--',lw=2)
    py.axis([-15,15,0,maxN+5])
    py.xlabel('Tangential Velocity (mas yr$^{-1}$)',fontsize=18)
    py.ylabel('N',fontsize=18)
    #py.title('Old Stars')
    py.savefig(root + 'plots/tangVel_hist.png')
    py.savefig(root + 'plots/tangVel_hist.eps')
    py.close(6)

    print ''
    # Do the same for the young stars
    # Convert velocities to polar coordinates
    # and fit a Gaussian to one of the histograms
    vr_y, vt_y, vre_y, vte_y = rec2polar_sgra(-x[idYng],y[idYng],vx[idYng],vy[idYng],vxerr[idYng],vyerr[idYng],sgr_xe,sgr_ye)
    binsIn = py.arange(-11, 20, 1)
    prop = matplotlib.font_manager.FontProperties(size=12)
    py.figure(7,figsize=(6,6))
    py.clf()
    (nx,bx,ptx) = py.hist(vr_y,binsIn,color='r',histtype='step',label=r'$v_r$',lw=1.5)
    (ny,by,pty) = py.hist(vt_y,binsIn,color='b',histtype='step',label=r'$v_t$',lw=1.5)
    p0 = [15, 5, 5]
    print 'Young stars'
    print 'V_rad Gaussian fit:'
    rfit = fitGaussianMP(p0,[bx, nx, np.sqrt(nx)],1)
    rparams = rfit.params
    rline_r = np.arange(-11,20,.1)
    rline_v = modelGaussian(rline_r, rparams[0], rparams[1], rparams[2])
    py.plot(rline_r,rline_v,'r-',lw=1.5)
    py.plot([rparams[1],rparams[1]],[0,13],'r--',lw=2)
    print 'V_tan Gaussian fit:'
    tfit = fitGaussianMP(p0,[by, ny, np.sqrt(ny)],1)
    tparams = tfit.params
    tline_r = np.arange(-11,20,.1)
    tline_v = modelGaussian(tline_r, tparams[0], tparams[1], tparams[2])
    py.plot(tline_r,tline_v,'b-',lw=1.5)
    py.plot([tparams[1],tparams[1]],[0,13],'b--',lw=2)
    py.xlabel('Component of Velocity (mas/yr)')
    py.ylabel('N')
    py.legend(prop=prop,numpoints=1)
    py.savefig(root + 'plots/polarVels_hist_young.png')
    py.close(7)

    # Just plot tangential vels for UCB talk
    py.figure(8,figsize=(6,6))
    py.clf()
    py.hist(vt_y,binsIn,color='k',histtype='step',lw=1.5)
    py.plot([0,0],[0,10],'k--',lw=2)
    py.axis([-15,15,0,10])
    py.xlabel('Tangential Velocity (mas/yr)')
    py.ylabel('N')
    #py.title('Young Stars')
    py.savefig(root + 'plots/tangVel_hist_young.png')
    py.savefig(root + 'plots/tangVel_hist_young.eps')
    py.close(8)

    # Plot the angular velocities for the late-type stars
    ang_vel = vt / r[idLate]
    binsIn = py.arange(-7, 7, 0.2)
    py.figure(9)
    py.figure(figsize=(6,6))
    py.clf()
    (na,ba,pa) = py.hist(ang_vel,binsIn,color='k',histtype='step',lw=1.5)
    maxN = na.max()
    py.plot([0,0],[0,maxN+10],'k--',lw=2)
    py.xlabel('Angular Velocity (mas yr$^{-1}$ arcsec$^{-1}$)',fontsize=18)
    py.ylabel('N',fontsize=18)
    py.axis([-6,6,0,maxN+10])
    py.savefig(root + 'plots/angularVel_hist.png')
    py.savefig(root + 'plots/angularVel_hist.eps')
    py.rc('text', usetex=False)

    # Get weighted average tangential and angular velocity of late-type stars
    vt_wt = 1./vte**2
    vt_wavg = (vt*vt_wt).sum() / vt_wt.sum()
    vt_wavgerr = np.sqrt(1.0 / vt_wt.sum())

    wt = 1./vte**2
    ang_vel_wavg = (wt * ang_vel).sum() / wt.sum()
    ang_vel_wavgerr = np.sqrt(1. / wt.sum())
    print ''
    print 'Weighted average of tangential velocity for late-type stars:'
    print '%6.3f +- %6.3f mas/yr' % (vt_wavg, vt_wavgerr)
    print '%6.3f +- %6.3f km/s' % (vt_wavg*37.92, vt_wavgerr*37.92)
    print ''
    print 'Weighted average of angular velocity for late-type stars:'
    print '%6.3f +- %6.3f mas/yr/"' % (ang_vel_wavg, ang_vel_wavgerr)
    print '%6.3f +- %6.3f km/s/"' % (ang_vel_wavg*37.92, ang_vel_wavgerr*37.92)
    

def plot_vel_vectors(absDir='11_01_03/', absFile='lis/absolute_refs.dat',
                     youngOnly=False,oldOnly=False,unknownOnly=False,
                     excludeYng=False,oplot_osiris=False,
                     magCut=None,magRange=None,suffix=''):
    """
    Plot velocity vectors from an absolute (maser mosaic) analysis.
    youngOnly:   Plot up only known young stars
    oldOnly:     Plot up only known old stars
    unknownOnly: Plot up only stars not known to be either young or old
    excludeYng: set to True to remove young stars from calculation of averages
    magCut:      (integer) set to faintest magnitude to include in velocity maps
    magRange:    (integer) set to the delta of the magnitudes you want to plot
    		 For example, to plot K=15-16 stars, set magCut to 16 and magRange to 1.
    """
    root = '/u/syelda/research/gc/absolute/' + absDir

    data = asciidata.open(root+absFile)
    name = data[0].tonumpy()
    mag = data[1].tonumpy()
    x = data[2].tonumpy()
    y = data[3].tonumpy()
    xerr = data[4].tonumpy()
    yerr = data[5].tonumpy()
    vx = data[6].tonumpy()
    vy = data[7].tonumpy()
    vxerr = data[8].tonumpy()
    vyerr = data[9].tonumpy()
    r = np.hypot(x,y)
    use = data[11].tonumpy()
    print 'Maximum radius: %4.2f' % r.max()

    if magCut != None:
        if magRange == None:
            kp = np.where(mag <= magCut)[0]
            name = np.array([name[nn] for nn in kp])
            mag = mag[kp]
            x = x[kp]
            y = y[kp]
            xerr = xerr[kp]
            yerr = yerr[kp]
            vx = vx[kp]
            vy = vy[kp]
            vxerr = vxerr[kp]
            vyerr = vyerr[kp]
            r = r[kp]
            use = use[kp]
            suffix = suffix + '_mag%s' % str(magCut)
            print 'Number of stars to be used after magnitude cut: %s' % len(kp)
        else:
            kp = np.where((mag < magCut) & (mag > (magCut-1)))[0]
            name = np.array([name[nn] for nn in kp])
            mag = mag[kp]
            x = x[kp]
            y = y[kp]
            xerr = xerr[kp]
            yerr = yerr[kp]
            vx = vx[kp]
            vy = vy[kp]
            vxerr = vxerr[kp]
            vyerr = vyerr[kp]
            r = r[kp]
            use = use[kp]
            suffix = suffix + '_mag%s_to_mag%s' % (str(magCut), str(magCut-1))
            print 'Number of stars to be used after magnitude cut: %s' % len(kp)

    # Identify the young stars 
    _young = young.youngStarNames()
    yng = [str(yy) for yy in _young]
    yidx = []
    for ii in range(len(yng)):
        foo = np.where(name == yng[ii])[0]
        if len(foo) > 0:
            yidx = np.concatenate([yidx,foo])
    yidx = np.array([int(yy) for yy in yidx])
    

    # Identify the old stars 
    _old = oldNames.loadOldStars()
    old = [str(oo) for oo in _old]
    oidx = []
    for ii in range(len(old)):
        foo = np.where(name == old[ii])[0]
        if len(foo) > 0:
            oidx = np.concatenate([oidx,foo])
    oidx = np.array([int(oo) for oo in oidx])
    
    if youngOnly == True:
        idx = yidx
        suffix = suffix + '_yngOnly'
    elif oldOnly == True:
        idx = oidx
        suffix = suffix + '_oldOnly'
    elif unknownOnly == True:
        known = np.sort(np.concatenate([yidx,oidx]))
        unknown = np.setdiff1d(np.arange(len(name)),known)
        idx = unknown
        suffix = suffix + '_unknown'
    elif excludeYng == True:
        notYng = np.setdiff1d(np.arange(len(name)),yidx)
        idx = notYng
        suffix = suffix + '_noYng'
    else:
        idx = np.arange(len(name))
        suffix = suffix + '_all'

    if oplot_osiris == True:
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
            fx = row[12].split(',')
            fy = row[13].split(',')
            fx = np.concatenate([fx[0:4],[fx[0]]]) # repeat the first vertex to complete the box
            fy = np.concatenate([fy[0:4],[fy[0]]]) # repeat the first vertex to complete the box
            fx = [np.float(ii) for ii in fx]
            fy = [np.float(ii) for ii in fy]
            fx_all = np.concatenate([fx_all,fx])
            fy_all = np.concatenate([fy_all,fy])

        # Also plot the central 10" box
        cntrlX = [-5.33,5.56,5.56,-5.33,-5.33]
        cntrlY = [-6.39,-6.39,4.20,4.20,-6.39]

    print 'Plotting %i (%s)' % (len(idx), suffix) 
    py.figure(1)
    py.figure(figsize=(6,6))
    py.clf()
    qvr = py.quiver(x[idx], y[idx], vx[idx], vy[idx], color='black', angles='xy', \
                    units='y', scale=7)
    py.quiverkey(qvr,x.max()-0.5,y.max()-0.5,-2,'2 mas/yr',coordinates='data',
                 color='red',fontproperties={'size': 'smaller'})
    py.plot([0],[0],'ro')
    if oplot_osiris == True:
        for ii in range(0,len(fx_all/5),5):
            py.plot(fx_all[ii:ii+5],fy_all[ii:ii+5],'b-')
        py.plot(cntrlX,cntrlY,'r-')
    #py.axis([x[idx].max(),x[idx].min(),y[idx].min(),y[idx].max()])
    py.axis([16,-6,-10,12])
    py.xlabel('X (arcsec)')
    py.ylabel('Y (arcsec)')
    py.title('Stellar Velocities')
    py.savefig(root + 'plots/velocity_vectors%s.png' % suffix)
    py.close(1)




def rec2polar(x,y,vx,vy,vxe,vye):
    """
    Convert rectangular to polar coordinates.
    """

    vtot = np.hypot(vx,vy)
    r2d = np.hypot(x,y)

    vr = (vx*x + vy*y) / r2d
    vt = (vx*y - vy*x) / r2d
    vre = np.sqrt((vxe*x)**2 + (vye*y)**2) / r2d
    vte = np.sqrt((vxe*y)**2 + (vye*x)**2) / r2d

    return vr, vt, vre, vte

def rec2polar_sgra(x,y,vx,vy,vxe,vye,sgr_xe,sgr_ye):
    """
    Convert rectangular to polar coordinates.
    Include error in position of Sgr A* from Yelda et al. (2010).
    """

    vtot = np.hypot(vx,vy)
    r2d = np.hypot(x,y)

    vr = (vx*x + vy*y) / r2d
    vt = (vx*y - vy*x) / r2d
    vre = np.sqrt((vxe*x)**2 + (vye*y)**2 + (vx*sgr_xe)**2 + (vy*sgr_ye)**2) / r2d
    vte = np.sqrt((vxe*y)**2 + (vye*x)**2 + (vx*sgr_ye)**2 + (vy*sgr_xe)**2) / r2d

    return vr, vt, vre, vte

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
    print '   A      = %6.2f +/- %5.2f' % (p[0],perr[0])
    print '   mu     = %5.3f +/- %5.3f' % (p[1],perr[1])
    print '   sigma  = %5.3f +/- %5.3f' % (p[2],perr[2])
    print '   chi^2  = %5.2f' % m.fnorm
    print '   dof    = %2d' % m.dof
    print '   Rchi^2 = %5.2f' % Rchi2

    return m


def fitFuncCos(c, fjac=None, x=None, N=None, err=None):
    """
    Find residuals of fit
    c should be list of form [A, phi, vs]
    while data should be list of form [x, N, err]
    """

    num = len(x)
    model = np.zeros(num, dtype=float)
    devs = np.zeros(num, dtype=float)

    # Set parameters
    A = c[0]
    #b = c[1]
    #b=2.0
    phi = c[1]
    vs = c[2]

    #model = A*np.cos([b*xx for xx in x]-phi) + vs
    model = A*np.cos(x-phi) + vs
    residuals = (N - model)/err
    status = 0

    return [status, residuals]


def fitCos(c0=None, data=None, quiet=1):
    """
    Fits cosine function using mpfit.

    Inputs should be given as c0=[A,phi,vs] and
    data=[angleBins, N, err].  Returns object of class mpfit.
    """

    print 'Initial Guess:'
    print '   A = %6.2f' % c0[0]
    #print '   b = %5.3f' % c0[1]
    print ' phi = %5.3f' % c0[1]
    print '  vs = %5.3f' % c0[2]
    
    # Set data to be passed to fit
    functargs = {'x':data[0],'N':data[1],'err':data[2]}

    # Set initial values and limits (no limits on parameters)
    cinfo = [{'value':0,'fixed':0,'limited':[0,0],
              'limits':[0,0]}]*len(c0)

    for ii in range(len(c0)):
        cinfo[ii]['value'] = c0[ii]

    m = nmpfit_sy.mpfit(fitFuncCos, c0, functkw=functargs, parinfo=cinfo,
                    quiet=quiet)
    if (m.status <= 0):
        print 'Error message = ', m.errmsg

    p = m.params                # Best-fit parameters
    perr = m.perror             # Error in parameter fits
                                # from covariance matrix

    m.dof = len(data[0])-len(p) # Number of degrees of freedom
    Rchi2 = m.fnorm/m.dof       # Reduced Chi^2 statistic
    #x = (data[0])[np.nonzero(data[2])]  
    x = data[0]
    m.dof = len(x)-len(p) # Number of degrees of freedom

    p_deg = p[1]*180./math.pi
    perr_deg = perr[1]*180./math.pi
    print 'Final Solution:'
    print '   A    = %6.2f +/- %5.2f' % (p[0],perr[0])
    #print '   b  = %5.3f +/- %5.3f' % (p[1],perr[1])
    print '   phi  = %5.3f +/- %5.3f rad = %4.2f +- %4.2f degrees' % \
          (p[1],perr[1],p_deg,perr_deg)
    print '   vs  = %5.3f +/- %5.3f' % (p[2],perr[2]) # vertical shift
    print '   chi^2  = %5.2f' % m.fnorm
    print '   Rchi^2 = %5.2f' % Rchi2
    print ''

    return m


def plotLabel(root='/u/syelda/research/gc/absolute/10_06_22/',
              labelFile='source_list/label_new.dat'):
    """
    Creates various plots to examine a label.dat-type file.
    Useful when creating a new, updated label.dat file.
    """

    lbl = asciidata.open(root+labelFile)
    name = lbl[0].tonumpy()
    mag = lbl[1].tonumpy()
    x = lbl[2].tonumpy()
    y = lbl[3].tonumpy()
    xe = lbl[4].tonumpy()
    ye = lbl[5].tonumpy()
    vx = lbl[6].tonumpy()
    vy = lbl[7].tonumpy()
    vxe = lbl[8].tonumpy()
    vye = lbl[9].tonumpy()
    t0 = lbl[10].tonumpy()
    use = lbl[11].tonumpy()
    r2d = lbl[12].tonumpy()
    vtot = np.hypot(vx,vy)

    # position and error of sgr a from Yelda et al. 2010 (table 5)
    sgr_x = -0.31 # mas
    sgr_y = 1.44
    sgr_xe = 0.55 # mas
    sgr_ye = 0.59

    idx = np.where(r2d < 5.)[0]
    print 'There are %i stars out to %5.3f arcsec' % (len(name),r2d.max())
    print 'There are %i stars within 5 arcsec' % len(idx)

    # Only get stars with ref=1
    ref = np.where(use == 1)[0]
    print 'There are %i reference stars with use=1' % len(ref)
    print ''
    name = np.array([name[nn] for nn in ref])
    mag = mag[ref]
    x = x[ref]
    y = y[ref]
    xe = xe[ref]
    ye = ye[ref]
    vx = vx[ref]
    vy = vy[ref]
    vxe = vxe[ref]
    vye = vye[ref]
    t0 = t0[ref]
    use = use[ref]
    r2d = r2d[ref]
    vtot = vtot[ref]

    # Get weighted average vels for all known old stars; and all but young stars:
    oldNm = oldNames.loadOldStars()
    yngNm = youngNames.loadYoungStars()
    old = [str(oo) for oo in oldNm]
    oidx = []
    for ii in range(len(old)):
        foo = np.where(name == old[ii])[0]
        if len(foo) > 0:
            oidx = np.concatenate([oidx,foo])
    oidx = [int(oo) for oo in oidx]
    x_o = x[oidx]
    y_o = y[oidx]
    xe_o = xe[oidx]
    ye_o = ye[oidx]
    vx_o = vx[oidx]
    vy_o = vy[oidx]
    vxe_o = vxe[oidx]
    vye_o = vye[oidx]
    r2d_o = r2d[oidx]
    vtot_o = vtot[oidx]
    vx_wt_o = 1./vxe_o**2
    vx_wavg_o = (vx_wt_o * vx_o).sum() / vx_wt_o.sum()
    vx_wavgerr_o = np.sqrt(1. / vx_wt_o.sum())
    vy_wt_o = 1./vye_o**2
    vy_wavg_o = (vy_wt_o * vy_o).sum() / vy_wt_o.sum()
    vy_wavgerr_o = np.sqrt(1. / vy_wt_o.sum())
    # For known old stars, get cluster velocity in polar coords (tangential, radial, and angular)
    # Flip the x axis so numbers increase toward west when converting to angular
    vr_o, vt_o, vre_o, vte_o = rec2polar_sgra(-x_o, y_o, -vx_o, vy_o, vxe_o, vye_o, sgr_xe, sgr_ye)
    vang_o = vt_o / r2d_o

    # The intrinsic motion of the cluster dominates over measurement
    # error, so we'll use this term, vx_iw and vy_iw, as the
    # error (which includes measurement error).
    print 'Known old stars (N=%i):' % len(oidx)
    print '<vx> = %5.3f +- %5.3f mas/yr' %  (vx_o.mean(), vx_o.std(ddof=1))
    print '<vy> = %5.3f +- %5.3f mas/yr' %  (vy_o.mean(), vy_o.std(ddof=1))
    print ''
    print 'weighted <vx> = %5.3f +- %5.3f mas/yr' % (vx_wavg_o,vx_wavgerr_o)
    print 'weighted <vy> = %5.3f +- %5.3f mas/yr' % (vy_wavg_o,vy_wavgerr_o)
    print ''
    print 'Accounting for motion of Sgr A* from Yelda et al. (2010) Table 5:'
    sgr_vx = 0.02 # mas/yr
    sgr_vy = -0.06
    sgr_vxe = 0.09 # mas/yr
    sgr_vye = 0.14
    vx_new_o = vx_wavg_o - sgr_vx
    vy_new_o = vy_wavg_o - sgr_vy
    vx_iw_o = vx_o.std(ddof=1)/np.sqrt(len(oidx))
    vy_iw_o = vy_o.std(ddof=1)/np.sqrt(len(oidx))
    vxe_new_o = np.sqrt(vx_iw_o**2 + sgr_vxe**2)
    vye_new_o = np.sqrt(vy_iw_o**2 + sgr_vye**2)
    print 'weighted <vx> = %5.3f +- %5.3f mas/yr' % (vx_new_o, vxe_new_o)
    print 'weighted <vy> = %5.3f +- %5.3f mas/yr' % (vy_new_o, vye_new_o)
    print ''
    # Weighted average of these values
    vr_wt_o = 1./vre_o**2
    vr_wavg_o = (vr_wt_o * vr_o).sum() / vr_wt_o.sum()
    vr_wavgerr_o = np.sqrt(1. / vr_wt_o.sum())
    vt_wt_o = 1./vte_o**2
    vt_wavg_o = (vt_wt_o * vt_o).sum() / vt_wt_o.sum()
    vt_wavgerr_o = np.sqrt(1. / vt_wt_o.sum())
    vang_wavg_o = (vang_o * vt_o).sum() / vt_wt_o.sum()
    vang_wavgerr_o = np.sqrt(1. / vt_wt_o.sum())

    # Use bootstrap (sample w/ replacement) to determine error on cluster's mean velocity
    def sample_wr(population, k):
        "Chooses k random elements (with replacement) from a population"
        n = len(population)
        _random, _int = random.random, int  # speed hack 
        result = [None] * k
        for i in xrange(k):
            j = _int(_random() * n)
            result[i] = population[j]
        return result

    # Run MC to get errors on polar velocities
    print 'Running bootstrap to calculate uncertainty on cluster mean velocity'
    print 'in polar coords for just the known old stars'
    ntrials = 10 # want this set to 100000, but decrease temporarily to save time
    #ntrials = 100000 # want this set to 100000, but decrease temporarily to save time
    print 'Running %s trials' % str(ntrials)
    vr_o_wavg_t = []
    vt_o_wavg_t = []
    vang_o_wavg_t = []
    for nn in range(ntrials):
        sz = len(vx_o)
        ridx = sample_wr(range(sz), sz)
        ridx = [int(ii) for ii in ridx]

        vr_o_t = vr_o[ridx]
        vt_o_t = vt_o[ridx]
        vang_o_t = vang_o[ridx]
        vre_o_t = vre_o[ridx]
        vte_o_t = vte_o[ridx]

        # Get the weighted average vr, vt, and vang for this trial
        vr_wt_t = 1./vre_o_t**2
        vr_wa_t = (vr_wt_t * vr_o_t).sum() / vr_wt_t.sum()
        vr_wavgerr_t = np.sqrt(1. / vr_wt_t.sum())
        vt_wt_t = 1./vte_o_t**2
        vt_wa_t = (vt_wt_t * vt_o_t).sum() / vt_wt_t.sum()
        vt_waErr_t = np.sqrt(1. / vt_wt_t.sum())
        vang_wt_t = 1./vte_o_t**2
        vang_wa_t = (vang_wt_t * vang_o_t).sum() / vang_wt_t.sum()
        vang_waErr_t = np.sqrt(1. / vang_wt_t.sum())

        vr_o_wavg_t = np.concatenate([vr_o_wavg_t,[vr_wa_t]])
        vt_o_wavg_t = np.concatenate([vt_o_wavg_t,[vt_wa_t]])
        vang_o_wavg_t = np.concatenate([vang_o_wavg_t,[vang_wa_t]])

    # get the RMS of the weighted averages, using the
    # actual weighted average of the cluster as the 'mean'
    vr_o_wavg_boot = np.sqrt(((vr_o_wavg_t - vr_wavg_o)**2).sum() / ntrials)
    vt_o_wavg_boot = np.sqrt(((vt_o_wavg_t - vt_wavg_o)**2).sum() / ntrials)
    vang_o_wavg_boot = np.sqrt(((vang_o_wavg_t - vang_wavg_o)**2).sum() / ntrials)

    print ''
    print '**Polar Coordinates**'
    print '<vr> = %5.3f +- %5.3f mas/yr' %  (vr_o.mean(), vr_o.std(ddof=1))
    print '<vt> = %5.3f +- %5.3f mas/yr' %  (vt_o.mean(), vt_o.std(ddof=1))
    print '<vang> = %5.3f +- %5.3f mas/yr/asec' %  (vang_o.mean(), vang_o.std(ddof=1))
    print ''
    print 'weighted <vr> = %5.3f +- %5.3f mas/yr' % (vr_wavg_o,vr_wavgerr_o)
    print 'weighted <vt> = %5.3f +- %5.3f mas/yr' % (vt_wavg_o,vt_wavgerr_o)
    print 'weighted <vang> = %5.3f +- %5.3f mas/yr/asec' % (vang_wavg_o,vang_wavgerr_o)
    print ''
    print '**bootstrap (only known old stars)**'
    print 'Weighted average velocity (Radial, Tangential, Angular):'
    print 'RMS from bootstrap = (%5.3f, %5.3f, %5.3f) mas/yr' % \
          (vr_o_wavg_boot, vt_o_wavg_boot, vang_o_wavg_boot)
    print '*****************************************'
    print ''

    # Young stars only
    yng = [str(yy) for yy in yngNm]
    yidx = []
    for ii in range(len(yng)):
        foo = np.where(name == yng[ii])[0]
        if len(foo) > 0:
            yidx = np.concatenate([yidx,foo])
    yidx = [int(yy) for yy in yidx]
    x_y = x[yidx]
    y_y = y[yidx]
    xe_y = xe[yidx]
    ye_y = ye[yidx]
    vx_y = vx[yidx]
    vy_y = vy[yidx]
    vxe_y = vxe[yidx]
    vye_y = vye[yidx]
    r2d_y = r2d[yidx]
    vtot_y = vtot[yidx]
    vx_wt_y = 1./vxe_y**2
    vx_wavg_y = (vx_wt_y * vx_y).sum() / vx_wt_y.sum()
    vx_wavgerr_y = np.sqrt(1. / vx_wt_y.sum())
    vy_wt_y = 1./vye_y**2
    vy_wavg_y = (vy_wt_y * vy_y).sum() / vy_wt_y.sum()
    vy_wavgerr_y = np.sqrt(1. / vy_wt_y.sum())
    # For known young stars, get cluster velocity in polar coords (tangential, radial, and angular)
    # Flip the x axis so numbers increase toward west when converting to angular
    vr_y, vt_y, vre_y, vte_y = rec2polar_sgra(-x_y, y_y, -vx_y, vy_y, vxe_y, vye_y, sgr_xe, sgr_ye)
    vang_y = vt_y / r2d_y

    print 'Known young stars (N=%i):' % len(yidx)
    print '<vx> = %5.3f +- %5.3f mas/yr' %  (vx_y.mean(), vx_y.std(ddof=1))
    print '<vy> = %5.3f +- %5.3f mas/yr' %  (vy_y.mean(), vy_y.std(ddof=1))
    print ''
    print 'weighted <vx> = %5.3f +- %5.3f mas/yr' % (vx_wavg_y,vx_wavgerr_y)
    print 'weighted <vy> = %5.3f +- %5.3f mas/yr' % (vy_wavg_y,vy_wavgerr_y)
    print ''
    # Weighted average of these values
    vr_wt_y = 1./vre_y**2
    vr_wavg_y = (vr_wt_y * vr_y).sum() / vr_wt_y.sum()
    vr_wavgerr_y = np.sqrt(1. / vr_wt_y.sum())
    vt_wt_y = 1./vte_y**2
    vt_wavg_y = (vt_wt_y * vt_y).sum() / vt_wt_y.sum()
    vt_wavgerr_y = np.sqrt(1. / vt_wt_y.sum())
    vang_wavg_y = (vang_y * vt_y).sum() / vt_wt_y.sum()
    vang_wavgerr_y = np.sqrt(1. / vt_wt_y.sum())

    # run MC to get errors in angular velocity for young stars
    vr_y_wavg_t = []
    vt_y_wavg_t = []
    vang_y_wavg_t = []
    for nn in range(ntrials):
        sz = len(vx_y)
        ridx = sample_wr(range(sz), sz)
        ridx = [int(ii) for ii in ridx]

        vr_y_t = vr_y[ridx]
        vt_y_t = vt_y[ridx]
        vang_y_t = vang_y[ridx]
        vre_y_t = vre_y[ridx]
        vte_y_t = vte_y[ridx]

        # Get the weighted average vr, vt, and vang for this trial
        vr_wt_t = 1./vre_y_t**2
        vr_wa_t = (vr_wt_t * vr_y_t).sum() / vr_wt_t.sum()
        vr_wavgerr_t = np.sqrt(1. / vr_wt_t.sum())
        vt_wt_t = 1./vte_y_t**2
        vt_wa_t = (vt_wt_t * vt_y_t).sum() / vt_wt_t.sum()
        vt_waErr_t = np.sqrt(1. / vt_wt_t.sum())
        vang_wt_t = 1./vte_y_t**2
        vang_wa_t = (vang_wt_t * vang_y_t).sum() / vang_wt_t.sum()
        vang_waErr_t = np.sqrt(1. / vang_wt_t.sum())

        vr_y_wavg_t = np.concatenate([vr_y_wavg_t,[vr_wa_t]])
        vt_y_wavg_t = np.concatenate([vt_y_wavg_t,[vt_wa_t]])
        vang_y_wavg_t = np.concatenate([vang_y_wavg_t,[vang_wa_t]])

    # get the RMS of the weighted averages, using the
    # actual weighted average of the cluster as the 'mean'
    vr_y_wavg_boot = np.sqrt(((vr_y_wavg_t - vr_wavg_y)**2).sum() / ntrials)
    vt_y_wavg_boot = np.sqrt(((vt_y_wavg_t - vt_wavg_y)**2).sum() / ntrials)
    vang_y_wavg_boot = np.sqrt(((vang_y_wavg_t - vang_wavg_y)**2).sum() / ntrials)

    print ''
    print '**Polar Coordinates**'
    print '<vr> = %5.3f +- %5.3f mas/yr' %  (vr_y.mean(), vr_y.std(ddof=1))
    print '<vt> = %5.3f +- %5.3f mas/yr' %  (vt_y.mean(), vt_y.std(ddof=1))
    print '<vang> = %5.3f +- %5.3f mas/yr/asec' %  (vang_y.mean(), vang_y.std(ddof=1))
    print ''
    print 'weighted <vr> = %5.3f +- %5.3f mas/yr' % (vr_wavg_y,vr_wavgerr_y)
    print 'weighted <vt> = %5.3f +- %5.3f mas/yr' % (vt_wavg_y,vt_wavgerr_y)
    print 'weighted <vang> = %5.3f +- %5.3f mas/yr/asec' % (vang_wavg_y,vang_wavgerr_y)
    print ''
    print '**bootstrap (only known young stars)**'
    print 'Weighted average velocity (Radial, Tangential, Angular):'
    print 'RMS from bootstrap = (%5.3f, %5.3f, %5.3f) mas/yr' % \
          (vr_y_wavg_boot, vt_y_wavg_boot, vang_y_wavg_boot)
    print '*****************************************'


    # stars that aren't known to be young
    nyNm = np.setdiff1d(name, yngNm)
    nyng = [str(nn) for nn in nyNm]
    nidx = []
    for ii in range(len(nyng)):
        if nyng[ii] == 'SgrA':
            continue
        foo = np.where(name == nyng[ii])[0]
        if len(foo) > 0:
            nidx = np.concatenate([nidx,foo])
    nidx = [int(nn) for nn in nidx]
    x_n = x[nidx]
    y_n = y[nidx]
    xe_n = xe[nidx]
    ye_n = ye[nidx]
    vx_n = vx[nidx]
    vy_n = vy[nidx]
    vxe_n = vxe[nidx]
    vye_n = vye[nidx]
    vx_wt_n = 1./vxe_n**2
    vx_wavg_n = (vx_wt_n * vx_n).sum() / vx_wt_n.sum()
    vx_wavgerr_n = np.sqrt(1. / vx_wt_n.sum())
    vy_wt_n = 1./vye_n**2
    vy_wavg_n = (vy_wt_n * vy_n).sum() / vy_wt_n.sum()
    vy_wavgerr_n = np.sqrt(1. / vy_wt_n.sum())
    r2d_n = r2d[nidx]
    print 'All stars but known young stars (N=%i):' % len(nidx)
    print '<vx> = %5.3f +- %5.3f mas/yr' %  (vx_n.mean(), vx_n.std(ddof=1))
    print '<vy> = %5.3f +- %5.3f mas/yr' %  (vy_n.mean(), vy_n.std(ddof=1))
    print ''
    print 'weighted <vx> = %5.3f +- %5.3f mas/yr' % (vx_wavg_n,vx_wavgerr_n)
    print 'weighted <vy> = %5.3f +- %5.3f mas/yr' % (vy_wavg_n,vy_wavgerr_n)
    print ''
    print 'Accounting for motion of Sgr A* from Yelda et al. (2010) Table 5:'
    vx_new_n = vx_wavg_n - sgr_vx
    vy_new_n = vy_wavg_n - sgr_vy
    vx_iw_n = vx_n.std(ddof=1)/np.sqrt(len(vx_n))
    vy_iw_n = vy_n.std(ddof=1)/np.sqrt(len(vx_n))
    vxe_new_n = np.sqrt(vx_iw_n**2 + sgr_vxe**2)
    vye_new_n = np.sqrt(vy_iw_n**2 + sgr_vye**2)
    print 'weighted <vx> = %5.3f +- %5.3f mas/yr' % (vx_new_n,vxe_new_n)
    print 'weighted <vy> = %5.3f +- %5.3f mas/yr' % (vy_new_n,vye_new_n)
    print 'error on mean = (%5.3f, %5.3f) mas/yr' % (vx_iw_n, vy_iw_n)

    # Get cluster velocity in polar coords (tangential, radial, and angular)
    # Flip the x axis so numbers increase toward west when converting to angular
    vr_n, vt_n, vre_n, vte_n = rec2polar_sgra(-x_n, y_n, -vx_n, vy_n, vxe_n, vye_n, sgr_xe, sgr_ye)
    vang_n = vt_n / r2d_n
    # Weighted average of these values
    vr_wt_n = 1./vre_n**2
    vr_wavg_n = (vr_wt_n * vr_n).sum() / vr_wt_n.sum()
    vr_wavgerr_n = np.sqrt(1. / vr_wt_n.sum())
    vt_wt_n = 1./vte_n**2
    vt_wavg_n = (vt_wt_n * vt_n).sum() / vt_wt_n.sum()
    vt_wavgerr_n = np.sqrt(1. / vt_wt_n.sum())
    vang_wavg_n = (vang_n * vt_n).sum() / vt_wt_n.sum()
    vang_wavgerr_n = np.sqrt(1. / vt_wt_n.sum())
    print ''
    print '**Polar Coordinates**'
    print '<vr> = %5.3f +- %5.3f mas/yr' %  (vr_n.mean(), vr_n.std(ddof=1))
    print '<vt> = %5.3f +- %5.3f mas/yr' %  (vt_n.mean(), vt_n.std(ddof=1))
    print '<vang> = %5.3f +- %5.3f mas/yr/asec' %  (vang_n.mean(), vang_n.std(ddof=1))
    print ''
    print 'weighted <vr> = %5.3f +- %5.3f mas/yr' % (vr_wavg_n,vr_wavgerr_n)
    print 'weighted <vt> = %5.3f +- %5.3f mas/yr' % (vt_wavg_n,vt_wavgerr_n)
    print 'weighted <vang> = %5.3f +- %5.3f mas/yr/asec' % (vang_wavg_n,vang_wavgerr_n)

    # Get cluster velocity along and perpendicular to Galactic plane
    vtot_n = np.hypot(vx_n,vy_n)
    # Rotate the velocities so that we're in galactic coordinates
    # note this rotates CCW, so g_theta is multiplied by -1 since x
    # axis increases to left 
    g_theta = -31.4 # Reid & Brunthaler (2004)
    g_thetaErr = 0.1 # Reid & Brunthaler (2004)
    cosTh = math.cos(math.radians(g_theta))
    sinTh = math.sin(math.radians(g_theta))
    vpar = vx_n*cosTh - vy_n*sinTh
    vperp = vx_n*sinTh + vy_n*cosTh
    vparErr = vxe_n*cosTh - vye_n*sinTh
    vperpErr = vxe_n*sinTh + vye_n*cosTh

    # get weighted average of these values
    vpar_wt = 1./vparErr**2
    vpar_wavg = (vpar_wt * vpar).sum() / vpar_wt.sum()
    vpar_wavgerr = np.sqrt(1. / vpar_wt.sum())
    vperp_wt = 1./vperpErr**2
    vperp_wavg = (vperp_wt * vperp).sum() / vperp_wt.sum()
    vperp_wavgerr = np.sqrt(1. / vperp_wt.sum())
    print ''
    print 'Parallel/Perpendicular to Galactic Plane'
    print 'All stars but known young stars (N=%i):' % len(nidx)
    print '<v_parallel> = %6.4f +- %6.4f mas/yr' %  (vpar.mean(), vpar.std(ddof=1))
    print '<v_perp> = %6.4f +- %6.4f mas/yr' %  (vperp.mean(), vperp.std(ddof=1))
    print ''
    print 'weighted <v_parallel> = %6.4f +- %6.4f mas/yr' % (vpar_wavg,vpar_wavgerr)
    print 'weighted <v_perp> = %6.4f +- %6.4f mas/yr' % (vperp_wavg,vperp_wavgerr)
    print ''

    # Split the cluster into an inner and outer, and get velocity info
    inner = np.where(r2d_n < 5.)[0]
    outer = np.where(r2d_n >= 5.)[0]
    vx_n_in = vx_n[inner]
    vy_n_in = vy_n[inner]
    vx_n_out = vx_n[outer]
    vy_n_out = vy_n[outer]
    vxe_n_in = vxe_n[inner]
    vye_n_in = vye_n[inner]
    vxe_n_out = vxe_n[outer]
    vye_n_out = vye_n[outer]
    vt_n_in = vt_n[inner]
    vang_n_in = vang_n[inner]
    vt_n_out = vt_n[outer]
    vang_n_out = vang_n[outer]
    vte_n_in = vte_n[inner]
    vange_n_in = vte_n[inner]
    vte_n_out = vte_n[outer]
    vange_n_out = vte_n[outer]

    print 'Inner vs. outer cluster'
    print 'Average tangential velocity:'
    print 'for R < 5 arcsec (N=%i): %5.3f +- %5.3f mas/yr' % (len(inner),vt_n_in.mean(), vt_n_in.std(ddof=1))
    print 'for R > 5 arcsec (N=%i): %5.3f +- %5.3f mas/yr' % (len(outer),vt_n_out.mean(), vt_n_out.std(ddof=1))
    print 'Average angular velocity:'
    print 'for R < 5 arcsec: %5.3f +- %5.3f mas/yr/asec' % (vang_n_in.mean(), vang_n_in.std(ddof=1))
    print 'for R > 5 arcsec: %5.3f +- %5.3f mas/yr/asec' % (vang_n_out.mean(), vang_n_out.std(ddof=1))
    print ''
    print ''
    print 'Average x velocity:'
    print 'for R < 5 arcsec (N=%i): %5.3f +- %5.3f mas/yr' % (len(inner),vx_n_in.mean(), vx_n_in.std(ddof=1))
    print 'for R > 5 arcsec (N=%i): %5.3f +- %5.3f mas/yr' % (len(outer),vx_n_out.mean(), vx_n_out.std(ddof=1))
    print 'Average y velocity:'
    print 'for R < 5 arcsec: %5.3f +- %5.3f mas/yr/asec' % (vy_n_in.mean(), vy_n_in.std(ddof=1))
    print 'for R > 5 arcsec: %5.3f +- %5.3f mas/yr/asec' % (vy_n_out.mean(), vy_n_out.std(ddof=1))
    print ''

    # Weighted average for inner and outer
    vx_wt_n_in = 1./vxe_n_in**2
    vx_wavg_n_in = (vx_wt_n_in * vx_n_in).sum() / vx_wt_n_in.sum()
    vx_wavgerr_n_in = np.sqrt(1. / vx_wt_n_in.sum())
    vy_wt_n_in = 1./vye_n_in**2
    vy_wavg_n_in = (vy_wt_n_in * vy_n_in).sum() / vy_wt_n_in.sum()
    vy_wavgerr_n_in = np.sqrt(1. / vy_wt_n_in.sum())
    vt_wt_n_in = 1./vte_n_in**2
    vt_wavg_n_in = (vt_wt_n_in * vt_n_in).sum() / vt_wt_n_in.sum()
    vt_wavgerr_n_in = np.sqrt(1. / vt_wt_n_in.sum())
    vang_wavg_n_in = (vang_n_in * vt_n_in).sum() / vt_wt_n_in.sum()
    vang_wavgerr_n_in = np.sqrt(1. / vt_wt_n_in.sum())
    # Weighted average for inner and outer
    vx_wt_n_out = 1./vxe_n_out**2
    vx_wavg_n_out = (vx_wt_n_out * vx_n_out).sum() / vx_wt_n_out.sum()
    vx_wavgerr_n_out = np.sqrt(1. / vx_wt_n_out.sum())
    vy_wt_n_out = 1./vye_n_out**2
    vy_wavg_n_out = (vy_wt_n_out * vy_n_out).sum() / vy_wt_n_out.sum()
    vy_wavgerr_n_out = np.sqrt(1. / vy_wt_n_out.sum())
    vt_wt_n_out = 1./vte_n_out**2
    vt_wavg_n_out = (vt_wt_n_out * vt_n_out).sum() / vt_wt_n_out.sum()
    vt_wavgerr_n_out = np.sqrt(1. / vt_wt_n_out.sum())
    vang_wavg_n_out = (vang_n_out * vt_n_out).sum() / vt_wt_n_out.sum()
    vang_wavgerr_n_out = np.sqrt(1. / vt_wt_n_out.sum())

    # Call sample_wr to get the indices for each trial
    print 'Running bootstrap to calculate uncertainty on cluster mean velocity'
    print 'in x/y coords, polar coords, and par/perp coords'
    print '(No young stars included)'
    vx_wavg_t = []
    vy_wavg_t = []
    vr_wavg_t = []
    vt_wavg_t = []
    vang_wavg_t = []
    vpar_wavg_t = []
    vperp_wavg_t = []
    for nn in range(ntrials):
        sz = len(vx_n)
        ridx = sample_wr(range(sz), sz)
        ridx = [int(ii) for ii in ridx]

        vx_n_t = vx_n[ridx]
        vy_n_t = vy_n[ridx]
        vxe_n_t = vxe_n[ridx]
        vye_n_t = vye_n[ridx]

        vr_n_t = vr_n[ridx]
        vt_n_t = vt_n[ridx]
        vang_n_t = vang_n[ridx]
        vre_n_t = vre_n[ridx]
        vte_n_t = vte_n[ridx]

        vpar_t = vpar[ridx]
        vperp_t = vperp[ridx]
        vparE_t = vparErr[ridx]
        vperpE_t = vperpErr[ridx]

        # Get the weighted average vx and vy for this trial
        vx_wt_t = 1./vxe_n_t**2
        vx_wa_t = (vx_wt_t * vx_n_t).sum() / vx_wt_t.sum()
        vx_wavgerr_t = np.sqrt(1. / vx_wt_t.sum())
        vy_wt_t = 1./vye_n_t**2
        vy_wa_t = (vy_wt_t * vy_n_t).sum() / vy_wt_t.sum()
        vy_waErr_t = np.sqrt(1. / vy_wt_t.sum())

        vx_wavg_t = np.concatenate([vx_wavg_t,[vx_wa_t]])
        vy_wavg_t = np.concatenate([vy_wavg_t,[vy_wa_t]])

        # Get the weighted average vr, vt, and vang for this trial
        vr_wt_t = 1./vre_n_t**2
        vr_wa_t = (vr_wt_t * vr_n_t).sum() / vr_wt_t.sum()
        vr_wavgerr_t = np.sqrt(1. / vr_wt_t.sum())
        vt_wt_t = 1./vte_n_t**2
        vt_wa_t = (vt_wt_t * vt_n_t).sum() / vt_wt_t.sum()
        vt_waErr_t = np.sqrt(1. / vt_wt_t.sum())
        vang_wt_t = 1./vte_n_t**2
        vang_wa_t = (vang_wt_t * vang_n_t).sum() / vang_wt_t.sum()
        vang_waErr_t = np.sqrt(1. / vang_wt_t.sum())

        vr_wavg_t = np.concatenate([vr_wavg_t,[vr_wa_t]])
        vt_wavg_t = np.concatenate([vt_wavg_t,[vt_wa_t]])
        vang_wavg_t = np.concatenate([vang_wavg_t,[vang_wa_t]])

        # Get the weighted average vpar and vperp for this trial
        vpar_wt_t = 1./vparE_t**2
        vpar_wa_t = (vpar_wt_t * vpar_t).sum() / vpar_wt_t.sum()
        vpar_wavgerr_t = np.sqrt(1. / vpar_wt_t.sum())
        vperp_wt_t = 1./vperpE_t**2
        vperp_wa_t = (vperp_wt_t * vperp_t).sum() / vperp_wt_t.sum()
        vperp_waErr_t = np.sqrt(1. / vperp_wt_t.sum())

        vpar_wavg_t = np.concatenate([vpar_wavg_t,[vpar_wa_t]])
        vperp_wavg_t = np.concatenate([vperp_wavg_t,[vperp_wa_t]])


    # get the RMS of the weighted averages, using the
    # actual weighted average of the cluster as the 'mean'
    # First do this for the x/y coords
    vx_wavg_boot = np.sqrt(((vx_wavg_t - vx_wavg_n)**2).sum() / ntrials)
    vy_wavg_boot = np.sqrt(((vy_wavg_t - vy_wavg_n)**2).sum() / ntrials)
    print ''
    print '**bootstrap**'
    print 'Weighted average velocity (X, Y):'
    print 'RMS from bootstrap = (%5.3f, %5.3f) mas/yr' % \
          (vx_wavg_boot, vy_wavg_boot)

    vr_wavg_boot = np.sqrt(((vr_wavg_t - vr_wavg_n)**2).sum() / ntrials)
    vt_wavg_boot = np.sqrt(((vt_wavg_t - vt_wavg_n)**2).sum() / ntrials)
    vang_wavg_boot = np.sqrt(((vang_wavg_t - vang_wavg_n)**2).sum() / ntrials)
    print ''
    print '**bootstrap**'
    print 'Weighted average velocity (Radial, Tangential, Angular):'
    print 'RMS from bootstrap = (%5.3f, %5.3f, %5.3f) mas/yr' % \
          (vr_wavg_boot, vt_wavg_boot, vang_wavg_boot)

    vpar_wavg_boot = np.sqrt(((vpar_wavg_t - vpar_wavg)**2).sum() / ntrials)
    vperp_wavg_boot = np.sqrt(((vperp_wavg_t - vperp_wavg)**2).sum() / ntrials)
    print ''
    print '**bootstrap**'
    print 'Weighted average velocity (PARALLEL, PERPENDICULAR):'
    print 'RMS from bootstrap = (%5.3f, %5.3f) mas/yr' % \
          (vpar_wavg_boot, vperp_wavg_boot)


    # Call sample_wr to get the indices for each trial
    print ''
    # First run the inner part of cluster thru the MC
    vx_in_wavg_t = []
    vy_in_wavg_t = []
    vt_in_wavg_t = []
    vang_in_wavg_t = []
    for nn in range(ntrials):
        sz = len(vx_n_in)
        ridx = sample_wr(range(sz), sz)
        ridx = [int(ii) for ii in ridx]

        vx_in_t = vx_n_in[ridx]
        vy_in_t = vy_n_in[ridx]
        vxe_in_t = vxe_n_in[ridx]
        vye_in_t = vye_n_in[ridx]

        vt_in_t = vt_n_in[ridx]
        vang_in_t = vang_n_in[ridx]
        vte_in_t = vte_n_in[ridx]

        # Get the weighted average vx and vy for this trial
        vx_wt_t = 1./vxe_in_t**2
        vx_wa_t = (vx_wt_t * vx_in_t).sum() / vx_wt_t.sum()
        vx_wavgerr_t = np.sqrt(1. / vx_wt_t.sum())
        vy_wt_t = 1./vye_in_t**2
        vy_wa_t = (vy_wt_t * vy_in_t).sum() / vy_wt_t.sum()
        vy_waErr_t = np.sqrt(1. / vy_wt_t.sum())

        vx_in_wavg_t = np.concatenate([vx_wavg_t,[vx_wa_t]])
        vy_in_wavg_t = np.concatenate([vy_wavg_t,[vy_wa_t]])

        # Get the weighted average vr, vt, and vang for this trial
        vt_wt_t = 1./vte_in_t**2
        vt_wa_t = (vt_wt_t * vt_in_t).sum() / vt_wt_t.sum()
        vt_waErr_t = np.sqrt(1. / vt_wt_t.sum())
        vang_wt_t = 1./vte_in_t**2
        vang_wa_t = (vang_wt_t * vang_in_t).sum() / vang_wt_t.sum()
        vang_waErr_t = np.sqrt(1. / vang_wt_t.sum())

        vt_in_wavg_t = np.concatenate([vt_wavg_t,[vt_wa_t]])
        vang_in_wavg_t = np.concatenate([vang_wavg_t,[vang_wa_t]])

    # Now do the outer part of the cluster
    vx_out_wavg_t = []
    vy_out_wavg_t = []
    vt_out_wavg_t = []
    vang_out_wavg_t = []
    for nn in range(ntrials):
        sz = len(vx_n_out)
        ridx = sample_wr(range(sz), sz)
        ridx = [int(ii) for ii in ridx]

        vx_out_t = vx_n_out[ridx]
        vy_out_t = vy_n_out[ridx]
        vxe_out_t = vxe_n_out[ridx]
        vye_out_t = vye_n_out[ridx]

        vt_out_t = vt_n_out[ridx]
        vang_out_t = vang_n_out[ridx]
        vte_out_t = vte_n_out[ridx]

        # Get the weighted average vx and vy for this trial
        vx_wt_t = 1./vxe_out_t**2
        vx_wa_t = (vx_wt_t * vx_out_t).sum() / vx_wt_t.sum()
        vx_wavgerr_t = np.sqrt(1. / vx_wt_t.sum())
        vy_wt_t = 1./vye_out_t**2
        vy_wa_t = (vy_wt_t * vy_out_t).sum() / vy_wt_t.sum()
        vy_waErr_t = np.sqrt(1. / vy_wt_t.sum())

        vx_out_wavg_t = np.concatenate([vx_wavg_t,[vx_wa_t]])
        vy_out_wavg_t = np.concatenate([vy_wavg_t,[vy_wa_t]])

        # Get the weighted average vr, vt, and vang for this trial
        vt_wt_t = 1./vte_out_t**2
        vt_wa_t = (vt_wt_t * vt_out_t).sum() / vt_wt_t.sum()
        vt_waErr_t = np.sqrt(1. / vt_wt_t.sum())
        vang_wt_t = 1./vte_out_t**2
        vang_wa_t = (vang_wt_t * vang_out_t).sum() / vang_wt_t.sum()
        vang_waErr_t = np.sqrt(1. / vang_wt_t.sum())

        vt_out_wavg_t = np.concatenate([vt_wavg_t,[vt_wa_t]])
        vang_out_wavg_t = np.concatenate([vang_wavg_t,[vang_wa_t]])

    # get the RMS of the weighted averages, using the
    # actual weighted average of the cluster as the 'mean'
    # First do this for the x/y coords
    vx_in_wavg_boot = np.sqrt(((vx_in_wavg_t - vx_wavg_n_in)**2).sum() / ntrials)
    vy_in_wavg_boot = np.sqrt(((vy_in_wavg_t - vy_wavg_n_in)**2).sum() / ntrials)
    vx_out_wavg_boot = np.sqrt(((vx_out_wavg_t - vx_wavg_n_out)**2).sum() / ntrials)
    vy_out_wavg_boot = np.sqrt(((vy_out_wavg_t - vy_wavg_n_out)**2).sum() / ntrials)
    print ''
    print '**bootstrap**'
    print 'Weighted average velocity (X, Y):'
    print 'INNER 5 arcsec: RMS from bootstrap = (%5.3f, %5.3f) mas/yr' % \
          (vx_in_wavg_boot, vy_in_wavg_boot)
    print 'OUTER 5 arcsec: RMS from bootstrap = (%5.3f, %5.3f) mas/yr' % \
          (vx_out_wavg_boot, vy_out_wavg_boot)

    vt_in_wavg_boot = np.sqrt(((vt_in_wavg_t - vt_wavg_n_in)**2).sum() / ntrials)
    vang_in_wavg_boot = np.sqrt(((vang_in_wavg_t - vang_wavg_n_in)**2).sum() / ntrials)
    vt_out_wavg_boot = np.sqrt(((vt_out_wavg_t - vt_wavg_n_out)**2).sum() / ntrials)
    vang_out_wavg_boot = np.sqrt(((vang_out_wavg_t - vang_wavg_n_out)**2).sum() / ntrials)
    print ''
    print '**bootstrap**'
    print 'Weighted average velocity (Tangential, Angular):'
    print 'INNER 5 arcsec: RMS from bootstrap = (%5.3f, %5.3f) mas/yr' % \
          (vt_in_wavg_boot, vang_in_wavg_boot)
    print 'OUTER 5 arcsec: RMS from bootstrap = (%5.3f, %5.3f) mas/yr' % \
          (vt_out_wavg_boot, vang_out_wavg_boot)


    # Plot distribution of stars on an image
    imgFile = '/u/ghezgroup/data/gc/06maylgs1/combo/mag06maylgs1_msr_kp.fits'
    scale = 0.00993
    sgra = [1596.,1006.]
    img = pyfits.getdata(imgFile)
    imgsize = (img.shape)[0]
    x_new = sgra[0]-x/scale
    y_new = sgra[1]+y/scale

    py.figure(1)
    py.clf()
    py.figure(figsize=(8,8))
    py.imshow(np.log10(img+1), aspect='equal', interpolation='bicubic',
              vmin=2.2,vmax=5,origin='lowerleft', cmap=py.cm.gray_r)
    py.plot(x_new[nidx], y_new[nidx], 'r.')
    py.axis([0,imgsize,0,imgsize])
    py.xlabel('X (pix)')
    py.ylabel('Y (pix)')
    py.title('Positions of Secondary Stars')
    py.savefig(root+'plots/secondary_positions.png')
    py.close(1)

    # Plot the velocities on an image
    py.figure(2)
    py.figure(figsize=(8,8))
    py.clf()
    py.imshow(np.log10(img+1), aspect='equal', interpolation='bicubic',
              vmin=2.2,vmax=5,origin='lowerleft', cmap=py.cm.gray_r)

    qvr = py.quiver([x_new[nidx]],[y_new[nidx]],[-vx[nidx]],[vy[nidx]],color='black',units='y',\
                    width=2,headwidth=6,headlength=8,headaxislength=8, scale=0.15)

    py.quiverkey(qvr,150,50,5,'5 mas/yr',coordinates='data',
              fontproperties={'size': 'smaller'})
    py.xlabel('X (pix)')
    py.ylabel('Y (pix)')
    py.title('Absolute Velocities of Secondary Stars')
    py.axis([0,imgsize,0,imgsize])
    py.savefig(root+'plots/secondary_absVels.png')
    py.close(2)

    # Plot histogram of the velocities in x/y coords
    import matplotlib.font_manager
    prop = matplotlib.font_manager.FontProperties(size=12)
    binsIn = py.arange(-20, 20, 0.5)
    py.figure(3)
    py.figure(figsize=(10,5))
    py.clf()
    py.subplots_adjust(left=0.1,right=0.98,top=0.95,
                    wspace=0.3,hspace=0.5)
    py.subplot(1,2,1)
    (nhx,bhx,phx) = py.hist(vx[nidx],binsIn,color='r',histtype='step',label='x')
    (nhy,bhy,phy) = py.hist(vy[nidx],binsIn,color='b',histtype='step',label='y')
    py.xlabel('Cluster Velocity (mas/yr)',fontsize=16)
    py.ylabel('N',fontsize=16)
    py.legend(prop=prop)
    py.axis([-12,12,0,max(nhx.max(),nhy.max())+10])
    py.subplot(1,2,2)
    py.plot(sgr_vx,sgr_vy,'k+',mew=2)
    py.plot(vx_wavg_n,vy_wavg_n,'kx',mew=2)
    # Plot the 1,2,3 sigma contours of the velocities
    an = np.linspace(0, 2*np.pi, 100)
    py.plot(vx_wavg_boot*np.cos(an)+vx_wavg_n, vy_wavg_boot*np.sin(an)+vy_wavg_n,
            'k--', mew=5,label='Cluster')
    py.plot(2.*vx_wavg_boot*np.cos(an)+vx_wavg_n, 2.*vy_wavg_boot*np.sin(an)+vy_wavg_n,
            'k--', mew=5)
    py.plot(3.*vx_wavg_boot*np.cos(an)+vx_wavg_n, 3.*vy_wavg_boot*np.sin(an)+vy_wavg_n,
            'k--', mew=5)
    py.plot(sgr_vxe*np.cos(an)+sgr_vx, sgr_vye*np.sin(an)+sgr_vy,
            'k-', mew=5,label='Sgr A*')
    py.plot(2.*sgr_vxe*np.cos(an)+sgr_vx, 2.*sgr_vye*np.sin(an)+sgr_vy,
            'k-', mew=5)
    py.plot(3.*sgr_vxe*np.cos(an)+sgr_vx, 3.*sgr_vye*np.sin(an)+sgr_vy,
            'k-', mew=5)
    py.xlabel(r'V$_x$ (mas/yr)',fontsize=16)
    py.ylabel(r'V$_y$ (mas/yr)',fontsize=16)
    py.legend(prop=prop,loc=4,numpoints=1)
    py.axis([-0.7,0.7,-0.7,0.7])
    py.savefig(root+'plots/secondary_hist_sgrCluster_xy.png')
    py.savefig(root+'plots/secondary_hist_sgrCluster_xy.eps')
    py.close(3)

    # Plot histogram of the velocities in parallel/perpendicular coords
    py.figure(5)
    py.figure(figsize=(6,6))
    py.clf()
    py.subplots_adjust(left=0.15,right=0.9,top=0.9,bottom=0.1)
    py.plot(sgr_vx,sgr_vy,'k+',mew=2)
    py.plot(vx_wavg_n,vy_wavg_n,'kx',mew=2)
    # Plot the 1,2,3 sigma contours of the velocities
    an = np.linspace(0, 2*np.pi, 100)
    py.plot(vx_wavg_boot*np.cos(an)+vx_wavg_n, vy_wavg_boot*np.sin(an)+vy_wavg_n,
            'k--', mew=5,label='Cluster')
    py.plot(2.*vx_wavg_boot*np.cos(an)+vx_wavg_n, 2.*vy_wavg_boot*np.sin(an)+vy_wavg_n,
            'k--', mew=5)
    py.plot(3.*vx_wavg_boot*np.cos(an)+vx_wavg_n, 3.*vy_wavg_boot*np.sin(an)+vy_wavg_n,
            'k--', mew=5)
    py.plot(sgr_vxe*np.cos(an)+sgr_vx, sgr_vye*np.sin(an)+sgr_vy,
            'k-', mew=5,label='Sgr A*')
    py.plot(2.*sgr_vxe*np.cos(an)+sgr_vx, 2.*sgr_vye*np.sin(an)+sgr_vy,
            'k-', mew=5)
    py.plot(3.*sgr_vxe*np.cos(an)+sgr_vx, 3.*sgr_vye*np.sin(an)+sgr_vy,
            'k-', mew=5)
    py.xlabel(r'V$_x$ (mas yr$^{-1}$)',fontsize=18)
    py.ylabel(r'V$_y$ (mas yr$^{-1}$)',fontsize=18)
    py.legend(prop=prop,loc=4,numpoints=1)
    py.axis([-0.7,0.7,-0.7,0.7])
    py.savefig(root+'plots/sgrA_vs_cluster_vel.png')
    py.savefig(root+'plots/sgrA_vs_cluster_vel.eps')
    py.close(5)


    # Plot histogram of the velocity errors
    binsIn = py.arange(0, 3, 0.1)
    py.figure(6)
    py.figure(figsize=(6,6))
    py.clf()
    (nhx,bhx,phx) = py.hist(vxe[nidx],binsIn,color='r',histtype='step',label='x')
    (nhy,bhy,phy) = py.hist(vye[nidx],binsIn,color='b',histtype='step',label='y')
    py.plot([0,0],[0,max(nhx)],'k--')
    py.title('Absolute Velocity Errors of Secondary Stars')
    py.xlabel('Absolute Velocity Error (mas/yr)')
    py.ylabel('N')
    py.legend()
    py.savefig(root+'plots/secondary_hist_absVelErr.png')
    py.close(6)

    ##########
    #
    # Make 2D histograms of absolute velocities
    #
    ##########
    bincnt = 20
    # First get the non-known-young stars
    (hist, vxbins, vybins) = h2d.histogram2d(vx[nidx], vy[nidx], bins=(bincnt, bincnt))
    # Need to convert the 2d histogram into floats
    probDist = np.array(hist, dtype=float)
    # We can turn the histogram into a probability distribution
    # just by dividing by the total number of measurements
    probDist /= float(len(vx))
    py.rc('text', usetex=True)

    # Clear the plot
    py.figure(7)
    py.figure(figsize=(7,7))
    py.clf()
    # Display the 2D histogram
    py.imshow(probDist, cmap=py.cm.hot_r, origin='lower', aspect='auto',
           extent=[vxbins[0], vxbins[-1], vybins[0], vybins[-1]])
    # Plot velocity of Sgr A* from Table 5 (Yelda et al. 2010)
    py.plot(sgr_vx,sgr_vy,'k+',ms=7,mew=2)
    # Plot the 1 sigma contours of the velocity intrinsic width of the cluster (vx_iw, vy_iw)
    an = np.linspace(0, 2*np.pi, 100)
    py.plot(vx_iw_n*np.cos(an), vy_iw_n*np.sin(an), 'k--', mew=5)
    py.axis([-10,10,-10,10])
    # Draw labels using latex by putting "r" before the string
    py.xlabel(r'V$_x$ (mas/yr)', fontsize=17)
    py.ylabel(r'V$_y$ (mas/yr)', fontsize=17)
    # Set the label axis formatting.
    thePlot = py.gca()
    py.setp( thePlot.get_xticklabels(), fontsize=16, fontweight='bold')
    py.setp( thePlot.get_yticklabels(), fontsize=16, fontweight='bold')
    py.savefig(root+'plots/abs_vels_pdf_noYng.eps')
    py.savefig(root+'plots/abs_vels_pdf_noYng.png')
    py.close(7)

    # Now plot the known late-type stars
    (hist, vxbins, vybins) = h2d.histogram2d(vx_o, vy_o, bins=(bincnt, bincnt))
    # Need to convert the 2d histogram into floats
    probDist = np.array(hist, dtype=float)
    # We can turn the histogram into a probability distribution
    # just by dividing by the total number of measurements
    probDist /= float(len(vx))
    py.rc('text', usetex=True)

    # Clear the plot
    py.figure(8)
    py.figure(figsize=(7,7))
    py.clf()
    # Display the 2D histogram
    py.imshow(probDist, cmap=py.cm.hot_r, origin='lower', aspect='auto',
           extent=[vxbins[0], vxbins[-1], vybins[0], vybins[-1]])
    # Plot velocity of Sgr A* from Table 5 (Yelda et al. 2010)
    py.plot(sgr_vx,sgr_vy,'k+',ms=7,mew=2)
    # Plot the 1 sigma contours of the velocity intrinsic width of the cluster (vx_iw, vy_iw)
    an = np.linspace(0, 2*np.pi, 100)
    py.plot(vx_iw_n*np.cos(an), vy_iw_n*np.sin(an), 'k--', mew=5)

    py.axis([-10,10,-10,10])
    # Draw labels using latex by putting "r" before the string
    py.xlabel(r'V$_x$ (mas/yr)', fontsize=17)
    py.ylabel(r'V$_y$ (mas/yr)', fontsize=17)
    # Set the label axis formatting.
    thePlot = py.gca()
    py.setp( thePlot.get_xticklabels(), fontsize=16, fontweight='bold')
    py.setp( thePlot.get_yticklabels(), fontsize=16, fontweight='bold')
    py.savefig(root+'plots/abs_vels_pdf_knownOld.eps')
    py.savefig(root+'plots/abs_vels_pdf_knownOld.png')
    py.close(8)

    # Plot the cumulative weighted average error vs. R
    # Sort by radius (only sort the non-known-young stars)
    rsrt = r2d_n.argsort()
    r2d_ns = r2d_n[rsrt]
    x_nr = x_n[rsrt]
    y_nr = y_n[rsrt]
    vx_nr = vx_n[rsrt]
    vxe_nr = vxe_n[rsrt]
    vy_nr = vy_n[rsrt]
    vye_nr = vye_n[rsrt]
    vx_wavgerr_c = [0] # set first index to zero, since it's error w/ one (1st) source in CDF
    vy_wavgerr_c = [0]
    vr_wavgerr_c = [0]
    vt_wavgerr_c = [0]
    vang_wavgerr_c = [0]
    vx_eom_c = [0]
    vy_eom_c = [0]
    vr_eom_c = [0]
    vt_eom_c = [0]
    vang_eom_c = [0]
    # Loop thru the radii and calculate the cumulative weighted average error
    for rr in range(1,len(r2d_ns)):
        # first get error on weighted mean
        vx_wt_tmp = 1./vxe_nr[0:rr]**2
        vx_wavgerr_tmp = np.sqrt(1. / vx_wt_tmp.sum())
        vx_wavgerr_c = np.concatenate([vx_wavgerr_c, [vx_wavgerr_tmp]])
        vy_wt_tmp = 1./vye_nr[0:rr]**2
        vy_wavgerr_tmp = np.sqrt(1. / vy_wt_tmp.sum())
        vy_wavgerr_c = np.concatenate([vy_wavgerr_c, [vy_wavgerr_tmp]])

        # also get error on mean
        vx_eom_tmp = vx_nr[0:rr].std(ddof=1) / np.sqrt(rr)
        vy_eom_tmp = vy_nr[0:rr].std(ddof=1) / np.sqrt(rr)
        vx_eom_c = np.concatenate([vx_eom_c, [vx_eom_tmp]])
        vy_eom_c = np.concatenate([vy_eom_c, [vy_eom_tmp]])

        # Get vels in polar coords 
        # Get cluster velocity in polar coords (tangential, radial, and angular)
        # Flip the x axis so numbers increase toward west when converting to angular
        vr_n_c, vt_n_c, vre_n_c, vte_n_c = \
                rec2polar_sgra(-x_nr[0:rr], y_nr[0:rr], -vx_nr[0:rr], vy_nr[0:rr], vxe_nr[0:rr], vye_nr[0:rr],sgr_xe,sgr_ye)
        vang_n_c = vt_n_c / r2d_ns[0:rr]
        # first get error on weighted mean
        vr_wt_tmp = 1./vre_n_c**2
        vr_wavgerr_tmp = np.sqrt(1. / vr_wt_tmp.sum())
        vr_wavgerr_c = np.concatenate([vr_wavgerr_c, [vr_wavgerr_tmp]])
        vt_wt_tmp = 1./vte_n_c**2
        vt_wavgerr_tmp = np.sqrt(1. / vt_wt_tmp.sum())
        vt_wavgerr_c = np.concatenate([vt_wavgerr_c, [vt_wavgerr_tmp]])
        vang_wt_tmp = 1./vte_n_c**2
        vang_wavgerr_tmp = np.sqrt(1. / vang_wt_tmp.sum())
        vang_wavgerr_c = np.concatenate([vang_wavgerr_c, [vang_wavgerr_tmp]])

        # also get error on mean
        vr_eom_tmp = vr_n_c.std(ddof=1) / np.sqrt(rr)
        vt_eom_tmp = vt_n_c.std(ddof=1) / np.sqrt(rr)
        vang_eom_tmp = vang_n_c.std(ddof=1) / np.sqrt(rr)
        vr_eom_c = np.concatenate([vr_eom_c, [vr_eom_tmp]])
        vt_eom_c = np.concatenate([vt_eom_c, [vt_eom_tmp]])
        vang_eom_c = np.concatenate([vang_eom_c, [vang_eom_tmp]])

    # How do things change if we include the young stars?
    # Plot the cumulative weighted average error vs. R
    # Sort by radius 
    rsrt = r2d.argsort()
    r2d_as = r2d[rsrt]
    x_as = x[rsrt]
    y_as = y[rsrt]
    vx_as = vx[rsrt]
    vxe_as = vxe[rsrt]
    vy_as = vy[rsrt]
    vye_as = vye[rsrt]
    vx_wavgerr_ca = [0] # set first index to zero, since it's error w/ one (1st) source in CDF
    vy_wavgerr_ca = [0]
    vr_wavgerr_ca = [0]
    vt_wavgerr_ca = [0]
    vang_wavgerr_ca = [0]
    vx_eom_ca = [0]
    vy_eom_ca = [0]
    vr_eom_ca = [0]
    vt_eom_ca = [0]
    vang_eom_ca = [0]
    # Loop thru the radii and calculate the cumulative weighted average error
    for rr in range(1,len(r2d_as)):
        # first get error on weighted mean
        vx_wt_tmp = 1./vxe_as[0:rr]**2
        vx_wavgerr_tmp = np.sqrt(1. / vx_wt_tmp.sum())
        vx_wavgerr_ca = np.concatenate([vx_wavgerr_ca, [vx_wavgerr_tmp]])
        vy_wt_tmp = 1./vye_as[0:rr]**2
        vy_wavgerr_tmp = np.sqrt(1. / vy_wt_tmp.sum())
        vy_wavgerr_ca = np.concatenate([vy_wavgerr_ca, [vy_wavgerr_tmp]])

        # also get error on mean
        vx_eom_tmp = vx_as[0:rr].std(ddof=1) / np.sqrt(rr)
        vy_eom_tmp = vy_as[0:rr].std(ddof=1) / np.sqrt(rr)
        vx_eom_ca = np.concatenate([vx_eom_ca, [vx_eom_tmp]])
        vy_eom_ca = np.concatenate([vy_eom_ca, [vy_eom_tmp]])

        # Get vels in polar coords 
        # Get cluster velocity in polar coords (tangential, radial, and angular)
        # Flip the x axis so numbers increase toward west when converting to angular
        vr_c, vt_c, vre_c, vte_c = \
                rec2polar_sgra(-x_as[0:rr], y_as[0:rr], -vx_as[0:rr], vy_as[0:rr], vxe_as[0:rr], vye_as[0:rr],sgr_xe,sgr_ye)
        vang_c = vt_c / r2d_as[0:rr]
        # first get error on weighted mean
        vr_wt_tmp = 1./vre_c**2
        vr_wavgerr_tmp = np.sqrt(1. / vr_wt_tmp.sum())
        vr_wavgerr_ca = np.concatenate([vr_wavgerr_ca, [vr_wavgerr_tmp]])
        vt_wt_tmp = 1./vte_c**2
        vt_wavgerr_tmp = np.sqrt(1. / vt_wt_tmp.sum())
        vt_wavgerr_ca = np.concatenate([vt_wavgerr_ca, [vt_wavgerr_tmp]])
        vang_wt_tmp = 1./vte_c**2
        vang_wavgerr_tmp = np.sqrt(1. / vang_wt_tmp.sum())
        vang_wavgerr_ca = np.concatenate([vang_wavgerr_ca, [vang_wavgerr_tmp]])

        # also get error on mean
        vr_eom_tmp = vr_c.std(ddof=1) / np.sqrt(rr)
        vt_eom_tmp = vt_c.std(ddof=1) / np.sqrt(rr)
        vang_eom_tmp = vang_c.std(ddof=1) / np.sqrt(rr)
        vr_eom_ca = np.concatenate([vr_eom_ca, [vr_eom_tmp]])
        vt_eom_ca = np.concatenate([vt_eom_ca, [vt_eom_tmp]])
        vang_eom_ca = np.concatenate([vang_eom_ca, [vang_eom_tmp]])

    # How much does the error on weighted mean improve when we include the young stars?
    # Calculate this at the speckle and the AO field-of-view distances
    sp_ny = np.where(np.abs(r2d_ns - 2.5) == np.abs(r2d_ns - 2.5).min())[0]
    sp_all = np.where(np.abs(r2d_as - 2.5) == np.abs(r2d_as - 2.5).min())[0]
    sp_impX = vx_wavgerr_c[sp_ny] / vx_wavgerr_ca[sp_all]
    sp_impY = vy_wavgerr_c[sp_ny] / vy_wavgerr_ca[sp_all]

    ao_ny = np.where(np.abs(r2d_ns - 5) == np.abs(r2d_ns - 5).min())[0]
    ao_all = np.where(np.abs(r2d_as - 5) == np.abs(r2d_as - 5).min())[0]
    ao_impX = vx_wavgerr_c[ao_ny] / vx_wavgerr_ca[ao_all]
    ao_impY = vy_wavgerr_c[ao_ny] / vy_wavgerr_ca[ao_all]
    print ''
    print 'Improvement in error on weighted mean when including young stars:'
    print 'Speckle FOV (R=2.5"): %5.3f in X, %5.3f in Y' % (sp_impX, sp_impY)
    print 'AO FOV (R=5"): %5.3f in X, %5.3f in Y' % (ao_impX, ao_impY)
          

    # For the all-star case, calculate the predicted improvement in error on
    # weighted avg with time; do this at speckle and AO field of views
    # This should go as (1/t)^(3/2)
    t_now = 2010.339 # last maser mosaic obs
    t_init = 2005.495 # first maser mosaic obs
    t = np.arange(t_now,t_now+25,0.1)
    dt_factor = ((t_now - t_init) / (t - t_init))**(3./2.)
    vx_wavgerr_spImp = vx_wavgerr_ca[sp_all] * dt_factor
    vy_wavgerr_spImp = vy_wavgerr_ca[sp_all] * dt_factor
    vx_wavgerr_aoImp = vx_wavgerr_ca[ao_all] * dt_factor
    vy_wavgerr_aoImp = vy_wavgerr_ca[ao_all] * dt_factor

    # What is the error on the mean at the speckle FOV limit? (do not include young stars)
    vx_eom_sp = vx_eom_c[sp_ny]
    vy_eom_sp = vy_eom_c[sp_ny]
    vang_eom_sp = vang_eom_c[sp_ny]
    # What is the error on the mean at the AO FOV limit? (do not include young stars)
    vx_eom_ao = vx_eom_c[ao_ny]
    vy_eom_ao = vy_eom_c[ao_ny]
    vang_eom_ao = vang_eom_c[ao_ny]

    print ''
    print 'Cluster vs. Maser Method (Translational):'
    print 'Speckle FOV -- Maser method is %4.2f times better than cluster method' % \
         (((vx_eom_sp + vy_eom_sp) / 2) / ((vx_wavgerr_ca[sp_all] + vy_wavgerr_ca[sp_all]) / 2))
    print 'AO FOV -- Maser method is %4.2f times better than cluster method' % \
         (((vx_eom_ao + vy_eom_ao) / 2) / ((vx_wavgerr_ca[ao_all] + vy_wavgerr_ca[ao_all]) / 2))
    print ''
    print 'Cluster vs. Maser Method (Rotational):'
    print 'Speckle FOV -- Maser method is %4.2f times better than cluster method' % \
         (vang_eom_sp / vang_wavgerr_ca[sp_all])
    print 'AO FOV -- Maser method is %4.2f times better than cluster method' % \
         (vang_eom_ao / vang_wavgerr_ca[ao_all])

    # What is the error on weighted mean (improvement with time) for just the IR masers?
    maserNames = ['irs9', 'irs7', 'irs12N', 'irs28',
                  'irs10EE', 'irs15NE', 'irs17']
    midx = np.zeros(len(maserNames))
    for i in range(len(maserNames)):
        midx[i] = np.where(name == maserNames[i])[0]
    midx = [int(mm) for mm in midx]
    vxe_IRmsr = vxe[midx]
    vye_IRmsr = vye[midx]
    vx_wt_msr = 1./vxe_IRmsr**2
    vx_wavgerr_msr = np.sqrt(1. / vx_wt_msr.sum())
    vy_wt_msr = 1./vye_IRmsr**2
    vy_wavgerr_msr = np.sqrt(1. / vy_wt_msr.sum())
    vx_wavgerr_msr_dt = vx_wavgerr_msr * dt_factor
    vy_wavgerr_msr_dt = vy_wavgerr_msr * dt_factor

    # What is the error on weighted mean for the masers in the radio?
    # Read in the Reid values -- vel errors drop as 1/t^(3/2), then
    # take the weighted average at each increment in time
    reidFile='/u/ghezgroup/data/gc/source_list/reid2010Masers.dat'
    reidTab = asciidata.open(reidFile)
    vxraErr = reidTab[7].tonumpy()
    vyraErr = reidTab[9].tonumpy()
    t_now_rad = 2008.86 # latest radio measurement
    t_rad = np.arange(t_now_rad,t_now_rad+25,0.1)
    # initial observations for each maser in the radio (from data Reid gave us)
    t_initRad = [1998.41, 1995.49, 1996.413, 1998.41, 1995.49, 1995.49, 2000.85]
    vxraErr_t = np.zeros((len(vxraErr),len(t_rad)))
    vyraErr_t = np.zeros((len(vyraErr),len(t_rad)))
    # Loop thru each maser and calculate the improvement with time
    for ii in range(len(vxraErr)):
        dt_radio = ((t_now_rad - t_initRad[ii]) / (t_rad - t_initRad[ii]))**(3./2.)
        vxraErr_t[ii,:] = vxraErr[ii] * dt_radio
        vyraErr_t[ii,:] = vyraErr[ii] * dt_radio
    # Now loop thru the times and calculate the error on the weighted average
    vxraErr_t_wavgerr = np.zeros(len(t_rad))
    vyraErr_t_wavgerr = np.zeros(len(t_rad))
    for tt in range(len(t_rad)):
        vxraErr_t_wt = 1./vxraErr_t[:,tt]**2
        vxraErr_t_wavgerr[tt] = np.sqrt(1. / vxraErr_t_wt.sum())
        vyraErr_t_wt = 1./vyraErr_t[:,tt]**2
        vyraErr_t_wavgerr[tt] = np.sqrt(1. / vyraErr_t_wt.sum())
        
    # Quad sum the IR and radio maser errors 
    # use the current error for IR masers and current error for radio
    # (note: this is not 2010 for radio. last radio measurement was in 2008)
    msr_xErr = np.sqrt(vx_wavgerr_msr**2 + vxraErr_t_wavgerr[0]**2)
    msr_yErr = np.sqrt(vy_wavgerr_msr**2 + vyraErr_t_wavgerr[0]**2)

    # print out current total error (at AO FOV)
    vx_tot_err_now = np.sqrt(vx_wavgerr_ca[ao_all]**2 + vx_wavgerr_msr**2 + vxraErr_t_wavgerr[0]**2) 
    vy_tot_err_now = np.sqrt(vy_wavgerr_ca[ao_all]**2 + vy_wavgerr_msr**2 + vyraErr_t_wavgerr[0]**2) 
    print ''
    print 'Current reference frame stability (IR masers + Radio masers + Secondary Standards):'
    print '(x,y) = (%5.3f, %5.3f) mas/yr' % (vx_tot_err_now,vy_tot_err_now)

    # The total velocity error that sets the reference frame stability
    # is the quad sum of the maser method, IR maser, and radio maser errors
    # But need to shift the array of radio values so the times match up!
    #foo = np.where(np.abs(t_rad-2010).round() == 0)[0]
    foo = 11 # index of the radio time array that gets us closest to 2010
    vxraErr_t_wavgerr = vxraErr_t_wavgerr[foo:] 
    vyraErr_t_wavgerr = vyraErr_t_wavgerr[foo:]
    t_rad = t_rad[foo:]
    # Now that this was trimmed, need to adjust the other arrays so we can add them together!
    ll = len(vxraErr_t_wavgerr)
    vx_tot_err = np.sqrt(vx_wavgerr_aoImp[0:ll]**2 + vx_wavgerr_msr_dt[0:ll]**2 + vxraErr_t_wavgerr**2)
    vy_tot_err = np.sqrt(vy_wavgerr_aoImp[0:ll]**2 + vy_wavgerr_msr_dt[0:ll]**2 + vyraErr_t_wavgerr**2)
    tt = t[0:ll]
    
    # When will the total error be below the required error for GR?
    gr = np.where(vy_tot_err < 0.02)[0] # Y is dominant error term
    print ''
    print 'GR can be detected beginning in year %7.2f' % tt[gr[0]]

    py.clf()
    py.figure(9)
    py.figure(figsize=(6,6))
    py.subplots_adjust(left=0.15,right=0.9,top=0.9,
                    wspace=0.2,hspace=0.2)
    py.clf()
    #py.loglog(r2d_ns,vx_wavgerr_c,'r-',label='X') # error on weighted mean (no yng)
    #py.loglog(r2d_ns,vy_wavgerr_c,'b-',label='Y') # error on weighted mean (no yng)
    py.loglog(r2d_ns,vx_eom_c,'r--',lw=2) # error on mean
    py.loglog(r2d_ns,vy_eom_c,'b--',lw=2) # error on mean
    py.loglog(r2d_as,vx_wavgerr_ca,'r-',lw=2,label='X') # error on weighted mean (all stars)
    py.loglog(r2d_as,vy_wavgerr_ca,'b-',lw=2,label='Y') # error on weighted mean (all stars)
    # Overplot arrows indicating speckle and AO FOVs
    py.text(2.5,0.001,r'$\downarrow$',fontsize=24)
    py.text(5,0.001,r'$\downarrow$',fontsize=24)
    # Overplot the quad sum of IR and radio maser errors
    #py.plot([0.1,100],[msr_xErr,msr_xErr],'r:')
    #py.plot([0.1,100],[msr_yErr,msr_yErr],'b:')
    #py.text(0.12,0.052,'IR + Radio Masers',fontsize=10)
    # Overplot the total (all errors) as we currently measure it
    #py.plot([0.1,100],[vx_tot_err_now,vx_tot_err_now],'r-',lw=3)
    #py.plot([0.1,100],[vy_tot_err_now,vy_tot_err_now],'b-',lw=3)
    py.plot([0.1,100],[msr_xErr,msr_xErr],'r-',lw=3)
    py.plot([0.1,100],[msr_yErr,msr_yErr],'b-',lw=3)
    py.text(0.15,0.048,'Maser Method',fontsize=12)
    py.text(0.125,0.035,'(maser standards)',fontsize=12)
    py.legend(prop=prop,numpoints=1)
    py.text(8.6, 0.28, 'Cluster Method', fontsize=12)
    py.text(8, 0.2, '(zero net cluster', fontsize=12)
    py.text(8, 0.15, 'motion assumed)', fontsize=12)
    py.text(8, 0.005, 'Maser Method', fontsize=12)
    py.text(5.8, 0.0037, '(secondary standards)', fontsize=12)
    py.xlabel('R (arcsec)',fontsize=16)
    py.ylabel('Reference Frame Stability (mas yr$^{-1}$)',fontsize=16)
    py.ylim(1e-3,3)
    # Set the label axis formatting.
    thePlot = py.gca()
    py.setp( thePlot.get_xticklabels(), fontsize=16, fontweight='bold')
    py.setp( thePlot.get_yticklabels(), fontsize=16, fontweight='bold')
    py.savefig(root+'plots/error_mean_vel_cdf.png')
    py.savefig(root+'plots/error_mean_vel_cdf.eps')
    py.close(9)

    py.figure(10)
    py.figure(figsize=(12,6))
    py.clf()
    py.subplot(1,2,1)
    # Overplot 'cluster method' -- error on mean (constant w/ time)
    py.plot([t[0],t[-1]],[vx_eom_ao,vx_eom_ao],'r--')
    py.plot([t[0],t[-1]],[vy_eom_ao,vy_eom_ao],'b--')
    py.text(2020, vy_eom_ao+0.01, 'Cluster Method', fontsize=12)
    # Overplot what we need to detect relativistic precession
    py.plot([t[0],t[-1]],[0.02,0.02],'k-')
    py.text(2022.5,0.025,'Required Stability for',fontsize=12)
    py.text(2022.5,0.020,'GR Effects (S0-2)',fontsize=12)
    # Overplot the total error
    py.semilogy(tt, vx_tot_err,'r-',lw=3) 
    py.semilogy(tt, vy_tot_err,'b-',lw=3)
    py.text(2012.5, 0.07, 'Maser Method',fontsize=12)
    py.axis([2010,2030,5e-3,1])
    py.xlabel('Year',fontsize=14)
    py.ylabel('Reference Frame Stability (mas yr$^{-1}$)',fontsize=14)
    py.subplot(1,2,2)
    py.text(2015.5,0.12,'Maser Method',fontsize=22)
    # Plot the improvement with time for the ref stars for each FOV
    py.semilogy(t, vx_wavgerr_aoImp,'r-.',lw=2)
    py.semilogy(t, vy_wavgerr_aoImp,'b-.',lw=2)
    py.text(2010.8, 0.0043, 'Secondary IR Standards', fontsize=12)
    # Overplot the improvement with time for just the masers (in the IR)
    py.semilogy(t, vx_wavgerr_msr_dt,'r-',lw=2)
    py.semilogy(t, vy_wavgerr_msr_dt,'b-',lw=2)
    py.text(2014,0.045,'Infrared Masers',fontsize=12)
    # Overplot the improvement with time for just the masers (in the radio)
    for ii in range(0,len(t_rad),3):
        py.semilogy(t_rad[ii],vxraErr_t_wavgerr[ii],'r.',ms=3,lw=2)
        py.semilogy(t_rad[ii],vyraErr_t_wavgerr[ii],'b.',ms=3,lw=2)
    py.text(2015.6,0.01,'Radio Masers',fontsize=12)
    py.axis([2010,2030,1e-3,0.2])
    py.xlabel('Year',fontsize=14)
    py.ylabel('Reference Frame Stability (mas yr$^{-1}$)',fontsize=14)
    #py.title('AO Field of View')
    py.savefig(root+'plots/refFrame_stability_2panel.png')
    py.savefig(root+'plots/refFrame_stability_2panel.eps')
    py.savefig(root+'plots/refFrame_stability_2panel.ps')
    py.close(10)

    py.figure(11)
    py.figure(figsize=(6,6))
    py.clf()
    binsIn = py.arange(-10, 10, 0.4)
    (nhx,bhx,phx) = py.hist(vpar,binsIn,color='r',lw=2,histtype='step',label='$v_{\parallel}$')
    (nhy,bhy,phy) = py.hist(vperp,binsIn,color='b',ls='dashed',lw=2,histtype='step',label='$v_{\perp}$')
    py.xlabel(r'Cluster Velocity (mas yr$^{-1}$)',fontsize=18)
    py.ylabel('N',fontsize=18)
    #py.title(r'Velocity Parallel \& Perpendicular to Galactic Plane')
    py.legend(prop=prop)
    py.axis([-12,12,0,max(nhx.max(),nhy.max())+10])
    py.savefig(root+'plots/secondary_hist_absVels_parPerp.png')
    py.savefig(root+'plots/secondary_hist_absVels_parPerp.eps')
    py.close(11)

    py.figure(12)
    py.figure(figsize=(12,6))
    py.clf()
    py.subplots_adjust(left=0.1,right=0.98,top=0.95,
                    wspace=0.3,hspace=0.5)
    py.subplot(1,2,1)
    binsIn = py.arange(-10, 10, 0.4)
    (nhx,bhx,phx) = py.hist(vt_n_in,binsIn,color='r',histtype='step',label='$v_{inner}$')
    (nhy,bhy,phy) = py.hist(vt_n_out,binsIn,color='b',histtype='step',label='$v_{outer}$')
    py.text(-10,40,'Inner (r $<$ 5"): %6.3f +- %5.3f mas/yr' % (vt_n_in.mean(),vt_in_wavg_boot))
    py.text(-10,38,'Outer (r $>=$ 5"): %6.3f +- %5.3f mas/yr' % (vt_n_out.mean(),vt_out_wavg_boot))
    py.xlabel('Tangential Velocity (mas/yr)',fontsize=16)
    py.ylabel('N',fontsize=16)
    py.legend(prop=prop)
    py.axis([-12,12,0,max(nhx.max(),nhy.max())+10])
    py.subplot(1,2,2)
    (nhx,bhx,phx) = py.hist(vang_n_in,binsIn,color='r',histtype='step',label='$v_{inner}$')
    (nhy,bhy,phy) = py.hist(vang_n_out,binsIn,color='b',histtype='step',label='$v_{outer}$')
    py.text(-10,250,'Inner (r $<$ 5"): %6.3f +- %5.3f mas/yr/asec' % (vang_n_in.mean(),vang_in_wavg_boot))
    py.text(-10,235,'Outer (r $>=$ 5"): %6.3f +- %5.3f mas/yr/asec' % (vang_n_out.mean(),vang_out_wavg_boot))
    py.xlabel('Angular Velocity (mas/yr/asec)',fontsize=16)
    py.ylabel('N',fontsize=16)
    py.legend(prop=prop)
    py.axis([-12,12,0,max(nhx.max(),nhy.max())+10])
    py.savefig(root+'plots/secondary_hist_absVels_parPerp_inVsOut.png')
    py.savefig(root+'plots/secondary_hist_absVels_parPer_inVsOut.eps')
    py.close(12)

    py.figure(13)
    py.figure(figsize=(12,6))
    py.clf()
    py.subplots_adjust(left=0.1,right=0.98,top=0.95,
                    wspace=0.3,hspace=0.5)
    py.subplot(1,2,1)
    binsIn = py.arange(-10, 10, 0.4)
    (nhx,bhx,phx) = py.hist(vx_n_in,binsIn,color='r',histtype='step',label='$v_{inner}$')
    (nhy,bhy,phy) = py.hist(vx_n_out,binsIn,color='b',histtype='step',label='$v_{outer}$')
    py.text(-10,48,'Inner (r $<$ 5"): %6.3f +- %5.3f mas/yr/asec' % (vx_n_in.mean(),vx_in_wavg_boot))
    py.text(-10,45,'Outer (r $>=$ 5"): %6.3f +- %5.3f mas/yr/asec' % (vx_n_out.mean(),vx_out_wavg_boot))
    py.xlabel('X Velocity (mas/yr)',fontsize=16)
    py.ylabel('N',fontsize=16)
    py.legend(prop=prop)
    py.axis([-12,12,0,max(nhx.max(),nhy.max())+10])
    py.subplot(1,2,2)
    (nhx,bhx,phx) = py.hist(vy_n_in,binsIn,color='r',histtype='step',label='$v_{inner}$')
    (nhy,bhy,phy) = py.hist(vy_n_out,binsIn,color='b',histtype='step',label='$v_{outer}$')
    py.text(-10,43,'Inner (r $<$ 5"): %6.3f +- %5.3f mas/yr' % (vy_n_in.mean(),vy_in_wavg_boot))
    py.text(-10,40,'Outer (r $>=$ 5"): %6.3f +- %5.3f mas/yr' % (vy_n_out.mean(),vy_out_wavg_boot))
    py.xlabel('Y Velocity (mas/yr)',fontsize=16)
    py.ylabel('N',fontsize=16)
    py.legend(prop=prop)
    py.axis([-12,12,0,max(nhx.max(),nhy.max())+10])
    py.savefig(root+'plots/secondary_hist_absVels_xy_inVsOut.png')
    py.savefig(root+'plots/secondary_hist_absVels_xy_inVsOut.eps')
    py.close(13)

    # Turn off Latex Processing
    py.rc('text', usetex=False)
    
    

#    # Can compare the central 10" label file to the maser mosaic label file
#    # if the former was passed in:
#    if labelFile == 'lis/label_new.dat':
#        # Compare positions from the new label_new.dat (central 10" stars)
#        # to those in allStars_absolute.dat (stars from maser mosaic).
#        # Only look at named sources in the two lists.
#        msr = asciidata.open(root+'source_list/allStars_absolute.dat')
#        name_m = msr[0].tonumpy()
#        mag_m = msr[1].tonumpy()
#        x_m = msr[2].tonumpy()
#        y_m = msr[3].tonumpy()
#        xe_m = msr[4].tonumpy()
#        ye_m = msr[5].tonumpy()
#        vx_m = msr[6].tonumpy()
#        vy_m = msr[7].tonumpy()
#        vxe_m = msr[8].tonumpy()
#        vye_m = msr[9].tonumpy()
#        t0_m = msr[10].tonumpy()
#        r2d_m = msr[12].tonumpy()
#    
#        # Find the stars in common across the 2 lists
#        mtch_m = []
#        mtch_l = []
#        noName = 0
#        for ii in range(len(name)):
#            if 'star' not in name[ii]:
#                idx = np.where(name_m == name[ii])[0]
#                if len(idx) > 0:
#                    mtch_m = np.concatenate([mtch_m,[idx[0]]])
#                    mtch_l = np.concatenate([mtch_l,[ii]])
#    
#            # How many un-named stars are there?
#            if 'star' in name[ii]:
#                noName += 1
#  
#
#        print '%i stars in label_absolute.dat do not have a name' % noName
#
#        mtch_l = [int(ll) for ll in mtch_l]
#        name_l = [name[ll] for ll in mtch_l]
#        mag_l = mag[mtch_l]
#        x_l = x[mtch_l]
#        y_l = y[mtch_l]
#        xe_l = xe[mtch_l]
#        ye_l = ye[mtch_l]
#        vx_l = vx[mtch_l]
#        vy_l = vy[mtch_l]
#        vxe_l = vxe[mtch_l]
#        vye_l = vye[mtch_l]
#        t0_l = t0[mtch_l]
#        r2d_l = r2d[mtch_l]
#    
#        mtch_m = [int(mm) for mm in mtch_m]
#        name_m = [name_m[mm] for mm in mtch_m]
#        mag_m = mag_m[mtch_m]
#        x_m = x_m[mtch_m]
#        y_m = y_m[mtch_m]
#        xe_m = xe_m[mtch_m]
#        ye_m = ye_m[mtch_m]
#        vx_m = vx_m[mtch_m]
#        vy_m = vy_m[mtch_m]
#        vxe_m = vxe_m[mtch_m]
#        vye_m = vye_m[mtch_m]
#        t0_m = t0_m[mtch_m]
#        r2d_m = r2d_m[mtch_m]
#    
#        dvx = vx_l - vx_m
#        dvy = vy_l - vy_m
#    
#        # remove outliers
#        kp = np.where((abs(dvx) < 1.5) & (abs(dvy) < 1.5))[0]
#        x_l = x_l[kp]
#        y_l = y_l[kp]
#        x_m = x_m[kp]
#        y_m = y_m[kp]
#        dvx = dvx[kp]
#        dvy = dvy[kp]
#    
#        print 'Found %i named stars in common between the two lists' % len(kp)
#        print 'Average X velocity difference = %5.3f +- %5.3f mas/yr' % \
#              (dvx.mean(),dvx.std(ddof=1))
#        print 'Average Y velocity difference = %5.3f +- %5.3f mas/yr' % \
#              (dvy.mean(),dvy.std(ddof=1))
#    
#        # Plot the vector differences (velocities)
#        py.figure(9)
#        py.clf()
#        qvr = py.quiver(x_l,y_l,dvx,dvy,color='black', units='y',scale=1.3)
#        py.quiverkey(qvr,-5,-7,1,'1 mas/yr',coordinates='data',
#                  fontproperties={'size': 'smaller'})
#        py.xlabel('X (pix)')
#        py.ylabel('Y (pix)')
#        py.title('Velocity Deltas of Secondary Stars')
#        py.savefig(root+'plots/secondary_maser_deltaVel.png')
    


#----------
# Contours
#----------
def getContourLevels(probDist,numSig=3):
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

    # Determine point at which we reach 68% level
    if numSig == 1:
        percents = np.array([0.6827])
    if numSig == 2:
        percents = np.array([0.6827, 0.9545])
    if numSig == 3:
        percents = np.array([0.6827, 0.9545, 0.9973])
    levels = np.zeros(len(percents), dtype=float)
    for ii in range(len(levels)):
        # Get the index of the pixel at which the CDF
        # reaches this percentage (the first one found)
        idx = (np.where(cdf < percents[ii]))[0]
        
        # Now get the level of that pixel
        levels[ii] = pixSort[idx[-1]]
        
    return levels
    

def refStar_distribution(absDir='10_06_22',step=0.5):
    """
    Plots the number of reference stars as a function of radius
    """
    root = '/u/syelda/research/gc/absolute/'
    absFile = root + absDir + '/lis/absolute_refs.dat'
    data = asciidata.open(absFile)

    name = data[0].tonumpy()
    x = data[2].tonumpy()
    y = data[3].tonumpy()
    r = np.hypot(x,y)
    print 'Maximum radius: %4.2f' % r.max()
    print 'Number of stars in %s = %i' % (absFile,len(name))

    # Determine which stars are young
    young = youngNames.loadYoungStars()

    # Determine which stars are old
    old = oldNames.loadOldStars()

    # Get number of stars in each radial bin
    rbin = np.arange(0,r.max(),step)
    numBin = 0
    yngBin = 0
    oldBin = 0
    nyBin = 0 # non-known-young bin
    unkBin = 0 # unknown spectal IDs
    numStars = []
    numYng = []
    numNotYng = []
    numOld = []
    numUnk = []
    for rr in range(len(rbin)):
        idx = np.where((r > rbin[rr]) & (r < rbin[rr]+step))[0]
        numBin += len(idx)
        numStars = np.concatenate([numStars,[numBin]])

        # Only young stars:
        for ii in range(len(idx)):
            if name[idx[ii]] in young:
                yngBin += 1
            else:
                # Non-known-young stars
                nyBin += 1
        numYng = np.concatenate([numYng, [int(yngBin)]])
        numNotYng = np.concatenate([numNotYng, [int(nyBin)]])

        # Only known old stars:
        for ii in range(len(idx)):
            if name[idx[ii]] in old:
                oldBin += 1
        numOld = np.concatenate([numOld, [int(oldBin)]])

        # Get the unknown sources:
        for ii in range(len(idx)):
            if ((name[idx[ii]] not in old) & (name[idx[ii]] not in young)):
                unkBin += 1
        numUnk = np.concatenate([numUnk, [int(unkBin)]])
    print 'Number of known young stars = %i' % yngBin
    print 'Number of known old stars = %i' % oldBin
    print 'Number of stars with unknown spectroscopic IDs = %i' % unkBin

    rmid = rbin + step/2.0

    import matplotlib.font_manager
    prop = matplotlib.font_manager.FontProperties(size=16)
    py.figure(1,figsize=(6,6))
    py.clf()
    py.subplots_adjust(left=0.15,right=0.9,top=0.9,
                    wspace=0.2,hspace=0.2)
    py.loglog(rmid,numStars,'k-',label='All',lw=3)
    py.loglog(rmid,numYng,'b-',label='Young',lw=1) 
    py.loglog(rmid,numOld,'r-',label='Old',lw=2)
    py.loglog(rmid,numUnk,'g-',label='Unknown')
    #py.loglog(rmid,numNotYng,'gd',label='Not Young',ms=4)
    py.loglog([2.5,2.5],[0.5,numStars.max()+100],'k--',lw=2) # Speckle FOV
    py.loglog([5,5],[0.5,numStars.max()+100],'k--',lw=2) # AO FOV
    py.gca().yaxis.set_major_formatter(py.FormatStrFormatter('%i'))
    py.axis([0.1,50,0.5,numStars.max()+200])
    py.xlabel('R (arcsec)',fontsize=16)
    py.ylabel('Cumulative Number of Reference Stars',fontsize=16)
    py.legend(prop=prop,numpoints=1,loc=2)
    # Set the label axis formatting.
    thePlot = py.gca()
    py.setp( thePlot.get_xticklabels(), fontsize=16)
    py.setp( thePlot.get_yticklabels(), fontsize=16)
    #thePlot.xaxis.set_major_formatter(py.FormatStrFormatter('%g'))
    #thePlot.yaxis.set_major_formatter(py.FormatStrFormatter('%g'))
    py.savefig(root + absDir + '/plots/numRefStars_radius.png')
    py.savefig(root + absDir + '/plots/numRefStars_radius.eps')
    py.close(1)


def sgraPos(root='/u/syelda/research/gc/aligndir/',alnDir='06setup_only/10_08_26_fixVel'):
    """
    Plots the location of Sgr A* IR from the AO maps relative to the origin
    (so from align_d_rms_abs.* files).
    """

    align = '/align/align_d_rms_abs'
    s = starset.StarSet(root + alnDir + align)

    name = np.array(s.getArray('name'))

    starCnt = len(s.stars)
    epochCnt = len(s.stars[0].e)

    sgr = np.where(name == 'SgrA')[0]

    sgrx = []
    sgry = []
    sgrx_e = []
    sgry_e = []
    epoch = []

    for ee in range(epochCnt):
        xa = s.getArrayFromEpoch(ee, 'xpix')
        ya = s.getArrayFromEpoch(ee, 'ypix')
        xa_err = s.getArrayFromEpoch(ee, 'xerr_a')
        ya_err = s.getArrayFromEpoch(ee, 'yerr_a')

        # Get SgrA's positions
        if xa[sgr] < -900:
            continue
        else:
            sgrx = np.concatenate([sgrx,xa[sgr]])
            sgry = np.concatenate([sgry,ya[sgr]])
            sgrx_e = np.concatenate([sgrx_e,xa_err[sgr]])
            sgry_e = np.concatenate([sgry_e,ya_err[sgr]])
            epoch = np.concatenate([epoch,[ee]])

    epoch = [int(ee) for ee in epoch]

    epochID = np.array(['06maylgs1','06junlgs','06jullgs','07maylgs','07auglgs',
               '08maylgs1','08jullgs','09maylgs','09jullgs','09seplgs',
               '10maylgs','10jullgs1','10auglgs'])

    epochID = epochID[epoch]

    py.figure(1,figsize=(6,6))
    py.clf()
    py.subplots_adjust(left=0.15,right=0.9,top=0.9,wspace=0.2,hspace=0.2)
    py.plot(sgrx,sgry,'r.')
    py.plot([0],[0],'kx',ms=10)
    py.plot([0],[0],'k+',ms=10)
    for ii in range(len(epoch)):
        py.text(sgrx[ii]+0.001,sgry[ii]+0.001,'%s' % epochID[ii],fontsize=9)
    py.axis([-0.005,0.015,-0.015,0.012])
    py.xlabel('X (arcsec)')
    py.ylabel('Y (arcsec)')
    py.title(alnDir)
    py.savefig(root+alnDir+'/plots/sgraPos.png')



def velocity_map_nearestNbr(absDir='11_01_03/', magCut=None, magRange=None,
                            absFile='lis/absolute_refs.dat',excludeYng=True,
                            onlyKnownOld=False, onlyKnownYng=False,
                            nstars=50, step=0.25,
                            run_multi_nNbrs=False, cntrl10=False, writeFile=True,
                            plotMsrs=True):
    """
    Plot a smooth map of the average velocities (X and Y separately)
    over the field, where the velocities are averaged over patches of the field

    magCut:     (integer) set to faintest magnitude to include in velocity maps
    magRange:   (integer) set to the delta of the magnitudes you want to plot
    		For example, to plot K=15-16 stars, set magCut to 16 and magRange to 1.
    onlyKnownOld: set to True to use ONLY stars that have been spectroscopically
    		  identified (by us) as old.
    excludeYng: set to True to remove young stars from calculation of averages
    nstars: 	number of nearest neighbors to use in average
    step:	distance in arcseconds at which the average is calculated for nstars
    cntrl10:	set to True to only plot the central 10 arcsec, otherwise, the maser
    		mosaic FOV will be plotted. (def = False)
    plotMsrs:   set to True to plot X's where the masers are located
    """
    root = '/u/syelda/research/gc/absolute/' + absDir

    suffix = ''

    data = asciidata.open(root+absFile)
    name = np.array(data[0].tonumpy())
    mag = data[1].tonumpy()
    x = data[2].tonumpy()
    y = data[3].tonumpy()
    xerr = data[4].tonumpy()
    yerr = data[5].tonumpy()
    vx = data[6].tonumpy()
    vy = data[7].tonumpy()
    vxerr = data[8].tonumpy()
    vyerr = data[9].tonumpy()
    r = np.hypot(x,y)
    use = data[11].tonumpy()

    if onlyKnownOld == True:
        # Only include stars known to be old
        old = oldNames.loadOldStars()

        idOld = []
        for ii in range(len(x)):
            # Find old stars
            if name[ii] in old:
                idOld = np.concatenate([idOld, [ii]])

        nOld = len(idOld)
        print 'Number of old stars to be used: %s' % nOld
        idOld = [int(ii) for ii in idOld]

        name = [name[ii] for ii in idOld]       
        mag = mag[idOld]       
        x = x[idOld]       
        y = y[idOld]       
        xerr = xerr[idOld]       
        yerr = yerr[idOld]       
        vx = vx[idOld]       
        vy = vy[idOld]       
        vxerr = vxerr[idOld]       
        vyerr = vyerr[idOld]       
        r = r[idOld]       
        use = use[idOld]
        suffix = suffix + '_knownOld'

    if onlyKnownYng == True:
        # Only include stars known to be young
        yng = youngNames.loadYoungStars()

        idYng = []
        for ii in range(len(x)):
            # Find young stars
            if name[ii] in yng:
                idYng = np.concatenate([idYng, [ii]])

        nYng = len(idYng)
        print 'Number of young stars to be used: %s' % nYng
        idYng = [int(ii) for ii in idYng]

        name = [name[ii] for ii in idYng]       
        mag = mag[idYng]       
        x = x[idYng]       
        y = y[idYng]       
        xerr = xerr[idYng]       
        yerr = yerr[idYng]       
        vx = vx[idYng]       
        vy = vy[idYng]       
        vxerr = vxerr[idYng]       
        vyerr = vyerr[idYng]       
        r = r[idYng]       
        use = use[idYng]
        suffix = suffix + '_knownYng'
        excludeYng = False

    if excludeYng == True:
        # Do not include young stars in alignment b/c of known net rotation
        young = youngNames.loadYoungStars()

        for ii in range(len(x)):
            # Exclude young stars
            if name[ii] in young:
                use[ii] = 0
           

        idYng = np.where(use == 0)[0]
        nYng = len(idYng)
        print 'Number of young stars to be excluded: %s' % nYng

        idLate = np.where(use == 1)[0]
        print 'Number of non-known-young stars: %s' % len(idLate)

        name = [name[ii] for ii in idLate]       
        mag = mag[idLate]       
        x = x[idLate]       
        y = y[idLate]       
        xerr = xerr[idLate]       
        yerr = yerr[idLate]       
        vx = vx[idLate]       
        vy = vy[idLate]       
        vxerr = vxerr[idLate]       
        vyerr = vyerr[idLate]       
        r = r[idLate]       
        use = use[idLate]       


    if magCut != None:
        if magRange == None:
            kp = np.where(mag <= magCut)[0]
            name = [name[nn] for nn in kp]
            mag = mag[kp]
            x = x[kp]
            y = y[kp]
            xerr = xerr[kp]
            yerr = yerr[kp]
            vx = vx[kp]
            vy = vy[kp]
            vxerr = vxerr[kp]
            vyerr = vyerr[kp]
            r = r[kp]
            use = use[kp]
            suffix = suffix + '_mag%s' % str(magCut)
            print 'Number of stars to be used after magnitude cut: %s' % len(kp)
        else:
            kp = np.where((mag < magCut) & (mag > (magCut-1)))[0]
            name = [name[nn] for nn in kp]
            mag = mag[kp]
            x = x[kp]
            y = y[kp]
            xerr = xerr[kp]
            yerr = yerr[kp]
            vx = vx[kp]
            vy = vy[kp]
            vxerr = vxerr[kp]
            vyerr = vyerr[kp]
            r = r[kp]
            use = use[kp]
            suffix = suffix + '_mag%s_to_mag%s' % (str(magCut), str(magCut-1))
            print 'Number of stars to be used after magnitude cut: %s' % len(kp)


    # save the maser info to plot their locations
    msrX = x[0:7]
    msrY = y[0:7]

    # Pull the chi2 values from the polyfit files
    s = starset.StarSet(root + '/align/align_abs_t')
    s.loadPolyfit(root + '/polyfit_noDistErr/fit', arcsec=0)
    nameP = np.array(s.getArray('name'))
    xchi2tmp = s.getArray('fitpXv.chi2')
    xchi2rtmp = s.getArray('fitpXv.chi2red')
    ychi2tmp = s.getArray('fitpYv.chi2')
    ychi2rtmp = s.getArray('fitpYv.chi2red')

    xchi2 = np.zeros(len(name),dtype=float)
    ychi2 = np.zeros(len(name),dtype=float)
    xchi2r = np.zeros(len(name),dtype=float)
    ychi2r = np.zeros(len(name),dtype=float)
    # Need to match star names from absFile and from polyfit files
    for ii in range(len(name)):
        # Find it in polyfit
        idx = np.where(nameP == name[ii])[0]
        xchi2[ii] = xchi2tmp[idx]
        ychi2[ii] = ychi2tmp[idx]
        xchi2r[ii] = xchi2rtmp[idx]
        ychi2r[ii] = ychi2rtmp[idx]
        

    xmin = x.min()
    xmax = x.max()
    ymin = y.min()
    ymax = y.max()

    print 'The full field extends from E to W (%5.2f, %5.2f) and from N to S (%5.2f, %5.2f)' % \
          (xmax, xmin, ymax, ymin)

    imgFile = '/u/ghezgroup/data/gc/06maylgs1/combo/mag06maylgs1_msr_kp.fits'
    scale = 0.00993
    sgra = [1596.,1006.]
    img = pyfits.getdata(imgFile)
    imgsize = (img.shape)[0]

    if cntrl10 == True:
        xmin = -5.0
        xmax = 5.0
        ymin = -5.0
        ymax = 5.0
        print 'Plotting only the AO field of view'
        fov = np.hypot(xmax,ymax)

        ao = np.where((np.abs(x) < xmax) & (np.abs(y) < ymax))[0]
        mag = mag[ao]       
        x = x[ao]       
        y = y[ao]       
        xerr = xerr[ao]       
        yerr = yerr[ao]       
        vx = vx[ao]       
        vy = vy[ao]       
        vxerr = vxerr[ao]       
        vyerr = vyerr[ao]       
        r = r[ao]       
        use = use[ao]
        print '%i reference stars in the AO field of view' % len(ao)

        suffix = suffix + '_cntrl10'

    else:
        fov = 18.0


    # Calculate average velocities
    xbins = np.arange(np.floor(xmin), np.ceil(xmax)+step, step)
    ybins = np.arange(np.floor(ymin), np.ceil(ymax)+step, step)
    rmax =  np.zeros((len(ybins),len(xbins)),dtype='f')
    med_vx = np.zeros((len(ybins),len(xbins)),dtype='f')
    med_vy = np.zeros((len(ybins),len(xbins)),dtype='f')
    med_vxe = np.zeros((len(ybins),len(xbins)),dtype='f')
    med_vye = np.zeros((len(ybins),len(xbins)),dtype='f')
    ave_vx = np.zeros((len(ybins),len(xbins)),dtype='f')
    ave_vy = np.zeros((len(ybins),len(xbins)),dtype='f')
    eom_vx = np.zeros((len(ybins),len(xbins)),dtype='f')
    eom_vy = np.zeros((len(ybins),len(xbins)),dtype='f')
    med_xchi2r = np.zeros((len(ybins),len(xbins)),dtype='f')
    med_ychi2r = np.zeros((len(ybins),len(xbins)),dtype='f')
    rad = np.zeros((len(ybins),len(xbins)),dtype='f')

    if writeFile == True:
        # Output text file
        _out = root + 'lis/mean_velocities_nearestNbr.txt'
        out = open(_out, 'w')
        fmt = '%8.2f  %8.2f  %11.3f  %11.3f  %6.3f  %6.3f\n'
        hdr = '#%8s  %8s  %11s  %11s  %6s  %6s\n'
        out.write(hdr % ('X (")','Y (")','Vx (mas/yr)','Vy (mas/yr)','Vxe','Vye'))

    def build_map(nstars, writeFile):
        # Build mean and median velocity arrays
        outl = 0
        for jj in range(len(ybins)):
	    for ii in range(len(xbins)):
	        rmax[jj][ii] = np.sort(np.sqrt((x-xbins[ii])**2+
	    		                (y-ybins[jj])**2))[nstars-1]
                #print rmax[jj][ii]
	        idx = (np.where(np.sqrt((x-xbins[ii])**2+(y-ybins[jj])**2)
	    	   <=rmax[jj][ii]))[0]

	        # Calculate median velocity in this bin
	        med_vx[jj][ii] = np.median(vx[idx])
	        med_vy[jj][ii] = np.median(vy[idx])

	        # Calculate median velocity error in this bin
	        med_vxe[jj][ii] = np.median(vxerr[idx])
	        med_vye[jj][ii] = np.median(vyerr[idx])
    
    	        # Calculate mean velocity in this bin
    	        ave_vx[jj][ii] = np.mean(vx[idx])
	        ave_vy[jj][ii] = np.mean(vy[idx])

                # Get the error on the mean of the N neighbors' velocities
                eom_vx[jj][ii] = vx[idx].std(ddof=1) / np.sqrt(len(idx))
                eom_vy[jj][ii] = vy[idx].std(ddof=1) / np.sqrt(len(idx))

                # Save the radius of each bin
                rad[jj][ii] = np.hypot(xbins[ii],ybins[jj])

                # Sigma clipping: throw out any of the neighbors that are
                # more than 3 sigma off the mean
                sigX = np.abs(vx[idx] - ave_vx[jj][ii])/vx[idx].std(ddof=1) 
                sigY = np.abs(vy[idx] - ave_vy[jj][ii])/vy[idx].std(ddof=1) 
                kp = np.where((sigX < 3.0) & (sigY < 3.0))[0]
                if len(kp) < len(idx):
                    #print 'found a 5 sigma outlier'
                    outl += (len(idx) - len(kp))
    
                    # Recompute the mean and eom after sigma clipping if outliers found
    	            ave_vx[jj][ii] = np.mean(vx[idx[kp]])
	            ave_vy[jj][ii] = np.mean(vy[idx[kp]])

                if writeFile == True:
                    # Save these values off to a text file for later use
                    out.write(fmt % (xbins[ii], ybins[jj],
                                     ave_vx[jj][ii], ave_vy[jj][ii],
                                     eom_vx[jj][ii], eom_vy[jj][ii]))  

	        # Calculate median chi2 in this bin
	        med_xchi2r[jj][ii] = np.median(xchi2r[idx])
	        med_ychi2r[jj][ii] = np.median(ychi2r[idx])

        #print 'Found a total of %d outliers' % outl

        if writeFile == True:
            out.close()

        if run_multi_nNbrs == True:
            return med_vx, med_vy, (med_vx.flatten()).std(ddof=1), (med_vy.flatten()).std(ddof=1), rad


    def plot_dither_pattern():
        # Create arrays for overplotting the maser mosaic dither pattern
        msrCx = np.array([-0.5, 9.7, 9.7, -0.5, -0.5]) # 10.2 arcsecond width
        msrCy = np.array([-3.8, -3.8, 6.4, 6.4, -3.8]) # 10.2 arcsecond height

        py.plot(msrCx, msrCy, 'k--') # Central dither
        py.plot(msrCx+6.0,msrCy+6.0,'b--')
        py.plot(msrCx-6.0,msrCy+6.0,'b--')
        py.plot(msrCx+6.0,msrCy-6.0,'b--')
        py.plot(msrCx-6.0,msrCy-6.0,'b--')
        py.plot(msrCx+6.0,msrCy,'r--')
        py.plot(msrCx-6.0,msrCy,'r--')
        py.plot(msrCx,msrCy+6.0,'r--')
        py.plot(msrCx,msrCy-6.0,'r--')

    def compute_ratio(med_vx,med_vy,rad,rbins,vdisp,occ):
        #med_vx_over_rms = np.zeros((len(med_vx.flatten())), dtype='f')
        #med_vy_over_rms = np.zeros((len(med_vy.flatten())), dtype='f')
        med_vx_over_rms = []
        med_vy_over_rms = []
        for ii in range(len(med_vx.flatten())):
            rdx = np.where(np.abs(rad.flatten()[ii] - rbins) == np.abs(rad.flatten()[ii] - rbins).min())[0]
            disp = vdisp[rdx]
            nocc = occ[rdx]
            med_vx_r = med_vx.flatten()[ii] / (disp/np.sqrt(nocc))
            med_vy_r = med_vy.flatten()[ii] / (disp/np.sqrt(nocc))
            #med_vx_r = med_vx.flatten()[ii] / disp
            #med_vy_r = med_vy.flatten()[ii] / disp
            med_vx_over_rms = np.concatenate([med_vx_over_rms, med_vx_r])
            med_vy_over_rms = np.concatenate([med_vy_over_rms, med_vy_r])

        return med_vx_over_rms, med_vy_over_rms

    # Build the nearest neighbor maps
    build_map(nstars, writeFile=writeFile)

    # Get the velocity dispersion as a function of radius
    rbins, vdisp, vdisperr, occ = vel_disp(x,y,r,vx,vy,vxerr,vyerr,fov)

    # Compute the median nearest neighbor velocity / velocity dispersion
    # Have to figure out what the dispersion is at each point in the NN map
    med_vx_over_rms, med_vy_over_rms = compute_ratio(med_vx,med_vy,rad,rbins,vdisp,occ)

    # Print out some details for the location of Sgr A*
    sdx = np.where(rad.flatten() == 0)[0]
    print '-----------'
    print 'At location of Sgr A*:'
    print 'Velocity dispersion / Sqrt(N stars within 1 arcsec): %6.3f mas/yr' % vdisp[0]
    print 'Median (X, Y) Velocity divided by number above: (%6.3f, %6.3f)' % \
          (med_vx_over_rms[sdx], med_vy_over_rms[sdx])
    print '-----------'


    #################
    # PLOTS 
    #################

    # Plot median velocities
    py.figure(1)
    py.figure(figsize=(12,6))
    py.subplots_adjust(left=0.07,right=0.98,top=0.95,wspace=0.2)
    py.clf()
    py.subplot(1,2,1)
    py.imshow(med_vx,interpolation='nearest',vmin=-1.5,vmax=2.0,
              extent=[xbins[0],xbins[-1],ybins[0],ybins[-1]],
              origin='lowerleft',cmap=py.cm.gist_rainbow_r)
    py.plot([0],[0],'kx') # Sgr A*
    plot_dither_pattern()
    if (plotMsrs == True):
        py.plot(msrX,msrY,'kx',ms=8,mew=2)
    py.axis([np.ceil(xmax),np.floor(xmin),np.floor(ymin),np.ceil(ymax)])
    cbar = py.colorbar(shrink=.75)
    cbar.set_label('(mas/yr)')
    py.title('Median X Velocity: %i Nearest Neighbors' % nstars, fontsize=12)
    py.xlabel('RA from Sgr A* (arcsec)')
    py.ylabel('Dec from Sgr A* (arcsec)')
    py.subplot(1,2,2)
    py.imshow(med_vy,interpolation='nearest',vmin=-2.5,vmax=2.5,
              extent=[xbins[0],xbins[-1],ybins[0],ybins[-1]],
              origin='lowerleft',cmap=py.cm.gist_rainbow_r)
    py.plot([0],[0],'kx') # Sgr A*
    plot_dither_pattern()
    if (plotMsrs == True):
        py.plot(msrX,msrY,'kx',ms=8,mew=2)
    py.axis([np.ceil(xmax),np.floor(xmin),np.floor(ymin),np.ceil(ymax)])
    cbar = py.colorbar(shrink=.75)
    cbar.set_label('(mas/yr)')
    py.title('Median Y Velocity: %i Nearest Neighbors' % nstars, fontsize=12)
    py.xlabel('RA from Sgr A* (arcsec)')
    py.ylabel('Dec from Sgr A* (arcsec)')
    py.savefig(root + 'plots/secondary_median_vel_nearestNbr%s%s.png' % (nstars, suffix))
    py.close(1)

    # Plot median velocity errors
    py.figure(2)
    py.figure(figsize=(12,6))
    py.subplots_adjust(left=0.07,right=0.98,top=0.95,wspace=0.2)
    py.clf()
    py.subplot(1,2,1)
    py.imshow(med_vxe,interpolation='nearest',
              extent=[xbins[0],xbins[-1],ybins[0],ybins[-1]],
              origin='lowerleft',cmap=py.cm.gist_rainbow_r)
    py.plot([0],[0],'kx') # Sgr A*
    plot_dither_pattern()
    if (plotMsrs == True):
        py.plot(msrX,msrY,'kx',ms=8,mew=2)
    py.axis([np.ceil(xmax),np.floor(xmin),np.floor(ymin),np.ceil(ymax)])
    cbar = py.colorbar(shrink=.75)
    cbar.set_label('(mas/yr)')
    py.title('Median X Velocity Error: %i Nearest Neighbors' % nstars, fontsize=12)
    py.xlabel('RA from Sgr A* (arcsec)')
    py.ylabel('Dec from Sgr A* (arcsec)')
    py.subplot(1,2,2)
    py.imshow(med_vye,interpolation='nearest',
              extent=[xbins[0],xbins[-1],ybins[0],ybins[-1]],
              origin='lowerleft',cmap=py.cm.gist_rainbow_r)
    py.plot([0],[0],'kx') # Sgr A*
    plot_dither_pattern()
    if (plotMsrs == True):
        py.plot(msrX,msrY,'kx',ms=8,mew=2)
    py.axis([np.ceil(xmax),np.floor(xmin),np.floor(ymin),np.ceil(ymax)])
    cbar = py.colorbar(shrink=.75)
    cbar.set_label('(mas/yr)')
    py.title('Median Y Velocity Error: %i Nearest Neighbors' % nstars, fontsize=12)
    py.xlabel('RA from Sgr A* (arcsec)')
    py.ylabel('Dec from Sgr A* (arcsec)')
    py.savefig(root + 'plots/secondary_median_velErr_nearestNbr%s%s.png' % (nstars,suffix))
    py.close(2)

    # Plot velocity error on the mean
    py.figure(3)
    py.figure(figsize=(12,6))
    py.subplots_adjust(left=0.07,right=0.98,top=0.95,wspace=0.2)
    py.clf()
    py.subplot(1,2,1)
    py.imshow(eom_vx,interpolation='nearest',
              extent=[xbins[0],xbins[-1],ybins[0],ybins[-1]],
              origin='lowerleft',cmap=py.cm.gist_rainbow_r)
    py.plot([0],[0],'kx') # Sgr A*
    plot_dither_pattern()
    if (plotMsrs == True):
        py.plot(msrX,msrY,'kx',ms=8,mew=2)
    py.axis([np.ceil(xmax),np.floor(xmin),np.floor(ymin),np.ceil(ymax)])
    cbar = py.colorbar(shrink=.75)
    cbar.set_label('(mas/yr)')
    py.title('X Velocity Error on the Mean: %i Nearest Neighbors' % nstars, fontsize=12)
    py.xlabel('RA from Sgr A* (arcsec)')
    py.ylabel('Dec from Sgr A* (arcsec)')
    py.subplot(1,2,2)
    py.imshow(eom_vy,interpolation='nearest',
              extent=[xbins[0],xbins[-1],ybins[0],ybins[-1]],
              origin='lowerleft',cmap=py.cm.gist_rainbow_r)
    py.plot([0],[0],'kx') # Sgr A*
    plot_dither_pattern()
    if (plotMsrs == True):
        py.plot(msrX,msrY,'kx',ms=8,mew=2)
    py.axis([np.ceil(xmax),np.floor(xmin),np.floor(ymin),np.ceil(ymax)])
    cbar = py.colorbar(shrink=.75)
    cbar.set_label('(mas/yr)')
    py.title('Y Velocity Error on the Mean: %i Nearest Neighbors' % nstars, fontsize=12)
    py.xlabel('RA from Sgr A* (arcsec)')
    py.ylabel('Dec from Sgr A* (arcsec)')
    py.savefig(root + 'plots/secondary_velErrOnMean_nearestNbr%s%s.png' % (nstars,suffix))
    py.close(3)

    # Plot median reduced chi2
    py.figure(4)
    py.figure(figsize=(12,6))
    py.subplots_adjust(left=0.07,right=0.98,top=0.95,wspace=0.2)
    py.clf()
    py.subplot(1,2,1)
    py.imshow(med_xchi2r,interpolation='nearest',
              extent=[xbins[0],xbins[-1],ybins[0],ybins[-1]],
              origin='lowerleft',cmap=py.cm.gist_rainbow_r)
    py.plot([0],[0],'kx') # Sgr A*
    plot_dither_pattern()
    if (plotMsrs == True):
        py.plot(msrX,msrY,'kx',ms=8,mew=2)
    py.axis([np.ceil(xmax),np.floor(xmin),np.floor(ymin),np.ceil(ymax)])
    py.colorbar(shrink=.75)
    py.title('Median X Reduced Chi2: %i Nearest Neighbors' % nstars, fontsize=12)
    py.xlabel('RA from Sgr A* (arcsec)')
    py.ylabel('Dec from Sgr A* (arcsec)')
    py.subplot(1,2,2)
    py.imshow(med_ychi2r,interpolation='nearest',
              extent=[xbins[0],xbins[-1],ybins[0],ybins[-1]],
              origin='lowerleft',cmap=py.cm.gist_rainbow_r)
    py.plot([0],[0],'kx') # Sgr A*
    plot_dither_pattern()
    if (plotMsrs == True):
        py.plot(msrX,msrY,'kx',ms=8,mew=2)
    py.axis([np.ceil(xmax),np.floor(xmin),np.floor(ymin),np.ceil(ymax)])
    py.colorbar(shrink=.75)
    py.title('Median Y Reduced Chi2: %i Nearest Neighbors' % nstars, fontsize=12)
    py.xlabel('RA from Sgr A* (arcsec)')
    py.ylabel('Dec from Sgr A* (arcsec)')
    py.savefig(root + 'plots/secondary_median_chi2r_nearestNbr%s%s.png' % (nstars,suffix))
    py.close(4)

    # Plot mean velocities
    py.figure(5)
    py.figure(figsize=(12,6))
    py.subplots_adjust(left=0.07,right=0.98,top=0.95,wspace=0.2)
    py.clf()
    py.subplot(1,2,1)
    py.imshow(ave_vx,interpolation='nearest',vmin=-2.0,vmax=2.5,
              extent=[xbins[0],xbins[-1],ybins[0],ybins[-1]],
              origin='lowerleft',cmap=py.cm.gist_rainbow_r)
    py.plot([0],[0],'kx') # Sgr A*
    plot_dither_pattern()
    if (plotMsrs == True):
        py.plot(msrX,msrY,'kx',ms=8,mew=2)
    py.axis([np.ceil(xmax),np.floor(xmin),np.floor(ymin),np.ceil(ymax)])
    cbar = py.colorbar(shrink=.75)
    cbar.set_label('(mas/yr)')
    py.title('Mean X Velocity: %i Nearest Neighbors' % nstars, fontsize=14)
    py.xlabel('RA from Sgr A* (arcsec)',fontsize=14)
    py.ylabel('Dec from Sgr A* (arcsec)',fontsize=14)
    py.subplot(1,2,2)
    py.imshow(ave_vy,interpolation='nearest',vmin=-2.0,vmax=2.5,
              extent=[xbins[0],xbins[-1],ybins[0],ybins[-1]],
              origin='lowerleft',cmap=py.cm.gist_rainbow_r)
    py.plot([0],[0],'kx') # Sgr A*
    plot_dither_pattern()
    if (plotMsrs == True):
        py.plot(msrX,msrY,'kx',ms=8,mew=2)
    py.axis([np.ceil(xmax),np.floor(xmin),np.floor(ymin),np.ceil(ymax)])
    cbar = py.colorbar(shrink=.75)
    cbar.set_label('(mas/yr)')
    py.title('Mean Y Velocity: %i Nearest Neighbors' % nstars, fontsize=14)
    py.xlabel('RA from Sgr A* (arcsec)',fontsize=14)
    py.ylabel('Dec from Sgr A* (arcsec)',fontsize=14)
    py.savefig(root + 'plots/secondary_mean_vel_nearestNbr%s%s.png' % (nstars,suffix))
    py.close(5)

    py.figure(13)
    py.figure(figsize=(7,7))
    py.subplots_adjust(left=0.1,right=0.98,top=0.95,wspace=0.2)
    py.clf()
    qvr = py.quiver(xbins, ybins, ave_vx, ave_vy, color='black', angles='xy', \
                    units='y', scale=2.0)
    py.plot([0],[0],'ro')
    py.xlabel('RA from Sgr A* (arcsec)',fontsize=14)
    py.ylabel('Dec from Sgr A* (arcsec)',fontsize=14)
    py.title('Mean Velocity: %i Nearest Neighbors' % nstars, fontsize=14)
    py.axis([np.ceil(xmax),np.floor(xmin),np.floor(ymin),np.ceil(ymax)])
    py.savefig(root + 'plots/secondary_mean_vel_vectors_nearestNbr%s%s.png' % (nstars,suffix))
    py.close(13)

    py.figure(14)
    py.figure(figsize=(7,7))
    py.subplots_adjust(left=0.1,right=0.98,top=0.95,wspace=0.2)
    py.clf()
    qvr = py.quiver(xbins, ybins, med_vx, med_vy, color='black', angles='xy', \
                    units='y', scale=2.0)
    py.plot([0],[0],'ro')
    py.xlabel('RA from Sgr A* (arcsec)',fontsize=14)
    py.ylabel('Dec from Sgr A* (arcsec)',fontsize=14)
    py.title('Median Velocity: %i Nearest Neighbors' % nstars, fontsize=14)
    py.axis([np.ceil(xmax),np.floor(xmin),np.floor(ymin),np.ceil(ymax)])
    py.savefig(root + 'plots/secondary_median_vel_vectors_nearestNbr%s%s.png' % (nstars,suffix))
    py.close(14)
    

    # Histogram of the median velocities (flattened)
    py.figure(6)
    py.figure(figsize=(10,5))
    py.subplots_adjust(left=0.07,right=0.98,top=0.95,wspace=0.2)
    py.clf()
    binsIn = py.arange(-3, 3, 0.2)
    py.subplot(1,2,1)
    (nx,bx,ptx) = py.hist(med_vx.flatten(),binsIn,histtype='step',lw=1.5)
    py.text(-3,1000, '<vx> = %6.3f +- %5.3f mas/yr' % \
            ((med_vx.flatten()).mean(),(med_vx.flatten()).std(ddof=1)),fontsize=10)
    py.xlabel('Median X Velocity (mas/yr)')
    py.ylabel('N')
    py.title('%i Nearest Neighbors' % nstars, fontsize=12)
    py.subplot(1,2,2)
    (ny,by,pty) = py.hist(med_vy.flatten(),binsIn,histtype='step',lw=1.5)
    py.text(-3,1000, '<vy> = %6.3f +- %5.3f mas/yr' % \
            ((med_vy.flatten()).mean(),(med_vy.flatten()).std(ddof=1)),fontsize=10)
    py.xlabel('Median Y Velocity (mas/yr)')
    py.ylabel('N')
    py.title('%i Nearest Neighbors' % nstars, fontsize=12)
    py.savefig(root + 'plots/secondary_median_histVel_nearestNbr%s%s.png' % (nstars,suffix))
    py.close(6)
    
    # Histogram of the median velocities (flattened)
    py.figure(7)
    py.figure(figsize=(10,5))
    py.subplots_adjust(left=0.07,right=0.98,top=0.95,wspace=0.2)
    py.clf()
    binsIn = py.arange(-3, 3, 0.2)
    py.subplot(1,2,1)
    (nx,bx,ptx) = py.hist(ave_vx.flatten(),binsIn,histtype='step',lw=1.5)
    py.text(-3,1000, '<vx> = %6.3f +- %5.3f mas/yr' % \
            ((ave_vx.flatten()).mean(),(ave_vx.flatten()).std(ddof=1)),fontsize=10)
    py.xlabel('Mean X Velocity (mas/yr)')
    py.ylabel('N')
    py.title('%i Nearest Neighbors' % nstars, fontsize=12)
    py.subplot(1,2,2)
    (ny,by,pty) = py.hist(ave_vy.flatten(),binsIn,histtype='step',lw=1.5)
    py.text(-3,1000, '<vy> = %6.3f +- %5.3f mas/yr' % \
            ((ave_vy.flatten()).mean(),(ave_vy.flatten()).std(ddof=1)),fontsize=10)
    py.xlabel('Mean Y Velocity (mas/yr)')
    py.ylabel('N')
    py.title('%i Nearest Neighbors' % nstars, fontsize=12)
    py.savefig(root + 'plots/secondary_mean_histVel_nearestNbr%s%s.png' % (nstars,suffix))
    py.close(7)

    py.figure(8)
    py.figure(figsize=(7,7))
    py.clf()
    #py.plot(rbins, vdisp/np.sqrt(occ), 'k.')
    py.errorbar(rbins, vdisp, vdisperr, fmt='k.')
    py.xlabel('Radius (arcsec)')
    py.ylabel('Total Velocity Dispersion (mas/yr)')
    py.savefig(root + 'plots/secondary_vel_disp_rad%s.png' % suffix)
    py.close(8)

    # Plot histogram of median velocity / vel dispersion/sqrt(N)
    py.figure(9)
    py.figure(figsize=(10,5))
    py.subplots_adjust(left=0.07,right=0.98,top=0.95,wspace=0.2)
    py.clf()
    binsIn = py.arange(-10, 10, 0.2)
    #binsIn = py.arange(-3, 3, 0.2)
    py.subplot(1,2,1)
    (nx,bx,ptx) = py.hist(med_vx_over_rms,binsIn,histtype='step',lw=1.5)
    #py.text(-3,1000, '<vx> = %6.3f +- %5.3f mas/yr' % \
    #        ((ave_vx.flatten()).mean(),(ave_vx.flatten()).std(ddof=1)),fontsize=10)
    py.xlabel('Median X Velocity / Dispersion / Sqrt(N)')
    py.ylabel('N')
    py.title('%i Nearest Neighbors' % nstars, fontsize=12)
    py.subplot(1,2,2)
    (ny,by,pty) = py.hist(med_vy_over_rms,binsIn,histtype='step',lw=1.5)
    #py.text(-3,1000, '<vy> = %6.3f +- %5.3f mas/yr' % \
    #        ((ave_vy.flatten()).mean(),(ave_vy.flatten()).std(ddof=1)),fontsize=10)
    py.xlabel('Median Y Velocity / Dispersion / Sqrt(N)')
    py.ylabel('N')
    py.title('%i Nearest Neighbors' % nstars, fontsize=12)
    py.savefig(root + 'plots/secondary_medianVel_over_dispersion_nearestNbr%s%s.png' % (nstars,suffix))
    py.close(9)
    
    py.figure(10)
    py.figure(figsize=(7,7))
    py.subplots_adjust(left=0.1,right=0.98,top=0.95)
    py.clf()
    py.imshow(rmax,interpolation='nearest',
              extent=[xbins[0],xbins[-1],ybins[0],ybins[-1]],
              origin='lowerleft',cmap=py.cm.gist_rainbow_r)
    py.plot([0],[0],'kx') # Sgr A*
    plot_dither_pattern()
    if (plotMsrs == True):
        py.plot(msrX,msrY,'kx',ms=8,mew=2)
    py.axis([np.ceil(xmax),np.floor(xmin),np.floor(ymin),np.ceil(ymax)])
    py.colorbar(shrink=.75)
    py.title('Max Radius to Reach %i Nearest Neighbors' % nstars, fontsize=12)
    py.xlabel('RA from Sgr A* (arcsec)')
    py.ylabel('Dec from Sgr A* (arcsec)')
    py.savefig(root + 'plots/secondary_rmax_nearestNbr%s%s.png' % (nstars,suffix))
    py.close(10)

    # See how the RMS of the velocities changes with N nbrs
    if run_multi_nNbrs == True:
        nNbr_vx_rms = []
        nNbr_vy_rms = []
        med_vx_disp_1sig = []
        med_vy_disp_1sig = []

        nbr_array = [10, 25, 50, 75, 100, 150, 200, 250, 500]
        for ii in range(len(nbr_array)):
            nn = nbr_array[ii]
            print 'Building velocity maps for %d nearest neighbors' % nn
            med_vx, med_vy, rms_vx, rms_vy, rad = build_map(nn, writeFile=False)
            nNbr_vx_rms = np.concatenate([nNbr_vx_rms, [rms_vx]])
            nNbr_vy_rms = np.concatenate([nNbr_vy_rms, [rms_vy]])

            # Get the velocity dispersion as a function of radius
            rbins, vdisp, vdisperr, occ = vel_disp(x,y,r,vx,vy,vxerr,vyerr,fov)
            med_vx_over_rms, med_vy_over_rms = compute_ratio(med_vx,med_vy,rad,rbins,vdisp,occ)

            # Now find the width of these distributions
            med_vx_disp_1sig = np.concatenate([med_vx_disp_1sig, [med_vx_over_rms.std(ddof=1)]])
            med_vy_disp_1sig = np.concatenate([med_vy_disp_1sig, [med_vy_over_rms.std(ddof=1)]])

        # Plot the RMS of the velocities as a function of N nearest nbrs
        py.figure(11)
        py.figure(figsize=(8,6))
        py.subplots_adjust(left=0.15,right=0.95,top=0.9,wspace=0.2)
        py.clf()
        py.plot(nbr_array, nNbr_vx_rms, 'r-', label='x')
        py.plot(nbr_array, nNbr_vy_rms, 'b-', label='y')
        py.plot(nbr_array, 3.0/np.sqrt(nbr_array), 'k-', label='3/sqrt(N)')
        py.plot(nbr_array, nNbr_vx_rms / (3.0/np.sqrt(nbr_array)), 'r--', label='x ratio')
        py.plot(nbr_array, nNbr_vy_rms / (3.0/np.sqrt(nbr_array)), 'b--', label='y ratio')
        py.plot([50,50],[0,2.5],'k--') # vertical lines to guide the eye
        py.plot([100,100],[0,2.5],'k--')
        #py.axis([0,nbr_array[-1],0,1.5])
        py.xlabel('N Nearest Neighbors')
        py.ylabel('Velocity RMS (mas/yr)')
        py.title(absDir)
        py.legend(fancybox=True,numpoints=1)
        py.savefig(root + 'plots/rms_vel_N_nbrs%s.png' % suffix)
        py.close(11)

        # Plot the RMS of the ratio of median vel/dispersion as a function of N nearest nbrs
        py.figure(12)
        py.figure(figsize=(8,6))
        py.subplots_adjust(left=0.15,right=0.95,top=0.9,wspace=0.2)
        py.clf()
        py.plot(nbr_array, med_vx_disp_1sig, 'r-', label='x')
        py.plot(nbr_array, med_vy_disp_1sig, 'b-', label='y')
        py.xlabel('N Nearest Neighbors')
        py.ylabel('RMS of Median Velocity / Dispersion / Sqrt(N)')
        py.title(absDir)
        py.legend(fancybox=True,numpoints=1)
        py.savefig(root + 'plots/rms_vel_disp_N_nbrs%s.png' % suffix)
        py.close(12)


def vel_disp(x,y,r,vx,vy,vxerr,vyerr,fov):
    step = 1.0

    vrad = (vx*x+vy*y)/r
    vtan = (vx*y-vy*x)/r

    vraderr = np.sqrt((vxerr*x)**2+(vyerr*y)**2)/r
    vtanerr = np.sqrt((vxerr*y)**2+(vyerr*x)**2)/r

    # Calculate dispersions and errors
    rbins = np.arange(0, fov, step)
    vdisprad = np.zeros(len(rbins))
    vdisptan = np.zeros(len(rbins))
    vdisptot = np.zeros(len(rbins))
    vbiasrad2 = np.zeros(len(rbins))
    vbiastan2 = np.zeros(len(rbins))
    occupation = np.zeros(len(rbins))
    raderr =  np.zeros(len(rbins))
    tanerr =  np.zeros(len(rbins))
    toterr =  np.zeros(len(rbins))

    for ii in range(len(rbins)):
	idx = (np.where((r>rbins[ii]) & (r<rbins[ii]+step)))[0]
	occupation[ii] = len(idx)
	if len(idx) >= 2:  # set dispersion to zero if < 2 data points
	    # Calculate bias in dispersions from velocity errors
	    vbiasrad2[ii] = (vraderr[idx]**2).sum()/(2*(len(idx)))
	    vbiastan2[ii] = (vtanerr[idx]**2).sum()/(2*(len(idx)))
	    vdisprad[ii] = vrad[idx].std()
	    vdisptan[ii] = vtan[idx].std()
	    # If bias is larger than measured dispersion, set disp to zero
	    if vdisprad[ii] > np.sqrt(vbiasrad2[ii]):
		vdisprad[ii] = np.sqrt((vdisprad[ii]**2)-vbiasrad2[ii])
	    else:
		vdisprad[ii] = 0

	    if vdisptan[ii] > np.sqrt(vbiastan2[ii]):
		vdisptan[ii] = np.sqrt((vdisptan[ii]**2)-vbiastan2[ii])
	    else:
		vdisptan[ii] = 0

	    raderr[ii] = vdisprad[ii]/np.sqrt(2*len(idx)-1)
	    tanerr[ii] = vdisptan[ii]/np.sqrt(2*len(idx)-1)
	else:
	    vdisprad[ii] = 0
	    vdisptan[ii] = 0
	    raderr[ii] = 0
	    tanerr[ii] = 0

	# Calculate total dispersion
	vdisptot[ii] = np.sqrt(vdisprad[ii]**2+vdisptan[ii]**2)
	if vdisptot[ii] > 0:
	    toterr[ii] = np.sqrt((vdisprad[ii]/vdisptot[ii]*raderr[ii])**2+
				(vdisptan[ii]/vdisptot[ii]*tanerr[ii])**2)
	else:
	    toterr[ii] = 0

    # Adjust rbin values to center of bin
    rbins = rbins+step/2

    return rbins, vdisptot, toterr, occupation



def examineOutliers(root='/u/syelda/research/gc/absolute/10_06_22/',
                    align='align/align_abs_t', poly='polyfit_noDistErr/fit',
                    absFile='source_list/label_new.dat',excludeYng=False):
    """
    Look at the astrometric reference stars' velocities and
    quality of the velocity fits.  There were some stars with very low
    chi2 values in Figure 14 of Yelda et al (2010), indicating the
    positional uncertainties were relatively high. These could be giving
    rise to the net motion in +Y of the ref stars in the central 10 arcsec
    alignments.
    """

    s = starset.StarSet(root + align)
    s.loadPolyfit(root + poly, accel=0, arcsec=1)

    names = s.getArray('name')
    mag = s.getArray('mag')

    t0 = s.getArray('fitpXv.t0')
    velCnt = s.getArray('velCnt')
    numEpochs = velCnt[0]

    x = s.getArray('fitpXv.p') * -1.0
    xerr = s.getArray('fitpXv.perr') * 10**3 # mas
    vx = s.getArray('fitpXv.v') * 10**3 * -1.0
    vxerr = s.getArray('fitpXv.verr') * 10**3
    xchi2 = s.getArray('fitpXv.chi2')
    xchi2r = s.getArray('fitpXv.chi2red')

    y = s.getArray('fitpYv.p')
    yerr = s.getArray('fitpYv.perr') * 10**3 # mas
    vy = s.getArray('fitpYv.v') * 10**3
    vyerr = s.getArray('fitpYv.verr') * 10**3
    ychi2 = s.getArray('fitpYv.chi2')
    ychi2r = s.getArray('fitpYv.chi2red')

    # Make the same cuts we made for our final list of reference stars
    bad = np.where((np.abs(vx) > 10.) | (np.abs(vy) > 10.) | (vxerr > 1.5) | (vyerr > 1.5))[0]
    
    kp = np.setdiff1d(np.arange(len(x)), bad)
    names = [names[nn] for nn in kp]
    mag = mag[kp]
    t0 = t0[kp]
    velCnt = velCnt[kp]
    x = x[kp]
    y = y[kp]
    xerr = xerr[kp]
    yerr = yerr[kp]
    vx = vx[kp]
    vy = vy[kp]
    vxerr = vxerr[kp]
    vyerr = vyerr[kp]
    xchi2 = xchi2[kp]
    xchi2r = xchi2r[kp]
    ychi2 = ychi2[kp]
    ychi2r = ychi2r[kp]

    r2d = np.sqrt(x**2 + y**2)

    vtot = np.sqrt(vx**2 + vy**2)
    vtoterr = np.sqrt((vx*vxerr)**2 + (vy*vyerr)**2) / vtot

    all_chi2 = np.array([xchi2,ychi2]).flatten()
    allMsr_chi2 = np.array([xchi2[0:7],ychi2[0:7]]).flatten()

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

    # Expected chi2 for n dof:
    binsIn = py.arange(0, 15, 0.4)
    dof = numEpochs - 2
    xpctd6 = stats.chi2.pdf(binsIn,dof) # for stars in 6 epochs
    xpctd5 = stats.chi2.pdf(binsIn,3) # for stars in 5 epochs
    xpctd4 = stats.chi2.pdf(binsIn,2) # for stars in 4 epochs


    # Bin up the stars by magnitude, and find the RMS error
    # of the positional uncertainties and the velocity uncertainties
    # (this is a binning-up of the plot in Yelda+2010, Fig 15)
    magStep = 1.0
    magBins = np.arange(8.0, 16.0, magStep)
    xeMag = np.zeros(len(magBins), float)
    yeMag = np.zeros(len(magBins), float)
    vxeMag = np.zeros(len(magBins), float)
    vyeMag = np.zeros(len(magBins), float)
    xeMagRMS = np.zeros(len(magBins), float)
    yeMagRMS = np.zeros(len(magBins), float)
    vxeMagRMS = np.zeros(len(magBins), float)
    vyeMagRMS = np.zeros(len(magBins), float)

    for mm in range(len(magBins)):
        mMin = magBins[mm] - (magStep / 2.0)
        mMax = magBins[mm] + (magStep / 2.0)
        idx = np.where((mag >= mMin) & (mag < mMax))[0]

        if (len(idx) > 0):
            xeMag[mm] = np.median(xerr[idx])
            xeMagRMS[mm] = xerr[idx].std(ddof=1)

            yeMag[mm] = np.median(yerr[idx])
            yeMagRMS[mm] = yerr[idx].std(ddof=1)

            vxeMag[mm] = np.median(vxerr[idx])
            vxeMagRMS[mm] = vxerr[idx].std(ddof=1)

            vyeMag[mm] = np.median(vyerr[idx])
            vyeMagRMS[mm] = vyerr[idx].std(ddof=1)

            print '%4.1f < K < %4.1f: N=%i' % (mMin, mMax, len(idx))
            print 'x err + 1sigma = %4.2f mas' % (xeMag[mm]+xeMagRMS[mm])
            print 'y err + 1sigma = %4.2f mas' % (yeMag[mm]+yeMagRMS[mm])
            print 'vx err + 1sigma = %4.2f mas/yr' % (vxeMag[mm]+vxeMagRMS[mm])
            print 'vy err + 1sigma = %4.2f mas/yr' % (vyeMag[mm]+vyeMagRMS[mm])
            print ''
        
    # Print out the high velocity stars
    hiV = np.where((np.abs(vx) > 3.0) | (np.abs(vy) > 3.0))[0]

    py.figure(3)
    py.clf()
    py.figure(figsize=(6,6))
    if len(hiV) > 0:
        fmt = '%12s  %7.3f  %7.3f  %7.3f  %6.2f  %6.2f  %6.2f  %6.2f'
        hdr = '%12s  %7s  %7s  %7s  %6s  %6s  %6s  %6s'
        print 'Stars with V > 3 mas/yr:'
        print hdr % ('Name', 'x', 'y', 'r2d', 'Vx', 'Vx err', 'Vy', 'Vy err (mas/yr)')
        for ii in hiV:
            print fmt % (names[ii], x[ii], y[ii], r2d[ii], vx[ii], vxerr[ii], vy[ii], vyerr[ii])

            py.plot(x[ii],y[ii],'k.',ms=2.*np.abs(vy[ii]))
        py.axis([5,-5,-5,5])
        py.xlabel('X (arcsec)')
        py.ylabel('Y (arcsec)')
        py.title('Locations of V>3 mas/yr Sources (scaled by Vy)', fontsize=12)
        py.savefig(root + 'plots/hiVelocity_scaledByVy.png')

        # Look at these stars' velocity fits
        

    # Plots
    py.figure(1)
    py.figure(figsize=(7,7))
    py.clf()
    py.plot(mag, xerr, 'rx')
    py.plot(mag, yerr, 'b+')
    py.plot(magBins, xeMag+xeMagRMS, 'r--')
    py.plot(magBins, yeMag+yeMagRMS, 'b--')
    py.xlabel('K Magnitude')
    py.ylabel('Positional Uncertainty (mas)')
    py.axis([7,17,0,2.5])
    py.savefig(root + 'plots/posErr_mag.png')
    py.close(1)

    py.figure(2)
    py.figure(figsize=(7,7))
    py.clf()
    py.plot(mag, vxerr, 'rx')
    py.plot(mag, vyerr, 'b+')
    py.plot(magBins, vxeMag+vxeMagRMS, 'r--')
    py.plot(magBins, vyeMag+vyeMagRMS, 'b--')
    py.xlabel('K Magnitude')
    py.ylabel('Velocity Uncertainty (mas)')
    py.axis([7,17,0,1.6])
    py.savefig(root + 'plots/velErr_mag.png')
    py.close(2)


#    data = asciidata.open(root+absFile)
#    name = data[0]._data
#    mag = data[1].tonumpy()
#    x = data[2].tonumpy()
#    y = data[3].tonumpy()
#    xerr = data[4].tonumpy()
#    yerr = data[5].tonumpy()
#    vx = data[6].tonumpy()
#    vy = data[7].tonumpy()
#    vxerr = data[8].tonumpy()
#    vyerr = data[9].tonumpy()
#    r2d = np.hypot(x,y)
#    use = data[11].tonumpy()
#
#    # We only want the reference stars outside of 0.5"
#    ref = np.where((use != 0) & (r2d > 0.5))[0]
#    name = [name[ii] for ii in ref]
#    mag = mag[ref]       
#    x = x[ref]       
#    y = y[ref]       
#    xerr = xerr[ref]       
#    yerr = yerr[ref]       
#    vx = vx[ref]       
#    vy = vy[ref]       
#    vxerr = vxerr[ref]       
#    vyerr = vyerr[ref]       
#    r2d = r2d[ref]       
#    use = use[ref]       
#
#    if excludeYng == True:
#        # Do not include young stars 
#        young = youngNames.loadYoungStars()
#
#        for ii in range(len(x)):
#            # Exclude young stars
#            if name[ii] in young:
#                use[ii] = 0
#
#        idYng = np.where(use == 0)[0]
#        nYng = len(idYng)
#        print 'Number of young stars to be excluded: %s' % nYng
#
#        idLate = np.where(use == 1)[0]
#        print 'Number of non-known-young stars: %s' % len(idLate)
#
#        name = [name[ii] for ii in idLate]       
#        mag = mag[idLate]       
#        x = x[idLate]       
#        y = y[idLate]       
#        xerr = xerr[idLate]       
#        yerr = yerr[idLate]       
#        vx = vx[idLate]       
#        vy = vy[idLate]       
#        vxerr = vxerr[idLate]       
#        vyerr = vyerr[idLate]       
#        r2d = r2d[idLate]       
#        use = use[idLate]       
#
#    outV = np.where((vx > (vx.mean() + 2.*vx.std(ddof=1))) |
#                    (vx < (vx.mean() - 2.*vx.std(ddof=1))) |
#                    (vy > (vy.mean() + 2.*vy.std(ddof=1))) |
#                    (vy < (vy.mean() - 2.*vy.std(ddof=1))))[0]
#
#    fmt = '%12s  %10.3f         %10.3f'
#    hdr = '%12s  %7s (mas/yr)  %7s (mas/yr)' % ('Name','Vx','Vy')
#    print '%i outliers in velocity:' % len(outV)
#    print '%12s  %7s (mas/yr)  %7s (mas/yr)' % ('Name','Vx','Vy')
#
#    for ii in range(len(outV)):
#        print fmt % (name[ii], vx[ii], vy[ii])
#
#    outVerr = np.where((vxerr > (vxerr.mean() + 2.*vxerr.std(ddof=1))) |
#                       (vxerr < (vxerr.mean() - 2.*vxerr.std(ddof=1))) |
#                       (vyerr > (vyerr.mean() + 2.*vyerr.std(ddof=1))) |
#                       (vyerr < (vyerr.mean() - 2.*vyerr.std(ddof=1))))[0]
#
#    print ''
#    print '%i outliers in velocity errors:' % len(outVerr)
#    print '%12s  %7s (mas/yr)  %7s (mas/yr)' % ('Name','VxErr','VyErr')
#    for ii in range(len(outV)):
#        print fmt % (name[ii], vxerr[ii], vyerr[ii])



def compare_abs_lists(dir1='10_06_22',dir2='11_01_03',filename1='/lis/absolute_refs.dat',
                      filename2='/lis/absolute_refs.dat',suffix='_mosaic_img_vs_lis'):
    """
    Makes a delta map of velocities in two different
    absolute_refs.dat files.
    """
    root = '/u/syelda/research/gc/absolute/'
    f1 = asciidata.open(root + dir1 + filename1)
    n1 = f1[0].tonumpy()
    mag1 = f1[1].tonumpy()
    x1 = f1[2].tonumpy()
    y1 = f1[3].tonumpy()
    xe1 = f1[4].tonumpy()
    ye1 = f1[5].tonumpy()
    vx1 = f1[6].tonumpy()
    vy1 = f1[7].tonumpy()
    vxe1 = f1[8].tonumpy()
    vye1 = f1[9].tonumpy()
    t01 = f1[10].tonumpy()
    use1 = f1[11].tonumpy()
    r2d1 = f1[12].tonumpy()

    f2 = asciidata.open(root + dir2 + filename2)
    n2 = f2[0].tonumpy()
    mag2 = f2[1].tonumpy()
    x2 = f2[2].tonumpy()
    y2 = f2[3].tonumpy()
    xe2 = f2[4].tonumpy()
    ye2 = f2[5].tonumpy()
    vx2 = f2[6].tonumpy()
    vy2 = f2[7].tonumpy()
    vxe2 = f2[8].tonumpy()
    vye2 = f2[9].tonumpy()
    t02 = f2[10].tonumpy()
    use2 = f2[11].tonumpy()
    r2d2 = f2[12].tonumpy()

    # We only want the reference stars:
    idx1 = np.where(use1 == 1)[0]
    n1 = np.array([n1[nn] for nn in idx1])
    mag1 = mag1[idx1]
    x1 = x1[idx1]
    y1 = y1[idx1]
    xe1 = xe1[idx1]
    ye1 = ye1[idx1]
    vx1 = vx1[idx1]
    vy1 = vy1[idx1]
    vxe1 = vxe1[idx1]
    vye1 = vye1[idx1]
    t01 = t01[idx1]
    r2d1 = r2d1[idx1]
    idx2 = np.where(use2 == 1)[0]
    n2 = np.array([n2[nn] for nn in idx2])
    mag2 = mag2[idx2]
    x2 = x2[idx2]
    y2 = y2[idx2]
    xe2 = xe2[idx2]
    ye2 = ye2[idx2]
    vx2 = vx2[idx2]
    vy2 = vy2[idx2]
    vxe2 = vxe2[idx2]
    vye2 = vye2[idx2]
    t02 = t02[idx2]
    r2d2 = r2d2[idx2]
   
    # Note that the t0's may be different between the two lists, but
    # this shouldn't affect the velocity comparison so much, given
    # that most stars are not accelerating over the 5 year mosaic baseline

    # Have to match the stars first
    x_all = []
    y_all = []
    dvx_all = []
    dvy_all = []
    x1_m = []
    y1_m = []
    x2_m = []
    y2_m = []
    vx1_m = []
    vy1_m = []
    vx2_m = []
    vy2_m = []
    vxe1_m = []
    vye1_m = []
    vxe2_m = []
    vye2_m = []
    n1m = []
    n2m = []
    mag1m = []
    mag2m = []
    for ii in range(len(n1)):
        idx = np.where(n2 == n1[ii])[0]

        if len(idx) > 0:
            dvx = vx1[ii] - vx2[idx]
            dvy = vy1[ii] - vy2[idx]
            
            x_all = np.concatenate([x_all,[x1[ii]]]) # save the pos from first file
            y_all = np.concatenate([y_all,[y1[ii]]])
            dvx_all = np.concatenate([dvx_all,dvx])
            dvy_all = np.concatenate([dvy_all,dvy])

            n1m = np.concatenate([n1m,[n1[ii]]])
            mag1m = np.concatenate([mag1m,[mag1[ii]]])
            x1_m = np.concatenate([x1_m, [x1[ii]]])
            y1_m = np.concatenate([y1_m, [y1[ii]]])
            vx1_m = np.concatenate([vx1_m, [vx1[ii]]])
            vy1_m = np.concatenate([vy1_m, [vy1[ii]]])
            vxe1_m = np.concatenate([vxe1_m, [vxe1[ii]]])
            vye1_m = np.concatenate([vye1_m, [vye1[ii]]])

            n2m = np.concatenate([n2m,n2[idx]])
            mag2m = np.concatenate([mag2m,mag2[idx]])
            x2_m = np.concatenate([x2_m, x2[idx]])
            y2_m = np.concatenate([y2_m, y2[idx]])
            vx2_m = np.concatenate([vx2_m, vx2[idx]])
            vy2_m = np.concatenate([vy2_m, vy2[idx]])
            vxe2_m = np.concatenate([vxe2_m, vxe2[idx]])
            vye2_m = np.concatenate([vye2_m, vye2[idx]])

    # Identify 16C
    i16c1 = np.where(n1m == 'irs10W')[0]
    x16c1 = x1_m[i16c1]
    y16c1 = y1_m[i16c1]
    vxe16c1 = vxe1_m[i16c1]
    vye16c1 = vye1_m[i16c1]

    i16c2 = np.where(n2m == 'irs10W')[0]
    x16c2 = x2_m[i16c2]
    y16c2 = y2_m[i16c2]
    vxe16c2 = vxe2_m[i16c2]
    vye16c2 = vye2_m[i16c2]

    # Print out some info
    print 'Average differences in velocities:'
    print '<dvx> = %5.3f +- %5.3f mas/yr' % (dvx_all.mean(),dvx_all.std(ddof=1))
    print '<dvy> = %5.3f +- %5.3f mas/yr' % (dvy_all.mean(),dvy_all.std(ddof=1))
    print
    print 'Maximum velocity offset (Vx,Vy) = %5.3f, %5.3f mas/yr' % \
          (np.abs(dvx_all).max(),np.abs(dvy_all).max())
    print
    # Determine which are velocity outliers:
    ol = np.where((np.abs(dvx_all) > 3.0*dvx_all.std(ddof=1)) | (np.abs(dvy_all) > 3.0*dvy_all.std(ddof=1)))[0]
    if len(ol) > 0:
        print '3 sigma outliers:'
        hdr = '%10s  %8s    %8s %8s'
        print hdr % ('Name', 'dvx', 'dvy', '(mas/yr)')
        fmt = '%10s  %8.3f    %8.3f'
        for oo in ol:
            print fmt % (n1m[oo], dvx_all[oo], dvy_all[oo])
    print
    print 'NOTE: These may have been mis-matched in the latest absolute alignment.'
    print 'Check against Yelda et al. 2010 published velocities.'

    # Find K magnitude outliers
    print
    print 'Magnitude Outliers:'
    mol = np.where((np.abs(mag1m - mag2m) > 0.5))[0]
    if len(mol) > 0:
        hdr = '%10s  %6s  %6s'
        print hdr % ('Name', 'Mag1', 'Mag2')
        print '%8s  %10s  %10s' % ('', '('+dir1+')', '('+dir2+')')
        fmt = '%10s   %5.2f   %5.2f'
    for mm in mol:
        print fmt % (n1m[mm], mag1m[mm], mag2m[mm])
    
    prop = matplotlib.font_manager.FontProperties(size=10)

    py.figure(1)
    py.figure(figsize=(6,6))
    py.clf()
    qvr = py.quiver(x_all, y_all, -dvx_all, dvy_all, color='black', \
                    units='y', scale=2)
    # mark the masers with a different color
    py.plot(x_all[0:7], y_all[0:7], 'ro', mec='r', mfc='None')
    #py.plot(x1[irs16c], y1[irs16c], 'rx')
    py.quiverkey(qvr,12,10,-2,'2 mas/yr',coordinates='data',
                 color='red',fontproperties={'size': 'smaller'})
    py.axis([x_all.max(),x_all.min(),y_all.min(),y_all.max()])
    py.xlabel('X (arcsec)')
    py.ylabel('Y (arcsec)')
    py.title('Velocity Differences btwn %s & %s' % (dir1, dir2))
    #py.show()
    py.savefig(root + 'absolute_refs%s.png' % suffix)
    py.close(1)

    py.figure(2)
    py.figure(figsize=(6,6))
    py.clf()
    py.plot(vx1_m, vx2_m, 'r.', label='x')
    py.plot(vy1_m, vy2_m, 'b.', label='y')
    py.plot([-10,10],[-10,10],'k--')
    py.xlabel(dir1 + ' Velocity (mas/yr)')
    py.ylabel(dir2 + ' Velocity (mas/yr)')
    py.axis([-10,10,-10,10])
    py.legend(loc=2,numpoints=1,fancybox=True)
    py.savefig('%s/compare_labelVels_%s_%s.png' % (root, dir1, dir2))
    py.close(2)

    py.figure(3)
    py.figure(figsize=(6,6))
    py.clf()
    binsIn = py.arange(-11, 11, 0.25)
    (nx,bx,ptx) = py.hist(dvx_all,binsIn,color='r',histtype='step',label='x',lw=1.5)
    (ny,by,pty) = py.hist(dvy_all,binsIn,color='b',histtype='step',label='y',lw=1.5)
    py.xlabel('Velocity Offset (mas/yr)')
    py.ylabel('N')
    py.axis([min(dvx_all.min(),dvy_all.min())-0.5,max(dvx_all.max(),dvy_all.max())+0.5,0,max(nx.max(),ny.max())])
    py.legend(numpoints=1,fancybox=True)
    py.savefig('%s/compare_labelVels_hist_%s_%s.png' % (root, dir1, dir2))
    py.close(3)

    py.figure(4)
    py.figure(figsize=(6,6))
    py.clf()
    py.plot(mag1m, mag2m, 'k.')
    py.plot(mag1m[mol], mag2m[mol], 'rx',ms=8)
    py.plot([5,17],[5,17],'k--')
    py.axis([7,16,7,16])
    py.xlabel(dir1 + ' K Magnitude')
    py.ylabel(dir2 + ' K Magnitude')
    py.savefig('%s/compare_labelMags_%s_%s.png' % (root, dir1, dir2))
    py.close(4)

    py.figure(5)
    py.figure(figsize=(6,6))
    py.clf()
    py.plot(vxe1_m, vxe2_m, 'r.', label='x')
    py.plot(vye1_m, vye2_m, 'b.', label='y')
    py.plot([0,1.6],[0,1.6],'k--')
    py.xlabel(dir1 + ' Velocity Error (mas/yr)')
    py.ylabel(dir2 + ' Velocity Error(mas/yr)')
    py.axis([0,1.16,0,1.6])
    py.legend(loc=2,numpoints=1,fancybox=True)
    py.savefig('%s/compare_labelVelErrs_%s_%s.png' % (root, dir1, dir2))
    py.close(5)

    py.figure(6)
    py.figure(figsize=(6,6))
    py.clf()
    py.plot(np.hypot(x1_m,y1_m),np.hypot(vxe1_m,vye1_m), 'r.', label=dir1)
    py.plot(np.hypot(x2_m,y2_m),np.hypot(vxe2_m,vye2_m), 'b.', label=dir2)
    py.xlabel('Radius (arcsec)')
    py.ylabel('Total Velocity Error (mas/yr)')
    py.axis([0,20,0,2])
    py.legend(loc=4,numpoints=1,fancybox=True,prop=prop)
    py.savefig('%s/compare_label_VelErrVsR2d_%s_%s.png' % (root, dir1, dir2))
    py.close(6)

    from matplotlib.collections import EllipseCollection
    py.figure(7)
    py.figure(figsize=(10,5))
    py.clf()
    ax = py.subplot(1, 2, 1)
    scale = 0.02
    angles = np.zeros(len(x1_m))
    xy = np.column_stack((x1_m, y1_m))
    ec = EllipseCollection(vxe1_m*scale, vye1_m*scale, 
                           angles, units='width', offsets=xy,
                           transOffset=ax.transData, facecolors='none')
    ecRef = EllipseCollection(np.array([0.5])*scale, np.array([0.5])*scale, 
                              [0], units='width', offsets=[[12, 9.5]],
                              transOffset=ax.transData,
                              facecolors='none', edgecolors='red') 
    ax.add_collection(ec)
    ax.add_collection(ecRef)
    ax.autoscale_view()
    py.plot(x16c1,y16c1,'rx',ms=10)
    py.xlabel('X (arcsec)')
    py.ylabel('Y (arcsec)')
    py.title(dir1)
    py.text(12, 10, '0.5 mas/yr', color='red', 
            horizontalalignment='center', verticalalignment='bottom')
    py.axis([15.5, -6.5, -9, 12])
    ax = py.subplot(1, 2, 2)
    angles = np.zeros(len(x2_m))
    xy = np.column_stack((x2_m, y2_m))
    ec = EllipseCollection(vxe2_m*scale, vye2_m*scale, 
                           angles, units='width', offsets=xy,
                           transOffset=ax.transData, facecolors='none')
    ecRef = EllipseCollection(np.array([0.5])*scale, np.array([0.5])*scale, 
                              [0], units='width', offsets=[[12, 9.5]],
                              transOffset=ax.transData,
                              facecolors='none', edgecolors='red')
    ax.add_collection(ec)
    ax.add_collection(ecRef)
    ax.autoscale_view()
    py.plot(x16c2,y16c2,'rx',ms=10)
    py.xlabel('X (arcsec)')
    py.ylabel('Y (arcsec)')
    py.title(dir2)
    py.text(12, 10, '0.5 mas/yr', color='red', 
            horizontalalignment='center', verticalalignment='bottom')
    py.axis([15.5, -6.5, -9, 12])
    py.savefig('%s/compare_label_VelErrEllipse_%s_%s.png' % (root, dir1, dir2))
    py.close(7)

    # Pull the polyfit results from each of these analyses to compare chi2's
    p1 = starset.StarSet(dir1 + '/align/align_abs_t')
    p1.loadPolyfit(dir1 + '/polyfit_noDistErr/fit')
    name1 = p1.getArray('name')
    name1 = np.array([nn for nn in name1])
    mag1 = p1.getArray('mag')
    t01 = p1.getArray('fitpXv.t0')
    vx1 = p1.getArray('fitpXv.v') 
    vxerr1 = p1.getArray('fitpXv.verr')
    xchi21 = p1.getArray('fitpXv.chi2')
    xchi2r1 = p1.getArray('fitpXv.chi2red')
    vy1 = p1.getArray('fitpYv.v') 
    vyerr1 = p1.getArray('fitpYv.verr')
    ychi21 = p1.getArray('fitpYv.chi2')
    ychi2r1 = p1.getArray('fitpYv.chi2red')
    cnt1 = p1.getArray('velCnt')

    p2 = starset.StarSet(dir2 + '/align/align_abs_t')
    p2.loadPolyfit(dir2 + '/polyfit_noDistErr/fit')
    name2 = p2.getArray('name')
    name2 = np.array([nn for nn in name2])
    mag2 = p2.getArray('mag')
    t02 = p2.getArray('fitpXv.t0')
    vx2 = p2.getArray('fitpXv.v') 
    vxerr2 = p2.getArray('fitpXv.verr')
    xchi22 = p2.getArray('fitpXv.chi2')
    xchi2r2 = p2.getArray('fitpXv.chi2red')
    vy2 = p2.getArray('fitpYv.v') 
    vyerr2 = p2.getArray('fitpYv.verr')
    ychi22 = p2.getArray('fitpYv.chi2')
    ychi2r2 = p2.getArray('fitpYv.chi2red')
    cnt2 = p2.getArray('velCnt')

    # Match the stars by name
    n1m = []
    mag1m = []
    n2m = []
    mag2m = []
    dvx_all = []
    dvy_all = []
    dvx_all4 = []
    dvy_all4 = []
    dvx_all5 = []
    dvy_all5 = []
    dvx_all6 = []
    dvy_all6 = []
    xchi2r_all = []
    ychi2r_all = []
    xchi2_4 = []
    ychi2_4 = []
    xchi2r_4 = []
    ychi2r_4 = []
    xchi2_5 = []
    ychi2_5 = []
    xchi2r_5 = []
    ychi2r_5 = []
    xchi2_6 = []
    ychi2_6 = []
    xchi2r_6 = []
    ychi2r_6 = []
    cnt1m = []
    cnt2m = []
    for ii in range(len(name1)):
        idx = np.where(name2 == name1[ii])[0]

        if len(idx) > 0:
            n1m = np.concatenate([n1m,[name1[ii]]])
            mag1m = np.concatenate([mag1m,[mag1[ii]]])
            cnt1m = np.concatenate([cnt1m,[cnt1[ii]]])
            
            n2m = np.concatenate([n2m,name2[idx]])
            mag2m = np.concatenate([mag2m,mag2[idx]])
            cnt2m = np.concatenate([cnt2m,cnt2[idx]])
            dvx = np.abs(vx1[ii] - vx2[idx])
            dvx_sig = dvx / np.sqrt(vxerr1[ii]**2 + vxerr2[idx]**2)
            dvx_all = np.concatenate([dvx_all, dvx_sig])
            dvy = np.abs(vy1[ii] - vy2[idx])
            dvy_sig = dvy / np.sqrt(vyerr1[ii]**2 + vyerr2[idx]**2)
            dvy_all = np.concatenate([dvy_all, dvy_sig])
            xchi2r_all = np.concatenate([xchi2r_all, xchi2r2[idx]])
            ychi2r_all = np.concatenate([ychi2r_all, ychi2r2[idx]])
   
            # Separate out by degrees of freedom
            if (cnt1[ii] == 4) & (cnt2[idx] == 4):
                # Compare the velocities
                dvx4 = np.abs(vx1[ii] - vx2[idx])
                dvx_sig4 = dvx4 / np.sqrt(vxerr1[ii]**2 + vxerr2[idx]**2)
                dvx_all4 = np.concatenate([dvx_all4, dvx_sig4])
                dvy4 = np.abs(vy1[ii] - vy2[idx])
                dvy_sig4 = dvy4 / np.sqrt(vyerr1[ii]**2 + vyerr2[idx]**2)
                dvy_all4 = np.concatenate([dvy_all4, dvy_sig4])
                xchi2_4 = np.concatenate([xchi2_4, xchi22[idx]]) # get chi2 from 2nd align (star lists)
                ychi2_4 = np.concatenate([ychi2_4, ychi22[idx]]) 
                xchi2r_4 = np.concatenate([xchi2r_4, xchi2r2[idx]])
                ychi2r_4 = np.concatenate([ychi2r_4, ychi2r2[idx]])

            if (cnt1[ii] == 5) & (cnt2[idx] == 5):
                # Compare the velocities
                dvx5 = np.abs(vx1[ii] - vx2[idx])
                dvx_sig5 = dvx5 / np.sqrt(vxerr1[ii]**2 + vxerr2[idx]**2)
                dvx_all5 = np.concatenate([dvx_all5, dvx_sig5])
                dvy5 = np.abs(vy1[ii] - vy2[idx])
                dvy_sig5 = dvy5 / np.sqrt(vyerr1[ii]**2 + vyerr2[idx]**2)
                dvy_all5 = np.concatenate([dvy_all5, dvy_sig5])
                xchi2_5 = np.concatenate([xchi2_5, xchi22[idx]]) # get chi2 from 2nd align (star lists)
                ychi2_5 = np.concatenate([ychi2_5, ychi22[idx]]) 
                xchi2r_5 = np.concatenate([xchi2r_5, xchi2r2[idx]])
                ychi2r_5 = np.concatenate([ychi2r_5, ychi2r2[idx]])

            if (cnt1[ii] == 6) & (cnt2[idx] == 6):
                # Compare the velocities
                dvx6 = np.abs(vx1[ii] - vx2[idx])
                dvx_sig6 = dvx6 / np.sqrt(vxerr1[ii]**2 + vxerr2[idx]**2)
                dvx_all6 = np.concatenate([dvx_all6, dvx_sig6])
                dvy6 = np.abs(vy1[ii] - vy2[idx])
                dvy_sig6 = dvy6 / np.sqrt(vyerr1[ii]**2 + vyerr2[idx]**2)
                dvy_all6 = np.concatenate([dvy_all6, dvy_sig6])
                xchi2_6 = np.concatenate([xchi2_6, xchi22[idx]]) # get chi2 from 2nd align (star lists)
                ychi2_6 = np.concatenate([ychi2_6, ychi22[idx]]) 
                xchi2r_6 = np.concatenate([xchi2r_6, xchi2r2[idx]])
                ychi2r_6 = np.concatenate([ychi2r_6, ychi2r2[idx]])
 

    # Plot up the velocity delta (units of sigma) vs. chi2 (as found in mosaicked star list)
    # for stars in 4, 5, and 6 epochs
    print
    print 'Found %i stars in common between the two aligns' % len(n2m)
    print 'Number of stars in 4, 5, 6 epochs: %i, %i, %i' % (len(xchi2_4), len(xchi2_5), len(xchi2_6))
    py.figure(8)
    py.figure(figsize=(10,4))
    py.clf()
    py.subplots_adjust(left=0.06,bottom=0.15,right=0.98,top=0.9,
                       wspace=0.25,hspace=0.2)
    py.subplot(1,3,1)
    py.plot(xchi2_4,dvx_all4,'r.',label='X')
    py.plot(ychi2_4,dvy_all4,'b.',label='Y')
    py.xlabel('Chi Squared')
    py.ylabel('Velocity Difference (sigma)')
    py.title('4 Epochs')
    py.legend(loc=1,numpoints=1,fancybox=True)
    py.subplot(1,3,2)
    py.plot(xchi2_5,dvx_all5,'r.')
    py.plot(ychi2_5,dvy_all5,'b.')
    py.xlabel('Chi Squared')
    py.title('5 Epochs')
    py.subplot(1,3,3)
    py.plot(xchi2_6,dvx_all6,'r.')
    py.plot(ychi2_6,dvy_all6,'b.')
    py.xlabel('Chi Squared')
    py.title('6 Epochs')
    py.savefig('%s/compare_label_VelChi2_%s_%s.png' % (root, dir1, dir2))
    py.close(8)

    py.figure(9)
    py.figure(figsize=(10,4))
    py.clf()
    py.subplots_adjust(left=0.06,bottom=0.15,right=0.98,top=0.9,
                       wspace=0.25,hspace=0.2)
    py.subplot(1,3,1)
    py.plot(xchi2r_4,dvx_all4,'r.',label='X')
    py.plot(ychi2r_4,dvy_all4,'b.',label='Y')
    py.xlabel('Reduced Chi Squared')
    py.ylabel('Velocity Difference (sigma)')
    py.title('4 Epochs')
    py.legend(loc=1,numpoints=1,fancybox=True)
    py.subplot(1,3,2)
    py.plot(xchi2r_5,dvx_all5,'r.')
    py.plot(ychi2r_5,dvy_all5,'b.')
    py.xlabel('Reduced Chi Squared')
    py.title('5 Epochs')
    py.subplot(1,3,3)
    py.plot(xchi2r_6,dvx_all6,'r.')
    py.plot(ychi2r_6,dvy_all6,'b.')
    py.xlabel('Reduced Chi Squared')
    py.title('6 Epochs')
    py.savefig('%s/compare_label_VelChi2Red_DOF_%s_%s.png' % (root, dir1, dir2))
    py.close(9)

    py.figure(10)
    py.figure(figsize=(6,6))
    py.clf()
    py.plot(xchi2r_all,dvx_all,'r.')
    py.plot(ychi2r_all,dvy_all,'b.')
    py.axis([0,10,0,6])
    py.xlabel('Reduced Chi Squared')
    py.ylabel('Velocity Difference (sigma)')
    py.savefig('%s/compare_label_VelChi2Red_%s_%s.png' % (root, dir1, dir2))
    py.close(10)

    cnt_diff = cnt1m - cnt2m
    # Which stars had different number of detections in the 2 analyses?
    cdx = np.where(cnt_diff != 0)[0]
    print 'Number of stars with different number of counts in the 2 analyses: %i' % len(cdx)
    hdr = '%10s   %5s   %5s  %4s  %4s'
    fmt = '%10s   %5.2f   %5.2f  %4i  %4i'
    print hdr % ('Name','Mag1','Mag2','Cnt1','Cnt2')
    for cc in cdx:
        print fmt % (n1m[cc], mag1m[cc], mag2m[cc], cnt1m[cc], cnt2m[cc])
    py.figure(11)
    py.figure(figsize=(6,6))
    py.clf()
    py.plot(mag1m[cdx],cnt_diff[cdx],'k.')
    py.plot([10,16],[0,0],'k--')
    py.text(10.2,0.5,'More matches in %s' % dir1,fontsize=10)
    py.text(10.2,-0.5,'More matches in %s' % dir2,fontsize=10)
    py.axis([10,16,-2,3])
    py.xlabel('K Magnitude')
    py.ylabel('Difference in Counts')
    py.savefig('%s/compare_label_diffCountsMag_%s_%s.png' % (root, dir1, dir2))
    py.close(11)

    py.figure(12)
    py.figure(figsize=(6,6))
    py.clf()
    py.plot(mag1m, xchi2r_all,'r.')
    py.plot(mag1m, ychi2r_all,'b.')
    #py.axis([7,16,0,10])
    py.xlabel('K Magnitude')
    py.ylabel('Reduced Chi Squared')
    py.savefig('%s/compare_label_Chi2RedMag_%s_%s.png' % (root, dir1, dir2))
    py.close(12)


def plotPosErr(dateDir='10_06_22',numEpochs=6,radius=20.0,magCutOff=16.0,
               subtractDist=False):
    """
    Plots positional error vs. K mag for all the maser
    mosaics, and marks the masers.
    """

    root = '/u/syelda/research/gc/absolute/' + dateDir
    epochID = ['05junlgs','06maylgs1','07auglgs','08maylgs1','09junlgs','10maylgs']

    scale = 0.00995

    fig = py.figure(1, figsize=(10,8))
    fig.subplots_adjust(left=0.1,right=0.98,top=0.95,
                    wspace=0.25,hspace=0.25)
    fig.clf()
    fig = py.figure(2, figsize=(10,8))
    fig.subplots_adjust(left=0.1,right=0.98,top=0.95,
                    wspace=0.25,hspace=0.25)
    fig.clf()
    for ii in range(numEpochs):
        # read in positional errors
        lisFile = 'mag%s_msr_kp_rms_msr.lis' % epochID[ii]
        lis = asciidata.open('%s/lis/%s' % (root, lisFile))
        name = lis[0].tonumpy()
        mag = lis[1].tonumpy()
        x = lis[3].tonumpy()
        y = lis[4].tonumpy()
        xerr = lis[5].tonumpy()
        yerr = lis[6].tonumpy()

        # Convert into arsec offset from field center
        # We determine the field center by assuming that stars
        # are detected all the way out the edge.
        xhalf = x.max() / 2.0
        yhalf = y.max() / 2.0
        x = (x - xhalf) * scale
        y = (y - yhalf) * scale
        xerr *= scale * 1000.0 # mas
        yerr *= scale * 1000.0 # mas
 
        if subtractDist == True:
            # We should remove distortion error term from errors
            # in /absolute/11_01_03/ lis files
            dx2 = 1.0**2 + 0.5**2 # mas (squared)
            dy2 = 1.0**2 + 0.4**2 # mas (squared)

            xerr = np.sqrt(xerr**2 - dx2)
            yerr = np.sqrt(yerr**2 - dy2)

        r = np.hypot(x, y)
        err = (xerr + yerr) / 2.0


        ##########
        # Compute errors in magnitude bins
        ########## 
        magStep = 1.0
        magBins = np.arange(10.0, 20.0, magStep)
        errMag = np.zeros(len(magBins), float)
        for mm in range(len(magBins)):
            mMin = magBins[mm] - (magStep / 2.0)
            mMax = magBins[mm] + (magStep / 2.0)
            idx = (np.where((mag >= mMin) & (mag < mMax) & (r < radius)))[0]

            if (len(idx) > 0):
                errMag[mm] = np.median(err[idx])
        
        idx = (np.where((mag < magCutOff) & (r < radius)))[0]
        errMedian = np.median(err[idx])

        py.figure(1)
        py.subplot(2,3,ii+1)
        idx = (np.where(r < radius))[0]
        py.semilogy(mag[0:7], err[0:7], 'rx', ms=7)
        py.semilogy(mag[idx], err[idx], 'k.')
        py.semilogy(magBins, errMag, 'g.-')
        py.axis([8, 17, 5e-2, 20.0])
        if (ii == 0 | ii == 3):
            py.ylabel('Positional Uncertainty (mas)', fontsize=14)
        if (ii == 4):
            py.xlabel('K Magnitude for r < %4.1f"' % radius, fontsize=14)
        py.title(epochID[ii], fontsize=12)

        py.text(9.,0.1, 'Median err for K < %4.1f:' % magCutOff, color='green', fontsize=10)
        py.text(11, 0.07, '%4.2f mas' % errMedian, color='green', fontsize=10)


        # For stars with K < magCutOff, plot their pos errors vs. radius
        idx = (np.where((mag < magCutOff) & (r < radius)))[0]
        py.figure(2)
        py.subplot(2,3,ii+1)
        py.semilogy(r[idx],err[idx],'k.')
        if (ii == 0 | ii == 3):
            py.ylabel('Positional Uncertainty (mas)', fontsize=14)
        if (ii == 4):
            py.xlabel('Radius for K < %4.1f' % magCutOff, fontsize=14)
        py.title(epochID[ii], fontsize=12)


    py.figure(1)
    py.savefig('%s/plots/plotPosErr_allMosaics_%s.png' % (root, dateDir))
    py.close(1)

    py.figure(2)
    py.savefig('%s/plots/plotPosErr_radius_allMosaics_%s.png' % (root, dateDir))
    py.close(2)


def plotPosErr_dithers(absDir='13_07_24/'):
    """
    Plot positional errors by dither position
    """

    root = '/u/syelda/research/gc/absolute/' + absDir
    table = asciidata.open(root + '/scripts/epochsInfo.txt')

    # List of columns in the table. Make and array for each one.
    epoch = [table[0][ss].strip() for ss in range(table.nrows)]
    ep = [epoch[ss].split('_')[0] for ss in range(len(epoch))]
    dirs = [table[1][ss].strip() for ss in range(table.nrows)]

    #fld = ['C', 'E', 'W', 'N', 'S', 'NE', 'SE', 'NW', 'SW']
    fld = ['NE','N','NW','E','C','W','SE','S','SW']
    clr = ['black','orange', 'red', 'turquoise', 'purple', 'lightgreen', 'navy',
           'green', 'mediumorchid']#, 'cyan', 'magenta', 'purple', 'mediumorchid',
           #'deeppink', 'tomato', 'salmon', 'lightgreen', 'sienna']

    itime = 0.181
    scale = 0.00995
    # Number of coadds from each epoch
    # Note: 2011 had 2 different setups, one with 60 and one with 10 coadds
    # So I'm taking the average and using 35 for this calculation
    coadds = [60, 60, 60, 60, 60, 10, 35, 60, 120]
    expTime = np.zeros(len(ep), dtype=int)
    errMedian = np.zeros((len(ep), len(fld)), dtype=float)
    aveErr = np.zeros(len(ep), dtype=float)

    py.close('all')
    py.clf()
    py.figure(figsize=(10,10))
    py.subplots_adjust(left=0.1,right=0.85,top=0.95,bottom=0.1,
                       wspace=0.3,hspace=0.3)

    for ii in range(len(epoch)):
        for jj in range(len(fld)):
            lisFile = '%s/combo/starfinder/mag%s_msr_%s_kp_rms.lis' % (dirs[ii], ep[ii], fld[jj])
            lis = asciidata.open(lisFile)
            name = lis[0].tonumpy()
            mag = lis[1].tonumpy()
            x = lis[3].tonumpy()
            y = lis[4].tonumpy()
            xerr = lis[5].tonumpy()
            yerr = lis[6].tonumpy()
            numstars = len(xerr)
            images = asciidata.open('%s/combo/mag%s_msr_%s_kp.log' % (dirs[ii], ep[ii], fld[jj]))
            frames = images[0].tonumpy()
            numframes = len(frames)
            expTime[ii] = numframes * itime * coadds[ii]

            # Convert into arsec offset from field center
            # We determine the field center by assuming that stars
            # are detected all the way out the edge.
            xerr *= scale * 1000.0 # mas
            yerr *= scale * 1000.0 # mas
            err = (xerr + yerr) / 2.0

            ##########
            # Compute errors in magnitude bins
            ########## 
            magStep = 1.0
            magBins = np.arange(10.0, 20.0, magStep)
            errMag = np.zeros(len(magBins), float)
            for mm in range(len(magBins)):
                mMin = magBins[mm] - (magStep / 2.0)
                mMax = magBins[mm] + (magStep / 2.0)
                idx = np.where((mag >= mMin) & (mag < mMax))[0]
    
                if (len(idx) > 0):
                    errMag[mm] = np.median(err[idx])
            
            idx = np.where(mag < 14)[0]
            errMedian[ii,jj] = np.median(err[idx])
                
            py.subplot(3,3,jj+1)
            py.semilogy(magBins, errMag, color=clr[ii],label='20%s (%3i s)' % (str(ep[ii][0:2]),expTime[ii]))
            thePlot = py.gca()
            thePlot.get_xaxis().set_major_locator(py.MultipleLocator(2))
            py.axis([9,18,0.05,10])
            if fld[jj] == 'S':
                py.xlabel('K Magnitude')
            if fld[jj] == 'E':
                py.ylabel('Positional Error (mas)')
            py.title(fld[jj])

        # Compute average over all 9 dither positions of the pos errors for each epoch
        aveErr[ii] = errMedian[ii,:].mean()

    import matplotlib.font_manager
    prop = matplotlib.font_manager.FontProperties(size=10)
    py.legend(bbox_to_anchor=(1.73,2.25), prop=prop,fancybox=True)
    py.savefig('%s/plots/posErr_dithers.png' % root)
    py.close()

    py.clf()
    py.figure(2)
    py.figure(figsize=(6,6))
    py.subplots_adjust(left=0.15,right=0.95,top=0.95,bottom=0.12)
    py.semilogy(expTime, aveErr, 'k.')
    py.xlabel('Total Exposure Time (s)')
    py.ylabel('Average Pos Error for K<14 (mas)')
    py.axis([0,360,0.1,1.0])
    py.savefig('%s/plots/posErr_expTime.png' % root)
    py.close()


def check_masers_vel_chi2(absDir='13_07_24/',epochsCut=4,noJknife=False):
    """
    Check that the maser velocities that we derive in the
    radio reference frame have sensible chi^2 values.
    """
    if (noJknife):
        suffix = '_nojknife'
    else:
        suffix = ''

    root = '/u/syelda/research/gc/absolute/' + absDir

    s = starset.StarSet(root + '/align/align_abs_t')
    s.loadPolyfit(root + '/polyfit_noDistErr/fit', arcsec=0)

    names = s.getArray('name')
    mag = s.getArray('mag')

    # Polyfit Fit in Pixels
    t0 = s.getArray('fitpXv.t0')
    velCnt = s.getArray('velCnt')
    numEpochs = velCnt[0]

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

    sampleFactor = np.sqrt(len(vx)/(len(vx)-1.))

    all_chi2 = np.array([xchi2,ychi2]).flatten()
    allMsr_chi2 = np.array([xchi2[0:7],ychi2[0:7]]).flatten()

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

    # Expected chi2 for n dof:
    binsInX = py.arange(0, 25, 0.1) # to calculate theoretical curve
    binsIn = py.arange(0, 25, 0.75) # for binning the actual data
    dof = numEpochs - 2

    usetexTrue()

    # Be smarter about the way the chi2 distributions are created
    numRows = np.ceil((np.sqrt(numEpochs - epochsCut + 1)))
    py.clf()
    py.figure(figsize=(10,10))
    py.subplots_adjust(left=0.1,bottom=0.1,right=0.98,top=0.95,
                       wspace=0.3,hspace=0.35)
    for ii in range(numEpochs - epochsCut + 1):
        idx = np.where(velCnt == (numEpochs-ii))[0]
        dof = numEpochs - ii - 2
        xpctd = stats.chi2.pdf(binsInX,dof)
        
        py.subplot(numRows,numRows,ii+1)
        py.hist(xchi2[idx],binsIn,color='r',lw=2,histtype='step',normed=True,label='X')
        py.hist(ychi2[idx],binsIn,color='b',lw=2,ls='dashed',histtype='step',normed=True,label='Y')
        py.plot(binsInX,xpctd,'k-',label='Expected',lw=1)
        py.title('%i epochs (N=%i stars)' % (numEpochs-ii, len(idx)),fontsize=12)
        py.xlabel(r'$\chi^2$')
        py.axis([0,25,0,0.3])
        print 'Number of stars in %i epochs: %i' % ((numEpochs-ii), len(idx))

    py.savefig(root + 'plots/abs_plots_allStars_chi2hists.png')
    usetexFalse()


def changeUseLabel(inputLabel, outputLabel):
    """
    Reads in a label.dat and changes the use column of
    reference stars from '1' to '2,8'
    """
    
    labels = starTables.Labels(labelFile=inputLabel)

    # Now lets write the output
    _out = open(outputLabel, 'w')
    _out.write('%-10s  %5s   ' % ('#Name', 'K'))
    _out.write('%7s %7s %7s %7s   ' % ('x', 'y', 'xerr', 'yerr'))
    _out.write('%8s %8s %8s %8s   ' % ('vx', 'vy', 'vxerr', 'vyerr'))
    _out.write('%8s %4s %7s\n' %  ('t0', 'use?', 'r2d'))

    _out.write('%-10s  %5s   ' % ('#()', '(mag)'))
    _out.write('%7s %7s %7s %7s   ' % \
               ('(asec)', '(asec)', '(asec)', '(asec)'))
    _out.write('%8s %8s %8s %8s   ' % \
               ('(mas/yr)', '(mas/yr)', '(mas/yr)', '(mas/yr)'))
    _out.write('%8s %4s %7s\n' %  ('(year)', '()', '(asec)'))

    for i in range(len(labels.name)):

        if (labels.useToAlign[i] == 1):
            # Speckle and AO
            use = '2,8'
        else:
            use = 0 

	_out.write('%-10s  %5.1f   ' % (labels.name[i], labels.mag[i]))
	_out.write('%9.5f %9.5f ' % (labels.x[i], labels.y[i]))
        _out.write('%9.5f %9.5f   ' % (labels.xerr[i], labels.yerr[i]))
	_out.write('%8.3f %8.3f ' % (labels.vx[i], labels.vy[i]))
        _out.write('%8.3f %8.3f   ' % (labels.vxerr[i], labels.vyerr[i]))
	_out.write('%8.3f %4s %7.3f\n' %  \
                   (labels.t0[i], use, labels.r[i]))


    _out.close()


def removeAcceleratingRefStars(absDir, alnDir, oldLabel='label_restrict.dat',
                               newLabel='label_restrict_noAccel.dat',
                               sigma=5.0,pvalue=5.0):
    """
    Finds accelerating stars (from a full alignment of speckle+AO narrow
    field data) that are also reference stars in label.dat and absolute_refs.dat.
    They are removed as reference stars, but kept in these files for naming
    purposes. This code should be run after producing a new absolute_refs.dat
    from a new IR-Radio absolute analysis.

    A new absolute_refs.dat file is also made, without the accelerating stars.
    The file is saved to absolute_refs_noAccel.dat.

    Inputs:
       absDir (str):    Full path to where the new absolute_refs.dat and label.dat are
       		        (e.g., '/u/syelda/research/gc/absolute/13_07_24/new_label_output/')
       alnDir (str):    Full path to where the speckle+AO alignment is
                        located (e.g., '/g/ghez/align/13_08_08/')
       oldLabel (str):  The label.dat file name (in absDir) that will be copied to newLabel,
       			but without the accelerating sources.
       newLabel (str):  The new label.dat file that will be the same as oldLabel, but without
       			accelerating reference stars.
       sigma (float):   The acceleration significance level above which stars are removed as
       			reference stars.
    """

    # The label.dat file
    labels = starTables.Labels(labelFile=absDir+oldLabel)

    dateDir = alnDir.split('/')[-2] + '/'
    aRoot = 'align/align_d_rms_1000_abs_t'
    poly = 'polyfit_c/fit'
    points = 'points_c/'

    # Read in the full alignment and get the accelerating stars
    s = starset.StarSet(alnDir + aRoot)
    s.loadPolyfit(alnDir + poly, accel=0, arcsec=1)
    s.loadPolyfit(alnDir + poly, accel=1, arcsec=1)

    names = s.getArray('name')
    r2d = s.getArray('r2d')
    mag = s.getArray('mag')
    x0 = s.getArray('fitXa.p')
    y0 = s.getArray('fitYa.p')
    x0e = s.getArray('fitXa.perr')
    y0e = s.getArray('fitYa.perr')
    ax = s.getArray('fitXa.a')
    ay = s.getArray('fitYa.a')
    axe = s.getArray('fitXa.aerr')
    aye = s.getArray('fitYa.aerr')
    numEps = s.getArray('velCnt')
    vxchi2r = s.getArray('fitXv.chi2red')
    vychi2r = s.getArray('fitYv.chi2red')

    # Convert accels to radial and tangential
    (ar, at, are, ate) = util.xy2circErr(x0, y0, ax, ay,
                                         x0e, y0e, axe, aye)

    # Make sure the masers and some irs16 sources are kept
    keepers = ['irs9', 'irs7', 'irs12N', 'irs28', 'irs10EE', 'irs15NE', 'irs17',
               'irs16C', 'irs16NW']
    
    # Cludge - treat the tangential accel as a source of noise, and
    # add it to the radial accel *noise*.
    #noise = np.hypot(at, are)
    #arsig_t = ar / noise

    # Accelerations in sigma
    arsig = ar / are
    atsig = at / ate

    # Find the following stars and remove them if they are reference stars:
    #   1. passed the F test for acceleration (in x or y direction)
    #   2. significantly accelerating in either radial or tangential direction

    # Run the F-test for accel vs. velocity
    passFnames, xfp, yfp = syAccel.run_f_test(names,pvalue,dateDir,aRoot,poly,points,
                            returnAcc=True,verbose=True)

   
    # Find the significantly accelerating stars, whether negative or positive,
    # or radial or tangential. They cannot be reference stars b/c align.java
    # only allows for linearly-moving reference stars.
    sig = np.where((np.abs(arsig) > sigma) | (np.abs(atsig) > sigma))[0]
    sig = [np.int(ss) for ss in sig]
    accel = np.array([names[ss] for ss in sig])


    print 'Found %i accelerating stars' % len(sig)
    print 'Found %i stars that passed the F test' % len(passFnames)
    print

    # Now lets write the output to a new label file, setting the above stars to use=0
    _out = open(absDir+newLabel, 'w')
    _out.write('%-10s  %5s   ' % ('#Name', 'K'))
    _out.write('%7s %7s %7s %7s   ' % ('x', 'y', 'xerr', 'yerr'))
    _out.write('%8s %8s %8s %8s   ' % ('vx', 'vy', 'vxerr', 'vyerr'))
    _out.write('%8s %4s %7s\n' %  ('t0', 'use?', 'r2d'))

    _out.write('%-10s  %5s   ' % ('#()', '(mag)'))
    _out.write('%7s %7s %7s %7s   ' % \
               ('(asec)', '(asec)', '(asec)', '(asec)'))
    _out.write('%8s %8s %8s %8s   ' % \
               ('(mas/yr)', '(mas/yr)', '(mas/yr)', '(mas/yr)'))
    _out.write('%8s %4s %7s\n' %  ('(year)', '()', '(asec)'))

    # Also write out a new absolute_refs.dat file
    newAbs = 'absolute_refs_noAccel.dat'
    _abs = open(absDir+newAbs, 'w')
    _abs.write('%-10s  %5s   ' % ('#Name', 'K'))
    _abs.write('%7s %7s %7s %7s   ' % ('x', 'y', 'xerr', 'yerr'))
    _abs.write('%8s %8s %8s %8s   ' % ('vx', 'vy', 'vxerr', 'vyerr'))
    _abs.write('%8s %4s %7s\n' %  ('t0', 'use?', 'r2d'))

    _abs.write('%-10s  %5s   ' % ('#()', '(mag)'))
    _abs.write('%7s %7s %7s %7s   ' % \
               ('(asec)', '(asec)', '(asec)', '(asec)'))
    _abs.write('%8s %8s %8s %8s   ' % \
               ('(mas/yr)', '(mas/yr)', '(mas/yr)', '(mas/yr)'))
    _abs.write('%8s %4s %7s\n' %  ('(year)', '()', '(asec)'))


    cnt = 0
    for i in range(len(labels.name)):

        if labels.useToAlign[i].strip() == '2,8': # we only care about ref stars

            # Is this an accelerating star?
            if ((labels.name[i] not in keepers) & ((labels.name[i] in accel) | (labels.name[i] in passFnames))):
                use = '0'
                absUse = '0'
                print 'Removing %s' % labels.name[i]
                cnt += 1
            else:
                use = labels.useToAlign[i]
                absUse = '1'

                # We only want to write out reference stars to the absolute_refs file:
	        _abs.write('%-10s  %5.1f   ' % (labels.name[i], labels.mag[i]))
    	        _abs.write('%9.5f %9.5f ' % (labels.x[i], labels.y[i]))
                _abs.write('%9.5f %9.5f   ' % (labels.xerr[i], labels.yerr[i]))
	        _abs.write('%8.3f %8.3f ' % (labels.vx[i], labels.vy[i]))
                _abs.write('%8.3f %8.3f   ' % (labels.vxerr[i], labels.vyerr[i]))
	        _abs.write('%8.3f %4s %7.3f\n' %  \
                           (labels.t0[i], absUse, labels.r[i]))

        else: # all the non-ref stars
            use = labels.useToAlign[i]

        # Write out the label file, which includes ref and non-ref stars
	_out.write('%-10s  %5.1f   ' % (labels.name[i], labels.mag[i]))
	_out.write('%9.5f %9.5f ' % (labels.x[i], labels.y[i]))
        _out.write('%9.5f %9.5f   ' % (labels.xerr[i], labels.yerr[i]))
	_out.write('%8.3f %8.3f ' % (labels.vx[i], labels.vy[i]))
        _out.write('%8.3f %8.3f   ' % (labels.vxerr[i], labels.vyerr[i]))
	_out.write('%8.3f %4s %7.3f\n' %  \
                   (labels.t0[i], use, labels.r[i]))

    _out.close()
    _abs.close()

    print
    print 'Removed %i reference stars' % cnt
