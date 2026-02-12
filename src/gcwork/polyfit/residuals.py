import shutil, os
import pylab as plt
import numpy as np
from matplotlib.pyplot import cm
import scipy
import scipy.stats
import os
import imageio
from PIL import Image
from gcwork import starset
from gcwork import young
from gcwork import starTables
from gcwork import orbits
import pdb
from astropy.table import Table,join

def confusionThreshold(starName1, starName2, root='./',
                       align='align/align_d_rms1000_t',
                       poly='polyfit_d/fit'):
    s = starset.StarSet(root + align)
    s.loadPolyfit(root + poly, arcsec=1, accel=0)

    names = s.getArray('name')
    id1 = names.index(starName1)
    id2 = names.index(starName2)

    star1 = s.stars[id1]
    star2 = s.stars[id2]

    years = np.array(star1.years)
    xfit1 = star1.fitXv.p + (star1.fitXv.v * (years - star1.fitXv.t0))
    yfit1 = star1.fitYv.p + (star1.fitYv.v * (years - star1.fitYv.t0))
    xfit2 = star2.fitXv.p + (star2.fitXv.v * (years - star2.fitXv.t0))
    yfit2 = star2.fitYv.p + (star2.fitYv.v * (years - star2.fitYv.t0))

    xdiff = xfit1 - xfit2
    ydiff = yfit1 - yfit2
    diff = np.hypot(xdiff, ydiff)

    for ee in range(len(years)):
        detected1 = (star1.e[ee].xpix > -999)
        detected2 = (star2.e[ee].xpix > -999)

        if (diff[ee] < 0.075):
            print('%8.3f   Close Approach: sep = %5.3f' % \
                  (years[ee], diff[ee]))

            if ((detected1 == False) and (detected2 == False)):
                print('\tNeither source found... do nothing')
            if ((detected1 == False) and (detected2 == True)):
                print('\t%13s: Not found in this epoch' % (starName1))
                print('\t%13s: Remove point for this epoch' % (starName2))
            if ((detected2 == False) and (detected1 == True)):
                print('\t%13s: Not found in this epoch' % (starName2))
                print('\t%13s: Remove point for this epoch' % (starName1))
            if ((detected1 == True) and (detected2 == True)):
                print('\t   Found both sources... do nothing')
    

def plotAllStar(epoch_cut=30, poly='polyfit_c/fit', points='points_c/', 
         rootDir='./', align='align/align_d_rms_1000_abs_t', accel=True):
    """
    plot star's track for all stars detected in more than epoch_cut epochs"""
    # find stars detected in more than epoch_cut epochs
    s = starset.StarSet(rootDir + align)
    s.loadPolyfit(rootDir + poly, accel=0, arcsec=0)
    s.loadPolyfit(rootDir + poly, accel=1, arcsec=0)
    nEpochs = s.getArray('velCnt')
    idx_star = np.where(nEpochs>epoch_cut)[0]
    names = s.getArray('name')
    star_names = np.array(names)[idx_star]
    for star in star_names:
        plotStar(star, accel=accel, poly=poly, points=points)
    return


def plotStar(starName, rootDir='./', align='align/align_d_rms_1000_abs_t',
             poly='polyfit_4_trim/fit', points='points_4_trim/', radial=False,
             accel=True, subdir=None, starset_obj=None,plotdir=None):
    """
    plot track for starName"""
    
    # Load new StarSet object if not provided
    s = starset_obj
    
    if starset_obj is None:
        s = starset.StarSet(rootDir + align)
        s.loadPolyfit(rootDir + poly, accel=0, arcsec=1)
        s.loadPolyfit(rootDir + poly, accel=1, arcsec=1)
        points_dir = rootDir + points
    else:
        # if the starset object is given, then update the points
        # directory from the starset object to keep it consistent
        points_dir = starset_obj.pointsDir
    
    names = s.getArray('name')
    if not np.in1d(starName, np.array(names)):
        print('%s not found in %s' %(starName, points))
        return
    ii = names.index(starName)
    star = s.stars[ii]


    # read in the phot file
    pointsTab = starTables.read_phot_points( os.path.join(points_dir,starName+ '.phot'))
    time = pointsTab['epoch']
    x = pointsTab['x']
    y = pointsTab['y']
    xerr = pointsTab['xerr']
    yerr = pointsTab['yerr']
    m = pointsTab['mag']
    merr = pointsTab['merr']

    chi2x = s.getArray('fitXa.chi2')[ii]/(len(time)-3)
    chi2y = s.getArray('fitYa.chi2')[ii]/(len(time)-3)
    chi2x_v = s.getArray('fitXv.chi2')[ii]/(len(time)-2)
    chi2y_v = s.getArray('fitYv.chi2')[ii]/(len(time)-2)

    if not os.path.exists('plots/plotStar'):
        os.mkdir('plots/plotStar/')

    if (subdir != None) and (not os.path.exists('plots/plotStar/' + subdir)):
        os.mkdir('plots/plotStar/' + subdir)

    refStars = find_ref_star(rootDir=rootDir)

    #################
    ## plot acceleration fit
    ###################
    if accel:
        x0_a = star.fitXa.p
        y0_a = star.fitYa.p
        x0e_a = star.fitXa.perr
        y0e_a = star.fitYa.perr
        vx_a = star.fitXa.v
        vy_a = star.fitYa.v
        vxe_a = star.fitXa.verr
        vye_a = star.fitYa.verr
        ax = star.fitXa.a
        ay = star.fitYa.a
        axe = star.fitXa.aerr
        aye = star.fitYa.aerr

        r_a = np.hypot(x0_a, y0_a)
        ar = ((ax*x0_a) + (ay*y0_a)) / r_a
        at = ((ax*y0_a) - (ay*x0_a)) / r_a
        are = np.sqrt((axe*x0_a)**2 + (aye*y0_a)**2) / r_a
        ate = np.sqrt((axe*y0_a)**2 + (aye*x0_a)**2) / r_a
        print('%s has ar = %.2f +/- %.3f mas/yr^2   at = %.2f +/- %.3f mas/yr^2 ' %(starName, ar*1000, are*1000, at*1000, ate*1000))

        dt = time - star.fitXa.t0
        fitX = x0_a + vx_a*dt + 0.5*ax*(dt**2) 
        fitY = y0_a + vy_a*dt + 0.5*ay*(dt**2) 
        fitErrX = np.sqrt(x0e_a**2 + (vxe_a*dt)**2 + (0.5*axe*dt*dt)**2)
        fitErrY = np.sqrt(y0e_a**2 + (vye_a*dt)**2 + (0.5*aye*dt*dt)**2)

    # plot linear fit
    else:
        fitx = star.fitXv
        fity = star.fitYv
        dt = time - fitx.t0
        fitX = fitx.p + fitx.v * dt
        fitY = fity.p + (fity.v * dt)
        fitErrX = np.sqrt( fitx.perr**2 + (dt * fitx.verr)**2 )
        fitErrY = np.sqrt( fity.perr**2 + (dt * fity.verr)**2 )

    diffX = x - fitX
    diffY = y - fitY
    diff = np.hypot(diffX, diffY)

    # star residual plot
    plt.figure(2, figsize=(15, 18))
    plt.clf()
    plt.subplots_adjust(left=0.15, bottom=0.09)

    dateTicLoc = plt.MultipleLocator(5)
    dateTicRng = [np.floor(time[0])-1, np.ceil(time[-1])+1]

    maxErr = np.array([xerr, yerr]).max()
    resTicRng = [-3*maxErr, 3*maxErr]

    from matplotlib.ticker import FormatStrFormatter
    fmtX = FormatStrFormatter('%5i')
    fmtY = FormatStrFormatter('%6.2f')

    paxes = plt.subplot(4, 3, 1)
    plt.errorbar(time, x, yerr=xerr, fmt='k.')
    plt.plot(time, fitX, 'b-')
    plt.plot(time, fitX + fitErrX, 'b--')
    plt.plot(time, fitX - fitErrX, 'b--')
    rng = plt.axis()
    plt.axis(dateTicRng + [rng[2], rng[3]])
    plt.xlabel('Date (yrs)')
    plt.ylabel('X (arcsec)')
    if accel:
        if vx_a>0:
            plt.annotate(r'$\chi^2_{red}$_a = %.1f' %chi2x, xy=(0.1, 0.8),xycoords='axes fraction')
            plt.annotate(r'$\chi^2_{red}$_v = %.1f' %chi2x_v, xy=(0.1, 0.6),xycoords='axes fraction')
        if vx_a<0:
            plt.annotate(r'$\chi^2_{red}$_a = %.1f' %chi2x, xy=(0.1, 0.4), xycoords='axes fraction')
            plt.annotate(r'$\chi^2_{red}$_v = %.1f' %chi2x_v, xy=(0.1, 0.2), xycoords='axes fraction')
        if np.in1d(starName, refStars):
            plt.title(starName+'_accelFit (ref)')
        else:
            plt.title(starName+'_accelFit')
    else:
        if fitx.v>0:
            plt.annotate(r'$\chi^2_{red}$_a = %.1f' %chi2x, xy=(0.1, 0.8),xycoords='axes fraction')
            plt.annotate(r'$\chi^2_{red}$_v = %.1f' %chi2x_v, xy=(0.1, 0.6),xycoords='axes fraction')
        if fitx.v<0:
            plt.annotate(r'$\chi^2_{red}$_a = %.1f' %chi2x, xy=(0.1, 0.4), xycoords='axes fraction')
            plt.annotate(r'$\chi^2_{red}$_v = %.1f' %chi2x_v, xy=(0.1, 0.2), xycoords='axes fraction')
        if np.in1d(starName, refStars):
            plt.title(starName+'_linearFit (ref)')
        else:
            plt.title(starName+'_linearFit')

    paxes.xaxis.set_major_formatter(fmtX)
    paxes.get_xaxis().set_major_locator(dateTicLoc)
    paxes.yaxis.set_major_formatter(fmtY)
    
    paxes = plt.subplot(4, 3, 2)
    plt.errorbar(time, y, yerr=yerr, fmt='k.')
    plt.plot(time, fitY, 'b-')
    plt.plot(time, fitY + fitErrY, 'b--')
    plt.plot(time, fitY - fitErrY, 'b--')
    rng = plt.axis()
    plt.axis(dateTicRng + [rng[2], rng[3]])
    plt.xlabel('Date (yrs)')
    plt.ylabel('Y (arcsec)')
    if accel:
        if vy_a>0:
            plt.annotate(r'$\chi^2_{red}$_a = %.1f' %chi2y, xy=(0.1, 0.8), xycoords='axes fraction')
            plt.annotate(r'$\chi^2_{red}$_v = %.1f' %chi2y_v, xy=(0.1, 0.6),xycoords='axes fraction')
        if vy_a<0:
            plt.annotate(r'$\chi^2_{red}$_a = %.1f' %chi2y, xy=(0.1, 0.4),xycoords='axes fraction')
            plt.annotate(r'$\chi^2_{red}$_v = %.1f' %chi2y_v, xy=(0.1, 0.2),xycoords='axes fraction')
    else:
        if fity.v>0:
            plt.annotate(r'$\chi^2_{red}$_a = %.1f' %chi2y, xy=(0.1, 0.8), xycoords='axes fraction')
            plt.annotate(r'$\chi^2_{red}$_v = %.1f' %chi2y_v, xy=(0.1, 0.6),xycoords='axes fraction')
        if fity.v<0:
            plt.annotate(r'$\chi^2_{red}$_a = %.1f' %chi2y, xy=(0.1, 0.4),xycoords='axes fraction')
            plt.annotate(r'$\chi^2_{red}$_v = %.1f' %chi2y_v, xy=(0.1, 0.2),xycoords='axes fraction')
    paxes.xaxis.set_major_formatter(fmtX)
    paxes.get_xaxis().set_major_locator(dateTicLoc)
    paxes.yaxis.set_major_formatter(fmtY)
    
    paxes = plt.subplot(4, 3, 3)
    plt.errorbar(x, y, xerr=xerr, yerr=yerr, fmt='k.')
    plt.plot(fitX, fitY, 'b-')
    plt.plot(fitX + fitErrX, fitY + fitErrY, 'b--')
    plt.plot(fitX - fitErrX, fitY - fitErrY, 'b--')
    plt.xlabel('X (arcsec)')
    plt.ylabel('Y (arcsec)')
    paxes.xaxis.set_major_formatter(fmtY)
    paxes.yaxis.set_major_formatter(fmtY)
 
    import matplotlib.cm as cm
    Reds = cm.get_cmap('Reds_r') 
    m_per = (m - m.min())/ (m.max() - m.min()) * 0.8
    plt.figure(1)
    temp = plt.scatter(m,m,c=m, cmap=Reds)
    plt.close(1)

    paxes = plt.subplot(4, 3, 4)
    for i in range(len(time)):
        plt.errorbar(time[i], diffX[i]*1000, yerr=xerr[i]*1000, fmt='o', color=Reds(m_per[i]))
    plt.plot(time, np.zeros(len(time)), 'b-')
    plt.plot(time, fitErrX*1000, 'b--')
    plt.plot(time, -fitErrX*1000, 'b--')
    plt.xlabel('Date (yrs)')
    plt.ylabel('X Residuals (mas)')
    rng = plt.axis()
    plt.axis(dateTicRng + [rng[2], rng[3]])
    paxes.get_xaxis().set_major_locator(dateTicLoc)
    paxes.xaxis.set_major_formatter(fmtX)
    paxes.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    paxes = plt.subplot(4, 3, 5)
    for i in range(len(time)):
        plt.errorbar(time[i], diffY[i]*1000., yerr=yerr[i]*1000., fmt='^', color=Reds(m_per[i]))
    plt.plot(time, np.zeros(len(time)), 'b-')
    plt.plot(time, fitErrY*1000., 'b--')
    plt.plot(time, -fitErrY*1000., 'b--')
    plt.xlabel('Date (yrs)')
    plt.ylabel('Y Residuals (mas)')
    rng = plt.axis()
    plt.axis(dateTicRng + [rng[2], rng[3]])
    paxes.get_xaxis().set_major_locator(dateTicLoc)
    paxes.xaxis.set_major_formatter(fmtX)
    paxes.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    paxes = plt.subplot(4, 3, 6)
    paxes.axis('off')
    cb = plt.colorbar(temp, fraction=0.5)
    cb.set_label('mag')
    cb.ax.invert_yaxis()

    paxes = plt.subplot(4, 3, 7)
    plt.plot(time, diffX/xerr, 'o')
    plt.plot(time, np.zeros(len(time)), 'b-')
    plt.xlabel('Date (yrs)')
    plt.ylabel('X Residuals (sigma)')
    rng = plt.axis()
    plt.axis(dateTicRng + [rng[2], rng[3]])
    paxes.get_xaxis().set_major_locator(dateTicLoc)
    paxes.xaxis.set_major_formatter(fmtX)
    #paxes.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    paxes = plt.subplot(4, 3, 8)
    plt.plot(time, diffY/yerr, 'o')
    plt.plot(time, np.zeros(len(time)), 'b-')
    rng = plt.axis()
    plt.axis(dateTicRng + [rng[2], rng[3]])
    plt.xlabel('Date (yrs)')
    plt.ylabel('Y Residuals (sigma)')
    paxes.get_xaxis().set_major_locator(dateTicLoc)
    paxes.xaxis.set_major_formatter(fmtX)
    #paxes.yaxis.set_major_formatter(FormatStrFormatter('%d'))

    paxes = plt.subplot(4, 3, 9)
    for i in range(len(m)):
        plt.errorbar(time[i], m[i], yerr=merr[i], fmt='.', color=Reds(m_per[i]))
    plt.xlabel('Date (yrs)')
    plt.ylabel('Mag')
    rng = plt.axis()
    plt.axis(dateTicRng + [rng[2], rng[3]])
    paxes.get_xaxis().set_major_locator(dateTicLoc)
    paxes.xaxis.set_major_formatter(fmtX)
    paxes.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    plt.gca().invert_yaxis()

    plt.subplot(4, 3, 10)
    bins = np.arange(-7, 7, 1)
    sigX =  diffX/xerr
    (n, b, p) = plt.hist(sigX, bins)
    plt.setp(p, 'facecolor', 'k')
    plt.axis([-5, 5, 0, 20])
    plt.xlabel('X Residuals (sigma)')
    plt.ylabel('Number of Epochs')

    plt.subplot(4, 3, 11)
    sigY =  diffY/yerr
    (n, b, p) = plt.hist(sigY, bins)
    plt.axis([-5, 5, 0, 20])
    plt.setp(p, 'facecolor', 'k')
    plt.xlabel('Y Residuals (sigma)')
    plt.ylabel('Number of Epochs')

    paxes = plt.subplot(4, 3, 12)
    plt.errorbar(time, diff*1000., yerr=np.hypot(xerr, yerr)*1000., fmt='k.')
    plt.xlabel('Date (yrs)')
    plt.ylabel('Total Residuals')
    rng = plt.axis()
    plt.axis(dateTicRng + [rng[2], rng[3]])
    paxes.get_xaxis().set_major_locator(dateTicLoc)
    paxes.xaxis.set_major_formatter(fmtX)
    paxes.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
 

    plt.subplots_adjust(wspace=0.4, hspace=0.4, right=0.95, top=0.95)
    if plotdir is None:
        plotdir = rootDir+'plots/plotStar/'
        
    if subdir == None:
        if accel:
            plt.savefig(os.path.join(plotdir,starName + '_accel.png'))
            plt.close(2)
        else:
            plt.savefig(os.path.join(plotdir,starName + '_linear.png'))
            plt.close(2)
    else:
        if accel:
            plt.savefig(os.path.join(plotdir,subdir+ starName + '_accel.png'))
            plt.close(2)
        else:
            plt.savefig(os.path.join(plotdir,subdir+ starName + '_linear.png'))
            plt.close(2)

    ####
    #radial/tangential fit
    ####
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

        # plot radial/tangential track
        plt.clf()
        dateTicLoc = plt.MultipleLocator(5)
        maxErr = np.array([rerr, terr]).max()
        resTicRng = [-3*maxErr, 3*maxErr]
        
        from matplotlib.ticker import FormatStrFormatter
        fmtX = FormatStrFormatter('%5i')
        fmtY = FormatStrFormatter('%6.2f')
        
        paxes = plt.subplot(3, 2, 1)
        plt.plot(time, fitLineR, 'b-')
        plt.plot(time, fitLineR + fitSigR, 'b--')
        plt.plot(time, fitLineR - fitSigR, 'b--')
        plt.errorbar(time, r, yerr=rerr, fmt='k.')
        rng = plt.axis()
        plt.axis(dateTicRng + [rng[2], rng[3]])
        plt.xlabel('Date (yrs)')
        plt.ylabel('R (pix)')
        paxes.xaxis.set_major_formatter(fmtX)
        paxes.get_xaxis().set_major_locator(dateTicLoc)
        paxes.yaxis.set_major_formatter(fmtY)
        
        paxes = plt.subplot(3, 2, 2)
        plt.plot(time, fitLineT, 'b-')
        plt.plot(time, fitLineT + fitSigT, 'b--')
        plt.plot(time, fitLineT - fitSigT, 'b--')
        plt.errorbar(time, t, yerr=terr, fmt='k.')
        rng = plt.axis()
        plt.axis(dateTicRng + [rng[2], rng[3]])
        plt.xlabel('Date (yrs)')
        plt.ylabel('T (pix)')
        paxes.xaxis.set_major_formatter(fmtX)
        paxes.get_xaxis().set_major_locator(dateTicLoc)
        paxes.yaxis.set_major_formatter(fmtY)
        
        paxes = plt.subplot(3, 2, 3)
        plt.plot(time, np.zeros(len(time)), 'b-')
        plt.plot(time, fitSigR, 'b--')
        plt.plot(time, -fitSigR, 'b--')
        plt.errorbar(time, r - fitLineR, yerr=rerr, fmt='k.')
        plt.axis(dateTicRng + resTicRng)
        plt.xlabel('Date (yrs)')
        plt.ylabel('R Residuals (pix)')
        paxes.get_xaxis().set_major_locator(dateTicLoc)
        
        paxes = plt.subplot(3, 2, 4)
        plt.plot(time, np.zeros(len(time)), 'b-')
        plt.plot(time, fitSigT, 'b--')
        plt.plot(time, -fitSigT, 'b--')
        plt.errorbar(time, t - fitLineT, yerr=terr, fmt='k.')
        plt.axis(dateTicRng + resTicRng)
        plt.xlabel('Date (yrs)')
        plt.ylabel('T Residuals (pix)')
        paxes.get_xaxis().set_major_locator(dateTicLoc)
        
        bins = np.arange(-7, 7, 1)
        plt.subplot(3, 2, 5)
        (n, b, p) = plt.hist(sigR, bins)
        plt.setp(p, 'facecolor', 'k')
        plt.axis([-5, 5, 0, 20])
        plt.xlabel('T Residuals (sigma)')
        plt.ylabel('Number of Epochs')
        
        plt.subplot(3, 2, 6)
        (n, b, p) = plt.hist(sigT, bins)
        plt.axis([-5, 5, 0, 20])
        plt.setp(p, 'facecolor', 'k')
        plt.xlabel('Y Residuals (sigma)')
        plt.ylabel('Number of Epochs')
        
        plt.subplots_adjust(wspace=0.4, hspace=0.4, right=0.95, top=0.95)
        plt.savefig(rootDir+'plots/plotStar/' + starName + '_radial.png')
    return

def find_ref_star(rootDir='./'):
    t = Table.read(os.path.join(rootDir,'source_list/label.dat'), format='ascii')
    idx_ref = np.where(t['col12']!='0')[0]
    idx_star = t['col1'][idx_ref]
    return np.array(idx_star)
        
def sumAllStars(root='./', align='align/align_d_rms_1000_abs_t',
                poly='polyfit_d/fit', points='points_d/',
                youngOnly=False, trimOutliers=False, trimSigma=4,
                useAccFits=False, magCut=None, radCut=None):
    """Analyze the distribution of points relative to their best
    fit velocities. Optionally trim the largest outliers in each
    stars *.points file.  Optionally make a magnitude cut with
    magCut flag and/or a radius cut with radCut flag."""

    s = starset.StarSet(root + align)
    s.loadPolyfit(root + poly, accel=0, arcsec=0)
    s.loadPolyfit(root + poly, accel=1, arcsec=0)
    if (youngOnly):
        s.onlyYoungDisk()
    
    # Re-get the names array since we may have trimmed down to
    # only the young disk stars.
    names = s.getArray('name')

    # Check if we're doing any cutting
    if ((magCut != None) and (radCut != None)):
        mag = s.getArray('mag')
        x = s.getArray('x')
        y = s.getArray('y')
        r = np.hypot(x,y)
        idx = np.where((mag < magCut) & (r < radCut))[0]
        newstars = []
        for i in idx:
            newstars.append(s.stars[i])
        s.stars = newstars
        mag = mag[idx]
        x = x[idx]
        y = y[idx]
        r = r[idx]
        names = [names[ii] for ii in idx]
    elif ((magCut != None) and (radCut == None)):
        mag = s.getArray('mag')
        x = s.getArray('x')
        y = s.getArray('y')
        r = np.hypot(x,y)
        idx = np.where(mag < magCut)[0]
        newstars = []
        for i in idx:
            newstars.append(s.stars[i])
        s.stars = newstars
        mag = mag[idx]
        x = x[idx]
        y = y[idx]
        r = r[idx]
        names = [names[ii] for ii in idx]
    elif ((magCut == None) and (radCut != None)):
        mag = s.getArray('mag')
        x = s.getArray('x')
        y = s.getArray('y')
        r = np.hypot(x,y)
        idx = np.where(r < radCut)[0]
        newstars = []
        for i in idx:
            newstars.append(s.stars[i])
        s.stars = newstars
        mag = mag[idx]
        x = x[idx]
        y = y[idx]
        r = r[idx]
        names = [names[ii] for ii in idx]

    # Make some empty arrays to hold all our results.
    sigmaX = np.arange(0, dtype=float)
    sigmaY = np.arange(0, dtype=float)
    sigma  = np.arange(0, dtype=float)
    diffX_all = np.arange(0, dtype=float)
    diffY_all = np.arange(0, dtype=float)
    xerr_all = np.arange(0, dtype=float)
    yerr_all = np.arange(0, dtype=float)

    # Loop through all the stars and combine their residuals.
    for star in s.stars:
        starName = star.name
        
        pointsFile = root + points + starName + '.points'
        if os.path.exists(pointsFile + '.orig'):
            pointsTab = starTables.read_points(pointsFile + '.orig')
        else:
            pointsTab = starTables.read_points(pointsFile)

        # Observed Data
        t = pointsTab['epoch']
        x = pointsTab['x']
        y = pointsTab['y']
        xerr = pointsTab['xerr']
        yerr = pointsTab['yerr']

        # Best fit velocity model
        if (useAccFits == True):
            fitx = star.fitXa
            fity = star.fitYa
        else:
            fitx = star.fitXv
            fity = star.fitYv

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

        # Residuals
        diffX = x - fitLineX
        diffY = y - fitLineY
        diff = np.hypot(diffX, diffY)
        rerr = np.sqrt((diffX*xerr)**2 + (diffY*yerr)**2) / diff
        sigX = diffX / xerr
        sigY = diffY / yerr
        sig = diff / rerr

        idxX = (np.where(abs(sigX) > trimSigma))[0]
        idxY = (np.where(abs(sigY) > trimSigma))[0]
        idx  = (np.where(abs(sig) > trimSigma))[0]


#         if (len(idxX) > 0):
#             print 'X %d sigma outliers for %13s: ' % \
#                   (trimSigma, starName), sigX[idxX], t[idxX]
#         if (len(idxY) > 0):
#             print 'Y %d sigma outliers for %13s: ' % \
#                   (trimSigma, starName), sigY[idxY], t[idxY]
#         if (len(idx) > 0):
#             print 'T %d sigma outliers for %13s: ' % \
#                   (trimSigma, starName), sig[idx], t[idx]

        if ((trimOutliers == True) and (len(idx) > 0)):
            if not os.path.exists(pointsFile + '.orig'):
                shutil.copyfile(pointsFile, pointsFile + '.orig')

            for ii in idx[::-1]:
                pointsTab.delete(ii)

            pointsTab.writeto(pointsFile)

        # Combine this stars information with all other stars.
        sigmaX = concatenate((sigmaX, sigX))
        sigmaY = concatenate((sigmaY, sigY))
        sigma = concatenate((sigma, sig))
        diffX_all = concatenate((diffX_all,diffX))
        diffY_all = concatenate((diffY_all,diffY))
        xerr_all = concatenate((xerr_all,xerr))
        yerr_all = concatenate((yerr_all,yerr))

    rmsDiffXY = (diffX_all.std() + diffY_all.std()) / 2.0 * 1000.0
    aveDiffR = np.sqrt(diffX_all**2 + diffY_all**2).mean()
    medDiffR = np.median(np.sqrt(diffX_all**2 + diffY_all**2))

    print(diffX_all.mean(), diffY_all.mean())
    print(diffX_all.std(), diffY_all.std())
    print(rmsDiffXY, aveDiffR, medDiffR)
    print(np.median(xerr_all))

    # Residuals should have a gaussian probability distribution
    # with a mean of 0 and a sigma of 1. Overplot this to be sure.
    ggx = np.arange(-7, 7, 0.25)
    ggy = normpdf(ggx, 0, 1)

    print('Mean   RMS residual: %5.2f sigma' % (sigma.mean()))
    print('Stddev RMS residual: %5.2f sigma' % (sigma.std()))
    print('Median RMS residual: %5.2f sigma' % (np.median(sigma)))
    
    print('Mean X centroiding error: %5.4f mas (median %5.4f mas)' % \
        ((xerr_all*1000.0).mean(), np.median(xerr_all)*10**3))
    print('Mean Y centroiding error: %5.4f mas (median %5.4f mas)' % \
        ((yerr_all*1000.0).mean(), np.median(yerr_all)*10**3))
    print('Mean distance from velocity fit: %5.4f mas (median %5.4f mas)' % \
        (aveDiffR*10**3, medDiffR*10**3))

    ##########
    # Plot
    ##########
    bins = np.arange(-7, 7, 1.0)
    plt.figure(1)
    plt.clf()
    plt.subplot(3, 1, 1)
    (nx, bx, px) = plt.hist(sigmaX, bins)
    ggamp = ((sort(nx))[-2:]).sum() / (2.0 * ggy.max())
    plt.plot(ggx, ggy*ggamp, 'k-')
    plt.xlabel('X Residuals (sigma)')

    plt.subplot(3, 1, 2)
    (ny, by, py) = plt.hist(sigmaY, bins)
    ggamp = ((sort(ny))[-2:]).sum() / (2.0 * ggy.max())
    plt.plot(ggx, ggy*ggamp, 'k-')
    plt.xlabel('Y Residuals (sigma)')

    plt.subplot(3, 1, 3)
    (ny, by, py) = plt.hist(sigma, np.arange(0, 7, 0.5))
    plt.xlabel('Total Residuals (sigma)')

    plt.subplots_adjust(wspace=0.34, hspace=0.33, right=0.95, top=0.97)
    plt.savefig(root+'plots/residualsDistribution.eps')
    plt.savefig(root+'plots/residualsDistribution.png')

    # Put all residuals together in one histogram
    plt.clf()
    sigmaA = []
    for ss in range(len(sigmaX)):
        sigmaA = np.concatenate([sigmaA,[sigmaX[ss]]])
        sigmaA = np.concatenate([sigmaA,[sigmaY[ss]]])
    (na, ba, pa) = plt.hist(sigmaA, bins)
    ggamp = ((sort(na))[-2:]).sum() / (2.0 * ggy.max())
    plt.plot(ggx, ggy*ggamp, 'k-')
    plt.xlabel('Residuals (sigma)')
    plt.savefig(root+'plots/residualsAll.eps')
    plt.savefig(root+'plots/residualsAll.png')
    return
    
def residual_quiver(root='./', align='align/align_d_rms_1000_abs_t',
                 poly='polyfit_4_trim/fit', epochsFile='scripts/epochsInfo.txt',
                 useAccFits=True, 
                 magCut=16, rMax=10, rMin=0.4, epochCut='max', 
                 qscale_spk=0.005, qscale_ao=0.0005):
    """
    plot residual in each epoch and make a quier plot"""
    s = starset.StarSet(root + align, relErr=1)
    s.loadPolyfit(root + poly, accel=0, arcsec=1)
    s.loadPolyfit(root + poly, accel=1, arcsec=1)

    # Use only stars detected in epochs more than epochCut
    epochCnt = s.getArray('velCnt')
    mag = s.getArray('mag')
    x = s.getArray('x')
    y = s.getArray('y')
    r = np.hypot(x, y)
    names = s.getArray('name')

    if epochCut == 'max':
        epochCut = epochCnt.max()
    idx = np.where((mag < magCut) & (r<rMax) & (r>rMin) & (epochCnt >= epochCut))[0]
    newStars = []
    for ii in idx:
        newStars.append(s.stars[ii])
    s.stars = newStars
    names = s.getArray('name')
    print('Using %d out of %d stars detected in more than %d epochs' 
          %(len(newStars), len(epochCnt), epochCut))

    # Make some empty arrays to hold all our results.
    numEpochs = len(s.stars[0].years)
    sigmaX = np.zeros(numEpochs, float)
    sigmaY = np.zeros(numEpochs, float)
    sigma  = np.zeros(numEpochs, float)
    diffEpX = np.zeros(numEpochs, float)
    diffEpY = np.zeros(numEpochs, float)
    diffEp  = np.zeros(numEpochs, float)

    # find speckle epoch and ao epoch
    epochsInfo = Table.read(epochsFile, format='ascii')
    epoch_used = np.where(epochsInfo['doAlign']==1)[0]
    epoch_names = epochsInfo['epoch'][epoch_used]
    epoch_name = [i[0:8] for i in epoch_names]

    epoch_ao = np.where(epochsInfo['isAO']==1)[0]
    epoch_ao = np.intersect1d(epoch_used, epoch_ao)
    epoch_ao_idx = [list(epoch_used).index(i) for i in epoch_ao]

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
    if not os.path.exists(root+'plots/residual'):
        os.mkdir(root + 'plots/residual')

    t_ref = Table.read('align/align_d_rms_1000_abs_t.starsUsed', format='ascii')
    # Clean column names for subsequent code
    if t_ref.colnames[0] is not 'col1':
        for (i, cur_name) in zip(range(1, len(t_ref.colnames) + 1), t_ref.colnames):
            t_ref.rename_column(cur_name, 'col' + str(i))
    
    for ee in range(numEpochs):
        # Observed data
        x = s.getArrayFromEpoch(ee, 'x')
        y = s.getArrayFromEpoch(ee, 'y')
        xerr_p = s.getArrayFromEpoch(ee, 'xerr_p')
        yerr_p = s.getArrayFromEpoch(ee, 'yerr_p')
        xerr_a = s.getArrayFromEpoch(ee, 'xerr_a')
        yerr_a = s.getArrayFromEpoch(ee, 'yerr_a')
        xerr = np.hypot(xerr_p, xerr_a)
        yerr = np.hypot(yerr_p, yerr_a)
        t = s.stars[0].years[ee]

        dt = t - t0
        fitX = x0 + (vx * dt)
        fitSigX = np.sqrt( x0e**2 + (dt * vxe)**2 )

        fitY = y0 + (vy * dt)
        fitSigY = np.sqrt( y0e**2 + (dt * vye)**2 )

        if (useAccFits == True):
            fitX += (ax * dt**2) / 2.0
            fitY += (ay * dt**2) / 2.0
            fitSigX = np.sqrt(fitSigX**2 + (dt**2 * axe / 2.0)**2)
            fitSigY = np.sqrt(fitSigY**2 + (dt**2 * aye / 2.0)**2)

        # Residuals
        diffX = x - fitX
        diffY = y - fitY
        diff = np.hypot(diffX, diffY)

        sigX = diffX / xerr
        sigY = diffY / yerr
        sigX_abs = np.abs(diffX / xerr)
        sigY_abs = np.abs(diffY / yerr)
        #rerr = np.hypot(diffX*xerr, diffY*yerr) / diff
        rerr = np.hypot(xerr, yerr)
        sig = np.abs(diff / rerr)

        print('Epoch %d' % ee)
        print('p    p_err   fitP    fitP_err    diffP   ')
        print('%8.5f +/- %8.5f   %8.5f +/- %8.5f  %8.5f (%5.1f sigma)' % \
              (x[0], xerr[0], fitX[0], fitSigX[0], diffX[0], sigX[0]))
        print('%8.5f +/- %8.5f   %8.5f +/- %8.5f  %8.5f (%5.1f sigma)' % \
              (y[0], yerr[0], fitY[0], fitSigY[0], diffY[0], sigY[0]))

        good = np.where((x > -1e2) & (y > -1e2))[0]
        sigmaX[ee] = np.median(sigX_abs[good])
        sigmaY[ee] = np.median(sigY_abs[good])
        sigma[ee] = np.median(sig[good])

        diffEpX[ee] = np.median(abs(diffX[good]))
        diffEpY[ee] = np.median(abs(diffY[good]))
        diffEp[ee] = np.median(diff[good])

        # Plot residuals in each epoch
        ref_epoch = t_ref['col' + str(ee+1)]
        idx_ref = np.where(ref_epoch!='---')
        ref = np.intersect1d(names, ref_epoch[idx_ref])
        idx_ref = [names.index(i) for i in ref] 

        # make plots for AO 
        if np.in1d(ee, np.array(epoch_ao_idx)):
            plt.figure(figsize=(10,10))
            qscale = qscale_ao
            bins = np.linspace(0, 0.002, 20)
            plt.quiver(x[good],y[good],diffX[good],diffY[good], units='xy', scale=qscale, color='k')
            if len(idx_ref) != 0:
                plt.quiver(x[idx_ref],y[idx_ref],diffX[idx_ref],diffY[idx_ref], units='xy', scale=qscale, color='r')
            plt.quiver([0,4], [0,4.2], [0,0.0005], [0,0], units='xy', scale = qscale, color='b')
            plt.xlabel('x(arcsec)')
            plt.ylabel('y(arcsec)')
            plt.text(3.5, 3.8, '0.5 mas', color = 'b')
            plt.text(-4.8, 4.2, 'stars with mag<%.1f, %.1f<r<%.1f,epochs>%d'%(magCut, rMin, rMax, epochCut), color = 'k')
            plt.text(-4.8, 3.8, 'reference stars', color = 'r')
            if useAccFits:
                plt.title('acc_residual_%s_%depoch_%03dstars' %(epoch_name[ee],ee,len(good)))
            else:
                plt.title('lin_residual_%s_%depoch_%03dstars' %(epoch_name[ee],ee,len(good)))
            plt.axis('equal')
            plt.xlim([-5.1,5.1])
            plt.ylim([-5.1,5.1])
            plt.savefig(root+'plots/residual/residual_%02d_epoch.png' %ee)
            plt.close()

            # plot the histogram of the residual
            plt.figure(figsize=(10,10))
            plt.clf()
            plt.hist(diff[good], color='k', histtype='step', bins=bins, label='all stars')
            if len(idx_ref)!=0:
                idx_area_cut = np.where((np.abs(x[idx_ref])<5.1)|(np.abs(y[idx_ref])<5.1))[0]
                plt.hist(diff[idx_ref][idx_area_cut], color='r', histtype='step', bins=bins, label='reference')
            plt.legend()
            plt.title('residual_%s_%depoch_%03dstars' %(epoch_name[ee],ee,len(good)))
            plt.ylim(0, 70)
            plt.xlim(0, 0.002)
            plt.savefig(root+'plots/residual/hist_residual_%02d_epoch.png' %ee)
            plt.close()

            # plot the histogram of the residual in x and y in units of sigma
            plt.figure(figsize=(10,10))
            plt.clf()
            bins = np.linspace(-5,5,20)
            plt.hist(sigX, color='r', histtype='step', bins=bins, label='X(sigma)')
            plt.hist(sigY, color='g', histtype='step', bins=bins, label='Y(sigma)')
            plt.legend()
            plt.title('residual_sigma_%s_%depoch_%03dstars' %(epoch_name[ee],ee,len(good)))
            plt.ylim(0, 70)
            plt.savefig(root+'plots/residual/hist_sigma_%02d_epoch.png' %ee)
            plt.close()

        # make plots for Speckle 
        else:
            qscale = qscale_spk
            bins = np.linspace(0, 0.02, 20)
            fig = plt.figure(figsize=(10,10))
            plt.quiver(x[good],y[good], diffX[good], diffY[good], units='xy', scale=qscale, color='k')
            if len(idx_ref)!= 0:
                plt.quiver(x[idx_ref],y[idx_ref],diffX[idx_ref],diffY[idx_ref], units='xy', scale=qscale, color='r')
            plt.quiver([0,4], [0,4.2], [0,0.005], [0,0], units='xy', scale = qscale, color='b')
            plt.xlabel('x(arcsec)')
            plt.ylabel('y(arcsec)')
            plt.text(3.5, 3.8, '5 mas', color = 'b')
            plt.text(-4.8, 4.2, 'stars with mag<%.1f, %.1f<r<%.1f,epochs>%d'%(magCut, rMin, rMax, epochCut), color = 'k')
            plt.text(-4.8, 3.8, 'reference stars', color = 'r')
            if useAccFits:
                plt.title('acc_residual_%s_%depoch_%03dstars' %(epoch_name[ee],ee,len(good)))
            else:
                plt.title('lin_residual_%s_%depoch_%03dstars' %(epoch_name[ee],ee,len(good)))
            plt.axis('scaled')
            plt.xlim([-5.1,5.1])
            plt.ylim([-5.1,5.1])
            #plt.axis([-5.1, 5.1, -5.1, 5.1])
            #plt.set_autoscale_on(False)
            fig.savefig(root+'plots/residual/residual_%02d_epoch.png' %ee)
            plt.close(fig)

            # plot the histogram of the residual
            plt.figure(figsize=(10,10))
            plt.clf()
            plt.hist(diff[good], color='k', histtype='step', bins=bins, label='all stars')
            if len(idx_ref)!=0:
                idx_area_cut = np.where((np.abs(x[idx_ref])<5.1)|(np.abs(y[idx_ref])<5.1))[0]
                plt.hist(diff[idx_ref][idx_area_cut], color='r', histtype='step', bins=bins, label='reference')
            plt.legend()
            plt.title('residual_%s_%depoch_%03dstars' %(epoch_name[ee],ee,len(good)))
            plt.ylim(0, 70)
            plt.xlim(0, 0.02)
            plt.savefig(root+'plots/residual/hist_residual_%02d_epoch.png' %ee)
            plt.close()

            # plot the histogram of the residual in x and y in units of sigma
            plt.figure(figsize=(10,10))
            plt.clf()
            bins = np.linspace(-5,5,20)
            plt.hist(sigX, color='r', histtype='step', bins=bins, label='X(sigma)')
            plt.hist(sigY, color='g', histtype='step', bins=bins, label='Y(sigma)')
            plt.legend()
            plt.title('residual_sigma_%s_%depoch_%03dstars' %(epoch_name[ee],ee,len(good)))
            plt.ylim(0,70)
            plt.savefig(root+'plots/residual/hist_sigma_%02d_epoch.png' %ee)
            plt.close()

        file_names = sorted((fn for fn in os.listdir('./plots/residual') if (fn.startswith('residual_') & fn.endswith('.png'))))
        images = []
        for filename in file_names:
            images.append(imageio.imread('plots/residual/' + filename))
        imageio.mimsave('plots/' + "residual_each_epoch.gif", images, duration=0.2)


def sigmaVsEpoch(root='./', align='align/align_d_rms_1000_abs_t',
                 poly='polyfit_c/fit', epochsFile='scripts/epochsInfo.txt',
                 useAccFits=True, magCut=16, rMax=2, rMin=0.4, epochCut='max'):
    """
    Plot the average offset (in sigma) from the best fit
    velocity as a function of epoch.
    """
    s = starset.StarSet(root + align, relErr=1)
    s.loadPolyfit(root + poly, accel=0, arcsec=1)
    s.loadPolyfit(root + poly, accel=1, arcsec=1)

    # Use only stars detected in epochs more than epochCut
    epochCnt = s.getArray('velCnt')
    mag = s.getArray('mag')
    x = s.getArray('x')
    y = s.getArray('y')
    r = np.hypot(x, y)
    names = s.getArray('name')

    if epochCut == 'max':
        epochCut = epochCnt.max()
    idx = np.where((mag < magCut) & (r<rMax) & (r>rMin) & (epochCnt >= epochCut))[0]
    newStars = []
    for ii in idx:
        newStars.append(s.stars[ii])
    s.stars = newStars
    names = s.getArray('name')
    print('Using %d out of %d stars detected in more than %d epochs' 
          %(len(newStars), len(epochCnt), epochCut))

    # Make some empty arrays to hold all our results.
    numEpochs = len(s.stars[0].years)
    sigmaX = np.zeros(numEpochs, float)
    sigmaY = np.zeros(numEpochs, float)
    sigma  = np.zeros(numEpochs, float)
    diffEpX = np.zeros(numEpochs, float)
    diffEpY = np.zeros(numEpochs, float)
    diffEp  = np.zeros(numEpochs, float)

    # find speckle epoch and ao epoch
    epochsInfo = Table.read(epochsFile, format='ascii')
    epoch_used = np.where(epochsInfo['doAlign']==1)[0]
    epoch_names = epochsInfo['epoch'][epoch_used]
    epoch_name = [i[0:8] for i in epoch_names]

    epoch_ao = np.where(epochsInfo['isAO']==1)[0]
    epoch_ao = np.intersect1d(epoch_used, epoch_ao)
    epoch_ao_idx = [list(epoch_used).index(i) for i in epoch_ao]

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

    t_ref = Table.read('align/align_d_rms_1000_abs_t.starsUsed', format='ascii')
    for ee in range(numEpochs):
        # Observed data
        x = s.getArrayFromEpoch(ee, 'x')
        y = s.getArrayFromEpoch(ee, 'y')
        xerr_p = s.getArrayFromEpoch(ee, 'xerr_p')
        yerr_p = s.getArrayFromEpoch(ee, 'yerr_p')
        xerr_a = s.getArrayFromEpoch(ee, 'xerr_a')
        yerr_a = s.getArrayFromEpoch(ee, 'yerr_a')
        xerr = np.hypot(xerr_p, xerr_a)
        yerr = np.hypot(yerr_p, yerr_a)
        t = s.stars[0].years[ee]

        dt = t - t0
        fitX = x0 + (vx * dt)
        fitSigX = np.sqrt( x0e**2 + (dt * vxe)**2 )

        fitY = y0 + (vy * dt)
        fitSigY = np.sqrt( y0e**2 + (dt * vye)**2 )

        if (useAccFits == True):
            fitX += (ax * dt**2) / 2.0
            fitY += (ay * dt**2) / 2.0
            fitSigX = np.sqrt(fitSigX**2 + (dt**2 * axe / 2.0)**2)
            fitSigY = np.sqrt(fitSigY**2 + (dt**2 * aye / 2.0)**2)

        # Residuals
        diffX = x - fitX
        diffY = y - fitY
        diff = np.hypot(diffX, diffY)

        sigX = diffX / xerr
        sigY = diffY / yerr
        sigX_abs = np.abs(diffX / xerr)
        sigY_abs = np.abs(diffY / yerr)
        #rerr = np.hypot(diffX*xerr, diffY*yerr) / diff
        rerr = np.hypot(xerr, yerr)
        sig = np.abs(diff / rerr)

        print('Epoch %d' % ee)
        print('p    p_err   fitP    fitP_err    diffP   ')
        print('%8.5f +/- %8.5f   %8.5f +/- %8.5f  %8.5f (%5.1f sigma)' % \
              (x[0], xerr[0], fitX[0], fitSigX[0], diffX[0], sigX[0]))
        print('%8.5f +/- %8.5f   %8.5f +/- %8.5f  %8.5f (%5.1f sigma)' % \
              (y[0], yerr[0], fitY[0], fitSigY[0], diffY[0], sigY[0]))

        good = np.where((x > -1e2) & (y > -1e2))[0]
        sigmaX[ee] = np.median(sigX_abs[good])
        sigmaY[ee] = np.median(sigY_abs[good])
        sigma[ee] = np.median(sig[good])

        diffEpX[ee] = np.median(abs(diffX[good]))
        diffEpY[ee] = np.median(abs(diffY[good]))
        diffEp[ee] = np.median(diff[good])

    ##########
    # Plot
    ##########
    # for all the epochs in a year, only show the name of the first epoch in clear
    years = np.array(s.stars[0].years)
    current_year = epoch_name[0][0:2]
    year_show=[0]
    for i in np.arange(1, len(years)):
        next_year = epoch_name[i][0:2]
        if current_year != next_year:
            year_show.append(i)
            current_year = next_year

    plt.figure(figsize=(10,5))
    plt.plot(years, sigmaX, 'rx')
    plt.plot(years, sigmaY, 'bx')
    plt.plot(years, sigma, 'ko')
    plt.xlabel('Epoch (years)')
    plt.ylabel('Median Residual Error (sigma)')
    plt.legend(('X', 'Y', 'Total'))
    epoch_name = np.array(epoch_name) 
    plt.xticks(years[year_show], epoch_name[year_show], rotation='vertical')
    plt.grid(axis='x')
    plt.ylim(0, 2.2)
    plt.title('mag<%.1f, %.1f<r<%.1f, epoch>%d' %(magCut, rMin, rMax, epochCut))
    plt.tight_layout()
    plt.savefig(root+'plots/residualsVsEpoch.png')
    plt.close()

    #plt.clf()
    plt.figure(figsize=(10,5))
    plt.plot(years, diffEpX*1000.0, 'rx')
    plt.plot(years, diffEpY*1000.0, 'bx')
    plt.plot(years, diffEp*1000.0, 'ko')
    plt.xlabel('Epoch (years)')
    plt.ylabel('Median Residual Error (mas)')
    plt.legend(('X', 'Y', 'Total'))
    plt.xticks(years[year_show], epoch_name[year_show], rotation='vertical')
    plt.grid(axis='x')
    plt.ylim(-2, 6)
    plt.title('mag<%.1f, %.1f<r<%.1f, epoch>%d' %(magCut, rMin, rMax, epochCut))
    plt.tight_layout()
    plt.savefig(root+'plots/residualsVsEpochMAS.png')
    plt.close()


    # Print out epochs with higher than 3 sigma median residuals
    sigma_value = 2
    hdx = (np.where(sigma > sigma_value))[0]
    print('Epochs with median residuals > %.1f sigma:' %sigma_value)
    for hh in hdx:
        print('%s  residual = %4.1f' % (epoch_name[hh], sigma[hh]))

    #print('chi2 = %.3f, Ndata=%d, Npar=%d' %(chi2, Ndata, Npar))
    return 

def writeGoodStars(starNames, root='./', align='align/align_d_rms_1000_abs_t',
                 poly='polyfit_1000/fit', points='points_1000'):
    """
    This function is used to find qualified stars as reference stars in next pass run
    Input:
        starNames: good stars from previous output
                   good stars = detected in all epochs +  radius cut + magnitude cut

    Output:
        name:
        mag:
        x,y: in arcsec
        xerr, yerr: mean observation postional error in arcsec
        diffx, diffy: mean observe - fitting in arcsec

    Our criterial:
        nstars: use the most n brightest stars
        disMin(arcsec): to evenly distribute the stars, require the minimun distance between two stars
        errMax(pix): xerr, yerr is not so large, require the maximum x/y err
        diffMax(pix): xobs-xfit, yobs-yfit is not so large, require maximum diffx/diffy
    """
    s = starset.StarSet(root + align)
    s.loadPolyfit(root + poly, accel=0)
 
    # read names, mag, epoch
    name = s.getArray('name')
    mag = np.array(s.getArray('mag') * 1.0)
    t = np.array(s.stars[0].years)
    
    # read the position 
    x = s.getArray('x')
    y = s.getArray('y')

    # fine the index for specific stars in starNames
    Ind = [name.index(i) for i in starNames]   
    name = np.array(name)[Ind] 
    mag = np.array(mag)[Ind]
    x = np.array(x)[Ind]
    y = np.array(y)[Ind]

    xerr = np.zeros(len(starNames), float)
    yerr = np.zeros(len(starNames), float)

    chi2x = np.zeros(len(starNames), float)  
    chi2y = np.zeros(len(starNames), float)  

    #diffx = np.zeros(len(starNames), float)
    #diffy = np.zeros(len(starNames), float)

    # go through each star in starNames
    for i in range(len(starNames)):
        starName = starNames[i]
        ind = Ind[i]
        star = s.stars[ind]
        # read observed position in arcsec
        pointsTab = starTables.read_points(root + points + starName + '.points')
        X = pointsTab['x']
        Y = pointsTab['y']
        Xerr = pointsTab['xerr']
        Yerr = pointsTab['yerr']

        # read linear fitting result in arcsec
        t0 = star.fitXv.t0
        dt = t- t0
        x0 = star.fitXv.p
        y0 = star.fitYv.p
        x0e = star.fitXv.perr
        y0e = star.fitYv.perr
        vx = star.fitXv.v
        vy = star.fitYv.v
        vxe = star.fitXv.verr
        vye = star.fitYv.verr
        fitLineX = x0 +  vx * dt
        fitSigX = np.sqrt( x0e**2 + (dt * vxe)**2 )
        fitLineY = y0 + vy * dt
        fitSigY = np.sqrt( y0e**2 + (dt * vye)**2 )

        # calculate the difference 
        diffX = X - fitLineX
        diffY = Y - fitLineY

        # calculate reduced chi2
        sigX = diffX**2/Xerr**2
        sigY = diffY**2/Yerr**2

        dof = len(t) - 2
        chi2X = np.sum(sigX)/dof
        chi2Y = np.sum(sigY)/dof

        # keep the info of position err, chi2 

        xerr[i] = np.mean(Xerr)
        yerr[i] = np.mean(Yerr)
        chi2x[i] = chi2X
        chi2y[i] = chi2Y

        #diffx[i] = np.mean(np.abs(diffX))
        #diffy[i] = np.mean(np.abs(diffY))

    t = Table()
    t['name'] = name 
    t['mag'] = mag
    t['x'] = x
    t['y'] = y
    t['xerr'] = xerr
    t['yerr'] = yerr
    t['chi2x'] = chi2x
    t['chi2y'] = chi2y
    #t['diffx'] = diffx
    #t['diffy'] = diffy
    t.write('goodStars.dat',format='ascii.fixed_width')

def analyzeGoodStars(starTable, nstars = None, disMin = 0.003, errMax =  0.1, chi2Max = 10, #diffMax=0.03
                    rootDir='./', align='align/align_d_rms_1000_abs_t', poly='polyfit_d/fit', points='points_d/'):
    """
    This function is used to find qualified stars as reference stars in next pass run
    Input:
        starTable with good stars from previous output
        good stars = detected in all epochs +  radius cut + magnitude cut
    Our criterial:
        nstars: use the most n brightest stars
        disMin(arcsec): to evenly distribute the stars, require the minimun distance between two stars
        errMax(arcsec): maximum observation position err
        chi2Max: masimum reduced chi2
        #diffMax(pix): xobs-xfit, yobs-yfit is not so large, require maximum diffx/diffy
    Return:
        qualified starNames
    """
    t = Table.read(starTable,format='ascii.fixed_width')
    t.sort('mag')
    name = list(t['name'])
    x = list(t['x'])
    y = list(t['y'])
    mag = list(t['mag'])
    xerr = t['xerr']
    yerr = t['yerr']
    chi2x = t['chi2x']
    chi2y = t['chi2y']
    #diffx = t['diffx']
    #diffy = t['diffy']

    if nstars == None:
        nstars = len(name)

    useInd = []
    for i in range(nstars):
        if xerr[i] > errMax or yerr[i] > errMax:
            continue
        elif chi2x[i] > chi2Max or chi2y[i] > chi2Max:
            continue
        elif i ==0:
            useInd.append(i)
        else:
            for j in range(0,i):
                dis = np.sqrt((x[i]-x[j])**2 +(y[i]-y[j])**2)
                if dis < disMin:
                    break
            else:
                useInd.append(i)
    t = t[useInd]
    
    print('%d stars are found to be bright and evenly distributed' %len(useInd))
    print('Minimum distance required between two stars: %f arcsec' %disMin)
    print('faintest star\'s magnitude: %f' %max(t['mag']))
    print('biggest postion error in x and y : %f mas, %f mas' %(max(t['xerr'])*1000, max(t['yerr'])*1000))
    print('biggest reduced chi2 in x and y: %f , %f' %(max(t['chi2x']), max(t['chi2y'])))
    #print 'biggest fittind difference in x and y : %f pixel, %f pixel' %(max(t['diffx']), max(t['diffy']))

    t.write('useStar.dat', format='ascii.fixed_width')
    print('stars used next pass: ')
    print(list(t['name']))
    plt.clf()
    plt.figure(figsize=(10,10))
    plt.plot(t['x'],t['y'],'o')
    plt.xlabel('x (arcsec)')
    plt.ylabel('y (arcsec)')
    plt.axis('equal')
    plt.savefig('used_star.png',format='png')

    return 

def plot_mag_residuals(starName, rootDir='./', align='align/align_d_rms_1000_abs_t',
                       poly='polyfit_d/fit', points='points_d/', radial=False):
    """
    Plot the magnitude vs. time for the specified star. 
    """
    s = starset.StarSet(rootDir + align)
    names = s.getArray('name')
    
    ii = names.index(starName)
    star = s.stars[ii]
    
    epoch = s.years
    mag_avg = star.mag
    mag_avg_err = star.magerr
    mag = star.getArrayAllEpochs('mag')
    magerr =  1.0 / star.getArrayAllEpochs('snr')

    plt.clf()
    plt.errorbar(epoch, mag, yerr=magerr, marker='o', linestyle='none')
    plt.title(star.name)

    plt.ylim(mag_avg - 10*mag_avg_err, mag_avg + 10*mag_avg_err)

    return

def compare_label_align(label_file = 'source_list/absolute_refs.dat',\
                align='align/align_d_rms_1000_abs_t', poly='polyfit_4_trim/fit', \
                rMin=0.4, rMax=100, KpMax=16, verrMax=2, \
                qscale_v=0.05, deltaKey_v=0.05, \
                qscale_p=0.001, deltaKey_p = 0.001, \
                bins_delV = None, bins_delV_ratio = None):
    """
    this function is used to compare between alignment and label.dat.
    both position difference and velocity difference.
    Input: 
        align: the align file that is used to read alignment file
        label_file: label.dat file path
        Stars that are used in comparison
            rMin, rMax: radius range (arcsec)
            KpMax:      magnitude cut
            verrMax:    velocity error less than verrMax (mas/yr)
        deltaKey_v: arrow key in velocity quiver plot
        deltaKey_p: arrow key in position quiver plot
        qscale_v: change the arrow size in velocity quiver plot
        qscale_p: change the arrow size in postion quiver plot
        bins_delV: velocity difference histogram bins range
        bins_delV_ratio: velocity difference weighted by error histogram bins range

    Output:
        velocity comparison
        quiver plot of (x, y, del_vx, del_vy)
        quiver plot of (x, y, del_x, del_y)
    """
    ###############################==
    # read tables
    ###############################=
    # read label.dat vel in arcsec & mas/yr
    t1 = Table.read(label_file, format ='ascii')
    cols = t1.colnames
    if len(cols) == 13:
        t1.rename_columns(cols, ['Name','K','x','y','xerr','yerr','vx','vy','vxerr','vyerr','t0','use','r'])
    elif len(cols) == 17:
        t1.rename_columns(cols, ['Name','K','x','y','xerr','yerr','vx','vy','vxerr','vyerr','ax','ay','axerr','ayerr','t0','use','r'])
    
    # remove verr
    idx = np.where(t1['vxerr']!=0)[0]

    # read align in arcsec & arcsec/yr and convert it to arcsec & mas/yr
    s = starset.StarSet(align)
    s.loadPolyfit(poly, arcsec=1)
    t2 = Table()
    t2['Name'] = s.getArray('name')
    t2['K'] = s.getArray('mag') * 1.0
    t2['x'] = s.getArray('fitXv.p')
    t2['y'] = s.getArray('fitYv.p')
    t2['xerr'] = s.getArray('fitXv.perr')
    t2['yerr'] = s.getArray('fitYv.perr')
    t2['vx'] = s.getArray('fitXv.v')*1000.
    t2['vy'] = s.getArray('fitYv.v')*1000.
    t2['vxerr'] = s.getArray('fitXv.verr')*1000.
    t2['vyerr'] = s.getArray('fitYv.verr')*1000.
    t2['t0'] = s.getArray('fitXv.t0')
    #t2.write('plots/polyfit.dat', format='ascii.fixed_width')

    plot_compare(t1[idx], t2, 'label', 'align', rMin=rMin, rMax=rMax, KpMax=KpMax, verrMax=verrMax,
            qscale_v=qscale_v, deltaKey_v=deltaKey_v, qscale_p=qscale_p, deltaKey_p=deltaKey_p, 
            bins_delV=bins_delV, bins_delV_ratio=bins_delV_ratio)
            


def plot_compare(t1, t2, name1, name2, \
                rMin=0.4, rMax=100, KpMax=16, verrMax=2, \
                qscale_v=0.05, deltaKey_v=0.05, \
                qscale_p=0.001, deltaKey_p = 0.001, \
                bins_delV = None, bins_delV_ratio = None):

    # merge two tables into mas/yr
    t = join(t1, t2, keys='Name', table_names=['1','2'])
    x0_1 = t['x_1']
    y0_1 = t['y_1']
    t0_1 = t['t0_1']
    K_1 = t['K_1'] 
    vx_1 = t['vx_1']
    vy_1 = t['vy_1']
    vxerr_1 = t['vxerr_1']
    vyerr_1 = t['vyerr_1']
    r_1 = np.hypot(x0_1, y0_1)

    x0_2 = t['x_2']
    y0_2 = t['y_2']
    t0_2 = t['t0_2']
    K_2 = t['K_2'] 
    vx_2 = t['vx_2']
    vy_2 = t['vy_2']
    vxerr_2 = t['vxerr_2']
    vyerr_2 = t['vyerr_2']

    print('among %d stars in %s, %d stars in %s, %d stars exist in both' %(len(t1), name1, len(t2), name2, len(t)))

    # stars out of rMin and K<KpMax and verr<verrMax
    idx = np.where((r_1>rMin) & (r_1<rMax) & (K_1<KpMax) & 
                    (vxerr_1<verrMax) & (vyerr_1<verrMax) &
                    (vxerr_2<verrMax) & (vyerr_2<verrMax))[0]

    print('among %d stars exist in both, %d stars are: %.1f arcsec <r< %.1f arcsec, Kp < %.1f, ve< %f mas/yr '\
            %(len(t), len(idx), rMin, rMax, KpMax, verrMax))

    # make plot
    slope1, intercept1, r_value, p_value, std_err1 = scipy.stats.linregress(vx_1[idx],vx_2[idx])
    def fx(x):
        return intercept1 + x*slope1

    slope2, intercept2, r_value, p_value, std_err2 = scipy.stats.linregress(vy_1[idx],vy_2[idx])
    def fy(x):
        return intercept2 + x*slope2

    # plot v_1 vs v_2
    plt.clf()
    plt.figure(figsize=(22,10))
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.4, hspace=0.4)
    plt.subplot(1,2,1)
    #plt.plot(vx_1, vx_2, 'o', mec='b', mfc='none')
    plt.plot(vx_1[idx], vx_2[idx], 'bo')
    vxMin = min(vx_1[idx])
    vxMax = max(vx_1[idx])
    plt.plot([vxMin, vxMax], [fx(vxMin),fx(vxMax)], 'r-', label='linear fit')
    plt.xlabel('vx(mas/yr) in %s' %name1)
    plt.ylabel('vx(mas/yr) in %s' %name2)
    plt.axis('equal')
    plt.annotate('vx\' = (%5.2f +/- %5.2f)*vx + %6.3f' %(slope1, std_err1, intercept1),\
            xy=(0.5,0.8), xycoords='axes fraction', color='r')
    plt.legend(loc='upper left')
    
    plt.subplot(1,2,2)
    #plt.plot(vy_1, vy_2, 'o', mec='b', mfc='none')
    plt.plot(vy_1[idx], vy_2[idx], 'bo')
    vyMin = min(vy_1[idx])
    vyMax = max(vy_1[idx])
    plt.plot([vyMin, vyMax],[fy(vyMin),fy(vyMax)], 'r-', label='linear fit')
    plt.xlabel('vy(mas/yr) in %s' %name1)
    plt.ylabel('vy(mas/yr) in %s' %name2)
    plt.axis('equal')
    plt.annotate('vy\' = (%5.2f +/- %5.2f)*vy + %6.3f' %(slope2, std_err2, intercept2), \
            xy=(0.5,0.8), xycoords='axes fraction', color='r')
    plt.legend(loc='upper left')
    plt.savefig('plots/compare_%s_%s_v.png' %(name1, name2), format='png')


    # quiver plot of velocity difference between label.dat and polyfit
    delta_vx = (vx_2 - vx_1)
    delta_vy = (vy_2 - vy_1)
    delta_vx_ratio = (vx_2 - vx_1)/np.sqrt(vxerr_2**2 + vxerr_1**2)
    delta_vy_ratio = (vy_2 - vy_1)/np.sqrt(vyerr_2**2 + vyerr_1**2)

    xx = x0_1[idx]
    yy = y0_1[idx]
    delta_vxx = delta_vx[idx]
    delta_vyy = delta_vy[idx]
    delta_v = np.sqrt(delta_vxx**2 + delta_vyy**2)
    delta_vxx_ratio = delta_vx_ratio[idx]
    delta_vyy_ratio = delta_vy_ratio[idx]

    #idx1 = np.where(delta_v < 1)[0]
    #idx2 = np.where(delta_v >= 1)[0]
    plt.clf()
    plt.figure(figsize=(10,10))
    #plt.quiver(xx[idx2], yy[idx2], delta_vxx[idx2], delta_vyy[idx2], units='inches', scale=qscale_v, color='r')
    #plt.quiver(xx[idx1], yy[idx1], delta_vxx[idx1], delta_vyy[idx1], units='inches', scale=qscale_v, color='r')
    Q = plt.quiver(xx, yy, delta_vxx*(-1.), delta_vyy, units='inches', scale=qscale_v, color='r')
    plt.quiverkey(Q, 0.8, 0.9, deltaKey_v, r'$v_{%s}-v_{%s}$ =  %.2f mas/yr ' %(name2, name1, deltaKey_v), color='b', labelcolor='b')
    #plt.quiver([0,min(x0_label)], [0,max(y0_label)+1], [0, deltaKey_v], [0,0], units='inches', scale = qscale_v, color='b')
    #plt.text(min(x0_label), max(y0_label)+1.5, r'$v_{align}-v_{label}$ =  %.2f (mas/yr) ' %(deltaKey_v), color = 'b')
    plt.xlabel('x(parsec)')
    plt.ylabel('y(parsec)')
    plt.title(r'$v_{%s} - v_{%s}$' %(name2, name1))
    plt.axis('equal')
    plt.xlim(-6,6)
    plt.ylim(-6,6)
    plt.gca().invert_xaxis()
    plt.savefig('plots/compare_%s_%s_v_quiver.png' %(name1, name2), format='png')
    
    # histogram of velocity difference
    plt.figure(figsize=(22,10))
    plt.subplot(121)
    if bins_delV_ratio == None:
        bins_delV_ratio = np.linspace(-5,5,20)
    plt.hist(delta_vxx_ratio, label='vx', bins=bins_delV_ratio, alpha=0.5, histtype='step', linewidth = 2)
    plt.hist(delta_vyy_ratio, label='vy', bins=bins_delV_ratio, alpha=0.5, histtype='step', linewidth = 2)
    plt.xlabel(r'$(V_{%s} - V_{%s})$/${\sigma}_V$' %(name2, name1))
    plt.legend()
    plt.xlim(bins_delV_ratio[0],bins_delV_ratio[-1])

    plt.subplot(122)
    if bins_delV == None:
        bins_delV = np.linspace(-0.3,0.3,20)
    plt.hist(delta_vxx, label='vx', bins=bins_delV, alpha=0.5, histtype='step', linewidth = 2)
    plt.hist(delta_vyy, label='vy', bins=bins_delV, alpha=0.5, histtype='step', linewidth = 2)
    plt.xlabel(r'$(V_{%s} - V_{%s})$(mas/yr)'%(name2, name1))
    plt.legend()
    plt.xlim(bins_delV[0],bins_delV[-1])
    plt.savefig('plots/compare_%s_%s_v_hist.png' %(name1, name2), format='png')

    # quiver plot of position difference between label.dat and polyfit
    #time = 2009.5
    #fitx_label = x0_label + vx_label*0.001 * (time - t0_label)
    #fity_label = y0_label + vy_label*0.001 * (time - t0_label)
    #fitx_align = x0_align + vx_align*0.001 * (time - t0_align)
    #fity_align = y0_align + vy_align*0.001 * (time - t0_align)

    #delta_x = (fitx_align - fitx_label)
    #delta_y = (fity_align - fity_label)

    #delta_xx = delta_x[idx]
    #delta_yy = delta_y[idx]
    #delta_p = np.sqrt(delta_xx**2 + delta_yy**2) 
    #idx1 = np.where(delta_p < deltaKey_p)[0]
    #idx2 = np.where(delta_p >= deltaKey_p)[0]


    #plt.clf()
    #plt.figure(figsize=(10,10))
    #plt.quiver(xx[idx2], yy[idx2], delta_xx[idx2], delta_yy[idx2], units='inches', scale=qscale_p, color='lightpink')
    #plt.quiver(xx[idx1], yy[idx1], delta_xx[idx1], delta_yy[idx1], units='inches', scale=qscale_p, color='r')
    #plt.xlabel('x(parsec)')
    #plt.ylabel('y(parsec)')
    #plt.title(r'$P_{align} - P_{label}$')
    #plt.axis('equal')
    #plt.quiver([0,min(x0_label)], [0,max(y0_label)], [0, deltaKey_p], [0,0], units='inches', scale = qscale_p, color='b')
    #plt.text(min(x0_label), max(y0_label)+0.5, r'$P_{align}-P_{label}$ =  %.2f (mas) ' %(deltaKey_p*1000), color = 'b')
    #plt.annotate( 'align_error=%.3f mas' %(alignErr), xy=(0.7,0.95), xycoords='axes fraction',color='b')
    #plt.savefig('plots/compare_pos_quiver.png', format='png')
    
    plt.close('all')
    return
    
    
def compare_label_align_accel(label_file = 'source_list/absolute_refs.dat',\
                align='align/align_d_rms_1000_abs_t', poly='polyfit_4_trim/fit', \
                rMin=0.4, rMax=100, KpMax=16, aerrMax=0.2, \
                qscale_a=0.05, deltaKey_a=0.05, \
                bins_delA = None, bins_delA_ratio = None):
    """
    this function is used to compare between alignment and label.dat
    acceleration difference. WILL ONLY WORK WITH ACCEL ALIGN.
    Input:
        align: the align file that is used to read alignment file
        label_file: label.dat file path
        Stars that are used in comparison
            rMin, rMax: radius range (arcsec)
            KpMax:      magnitude cut
            verrMax:    velocity error less than verrMax (mas/yr)
        deltaKey_a: arrow key in acceleration quiver plot
        qscale_a: change the arrow size in acceleration quiver plot
        bins_delA: acceleration difference histogram bins range
        bins_delA_ratio: acceleration difference weighted by error histogram bins range

    Output:
        acceleration comparison
        quiver plot of (x, y, del_vx, del_vy)
    """
    ###############################==
    # read tables
    ###############################=
    # read label.dat vel in arcsec & mas/yr
    t1 = Table.read(label_file, format ='ascii')
    cols = t1.colnames
    t1.rename_columns(cols, ['Name','K','x','y','xerr','yerr','vx','vy','vxerr','vyerr','ax','ay','axerr','ayerr','t0','use','r'])
    
    # remove verr
    idx = np.where(t1['vxerr']!=0)[0]

    # read align in arcsec & arcsec/yr and convert it to arcsec & mas/yr
    s = starset.StarSet(align)
    s.loadPolyfit(poly, arcsec=1, accel=1)
    t2 = Table()
    t2['Name'] = s.getArray('name')
    t2['K'] = s.getArray('mag') * 1.0
    t2['x'] = s.getArray('fitXa.p')
    t2['y'] = s.getArray('fitYa.p')
    t2['xerr'] = s.getArray('fitXa.perr')
    t2['yerr'] = s.getArray('fitYa.perr')
    t2['vx'] = s.getArray('fitXa.v')*1000.
    t2['vy'] = s.getArray('fitYa.v')*1000.
    t2['vxerr'] = s.getArray('fitXa.verr')*1000.
    t2['vyerr'] = s.getArray('fitYa.verr')*1000.
    t2['ax'] = s.getArray('fitXa.a')*1000.
    t2['ay'] = s.getArray('fitYa.a')*1000.
    t2['axerr'] = s.getArray('fitXa.aerr')*1000.
    t2['ayerr'] = s.getArray('fitYa.aerr')*1000.
    t2['t0'] = s.getArray('fitXv.t0')
    #t2.write('plots/polyfit.dat', format='ascii.fixed_width')

    plot_compare_accel(t1[idx], t2, 'label', 'align', rMin=rMin, rMax=rMax, KpMax=KpMax, aerrMax=aerrMax,
            qscale_a=qscale_a, deltaKey_a=deltaKey_a,
            bins_delA=bins_delA, bins_delA_ratio=bins_delA_ratio)
            
            
            
def plot_compare_accel(t1, t2, name1, name2, \
                rMin=0.4, rMax=100, KpMax=16, aerrMax=0.2, \
                qscale_a=0.05, deltaKey_a=0.05, \
                bins_delA = None, bins_delA_ratio = None):

    # merge two tables into mas/yr
    t = join(t1, t2, keys='Name', table_names=['1','2'])
    x0_1 = t['x_1']
    y0_1 = t['y_1']
    t0_1 = t['t0_1']
    K_1 = t['K_1']
    ax_1 = t['ax_1']
    ay_1 = t['ay_1']
    axerr_1 = t['axerr_1']
    ayerr_1 = t['ayerr_1']
    r_1 = np.hypot(x0_1, y0_1)

    x0_2 = t['x_2']
    y0_2 = t['y_2']
    t0_2 = t['t0_2']
    K_2 = t['K_2']
    ax_2 = t['ax_2']
    ay_2 = t['ay_2']
    axerr_2 = t['axerr_2']
    ayerr_2 = t['ayerr_2']

    print('among %d stars in %s, %d stars in %s, %d stars exist in both' %(len(t1), name1, len(t2), name2, len(t)))

    # stars out of rMin and K<KpMax and verr<verrMax
    idx = np.where((r_1>rMin) & (r_1<rMax) & (K_1<KpMax) &
                    (axerr_1<aerrMax) & (ayerr_1<aerrMax) &
                    (axerr_2<aerrMax) & (ayerr_2<aerrMax))[0]

    print('among %d stars exist in both, %d stars are: %.1f arcsec <r< %.1f arcsec, Kp < %.1f, ae< %f mas/yr^2 '\
            %(len(t), len(idx), rMin, rMax, KpMax, aerrMax))

    # make plot
    slope1, intercept1, r_value, p_value, std_err1 = scipy.stats.linregress(ax_1[idx],ax_2[idx])
    def fx(x):
        return intercept1 + x*slope1

    slope2, intercept2, r_value, p_value, std_err2 = scipy.stats.linregress(ay_1[idx],ay_2[idx])
    def fy(x):
        return intercept2 + x*slope2

    # plot a_1 vs a_2
    plt.clf()
    plt.figure(figsize=(22,10))
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.4, hspace=0.4)
    plt.subplot(1,2,1)
    #plt.plot(vx_1, vx_2, 'o', mec='b', mfc='none')
    plt.plot(ax_1[idx], ax_2[idx], 'bo')
    axMin = min(ax_1[idx])
    axMax = max(ax_1[idx])
    plt.plot([axMin, axMax], [fx(axMin),fx(axMax)], 'r-', label='linear fit')
    plt.xlabel('ax(mas/yr^2) in %s' %name1)
    plt.ylabel('ax(mas/yr^2) in %s' %name2)
    plt.axis('equal')
    plt.annotate('ax\' = (%5.2f +/- %5.2f)*ax + %6.3f' %(slope1, std_err1, intercept1),\
            xy=(0.5,0.8), xycoords='axes fraction', color='r')
    plt.legend(loc='upper left')
    
    plt.subplot(1,2,2)
    #plt.plot(vy_1, vy_2, 'o', mec='b', mfc='none')
    plt.plot(ay_1[idx], ay_2[idx], 'bo')
    ayMin = min(ay_1[idx])
    ayMax = max(ay_1[idx])
    plt.plot([ayMin, ayMax],[fy(ayMin),fy(ayMax)], 'r-', label='linear fit')
    plt.xlabel('ay(mas/yr^2) in %s' %name1)
    plt.ylabel('ay(mas/yr^2) in %s' %name2)
    plt.axis('equal')
    plt.annotate('ay\' = (%5.2f +/- %5.2f)*ay + %6.3f' %(slope2, std_err2, intercept2), \
            xy=(0.5,0.8), xycoords='axes fraction', color='r')
    plt.legend(loc='upper left')
    plt.savefig('plots/compare_%s_%s_a.png' %(name1, name2), format='png')


    # quiver plot of velocity difference between label.dat and polyfit
    delta_ax = (ax_2 - ax_1)
    delta_ay = (ay_2 - ay_1)
    delta_ax_ratio = (ax_2 - ax_1)/np.sqrt(axerr_2**2 + axerr_1**2)
    delta_ay_ratio = (ay_2 - ay_1)/np.sqrt(ayerr_2**2 + ayerr_1**2)

    xx = x0_1[idx]
    yy = y0_1[idx]
    delta_axx = delta_ax[idx]
    delta_ayy = delta_ay[idx]
    delta_a = np.sqrt(delta_axx**2 + delta_ayy**2)
    delta_axx_ratio = delta_ax_ratio[idx]
    delta_ayy_ratio = delta_ay_ratio[idx]

    #idx1 = np.where(delta_v < 1)[0]
    #idx2 = np.where(delta_v >= 1)[0]
    plt.clf()
    plt.figure(figsize=(10,10))
    #plt.quiver(xx[idx2], yy[idx2], delta_vxx[idx2], delta_vyy[idx2], units='inches', scale=qscale_v, color='r')
    #plt.quiver(xx[idx1], yy[idx1], delta_vxx[idx1], delta_vyy[idx1], units='inches', scale=qscale_v, color='r')
    Q = plt.quiver(xx, yy, delta_axx*(-1.), delta_ayy, units='inches', scale=qscale_a, color='r')
    plt.quiverkey(Q, 0.8, 0.9, deltaKey_a, r'$a_{%s}-a_{%s}$ =  %.2f mas/yr^2 ' %(name2, name1, deltaKey_a), color='b', labelcolor='b')
    #plt.quiver([0,min(x0_label)], [0,max(y0_label)+1], [0, deltaKey_v], [0,0], units='inches', scale = qscale_v, color='b')
    #plt.text(min(x0_label), max(y0_label)+1.5, r'$v_{align}-v_{label}$ =  %.2f (mas/yr) ' %(deltaKey_v), color = 'b')
    plt.xlabel('x(parsec)')
    plt.ylabel('y(parsec)')
    plt.title(r'$a_{%s} - a_{%s}$' %(name2, name1))
    plt.axis('equal')
    plt.xlim(-6,6)
    plt.ylim(-6,6)
    plt.gca().invert_xaxis()
    plt.savefig('plots/compare_%s_%s_a_quiver.png' %(name1, name2), format='png')
    
    # histogram of velocity difference
    plt.figure(figsize=(22,10))
    plt.subplot(121)
    if bins_delA_ratio == None:
        bins_delA_ratio = np.linspace(-5,5,20)
    plt.hist(delta_axx_ratio, label='ax', bins=bins_delA_ratio, alpha=0.5, histtype='step', linewidth = 2)
    plt.hist(delta_ayy_ratio, label='ay', bins=bins_delA_ratio, alpha=0.5, histtype='step', linewidth = 2)
    plt.xlabel(r'$(A_{%s} - A_{%s})$/${\sigma}_V$' %(name2, name1))
    plt.legend()
    plt.xlim(bins_delA_ratio[0],bins_delA_ratio[-1])

    plt.subplot(122)
    if bins_delA == None:
        bins_delA = np.linspace(-0.3,0.3,20)
    plt.hist(delta_axx, label='ax', bins=bins_delA, alpha=0.5, histtype='step', linewidth = 2)
    plt.hist(delta_ayy, label='ay', bins=bins_delA, alpha=0.5, histtype='step', linewidth = 2)
    plt.xlabel(r'$(A_{%s} - A_{%s})$(mas/yr)'%(name2, name1))
    plt.legend()
    plt.xlim(bins_delA[0],bins_delA[-1])
    plt.savefig('plots/compare_%s_%s_a_hist.png' %(name1, name2), format='png')

    
    
    plt.close('all')
    return



# compare the align_velocity in two different align
def compare_aligns(align1,poly1, align2, poly2,
                align1_name=None, align2_name=None,\
                rMin=0.4, rMax=100, KpMax=16, verrMax=1, \
                alignErr=None, bins_delV = None, bins_delV_ratio = None,\
                qscale_v=0.1, deltaKey_v=0.001, qscale_p=0.1, deltaKey_p=0.1):
    """
    this function is used to compare alignment velocity with velocity in lable.dat.
    Input:
        goodStars that are used in the next comparison
            verrMax:    velocity error less than verrMax (mas/yr)
            rMin, rMax: radius range (arcsec)
            KpMax:      magnitude cut
        qscale_v: change the arrow size in velocity quiver plot
        deltaKey_v: arrow key in velocity quiver plot
        qscale_p: change the arrow size in postion quiver plot
        deltaKey_p: arrow key in position quiver plot
    Output:
        velocity comparison
        quiver plot of (x, y, del_vx, del_vy)
        quiver plot of (x, y, del_x, del_y)
    """
    if align1_name == None:
        align1_name = align1
    if align2_name == None:
        align2_name = align2
    # read align1 and align2 in arcsec and mas/yr
    t1 = Table()
    s1 = starset.StarSet(align1)
    s1.loadPolyfit(poly1, arcsec=1)
    t1['Name'] = s1.getArray('name')
    t1['K'] = s1.getArray('mag') * 1.0
    t1['x'] = s1.getArray('fitXv.p')
    t1['y'] = s1.getArray('fitYv.p')
    t1['xerr'] = s1.getArray('fitXv.perr')
    t1['yerr'] = s1.getArray('fitYv.perr')
    t1['vx'] = s1.getArray('fitXv.v')*1000.
    t1['vy'] = s1.getArray('fitYv.v')*1000.
    t1['vxerr'] = s1.getArray('fitXv.verr')*1000.
    t1['vyerr'] = s1.getArray('fitYv.verr')*1000.
    t1['t0'] = s1.getArray('fitXv.t0')
    

    t2 = Table()
    s2 = starset.StarSet(align2)
    s2.loadPolyfit(poly2, arcsec=1)
    t2['Name'] = s2.getArray('name')
    t2['K'] = s2.getArray('mag') * 1.0
    t2['x'] = s2.getArray('fitXv.p')
    t2['y'] = s2.getArray('fitYv.p')
    t2['xerr'] = s2.getArray('fitXv.perr')
    t2['yerr'] = s2.getArray('fitYv.perr')
    t2['vx'] = s2.getArray('fitXv.v')*1000.
    t2['vy'] = s2.getArray('fitYv.v')*1000.
    t2['vxerr'] = s2.getArray('fitXv.verr')*1000.
    t2['vyerr'] = s2.getArray('fitYv.verr')*1000.
    t2['t0'] = s2.getArray('fitXv.t0')

    plot_compare(t1, t2, align1_name, align2_name, rMin=rMin, rMax=rMax, KpMax=KpMax, verrMax=verrMax,
            qscale_v=qscale_v, deltaKey_v=deltaKey_v, qscale_p=qscale_p, deltaKey_p=deltaKey_p, 
            bins_delV=bins_delV, bins_delV_ratio=bins_delV_ratio)

def compare_labels(label1, label2, label1_name, label2_name,
                rMin=0.4, rMax=100, KpMax=16, verrMax=1, 
                alignErr=None, bins_delV = None, bins_delV_ratio = None,
                qscale_v=0.1, deltaKey_v=0.001, qscale_p=0.1, deltaKey_p=0.1):

    t1 = Table.read(label1, format ='ascii',\
          names=('Name','K','x','y','xerr','yerr','vx','vy','vxerr','vyerr','t0','use','r'))
    t2 = Table.read(label2, format ='ascii',\
          names=('Name','K','x','y','xerr','yerr','vx','vy','vxerr','vyerr','t0','use','r'))

    # remove things without ve
    idx1 = np.where(t1['vxerr']!=0)[0]
    idx2 = np.where(t2['vxerr']!=0)[0]

    plot_compare(t1[idx1], t2[idx2], label1_name, label2_name, rMin=rMin, rMax=rMax, KpMax=KpMax, verrMax=verrMax,
            qscale_v=qscale_v, deltaKey_v=deltaKey_v, qscale_p=qscale_p, deltaKey_p=deltaKey_p, 
            bins_delV=bins_delV, bins_delV_ratio=bins_delV_ratio)

    t = join(t1, t2, keys='Name', table_names=['1','2'])
    x0_1 = t['x_1']
    y0_1 = t['y_1']
    x0e_1 = t['xerr_1']
    y0e_1 = t['yerr_1']
    t0_1 = t['t0_1']
    K_1 = t['K_1'] 
    vx_1 = t['vx_1']
    vy_1 = t['vy_1']
    vxe_1 = t['vxerr_1']
    vye_1 = t['vyerr_1']
    r_1 = np.sqrt(x0_1**2 + y0_1**2)

    x0_2 = t['x_2']
    y0_2 = t['y_2']
    x0e_2 = t['xerr_2']
    y0e_2 = t['yerr_2']
    t0_2 = t['t0_2']
    K_2 = t['K_2'] 
    vx_2 = t['vx_2']
    vy_2 = t['vy_2']
    vxe_2 = t['vxerr_2']
    vye_2 = t['vyerr_2']
    r_2 = np.sqrt(x0_2**2 + y0_2**2)

    plt.figure(figsize=(22,22))
    plt.subplot(221)
    plt.plot(K_1, x0e_1*1000, 'bo', label=label1_name)
    plt.plot(K_2, x0e_2*1000, 'ro', label=label2_name)
    plt.xlabel('mag')
    plt.ylabel('xe (mas)')
    #plt.ylim([1, 1.5])
    plt.legend()
    plt.subplot(222)
    plt.plot(K_1, y0e_1*1000, 'bo', label=label1_name)
    plt.plot(K_2, y0e_2*1000, 'ro', label=label2_name)
    plt.xlabel('mag')
    plt.ylabel('ye (mas)')
    #plt.ylim([1, 1.5])
    plt.legend()
    plt.subplot(223)
    plt.plot(K_1, vxe_1, 'bo', label=label1_name)
    plt.plot(K_2, vxe_2, 'ro', label=label2_name)
    plt.xlabel('mag')
    plt.ylabel('vxe (mas/yr)')
    plt.ylim([0, 0.5])
    plt.legend()
    plt.subplot(224)
    plt.plot(K_1, vye_1, 'bo', label=label1_name)
    plt.plot(K_2, vye_2, 'ro', label=label2_name)
    plt.xlabel('mag')
    plt.ylabel('vye (mas/yr)')
    plt.ylim([0, 0.5])
    plt.legend()
    plt.savefig('compare_abs.png', format='png')

   
def S02_offset(align_dir='mine/', anna_dir='anna_ref/', anna_orig='../anna_orig/', points='points_1000/'):
    t1 = Table.read(align_dir+points+'S0-2.points', format='ascii')
    t2 = Table.read(anna_dir+points+'S0-2.points', format='ascii')
    t3 = Table.read(anna_orig+'points_1000/'+'S0-2.points', format='ascii')
    
    # find where t1 and t3 has both been detected
    t13 = join(t1, t3, keys='col1')
    t23 = join(t2, t3, keys='col1')
    fig, ax = plt.subplots(2,1,sharex=True, figsize=(20,20))
    #ax[0].errorbar(t3['col1'], t3['col2'], t3['col4'], color='y', fmt='o', alpha=0.5, ecolor='w', label='anna_orig')
    #ax[0].errorbar(t1['col1'], t1['col2'], t1['col4'], color='r', fmt='.', alpha=0.5, ecolor='w', label='siyao_ref')
    #ax[0].errorbar(t2['col1'], t2['col2'], t2['col4'], color='g', fmt='.', alpha=0.5, ecolor='w', label='anna_ref')
    #ax[1].errorbar(t3['col1'], t3['col3'], t3['col5'], color='y', fmt='.', alpha=0.5, ecolor='w', label='anna_orig')
    #ax[1].errorbar(t1['col1'], t1['col3'], t1['col5'], color='r', fmt='.', alpha=0.5, ecolor='w', label='siyao_ref')
    #ax[1].errorbar(t2['col1'], t2['col3'], t2['col5'], color='g', fmt='.', alpha=0.5, ecolor='w', label='anna_ref')
    #ax[0].plot(t3['col1'], t3['col2'], 'o', color='y', mew=0, label='anna_orig')
    #ax[0].plot(t1['col1'], t1['col2'], 'x', color='r', mew=1, label='siyao_ref')
    #ax[0].plot(t2['col1'], t2['col2'], '+', color='g', mew=1, label='anna_ref')
    #ax[1].plot(t3['col1'], t3['col3'], 'o', color='y', mew=0, label='anna_orig')
    #ax[1].plot(t1['col1'], t1['col3'], 'x', color='r', mew=1, label='siyao_ref')
    #ax[1].plot(t2['col1'], t2['col3'], '+', color='g', mew=1, label='anna_ref')
    ax[0].plot(t13['col1'], t13['col2_1']-t13['col2_2'], 'x', color='r', mew=1, label='siyao_ref')
    ax[0].plot(t23['col1'], t23['col2_1']-t23['col2_2'], '+', color='g', mew=1, label='anna_ref')
    ax[0].axhline(0, color='k', ls='--')
    ax[1].plot(t13['col1'], t13['col3_1']-t13['col3_2'], 'x', color='r', mew=1, label='siyao_ref')
    ax[1].plot(t23['col1'], t23['col3_1']-t23['col3_2'], '+', color='g', mew=1, label='anna_ref')
    ax[1].axhline(0, color='k', ls='--')
    ax[0].set_ylabel(r'$\Delta$x(arcsec)')
    ax[0].set_title('position offset between new align and anna\'s origial align')
    ax[0].legend()
    ax[1].set_ylabel(r'$\Delta$y(arcsec)')
    ax[1].set_xlabel('t(year)')
    ax[1].legend()
    plt.savefig('S02_offset.png', format='png')

def S02_orbit(align_dir='mine/', anna_dir='anna_ref/', anna_orig='../anna_orig/',points='points_1000/'):
    t1 = Table.read(align_dir+points+'S0-2.points', format='ascii')
    t2 = Table.read(anna_dir+points+'S0-2.points', format='ascii')
    t3 = Table.read(anna_orig+points+'S0-2.points', format='ascii')
    
    # orbit para from devin
    orb = orbits.Orbit()
    orb.w = 67.27
    orb.o = 229.1
    orb.i = 134.9
    orb.e = 0.8898
    orb.p = 16.09
    orb.t0 = 2002.344
    time = np.arange(1,20,0.01)+2000
    (r1, v, a) = orb.kep2xyz(np.array(time))

    # orbit para from anna 
    orb = orbits.Orbit()
    orb.w = 66.8
    orb.o = 228
    orb.i = 134.2
    orb.e = 0.892
    orb.p = 15.92
    orb.t0 = 2002.347
    time = np.arange(1,20,0.01)+2000
    (r2, v, a) = orb.kep2xyz(np.array(time))

    # plot orbit
    plt.figure(figsize=(10,30))
    plt.clf()
    l1, = plt.plot([i[0] for i in r1],[i[1] for i in r1], 'b', label='Devin\'s analysis')
    l2, = plt.plot([i[0] for i in r2],[i[1] for i in r2], 'm', label='Anna\'s analysis')
    legend1 = plt.legend(handles=[l1, l2], loc='lower right')

    # plot points
    ms = 6
    alpha = 0.8
    mew = 1
    idx_spe = np.where((t3['col1']<2005.580) & (t3['col1']!=2005.495))[0]
    l3, = plt.plot(t3['col2']*-1., t3['col3'],'yo', ms=ms,mec='y', mfc='none', mew=mew, alpha=alpha,label='anna_orig(ao)')
    l4, = plt.plot(t3['col2'][idx_spe]*-1., t3['col3'][idx_spe],'yo', ms=ms,mec='y', mfc='y', mew=mew, alpha=alpha,label='anna_orig(spe)')
    plt.gca().add_artist(legend1)
    legend2 = plt.legend(handles=[l4,l3], loc='lower left')
    l5, = plt.plot(t3['col2'][idx_spe]*-1., t3['col3'][idx_spe],'yo', ms=ms, mec='y', mfc='y', mew=mew, alpha=alpha,label='anna_orig')

    idx_spe = np.where((t1['col1']<2005.580) & (t1['col1']!=2005.495))[0]
    l6, = plt.plot(t1['col2']*-1., t1['col3'],'r^', ms=ms,mew=mew, mfc='none', alpha=alpha,mec='r')
    l7, = plt.plot(t1['col2'][idx_spe]*-1., t1['col3'][idx_spe],'r^',ms=ms, mew=0, mfc='r', alpha=alpha,label='siyao_ref')

    idx_spe = np.where((t2['col1']<2005.580) & (t2['col1']!=2005.495))[0]
    l8, = plt.plot(t2['col2']*-1., t2['col3'],'gv', ms=ms,mew=mew, mfc='none', alpha=alpha,mec='g')
    l9, = plt.plot(t2['col2'][idx_spe]*-1., t2['col3'][idx_spe],'gv',ms=ms, mew=0, mfc='g', alpha=alpha,label='anna_ref')

    plt.gca().add_artist(legend2)
    plt.legend(handles=[l5,l7,l9], loc='upper left')
    plt.title('S0-2 orbit')
    plt.xlabel('RA (arcsec)')
    plt.ylabel('Dec (arcsec)')
    plt.gca().invert_xaxis()
    plt.axis('equal')
    plt.savefig('S02_orbit.pdf',format='pdf')
    plt.close()
    return

def compare_pe(align1, align2, align1_name, align2_name, stars, points='/points_c/'):
    """
    plot median xe/ye for align1 and align2 for stars"""
    xe1 = np.zeros(len(stars))
    ye1 = np.zeros(len(stars))
    xe2 = np.zeros(len(stars))
    ye2 = np.zeros(len(stars))

    # loop through each star
    for i in range(len(stars)):
        starName = stars[i]
        pointsTab = starTables.read_points(align1 + points + starName + '.points')
        xerr = pointsTab['xerr']
        yerr = pointsTab['yerr']
        xe1[i] = np.median(xerr)
        ye1[i] = np.median(yerr)
        pointsTab = starTables.read_points(align2 + points + starName + '.points')
        xerr = pointsTab['xerr']
        yerr = pointsTab['yerr']
        xe2[i] = np.median(xerr)
        ye2[i]  = np.median(yerr)

    # plot
    plt.clf()
    plt.plot(xe1*1000, xe2*1000, 'rx', label='xe')
    plt.plot(ye1*1000, ye2*1000, 'b+', label='ye')
    plt.xlabel(align1_name + ' median position uncertainty (mas)')
    plt.ylabel(align2_name + ' median position uncertainty (mas)')
    plt.legend(loc='best')
    plt.plot([0,1.8], [0,1.8],'k--')
    plt.savefig('pe_'+align1_name+'_'+align2_name+'.png', format='png')

def plot_pos_err(align='align/align_d_rms_1000_abs_t', poly='polyfit_4_trim/fit'):
    s = starset.StarSet(align)
    s.loadPolyfit(poly)
    pe = s.getArrayFromAllEpochs('xerr_p')*1000 #in mas
    ae = s.getArrayFromAllEpochs('xerr_a')*1000 #in mas
    nEpochs, nStars = pe.shape
    K = s.getArray('mag')
    names = s.getArray('name')

    # epoch1 
    pe1 = pe[0]
    ae1 = ae[0]
    plt.plot(K, pe1, 'bo')



    plt.figure(figsize=(10,10))
    for i in range(nEpochs):
        plt.plot(np.tile(i, nStars), pe[i])
