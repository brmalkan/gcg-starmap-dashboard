import numpy as np
import pylab as py
import math
import healpy
import pdb
import healpy.rotator as R
import healpy.projector as P

def go(mcFile, npix, ntrials, plottrial=0, plotlog=False, olddisks=False, plot_range=[0,0.024]):

    #healpixTable = open(mcFile, 'rb') 

    nside = healpy.npix2nside(npix)

    #healpix = healpixTable[:,plottrial]
    #hp = healpixTable.read()
    #healpix = np.fromstring(hp, dtype=np.float64)

    #start_arr = plottrial*npix
    #end_arr = plottrial*npix+npix
    #healpix = healpix[start_arr:end_arr]
    healpix = np.fromfile(mcFile, dtype=float)
    healpix = healpix.reshape((ntrials, npix))

    healpix = healpix[plottrial]

    if plotlog == True:
        healpix = np.log(healpix)

    # all sky pixels sorted with highest pixels first
    idx = (np.argsort(healpix))[::-1]
    sortHP = healpix[idx]

    # Cumulatively sum up the pixels
    cumHP = np.cumsum(sortHP)
    totHP = sortHP.sum()

    # Make contours
    pixIdx = np.arange(npix)
    theta, phi = healpy.pix2ang(nside, pixIdx)

    # Peak disk value
    peakPixVal = healpix.max()
    peakPixIdx = healpix.argmax()
    peakIncl = np.degrees(theta[peakPixIdx])
    peakOmeg = np.degrees(phi[peakPixIdx])
    print('Disk candidate at i = %6.2f, O = %6.2f' % (peakIncl, peakOmeg))

    # Determine the background (avg and stddev)
    avgiter = 2
    idx = np.arange(len(healpix))

    for ii in range(avgiter):
        avgval = healpix[idx].mean()
        stddev = healpix[idx].std()

        hicut = avgval + 5.0 * stddev
        locut = avgval - 5.0 * stddev

        idx = np.where((healpix > locut) & (healpix < hicut))[0]

        print('BKG: iter = %d, rejecting %d out of %d pixels' % \
               (ii, (npix - len(idx)), npix))

    print('')
    print('Background mean = %f, stddev = %f' % (avgval, stddev))
    print('Peak = %f, significance = %f' % \
          (healpix.max(), (healpix.max() - avgval) / stddev))

    # Find pixels 1, 2, 3 sigma below the peak
    idx1 = np.where(healpix > (peakPixVal - (1.0 * stddev)))[0]
    idx2 = np.where(healpix > (peakPixVal - (2.0 * stddev)))[0]
    idx3 = np.where(healpix > (peakPixVal - (3.0 * stddev)))[0]
    cnt1 = len(idx1)
    cnt2 = len(idx2)
    cnt3 = len(idx3)
    inclRng1 = np.degrees([min(theta[idx1]), max(theta[idx1])])
    inclRng2 = np.degrees([min(theta[idx2]), max(theta[idx2])])
    inclRng3 = np.degrees([min(theta[idx3]), max(theta[idx3])])
    omegRng1 = np.degrees([min(phi[idx1]), max(phi[idx1])])
    omegRng2 = np.degrees([min(phi[idx2]), max(phi[idx2])])
    omegRng3 = np.degrees([min(phi[idx3]), max(phi[idx3])])

    fmt = '%d sigma ranges: i = [%5.1f - %5.1f]  o = [%5.1f - %5.1f] (N=%d)'
    print(fmt % (1, inclRng1[0], inclRng1[1], omegRng1[0], omegRng1[1], cnt1))
    print(fmt % (2, inclRng2[0], inclRng2[1], omegRng2[0], omegRng2[1], cnt2))
    print(fmt % (3, inclRng3[0], inclRng3[1], omegRng3[0], omegRng3[1], cnt3))

    # Now display gaussian fits along inclination and Omega to get
    # estimate of the disk thickness
    iidx = np.where((np.degrees(theta) > (peakIncl - 45.0)) &
                    (np.degrees(theta) < (peakIncl + 45.0)) &
                    (np.degrees(phi) > (peakOmeg - 1.0)) &
                    (np.degrees(phi) < (peakOmeg + 1.0)))[0]
    oidx = np.where((np.degrees(phi) > (peakOmeg - 45.0)) & 
                    (np.degrees(phi) < (peakOmeg + 45.0)) & 
                    (np.degrees(theta) > (peakIncl - 1.0)) &
                    (np.degrees(theta) < (peakIncl + 1.0)))[0]

    #py.clf()
    #py.plot(np.degrees(theta[iidx]),healpix[iidx],'r.')
    #py.plot(np.degrees(phi[oidx]),healpix[oidx],'b.')
    #py.show()

    # Plot the previously proposed disks
    if olddisks == True:
        # Bartko doesn't give an error or thickness for the CCW structure, just
        # says 'U-shaped'. So let's arbitrarily assign it 15 deg error
        # For the CCW disk thickness, use Paumard's estimate since
        # Bartko just says 'extended U-shaped'
        i = [129.0, 38.0] # CW then CCW from Bartko+09
        i_err = [3.0, 15.0] # CW err from B09, CCW err unknown
        i_thick = [18.0, 14.0] # CW thickness from B09, CCW from P06
        Om = [98.0, 160.0]
        Om_err = [3.0, 15.0]
        Om_thick = [18.0, 19.0]
        #disks = {'s1': ellipseOutline(i[0], Om[0], i_err[0], Om_err[0]), 
        #         's2': ellipseOutline(i[1], Om[1], i_err[1], Om_err[1]),
        #         's3': ellipseOutline(i[0], Om[0], i_thick[0], Om_thick[0], line='--'),
        #         's4': ellipseOutline(i[1], Om[1], i_thick[1], Om_thick[1], line='--')}
        s1 = ellipseOutline(i[0], Om[0], i_err[0], Om_err[0]) # bartko09 CW
        s2 = ellipseOutline(i[1], Om[1], i_err[1], Om_err[1]) # bartko09 CCW
        s3 = ellipseOutline(i[0], Om[0], i_thick[0], Om_thick[0])
        s4 = ellipseOutline(i[1], Om[1], i_thick[1], Om_thick[1])

        ourIO = [130.2, 96.3]
        ourIOerr = [3.0, 3.0]
        ourIOthick = [15.0, 15.0]
        s1ours = ellipseOutline(ourIO[0], ourIO[1], ourIOerr[0], ourIOerr[1]) # Yelda12 CW
        s2ours = ellipseOutline(ourIO[0], ourIO[1], ourIOthick[0], ourIOthick[1]) # Yelda12 CW

    # Plot up the orientation of the orbital plane for this star
    py.clf()
    py.figure(figsize=(8,6))
    healpy.mollview(healpix, 1, coord=['C','C'], flip='geo', rot=[0,180],
                    cmap=py.cm.gist_stern_r,title='i=180',
                    min=plot_range[0],max=plot_range[1],
                    #min=0.001,max=0.024,
                    #min=0.000252756,max=0.00202247,
                    #min=0.000303029,max=0.0139865,
                    #min=0.00116216,max=0.0240521,
                    notext=True)
    py.rcParams.update({'font.size': 12})
    healpy.graticule(dpar=30,dmer=45)
    if olddisks == True:
        #healpy.projplot(s1['ra'], s1['dec'], lonlat=True,
        #                direct=False, color='k', linewidth=1.5)
        #healpy.projplot(s2['ra'], s2['dec'], lonlat=True,
        #                direct=False, color='k', linewidth=1.5)
        #healpy.projplot(s3['ra'], s3['dec'], lonlat=True,
        #                direct=False, color='k', linewidth=1.5,
        #                linestyle='--')
        #healpy.projplot(s4['ra'], s4['dec'], lonlat=True,
        #                direct=False, color='k', linewidth=1.5,
        #                linestyle='--')
        healpy.projplot(s1ours['ra'], s1ours['dec'], lonlat=True,
                        direct=False, color='w', linewidth=1.5,
                        linestyle='-')
        #healpy.projplot(s2ours['ra'], s2ours['dec'], lonlat=True,
        #                direct=False, color='w', linewidth=1.5,
        #                linestyle='--')

    fs = 12
    py.text(0.02, -0.98, '0', fontsize=fs)
    py.text(0.02, -0.73, '30', fontsize=fs)
    py.text(0.02, -0.38, '60', fontsize=fs)
    py.text(0.02, 0.02, '90', fontsize=fs)
    py.text(0.00, 0.43, '120', fontsize=fs)
    py.text(0.00, 0.80, '150', fontsize=fs)
    
    py.text(-1.07, 0.02, 'N', fontsize=fs)
    py.text(-0.07, 0.02, 'W', fontsize=fs)
    py.text(0.93, 0.02, 'S', fontsize=fs)
    py.text(1.93, 0.02, 'E', fontsize=fs)

    py.savefig(mcFile + '.png')
    py.close()


def ellipseOutline(i, o, irad, orad, line='-'):
    
    # Go through 360.0 degrees and find the coordinates for each angle
    binsize = 10
    bincnt = 360 / binsize
    
    #angle = findgen(bincnt + 1) * binsize
    angle = np.arange(bincnt + 1) * binsize
    angle = angle * math.pi / 180.0
    
    x = np.zeros((len(angle)), dtype=float)
    y = np.zeros((len(angle)), dtype=float)
    for aa in range(len(angle)):
        x[aa] = o + (orad * math.sin(angle[aa]))
        y[aa] = i + (irad * math.cos(angle[aa]))

    y = 90.0 - y

    ellipse = {'coord': 'C', 'ra': x, 'dec': y, 'line': line}
    #ellipse = [x, y]

    return ellipse


