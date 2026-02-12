import os, sys
from numpy import *
import numpy as np
from gcwork import starTables as tabs
from gcwork import starset
from gcwork import objects
import sqlite3 as sqlite
import pdb

# temporary fix for MySQLdb so that other code can run
try:
    import MySQLdb as mdb
except ImportError:
    pass    

def youngStarNames():
    """
    Loads list of known young stars from database, which is
    continuously updated (per T. Do).
    """
    #dbfile = '/u/ghezgroup/data/gc/database/stars.sqlite'
    pw = open('/u/lucy/scratch/other/pw.txt').read().split()[0]

    # Create a connection to the database file
    #connection = sqlite.connect(dbfile)
    con = mdb.connect(host='169.232.151.194',user='dbread',passwd=pw,db='gcg')

    # Create a cursor object
    cur = con.cursor()

    # First look in our stars table
    #cur.execute('SELECT * FROM stars WHERE name NOT IN ("S3-3","S10-34","S2-76","S3-190","S7-30")')
    #cur.execute('SELECT * FROM stars WHERE name NOT IN ("S1-22","S1-24")')
    cur.execute('SELECT * FROM stars')
    yngNames = []
    for row in cur:
        yng = row[2]
        if (yng == 'T'):
            yngNames = np.concatenate([yngNames,[row[0]]])
        
    #skip = ['S2-66', 'S1-22', 'S1-24','irs13E3b']

    # Next look in Bartko et al. 2009 table
    # If they are in this table, they are young
    cur.execute('SELECT * FROM bartko2009')
    for row in cur:
        yng = row[1]
        yngNames = np.concatenate([yngNames,[yng]])

    # Next look in Paumard et al. 2006 table
    # If they are in this table, they are young
    cur.execute('SELECT * FROM paumard2006')
    for row in cur:
        yng = row[1]
        # Skip S2-66; Paumard calls it young, but we've
        # ID'ed it as old, and we take precedence.
        # Paumard also has irs13E3b as a source of unknown spectral type,
        # but Fritz et al. (2010) showed that irs13E3 is a concentration of
        # warm dust and gas. This is why we can't get a sensible proper motion
        # or orbit for this source. So we exclude it since it isn't a star.
        if ((yng == 'S2-66') | (yng == 'irs13E3b')):
            continue
        yngNames = np.concatenate([yngNames,[yng]])

    return np.unique(yngNames)


def loadYoungStars(root, align='align/align_d_rms_1000_abs_t',
                   fit='polyfit_c/fit', points='points_c/',
                   radiusCut=0.8, relErr=1, verbose=False,
                   withRVonly=False, silent=False, mosaic=False,
                   skipStar=['']):
    """
    skipStar: Option to not include a particular star; can pass
              in multiple stars as a list
    """

    if not os.path.exists(root + align + '.trans'):
        align = 'align/align_d_rms_100_abs_t'
        #align = 'align/align_d_rms1000_t'
        
    # Load up the absolute position of Sgr A* based on S0-2's orbit
    t = objects.Transform()
    t.loadAbsolute()

    # Load up position/velocity information
    s = starset.StarSet(root + align, relErr=relErr, trans=t)
    s.loadPolyfit(root + fit, arcsec=1, silent=silent)
    if mosaic == False:
        s.loadPolyfit(root + fit, arcsec=1, accel=1, silent=silent)

    yng = youngStarNames()
    # Tables in database w/ RV information
    ucla = tabs.UCLAstars()
    bart = tabs.Bartko2009()
    paum = tabs.Paumard2006()

    cc = objects.Constants()

    # Pull out from the set of stars those that are young
    # stars.
    stars = []
    names = [star.name for star in s.stars]

    for name in yng:
        if name in skipStar:
            continue

    # Find the star in our star lists
    try:
        idx = names.index(name)
        star = s.stars[idx]

        if (star.r2d >= radiusCut):
            stars.append(star)

            # Number of Epochs Detected should be corrected
            # for epochs trimmed out of the *.points files.
            pntsFile = '%s%s%s.points' % (root, points, star.name)
            _pnts = tabs.read_points(pntsFile)
            star.velCnt = len(_pnts)

            pntDate = _pnts['epoch']
            pntX = _pnts['x']
            pntY = _pnts['y']
            pntXe = _pnts['xerr']
            pntYe = _pnts['yerr']

            # Load up data from the points files.
            for ee in range(len(star.years)):
                tt = (where(abs(pntDate - star.years[ee]) < 0.001))[0]

                if (len(tt) == 0):
                    star.e[ee].x_pnt = -1000.0
                    star.e[ee].y_pnt = -1000.0
                    star.e[ee].xe_pnt = -1000.0
                    star.e[ee].ye_pnt = -1000.0
                else:
                    tt = tt[0]
                    star.e[ee].x_pnt = pntX[tt]
                    star.e[ee].y_pnt = pntY[tt]
                    star.e[ee].xe_pnt = pntXe[tt]
                    star.e[ee].ye_pnt = pntYe[tt]

            if mosaic == True: # We only have 3 epochs (as of 2011)
                star.fitXv.v *= cc.asy_to_kms
                star.fitYv.v *= cc.asy_to_kms
                star.fitXv.verr *= cc.asy_to_kms
                star.fitYv.verr *= cc.asy_to_kms

                star.vx = star.fitXv.v
                star.vy = star.fitYv.v
                star.vxerr = star.fitXv.verr
                star.vyerr = star.fitYv.verr
            else:
                star.fitXa.v *= cc.asy_to_kms
                star.fitYa.v *= cc.asy_to_kms
                star.fitXa.verr *= cc.asy_to_kms
                star.fitYa.verr *= cc.asy_to_kms

                star.vx = star.fitXa.v
                star.vy = star.fitYa.v
                star.vxerr = star.fitXa.verr
                star.vyerr = star.fitYa.verr
                

            star.rv_ref = 'None'

            def other_RV_tables():
                # If not found in UCLA tables, then check Bartko+2009
                idx = np.where(bart.ourName == name)[0]
                if len(idx) > 0:
                    star.vz = bart.vz[idx][0] 
                    star.vzerr = bart.vzerr[idx][0]
                    star.vzt0 = bart.t0_spectra[idx][0] 
                    star.rv_ref = 'Bartko+2009'
                    # Bartko took some RVs from Paumard, so we should
                    # give Paumard credit for these
                    temp = np.where(paum.ourName == name)[0]
                    if len(temp) > 0:
                        if ((star.vz == paum.vz[temp][0]) & (star.vzerr == paum.vzerr[temp][0])):
                            star.rv_ref += ' Paumard+2006'
                else:
                    # Next check Paumard+2006
                    idx = np.where(paum.ourName == name)[0]
                    if len(idx) > 0:
                        star.vz = paum.vz[idx][0] 
                        star.vzerr = paum.vzerr[idx][0]
                        star.vzt0 = paum.t0_spectra[idx][0]
                        star.altName = paum.name[idx][0]
                        star.rv_ref = 'Paumard+2006'
                    #else:
                    #    print( 'Could not find radial velocity for %s' % name)

            # Find the radial velocity for each star
            # First look in OSIRIS data, then Bartko, then Paumard
            idx = np.where(ucla.ourName == name)[0]
            if len(idx) > 0:
                if name == 'S1-24':
                    # Get the RV from the stars table; this star was in only one
                    # frame and so does not have errors. Given its high SNR, we'll
                    # estimate the RV error using Tuan's formula relating SNR to vz_err
                    dbfile = '/u/ghezgroup/data/gc/database/stars.sqlite'
                    # Create a connection to the database file
                    connection = sqlite.connect(dbfile)
                    # Create a cursor object
                    cur = connection.cursor()
                    cur.execute('SELECT name,vlsr,snr,ddate FROM spectra WHERE name in ("S1-24")')
                    for row in cur:
                        star.vz = row[1]
                        snr = row[2]
                        star.vzt0 = row[3]
                        star.vzerr = 175.4*snr**(-0.367)
                        star.rv_ref = 'UCLA'
                        cur.close()
                else:
                    star.vz = ucla.vz[idx][0]
                    star.vzerr = ucla.vzerr[idx][0]
                    star.vzt0 = ucla.t0_spectra[idx][0]
                    star.rv_ref = 'UCLA'

                # Temporary:
                #if name == 'S1-1':
                #    dbfile = '/u/ghezgroup/data/gc/database/stars.sqlite'
                #    # Create a connection to the database file
                #    connection = sqlite.connect(dbfile)
                #    # Create a cursor object
                #    cur = connection.cursor()
                #    cur.execute('SELECT name,vlsr,snr,ddate FROM spectra WHERE name in ("S1-1")')
                #    for row in cur:
                #        star.vz = 8.5 # From Kim's latest RV measurement
                #        snr = row[2]
                #        star.vzt0 = row[3]
                #        star.vzerr = 175.4*snr**(-0.367)
                #        star.rv_ref = 'UCLA'
                #        cur.close()
                #if name == 'S2-22':
                #    star.vz = 56.3
                #    star.vzerr = 30.2
                #    star.vzt0 = 2007.54
                #    star.rv_ref = 'UCLA'
                #if name == 'S2-19':
                #    star.vz = -78.4
                #    star.vzerr = 31.4
                #    star.vzt0 = 2008.44
                #    star.rv_ref = 'UCLA'
                #    
                # End temporary

                # A star could be in the stars table but still not have vz
                if (star.vz == None) | (star.vz == ''): # then check other tables
                    star.rv_ref = None
                    other_RV_tables()
            else:
                other_RV_tables()

            if withRVonly == True:
                if star.rv_ref == None:
                    # remove this star
                    stars.remove(star)

            if (verbose == True):
                print( 'Matched %15s to %12s' % (name, star.rv_ref))

        star.jz = (star.x * star.vy) - (star.y * star.vx)
        star.jz /= (star.r2d * hypot(star.vx, star.vy))        
    except ValueError as e:
        # Couldn't find the star in our lists
        pass

    # Set the starset's star list
    s.stars = stars

    print( 'Found %d young stars' % len(stars))
    return s
    

def loadAllYoungStars(root, radiusCut=0.8, withRVonly=False):
    cc = objects.Constants()
    
    # Use our data if available. Otherwise use Paumards.
    ours = loadYoungStars(root, radiusCut=radiusCut, withRVonly=withRVonly)

    # Now get all other young star info from Paumard2006
    paum = tabs.Paumard2006()

    # Paumard doesn't have positional uncertainties so we
    # need to figure out what to use. Do this by comparing
    # to stars that are in both.
    ourNames = ours.getArray('name')

    for ii in range(len(ourNames)):
        star = ours.stars[ii]

    # Find the star in our star lists
    try:
        #idx = paum.ourName.index(ourNames[ii])
        idx = np.where(paum.ourName == ourNames[ii])[0]
            
        star.paumDiffX = paum.x[idx] - star.fitXa.p
        star.paumDiffY = paum.y[idx] - star.fitYa.p
        star.paumDiff = sqrt(star.paumDiffX**2 + star.paumDiffY**2)
    except ValueError as e:
        pass
    
    # We now have young stars that are not in Paumard, so must account
    # for this
    diff = ours.getArray('paumDiff')
    mtch = np.where(diff > 0)[0] # This will find all but the NaN's
    avgDiff = diff[mtch].mean()
    print( 'The average error in Paumard distances is', avgDiff)

    # Loop through all the young stars in Paumard's list.
    for ii in range(len(paum.name)):
        name = paum.ourName[ii]
        # Find the star in our star lists
        try:
            idx = np.where(ourNames == name)[0]
            star = ours.stars[idx]
        except ValueError as e:
            # Couldn't find the star in our lists. Use Paumard info.
            if (name == ' ' or name == ''):
                name = 'paum' + paum.name[ii]

            # Only keep if there is 3D velocity
            if (paum.vx[ii] == 0 and paum.vy[ii] == 0):
                continue

            if (sqrt(paum.x[ii]**2 + paum.y[ii]**2) < radiusCut):
                continue
                
            star = objects.Star(name)
            ours.stars.append(star)

            star.x = paum.x[ii]
            star.y = paum.y[ii]
            star.vx = paum.vx[ii]
            star.vy = paum.vy[ii]
            star.vz = paum.vz[ii]

            star.mag = paum.mag[ii]

            star.xerr = avgDiff
            star.yerr = avgDiff
            star.vxerr = paum.vxerr[ii]
            star.vyerr = paum.vyerr[ii]
            star.vzerr = paum.vzerr[ii]

            star.velCnt = 0

            # Proper motions need to be fixed for a different distance.
            # Paumard assumed 8kpc.
            star.vx *= cc.dist / 8000.0
            star.vy *= cc.dist / 8000.0
            star.vxerr *= cc.dist / 8000.0
            star.vyerr *= cc.dist / 8000.0

            # Need to set the positional errors to something.

            # We don't know the epoch of the Paumard measurements:
            t0 = 2005.495
            star.setFitXv(t0, star.x, star.xerr,
                          star.vx/cc.asy_to_kms, star.vxerr/cc.asy_to_kms)
            star.setFitYv(t0, star.y, star.yerr,
                          star.vy/cc.asy_to_kms, star.vyerr/cc.asy_to_kms)

            star.jz = (star.x * star.vy) - (star.y * star.vx)
            star.jz /= (star.r2d * hypot(star.vx, star.vy))
            
    # Return the starlist 
    return ours


def idNewYoung(alignRoot):
    s = starset.StarSet(alignRoot)
    
    young_new_dat = '/u/ghezgroup/data/gc/source_list/young_new.dat'
    yng = youngStarNames(datfile=young_new_dat)

    ourName = s.getArray('name')
    x = s.getArray('x')
    y = s.getArray('y')
    mag = s.getArray('mag')

    # Load up Paumard
    paum = tabs.Paumard2006()

    for i in range(len(paum.name)):
        dx = x - paum.x[i]
        dy = y - paum.y[i]
        dm = mag - paum.mag[i]

        dr = sqrt(dx**2 + dy**2)

        idx = (where((dr < 0.5) & (abs(dm) < 0.5)))[0]

        if (len(idx) > 0):
            print( 'Possible matches for:')
            print( '  %-14s  %4.1f  %7.3f  %7.3f (%s)' % \
                  (paum.name[i], paum.mag[i], paum.x[i], paum.y[i],
                   paum.ourName[i]))
            for k in idx:
                print( '  %-14s  %4.1f  %7.3f  %7.3f   %5.2f  %5.2f  %3.1f' % \
                      (ourName[k], mag[k], x[k], y[k], dx[k], dy[k], dm[k]))

        else:
            print( 'No match for: %s' % (paum.name[i]))


def makeYoungDat():
    """
    Loads list of known young stars from database, which is
    continuously updated (per T. Do).
    """
    dbfile = '/u/ghezgroup/data/gc/database/stars.sqlite'

    # Create a connection to the database file
    connection = sqlite.connect(dbfile)

    # Create a cursor object
    cur = connection.cursor()
    
    # Output File
    _out = open('/u/ghezgroup/data/gc/source_list/young_new.dat', 'w')

    _out.write('# List of Young Stars\n')
    _out.write('# -- created with gcwork.youngNames.makeYoungDat()\n')

#     cur.execute('SELECT ref_id,author,year FROM references')
    _out.write('# References:\n')
#     for row in cur:
#         _out.write('#  %s = %s (%s)\n' % (row[0], row[1], row[2]))

    _out.write('#  %s = %s (%s)\n' % ('1', 'Paumard et al.', '2006'))
    _out.write('#  %s = %s (%s)\n' % ('2', 'Do et al.', '2009'))
    _out.write('#  %s = %s (%s)\n' % ('3', 'Ghez et al.', '2008'))
    _out.write('#  %s = %s (%s)\n' % ('4', 'Genzel et al.', '2000'))
    _out.write('#  %s = %s (%s)\n' % ('5', 'Gillessen et al.', '2009'))

    _out.write('#%-14s  %-4s  %s\n' % ('OurName', 'Ref', 'AltNames'))

    # Write out the young stars.
    cur.execute('SELECT name,young,type_ref FROM stars')

    for row in cur:
        name = row[0]
        isYoung = row[1]
        reference = row[2]
        if (isYoung == 'T'):
            _out.write('%-15s  %-4s  %s\n' % (name, reference, '-'))
            
    _out.close()
