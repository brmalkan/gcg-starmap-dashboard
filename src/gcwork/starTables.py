import os, sys
import math, copy
import numpy as np
from gcwork import objects
from astropy.table import Table
try:
    import MySQLdb as mdb
except:
    try:
        import mysql as mdb
    except:
        import pymysql as mdb
import pandas as pd
import pdb
import datetime

tablesDir = './'

class StarTable(object):
    def fixNames(self):
        for i in range(len(self.ourName)):
            self.ourName[i] = self.ourName[i].strip()
            self.name[i] = self.name[i].strip()

            if self.ourName[i] == '-':
                self.ourName[i] = ''

        return

    def take(self, indices):
        """
        Loop through all the numpy arrays on this StarTable
        object and pull out only the items with the specified
        indices. This modifies the lists on this object.

        This code assumes there is always a 'name' list on the
        StarTable object.
        """
        origNameCnt = len(self.name)

        for dd in dir(self):
            obj = getattr(self, dd)
            objType = type(obj)

            # Check that this is a listable object (numpy or list)
            if ((objType.__name__ == 'ndarray' or objType is 'list') and
                (len(obj) == origNameCnt)):
                # Trim this numpy list.
                setattr(self, dd, obj[indices])


class StarfinderList(StarTable):
    def __init__(self, listFile, hasErrors=False):
        self.file = listFile
        self.hasErrors = hasErrors

        # We can create a list file from scratch with nothing in it.
        if listFile is None:
            self.name = np.array([], dtype=str)
            self.mag = np.array([], dtype=float)
            self.epoch = np.array([], dtype=float)
            self.x = np.array([], dtype=float)
            self.y = np.array([], dtype=float)

            if self.hasErrors:
                self.xerr = np.array([], dtype=float)
                self.yerr = np.array([], dtype=float)

            self.snr = np.array([], dtype=float)
            self.corr = np.array([], dtype=float)
            self.nframes = np.array([], dtype=float)
            self.counts = np.array([], dtype=float)

        else:
            tab = np.genfromtxt(self.file, dtype=None)

            self.name = tab[tab.dtype.names[0]].astype('str')
            for rr in range(len(self.name)):
                self.name[rr] = self.name[rr].strip()
            self.mag = tab[tab.dtype.names[1]]
            self.epoch = tab[tab.dtype.names[2]]
            self.x = tab[tab.dtype.names[3]]
            self.y = tab[tab.dtype.names[4]]

            tabIdx = 5
            if self.hasErrors is True:
                self.xerr = tab[tab.dtype.names[tabIdx+0]]
                self.yerr = tab[tab.dtype.names[tabIdx+1]]
                tabIdx += 2

            self.snr = tab[tab.dtype.names[tabIdx+0]]
            self.corr = tab[tab.dtype.names[tabIdx+1]]
            self.nframes = tab[tab.dtype.names[tabIdx+2]]
            self.counts = tab[tab.dtype.names[tabIdx+3]]

    def append(self, name, mag, x, y, epoch=None, xerr=0, yerr=0,
               snr=0, corr=1, nframes=1, counts=0):
        if epoch is None:
            epoch = self.epoch[0]

        self.name = np.append(self.name, name)
        self.mag = np.append(self.mag, mag)
        self.epoch = np.append(self.epoch, epoch)
        self.x = np.append(self.x, x)
        self.y = np.append(self.y, y)

        if self.hasErrors:
            self.xerr = np.append(self.xerr, xerr)
            self.yerr = np.append(self.yerr, yerr)

        self.snr = np.append(self.snr, snr)
        self.corr = np.append(self.corr, corr)
        self.nframes = np.append(self.nframes, nframes)
        self.counts = np.append(self.counts, counts)


    def saveToFile(self, outfile):
        _out = open(outfile, 'w')

        for ii in range(len(self.x)):
            _out.write('%-13s  ' % self.name[ii])
            _out.write('%6.3f  ' % self.mag[ii])
            _out.write('%9.4f  ' % self.epoch[ii])
            _out.write('%11.5f  %11.5f   ' % (self.x[ii], self.y[ii]))

            if self.hasErrors is True:
                _out.write('%8.5f  %8.5f  ' % (self.xerr[ii], self.yerr[ii]))

            _out.write('%15.4f  ' % self.snr[ii])
            _out.write('%4.2f  ' % self.corr[ii])
            _out.write('%5d  ' % self.nframes[ii])
            _out.write('%15.3f\n' % self.counts[ii])

        _out.close()
def read_stf_list(list_file):
    t = Table.read(list_file, format='ascii')

    # Determine if our columns are already labeled.
    # If so, then leave them alone.
    if (t.colnames[0] != 'col1') and (t.colnames[0] == 'name'):
        return t

    t.rename_column('col1', 'name')
    t.rename_column('col2', 'mag')
    t.rename_column('col3', 'year')
    t.rename_column('col4', 'x')
    t.rename_column('col5', 'y')

    if len(t.colnames) == 11:
        t.rename_column('col6', 'xerr')
        t.rename_column('col7', 'yerr')
        t.rename_column('col8', 'snr')
        t.rename_column('col9', 'corr')
        t.rename_column('col10', 'nframes')
        t.rename_column('col11', 'counts')
    else:
        t.rename_column('col6', 'snr')
        t.rename_column('col7', 'corr')
        t.rename_column('col8', 'nframes')
        t.rename_column('col9', 'counts')

    return t

def write_stf_list(table, out_file):
    formats = {'name':     '%-13s',
               'mag':      '%6.3f',
               'year':    '%9.4f',
               'x':       '%13.7f',
               'y':       '%13.7f',
               'xerr':     '%10.7f',
               'yerr':     '%10.7f',
               'snr':      '%15.4f',
               'corr':     '%4.2f',
               'nframes':  '%5d',
               'counts':   '%15.3f'}

    table.write(out_file, format='ascii.fixed_width_no_header',
                delimiter=' ', delimiter_pad=None, bookend=False,
                formats=formats, overwrite=True)

    return

def read_points(points_file):
    t = Table.read(points_file, format='ascii')
    t.rename_column('col1', 'epoch')
    t.rename_column('col2', 'x')
    t.rename_column('col3', 'y')
    t.rename_column('col4', 'xerr')
    t.rename_column('col5', 'yerr')

    return t

def write_points(table, out_file):
    formats = {'epoch':    '%9.4f',
               'x':       '%11.5f',
               'y':       '%11.5f',
               'xerr':     '%8.5f',
               'yerr':     '%8.5f'}

    if len(table) > 1:
        table.write(out_file, format='ascii.fixed_width_no_header',
                    delimiter=' ', delimiter_pad=None, bookend=False,
                    formats=formats, overwrite=True)
    else:
        open(out_file, 'w').close()

    return

def read_phot_points(phot_file):
    t = Table.read(phot_file, format='ascii')
    t.rename_column('col1', 'epoch')
    t.rename_column('col2', 'r')
    t.rename_column('col3', 'x')
    t.rename_column('col4', 'y')
    t.rename_column('col5', 'xerr')
    t.rename_column('col6', 'yerr')
    t.rename_column('col7', 'mag')
    t.rename_column('col8', 'merr')

    return t

def write_phot_points(table, out_file):
    formats = {'epoch':    '%9.4f',
               'r':       '%11.5f',
               'x':       '%11.5f',
               'y':       '%11.5f',
               'xerr':     '%8.5f',
               'yerr':     '%8.5f',
               'mag':      '%6.3f',
               'merr':     '%6.3f'}

    if len(table) > 1:
        table.write(out_file, format='ascii.fixed_width_no_header',
                    delimiter=' ', delimiter_pad=None, bookend=False,
                    formats=formats, overwrite=True)
    else:
        open(out_file, 'w').close()

    return

class Genzel2000(StarTable):
    def __init__(self):
        self.file = tablesDir + 'ucla_genzel2000.dat'
        tab = np.genfromtxt(self.file, dtype=None)

        cols = tab.dtype.names

        self.ourName = [tab[cols[0]].astype('str')[d].strip() for d in range(tab.nrows)]
        self.name = [tab[cols[1]].astype('str')[d].strip() for d in range(tab.nrows)]
        self.r = tab[cols[2]]
        self.x = tab[cols[3]]
        self.y = tab[cols[4]]
        self.vx1 = tab[cols[5]]
        self.vx1err = tab[cols[6]]
        self.vy1 = tab[cols[7]]
        self.vy1err = tab[cols[8]]
        self.vx2 = tab[cols[9]]
        self.vx2err = tab[cols[10]]
        self.vy2 = tab[cols[11]]
        self.vy2err = tab[cols[12]]
        self.vx = tab[cols[13]]
        self.vxerr = tab[cols[14]]
        self.vy = tab[cols[15]]
        self.vyerr = tab[cols[16]]
        self.vz = tab[cols[17]]
        self.vzerr = tab[cols[18]]

        self.fixNames()

class Paumard2001(StarTable):
    def __init__(self):
        self.file = tablesDir + 'ucla_paumard2001.dat'
        tab = np.genfromtxt(self.file, dtype=None)
        cols = tab.dtype.names

        self.ourName = [tab[cols[0]][d].astype('str').strip() for d in range(tab.nrows)]
        self.name = [tab[cols[1]][d].astype('str').strip() for d in range(tab.nrows)]
        self.x = tab[cols[2]]
        self.y = tab[cols[3]]
        self.vz = tab[cols[4]]
        self.vzerr = tab[cols[5]]

        x = self.x
        y = self.y
        self.r = np.sqrt(x**2 + y**2)

        self.fixNames()


class UCLAstars(StarTable):
    def __init__(self,pwFile='/g/lu/scratch/siyao/other/pw.txt'):
        # read in password file
        pw = open(pwFile).read().split()[0]

        # Create a connection to the database file
        con = mdb.connect(host='galaxy1.astro.ucla.edu',user='dbread',passwd=pw,db='gcg')

        # query data
        df = pd.read_sql_query('SELECT * FROM stars;', con) 

        self.ourName = np.array(df['name'])
        self.Kmag = np.array(df['kp'])
        self.t0_astrom = np.array(df['ddate'])
        self.x = np.array(df['x'])
        self.xerr = np.array(df['x_err'])
        self.y = np.array(df['y'])
        self.yerr = np.array(df['y_err'])
        self.r2d = np.array(df['r2d'])
        self.vx = np.array(df['vx'])
        self.vxerr = np.array(df['vx_err'])
        self.vy = np.array(df['vy'])
        self.vyerr = np.array(df['vy_err'])
        self.vz = np.array(df['vz'])
        self.vzerr = np.array(df['vz_err'])
        self.t0_spectra = np.array(df['vz_ddate'])

class Bartko2009(StarTable):
    def __init__(self,pwFile='/g/lu/scratch/siyao/other/pw.txt'):
        # read in password file
        pw = open(pwFile).read().split()[0]

        # Create a connection to the database file
        con = mdb.connect(host='galaxy1.astro.ucla.edu',user='dbread',passwd=pw,db='gcg')

        # query data
        df = pd.read_sql_query('SELECT * FROM bartko2009;', con) 

        self.ourName = np.array(df['ucla_name'])
        self.Kmag = np.array(df['Kmag'])
        self.t0_astrom = np.array(df['t0_astrometry'])
        self.x = np.array(df['x'])
        self.y = np.array(df['y'])
        self.r2d = np.array(df['r'])
        self.vx = np.array(df['vx'])
        self.vxerr = np.array(df['vx_err'])
        self.vy = np.array(df['vy'])
        self.vyerr = np.array(df['vy_err'])
        self.t0_spectra = np.array(df['t0_spectra'])
        self.vz = np.array(df['vz'])
        self.vzerr = np.array(df['vz_err'])

class Paumard2006(StarTable):
    def __init__(self,pwFile='/g/lu/scratch/siyao/other/pw.txt'):
        # read in password file
        pw = open(pwFile).read().split()[0]

        # Create a connection to the database file
        con = mdb.connect(host='galaxy1.astro.ucla.edu',user='dbread',passwd=pw,db='gcg')

        # query data
        df = pd.read_sql_query('SELECT * FROM paumard2006;', con) 

        self.ourName = np.array(df['ucla'])
        self.name = np.array(df['name'])
        self.r2d = np.array(df['r2d'])
        self.x = np.array(df['x'])
        self.y = np.array(df['y'])
        self.z = np.array(df['z'])
        self.zerr = np.array(df['z_err'])
        self.Kmag = np.array(df['Kmag'])
        self.vx = np.array(df['vx'])
        self.vxerr = np.array(df['vx_err'])
        self.vy = np.array(df['vy'])
        self.vyerr = np.array(df['vy_err'])
        self.vz = np.array(df['vz'])
        self.vzerr = np.array(df['vz_err'])
        self.jz = np.array(df['jz'])
        self.jzerr = np.array(df['jz_err'])
        self.e = np.array(df['e'])
        self.eerr = np.array(df['e_err'])
        self.type = np.array(df['type'])
        self.quality = np.array(df['quality'])
        self.MK = np.array(df['MK'])
        self.MKerr = np.array(df['MK_err'])
        self.t0_astrom = np.array(df['t0_astrometry'])
        self.t0_spectra = np.array(df['t0_spectra'])


    def matchNames(self, labelFile=tablesDir+'label.dat'):
        cc = objects.Constants()

        # Load up our label.dat file
        labels = Labels(labelFile)

        # Convert Paumard Velocities to asec/yr.
        vxPaum = self.vx / cc.asy_to_kms
        vyPaum = self.vy / cc.asy_to_kms

        # Epoch to match at:
        t = 2008.0
        t0Paum = 2005.0  # just a guess

        xPaum = self.x + vxPaum * (t - t0Paum)
        yPaum = self.y + vyPaum * (t - t0Paum)

        xOurs = labels.x + (labels.vx * (t - labels.t0) / 1.0e3)
        yOurs = labels.y + (labels.vy * (t - labels.t0) / 1.0e3)

        for ii in range(len(xPaum)):
#             if self.ourName[ii] is not '-':
#                 continue

            dr = np.hypot(xOurs - xPaum[ii], yOurs - yPaum[ii])
            dm = labels.mag - self.Kmag[ii]

            # Find the closest source
            rdx = dr.argsort()[0]

            # Find thoses sources within 0.1"
            cdx = np.where(dr < 0.1)[0]

            print( '')
            print( 'Match %10s at [%5.2f, %5.2f] and mag = %5.2f (ourName = %s)' %
                (self.name[ii], xPaum[ii], yPaum[ii], self.Kmag[ii],
                 self.ourName[ii]))
            print( '   Closest Star:')
            print( '      %10s at [%5.2f, %5.2f] and mag = %5.2f' %
                (labels.name[rdx], xOurs[rdx], yOurs[rdx], labels.mag[rdx]))
            print( '   Stars within 0.1"')
            for kk in cdx:
                print( '      %10s at [%5.2f, %5.2f] and mag = %5.2f' %
                    (labels.name[kk], xOurs[kk], yOurs[kk], labels.mag[kk]))

class Ott2003(StarTable):
    def __init__(self):
        self.file = tablesDir + 'ucla_ott2003.dat'
        tab = np.genfromtxt(self.file, dtype=None)
        cols = tab.dtype.names

        self.ourName = [tab[cols[0]][d].astype('str').strip() for d in range(tab.nrows)]
        self.id = [tab[cols[1]][d].astype('str').strip() for d in range(tab.nrows)]
        self.name = [tab[cols[2]][d].astype('str').strip() for d in range(tab.nrows)]
        self.r = tab[cols[3]]
        self.x = tab[cols[4]]
        self.y = tab[cols[5]]
        self.xerr = tab[cols[6]]
        self.yerr = tab[cols[7]]
        self.mag = tab[cols[8]]
        self.magerr = tab[cols[9]]
        self.mHK = tab[cols[10]]
        self.mCO = tab[cols[11]]
        self.vx = tab[cols[12]]
        self.vy = tab[cols[13]]
        self.vz = tab[cols[14]]
        self.vxerr = tab[cols[15]]
        self.vyerr = tab[cols[16]]
        self.vzerr = tab[cols[17]]
        self.type = tab[cols[18]]

        self.fixNames()

class Tanner2006(StarTable):
    def __init__(self):
        self.file = tablesDir + 'ucla_tanner2006.dat'
        tab = np.genfromtxt(self.file, dtype=None)
        cols = tab.dtype.names

        self.ourName = [tab[cols[0]][d].astype('str').strip() for d in range(tab.nrows)]
        self.name = [tab[cols[1]][d].astype('str').strip() for d in range(tab.nrows)]
        self.x = tab[cols[2]]
        self.y = tab[cols[3]]
        self.xerr = tab[cols[4]]
        self.yerr = tab[cols[5]]
        self.vx = tab[cols[6]]
        self.vxerr = tab[cols[7]]
        self.vy = tab[cols[8]]
        self.vyerr = tab[cols[9]]
        self.vz = tab[cols[10]]
        self.vzerr = tab[cols[11]]

        self.fixNames()

def youngStarNames(datfile='/u/ghezgroup/data/gc/source_list/young.dat'):
    """Load list of young stars.

    Retrieves the list from /u/ghezgroup/data/gc/source_list/young.dat
    and returns a list of the names.
    """
    f_yng = open(datfile, 'r')

    names = []
    for line in f_yng:
        _yng = line.split()

        names.append(_yng[0])

    names.sort()
    return names

def lateStarNames(datfile='/u/ghezgroup/data/gc/source_list/late.dat'):
    """Load list of late-type stars.

    Retrieves the list from /u/ghezgroup/data/gc/source_list/late.dat
    and returns a list of the names.
    """
    f_yng = open(datfile, 'r')

    names = []
    for line in f_yng:
        _yng = line.split()

        names.append(_yng[0])

    names.sort()
    return names

class Orbits(StarTable):
    """
    Loads up an orbits.dat file. File is assumed to reside in
    /u/ghezgroup/data/gc/source_list/.

    Optional Input:
    orbitFile: Default is 'orbits.dat'
    """
    def __init__(self, orbitFile='orbits.dat'):
        self.file = tablesDir + orbitFile
        tab = np.genfromtxt(self.file, dtype=None)
        cols = tab.dtype.names

        self.ourName = [tab[cols[0]][d].astype('str').strip() for d in range(tab.nrows)]
        self.name = [tab[cols[0]][d].astype('str').strip() for d in range(tab.nrows)]
        self.p = tab[cols[1]]
        self.a = tab[cols[2]]
        self.t0 = tab[cols[3]]
        self.e = tab[cols[4]]
        self.i = tab[cols[5]]
        self.o = tab[cols[6]]
        self.w = tab[cols[7]]
        self.searchRadius = tab[cols[8]]

        return

class Labels(StarTable):
    """
    Loads up a label.dat file. File is assumed to reside in
    /u/ghezgroup/data/gc/source_list/.

    Optional Input:
    labelFile: Default is 'label.dat'
    """
    def __init__(self, labelFile=tablesDir+'label.dat'):
        self.file = labelFile

        if labelFile != None:
            tab = Table.read(labelFile, format='ascii')

            self.ourName = tab['Name']
            self.name = tab['Name']
            self.mag = tab['K']
            self.x = tab['x']
            self.y = tab['y']
            self.xerr = tab['xerr']
            self.yerr = tab['yerr']
            self.vx = tab['vx']
            self.vy = tab['vy']
            self.vxerr = tab['vxerr']
            self.vyerr = tab['vyerr']
            self.t0 = tab['t0']
            self.useToAlign = tab['use?']
            self.r = tab['r2d']

            # Now get the header string
            self.headerString = ''

            tmpfile = open(self.file, 'r')
            for line in tmpfile:
                if line[0] == '#':
                    self.headerString += line
                else:
                    break
        else:
            self.headerString = ''
            self.ourName = np.array([], dtype=str)
            self.name = np.array([], dtype=str)
            self.mag = np.array([], dtype=float)
            self.x = np.array([], dtype=float)
            self.y = np.array([], dtype=float)
            self.xerr = np.array([], dtype=float)
            self.yerr = np.array([], dtype=float)
            self.vx = np.array([], dtype=float)
            self.vy = np.array([], dtype=float)
            self.vxerr = np.array([], dtype=float)
            self.vyerr = np.array([], dtype=float)
            self.t0 = np.array([], dtype=float)
            self.useToAlign = np.array([], dtype=str)
            self.r = np.array([], dtype=float)

    def saveToFile(self, outfile):
        _out = open(outfile, 'w')

        _out.write(self.headerString)

        for ii in range(len(self.x)):
            _out.write('%-11s  ' % self.name[ii])
            _out.write('%4.1f    ' % self.mag[ii])
            _out.write('%9.5f  %9.5f   ' % (self.x[ii], self.y[ii]))
            _out.write('%8.5f  %8.5f  ' % (self.xerr[ii], self.yerr[ii]))
            _out.write('%8.3f %8.3f  ' % (self.vx[ii], self.vy[ii]))
            _out.write('%7.3f  %7.3f   ' % (self.vxerr[ii], self.vyerr[ii]))
            _out.write('%8.3f    ' % (self.t0[ii]))
            _out.write('%-10s   ' % str(self.useToAlign[ii]))
            _out.write('%6.3f\n' % self.r[ii])

        _out.close()

def makeLabelDat(root='./', align='align/align_d_rms_1000_abs_t', poly='polyfit_4_trim/fit',
                 oldLabelFile='source_list/label.dat',
                 newLabelFile='source_list/label_new.dat', 
                 oldInfoFile='source_list/label_info.dat',
                 newInfoFile='source_list/label_info_new.dat',
                 addNewStars=True, keepOldStars=True, updateStarPosVel=True,
                 newUse=0, stars=None, sample_cut=True):
    """
    Make a new label.dat file using output from align and polyfit.

    Optional Inputs:
    root: The root of align analysis (e.g. './' or '07_05_18.')
    align: The root filename of the align output.
    poly: The root filename of the polyfit output.
    stars: A starset.StarSet() object with polyfit already loaded.
           This overrides align/poly/root values and is useful for
           custom cuts that trim_align can't handle such as magnitude
           dependent velocity error cuts. BEWARE: stars may be modified.

    Outputs:
    source_list/label_new.dat

    Dependencies:
    Polyfit and align must contain the same numbers/names of stars. Also,
    making the label.dat file depends on having the absolute astrometry
    done correctly. See gcwork.starset to learn about how the absolute
    astrometry is loaded (it depends on a specific reference epoch in align).

    You MUST run this on something that has already been run through
    java align_absolute.
    """
    from gcwork import starset

    if stars is None:
        s = starset.StarSet(root + align, relErr=0)

        if (poly != None):
            s.loadPolyfit(root + poly)
            s.loadPolyfit(root + poly, accel=1)
    else:
        s = stars

    # Trim out the new stars if we aren't going to add them
    if not addNewStars:
        idx = []
        for ss in range(len(s.stars)):
            if 'star' not in s.stars[ss].name:
                idx.append(ss)
        s.stars = [s.stars[ss] for ss in idx]

    # Get the 2D radius of all stars and sort
    radius = s.getArray('r2d')
    ridx = radius.argsort()
    s.stars = [s.stars[ss] for ss in ridx]


    # Get info for all the stars.
    names = np.array(s.getArray('name'))

    if poly != None:
        t0 = s.getArray('fitXv.t0')
        x = s.getArray('fitXv.p') * -1.0
        y = s.getArray('fitYv.p')
        xerr = s.getArray('fitXv.perr')
        yerr = s.getArray('fitYv.perr')
        vx = s.getArray('fitXv.v') * 1000.0 * -1.0
        vy = s.getArray('fitYv.v') * 1000.0
        vxerr = s.getArray('fitXv.verr') * 1000.0
        vyerr = s.getArray('fitYv.verr') * 1000.0
    else:
        t0 = s.getArray('fitXalign.t0')
        x = s.getArray('fitXalign.p')# * -1.0
        y = s.getArray('fitYalign.p')
        xerr = s.getArray('fitXalign.perr')
        yerr = s.getArray('fitYalign.perr')
        vx = s.getArray('fitXalign.v') * 1000.0# * -1.0
        vy = s.getArray('fitYalign.v') * 1000.0
        vxerr = s.getArray('fitXalign.verr') * 1000.0
        vyerr = s.getArray('fitYalign.verr') * 1000.0

    r2d = np.sqrt(x**2 + y**2)
    mag = s.getArray('mag')

    # Fix Sgr A*
    idx = np.where(names == 'SgrA')[0]
    if (len(idx) > 0):
        x[idx] = 0
        y[idx] = 0
        vx[idx] = 0
        vy[idx] = 0
        vxerr[idx] = 0
        vyerr[idx] = 0
        r2d[idx] = 0

    # Clean up xerr and yerr so that they are at least 1 mas
    idx = np.where(xerr < 0.00001)[0]
    xerr[idx] = 0.00001
    idx = np.where(yerr < 0.00001)[0]
    yerr[idx] = 0.00001

    # only using stars r>0.4arcsec, mag<16, verr<2mas/yr

    if sample_cut:
        idx = np.where((r2d>0.4) & (mag<16) & (vxerr<2) & (vyerr<2))[0]
        print('among %d stars in align, %d stars are r>0.4as, mag<16, verr<2mas/yr' %(len(r2d), len(idx)))
    else:
        idx = range(len(mag))
        print('use all %d stars in align' %(len(idx)))
    t0 = t0[idx]
    x = x[idx]
    y = y[idx]
    xerr = xerr[idx]
    yerr = yerr[idx]
    vx = vx[idx]
    vy = vy[idx]
    vxerr = vxerr[idx]
    vyerr = vyerr[idx]
    r2d = r2d[idx]
    mag = mag[idx]
    names = names[idx]

    ##########
    # Load up the old star list and find the starting
    # point for new names.
    ##########
    oldLabels = Labels(labelFile=oldLabelFile)
    alnLabels = Labels(labelFile=oldLabelFile)
    newLabels = Labels(labelFile=oldLabelFile)

    if addNewStars:
        newNumber = calcNewNumbers(oldLabels.name, names)

    # Sort the old label list by radius just in case it
    # isn't already. We will update the radii first since
    # these sometimes get out of sorts.
    oldLabels.r = np.hypot(oldLabels.x, oldLabels.y)
    sidx = oldLabels.r.argsort()
    oldLabels.take(sidx)

    # Clean out the new label lists.
    newLabels.ourName = []
    newLabels.name = []
    newLabels.mag = []
    newLabels.x = []
    newLabels.y = []
    newLabels.xerr = []
    newLabels.yerr = []
    newLabels.vx = []
    newLabels.vy = []
    newLabels.vxerr = []
    newLabels.vyerr = []
    newLabels.t0 = []
    newLabels.useToAlign = []
    newLabels.r = []

    # Load up the align info into the alnLabels object
    alnLabels.ourName = names
    alnLabels.name = names
    alnLabels.mag = mag
    alnLabels.x = x
    alnLabels.y = y
    alnLabels.xerr = xerr
    alnLabels.yerr = yerr
    alnLabels.vx = vx
    alnLabels.vy = vy
    alnLabels.vxerr = vxerr
    alnLabels.vyerr = vyerr
    alnLabels.t0 = t0
    alnLabels.r = r2d


    def addStarFromAlign(alnLabels, ii, use):
        newLabels.ourName.append(alnLabels.ourName[ii])
        newLabels.name.append(alnLabels.name[ii])
        newLabels.mag.append(alnLabels.mag[ii])
        newLabels.x.append(alnLabels.x[ii])
        newLabels.y.append(alnLabels.y[ii])
        newLabels.xerr.append(alnLabels.xerr[ii])
        newLabels.yerr.append(alnLabels.yerr[ii])
        newLabels.vx.append(alnLabels.vx[ii])
        newLabels.vy.append(alnLabels.vy[ii])
        newLabels.vxerr.append(alnLabels.vxerr[ii])
        newLabels.vyerr.append(alnLabels.vyerr[ii])
        newLabels.t0.append(alnLabels.t0[ii])
        newLabels.useToAlign.append(use)
        newLabels.r.append(alnLabels.r[ii])

    def addStarFromOldLabels(oldLabels, ii):
        newLabels.ourName.append(oldLabels.name[ii])
        newLabels.ourName.append(oldLabels.ourName[ii])
        newLabels.name.append(oldLabels.name[ii])
        newLabels.mag.append(oldLabels.mag[ii])
        newLabels.x.append(oldLabels.x[ii])
        newLabels.y.append(oldLabels.y[ii])
        newLabels.xerr.append(oldLabels.xerr[ii])
        newLabels.yerr.append(oldLabels.yerr[ii])
        newLabels.vx.append(oldLabels.vx[ii])
        newLabels.vy.append(oldLabels.vy[ii])
        newLabels.vxerr.append(oldLabels.vxerr[ii])
        newLabels.vyerr.append(oldLabels.vyerr[ii])
        newLabels.t0.append(oldLabels.t0[ii])
        newLabels.useToAlign.append(oldLabels.useToAlign[ii])
        newLabels.r.append(oldLabels.r[ii])

    def deleteFromAlign(alnLabels, idx):
        # Delete them from the align lists.
        alnLabels.ourName = np.delete(alnLabels.ourName, idx)
        alnLabels.name = np.delete(alnLabels.name, idx)
        alnLabels.mag = np.delete(alnLabels.mag, idx)
        alnLabels.x = np.delete(alnLabels.x, idx)
        alnLabels.y = np.delete(alnLabels.y, idx)
        alnLabels.xerr = np.delete(alnLabels.xerr, idx)
        alnLabels.yerr = np.delete(alnLabels.yerr, idx)
        alnLabels.vx = np.delete(alnLabels.vx, idx)
        alnLabels.vy = np.delete(alnLabels.vy, idx)
        alnLabels.vxerr = np.delete(alnLabels.vxerr, idx)
        alnLabels.vyerr = np.delete(alnLabels.vyerr, idx)
        alnLabels.t0 = np.delete(alnLabels.t0, idx)
        alnLabels.r = np.delete(alnLabels.r, idx)

    nn = 0
    # keep record of the name change
    while nn < len(oldLabels.name):
        #
        # First see if there are any new stars that should come
        # before this star.
        #
        if addNewStars:
            def filterFunction(i):
                return ((alnLabels.r[i] < oldLabels.r[nn]) and ('star' in alnLabels.name[i]))
            idx = list(filter(filterFunction, range(len(alnLabels.name))))

            for ii in idx:
                rAnnulus = int(math.floor(alnLabels.r[ii]))
                number = newNumber[rAnnulus]
                print('{0} -> S{1}-{2}'.format(alnLabels.name[ii], rAnnulus, number))
                alnLabels.name[ii] = 'S%d-%d' % (rAnnulus, number)
                newNumber[rAnnulus] += 1

                # Insert these new stars.
                addStarFromAlign(alnLabels, ii, newUse)

            # Delete these stars from the align info.
            for ii in idx:
                deleteFromAlign(alnLabels, ii)

        #
        # Now look for this star in the new align info
        #
        idx = np.where(alnLabels.name == oldLabels.name[nn])[0]

        if len(idx) > 0:
            # Found the star

            if updateStarPosVel:
                # Update with align info
                addStarFromAlign(alnLabels, idx[0], oldLabels.useToAlign[nn])
            else:
                # Don't update with align info
                addStarFromOldLabels(oldLabels, nn)

            deleteFromAlign(alnLabels, idx[0])

        elif keepOldStars:
            # Did not find the star. Only keep if user said so.
            addStarFromOldLabels(oldLabels, nn)

        nn += 1

    # Quick verification that we don't have repeat names.
    uniqueNames = np.unique(newLabels.name)
    if len(uniqueNames) != len(newLabels.name):
        print( 'Problem, we have a repeat name!!')

    # Write to output
    newLabels.saveToFile(root +  newLabelFile)

    # add the current date
    import datetime
    with open(root + newLabelFile, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write('#'+ str(datetime.datetime.now()) + '\n' + content)


    # For changed stars: add current align directory to label_info.dat 
    label_new = Table.read(newLabelFile, format='ascii',
          names=('Name','K','x','y','xerr','yerr','vx','vy','vxerr','vyerr','t0','use','r'))
    label_old = Table.read(oldLabelFile, format='ascii',
          names=('Name','K','x','y','xerr','yerr','vx','vy','vxerr','vyerr','t0','use','r'))
    info_old = Table.read(oldInfoFile, data_start=1, format='ascii', names=('Name', 'update'))
    info_new_star = []
    info_new_update = []

    names_old = np.array(label_old['Name'])
    names_new = np.array(label_new['Name'])

    path = os.getcwd()
    
    # make sure label_old and info_old has the same stars
    if len(info_old) != len(label_old):
        print('Warning: label_info.dat has different stars as in label.dat')
        print('Not updating label_info.dat')
        print('---------PLEASE CHECK WHY------------------------------------') 
        return 
    elif not (info_old['Name']==label_old['Name']).all():
        print('Warning: label_info.dat has different stars as in label.dat')
        print('Not updating label_info.dat')
        print('---------PLEASE CHECK WHY------------------------------------') 
        return 

    # loop through all the stars in the new label.dat
    for i in range(len(names_new)):
        star = names_new[i]
        if not np.in1d(star, names_old):
            info_new_star.append(star)
            info_new_update.append(path)
        else:
            idx = np.where(names_old==star)[0][0]
            dvx = label_new[i]['vx'] - label_old[idx]['vx']
            dvy = label_new[i]['vy'] - label_old[idx]['vy']
            if (abs(dvx) < 0.00001) and  (abs(dvy) < 0.00001):
                info_new_star.append(star)
                info_new_update.append(info_old[idx]['update'])
            else:
                info_new_star.append(star)
                info_new_update.append(path)

    info_new = Table()
    info_new['Name'] = info_new_star
    info_new['update'] = info_new_update
    info_new['Name'].format ='12s'
    info_new['update'].format = '80s'
    info_new.write(newInfoFile, format='ascii.fixed_width', overwrite=True, delimiter=None)

    # add the current date
    with open(root + newInfoFile, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write('#'+ str(datetime.datetime.now()) + '\n' + content)


def calcNewNumbers(oldNames, newNames):
    # Loop through annuli of 1 arcsecond and find last name
    rRange = np.arange(20)
    newNumber = np.zeros(len(rRange))

    for rr in range(len(rRange)):
        substring = 'S%d-' % rr
        rNameOld = list(filter(lambda x: x.find(substring) != -1, oldNames))
        rNameNew = list(filter(lambda x: x.find(substring) != -1, newNames))

        if (len(rNameOld) == 0):
            newNumber[rr] = 1
        else:
            rNumberOld = np.zeros(len(rNameOld))
            for nn in range(len(rNameOld)):
                tmp = rNameOld[nn].split('-')
                rNumberOld[nn] = int(tmp[-1])
            rNumberOld.sort()

            if (len(rNameNew) != 0):
                rNumberNew = np.zeros(len(rNameNew))
                for nn in range(len(rNameNew)):
                    tmp = rNameNew[nn].split('-')
                    rNumberNew[nn] = int(tmp[-1])
                rNumberNew.sort()
            else:
                rNumberNew = np.array([1])

            newNumber[rr] = max([rNumberOld[-1], rNumberNew[-1]]) + 1

        print( 'First New Number is S%d-%d' % (rRange[rr], newNumber[rr]))

    return newNumber

def makeOrbitsDat(root='./', efit='efit3_d/output/efit3.log',
                  poly='polyfit_d/fit', onlyHighAccel=True):
    """
    Make a new orbits.dat file using output from polyfit and efit3.

    Optional Inputs:
    root: The root of align analysis (e.g. './' or '07_05_18.')
    poly: The root filename of the polyfit output.
    efit: The efit3.log file containing the new orbit solutions.

    Outputs:
    source_list/orbits_new.dat

    Dependencies:
    Only sources in the central arcsecond with significant accelerations
    are included in our list of stellar orbits. To determine which stars
    these are, we run

    gcwork.polyfit.accel.highSigSrcs(0.5, 4)

    and then use all the named sources in the resulting list.
    """
    from gcwork.polyfit import accel

    # Now read in the efit3.log file
    tab = np.genfromtxt(root + efit, dtype=None)
    cols = tab.dtype.names

    name = tab[cols[0]].astype('str')
    dist = tab[cols[1]]  # pc
    a = tab[cols[4]]     # mas
    p = tab[cols[5]]     # yr
    e = tab[cols[6]]     #
    t0 = tab[cols[7]]    # yr
    w = tab[cols[8]]     # deg
    i = tab[cols[9]]     # deg
    o = tab[cols[10]]    # deg

    if onlyHighAccel is True:
        # Find the significantly accelerating sources within the
        # central arcsecond.
        srcs = accel.highSigSrcs(0.5, 4, verbose=False, rootDir=root, poly=poly)
    else:
        # Use ALL stars in this list
        srcs = name

    _out = open(root + 'source_list/orbits_new.dat', 'w')
    _out.write('# Python gcwork.starTables.makeOrbitsDat()\n')
    _out.write('%-10s  %7s  %7s  %8s  %7s  %7s  %7s  %7s  %7s\n' % \
           ('#Star', 'P', 'A', 't0', 'e', 'i', 'Omega', 'omega', 'search'))
    _out.write('%-10s  %7s  %7s  %8s  %7s  %7s  %7s  %7s  %7s\n' % \
           ('#Name', '(yrs)', '(mas)', '(yrs)', '()',
        '(deg)', '(deg)', '(deg)', '(pix)'))


    # Loop through every src and if it is named, output into a
    # new orbits_new.dat file.
    for ss in range(len(srcs)):
        try:
            idx = name.index(srcs[ss])
        except ValueError:
            #print( 'Failed to find match for %s in %s' % (srcs[ss], efit))
            continue

        # Skip if this isn't a named source
        if (('star' in srcs[ss]) and (onlyHighAccel == True)):
            continue

    # Write output
    _out.write('%-10s  ' % (srcs[ss]))
    _out.write('%7.2f  %7.1f  %8.3f  ' % (p[idx], a[idx], t0[idx]))
    _out.write('%7.5f  %7.3f  %7.3f  ' % (e[idx], i[idx], o[idx]))
    _out.write('%7.3f  %7d\n' % (w[idx], 2))

    _out.close()


def labelNoYoung(input_labels, output_labels):
    """
    Take an existing label.dat file and set all the known young stars
    to NOT be used in alignment.
    """

    labels = Labels(labelFile=input_labels)

    # Load up the list of young stars
    yng = youngStarNames(datfile='/u/ghezgroup/data/gc/source_list/young_new.dat')

    _out = open(output_labels, 'w')
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
        if (labels.name[i] in yng):
            labels.useToAlign[i] = 0   # Don't use for alignment

        _out.write('%-10s  %5.1f   ' % (labels.name[i], labels.mag[i]))
        _out.write('%7.3f %7.3f ' % (labels.x[i], labels.y[i]))
        _out.write('%7.3f %7.3f   ' % (labels.xerr[i], labels.yerr[i]))
        _out.write('%8.3f %8.3f ' % (labels.vx[i], labels.vy[i]))
        _out.write('%8.3f %8.3f   ' % (labels.vxerr[i], labels.vyerr[i]))
        _out.write('%8.3f %4d %7.3f\n' %
                       (labels.t0[i], labels.useToAlign[i], labels.r[i]))


    _out.close()


def labelRestrict(inputLabel, outputLabel, alignInput,
                  numSpeck=None, numAO=None):
    """
    Modify an existing label.dat file to be used with the align
    -restrict flag. This is the program that chooses which stars
    are to be used for speckle alignment and which are to be
    used for AO alignment. The stars are chosen based on the
    number of detections in either speckle or AO.
    We will use the subset of stars in ALL speckle epochs as speckle
    alignment sources; and those stars that are in ALL AO epochs
    are used as AO alignment sources. The input align files should
    not have been trimmed for the most part.

    Be sure that there is an <alignInput>.list file containing
    the epochs and their data types.

    Paramters:
    inputLabel -- the input label.dat file. This will not be modified.
    outputLabel -- the output label.dat file. Only the use? column is changed.
    alignInput -- the root name of the align files to be used when
                  determining how many speckle and AO maps a stars is found in.
    numSpeck -- if None then only stars in ALL speckle epochs are used
                as alignment sources.
    numAO -- if None then only stars in ALL AO epochs are used as
             alignment sources.
    """
    from gcwork import starset

    labels = Labels(labelFile=inputLabel)
    s = starset.StarSet(alignInput)

    # Figure out the data/camera type for each epoch (speckle or AO)
    _list = open(alignInput + '.list', 'r')
    aoEpochs = []
    spEpochs = []

    i = 0
    for line in _list:
        info = line.split()
        aoType = int( info[1] )

        if ((aoType == 2) or (aoType == 3)):
            spEpochs.append(i)
        if ((aoType == 8) or (aoType == 9)):
            aoEpochs.append(i)

        i += 1

    if (numSpeck is None):
        numSpeck = len(spEpochs)
    if (numAO is None):
        numAO = len(aoEpochs)

    # For each star, count up the number of speckle and AO epochs it is
    # detected in.
    names = s.getArray('name')
    velCnt = s.getArray('velCnt')

    numStars = len(names)
    numEpochs = len(s.stars[0].years)

    print( 'Initial:  Nstars = %4d  Nepochs = %2d' % (numStars, numEpochs))
    print( 'Number of Epochs of Type:')
    print( '   Speckle = %d' % len(spEpochs))
    print( '   AO      = %d' % len(aoEpochs))


    aoCnt = np.zeros(numStars)
    spCnt = np.zeros(numStars)

    for e in range(numEpochs):
        pos = s.getArrayFromEpoch(e, 'x')

        idx = (np.where(pos > -1000))[0]

        if (e in aoEpochs):
            aoCnt[idx] += 1
        if (e in spEpochs):
            spCnt[idx] += 1

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

    spNumStars = 0
    aoNumStars = 0
    use = '1'
    for i in range(len(labels.name)):
        # Find this star in our align output
        try:
            foo = names.index(labels.name[i])

            if (labels.useToAlign[i] == 0):
                # Preserve any pre-existing use?=0 stars
                use = '0'

                if (spCnt[foo] >= numSpeck):
                    print( '%-13s is in all speckle epochs, but use=0' %
                          names[foo])
            else:
                if (spCnt[foo] >= numSpeck):
                    use = '2'
                    spNumStars += 1

                if (aoCnt[foo] >= numAO):
                    aoNumStars += 1

                    if (use == '2'):
                        # Speckle and AO
                        use += ',8'
                    else:
                        # AO only
                        use = '8'

        except ValueError:
            # Don't change anything if we didn't find it.
            # Reformat to string for ease of use
            use = str(labels.useToAlign[i])


    _out.write('%-10s  %5.1f   ' % (labels.name[i], labels.mag[i]))
    _out.write('%7.3f %7.3f ' % (labels.x[i], labels.y[i]))
    _out.write('%7.3f %7.3f   ' % (labels.xerr[i], labels.yerr[i]))
    _out.write('%8.3f %8.3f ' % (labels.vx[i], labels.vy[i]))
    _out.write('%8.3f %8.3f   ' % (labels.vxerr[i], labels.vyerr[i]))
    _out.write('%8.3f %-4s %7.3f\n' %
                   (labels.t0[i], use, labels.r[i]))

    _out.close()

    print( 'Final:   Nstars Speckle = %4d  AO = %4d' %
          (spNumStars, aoNumStars))




def updateLabelInfoWithAbsRefs(oldLabelFile, newLabelFile, outputFile,
                               newUse=1, oldUse=0, appendNew=False):
    """
    Modify an existing label.dat file with updated positions and
    velocities from a different label.dat (or absolute_refs.dat) file.

    Input Parameters:
    oldLabelFile - The name of the input old label.dat file
    newLabelFile - The name of the input file from which to pull new vel info.
    outputFile - Save the results to a new file.

    Optional Parameters:
    newUse - (def=1) set to this value for stars that are modified,
             or if None, preserve what was in the old label file.
    oldUse - (def=0) set to this value for stars that are not modified,
             or if None, preserve what was in the old label file.
    """
    old = Labels(oldLabelFile)
    new = Labels(newLabelFile)
    print( '%5d stars in list with old velocities' % len(old.name))
    print( '%5d stars in list with new velocities' % len(new.name))

    if oldUse != None:
        old.useToAlign[:] = oldUse

    if appendNew:
        newNumber = calcNewNumbers(old.name, new.name)

    updateCount = 0
    newCount = 0

    for nn in range(len(new.name)):
        idx = np.where(old.name == new.name[nn])[0]

        if len(idx) > 0:
            old.mag[idx] = new.mag[nn]
            old.x[idx] = new.x[nn]
            old.y[idx] = new.y[nn]
            old.xerr[idx] = new.xerr[nn]
            old.yerr[idx] = new.yerr[nn]
            old.vx[idx] = new.vx[nn]
            old.vy[idx] = new.vy[nn]
            old.vxerr[idx] = new.vxerr[nn]
            old.vyerr[idx] = new.vyerr[nn]
            old.t0[idx] = new.t0[nn]
            old.r[idx] = new.r[nn]

            if newUse != None:
                old.useToAlign[idx] = newUse

            updateCount += 1
        else:
            if appendNew:
                rAnnulus = int(math.floor(new.r[nn]))
                number = newNumber[rAnnulus]
                new.name[nn] = 'S%d-%d' % (rAnnulus, number)
                newNumber[rAnnulus] += 1

                old.name = np.append(old.name, new.name[nn])
                old.mag = np.append(old.mag, new.mag[nn])
                old.x = np.append(old.x, new.x[nn])
                old.y = np.append(old.y, new.y[nn])
                old.xerr = np.append(old.xerr, new.xerr[nn])
                old.yerr = np.append(old.yerr, new.yerr[nn])
                old.vx = np.append(old.vx, new.vx[nn])
                old.vy = np.append(old.vy, new.vy[nn])
                old.vxerr = np.append(old.vxerr, new.vxerr[nn])
                old.vyerr = np.append(old.vyerr, new.vyerr[nn])
                old.t0 = np.append(old.t0, new.t0[nn])
                old.r = np.append(old.r, new.r[nn])
                old.useToAlign = np.append(old.useToAlign, newUse)

                newCount += 1
    print( '%5d stars in the NEW starlist created' % len(old.name))
    print( '   %5d updated' % updateCount)
    print( '   %5d added'   % newCount)


    old.saveToFile(outputFile)

def checkLabelsForDuplicates(labels='/u/ghezgroup/data/gc/source_list/label.dat'):
    """
    Read in a label.dat file (or a labels object) and search for duplicates
    based on position and magnitude.
    """
    if type(labels) is 'gcwork.starTables.Labels':
        lab = labels
    else:
        lab = Labels(labelFile=labels)

    rdx = lab.r.argsort()
    lab.take(rdx)

    duplicateCnt = 0

    dummy = np.arange(len(lab.name))

    for ii in dummy:
        dx = lab.x - lab.x[ii]
        dy = lab.y - lab.y[ii]
        dm = np.abs(lab.mag - lab.mag[ii])

        dr = np.hypot(dx, dy)

        # Search for stars within 50 mas
        rdx = np.where((dr < 0.05) & (dm < 1) & (dummy >= ii))[0]

        if len(rdx) > 1:
            duplicateCnt += 1

            print( '')
            print( 'Found stars close to %s' % lab.name[ii])
            print( '    %-13s  %5s  %7s %7s  %7s %7s' %
                ('Name', 'mag', 'x', 'y', 'vx', 'vy'))

            for rr in rdx:
                print( '    %-13s  %5.2f  %7.3f %7.3f   %7.3f %7.3f' %
                    (lab.name[rr], lab.mag[rr], lab.x[rr], lab.y[rr],
                     lab.vx[rr], lab.vy[rr]))

    print( '')
    print( 'Found %d duplicates' % duplicateCnt)

def changeUseLabel(inputLabel, outputLabel):
    """
    Reads in a label.dat and changes the use column of
    reference stars from '1' to '2,8'
    """

    labels = Labels(labelFile=inputLabel)

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
    _out.write('%7.3f %7.3f ' % (labels.x[i], labels.y[i]))
    _out.write('%7.3f %7.3f   ' % (labels.xerr[i], labels.yerr[i]))
    _out.write('%8.3f %8.3f ' % (labels.vx[i], labels.vy[i]))
    _out.write('%8.3f %8.3f   ' % (labels.vxerr[i], labels.vyerr[i]))
    _out.write('%8.3f %4s %7.3f\n' %
                   (labels.t0[i], use, labels.r[i]))

    _out.close()


def add_accel_cols_to_label(label_file):
    """
    TEMPORARY stand-alone function to add acceleration columns to label.dat file.
    It will assign a value of "0" for accels and accel errs. All other values 
    will be the same

    NOTE: this should not be used permenantly, as these columns should
    ideally be added in the codes that actually make label.dat. This is 
    a short-term patch to allow for testing of the file format.

    New file is written as "label.dat" in current working directory.
    Header info (version number and date) will be updated.
    The "major version" will be increased by one.
    
    If original label.dat file is also in current working directory,
    will move into new sub-directory called "label_orig/" to prevent it
    from being overwritten

    Parameters:
    ----------
    label_file: label.dat file
        Label.dat file to add accelerations too

    """
    # Read in existing label.dat file data. Have 2 reads: a "data read" through
    # astropy table and and "header read" through general python reader. Clumsy but
    # I don't know how to do this better.
    orig = Table.read(label_file, format='ascii.commented_header', header_start=-2)
    orig_with_header = open(label_file, 'r')

    # If label_file is in CWD, then move into new sub-directory, so we don't overwrite it
    if os.path.exists('{0}/{1}'.format(os.getcwd(), label_file)):
        print('Moving input label.dat into protected dir')
        orig_dir = 'label_orig/'
        if not os.path.exists(orig_dir):
            os.mkdir(orig_dir)

        cmd = 'mv {0} {1}'.format(label_file, orig_dir)
        print(cmd)
        os.system(cmd)

    # Create new output file in CWD
    _out = open('label.dat', 'w')

    # First, we need to write the header. We'll take this from the original
    # label.dat file. Edit where necessary
    lines = orig_with_header.readlines()
    for line in lines:
        # Isolate header lines
        if line.startswith('#'):
            # Increment version
            if ('version' in line) & ('_' in line):
                tmp = line.split(':')
                version_orig = tmp[-1].split('_')
                
                major_ver = int(version_orig[0])
                major_ver += 1

                newline = '{0} {1}_{2}'.format(tmp[0], major_ver, version_orig[-1])
                
                _out.write(newline)
            # Update date
            elif '2023' in line:
                today = datetime.date.today()
                newline = '# {0}-{1}-{2}\n'.format(today.year, today.month, today.day)

                _out.write(newline)
            # If we get to column headers, stop because we are remaking these
            elif '# Name' in line:
                break
            # Otherwise, just copy the line
            else:
                _out.write(line)

    # Now, update the column header info to contain accels
    _out.write('# Name\t K\t x\t y\t xerr\t yerr\t vx\t vy\t vxerr\t vyerr\t ax\t ay\t axerr\t ayerr\t t0\t use?\t r2d\n')
    _out.write('#()\t (mag)  (asec)  (asec)  (asec)  (asec)  (mas/yr)  (mas/yr)  (mas/yr)  (mas/yr)  (mas/yr2) (mas/yr2) (mas/yr2) (mas/yr2) (year) () (asec)\n')

    # Add data, with zeros for accel cols
    for dd in range(len(orig)):
        star = orig[dd]
        str_fmt = '{:10s}   {:4.2f}   {:8.5f}   {:8.5f}   {:7.5f}   {:7.5f}   {:7.3f}   {:7.3f}   {:7.3f}   {:7.3f}   {:5.3f}   {:5.3f}   {:5.3f}   {:5.3f}   {:7.3f}   {:s}   {:5.3f}\n'
        newline = str_fmt.format(star['Name'], star['K'], star['x'], star['y'], star['xerr'], star['yerr'],
                                     star['vx'], star['vy'], star['vxerr'], star['vyerr'],
                                     0., 0., 0., 0., star['t0'], star['use?'], star['r2d'])
        _out.write(newline)

    # Close files
    _out.close()
    orig_with_header.close()

    # Finally, run a quick sanity check to make sure all non-accel columns
    # are exactly the same as they were originally
    new_label = Table.read('label.dat', format='ascii.commented_header', header_start = -2)
    test_cols = ['Name', 'K', 'x', 'y', 'xerr', 'yerr', 'vx', 'vy', 'vxerr', 'vyerr', 't0', 'use?', 'r2d']

    for ii in test_cols:
        assert np.all(orig[ii] == new_label[ii])

    return
