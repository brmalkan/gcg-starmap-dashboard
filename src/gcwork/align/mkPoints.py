import asciidata
import os
import numpy as np
from gcwork import starset
import gcutil
import pdb

def go(root='/g/ghez/align/13_08_21/',align='align/align_d_rms_1000_abs_t',outroot='', star='S0-2',trimS02=True):
    """
    Creates a points file for specified star from positions and uncertainties
    in align files. Points file is saved in current directory.
    Set trimS02 = True to trim the biased epochs from the points file.
    """

    s = starset.StarSet(root + align)
    epochCnt = len(s.stars[0].years)
    names = s.getArray('name')
    names = np.array(names)

    # Find the star of interest
    _points = open('%s%s.points' % (outroot, star), 'w')
    sidx = np.where(names == star)[0]

    years = s.stars[sidx].years

    if star == 'S0-2' and trimS02 == True:
        skipEp = [1998.251, 1998.366, 1998.505, 1998.590, 1998.771,
                  2002.309, 2002.391, 2002.547, 2006.470, 2006.541,
                  2007.374, 2007.612]

    # Loop over epochs and write to points file
    for ee in range(epochCnt):
        x = s.getArrayFromEpoch(ee, 'xpix')
        y = s.getArrayFromEpoch(ee, 'ypix')
        xerr_p = s.getArrayFromEpoch(ee, 'xerr_p')
        yerr_p = s.getArrayFromEpoch(ee, 'yerr_p')
        xerr_a = s.getArrayFromEpoch(ee, 'xerr_a')
        yerr_a = s.getArrayFromEpoch(ee, 'yerr_a')

        xerr_t = np.hypot(xerr_p, xerr_a)
        yerr_t = np.hypot(yerr_p, yerr_a)

        if star == 'S0-2' and trimS02 == True:
            # Trim epochs where S0-2 confused/biased
            if years[ee] in skipEp:
                continue

        _points.write('%8.3f  %9.5f  %9.5f  %9.5f  %9.5f\n' %
                      (years[ee], x[sidx], y[sidx], xerr_t[sidx],
                       yerr_t[sidx]))

    _points.close()
