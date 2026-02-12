import numpy as np
import pylab as py
import scipy as sp
import histogram2d as h2d
import asciidata
import math
from gcwork import objects
from scipy import stats
import pdb

def pullResults(p1,p2,inFile,star_num=0,useRel=False,fixPar=[],getAll=False):
    """
    Pull results from an efit run (MultiNest version). The results file will by
    default be in the following order:
    M, x0, y0, vx, vy, vz, R0, Mext, Periapse, alpha, Omega, omega, i, P, T0, e

    If a parameter is held fixed, this variable is popped out of the above list
    and appended at the end. This is accounted for here. NOTE that the order in
    which the parameters are placed at the end of the list may not be accurate,
    but this should not matter since the values are fixed anyway.

    Input:
    p1, p2 (str):  2 paramaters of interest; must match parameter names:
    		   ['M', 'x0', 'y0', 'vx', 'vy', 'vz', 'R0', 'Mext', 'peri', 'alpha']
    inFile (str):  Multinest file (e.g., efit_.txt) 

    Optional Inputs:
    star_num (int): Integer representing star to be plotted, in the case of fits
    		    with multiple stars. If only one star was fit, star_num=0 (default).
    useRel (bool):  If relativity was used in the orbital fit, set to True. (def = False)
    fixPar (list):  Parmameters that were held fixed in the fit. Default is an empty
    		    list. But if relativity was not used (useRel=False), then
                    it is assumed that Mext, peri, and alpha were held fixed and the
                    code automatically adds these parameters to the fixPar list.
    getAll (bool):  Instead of just getting 2 parameters, get all of the parameters
    		    from the orbital fit.
    """
    
    inFile = asciidata.open(inFile)

    params = ['M', 'x0', 'y0', 'vx', 'vy', 'vz', 'R0', 'Mext', 'peri',
              'alpha']
    orbElem = ['O', 'w', 'i', 'P', 'T0', 'e']

    # Must account for multiple stars used, whose 6 orbital elements are written
    # out similarly to the 1st star, one after the other
    for ii in range(star_num+1):
        nextStar = [oo+str(ii) for oo in orbElem]
        params.extend(nextStar)

    # Was anything held fixed?
    if useRel == False:
        fixPar.extend(['Mext','peri','alpha'])

        # In case the user included the above relativistic terms in fixPar already,
        # use a set to select unique elements, then convert back to list:
        fixPar = list(set(fixPar)) # CAUTION! set() does not preserve order!

    if len(fixPar) > 0:
        params = [x for x in params if x not in fixPar]

        # Now add the fixed parameters to the end of the list to reflect their
        # location in the results file
        params.extend(fixPar)

    # Now create dictionary storing (parameters, column num) as (key, value) pairs
    ii = range(2, len(params) + 2)
    idx = dict(zip(params,ii))

    #To Do: is there a better way to do the above step?? and below?
                    
    # Pull the data from the appropriate column
    weights = inFile[0].tonumpy()
    logLike = inFile[1].tonumpy()
    M = inFile[idx['M']].tonumpy()
    M /= 1.e6
    x0 = -inFile[idx['x0']].tonumpy()
    x0 *= 1000
    y0 = inFile[idx['y0']].tonumpy()
    y0 *= 1000
    Vx = -inFile[idx['vx']].tonumpy()
    Vx *= 1000
    Vy = inFile[idx['vy']].tonumpy()
    Vy *= 1000
    Vz = inFile[idx['vz']].tonumpy() # check units
    R0 = inFile[idx['R0']].tonumpy()
    R0 /= 1.e3
    Mext = inFile[idx['Mext']].tonumpy()
    Mext /= 1000
    peri = inFile[idx['peri']].tonumpy() # check units
    alpha = inFile[idx['alpha']].tonumpy() # check units
    O = inFile[idx['O'+str(star_num)]].tonumpy() # check units (radians)
    w = inFile[idx['w'+str(star_num)]].tonumpy() # check units (radians)
    i = inFile[idx['i'+str(star_num)]].tonumpy() # check units (radians)
    P = inFile[idx['P'+str(star_num)]].tonumpy() # check units
    T0 = inFile[idx['T0'+str(star_num)]].tonumpy() # check units
    e = inFile[idx['e'+str(star_num)]].tonumpy() 

    # What is to be plotted?  Create a dictionary with the data:
    data = {'M': M, 'x0': x0, 'y0': y0, 'vx': Vx, 'vy': Vy, 'vz': Vz,
            'R0': R0, 'Mext': Mext, 'peri': peri, 'alpha': alpha,
            'O': O, 'w': w, 'i': i, 'P': P, 'T0': T0, 'e': e}

    label = {'M': 'Mass ($10^6$ M$_\odot$)',
             'x0': 'Sgr A* $\Delta$RA Position (mas)',
             'y0': 'Sgr A* $\Delta$Dec Position (mas)', 
             'vx': 'Sgr A* $\Delta$RA Velocity (mas/yr)', 
             'vy': 'Sgr A* $\Delta$Dec Velocity (mas/yr)',
             'vz': 'Sgr A* Z Velocity (mas/yr)',
             'R0': 'R$_o$ (kpc)', 
             'Mext': 'Extended Mass ($10^3$ M$_\odot$)',
             'peri': 'Periapse',
             'alpha': '$\alpha$',
             'O': '$\Omega$',
             'w': '$\omega$',
             'i': '$i$',
             'P': '$P$',
             'T0': 'T$_o$',
             'e': 'e'}

    if getAll == False:
        return [data[p1], data[p2], weights, label[p1], label[p2]]
    else:
        return [data, weights, label]


def plot2_mn(p1,p2,outfile,adir='13_10_16/',mnestfile='chains/efit_.txt',star_num=0,
             title='',bin=50,color='black',useRel=False,fixPar=[]):
    """
    Plot a 2D histogram of 2 specified parameters. Results must be
    from MultiNest version of efit.

    INPUT:
    p1 (str):        parameter to plot on X axis; options are:
                     'M', 'x0', 'y0', 'vx', 'vy', 'vz', 'R0', 'Mext',
                     'peri', 'alpha', 'O', 'w', 'i', 'P', 'T0', 'e' 
    p2 (str):        parameter to plot on Y axis; options are same as above.
    outfile (str):   Name of output file (do not include extension). Will
                     be saved to adir+plots/.

    OPTIONAL INPUT:
    adir (str):      Path to align directory.
    mnestfile (str): File containing efit results. Assumed to be within adir + efit/.
    title (str):     Title for plot (default: blank)
    bin (int):       Number of bins for the 2 dimensions to be plotted.
    color (str):     Color of contours.

    OUTPUT:
    <outfile>.png/eps:   2D histogram of the 2 input parameters.
    
    """

    # Pull the data for the input parameters
    mnfile = '%s/efit/%s' % (adir, mnestfile)
    pltX, pltY, weights, labX, labY = pullResults(p1,p2,mnfile,star_num=star_num,
                                                  useRel=useRel,fixPar=fixPar)


    py.clf()
    py.figure(1,figsize=(8,6))
    py.subplots_adjust(left=0.13,right=0.95,bottom=0.13,top=0.9)
    (hist, pYbins, pXbins) = np.histogram2d(pltY, pltX,
                                        bins=(bin,bin),weights=weights)
    
    # Need to convert the 2d histogram into floats
    probDist = np.array(hist, dtype=float)
    levels = getContourLevels(probDist)

    py.imshow(probDist, cmap=py.cm.hot_r, origin='lower', aspect='auto',
                 extent=[pXbins[0], pXbins[-1], pYbins[0], pYbins[-1]])
    py.contour(probDist, levels, origin=None, colors=color,
                 extent=[pXbins[0], pXbins[-1], pYbins[0], pYbins[-1]])
    thePlot = py.gca()
    py.setp( thePlot.get_xticklabels(), fontsize=16, fontweight='bold')
    py.setp( thePlot.get_yticklabels(), fontsize=16, fontweight='bold')  
    font = {'fontsize' : 20}
    if ((p1 == 'T0') | (p2 == 'T0')):
        thePlot.xaxis.set_major_formatter(py.FormatStrFormatter('%6.1f'))
    #thePlot.get_xaxis().set_major_locator(py.MultipleLocator(2))
    py.xlabel(labX, font)
    py.ylabel(labY, font)
    py.title('%s' % title, font)
    py.savefig('%s/plots/%s.png' % (adir,outfile))
    py.savefig('%s/plots/%s.eps' % (adir,outfile))
    py.close()
  

def plot2_mn_multiFits(p1,p2,outfile,numFits,star_num=0,
                       outdir='./',sigma=3,bin=50,useRel=False,fixPar=[],
                       drop_maser=False):
    """
    Overplot the contours of a 2D histogram of 2 specified parameters for multiple efits.
    Results must be from MultiNest version of efit.

    INPUT:
    p1 (str):        parameter to plot on X axis; options are:
                     'M', 'x0', 'y0', 'vx', 'vy', 'vz', 'R0', 'Mext',
                     'peri', 'alpha', 'O', 'w', 'i', 'P', 'T0', 'e' 
    p2 (str):        parameter to plot on Y axis; options are same as above.
    outfile (str):   Name of output file (do not include extension).
    	 	     To save to a specific dir, must include outdir (see below).
    numFits (int):   Number of efit results to plot. The contours from
                     each efit 2D histogram will be overplotted.

    Input at prompt:
    adir (str):      Path to align directory from which efit results will be pulled.
                     Results are assumed to live in the specified mnestfile for
                     all adir's given at prompt.
    mnestfile (str): File containing efit results. Must include path.
    lbl (str):       Label for this fit for plotting.

    OPTIONAL INPUT:
    outdir (str):    Path where output files will be saved.
    sigma (int):     Sigma of contour to be plotted.
    bin (int):       Number of bins for the 2 dimensions to be plotted.

    OUTPUT:
    <outfile>.png/eps:  Contour plot of 2D hist for all align dirs given.
    
    """
    import matplotlib.font_manager
    prop = matplotlib.font_manager.FontProperties(size=9)
    py.clf()
    py.figure(1,figsize=(6,6))
    py.subplots_adjust(left=0.16,right=0.92,bottom=0.13,top=0.92)

    clrs = ['red','blue','green','magenta','cyan','black','orange']

    if drop_maser == True: # special case
        msrDir = '/u/syelda/research/gc/aligndir/drop_maser_tests/'
        msrTest = ['13_10_16_no10ee/','13_10_16_no12n/','13_10_16_no15ne/','13_10_16_no17/',
                   '13_10_16_no28/','13_10_16_no7/','13_10_16_no9/']
        outdir = msrDir + 'plots/'
        #lbl = np.empty([numFits])
        lbl = ['IRS 10EE', 'IRS 12N', 'IRS 15NE', 'IRS 17', 'IRS 28', 'IRS 7', 'IRS 9']

    for ii in range(numFits):
        if drop_maser == True:
            adir = msrDir + msrTest[ii]
            mnfile = '%s/efit/chains/efit_.txt' % adir
            #lbl[ii] = msrTest[ii]
        else:
            adir = raw_input("Full path to align directory %i: " % (ii+1))
            mnfile = raw_input("MultiNest file: ")
            lbl[ii] = raw_input("Legend label: ")

        # Pull the data for the input parameters
        pltX, pltY, weights, labX, labY = pullResults(p1,p2,mnfile,star_num=0,
                                                      useRel=useRel,fixPar=fixPar)
        (hist, pYbins, pXbins) = np.histogram2d(pltY, pltX,
                                            bins=(bin,bin),weights=weights)
    
        # Need to convert the 2d histogram into floats
        probDist = np.array(hist, dtype=float)
        levels = getContourLevels(probDist,sigma=np.int(sigma))

        py.contour(probDist, levels, origin=None, colors=clrs[ii],
                   extent=[pXbins[0], pXbins[-1], pYbins[0], pYbins[-1]])

        # Get the min and max of the bins to define the axis later
        if ii == 0:
            xmin = pltX.min()
            xmax = pltX.max()
            ymin = pltY.min()
            ymax = pltY.max()
        else:
            xmin = np.min(pltX.min(),xmin)
            xmax = np.max(pltX.max(),xmax)
            ymin = np.min(pltY.min(),ymin)
            ymax = np.max(pltY.max(),ymax)


    if ((p1 == 'x0') or (p1 == 'vx')):
        # Flip the X axis
        xtmp = xmin
        xmin = xmax
        xmax = xtmp
        
    py.axis([xmin,xmax,ymin,ymax]) # automatically determined ranges
    #py.axis([15,-10,-20,5]) # x0,y0
    #py.axis([0.8,-1.2,-0.4,1.6]) # vx,vy
    #py.axis([5,10,2,6]) # R0,mass
    # Where to add labels
    xinc = (xmax - xmin) / 6.
    yinc = (ymax - ymin) / 6.
    xx = xmax - xinc
    yy = ymax - yinc
    #for ii in range(numFits):
        #py.text(xx, yy-(yinc/3.*ii), lbl[ii], color=clrs[ii]) # auto
        #py.text(-4.5, yy-(yinc/3.*ii), lbl[ii], color=clrs[ii]) # x0, y0
        #py.text(-0.65, yy-(yinc/3.*ii), lbl[ii], color=clrs[ii]) # vx, vy
        #py.text(9.5, yy-(yinc/3.*ii), lbl[ii], color=clrs[ii]) # R0,mass
    thePlot = py.gca()
    py.setp( thePlot.get_xticklabels(), fontsize=16, fontweight='bold')
    py.setp( thePlot.get_yticklabels(), fontsize=16, fontweight='bold')  
    font = {'fontsize' : 20}
    py.xlabel(labX, font)
    py.ylabel(labY, font)
    #py.legend(numpoints=1,fancybox=True,prop=prop)
    py.savefig('%s/%s.png' % (outdir,outfile))
    py.savefig('%s/%s.eps' % (outdir,outfile))
    py.close()


def plot_all_1D(adir='/g/ghez/align/test_java/13_10_03/',mnestfile='chains/efit_.txt',star_num=0,
                bin=50,color='black',useRel=False,fixPar=[],suffix=''):
    """
    NOTE: Does not plot GR parameters. 
    """

    # Pull the data for the input parameters
    mnfile = '%s/efit/%s' % (adir, mnestfile)
    
    # This returns all the parameters 
    data, weights, labels = pullResults('M','R0',mnfile,star_num=0,
                                        useRel=useRel,fixPar=fixPar,
                                        getAll=True)

    # Set up placement and colors of the 13 parameters
    params = ['M', 'R0', 'x0', 'y0', 'vx', 'vy', 'vz', 
              'O', 'w', 'i', 'P', 'T0', 'e']
    pltArr = [1,2,4,4,5,5,6,7,8,9,10,11,12]
    clrs = ['k','k','r','b','r','b','k','k','k','k','k','k','k']
    rads = ['O','w','i'] # parameters in radians, so we can change to degrees
    rad2deg = 180.0 / np.pi

    py.clf()
    py.figure(figsize=(8,12))
    py.subplots_adjust(left=0.07,right=0.95,bottom=0.07,top=0.95,
                       wspace=0.35,hspace=0.35)

    fmt = '%s:  %10.5f (1 sigma = %10.5f)'
    for ii in range(len(params)):
        if params[ii] in rads:
            data[params[ii]] *= rad2deg
        plt = pltArr[ii]
        py.subplot(4,3,plt)
        nn,bb,pp = py.hist(data[params[ii]],bins=bin,color=clrs[ii],histtype='step',normed=True)
        py.xlabel(labels[params[ii]])
        print fmt % (labels[params[ii]], bb[nn.argmax()], data[params[ii]].std(ddof=1))

    py.savefig('%s/efit/plots/hist1D_allParams%s.png' % (adir,suffix))
    py.close()


def combinePDFs_mn_multiFits(p1,p2,outfile,numFits,binArray,
                             mnestfile='efit_.txt',star_num=0,
                             outdir='./',useRel=False,fixPar=[],
                             drop_maser=True,ref_star_test=False):
    """
    Combine the 2D histograms of interest for multiple efits that were
    run. Results must be from MultiNest version of efit.

    INPUT:
    p1 (str):         parameter to plot on X axis; options are:
                      'M', 'x0', 'y0', 'vx', 'vy', 'vz', 'R0', 'Mext',
                      'peri', 'alpha', 'O', 'w', 'i', 'P', 'T0', 'e' 
    p2 (str):         parameter to plot on Y axis; options are same as above.
    outfile (str):    Name of output file (do not include extension).
    	 	      To save to a specific dir, must include outdir (see below).
    numFits (int):    Number of efit results to plot. The contours from
                      each efit 2D histogram will be overplotted.
    binArray (float): [array, array] representing the bin edges in each dimension.
    	   	      NOTE!: Important to set this as a pre-determined array of edges so
                      that when the PDFs are combined from multiple fits, they are
                      added to the correct bin. Be sure the bins are both the same size!
                      Example: mass = np.arange(2.0, 6.0, 4.0/50) # will return 50 bins
                               r0 = np.arange(4.0, 10.0, 6./50) # will return 50 bins

                               x0 = np.arange(-5.0, 20.0, 25./50) # want same # of bins and same sizes
                               y0 = np.arange(-20.0, 5.0, 25./50) # want same # of bins and same sizes

                               vx = np.arange(-1.0, 0.5, 1.5/50) # want same # of bins and same sizes
                               vy = np.arange(0.0, 1.5, 1.5/50) # want same # of bins and same sizes

    Input at prompt:
    adir (str):      Path to align directory from which efit results will be pulled.
                     Results are assumed to live in the specified mnestfile for
                     all adir's given at prompt.

    OPTIONAL INPUT:
    mnestfile (str):      File containing efit results. Must include path.
    outdir (str):         Path where output files will be saved.
    bin (int):            Number of bins for the 2 dimensions to be plotted.
    drop_maser (bool):    Special case to combine efit results from the 7 different
    		          drop-a-maser tests.
    ref_star_test (bool): Special case to combine efit results from the 6 different
    		          tests in which different sets of reference stars were selected.

    OUTPUT:
    <outfile>.png/eps:  2D histogram of the 2 input parameters for all align dirs given.
    
    """
    import matplotlib.font_manager
    prop = matplotlib.font_manager.FontProperties(size=9)

    binX = binArray[0]
    binY = binArray[1]

    # Create the array that will hold the combined histogram
    combo_hist = np.zeros((len(binX)-1,len(binY)-1), dtype=float)

    if drop_maser == True: # special case
        msrDir = '/u/syelda/research/gc/aligndir/drop_maser_tests/'
        msrTest = ['13_10_16_no10ee/','13_10_16_no12n/','13_10_16_no15ne/','13_10_16_no17/',
                   '13_10_16_no28/','13_10_16_no7/','13_10_16_no9/']
        outdir = msrDir + 'plots/'

    if ref_star_test == True: # special case
        refDir = '/g/ghez/align/test_java/13_10_03/efit/'
        frac = [0.0, 0.2, 0.4, 0.6, 0.8]
        outdir = refDir + 'plots/'
    
    for ii in range(numFits):
        if drop_maser == True:
            adir = msrDir + msrTest[ii]
            mnfile = '%s/efit/chains/%s' % (adir, mnestfile)
        elif ref_star_test == True:
            adir = refDir 
            mnfile = '%s/chains_align_%s/%s' % (adir, str(frac[ii]), mnestfile)
        else:
            adir = raw_input("Full path to align directory %i: " % (ii+1))

        #mnfile = '%s/efit/%s' % (adir, mnestfile)

        # Pull the data for the input parameters
        pltX, pltY, weights, labX, labY = pullResults(p1,p2,mnfile,star_num=0,
                                                      useRel=useRel,fixPar=fixPar)
        (hist, pYbins, pXbins) = np.histogram2d(pltY, pltX,
                                            bins=(binY,binX),weights=weights)

        # Need to convert the 2d histogram into floats
        probDist = np.array(hist, dtype=float)

        # Stack this 2D histogram with the previous one that we looped through
        combo_hist += probDist

        # Also combine the 1D histograms so we can compute the average and sigmas
        py.figure(2)
        nnX, bbX, ppX = py.hist(pltX, bins=binX)
        nnY, bbY, ppY = py.hist(pltY, bins=binY)
        py.close(2)
        if ii == 0:
            combo_histX = nnX
            combo_histY = nnY
        else:
            combo_histX += nnX
            combo_histY += nnY

    combo_hist /= numFits
    levels = getContourLevels(combo_hist)

    # Create 1D histograms for each parameter
    combo_histX = [np.float(pp) / numFits for pp in combo_histX]
    combo_histY = [np.float(pp) / numFits for pp in combo_histY]

    for ii in range(2):
        if ii == 0:
            prob = np.array(combo_histX) / sum(combo_histX) # normalize
            label = p1
            bins = binX
        elif ii == 1:
            prob = np.array(combo_histY) / sum(combo_histY) # normalize
            label = p2
            bins = binY

        print
        print 'Computing errors on %s' % label

        # Calculate the peak of the probability distribution
        # and the confidence intervals from the 1D Probs.
        sid = (prob.argsort())[::-1]  #  indices for a reverse sort
        probSort = prob[sid]

        peakPix = sid[0]
        peakVal = bins[peakPix]
        peakProb = prob[peakPix]

        # Make a cumulative distribution function starting from the
        # highest pixel value. This way we can find the level above
        # which 68% of the trials will fall.
        cdf = np.cumsum(probSort)

        # Determine point at which we reach XX confidence
        idx1 = (np.where(cdf > 0.6827))[0] # 1 sigma
        idx2 = (np.where(cdf > 0.9545))[0] # 2 sigma
        idx3 = (np.where(cdf > 0.9973))[0] # 3 sigma

        level1 = probSort[idx1[0]]
        level2 = probSort[idx2[0]]
        level3 = probSort[idx3[0]]

        # Find the range of values 
        idx1 = (np.where((prob > level1)))[0]
        idx2 = (np.where((prob > level2)))[0]
        idx3 = (np.where((prob > level3)))[0]

        # Parameter Range:
        range1 = np.array([ bins[idx1[0]], bins[idx1[-1]] ])
        range2 = np.array([ bins[idx2[0]], bins[idx2[-1]] ])
        range3 = np.array([ bins[idx3[0]], bins[idx3[-1]] ])

        # Plus/Minus Errors:
        pmErr1 = np.abs(range1 - peakVal)
        pmErr2 = np.abs(range2 - peakVal)
        pmErr3 = np.abs(range3 - peakVal)
        

        # Find the min and max values for each confidence
        print ''
        print '1, 2, 3 Sigma Confidence Intervals for %s' % label
        print '   68.27%% = [%10.4f -- %10.4f] or -/+ [%10.4f, %10.4f]' % \
              (range1[0], range1[1], pmErr1[0], pmErr1[1])
        print '   95.45%% = [%10.4f -- %10.4f] or -/+ [%10.4f, %10.4f]' % \
              (range2[0], range2[1], pmErr2[0], pmErr2[1])
        print '   99.73%% = [%10.4f -- %10.4f] or -/+ [%10.4f, %10.4f]' % \
              (range3[0], range3[1], pmErr3[0], pmErr3[1])


    py.clf()
    py.figure(1)
    py.figure(figsize=(6,6))
    py.subplots_adjust(left=0.16,right=0.92,bottom=0.13,top=0.92)
    py.imshow(combo_hist, cmap=py.cm.hot_r, origin='lower', aspect='auto',
                 extent=[binX[0], binX[-1], binY[0], binY[-1]])
    py.contour(combo_hist, levels, origin=None, colors='black',
                 extent=[binX[0], binX[-1], binY[0], binY[-1]])
    thePlot = py.gca()
    py.setp(thePlot.get_xticklabels(), fontsize=16, fontweight='bold')
    py.setp(thePlot.get_yticklabels(), fontsize=16, fontweight='bold')  
    font = {'fontsize' : 20}
    if (p1 == 'T0'):
        thePlot.xaxis.set_major_formatter(py.FormatStrFormatter('%6.1f'))
    #rng = py.axis()
    #py.axis([rng[0],rng[1],rng[2],rng[3]])
    xmin = binX[0]
    xmax = binX[-1]
    ymin = binY[0]
    ymax = binY[-1]
    if ((p1 == 'x0') or (p1 == 'vx')):
        # Flip the X axis
        xmin = binX[-1]
        xmax = binX[0]
    #py.axis([binX[0],binX[-1],binY[0],binY[-1]])
    py.axis([xmin,xmax,ymin,ymax]) # auto
    #py.axis([15,-10,-20,5]) # x0,y0
    #py.axis([0.8,-1.2,-0.4,1.6]) # vx,vy
    #py.axis([5,10,2,6]) # R0,mass
    py.xlabel(labX, font)
    py.ylabel(labY, font)
    #py.title('Combined PDFs for %i efits' % numFits,  font)
    py.savefig('%s/%s.png' % (outdir,outfile))
    py.savefig('%s/%s.eps' % (outdir,outfile))
    py.close(1)


#----------
#
# Contours
#
#----------
def getContourLevels(probDist,sigma=None):
    """
    If we want to overlay countours, we need to figure out the
    appropriate levels. The algorithim is:
        1. Sort all pixels in the 2D histogram (largest to smallest)
        2. Make a cumulative distribution function
        3. Find the level at which 68% of trials are enclosed.

    The 1, 2, and 3 sigma contours are returned by default. To
    return only one contour, specify using the 'sigma' keyword.
    sigma keyword must be an integer (1, 2, or 3).
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
    percents = np.array([0.6827, 0.9545, 0.9973])
    if sigma != None:
        sig = np.int(sigma - 1)
        percents = np.array([percents[sig]])
        
    levels = np.zeros(len(percents), dtype=float)
    for ii in range(len(levels)):
        # Get the index of the pixel at which the CDF
        # reaches this percentage (the first one found)
        idx = (np.where(cdf < percents[ii]))[0]
        
        # Now get the level of that pixel
        levels[ii] = pixSort[idx[-1]]

    return levels


#e.plot2_mn('R0','M','R0_mass_no10EE',outdir='plots/',mnestfile='efit/chains/efit_.txt',title='IRS10EE Dropped')
#cd ../13_10_16_no12n
#e.plot2_mn('R0','M','R0_mass_no12N',outdir='plots/',mnestfile='efit/chains/efit_.txt',title='IRS12N Dropped')
#cd ../13_10_16_no15ne
#e.plot2_mn('R0','M','R0_mass_no15NE',outdir='plots/',mnestfile='efit/chains/efit_.txt',title='IRS15NE Dropped')
#cd ../13_10_16_no17
#e.plot2_mn('R0','M','R0_mass_no17',outdir='plots/',mnestfile='efit/chains/efit_.txt',title='IRS17 Dropped')
#cd ../13_10_16_no28
#e.plot2_mn('R0','M','R0_mass_no28',outdir='plots/',mnestfile='efit/chains/efit_.txt',title='IRS28 Dropped')
#cd ../13_10_16_no7
#e.plot2_mn('R0','M','R0_mass_no7',outdir='plots/',mnestfile='efit/chains/efit_.txt',title='IRS7 Dropped')
#cd ../13_10_16_no9
#e.plot2_mn('R0','M','R0_mass_no9',outdir='plots/',mnestfile='efit/chains/efit_.txt',title='IRS9 Dropped')

# Pre-determined binning arrays for mass, R0, vx, vy to input into combinePDFs_mn_multiFits() function
# binM = array([  6.51433735,   6.59405055,   6.67376375,   6.75347695, 6.83319016,   6.91290336,   6.99261656,   7.07232976, 7.15204296,   7.23175616,   7.31146936,   7.39118256, 7.47089576,   7.55060897,   7.63032217,   7.71003537, 7.78974857,   7.86946177,   7.94917497,   8.02888817, 8.10860137,   8.18831457,   8.26802778,   8.34774098, 8.42745418,   8.50716738,   8.58688058,   8.66659378, 8.74630698,   8.82602018,   8.90573338,   8.98544658, 9.06515979,   9.14487299,   9.22458619,   9.30429939, 9.38401259,   9.46372579,   9.54343899,   9.62315219, 9.70286539,   9.7825786 ,   9.8622918 ,   9.942005  , 10.0217182 ,  10.1014314 ,  10.1811446 ,  10.2608578 , 10.340571  ,  10.4202842 ,  10.49999741])



# binR0 = array([ 2.35168   ,  2.4446462 ,  2.53761239,  2.63057859,  2.72354479, 2.81651098,  2.90947718,  3.00244338,  3.09540957,  3.18837577, 3.28134197,  3.37430816,  3.46727436,  3.56024056,  3.65320675, 3.74617295,  3.83913915,  3.93210534,  4.02507154,  4.11803774, 4.21100393,  4.30397013,  4.39693633,  4.48990252,  4.58286872, 4.67583492,  4.76880111,  4.86176731,  4.95473351,  5.0476997 , 5.1406659 ,  5.2336321 ,  5.32659829,  5.41956449,  5.51253069, 5.60549689,  5.69846308,  5.79142928,  5.88439548,  5.97736167, 6.07032787,  6.16329407,  6.25626026,  6.34922646,  6.44219266, 6.53515885,  6.62812505,  6.72109125,  6.81405744,  6.90702364, 6.99998984])

    #binVx = [-0.67560452, -0.64982908, -0.62405364, -0.5982782 , -0.57250276, -0.54672732, -0.52095188, -0.49517644, -0.469401  , -0.44362556, -0.41785012, -0.39207468, -0.36629924, -0.3405238 , -0.31474836, -0.28897292, -0.26319748, -0.23742204, -0.2116466 , -0.18587116, -0.16009572, -0.13432028, -0.10854484, -0.0827694 , -0.05699396, -0.03121852, -0.00544308,  0.02033236,  0.0461078 ,  0.07188324, 0.09765868,  0.12343412,  0.14920956,  0.174985  ,  0.20076044, 0.22653588,  0.25231133,  0.27808677,  0.30386221,  0.32963765, 0.35541309,  0.38118853,  0.40696397,  0.43273941,  0.45851485, 0.48429029,  0.51006573,  0.53584117,  0.56161661,  0.58739205, 0.61316749]

    #binVy = [-0.96534659, -0.90655284, -0.84775908, -0.78896532, -0.73017156, -0.67137781, -0.61258405, -0.55379029, -0.49499653, -0.43620277, -0.37740902, -0.31861526, -0.2598215 , -0.20102774, -0.14223399, -0.08344023, -0.02464647,  0.03414729,  0.09294104,  0.1517348 , 0.21052856,  0.26932232,  0.32811608,  0.38690983,  0.44570359, 0.50449735,  0.56329111,  0.62208486,  0.68087862,  0.73967238, 0.79846614,  0.85725989,  0.91605365,  0.97484741,  1.03364117, 1.09243493,  1.15122868,  1.21002244,  1.2688162 ,  1.32760996, 1.38640371,  1.44519747,  1.50399123,  1.56278499,  1.62157874, 1.6803725 ,  1.73916626,  1.79796002,  1.85675378,  1.91554753, 1.97434129]
