import asciidata
import numpy as np
import pylab as py
import pyfits
from gcwork.plotgc import gccolors 
from gcwork import objects
from gcwork import starset
import young
from pylab import *
from matplotlib.colors import colorConverter
import colorsys
import pdb

def go(modelDir, outdir, radius=0.6, nodata=True, nolabel=False, noStarLabel=False, addScale=False,
       bnw=False, stars=None,clr=None,suffix='',ly=None,diffSyms=False,ngao=False):
    """
    Plots central arcsecond image with orbits overplotted
    (Originally written in IDL; converted to Python by S. Yelda)

    Latest model directory with most orbits is:
    /u/syelda/research/gc/aligndir/test_schoedel/11_06_08/efit/

    Options:
    bnw (bool):	             Make a black and white image
    stars (list of strings): If you only want to plot certain stars, specify with this keyword.
    clr (list of strings):   If you only want to select the color for certain stars,
                             specify with this keyword.
    ly (float):		     Location in arcseconds for the placement of the star's label
    			     along the Y axis. Only useful when also using stars option to
                             specify star to plot.
    diffSyms (bool):	     Set to True to give each star a different symbol. Used only with 
    			     black and white figure (bnw=True).
    """

    outroot = 'plot_central_image' + suffix
    #outroot = 'plot_central_image'
    if nolabel == True:
        outroot += '_nolabel'

    #refPosX = None
    #refPosY = None

    if stars == None:
        # Stars to be plotted
        #stars = ['S0-1','S0-2','S0-5','S0-16','S0-19','S0-20','S0-38','S0-49','S0-102','S0-103','S0-104']
        #stars = ['S0-1', 'S0-2', 'S0-5', 'S0-16', 'S0-19', 'S0-20', 'S0-38', 'S0-102']
        #stars = ['S0-1','S0-2','S0-5','S0-16','S0-19','S0-20', 'S0-38', 'S0-102','S0-104']
        stars = ['S0-2','S0-38']

        # Specific for NGAO proposal:
        #stars = ['S0-2','S0-102','S0-38','sim125006','sim188985','sim31961','sim63321']
        #refPosX = [514.560, 507.170, 538.25, 529.980, 516.576, 512.454, 524.610] 
        #refPosY = [632.250, 596.940, 610.25, 592.170, 614.747, 614.214, 612.047] 
        #refPosXa = np.zeros(len(refPosX),dtype=float)
        #refPosYa = np.zeros(len(refPosY),dtype=float)
        # End NGAO-specific code

    if clr == None:
        clr = ['red', 'blue', 'purple', 'steelblue', 'cyan',
               'magenta', 'green', 'salmon','orange','yellow']
    #clr = ['red', 'cyan', 'orange' , 'steelblue', 'yellowgreen', 'purple', 'green', 'magenta','yellow','salmon']
    # If the star was discovered during speckle, use solid
    # If discovered during AO, use dashed
    #linestyle = ['-', '-', '-', '-','--', '--', '-']  # ngao v1
    #linestyle = ['-', '-', '-', '--','--', '--', '--']  # ngao v2
    linestyle = ['-','-', '-', '-', '-','-', '--', '--','--'] # current ao
    discover = [1995.5, 1995.5, 1995.5, 1995.5, 1995.5, 1995.5, 1995.5, 2005.5, 2005.5]


    rng = radius
    xcen = 0.0
    ycen = 0.0
    delta = rng

    # Set up the background image
    angle = 0.0
    scale = 0.00995
    if ngao == True:
        imgFile = '/u/syelda/research/gc/TMT/TMTsims/iris_code_20120130/code/KeckSims/NGAO/new/galcenter_stars_noise___1.fits'
        sgra = [512.5, 614.0]
    else:
        #imgFile = '/u/ghezgroup/data/gc/14maylgs2/combo/mag14maylgs2_kp.fits'
        imgFile = '/u/ghezgroup/data/gc/15auglgs/combo/mag15auglgs_kp.fits'
        sgra = [ 591.91302, 703.6]
        #imgFile = '/u/syelda/research/gc/TMT/TMTsims/iris_code_20120130/code/KeckSims/NGAO/currentAO/galcenter_stars_noise___1.fits'
        #sgra = [512.5, 614.0]

    img = pyfits.getdata(imgFile)
    imgsize = (img.shape)[0]
    # Make axes for images in arcsec
    pixL = np.arange(0,imgsize)
    xL = [-1*(xpos - sgra[0])*scale for xpos in pixL]
    yL = [(ypos - sgra[1])*scale for ypos in pixL]
    #if refPosX != None:
    #    for ii in range(len(refPosX)):
    #        refPosXa[ii] = -1*(refPosX[ii] - sgra[0])*scale
    #        refPosYa[ii] = (refPosY[ii] - sgra[1])*scale

    if bnw == True:
        tclr = 'black'
        # Define symbols for each star
        #sym = ['s', 'o', '^', 'h','p', 'v', '<', '*']
        sym = ['o']*len(stars)
    else:
        tclr = 'white'
        sym = ['o']*len(stars)

    # Plot
    py.figure(1)
    py.clf()
    py.figure(figsize=(6,6))
    #py.figure(figsize=(7,5.5))
    py.subplots_adjust(left=0.02,right=0.98,top=0.98,bottom=0.02)
    if bnw == True:
        py.imshow(np.log10(img), aspect='equal', interpolation='bicubic',
                  #extent=[max(xL), min(xL), min(yL), max(yL)],vmin=4.4,vmax=6.1, # good for ngao bnw
                  #extent=[max(xL), min(xL), min(yL), max(yL)],vmin=4.4,vmax=6.0, # good for ao bnw
                  extent=[max(xL), min(xL), min(yL), max(yL)],vmin=2.5,vmax=3.5,
                  origin='lowerleft', cmap=py.cm.gray_r)
    else:
        py.imshow(np.log10(img), aspect='equal', interpolation='bicubic',
                  extent=[max(xL), min(xL), min(yL), max(yL)],vmin=2.6,vmax=4.0,
                  origin='lowerleft')#,cmap=py.cm.spectral)

    # Loop through the stars and get the model fits
    for ss in range(len(stars)):
        #mod = asciidata.open(modelDir + 'orbit.' + stars[ss] + '_fixed.model')
        # Temporary -- use our latest orbit for S0-2
        if stars[ss] == 'S0-2':
            mod = asciidata.open('/g/ghez/bsitarski/S0-2_orbits/efit_w_S0-38/orbit.S0-2_2.model')
            #mod = asciidata.open('/u/bsitarski/orbits/S0-2/orbit.' + stars[ss] + '.model')
        if stars[ss] == 'S0-4':
            mod = asciidata.open('/u/bsitarski/orbits/S0-4/orbit.' + stars[ss] + '.model')
        elif stars[ss] == 'S0-16':
            mod = asciidata.open('/u/bsitarski/orbits/S0-16/orbit.' + stars[ss] + '.model')
        elif stars[ss] == 'S0-19':
            mod = asciidata.open('/u/syelda/research/gc/aligndir/test_schoedel/12_07_03/efit/orbitsdat/orbit.' + stars[ss] + '.model')
        elif stars[ss] == 'S0-20':
            mod = asciidata.open('/u/bsitarski/orbits/S0-20/orbit.' + stars[ss] + '.model')
        elif (stars[ss] == 'S0-104'):
            #mod = asciidata.open('/u/syelda/research/gc/aligndir/test_schoedel/12_05_22/efit/orbit.' + \
            #                     stars[ss] + '_fixedto0.model')
            mod = asciidata.open('/g/ghez/bsitarski/S0-2_orbits/efit_w_S0-38/orbit.S0-38_2.model')

        elif (stars[ss] == 'S0-102'):
            mod = asciidata.open('/u/bsitarski/orbits/S0-102/orbit.S0-102_new.model')
            #mod = asciidata.open('/u/syelda/research/gc/aligndir/test_schoedel/12_05_22/efit/orbit.' + \
                                 #stars[ss] + '_fixed.model')
        elif stars[ss] == 'S0-103':
            mod = asciidata.open('/u/bsitarski/orbits/S0-103/orbit.' + stars[ss] + '.model')
            #mod = asciidata.open('/u/aboehle/research/orbits/model_files/orbit.' + \
            #                     stars[ss] + '_no2006_wrest.model')
        elif stars[ss] == 'S0-38':
            #####mod = asciidata.open('/u/aboehle/research/orbits/model_files/orbit.' + \
            #####                     stars[ss] + '_all.model')
            #mod = asciidata.open('/g/ghez/align/test_schoedel/14_01_15/efit/orbit.S0-38_97may_1err.model')
            mod = asciidata.open('/u/bsitarski/orbits/S0-38/orbit.S0-38.model')
        elif stars[ss] == 'S0-49':
            mod = asciidata.open('/u/aboehle/research/orbits/model_files/orbit.' + \
                                 stars[ss] + '.model')
        elif stars[ss] == 'S0-1':
            mod = asciidata.open('/u/bsitarski/orbits/S0-1/orbit.S0-1_maybe.model') 
        elif stars[ss] == 'S0-5':
            mod = asciidata.open('/u/bsitarski/orbits/S0-5/orbit.S0-5_new.model')
        elif 'sim' in stars[ss]:
            mod = asciidata.open('/u/syelda/research/gc/TMT/TMTsims/iris_code_20120130/code/KeckSims/NGAO/' + \
                  stars[ss] + '.points')
        else:
            mod = asciidata.open(modelDir + 'orbit.' + stars[ss] + '.model')
        mt = mod[0].tonumpy()
        mx = mod[1].tonumpy() * -1.0
        if 'sim' in stars[ss]:
            mx = mod[1].tonumpy() 
        my = mod[2].tonumpy()

        print 'Plotting star %s' % stars[ss]

        idx = np.where((mt >= discover[ss]) & (mt < 2015.8))[0]
        emt = mt[idx]
        emx = mx[idx]
        emy = my[idx]

        # Plot the orbit model
        if bnw == True:
            #clr[ss] = 'k'
            if stars[ss] == 'S0-2':
                py.plot(mx[idx], my[idx], color=clr[ss], linestyle=linestyle[ss], lw=2.0)
            else:
                py.plot(mx[idx], my[idx], color=clr[ss], linestyle=linestyle[ss], lw=2.0)
        else:
            if stars[ss] == 'S0-2':
                py.plot(mx[idx], my[idx], color=clr[ss], linestyle=linestyle[ss], lw=2.0)
            else:
                py.plot(mx[idx], my[idx], color=clr[ss], linestyle=linestyle[ss], lw=2.0)
        # Plot the data points over the model
        epochs = np.arange(discover[ss], discover[ss] + 23, 1)
        numE = len(epochs)
        starname = stars[ss]

        # Get the colors for each star
        if bnw == True:
            rgb = asarray(colorConverter.to_rgb('0.3'))
        else:
            rgb = asarray(colorConverter.to_rgb(clr[ss]))
        hue = colorsys.rgb_to_hsv(rgb[0],rgb[1],rgb[2])

        for ee in range(numE):
            if nodata == True:
                # Plot the model
                tdiff = np.abs(emt - epochs[ee])
                t = np.where(tdiff == tdiff.min())[0]
                x = emx[t]
                y = emy[t]

                # Circles for data points
                # Make the colors fade
                rgb = colorsys.hsv_to_rgb(hue[0],hue[1]/((numE-ee)/8.0+1.0),hue[2])
                if stars[ss] == 'S0-2':
                    py.plot(x, y, color=rgb, marker=sym[ss], ms=5)
                    #pdb.set_trace()
                else:
                    py.plot(x, y, color=rgb, marker=sym[ss], ms=5)
                # Black edge around data points
                py.plot(x, y, mfc='none', mec='k', ms=5)

        #if refPosXa != None:
        #    py.plot(refPosXa[ss], refPosYa[ss], color=clr[ss], marker=sym[ss], ms=7)
        #    py.plot(refPosXa[ss], refPosYa[ss], mfc='none', mec='k', ms=7)

        lx = -0.44
        if ly == None:
            ly = 0.14
        
        if ((nolabel == False) & (noStarLabel == False)):
            py.text(lx,ly-(0.05*ss),starname,weight='bold',fontsize=13, color=tclr)
            py.plot(lx+0.03, ly+0.01-(0.05*ss), color=clr[ss], marker=sym[ss], ms=7)
            py.plot(lx+0.03, ly+0.01-(0.05*ss), mfc='none', mec='k', ms=7)

    if nolabel == False:
        # Add some text to plot
        #py.text(-rng+0.38, -rng+0.14, 'Keck/UCLA', weight='bold', color=tclr, fontsize=14)
        #py.text(-rng+0.53, -rng+0.09, 'Galactic Center Group', weight='bold', color=tclr, fontsize=14)
        #py.text(-rng+0.38, -rng+0.05, '1995-2012', weight='bold', color=tclr, fontsize=14)
        py.text(rng-0.17, -rng+0.10, 'Keck/UCLA', weight='bold', color=tclr, fontsize=16)
        py.text(rng-0.05, -rng+0.05, 'Galactic Center Group', weight='bold', color=tclr, fontsize=16)
        py.text(-rng+0.33, -rng+0.05, '1995-2015', weight='bold', color=tclr, fontsize=16)

        # Add scale
        if bnw == True:
            py.plot([0.45, 0.35], [0.55, 0.55], 'k-', lw=3)
        else:
            py.plot([0.45, 0.35], [0.55, 0.55], 'w-', lw=3)
        py.text(0.44,0.5,'0.1"',fontsize=14, weight='bold', color=tclr)

        # Add compass
        qvr = py.quiver([-0.48], [0.426], [0], [0.1], color=tclr, units='width', scale=1)
        py.text(-0.43,0.53,'N',fontsize=14, weight='bold', color=tclr)
        qvr = py.quiver([-0.4802], [0.43], [-0.1], [0], color=tclr, units='width', scale=1)
        py.text(-0.35,0.45,'E',fontsize=14, weight='bold', color=tclr)

    # Do we just want a scale?
    if ((addScale == True) and (nolabel == True)):
        # Add scale
        if bnw == True:
            py.plot([0.25, 0.15], [-0.25, -0.25], 'k-', lw=3)
            #py.plot([0.45, 0.35], [0.55, 0.55], 'k-', lw=3)
        else:
            py.plot([0.45, 0.35], [0.55, 0.55], 'w-', lw=3)
        py.text(0.22,-0.235,'0.1"',fontsize=14, weight='bold', color=tclr)

    #py.text(0.05,0.26,'NGAO',fontsize=22, weight='bold', color=tclr)
    #py.text(0.1,0.26,'Current AO',fontsize=22, weight='bold', color=tclr)

    py.axis([rng,-rng,-rng,rng])
    thePlot = py.gca()
    py.setp(thePlot.set_xticks([]))
    py.setp(thePlot.set_yticks([]))
    py.setp(thePlot.get_xticklabels(),visible=False)
    py.setp(thePlot.get_yticklabels(),visible=False)
    
    py.savefig(outdir + outroot + '_hires.png', dpi=400)
    #py.savefig(outdir + outroot + '_hires.png', dpi=300)
    py.savefig(outdir + outroot + '.png')



def plot_S02_S0102(modelDir, outdir, nodata=True, nolabel=False, transparent=False, suffix=''):
    """
    Plots central arcsecond image with S0-2's and S0-102's orbits overplotted
    """

    outroot = 'plot_central_image' + suffix
    if nolabel == True:
        outroot += '_nolabel'
    
    root = '/u/syelda/research/gc/aligndir/'

    # Stars to be plotted
    stars = ['S0-2','S0-102'] #,'S0-38']
    clr = ['red', 'magenta', 'magenta']
    # If the star was discovered during speckle, use solid
    # If discovered during AO, use dashed
    linestyle = ['-', '-', '-', '-', '-', '-', '-']
    discover = [1995.5, 1995.5, 1995.5, 1995.5, 1995.5, 1995.5, 1995.5]

    rng = 0.45
    xcen = 0.0
    ycen = 0.0
    delta = rng

    # Set up the background image
    angle = 0.0
    scale = 0.00995
    sgra = [575., 677.]
    imgFile = '/u/ghezgroup/data/gc/12maylgs/combo/mag12maylgs_kp.fits'
    img = pyfits.getdata(imgFile)
    imgsize = (img.shape)[0]
    # Make axes for images in arcsec
    pixL = np.arange(0,imgsize)
    xL = [-1*(xpos - sgra[0])*scale for xpos in pixL]
    yL = [(ypos - sgra[1])*scale for ypos in pixL]

    # Plot
    py.figure(1)
    py.clf()
    py.figure(figsize=(6,6))
    py.subplots_adjust(left=0.02,right=0.98,top=0.98,bottom=0.02)
    py.imshow(np.log10(img), aspect='equal', interpolation='bicubic',
              extent=[max(xL), min(xL), min(yL), max(yL)],vmin=2.3,vmax=4.1,
              origin='lowerleft')

    # Loop through the stars and get the model fits
    for ss in range(len(stars)):
        #mod = asciidata.open(modelDir + 'orbit.' + stars[ss] + '_fixed.model')
        # Temporary -- use our latest orbit for S0-2
        if stars[ss] == 'S0-2':
            mod = asciidata.open(root + '/test_schoedel/11_06_08/efit/orbit.' + \
                                 stars[ss] + '.fixed.model')
        elif stars[ss] == 'S0-102':
            mod = asciidata.open(root + '/test_schoedel/12_05_22/efit/orbit.' + \
                                 stars[ss] + '_fixed.model')
        elif stars[ss] == 'S0-38':
            mod = asciidata.open(root + '/test_schoedel/12_07_03/efit/orbit.' + \
                                 stars[ss] + '_all.model')
        mt = mod[0].tonumpy()
        mx = mod[1].tonumpy() * -1.0
        my = mod[2].tonumpy()

        idx = np.where((mt >= discover[ss]) & (mt < 2015.8))[0]
        emt = mt[idx]
        emx = mx[idx]
        emy = my[idx]

        # Plot the orbit model
        py.plot(mx, my, color=clr[ss], linestyle=linestyle[ss], lw=1.5)

        # Plot the data points over the model
        epochs = np.arange(discover[ss], discover[ss] + 17, 1)
        numE = len(epochs)
        starname = stars[ss]

        # Get the colors for each star
        rgb = asarray(colorConverter.to_rgb(clr[ss]))
        hue = colorsys.rgb_to_hsv(rgb[0],rgb[1],rgb[2])

        for ee in range(numE):
            if nodata == True:
                # Plot the model
                tdiff = np.abs(emt - epochs[ee])
                t = np.where(tdiff == tdiff.min())[0]
                x = emx[t]
                y = emy[t]

                # Circles for data points
                # Make the colors fade
                rgb = colorsys.hsv_to_rgb(hue[0],hue[1]/((numE-ee)/8.0+1.0),hue[2])
                py.plot(x, y, color=rgb, marker='o', ms=6)
                # Black edge around data points
                py.plot(x, y, mfc='none', mec='k', ms=6)

        lx = -rng+0.15
        ly = 0.05
        if nolabel == False:
            py.text(lx,ly-(0.05*ss),starname,weight='bold',fontsize=14, color='white')
            py.plot(lx+0.03, ly+0.01-(0.05*ss), color=clr[ss], marker='o', ms=9)
            py.plot(lx+0.03, ly+0.01-(0.05*ss), mfc='none', mec='k', ms=9)

    if nolabel == False:
        # Add some text to plot
        py.text(-rng+0.3, -rng+0.14, 'Keck/UCLA', weight='bold', color='white', fontsize=14)
        py.text(-rng+0.38, -rng+0.09, 'Galactic Center Group', weight='bold', color='white', fontsize=14)
        py.text(-rng+0.3, -rng+0.05, '1995-2015', weight='bold', color='white', fontsize=14)
        #py.text(rng-0.12, -rng+0.07, 'Keck/UCLA', weight='bold', color='white', fontsize=17)
        #py.text(rng-0.02, -rng+0.02, 'Galactic Center Group', weight='bold', color='white', fontsize=17)
        #py.text(rng-0.57, -rng+0.04, '1995-2012', weight='bold', color='white', fontsize=17)

        # Add scale
        py.plot([rng-0.05, rng-0.15], [rng-0.03, rng-0.03], 'w-', lw=3)
        py.text(rng-0.07,rng-0.08,'0.1"',fontsize=16, weight='bold', color='white')

        # Add compass
        qvr = py.quiver([-rng+0.10], [rng-0.11], [0], [0.1], color='white', units='width', scale=1)
        py.text(-rng+0.14,rng-0.04,'N',fontsize=18, weight='bold', color='white')
        qvr = py.quiver([-rng+0.102], [rng-0.107], [-0.1], [0], color='white', units='width', scale=1)
        py.text(-rng+0.215,rng-0.12,'E',fontsize=18, weight='bold', color='white')

    py.axis([rng,-rng,-rng,rng])
    thePlot = py.gca()
    py.setp(thePlot.set_xticks([]))
    py.setp(thePlot.set_yticks([]))
    py.setp(thePlot.get_xticklabels(),visible=False)
    py.setp(thePlot.get_yticklabels(),visible=False)
    
    py.savefig(outdir + outroot + '_hires.png', dpi=300, transparent=transparent)
    #py.savefig(outdir + outroot + '_hires.png', dpi=300)
    py.savefig(outdir + outroot + '.png',transparent=transparent)




def temp():
    f = asciidata.open('/u/syelda/research/gc/anim/orbits/science_press/orbits_movie_science_press2012.dat')
    n = f[0].tonumpy()
    p = f[1].tonumpy()
    d = f[9].tonumpy()

    frac = (2012.5 - d) / p

    for ff in range(len(n)):
        print '%6s   fraction covered = %6.4f' % (n[ff], frac[ff])

