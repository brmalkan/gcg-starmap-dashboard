import astropy
from astropy.table import Table
import os
from gcreduce import align_epochs
from gcwork import manual_points_change,accel_class,starset
from gcwork.polyfit import residuals as res
from gcwork.align import compare_pos
import numpy as np
import matplotlib
import matplotlib.colors as colors
import matplotlib.cm as cm
import pylab as plt
import os
from scipy.stats import chi2
from collections import OrderedDict
import pandas as pd
import matplotlib.lines as mlines

def find_star(starname, starset_obj, ref_points,show_plots=False,delta_r=0.1,delta_mag=1.0):
    '''

    Using a starset object and a reference points list, try to
    determine which star in each epoch best matches the reference
    points file. The matching will be done via the epoch year and
    closest distance (in arcseconds) and magnitude.

    Inputs
    ------
    starname - name of the star
    starset_obj - a StarSet object that has an align & points files loaded
    ref_points - a .points file that has the reference points

    Keywords
    --------
    delta_r - distance from the a point to be called a match
    delta_mag - magntiude difference from a point to be called a match



    '''


    name = np.array(starset_obj.getArray('name'))
    mag = starset_obj.getArray('mag') * 1.0
    years = starset_obj.getArray('years')[0]
    r = starset_obj.getArray('r2d')
    x = -np.array(starset_obj.getArray('x'))
    y = np.array(starset_obj.getArray('y'))
    #nEpochs = starset_obj.getArray('velCnt')
    nEpochs = len(years)

    print('n stars: '+str(len(name)))
    print('n epochs: '+str(len(years)))


    # Set up a color scheme
    cnorm = colors.Normalize(years.min(), years.max()+1)
    #cmap = cm.gist_ncar
    #cmap = cm.inferno_r
    cmap = cm.nipy_spectral
    yearsInt = np.floor(years)

    colorlist = []
    for ee in range(nEpochs):
        colorlist.append( cmap(cnorm(yearsInt[ee])) )

    # find the current detections
    ind = np.where(name == starname)[0]
    print(starname,ind)
    if len(ind) > 0:

        # positional errors
        xerr_p = starset_obj.stars[ind[0]].getArrayAllEpochs('xerr_p')
        yerr_p = starset_obj.stars[ind[0]].getArrayAllEpochs('yerr_p')

        # alignment errors
        xerr_a = starset_obj.stars[ind[0]].getArrayAllEpochs('xerr_a')
        yerr_a = starset_obj.stars[ind[0]].getArrayAllEpochs('yerr_a')

        xerr = np.sqrt(xerr_p**2 + xerr_a**2)
        yerr = np.sqrt(yerr_p**2 + yerr_a**2)

        mag = starset_obj.stars[ind[0]].getArrayAllEpochs('mag')
        indict = OrderedDict([('date',years),('mag',mag),
                             ('x',-starset_obj.stars[ind[0]].getArrayAllEpochs('x')),
                             ('xerr',xerr),
                             ('y',starset_obj.stars[ind[0]].getArrayAllEpochs('y')),
                             ('yerr',yerr)])

        intab = pd.DataFrame(indict)
        valid = intab[intab['y'] > -1000]
        
        # read in the reference file
        ref = pd.read_csv(ref_points,delim_whitespace=True,header=None,float_precision='round_trip',
                          names=['date','x','y','xerr','yerr'],usecols=[0,1,2,3,4])
        #print(ref)
        #print(intab)

        combo = ref.merge(intab,on='date',how='left',suffixes=('_ref','_align'))
        #print(len(combo))
        #print(combo)
        not_align = combo[combo['y_align'] < -1000]
        in_align = combo[combo['y_align'] > -1000]
        
        # make a plot of the align vs the reference points
        plt.subplot(2,1,1)
        #print(valid.index)
        #print(colorlist)
        for jj in np.arange(len(ref)):
            epoch = ref['date'].iloc[jj]
            year_ind = np.where(years == epoch)[0]
            year_ind = year_ind[0]
            current_x = ref['x'].iloc[jj]
            current_y = ref['y'].iloc[jj]
            plt.plot(current_x,current_y,'o',color=colorlist[year_ind],ms=10.0,markeredgecolor='k')            

        for ii in np.arange(len(valid)):
            plt.plot(valid['x'].iloc[ii],valid['y'].iloc[ii],marker='^',c=colorlist[valid.index[ii]],markeredgecolor='k',ms=10)
            
        plt.xlabel('$\Delta$ RA (arcsec)')
        plt.ylabel('$\Delta$ DEC (arcsec)')            
        plt.axis('equal')



        circle = mlines.Line2D([],[],linewidth=0,color='k', marker='o',
                          markersize=15, label=starname+' Reference Points File')
        triangle = mlines.Line2D([], [], linewidth=0,color='k', marker='^',
                          markersize=15, label=starname+' Points in Align')
        plt.legend(handles=[circle,triangle])
        
        # make year annotation
        space = 1./int(years.max()-years.min())
        previous_year = 0
        n_year = 0
        for i in range(len(years)):
            year = years[i]
            if int(previous_year) != int(year):
                plt.annotate(str(int(year)), (1.03, -0.05 + n_year*space), color=colorlist[i],
                             xycoords='axes fraction', fontsize=15)
                previous_year = year
                n_year += 1
        
        print(not_align)
        for i in np.arange(len(not_align)):

            epoch = not_align['date'].iloc[i]
            year_ind = np.where(years == epoch)[0]
            year_ind = year_ind[0]

            current_x = not_align['x_ref'].iloc[i]
            current_y = not_align['y_ref'].iloc[i]
            current_mag = not_align['mag'].iloc[i]

            epoch_x = starset_obj.getArrayFromEpoch(year_ind,'x')
            epoch_y = starset_obj.getArrayFromEpoch(year_ind,'y')
            epoch_mag = starset_obj.getArrayFromEpoch(year_ind,'mag')
            epoch_name = starset_obj.getArrayFromEpoch(year_ind,'name')
            dist = np.sqrt((epoch_x-current_x)**2 + (epoch_y-current_y)**2)
            good = np.where(dist <= delta_r)[0]
            if i == 0:
                plt.subplot(2,1,2)
            plt.plot(current_x,current_y,'o',color=colorlist[year_ind],ms=10.0,markeredgecolor='k')
            plt.text(current_x,current_y,str(epoch))
            if len(good) > 0:
                plt.plot(epoch_x[good],epoch_y[good],'^',ms=8,color=colorlist[year_ind])
                for g in good:
                    plt.plot([current_x,epoch_x[g]],[current_y,epoch_y[g]],
                             '--',color=colorlist[year_ind])
                    txt = '%s, %3.1f ' % (name[g],epoch_mag[g])
                    plt.text(epoch_x[g],epoch_y[g],txt,fontsize=11)
        plt.xlabel('$\Delta$ RA (arcsec)')
        plt.ylabel('$\Delta$ DEC (arcsec)')            
        plt.axis('equal')
