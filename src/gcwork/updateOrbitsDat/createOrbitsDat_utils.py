import NStarOrbits as NSO
import sys
from NStarOrbits import utils
from NStarOrbits import plot
import pandas as pd
import createOrbitsDat_utils as utl
import numpy as np
import math
import matplotlib.pyplot as plt

star = sys.argv



## Write median and 1-sigma quantiles to file ##
def arrays_to_dict(names, data):
    '''
    names = array of strings, 
    data = array or array of arrays
    each element in names becomes a key, with a value equal to the respective array or item in data
    
    '''
    dic = {}
    for n,d in zip(names,data):
        dic[n]=d
    return dic

def get_df_csv(in_dict, headers = ['-1sig', 'med', '+1sig'], out_fname = 'fit_results/'+star[1]+'_weighted_quantiles.csv', save = True): 
    '''
    in_dict must be an OrderedDict
    headers = list of strings indicating column names
    out_fname = str, folder in which to save the csv file
    save = bool, whether you want to save the df to a csv file, or not
    
    #In this function, the keys from in_dict will be the row index name, wherease headers is the key for each column

    '''
    df = pd.DataFrame.from_dict(in_dict, orient = 'index', columns = headers)
    print(df)
    
    if save == True:
        df.to_csv(out_fname)
    return df



## Evaluate deviation in photometry ##


def get_rmse(arr, unbiased_estimator = False, variance = False):
    '''
    Get Root-mean-square error (or RMS deviation) for a set of values
    RMS deviation is the square root of the average of the squared deviation, in other words, the square root of the variance
    If you have data on the whole population, just divide by N (regular way of getting standard deviation). But if you only have a sample and are trying to determine the standard deviation of a whole population, then use N-1

    INPUT: 
    arr, array of values for which you want the rms deviation
    unbiased_estimator: Bool, True if you have data on the whole population (in which case, RMS is calculated by dividing by N), False if you only have a sample and are trying to determine the RMS of a whole population (in which case it is calculated by dividing by N-1)
    variance: Bool, True if you want to get the variance (sigma squared), False if you want to get the square root of the variance, which is the RMS error
    '''
    avg = np.mean(arr)
    resid = 0.
    for i in range(len(arr)):
        resid += (arr[i]-avg)*(arr[i]-avg)
    N = len(arr)
    
    if unbiased_estimator == False:
        var = resid/(N-1)
        rmse = math.sqrt(var)
    if unbiased_estimator == True:
        var = resid/N
        rmse = math.sqrt(var)
        
    if variance == False: 
        return rmse
    
    if variance == True:
        return var



def get_photometry(star, root_dir):
    '''
    star: string, name of star
    root_dir: string, path to STAR.phot file
    
    returns:
    array of dates of observations, array of measured magnitude at each epoch, and array of corresponding uncertainties
    kp_h_change: float, represents the approximate magnitude above which would be h band, below which would be k band. H-band observations are roughly 2 mags fainter in the GC than k observations are.
    '''
    phot_file = np.loadtxt(root_dir + star + '.phot')
    t = phot_file[:,0]
    phot = phot_file[:,6]
    phot_err = phot_file[:,7]

    #kp = []
    #for epoch, mag in zip(t, phot):
     #   if epoch < 2015:
      #      kp.append(mag)
    #kp_h_change = np.median(kp) + 2.   
    return t, phot, phot_err#, kp_h_change


def separate_h_k(star, root_dir, kp_h_change, k_kp_change):#k_kp_change = 2006.):
    '''
    Use a star.phot file to determine which measurements are taken in K (speckle), Kp (AO), and which are taken in H. S-Stars in Hband tend to be about 2 magnitudes fainter than in Kp, so can determine the dates for each based on where you see a clear split. Split between K and Kp is ~2006.
    Note: H band measurements were taken in 2017, 2018 for GR18
    
    INPUT:
    star: string, name of star
    root_dir: string, path to STAR.phot file
    kp_h_change: array of two dates, the date of the first and last h-band epochs, inclusive. For example, as of 2020 data, h-band data are taken between kp_h_change = [2017.3480, 2018.6730]
    k_kp_change: approximate date at which speckle transitioned to AO (K versus Kp filter). Default = 2006.
    RETURNS:
    Three 1D arrays containing the dates of observations, magnitude of observaations, and associated photometric error for each of the three bands(k, kp, and h)
    '''
    t, phot, phot_err = get_photometry(star, root_dir)
    k_phot = []
    k_err = []
    k_dates = []
    h_phot = []
    h_err = []
    h_dates = []
    kp_phot = []
    kp_err = []
    kp_dates = []
    for i in range(len(t)):
        #if before k+kp, it's k
         #   if it's afterk_kp and before kp_h OR after kp_h, it's kp
        #if it's between (inclusive) kp_h[0] and [1] it's h

        if t[i] <= k_kp_change:
            k_phot.append(phot[i])
            k_err.append(phot_err[i])
            k_dates.append(t[i])

        if t[i] > k_kp_change and t[i] < kp_h_change[0]:
            kp_phot.append(phot[i])
            kp_err.append(phot_err[i])
            kp_dates.append(t[i])
        if t[i] > kp_h_change[1]:
            kp_phot.append(phot[i])
            kp_err.append(phot_err[i])
            kp_dates.append(t[i])

        if t[i] >= kp_h_change[0] and t[i] <= kp_h_change[1]:
            h_phot.append(phot[i])
            h_err.append(phot_err[i])
            h_dates.append(t[i])           
        '''
            ##if kp_h change is a magnitude limit as opposed to a date range, can use these criteria. In that case kp_h_change would be a rough magnitude at which Kp mags will be smaller than this value and H band mags will be larger. This value just has to be determined by eye by looking at the photometry vs time plot 
        if phot[i] > kp_h_change:
            h_phot.append(phot[i])
            h_err.append(phot_err[i])
            h_dates.append(t[i])
        if phot[i] < kp_h_change and t[i] > k_kp_change:
            kp_phot.append(phot[i])
            kp_err.append(phot_err[i])
            kp_dates.append(t[i])
        if phot[i] < kp_h_change and t[i] <= k_kp_change:
            k_phot.append(phot[i])
            k_err.append(phot_err[i])
            k_dates.append(t[i])
        '''
    return h_dates, h_phot, h_err, kp_dates, kp_phot, kp_err, k_dates, k_phot, k_err



def get_delta_phot(star, root_dir, kp_h_change, k_kp_change , rmse):
    '''
    star: string, name of star
    root_dir: string, path to STAR.phot file
    rmse: bool. True if you want to include the scatter in mag between epochs in the calculation of sigma; False if you want to calculate sigma for each epoch just based on measurement error
    
    returns: array of photometric change with respect to the median magnitude, in units of sigma. Avg calculated separately for speckle, A0 Kp, and AO H.

    '''
    h_dates, h_phot, h_err, kp_dates, kp_phot, kp_err, k_dates, k_phot, k_err =  separate_h_k(star, root_dir, kp_h_change, k_kp_change)
    
    h_avg = np.median(h_phot)
    h_delta = h_phot - h_avg
    h_rmse = get_rmse(h_phot, unbiased_estimator = False, variance = False)
    
    kp_avg = np.median(kp_phot)
    kp_delta = kp_phot - kp_avg
    kp_rmse = get_rmse(kp_phot, unbiased_estimator = False, variance = False)
    
    k_avg = np.median(k_phot)
    k_delta = k_phot - k_avg
    k_rmse = get_rmse(k_phot, unbiased_estimator = False, variance = False)  
    
    if rmse == True:    #include the scatter between epochs in calculation of 'sigma'
        h_quad_err = []
        for i in range(len(h_err)):
            h_quad_err.append(math.sqrt(h_err[i]*h_err[i] + h_rmse*h_rmse))
        kp_quad_err = []
        for i in range(len(kp_err)):
            kp_quad_err.append(math.sqrt(kp_err[i]*kp_err[i] + kp_rmse*kp_rmse))
        k_quad_err = []
        for i in range(len(k_err)):
            k_quad_err.append(math.sqrt(k_err[i]*k_err[i] + k_rmse*k_rmse))
            
    else:  #only consider measurement error for each epoch in calculation of 'sigma'
        h_quad_err = h_err
        kp_quad_err = kp_err
        k_quad_err = k_err
            
    return h_dates, h_avg, h_delta, h_quad_err, kp_dates, kp_avg, kp_delta, kp_quad_err, k_dates, k_avg, k_delta, k_quad_err



def get_delta_phot_chrono(star, root_dir, kp_h_change, k_kp_change, rmse):
    h_dates, h_avg, h_delta, h_quad_err, kp_dates, kp_avg, kp_delta, kp_quad_err, k_dates, k_avg, k_delta, k_quad_err = get_delta_phot(star, root_dir, kp_h_change, k_kp_change, rmse)

    k = k_delta/k_quad_err
    k = np.sqrt(k*k)
    kp = kp_delta/kp_quad_err
    kp = np.sqrt(kp*kp)
    h = h_delta/h_quad_err
    h = np.sqrt(h*h)

    date_tot = []
    dm_tot = []  #diff in mag for all dates
    for i,j in zip(k_dates,k):
        dm_tot.append(j)
        date_tot.append(i)
    for i,j in zip(kp_dates,kp):
        dm_tot.append(j)
        date_tot.append(i)
    for i,j in zip(h_dates,h):
        dm_tot.append(j)
        date_tot.append(i)

    sorted_dm = sorted(zip(date_tot, dm_tot))  #sort by date
    date_tot = np.array(sorted_dm)[:,0]
    dm_tot = np.array(sorted_dm)[:,1]

    return (date_tot, dm_tot)


def plot_phot(star, root_dir, plot_err = True, save_to = False):
    '''
    star: string, name of star
    root_dir: string, path to STAR.phot file
    save_to = either False if you don't want to save the fig, or path to the folder where you'd like to save the fig if you do want to save it 
    '''
    fig = plt.figure(figsize = (12,9))
    t, phot, phot_err = get_photometry(star, root_dir)
    if plot_err == True:
        plt.errorbar(t,phot, yerr = phot_err, ls = 'none', marker = '+')
    else:
        plt.scatter(t,phot)
    
    fs = 18 #fontsize
    ts = 16 #ticksize
    plt.xlabel('Year', fontsize = fs)
    plt.ylabel('Magnitude', fontsize = fs)
    plt.xticks(fontsize = ts)
    plt.yticks(fontsize = ts)

    plt.tight_layout()
    
    if save_to == False:
        plt.show()
    else:
        plt.savefig(save_to+star+'_phot.png')
    return 0
        
def plot_pvalue_vs_delta_phot(delta_phot, pvalue_x, pvalue_y, epochs, p_cutoff, dm_cutoff, save_to):    
    px = np.array(pvalue_x)
    py = np.array(pvalue_y)

    #if pvalue is less than zero, it is a numerical error, just means it has very very low probability, so just set to zero. Otherwise the quad sum of x and y will make it seem like it has high probability. 
    for i in range(len(px)):
        if px[i] < 0.:
            px[i] = 0.
        if py[i] < 0.:
            py[i] = 0.
            
    p_sq = np.sqrt(px**2 + py**2)
    p_sq_normed = p_sq / np.max(p_sq)

    y = [p_sq_normed, 100.*px, 100.*py]
    ylabels = ['P-Value (x and y, normalized)', 'X P-Value (Percentage)', 'Y P-Value (Percentage)']
    
    fig, axes = plt.subplots(nrows = len(y), ncols = 1, figsize = (9,12))
    for ax,i in zip(axes.flat, range(len(y))):
        im = ax.scatter(delta_phot, y[i], c = epochs)
        ax.set_ylabel(ylabels[i])#, fontsize = 16)
        #ax.set_ylim(ymin=0.)
        ax.axvline(dm_cutoff, linestyle = 'dashed', color = 'gray')
        if i > 0: #indicate pvalue cutoff for the x,y individual plots, i.e. not for the first subplot, which is the quad sum of x and y
            ax.axhline(p_cutoff*100., linestyle = 'dashed', color = 'gray')
        

    plt.xlabel('Photometric Deviation (sigma)')#, fontsize = 16)
    fig.colorbar(im, ax=axes.ravel().tolist(), location = 'right')
    plt.savefig(save_to+'_pvalue_vs_DeltaPhot.png')


def get_points(star, root_dir):
    points_file = np.loadtxt(root_dir + star + '.points')
    t = points_file[:,0]
    x = points_file[:,1]
    y = points_file[:,2]
    dx = points_file[:,3]
    dy = points_file[:,4]
    return t,x,y,dx,dy


def plot_orbit(star, root_dir, save_to):
    t,x,y,dx,dy = get_points(star, root_dir)

    fig = plt.figure()
    plt.errorbar(-x,y,xerr=dx,yerr=dy, marker = 'o', mfc = 'none', ms = 6, ls='none')
    plt.xlabel('$\Delta$RA (arcsec)', fontsize = 16)
    plt.ylabel('$\Delta$Dec (arcsec)', fontsize = 16)
    plt.xticks(fontsize = 16)
    plt.yticks(fontsize = 16)
    xmin = np.min(-x)- 0.01
    xmax = np.max(-x)+ 0.01
    plt.xlim(xmax, xmin)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(save_to+star+'_orbit.png')
