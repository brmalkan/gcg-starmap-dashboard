import numpy as np
import matplotlib.pyplot as plt
import os
from math import sqrt

import gcwork.efit.efit5_results as ef

def getContourLevels(probDist,percents = np.array([0.6827, .95, .997])):
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
    cdf = cdf/max(cdf)
    
    # Determine point at which we reach 68% level
    
    levels = np.zeros(len(percents), dtype=float)
    for ii in range(len(levels)):
        # Get the index of the pixel at which the CDF
        # reaches this percentage (the first one found)
        idx = (np.where(cdf < percents[ii]))[0]
        
        # Now get the level of that pixel
        levels[ii] = pixSort[idx[-1]]
    return levels

def weighted_percentile(data, percents, weights=None):
    ''' percents in units of 1%
        weights specifies the frequency (count) of data.
    '''
    if weights is None:
        return np.percentile(data, 100*percents)
    ind=np.argsort(data)
    d=data[ind]
    w=weights[ind]
    p=1.*w.cumsum()/w.sum()
    y=np.interp(percents, p, d)
    return y



def produceFig(rootDir,with3sigma=False,overwrite=False,processes=1):
    dir_all='S0-2_all/'
    dir_s='S0-2_speckle/'
    dir_a='S0-2_AO/'
    
    cwd = os.getcwd()
    os.chdir(rootDir)
    
    # loading the chains
    dat_all = ef.efit5(dir_all+'orbit.S0-2')
    dat_s = ef.efit5(dir_s+'orbit.S0-2')
    dat_a = ef.efit5(dir_a+'orbit.S0-2')
    
    # computing residuals (for the case of all data)
    if overwrite or not os.path.exists(dat_all.stars[0].path_res+'_res.rv'):
        print('processing residuals from orbit')
        dat_all.all_residuals(processes)
    else:
        print('Already processed residuals, file exists: '+dat_all.stars[0].path_res+'_res.rv')

    # ploting residuals (for the case of all data)
    dat_all.stars[0].plot_astro_residuals(outname='S0-2_all')


    # percentiles recorded
    perc = np.array([ 0.15865, 0.84135, 0.0015, 0.9985])
    
    if with3sigma:
        levels=np.array([.997,0.6827])
        ls = ['--','-']
    else:
        levels=np.array([0.6827])
        ls= ['-']

    # factors used for plots
    fmas=1e3
    nbin=50
    

    n_all='All'
    n_a='AO'
    n_s='Speckle'
    col_all='red'
    col_s='blue'
    col_a='green'
    
    #
    f = plt.figure()
    a = plt.gca()
    dat_all.plot_2D_contour('x0','y0',fac=fmas,CL=levels,col=col_all,ax=a,linestyles=ls,nbins=nbin,withSave=False)
    dat_a.plot_2D_contour  ('x0','y0',fac=fmas,CL=levels,col=col_a,  ax=a,linestyles=ls,nbins=nbin,withSave=False)
    dat_s.plot_2D_contour  ('x0','y0',fac=fmas,CL=levels,col=col_s,  ax=a,linestyles=ls,nbins=nbin,withSave=False)
    proxy = [plt.Rectangle((0,0),1,1,fc = pc) for pc in [col_all, col_s, col_a]]
    plt.legend(proxy,[n_all,n_s,n_a])
    plt.ylabel('$y_0$ [mas]')
    plt.xlabel('$x_0$ [mas]')
    #plt.title('Align ')
    plt.plot([0,0],[-50,50],'gray',ls='--')
    plt.plot([-100,100],[0,0],'gray',ls='--')
    
    plt.xlim([-20,10])
    plt.ylim([-20, 10])
    
    
    plt.savefig('AO_speck_x0_y0.png')
    
    # get quantiles
    res_all = np.append(weighted_percentile(dat_all.BH.chain_glob['x0']*fmas,perc,dat_all.weights) , weighted_percentile(dat_all.BH.chain_glob['y0']*fmas,perc,dat_all.weights)) #x0, y0
    res_s = np.append(weighted_percentile(dat_s.BH.chain_glob['x0']*fmas,perc,dat_s.weights) , weighted_percentile(dat_s.BH.chain_glob['y0']*fmas,perc,dat_s.weights)) #x0, y0
    res_a = np.append(weighted_percentile(dat_a.BH.chain_glob['x0']*fmas,perc,dat_a.weights) , weighted_percentile(dat_a.BH.chain_glob['y0']*fmas,perc,dat_a.weights)) #x0, y0


    #2nd figure vx0 vs vy0
    plt.figure()
    a = plt.gca()
    dat_all.plot_2D_contour('vx0','vy0',fac=fmas,CL=levels,col=col_all,ax=a,linestyles=ls,nbins=nbin,withSave=False)
    dat_a.plot_2D_contour  ('vx0','vy0',fac=fmas,CL=levels,col=col_a,  ax=a,linestyles=ls,nbins=nbin,withSave=False)
    dat_s.plot_2D_contour  ('vx0','vy0',fac=fmas,CL=levels,col=col_s,  ax=a,linestyles=ls,nbins=nbin,withSave=False)
    
    proxy = [plt.Rectangle((0,0),1,1,fc = pc) for pc in [col_all, col_s, col_a]]
    plt.legend(proxy,[n_all,n_s,n_a])
    plt.ylabel('$vy_0$ [mas/yr]')
    plt.xlabel('$vx_0$ [mas/yr]')
    #plt.title('Align '+alignDir)
    plt.ylim([-2.,2.])
    plt.xlim([-2,2])


    plt.plot([0,0],[-10,10],'gray',ls='--')
    plt.plot([-10,10],[0,0],'gray',ls='--')
    plt.savefig('AO_speck_vx0_vy0.png')
    plt.close("all")

    res_all = np.append(res_all , weighted_percentile(dat_all.BH.chain_glob['vx0']*fmas,perc,dat_all.weights)) #vx0
    res_all = np.append(res_all , weighted_percentile(dat_all.BH.chain_glob['vy0']*fmas,perc,dat_all.weights)) #vy0
    
    res_s = np.append(res_s , weighted_percentile(dat_s.BH.chain_glob['vx0']*fmas,perc,dat_s.weights)) #vx0
    res_s = np.append(res_s , weighted_percentile(dat_s.BH.chain_glob['vy0']*fmas,perc,dat_s.weights)) #vy0
    
    res_a = np.append(res_a , weighted_percentile(dat_a.BH.chain_glob['vx0']*fmas,perc,dat_a.weights)) #vx0
    res_a = np.append(res_a , weighted_percentile(dat_a.BH.chain_glob['vy0']*fmas,perc,dat_a.weights)) #vy0
    
                      
    fout=open('res.txt','w')
    fout.write('        x0 1sig            x0 3 sig            y0 1 sig            y0 3 sig            ')
    fout.write(' vx0 1sig           vx0 3 sig           vy0 1 sig           vy0 3 sig\n')
    #print tuple(res_all.tolist())
    fout.write('All -- '+'%9.6f '*16 % tuple(res_all.tolist())+'\n')
    fout.write('Spe -- '+'%9.6f '*16 % tuple(res_s.tolist())+'\n')
    fout.write('AO  -- '+'%9.6f '*16 % tuple(res_a.tolist())+'\n')
    fout.close()
    os.chdir(cwd)
    
    #3rd figure GM vs R0
    fac=[1e-6,1e-3]
    plt.figure()
    a = plt.gca()
    dat_all.plot_2D_contour('GM','R0',fac=fac,CL=levels,col=col_all,ax=a,linestyles=ls,nbins=nbin,withSave=False)
    dat_a.plot_2D_contour  ('GM','R0',fac=fac,CL=levels,col=col_a,  ax=a,linestyles=ls,nbins=nbin,withSave=False)
    dat_s.plot_2D_contour  ('GM','R0',fac=fac,CL=levels,col=col_s,  ax=a,linestyles=ls,nbins=nbin,withSave=False)
    
    proxy = [plt.Rectangle((0,0),1,1,fc = pc) for pc in [col_all, col_s, col_a]]
    plt.legend(proxy,[n_all,n_s,n_a])
    plt.ylabel('$R_0$ [kpc]')
    plt.xlabel('$GM$ [$10^6$ Sun GM]')
    #plt.title('Align '+alignDir)
    plt.xlim([3,5.5])
    plt.ylim([6,10])
    
    plt.plot([0,0],[-10,10],'gray',ls='--')
    plt.plot([-10,10],[0,0],'gray',ls='--')
    plt.savefig('AO_speck_GM_R0.png')


    # figure with the 2D orbits and residuals
    mod=np.loadtxt('S0-2_all/chains/S0-2_mod.points')
    
    fig=plt.figure(figsize=(8,8))

    plt.errorbar(dat_all.stars[0].astro[:,1],dat_all.stars[0].astro[:,2],xerr=dat_all.stars[0].astro[:,3],yerr=dat_all.stars[0].astro[:,4],ls='None',lw=2,marker='o')

    plt.plot(mod[:,2],mod[:,5],lw=1,ls='-')
    plt.axis('equal')

    plt.xlabel('x [as]')
    plt.ylabel('y [as]')
    plt.savefig('S0-2_orbit.png')

    os.chdir(cwd)



