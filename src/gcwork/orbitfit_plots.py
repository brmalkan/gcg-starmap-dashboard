import numpy as np
#from gcwork.efit import efit5_results
import matplotlib
import pylab as plt
from corner import corner
import scipy
from astropy.time import Time
import os
from scipy.stats import chi2
from collections import OrderedDict
import pandas as pd
from gcwork import Chains
from gcwork.efit import efit5_results


# adapted from efit5_results. These functions are for plotting the
# chains from the fitter used for GR2018 (the fortran code from
# Aurelien)

def plot_rv_residuals(star,path='./',ylim=None,xlim=None,rv_file=None,
    pdf_file=None,figsize=(11,8.5),fitter=None,resfile=None,modfile=None):
    '''
    Plot the RV fit residuals

    fitter - set to be 'efit5' if using residuals from that. Defaults to using fortran fitter (default: None)
    '''

    if resfile is None:
        resfile = os.path.join(path,star+'_res.rv')
    if modfile is None:
        modfile = os.path.join(path,star+'_mod.rv')

    res = np.loadtxt(resfile)
    mod = np.loadtxt(modfile)
    
    plt.clf()
    plt.figure(figsize=figsize)
    plt.subplot(2,1,1)

    if rv_file  is not None:
        rv = pd.read_csv(rv_file,header=None,comment='#',names=['date','vz','vz_err'],delim_whitespace=True,usecols=[0,1,2])
        plt.errorbar(rv['date'], rv['vz'],yerr=rv['vz_err'],fmt='o',capsize=4.0,markerfacecolor='white',
                 markersize=10,markeredgecolor='k',ecolor='k')


    #rv_time = np.linspace(np.min(mod[:,0]),np.max(mod[:,0]),10000)
    #rv_model = s1.get_RV_opt(rv_time)
    plt.plot(mod[:,0],mod[:,3],color='grey')
    #plt.plot(mod[:,0],mod[:,1],color='grey')
    plt.title('RV for '+star)
    plt.ylabel('RV [km/s]')
    if xlim is not None:
        plt.xlim(*xlim)
    else:
        plt.xlim([np.min(mod[:,0]),np.max(mod[:,0])])

    plt.xlabel('Time [yr]')
    
    plt.subplot(2,1,2)

   
    plt.errorbar(res[:,0], res[:,1],yerr=res[:,2],fmt='o',capsize=4.0,markerfacecolor='white',
                 markersize=10,markeredgecolor='k',ecolor='k')
    chi2 = np.sum((res[:,1]/res[:,2])**2 )
    n = len(res[:,1])

    #plt.plot(mod[:,0],(mod[:,2]-mod[:,1]),'g--')
    #plt.plot(mod[:,0],(mod[:,3]-mod[:,1]),'g--')
    if fitter == 'efit5':
            plt.fill_between(mod[:,0],(mod[:,3]-mod[:,1]),(mod[:,2]-mod[:,1]),alpha=0.5,color='grey')
    else:
        plt.fill_between(mod[:,0],mod[:,4]-mod[:,3],mod[:,2]-mod[:,3],alpha=0.5,color='grey')

    plt.title('RV for '+star)
    plt.ylabel('$\Delta$ RV [km/s]')
    if xlim is not None:
        plt.xlim(*xlim)
    else:
        plt.xlim([np.min(mod[:,0]),np.max(mod[:,0])])

    plt.xlabel('Time [yr]')
    if ylim is not None:
        plt.ylim(*ylim)
    if rv_file is not None:
        plt.title('RV for '+star+' '+ 'n='+str(n)+' chi2=%.2f  chi2/n=%.2f' %(chi2,chi2/(n*1.)))
    else:
        plt.title('RV for '+star)
    
    if pdf_file is not None:
        plt.savefig(pdf_file)

def plot_astro_residuals(star,print_tab=False,xlim=None,path='./',
    astro_file=None,pdf_file=None,figsize=(10,5),fitter=None,resfile=None,modfile=None):
    '''
    Plot the best-fit astrometry and the residuals in x and y
    '''
    if resfile is None:
        resfile = os.path.join(path,star+'_res.points')
    if modfile is None:
        modfile = os.path.join(path,star+'_mod.points')

    if fitter == 'efit5':
        mod = pd.read_csv(modfile,header=None,delim_whitespace=True,
                      names=['date','x0.16','x0.5','x0.84',
                             'y0.16','y0.5','y0.84'],comment='#')
        res = pd.read_csv(resfile,header=None,delim_whitespace=True,
                          names=['date','x','y','xerr','yerr','xelpd','yelpd'],comment='#')

    else:
        mod = pd.read_csv(modfile,header=None,delim_whitespace=True,
                        names=['date','x0.025','x0.16','x0.5','x0.84','x0.975',
                                'y0.025','y0.16','y0.5','y0.84','y0.975'],comment='#')
        res = pd.read_csv(resfile,header=None,delim_whitespace=True,
                        names=['date','x','y','xerr','yerr','xelpd','yelpd'],comment='#')
    astro = pd.read_csv(astro_file,header=None,delim_whitespace=True,usecols=[0,1,2,3,4],
                        names = ['date','x','y','xerr','yerr'],comment='#')
    #res = np.loadtxt(os.path.join(path,star+'_res.points'))
    #mod = np.loadtxt(os.path.join(path,star+'_mod.points'))
        

    # plot the x residuals
    plt.clf()
    plt.figure(figsize=figsize)

    plt.subplot(2,2,1)
    #plt.errorbar(res[:,0], res[:,1]*1000.,yerr=res[:,3]*1000.,fmt='ko',capsize=3)
    plt.errorbar(res['date'], res['x']*1000.,yerr=res['xerr']*1000.,fmt='ko',capsize=3)
    
    #plt.plot(mod[:,0],(mod[:,2]-mod[:,1])*1000.,'g--')
    #plt.plot(mod[:,0],(mod[:,3]-mod[:,1])*1000.,'g--')
    #plt.fill_between(mod[:,0],(mod[:,4]-mod[:,3])*1000.,(mod[:,2]-mod[:,3])*1000.,facecolor='grey',alpha=0.5)
    plt.fill_between(mod['date'],(mod['x0.16']-mod['x0.5'])*1000.,(mod['x0.84']-mod['x0.5'])*1000.,facecolor='grey',alpha=0.5)


    plt.ylabel('$\Delta$ x [mas]')
    plt.xlabel('Time [yr]')
    if xlim is None:
        plt.xlim([np.min(mod['date']),np.max(mod['date'])])
    else: 
        plt.xlim(*xlim)
        
    chi2 = np.sum((res['x']/res['xerr'])**2 +(res['y']/res['yerr'])**2)
    n = len(res)
    plt.title('Astro for '+star+' '+ 'n='+str(n)+' chi2=%.2f  chi2/2n=%.2f' %(chi2,chi2*0.5/(n*1.) ),fontsize=12)

    # plot y residuals
    plt.subplot(2,2,2)
    plt.errorbar(res['date'], res['y']*1000.,yerr=res['yerr']*1000.,fmt='ko',capsize=3)
    #plt.plot(mod[:,0],(mod[:,5]-mod[:,4])*1000.,'g--')
    #plt.plot(mod[:,0],(mod[:,6]-mod[:,4])*1000.,'g--')
    plt.fill_between(mod['date'],(mod['y0.16']-mod['y0.5'])*1000.,(mod['y0.84']-mod['y0.5'])*1000.,facecolor='grey',alpha=0.5)
    plt.ylabel('$\Delta$ y [mas]')
    plt.xlabel('Time [yr]')
    if xlim is None:
        plt.xlim([np.min(mod['date']),np.max(mod['date'])])
    else: 
        plt.xlim(*xlim)

    #plt.show()

    # X measurements + model
    plt.subplot(2,2,3)
    plt.plot(mod['date'],mod['x0.5']*1000.,'grey')
    plt.errorbar(astro['date'], astro['x']*1000.,yerr=astro['xerr']*1000.,fmt='ko')
    #ax1.plot(mod[:,0],(mod[:,2])*1000.,'g--')
    #ax1.plot(mod[:,0],(mod[:,3])*1000.,'g--')
    #ax1.fill_between(mod[:,0],(mod[:,3])*1000.,(mod[:,2])*1000.,facecolor='green',alpha=0.5)
    #plt.title('Astro for '+star.name)
    plt.ylabel('x [mas]')
    plt.xlabel('Time [yr]')
    if xlim is None:
        plt.xlim([np.min(mod['date']),np.max(mod['date'])])
    else: 
        plt.xlim(*xlim)
        
    plt.subplot(2,2,4)
    plt.plot(mod['date'],mod['y0.5']*1000.,'grey')
    plt.errorbar(astro['date'], astro['y']*1000.,yerr=astro['yerr']*1000.,fmt='ko')
    plt.ylabel('y [mas]')
    plt.xlabel('Time [yr]')
    if xlim is None:
        plt.xlim([np.min(mod['date']),np.max(mod['date'])])
    else: 
        plt.xlim(*xlim)
    if pdf_file is not None:
        plt.savefig(pdf_file)

def plot_orbit(star,xlim=None,path='./',astro_file=None,ylim=None,pdf_file=None,figsize=(8,8),fitter=None,
               resfile=None,modfile=None):
    '''
    plot the 2D orbit of the model and data
    '''
    if resfile is None:
        resfile = os.path.join(path,star+'_res.points')
    if modfile is None:
        modfile = os.path.join(path,star+'_mod.points')

    if fitter == 'efit5':
        mod = pd.read_csv(modfile,header=None,delim_whitespace=True,
                      names=['date','x0.16','x0.5','x0.84',
                             'y0.16','y0.5','y0.84'],comment='#')
        res = pd.read_csv(resfile,header=None,delim_whitespace=True,
                          names=['date','x','y','xerr','yerr','xelpd','yelpd'],comment='#')

    else:
        mod = pd.read_csv(modfile,header=None,delim_whitespace=True,
                        names=['date','x0.025','x0.16','x0.5','x0.84','x0.975',
                                'y0.025','y0.16','y0.5','y0.84','y0.975'],comment='#')
        res = pd.read_csv(resfile,header=None,delim_whitespace=True,
                        names=['date','x','y','xerr','yerr','xelpd','yelpd'],comment='#')


    # assume the points file is in the same default location if not given
    if astro_file is None:
        astro_file = os.path.join(path,star+'.points')
        
    astro = pd.read_csv(astro_file,header=None,delim_whitespace=True,usecols=[0,1,2,3,4],
                        names = ['date','x','y','xerr','yerr'],comment='#')


    plt.clf()
    plt.figure(figsize=figsize)
    plt.scatter(astro['x'],astro['y'],c=astro['date'],cmap='tab20c',
                s=120,alpha=1,edgecolor='k')
    plt.colorbar(label='Time',orientation='horizontal',shrink=0.5)

    #plt.errorbar(astro['x'],astro['y'],fmt='ko',xerr=astro['xerr'],yerr=astro['yerr'])
    plt.plot(mod['x0.5'],mod['y0.5'],color='grey',alpha=0.7)
    plt.axis('equal')
    plt.xlabel('$\Delta RA$')
    plt.ylabel('$\Delta Dec$')
    plt.tight_layout()
    if pdf_file is not None:
        plt.savefig(pdf_file)    
    
    
def load_chains(chain_file,chain_type='orbitfit',fitmodel='GR',starname='S-star_name',save_summary=False,save_dir='./'):
    '''
    Loads Multinest chains from orbit fitters

    chain_type - type of fitter that was used (default: orbitfit)
               - if using 'efit5' as chain_type, then use the orbit config file as the input

    fitmodel - type of orbit fit run (options are 'GR' (General Relativity), 'Newton','Newton_Redshift' (Newton + Redshift))
    
    save_summary - save the summary of the chains. Default: False

    '''
    if os.path.exists(chain_file):
        names = []
        if chain_type == 'efit5':
            k = efit5_results.efit5(chain_file)
            tab = {'GM':k.BH.chain_glob['GM']/1e6,
                   'R0':k.BH.chain_glob['R0']/1e3,
                   'x0':k.BH.chain_glob['x0']*1e3,
                   'y0':k.BH.chain_glob['y0']*1e3,
                   'vx0':k.BH.chain_glob['vx0']*1e3,
                   'vy0':k.BH.chain_glob['vy0']*1e3,
                   'vz0':k.BH.chain_glob['vz0'],
                   'weights':k.weights}
            if 'Redshift' in k.BH.chain_glob.keys():
                tab['redshift']=k.BH.chain_glob['Redshift']
            
            star = k.stars[0].chain

            tab['P'] = star['P']
            tab['inc'] = star['i']
            tab['T0'] = star['T0']
            tab['e'] = star['e']
            tab['w'] = star['w']
            tab['O'] = star['O']

            tab = pd.DataFrame(tab)
        else:   
            if fitmodel=='GR':
                names = ['weights','likelihood','GM','R0','redshift','GR','EM','x0','y0','vx0','vy0','vz0',\
                        'P','T0','e','inc','w','O','xAO','yAO','rho','lambda','deltaNIRSPEC','sigNIRSPEC',\
                        'deltaNIRC2','sigNIRC2','deltaKbb','sigKbb','deltaKn3','sigKn3','deltaVLT','sigVLT',\
                        'deltaNIFS','sigNIFS','deltaSUBARU','sigSUBARU']
            if fitmodel=='EM':
                names = ['weights','likelihood','GM','R0','redshift','GR','EM','x0','y0','vx0','vy0','vz0',\
                        'P','T0','e','inc','w','O','xAO','yAO','rho','lambda','deltaNIRSPEC','sigNIRSPEC',\
                        'deltaNIRC2','sigNIRC2','deltaKbb','sigKbb','deltaKn3','sigKn3','deltaVLT','sigVLT',\
                        'deltaNIFS','sigNIFS','deltaSUBARU','sigSUBARU']
            if fitmodel=='Newton':
                names = ['weights','likelihood','GM','R0','x0','y0','vx0','vy0','vz0',\
                        'P','T0','e','inc','w','O','xAO','yAO','rho','lambda','deltaNIRSPEC','sigNIRSPEC',\
                        'deltaNIRC2','sigNIRC2','deltaKbb','sigKbb','deltaKn3','sigKn3','deltaVLT','sigVLT',\
                        'deltaNIFS','sigNIFS','deltaSUBARU','sigSUBARU']
            if fitmodel=='Newton_Redshift':
                names = ['weights','likelihood','GM','R0','redshift','x0','y0','vx0','vy0','vz0',\
                'P','T0','e','inc','w','O','xAO','yAO','rho','lambda','deltaNIRSPEC','sigNIRSPEC',\
                'deltaNIRC2','sigNIRC2','deltaKbb','sigKbb','deltaKn3','sigKn3','deltaVLT','sigVLT',\
                'deltaNIFS','sigNIFS','deltaSUBARU','sigSUBARU']
            tab = pd.read_csv(chain_file,header=None,delim_whitespace=True,names=names,comment='#')
            tab = tab.drop(['likelihood'],axis=1)

        '''
        Write out file to a txt file
        '''
        c = Chains.Chains(tab)
        if save_summary:
            outfile = open(os.path.join(save_dir,starname+'_chains_summary.txt'),'w')
            outfile.write(str(c))
            outfile.close()
        return c
    else:
        print("file does not exist: "+chain_file)

def plot_joint_pdf(chains_obj,plot_dir='./',prefix='',plot_em=False,sigma=4):
    '''
    Plot the joint posteriors of parameters in the chains. 
    This is aimed at looking at correlations in the redshift

    Inputs
    ------
    chains_obj - chains object from using load_chains()

    Output
    ------
    Plots of the joint-posteriors of the dataset

    Keywords
    --------
    plot_dir - the directory where to place the plots (Default: './')
    prefix - the prefix to the file names to plut (Default: '')
    '''

    plt.clf()
    plt.figure(figsize=(8.5,8.5))
    # first plot the black hole related parameters
    if 'GR' in chains_obj.parameter_names:
        if plot_em:
            chains_obj.plot_triangle(parameters=['GM','R0','EM','redshift','x0','y0','vx0','vy0','vz0'],sigma=sigma)
        else:
            chains_obj.plot_triangle(parameters=['GM','R0','GR','redshift','x0','y0','vx0','vy0','vz0'], sigma=sigma)
    
    elif 'redshift' in chains_obj.parameter_names:
        chains_obj.plot_triangle(parameters=['GM','R0','redshift','x0','y0','vx0','vy0','vz0'],
                                sigma=sigma)
    else:
        chains_obj.plot_triangle(parameters=['GM','R0','x0','y0','vx0','vy0','vz0'],
                                sigma=sigma)

    outfile1 = os.path.join(plot_dir,prefix+'_bh_params.pdf')
    plt.savefig(outfile1)

    plt.clf()
    plt.figure(figsize=(8.5,8.5))
    # plot stellar orbital elements
    chains_obj.plot_triangle(parameters=['P','T0','e','inc','w','O'],sigma=sigma)
    outfile2 = os.path.join(plot_dir,prefix+'_star_params.pdf')
    plt.savefig(outfile2)
    

    if 'redshift' in chains_obj.parameter_names:
        plt.clf()
        plt.figure(figsize=(8.5,8.5))
        # plot parameters correlated with redshift
        chains_obj.plot_triangle(parameters=['redshift','GM','R0','vx0','vy0','vz0','T0','e','w','O'],
                                sigma=sigma)
        outfile2 = os.path.join(plot_dir,prefix+'_redshift_corr.pdf')
        plt.savefig(outfile2)

    if 'GR' in chains_obj.parameter_names:
        plt.clf()
        plt.figure(figsize=(8.5,8.5))
        # plot parameters correlated with redshift
        if plot_em:
            chains_obj.plot_triangle(parameters=['GR','redshift','GM','R0','EM','vx0','vy0','vz0','T0','e','w','O'],sigma=sigma)
        else:
            chains_obj.plot_triangle(parameters=['GR','redshift','GM','R0','vx0','vy0','vz0','T0','e','w','O'], sigma=sigma)
        outfile2 = os.path.join(plot_dir,prefix+'_GR_corr.pdf')
        plt.savefig(outfile2)
        
    # # plot a corner plot of all parameters
    # plt.clf()
    # plt.figure(figsize=(8.5,8.5))
    # outfile3 = os.path.join(plot_dir,prefix+'_all_params.pdf')
    # chains_obj.plot_corner()

    
    
