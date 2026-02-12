#!/opt/local/bin/python3
import sys
import os
from shutil import copy2, copyfile, rmtree
import subprocess as sb
from math import pi
import nstarorbits as NSO
from nstarorbits import utils
from nstarorbits import plot
from nstarorbits import stats
import yaml
import pandas as pd

'''
********************SUMMARY OF STEPS********************
For first iteration (if run_num == 1)
1. create subdirectory directory for each star
2. create subdirs within each star directory, e.g. data, config_files, etc. 
3. copy data files
    3a. copy .points file and .phot file from specified align directory
    3b. pull .rv file from findrv.py 
4. create config file for each star
    4a. generate efit config file based on the star's estimated period (if known) and whether or not it has rv information.
    4b. convert efit config file to NStarOrbits yaml file using efit5tonstarorbits
    4c. edit yaml file to include an astrometric systematic error parameter, set t_origin_BH, and define run_name so the fit saves the .nstar oject 
5. run nstarorbits and post processing script for each star
    5a. Fit orbit
    5b. Save fit products: chains saved as hdf5 file, cpu run time (in seconds) saved as txt file, orbit parameters and uncertainties saved as csv file, output plots
    5d. Identify astrometric or photometric outliers, and record which epochs to cut for future iterations
6. Generage an orbits.dat file with MAP orbit params for all stars in the starlist

For subsequent iterations (if run_num > 1):
1. create subdirectory directory for each star
2. create subdirs within each star directory, e.g. data, config_files, etc. 
3. copy files for each star
    3a. If any epochs were flagged as outliers in the previous iteration, copy over the config file and new data files with outlier epochs removed
    3b. If no epochs were flagged as outliers, just copy over the file with parameter estimates from the previous fit (so that this star is still included in the resulting orbits.dat file even though it is not fit in this iteration), and the empty file that says which epochs to remove (so that if yet another iteration is run later, this star does not get re-fit again)
4. run nstarorbits and post processing script for each star that has remaining outliers (saves same data products for each star as in the first iteration)
5. generate orbits.dat file with MAP orbit params for all stars in the starlist. For stars that were not re-fit, the values will be the same as the orbits.dat file generated in the previous iteration. For stars that were re-fit, values will be updated based on this round of fits.

Run this script after incrememnting run_num for as long as there reamins stars with remaining outliers. 

********************DEPENDENCIES********************
1. NStarOrbits
2. efit5tonstarorbits
3. python3

'''

#*****************DEFINE INPUT SETTINGS (USER INPUT)************************#
'''
USER INPUT:
- run_num: iteration number (int). Start with run_num = 1 for the first set of orbit fits, and continue incrementing until none of the stars in the sample have remaining outliers (i.e. until all resulting STAR_remove.txt files are empty).
- setup: Bool. True to create subdirs for each star, False to not create subdirs if directories are already set up.
- run: Bool. True to fit orbit and run post-processing script for each star, False to not fit the orbits.
- t_origin:  Set dynamical t0 at which BH params are specified. Example: t_origin = 2014.165 (weighted t0 for Sgr A* as of 2020 data)
- dist: R0 in pc (float). This value is only used for the orbit fit for stars that don't have RVs, or if BH params are fixed.
- mass: M_BH in solar masses (float). This value is only used for orbit fits if BH params are fixed.
- BHpriors: (Bool). Specify whether BH params will be fit or fixed. Only three options that work are 'fixed' if you are fixing the BH params for the orbit fits, 'fixed_MR' if you want to fix M and R0 but keep the reference frame parameters free,  or 'uniform' if you are leaving BH params free. In the latter case, it will still fix R0 and vz0 if there are no RVs  for a given star.
- sys_err: (bool). True if you want to include a systematic error parameter in the fit (as a fitted param), False if not. 
'''
run_num = 1
setup = True 
run = True    

t_origin =  {"t_origin_BH": 2014.165}

#S0-2 median values below (from fit with GR18corr and NIRC2 RV offset param with GR True, EM free - the canonical model for thesis work). 
dist=8190 
mass = 4200000

BHpriors = 'fixed'
sys_err = True

#*****************DEFINE STARLIST AND ALIGN DIRECTORIES (USER INPUT)************************#
# Specify list of stars to include
# starList: Array/list of 3-element arrays/lists, where the three elements for each star are: star_name (str), estimated period (float with estimated period in years if estimate is known, or 'min' if there is not a reasonable starting guess for that star's period), and whether or not there are RV data for that star (either 'RV' if RV measurements exist for that star, or "no" if not).
#The period does not need to be exact. It is just used to set a reasonable prior range on P and T0 for each star. 
#If no period estimate is known, use 'min' instead of specifying a value. This will calculate the prior range for P and T0 using the the minimum possible period, calculated from maximum r2d from the points file for that star. In this case, it is recommended to evaluate the posteriors after the first iteration, and adjust P and T0 ranges for subsequent iterationss accordingly. 
# Warning: these flags are case sensitive. 

starList=[
              ['S0-3', 80., 'no'],
              ['S0-4', 92.,'no'],
              ['S0-5', 55.,'no'],
              ['S0-8', 100., 'no'],
              ['S0-20', 50.,'no'],
              ['S0-16', 60.,'no'],
              ['S0-17', 70.,'no'],
              ['S0-19', 60.,'no'], 
              ['S0-28', 310. , 'no'],
              ['S0-38', 19.,'no'],
              ['S0-48', 100., 'no'],
              ['S0-49', 40.,'no'],
              ['S0-102', 15.,'no'],
              ['S0-1', 170.,'no'],
              ]
  
alignDir = '/u/shoko/align/2023_05_26/leg_acc/Dist/efit_mjd_err/'
photDir = '/u/shoko/align/2023_05_26/leg_acc/points_3_c/'
previous_dir = '/u/kkosmo/NStar_runs/S-stars/thesis/iterative_fits/2022data_legacyPSF/accelAlign_Dist/2023_05_26/kepler/BHfixed_Sys/run1/'

# all stars that have RV files in the gcg data release
rv_dr_list = ['S0-2', 'S0-1', 'S0-3', 'S0-5', 'S0-8', 'S0-16', 'S0-17', 'S0-19', 'S0-20', 'S0-38', 'S0-49']
rvDir = '/u/ghezgroup/data/rv_dr/pdr2/'

# stars that have points files already manually cleaned:
cleaned_stars = ['S0-8', 'S0-20', 'S0-38','S0-1']
cleaned_dir = '/u/kkosmo/NStar_runs/S-stars/thesis/final_thesis_fits/data/'

#********************FUNCTIONS*********************#
def createDir(n):
    if not os.path.isdir(n):
        os.makedirs(n)

def createSubdirs(subdirs):
    for dir in subdirs:
        createDir(dir)

def setup_dirs(starList, run_num = 1):
    if not os.path.isdir('run'+str(run_num)):
        createDir('run'+str(run_num))
        os.chdir('run'+str(run_num))
    else:
        print('WARNING: run'+str(run_num)+' directory already exists. Are you sure you want to over-ride it?')
        
    for s in starList:
        createDir(s[0])

        if run_num == 1:
            os.chdir(s[0]) #go into directory for each star (directory name is name of star)
            createSubdirs(subdirs=['data', 'config_files', 'plots', 'to_cut', 'fit_results'])

            if s[0] in cleaned_stars:
                copy2(cleaned_dir + s[0]+'.points','data/'+s[0]+'.points')
                copy2(cleaned_dir+s[0]+'.phot','data/'+s[0]+'.phot')
            else:
                #copy points and phot data for a given star from a given align
                copy2(previous_dir+s[0]+'/to_cut/'+s[0]+'_keep_points.txt','data/'+s[0]+'.points')
                copy2(previous_dir+s[0]+'/to_cut/'+s[0]+'_keep_phot.txt','data/'+s[0]+'.phot')

            #Get rv data for a given star
            #NOTE: for all stars that have RV files in the gcg data release, copy rv files from there. Otherwise, pull rv file from database using findrv.py
            if s[0] in rv_dr_list:
                copy2(rvDir+s[0]+'.rv','./data/'+s[0]+'.rv')
            else:
                os.chdir('data')
                os.system('python3 /u/kkosmo/python/database/code/findrv.py -p t36fCEtw --snr 20 --non_rel -o '+s[0])
                os.chdir('../')
            os.chdir('../')

        if run_num > 1: #for subsequent iterative fits
            #check if any outliers were flagged for removal in previous iteration. If yes, remove those epochs and refit. If no, just copy over the param_stats file so that the star is still included in the new orbits.dat file.
            previous_cut_file = open('../run'+str(run_num-1)+'/'+s[0]+'/to_cut/'+s[0]+'_remove.txt', 'r')
            rm_len = 0. 
            for l in previous_cut_file:
                if '#' not in l:
                    rm_len += 1

            os.chdir(s[0])
            if rm_len > 0.:
                createSubdirs(subdirs=['data', 'config_files', 'plots', 'to_cut', 'fit_results'])
                
                #copy data after outlier rejection from previous run
                copy2('../../run'+str(run_num-1)+'/'+s[0]+'/to_cut/'+s[0]+'_keep_points.txt','data/'+s[0]+'.points')
                copy2('../../run'+str(run_num-1)+'/'+s[0]+'/to_cut/'+s[0]+'_keep_phot.txt','data/'+s[0]+'.phot')
                copy2('../../run'+str(run_num-1)+'/'+s[0]+'/data/'+s[0]+'.rv', 'data/'+s[0]+'.rv')
    
                #copy config file
                copy2('../../run'+str(run_num-1)+'/'+s[0]+'/config_files/'+s[0]+'.yaml', 'config_files/'+s[0]+'.yaml')
            if rm_len == 0.:
                createSubdirs(subdirs=['fit_results', 'to_cut'])
                copy2('../../run'+str(run_num-1)+'/'+s[0]+'/fit_results/'+s[0]+'_param_stats.csv',  'fit_results/')
                copy2('../../run'+str(run_num-1)+'/'+s[0]+'/to_cut/'+s[0]+'_remove.txt', 'to_cut/')  #copy empty _remove.txt file so it knows to skip this star again in future iterations. 
            os.chdir('../') #go back to run_num directory
        
    os.chdir('../') #go back to root directory, i.e. where 'run1/' , 'run2/' etc live 
       
def createEfitFile_noRV(s,per, BHpriors = 'uniform'):
    # use efit units, will convert to NSO units when it converts to yaml file
    f = open('./config_files/orbit.'+s,'w')
    f.write('dataSource = individual("./data/"); \n')
    f.write('fitter = multinest; \n')
    f.write('efficiency = 0.3; \n')
    f.write('livePoints = 3000;\n')
    #f.write('chainDir="chains_'+s+'";\n\n')  #breaks efit5tonstarobits. Comment out to use this for nso. If you ultimately just want to run this with efit, would need to specify chains dir
    f.write('pix2arc = 1.0;\n')

    if BHpriors == 'center_free':
        f.write('mass =' +str(mass)+';\n') #input in efit is in solar masses. Converts to million solar masses for NStarOrbits config files. 
        f.write('distance =' +str(dist)+';\n\n') #input in efit is in pc. Converts to kpc for NStarOrbits config files.
        f.write('focusX= [-0.03,0.03];\n') #input in efit is in arcsec. Converts to mas for NStarOrbits config files.
        f.write('focusY= [-0.03,0.03];\n')
        f.write('focusX_velocity= [-0.03,0.03];\n')
        f.write('focusY_velocity= [-0.03,0.03];\n')
        f.write('focus_radial ='+str(vz0)+';\n') 
    if BHpriors == 'fixed_Rvz':
        f.write('mass = [1000000.,7500000.];\n') #input in efit is in solar masses. Converts to million solar masses for NStarOrbits config files. 
        f.write('distance =' +str(dist)+';\n\n') #input in efit is in pc. Converts to kpc for NStarOrbits config files.
        f.write('focusX= [-0.03,0.03];\n') #input in efit is in arcsec. Converts to mas for NStarOrbits config files.
        f.write('focusY= [-0.03,0.03];\n')
        f.write('focusX_velocity= [-0.03,0.03];\n')
        f.write('focusY_velocity= [-0.03,0.03];\n')
        f.write('focus_radial ='+str(vz0)+';\n') 
    if BHpriors == 'fixed_onlyMfree':
        f.write('mass = [1000000.,7500000.];\n') #input in efit is in solar masses. Converts to million solar masses for NStarOrbits config files. 
        f.write('distance =' +str(dist)+';\n\n') #input in efit is in pc. Converts to kpc for NStarOrbits config files.
        f.write('focusX='+str(x0)+';\n') #input in efit is in arcsec. Converts to mas for NStarOrbits config files.
        f.write('focusY='+str(y0)+';\n')
        f.write('focusX_velocity='+str(vx0)+';\n')
        f.write('focusY_velocity='+str(vy0)+';\n')
        f.write('focus_radial ='+str(vz0)+';\n')    
    if BHpriors == 'fixed_S02':
        f.write('mass =' +str(mass)+';\n') #input in efit is in solar masses. Converts to million solar masses for NStarOrbits config files. 
        f.write('distance =' +str(dist)+';\n\n') #input in efit is in pc. Converts to kpc for NStarOrbits config files.
        f.write('focusX='+str(x0)+';\n') #input in efit is in arcsec. Converts to mas for NStarOrbits config files.
        f.write('focusY='+str(y0)+';\n')
        f.write('focusX_velocity='+str(vx0)+';\n')
        f.write('focusY_velocity='+str(vy0)+';\n')
        f.write('focus_radial ='+str(vz0)+';\n')
    if BHpriors == 'uniform':
        f.write('mass = [1000000.,7500000.];\n') #input in efit is solar masses. Converts to million solar masses in NstarOrbits config files. 
        f.write('distance ='+str(dist)+';\n\n')#input in efit is in pc. Converts to kpc for NStarOrbits config files. 
        f.write('focusX= [-0.03,0.03];\n') #input in efit is in arcsec. Converts to mas for NStarOrbits config files. 
        f.write('focusY= [-0.03,0.03];\n')
        f.write('focusX_velocity= [-0.03,0.03];\n')
        f.write('focusY_velocity= [-0.03,0.03];\n')
        f.write('focus_radial =  0.;\n')

    if BHpriors == 'fixed':
        f.write('mass =' +str(mass)+';\n') #input in efit is in solar masses. Converts to million solar masses for NStarOrbits config files. 
        f.write('distance =' +str(dist)+';\n\n') #input in efit is in pc. Converts to kpc for NStarOrbits config files.
        f.write('focusX=0.  ;\n') #input in efit is in arcsec. Converts to mas for NStarOrbits config files.
        f.write('focusY= 0. ;\n')
        f.write('focusX_velocity= 0. ;\n')
        f.write('focusY_velocity= 0. ;\n')
        f.write('focus_radial = 0. ;\n')

    if BHpriors == 'fixed_MR':
        f.write('mass =' +str(mass)+';\n') #input in efit is in solar masses. Converts to million solar masses for NStarOrbits config files. 
        f.write('distance =' +str(dist)+';\n\n') #input in efit is in pc. Converts to kpc for NStarOrbits config files.
        f.write('focusX= [-0.03,0.03];\n') #input in efit is in arcsec. Converts to mas for NStarOrbits config files. 
        f.write('focusY= [-0.03,0.03];\n')
        f.write('focusX_velocity= [-0.03,0.03];\n')
        f.write('focusY_velocity= [-0.03,0.03];\n')
        f.write('focus_radial =  0.;\n')
    
    f.write('object "'+s+'" {\n')
    f.write('   radialVelocityData = "./data/'+s+'.rv";\n')   #need blank rv file for efit5tonstarorbits even if no rv data
    f.write('   eccentricity = [0.01 , 0.99];\n')
    f.write('   period = ['+ str(0.1*per)+ ', '+str(2*per)+'];\n')
    f.write('   periastronPassage = [1990.0 , '+str(1990.+1.5*per)+'];\n')
    f.write('   smallOmega = [-180., 180.];\n')
    f.write('   bigOmega = [-180., 180.];\n')
    f.write('   inclination = [0. , 180.];\n')
    f.write('}');
    f.close()

def createEfitFile_withRV(s,per, BHpriors = 'uniform'):
    # use efit units, will convert to NSO units when it converts to yaml file
    f = open('./config_files/orbit.'+s,'w')
    f.write('dataSource = individual("./data/"); \n')
    f.write('fitter = multinest; \n')
    f.write('efficiency = 0.3; \n')
    f.write('livePoints = 1000;\n')
    f.write('pix2arc = 1.0;\n')
    #f.write('chainDir="chains_'+s+'";\n\n')  #breaks efit5tonstarobits. Comment out to use this for nso. If you ultimately just want to run this with efit, would need to specify chains dir

    if BHpriors == 'uniform':
        f.write('mass = [1000000.,7500000.];\n') #input in efit is in solar masses. Converts to million solar masses for NStarOrbits config files. 
        f.write('distance = [5000.,11000.];\n\n') #input in efit is in pc. Converts to kpc for NStarOrbits config files.
        f.write('focusX= [-0.03,0.03];\n') #input in efit is in arcsec. Converts to mas for NStarOrbits config files.
        f.write('focusY= [-0.03,0.03];\n')
        f.write('focusX_velocity= [-0.03,0.03];\n')
        f.write('focusY_velocity= [-0.03,0.03];\n')
        f.write('focus_radial =  [-100.,100.];\n')

    if BHpriors == 'fixed':
        f.write('mass =' +str(mass)+';\n') #input in efit is in solar masses. Converts to million solar masses for NStarOrbits config files. 
        f.write('distance =' +str(dist)+';\n\n') #input in efit is in pc. Converts to kpc for NStarOrbits config files.
        f.write('focusX=0.  ;\n') #input in efit is in arcsec. Converts to mas for NStarOrbits config files.
        f.write('focusY= 0. ;\n')
        f.write('focusX_velocity= 0. ;\n')
        f.write('focusY_velocity= 0. ;\n')
        f.write('focus_radial = 0. ;\n')

    if BHpriors == 'fixed_MR':
        f.write('mass =' +str(mass)+';\n') #input in efit is in solar masses. Converts to million solar masses for NStarOrbits config files. 
        f.write('distance =' +str(dist)+';\n\n') #input in efit is in pc. Converts to kpc for NStarOrbits config files.
        f.write('focusX= [-0.03,0.03];\n') #input in efit is in arcsec. Converts to mas for NStarOrbits config files. 
        f.write('focusY= [-0.03,0.03];\n')
        f.write('focusX_velocity= [-0.03,0.03];\n')
        f.write('focusY_velocity= [-0.03,0.03];\n')
        f.write('focus_radial =  [-100.,100.];\n')

    if BHpriors == 'fixed_S02':
        f.write('mass =' +str(mass)+';\n') #input in efit is in solar masses. Converts to million solar masses for NStarOrbits config files. 
        f.write('distance =' +str(dist)+';\n\n') #input in efit is in pc. Converts to kpc for NStarOrbits config files.
        f.write('focusX='+str(x0)+';\n') #input in efit is in arcsec. Converts to mas for NStarOrbits config files.
        f.write('focusY='+str(y0)+';\n')
        f.write('focusX_velocity='+str(vx0)+';\n')
        f.write('focusY_velocity='+str(vy0)+';\n')
        f.write('focus_radial ='+str(vz0)+';\n')

    if BHpriors == 'fixed_onlyMfree':
        f.write('mass = [1000000.,7500000.];\n') #input in efit is in solar masses. Converts to million solar masses for NStarOrbits config files. 
        f.write('distance =' +str(dist)+';\n\n') #input in efit is in pc. Converts to kpc for NStarOrbits config files.
        f.write('focusX='+str(x0)+';\n') #input in efit is in arcsec. Converts to mas for NStarOrbits config files.
        f.write('focusY='+str(y0)+';\n')
        f.write('focusX_velocity='+str(vx0)+';\n')
        f.write('focusY_velocity='+str(vy0)+';\n')
        f.write('focus_radial ='+str(vz0)+';\n')

    if BHpriors == 'fixed_Rvz':
        f.write('mass = [1000000.,7500000.];\n') #input in efit is in solar masses. Converts to million solar masses for NStarOrbits config files. 
        f.write('distance =' +str(dist)+';\n\n') #input in efit is in pc. Converts to kpc for NStarOrbits config files.
        f.write('focusX= [-0.03,0.03];\n') #input in efit is in arcsec. Converts to mas for NStarOrbits config files.
        f.write('focusY= [-0.03,0.03];\n')
        f.write('focusX_velocity= [-0.03,0.03];\n')
        f.write('focusY_velocity= [-0.03,0.03];\n')
        f.write('focus_radial ='+str(vz0)+';\n')
        
    if BHpriors == 'center_free':
        f.write('mass =' +str(mass)+';\n') #input in efit is in solar masses. Converts to million solar masses for NStarOrbits config files. 
        f.write('distance =' +str(dist)+';\n\n') #input in efit is in pc. Converts to kpc for NStarOrbits config files.
        f.write('focusX= [-0.03,0.03];\n') #input in efit is in arcsec. Converts to mas for NStarOrbits config files.
        f.write('focusY= [-0.03,0.03];\n')
        f.write('focusX_velocity= [-0.03,0.03];\n')
        f.write('focusY_velocity= [-0.03,0.03];\n')
        f.write('focus_radial ='+str(vz0)+';\n') 
        
    #KEY ERROR WHEN TRYING TO USE GAUSSIAN PRIORS. Either incorrect syntax for efit, or efit5tonstarorbits not set up for this. 
    #if BHpriors == 'gaussian':
     #   f.write('useMPrior; \n')
      #  f.write('useR0Prior; \n')
       # f.write('meanMPrior = 4000000.; \n')
        #f.write('sigMPrior = 20000.; \n')
        #f.write('meanR0Prior = 8000.; \n')
        #f.write('sigR0Prior = 500.; \n')

    f.write('object "'+s+'" {\n')
    f.write('   radialVelocityData = "./data/'+s+'.rv";\n')  
    f.write('   eccentricity = [0.01 , 0.99];\n')
    f.write('   period = ['+ str(0.1*per)+ ', '+str(2*per)+'];\n')
    f.write('   periastronPassage = [1990.0 , '+str(1990.+1.5*per)+'];\n')
    f.write('   smallOmega = [-180., 180.];\n')
    f.write('   bigOmega = [-180., 180.];\n')
    f.write('   inclination = [0. , 180.];\n')
    f.write('}');
    f.close()
    
def editYamlFile(s, t_origin, sys_err = True):
    '''
    add some keywords to the yaml file before running NSO

    INPUT: 
    - t_origin: float, epoch that you want to use as dynamical t0 (date at which reference frame params are calculated.  
    - sys_err: bool, true if you want to include an additive systematic error parameter in the fit. False if not. 
    '''
    fname = './config_files/orbit.'+s[0]+'.yaml'
    f = open(fname)
    d = yaml.load(f, Loader=yaml.FullLoader)
    
    #add ability to save the fit
    d['sampler']['run_name']= './fit_results/'+s[0]+'_saved_fit'

    #adjust dynamical T0
    d['evaluators']['eval1'][s[0]]['kepler'].append(t_origin)   #adding dictionary to a list within a dictionary to set t_origin_BH

    if sys_err == True: 
        #include systematic additive error in the fit
        d['evaluators']['eval1'][s[0]+'_sys']={'err.sys':[s[0]+'_sys',s[0]+'::position']}

        d['priors']['sys_err_prior'] = {'parameters': {'lower': 0., 'upper': 5}, 'prior_type': 'uniform'}
        d['parameters'][s[0]+'_sys'] = {'err': 'sys_err_prior'}
        
    if s[2] != 'RV':
        d['parameters'][s[0]]['R0'] = dist/1000.
        d['parameters'][s[0]]['vz0'] =0.
    
    out_yaml = open('./config_files/'+s[0]+'.yaml', 'w')
    yaml.dump(d, out_yaml)

def create_configs(starList, BHpriors = 'uniform', sys_err = True):
    # create efit config file
    for s in starList:
        os.chdir(s[0])
    
        if s[1] == 'min':
            period = 2.* utils.get_min_period_kepler('./data/'+s[0]+'.points')
            #this is a rough approximation. If period estimate is not known initially, find the minimum period based on r2d, then multiply it by 2 so that the period prior range is from [min_period, 4xmin_period] (from the createEfitFile functions)
        else:
            period = s[1]
        
        if s[2] == 'RV':
            createEfitFile_withRV(s[0],period, BHpriors = BHpriors)
    
        else:
            createEfitFile_noRV(s[0],period, BHpriors = BHpriors)

        # turn efit config file into nso yaml file
        os.system('python3 /u/kkosmo/python/efit5tonstarorbits/efit5toNStarOrbits.py ./config_files/orbit.'+s[0])

        #edit yaml file with sys error param, etc. 
        editYamlFile(s, t_origin, sys_err)

        os.chdir('../')

        
def writeOrbitsDatOutput(s, outfile, statistic = 'MAP', Mass = 4.e6, R0 = 8000.):
    '''
    run this function for all stars after creating a blank output file (here called outfile), then close the output file at the end

    statistic: string, either 'MAP' or 'Med' depending on whether you want to use MAP or median parameter values

    Mass, R0: either str ('free') if either Mass or R0 was left as a free parameter, or float, fixed to whatever value Mass and R0 are for each fit. 
    '''
    #write output:
    csvfile = './fit_results/'+s[0]+'_param_stats.csv' #file with MAP params (maximum a posteriori), median, and 1-sig (68% confidence on central confidence interval) parameter estimates, created in runNSO.py stage
    tab = pd.read_csv(csvfile, index_col = 0)  #index_col = 0 means it will use the 0th column (param names) as the row labels

    #name_header = 'eval[0].'+s[0]+'.'  #if the csv file does not have cleaned param names, (for example, if the csv file gives param names in the format eval[0].S0-8.P), then need to use loc[name_header+'P', etc. 

    P = tab.loc['P'][statistic]
    T0 = tab.loc['T0'][statistic]
    e = tab.loc['e'][statistic]
    I = tab.loc['I'][statistic]
    O = tab.loc['Omega'][statistic]
    w = tab.loc['omega'][statistic]
    
    if Mass == 'free': #if BH mass is left as a free param, get  value from the fit and convert to solar masses from million solar masses. Otherwise, use inputted value for Mass in solar masses.
        Mass = tab.loc['Mass'][statistic] * 1.e6
    if R0 == 'free': #if R0 is left as a free param, get value from the fit and convert to pc from kpc. Otherwise, use inputted value for R0 in pc.
        R0 = tab.loc['R0'][statistic] * 1.e3
        
    sma_AU = (Mass * P**2)**(1./3.)   #calculate semi-major axis in AU from period and mass in yrs and solar masses
    sma_mas = 1000.*sma_AU / R0   #convert sma to units of mas using R0 in pc
    
    outfile.write(s[0]+' '*(9-len(s[0]))+ '   %6.3f   %5.3f   %8.3f   %6.4f   %5.1f   %5.1f   %5.1f   2 \n'%(P, sma_mas, T0, e, I, O, w))



def run_stars(starList, run_num):
    outfile= open('_orbits.dat','w')
    
    if run_num == 1:
        for s in starList:
            os.chdir(s[0])
            os.system('python3 ../../_run.py {0} {1} {2}'.format(s[0], s[2], BHpriors))
            writeOrbitsDatOutput(s, outfile, statistic = 'MAP')
            os.chdir('../')

    if run_num > 1: #for subsequent iterative fits
        for s in starList:
            #check if any outliers were flagged for removal in previous iteration. If yes, refit. If no, just move to next star.
            previous_cut_file = open('../run'+str(run_num-1)+'/'+s[0]+'/to_cut/'+s[0]+'_remove.txt', 'r')
            rm_len = 0. 
            for l in previous_cut_file:
                if '#' not in l:
                    rm_len += 1

            os.chdir(s[0])
            if rm_len == 0.:
                writeOrbitsDatOutput(s, outfile, statistic = 'MAP')
                
            if rm_len > 0.:
                os.system('python3 ../../_run.py {0} {1} {2}'.format(s[0], s[2], BHpriors))
                writeOrbitsDatOutput(s, outfile, statistic = 'MAP')
            os.chdir('../')

    outfile.close()


#********************RUN FIRST TIME*********************#

if setup == True:
    setup_dirs(starList, run_num = run_num)
    if run_num == 1:
        os.chdir('run'+str(run_num)+'/')
        create_configs(starList, BHpriors = BHpriors, sys_err = sys_err)
        os.chdir('../')

if run == True:
    os.chdir('run'+str(run_num)+'/')
    #os.system('pwd')
    run_stars(starList, run_num)
