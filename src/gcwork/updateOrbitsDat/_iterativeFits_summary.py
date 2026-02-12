import phase_coverage as pc
from nstarorbits import stats
import numpy as np
import pandas as pd
import math
import csv
import sys
import os

##NOTE: NEED TO MANUALLY CHECK FOR STARS WHERE PHASE COVERAGE IS GREATER THAN 100!



#User input: copy over starlist from the fit directory, and define total_runs and other keywords
#to do: take in starList and other keywords as sys args so user does not have to define

fit_root = './'
alignDir = '/u/shoko/align/2023_05_26/leg_acc/'


total_runs = 1
sys_err = True  #True if sys error param was fitted and you want to include it's value in the output file
use_resids = True #True if loo_resids_...txt files exist and you want to include rmse of residuals in output file
use_mag = False #True if you want to include info on magnitude. Must be manually input as 4th column in starList if so.


starList=[
              ['S0-1', 170.,'RV'],
              ['S0-3', 80., 'RV'],
              ['S0-4', 92.,'RV'],
              ['S0-5', 55.,'RV'],
              ['S0-8', 100., 'RV'],
              ['S0-20', 50.,'RV'],
              ['S0-16', 60.,'RV'],
              ['S0-17', 70.,'RV'],
              ['S0-19', 60.,'RV'], 
              ['S0-28', 310. , 'RV'],
              ['S0-38', 19.,'RV'],
              ['S0-48', 100., 'no'],
              ['S0-49', 40.,'RV'],
              ['S0-102', 15.,'no'],

              ]


confuseCorrected_dir = alignDir + 'points_3_c/'
initialMatched_dir = alignDir + 'points_1_1000/'

#get total number of reference epochs for the given align
num_ref_epochs_tot = len(np.loadtxt(alignDir+'align/align_d_rms.trans')[:,0])



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


def get_runNum_from_iterativeFits(starList, total_runs, fit_root = './'):
    star_info = {}
    for s in starList:
        #get run number of "final" fit for each star
        star_info[s[0]] = {}
        star_info[s[0]]['run_num']=0 #set all to zero initially
        
        cut_file = open(fit_root+'run1/'+s[0]+'/to_cut/'+s[0]+'_remove.txt', 'r')
        rm_len = 0 
        for l in cut_file:
            if '#' not in l:
                rm_len += 1
        if rm_len == 0:
            star_info[s[0]]['run_num']=1

    if total_runs >1:
        for i in range(1, total_runs+1):
            for s in starList:
                if star_info[s[0]]['run_num']==0:
                    #print(s[0])
                    cut_file = open(fit_root+'run'+str(i)+'/'+s[0]+'/to_cut/'+s[0]+'_remove.txt', 'r')
                    rm_len = 0 
                    for l in cut_file:
                        if '#' not in l:
                            rm_len += 1
                    if rm_len == 0:
                        star_info[s[0]]['run_num']=i
                #print(s[0], star_info[s[0]]['run_num'])
            i += 1
    
    for s in starList:
        if star_info[s[0]]['run_num']==0:  #if there remains outliers for a star but no more iterations were run, just use the last one
            star_info[s[0]]['run_num']=total_runs
    
    return star_info



def get_phaseCoverage_from_iterativeFits(starList, total_runs, fit_root = './', sys_err = False, use_resids = False, use_mag = False):
    star_info = get_runNum_from_iterativeFits(starList, total_runs, fit_root)   
    params = ["P","T0",'e']
    quantiles = [.5, .16,.84]
    
    for s in starList:   
        #get points file from final iterative fit for each star
        points_file = fit_root + 'run'+str(star_info[s[0]]['run_num'])+'/'+s[0]+'/data/'+s[0]+'.points'        
        star_info[s[0]]['ast_start'] = np.loadtxt(points_file)[:,0][0]
        star_info[s[0]]['ast_end'] = np.loadtxt(points_file)[:,0][-1]
        #get length of points file
        star_info[s[0]]['num_astro_epochs_used'] = len(np.loadtxt(points_file)[:,0])
        star_info[s[0]]['percentPossible_astro_epochs_used'] = 100.*len(np.loadtxt(points_file)[:,0])/num_ref_epochs_tot

        #get number of epochs before outliers or confused epochs were cut
        #returns percentage of the total possible number of reference epochs for that align
        star_info[s[0]]['percentPossible_confuseCorrected_astro_epochs'] = 100.*len(np.loadtxt(confuseCorrected_dir+s[0]+'.points')[:,0])/num_ref_epochs_tot
        star_info[s[0]]['percentPossible_initialMatched_astro_epochs'] = 100.*len(np.loadtxt(initialMatched_dir+s[0]+'.points')[:,0])/num_ref_epochs_tot
        


        
        #get P, e, T0 from chains
        ch = pd.read_hdf(fit_root + 'run'+str(star_info[s[0]]['run_num'])+'/'+s[0]+'/fit_results/'+s[0]+'_chains.h5')
        weights = ch['weights']
        header = 'eval[0].'+s[0]+'.'

        if sys_err == True:
            header_sys = 'eval[0].'+s[0]+'_sys.'
            params.append('err')
            
        for p in params:
            if p != 'err':
                param = ch[header+p]
            else:
                param = ch[header_sys+p]
            med, lo, hi = stats.weighted_quantile(param, quantiles, sample_weight = weights )
            ave_sig = ((med - lo) + (hi - med))/2.
            star_info[s[0]][p] = med
            star_info[s[0]][p+"_sig"] = ave_sig
            
        
        t1 = star_info[s[0]]['ast_start']
        t2 = star_info[s[0]]['ast_end']
        P = star_info[s[0]]['P']
        e = star_info[s[0]]['e']
        T0 = star_info[s[0]]['T0']
        
        #shift T0 closest to observed range
        if T0 > t2:
            if (T0 - 3.*P) >= t1:
                T0 = T0 - 3.*P
            if (T0 - 2.*P) >= t1:
                T0 = T0 - 2.*P
            if (T0 - P) >= t1:
                T0 = T0 - P
            star_info[s[0]]['T0']=T0
            
        phase_cov_ast = pc.get_phase_coverage(t1,t2,T0,e,P)
        star_info[s[0]]['phase_cov_ast_median'] = phase_cov_ast
    
        #get MAP values to compare how much it changes phase coverage
        csvfile = fit_root + 'run'+str(star_info[s[0]]['run_num'])+'/'+s[0]+'/fit_results/'+s[0]+'_param_stats.csv' #file with MAP params (maximum a posteriori), median, and 1-sig (68% confidence on central confidence interval) parameter estimates, created in runNSO.py stage
        tab = pd.read_csv(csvfile, index_col = 0)  #index_col = 0 means it will use the 0th column (param names) as the row labels
        P_MAP = tab.loc['P']['MAP']
        T0_MAP = tab.loc['T0']['MAP']
        e_MAP = tab.loc['e']['MAP']
        #shift T0 closest to observed range        
        if T0_MAP > t2:
            if (T0_MAP - 2.*P_MAP) >= t1:
                T0_MAP = T0_MAP - 2.*P_MAP
            if (T0_MAP - P_MAP) >= (t1-1):  #give 1 year of grace for t1 for S0-103 
                T0_MAP = T0_MAP - P_MAP
    
        star_info[s[0]]['T0_MAP']=T0_MAP
        star_info[s[0]]['P_MAP']=P_MAP
        star_info[s[0]]['e_MAP']=e_MAP

        phase_cov_ast = pc.get_phase_coverage(t1,t2,T0_MAP,e_MAP,P_MAP)
        star_info[s[0]]['phase_cov_ast_MAP'] = phase_cov_ast

        star_info[s[0]]['percent_of_period_med_ast'] = 100.*(t2-t1) / P
        star_info[s[0]]['percent_of_period_MAP_ast'] = 100.*(t2-t1) / P_MAP
    
        #rv file doesn't change with iterative fits, so can just take rv file from first iteration
        if s[2] == 'RV': 
            rv_file = fit_root + 'run1/'+s[0]+'/data/'+s[0]+'.rv'
            star_info[s[0]]['rv_start'] = np.loadtxt(rv_file)[:,0][0]
            star_info[s[0]]['rv_end'] = np.loadtxt(rv_file)[:,0][-1]
            star_info[s[0]]['num_rv_epochs'] = len(np.loadtxt(rv_file)[:,0])

            
            t1 = star_info[s[0]]['rv_start']
            t2 = star_info[s[0]]['rv_end']            
            phase_cov_rv = pc.get_phase_coverage(t1,t2,T0,e,P)
            star_info[s[0]]['phase_cov_rv_median'] = phase_cov_rv
            phase_cov_rv = pc.get_phase_coverage(t1,t2,T0_MAP,e_MAP,P_MAP)
            star_info[s[0]]['phase_cov_rv_MAP'] = phase_cov_rv

            star_info[s[0]]['percent_of_period_med_rv'] = 100.*(t2-t1) / P
            star_info[s[0]]['percent_of_period_MAP_rv'] = 100.*(t2-t1) / P_MAP
            
        else:
            star_info[s[0]]['phase_cov_rv_median'] = 0.
            star_info[s[0]]['phase_cov_rv_MAP'] = 0.
            star_info[s[0]]['rv_start'] = 0.
            star_info[s[0]]['rv_end'] = 0.
            star_info[s[0]]['percent_of_period_med_rv'] = 0.
            star_info[s[0]]['percent_of_period_MAP_rv'] = 0.
            star_info[s[0]]['num_rv_epochs'] = 0.
            
        star_info[s[0]]['phase_cov_tot_median'] = math.sqrt(star_info[s[0]]['phase_cov_ast_median']**2 + star_info[s[0]]['phase_cov_rv_median']**2)
        star_info[s[0]]['phase_cov_tot_MAP'] = math.sqrt(star_info[s[0]]['phase_cov_ast_MAP']**2 + star_info[s[0]]['phase_cov_rv_MAP']**2)


        #get rmse of residuals for final fit, as long as loo_resids...txt files exist
        if use_resids == True:
            x_resids = np.loadtxt(fit_root + 'run'+str(star_info[s[0]]['run_num'])+'/'+s[0]+'/plots/loo_resids_pos.x.txt')[:,1]
            y_resids = np.loadtxt(fit_root + 'run'+str(star_info[s[0]]['run_num'])+'/'+s[0]+'/plots/loo_resids_pos.y.txt')[:,1]
            star_info[s[0]]['x_resid_rmse'] = get_rmse(x_resids, unbiased_estimator = False, variance = False)
            star_info[s[0]]['y_resid_rmse'] = get_rmse(y_resids, unbiased_estimator = False, variance = False)
        
        #include magnitude in star_info dictionary, if mag has been manually inputted as a fourth column in the starList
        if use_mag == True:
            star_info[s[0]]['kp_mag'] = s[3]



    
            
    return star_info





#get_runNum_from_iterativeFits(starList, total_runs, fit_root)    
star_info = get_phaseCoverage_from_iterativeFits(starList, total_runs, fit_root,
                                                 sys_err = sys_err, use_resids = use_resids, use_mag = use_mag)

#BHpriors = 'fixed'
#for s in starList:  
 #   #just for running the post process analysis after everything was intially run
  #  os.chdir('run'+str(star_info[s[0]]['run_num'])+'/'+s[0]+'/')
   # os.system('python3 ../../_run.py {0} {1} {2}'.format(s[0], s[2], BHpriors))
    #os.chdir('../../')
    
for star in star_info:
    print(star, ': \n', star_info[star], '\n')

print('NOTE: NEED TO MANUALLY CHECK FOR STARS WHERE PHASE COVERAGE IS GREATER THAN 100!')

#output to csv file
headers = ['phase_cov_tot_MAP', 'phase_cov_tot_median', 'phase_cov_ast_MAP', 'phase_cov_ast_median', 'phase_cov_rv_MAP', 'phase_cov_rv_median', 'P', 'P_MAP', 'P_sig', 'T0', 'T0_MAP', 'T0_sig', 'e', 'e_MAP', 'e_sig','ast_start', 'ast_end', 'rv_start', 'rv_end', 'percent_of_period_MAP_ast', 'percent_of_period_med_ast', 'percent_of_period_MAP_rv', 'percent_of_period_med_rv', 'num_rv_epochs', 'num_astro_epochs_used', 'percentPossible_astro_epochs_used', 'percentPossible_confuseCorrected_astro_epochs', 'percentPossible_initialMatched_astro_epochs']
if sys_err == True:
    headers.append('err')
    headers.append('err_sig')

if use_resids == True:
    headers.append('x_resid_rmse')
    headers.append('y_resid_rmse')
if use_mag == True:
    headers.append('kp_mag')
df = pd.DataFrame.from_dict(star_info, orient = 'index', columns = headers)

if total_runs == 1:
    df.to_csv(fit_root + 'star_info_firstFits.csv')
if total_runs >1:
    df.to_csv(fit_root + 'star_info_afterIterativeFits.csv')

