import nstarorbits as NSO
import sys
import nstarorbits as NSO
from nstarorbits import utils
from nstarorbits import plot
from nstarorbits import stats
import pandas as pd
import createOrbitsDat_utils as utl
import numpy as np
import matplotlib.pyplot as plt
from shutil import copy2
import os
from nstarorbits import use_loo_example as loo
import yaml
import time


star = sys.argv
print(star[1])
print(star)


run = True
#run: bool, True if you want to run the orbit fit, False if you just want to do the analysis (if orbit has already been fit)
cut = True
#cut: bool, True if you want to cut epochs and generate new _keep and _remove points files, False if not. 
get_corner = True
#get_corner: bool, True if you want to plot corner plots, false if not (option exists because it takes a long time to plot corner plots, and is unnecessary if you're just interested in the other post processing)
#add_tag = False
#add_tag: bool, True if the fit includes more than one param map (e.g. if sys.err used,or additional star). False if not. This just affects which parameters are plotted in the corner plots.


#***************DEFINE CUT CRITERIA*********************#

#Astrometric criteria is based on p-value. Example: p_cutoff = 0.003 corresponds to a 3 sigma cutoff; in other words, if p-value is less than 0.003, there is a 0.3%chance of that data point being that value, given all other data points except for that one, or a 99.7% chance that the point would not be replicated

#photometric criteria is based on deviation in magnitude in units of sigma. Example: dm_cutoff = 3 corresponds to 3 sigma deviation between that measurement and the model (crudely, a constant magnitude model, taking into account both measurement uncertainty and rms between data points)
ast_cutoff = 0.003
phot_cutoff = 2.  

#***************RUN NSTARORBITS****************#

#Load observables, objective, and sampler from yaml config file
obs, obj, samp = NSO.load_all('./config_files/'+star[1]+'.yaml') 

if run == True:
    #run NSO and record run time
    start_time = time.process_time()
    samp.run()
    end_time = time.process_time()
    run_time = end_time-start_time
    time_file = open('./fit_results/runtime.txt', 'w')
    time_file.write(str(end_time-start_time))
    time_file.close() #gives run time in cpu seconds
    # can load the timing data as: f=np.loadtxt('runtime.txt')
    
    
    #Save chains
    utils.save_nstarorbits_chains(samp, savefile='./fit_results/'+star[1]+'_chains.h5')

#************* GET FIT RESULTS ***********************#

#get printer object
pr = samp.get_printer()

#access chains
print('Accessing Chains...')
ch = pr.chain(weights = 'weights', priors_only=True)  #ch is a dictionary
ch_cut = pr.chain(weights = 'weights',  min_wt = 0.000001,  priors_only=True)
#ch_thin = pr.chain(weights = 'weights', thin = 10,  priors_only=True) #gets 'evenly weighted' posteriors
##TO DO: have it output eccentricity even weighted chains as .txt file
#TO DO: Change to BH gaussian priors

if get_corner == True:
    names = []
    data = []
    for a, b in ch.items():
        if a != "weights" and a != "lnlike" and a != '__obj':
            names.append(a)
            data.append(b)
            print("{0}: {1}".format(a, stats.weighted_quantile(b, [0.1585, 0.5, 0.8415], sample_weight=ch["weights"])))
 
    #plot.makeTriPlot(np.transpose(np.array(data)), ch["weights"], "./plots/corner.png", names, star[1])

    #corner plot with everything
    names_cut = []
    data_cut = []
    for a, b in ch_cut.items():
        if a != "weights" and a != "lnlike" and a != '__obj':
            names_cut.append(a)
            data_cut.append(b)
    #plot.makeTriPlot(np.transpose(np.array(data_cut)), ch_cut["weights"], "./plots/corner_weights_cut.png", names_cut, star[1]) 

    #TO DO: Edit so utils can take in params with different tags to get sys err param included in corner plots. Old way with everything above. 
    
    #Specify desired parameters for which to make corner plots and save output values
    star_header = 'eval[0].'+star[1]+'.'
    include_params_orbit = ['P','T0','e','omega','Omega', 'I']

    names_orbit, labels_orbit, data_orbit = utils.access_chains_arrays(ch, star_header = star_header, include_params = include_params_orbit)
    names_orbit, labels_orbit, data_cut_orbit = utils.access_chains_arrays(ch_cut, star_header = star_header, include_params = include_params_orbit)

    #make corner plots for orbit params (here, plots are created using cut weights for clarity of the figure)
    plot.makeTriPlot(data_cut_orbit, ch_cut["weights"], "./plots/corner_orbitParms_weightsCut.png", labels_orbit, star[1]+ " Orbit Params (weights cut)")
    plot.makeTriPlot(data_orbit, ch["weights"], "./plots/corner_orbitParms.png", labels_orbit, star[1]+" Orbit Params")

    #only make corner plot for global params if they are not fixed
    if star[3] != 'fixed' and star[3] != 'fixed_S02' and star[3]!='fixed_onlyMfree':
        if star[3] == 'fixed_Rvz':
            include_params_global = ['x0', 'y0', 'vx0', 'vy0', 'Mass']            
        if star[3] == 'fixed_MR':
            if star[2] =='RV':
                include_params_global = ['x0', 'y0', 'vx0', 'vy0', 'vz0']
            else:
                include_params_global = ['x0', 'y0', 'vx0', 'vy0']
        if star[3] == 'uniform':    
            if star[2] =='RV':
                include_params_global = ['Mass', 'R0', 'x0', 'y0', 'vx0', 'vy0', 'vz0']
            else:
                include_params_global = ['Mass', 'x0', 'y0', 'vx0', 'vy0']#,'err']
        if star[3] == 'center_free':
            include_params_global = ['x0', 'y0', 'vx0', 'vy0']
        names_global, labels_global, data_global = utils.access_chains_arrays(ch, star_header = star_header, include_params = include_params_global)
        names_global, labels_global, data_cut_global = utils.access_chains_arrays(ch_cut, star_header = star_header, include_params = include_params_global)
        
        plot.makeTriPlot(data_cut_global, ch_cut["weights"], "./plots/corner_globalParams_weightsCut.png", labels_global, star[1]+" Global Params (weights cut)")
        plot.makeTriPlot(data_global, ch["weights"], "./plots/corner_globalParams.png", labels_global, star[1]+" Global Params")

        all_names = names_orbit + names_global
        all_labels = labels_orbit + labels_global
      
    if star[3] == 'fixed_onlyMfree': #no corner plot since only one free BH param, but still add Mass to param_stats file 
        include_params_global = ['Mass']
        names_global, labels_global, data_global = utils.access_chains_arrays(ch, star_header = star_header, include_params = include_params_global)
        names_global, labels_global, data_cut_global = utils.access_chains_arrays(ch_cut, star_header = star_header, include_params = include_params_global)
        all_names = names_orbit + names_global
        all_labels = labels_orbit + labels_global
        
    if star[3] == 'fixed' or star[3] == 'fixed_S02': #no bh corner plot or param stats since they're all fixed
        all_names = names_orbit
        all_labels = labels_orbit
    #write output file with MAP and median quantiles
    utils.write_param_stats_file(ch, quantiles = [0.5, 0.00135, 0.02275, 0.1585, 0.8415, 0.97725, 0.99865], star_header = star_header, include_params = all_labels, out_fname = './fit_results/'+star[1]+'_param_stats.csv', save = True)
    


#***************ACCESS DATA****************#
print('Getting Photometric Data')
#get photometric data
t, phot, phot_err = utl.get_photometry(star[1], root_dir = './data/')
#Calculate photometric deviation for each epoch
t, delta_phot = utl.get_delta_phot_chrono(star[1], root_dir = './data/', kp_h_change = [2017.3480, 2018.6730], k_kp_change = 2006., rmse = True)

#get astrometric observables 
x = obs.get(star[1]).get('position')['x']
y = obs.get(star[1]).get('position')['y']
xerr = obs.get(star[1]).errors('position::x') 
yerr = obs.get(star[1]).errors('position::y') 
positions = obs.get(star[1]).get("position")
pos_time = obs.get(star[1]).get('position')['times']
#if obs defined in python not loaded from yaml: pos_time = obs.get("position")["times"]

#get spectroscopic observables
if star[2] == 'RV':
    rv = obs.get(star[1]).get("velocity")['z'] #
    rverr = obs.get(star[1]).errors('velocity::z')
    velocities = obs.get(star[1]).get('velocity')
    vel_time = obs.get(star[1]).get('velocity')['times']

#***************GET MODEL****************#

print('Getting leave-one-out model')
#get leave-one-out information
if star[2] =='RV':
    cls = pr.loo(obs={"pos": positions, "vel": velocities}, weights= "weights", mc_num=1000, err_env = False)
    #Write goodness of fit metrics to file
    stats.get_fit_statistics(cls, pos = True, vel = True, save_dir = './fit_results/')
else:
    cls = pr.loo(obs={"pos": positions}, weights= "weights", mc_num=1000, err_env = False)
    #Write goodness of fit metrics to file
    stats.get_fit_statistics(cls, pos = True, vel = False, save_dir = './fit_results/')  #if no RV data

#get pvalues and save to file
dates, pvalue_x, pvalue_y = stats.get_pvalue(cls, save_dir = './fit_results/')

#load info from yaml file
ini = yaml.load(open('./config_files/'+star[1]+".yaml"), Loader=yaml.FullLoader)

#get name of star and model from the yaml file
#NOTE(nstars): this is only setup for single star fits so far
#This requires that the yaml file follows a standard structure
tags = []
models=[]
    
for item in ini['evaluators']['eval1']:
    tags.append(item)
    for mod in ini['evaluators']['eval1'][item]:
        models.append(mod)
#star = tags[0]
model = models[0]
    
#Load parameters
params = NSO.load_params('./config_files/'+star[1]+".yaml")
param_names = []
for tag in tags:
    for p in params[tag]:
        param_names.append(p)

#Get dynamical T0 that was used in the fit
#This is the epoch at which fitted parameters are estimated
if 't_origin_BH' in ini['evaluators']['eval1'][star[1]][model][2]:
    t_origin_BH = ini['evaluators']['eval1'][star[1]][model][2]['t_origin_BH']
else: #if not specified, assume the NStarOrbits default for dynamical T0
    t_origin_BH = 2000.

model_obs_format = {"position": (NSO.variable("x [marcsec]"), NSO.variable("y [marcsec]"), NSO.conditional("times [yrs]"), NSO.length(4000)),
                        "velocity": (NSO.variable("z [km/sec]"), NSO.conditional("times [yrs]"), NSO.length(4000))}
model_obs = NSO.observable( model_obs_format ) 
model_obs.get("position")["times"][:] = np.linspace(1995, 2025, 4000)
model_obs.get("velocity")["times"][:] = np.linspace(1995, 2025, 4000)

model_positions= model_obs.get("position")
model_velocities= model_obs.get("velocity")

#Model objective needs to be updated in order to accomodate multiple models.
# Must match the format of the objective from the original orbit fit, else error envelopes will not represent the exact model.
if len(models) > 1:
    model_ev_format = {star[1]: {model: (params[star[1]], model_obs,  {"t_origin_BH": t_origin_BH})},
                  tags[1]: {models[1]: (params[tags[1]], model_obs.get("position"))} }
else:
    model_ev_format = {star[1]: {model: (params[star[1]], model_obs,  {"t_origin_BH": t_origin_BH})}}
model_ev = NSO.evaluator(model_ev_format, init_models = True)

model_obj = NSO.objective("normal", {}, model_obs.get("position"), model_ev) + NSO.objective("normal", {}, model_obs.get("velocity"), model_ev)

model_obj.gen() #generate new data from the objective
model_cls = pr.loo(obs={"pos": model_obs.get("position"), "vel":model_obs.get("velocity")}, weights="weights", mc_num=1000, err_env = True, lnlike = model_obj)
#model_cls = loo.load_cls_fromPy(model_obs, model_obj, samp, run = False, err_env = True, lnlike = model_obj)
#model_cls = loo.gen_model_cls(star[1], model, params[star[1]], samp, tmin = 2000, tmax = 2025, t_origin_BH = t_origin_BH, n = 1000)
#NOTE: if useing the above shortcut, plotted envelope does not include sys err. Not completely accurate


#*************MAKE ADDITIONAL PLOTS  ...  *************#

utl.plot_phot(star[1], root_dir = './data/', plot_err = True, save_to = './plots/')
utl.plot_pvalue_vs_delta_phot(delta_phot, pvalue_x, pvalue_y, t, ast_cutoff, phot_cutoff, save_to = './plots/')
utl.plot_orbit(star[1], root_dir = './data/', save_to = './plots/')


#************* GENERATE MODEL FILES ***********************#

#get continuous loo model and error envelope between a desired date range (see docstrings for additional keywords to set date range):
model_t, model_x, model_x_low, model_x_hi = loo.access_loo_model(model_cls, obs_tag = 'pos.x')
model_t, model_y, model_y_low, model_y_hi = loo.access_loo_model(model_cls, obs_tag = 'pos.y')

f=open('./model.points', 'w+')
for i in range(len(model_t)):
     f.write('%f %f %f %f %f %f %f \n' % (model_t[i], model_x[i], model_y[i], model_x_low[i], model_y_low[i], model_x_hi[i], model_y_hi[i]))

if star[2] =='RV':
    model_rv_t, model_vz, model_vz_low, model_vz_hi = loo.access_loo_model(model_cls, obs_tag = 'vel.z')
    f=open('./model.rv', 'w+')
    for i in range(len(model_rv_t)):
        f.write('%f %f %f %f \n' % (model_rv_t[i], model_vz[i], model_vz_low[i], model_vz_hi[i]))

#************* POST PROCESSING ***********************#

#plot LOO resiuals, and x or y data vs time
plot.plot(cls, obs=positions, 
          names={"pos.x": "x", "pos.y": "y"}, 
          ylabels={"pos.x": "x [mas]", "pos.y": "y [mas]"},
          titles={"pos.x": star[1], "pos.y": star[1]},
          env = model_cls,
          save_dir = './plots/')

#plot LOO resiuals, and model overlaid with RV data vs time
#Need to remove duplicated epochs
if star[2] == 'RV':
    plot.plot(cls, obs=velocities, 
          names={"vel.z": "z"}, 
          ylabels={"vel.z": "RV [km/s]"},
          titles={"vel.z": star[1]},
          env = model_cls,
          save_dir = './plots/')



#***************CUT OUTLIER EPOCHS*********************#

obs_list = [x,y,xerr,yerr,phot,phot_err]  #for now, only astrometry and photometry are being cut, not rv
#create new files with epochs that do or don't meet cutoff criteria
if cut == True:
    utl.cut_outliers(ast_cutoff=ast_cutoff, phot_cutoff=phot_cutoff, dates=t, obs_list = obs_list, delta_phot=delta_phot, cls=cls, outfile_root='./to_cut/'+star[1])


