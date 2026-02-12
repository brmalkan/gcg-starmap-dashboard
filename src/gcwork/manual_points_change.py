import pylab as py
import numpy as np
from astropy.table import Table
from astropy.io import fits
from astropy.time import Time
import os, shutil
import subprocess
from gcwork import starset
from gcwork import starTables
import gcreduce
from gcreduce import applyLocalDist
from gcreduce import gcutil
from gcwork import accel_class as acc
import sqlite3 as sqlite
import datetime
import glob
from collections import OrderedDict
import pdb
import pandas as pd

def add_point(
        name,name_epoch,epoch,lis_file=None,align_dir=None,align_name=None,
        align_filename=None,align_x=None,align_y=None,align_xerr=None,
        align_yerr=None,align_mag=None,dbfile='manual_star_match.sqlite',
        ao_version=None,
        speckle_version=None,
    ):
    """
    Add a new manual identification into the database to give it information about
    what the names of stars should be in a starfiner .lis file from different
    epochs. This is useful for keeping the information from manual identification
    of stars. This works well in conjunction with manual searches with compare_pos.

    Inputs:
    -------
    name - name of the star (e.g. "S0-2")
    name_epoch - the name given in the starfinder .lis file for that epoch
                 (e.g. "star_74")
    epoch - The Julian epoch date (e.g. 2000.123)

    Keywords:
    --------
    lis_file - the .lis file to use for additional information (e.g the x,y,xerr,
    yerr info from the rms.lis file) to put into the database

    dbfile - the database with to store this information
    (Default: 'manual_star_match.sqlite')
    """
    
    table_name='matched_stars'
    if ao_version is not None:
        table_name = 'ao_v' + ao_version
    elif speckle_version is not None:
        table_name = 'speckle_v' + speckle_version
    
    if os.path.exists(dbfile):
        print("using database: "+dbfile)
        connection = sqlite.connect(dbfile,detect_types=sqlite.PARSE_DECLTYPES)
        cur = connection.cursor()
        today = datetime.datetime.now()
        if lis_file is None:
            # no .lis file is given so, we can only fill in a few of the fields
            query = 'SELECT name,name_epoch,epoch FROM ' + table_name + ' WHERE name=? AND name_epoch=? AND epoch=?'
            cur.execute(query,(name,name_epoch,epoch))
            stack = cur.fetchall()
            if len(stack) == 0:
                print("%s / %s is not in the database, adding it" % (name,name_epoch))
                query2 = 'INSERT INTO ' + table_name + ' (name, name_epoch, epoch,date_added,align_dir,align_name, align_filename,align_x,align_y,align_xerr,align_yerr,align_mag) VALUES (?, ? ,?, ?,?,?,?,?,?,?,?,?)'
                cur.execute(query2,(name,name_epoch,epoch,today,align_dir,align_name, align_filename,align_x,align_y,align_xerr,align_yerr,align_mag))
            else:
                print("%s / %s is already in the database, updating it" % (name,name_epoch))
                query2 = 'UPDATE ' + table_name + ' SET name=?, name_epoch=?, epoch=?,date_added=?,align_dir=?,align_name=?, align_filename=?,align_x=?,align_y=?,align_xerr=?,align_yerr=?,align_mag=? WHERE name=? AND name_epoch = ? AND epoch = ?'

                cur.execute(query2,(name,name_epoch,epoch,today,align_dir,align_name, align_filename,align_x,align_y,align_xerr,align_yerr,align_mag,name,name_epoch,epoch))
            connection.commit()
            cur.close()
            connection.close()

    else:
        print('dbfile: '+dbfile+' not found.')

def test_add_point(dbfile='manual_star_match.sqlite'):
    # test adding into db

    add_point('S0-38','star_1120',2010.342,dbfile=dbfile)

def run_manual_star_match(rootDir,points='points_1_1000/',poly='polyfit_1_1000/fit',
                          outputdir=None,dbfile='manual_star_match.sqlite',
                          speckle_version='2_2',ao_version='1_1'):
    '''
    Run this function to change points files based on manually determined
    point identifications in manual_star_match.sqlite. This is basically just a
    wrapper to load accel_class for input into modify_points_files()

    NOTE: this needs both an align and a points directory to target. Will not work
    on a points_mjd directory because the align doesn't know about MJDs at the
    moment (2017-08-04)

    Inputs
    ------
    rootDir - same as argument for accel_class (the root name of the align
    directory (e.g: /u/ghezgroup/align/17_08_01/))

    Keywords
    --------
    points - points directory to load that will be used to modify later
    (Default:'points_1_1000/')
    outputdir -

    '''

    a = acc.accelClass(rootDir = rootDir,points=points,poly=poly)
    if outputdir is None:
        outputdir = points
    modify_points_files(a,dbfile=dbfile,outdir=os.path.join(rootDir,outputdir),
                        speckle_version=speckle_version,ao_version=ao_version)

def modify_points_files(accel_class_obj,dbfile='manual_star_match.sqlite',outdir='./',
                        speckle_version='2_2',ao_version='1_1'):
    '''
    Modify points files directory based on the manual star matching database.
    This function also requires an align in order to know what points are
    matched.

    Inputs:
    -------
    accel_class_obj - accel_class.accelClass object
    (this is so that we don't have to load the align each time)

    Keywords
    --------
    dbfile - the sqlite database files

    '''
    connection = sqlite.connect(dbfile,detect_types=sqlite.PARSE_DECLTYPES)
    cur = connection.cursor()
    # select both the speckle and AO tables for manual matches
    query = 'SELECT * FROM speckle_v'+speckle_version+' UNION ALL select * FROM ao_v'+ao_version
    cur.execute(query,())
    stack = cur.fetchall()
    print(stack[0])
    epochs = accel_class_obj.allEpochs
    changed_stars = []
    data = {}   # empty dictionary to store data
    failed = []
    for i in np.arange(len(stack)):
        row = stack[i]

        starind = np.where(accel_class_obj.names == row[0])[0]
        if len(starind) > 0:
            starind = starind[0]
        else:
            print("Star was not found in align: "+row[0])
            continue
        print("analyzing star: "+row[0]+' in index of main align: '+str(starind))

        temp_epoch = row[3]
        name_epoch = row[1]   # the name of the star in that epoch
        epochind = np.where(epochs == temp_epoch)[0]
        if len(epochind) > 0:
            epochind = epochind[0]
        else:
            print('epoch not found: '+str(temp_epoch))
            continue

        print("in epoch "+str(temp_epoch)+" this was is named: "+name_epoch+" in epoch number: "+str(epochind))

        # look up the location of the star in the right epoch
        name_epoch_arr  = np.array(accel_class_obj.starSet.getArrayFromEpoch(epochind,'name'))
        epoch_starind = np.where(name_epoch_arr == name_epoch)[0]
        if len(epoch_starind) > 0:
            epoch_starind = epoch_starind[0]
        else:
            print('star not found in the given epoch: '+name_epoch, epochs[epochind])
            failed.append(row[0]+' not found in the given epoch: '+name_epoch+' '+str(epochs[epochind]))
            continue

        print("This star was found in index "+str(epoch_starind)+" corresponding to the star in original align: "+accel_class_obj.names[epoch_starind])


        # get all the x and y positions of a given epoch and select the one that is relevant
        temp_x = accel_class_obj.starSet.getArrayFromEpoch(epochind,'pnt_x')[epoch_starind]
        temp_y = accel_class_obj.starSet.getArrayFromEpoch(epochind,'pnt_y')[epoch_starind]
        temp_xerr = accel_class_obj.starSet.getArrayFromEpoch(epochind,'pnt_xe')[epoch_starind]
        temp_yerr = accel_class_obj.starSet.getArrayFromEpoch(epochind,'pnt_ye')[epoch_starind]


        temp_phot_x = accel_class_obj.starSet.getArrayFromEpoch(epochind,'phot_x')[epoch_starind]
        temp_phot_y = accel_class_obj.starSet.getArrayFromEpoch(epochind,'phot_y')[epoch_starind]
        temp_phot_xerr = accel_class_obj.starSet.getArrayFromEpoch(epochind,'phot_xe')[epoch_starind]
        temp_phot_yerr = accel_class_obj.starSet.getArrayFromEpoch(epochind,'phot_ye')[epoch_starind]
        temp_phot_mag = accel_class_obj.starSet.getArrayFromEpoch(epochind,'phot_mag')[epoch_starind]
        temp_phot_magerr = accel_class_obj.starSet.getArrayFromEpoch(epochind,'phot_mage')[epoch_starind]
        temp_phot_r = accel_class_obj.starSet.getArrayFromEpoch(epochind,'phot_r')[epoch_starind]

        print(temp_epoch,temp_x,temp_y,temp_xerr,temp_yerr)
        matched_star =accel_class_obj.names[epoch_starind]
        if accel_class_obj.names[epoch_starind] != row[0]:

            # NOTE: only add to the changed_epoch list if the epoch has been
            # replaced with a CORRECT value. This is to avoid the problem of
            # removing those points when swapping two sources
            if row[0] in changed_stars:

                print(row[0]+" is one of the stars that have already been changed, editing.")
                data[row[0]+'_changed_epoch'].append(epochs[epochind])
            else:
                print(row[0]+" is *not* of the stars that have already been changed, adding.")
                changed_stars.append(row[0])
                data[row[0]+'_x'] = accel_class_obj.starSet.stars[starind].getArrayAllEpochs('pnt_x')
                data[row[0]+'_y'] = accel_class_obj.starSet.stars[starind].getArrayAllEpochs('pnt_y')
                data[row[0]+'_xe'] = accel_class_obj.starSet.stars[starind].getArrayAllEpochs('pnt_xe')
                data[row[0]+'_ye'] = accel_class_obj.starSet.stars[starind].getArrayAllEpochs('pnt_ye')

                data[row[0]+'_phot_x'] = accel_class_obj.starSet.stars[starind].getArrayAllEpochs('phot_x')
                data[row[0]+'_phot_y'] = accel_class_obj.starSet.stars[starind].getArrayAllEpochs('phot_y')
                data[row[0]+'_phot_xe'] = accel_class_obj.starSet.stars[starind].getArrayAllEpochs('phot_xe')
                data[row[0]+'_phot_ye'] = accel_class_obj.starSet.stars[starind].getArrayAllEpochs('phot_ye')
                data[row[0]+'_phot_mag'] = accel_class_obj.starSet.stars[starind].getArrayAllEpochs('phot_mag')
                data[row[0]+'_phot_mage'] = accel_class_obj.starSet.stars[starind].getArrayAllEpochs('phot_mage')
                data[row[0]+'_phot_r'] = accel_class_obj.starSet.stars[starind].getArrayAllEpochs('phot_r')

                data[row[0]+'_changed_epoch'] = [epochs[epochind],]

            data[row[0]+'_x'][epochind] = temp_x
            data[row[0]+'_y'][epochind] = temp_y
            data[row[0]+'_xe'][epochind] = temp_xerr
            data[row[0]+'_ye'][epochind] = temp_yerr

            data[row[0]+'_phot_x'][epochind] = temp_phot_x
            data[row[0]+'_phot_y'][epochind] = temp_phot_y
            data[row[0]+'_phot_xe'][epochind] = temp_phot_xerr
            data[row[0]+'_phot_ye'][epochind] = temp_phot_yerr
            data[row[0]+'_phot_mag'][epochind] = temp_phot_mag
            data[row[0]+'_phot_mage'][epochind] = temp_phot_magerr
            data[row[0]+'_phot_r'][epochind] = temp_phot_r

            if matched_star in changed_stars:
                print(matched_star+" is one of the stars that have already been changed, editing.")
            else:
                print(matched_star+" is *not* of the stars that have already been changed, adding.")
                changed_stars.append(matched_star)
                data[matched_star+'_x'] = accel_class_obj.starSet.stars[epoch_starind].getArrayAllEpochs('pnt_x')
                data[matched_star+'_y'] = accel_class_obj.starSet.stars[epoch_starind].getArrayAllEpochs('pnt_y')
                data[matched_star+'_xe'] = accel_class_obj.starSet.stars[epoch_starind].getArrayAllEpochs('pnt_xe')
                data[matched_star+'_ye'] = accel_class_obj.starSet.stars[epoch_starind].getArrayAllEpochs('pnt_ye')

                data[matched_star+'_phot_x'] = accel_class_obj.starSet.stars[epoch_starind].getArrayAllEpochs('phot_x')
                data[matched_star+'_phot_y'] = accel_class_obj.starSet.stars[epoch_starind].getArrayAllEpochs('phot_y')
                data[matched_star+'_phot_xe'] = accel_class_obj.starSet.stars[epoch_starind].getArrayAllEpochs('phot_xe')
                data[matched_star+'_phot_ye'] = accel_class_obj.starSet.stars[epoch_starind].getArrayAllEpochs('phot_ye')
                data[matched_star+'_phot_mag'] = accel_class_obj.starSet.stars[epoch_starind].getArrayAllEpochs('phot_mag')
                data[matched_star+'_phot_mage'] = accel_class_obj.starSet.stars[epoch_starind].getArrayAllEpochs('phot_mage')
                data[matched_star+'_phot_r'] = accel_class_obj.starSet.stars[epoch_starind].getArrayAllEpochs('phot_r')
                data[matched_star+'_changed_epoch'] = []

            if epochs[epochind] in data[matched_star+'_changed_epoch']:
                print("This epoch in the mismatched star has already been changed, don't edit")
            else:
                print("Removing this epoch from the mismatched star ")
                data[matched_star+'_x'][epochind] = -1000.0
                data[matched_star+'_y'][epochind] = -1000.0
                data[matched_star+'_xe'][epochind] = -1000.0
                data[matched_star+'_ye'][epochind] = -1000.0

                data[matched_star+'_phot_x'][epochind] = -1000.0
                data[matched_star+'_phot_y'][epochind] = -1000.0
                data[matched_star+'_phot_xe'][epochind] = -1000.0
                data[matched_star+'_phot_ye'][epochind] = -1000.0
        else:
            print("this star is already correctly matched. Do nothing.\n")

    print(changed_stars)
    # writout the new points files
    for s in changed_stars:
        filename = os.path.join(outdir,s+'.points')
        filename_phot = os.path.join(outdir,s+'.phot')
        if os.path.exists(filename):
            print("File "+filename+" exists, making a copy")
            orig_file = os.path.join(outdir,s+'_orig.points')
            try:
                shutil.copy(filename,orig_file)
            except OSError as err:
                print("OS error: {0}".format(err))                
        if os.path.exists(filename_phot):
            print("File "+filename_phot+" exists, making a copy")
            orig_file_phot = os.path.join(outdir,s+'_orig.phot')
            try:
                shutil.copy(filename,orig_file_phot)
            except OSError as err:
                print("OS error: {0}".format(err))                
               

        # remove undefined points in each star and write out the .points and
        # .phot files
        good = np.where(data[s+'_xe'] != -1000.0)[0]
        points_dict = OrderedDict([('epoch',epochs[good]),
                       ('x',data[s+'_x'][good]),
                       ('y',data[s+'_y'][good]),
                       ('xerr',data[s+'_xe'][good]),
                       ('yerr',data[s+'_ye'][good])])
        points_tab = Table(points_dict)

        phot_dict = OrderedDict([('epoch',epochs[good]),
                     ('r',data[s+'_phot_r'][good]),
                     ('x',data[s+'_phot_x'][good]),
                     ('y',data[s+'_phot_y'][good]),
                     ('xerr',data[s+'_phot_xe'][good]),
                     ('yerr',data[s+'_phot_ye'][good]),
                     ('mag',data[s+'_phot_mag'][good]),
                     ('mage',data[s+'_phot_mage'][good])])
        phot_tab = Table(phot_dict)

        print("Outputing Star: "+s)
        starTables.write_points(points_tab,filename)
        starTables.write_phot_points(phot_tab,filename_phot)


    print('the following stars failed')
    print(failed)

def test_manual_star_match(points):
    # run a test
    rootDir = '/u/ghezgroup/align/17_08_01/'

    # make a test directory if it doesn't exists
    test_dir = os.path.join(rootDir,'points_1_1000_test')
    if os.path.exists(test_dir):
        pass
    else:
        os.mkdir(test_dir)

    # delete files in this directory so we can have a clean test
    files = glob.glob(test_dir+'/*.points')
    for f in files:
        os.remove(f)

    dbfile = os.path.join(rootDir,'source_list/manual_star_match.sqlite')
    # to do this test, we'll load from one points directory but write into
    # another so that we don't overwrite anything
    print("testing using align directory "+rootDir+" with points directory points_1_1000")
    print("will output in "+test_dir)
    run_manual_star_match(rootDir,points='points_1_1000/',outputdir=test_dir,
    dbfile=dbfile)

def remove_points(rootDir='../', points='points_3_c/', manualepochsfile='source_list/confused_epochs_astrometry.txt'):
    infile = open(rootDir+manualepochsfile,'r')
    workDir = rootDir + points

    for line in infile:
        parts = line.split(' ')
        star = parts[0]
        remove_temp = np.array(parts[1:],dtype=float)

        ptFile = workDir + star + '.points'
        phFile = workDir + star +'.phot'

        if not os.path.exists(ptFile):
            print('when manually removing: %s not found in %s' %(star, workDir))
            continue

        if os.stat(ptFile).st_size == 0:
            print('%s is empty' %ptFile)
            continue

        tab = starTables.read_points(ptFile)
        phot = starTables.read_points(phFile)

        # save the original file
        ptFile_orig = workDir + star + '_orig.points'
        phFile_orig = workDir + star +'_orig.phot'

        starTables.write_points(tab, ptFile)
        starTables.write_points(phot, phFile)

        #print('Those epochs are removed in %s:' %ptFile)
        for ee in remove_temp:
            epochs = tab['epoch']
            bad = np.where(epochs == ee)[0]
            if len(bad) > 0:
                tab.remove_rows(bad[0])
                phot.remove_rows(bad[0])
                #print(ee)

        starTables.write_points(tab, ptFile)
        starTables.write_points(phot, phFile)

def shift_residual_distortion(starnames,points_dir,out_dir=None,metadata=True,
                              add_errors = True, use_orbits = True,use_bootstrap=True,
                              residual_file='/u/tdo/research/source_list/astrometry/epoch_wgtd_resids_18_07_02.txt',
                              use_res_as_error=False):
    '''
    Shifts points files based on residual distortion offsets as
    calculated using an align without the local distortion.

    Inputs
    ------
    starnames - list of stars to apply the correction 
    points_dir - the points directory that contains the points files to be shifted

    Keywords
    --------
    add_errors - add the errors from the residual offsets in quadrature (default: True)
    use_orbits - use the residual offsets that include orbiting stars (default: True)
    out_dir - optional output directory to write out the points files
    metadata - write out meta data at the top of the file about the shifts (default: True)
    residual_file - the file where the offsets are located (default: '/u/tdo/research/source_list/astrometry/epoch_wgtd_resids_18_07_02.txt')
    use_bootstrap - use bootstrap errors (default: True)
    use_res_as_error - add an additional error using the absolute value of the distortion (default: False)

    History
    -------
    2018-07-12 - T. Do

    '''

    ## check that the points_dir exists
    if os.path.exists(points_dir):
        
        tab = pd.read_csv(residual_file,delim_whitespace=True,float_precision='round_trip',comment='#')
        #tab[]/tab['resid_x_wgtdMean_unc_noOrbitStars']
        if use_orbits:
            xoffset_key = 'resid_x_wgtdMean_wOrbitStars'
            yoffset_key = 'resid_y_wgtdMean_wOrbitStars'

            if use_bootstrap:
                xoffset_err_key = 'resid_x_wgtdMean_unc_bs_wOrbitStars'
                yoffset_err_key = 'resid_y_wgtdMean_unc_bs_wOrbitStars'
            else:
                xoffset_err_key = 'resid_x_wgtdMean_unc_wOrbitStars'
                yoffset_err_key = 'resid_y_wgtdMean_unc_wOrbitStars'            
        else:
            xoffset_key = 'resid_x_wgtdMean_noOrbitStars'
            yoffset_key = 'resid_y_wgtdMean_noOrbitStars'

            if use_bootstrap:
                xoffset_err_key = 'resid_x_wgtdMean_unc_bs_noOrbitStars'
                yoffset_err_key = 'resid_y_wgtdMean_unc_bs_noOrbitStars'
            else:
                xoffset_err_key = 'resid_x_wgtdMean_unc_noOrbitStars'
                yoffset_err_key = 'resid_y_wgtdMean_unc_noOrbitStars'            

        if out_dir is None:
            print('No output directory is specified. All results will just be displayed on the screen')

        for i in np.arange(len(starnames)):
            starfile = os.path.join(points_dir,starnames[i]+'.points')
            try:
                startab = pd.read_csv(starfile,header=None,delim_whitespace=True,float_precision='round_trip')
            except pd.io.common.EmptyDataError:
                print("File is empty: "+starfile)

            else:
                if len(startab.columns) == 5:
                    startab.columns = ['epoch_yearDate','x','y','xerr','yerr']
                if len(startab.columns) == 6:
                    startab.columns = ['epoch_yearDate','x','y','xerr','yerr', 'epoch_mjd']

                #else:
                 #   startab.columns = ['epoch_yearDate','x','y','xerr','yerr'] + (len(startab.columns)-5)

                if out_dir is None:
                    print(startab)

                    print('----------')
                temptab = startab.merge(tab,on='epoch_yearDate',how='left')
                #print(temptab[['epoch_yearDate','x','y','xerr','yerr',xoffset_key,xoffset_err_key,yoffset_key,yoffset_err_key]])

                temptab['x'] = temptab['x'] - temptab[xoffset_key]/1000.0
                temptab['y'] = temptab['y'] - temptab[yoffset_key]/1000.0

                if add_errors:
                    if use_res_as_error:
                        res_err_x = np.abs(temptab[xoffset_key]/1000.0)
                        res_err_y = np.abs(temptab[yoffset_key]/1000.0)
                    else:
                        res_err_x = 0.0
                        res_err_y = 0.0
                    temptab['xerr'] = np.sqrt(temptab['xerr']**2 + (temptab[xoffset_err_key]/1000.0)**2+ res_err_x**2)
                    temptab['yerr'] = np.sqrt(temptab['yerr']**2 + (temptab[yoffset_err_key]/1000.0)**2+ res_err_y**2)

                if out_dir is None:
                    print(temptab[startab.columns])
                else:
                    out_file = os.path.join(out_dir,os.path.basename(starfile))
                    output = open(out_file,'w')
                    if metadata:
                        output.write('#'+str(datetime.datetime.now())+'\n')
                        output.write('#input : '+os.path.realpath(starfile)+'\n')
                        output.write('#residual_file : '+residual_file+'\n')
                        output.write('#add_errors : '+str(add_errors)+'\n')
                        output.write('#use_orbits : '+str(use_orbits)+'\n')
                        output.write('#use_res_as_error:'+str(use_res_as_error)+'\n')
                    temptab[startab.columns].to_csv(output,index=False,header=False,sep=' ')
                    output.close()
                # for j in np.arange(len(startab)):
                #     temptab = tab[tab['epoch_yearDate'] == startab[0].iloc[j]]
                #     xoffset,xoffset_err,yoffset,yoffset_err = temptab[[xoffset_key,xoffset_err_key,yoffset_key,yoffset_err_key]].iloc[0]
                #     #print(xoffset,xoffset_err,yoffset,yoffset_err)

                #     # we will assume that the residuals are calculated based on Data - Model
                #     # also, the residuals are in units of mas
                #     startab[1].iloc[j] = float(startab[1].iloc[j]- xoffset/1000.0)
                #     startab[2].iloc[j] = float(startab[2].iloc[j]- yoffset/1000.0)

                #     if add_errors:
                #         startab[3].iloc[j] = np.sqrt(float(startab[3].iloc[j])**2 + (xoffset_err/1000.0)**2)
                #         startab[4].iloc[j] = np.sqrt(float(startab[4].iloc[j])**2 + (yoffset_err/1000.0)**2)
                # print(startab)
                    
                

        

def swap_points_files(swap_in,swap_out,epochs,mjd=False,metadata=True):
    '''
    Take points from one points files and directoy move them to
    another points file, selected by the epoch.

    INPUT
    -----
    swap_in - file to ADD points into
    swap_out - files to REMOVE points from
    epochs - epochs to move

    KEYWORDS
    --------
    mjd - the epoch inputs are in MJD (default: False)
    metadata - save meta data at the top (default: True)

    HISTORY
    -------
    2018-08-24 - T. Do

    '''

    # read in the points file
    tab_in = np.loadtxt(swap_in)
    tab_out = np.loadtxt(swap_out)

    tab_in = pd.read_csv(swap_in,header=None,delim_whitespace=True,float_precision='round_trip',comment='#')
    tab_out = pd.read_csv(swap_out,header=None,delim_whitespace=True,float_precision='round_trip',comment='#')    
    if len(tab_in.columns) == 5:
        tab_in.columns = ['epoch_yearDate','x','y','xerr','yerr']
        tab_out.columns = ['epoch_yearDate','x','y','xerr','yerr']
    else:

        tab_in.columns = ['epoch_yearDate','x','y','xerr','yerr','mjd']
        tab_out.columns = ['epoch_yearDate','x','y','xerr','yerr','mjd']

    if mjd:
        epoch_str = 'mjd'
    else:
        epoch_str = 'epoch_yearDate'

    ind = []    
    for dd in epochs:
        ind = ind + list(np.where(tab_out[epoch_str] == dd)[0])

    tab_in = tab_in.append(tab_out.iloc[ind]) # add points
    tab_out = tab_out.drop(ind)               # remove points from the other table
    tab_in = tab_in.sort_values(epoch_str)
    tab_out = tab_out.sort_values(epoch_str)

    

    outfile1 = os.path.splitext(swap_in)[0]+'_swap.points'
    outfile2 = os.path.splitext(swap_out)[0]+'_swap.points'


    print('output: '+outfile1)
    output1 = open(outfile1,'w')
    if metadata:
        output1.write('#'+str(datetime.datetime.now())+'\n')
        output1.write('#input : '+os.path.realpath(swap_in)+'\n')
        output1.write('#input 2 : '+os.path.realpath(swap_out)+'\n')
        outstr = ''
        for k in epochs:
            outstr = outstr+','+str(k)
        output1.write('# epochs : '+outstr+'\n')

    tab_in.to_csv(output1,index=False,header=False,sep=' ')
    output1.close()
    
    print('output: '+outfile2)
    output2 = open(outfile2,'w')
    if metadata:
        output2.write('#'+str(datetime.datetime.now())+'\n')
        output2.write('#input : '+os.path.realpath(swap_in)+'\n')
        output2.write('#input 2 : '+os.path.realpath(swap_out)+'\n')
        outstr = ''
        for k in epochs:
            outstr = outstr+','+str(k)
        output2.write('# epochs : '+outstr+'\n')

    tab_out.to_csv(output2,index=False,header=False,sep=' ')
    output2.close()
    
