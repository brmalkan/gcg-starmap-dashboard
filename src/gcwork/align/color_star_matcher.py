#!/usr/bin/env python

# Color Star Matcher
# ---
# Function to carry out matching and naming stars with very different colors
# across aligns
# ---
# Abhimat Gautam

import numpy as np
from astropy.table import Table
from flystar import starlists, align
from flystar.starlists import StarList
from flystar import transforms
# from flystar import analysis, plots
from glob import glob
import shutil
import os
import copy

def read_b0_table(align_dir, align_base_name='align_d_rms_1000_abs_t'):
    b0_file = '{0}/align/{1}.b0'.format(align_dir, align_base_name)
    b0_table = Table.read(b0_file, format='ascii', delimiter='\s')
    
    b0_table.rename_column('col1', 'star')
    b0_table.rename_column('col2', 't0')
    
    b0_table.rename_column('col3', 'x')
    b0_table.rename_column('col4', 'x_unc')
    b0_table.rename_column('col5', 'x_vel')
    b0_table.rename_column('col6', 'x_vel_unc')
    
    b0_table.rename_column('col9', 'y')
    b0_table.rename_column('col10', 'y_unc')
    b0_table.rename_column('col11', 'y_vel')
    b0_table.rename_column('col12', 'y_vel_unc')
    
    return b0_table

def read_vel_table(align_dir, align_base_name='align_d_rms_1000_abs_t'):
    vel_file = '{0}/align/{1}.vel'.format(align_dir, align_base_name)
    vel_table = Table.read(vel_file, format='ascii', delimiter='\s')
    
    vel_table.rename_column('col1', 'star')
    vel_table.rename_column('col2', 'num_epochs')
    
    vel_table.rename_column('col3', 'mag')
    
    return vel_table


def make_align_ref_table(b0_table, vel_table):
    # Make new table, and then add relevent columns for ref table
    align_starlist_table = Table()
    
    align_starlist_table.add_column(b0_table['star'], name='name')
    
    align_starlist_table.add_column(b0_table['x'], name='x0')
    align_starlist_table.add_column(b0_table['y'], name='y0')
    align_starlist_table.add_column(b0_table['x_unc'], name='x0e')
    align_starlist_table.add_column(b0_table['y_unc'], name='y0e')
    
    align_starlist_table.add_column(b0_table['x_vel'], name='vx')
    align_starlist_table.add_column(b0_table['y_vel'], name='vy')
    align_starlist_table.add_column(b0_table['x_vel_unc'], name='vxe')
    align_starlist_table.add_column(b0_table['y_vel_unc'], name='vye')
    
    align_starlist_table.add_column(b0_table['t0'], name='t0')
    
    align_starlist_table.add_column(vel_table['mag'], name='m')
    
    # Take out sources that don't have names assigned
    align_starlist_table = align_starlist_table[np.where(
        np.char.find(align_starlist_table['name'], 'star_') == -1)]
    
    
    return align_starlist_table



def make_align_starlist(b0_table, vel_table):
    # Make new table
    align_starlist_table = Table()
    
    align_starlist_table.add_column(b0_table['star'], name='name')
    
    # Propagate all positions to the same time
    t0 = b0_table['t0'][0]
    
    propagated_x = b0_table['x'] + ((t0 - b0_table['t0']) * b0_table['x_vel'])
    propagated_x_unc = b0_table['x_unc'] + ((t0 - b0_table['t0']) * 
                                            b0_table['x_vel_unc'])
    propagated_y = b0_table['y'] + ((t0 - b0_table['t0']) * b0_table['y_vel'])
    propagated_y_unc = b0_table['y_unc'] + ((t0 - b0_table['t0']) * 
                                            b0_table['y_vel_unc'])    
    
    
    
    # Add position, mag, time columns to the table
    # align_starlist_table.add_column(propagated_x, name='x')
    # align_starlist_table.add_column(propagated_y, name='y')
    # align_starlist_table.add_column(propagated_x_unc, name='xe')
    # align_starlist_table.add_column(propagated_y_unc, name='ye')
    
    align_starlist_table.add_column(b0_table['x'], name='x')
    align_starlist_table.add_column(b0_table['x_unc'], name='xe')
    align_starlist_table.add_column(b0_table['y'], name='y')
    align_starlist_table.add_column(b0_table['y_unc'], name='ye')
    
    align_starlist_table.add_column(vel_table['mag'], name='m')
    
    # align_starlist_table.add_column(np.full(len(b0_table), t0), name='t')
    align_starlist_table.add_column(b0_table['t0'], name='t')
    
    align_starlist = StarList.from_table(align_starlist_table)
    
    # Make sure there are unique names in starlist, otherwise align breaks
    
    unique_names, unique_inds = np.unique(align_starlist['name'], return_index=True)
    align_starlist = align_starlist[unique_inds]
    
    return align_starlist
    

def color_star_matcher(ref_align_dir, match_align_dir,
                       align_base_name='align_d_rms_1000_abs_t',
                       iters=2, dr_tol=[0.04, 0.04], dm_tol=[5.0, 4.0],
                       init_guess_mode='name'):
    """
    Run flystar to match unnamed sources in an align to those in another
    (reference) align
    
    Parameters
    ----------
    ref_align_dir : str
    match_align_dir : str
    align_base_name : str, default='align_d_rms_1000_abs_t'
    iters : int, default=2
    dr_tol : list of float, default=[0.04, 0.04]
    dm_tol : list of float, default=[5.0, 4.0]
    
    Returns
    -------
    matches_table : astropy.Table
        Astropy Table object, listing all matched sources with reference align
        names, current align names, and magnitude in both aligns.
    """
    
    # Read in positions and velocity from both aligns
    # Reference align directory (i.e. align being used as name reference)
    ref_b0_table = read_b0_table(ref_align_dir,
                                 align_base_name=align_base_name)
    
    ref_vel_table = read_vel_table(ref_align_dir,
                                   align_base_name=align_base_name)
    
    # Match align directory (i.e. align that needs new names assigned)
    match_b0_table = read_b0_table(match_align_dir,
                                   align_base_name=align_base_name)
    
    match_vel_table = read_vel_table(match_align_dir,
                                     align_base_name=align_base_name)
    
    # Convert input tables into tables and StarLists that Flystar needs
    ref_table = make_align_ref_table(ref_b0_table, ref_vel_table)
    
    match_starlist = make_align_starlist(match_b0_table, match_vel_table)
    
    # Set up and run align
    ref_table.write('ref_table.txt', format='ascii.fixed_width')
    match_starlist.write('match_starlist.txt', format='ascii.fixed_width')
    
    msc = align.MosaicToRef(ref_table, [match_starlist], iters=iters,
                            dr_tol=dr_tol, dm_tol=dm_tol,
                            trans_class=transforms.PolyTransform,
                            trans_args=[{'order': 1}, {'order': 1}], 
                            use_vel=True,
                            use_ref_new=False,
                            update_ref_orig=False, 
                            mag_trans=True,
                            init_guess_mode=init_guess_mode, verbose=False)
    
    msc.fit()
    align_tab = msc.ref_table
    print(align_tab)
    print(align_tab.colnames)
    
    # Construct matches table
    matches_table = Table()
    
    matches_table.add_column(align_tab['name'], name='ref_align_name')
    matches_table.add_column(align_tab['name_in_list'][:,0], name='cur_align_name')
    matches_table.add_column(align_tab['m0'], name='ref_align_mag')
    matches_table.add_column(align_tab['m_orig'][:,0], name='cur_align_mag')
    
    # Filter out stars that were not matched
    matches_table = matches_table[np.where(
        np.char.find(matches_table['ref_align_name'], '0_') == -1)]
    
    # Filter out rows in ref_align_name that are already in cur_align_name
    matches_table = matches_table[np.where(
        np.isin(matches_table['ref_align_name'],
                matches_table['cur_align_name'],
                invert=True))]
    
    # Filter out rows where cur_align_name is already a name assigned by align
    # (assuming that align has correctly identified and named these sources;
    #  otherwise can missname already named sources, e.g. S0-2 -> Sgr A*)
    matches_table = matches_table[np.where(
        np.char.find(matches_table['cur_align_name'], 'star_') != -1)]
    
    
    print(matches_table)
    
    matches_table.write('matches_table.txt', format='ascii.fixed_width',
                        overwrite=True)
    
    return matches_table


def process_matched_points_file(orig_file_loc, new_file_loc, matches_table,
                                num_header_lines=1):
    """
    Process a file by replacing star name in the beginning of each line with
    its matched name.
    """
    
    out_str = ''
    
    # Open existing file
    with open(orig_file_loc, 'r') as inp_file:
        # Keep number of header lines
        inp_file_lines = inp_file.readlines()
        
        for cur_line in inp_file_lines[:num_header_lines]:
            out_str = cur_line
        
        # Process each line
        for cur_line in inp_file_lines[num_header_lines:]:
            if cur_line == '':
                continue
            
            # Split out cur star
            split_line = (cur_line.strip()).split()
            cur_star = split_line[0]
            
            match_star = (matches_table['ref_align_name'])[
                            np.where(matches_table['cur_align_name'] == cur_star)]
            
            out_star = ''
            
            if len(match_star) == 0:
                out_star = cur_star
            else:
                len_cur_star = len(cur_star)
                out_star = (match_star[0]).ljust(len_cur_star)
            
            # Create new line and append
            out_line = cur_line.replace(cur_star, out_star, 1)
            out_str = out_str + out_line
        
    # Write out new, processed file
    with open(new_file_loc, 'w') as dest_file:
        dest_file.write(out_str)
    
    return new_file_loc


def create_matched_points(input_points_dir = 'points_3_c',
                          output_points_dir = 'points_3_c_name_match_Kp',
                          align_root = '/g/ghez/align/phot_19_07_2_H'):
    """
    Create matched points and polyfit files
    """
    
    src_loc = '{0}/{1}/'.format(align_root, input_points_dir)
    dest_loc = '{0}/{1}/'.format(align_root, output_points_dir)
    
    # First copy every .points and .phot file in input points dir
    all_points = glob(src_loc + '*.points'.format(src_loc))
    all_phots = glob(src_loc + '*.phot'.format(src_loc))
    
    for points_file in all_points:
        shutil.copy(points_file, dest_loc)
    for phot_file in all_phots:
        shutil.copy(phot_file, dest_loc)
    
    # Load in matches table
    matches_table_loc = '{0}/{1}/matches_table.txt'.format(
                            align_root, output_points_dir)
    
    matches_table = Table.read(matches_table_loc, format='ascii.fixed_width')
    
    # Rename .points and .phot files
    for cur_star_index in range(len(matches_table)):
        ref_name = matches_table['ref_align_name'][cur_star_index]
        cur_name = matches_table['cur_align_name'][cur_star_index]
        
        ref_points_name = '{0}/{1}/{2}.points'.format(
                                align_root, output_points_dir, ref_name)
        cur_points_name = '{0}/{1}/{2}.points'.format(
                                align_root, output_points_dir, cur_name)
        
        ref_phot_name = '{0}/{1}/{2}.phot'.format(
                                align_root, output_points_dir, ref_name)
        cur_phot_name = '{0}/{1}/{2}.phot'.format(
                                align_root, output_points_dir, cur_name)
        
        if os.path.exists(ref_points_name):
            # If points file already exists with ref name
            continue
        elif os.path.exists(cur_points_name):
            # If points file exists with cur name
            os.rename(cur_points_name, ref_points_name)
            os.rename(cur_phot_name, ref_phot_name)
        else:
            # If neither points file exists
            continue
    
    # Copy and rename sources in txt files if they exist
    # Check and replace all star names in confusing sources list
    if os.path.exists(src_loc + 'confusingSources.txt'):
        out_str = ''
        
        # Open existing file
        with open(src_loc + 'confusingSources.txt', 'r') as inp_file:
            # Process each line
            for cur_line in inp_file:
                out_stars = []
                
                # Split out all stars in each line and rename
                split_stars = cur_line.split()
                for star in split_stars:
                    match_star = (matches_table['ref_align_name'])[
                                    np.where(matches_table['cur_align_name'] == star)]
                    
                    if len(match_star) == 0:
                        out_stars.append(star)
                    else:
                        out_stars.append(match_star[0])
                
                # Create new line and append
                out_line = ' '.join(out_stars)
                out_str = out_str + out_line + '\n'
            
        # Write out new, processed file
        with open(dest_loc + 'confusingSources.txt', 'w') as dest_file:
            dest_file.write(out_str)
    
    if os.path.exists(src_loc + 'epochsRemoved.txt'):
        out_str = ''
        
        # Open existing file
        with open(src_loc + 'epochsRemoved.txt', 'r') as inp_file:
            # Process each line
            for cur_line in inp_file:
                out_star_and_dates = []
                
                # Split out cur star
                split_line = cur_line.split()
                cur_star = split_line[0]
                dates = split_line[1:]
                
                match_star = (matches_table['ref_align_name'])[
                                np.where(matches_table['cur_align_name'] == cur_star)]
                
                out_star = ''
                
                if len(match_star) == 0:
                    out_star = cur_star
                else:
                    out_star = match_star[0]
                
                # Create new line and append
                out_line = ' '.join([out_star] + dates)
                out_str = out_str + out_line + '\n'
            
        # Write out new, processed file
        with open(dest_loc + 'epochsRemoved.txt', 'w') as dest_file:
            dest_file.write(out_str)
    
    
    # Copy and rename sources in the polyfit
    input_polyfit_dir = input_points_dir.replace('points', 'polyfit')
    output_polyfit_dir = output_points_dir.replace('points', 'polyfit')
    
    if not os.path.exists(align_root + '/' + output_polyfit_dir):
        os.mkdir(align_root + '/' + output_polyfit_dir)
    
    polyfit_files = ['linear.txt', 'accel.txt',
                     'fit.lt0', 'fit.linearFormal',
                     'fit.t0', 'fit.accelFormal', 'fit.accelPolar']
    polyfit_files_header_lines = [1, 1,
                                  1, 1,
                                  1, 1, 2]
    
    for (cur_file,
         cur_file_head_lines) in zip(polyfit_files,
                                     polyfit_files_header_lines):
        process_matched_points_file('{0}/{1}/{2}'.format(align_root,
                                        input_polyfit_dir, cur_file),
                                    '{0}/{1}/{2}'.format(align_root,
                                        output_polyfit_dir, cur_file),
                                    matches_table,
                                    num_header_lines = cur_file_head_lines)

def create_matched_align_files(
    input_base_align_name = 'align_d_rms_1000_abs',
    output_base_align_name = 'align_d_rms_1000_abs_name_match_Kp',
    matches_table_loc = './matches_table.txt',
    align_root = '/g/ghez/align/phot_19_07_2_H'):
    """
    Create files in align directory that have matched star names
    
    Parameters
    ----------
    input_base_align_name : str, default='align_d_rms_1000_abs'
        The root names of the align files that are serving as the source
    output_base_align_name : str, default='align_d_rms_1000_abs_name_match_Kp'
        The root names of the align files that are serving as the destination of
        the align files re-written with the matched names
    matches_table_loc : str, default='./matches_table.txt',
        The location of the txt table with the star name matches
    align_root : str, default='/g/ghez/align/phot_19_07_2_H'
        The root path of the align epochs run
    """
    
    # Roots for input and output align files
    src_align_file_root = '{0}/align/{1}'.format(align_root, input_base_align_name)
    dest_align_file_root = '{0}/align/{1}'.format(align_root, output_base_align_name)
    
    # First pull out all the matching align files with the given root
    all_align_files = glob(src_align_file_root + '.*')
    all_align_fileEnds = copy.deepcopy(all_align_files)
    
    for cur_ind in range(len(all_align_fileEnds)):
        all_align_fileEnds[cur_ind] =\
            (all_align_fileEnds[cur_ind])[len(src_align_file_root):]
    
    # Load in matches table
    matches_table = Table.read(matches_table_loc, format='ascii.fixed_width')
    
    # Copy each align file and rename sources if necessary
    align_files_header_lines =\
        {'.b0': 0,
         '.err': 0,
         '.mag': 0,
         '.miss': 0,
         '.name': 0,
         '.param': 0,
         '.origpos': 0,
         '.pos': 0,
         '.vel': 0}
    
    for cur_fileEnd in all_align_fileEnds:
        if cur_fileEnd.startswith('.miss'):
            # File is .miss file
            process_matched_points_file(src_align_file_root + cur_fileEnd,
                                        dest_align_file_root + cur_fileEnd,
                                        matches_table,
                                        num_header_lines = align_files_header_lines['.miss'])
        elif cur_fileEnd in align_files_header_lines:
            # File is another align file that needs to have names changed
            process_matched_points_file(src_align_file_root + cur_fileEnd,
                                        dest_align_file_root + cur_fileEnd,
                                        matches_table,
                                        num_header_lines = align_files_header_lines[cur_fileEnd])
        else:
            # File does not need to have star names changed; just copy
            shutil.copy(src_align_file_root + cur_fileEnd,
                        dest_align_file_root + cur_fileEnd)
    
    