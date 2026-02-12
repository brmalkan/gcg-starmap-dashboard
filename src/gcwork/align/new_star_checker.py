#!/usr/bin/env python

# New Star Checker
# ---
# Shows most closely matching stars in label.dat to target star.
# Constructs line and star name to be written to label.dat
# in case it needs to be added a new star.
# ---
# Abhimat Gautam

import numpy as np
from astropy.table import Table

def new_star_checker_example():
    align_name = 'phot_18_05_14'
    star_name='star_151'
    
    new_star_checker(star_name, align_name)

def new_star_checker(
        star_name, align_name,
        align_dir_loc='/g/ghez/align/',
        align_base_name='align_d_rms_1000_abs_t',
        source_list_dir='/Users/abhimat/software/ghezcode/source_list/',
        dist_cutoff=0.5,
    ):
    """
    Function to check if a star in align already exists in label.dat.
    Shows most closely matching stars in label.dat to `star_name`.
    Offers a line with a new star name to include in label.dat to in case it
    needs to be added into label.dat
    
    Parameters
    ----------
    star_name : str
        Name of possible new star in align
    align_name : str
        Name of the align epochs directory
    align_dir_loc : str, default='/g/ghez/align/'
        File path location of the align epochs directory
    align_base_name : str, default='align_d_rms_1000_abs_t'
        Base file name of the align files to use. These align files are
        expected to be located in the `align/` directory inside the align
        epochs directory.
    source_list_dir : str, default='/Users/abhimat/software/ghezcode/source_list/'
        Location of source_list repo, to find the label.dat file
    dist_cutoff : float, default=0.5
        Arcseconds around the target star to search for possible matches to
        already named stars in label.dat
    """
    
    # Info about star from align
    vel_file = '{0}{1}/align/{2}.vel'.format(
        align_dir_loc, align_name, align_base_name)
    vel_table = Table.read(vel_file, format='ascii', delimiter='\s')
    star_vel_row = (vel_table[np.where(vel_table['col1'] == star_name)])[0]

    b0_file = '{0}{1}/align/{2}.b0'.format(
        align_dir_loc, align_name, align_base_name)
    b0_table = Table.read(b0_file, format='ascii', delimiter='\s')
    star_b0_row = (b0_table[np.where(b0_table['col1'] == star_name)])[0]

    star_ref_year = star_b0_row['col2']

    star_mag = star_vel_row['col3']

    star_ref_xpos = -1. * star_b0_row['col3']
    star_ref_xpos_err = star_b0_row['col4']
    star_xvel = -1. * star_b0_row['col5']
    star_xvel_err = star_b0_row['col6']

    star_ref_ypos = star_b0_row['col9']
    star_ref_ypos_err = star_b0_row['col10']
    star_yvel = star_b0_row['col11']
    star_yvel_err = star_b0_row['col12']

    star_r2d = (star_ref_xpos**2. + star_ref_ypos**2.)**0.5

    # Check with label.dat file for existing stars near target star
    label_file_loc = source_list_dir + 'label.dat'

    label_table = Table.read(label_file_loc, format='ascii')

    ## Propogate position to star's ref year
    label_table['col3'] = (label_table['col7'] / 1000.) *\
        (star_ref_year - label_table['col15']) + label_table['col3']
    label_table['col4'] = (label_table['col8'] / 1000.) *\
        (star_ref_year - label_table['col15']) + label_table['col4']

    label_table['col15'] = star_ref_year
    
    ## Proposed name for star!
    proposed_name_prefix = 'S{0}-'.format(int(np.floor(star_r2d)))
    
    star_names = np.array(label_table['col1'])
    
    current_prefix_star_names = star_names[
        np.where(np.char.startswith(star_names, proposed_name_prefix))
    ]

    current_prefix_star_numbers = (np.char.replace(
        current_prefix_star_names, proposed_name_prefix, ''
    )).astype(np.int)

    proposed_name_number = np.max(current_prefix_star_numbers) + 1

    proposed_name = '{0}{1}'.format(proposed_name_prefix, proposed_name_number)


    ## Check which stars match closest to the star
    label_table_dists = ((label_table['col3'] - star_ref_xpos)**2. +
        (label_table['col4'] - star_ref_ypos)**2.) ** 0.5

    matched_label_table = label_table[
        np.where(label_table_dists < dist_cutoff)
    ]
    matched_label_table['dist'] = label_table_dists[
        np.where(label_table_dists < dist_cutoff)
    ]
    
    # Print current star and closely matching entries in label.dat
    print('Proposed new star line for label.dat')
    
    # Construct proposed line for the new star to go into label.dat
    label_line = ''
    
    label_line += f'{proposed_name:<13}'
    label_line += f'{star_mag:.1f}'
    label_line += f'{star_ref_xpos:>13.5f}'
    label_line += f'{star_ref_ypos:>11.5f}'
    label_line += f'{star_ref_xpos_err:>11.5f}'
    label_line += f'{star_ref_ypos_err:>10.5f}'
    label_line += f'{star_xvel * 1000.:>10.3f}'
    label_line += f'{star_yvel * 1000.:>9.3f}'
    label_line += f'{star_xvel_err * 1000.:>9.3f}'
    label_line += f'{star_yvel_err * 1000.:>9.3f}'
    label_line += f'{star_ref_year:>11.3f}'
    label_line += '    0'
    label_line += f'{star_r2d:>18.3f}'
    
    print(label_line)

    print('\nNearby existing stars in label.dat')
    print(matched_label_table)
