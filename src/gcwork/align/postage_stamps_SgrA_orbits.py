#!/usr/bin/env python

# Postage Stamps: Star Field Plotter
# Shows target star and nearby stars in original image
# ---
# Abhimat Gautam

from gcwork import starset

import numpy as np
from astropy.io import fits
from astropy.table import Table

from astropy.time import Time
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib.ticker import MultipleLocator

import imageio

import os
import warnings
import cPickle as pickle

from tqdm import tqdm


def postage_stamps_example():
    center_stars = ['irs16SW', 'S4-258', 'S2-36']
    align_name = 'phot_17_11_07'
    
    out_dir = '/u/abhimat/postage_stamps_test/'
    
    neighbor_rad=0.5    # Radius for neighbors (any star passing within this radius of target star in any epoch qualifies as a neighbor and is labeled on all images)
    circle_size=0.063   # 63 mas (current confusion radius)
    
    make_postage_stamps(center_stars, align_name, out_dir=out_dir)


def make_postage_stamps(center_stars, align_name,
                        align_dir_loc='/g/ghez/align/', align_base_name='align_d_rms_1000_abs_t',
                        align_points_dir='points_4_trim',
                        out_dir='~/',
                        use_res_file=False,
                        image_center_buffer=100., scale_edge=20., draw_credits=True,
                        label_offset=8., label_side=False, no_labels=False, no_circles=False, label_end_frames=False, 
                        neighbor_stars_sample = [], neighbor_rad=0.5, circle_size=0.063, out_dpi=150,
                        make_pdfs=True,
                        save_starset=False, load_starset=False, starset_pickle_loc=None,
                        calc_SgrA_pos=False, draw_orbit=False,
                        bw_image=False, AO_high_contrast=False,
                        animate_epochs=None, year_span=0.1, last_frame_time=2.0,
                        diagnostic_output=False):
    align_dir = '{0}{1}/'.format(align_dir_loc, align_name)

    # Plotting Parameters
    plate_scale = 0.00993   ## NIRC2 Plate Scale
    
    star_names = np.array([])
    epoch_dates = np.array([])

    star_Xs_Array = np.array([])
    star_Ys_Array = np.array([])
    star_mags_Array = np.array([])
    
    # Read in data from starset (or pickled starset)
    if load_starset:
        pickle_loc = out_dir
        if starset_pickle_loc is not None:
            pickle_loc = starset_pickle_loc
        
        with open('{0}/postage_stamps_starset_pickle.pkl'.format(pickle_loc), 'rb') as input_pickle:
            starset_obj = pickle.load(input_pickle)
            
            star_names = pickle.load(input_pickle)
            epoch_dates = pickle.load(input_pickle)

            star_Xs_Array = pickle.load(input_pickle)
            star_Ys_Array = pickle.load(input_pickle)
            star_mags_Array = pickle.load(input_pickle)
    else:
        # Open starset object, and load in array of positions
        warnings.simplefilter('ignore', UserWarning)
        starset_obj = starset.StarSet('{0}align/{1}'.format(align_dir, align_base_name))
        starset_obj.loadPoints('{0}/{1}/'.format(align_dir, align_points_dir))

        star_names = np.array(starset_obj.getArray('name'))
        epoch_dates = np.array(starset_obj.stars[0].years)

        star_Xs_Array = starset_obj.getArrayFromAllEpochs('pnt_x')
        star_Ys_Array = starset_obj.getArrayFromAllEpochs('pnt_y')
        star_mags_Array = starset_obj.getArrayFromAllEpochs('phot_mag')
        
        if save_starset:
            pickle_loc = out_dir
            if starset_pickle_loc is not None:
                pickle_loc = starset_pickle_loc
            
            with open('{0}/postage_stamps_starset_pickle.pkl'.format(pickle_loc), 'wb') as output_pickle:
                pickle.dump(starset_obj, output_pickle)
                
                pickle.dump(star_names, output_pickle)
                pickle.dump(epoch_dates, output_pickle)

                pickle.dump(star_Xs_Array, output_pickle)
                pickle.dump(star_Ys_Array, output_pickle)
                pickle.dump(star_mags_Array, output_pickle)
    
    
    # Read in orbit file, if draw orbits is set to True
    if draw_orbit:
        orbit_file = out_dir + 'S0-2_astro_model.txt'
        orbit_table = Table.read(orbit_file, format='ascii', header_start=0)
        
        ## Convert x and y coordinates to pixels
        orbit_table['x'] = orbit_table['x'] / plate_scale
        orbit_table['y'] = orbit_table['y'] / plate_scale
    
    for center_star in center_stars:
        # Center target star
        # center_star = 'S3-38'
        file_name = center_star
    
        if not os.path.exists(out_dir + center_star):
            os.makedirs(out_dir + center_star)
        
        if make_pdfs:
            if not os.path.exists(out_dir + center_star + '/pdf'):
                os.makedirs(out_dir + center_star + '/pdf')
    
        if not os.path.exists(out_dir + center_star + '/png'):
            os.makedirs(out_dir + center_star + '/png')
    
    
        # Plot Stars
        ## Determine current star's positions
        if len(neighbor_stars_sample) == 0:
            center_star_index = np.where(star_names == center_star)[0][0]
            center_star_xs = star_Xs_Array[:, center_star_index]
            center_star_ys = star_Ys_Array[:, center_star_index]
            center_star_mags = star_mags_Array[:, center_star_index]
    
            ## Find stars matching search parameters in any epoch
            diff_xs = star_Xs_Array - center_star_xs[:, None]
            diff_ys = star_Ys_Array - center_star_ys[:, None]
            diff_pos = np.hypot(diff_xs, diff_ys)
    
            where_matches = np.where(np.logical_and(diff_pos <= neighbor_rad, star_mags_Array > 0.))

            ## Unique matching stars
            unique_matches = np.unique(where_matches[1], return_counts=True)

            ## Construct list
            for cur_star in ((np.array(star_names))[unique_matches[0]]):
                if cur_star not in neighbor_stars_sample:
                    neighbor_stars_sample.append(cur_star)
        
        if diagnostic_output:
            print('Neighbor Stars: {0}'.format(neighbor_stars_sample))

        # Background image info
        image_epochs = []
        epoch_dirs = []
        
        epochsInfo_table = Table.read('{0}{1}/scripts/epochsInfo.txt'.format(align_dir_loc, align_name), format='ascii')
        
        ## Remove epochs not getting aligned
        idx = np.where(epochsInfo_table['doAlign'] == 1)
        epochsInfo_table = epochsInfo_table[idx]
        
        ## Read out relevant information from epochsInfo file
        image_epochs = epochsInfo_table['epoch']
        epoch_dirs = epochsInfo_table['directory']
        
        image_colnums = (range(len(image_epochs) + 1))[1:]

        for cur_epoch_index in tqdm(range(len(image_epochs))):
            image_epoch = image_epochs[cur_epoch_index]
            image_colnum = image_colnums[cur_epoch_index]
            epoch_dir = epoch_dirs[cur_epoch_index]

            if diagnostic_output:
                print(image_epoch)


            ## Make conversion of align names to starlist name
            align_to_starlist = {}
            with open("{0}{1}/align/{2}.name".format(align_dir_loc, align_name, align_base_name), "r") as name_file:
                for cur_line in name_file:
                    split_line = cur_line.split()

                    align_star_name = split_line[0]
                    starlist_star_name = split_line[image_colnum]

                    if not starlist_star_name.startswith('-'):
                        align_to_starlist[align_star_name] = starlist_star_name

            ## Get position of stars in starlist
            starlist_x_pos = {}
            starlist_y_pos = {}

            with open("{0}{1}/lis/mag{2}_rms.lis".format(align_dir_loc, align_name, image_epoch), "r") as starlist:
                for cur_line in starlist:
                    split_line = cur_line.split()

                    star_name = split_line[0]

                    starlist_x_pos[star_name] = float(split_line[3])
                    starlist_y_pos[star_name] = float(split_line[4])
            
            ## Get positions of neighbor stars from .miss file if not in align
            miss_table = Table.read("{0}{1}/align/{2}.miss{3}".format(align_dir_loc, align_name, align_base_name, cur_epoch_index), 
                                    format = 'ascii')
            
            for cur_star in neighbor_stars_sample:
                if cur_star != 'SgrA' and cur_star not in align_to_starlist:
                    if diagnostic_output:
                        print('Using miss table for {0} in epoch {1}'.format(cur_star, image_colnum))
                    
                    miss_index = np.where(miss_table['col1'] == cur_star)[0][0]
                    
                    miss_x = miss_table['col4'][miss_index]
                    miss_y = miss_table['col5'][miss_index]
                    
                    align_to_starlist[cur_star] = cur_star + '_miss'
                    
                    starlist_x_pos[cur_star + '_miss'] = miss_x
                    starlist_y_pos[cur_star + '_miss'] = miss_y
                    
                    
            ## Calculate SgrA position from S0-2, if necessary
            if ('SgrA' not in align_to_starlist) and calc_SgrA_pos:
                neighbor_star = 'S0-2'
                neighbor_index = np.where(star_names == neighbor_star)[0][0]
                neighbor_xs = star_Xs_Array[:, neighbor_index]
                neighbor_ys = star_Ys_Array[:, neighbor_index]
                
                ### Check if align detects the neighbor star in this epoch
                align_detected = True
                # if neighbor_star not in align_to_starlist:
                #     align_detected = False
                # elif (neighbor_xs[cur_epoch_index] == -1000.0) or (neighbor_ys[cur_epoch_index] == -1000.0):
                #     align_detected = False
                if (neighbor_xs[cur_epoch_index] == -1000.0) or (neighbor_ys[cur_epoch_index] == -1000.0):
                    align_detected = False
                
                ### Switch to backup stars if neighbor star not detected
                poss_neighbor_stars = ['S0-1', 'S0-17', 'S0-6']
                neighbor_ind = 0
                while not align_detected and (neighbor_ind) < len(poss_neighbor_stars):
                    neighbor_star = poss_neighbor_stars[neighbor_ind]
                    
                    neighbor_index = np.where(star_names == neighbor_star)[0][0]
                    neighbor_xs = star_Xs_Array[:, neighbor_index]
                    neighbor_ys = star_Ys_Array[:, neighbor_index]
                    
                    #### Check if align detects new neighbor star in this epoch
                    align_detected = True
                    if (neighbor_xs[cur_epoch_index] == -1000.0) or (neighbor_ys[cur_epoch_index] == -1000.0):
                        align_detected = False
                    
                    neighbor_ind += 1

                if neighbor_star not in align_to_starlist:
                    continue

                align_to_starlist['SgrA'] = 'SgrA_calc'
                
                x_offset = neighbor_xs[cur_epoch_index] / plate_scale
                y_offset = neighbor_ys[cur_epoch_index] / plate_scale
                
                starlist_x_pos['SgrA_calc'] = starlist_x_pos[align_to_starlist[neighbor_star]] - x_offset
                starlist_y_pos['SgrA_calc'] = starlist_y_pos[align_to_starlist[neighbor_star]] - y_offset
                
                if diagnostic_output:
                    print('Neighbor Star: {0}'.format(neighbor_star))
                    print('Neighbor Star Position (abs) ({0:.3f}, {1:.3f})'.format(neighbor_xs[cur_epoch_index], neighbor_ys[cur_epoch_index]))
                    print('Neighbor Star Position (img) ({0:.3f}, {1:.3f})'.format(starlist_x_pos[align_to_starlist[neighbor_star]], starlist_y_pos[align_to_starlist[neighbor_star]]))
                    print('Calculated Sgr A* Position: ({0:.3f}, {1:.3f})'.format(starlist_x_pos['SgrA_calc'], starlist_y_pos['SgrA_calc']))

            # ## Get position of Sgr A* (if detected) from named starlist
            # sgrA_x_pos = 0.
            # sgrA_y_pos = 0.
            #
            # with open("/g/ghez/data/gc/{0}/combo/starfinder/mag{0}_kp_rms_named.lis".format(image_epoch), "r") as starlist:
            #     for cur_line in starlist:
            #         split_line = cur_line.split()
            #
            #         star_name = split_line[0]
            #
            #         if star_name != 'SgrA':
            #             continue
            #         else:
            #             sgrA_x_pos = float(split_line[3])
            #             sgrA_y_pos = float(split_line[4])
            #             break

            ## Construct lists of coordinates to plot
            ### Variable Stars Sample
            neighbor_stars_plot_x = np.array([])
            neighbor_stars_plot_y = np.array([])

            center_star_plot_x = 0.
            center_star_plot_y = 0.

            for cur_star in neighbor_stars_sample:
                if cur_star not in align_to_starlist:
                    continue

                neighbor_stars_plot_x = np.append(neighbor_stars_plot_x, starlist_x_pos[align_to_starlist[cur_star]])
                neighbor_stars_plot_y = np.append(neighbor_stars_plot_y, starlist_y_pos[align_to_starlist[cur_star]])

                if cur_star == center_star:
                    center_star_plot_x = starlist_x_pos[align_to_starlist[cur_star]]
                    center_star_plot_y = starlist_y_pos[align_to_starlist[cur_star]]

            ## Drawing image plot
            # ### Font Nerdery
            # plt.rc('font', family='serif')
            # plt.rc('font', serif='Computer Modern Roman')
            # plt.rc('text', usetex=True)

            fig = plt.figure(figsize=(8., 8.))

            ax1 = fig.add_subplot(1, 1, 1)


            ### Import image from FITS file
            fits_file = epoch_dir + "/combo/mag" + image_epoch + ".fits"
            if use_res_file:
                fits_file = epoch_dir + "/combo/mag" + image_epoch + "_res.fits"

            if 'holography' in epoch_dir:
                fits_file = epoch_dir + "/final/" + image_epoch + "_holo.fits"


            warnings.simplefilter('ignore', UserWarning)
            with fits.open(fits_file) as hdulist:
                image_data = hdulist[0].data

            # print(image_data[1100, 1100])

            ## Normalize to a stable calibrator star
            norm_star = 'S1-1'
            norm_value = 4500.

            norm_star_x = int(starlist_x_pos[align_to_starlist[norm_star]])
            norm_star_y = int(starlist_y_pos[align_to_starlist[norm_star]])

            norm_rad = 3

            pre_norm = np.mean(image_data[norm_star_y-norm_rad:norm_star_y+norm_rad, norm_star_x-norm_rad:norm_star_x+norm_rad])

            norm_factor = pre_norm/norm_value
            image_data = image_data/norm_factor

            post_norm = np.mean(image_data[norm_star_y-norm_rad:norm_star_y+norm_rad, norm_star_x-norm_rad:norm_star_x+norm_rad])

            # Default AO image scaling
            im_add = 0.
            im_floor = 100.

            im_add = 2000.
            im_add = 5000.
            im_floor = 1200.
            im_ceil = 1.e6
            im_ceil = 1.e4
            im_ceil = 9.e3

            im_mult = 400.
            im_invert = 1.
            
            if AO_high_contrast:
                im_add = 0.
                
                im_floor = 0.
                im_ceil = 1.25e3
                
                im_mult = 200.
                im_invert = -1.

            if 'holography' in epoch_dir:
                im_add = 20.

                im_add = 2000.
                im_floor = 0.
                im_ceil = 1.e6
                im_ceil = 5.e5
                im_ceil = 5.e3
                
                im_mult = 200.
                im_invert = -1.
            else:
                if use_res_file:
                    im_add = 0.
                    im_floor = 1.
                    im_mult = 40.
                    im_invert = -1.


            ## Put in image floor
            image_data = image_data + im_add
            image_data[np.where(image_data <= im_floor)] = im_floor
            image_data[np.where(image_data >= im_ceil)] = im_ceil

            image_data *= im_mult
            
            im_cmap = plt.get_cmap('hot')
            if bw_image:
                im_cmap = plt.get_cmap('gray')
            
            if 'holography' in epoch_dir:
                
                ax1.imshow(im_invert * np.sqrt(image_data), cmap=im_cmap, interpolation='nearest')
            else:
                ax1.imshow(im_invert * np.sqrt(image_data), cmap=im_cmap, interpolation='nearest')

            # ax1.imshow(np.log(image_data), cmap='hot')
            # ax1.imshow(-1. * np.log(image_data), cmap='gray')
            # ax1.imshow(np.sqrt(image_data), cmap='PuBuGn')
            # ax1.imshow(image_data, cmap='inferno')
            # ax1.set_xlabel(r"$x$")
            # ax1.set_ylabel(r"$y$")


            ## Stars
            if not no_circles:
                for star_index in range(len(neighbor_stars_plot_x))[1:]:
                    # ax1.add_artist(plt.Circle((neighbor_stars_plot_x[star_index], neighbor_stars_plot_y[star_index]), radius=(circle_size * 1./plate_scale), linestyle='-', edgecolor='white', linewidth=1.5, fill=False))
                    ax1.add_artist(plt.Circle((neighbor_stars_plot_x[star_index], neighbor_stars_plot_y[star_index]), radius=(circle_size * 1./plate_scale), linestyle='-', edgecolor=None, fc='white', fill=True))

                # ax1.add_artist(plt.Circle((center_star_plot_x, center_star_plot_y), radius=(circle_size * 1./plate_scale), linestyle='-', edgecolor='darkgreen', linewidth=1.5, fill=False))
            
            ## Draw orbit
            if draw_orbit:
                draw_x = (orbit_table[orbit_table['date'] < epoch_dates[cur_epoch_index] - 0.])['x']
                draw_y = (orbit_table[orbit_table['date'] < epoch_dates[cur_epoch_index] - 0.])['y']
                
                ax1.plot(draw_x + center_star_plot_x, draw_y + center_star_plot_y, color='white', linewidth=1.5, alpha=0.7)
            
            ### Axes Limits

            #### Make stamp at avg. spot if star not detected in this epoch
            if center_star_plot_x == 0.:
                center_star_plot_x = np.mean(neighbor_stars_plot_x[np.where(neighbor_stars_plot_x > 0.)])
                center_star_plot_y = np.mean(neighbor_stars_plot_y[np.where(neighbor_stars_plot_y > 0.)])

            if np.isnan(center_star_plot_x) or np.isnan(center_star_plot_y):
                continue

            image_x_bounds = [center_star_plot_x - image_center_buffer, center_star_plot_x + image_center_buffer]
            image_y_bounds = [center_star_plot_y - image_center_buffer, center_star_plot_y + image_center_buffer]

            ax1.set_xlim(image_x_bounds)
            ax1.set_ylim(image_y_bounds)
            ax1.get_xaxis().set_visible(False)
            ax1.get_yaxis().set_visible(False)


            ### Size Scale
            scale_xEnd = image_x_bounds[1] - scale_edge
            scale_y = image_y_bounds[0] + scale_edge
            
            obs_time = Time(epoch_dates[cur_epoch_index], format='jyear')
            date_str = obs_time.datetime.strftime('%d %b %Y')
            
            scale_text_size = 'medium'
            if scale_edge <= 10:
                scale_text_size = 'small'

            ax1.plot([scale_xEnd - 0.25/plate_scale, scale_xEnd], [scale_y, scale_y], '-', color='white', linewidth=4)
            ax1.plot([scale_xEnd - 0.25/plate_scale, scale_xEnd], [scale_y, scale_y], '-', color='black', linewidth=2)
            ax1.text((scale_xEnd - 0.25/plate_scale + scale_xEnd)/2., 2. + scale_y, '1/4 arcsec', ha='center', va='bottom', size=scale_text_size, color='white', bbox = dict(boxstyle = 'round,pad=0.25', edgecolor='none', facecolor = 'black', alpha = 0.25))
            
            ax1.text((scale_xEnd - 0.25/plate_scale + scale_xEnd)/2., scale_y - 2., '{0:.3f}'.format(epoch_dates[cur_epoch_index], date_str), ha='center', va='top', size=scale_text_size, color='white', bbox = dict(boxstyle = 'round,pad=0.25', edgecolor='none', facecolor = 'black', alpha = 0.25))
            
            # if cur_epoch_index == 0 or cur_epoch_index == range(len(image_epochs))[-1]:
            #     ax1.text((scale_xEnd - 0.5/plate_scale + scale_xEnd)/2., scale_y - 5., '{0:.3f}\n{1}'.format(epoch_dates[cur_epoch_index], date_str), ha='center', va='top', size='medium', color='white', bbox = dict(boxstyle = 'round,pad=0.15', edgecolor='none', facecolor = 'black', alpha = 0.25))
            # else:
            #     ax1.text((scale_xEnd - 0.5/plate_scale + scale_xEnd)/2., scale_y - 5., '{0}'.format(np.floor(epoch_dates[cur_epoch_index]), date_str), ha='center', va='top', size='medium', color='white', bbox = dict(boxstyle = 'round,pad=0.15', edgecolor='none', facecolor = 'black', alpha = 0.25))
            
            
            ### Credits
            if draw_credits:                
                credits_x = image_x_bounds[0] + 60.
                credits_y = image_y_bounds[0] + scale_edge
            
                ax1.text(credits_x, credits_y, 'Keck / UCLA\nGalactic Center Group', ha='center', va='center', size='large', color='white', bbox = dict(boxstyle = 'round,pad=0.25', edgecolor='none', facecolor = 'black', alpha = 0.25))
            
            
            ## Star Labels
            if not no_labels:
                for cur_star in neighbor_stars_sample:
                    try:
                        if label_side:
                            ax1.text(label_offset + starlist_x_pos[align_to_starlist[cur_star]],
                                starlist_y_pos[align_to_starlist[cur_star]], cur_star.replace('irs', 'IRS '),
                                ha='left', va='center', size='xx-small', color='white',
                                bbox = dict(boxstyle = 'round,pad=0.2', edgecolor='none',
                                facecolor = 'black', alpha = 0.25))  # .replace('_', '\_')
                        else:
                            ax1.text(starlist_x_pos[align_to_starlist[cur_star]], label_offset +
                                starlist_y_pos[align_to_starlist[cur_star]], cur_star.replace('irs', 'IRS '), 
                                ha='center', va='bottom', size='xx-small', color='white',
                                bbox = dict(boxstyle = 'round,pad=0.2', edgecolor='none',
                                facecolor = 'black', alpha = 0.25))  # .replace('_', '\_')
                    except:
                        continue
                        # print(cur_star + ' not found')
            
            ## If the last epoch, label S0-2 and Sgr A*
            if label_end_frames and (cur_epoch_index == 0 or cur_epoch_index == range(len(image_epochs))[-2]):
                x_coord = starlist_x_pos[align_to_starlist['S0-2']]
                y_coord = starlist_y_pos[align_to_starlist['S0-2']]
                
                if cur_epoch_index == 0:
                    label_x_offset = 20
                    label_y_offset = 20
                    label_color='white'
                else:
                    label_x_offset = 20
                    label_y_offset = -20
                    label_color='white'
                
                ax1.annotate('S0-2', xy=(x_coord, y_coord),
                             xytext=(x_coord + label_x_offset, y_coord + label_y_offset),
                             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color=label_color, shrinkB=20),
                             fontsize='large', color=label_color)
                
                x_coord = starlist_x_pos[align_to_starlist['SgrA']]
                y_coord = starlist_y_pos[align_to_starlist['SgrA']]
                
                if cur_epoch_index == 0:
                    label_x_offset = -17
                    label_y_offset = -18
                    label_color='white'
                else:
                    label_x_offset = -20
                    label_y_offset = 20
                    label_color='white'
                
                if cur_epoch_index == 0:
                    ax1.annotate('Sgr A*', xy=(x_coord, y_coord),
                                 xytext=(x_coord + label_x_offset, y_coord + label_y_offset),
                                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color=label_color, shrinkB=7),
                                 fontsize='large', color=label_color)
            
            fig.tight_layout()

            if make_pdfs:
                fig.savefig('{0}/{1}/pdf/PostageStamps_{1}_{2}_{3}.pdf'.format(out_dir, file_name, align_name, image_colnum))

            fig.savefig('{0}/{1}/png/PostageStamps_{1}_{2}_{3}.png'.format(out_dir, file_name, align_name, image_colnum), dpi=out_dpi)
            plt.close(fig)
    
        # Generate gif animation
        animate_epoch_nums = image_colnums
        
        if animate_epochs is not None:
            animate_epoch_nums = animate_epochs
        
        frame_times = []
        
        for image_colnum_index in tqdm(range(len(animate_epoch_nums))):
            frame_time = last_frame_time
            
            if image_colnum_index != 0 and image_colnum_index+1 < len(animate_epoch_nums):
                frame_time = (epoch_dates[animate_epoch_nums[image_colnum_index + 1] - 1] - epoch_dates[animate_epoch_nums[image_colnum_index] - 1]) * year_span
            
            frame_times.append(frame_time)
            
        
        with imageio.get_writer('{0}/{1}/PostageStamps_{1}_{2}.gif'.format(out_dir, file_name, align_name), mode='I', duration=frame_times) as gif_writer:
            for image_colnum_index in tqdm(range(len(animate_epoch_nums))):
                image_colnum = animate_epoch_nums[image_colnum_index]
                
                cur_image = imageio.imread('{0}/{1}/png/PostageStamps_{1}_{2}_{3}.png'.format(out_dir, file_name, align_name, image_colnum))
                
                gif_writer.append_data(cur_image)
    


def make_postage_stamps_combo(unlabeled_dir, labeled_dir, orbit_draw_dir, align_name, align_dir_loc='/g/ghez/align/', align_base_name='align_d_rms_1000_abs_t', align_points_dir='points_4_trim', out_dir='~/',
save_starset=False, load_starset=False, starset_pickle_loc=None,
animate_epochs=None, year_span=0.1, last_frame_time=2.0,
diagnostic_output=False):
    file_name = 'SgrA'
    
    # Plotting Parameters
    plate_scale = 0.00993   ## NIRC2 Plate Scale
    
    star_names = np.array([])
    epoch_dates = np.array([])

    star_Xs_Array = np.array([])
    star_Ys_Array = np.array([])
    star_mags_Array = np.array([])
    
    # Read in data from starset (or pickled starset)
    if load_starset:
        pickle_loc = out_dir
        if starset_pickle_loc is not None:
            pickle_loc = starset_pickle_loc
        
        with open('{0}/postage_stamps_starset_pickle.pkl'.format(pickle_loc), 'rb') as input_pickle:
            starset_obj = pickle.load(input_pickle)
            
            star_names = pickle.load(input_pickle)
            epoch_dates = pickle.load(input_pickle)

            star_Xs_Array = pickle.load(input_pickle)
            star_Ys_Array = pickle.load(input_pickle)
            star_mags_Array = pickle.load(input_pickle)
    else:
        # Open starset object, and load in array of positions
        warnings.simplefilter('ignore', UserWarning)
        starset_obj = starset.StarSet('{0}align/{1}'.format(align_dir, align_base_name))
        starset_obj.loadPoints('{0}/{1}/'.format(align_dir, align_points_dir))

        star_names = np.array(starset_obj.getArray('name'))
        epoch_dates = np.array(starset_obj.stars[0].years)

        star_Xs_Array = starset_obj.getArrayFromAllEpochs('pnt_x')
        star_Ys_Array = starset_obj.getArrayFromAllEpochs('pnt_y')
        star_mags_Array = starset_obj.getArrayFromAllEpochs('phot_mag')
        
        if save_starset:
            pickle_loc = out_dir
            if starset_pickle_loc is not None:
                pickle_loc = starset_pickle_loc
            
            with open('{0}/postage_stamps_starset_pickle.pkl'.format(pickle_loc), 'wb') as output_pickle:
                pickle.dump(starset_obj, output_pickle)
                
                pickle.dump(star_names, output_pickle)
                pickle.dump(epoch_dates, output_pickle)

                pickle.dump(star_Xs_Array, output_pickle)
                pickle.dump(star_Ys_Array, output_pickle)
                pickle.dump(star_mags_Array, output_pickle)
    
    # Generate gif animation
    animate_epoch_nums = []
    
    if animate_epochs is not None:
        animate_epoch_nums = animate_epochs
    
    frame_times = []
    
    for labeled_group_index in ['unlabeled', 'labeled', 'orbit']:
        for image_colnum_index in tqdm(range(len(animate_epoch_nums))):
            frame_time = last_frame_time
            
            if (not (image_colnum_index == 0 and labeled_group_index == 'labeled')) and image_colnum_index+1 < len(animate_epoch_nums):
                frame_time = (epoch_dates[animate_epoch_nums[image_colnum_index + 1] - 1] - epoch_dates[animate_epoch_nums[image_colnum_index] - 1]) * year_span
        
            frame_times.append(frame_time)
        
    
    with imageio.get_writer('{0}/PostageStamps_combo_{1}_{2}.gif'.format(out_dir, file_name, align_name), mode='I', duration=frame_times) as gif_writer:
        for labeled_group_index in ['unlabeled', 'labeled', 'orbit']:
            for image_colnum_index in tqdm(range(len(animate_epoch_nums))):
                image_colnum = animate_epoch_nums[image_colnum_index]
                
                labeled_dir_name = labeled_dir
                if labeled_group_index == 'unlabeled':
                    labeled_dir_name = unlabeled_dir
                if labeled_group_index == 'orbit':
                    labeled_dir_name = orbit_draw_dir
                
                cur_image = imageio.imread('{0}/{1}/png/PostageStamps_{2}_{3}_{4}.png'.format(out_dir, labeled_dir_name, file_name, align_name, image_colnum))
                
                gif_writer.append_data(cur_image)
