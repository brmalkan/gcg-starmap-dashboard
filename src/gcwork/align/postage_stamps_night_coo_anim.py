#!/usr/bin/env python

# Postage Stamps: Star Field Plotter
# Shows target star and nearby stars in original image
# ---
# Abhimat Gautam

import numpy as np
from astropy.io import fits
from astropy.table import Table

from astropy.time import Time
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib.ticker import MultipleLocator
from astropy.visualization import simple_norm

import glob

import imageio
import moviepy.editor as mp

import os
import warnings

from tqdm import tqdm


def postage_stamps_example():
    center_stars = ['irs16SW', 'S4-258', 'S2-36']
    align_name = 'phot_17_11_07'
    
    out_dir = '/u/abhimat/postage_stamps_test/'
    
    neighbor_rad=0.5    # Radius for neighbors (any star passing within this radius of target star in any epoch qualifies as a neighbor and is labeled on all images)
    circle_size=0.063   # 63 mas (current confusion radius)
    
    make_postage_stamps(center_stars, align_name, out_dir=out_dir)


def make_postage_stamps(night_name, filt, coo_offset_x, coo_offset_y,
                        dr_loc = '/g/ghez/data/dr/dr1/',
                        out_dir='./', star_name=None,
                        use_res_file=False,
                        image_center_buffer=200., scale_edge=40.,
                        draw_credits=True,
                        # label_offset=8., label_side=False, no_labels=False, no_circles=False, label_end_frames=False,
                        # neighbor_stars_sample = [], neighbor_rad=0.5, circle_size=0.063, out_dpi=150,
                        make_pdfs=True,
                        # save_starset=False, load_starset=False, starset_pickle_loc=None,
                        # calc_SgrA_pos=False, draw_orbit=False,
                        bw_image=True, AO_high_contrast=False,
                        skip_plots=False,
                        anim_strehl_cut=0.1,
                        animate_epochs=None, day_span=90.0, last_frame_time=2.0,
                        diagnostic_output=False):
    # Plotting Parameters
    plate_scale = 0.00993   ## NIRC2 Plate Scale
    
    # Make output directories
    star_dir = ''
    if star_name != None:
        star_dir = f'_{star_name}'
    
    file_name = f'coo_postage_stamps_{night_name}{star_dir}'
    
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    if make_pdfs:
        if not os.path.exists(out_dir + '/pdf'):
            os.makedirs(out_dir + '/pdf')
    
    if not os.path.exists(out_dir + '/png'):
        os.makedirs(out_dir + '/png')

    # Determine list of images
    night_filt_clean_dir = f'{dr_loc}/clean/{night_name}/{filt}/'
    
    strehl_table = Table.read(night_filt_clean_dir + 'strehl_source.txt',
                              format='ascii')
    
    images_mjd = np.empty(len(strehl_table))
    
    # image_list = glob.glob(night_filt_clean_dir + 'c*.fits')
    # coo_list = glob.glob(night_filt_clean_dir + 'c*.coo')
    
    # image_list.sort()
    # coo_list.sort()
    
    # Go through each image
    if not skip_plots:
        for (cur_image_index, cur_row) in tqdm(enumerate(strehl_table)):
            cur_coo_file = cur_row['Filename'].replace('.fits', '.coo')
            cur_image_file = cur_row['Filename']
            cur_image_strehl = cur_row['Strehl']
        
            # Read in coo coordinates
            coo_x = 0.
            coo_y = 0.
        
            with open(night_filt_clean_dir + cur_coo_file, 'r') as coo_file:
                coords = coo_file.readline().split()
                coo_x = float(coords[0])
                coo_y = float(coords[1])
        
            print(coo_x, coo_y)
        
            center_star_plot_x = coo_x + coo_offset_x
            center_star_plot_y = coo_y + coo_offset_y
        
            # Drawing image plot
            # ### Font Nerdery
            # plt.rc('font', family='serif')
            # plt.rc('font', serif='Computer Modern Roman')
            # plt.rc('text', usetex=True)
        
            fig = plt.figure(figsize=(6., 6.))

            ax1 = fig.add_subplot(1, 1, 1)

            # Import image and header info from FITS file
            warnings.simplefilter('ignore', UserWarning)
            with fits.open(night_filt_clean_dir + cur_image_file) as hdulist:
                image_data = hdulist[0].data
                image_header = hdulist[0].header
        
            image_filename = image_header['FILENAME']
            image_mjd = image_header['MJD-OBS']
            images_mjd[cur_image_index] = image_mjd
        
            print(image_data[100, 100])
            print(image_filename)
            print(image_mjd)
        
            ## Normalize to the COO star
            norm_value = 2_000_000.     # / (cur_image_strehl ** 2)

            norm_star_x = int(coo_x)
            norm_star_y = int(coo_y)

            norm_rad = 10

            pre_norm = np.sum(image_data[
                                    norm_star_y-norm_rad:norm_star_y+norm_rad,
                                    norm_star_x-norm_rad:norm_star_x+norm_rad])

            print(f'Pre-norm value: {pre_norm}')

            norm_factor = pre_norm/norm_value
            image_data = image_data/norm_factor

            post_norm = np.sum(image_data[
                                    norm_star_y-norm_rad:norm_star_y+norm_rad,
                                    norm_star_x-norm_rad:norm_star_x+norm_rad])

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
            
            if filt == 'kp':
                im_add = 100
                im_floor = 50
                im_ceil = 2.5e3
            
                im_mult = 1000.0
            
            if filt == 'h':
                im_add = 100
                im_floor = 50
                im_ceil = 2.5e3
            
                im_mult = 1000.0
            
            if filt == 'lp':
                im_add = 100
                im_floor = 50
                im_ceil = 2.5e3
            
                im_mult = 1000.0
        

            if AO_high_contrast:
                im_add = 0.

                im_floor = 0.
                im_ceil = 1.25e3

                im_mult = 200.
                im_invert = -1.
        
            # Put in image floor
            image_data = image_data + im_add
            image_data[np.where(image_data <= im_floor)] = im_floor
            image_data[np.where(image_data >= im_ceil)] = im_ceil

            image_data *= im_mult
        
        
            norm = simple_norm(image_data, 'sqrt')
        
            im_cmap = plt.get_cmap('hot')
            if bw_image:
                im_cmap = plt.get_cmap('gray')        
        
            ax1.imshow(image_data, cmap=im_cmap, norm=norm)
        
            # if 'holography' in epoch_dir:
            #     im_add = 20.
            #
            #     im_add = 2000.
            #     im_floor = 0.
            #     im_ceil = 1.e6
            #     im_ceil = 5.e5
            #     im_ceil = 5.e3
            #
            #     im_mult = 200.
            #     im_invert = -1.
            # else:
            #     if use_res_file:
            #         im_add = 0.
            #         im_floor = 1.
            #         im_mult = 40.
            #         im_invert = -1.
            #
        
            # if 'holography' in epoch_dir:
            #
            #     ax1.imshow(im_invert * np.sqrt(image_data), cmap=im_cmap, interpolation='nearest')
            # else:
            #     ax1.imshow(im_invert * np.sqrt(image_data), cmap=im_cmap, interpolation='nearest')
        
            # ax1.imshow(im_invert * np.sqrt(image_data),
            #            cmap=im_cmap, interpolation='nearest')
        
            # ax1.imshow(np.log(image_data), cmap='hot')
            # ax1.imshow(-1. * np.log(image_data), cmap='gray')
            # ax1.imshow(np.sqrt(image_data), cmap='PuBuGn')
            # ax1.imshow(image_data, cmap='inferno')
            # ax1.set_xlabel(r"$x$")
            # ax1.set_ylabel(r"$y$")

            # Axes Limits

            # #### Make stamp at avg. spot if star not detected in this epoch
            # if center_star_plot_x == 0.:
            #     center_star_plot_x = np.mean(neighbor_stars_plot_x[np.where(neighbor_stars_plot_x > 0.)])
            #     center_star_plot_y = np.mean(neighbor_stars_plot_y[np.where(neighbor_stars_plot_y > 0.)])
            #
            # if np.isnan(center_star_plot_x) or np.isnan(center_star_plot_y):
            #     continue

            image_x_bounds = [center_star_plot_x - image_center_buffer,
                              center_star_plot_x + image_center_buffer]
            image_y_bounds = [center_star_plot_y - image_center_buffer,
                              center_star_plot_y + image_center_buffer]

            ax1.set_xlim(image_x_bounds)
            ax1.set_ylim(image_y_bounds)
            ax1.get_xaxis().set_visible(False)
            ax1.get_yaxis().set_visible(False)


            ### Size Scale
            scale_xEnd = image_x_bounds[1] - scale_edge
            scale_y = image_y_bounds[0] + scale_edge

            obs_time = Time(image_mjd, format='mjd')
            time_str = obs_time.datetime.strftime('%Y-%m-%d %H:%M')

            scale_text_size = 'medium'
            if scale_edge <= 10:
                scale_text_size = 'small'

            ax1.plot([scale_xEnd - 0.25/plate_scale, scale_xEnd],
                     [scale_y, scale_y], '-', color='white', linewidth=4)
            ax1.plot([scale_xEnd - 0.25/plate_scale, scale_xEnd],
                     [scale_y, scale_y], '-', color='black', linewidth=2)
            ax1.text((scale_xEnd - 0.25/plate_scale + scale_xEnd)/2.,
                     8. + scale_y,
                     '1/4 arcsec',
                     ha='center', va='bottom',
                     size=scale_text_size, color='white',
                     bbox = dict(boxstyle = 'round,pad=0.25', edgecolor='none',
                                 facecolor = 'black', alpha = 0.25))

            ax1.text((scale_xEnd - 0.25/plate_scale + scale_xEnd)/2., scale_y - 8.,
                     '{0}'.format(time_str),
                     ha='center', va='top', size=scale_text_size, color='white',
                     bbox = dict(boxstyle = 'round,pad=0.25', edgecolor='none',
                                 facecolor = 'black', alpha = 0.25))

            # if cur_epoch_index == 0 or cur_epoch_index == range(len(image_epochs))[-1]:
            #     ax1.text((scale_xEnd - 0.5/plate_scale + scale_xEnd)/2., scale_y - 5., '{0:.3f}\n{1}'.format(epoch_dates[cur_epoch_index], date_str), ha='center', va='top', size='medium', color='white', bbox = dict(boxstyle = 'round,pad=0.15', edgecolor='none', facecolor = 'black', alpha = 0.25))
            # else:
            #     ax1.text((scale_xEnd - 0.5/plate_scale + scale_xEnd)/2., scale_y - 5., '{0}'.format(np.floor(epoch_dates[cur_epoch_index]), date_str), ha='center', va='top', size='medium', color='white', bbox = dict(boxstyle = 'round,pad=0.15', edgecolor='none', facecolor = 'black', alpha = 0.25))


            ### Credits
            if draw_credits:
                credits_x = image_x_bounds[0] + 80.0
                credits_y = image_y_bounds[0] + scale_edge

                ax1.text(credits_x, credits_y,
                         'Keck / UCLA\nGalactic Center Group',
                         ha='center', va='center',
                         size='large', color='white',
                         bbox = dict(boxstyle = 'round,pad=0.25', edgecolor='none',
                                     facecolor = 'black', alpha = 0.25),
                        )
        
        
            # ## Star Labels
            # if not no_labels:
            #     for cur_star in neighbor_stars_sample:
            #         try:
            #             if label_side:
            #                 ax1.text(label_offset + starlist_x_pos[align_to_starlist[cur_star]],
            #                     starlist_y_pos[align_to_starlist[cur_star]], cur_star.replace('irs', 'IRS '),
            #                     ha='left', va='center', size='xx-small', color='white',
            #                     bbox = dict(boxstyle = 'round,pad=0.2', edgecolor='none',
            #                     facecolor = 'black', alpha = 0.25))  # .replace('_', '\_')
            #             else:
            #                 ax1.text(starlist_x_pos[align_to_starlist[cur_star]], label_offset +
            #                     starlist_y_pos[align_to_starlist[cur_star]], cur_star.replace('irs', 'IRS '),
            #                     ha='center', va='bottom', size='xx-small', color='white',
            #                     bbox = dict(boxstyle = 'round,pad=0.2', edgecolor='none',
            #                     facecolor = 'black', alpha = 0.25))  # .replace('_', '\_')
            #         except:
            #             continue
            #             # print(cur_star + ' not found')

            # ## If the last epoch, label S0-2 and Sgr A*
            # if label_end_frames and (cur_epoch_index == 0 or cur_epoch_index == range(len(image_epochs))[-2]):
            #     x_coord = starlist_x_pos[align_to_starlist['S0-2']]
            #     y_coord = starlist_y_pos[align_to_starlist['S0-2']]
            #
            #     if cur_epoch_index == 0:
            #         label_x_offset = 20
            #         label_y_offset = 20
            #         label_color='white'
            #     else:
            #         label_x_offset = 20
            #         label_y_offset = -20
            #         label_color='white'
            #
            #     ax1.annotate('S0-2', xy=(x_coord, y_coord),
            #                  xytext=(x_coord + label_x_offset, y_coord + label_y_offset),
            #                  arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color=label_color, shrinkB=20),
            #                  fontsize='large', color=label_color)
            #
            #     x_coord = starlist_x_pos[align_to_starlist['SgrA']]
            #     y_coord = starlist_y_pos[align_to_starlist['SgrA']]
            #
            #     if cur_epoch_index == 0:
            #         label_x_offset = -17
            #         label_y_offset = -18
            #         label_color='white'
            #     else:
            #         label_x_offset = -20
            #         label_y_offset = 20
            #         label_color='white'
            #
            #     if cur_epoch_index == 0:
            #         ax1.annotate('Sgr A*', xy=(x_coord, y_coord),
            #                      xytext=(x_coord + label_x_offset, y_coord + label_y_offset),
            #                      arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color=label_color, shrinkB=7),
            #                      fontsize='large', color=label_color)

            fig.tight_layout()

            if make_pdfs:
                fig.savefig('{0}/pdf/PostageStamps_{1}.pdf'.format(
                                out_dir, image_filename.replace('.fits', '')))

            fig.savefig('{0}/png/PostageStamps_{1}.png'.format(
                            out_dir, image_filename.replace('.fits', '')),
                        dpi=67,)
            plt.close(fig)

    # Generate gif animation
    # Perform the cut on which to generate animation
    animation_cut = np.where(strehl_table['Strehl'] >= anim_strehl_cut)
    
    if animate_epochs != None:
        animation_cut = animate_epochs
    
    animate_strehl_table = strehl_table[animation_cut]
    animate_images_mjd = images_mjd[animation_cut]
    
    total_anim_frames = len(animate_strehl_table)
    
    # Calculate the time for each frame
    frame_times = []
    
    for (row_index, image_mjd) in enumerate(animate_images_mjd):
        frame_time = last_frame_time
        
        if (row_index+1 < total_anim_frames):
            frame_time = (animate_images_mjd[row_index + 1] - 
                          animate_images_mjd[row_index]) * day_span
        
        frame_times.append(frame_time)

    # Construct final gif
    with imageio.get_writer('{0}/{1}_{2}.gif'.format(out_dir, file_name, filt),
                            mode='I', duration=frame_times) as gif_writer:
        for strehl_table_row in animate_strehl_table:
            im_name = strehl_table_row['Filename'].replace('c', 'n').replace('.fits', '')
            
            cur_image = imageio.imread(
                            '{0}/png/PostageStamps_{1}.png'.format(
                                out_dir, im_name)
                            )
            
            gif_writer.append_data(cur_image)
    
    # Make mp4 from gif
    mesh_clip = mp.VideoFileClip('{0}/{1}_{2}.gif'.format(out_dir, file_name, filt))
    mesh_clip = mesh_clip.loop(mesh_clip.duration)
    mesh_clip.write_videofile('{0}/{1}_{2}.mp4'.format(out_dir, file_name, filt))


