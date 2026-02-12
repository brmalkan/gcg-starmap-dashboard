#!/usr/bin/env python

# Postage Stamps: Star Field Plotter
# Shows target star and nearby stars in original image
# ---
# Abhimat Gautam

from gcwork import starset
import numpy as np
from astropy.io import fits
from astropy.table import Table
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib.ticker import MultipleLocator
import imageio
import os
import warnings
import pickle


def postage_stamps_example():
    center_stars = ['irs16SW', 'S4-258', 'S2-36']
    align_name = 'phot_17_11_07'
    
    out_dir = '/u/abhimat/postage_stamps_test/'
    
    neighbor_rad=0.5    # Radius for neighbors (any star passing within this radius of target star in any epoch qualifies as a neighbor and is labeled on all images)
    circle_size=0.063   # 63 mas (current confusion radius)
    
    make_postage_stamps(center_stars, align_name, out_dir=out_dir)


def make_postage_stamps(center_stars, align_name,
        align_dir_loc='/g/ghez/align/', align_base_name='align_d_rms_1000_abs_t',
        align_points_dir='points_3_c',
        out_dir='~/',
        use_res_file=False,
        image_center_buffer=100.,
        scale_edge=80.,
        label_offset=8., label_side=False,
        neighbor_rad=0.5,
        circle_size=0.063,
        make_pdfs=True,
        save_starset=False, load_starset=False,
        calc_SgrA_pos=False,
        no_labels=False,
        animate_epochs=None,
        bw_image=False, AO_high_contrast=False, black_sky=False,
        print_diagnostics=False):
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
        with open('{0}/postage_stamps_starset_pickle.pkl'.format(out_dir), 'rb') as input_pickle:
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
            with open('{0}/postage_stamps_starset_pickle.pkl'.format(out_dir), 'wb') as output_pickle:
                pickle.dump(starset_obj, output_pickle)
                
                pickle.dump(star_names, output_pickle)
                pickle.dump(epoch_dates, output_pickle)

                pickle.dump(star_Xs_Array, output_pickle)
                pickle.dump(star_Ys_Array, output_pickle)
                pickle.dump(star_mags_Array, output_pickle)
    
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
        neighbor_stars_sample = []
    
        center_star_index = np.where(star_names == center_star)[0][0]
        center_star_xs = star_Xs_Array[:, center_star_index]
        center_star_ys = star_Ys_Array[:, center_star_index]
        center_star_mags = star_mags_Array[:, center_star_index]
        
        if print_diagnostics:
            print(center_star + ' align info')
            print(center_star_index)
            print(center_star_xs)
            print(center_star_ys)
            print(center_star_mags)
            
        
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

        # print(len(neighbor_stars_sample))

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

        for cur_epoch_index in range(len(image_epochs)):
            image_epoch = image_epochs[cur_epoch_index]
            image_colnum = image_colnums[cur_epoch_index]
            epoch_dir = epoch_dirs[cur_epoch_index]
            # print(image_epoch)
    
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
            
            ## Calculate SgrA position from S0-2, if necessary
            if ('SgrA' not in align_to_starlist) and calc_SgrA_pos:
                neighbor_star = 'S0-2'
                
                poss_neighbor_stars = ['S0-1', 'S0-17', 'S0-6']
                neighbor_ind = 0
                while neighbor_star not in align_to_starlist and (neighbor_ind) < len(poss_neighbor_stars):
                    neighbor_star = poss_neighbor_stars[neighbor_ind]
                    neigbhor_ind += 1
                
                if neighbor_star not in align_to_starlist:
                    continue
                    
                align_to_starlist['SgrA'] = 'SgrA_calc'
                
                neighbor_index = np.where(star_names == neighbor_star)[0][0]
                neighbor_xs = star_Xs_Array[:, neighbor_index]
                neighbor_ys = star_Ys_Array[:, neighbor_index]
                
                x_offset = neighbor_xs[cur_epoch_index] / plate_scale
                y_offset = neighbor_ys[cur_epoch_index] / plate_scale
                
                starlist_x_pos['SgrA_calc'] = starlist_x_pos[align_to_starlist[neighbor_star]] - x_offset
                starlist_y_pos['SgrA_calc'] = starlist_y_pos[align_to_starlist[neighbor_star]] - y_offset
            
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

            fig = plt.figure(figsize=(5, 5))

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
            norm_star = 'S1-34'
            norm_value = 4500.
            
            if norm_star in align_to_starlist:
                norm_star_x = int(starlist_x_pos[align_to_starlist[norm_star]])
                norm_star_y = int(starlist_y_pos[align_to_starlist[norm_star]])

                norm_rad = 3

                pre_norm = np.mean(image_data[norm_star_y-norm_rad:norm_star_y+norm_rad, norm_star_x-norm_rad:norm_star_x+norm_rad])

                norm_factor = pre_norm/norm_value
                image_data = image_data/norm_factor
            
                post_norm = np.mean(image_data[norm_star_y-norm_rad:norm_star_y+norm_rad, norm_star_x-norm_rad:norm_star_x+norm_rad])
            else:
                print(f'Norm star, {norm_star}, not identifed in epoch {image_epoch}.')
                print('Not normalizing image scale')
            
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
            im_invert = -1.
            
            if black_sky:
                im_invert = 1.
            
            if AO_high_contrast:
                im_add = 5000.
                im_floor = 1200.
                im_ceil = 9.e3
                
                im_mult = 400.
                im_invert = 0.
                
                im_add = 0.
                im_floor = 60.
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
                im_invert = 1.
            else:
                if use_res_file:
                    im_add = 0.
                    im_floor = 0.
                    im_mult = 1.
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
            elif use_res_file:
                ax1.imshow(im_invert * image_data, cmap=im_cmap, interpolation='nearest')
            else:
                ax1.imshow(im_invert * np.sqrt(image_data), cmap=im_cmap, interpolation='nearest')
            
            # ax1.imshow(np.log(image_data), cmap='hot')
            # ax1.imshow(-1. * np.log(image_data), cmap='gray')
            # ax1.imshow(np.sqrt(image_data), cmap='PuBuGn')
            # ax1.imshow(image_data, cmap='inferno')
            # ax1.set_xlabel(r"$x$")
            # ax1.set_ylabel(r"$y$")
    
    
            ## Stars
            if not no_labels:
                for star_index in range(len(neighbor_stars_plot_x)):
                    ax1.add_artist(plt.Circle((neighbor_stars_plot_x[star_index], neighbor_stars_plot_y[star_index]), radius=(circle_size * 1./plate_scale), linestyle='-', edgecolor='red', linewidth=1.5, fill=False))
                
                ax1.add_artist(plt.Circle((center_star_plot_x, center_star_plot_y), radius=(circle_size * 1./plate_scale), linestyle='-', edgecolor='green', linewidth=1.5, fill=False))
    
            ### Axes Limits
    
            #### Make stamp at avg. spot if star not detected in this epoch
            if center_star_plot_x == 0.:
                center_star_plot_x = np.mean(neighbor_stars_plot_x[np.where(neighbor_stars_plot_x > 0.)])
                center_star_plot_y = np.mean(neighbor_stars_plot_y[np.where(neighbor_stars_plot_y > 0.)])
            
            if np.isnan(center_star_plot_x) or np.isnan(center_star_plot_y):
                print("No center star")
                print("Epoch: {0}".format(image_epoch))
                print("Neighbor stars: {0}\n".format(neighbor_stars_sample))
                plt.close(fig)
                continue
            
                    
            image_x_bounds = [center_star_plot_x - image_center_buffer, center_star_plot_x + image_center_buffer]
            image_y_bounds = [center_star_plot_y - image_center_buffer, center_star_plot_y + image_center_buffer]

            ax1.set_xlim(image_x_bounds)
            ax1.set_ylim(image_y_bounds)
            ax1.get_xaxis().set_visible(False)
            ax1.get_yaxis().set_visible(False)


            ### Size Scale
            scale_xEnd = center_star_plot_x + scale_edge
            scale_y = center_star_plot_y - scale_edge

            ax1.plot([scale_xEnd - 0.5/plate_scale, scale_xEnd], [scale_y, scale_y], '-', color='white', linewidth=4)
            ax1.plot([scale_xEnd - 0.5/plate_scale, scale_xEnd], [scale_y, scale_y], '-', color='black', linewidth=2)
            ax1.text((scale_xEnd - 0.5/plate_scale + scale_xEnd)/2., 3. + scale_y, '1/2 arcsec', ha='center', va='bottom', size='medium', color='black')
            ax1.text((scale_xEnd - 0.5/plate_scale + scale_xEnd)/2., scale_y - 5., '{0:.3f}'.format(epoch_dates[cur_epoch_index]), ha='center', va='top', size='medium', color='black', bbox = dict(boxstyle = 'round,pad=0.1', edgecolor='none', facecolor = 'white', alpha = 0.25))

            ## Star Labels
            if not no_labels:
                for cur_star in neighbor_stars_sample:
                    try:
                        if label_side:
                            ax1.text(label_offset + starlist_x_pos[align_to_starlist[cur_star]], starlist_y_pos[align_to_starlist[cur_star]], cur_star.replace('irs', 'IRS '), ha='left', va='center', size='x-small', bbox = dict(boxstyle = 'round,pad=0.1', edgecolor='none', facecolor = 'white', alpha = 0.25))  # .replace('_', '\_')
                        else:
                            ax1.text(starlist_x_pos[align_to_starlist[cur_star]], label_offset + starlist_y_pos[align_to_starlist[cur_star]], cur_star.replace('irs', 'IRS '), ha='center', va='bottom', size='x-small', bbox = dict(boxstyle = 'round,pad=0.1', edgecolor='none', facecolor = 'white', alpha = 0.25))  # .replace('_', '\_')
                    except:
                        continue
                        # print(cur_star + ' not found')

            fig.tight_layout()

            if make_pdfs:
                fig.savefig('{0}/{1}/pdf/PostageStamps_{1}_{2}_{3}.pdf'.format(out_dir, file_name, align_name, image_colnum))
            
            fig.savefig('{0}/{1}/png/PostageStamps_{1}_{2}_{3}.png'.format(out_dir, file_name, align_name, image_colnum))
            plt.close(fig)
    
        # Generate gif animation
        animate_epoch_nums = image_colnums
        
        if animate_epochs is not None:
            animate_epoch_nums = animate_epochs
        
        with imageio.get_writer('{0}/{1}/PostageStamps_{1}_{2}.gif'.format(out_dir, file_name, align_name), mode='I') as gif_writer:
            for image_colnum in animate_epoch_nums:
                cur_image = imageio.imread('{0}/{1}/png/PostageStamps_{1}_{2}_{3}.png'.format(out_dir, file_name, align_name, image_colnum))
                gif_writer.append_data(cur_image)



def calc_rotated_coords(
        orig_coords, epoch_angle,
        orig_image_center, rotated_image_center
    ):
    """
    Given a set of original image coordinates, the angle of rotation,
    and the angle of rotation in both the original and rotated image,
    return corresponding image coordinates in the rotated image.
    
    Parameters
    ----------
    """
    
    orig_coords_origin_rem = orig_coords - orig_image_center
    
    rotate_angle_rad = np.deg2rad(epoch_angle)
    
    rotated_coords_origin_rem = np.array([
        orig_coords_origin_rem[0] * np.cos(rotate_angle_rad) +\
        orig_coords_origin_rem[1] * np.sin(rotate_angle_rad),
        -orig_coords_origin_rem[0] * np.sin(rotate_angle_rad) +\
        orig_coords_origin_rem[1] * np.cos(rotate_angle_rad),
    ])
    
    rotated_coords = rotated_coords_origin_rem + rotated_image_center
    
    return rotated_coords
    

def make_postage_stamps_dr(
        center_stars, align_name,
        find_center_stars=None,
        dr_loc='/g/ghez/data/dr/dr1/',
        align_dir_loc='/g/ghez/align/',
        align_base_name='align_d_rms_1000_abs_t',
        align_points_dir='points_3_c',
        out_dir='~/',
        use_res_file=False,
        image_center_buffer=100.,
        scale_edge=80.,
        no_labels=False,
        label_offset=8., label_side=False,
        stamp_size=(5, 5),
        neighbor_rad=0.5,
        circle_size=0.063,
        make_pdfs=True,
        png_dpi=100,
        save_starset=False, load_starset=False,
        calc_SgrA_pos=False,
        prefer_calc_SgrA_pos=False,
        animate_epochs=None, year_span=0.1, last_frame_time=2.0,
        bw_image=False, AO_high_contrast=False, black_sky=True,
        specify_image_normalization={},
        override_stf_version=None,
        print_diagnostics=False,
    ):
    """Postage stamps code, with support for images in data release
    
    Parameters
    ----------
    center_stars : lis[str]
        List of star names to make postage stamps centered around
    find_center_stars : lis[str], default=None
        If specified, use these stars to find neighbor stars around instead of
        the center star
    align_name : str
        Name of the align epochs directory, used for the matching and naming
        of stars
    dr_loc : str, default='/g/ghez/data/dr/dr1/'
        File path location of the data release to use for the original images.
    align_dir_loc : str, default='/g/ghez/align/'
        File path location of the align epochs directory
    align_base_name : str, default='align_d_rms_1000_abs_t'
        Base file name of the align files to use. These align files are
        expected to be located in the `align/` directory inside the align
        epochs directory.
    align_points_dir : str, default='points_3_c'
        The points directory in align epochs to read in
    out_dir : str, default='~/'
        The output directory for plots and animations. If not specified,
        outputs are written at the user's home directory.
    use_res_file : bool, default=False
        Substitue in the starfinder residual image file instead of the combo
        epoch file. The residual file is taken from the data release by reading
        in the starfinder version from the align epoch run's
        epochsinfo.txt file.
    image_center_buffer : int, default=100
        The size of the postage stamp cutout around each center star. Variable
        sets the distance from the image edge to the center star location, in
        units of image pixels.
    scale_edge : float, default=80.
        Distance from the image center to draw the image scale legend. Distance
        is measured from the image center in units of image pixels.
    no_labels : bool, default=False
        Set to true for skipping drawing labels on stars
    label_offset : float, default=8.
        Offset of the label from the location of the star. Distance is measured
        in units of image pixels.
    label_side : bool, default=False
        Specify if star labels should be drawn to the side of the star. Default
        is False, where labels will be drawn above each star.
    neighbor_rad : float, default=0.5
        Any stars passing within the neighbor radius of the center star are
        labelled on the postage stamp. The neighbor radius is specified in units
        of arcseconds.
    circle_size : float, default=0.063
        The size of the circle drawn around detected stars, specified in units
        of arcseconds. The default is 63 mas, indicating the confusion radius at
        Kp in Keck NIRC2.
    make_pdfs : bool, default=True
        Output PDF postage stamps in addition to PNG files
    save_starset : bool, default=False
        Whether or not to create a starset pickle file. Creating a starset
        pickle file for the align epochs used will allow subsequent runs of the
        postage stamp code to be faster.
    load_starset : bool, default=False
        Whether or not to use a starset pickle file created in a previous run of
        the postage stamp code. Using a starset pickle file will speed up
        running the postage stamp code.
    specify_image_normalization : dic[str: float], default = {}
        Dictionary of epochs to manually specify image normalization.
        Specify if there are particular epochs where image normalization
        computation is not sufficient (e.g., shallow images).
    """
    
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
        with open('{0}/postage_stamps_starset_pickle.pkl'.format(out_dir), 'rb') as input_pickle:
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
            with open('{0}/postage_stamps_starset_pickle.pkl'.format(out_dir),
                      'wb') as output_pickle:
                pickle.dump(starset_obj, output_pickle)
                
                pickle.dump(star_names, output_pickle)
                pickle.dump(epoch_dates, output_pickle)

                pickle.dump(star_Xs_Array, output_pickle)
                pickle.dump(star_Ys_Array, output_pickle)
                pickle.dump(star_mags_Array, output_pickle)
    
    
    # Go through each sample star
    for center_star_index, center_star in enumerate(center_stars):
        # If animate epochs are not specified, then add each epoch
        # where image is successfully drawn
        add_animate_epochs = False
        if animate_epochs == None:
            add_animate_epochs = True
            animate_epochs = []
        
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
        neighbor_stars_sample = []
        
        if find_center_stars is None:
            center_star_index = np.where(star_names == center_star)[0][0]
            center_star_xs = star_Xs_Array[:, center_star_index]
            center_star_ys = star_Ys_Array[:, center_star_index]
            center_star_mags = star_mags_Array[:, center_star_index]
        else:
            find_star = find_center_stars[center_star_index]
            center_star_index = np.where(star_names == find_star)[0][0]
            center_star_xs = star_Xs_Array[:, center_star_index]
            center_star_ys = star_Ys_Array[:, center_star_index]
            center_star_mags = star_mags_Array[:, center_star_index]
        
        
        if print_diagnostics:
            print(center_star + ' align info')
            print(center_star_index)
            print(center_star_xs)
            print(center_star_ys)
            print(center_star_mags)
            
        
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

        # print(len(neighbor_stars_sample))
        
        # Construct a conversion from align name to background image position
        epochsInfo_table = Table.read(
            '{0}{1}/scripts/epochsInfo.txt'.format(
                align_dir_loc, align_name),
            format='ascii',
        )
        
        if print_diagnostics:
            print('{0}{1}/scripts/epochsInfo.txt'.format(align_dir_loc, align_name))
            print(epochsInfo_table)
        
        ## Remove epochs not getting aligned
        align_filt = np.where(epochsInfo_table['doAlign'] == 1)
        epochsInfo_table = epochsInfo_table[align_filt]
    
        ## Read out relevant information from epochsInfo file
        image_epochs = epochsInfo_table['epoch']
        epoch_dirs = epochsInfo_table['directory']
        epoch_stf_versions = epochsInfo_table['version']
        epoch_passbands = epochsInfo_table['wave']
        epoch_angles = epochsInfo_table['angle']
    
        image_colnums = range(1, len(image_epochs) + 1)
        
        if print_diagnostics:
            print(epochsInfo_table)
        
        for cur_epoch_index in range(len(image_epochs)):
            image_epoch = image_epochs[cur_epoch_index]
            
            filt_suffix_index = image_epoch.rfind('_')
            
            image_filt = image_epoch[filt_suffix_index + 1:]
            epoch_name = image_epoch[:filt_suffix_index]
            
            image_colnum = image_colnums[cur_epoch_index]
        
            epoch_dir = epoch_dirs[cur_epoch_index]
            epoch_stf_version = epoch_stf_versions[cur_epoch_index]
            epoch_passband = epoch_passbands[cur_epoch_index]
            epoch_angle = epoch_angles[cur_epoch_index]
            
            if override_stf_version != None:
                epoch_stf_version=override_stf_version
            
            if print_diagnostics:
                print(image_epoch)
            
            # Read in image from FITS file
            fits_file = f'{dr_loc}/combo/{epoch_name}/mag{image_epoch}.fits'
            if use_res_file:
                fits_file = f'{dr_loc}/starlists/combo/{epoch_name}/'
                fits_file += f'starfinder_v{epoch_stf_version}/'
                fits_file += f'mag{image_epoch}_res.fits'
            
            # Substitute holography file location for speckle data
            if 'holography' in epoch_dir:
                fits_file = epoch_dir + '/final/' + image_epoch + '_holo.fits'
            
            # Check for epochs that are not in data release
            if not os.path.isfile(fits_file):
                # Try old data location
                fits_file = epoch_dir + '/combo/mag' + image_epoch + '.fits'
                
                if not os.path.isfile(fits_file):
                    print('Cannot locate image file for epoch ' + epoch_name)
                
            warnings.simplefilter('ignore', UserWarning)
            with fits.open(fits_file) as hdulist:
                image_data = hdulist[0].data
            
            # Rotate the image if angle is non zero
            orig_image_center = (
                np.array(image_data.shape[:2][::-1]) - 1.0
            ) / 2.
            
            rotated_image_center = orig_image_center
            
            if epoch_angle != 0.0:
                image_data = ndimage.rotate(
                    image_data, -1 * epoch_angle, reshape=True,
                )
                
                rotated_image_center = (
                    np.array(image_data.shape[:2][::-1]) - 1.0
                ) / 2.
            
            # Make conversion of align names to starlist name
            align_to_starlist = {}
            with open("{0}{1}/align/{2}.name".format(
                          align_dir_loc, align_name,
                          align_base_name), "r") as name_file:
                for cur_line in name_file:
                    split_line = cur_line.split()
    
                    align_star_name = split_line[0]
                    starlist_star_name = split_line[image_colnum]
    
                    if not starlist_star_name.startswith('-'):
                        align_to_starlist[align_star_name] = starlist_star_name

            ## Get position of stars in starlist
            starlist_x_pos = {}
            starlist_y_pos = {}

            with open("{0}{1}/lis/mag{2}_rms.lis".format(
                          align_dir_loc, align_name,
                          image_epoch), "r") as starlist:
                for cur_line in starlist:
                    split_line = cur_line.split()
    
                    star_name = split_line[0]
                    
                    x_pos = float(split_line[3])
                    y_pos = float(split_line[4])
                    
                    plot_coords = np.array([
                        x_pos, y_pos
                    ])
                    
                    if epoch_angle != 0.0:
                        plot_coords = calc_rotated_coords(
                            plot_coords, -1 * epoch_angle,
                            orig_image_center, rotated_image_center,
                        )
                    
                    starlist_x_pos[star_name] = plot_coords[0]
                    starlist_y_pos[star_name] = plot_coords[1]
            
            if print_diagnostics:
                print(image_epoch)
            
            ## Calculate SgrA position from S0-2, if necessary
            if ((('SgrA' not in align_to_starlist) and calc_SgrA_pos) or
                (calc_SgrA_pos and prefer_calc_SgrA_pos)):
                neighbor_star = 'S0-2'
                
                poss_neighbor_stars = ['S0-1', 'S0-17', 'S0-6']
                neighbor_ind = 0
                while neighbor_star not in align_to_starlist and (neighbor_ind) < len(poss_neighbor_stars):
                    neighbor_star = poss_neighbor_stars[neighbor_ind]
                    neighbor_ind += 1
                
                if neighbor_star not in align_to_starlist:
                    continue
                
                align_to_starlist['SgrA'] = 'SgrA_calc'
                
                neighbor_index = np.where(star_names == neighbor_star)[0][0]
                neighbor_xs = star_Xs_Array[:, neighbor_index]
                neighbor_ys = star_Ys_Array[:, neighbor_index]
                
                x_offset = neighbor_xs[cur_epoch_index] / plate_scale
                y_offset = neighbor_ys[cur_epoch_index] / plate_scale
                
                starlist_x_pos['SgrA_calc'] = starlist_x_pos[align_to_starlist[neighbor_star]] - x_offset
                starlist_y_pos['SgrA_calc'] = starlist_y_pos[align_to_starlist[neighbor_star]] - y_offset
            
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
                
                neighbor_stars_plot_x = np.append(
                    neighbor_stars_plot_x,
                    starlist_x_pos[align_to_starlist[cur_star]])
                neighbor_stars_plot_y = np.append(
                    neighbor_stars_plot_y,
                    starlist_y_pos[align_to_starlist[cur_star]])
        
                if cur_star == center_star:
                    center_star_plot_x = starlist_x_pos[align_to_starlist[cur_star]]
                    center_star_plot_y = starlist_y_pos[align_to_starlist[cur_star]]
    
            ## Drawing image plot
            fig = plt.figure(figsize=stamp_size)
            
            ax1 = fig.add_subplot(1, 1, 1)
    
            
            # Normalize image scale to a stable calibrator star
            norm_star = 'S1-1'
            norm_value = 4500.
            
            if norm_star in align_to_starlist:
                norm_star_x = int(np.round(starlist_x_pos[align_to_starlist[norm_star]]))
                norm_star_y = int(np.round(starlist_y_pos[align_to_starlist[norm_star]]))

                norm_rad = 3

                pre_norm = np.median(image_data[
                    norm_star_y-norm_rad:norm_star_y+norm_rad,
                    norm_star_x-norm_rad:norm_star_x+norm_rad,
                ],)
                
                norm_factor = pre_norm/norm_value
                image_data = image_data/norm_factor
                
                post_norm = np.median(image_data[
                    norm_star_y-norm_rad:norm_star_y+norm_rad,
                    norm_star_x-norm_rad:norm_star_x+norm_rad,
                ])
                
                if print_diagnostics:
                    print(f'Norm star: {norm_star}')
                    print(f'at coords ({norm_star_x}, {norm_star_y})')
                    print(f'Norm star median value pre norm: {pre_norm:.3f}')
                    print(f'Norm star median value post norm: {post_norm:.3f}')
                    print(f'Computed norm factor: {norm_factor}')
                
                if image_epoch in specify_image_normalization:
                    norm_factor = specify_image_normalization[image_epoch]
                    image_data = image_data/norm_factor
                    
                    if print_diagnostics:
                        print(f'Using manually specified normalization: {norm_factor}')
            else:
                print(f'Norm star, {norm_star}, not identifed in epoch {epoch_name}.')
                print('Not normalizing image scale')

            # Default AO image scaling
            im_add = 0.
            im_floor = 100.
            im_ceil = 2.e3
            
            im_mult = 1.
            im_invert = -1.
            
            if black_sky:
                im_invert = 1.
            
            if AO_high_contrast:
                im_add = 5000.
                im_floor = 1200.
                im_ceil = 9.e3
                
                im_mult = 400.
                im_invert = 0.
                
                im_add = 0.
                im_floor = 60.
                im_mult = 200.
                im_invert = -1.
            
            if epoch_passband == 'lp':
                im_add = 0.
                im_floor = 300.
                im_mult = 200.
            
            if 'holography' in epoch_dir:
                im_add = 20.

                im_add = 1800.
                im_floor = 0.
                im_ceil = 1.e6
                im_ceil = 5.e5
                im_ceil = 4.e3
                
                im_mult = 200.
                im_mult = 1.0
                im_invert = 1.
            else:
                if use_res_file:
                    if print_diagnostics:
                        print(f'Im Max = {np.max(image_data)}')
                        print(f'Im Min = {np.min(image_data)}')
                    
                    im_add = 0.
                    [im_floor, im_ceil] = np.percentile(image_data, [5, 95])
                    
                    im_mult = 1.
                    im_invert = -1.

            
            # Put in image floor
            image_data = image_data + im_add
            image_data[np.where(image_data <= im_floor)] = im_floor
            image_data[np.where(image_data >= im_ceil)] = im_ceil

            image_data *= im_mult
            
            # Color Map for the image
            im_cmap = plt.get_cmap('hot')
            if bw_image:
                im_cmap = plt.get_cmap('gray')
            
            # Display image
            if 'holography' in epoch_dir:
                ax1.imshow(
                    im_invert * np.sqrt(image_data),
                    cmap=im_cmap,
                    interpolation='nearest',
                )
            elif use_res_file:
                ax1.imshow(
                    im_invert * image_data,
                    cmap=im_cmap,
                    interpolation='nearest',
                )
            else:
                ax1.imshow(
                    im_invert * np.sqrt(image_data),
                    cmap=im_cmap,
                    interpolation='nearest',
                )
            
            # Stars
            if not no_labels:            
                for star_index in range(len(neighbor_stars_plot_x)):
                    plot_coords = np.array([
                        neighbor_stars_plot_x[star_index],
                        neighbor_stars_plot_y[star_index],
                    ])
                    
                    ax1.add_artist(plt.Circle(
                        (plot_coords[0], plot_coords[1]),
                        radius=(circle_size * 1./plate_scale),
                        linestyle='-', edgecolor='red',
                        linewidth=1.5, fill=False,
                    ))
                
                # Green circle for target star
                ax1.add_artist(plt.Circle(
                    (center_star_plot_x, center_star_plot_y),
                    radius=(circle_size * 1./plate_scale),
                    linestyle='-', edgecolor='green',
                    linewidth=1.5, fill=False,
                ))
            
            ### Axes Limits
            
            #### Make stamp at avg. spot if star not detected in this epoch
            if center_star_plot_x == 0.:
                center_star_plot_x = np.mean(neighbor_stars_plot_x[np.where(neighbor_stars_plot_x > 0.)])
                center_star_plot_y = np.mean(neighbor_stars_plot_y[np.where(neighbor_stars_plot_y > 0.)])
            
            if np.isnan(center_star_plot_x) or np.isnan(center_star_plot_y):
                print(f'No center star in epoch {image_epoch}')
                print('Neighbor stars: {0}\n'.format(neighbor_stars_sample))
                
                plt.close(fig)
                continue
            
            # # Rotate center coordinates if image has nonzero position angle
            # if epoch_angle != 0.0:
            #     orig_coords = np.array([
            #         center_star_plot_x, center_star_plot_y,
            #     ])
            #     
            #     rotated_coords = calc_rotated_coords(
            #         orig_coords, epoch_angle,
            #         orig_image_center, rotated_image_center,
            #     )
            #     
            #     center_star_plot_x = rotated_coords[0]
            #     center_star_plot_y = rotated_coords[1]
            #     
            #     ax1.add_artist(plt.Circle(
            #         (center_star_plot_x, center_star_plot_y),
            #         radius=(circle_size * 1./plate_scale),
            #         linestyle='-', edgecolor='green',
            #         linewidth=1.5, fill=False,
            #     ))
                        
            # Set image bounds around the center coordinates
            image_x_bounds = [
                center_star_plot_x - image_center_buffer,
                center_star_plot_x + image_center_buffer,
            ]
            image_y_bounds = [
                center_star_plot_y - image_center_buffer,
                center_star_plot_y + image_center_buffer,
            ]

            ax1.set_xlim(image_x_bounds)
            ax1.set_ylim(image_y_bounds)
            ax1.get_xaxis().set_visible(False)
            ax1.get_yaxis().set_visible(False)
            

            ### Size Scale
            scale_xEnd = center_star_plot_x + scale_edge
            scale_y = center_star_plot_y - scale_edge

            ax1.plot([scale_xEnd - 0.5/plate_scale, scale_xEnd], [scale_y, scale_y], '-', color='white', linewidth=4)
            ax1.plot([scale_xEnd - 0.5/plate_scale, scale_xEnd], [scale_y, scale_y], '-', color='black', linewidth=2)
            ax1.text((scale_xEnd - 0.5/plate_scale + scale_xEnd)/2., 3. + scale_y, '1/2 arcsec', ha='center', va='bottom', size='medium', color='black')
            ax1.text((scale_xEnd - 0.5/plate_scale + scale_xEnd)/2., scale_y - 5., '{0:.3f}'.format(epoch_dates[cur_epoch_index]), ha='center', va='top', size='medium', color='black', bbox = dict(boxstyle = 'round,pad=0.1', edgecolor='none', facecolor = 'white', alpha = 0.25))

            ## Star Labels
            if not no_labels:
                for cur_star in neighbor_stars_sample:
                    try:
                        plot_coords = np.array([
                            starlist_x_pos[align_to_starlist[cur_star]],
                            starlist_y_pos[align_to_starlist[cur_star]],
                        ])
                        
                        if label_side:
                            ax1.text(
                                label_offset + plot_coords[0],
                                plot_coords[1],
                                cur_star.replace('irs', 'IRS '),
                                ha='left', va='center', size='x-small',
                                bbox = dict(
                                    boxstyle = 'round,pad=0.1',
                                    edgecolor='none', facecolor = 'white',
                                    alpha = 0.25,
                                ),
                            )  # .replace('_', '\_')
                        else:
                            ax1.text(
                                plot_coords[0],
                                label_offset + plot_coords[1],
                                cur_star.replace('irs', 'IRS '),
                                ha='center', va='bottom', size='x-small',
                                bbox = dict(
                                    boxstyle = 'round,pad=0.1',
                                    edgecolor='none', facecolor = 'white',
                                    alpha = 0.25,
                                ),
                            )  # .replace('_', '\_')
                    except:
                        continue
                        # print(cur_star + ' not found')

            fig.tight_layout()

            if make_pdfs:
                fig.savefig('{0}/{1}/pdf/PostageStamps_{1}_{2}_{3}.pdf'.format(out_dir, file_name, align_name, image_colnum))
            
            fig.savefig('{0}/{1}/png/PostageStamps_{1}_{2}_{3}.png'.format(out_dir, file_name, align_name, image_colnum),
                        dpi=png_dpi)
            plt.close(fig)
            
            if add_animate_epochs:
                animate_epochs.append(image_colnum)
    
        # Generate gif animation
        animate_epoch_nums = image_colnums
        
        if print_diagnostics:
            print('animate_epochs = ')
            print(animate_epochs)
        if animate_epochs is not None:
            animate_epoch_nums = animate_epochs
        
        # Assign frame times based on image time separation
        frame_times = []
        
        for image_colnum_index in range(len(animate_epoch_nums)):
            frame_time = last_frame_time
            
            if image_colnum_index != 0 \
               and image_colnum_index+1 < len(animate_epoch_nums):
                frame_time = (epoch_dates[animate_epoch_nums[image_colnum_index + 1] - 1]
                              - epoch_dates[animate_epoch_nums[image_colnum_index] - 1]) \
                             * year_span
            
            frame_times.append(frame_time)
        
        # Write GIF
        with imageio.get_writer('{0}/{1}/PostageStamps_{1}_{2}.gif'.format(out_dir,
                                    file_name, align_name),
                                mode='I', duration=frame_times) as gif_writer:
            for image_colnum in animate_epoch_nums:
                cur_image = imageio.imread('{0}/{1}/png/PostageStamps_{1}_{2}_{3}.png'.format(out_dir, file_name, align_name, image_colnum))
                gif_writer.append_data(cur_image)
