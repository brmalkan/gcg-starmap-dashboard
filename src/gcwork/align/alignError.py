from gcwork import starset
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
import pdb

adderr_ao = [0.057913912713861204, 11.688961885128897, 1.3745074553463132, -1.0741825191131915]

def adderr_ao_func(adderr_ao,x):
    a = adderr_ao[0]
    b = adderr_ao[1]
    alpha = adderr_ao[2]
    c = adderr_ao[3]
    if x>b:
        error =  10**(a*(x-b)**alpha+c)
    else:
        error = 0.1
    return error

 
def ErrVsEpoch(root='./', align='align/align_d_rms_1000_abs_t', points='points_4_trim/', poly='polyfit_4_trim/fit',
        mag_limit=16, r_limit=2, starset_obj=None):
    """
    plot the meadian position error and alignment error for all stars in each epoch"""
    # find the index for speckle epochs and AO epochs
    t_label = Table.read(root+'scripts/epochsInfo.txt', format='ascii')
    idx_align = np.where(t_label['doAlign']==1)[0]
    idx_spe = np.where(t_label['isAO'][idx_align]==0)[0]
    idx_ao = np.where(t_label['isAO'][idx_align]==1)[0]

    # Load new StarSet object if not provided
    s = starset_obj
    
    if starset_obj is None:
        s = starset.StarSet(root + align)
        s.loadPolyfit(root + poly, accel=0, arcsec=1)
        s.loadPolyfit(root + poly, accel=1, arcsec=1)
        s.loadPoints(root + points)
    
    years = s.getArray('years')[0]
    r2d = s.getArray('r2d')

    # initiate error term for all epochs
    pe_aln_med = np.zeros(len(idx_align))
    pe_add_med = np.zeros(len(idx_align))
    pe_pos_med = np.zeros(len(idx_align))
    pe_dis_med = np.zeros(len(idx_align))

    # loop through epochs for errors
    for epoch in range(len(idx_align)):
        # e_p is the combined error in: 3 submap error & distortion error & additive error
        # e_a is the alignment bootstrap error

        xe_p_epoch = s.getArrayFromEpoch(epoch, 'xerr_p') * 1000.
        ye_p_epoch = s.getArrayFromEpoch(epoch, 'yerr_p') * 1000.
        xe_a_epoch = s.getArrayFromEpoch(epoch, 'xerr_a') * 1000.
        ye_a_epoch = s.getArrayFromEpoch(epoch, 'yerr_a') * 1000.
        mag_epoch = s.getArrayFromEpoch(epoch, 'mag')
        name_epoch = np.array(s.getArrayFromEpoch(epoch, 'name'))

        # cut the sample: non-zero position error (detected in epoch)
        # mag and radius cut 
        idx_real = np.where((xe_p_epoch>0) & (ye_p_epoch>0) & (mag_epoch<mag_limit) & (r2d<r_limit))[0]
        xe_a_epoch = xe_a_epoch[idx_real]
        ye_a_epoch = ye_a_epoch[idx_real]
        xe_p_epoch = xe_p_epoch[idx_real]
        ye_p_epoch = ye_p_epoch[idx_real]
        mag_epoch = mag_epoch[idx_real]
        name_epoch = name_epoch[idx_real]

        # alignment error in mas
        xe_aln = xe_a_epoch
        ye_aln = ye_a_epoch
        pe_aln_med[epoch] = np.median([xe_aln, ye_aln])

        # For speckle: positional error, no additive error or distortion error
        if np.in1d(epoch, idx_spe):
            xe_pos = xe_p_epoch
            ye_pos = ye_p_epoch
            pe_pos_med[epoch] = np.median([xe_pos, ye_pos])
            continue

        # For AO: poistional error + additive error (all AO) + distortion error (non-06 AO)

        # For 06-14 AO epochs: positional error + additive error 
        elif t_label[idx_align][epoch]['06setup'] == 1:
            t_rms_noadd = Table.read(root + '/lis/mag' + t_label[idx_align][epoch]['epoch'] + '_rms_noadd.lis', format='ascii')
            name_rms_noadd = np.array(t_rms_noadd['col1'])
            idx_tmp = [np.where(name_rms_noadd==i)[0][0] for i in name_epoch]
            t_rms_noadd = t_rms_noadd[idx_tmp]

            t_rms = Table.read(root + '/lis/mag' + t_label[idx_align][epoch]['epoch'] + '_rms.lis', format='ascii')
            name_rms = np.array(t_rms['col1'])
            idx_tmp = [np.where(name_rms==i)[0][0] for i in name_epoch]
            t_rms = t_rms[idx_tmp]

            # read in errors in pixel
            xe_pos_pixel = t_rms_noadd['col6']
            ye_pos_pixel = t_rms_noadd['col7']
            xe_add_pixel = np.sqrt(t_rms['col6']**2 - t_rms_noadd['col6']**2)
            ye_add_pixel = np.sqrt(t_rms['col7']**2 - t_rms_noadd['col7']**2)

            scale_x = xe_p_epoch/t_rms['col6']
            scale_y = ye_p_epoch/t_rms['col7']

            # read in errors in arcsec
            xe_pos = xe_pos_pixel * scale_x
            ye_pos = ye_pos_pixel * scale_y
            xe_add = xe_add_pixel * scale_x
            ye_add = ye_add_pixel * scale_y

            pe_pos_med[epoch] = np.median([xe_pos, ye_pos])
            pe_add_med[epoch] = np.median([xe_add, ye_add])
            continue

        # For non-06 epochs: positional error + additive error + local distortion error
        else:
            t_rms_noadd = Table.read(root + '/lis/mag' + t_label[idx_align][epoch]['epoch'] + '_rms_noadd.lis', format='ascii')
            name_rms_noadd = np.array(t_rms_noadd['col1'])
            idx_tmp = [np.where(name_rms_noadd==i)[0][0] for i in name_epoch]
            t_rms_noadd = t_rms_noadd[idx_tmp]

            t_rms = Table.read(root + '/lis/mag' + t_label[idx_align][epoch]['epoch'] + '_rms.lis', format='ascii')
            name_rms = np.array(t_rms['col1'])
            idx_tmp = [np.where(name_rms==i)[0][0] for i in name_epoch]
            t_rms = t_rms[idx_tmp]

            t_rms_ld = Table.read(root + '/lis/mag' + t_label[idx_align][epoch]['epoch'] + '_rms_ld.lis', format='ascii')
            name_rms_ld = np.array(t_rms_ld['col1'])
            idx_tmp = [np.where(name_rms_ld==i)[0][0] for i in name_epoch]
            t_rms_ld = t_rms_ld[idx_tmp]

            # read in errors in pixel
            xe_pos_pixel = t_rms_noadd['col6']
            ye_pos_pixel = t_rms_noadd['col7']
            xe_add_pixel = np.sqrt(t_rms['col6']**2 - t_rms_noadd['col6']**2)
            ye_add_pixel = np.sqrt(t_rms['col7']**2 - t_rms_noadd['col7']**2)
            xe_dis_pixel = np.sqrt(t_rms_ld['col6']**2 - t_rms['col6']**2)
            ye_dis_pixel = np.sqrt(t_rms_ld['col7']**2 - t_rms['col7']**2)

            scale_x = xe_p_epoch/t_rms_ld['col6']
            scale_y = ye_p_epoch/t_rms_ld['col7']

            # read in errors in arcsec
            xe_pos = xe_pos_pixel * scale_x
            ye_pos = ye_pos_pixel * scale_y
            xe_add = xe_add_pixel * scale_x
            ye_add = ye_add_pixel * scale_y
            xe_dis = xe_dis_pixel * scale_x
            ye_dis = ye_dis_pixel * scale_y

            pe_pos_med[epoch] = np.nanmedian([xe_pos, ye_pos])
            pe_add_med[epoch] = np.nanmedian([xe_add, ye_add])
            pe_dis_med[epoch] = np.nanmedian([xe_dis, ye_dis])

    idx_non06 = np.where((t_label[idx_align]['isAO'] == 1) & (t_label[idx_align]['06setup'] == 0))[0]
    plt.clf()
    plt.plot(years, pe_aln_med, 'ko', label=r'$\sigma_{aln}$')
    plt.plot(years, pe_pos_med, 'rs', mfc='none', label=r'$\sigma_{pos}$')
    plt.plot(years[idx_non06], pe_dis_med[idx_non06], 'b^', mfc='none', label=r'$\sigma_{distortion}$')
    plt.plot(years[idx_ao], pe_add_med[idx_ao], 'g--', label=r'$\sigma_{add}$')
    plt.legend()
    plt.xlabel('Date(year)')
    plt.ylabel('Positional error in individual epochs(mas)')
    plt.ylim(0.01, 5)
    plt.gca().set_yscale('log')
    plt.tick_params(axis='y', which='both', right=True, direction='in')    
    plt.title('stars with K<%.1f and r<%.1f arcsec' %(mag_limit, r_limit))
    plt.savefig('plots/ErrVsEpoch.png')
    
    f = open('plots/ErrVsEpoch.txt', 'w')
    f.write('#Epoch  Aln_Err  Pos_Err')
    for i in range(len(years)):
        f.write('\n')
        f.write(str(years[i]))
        f.write(' ')
        f.write(str(pe_aln_med[i]))
        f.write(' ')
        f.write(str(pe_pos_med[i]))
    f.close()
        


def avErrVsMag(root='./', align='align/align_d_rms_1000_abs_t', poly='polyfit_4_trim/fit'):
    # Get the star set info
    s = starset.StarSet(root+align)
    s.loadPolyfit(root+poly, arcsec=1)
    s.loadPolyfit(root+poly, arcsec=1, accel=1)

    vxe = s.getArray('fitXa.verr') * 1000.
    vye = s.getArray('fitYa.verr') * 1000.
    axe = s.getArray('fitXa.aerr') * 1000.
    aye = s.getArray('fitYa.aerr') * 1000.
    mag = s.getArray('mag')
    r2d = s.getArray('r2d')
    nEpochs = s.getArray('velCnt')
    names = np.array(s.getArray('name'))

    # cut mag and radius
    idx = np.where((r2d<5) & (r2d>0.8) & (mag<16.5) & (nEpochs>20))[0]

    # plot
    plt.clf()
    plt.semilogy(mag[idx], vxe[idx], 'bx', label='vxe')
    plt.semilogy(mag[idx], vye[idx], 'r+', label='vye')
    plt.xlabel('Magnitude')
    plt.ylabel('Vel Error (mas/yr)')
    plt.title('0.8<r<5 and mag<16.5 and detect>20: N=%d' %len(idx))
    plt.legend()
    plt.savefig('plots/VelErrVsMag.png')

    plt.clf()
    plt.semilogy(mag[idx], axe[idx], 'bx', label='axe')
    plt.semilogy(mag[idx], aye[idx], 'r+', label='aye')
    plt.xlabel('Magnitude')
    plt.ylabel(r'Acc Error (mas/yr$^2$)')
    plt.title('0.8<r<5 and mag<16.5 and detect>20: N=%d' %len(idx))
    plt.legend()
    plt.savefig('plots/AccErrVsMag.png')


def alnErr2D(root, epoch):
    s = starset.StarSet(root)

    x = s.getArray('x')
    y = s.getArray('y')
    xerr = s.getArrayFromEpoch(epoch, 'xerr_a')
    yerr = s.getArrayFromEpoch(epoch, 'yerr_a')

    idx = (np.where(xerr > 0))[0]
    x = x[idx]
    y = y[idx]
    xerr = xerr[idx] * 1000.0
    yerr = yerr[idx] * 1000.0

    p.clf()
    #p.quiver([x], [y], [xerr], [yerr], 0.4)
    scale = 3.0e2
    p.scatter(x, y, xerr*scale, c='b', marker='o', alpha=0.5)
    p.scatter(x, y, yerr*scale, c='r', marker='s', alpha=0.5)
    p.scatter([-3.5], [-3.5], 0.5 * scale, c='k', marker='s')
    p.text(-3.1, -3.6, '0.5 mas')

    p.xlabel('X')
    p.ylabel('Y')
    p.title('Epoch #%d:  %8.3f' % (epoch, s.stars[0].years[epoch]))
    p.savefig('plot/alnErr2D.png')

def alnErrVsRadius(root, epoch):
    # Get the star set info
    s = starset.StarSet(root)

    x = s.getArray('x')
    y = s.getArray('y')

    xerr = s.getArrayFromEpoch(epoch, 'xerr_a')
    yerr = s.getArrayFromEpoch(epoch, 'yerr_a')

    idx = (np.where(xerr > 0))[0]

    x = x[idx]
    y = y[idx]
    xerr = xerr[idx] * 1000.0
    yerr = yerr[idx] * 1000.0

    r = np.hypot(x, y)

    p.clf()
    p.semilogy(r, xerr, 'bx')
    p.semilogy(r, yerr, 'r+')
    p.axis([0, 5, 0.05, 5])

    p.xlabel('Radius (arcsec)')
    p.ylabel('Alignment Error (mas)')

    p.title('Epoch #%d:  %8.3f' % (epoch, s.stars[0].years[epoch]))
    p.savefig('plots/alnErrVsRadius.png')


def alnErrVsMag(root, epoch):
    # Get the star set info
    s = starset.StarSet(root)

    xerr = s.getArrayFromEpoch(epoch, 'xerr_a')
    yerr = s.getArrayFromEpoch(epoch, 'yerr_a')
    mag = s.getArray('mag')

    idx = (np.where(xerr > 0))[0]
    mag = mag[idx]
    xerr = xerr[idx] * 1000.0
    yerr = yerr[idx] * 1000.0

    p.clf()
    p.semilogy(mag, xerr, 'bx')
    p.semilogy(mag, yerr, 'r+')
    p.axis([8, 17, 0.05, 5])

    p.xlabel('Magnitude')
    p.ylabel('Alignment Error (mas)')

    p.title('Epoch #%d:  %8.3f' % (epoch, s.stars[0].years[epoch]))
    p.savefig('plots/alnErrVsMag.png')



def compare_a3(root1, root2, epoch):
    s1 = starset.StarSet(root1)
    s2 = starset.StarSet(root2)

    x1 = s1.getArrayFromEpoch(epoch, 'x')
    y1 = s1.getArrayFromEpoch(epoch, 'y')
    xerr_p1 = s1.getArrayFromEpoch(epoch, 'xerr_p')
    yerr_p1 = s1.getArrayFromEpoch(epoch, 'yerr_p')
    xerr_a1 = s1.getArrayFromEpoch(epoch, 'xerr_a')
    yerr_a1 = s1.getArrayFromEpoch(epoch, 'yerr_a')
    name1 = s1.getArray('name')

    x2 = s2.getArrayFromEpoch(epoch, 'x')
    y2 = s2.getArrayFromEpoch(epoch, 'y')
    xerr_p2 = s2.getArrayFromEpoch(epoch, 'xerr_p')
    yerr_p2 = s2.getArrayFromEpoch(epoch, 'yerr_p')
    xerr_a2 = s2.getArrayFromEpoch(epoch, 'xerr_a')
    yerr_a2 = s2.getArrayFromEpoch(epoch, 'yerr_a')
    name2 = s2.getArray('name')

    idx1 = []
    idx2 = []
    for id1 in range(len(name1)):
        # Skip if this star wasn't detected in this epoch for set #1
        if (xerr_p1[id1] <= 0):
            continue
        
        xdiff = x1[id1] - x2
        ydiff = y1[id1] - y2
        diff = p.sqrt(xdiff**2 + ydiff**2)

        id2 = (diff.argsort())[0]

        # Skip if this star wasn't detected in this epoch for set #2
        if (xerr_p2[id2] <= 0):
            continue

        idx1.append(int(id1))
        idx2.append(int(id2))

    x1 = x1[idx1]
    y1 = y1[idx1]
    xerr_p1 = xerr_p1[idx1]
    yerr_p1 = yerr_p1[idx1]
    xerr_a1 = xerr_a1[idx1]
    yerr_a1 = yerr_a1[idx1]

    x2 = x2[idx2]
    y2 = y2[idx2]
    xerr_p2 = xerr_p2[idx2]
    yerr_p2 = yerr_p2[idx2]
    xerr_a2 = xerr_a2[idx2]
    yerr_a2 = yerr_a2[idx2]

    ##########
    #
    #  Plots
    #
    ##########
    p.clf()
    p.quiver([x1], [y1], [x1 - x2], [y1 - y2], 0.6)
    p.quiver([[-7]], [[-5]], [[0.001]], [[0]], 0.6)
    p.text(-6, -5.2, '1 mas')

    p.xlabel('X')
    p.ylabel('Y')
    p.title('Compare -a 2 vs. -a 3')

    
