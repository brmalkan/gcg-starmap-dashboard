#!/usr/bin/env python
import optparse
import textwrap
import numpy as np
import pylab as py
import math
import sys
from astropy.table import Table
from gcwork import starset
from gcwork import orbits
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib.lines as lines
import matplotlib.pyplot as plt
#from mpldatacursor import datacursor
import pdb
import os
from itertools import cycle

##################################################
#
# Main body of compare_pos
#
##################################################
def run(align_root='align/align_d_rms_1000_abs',
        plot_errors=False, center_star=None, range=0.4,
        xcenter=0, ycenter=0, show_names=False,saveplot=False,mag_range=None,with_orbits=True,
        orbits_file=None,manual_print=False,efit_points='efit_mjd/'):
    """
    Parameters
    ----------
    align_root : str
        The root name of the align to load up and plot.


    Optional Keywords
    -----------------
    plot_errors : bool
        (def=False) Plot error bars on all the points (quad sum of positional
        and alignment errors).
    center_star : str
        (def=None) Named star to center initial plot on.
    show_names : bool
        (def=False) label the name of the star in the first epoch
    range : float
        (def=0.4) Sets the half width of the X and Y axis in
        arcseconds from xcenter and ycenter.
    xcenter : float
        (def=0.0) The X center point of the plot in arcseconds
        offset from Sgr A*.
    ycenter : float
        (def=0.0) The Y center point of the plot in arcseconds
        offset from Sgr A*.
    saveplot : bool
        (def=False) Save plot as .png and .pdf.
    mag_range: intervals
        (def=None) Magnitude cuts performed using the given interval
    with_orbits : bool
        (def=True) will print the orbits using the file orbits.dat (see next keyword)
    orbits_file : str
        (def=None) used if with_orbits=True. If None, will use the orbits.dat file related to the align_root. Otherwise, will use this as the name of the orbits.dat file to use
    """

    option_string = ''

    if plot_errors:
        option_string += ' -e'

    if center_star is not None:
        option_string += ' -s {0}'.format(center_star)

    option_string += ' -r {0:.3f}'.format(range)
    option_string += ' -x {0:.3f}'.format(xcenter)
    option_string += ' -y {0:.3f}'.format(ycenter)

    if show_names:
        option_string += ' -n'

    if saveplot:
        option_string += ' --saveplot'

    if manual_print:
        option_string += ' -m'

    option_string += ' {0}'.format(align_root)

    print(option_string)

    main(argv=option_string.split(),mag_range=mag_range,with_orbits=with_orbits,orbits_file=orbits_file,efit_points=efit_points)

    return


def main(argv=None,mag_range=None,with_orbits=False,orbits_file=None,efit_points='efit_mjd/'):
    if argv is None:
        argv = sys.argv[1:]
    # Read options and check for errors.
    options = read_command_line(argv)
    if (options == None):
        return

    s = starset.StarSet(options.align_root)

    nEpochs = len(s.stars[0].years)
    nStars = len(s.stars)

    names = np.array(s.getArray('name'))

    if (options.center_star != None):
        idx = np.where(names == options.center_star)[0]

        if (len(idx) > 0):
            options.xcenter = s.stars[idx[0]].x
            options.ycenter = s.stars[idx[0]].y
        else:
            print('Could not find star to center, %s. Reverting to Sgr A*.' % \
                  (options.center_star))

    # Create a combined error term (quad sum positional and alignment)
    combineErrors(s)

    yearsInt = np.floor(s.years)

    # Set up a color scheme
    cnorm = colors.Normalize(s.years.min(), s.years.max()+1)
    cmap = cm.gist_ncar

    colorList = []
    for ee in range(nEpochs):
        colorList.append( cmap(cnorm(yearsInt[ee])) )

    py.close(2)
    fig = py.figure(2, figsize=(13,10))


    previousYear = 0.0

    point_labels = {}
    epochs_legend=[]

    for ee in range(nEpochs):
        x = s.getArrayFromEpoch(ee, 'x')
        y = s.getArrayFromEpoch(ee, 'y')

        xe = s.getArrayFromEpoch(ee, 'xerr')
        ye = s.getArrayFromEpoch(ee, 'yerr')

        mag = s.getArrayFromEpoch(ee, 'mag')
        name_epoch  = np.array(s.getArrayFromEpoch(ee,'name'))

        if mag_range is None:
            idx = np.where((x > -1000) & (y > -1000))[0]
        else:
            idx = np.where((x > -1000) & (y > -1000) & (mag <= np.max(mag_range)) & (mag >= np.min(mag_range)))[0]

        x = x[idx]
        y = y[idx]
        xe = xe[idx]
        ye = ye[idx]
        mag = mag[idx]
        name_epoch = name_epoch[idx]

        tmpNames = names[idx]

        if yearsInt[ee] != previousYear:
            previousYear = yearsInt[ee]
            label = '%d' % yearsInt[ee]
        else:
            label = '_nolegend_'

        if options.plot_errors:
            (line, foo1, foo2) = py.errorbar(x, y, xerr=xe, yerr=ye,
                                            color=colorList[ee], fmt='k^',
                                            markeredgecolor=colorList[ee],
                                            markerfacecolor=colorList[ee],
                                            label=label, picker=4)
        else:
            (line, foo1, foo2) = py.errorbar(x, y, xerr=None, yerr=None,
                                            color=colorList[ee], fmt='k^',
                                            markeredgecolor=colorList[ee],
                                            markerfacecolor=colorList[ee],
                                            label=label, picker=4)

        #for legend
        if label != '_nolegend_':
            line.set_label(str(label))
            epochs_legend.append(line)


        points_info = {'year': s.years[ee],
                       'name': tmpNames, 'x': x, 'y': y, 'xe': xe, 'ye': ye, 'mag': mag,'name_epoch':name_epoch}

        point_labels[line] = points_info

    foo = PrintSelected(point_labels,fig,s,mag_range,manual_print=options.manual_print)
    py.connect('pick_event', foo)

    if with_orbits:

        orb, starNames = generate_Orbits(options.align_root,orbits_file,tEnd=s.years[-1])
        normStar = colors.Normalize(0,len(orb)+2)
        orbitPlots=[]
        line_cycle = cycle(['-', '--', ':', '-.'])
        for idxStar in range(len(orb)):
            
            tmp,=py.plot(orb[idxStar][:,0],orb[idxStar][:,1],next(line_cycle),color=cmap(normStar(idxStar)), label=str(starNames[idxStar] ))

            orbitPlots.append(tmp)
            #orbLegend=py.legend(handles=orbitPlots,bbox_to_anchor=(1.02, 1.),loc=2,fontsize=12)
            orbLegend=py.legend(handles=orbitPlots,loc=1,fontsize=12)
            ax = py.gca().add_artist(orbLegend)

            if os.path.isfile(os.path.dirname(options.align_root)+'/../'+efit_points+'/'+starNames[idxStar]+'.points'):
                points = np.loadtxt(os.path.dirname(options.align_root)+'/../'+efit_points+'/'+starNames[idxStar]+'.points')
                # only plot if there are actually points
                if len(points) > 0:
                    tmp = py.plot(-points[:,1],points[:,2],markerfacecolor='none',markeredgecolor=cmap(normStar(idxStar)),marker='s',linestyle='None',ms='8')
            else:
                print(os.path.dirname(options.align_root)+'/../'+efit_points+'/'+starNames[idxStar]+'.points   files not found')





    xlo = options.xcenter + (options.range)
    xhi = options.xcenter - (options.range)
    ylo = options.ycenter - (options.range)
    yhi = options.ycenter + (options.range)

    py.axis('equal')
    py.axis([xlo, xhi, ylo, yhi])
    py.xlabel('R.A. Offset from Sgr A* (arcsec)')
    py.ylabel('Dec. Offset from Sgr A* (arcsec)')
    py.title(options.align_root)


    py.legend(handles=epochs_legend,numpoints=1, loc='lower left', fontsize=12)


    if options.show_names:
        xpos = s.getArray('x')
        ypos = s.getArray('y')
        goodind = np.where((xpos <= xlo) & (xpos >= xhi) &
                           (ypos >= ylo) & (ypos <= yhi))[0]
        for ind in goodind:
            py.text(xpos[ind],ypos[ind],names[ind],size=10)

    py.tight_layout()
    if options.saveplot:
        py.show(block=0)
        if (options.center_star != None):
            py.savefig(options.center_star + '_compare_pos.png')
            #py.savefig(options.center_star + '_compare_pos.pdf')
        else:
            py.savefig('compare_pos.png')
            #py.savefig('compare_pos.pdf')
    else:
        py.show()

    return


#function that reads orbit.dat and generates orbits for each stars included in orbit.dat
def generate_Orbits(align_root, orbits_file=None,tStart=1994. , tEnd=2020. , dt=0.01):

    t = np.linspace(tStart,tEnd,int(np.ceil((tEnd-tStart)/dt)))

    if orbits_file==None:
        tab = np.genfromtxt(os.path.dirname(align_root)+'/../source_list/orbits.dat', dtype=str)
    else:
        tab = np.genfromtxt(orbits_file, dtype=str)
#tab = np.genfromtxt('/u/ahees/ahees_server/align_2017_01_23/orbits.dat', dtype=None)

    res = []
    names = []
    for star in tab:
        orb = orbits.Orbit()
        orb.p  = float(star[1])
        orb.t0 = float(star[3])
        orb.e  = float(star[4])
        orb.i  = float(star[5])
        orb.o  = float(star[6])
        orb.w  = float(star[7])

        (x,v,a) = orb.kep2xyz(epochs=t,mass=4.e6,dist=8000.) #possible to change mass and R0 here if needed, current values are hardcoded in gcwork.objects.Constants
        names.append(star[0])
        res.append(x[:,0:2])

    return res,names

def read_command_line(argv):
    p = optparse.OptionParser(usage='usage: %prog [options] [starlist]',
                              formatter=IndentedHelpFormatterWithNL())

    p.add_option('-e', '--errors', dest='plot_errors', default=False,
                 action='store_true',
                 help='Plot error bars on all the points (quad sum of '+
                 'positional and alignment errors.')
    p.add_option('-s', '--star', dest='center_star', default=None,
                 metavar='[star]',
                 help='Named star to center initial plot on.')
    p.add_option('-r', '--range', dest='range', default=0.4, type=float,
                 metavar='[arcsec]',
                 help='Sets the half width of the X and Y axis in arcseconds'+
                 'from -xcen and -ycen  or 0,0 (default: %default)')
    p.add_option('-x', '--xcen', dest='xcenter', default=0, type=float,
                 metavar='[arcsec]',
                 help='The X center point of the plot in arcseconds offset' +
                 'from Sgr A*.')
    p.add_option('-y', '--ycen', dest='ycenter', default=0, type=float,
                 metavar='[arcsec]',
                 help='The Y center point of the plot in arcseconds offset' +
                 'from Sgr A*.')
    p.add_option('--saveplot', dest='saveplot', default=False,
                 action='store_true',
                 help='Save the plot')
    p.add_option('-n',dest='show_names',action='store_true',
                 default=False,help='Label the star names')
    p.add_option('-m',dest='manual_print',action='store_true',default=False,
                 help='Print info in correct format for manual_star_match.py')

    options, args = p.parse_args(argv)

    # Keep a copy of the original calling parameters
    options.originalCall = ' '.join(argv)

    # Read the input filename
    options.align_root = None
    if len(args) == 1:
        options.align_root = args[0]
    else:
        print('')
        p.print_help()
        return None

    return options


##################################################
#
# Help formatter for command line arguments.
# This is very generic... skip over for main code.
#
##################################################
class IndentedHelpFormatterWithNL(optparse.IndentedHelpFormatter):
    def format_description(self, description):
        if not description: return ""
        desc_width = self.width - self.current_indent
        indent = " "*self.current_indent
        # the above is still the same
        bits = description.split('\n')
        formatted_bits = [
            textwrap.fill(bit,
                          desc_width,
                          initial_indent=indent,
                          subsequent_indent=indent)
            for bit in bits]
        result = "\n".join(formatted_bits) + "\n"
        return result

    def format_option(self, option):
        # The help for each option consists of two parts:
        #   * the opt strings and metavars
        #   eg. ("-x", or "-fFILENAME, --file=FILENAME")
        #   * the user-supplied help string
        #   eg. ("turn on expert mode", "read data from FILENAME")
        #
        # If possible, we write both of these on the same line:
        #   -x    turn on expert mode
        #
        # But if the opt string list is too long, we put the help
        # string on a second line, indented to the same column it would
        # start in if it fit on the first line.
        #   -fFILENAME, --file=FILENAME
        #       read data from FILENAME
        result = []
        opts = self.option_strings[option]
        opt_width = self.help_position - self.current_indent - 2
        if len(opts) > opt_width:
            opts = "%*s%s\n" % (self.current_indent, "", opts)
            indent_first = self.help_position
        else: # start help on same line as opts
            opts = "%*s%-*s  " % (self.current_indent, "", opt_width, opts)
            indent_first = 0
        result.append(opts)
        if option.help:
            help_text = self.expand_default(option)
            # Everything is the same up through here
            help_lines = []
            for para in help_text.split("\n"):
                help_lines.extend(textwrap.wrap(para, self.help_width))
            # Everything is the same after here
            result.append("%*s%s\n" % (
                    indent_first, "", help_lines[0]))
            result.extend(["%*s%s\n" % (self.help_position, "", line)
                           for line in help_lines[1:]])
        elif opts[-1] != "\n":
            result.append("\n")
        return "".join(result)



def combineErrors(s):
    for ss in range(len(s.stars)):
        star = s.stars[ss]

        for ee in range(len(s.stars[0].e)):
            epoch = star.e[ee]

            epoch.xerr = math.sqrt(epoch.xerr_p**2 + epoch.xerr_a**2)
            epoch.yerr = math.sqrt(epoch.yerr_p**2 + epoch.yerr_a**2)


class PrintSelected(object):
    def __init__(self, points_info,fig,s,mag_range,manual_print=False):
        self.points_info = points_info
        self.selected, = fig.gca().plot([0.],[0.], 'o',ms=12,
                                 markerfacecolor='none', markeredgecolor='red', visible=False)
        self.selected_same_year, = fig.gca().plot([0.],[0.], '*',ms=15,
                                markerfacecolor='red', markeredgecolor='red', visible=False)
        self.fig = fig
        self.s = s
        self.manual_print=manual_print
        self.mag_range=mag_range
        return

    def __call__(self, event):
        if event.mouseevent.button == 1:
            indices = event.ind

            data = self.points_info[event.artist]

            if self.manual_print:
                fmt = 'align_name="{:s}",epoch={:f},align_mag={:4.2f},align_x={:10.4f},align_xerr={:7.4f},align_y={:10.4f},align_yerr={:7.4f},name_epoch="{:s}"'
            else:
                fmt = '{:15s}  t={:10.6f}  m={:5.2f}  x={:10.4f} +/- {:7.4f}  y={:10.4f} +/- {:7.4f}  Epoch name: {:15s}'

            for ii in indices:
                print(fmt.format(data['name'][ii], data['year'], data['mag'][ii],
                            data['x'][ii], data['xe'][ii],
                            data['y'][ii], data['ye'][ii],data['name_epoch'][ii]))

            idx = self.s.getArray('name').index(data['name'][indices[0]])
            xs = self.s.stars[idx].getArrayAllEpochs('x')
            ys = self.s.stars[idx].getArrayAllEpochs('y')
            #n  = self.s.stars[idx].getArrayAllEpochs('name')
            #print n
            self.selected.set_visible(True)
            self.selected.set_data(xs,ys)
            self.fig.canvas.draw()
        elif event.mouseevent.button == 3:
            indices = event.ind
            data = self.points_info[event.artist]

            if self.manual_print:
                fmt = 'align_name="{:s}",epoch={:f},align_mag={:4.2f},align_x={:10.4f},align_xerr={:7.4f},align_y={:10.4f},align_yerr={:7.4f},name_epoch="{:s}"'
            else:
                fmt = '{:15s}  t={:10.6f}  m={:5.2f}  x={:10.4f} +/- {:7.4f}  y={:10.4f} +/- {:7.4f}  Epoch name: {:15s}'

            ii =indices[0]
            print(fmt.format(data['name'][ii], data['year'], data['mag'][ii],
                             data['x'][ii], data['xe'][ii],
                             data['y'][ii], data['ye'][ii],data['name_epoch'][ii]))

            idxEpoch = np.where(self.s.years==data['year'])[0][0]
            x = self.s.getArrayFromEpoch(idxEpoch, 'x')
            y = self.s.getArrayFromEpoch(idxEpoch, 'y')
            mag = self.s.getArrayFromEpoch(idxEpoch, 'mag')
            if self.mag_range is None:
                idx = np.where((x > -1000) & (y > -1000))[0]
            else:
                idx = np.where((x > -1000) & (y > -1000) & (mag <= np.max(self.mag_range)) & (mag >= np.min(self.mag_range)))[0]

            x = x[idx]
            y = y[idx]
            self.selected_same_year.set_visible(True)
            self.selected_same_year.set_data(x,y)
            self.fig.canvas.draw()

        return

if __name__ == '__main__':
    main()



############################
#
## plot star's track for one star
#
############################
def plot_stars(stars, accel=False, save_path='plots/plotStar', plot_label_fit=False,
               root='./', align='align/align_d_rms_1000_abs_t', poly='polyfit_1_1000/fit',
               plotdir=None):
    """plot star's track and the nearest 5 star's track to check align matching

    plotdir - set this to a directory to override the default location
    where the plots are plotted in the align directory

    """

    if plotdir is None:
        if not os.path.exists(root+'/plots'):
            os.mkdir(root+'/plots')

        if not os.path.exists(root+save_path):
            os.mkdir(root+save_path)

    # read in all stars
    s = starset.StarSet(root + align)
    name = np.array(s.getArray('name'))
    mag = s.getArray('mag') * 1.0
    years = s.getArray('years')[0]
    r = s.getArray('r2d')
    x = s.getArray('x')
    y = s.getArray('y')
    nEpochs = s.getArray('velCnt')
 
    # load polyfit
    # notice this will only read stars that are detected in more than 3 epochs.
    if accel:
        s.loadPolyfit(root + poly, arcsec=1, accel=1, silent=True)
        name_fit = np.array(s.getArray('name'))
        x0 = s.getArray('fitXa.p')
        y0 = s.getArray('fitYa.p')
        x0e = s.getArray('fitXa.perr')
        y0e = s.getArray('fitYa.perr')
        vx = s.getArray('fitXa.v')
        vy = s.getArray('fitYa.v')
        vxe = s.getArray('fitXa.verr')
        vye = s.getArray('fitYa.verr')
        ax = s.getArray('fitXa.a')
        ay = s.getArray('fitYa.a')
        axe = s.getArray('fitXa.aerr')
        aye = s.getArray('fitYa.aerr')
        t0 = s.getArray('fitXa.t0')
    else:
        s.loadPolyfit(root + poly, arcsec=1, silent=True)
        name_fit = np.array(s.getArray('name'))
        x0 = s.getArray('fitXv.p')
        y0 = s.getArray('fitYv.p')
        x0e = s.getArray('fitXv.perr')
        y0e = s.getArray('fitYv.perr')
        vx = s.getArray('fitXv.v')
        vy = s.getArray('fitYv.v')
        vxe = s.getArray('fitXv.verr')
        vye = s.getArray('fitYv.verr')
        t0 = s.getArray('fitXv.t0')

    # load label.dat
    t = Table.read(os.path.join(root,'source_list/label_abs.dat'), format='ascii')
    name_label = t['col1']
    x0_label = t['col3']
    y0_label = t['col4']
    vx_label = t['col7']*0.001
    vy_label = t['col8']*0.001
    if len(t.colnames) == 13:
        t0_label = t['col11']
    else:
        t0_label = t['col15']

    # go through each star
    for star in stars:
        plt.figure(1, figsize=(10,10))

        # check if this star is detected
        i_star = np.where(name==star)[0]
        if len(i_star) == 0:
            continue

        # plot the polynomial fit
        if np.in1d(star, name_fit):
            i_star_fit = np.where(name_fit==star)[0]
            time = np.arange(years.min(),years.max()+1,0.01)
            dt = time-t0[i_star_fit]
            if accel:
                xpos = x0[i_star_fit] + vx[i_star_fit]*dt + 0.5*ax[i_star_fit]*dt**2
                ypos = y0[i_star_fit] + vy[i_star_fit]*dt + 0.5*ay[i_star_fit]*dt**2
                plt.plot(xpos, ypos, 'k--', lw=1, label='accel fit')
            else:
                xpos = x0[i_star_fit] + vx[i_star_fit]*dt
                ypos = y0[i_star_fit] + vy[i_star_fit]*dt
                plt.plot(xpos, ypos, 'k--', lw=1, label='linear fit')

        # plot fit from label.dat
        if plot_label_fit:
            if np.in1d(name[i_star], name_label):
                i_star_fit = np.where(name_label==star)[0]
                time = np.arange(years.min(),years.max()+1,0.01)
                dt = time-t0_label[i_star_fit]
                xpos = x0_label[i_star_fit] + vx_label[i_star_fit]*dt
                ypos = y0_label[i_star_fit] + vy_label[i_star_fit]*dt
                plt.plot(xpos, ypos, 'r--', lw=1, label='label.dat')
            plt.legend(loc='upper left')


        # plot star's track
        markers = ['o', '^', 'd', 'v', 's', '<']
        plt.subplots_adjust(right=0.9)

        # find the nearst 5 stars
        distances = np.sqrt((x[i_star] - x)**2 + (y[i_star] - y)**2)
        i_near = distances.argsort()[1:20]
        star_near = []
        num_star_near = 0
        for ii in i_near:
            star_near.append(name[ii])
            if nEpochs[ii] >= 4:
                num_star_near +=1
            if num_star_near == 5:
                break
                    
        star_near = np.array(star_near)
        #star_near = name[i_near]

        # return the min and max x y for plotting range
        x1_min = np.zeros(len(star_near)+1)
        x1_max = np.zeros(len(star_near)+1)
        y1_min = np.zeros(len(star_near)+1)
        y1_max = np.zeros(len(star_near)+1)

        x1_min[0], x1_max[0], y1_min[0], y1_max[0] = plot_near_1(star, mag[i_star], root, years, markers[0], withLabel=True)



        # plot nearest 5 stars
        num_marker_x = 0
        for ii in range(len(star_near)):
            star_ii = star_near[ii]
            i_star = np.where(name==star_ii)[0]

            # plot polynomial fit
            if np.in1d(star_ii, name_fit):
                i_star_fit = np.where(name_fit==star_ii)[0]
                time = np.arange(years.min(),years.max()+1,0.01)
                dt = time-t0[i_star_fit]
                if accel:
                    xpos = x0[i_star_fit] + vx[i_star_fit]*dt + 0.5*ax[i_star_fit]*dt**2
                    ypos = y0[i_star_fit] + vy[i_star_fit]*dt + 0.5*ay[i_star_fit]*dt**2
                else:
                    xpos = x0[i_star_fit] + vx[i_star_fit]*dt
                    ypos = y0[i_star_fit] + vy[i_star_fit]*dt
                plt.plot(xpos, ypos, 'k--', lw=1)
            # plot fit from label.dat
            if plot_label_fit:
                if np.in1d(star_ii, name_label):
                    i_star_fit = np.where(name_label==star_ii)[0]
                    time = np.arange(years.min(),years.max()+1,0.01)
                    dt = time-t0_label[i_star_fit]
                    xpos = x0_label[i_star_fit] + vx_label[i_star_fit]*dt
                    ypos = y0_label[i_star_fit] + vy_label[i_star_fit]*dt
                    plt.plot(xpos, ypos, 'r--', lw=1)

            if nEpochs[i_star] < 4:
                marker = ''
                num_marker_x +=1
            else:
                j = ii+1-num_marker_x
                while j > len(markers):
                    j -= len(markers)
                marker = markers[j]


            x1_min[ii+1], x1_max[ii+1], y1_min[ii+1], y1_max[ii+1] = plot_near_1(star_ii, mag[i_star], root, years, marker)

        plt.xlim(x1_min.min()-0.1, x1_max.max()+0.1)
        plt.ylim(y1_min.min()-0.1, y1_max.max()+0.1)
        plt.gca().invert_xaxis()

        plt.xlabel('X offset from Sgr A* (arcsec)')
        plt.ylabel('Y offset from Sgr A* (arcsec)')
        plt.gca().xaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
        plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
        if plotdir is None:
            plt.savefig(root+ save_path + '/%s.png' %star, format='png', dpi=300)
        else:
            plt.savefig(os.path.join(plotdir, '%s.png' %star), format='png', dpi=300)
        plt.close(1)
    return


def plot_star(star, accel=False, plot_label_fit=False, save_path='/plots/plotStar/',
       root='./', align='align/align_d_rms_1000_abs_t', poly='polyfit_1_1000/fit'):
    """
    plot star's track with its neibors to see if the star is matched correctly"""
    if not os.path.exists(root+'/plots'):
        os.mkdir(root+'/plots')

    if not os.path.exists(root+save_path):
        os.mkdir(root+save_path)

    # Load new StarSet object if not provided
    s = starset.StarSet(root + align)

    plt.figure(1, figsize=(10,10))
    plot_near_5(s, poly, star, root=root, acc=accel, plot_label_fit=plot_label_fit)
    plt.xlabel('X offset from Sgr A* (arcsec)')
    plt.ylabel('Y offset from Sgr A* (arcsec)')
    plt.gca().xaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    plt.savefig(root+ save_path + '/%s.png' %star, format='png', dpi=300)
    plt.close(1)
    return

def plot_near_5(s, poly, star, root='./', acc=False, plot_label_fit=False):
    """
    plot star's track and the nearest 5 star's track to check align matching"""
    # read in all stars
    name = np.array(s.getArray('name'))
    mag = s.getArray('mag') * 1.0
    years = s.getArray('years')[0]
    r = s.getArray('r2d')
    x = s.getArray('x')
    y = s.getArray('y')
    nEpochs = s.getArray('velCnt')

    # load polyfit
    # notice this will only read stars that are detected in more than 3 epochs.
    if acc:
        s.loadPolyfit(root + poly, arcsec=1, accel=1, silent=True)
        name_fit = np.array(s.getArray('name'))
        x0 = s.getArray('fitXa.p')
        y0 = s.getArray('fitYa.p')
        x0e = s.getArray('fitXa.perr')
        y0e = s.getArray('fitYa.perr')
        vx = s.getArray('fitXa.v')
        vy = s.getArray('fitYa.v')
        vxe = s.getArray('fitXa.verr')
        vye = s.getArray('fitYa.verr')
        ax = s.getArray('fitXa.a')
        ay = s.getArray('fitYa.a')
        axe = s.getArray('fitXa.aerr')
        aye = s.getArray('fitYa.aerr')
        t0 = s.getArray('fitXa.t0')
    else:
        s.loadPolyfit(root + poly, arcsec=1, silent=True)
        name_fit = np.array(s.getArray('name'))
        x0 = s.getArray('fitXv.p')
        y0 = s.getArray('fitYv.p')
        x0e = s.getArray('fitXv.perr')
        y0e = s.getArray('fitYv.perr')
        vx = s.getArray('fitXv.v')
        vy = s.getArray('fitYv.v')
        vxe = s.getArray('fitXv.verr')
        vye = s.getArray('fitYv.verr')
        t0 = s.getArray('fitXv.t0')

    # load label.dat
    t = Table.read('source_list/label_abs.dat', format='ascii')
    name_label = t['col1']
    x0_label = t['col3']
    y0_label = t['col4']
    vx_label = t['col7']*0.001
    vy_label = t['col8']*0.001
    t0_label = t['col11']


    # check if this star is detected
    i_star = np.where(name==star)[0]
    if len(i_star) == 0:
        return

    # plot the polynomial fit
    if np.in1d(star, name_fit):
        i_star_fit = np.where(name_fit==star)[0]
        time = np.arange(years.min(),years.max()+1,0.01)
        dt = time-t0[i_star_fit]
        if acc:
            xpos = x0[i_star_fit] + vx[i_star_fit]*dt + 0.5*ax[i_star_fit]*dt**2
            ypos = y0[i_star_fit] + vy[i_star_fit]*dt + 0.5*ay[i_star_fit]*dt**2
            plt.plot(xpos, ypos, 'k--', lw=1, label='accel fit')
        else:
            xpos = x0[i_star_fit] + vx[i_star_fit]*dt
            ypos = y0[i_star_fit] + vy[i_star_fit]*dt
            plt.plot(xpos, ypos, 'k--', lw=1, label='linear fit')

    # plot fit from label.dat
    if plot_label_fit:
        if np.in1d(name[i_star], name_label):
            i_star_fit = np.where(name_label==star)[0]
            time = np.arange(years.min(),years.max()+1,0.01)
            dt = time-t0_label[i_star_fit]
            xpos = x0_label[i_star_fit] + vx_label[i_star_fit]*dt
            ypos = y0_label[i_star_fit] + vy_label[i_star_fit]*dt
            plt.plot(xpos, ypos, 'r--', lw=1, label='label.dat')
        plt.legend(loc='upper left')


    # plot star's track
    markers = ['o', '^', 'd', 'v', 's', '<']
    plt.subplots_adjust(right=0.9)

    # find the nearst 5 stars
    distances = np.sqrt((x[i_star] - x)**2 + (y[i_star] - y)**2)
    i_near = distances.argsort()[1:20]
    star_near = []
    num_star_near = 0
    for ii in i_near:
        star_near.append(name[ii])
        if nEpochs[ii] >= 4:
            num_star_near +=1
        if num_star_near == 5:
            break
                
    star_near = np.array(star_near)
    #star_near = name[i_near]

    # return the min and max x y for plotting range
    x1_min = np.zeros(len(star_near)+1)
    x1_max = np.zeros(len(star_near)+1)
    y1_min = np.zeros(len(star_near)+1)
    y1_max = np.zeros(len(star_near)+1)

    x1_min[0], x1_max[0], y1_min[0], y1_max[0] = plot_near_1(star, mag[i_star], root, years, markers[0], withLabel=True)


    # plot nearest 5 stars
    num_marker_x = 0
    for ii in range(len(star_near)):
        star_ii = star_near[ii]
        i_star = np.where(name==star_ii)[0]

        # plot polynomial fit
        if np.in1d(star_ii, name_fit):
            i_star_fit = np.where(name_fit==star_ii)[0]
            time = np.arange(years.min(),years.max()+1,0.01)
            dt = time-t0[i_star_fit]
            if acc:
                xpos = x0[i_star_fit] + vx[i_star_fit]*dt + 0.5*ax[i_star_fit]*dt**2
                ypos = y0[i_star_fit] + vy[i_star_fit]*dt + 0.5*ay[i_star_fit]*dt**2
            else:
                xpos = x0[i_star_fit] + vx[i_star_fit]*dt
                ypos = y0[i_star_fit] + vy[i_star_fit]*dt
            plt.plot(xpos, ypos, 'k--', lw=1)
        # plot fit from label.dat
        if plot_label_fit:
            if np.in1d(star_ii, name_label):
                i_star_fit = np.where(name_label==star_ii)[0]
                time = np.arange(years.min(),years.max()+1,0.01)
                dt = time-t0_label[i_star_fit]
                xpos = x0_label[i_star_fit] + vx_label[i_star_fit]*dt
                ypos = y0_label[i_star_fit] + vy_label[i_star_fit]*dt
                plt.plot(xpos, ypos, 'r--', lw=1)

        if nEpochs[i_star] < 4:
            marker = 'x'
            num_marker_x +=1
        else:
            j = ii+1-num_marker_x
            while j > len(markers):
                j -= len(markers)
            marker = markers[j]


        x1_min[ii+1], x1_max[ii+1], y1_min[ii+1], y1_max[ii+1] = plot_near_1(star_ii, mag[i_star], root, years, marker)

    plt.xlim(x1_min.min()-0.1, x1_max.max()+0.1)
    plt.ylim(y1_min.min()-0.1, y1_max.max()+0.1)
    plt.gca().invert_xaxis()
    return

####
# plot track for one star
####
def plot_near_1(star, mag, root, years, marker, withLabel=False):

    # define colors
    cnorm = colors.Normalize(years.min(), years.max()+1)
    cmap = cm.gist_ncar
    linecolors = []
    for ee in range(len(years)):
        linecolors.append(cmap(cnorm(years[ee])))

    # plot points_3_c
    if os.stat(root+'points_3_c/'+star+'.points').st_size == 0:
        time3 = []
    else:
        star_track = np.genfromtxt(root+'points_3_c/'+star+'.points')
        if len(star_track.shape)==1:
            time3 = [star_track[0]]
            idx_color = [list(years).index(time3)]
            x3 = np.array([star_track[1]])*-1.
            y3 = np.array([star_track[2]])
            xe3 = np.array([star_track[3]])
            ye3 = np.array([star_track[4]])
        else:
            time3 = star_track[:,0]
            idx_color = [list(years).index(i) for i in time3]
            x3 = star_track[:,1]*-1.
            y3 = star_track[:,2]
            xe3 = star_track[:,3]
            ye3 = star_track[:,4]

        for i in range(len(x3)):
            plt.errorbar(x3[i], y3[i], xerr=xe3[i], yerr=ye3[i], color=linecolors[idx_color[i]], fmt=marker, ms=5)


    # plot points_2_s
    if os.stat(root+'points_2_s/'+star+'.points').st_size == 0:
        time2 = []
    else:
        star_track = np.genfromtxt(root+'points_2_s/'+star+'.points')
        if len(star_track.shape)==1:
            time2 = [star_track[0]]
            idx_color = [list(years).index(time2)]
            x2 = np.array([star_track[1]])*-1.
            y2 = np.array([star_track[2]])
            xe2 = np.array([star_track[3]])
            ye2 = np.array([star_track[4]])
        else:
            time2 = star_track[:,0]
            idx_color = [list(years).index(i) for i in time2]
            x2 = star_track[:,1]*-1.
            y2 = star_track[:,2]
            xe2 = star_track[:,3]
            ye2 = star_track[:,4]

        time_c = np.setdiff1d(time2, time3)
        idx_c = [list(time2).index(i) for i in time_c]
        for i in idx_c:
            plt.plot(x2[i], y2[i], color=linecolors[idx_color[i]], fillstyle='none', marker=marker, ms=5, ls='none')


    # plot points_1_1000
    if not os.stat(root+'points_1_1000/'+star+'.points').st_size == 0:
        star_track = np.genfromtxt(root+'points_1_1000/'+star+'.points')
        if len(star_track.shape)==1:
            time1 = [star_track[0]]
            idx_color = [list(years).index(time1)]
            x1 = np.array([star_track[1]])*-1.
            y1 = np.array([star_track[2]])
            xe1 = np.array([star_track[3]])
            ye1 = np.array([star_track[4]])
        else:
            time1 = star_track[:,0]
            idx_color = [list(years).index(i) for i in time1]
            x1 = star_track[:,1]*-1.
            y1 = star_track[:,2]
            xe1 = star_track[:,3]
            ye1 = star_track[:,4]

        time_s = np.setdiff1d(time1, time2)
        idx_s = [list(time1).index(i) for i in time_s]

        if marker == 'x':
            plt.plot(x1[idx_s], y1[idx_s], color='grey', alpha=0.5, marker=marker, ms=5, ls='none')
        else:
            plt.plot(x1[idx_s], y1[idx_s], color='grey', alpha=0.5, marker=marker, ms=5, mec='none', ls='none')

    if withLabel:
        plt.annotate(star + ': K=%.1f' %mag, (x1[x1.argmax()], y1[x1.argmax()]), fontweight='bold')
    else:
        plt.annotate(star + ': K=%.1f' %mag, (x1[x1.argmax()], y1[x1.argmax()]))


    if withLabel:
        plt.plot([], [], color='r', marker='o', ms=5, mew=0.5, ls='none', fillstyle='none', label='confusion')
        plt.plot([], [], color='grey', alpha=0.3, marker='o', ms=5, mew=0.5, ls='none', label='speckle edge')
        plt.legend()

        # make year annotation
        space = 1./int(years.max()-years.min())
        previous_year = 0
        n_year = 0
        for i in range(len(years)):
            year = years[i]
            if int(previous_year) != int(year):
                plt.annotate(str(int(year)), (1.03, 0 + n_year*space), color=linecolors[i],
                        xycoords='axes fraction', fontsize=15)
                previous_year = year
                n_year += 1

    return x1.min(), x1.max(), y1.min(), y1.max()
 
def compare_align(root1, root2, stars, acc=False):
    """
    plot the star's track in two different align to see which align is better
    plot this star and any star that's around this star"""
    align='align/align_d_rms_1000_abs_t'
    poly='polyfit_1_1000/fit'

    # load the first align
    s1 = starset.StarSet(root1 + align)
    s1.loadPolyfit(root1 + poly, arcsec=1, silent=True)
    s1.loadPolyfit(root1 + poly, arcsec=1, accel=1, silent=True)

    s2 = starset.StarSet(root2 + align)
    s2.loadPolyfit(root2 + poly, arcsec=1, silent=True)
    s2.loadPolyfit(root2 + poly, arcsec=1, accel=1, silent=True)

    # make the plots
    if not os.path.exists('compare_plot'):
        os.mkdir('compare_plot')

    for star in stars:
        plt.subplots(1,2, figsize=(20,10), sharex=True, sharey=True)
        ax1 = plt.subplot(1,2,1)
        plot_near_5(s1, star, root=root1, acc=acc)
        plt.xlabel('X offset from Sgr A* (arcsec)')
        plt.ylabel('Y offset from Sgr A* (arcsec)')
        plt.title(root1+star)
        ax2 = plt.subplot(1,2,2)
        plot_near_5(s2, star, root=root2, acc=acc)
        plt.title(root2+star)
        ax2.set_xticks([])
        ax2.set_yticks([])

        plt.savefig('compare_plot/%s.png' %star, format='png', dpi=300)
        plt.close()
