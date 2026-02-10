from astropy.io import fits
from astropy.time import Time
import matplotlib.pyplot as plt
import matplotlib.patheffects as mpe
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.offsetbox import AnchoredText
from matplotlib.offsetbox import (
    AnchoredOffsetbox, AuxTransformBox, DrawingArea, TextArea, VPacker)
import matplotlib.lines as lines
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib import rc
from matplotlib.patches import Rectangle, Ellipse
from matplotlib.dates import DateFormatter
# from matplotlib.dates import YearLocator, DateFormatter
import matplotlib.dates as mdates
import numpy as np
import os
import pandas as pd
import pdb
import textalloc as ta


class AnchoredEllipse(AnchoredOffsetbox):
    def __init__(
        self,
        transform,
        width,
        height,
        angle,
        loc,
        pad=0.5,
        borderpad=0.2,
        prop=None,
        frameon=True,
        color='blue',
    ):
        '''
        Draw an ellipse the size in data coordinate of the given axes.

        pad, borderpad in fraction of the legend font size (or prop).
        '''
        self._box = AuxTransformBox(transform)
        self.ellipse = Ellipse(
            xy=(0, 0),
            width=width,
            height=height,
            angle=angle,
            color='grey',
            ec='black',
            fill=True,
            hatch='\\\\\\\\\\\\',
        )
        self._box.add_artist(self.ellipse)
        super(AnchoredEllipse, self).__init__(
            loc,
            pad=pad,
            borderpad=borderpad,
            child=self._box,
            prop=prop,
            frameon=frameon,
        )



def PlotComp(
    mfitcomps,
    comp_color='blue',
    comp_lw=1.3,
    comp_labelling='textalloc',
    comp_label_pos=None,
):
    '''
    # Purpose:
        Function to overplot Gaussian modelfit components over an image
        that is already specified. Ultimately the script is used by the
        plot_image_comps script, that defines that axis that this script adds
        the component plotting to. Includes new functions (circle, ellipse, or
        cros) to draw modelfit components.
    
    # Args:
        mfitcomps (Components type object): ReadComp('file.mfitid') from
          read_components script.
        comp_color (str): color for components to plot. Default is 'blue'.
        comp_lw (float): line width for components to plot. Default is 1.3.
        comp_labelling (str): determines how component labels are added to the
          plot. Options are 'textalloc' (default), which uses the textalloc
          package (needs to be installed), 'auto', which puts the labels based
          on the component positions. For manual labelling, see comp_label_pos.
        comp_label_pos (list): provide the coordinates in [x,y] for each model-
          fit component in a list. Unit should be in mas.
    
    # Returns:
        Adds plotting of modelfit components to existing axis object from
          matplotlib.
    '''
    
    # outline=mpe.withStroke(linewidth=1, foreground='black')
    outline=mpe.withStroke(linewidth=min(1,0.5*comp_lw), foreground='black')
    
    def circros(x, y, r):
        circ = mpl.patches.Circle(
            xy=(x,y),
            radius=r,
            fill=False,
            color=comp_color,
            linewidth=comp_lw,
            zorder=4,
            path_effects=[outline],
        )
        ax=plt.gca()
        ax.add_artist(circ)
        plt.vlines(
            x,
            y+r,
            y-r,
            linestyle='-',
            color=comp_color,
            linewidth=comp_lw,
            zorder=4,
            path_effects=[outline],
        )
        plt.hlines(
            y,
            x+r,
            x-r,
            linestyle='-',
            color=comp_color,
            linewidth=comp_lw,
            zorder=4,
            path_effects=[outline],
        )

    def elcros(x, y, a, b, phi):
        el = mpl.patches.Ellipse(
            xy=(x,y),
            width=a,
            height=b,
            angle=phi,
            color=color,
            fill=False,
            hatch='\\',
            zorder=4,
            linewidth=comp_lw,
        )
        ax=plt.gca()
        ax.add_artist(el)
        plt.vlines(
            x,
            y + b/2,
            y - b/2,
            linestyle='-',
            color=comp_color,
            linewidth=comp_lw,
            zorder=4,
            path_effects=[outline],
        )
        plt.hlines(
            y,
            x + a/2,
            x - a/2,
            linestyle='-',
            color=comp_color,
            linewidth=comp_lw,
            zorder=4,
            path_effects=[outline],
        )

    def cros(x, y, r):
        plt.vlines(
            x,
            y + r,
            y - r,
            linestyle='-',
            color=comp_color,
            linewidth=comp_lw,
            zorder=4,
            path_effects=[outline],
        )
        plt.hlines(
            y,
            x + r,
            x - r,
            linestyle='-',
            color=comp_color,
            linewidth=comp_lw,
            zorder=4,
            path_effects=[outline],
        )
    
    for comp in mfitcomps:
        if comp.axratio == 1:
            if comp.major >= comp.dlim:
                circros(comp.ra,comp.dec,comp.a)
            else:
                cros(comp.ra,comp.dec,0.2)
        else:
            if comp.major >= comp.dlim/10:
                elcros(comp.ra,comp.dec,comp.a,comp.b,comp.phi)
            else:
                cros(comp.ra,comp.dec,comp.a)
    
    if comp_label_pos != None:
        # Use manual component label positions given in rel. RA and Dec
        # comp_label_pos = kwargs.get('comp_label_pos', False)
        for i, comp in enumerate(mfitcomps):
            plt.text(
                comp_label_pos[i][0],
                comp_label_pos[i][1],
                str(comp.name),
                color=comp_color,
                size=15,
                weight='bold',
            )
            plt.plot(
                [comp.ra, comp_label_pos[i][0]],
                [comp.dec, comp_label_pos[i][1]],
                color=comp_color,
                lw=comp_lw,
            )
    
    elif comp_labelling == 'textalloc':
        '''
        This piece of code uses the textalloc package to put component labels
        automatically without overlap for clarity. The old annotations have been
        commented out above.
        '''
        text = []
        x_pos = []
        y_pos = []
        size = []
        for comp in mfitcomps:
            if comp.name != 'None':
                do_text = True
                text.append(str(comp.name))
                x_pos.append(comp.ra)
                y_pos.append(comp.dec)
                size.append(comp.a)
            else:
                do_text = False
        
        if do_text == True:
            vermillion = '#E34234'
            ax = plt.gca()
            try:  # catch exception related to keyword changes in matplotlib
                ta.allocate(
                    ax=ax,
                    x=x_pos,
                    y=y_pos,
                    text_list=text,
                    x_scatter=np.array(x_pos)+10*np.array(size),
                    y_scatter=np.array(y_pos)+10*np.array(size),
                    textsize=15,
                    textcolor=comp_color,
                    linecolor=vermillion,
                    weight='bold',
                    avoid_label_lines_overlap=True,
                    min_distance=0.035,
                    # max_distance=0.05,
                    # # margin=0.0001,
                    # # avoid_crossing_label_lines=True,
                    # # x_scatter=x_pos,
                    # # y_scatter=y_pos,
                    # min_distance=0.0020,
                    # max_distance=0.0030,
                    # # direction='east',
                    # path_effects=[outline],
                    # zorder=3,
                )
            except TypeError:
                ta.allocate(
                    ax=ax,
                    x=x_pos,
                    y=y_pos,
                    text_list=text,
                    x_scatter=np.array(x_pos)+10*np.array(size),
                    y_scatter=np.array(y_pos)+10*np.array(size),
                    textsize=15,
                    textcolor=comp_color,
                    linecolor=vermillion,
                    fontweight='bold',
                    avoid_label_lines_overlap=True,
                    min_distance=0.035,
                )
    
    elif comp_labelling == 'auto':
        # Determine position of component labels based on the component positions
        for comp in mfitcomps:
            if comp.axratio == 1:
                if comp.major >= comp.dlim:
                    plt.text(
                        comp.ra - 0.5*comp.a,
                        comp.dec + 1.5*comp.a,
                        str(comp.name),
                        color=comp_color,
                        size=15,
                        weight='bold',
                    )
                else:
                    plt.text(
                        comp.ra - 0.3,
                        comp.dec + 0.3,
                        str(comp.name),
                        color=comp_color,
                        size=15,
                        weight='bold',
                    )
            else:
                if comp.major >= comp.dlim/10:
                    plt.text(
                        comp.ra - 1.5*comp.a,
                        comp.dec + 1.5*comp.a,
                        str(comp.name),
                        color=comp_color,
                        size=15,
                        weight='bold',
                    ) 
                else:
                    plt.text(
                        comp.ra - 0.3,
                        comp.dec + 0.3,
                        str(comp.name),
                        color=comp_color,
                        size=15,
                        weight='bold',
                    )


def plot_image_comps(
    components,
    IMAP_file,
    rms,
    plt_xlim,
    plt_ylim,
    do_shift_max=False,
    rms_factor=4,
    bmin=None,
    bmax=None,
    beam_angle=None,
    srcname_fontsize=14,
    epochname_fontsize=14,
    freq_fontsize=14,
    comp_color='blue',
    comp_lw=1.3,
    comp_labelling='textalloc',
    comp_label_pos=None,
    outfile_path=None,
    show=False,
    **kwargs
):
    """
    # Purpose:
        Small script to plot a fits image with Gaussian modelfit components.
    
    # Args:
        components (Components type object): ReadComp('file.mfitid') from
          read_components script. Incorporated by importing plot_components/
        IMAP_file (str): path to .fits image file.
        plt_xlim (list): minimum and maximum value [mas] in RA to be plotted.
        plt_ylim (list): minimum and maximum value [mas] in DEC to be plotted.
        do_shift_max (bool): determines whether or not to shift the map to the
          pixel with maximum intensity.
        rms_factor (int): factor to multiply the rms level with to determine
          the first contour to be displayed. Default is 4.
        srcname_fontsize (int): font size for source name in plot (default 14).
        epochname_fontsize (int): font size for epoch in plot (default 14).
        freq_fontsize (int): font size for frequency in plot (default 14).
        comp_color (str): color for components to plot. Default is 'blue'.
        comp_lw (float): line width for components to plot. Default is 1.3.
        comp_labelling (str): determines how component labels are added to the
          plot. Options are 'textalloc' (default), which uses the textalloc
          package (needs to be installed), 'auto', which puts the labels based
          on the component positions. For manual labelling, see comp_label_pos.
        comp_label_pos (list): gives the coordinates in [x,y] for each model-
        fit component in a list. Unit should be in mas.
        **kwargs (dict): some parameters that are now obsolete, but font size
           parameters (see below) might still be set manually.

    # Returns:
        Saves a .pdf and .png file with the desired map.
    """
    
    plt_xmin = plt_xlim[0]
    plt_xmax = plt_xlim[1]
    plt_ymin = plt_ylim[0]
    plt_ymax = plt_ylim[1]
    
    if not 'beam_xpos' in kwargs.keys():
        beam_xpos = plt_xmin*0.8
    else:
        beam_xpos = kwargs.get('beam_xpos', False)
    if not 'beam_ypos' in kwargs.keys():
        beam_ypos = plt_ymin*0.8
    else:
        beam_ypos = kwargs.get('beam_ypos', False)
    
    if not 'srcname_xpos' in kwargs.keys():
        srcname_xpos = plt_xmin*0.85
    else:
        srcname_xpos = kwargs.get('srcname_xpos', False)
    if not 'srcname_ypos' in kwargs.keys():
        srcname_ypos = plt_ymax*0.85
    else:
        srcname_ypos = kwargs.get('srcname_ypos', False)
    
    if not 'epochname_xpos' in kwargs.keys():
        epochname_xpos = plt_xmax*0.5
    else:
        epochname_xpos = kwargs.get('epochname_xpos', False)
    if not 'epochname_ypos' in kwargs.keys():
        epochname_ypos = plt_ymax*0.85
    else:
        epochname_ypos = kwargs.get('epochname_ypos', False)

    if not 'freq_xpos' in kwargs.keys():
        freq_xpos = plt_xmax*0.5
    else:
        freq_xpos = kwargs.get('freq_xpos', False)
    if not 'freq_ypos' in kwargs.keys():
        freq_ypos = plt_ymax*0.7
    else:
        freq_ypos = kwargs.get('freq_ypos', False)

    if 'overplot_RA' in kwargs.keys():
        do_overplot = True
        overplot_RA = kwargs.get('overplot_RA', False)
    else:
        do_overplot = False
    if 'overplot_RA_err' in kwargs.keys():
        overplot_RA_err = kwargs.get('overplot_RA_err', False)
    if 'overplot_Dec' in kwargs.keys():
        overplot_Dec = kwargs.get('overplot_Dec', False)
    if 'overplot_Dec_err' in kwargs.keys():
        overplot_Dec_err = kwargs.get('overplot_Dec_err', False)
    
    # Load up the fits images
    IMAP_data = fits.getdata(IMAP_file)

    # Extracting information from the fits header
    hdulist = fits.open(IMAP_file)
    hdu0 = hdulist[0]
    xdim = hdu0.header.cards['naxis1'][1]  # the dimensions of the map
        # in pixels
    ydim = hdu0.header.cards['naxis2'][1]  # the dimensions of the map
        # in pixels
    delt = round(hdu0.header.cards['cdelt2'][1]*3600*1000,10)  # mas per
        # pixel (taken from the pixel increment in DEC axis because it's
        # positive)
    freq = round((hdu0.header.cards['crval3'][1]*1e-6)/1e3, 1)  # frequency
        # in GHz rounded to one decimal
    if not 'srcname' in kwargs.keys():
        srcname = hdu0.header.cards['object'][1]  # name of source
    else:
        srcname = kwargs.get('srcname', False)
    epochname = hdu0.header.cards['date-obs'][1]  # observing epoch
    try:
        bmin = hdu0.header.cards['bmin'][1]*3.6e6  # bmin in mas
        bmaj = hdu0.header.cards['bmaj'][1]*3.6e6  # bmaj in mas
        beam_angle = hdu0.header.cards['bpa'][1]  # beam angle in degrees
    except KeyError:
        print('Could not extract beam information from header, needs to be provided.')
        bmin = bmin
        bmaj = bmaj
        beam_angle = beam_angle
    crpix1 = hdu0.header.cards['crpix1'][1]  # central pixel in x
    crpix2 = hdu0.header.cards['crpix2'][1]  # central pixel in y
    hdulist.close()
    if hdulist:
        del hdulist

    # Create the map grid with physical size
    xp = np.zeros((xdim, ydim), float)
    yp = np.zeros((xdim, ydim), float)
    
    if do_shift_max == True:
        # Locate the maximum of the total intensity
        IMAP_max = 0.0
        j_max = 0
        i_max = 0
        for i in range(0, xdim):
            for j in range(0, ydim):
                if IMAP_data[0,0,j,i] > IMAP_max:
                    IMAP_max = IMAP_data[0,0,j,i]
                    j_max = j
                    i_max = i
        # Shift coordinates of the map so that the maximum total intensity is
        # located at the origin
        for i in range(0, xdim):
            xp[:,i] = (i_max - i)*delt
        for j in range(0, ydim):
            yp[j,:] = (j - j_max)*delt
            
        xmin = xp[0,0]
        xmax = xp[0,xdim-1]
        ymin = yp[ydim-1,0]
        ymax = yp[0,0]
        
        shift_x = i_max - xdim/2.
        shift_y = ydim/2. - j_max
        
    elif 'shift' in kwargs.keys():
        shift = kwargs.get('shift', False)
        j_max = ydim/2. - shift[1]/delt
        i_max = xdim/2. + shift[0]/delt
        
        # Locate the maximum of the total intensity
        IMAP_max = 0.0
        for i in range(0, xdim):
            for j in range(0, ydim):
                if IMAP_data[0,0,j,i] > IMAP_max:
                    IMAP_max = IMAP_data[0,0,j,i]
        for i in range(0, xdim):
            xp[:,i] = (i_max - i)*delt
        for j in range(0, ydim):
            yp[j,:] = (j - j_max)*delt
        
        shift_x = shift[0]/delt
        shift_y = shift[1]/delt
        
    else:
        j_max = crpix2
        i_max = crpix1
        # j_max = ydim/2.
        # i_max = xdim/2.
        
        # Locate the maximum of the total intensity
        IMAP_max = 0.0
        for i in range(0, xdim):
            for j in range(0, ydim):
                if IMAP_data[0,0,j,i] > IMAP_max:
                    IMAP_max = IMAP_data[0,0,j,i]
        for i in range(0, xdim):
            xp[:,i] = (i_max - i)*delt
        for j in range(0, ydim):
            yp[j,:] = (j - j_max)*delt
                
        xmin = xp[0,0]
        xmax = xp[0,xdim-1]
        ymin = yp[ydim-1,0]
        ymax = yp[0,0]
        
        shift_x = i_max - xdim/2.
        shift_y = ydim/2. - j_max

    # print('Shift in RA: {0:d} px'.format(int(shift_x)))
    # print('Shift in Dec: {0:d} px'.format(int(shift_y)))
    
    # Rotate the beam angle so that it runs negative clock wise from North
    beam_angle_rot = (-1.0)*beam_angle  # required rotation for Ellipse

    fig = plt.figure(figsize=(7.3,7), dpi=250)  # size of the image in inches

    plt.xlim(plt_xmin, plt_xmax)
    plt.ylim(plt_ymin, plt_ymax)

    plt.xlabel('RA [mas]', fontsize=13)
    plt.ylabel('DEC [mas]', fontsize=13)

    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)

    ax = plt.gca()
    if rms < IMAP_max:
        firstcont = rms*rms_factor/IMAP_max*100
        # percentage of the first cont set at rms_factor*sigma,
        # then doubling the following ones in percentage
    else:
        print('! Warning: provided rms is higher than peak intensity! That '
            'should be checked. Take 10 percent as lowest contour.')
        firstcont = 10
    lastcont = 90  # percent of the peak to set the last contour

    Ilevs = np.array([0.0]*30)  # initialize the array with zero values

    count = 0
    Ilevs[0] = firstcont 
    for q in range(0,20):                
        Ilevs[count+1] = Ilevs[count]*2
        count = count + 1
        if Ilevs [count] >= lastcont:
            Ilevs[count] = lastcont
            break

    Ilevs = Ilevs [Ilevs != 0.0]  # Ilevs now contains only contours values

    Ilevsfrac = Ilevs/100
    contours = Ilevsfrac*IMAP_max

    # Make an overplot of the convolving beam (old)
    # ellipse = mpl.patches.Ellipse(
        # xy=( beam_xpos, beam_ypos ),
        # width=bmin, height=bmaj,
        # angle=beam_angle_rot, ec= 'black',
        # color= 'grey', fill=True,
        # hatch='\\\\\\\\\\\\'
    # )
    # ax.add_artist(ellipse)

    # NEW beam plotter
    if not any(x == None for x in (bmaj, bmin, beam_angle)):
        ae = AnchoredEllipse(
            ax.transData,
            width=bmin,
            height=bmaj,
            angle=beam_angle_rot,
            loc='lower left',
            pad=0.8,
            borderpad=0.3,
            frameon=False,
            prop=dict(),
        )
        ax.add_artist(ae)
    else:
        print('! Warning: could not plot beam because no info was provided.')

    # Annotate source name (old)
    # plt.annotate(
        # srcname,
        # xy=(srcname_xpos, srcname_ypos),
        # color='black',
        # fontsize=srcname_fontsize
    # )

    # Annotate source name (NEW, automatically adjusted to figure)
    at_1 = AnchoredText(
        srcname,
        loc='upper left',
        pad=0.8,
        borderpad=0.2,
        frameon=False,
        prop=dict(size=srcname_fontsize, color='black'),
    )
    ax.add_artist(at_1)

    # Annotate epoch name (old)
    # plt.annotate(epochname, xy=(epochname_xpos, epochname_ypos),
                 # color='black', fontsize=epochname_fontsize)
     
    # Annotate epoch name and frequency (NEW, automatically adjusted to figure)
    at_2 = AnchoredText(
        epochname+'\n\n'+str(freq)+' GHz',
        loc='upper right',
        pad=0.8,
        borderpad=0.2,
        frameon=False,
        prop=dict(size=epochname_fontsize, color='black'),
    )
    ax.add_artist(at_2)

    # Annotate observing frequency (old)
        # plt.annotate(
        # str(freq)+' GHz',
        # xy=(freq_xpos, freq_ypos),
        # color='black',
        # fontsize=freq_fontsize
    # )

    # List with total intensity contour levels
    levels_listed = np.array(['{:0.2f}'.format(x) for x in Ilevs])  

    Ipeak=np.around(IMAP_max, decimals=3)  # list total intensity peaks

    # Plot contours
    ax = plt.gca()
    try:
        ax.contour(
            xp,
            yp,
            IMAP_data[0,0,:,:],
            levels=contours,
            colors='grey',
            linewidths=1,
        )
    except ValueError:
        print(contours)
        ax.contour(
            xp,
            yp,
            IMAP_data[0,0,:,:],
            levels=contours,
            colors='grey',
            linewidths=1,
    )

    # List contour levels on top of image
    plt.text(
        -0.01,
        1.05,
        ('Contours: '+', '.join(levels_listed[0:4])+'\n'+', '
        .join(levels_listed[4:])+'$\%$ of '+ str(Ipeak)+' Jy/beam '),
        multialignment='left',
        fontsize=14,
        transform=ax.transAxes,
    )
    
    if components != None:
        # Plot modelfit components overlaid
        for i, comp in enumerate(components):
            # shift components by the same amount as the map (+1 pixel in RA
            # due to difmap not counting the first row of values
            if comp.is_shifted == False:
                components[i] = components[i].shift(
                    (shift_x+1)*delt,
                    shift_y*delt,
                )
            
        PlotComp(components, comp_color=comp_color, comp_lw=comp_lw,
                 comp_labelling=comp_labelling, comp_label_pos=comp_label_pos)
    
    if do_overplot == True:
        ### Overplot cross (for example for GAIA position) ###
        RA_line_x = np.linspace(overplot_RA-overplot_RA_err,
                                overplot_RA+overplot_RA_err, 10)
        RA_line_y = np.repeat(overplot_Dec, 10)
        Dec_line_y = np.linspace(overplot_Dec-overplot_Dec_err,
                                 overplot_Dec+overplot_Dec_err, 10)
        Dec_line_x = np.repeat(overplot_RA, 10)
        ax.plot(RA_line_x, RA_line_y, color='orange', linewidth=3.0,
                label='GAIA position', zorder=4)
        ax.plot(Dec_line_x, Dec_line_y, color='orange', linewidth=3.0, zorder=4)
        plt.legend(loc='lower right')

    ### Save the plot ###
    if show == True:
        plt.show()
    fig.savefig(outfile_path + IMAP_file[0:-5].rsplit('/', 1)[-1]+'_map.png', dpi=300, bbox_inches="tight")
    fig.savefig(outfile_path + IMAP_file[0:-5].rsplit('/', 1)[-1]+'_map.pdf')
    plt.close(fig)



def plot_grouped_by_freq(df, y_col, y_err_col, freq_tolerance=1E9, y_label='',
                         outfile_path='./', show=False, SBR_thresh=4,
                         plot_line=False):
    """
    Plots grouped data by frequency and saves each group as a PDF.
    
    Parameters:
        df : pandas.DataFrame
            Must contain 'Epoch', 'Freq [Hz]', y_col, and y_err_col columns.
        y_col : str
            Name of the column for y-axis values.
        y_err_col : str
            Name of the column for y-axis errors.
        y_label : str
            Label for the y-axis and also used in output file name.
        output_dir : str
            Directory to save plots.
        freq_tolerance : float
            Maximum difference in frequency (Hz) to consider as same group.
    """
    os.makedirs(outfile_path, exist_ok=True)
    
    # Convert epoch to datetime and MJD
    df = df.copy()
    df['Epoch'] = pd.to_datetime(df['Epoch'])
    # df['MJD'] = df['Epoch'].dt.to_julian_date() - 2400000.5
    df['MJD'] = Time(df['Epoch']).mjd# - 2400000.5
    
    # Unique frequencies sorted
    freqs = np.sort(df['Freq [Hz]'].unique())
    grouped_freqs = []
    
    # Group frequencies within tolerance
    while len(freqs) > 0:
        f0 = freqs[0]
        group = freqs[np.isclose(freqs, f0, atol=freq_tolerance)]
        grouped_freqs.append(group)
        freqs = freqs[~np.isclose(freqs, f0, atol=freq_tolerance)]
    
    bluish_green = '#009E73'
    vermillion = '#E34234'
    colors = {
        '2.3':bluish_green,
        '4.5':'black',
        '7.8':'orange',
        '15.1':'blue',
        '22.2':vermillion
        }
    
    # lolims = np.array([0, 0, 1, 0, 1, 0, 0, 0, 1, 0], dtype=bool)
    # uplims = np.array([0, 1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=bool)
    
    # Iterate over frequency groups
    for group in grouped_freqs:
        
        fig, ax1 = plt.subplots(figsize=(8,5))
        
        df_group = df[df['Freq [Hz]'].isin(group)]
        freq_lowest = np.min(group)

        keys = [float(key) for key in colors.keys()]
        diffs = np.abs(np.array(keys - freq_lowest/1E9))
        if np.any(diffs <= freq_tolerance):
            closest_key = keys[np.argmin(diffs)]
            color = colors[str(closest_key)]
        else:
            color = '#1f77b4'
        
        # Implement plotting lower and upper limits
        # lolims = []
        uplims = []
        for i, y_err in enumerate(df_group[y_err_col]):
            if np.isnan(y_err) == True:
                uplims.append(1)
            else:
                uplims.append(0)
        uplims = np.array(uplims, dtype=bool)
        # print(uplims)
        
        ax1.errorbar(
            df_group["MJD"],
            df_group[y_col],
            yerr=df_group[y_err_col],
            uplims=uplims,
            # lolims=lolims,
            fmt='o', capsize=3,
            markeredgecolor='black',
            markeredgewidth=0.5,
            color=color,
            label=f"{freq_lowest/1E9:.1f} GHz"
            )
        
        if y_col == 'SBR':
            plt.axhline(1, color="gray", linestyle="--", linewidth=2)
            plt.axhline(SBR_thresh, color="gray", linestyle=":", linewidth=1.5)
            plt.axhline(1/SBR_thresh, color="gray", linestyle=":", linewidth=1.5)
        
        if "Kin. fit slope" in df.columns:
            if y_col == 'Sep. [mas]' and plot_line == True:
                mjds = np.linspace(np.min(df_group["MJD"]), np.max(df_group["MJD"]), 1000)
                m = df_group["Kin. fit slope"].to_numpy()[0]
                m_err = df_group["Kin. fit slope err"].to_numpy()[0]
                if np.isinf(m_err) == False:
                    label = f'Fit with slope {m*365:.3f} +/- {m_err*365:.3f} mas/yr'
                else:
                    label = f'Fit with slope {m*365:.3f} mas/yr'
                    
                b = df_group["Kin. fit y-int."].to_numpy()[0]
                b_err = df_group["Kin. fit y-int. err"].to_numpy()[0]
                ax1.plot(
                    mjds,
                    mjds*m+b,
                    color='gray',
                    ls='dashed',
                    linewidth=1.5,
                    label=label
                    )
        
        ax1.set_xlabel("MJD", fontsize=17)
        ax1.set_ylabel(y_label, fontsize=17)
        # ax1.set_xticklabels(ax1.get_xticklabels(), fontsize=16)
        # ax1.set_yticklabels(ax1.get_yticklabels(), fontsize=16)
        # ax1.tick_params(axis="x", labelsize=16)
        # ax1.tick_params(axis="y", labelsize=16)
        # plt.title(f"{y_label} vs Time (Group: {freq_lowest/1E9:.1f} GHz)")
        
        if len(df_group) == 1:
            ax1.set_xlim(df_group["MJD"].values[0]-185,df_group["MJD"].values[0]+185)
        
        # Top axis with years
        ax2 = ax1.twiny()
        ax2.set_xlim(ax1.get_xlim())

        # Get min/max MJD in view
        mjd_min, mjd_max = ax1.get_xlim()

        # Convert to datetime range
        start_year = pd.to_datetime(mjd_min + 2400000.5, origin="julian", unit="D").year
        end_year = pd.to_datetime(mjd_max + 2400000.5, origin="julian", unit="D").year

        # Compute timespan
        timespan = end_year - start_year

        # Choose step size based on timespan
        if timespan > 20:
            step = 4
        elif timespan > 10:
            step = 2
        else:
            step = 1

        # Generate Jan 1 of each year in range with step
        year_starts = pd.date_range(f"{start_year}-01-01", f"{end_year}-01-01", freq=pd.DateOffset(years=step))
        
        # Convert back to MJD
        year_mjds = (year_starts.to_julian_date() - 2400000.5).values

        # Apply to axis
        ax2.set_xticks(year_mjds)
        # ax2.set_xticks(year_mjds, fontsize=16)
        # ax2.set_xticklabels(year_starts.year)
        ax2.set_xticklabels(year_starts.year.astype(str))
        ax2.set_xlabel("Year", fontsize=17)
        
        for label in ax2.get_xticklabels():
            label.set_rotation(40)
            label.set_horizontalalignment("left")
        
        handles, labels = ax1.get_legend_handles_labels()
        ax1.legend(handles, labels, loc="best",# title="Frequency group",
                   title_fontsize=14, fontsize=14)
        
        plt.tight_layout()
        
        # Build filename
        safe_label = y_label.replace(" ", "_")
        if safe_label == 'Separation_[mas]':
            safe_label = 'Separation'
        
        if show == True:
            plt.show()
        filename = f"{outfile_path}/{freq_lowest/1E9:.1f}GHz_{safe_label}.pdf"
        fig.savefig(filename[:-4]+'.png', dpi=300, bbox_inches="tight")
        fig.savefig(filename)
        plt.close()
        
        print(f"Saved: {filename}")
    
    if y_col == "SBR" or y_col == 'FR' or y_col == 'Sep. [mas]':
        print('\n')
        # if y_col == 'Sep. [mas]':
        #     y_col_ = 'Sep.'
        # else:
        #     y_col_ = y_col
        # print('# Plotting all data for ' + y_col_ + ' #')
        # Plotting all data together for one quantity #
        fig, ax3 = plt.subplots(figsize=(8,5))
        for group in grouped_freqs:
            df_group = df[df['Freq [Hz]'].isin(group)]
            freq_lowest = np.min(group)
            keys = [float(key) for key in colors.keys()]
            diffs = np.abs(np.array(keys - freq_lowest/1E9))
            if np.any(diffs <= freq_tolerance):
                closest_key = keys[np.argmin(diffs)]
                color = colors[str(closest_key)]
            else:
                color = '#1f77b4'
            
            # Implement plotting lower and upper limits
            lolims = np.zeros_like(df_group[y_err_col])
            uplims = np.zeros_like(df_group[y_err_col])
            if y_col == "SBR":
                for i, lim in enumerate(df_group['SBR is limit']):
                    if lim == 'upper':
                        uplims[i] = 1
                        # print(type(df_group[y_err_col]))
                        # input()
                        # df_group[y_err_col][i] = 1
                        # df_group.at[i, y_err_col] = 1
                    elif lim == 'lower':
                        lolims[i] = 1
                        # df_group.at[i, y_err_col] = 1
            uplims = np.array(uplims, dtype=bool)
            lolims = np.array(lolims, dtype=bool)
            
            y = np.array(df_group[y_col])
            yerrs = np.array(df_group[y_err_col])
            
            y_err = np.array([yerr if np.isnan(yerr) == False else y[i]/4 for i, yerr in enumerate(yerrs)])
            
            ax3.errorbar(
                df_group["MJD"],
                df_group[y_col],
                yerr=y_err,
                uplims=uplims,
                lolims=lolims,
                fmt='o', capsize=3,
                markeredgecolor='black',
                markeredgewidth=0.5,
                color=color,
                label=f"{freq_lowest/1E9:.1f} GHz"
                )

            # overplot kinematic fits for all frequencies
            if "Kin. fit slope" in df.columns:
                if y_col == 'Sep. [mas]' and plot_line == True:
                    mjds = np.linspace(np.min(df_group["MJD"]), np.max(df_group["MJD"]), 1000)
                    m = df_group["Kin. fit slope"].to_numpy()[0]
                    m_err = df_group["Kin. fit slope err"].to_numpy()[0]
                    if np.isinf(m_err) == False:
                        label = f'Fit with slope {m*365:.3f} +/- {m_err*365:.3f} mas/yr'
                    else:
                        if np.isinf(m) == False:
                            label = f'Fit with slope {m*365:.3f} mas/yr'
                    
                    b = df_group["Kin. fit y-int."].to_numpy()[0]
                    b_err = df_group["Kin. fit y-int. err"].to_numpy()[0]
                    ax3.plot(
                        mjds,
                        mjds*m+b,
                        color=color,
                        ls='dashed',
                        linewidth=1.5,
                        label=label
                        )

        if y_col == 'SBR':
            ax3.axhline(1, color="gray", linestyle="--", linewidth=2)
            ax3.axhline(SBR_thresh, color="gray", linestyle=":", linewidth=1.5)
            ax3.axhline(1/SBR_thresh, color="gray", linestyle=":", linewidth=1.5)
        
        ax3.set_xlabel("MJD", fontsize=17)
        ax3.set_ylabel(y_label, fontsize=17)
        # ax3.set_xticklabels(ax3.get_xticklabels(), fontsize=16)
        # ax3.set_yticklabels(ax3.get_yticklabels(), fontsize=16)
        # ax3.tick_params(axis="x", labelsize=16)
        # ax3.tick_params(axis="y", labelsize=16)
        # plt.title(f"{y_label} vs Time (Group: {freq_lowest/1E9:.1f} GHz)")
        
        # Top axis with years
        ax4 = ax3.twiny()
        ax4.set_xlim(ax3.get_xlim())

        # Get min/max MJD in view
        mjd_min, mjd_max = ax3.get_xlim()

        # Convert to datetime range
        start_year = pd.to_datetime(mjd_min + 2400000.5, origin="julian", unit="D").year
        end_year = pd.to_datetime(mjd_max + 2400000.5, origin="julian", unit="D").year
        
        # Compute timespan
        timespan = end_year - start_year

        # Choose step size based on timespan
        if timespan > 20:
            step = 4
        elif timespan > 10:
            step = 2
        else:
            step = 1

        # Generate Jan 1 of each year in range with step
        year_starts = pd.date_range(f"{start_year}-01-01", f"{end_year}-01-01", freq=pd.DateOffset(years=step))
        
        # Convert back to MJD
        year_mjds = (year_starts.to_julian_date() - 2400000.5).values

        # Apply to axis
        ax4.set_xticks(year_mjds)
        # ax4.set_xticks(year_mjds, fontsize=16)
        # ax4.set_xticklabels(year_starts.year)
        ax4.set_xticklabels(year_starts.year.astype(str))
        ax4.set_xlabel("Year", fontsize=17)
        
        for label in ax4.get_xticklabels():
            label.set_rotation(40)
            label.set_horizontalalignment("left")
        
        handles, labels = ax3.get_legend_handles_labels()
        ax3.legend(handles, labels, loc="best",# title="Frequency group",
                   title_fontsize=14, fontsize=14)
        
        plt.tight_layout()
        
        safe_label = y_label.replace(" ", "_")
        if safe_label == 'Separation_[mas]':
            safe_label = 'Separation'
        
        filename = f"{outfile_path}/all_freqs_{safe_label}.pdf"
        fig.savefig(filename[:-4]+'.png', dpi=300, bbox_inches="tight")
        fig.savefig(filename)
        if show == True:
            plt.show()
        plt.close()
        print(f"Saved combined {safe_label} plot: {filename}")

# Wrapper functions
def plot_FR(df, freq_tolerance=1E9, outfile_path='', show=False):
    plot_grouped_by_freq(
        df, 'FR', 'FR err',
        freq_tolerance,
        y_label='FR',
        outfile_path=outfile_path,
        show=show
        )

def plot_SBR(df, freq_tolerance=1E9, outfile_path='', show=False, SBR_thresh=4):
    plot_grouped_by_freq(
        df, 'SBR', 'SBR err',
        freq_tolerance,
        y_label='SBR',
        outfile_path=outfile_path,
        show=show,
        SBR_thresh=SBR_thresh
        )

def plot_fluxes(df, freq_tolerance=1E9, outfile_path='', show=False):
    plot_grouped_by_freq(
        df, 'Flux A [Jy]', 'Flux A err [Jy]',
        freq_tolerance,
        y_label='Flux A',
        outfile_path=outfile_path,
        show=show
        )
    plot_grouped_by_freq(
        df, 'Flux B [Jy]', 'Flux B err [Jy]',
        freq_tolerance,
        y_label='Flux B',
        outfile_path=outfile_path,
        show=show
        )

def plot_separation(df, freq_tolerance=1E9, outfile_path='', show=False,
                    plot_line=False):
    plot_grouped_by_freq(
        df,
        'Sep. [mas]',
        'Sep. err [mas]',
        freq_tolerance,
        y_label='Separation [mas]',
        outfile_path=outfile_path,
        show=show,
        plot_line=plot_line
        )

