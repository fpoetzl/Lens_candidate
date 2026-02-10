from astropy.io import ascii
from astropy.io import fits
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import os
import pandas as pd
import pexpect
from pexpect import replwrap
import re
import scipy
from scipy.optimize import curve_fit
from scipy.optimize import leastsq
from scipy.optimize import least_squares

import lens_candidate



def ReadComp(filename):
    '''
    # Purpose:
        Read data from modelfit file with ID components.
        ### WARNING: make sure the modelfit wasn't obtained when the image was
         shifted from the default 0,0 by some quantity;
         otherwise the Radius in mfit doesn't correspond in the image plane here.
         Use the shift method above if that is the case! ###
    
    # Args:
        filename (str): the .mfitid file with id of modelfit components
        
    # Returns:
        mfitcomps (list): list with all components with the component class.
    '''
    
    mfit = open(filename)
    mfitcomps = []
    for line in mfit:
        if not line.startswith("!"):
            comp = lens_candidate.components()
            sl = line.split()
            comp.flux = float(sl[0][:-1])
            # comp.flux_err = None
            # comp.Sp = None    # intensity of the pixel at the position of the
                # component
            # comp.Sp_err = None
            comp.dist = float(sl[1][:-1])
            # comp.dist_err = None
            comp.theta = float(sl[2][:-1])
            # comp.theta_err = None
            comp.major = float(sl[3][:-1])
            # comp.major_err = None
            comp.axratio = float(sl[4][:-1])
            comp.minor = comp.major*comp.axratio
            comp.phi = float(sl[5][:-1])
            # comp.T_b_obs = None
            # comp.T_b_obs_err = None
            comp.freq = float(sl[7])
            comp.name = sl[8] 
            # comp.dlim = None
            comp.calc()
            mfitcomps.append(comp)
    
    mfit.close()
    return mfitcomps



def fit_image_comps(components, IMAP_file, output=False, outpath='./',
                    do_shift_max=False, bmin=None, bmaj=None, beam_angle=None,
                    plt_xlim=[40,-40], plt_ylim=[-40,40], rms_factor=4):
    '''
    # Purpose: Script to fit Gaussian components read in from a modelfit
    file to an image and compute the residual rms noise in the image.

    # Args:
        components (Components type object): ReadComp('file.mfitid') from
          read_components script.
        IMAP_file (str): Path to final .fits image in Stokes I saved in difmap.
        output (bool): if True, create output .fits and .png files.
        do_shift_max: if True, shifts the map (for output=True) to the pixels
          where the flux is maximum.
        bmin (float): clean beam minor axis in mas. Default is None (read from
          header).
        bmaj (float): clean beam major axis in mas. Default is None (read from
          header).
        bmin (float): clean beam position angle (defined North to East) in
          degrees. Default is None (read from header).
    # Returns:
        rms (float): image root-mean-square noise in mJy/beam after component
          subtraction.
    '''
    
    ### Read data and set some initial quantities ###
    img_data = fits.getdata(IMAP_file)    # load up the fits image
    img_data = np.squeeze(img_data)    # remove redundant axes

    ### Extracting information from the fits header ###
    with fits.open(IMAP_file) as hdul:
        # hdulist = fits.open(IMAP_file)
        header = hdul[0].header
        hdu0 = hdul[0]
        try:
            xdim = hdu0.header.cards['naxis1'][1]    # the dimensions of the map in pixels
            ydim = hdu0.header.cards['naxis2'][1]    # the dimensions of the map in pixels
        except KeyError:
            xdim = len(np.array(img_data)[0])
            ydim = len(np.array(img_data)[:,0])
        delt = round(hdu0.header.cards['cdelt2'][1]*3600*1000,10)    # mas per pixel
            # (taken from the pixel increment in DEC axis because it's positive)
        try:
            bmin = hdu0.header.cards['bmin'][1]*3.6e6    # bmin in mas
            bmaj = hdu0.header.cards['bmaj'][1]*3.6e6    # bmaj in mas
            beam_angle = hdu0.header.cards['bpa'][1]    # beam angle in degrees
        except KeyError:
            print('Could not extract beam information from header, needs to be provided.')
            bmin = bmin
            bmaj = bmaj
            beam_angle = beam_angle
        crpix1 = hdu0.header.cards['crpix1'][1]    # max pixel
        crpix2 = hdu0.header.cards['crpix2'][1]

    '''
    We masked the brightness distribution in the clean map and compute the rms
    of the background. 
    For this, we fit gaussian distributions on the clean map, in correspondance
    of each modelfit components. 
    Then we substract the gaussian from the image and remained with a 'masked'
    array that we use to compute the rms.
    '''
    x1 = np.arange(0, xdim)
    y1 = np.arange(0, ydim)
    X, Y = np.meshgrid(x1,y1)    # it generates a grid starting from two 1-d
        # arrays

    guess_prms = []    # the guess parameters are taken from the modelfits

    if not isinstance(components, (list, np.ndarray)):
        components = [components]
    
    for comp in components:
        cent_x = xdim/2 - int(comp.ra/delt)
        cent_y = ydim/2 + int(comp.dec/delt)
        sigma_x = np.sqrt((comp.a/delt)**2 + (bmaj/2./delt)**2)
        sigma_y = np.sqrt((comp.b/delt)**2 + (bmin/2./delt)**2)

        theta = np.radians(beam_angle+90)    # rotating wrt x axis and then
            # translating from deg to radians

        guess_prms.append((cent_x, cent_y, sigma_x, sigma_y, theta, comp.flux))

    p0 = [p for prms in guess_prms for p in prms]    # Flatten the initial
        # guess parameter list.

    xdata = np.vstack((X.ravel(), Y.ravel()))    # We need to ravel the
        # meshgrids of X, Y points to a pair of 1-D arrays.

    ### Do the fit, using our custom _gaussian function which understands our
      # flattened (ravelled) ordering of the data points. ###
    
    lower = [0, 0, 0, 0, 0, 0]*len(components)
    upper = [xdim, ydim, xdim/2, ydim/2, 2*np.pi, np.inf]*len(components)

    try:
        popt, pcov = curve_fit(_gaussian, xdata, img_data.ravel(), p0, bounds=(lower, upper))
    except RuntimeError:    # if fit did not converge, slightly change start params
        print('!! Image-plane Gaussian modelfit did not converge: redo with slightly different start parameters')
        popt, pcov = curve_fit(_gaussian, xdata, img_data.ravel(), np.array(p0)*1.01, bounds=(lower, upper))
    
    fit = np.zeros(img_data.shape)
    for i in range(len(popt)//6):
        fit += gauss2d(X, Y, *popt[i*6:i*6+6])

    img_data_masked = img_data - fit
    rms = np.sqrt(np.mean((img_data_masked)**2))

    if output == True:
        xp = np.zeros((xdim, ydim), float)
        yp = np.zeros((xdim, ydim), float)
        
        ### Locate the maximum of the total intensity ###
        IMAP_max = 0.0
        i_max = 0
        j_max = 0
        for i in range(0, xdim):
            for j in range(0, ydim):
                if img_data[j,i] > IMAP_max:
                    IMAP_max = img_data[j,i]
                    j_max = j
                    i_max = i
        
        ### Create the map grid with physical size ###
        if do_shift_max == False:
            j_max = crpix2
            i_max = crpix1

            for i in range(0, xdim):
                xp[:,i] = (i_max - i)*delt
            for j in range(0, ydim):
                yp[j,:] = (j - j_max)*delt
        else:
            # Shift coordinates of the map so that the maximum total intensity is
              # located at the origin #
            for i in range(0, xdim):
                xp[:,i] = (i_max - i)*delt
            for j in range(0, ydim):
                yp[j,:] = (j - j_max)*delt

        xmin = xp[0,0]
        xmax = xp[0,xdim-1]
        ymin = yp[ydim-1,0]
        ymax = yp[0,0]

        ### Rotate the beam angle so that it runs negative clock wise from North ###
        beam_angle_rot = (-1.0)*(beam_angle)    # required rotation for
            # mpl.patches.Ellipse

        img_name = IMAP_file[0:-5].rsplit('/', 1)[-1]
        if outpath[-1] != '/':
            outpath = outpath + '/'
        
        hdu = fits.PrimaryHDU(fit, header=header)
        hdulist = fits.HDUList([hdu])
        hdulist.writeto(outpath + img_name + '_fitted_gaussians.fits',
                        overwrite=True)

        hdu = fits.PrimaryHDU(img_data_masked, header=header)
        hdulist = fits.HDUList([hdu])
        hdulist.writeto(outpath + img_name + '_fit_residuals.fits',
                        overwrite=True)
    
        ### Calculate contour levels ###
        if rms < IMAP_max:
            firstcont = rms*rms_factor/IMAP_max*100    # percentage of the first cont set at
                # rms_factor*sigma, then doubling the following ones in percentage
        else:
            print('! Warning: provided rms is higher than peak intensity! That should be checked. Take 10 percent as lowest contour.')
            rms = IMAP_max/20
            firstcont = 10
        lastcont = 90    # percent of the peak to set the last contour
        levs = np.array([0.0]*30)    # initialize the array with zero values

        count = 0
        levs[0] = firstcont 
        for q in range(0, 20):                
            levs[count+1] = levs[count]*2 
            count = count + 1
            if levs[count] >= lastcont:
                levs[count] = lastcont
                break

        levs = levs[levs != 0.0]  # Ilevs now contains only contours values

        levs_frac = levs/100
        contours = levs_frac*IMAP_max
        
        img_data_plot = img_data.copy()
        # img_data_plot[img_data_plot<=0] = 1e-9
        
        fig, ax = plt.subplots()
        plt.contour(xp, yp, fit, colors='black', levels=contours)
        ax.axis('scaled')
        plt.title('Gaussfit', y=1.20)
        plt.xlabel('Relative RA [mas]', fontsize=14)
        plt.ylabel('Relative DEC [mas]', fontsize=14)
        plt.xlim(plt_xlim[0], plt_xlim[1])
        plt.ylim(plt_ylim[0], plt_ylim[1])
        plt.savefig(outpath + img_name + '_gaussfit.png', dpi=300)
        # plt.show()
        plt.close(fig)
        plt.close('all')

        fig, ax = plt.subplots()
        axIm = plt.imshow(img_data_plot, cmap='plasma',
                          norm='log', extent=[xmin, xmax, ymin, ymax],
                          vmin=rms, vmax=IMAP_max)
        ax.axis('scaled')
        plt.title('Image', y=1.20)
        plt.xlabel('Relative RA [mas]', fontsize=14)
        plt.ylabel('Relative DEC [mas]', fontsize=14)
        plt.xlim(plt_xlim[0], plt_xlim[1])
        plt.ylim(plt_ylim[0], plt_ylim[1])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("top", size="8%", pad=0.01)
        cbar = plt.colorbar(orientation = "horizontal", cax=cax)
        cbar.ax.xaxis.set_ticks_position("top")
        cbar.ax.tick_params(labelsize=12)
        cbar.update_ticks()
        plt.tight_layout()
        plt.savefig(outpath + img_name + '_img.png', dpi=300)
        # plt.show()
        plt.close('all')
        
        fig, ax = plt.subplots()
        axIm = plt.imshow(img_data_plot, cmap='plasma',
                          norm='log', extent=[xmin, xmax, ymin, ymax],
                          vmin=rms, vmax=IMAP_max)
        ax.contour(xp, yp, fit, colors='w', levels=contours)
        ax.axis('scaled')
        plt.title('Image + Gaussfit', y=1.20)
        plt.xlabel('Relative RA [mas]', fontsize=14)
        plt.ylabel('Relative DEC [mas]', fontsize=14)
        plt.xlim(plt_xlim[0], plt_xlim[1])
        plt.ylim(plt_ylim[0], plt_ylim[1])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("top", size="8%", pad=0.01)
        cbar = plt.colorbar(orientation = "horizontal", cax=cax)
        cbar.ax.xaxis.set_ticks_position("top")
        cbar.ax.tick_params(labelsize=12)
        cbar.update_ticks()
        plt.tight_layout()
        plt.savefig(outpath + img_name + '_img+gaussfit.png', dpi=300)
        # plt.show()
        plt.close(fig)
        plt.close('all')

        fig, ax = plt.subplots()
        axIm = plt.imshow(img_data_masked, cmap='plasma',
                          norm='log', extent=[xmin, xmax, ymin, ymax],
                          vmin=rms, vmax=IMAP_max)
        ax.axis('scaled')
        plt.title('Masked image', y=1.20)
        plt.xlabel('Relative RA [mas]', fontsize=14)
        plt.ylabel('Relative DEC [mas]', fontsize=14)
        plt.xlim(plt_xlim[0], plt_xlim[1])
        plt.ylim(plt_ylim[0], plt_ylim[1])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("top", size="8%", pad=0.01)
        cbar = plt.colorbar(orientation = "horizontal", cax=cax)
        cbar.ax.xaxis.set_ticks_position("top")
        cbar.ax.tick_params(labelsize=12)
        cbar.update_ticks()
        plt.tight_layout()
        plt.savefig(outpath + img_name + '_img_masked.png', dpi=300)
        # plt.show()
        plt.close('all')

    return rms



def Gauss(xs, mu, sigma):
    return np.exp(-0.5*((xs - mu)/sigma)**2)/np.sqrt(2*np.pi*sigma**2)



def gauss2d(x, y, x0, y0, sigma_X, sigma_Y, theta, A):
    '''
    We define a 2d gaussian function (gauss2d) that we use to fit the brightness
    distribution associated with each set of modelfit components in the image
    plane (the clean image). We then extract such gaussian from the clean image
    with the "_gaussian" function. The goal is to finally compute the rms on the
    background.
    '''
    a = np.cos(theta)**2 / (2*sigma_X**2) + np.sin(theta)**2 / (2*sigma_Y**2)
    b = np.sin(2*theta) / (4*sigma_X**2) - np.sin(2 * theta) / (4*sigma_Y**2)
    c = np.sin(theta)**2 / (2*sigma_X**2) + np.cos(theta)**2 / (2*sigma_Y**2)
    return A * np.exp(-(a*(x-x0)**2 + 2*b*(x-x0)*(y-y0) + c*(y-y0)**2))



def _gaussian(M, *args):
    '''
    This is the callable that is passed to curve_fit. M is a (2,N) array
    where N is the total number of data points in Z, which will be ravelled
    to one dimension.
    '''
    x, y = M
    arr = np.zeros(x.shape)
    for i in range(len(args)//6):
        arr += gauss2d(x, y, *args[i*6:i*6+6])
    return arr



def get_ms_ps(fits_file):
    ### Extract necessary information from fits header ###
    header = fits.getheader(fits_file)
    ms_x = header['NAXIS1']
    ms_y = header['NAXIS2']
    round_digit = 10    # round pixel size to two significant figures
    ps_x = round(np.abs(header['CDELT1']*3600*1000),
               int(np.ceil(np.log10(np.abs(header['CDELT1']*3600*1000))))+(round_digit-1))
    ps_y = round(np.abs(header['CDELT2']*3600*1000),
               int(np.ceil(np.log10(np.abs(header['CDELT2']*3600*1000))))+(round_digit-1))
    
    return ms_x, ps_x, ms_y, ps_y



def get_rms(fits_file, uvf_file, mod_file, shift=None, uv_weight=0,
            error_weight=-1, par_file='', out_path='',
            difmap_path='/usr/local/difmap/difmap'):
    '''
    # Purpose: Short program to read in a .fits image and corresponding .uvfits
    and .mfit file (containing Gaussian modelfits) from difmap and calculate the
    total image rms noise.
    
    # Args:
        par_file (str): Path to difmap .par file if existent.
        fits_file (str): Path to the .fits image file.
        uvf_file (str): Path to the .uvfits file containing the visibilities.
        mfit_file (str): Path to the text file containing the Gaussian
          modelfit components from difmap.
        mfitid_file (str): Path to the text file containing the Gaussian
          modelfit components from difmap, including component names in the
          lasy column.
        shift (list): two-element-list with the shift of the map in RA and Dec.
        uv_weight (int): Exponent with which to weight the visibilities
          according to their density in the uv-plane. Natural weighting
          corresponds to -2, uniform to 0 (default).
        error_weight (int): Exponent with which to weight the visibility
          amplitudes according to their initial weights. -2 corresponds to
          Gaussian, 0 no error weighting. Default is -1.
    
    # Returns:
        S_p (list): List with peak flux densities for each component in mJy/beam.
        rms (list): List with residual image root-mean square for each
          component in mJy/beam.
    '''
    
    # Add difmap to PATH
    if difmap_path != None and not difmap_path in os.environ['PATH']:
        os.environ['PATH'] = os.environ['PATH'] + ':{0}'.format(difmap_path)

    # Initialize difmap call
    child = pexpect.spawn('difmap', encoding='utf-8', echo=False)
    child.expect_exact('0>', None, 2)

    def send_difmap_command(command,prompt='0>'):
        child.sendline(command)
        child.expect_exact(prompt, None, 2)

    if par_file != '' and os.path.exists('./'+par_file):
        print('Running difmap using .par file')
        send_difmap_command('@' + par_file+'.par')
        send_difmap_command('rmod ' + mod_file)
        if out_path == '':
            send_difmap_command('wdmap '+par_file[0:-4]+'.res')
        else:
            send_difmap_command('wdmap '+out_path)
        
        if shift != None:
            shift_x = shift[0]
            shift_y = shift[1]
        # If no shift given, check for applied shift in difmap .par file
        else:
            with open('./'+par_file+'.par', 'r') as file:
                lines_ = file.readlines()
            for line in lines_:
                if line.startswith('shift'):
                    parts = line.split(',', 1)
                    shift_x = float(parts[0].strip()[6:])
                    shift_y = float(parts[1].strip())
                    print('! Warning: it has been detected that the original fits image has been shifted in difmap.')
                    print('For consistency, shift of [{0:.3f}, {1:.3f}] mas has been applied to components within difmap.'.format(shift_x, shift_y))
                    break
                else:
                    shift_x = 0
                    shift_y = 0
        
        send_difmap_command('rmod ' + mod_file)
        send_difmap_command('dev /NULL')
        send_difmap_command('mapl map')
        send_difmap_command('print imstat(rms)')
        rms = float(child.before)
        print(rms)
        input('test rms method')
        
    else:
        print('Running difmap using .fits and .uvf file')
        ms_x, ps_x, ms_y, ps_y = get_ms_ps(fits_file)
        send_difmap_command('observe ' + uvf_file)
        send_difmap_command('select I')
        send_difmap_command('uvw '+str(uv_weight)+','+str(error_weight))    # use natural weighting as default
        if shift is not None:
            send_difmap_command('shift '+str(shift[0])+','+str(shift[1]))
            print(f'Shifted map by [{shift[0]:d},{shift[1]:d}] pixels.')
        send_difmap_command('rmod ' + mod_file)
        send_difmap_command('mapsize '+str(2*ms_x)+','+str(ps_x)+','+ str(2*ms_y)+','+str(ps_y))
        if out_path == '':
            send_difmap_command('wdmap '+fits_file[0:-5]+'.res')
        else:
            send_difmap_command('wdmap '+out_path)
        
        send_difmap_command('dev /NULL')
        send_difmap_command('mapl map')
        send_difmap_command('print imstat(rms)')
        start = child.before.find('\r') + 1
        end = child.before.find('\r', start)
        rms = float(child.before[start:end])

    # print('RMS value from residual map in Difmap:')
    # print(rms)
    
    os.system('rm -rf difmap.log*')
    
    return rms



def get_comp_peak_rms(fits_file, uvf_file, mfit_file, mfitid_file, comp_name='',
                      shift=None, uv_weight=0, error_weight=-1, par_file='',
                      out_path='', difmap_path='/usr/local/difmap/difmap'):
    print(f'# Determining component SNR for comp {comp_name} #')
    '''
    # Purpose: Short program to read in a .fits image and corresponding .uvfits
    and .mfit file (containing Gaussian modelfits) from difmap, to estimate the
    uncertainties of the modelfit components based on the image plane. This
    implementation here is the exact way described in Schinzel+ 2012, in which
    each component is handled individually.
    
    # Args:
        par_file (str): Path to difmap .par file if existent.
        fits_file (str): Path to the .fits image file.
        uvf_file (str): Path to the .uvfits file containing the visibilities.
        mfit_file (str): Path to the text file containing the Gaussian
          modelfit components from difmap.
        mfitid_file (str): Path to the text file containing the Gaussian
          modelfit components from difmap, including component names in the
          lasy column.
        shift (list): two-element-list with the shift of the map in RA and Dec.
        uv_weight (int): Exponent with which to weight the visibilities
          according to their density in the uv-plane. Natural weighting
          corresponds to -2, uniform to 0 (default).
        error_weight (int): Exponent with which to weight the visibility
          amplitudes according to their initial weights. -2 corresponds to
          Gaussian, 0 no error weighting. Default is -1.
    
    # Returns:
        S_p (list): List with peak flux densities for each component in mJy/beam.
        rms (list): List with residual image root-mean square for each
          component in mJy/beam.
    '''
    
    # Add difmap to PATH
    if difmap_path != None and not difmap_path in os.environ['PATH']:
        os.environ['PATH'] = os.environ['PATH'] + ':{0}'.format(difmap_path) 
        
    # Initialize difmap call
    child = pexpect.spawn('difmap', encoding='utf-8', echo=False)
    child.expect_exact('0>', None, 2)

    def send_difmap_command(command,prompt='0>'):
        child.sendline(command)
        child.expect_exact(prompt, None, 2)
    
    if par_file != '' and os.path.exists('./'+par_file):
        print('Running difmap using .par file')
        send_difmap_command('@' + par_file+'.par')
        send_difmap_command('rmod ' + mfit_file)
        if out_path == '':
            send_difmap_command('wdmap '+par_file[0:-4]+'.res')
        else:
            send_difmap_command('wdmap '+out_path+par_file[0:-4].rsplit('/', 1)[-1]+'.res')
        
        ms_x, ps_x, ms_y, ps_y = get_ms_ps(fits_file)
        
        components = ReadComp(mfitid_file)    # read here the no_shift map
        S_p = []
        rms = []
        
        if shift != None:
            shift_x = shift[0]
            shift_y = shift[1]
        # If no shift given, check for applied shift in difmap .par file
        else:
            with open('./'+par_file+'.par', 'r') as file:
                lines_ = file.readlines()
            for line in lines_:
                if line.startswith('shift'):
                    parts = line.split(',', 1)
                    shift_x = float(parts[0].strip()[6:])
                    shift_y = float(parts[1].strip())
                    print('! Warning: it has been detected that the original fits image has been shifted in difmap.')
                    print('For consistency, shift of [{0:.3f}, {1:.3f}] mas has been applied to components within difmap.'.format(shift_x, shift_y))
                    break
                else:
                    shift_x = 0
                    shift_y = 0
        
        for comp in components:
            
            if comp.name != comp_name:
                continue
            
            ra = comp.ra + shift_x
            dec = comp.dec + shift_y
            
            send_difmap_command('rmod ' + mfit_file)
            send_difmap_command('dev /NULL')
            send_difmap_command('delwin')
            send_difmap_command('mapl cln')
            send_difmap_command('addwin '+str(ra-0.1*ps_x)
                                     +','+str(ra+0.1*ps_x)
                                     +','+str(dec-0.1*ps_y)
                                     +','+str(dec+0.1*ps_y))
            send_difmap_command('winmod true')
            send_difmap_command('mapl map')
            send_difmap_command('print mapvalue('+str(ra)
                                             +','+str(dec)+')')
            try:
                for i, str_ in enumerate(child.before[::-1]):
                    if str_ =='.':
                        i_end = i
                        break
                S_p.append(float(child.before[-i_end-2:]))
            except ValueError:
                print('Could not read off peak flux density for component {0:s}'.format(comp.name))
                print(child.before)
                S_p.append(np.nan)
            
            if out_path == '':
                resMAP_data = fits.getdata(par_file[0:-4]+'.res')
            else:
                resMAP_data = fits.getdata(out_path+par_file[0:-4].rsplit('/', 1)[-1]+'.res')
            
            resMAP_data = np.squeeze(resMAP_data)
            xdim = len(np.array(resMAP_data)[0])
            ydim = len(np.array(resMAP_data)[:,0])
            rms_box_x = xdim/20.
            rms_box_y = ydim/20.
            rms_ = np.std(resMAP_data[ int(round(ydim/2 + dec/ps_y, 0))  - int(rms_box_y/2)
                                      :int(round(ydim/2 + dec/ps_y, 0))  + int(rms_box_y/2),
                                       int(round(xdim/2 - ra/ps_x, 0))-1 - int(rms_box_x/2)
                                      :int(round(xdim/2 - ra/ps_x, 0))-1 + int(rms_box_x/2)])
            rms.append(rms_)
        
    else:
        print('Running difmap using .fits and .uvf file')
        ms_x, ps_x, ms_y, ps_y = get_ms_ps(fits_file)
        send_difmap_command('observe ' + uvf_file)
        send_difmap_command('select I')
        send_difmap_command('uvw '+str(uv_weight)+','+str(error_weight))    # use natural weighting as default
        if shift is not None:
            send_difmap_command('shift '+str(shift[0])+','+str(shift[1]))
            print(f'Shifted map by [{shift[0]:d},{shift[1]:d}] pixels.')
        send_difmap_command('rmod ' + mfit_file)
        send_difmap_command('mapsize '+str(2*ms_x)+','+str(ps_x)+','+ str(2*ms_y)+','+str(ps_y))
        if out_path == '':
            send_difmap_command('wdmap '+fits_file[0:-5]+'.res')
        else:
            send_difmap_command('wdmap '+out_path+fits_file[0:-5].rsplit('/', 1)[-1]+'.res')
        
        components = ReadComp(mfitid_file)
        S_p = []
        rms = []
        sizes = []
        
        send_difmap_command('dev /NULL')
        send_difmap_command('mapl map')
        
        for comp in components:
            if comp.name != comp_name:
                continue

            sizes.append(np.sqrt(comp.major*comp.minor))
            
            if shift is not None:
                ra = comp.ra + shift[0]
                dec = comp.dec + shift[1]
            else:
                ra = comp.ra
                dec = comp.dec
            
            send_difmap_command('dev /NULL')
            send_difmap_command('mapl cln')
            send_difmap_command('addwin '+str(ra-0.1*ps_x)
                                     +','+str(ra+0.1*ps_x)
                                     +','+str(dec-0.1*ps_y)
                                     +','+str(dec+0.1*ps_y))
            send_difmap_command('winmod true')
            send_difmap_command('mapl map')
            send_difmap_command('print mapvalue('+str(ra)
                                             +','+str(dec)+')')
            
            try:
                for i, str_ in enumerate(child.before[::-1]):
                    if str_ =='.':
                        i_end = i
                        break
                S_p.append(float(child.before[-i_end-2:]))
            except ValueError:
                print('Could not read off peak flux density for component {0:s}'.format(comp.name))
                print(child.before)
                S_p.append(np.nan)
            
            rms_box = 100
            if out_path == '':
                resMAP_data = fits.getdata(fits_file[0:-5]+'.res')
            else:
                resMAP_data = fits.getdata(out_path+fits_file[0:-5].rsplit('/', 1)[-1]+'.res')
            resMAP_data = np.squeeze(resMAP_data)
            xdim = len(np.array(resMAP_data)[0])
            ydim = len(np.array(resMAP_data)[:,0])
            rms_ = np.std(resMAP_data[ int(round(ydim/2 + dec/ps_y, 0))  - int(rms_box/2)
                                      :int(round(ydim/2 + dec/ps_y, 0))  + int(rms_box/2),
                                       int(round(xdim/2 - ra/ps_x, 0))-1 - int(rms_box/2)
                                      :int(round(xdim/2 - ra/ps_x, 0))-1 + int(rms_box/2)])
            rms.append(rms_)
    print('Peak flux density after component subtraction [Jy/beam]:')
    print(S_p[0])
    print('RMS value measured around component position in residual map [Jy/beam]:')
    print(rms[0])
    print('Component equivalent size [mas]:')
    print(sizes[0])
    
    os.system('rm -rf difmap.log*')
    
    return S_p[0], rms[0]



def calc_lens_params(components, outfile, freq, gain_err=0.0):
    '''
    # Purpose:
    Script to calculate parameters for potential lensing in given VLBI
    Gaussian modelfit component. Calculates surface brightness of components,
    the surface brightness ratio and flux ratio of two main components,
    that may be made up of several individual Gaussian components.
    
    # Args:
        IMAP_file (str): Path to final .fits image in Stokes I saved in difmap.
        components (Components type object): ReadComp('file.mfitid') from
          read_components script. Incorporated by importing plot_components.
        outfile (str): Output text file for calculated quantities.
        freq (float): Observing frequency in GHz.
        gain_err (float): Optional gain uncertainty for error calculation
          (default is 0).
    
    # Returns:
        SB_A (float): Surface brightness of region A in Jy/arcsec^2.
        sig_SB_A (float): 1-sigma uncertainty of surface brightness of region A.
        SB_B (float): Surface brightness of region B in Jy/arcsec^2.
        sig_SB_B (float): 1-sigma uncertainty of surface brightness of region B.
        SBR (float): Surface brightness ratio between region A and B.
        sig_SBR (float): 1-sigma uncertainty of SBR.
        FR (float): Flux ratio between region A and B.
        sig_FR (float): 1-sigma uncertainty of FR.
        SB_AB_max (float): Surface brightness of the brightest component in
          region B in Jy/arcsec^2.
        sig_SB_AB_max (float): 1-sigma uncertainty of SB_AB_max.
        SBR_AB_max (float): Surface brightness ratio between the brightest
          components in region A and B, respectively.
        sig_SBR_AB_max (float): 1-sigma uncertainty of SBR_AB_max.
        FR_AB_max (float): Flux ratio between the brightest components in
          region A and B, respectively.
        sig_FR_AB_max (float): 1-sigma uncertainty of FR_AB_max.
        # Also saves outfile+'_lens_params.latex' and outfile+'_lens_params.dat'
          files
    '''
    
    print('### CALCULATE SBR, FR AND SEPARATION ###')
    
    ### initialize arrays ###
    S_t_A = []
    S_t_B = []
    sig_S_t_A = []
    sig_S_t_A_write = []
    sig_S_t_B = []
    sig_S_t_B_write = []
    
    FWHM_A = []
    FWHM_B = []
    sig_FWHM_A = []
    sig_FWHM_B = []
    
    r_A = -1
    r_B = -1
    phi_A = 0
    phi_B = 0
    sig_r_A = 0
    sig_r_B = 0
    sig_phi_A = 0
    sig_phi_B = 0
    
    lim = ''
    
    for comp in components:
        if comp.name[0] == 'A':    # select components in region A
            ### first, identify brightest component in region A and save these data ###
            if len(S_t_A) == 0:    # there is only one component in region A
                name_A_max = comp.name
                S_t_A_max = comp.flux
                sig_S_t_A_max_write = comp.flux_err
                sig_S_t_A_max = np.sqrt(comp.flux_err**2 - (gain_err*S_t_A_max)**2)
                
                r_A = comp.dist
                ra_A = comp.ra
                dec_A = comp.dec
                phi_A = comp.theta_rot
                sig_r_A = comp.dist_err
                sig_phi_A = comp.theta_err
                
                x_A = r_A * -np.cos(phi_A*np.pi/180.)
                y_A = r_A * np.sin(phi_A*np.pi/180.)
                # sig_x_A = np.sqrt(  (sig_r_A*-np.cos(phi_A*np.pi/180.))**2
                                  # + (sig_phi_A*np.pi/180.*r_A*np.sin(phi_A*np.pi/180.))**2)
                # sig_y_A = np.sqrt(  (sig_r_A*np.sin(phi_A*np.pi/180.))**2
                                  # + (sig_phi_A*np.pi/180.*r_A*np.cos(phi_A*np.pi/180.))**2)
                
                # cov_r_phi_A = 0.5*np.sin(2*phi_A*np.pi/180.)*(sig_r_A**2+r_A**2*sig_phi_A**2)/(r_A*np.cos(2*phi_A*np.pi/180.))
                # cov_term_A = 2*r_A*np.sin(phi_A*np.pi/180.)*np.cos(phi_A*np.pi/180.)*cov_r_phi_A

                # sig_x_A = np.sqrt(  (sig_r_A* np.sin(np.pi * phi_A / 180.))**2
                                  # + (sig_phi_A*np.pi/180.*r_A*np.cos(np.pi * phi_A / 180.))**2
                                  # - cov_term_A)
                # sig_y_A = np.sqrt(  (sig_r_A* np.cos(np.pi * phi_A / 180.))**2
                                  # + (sig_phi_A*np.pi/180.*r_A*np.sin(np.pi * phi_A / 180.))**2
                                  # + cov_term_A)
                
                if comp.axratio == 1:
                    if comp.major > comp.dlim:
                        FWHM_A_max = comp.major
                    else:
                        print('Component {0:s} has size ({1:.2f} mas) smaller than d_lim ({2:.2f} mas), will calculate surface brightness lower limit'.format(comp.name, comp.major, comp.dlim))
                        print('! Surface brightness ratio will be a LOWER limit')
                        lim = ' (lower limit)'
                        FWHM_A_max = comp.dlim
                else:
                    if comp.major > comp.dlim:
                        FWHM_A_max = comp.major*np.sqrt(1+comp.axratio**2)/np.sqrt(2)
                    else:
                        print('Component {0:s} has size ({1:.2f} mas) smaller than d_lim ({2:.2f} mas), will calculate surface brightness lower limit'.format(comp.name, comp.major, comp.dlim))
                        print('! Surface brightness ratio will be a LOWER limit')
                        lim = ' (lower limit)'
                        FWHM_A_max = comp.dlim
                sig_FWHM_A_max = comp.major_err
            else:    # there are multiple components in region A
                if all(comp.flux > np.array(S_t_A)):    # identify brightest component in region A
                    name_A_max = comp.name
                    S_t_A_max = comp.flux
                    sig_S_t_A_max_write = comp.flux_err
                    sig_S_t_A_max = np.sqrt(comp.flux_err**2 - (gain_err*S_t_A_max)**2)
                    
                    r_A = comp.dist
                    ra_A = comp.ra
                    dec_A = comp.dec
                    phi_A = comp.theta_rot
                    sig_r_A = comp.dist_err
                    sig_phi_A = comp.theta_err
                    
                    x_A = r_A * -np.cos(phi_A*np.pi/180.)
                    y_A = r_A * np.sin(phi_A*np.pi/180.)
                    # sig_x_A = np.sqrt(  (sig_r_A*-np.cos(phi_A*np.pi/180.))**2
                                      # + (sig_phi_A*np.pi/180.*r_A*np.sin(phi_A*np.pi/180.))**2)
                    # sig_y_A = np.sqrt(  (sig_r_A*np.sin(phi_A*np.pi/180.))**2
                                      # + (sig_phi_A*np.pi/180.*r_A*np.cos(phi_A*np.pi/180.))**2)
                    
                    # cov_r_phi_A = 0.5*np.sin(2*phi_A*np.pi/180.)*(sig_r_A**2+r_A**2*sig_phi_A**2)/(r_A*np.cos(2*phi_A*np.pi/180.))
                    # cov_term_A = 2*r_A*np.sin(phi_A*np.pi/180.)*np.cos(phi_A*np.pi/180.)*cov_r_phi_A

                    # sig_x_A = np.sqrt(  (sig_r_A* np.sin(np.pi * phi_A / 180.))**2
                                      # + (sig_phi_A*np.pi/180.*r_A*np.cos(np.pi * phi_A / 180.))**2
                                      # - cov_term_A)
                    # sig_y_A = np.sqrt(  (sig_r_A* np.cos(np.pi * phi_A / 180.))**2
                                      # + (sig_phi_A*np.pi/180.*r_A*np.sin(np.pi * phi_A / 180.))**2
                                      # + cov_term_A)
                    
                    if comp.axratio == 1:
                        if comp.major > comp.dlim:
                            FWHM_A_max = comp.major
                        else:
                            print('Component {0:s} has size ({1:.2f} mas) smaller than d_lim ({2:.2f} mas), will calculate surface brightness lower limit'.format(comp.name, comp.major, comp.dlim))
                            print('! Surface brightness ratio will be a LOWER limit')
                            lim = ' (lower limit)'
                            FWHM_A_max = comp.dlim
                    else:
                        if comp.major > comp.dlim:
                            FWHM_A_max = comp.major*np.sqrt(1+comp.axratio**2)/np.sqrt(2)
                        else:
                            print('Component {0:s} has size ({1:.2f} mas) smaller than d_lim ({2:.2f} mas), will calculate surface brightness lower limit'.format(comp.name, comp.major, comp.dlim))
                            print('! Surface brightness ratio will be a LOWER limit')
                            lim = ' (lower limit)'
                            FWHM_A_max = comp.dlim
                    sig_FWHM_A_max = comp.major_err
            
            ### now, store all components for region A ###
            S_t_A.append(comp.flux)
            sig_S_t_A_write.append(comp.flux_err)
            sig_S_t_A.append(np.sqrt(comp.flux_err**2 - (gain_err*comp.flux)**2))
            
            if comp.axratio == 1:
                if comp.major > comp.dlim:
                    FWHM_A.append(comp.major)
                else:
                    print('Component {0:s} has size ({1:.2f} mas) smaller than d_lim ({2:.2f} mas), will calculate surface brightness lower limit'.format(comp.name, comp.major, comp.dlim))
                    print('! Surface brightness ratio will be a LOWER limit')
                    lim = ' (lower limit)'
                    FWHM_A.append(comp.dlim)
            else:
                if comp.major > comp.dlim:
                    FWHM_A.append(comp.major*np.sqrt(1+comp.axratio**2)/np.sqrt(2))
                else:
                    print('Component {0:s} has size ({1:.2f} mas) smaller than d_lim ({2:.2f} mas), will calculate surface brightness lower limit'.format(comp.name, comp.major, comp.dlim))
                    print('! Surface brightness ratio will be a LOWER limit')
                    lim = ' (lower limit)'
                    FWHM_A.append(comp.dlim)
            sig_FWHM_A.append(comp.major_err)
            
        elif comp.name[0] == 'B':    # select components in region B
            ### first, identify brightest component in region B and save these data ###
            if len(S_t_B) == 0:    # there is only one component in region B
                name_B_max = comp.name
                S_t_B_max = comp.flux
                sig_S_t_B_max_write = comp.flux_err
                sig_S_t_B_max = np.sqrt(comp.flux_err**2 - (gain_err*S_t_B_max)**2)
                
                r_B = comp.dist
                ra_B = comp.ra
                dec_B = comp.dec
                phi_B = comp.theta_rot
                sig_r_B = comp.dist_err
                sig_phi_B = comp.theta_err
                
                x_B = r_B * -np.cos(phi_B*np.pi/180.)
                y_B = r_B * np.sin(phi_B*np.pi/180.)
                # sig_x_B = np.sqrt(  (sig_r_B*-np.cos(phi_B*np.pi/180.))**2
                                    # + (sig_phi_B*np.pi/180.*r_B*np.sin(phi_B*np.pi/180.))**2)
                # sig_y_B = np.sqrt(  (sig_r_B* np.sin(phi_B*np.pi/180.))**2
                                    # + (sig_phi_B*np.pi/180.*r_B*np.cos(phi_B*np.pi/180.))**2)
                
                # cov_r_phi_B = 0.5*np.sin(2*phi_B*np.pi/180.)*(sig_r_B**2+r_B**2*sig_phi_B**2)/(r_B*np.cos(2*phi_B*np.pi/180.))
                # cov_term_B = 2*r_B*np.sin(phi_B*np.pi/180.)*np.cos(phi_B*np.pi/180.)*cov_r_phi_B

                # sig_x_B = np.sqrt(  (sig_r_B* np.sin(np.pi * phi_B / 180.))**2
                                  # + (sig_phi_B*np.pi/180.*r_B*np.cos(phi_B*np.pi/180.))**2
                                  # - cov_term_B)
                # sig_y_B = np.sqrt(  (sig_r_B* np.cos(np.pi * phi_B / 180.))**2
                                  # + (sig_phi_B*np.pi/180.*r_B*np.sin(phi_B*np.pi/180.))**2
                                  # + cov_term_B)
                
                if comp.axratio == 1:
                    if comp.major > comp.dlim:
                        FWHM_B_max = comp.major
                    else:
                        print('Component {0:s} has size ({1:.2f} mas) smaller than d_lim ({2:.2f} mas), will calculate surface brightness lower limit'.format(comp.name, comp.major, comp.dlim))
                        print('! Surface brightness ratio will be an UPPER limit')
                        lim = ' (upper limit)'
                        FWHM_B_max = comp.dlim
                else:
                    if comp.major > comp.dlim:
                        FWHM_B_max = comp.major*np.sqrt(1+comp.axratio**2)/np.sqrt(2)
                    else:
                        print('Component {0:s} has size ({1:.2f} mas) smaller than d_lim ({2:.2f} mas), will calculate surface brightness lower limit'.format(comp.name, comp.major, comp.dlim))
                        print('! Surface brightness ratio will be an UPPER limit')
                        lim = ' (upper limit)'
                        FWHM_B_max = comp.dlim
                sig_FWHM_B_max = comp.major_err
            else:    # there are multiple components in region B
                if all(comp.flux > np.array(S_t_B)):    # identify brightest component in region B
                    name_B_max = comp.name
                    S_t_B_max = comp.flux
                    sig_S_t_B_max_write = comp.flux_err
                    sig_S_t_B_max = np.sqrt(comp.flux_err**2 - (gain_err*S_t_B_max)**2)
                    
                    r_B = comp.dist
                    ra_B = comp.ra
                    dec_B = comp.dec
                    phi_B = comp.theta_rot
                    sig_r_B = comp.dist_err
                    sig_phi_B = comp.theta_err
                    
                    x_B = r_B * -np.cos(phi_B*np.pi/180.)
                    y_B = r_B * np.sin(phi_B*np.pi/180.)
                    # sig_x_B = np.sqrt(  (sig_r_B*-np.cos(phi_B*np.pi/180.))**2
                                        # + (sig_phi_B*np.pi/180.*r_B*np.sin(phi_B*np.pi/180.))**2)
                    # sig_y_B = np.sqrt(  (sig_r_B* np.sin(phi_B*np.pi/180.))**2
                                        # + (sig_phi_B*np.pi/180.*r_B*np.cos(phi_B*np.pi/180.))**2)
                    
                    # cov_r_phi_B = 0.5*np.sin(2*phi_B*np.pi/180.)*(sig_r_B**2+r_B**2*sig_phi_B**2)/(r_B*np.cos(2*phi_B*np.pi/180.))
                    # cov_term_B = 2*r_B*np.sin(phi_B*np.pi/180.)*np.cos(phi_B*np.pi/180.)*cov_r_phi_B

                    # sig_x_B = np.sqrt(  (sig_r_B* np.sin(np.pi * phi_B / 180.))**2
                                      # + (sig_phi_B*np.pi/180.*r_B*np.cos(phi_B*np.pi/180.))**2
                                      # - cov_term_B)
                    # sig_y_B = np.sqrt(  (sig_r_B* np.cos(np.pi * phi_B / 180.))**2
                                      # + (sig_phi_B*np.pi/180.*r_B*np.sin(phi_B*np.pi/180.))**2
                                      # + cov_term_B)
 
                    if comp.axratio == 1:
                        if comp.major > comp.dlim:
                            FWHM_B_max = comp.major
                        else:
                            print('Component {0:s} has size {1:.2f} smaller than d_lim {2:.2f}, will calculate surface brightness lower limit'.format(comp.name, comp.major, comp.dlim))
                            print('! Surface brightness ratio will be an UPPER limit')
                            lim = ' (upper limit)'
                            FWHM_B_max = comp.dlim
                    else:
                        if comp.major > comp.dlim:
                            FWHM_B_max = comp.major*np.sqrt(1+comp.axratio**2)/np.sqrt(2)
                        else:
                            print('Component {0:s} has size {1:.2f} smaller than d_lim {2:.2f}, will calculate surface brightness lower limit'.format(comp.name, comp.major, comp.dlim))
                            print('! Surface brightness ratio will be an UPPER limit')
                            lim = ' (upper limit)'
                            FWHM_B_max = comp.dlim
                    sig_FWHM_B_max = comp.major_err
            
            ### now, store all components for region B ###
            S_t_B.append(comp.flux)
            sig_S_t_B_write.append(comp.flux_err)
            sig_S_t_B.append(np.sqrt(comp.flux_err**2 - (gain_err*comp.flux)**2))
            
            if comp.axratio == 1:
                if comp.major > comp.dlim:
                    FWHM_B.append(comp.major)
                else:
                    print('Component {0:s} has size ({1:.2f} mas) smaller than d_lim ({2:.2f} mas), will calculate surface brightness lower limit'.format(comp.name, comp.major, comp.dlim))
                    print('! Surface brightness ratio will be an UPPER limit')
                    lim = ' (upper limit)'
                    FWHM_B.append(comp.dlim)
            else:
                if comp.major > comp.dlim:
                    FWHM_B.append(comp.major*np.sqrt(1+comp.axratio**2)/np.sqrt(2))
                else:
                    print('Component {0:s} has size ({1:.2f} mas) smaller than d_lim ({2:.2f} mas), will calculate surface brightness lower limit'.format(comp.name, comp.major, comp.dlim))
                    print('! Surface brightness ratio will be an UPPER limit')
                    lim = ' (upper limit)'
                    FWHM_B.append(comp.dlim)
            sig_FWHM_B.append(comp.major_err)
    
    S_t_A = np.array(S_t_A)
    S_t_B = np.array(S_t_B)
    sig_S_t_A = np.array(sig_S_t_A)
    sig_S_t_A_write = np.array(sig_S_t_A_write)
    sig_S_t_B = np.array(sig_S_t_B)
    sig_S_t_B_write = np.array(sig_S_t_B_write)
    FWHM_A = np.array(FWHM_A)
    FWHM_B = np.array(FWHM_B)
    sig_FWHM_A = np.array(sig_FWHM_A)
    sig_FWHM_B = np.array(sig_FWHM_B)
    
    ### Sum up flux densities and areas covered by components ###
    S_t_A_tot = np.sum(S_t_A)
    S_t_B_tot = np.sum(S_t_B)
    sig_S_t_A_tot = np.sqrt(np.sum(sig_S_t_A**2))
    sig_S_t_A_tot_write = np.sqrt(np.sum(sig_S_t_A_write**2))
    sig_S_t_B_tot = np.sqrt(np.sum(sig_S_t_B**2))
    sig_S_t_B_tot_write = np.sqrt(np.sum(sig_S_t_B_write**2))
    
    Area_A_tot = np.sum(FWHM_A**2/1000.**2)    # tot. area in units of arcsec^2
    Area_A_tot_ = np.sum(FWHM_A**2/1000.**2/3600**2*(np.pi/180.)**2)    # tot. area in units of radians^2
    Area_B_tot = np.sum(FWHM_B**2/1000.**2)    # tot. area in units of arcsec^2
    Area_B_tot_ = np.sum(FWHM_B**2/1000.**2/3600**2*(np.pi/180.)**2)    # tot. area in units of radians^2
    sig_Area_A_tot = np.sqrt(np.sum((2*FWHM_A/1000.*sig_FWHM_A/1000.)**2))
    sig_Area_A_tot_ = np.sqrt(np.sum((2*FWHM_A/1000./3600.*np.pi/180.*sig_FWHM_A/1000./3600.*np.pi/180.)**2))
    sig_Area_B_tot = np.sqrt(np.sum((2*FWHM_B/1000.*sig_FWHM_B/1000.)**2))
    sig_Area_B_tot_ = np.sqrt(np.sum((2*FWHM_B/1000./3600.*np.pi/180.*sig_FWHM_B/1000./3600.*np.pi/180.)**2))
    
    ### Calculate surface brightness ###
    SB_A = S_t_A_tot/Area_A_tot    # In units of Jy/arcsec^2
    SB_B = S_t_B_tot/Area_B_tot
    
    # print('TEBIST: lim')
    # print(lim)
    
    if lim == ' (lower limit)':
        sig_SB_A = np.nan
        sig_SB_A_write = np.nan
        A_is_lim = 'lower'
        SBR_is_lim = 'lower'
    else:
        sig_SB_A = np.sqrt( (sig_S_t_A_tot/Area_A_tot)**2
                           +(sig_Area_A_tot*S_t_A_tot/(Area_A_tot)**2)**2 )
        sig_SB_A_write = np.sqrt( (sig_S_t_A_tot_write/Area_A_tot)**2
                                 +(sig_Area_A_tot*S_t_A_tot/(Area_A_tot)**2)**2 )
        A_is_lim = ''
        SBR_is_lim = ''
    if lim == ' (upper limit)':
        sig_SB_B = np.nan
        sig_SB_B_write = np.nan
        B_is_lim = 'lower'
        SBR_is_lim = 'upper'
    else:
        sig_SB_B = np.sqrt( (sig_S_t_B_tot/Area_B_tot)**2
                           +(sig_Area_B_tot*S_t_B_tot/(Area_B_tot**2))**2 )
        sig_SB_B_write = np.sqrt( (sig_S_t_B_tot_write/Area_B_tot)**2
                                 +(sig_Area_B_tot*S_t_B_tot/(Area_B_tot**2))**2 )
        B_is_lim = ''
        SBR_is_lim = ''
    
    # print('TEBIST: B_is_lim')
    # print(B_is_lim)
    # print('TEBIST: SBR_is_lim')
    # print(SBR_is_lim)
    
    nu = freq#*1E9    # convert from GHz to Hz
    c = scipy.constants.c
    k_B = scipy.constants.k
    
    T_b_obs_A = S_t_A_tot*1E-26*c**2/(2*k_B*nu**2*Area_A_tot_)
    if lim == ' (lower limit)':
        sig_T_b_obs_A = np.nan
    else:
        sig_T_b_obs_A = np.sqrt(  (sig_S_t_A_tot_write*1E-26*c**2/(2*k_B*nu**2*Area_A_tot_))**2
                                + (-1*sig_Area_A_tot_*S_t_A_tot*1E-26*c**2/(2*k_B*nu**2*Area_A_tot_**2))**2
                                )
    T_b_obs_B = S_t_B_tot*1E-26*c**2/(2*k_B*nu**2*Area_B_tot_)
    if lim == ' (upper limit)':
        sig_T_b_obs_B = np.nan
    else:
        sig_T_b_obs_B = np.sqrt(  (sig_S_t_B_tot_write*1E-26*c**2/(2*k_B*nu**2*Area_B_tot_))**2
                                + (-1*sig_Area_B_tot_*S_t_B_tot*1E-26*c**2/(2*k_B*nu**2*Area_B_tot_**2))**2
                                )
    
    log_T_b_obs_A = np.log10(T_b_obs_A)
    sig_log_T_b_obs_A = np.sqrt((np.array(sig_T_b_obs_A)/(np.log(10)*np.array(T_b_obs_A)))**2)
        # calculate error in log space with standard error propagation
    log_T_b_obs_B = np.log10(T_b_obs_B)
    sig_log_T_b_obs_B = np.sqrt((np.array(sig_T_b_obs_B)/(np.log(10)*np.array(T_b_obs_B)))**2)
        # calculate error in log space with standard error propagation
    
    # ### Test standard error propagation against MCMC for T_b ###
    # _T_b_obs_A_ = np.random.normal(T_b_obs_A, sig_T_b_obs_A, 100000)
    # _log_T_b_obs_A_ = np.log10(_T_b_obs_A_)
    # _sig_log_T_b_obs_A_ = np.std(_log_T_b_obs_A_)
    # fig1 = plt.figure(1)
    # counts, bins, _ = plt.hist(_log_T_b_obs_A_, bins=10000, density=True,
                               # label='MC: T_b = {0:.2f} +/- {1:.2f}'.format(np.mean(_log_T_b_obs_A_), _sig_log_T_b_obs_A_))
    # # plt.xlim(-50, 300)
    # vals = np.linspace(min(bins), max(bins), 100000)
    # plt.plot(vals, Gauss(vals, log_T_b_obs_A, sig_log_T_b_obs_A),
             # label='Gaussian: T_b = {0:.2f} +/- {1:.2f}'.format(log_T_b_obs_A, sig_log_T_b_obs_A))
    # plt.axvline(np.mean(_log_T_b_obs_A_), c='g', linewidth=1, linestyle='solid', label='MC Mean')
    # plt.axvline(np.mean(_log_T_b_obs_A_) + _sig_log_T_b_obs_A_, c='g', linewidth=1, linestyle='dashed')
    # plt.axvline(np.mean(_log_T_b_obs_A_) - _sig_log_T_b_obs_A_, c='g', linewidth=1, linestyle='dashed')
    # plt.axvline(np.mean(_log_T_b_obs_A_), c='orange', linewidth=1, linestyle='solid', label='Gaussian Mean')
    # plt.axvline(np.mean(_log_T_b_obs_A_) + sig_log_T_b_obs_A, c='orange', linewidth=1, linestyle='dashed')
    # plt.axvline(np.mean(_log_T_b_obs_A_) - sig_log_T_b_obs_A, c='orange', linewidth=1, linestyle='dashed')
    # plt.xlabel('log(T_b/K)')
    # plt.legend(loc='best')
    # plt.savefig(outfile+'_test_T_b_A.png')
    # plt.close()
    
    ### Calculate surface brightness ratio ###
    SBR = SB_A/SB_B
    sig_SBR = np.sqrt( (sig_SB_A/SB_B)**2 
                      +(SB_A*sig_SB_B/SB_B**2)**2 )
    
    #### Calculate SBR error from Monte-Carlo simulations ###
    if len(FWHM_B) != 0:
        _S_t_A_tot_ = np.random.normal(S_t_A_tot, sig_S_t_A_tot, 100000)
        
        _FWHM_A_ = np.zeros((len(FWHM_A), 100000))
        for i, FWHM in enumerate(FWHM_A):
            _FWHM_A_[i] = np.random.normal(FWHM_A[i], sig_FWHM_A[i], 100000)

        _S_t_B_tot_ = np.random.normal(S_t_B_tot, sig_S_t_B_tot, 100000)
        
        _FWHM_B_ = np.zeros((len(FWHM_B), 100000))
        for i, FWHM in enumerate(FWHM_B):
            _FWHM_B_[i] = np.random.normal(FWHM_B[i], sig_FWHM_B[i], 100000)

        _SBR_ = (   _S_t_A_tot_/np.sum(_FWHM_A_**2/1000.**2, axis=0)
                 / (_S_t_B_tot_/np.sum(_FWHM_B_**2/1000.**2, axis=0)))
        ### Rejection of strong outliers ###
        _SBR_ = _SBR_[(_SBR_ < 10*np.median(_SBR_)) & (_SBR_ > 0)]
        
        SBR_ = np.mean(_SBR_)
        sig_SBR_ = np.std(_SBR_)
        # sig_SBR_ = np.sqrt(np.sum((np.median(_SBR_)-_SBR_)**2)/len(_SBR_))
        
        fig1 = plt.figure(1)
        counts, bins, _ = plt.hist(_SBR_, bins=10000, density=True,
                                   label='MC: SBR = {0:.2f} +/- {1:.2f}'.format(SBR_, sig_SBR_))
        # plt.xlim(-50, 300)
        vals = np.linspace(min(bins), max(bins), 100000)
        plt.plot(vals, Gauss(vals, SBR, sig_SBR),
                 label='Gaussian: SBR = {0:.2f} +/- {1:.2f}'.format(SBR, sig_SBR))
        plt.axvline(SBR_, c='g', linewidth=1, linestyle='solid', label='MC Mean')
        plt.axvline(SBR_ + sig_SBR_, c='g', linewidth=1, linestyle='dashed')
        plt.axvline(SBR_ - sig_SBR_, c='g', linewidth=1, linestyle='dashed')
        plt.axvline(SBR, c='orange', linewidth=1, linestyle='solid', label='Gaussian Mean')
        plt.axvline(SBR + sig_SBR, c='orange', linewidth=1, linestyle='dashed')
        plt.axvline(SBR - sig_SBR, c='orange', linewidth=1, linestyle='dashed')
        plt.xlabel('SBR')
        plt.legend(loc='best')
        fig1.savefig(outfile+'_test_SBR.png')
        plt.close()
    
    ### Calculate flux ratio ###
    FR = S_t_A_tot/S_t_B_tot
    sig_FR = np.sqrt( (sig_S_t_A_tot/S_t_B_tot)**2
                     + (S_t_A_tot*sig_S_t_B_tot/S_t_B_tot**2)**2 )

    if len(S_t_B) == 0:
        print('No second components assigned to region B found. Values set to NaN.')
        SB_B = np.nan
        sig_SB_B = np.nan
        log_T_b_obs_B = np.nan
        sig_log_T_b_obs_B = np.nan
        FWHM_B = np.nan
        sig_FWHM_B = np.nan
        SBR = np.nan
        sig_SBR = np.nan
        FR = np.nan
        sig_FR = np.nan
        x_B = np.nan
        y_B = np.nan
        sig_x_B = np.nan
        sig_y_B = np.nan
    
    ### Calculate component distance from Cartesian coordinates ###
    # (obsolete for now, as I can't figure out how to incorporate covariance properly)
    # delta_r = np.sqrt((x_A - x_B)**2 + (y_A - y_B)**2)
    # sig_delta_r = np.sqrt(  (sig_x_A*2*(x_A-x_B))**2
                          # + (sig_x_B*2*(x_A-x_B))**2
                          # + (sig_y_A*2*(y_A-y_B))**2
                          # + (sig_y_B*2*(y_A-y_B))**2
                         # )/delta_r
    
    ### Now calculate distance using polar coordinates directly ###
    if r_A != -1 and r_B != -1:
        delta_r_dir = np.sqrt(r_A**2 + r_B**2 - 2*r_A*r_B*np.cos((phi_A-phi_B)*np.pi/180.))
        cos = np.cos((phi_A-phi_B)*np.pi/180.)
        sin = np.sin((phi_A-phi_B)*np.pi/180.)
        sig_delta_r_dir = np.sqrt(   (sig_r_A*(r_A-r_B)*cos)**2
                                   + (sig_r_B*(r_B-r_A)*cos)**2
                                   + (sig_phi_A*np.pi/180.*r_A*r_B*sin)**2
                                   + (sig_phi_B*np.pi/180.*r_A*r_B*sin)**2
                                  )/delta_r_dir
    else:
        delta_r_dir = np.nan
        sig_delta_r_dir = np.nan
    
    '''
    The above formula for standard error propagation applies only when the
    measurement errors are small; this is not necessarily the case for
    measurements of the polar coordinates, especially when they are close to the
    origin. Then the errors become much larger than the mean values. Thus,
    I employ here a Monte-Carlo approach to get a more reliable error estimate.
    '''
    if r_A != -1 and r_B != -1:
        _r_A_ = np.random.normal(r_A, sig_r_A, 100000)
        _phi_A_ = np.random.normal(phi_A, sig_phi_A, 100000)
        _r_B_ = np.random.normal(r_B, sig_r_B, 100000)
        _phi_B_ = np.random.normal(phi_B, sig_phi_B, 100000)

        _delta_r_ = np.sqrt(_r_A_**2 + _r_B_**2 - 2*_r_A_*_r_B_*np.cos((_phi_A_-_phi_B_)*np.pi/180.))
        ### rejection of strong outliers ###
        _delta_r_ = _delta_r_[(_delta_r_ < 10*np.median(_delta_r_)) & (_delta_r_ > -10*np.median(_delta_r_))]
        
        delta_r_MC = np.mean(_delta_r_)
        sig_delta_r_MC = np.std(_delta_r_)
        
        fig2 = plt.figure(2)
        counts, bins, _ = plt.hist(_delta_r_, bins=10000, density=True,
                                   label='MC: Delta_r = {0:.2f} +/- {1:.2f}'.format(delta_r_MC, sig_delta_r_MC))
        vals = np.linspace(min(bins), max(bins), 100000)
        plt.plot(vals, Gauss(vals, delta_r_dir, sig_delta_r_dir),
                             label='Gaussian: Delta_r = {0:.2f} +/- {1:.2f}'.format(delta_r_dir, sig_delta_r_dir))
        plt.axvline(delta_r_MC, c='g', linewidth=1, linestyle='solid', label='MC Mean')
        plt.axvline(delta_r_MC + sig_delta_r_MC, c='g', linewidth=1, linestyle='dashed')
        plt.axvline(delta_r_MC - sig_delta_r_MC, c='g', linewidth=1, linestyle='dashed')
        plt.axvline(delta_r_dir, c='orange', linewidth=1, linestyle='solid', label='Gaussian Mean')
        plt.axvline(delta_r_dir + sig_delta_r_dir, c='orange', linewidth=1, linestyle='dashed')
        plt.axvline(delta_r_dir - sig_delta_r_dir, c='orange', linewidth=1, linestyle='dashed')
        plt.xlabel('Sep. [mas]')
        plt.legend(loc='best')
        fig2.savefig(outfile+'_test_delta_r.png')
        plt.close()
        
    else:
        delta_r_MC = np.nan
        sig_delta_r_MC = np.nan
    
    delta_r = delta_r_dir    # use standard value calculated directly
    sig_delta_r = sig_delta_r_MC    # use MC value
    
    print('Surface brightness of region A: {0:.3e} +/- {1:.3e} Jy/arcsec^2'.\
          format(SB_A, sig_SB_A))
    print('Surface brightness of region B: {0:.3e} +/- {1:.3e} Jy/arcsec^2'.\
          format(SB_B, sig_SB_B))
    print('Surface brightness ratio'+lim+': {0:.2f} +/- {1:.2f}'.\
          format(SBR, sig_SBR))
    print('Flux ratio: {0:.2f} +/- {1:.2f}'.\
          format(FR, sig_FR))
    print('Distance between components: {0:.2f} +/- {1:.2f} mas'.\
          format(delta_r, sig_delta_r))
    
    if len(S_t_A) == 1 and len(S_t_B) == 1:
        T_b_obs_A_max = T_b_obs_A
        sig_T_b_obs_A_max = sig_T_b_obs_A
        log_T_b_obs_A_max = log_T_b_obs_A
        sig_log_T_b_obs_A_max = sig_log_T_b_obs_A
        
        T_b_obs_B_max = T_b_obs_B
        sig_T_b_obs_B_max = sig_T_b_obs_B
        log_T_b_obs_B_max = log_T_b_obs_B
        sig_log_T_b_obs_B_max = sig_log_T_b_obs_B
        
        SBR_A_max = SBR
        sig_SBR_A_max = sig_SBR
        SBR_B_max = SBR
        sig_SBR_B_max = sig_SBR
        SBR_AB_max = SBR
        sig_SBR_AB_max = sig_SBR
        FR_A_max = FR
        sig_FR_A_max = sig_FR
        FR_B_max = FR
        sig_FR_B_max = sig_FR
        
    elif len(S_t_A) > 1 and len(S_t_B) == 1:
        Area_A_max = np.sum(FWHM_A_max**2/1000.**2)    # tot. area in units of arcsec^2
        sig_Area_A_max = np.sqrt(np.sum((2*FWHM_A_max/1000.*sig_FWHM_A_max/1000.)**2))
        Area_A_max_ = np.sum(FWHM_A_max**2/1000.**2/3600.**2*(np.pi/180.)**2)    # compute area in square radians
        sig_Area_A_max_ = np.sqrt(np.sum((2*FWHM_A_max/1000./3600.**2*(np.pi/180.)**2
                                          * sig_FWHM_A_max/1000./3600.**2*(np.pi/180.)**2)**2))
        
        SB_A_max = S_t_A_max/Area_A_max    # In units of Jy/arcsec^2
        sig_SB_A_max = np.sqrt( (sig_S_t_A_max/Area_A_max)**2
                               +(sig_Area_A_max*S_t_A_max/(Area_A_max)**2)**2 )
        sig_SB_A_max_write = np.sqrt( (sig_S_t_A_max_write/Area_A_max)**2
                               +(sig_Area_A_max*S_t_A_max/(Area_A_max)**2)**2 )
        
        ### Calculate observed brightness temperature
        T_b_obs_A_max = S_t_A_max*1E-26*c**2/(2*k_B*nu**2*Area_A_max_)
        sig_T_b_obs_A_max = np.sqrt(  (sig_S_t_A_max*1E-26*c**2/(2*k_B*nu**2*Area_A_max_))**2
                                    + (-1*sig_Area_A_max_*S_t_A_max*1E-26*c**2/(2*k_B*nu**2*Area_A_max_**2))**2
                                    )
        log_T_b_obs_A_max = np.log10(T_b_obs_A_max)
        sig_log_T_b_obs_A_max = np.sqrt((np.array(sig_T_b_obs_A_max)/(np.log(10)*np.array(T_b_obs_A_max)))**2)
            # calculate error in log space with standard error propagation
        
        ### Calculate surface brightness ratio ###
        SBR_A_max = SB_A_max/SB_B
        sig_SBR_A_max = np.sqrt( (sig_SB_A_max/SB_B)**2 
                          +(SB_A_max*sig_SB_B/SB_B**2)**2 )
        
        ### Calculate flux ratio ###
        FR_A_max = S_t_A_max/S_t_B_tot
        sig_FR_A_max = np.sqrt( (sig_S_t_A_max/S_t_B_tot)**2
                         + (S_t_A_max*sig_S_t_B_tot/S_t_B_tot**2)**2 )
        
        T_b_obs_B_max = T_b_obs_B
        sig_T_b_obs_B_max = sig_T_b_obs_B
        log_T_b_obs_B_max = log_T_b_obs_B
        sig_log_T_b_obs_B_max = sig_log_T_b_obs_B
        
        SBR_B_max = SBR
        sig_SBR_B_max = sig_SBR
        SBR_AB_max = SBR_A_max
        sig_SBR_AB_max = sig_SBR_A_max
        FR_B_max = FR
        sig_FR_B_max = sig_FR
    
    elif len(S_t_A) == 1 and len(S_t_B) > 1:
        Area_B_max = np.sum(FWHM_B_max**2/1000.**2)    # tot. area in units of arcsec^2
        sig_Area_B_max = np.sqrt(np.sum((2*FWHM_B_max/1000.*sig_FWHM_B_max/1000.)**2))
        Area_B_max_ = np.sum(FWHM_B_max**2/1000.**2/3600.**2*(np.pi/180.)**2)    # compute area in square radians
        sig_Area_B_max_ = np.sqrt(np.sum((2*FWHM_B_max/1000./3600.**2*(np.pi/180.)**2*sig_FWHM_B_max/1000./3600.**2*(np.pi/180.)**2)**2))
        
        SB_B_max = S_t_B_max/Area_B_max    # In units of Jy/arcsec^2
        sig_SB_B_max = np.sqrt( (sig_S_t_B_max/Area_B_max)**2
                               +(sig_Area_B_max*S_t_B_max/(Area_B_max)**2)**2 )
        sig_SB_B_max_write = np.sqrt( (sig_S_t_B_max_write/Area_B_max)**2
                               +(sig_Area_B_max*S_t_B_max/(Area_B_max)**2)**2 )
        
        ### Calculate observed brightness temperature
        T_b_obs_B_max = S_t_B_max*1E-26*c**2/(2*k_B*nu**2*Area_B_max_)
        sig_T_b_obs_B_max = np.sqrt(  (sig_S_t_B_max*1E-26*c**2/(2*k_B*nu**2*Area_B_max_))**2
                                    + (-1*sig_Area_B_max_*S_t_B_max*1E-26*c**2/(2*k_B*nu**2*Area_B_max_**2))**2
                                    )
        log_T_b_obs_B_max = np.log10(T_b_obs_B_max)
        sig_log_T_b_obs_B_max = np.sqrt((np.array(sig_T_b_obs_B_max)/(np.log(10)*np.array(T_b_obs_B_max)))**2)
            # calculate error in log space with standard error propagation
        
        ### Calculate surface brightness ratio ###
        SBR_B_max = SB_A/SB_B_max
        sig_SBR_B_max = np.sqrt( (sig_SB_A/SB_B_max)**2 
                          +(SB_A*sig_SB_B_max/SB_B_max**2)**2 )
        
        ### Calculate flux ratio ###
        FR_B_max = S_t_A_tot/S_t_B_max
        sig_FR_B_max = np.sqrt( (sig_S_t_A_tot/S_t_B_max)**2
                         + (S_t_A_tot*sig_S_t_B_max/S_t_B_max**2)**2 )
        
        T_b_obs_A_max = T_b_obs_A
        sig_T_b_obs_A_max = sig_T_b_obs_A
        log_T_b_obs_A_max = log_T_b_obs_A
        sig_log_T_b_obs_A_max = sig_log_T_b_obs_A
        
        SBR_A_max = SBR
        sig_SBR_A_max = sig_SBR
        SBR_AB_max = SBR_B_max
        sig_SBR_AB_max = sig_SBR_B_max
        FR_A_max = FR
        sig_FR_A_max = sig_FR
    
    elif len(S_t_A) > 1 and len(S_t_B) > 1:
        ### brightest component in region A ###
        Area_A_max = np.sum(FWHM_A_max**2/1000.**2)    # tot. area in units of arcsec^2
        sig_Area_A_max = np.sqrt(np.sum((2*FWHM_A_max/1000.*sig_FWHM_A_max/1000.)**2))
        Area_A_max_ = np.sum(FWHM_A_max**2/1000.**2/3600.**2*(np.pi/180.)**2)    # compute area in square radians
        sig_Area_A_max_ = np.sqrt(np.sum((2*FWHM_A_max/1000./3600.**2*(np.pi/180.)**2*sig_FWHM_A_max/1000./3600.**2*(np.pi/180.)**2)**2))
        
        SB_A_max = S_t_A_max/Area_A_max    # In units of Jy/arcsec^2
        sig_SB_A_max = np.sqrt( (sig_S_t_A_max/Area_A_max)**2
                               +(sig_Area_A_max*S_t_A_max/(Area_A_max)**2)**2 )
        sig_SB_A_max_write = np.sqrt( (sig_S_t_A_max_write/Area_A_max)**2
                               +(sig_Area_A_max*S_t_A_max/(Area_A_max)**2)**2 )
        
        ### Calculate surface brightness ratio ###
        SBR_A_max = SB_A_max/SB_B
        sig_SBR_A_max = np.sqrt( (sig_SB_A_max/SB_B)**2 
                          +(SB_A_max*sig_SB_B/SB_B**2)**2 )
        
        ### Calculate flux ratio ###
        FR_A_max = S_t_A_max/S_t_B_tot
        sig_FR_A_max = np.sqrt( (sig_S_t_A_max/S_t_B_tot)**2
                         + (S_t_A_max*sig_S_t_B_tot/S_t_B_tot**2)**2 )
        
        ### Calculate observed brightness temperature
        T_b_obs_A_max = S_t_A_max*1E-26*c**2/(2*k_B*nu**2*Area_A_max_)
        sig_T_b_obs_A_max = np.sqrt(  (sig_S_t_A_max*1E-26*c**2/(2*k_B*nu**2*Area_A_max_))**2
                                    + (-1*sig_Area_A_max_*S_t_A_max*1E-26*c**2/(2*k_B*nu**2*Area_A_max_**2))**2
                                    )
        log_T_b_obs_A_max = np.log10(T_b_obs_A_max)
        sig_log_T_b_obs_A_max = np.sqrt((np.array(sig_T_b_obs_A_max)/(np.log(10)*np.array(T_b_obs_A_max)))**2)
            # calculate error in log space with standard error propagation
        
        
        ### brightest component in region B ###
        Area_B_max = np.sum(FWHM_B_max**2/1000.**2)    # tot. area in units of arcsec^2
        sig_Area_B_max = np.sqrt(np.sum((2*FWHM_B_max/1000.*sig_FWHM_B_max/1000.)**2))
        Area_B_max_ = np.sum(FWHM_B_max**2/1000.**2/3600.**2*(np.pi/180.)**2)    # compute area in square radians
        sig_Area_B_max_ = np.sqrt(np.sum((2*FWHM_B_max/1000./3600.**2*(np.pi/180.)**2
                                          * sig_FWHM_B_max/1000./3600.**2*(np.pi/180.)**2)**2))
        
        SB_B_max = S_t_B_max/Area_B_max    # In units of Jy/arcsec^2
        sig_SB_B_max = np.sqrt( (sig_S_t_B_max/Area_B_max)**2
                               +(sig_Area_B_max*S_t_B_max/(Area_B_max)**2)**2 )
        sig_SB_B_max_write = np.sqrt( (sig_S_t_B_max_write/Area_B_max)**2
                               +(sig_Area_B_max*S_t_B_max/(Area_B_max)**2)**2 )
        
        ### Calculate surface brightness ratio ###
        SBR_B_max = SB_A/SB_B_max
        sig_SBR_B_max = np.sqrt( (sig_SB_A/SB_B_max)**2 
                          +(SB_A*sig_SB_B_max/SB_B_max**2)**2 )
        
        ### Calculate flux ratio ###
        FR_B_max = S_t_A_tot/S_t_B_max
        sig_FR_B_max = np.sqrt( (sig_S_t_A_tot/S_t_B_max)**2
                         + (S_t_A_tot*sig_S_t_B_max/S_t_B_max**2)**2 )
        
        ### Calculate observed brightness temperature
        T_b_obs_B_max = S_t_B_max*1E-26*c**2/(2*k_B*nu**2*Area_B_max_)
        sig_T_b_obs_B_max = np.sqrt(  (sig_S_t_B_max*1E-26*c**2/(2*k_B*nu**2*Area_B_max_))**2
                                    + (-1*sig_Area_B_max_*S_t_B_max*1E-26*c**2/(2*k_B*nu**2*Area_B_max_**2))**2
                                    )
        log_T_b_obs_B_max = np.log10(T_b_obs_B_max)
        sig_log_T_b_obs_B_max = np.sqrt((np.array(sig_T_b_obs_B_max)/(np.log(10)*np.array(T_b_obs_B_max)))**2)
            # calculate error in log space with standard error propagation
        
        # SBR_AB_max = SBR
        # sig_SBR_AB_max = sig_SBR
    
    elif len(S_t_A) == 1 and len(S_t_B) == 0:
        T_b_obs_A_max = T_b_obs_A
        sig_T_b_obs_A_max = sig_T_b_obs_A
        log_T_b_obs_A_max = log_T_b_obs_A
        sig_log_T_b_obs_A_max = sig_log_T_b_obs_A
        T_b_obs_B_max = T_b_obs_B
        sig_T_b_obs_B_max = sig_T_b_obs_B
        log_T_b_obs_B_max = log_T_b_obs_B
        sig_log_T_b_obs_B_max = sig_log_T_b_obs_B
        
        SBR_A_max = SBR
        sig_SBR_A_max = sig_SBR
        SBR_B_max = SBR
        sig_SBR_B_max = sig_SBR
        SBR_AB_max = SBR
        sig_SBR_AB_max = sig_SBR
        FR_A_max = FR
        sig_FR_A_max = sig_FR
        FR_B_max = FR
        sig_FR_B_max = sig_FR
    
    elif len(S_t_A) > 1 and len(S_t_B) == 0:
        ### brightest component in region A ###
        Area_A_max = np.sum(FWHM_A_max**2/1000.**2)    # tot. area in units of arcsec^2
        sig_Area_A_max = np.sqrt(np.sum((2*FWHM_A_max/1000.*sig_FWHM_A_max/1000.)**2))
        Area_A_max_ = np.sum(FWHM_A_max**2/1000.**2/3600.**2*(np.pi/180.)**2)    # compute area in square radians
        sig_Area_A_max_ = np.sqrt(np.sum((2*FWHM_A_max/1000./3600.**2*(np.pi/180.)**2
                                          * sig_FWHM_A_max/1000./3600.**2*(np.pi/180.)**2)**2))
        
        SB_A_max = S_t_A_max/Area_A_max    # In units of Jy/arcsec^2
        sig_SB_A_max = np.sqrt( (sig_S_t_A_max/Area_A_max)**2
                               +(sig_Area_A_max*S_t_A_max/(Area_A_max)**2)**2 )
        sig_SB_A_max_write = np.sqrt( (sig_S_t_A_max_write/Area_A_max)**2
                               +(sig_Area_A_max*S_t_A_max/(Area_A_max)**2)**2 )
        
        ### Calculate observed brightness temperature
        T_b_obs_A_max = S_t_A_max*1E-26*c**2/(2*k_B*nu**2*Area_A_max_)
        sig_T_b_obs_A_max = np.sqrt(  (sig_S_t_A_max*1E-26*c**2/(2*k_B*nu**2*Area_A_max_))**2
                                    + (-1*sig_Area_A_max_*S_t_A_max*1E-26*c**2/(2*k_B*nu**2*Area_A_max_**2))**2
                                    )
        log_T_b_obs_A_max = np.log10(T_b_obs_A_max)
        sig_log_T_b_obs_A_max = np.sqrt((np.array(sig_T_b_obs_A_max)/(np.log(10)*np.array(T_b_obs_A_max)))**2)
            # calculate error in log space with standard error propagation
        
        ### Calculate surface brightness ratio ###
        SBR_A_max = np.nan
        sig_SBR_A_max = np.nan
        
        ### Calculate flux ratio ###
        FR_A_max = np.nan
        sig_FR_A_max = np.nan
        
        T_b_obs_B_max = T_b_obs_B
        sig_T_b_obs_B_max = sig_T_b_obs_B
        log_T_b_obs_B_max = log_T_b_obs_B
        sig_log_T_b_obs_B_max = sig_log_T_b_obs_B
        
        SBR_B_max = SBR
        sig_SBR_B_max = sig_SBR
        # SBR_AB_max = SBR
        # sig_SBR_AB_max = sig_SBR
        FR_B_max = FR
        sig_FR_B_max = sig_FR
    
    results_fluxes = {
        'Flux A [Jy]': S_t_A_tot,
        'Flux A err [Jy]': sig_S_t_A_tot_write,
        'Flux B [Jy]': S_t_B_tot,
        'Flux B err [Jy]': sig_S_t_B_tot_write
        }
    
    results_SB_test = {
        'log(T_b,obs,A)':log_T_b_obs_A,
        'log(T_b,obs,A) err':sig_log_T_b_obs_A,
        'A is limit':A_is_lim,
        'log(T_b,obs,B)':log_T_b_obs_B,
        'log(T_b,obs,B) err':sig_log_T_b_obs_B,
        'B is limit':B_is_lim,
        'SBR':SBR,
        'SBR err':sig_SBR,
        'SBR is limit':SBR_is_lim,
        'log(T_b,obs,A,max)':log_T_b_obs_A_max,
        'log(T_b,obs,A,max) err':sig_log_T_b_obs_A_max,
        'log(T_b,obs,B,max)':log_T_b_obs_B,
        'log(T_b,obs,B,max) err':sig_log_T_b_obs_B_max,
        'SBR_A,max':SBR_A_max,
        'SBR_A,max err':sig_SBR_A_max,
        'SBR_B,max':SBR_B_max,
        'SBR_B,max err':sig_SBR_B_max
        }
    
    results_FR_test = {
        'FR':FR,
        'FR err':sig_FR,
        'FR_A,max':FR_A_max,
        'FR_A,max err':sig_FR_A_max,
        'FR_B,max':FR_B_max,
        'FR_B,max err':sig_FR_B_max
        }
    
    results_sep_test = {
        'Sep. [mas]':delta_r,
        'Sep. err [mas]':sig_delta_r
        }
    
    return results_fluxes, results_SB_test, results_FR_test, results_sep_test



def calc_spix(components_nu1, components_nu2, nu1, nu2, gain_err=0.0):
    
    print('\n')
    print('### CALCULATE SPECTRAL INDICES ###')
    
    S_nu1_A = 0
    S_nu1_B = 0
    S_nu2_A = 0
    S_nu2_B = 0
    errs_S_nu1_A = []
    errs_S_nu1_B = []
    errs_S_nu2_A = []
    errs_S_nu2_B = []
    
    for comp in components_nu1:
        if comp.name[0] == 'A':
            S_nu1_A += comp.flux
            errs_S_nu1_A.append(comp.flux_err)
        if comp.name[0] == 'B':
            S_nu1_B += comp.flux
            errs_S_nu1_B.append(comp.flux_err)
    for comp in components_nu2:
        if comp.name[0] == 'A':
            S_nu2_A += comp.flux
            errs_S_nu2_A.append(comp.flux_err)
        if comp.name[0] == 'B':
            S_nu2_B += comp.flux
            errs_S_nu2_B.append(comp.flux_err)
    
    if S_nu1_A == 0:
        S_nu1_A = np.nan
    if S_nu1_B == 0:
        S_nu1_B = np.nan
    if S_nu2_A == 0:
        S_nu2_A = np.nan
    if S_nu2_B == 0:
        S_nu2_B = np.nan
    
    sig_S_nu1_A = np.sqrt(np.sum(np.array(errs_S_nu1_A)**2))
    sig_S_nu1_B = np.sqrt(np.sum(np.array(errs_S_nu1_B)**2))
    sig_S_nu2_A = np.sqrt(np.sum(np.array(errs_S_nu2_A)**2))
    sig_S_nu2_B = np.sqrt(np.sum(np.array(errs_S_nu2_B)**2))
    
    alpha_A = np.log(float(S_nu2_A)/float(S_nu1_A))/np.log(float(nu2)/float(nu1))

    sig_alpha_A = np.abs(1/np.log(float(nu2)/float(nu1))*np.sqrt((float(sig_S_nu1_A)/float(S_nu1_A))**2\
            + (float(sig_S_nu2_A)/float(S_nu2_A))**2))
    
    alpha_B = np.log(float(S_nu2_B)/float(S_nu1_B))/np.log(float(nu2)/float(nu1))

    sig_alpha_B = np.abs(1/np.log(float(nu2)/float(nu1))*np.sqrt((float(sig_S_nu1_B)/float(S_nu1_B))**2\
            + (float(sig_S_nu2_B)/float(S_nu2_B))**2))

    return alpha_A, sig_alpha_A, alpha_B, sig_alpha_B



def build_results_df(dataset, epoch, freq, results_fluxes, results_SB_test, results_FR_test, results_sep_test):
    # Merge all dictionaries
    merged = {
        'Dataset': dataset,
        'Epoch': epoch,
        'Freq [Hz]': freq,
        **results_fluxes,
        **results_SB_test,
        **results_FR_test,
        **results_sep_test,
        'alpha A': np.nan,
        'alpha A err': np.nan,
        'alpha B': np.nan,
        'alpha B err': np.nan
        }
    
    df = pd.DataFrame([merged])
    
    return df



def restructure_df(df):
    
    if df.empty:
        print('Error: cannot restructure because dataframe is empty. Run calc_all first.')
    else:    
        df_new = df.copy()
        
        # Drop dataset column if present
        if "Dataset" in df_new.columns:
            df_new = df_new.drop(columns=["Dataset"])

        reshaped_rows = []

        # Find A/B paired columns
        a_cols = [c for c in df_new.columns if re.search(r"\bA\b", c)]
        b_cols = [c for c in df_new.columns if re.search(r"\bB\b", c)]

        # # Map collapsed column names (strip A/B)
        paired_map = {}
        # for a in a_cols:
            # base = a.replace(" A", "")
            # paired_map[base] = (a, a.replace(" A", " B"))
        for col in df_new.columns:
            if "A" in col:
                # try different separators before/after A
                for sep in [" A", "A ", "(A", "A)", ",A", "A,", "[A", "A]", "_A"]:
                    if sep in col:
                        base = col.replace(sep, "").strip()
                        bcol = col.replace("A", "B", 1)  # replace only first occurrence
                        paired_map[base] = (col, bcol)
                        break

        for _, row in df_new.iterrows():
            # Common values (those without A/B)
            common_vals = {c: row[c] for c in df_new.columns if c not in sum(paired_map.values(), ())}

            # --- Row A ---
            row_a = {"Component": "A"}
            # common values
            row_a.update(common_vals)
            # Fill collapsed A/B columns
            for base, (col_a, col_b) in paired_map.items():
                row_a[base] = row[col_a]
            reshaped_rows.append(row_a)

            # --- Row B ---
            row_b = {"Component": "B"}
            # common values replaced by "-"
            # row_b.update({c: "-" for c in common_vals})
            row_b.update({c: np.nan for c in common_vals})
            # Fill collapsed A/B columns
            for base, (col_a, col_b) in paired_map.items():
                row_b[base] = row[col_b]
            reshaped_rows.append(row_b)

        # Build new dataframe
        df_new = pd.DataFrame(reshaped_rows)
        
        df_new = df_new.rename(columns={"log(T_b,obs,": "log(T_b,obs)"})
        df_new = df_new.rename(columns={"log(T_b,obs, err": "log(T_b,obs) err"})
        df_new = df_new.rename(columns={"is limit": "T_b is limit"})
        
        column_order = [
            "Epoch",
            "Freq [Hz]",
            "Component",
            "Flux",
            "Flux [Jy]",
            "Flux err [Jy]",
            "FR",
            "FR err",
            "FR_max",
            "FR_max err",
            "FR OK?",
            "log(T_b,obs)",
            "log(T_b,obs) err",
            "T_b is limit",
            "log(T_b,obs,max)",
            "log(T_b,obs,max) err",
            "SBR",
            "SBR err",
            "SBR is limit",
            "SBR_max",
            "SBR_max err",
            "SBR OK?",
            "Sep. [mas]",
            "Sep. err [mas]",
            "Sep. OK?",
            "alpha",
            "alpha err",
            "alpha OK?"
            ]

        # Reorder columns, keep only those that exist in the DataFrame
        df_reordered = df_new[[col for col in column_order if col in df_new.columns]]
        
        return df_reordered



def write_mfiterr(components, outfile=''):
    '''
    # Purpose: small helper function to write a file containing Gaussian
      modelfit component quantities with errors, difmap-stype but not
      readable by difmap.
    
    # Args:
        components (Components object): Components objects as defined in
          lens_candidate.
        outfile (str): output file name.
    
    # Returns:
        Creates output file.
    '''

    fluxes = []
    flux_errs = []
    dists = []
    dist_errs = []
    thetas = []
    theta_errs = []
    majors = []
    major_errs = []
    axratios = []
    phis = []
    Ts = [1]*len(components)
    freqs = []
    names = []

    for comp in components:
        fluxes.append(comp.flux)
        flux_errs.append(comp.flux_err)
        dists.append(comp.dist)
        dist_errs.append(comp.dist_err)
        thetas.append(comp.theta)
        theta_errs.append(comp.theta_err)
        majors.append(comp.major)
        major_errs.append(comp.major_err)
        axratios.append(comp.axratio)
        phis.append(comp.phi)
        freqs.append(comp.freq)
        names.append(comp.name)
    
    ascii.write([fluxes, flux_errs, dists, dist_errs, thetas, theta_errs,
                 majors, major_errs, axratios, phis, Ts, freqs, names],
                outfile,
                names=['Flux (Jy)',
                       'Flux error (Jy)',
                       'Radius (mas)',
                       'Radius error (mas)',
                       'Theta (deg)',
                       'Theta error (deg)',
                       'Major FWHM (mas)',
                       'Major FWHM error (mas)',
                       'Axial ratio',
                       'Phi (deg)',
                       'T',
                       'Freq (Hz)',
                       'Name',
                       ],
                formats={'Flux (Jy)':'.6f',
                         'Flux error (Jy)':'.6f',
                         'Radius (mas)':'.6f',
                         'Radius error (mas)':'.6f',
                         'Theta (deg)':'.4f',
                         'Theta error (deg)':'.4f',
                         'Major FWHM (mas)':'.6f',
                         'Major FWHM error (mas)':'.6f',
                         'Axial ratio':'.5f',
                         'Phi (deg)':'.4f',
                         'T':'d',
                         'Freq (Hz)':'.5e',
                         'Name':'s',               
                         },
                overwrite=True, format='fixed_width', delimiter=' ')
    
    with open(outfile, 'r') as file:
        lines = file.readlines()
    # Add an exclamation mark at the beginning for difmap readability
    for i, line in enumerate(lines):
        if i == 0:
            lines[i] = '!' + lines[0]
        else:
            lines[i] = ' ' + lines[i]
    with open(outfile, 'w') as file:
        file.writelines(lines)



def get_clean_rms(fits_path, uvf_path, shift=None, uv_weight=0,
                  error_weight=-1, par_file='', out_path='',
                  difmap_path='/usr/local/difmap/difmap'):

    # Check if fits file has AIPS CC table in it
    with fits.open(fits_path) as hdul:
        cc_hdu = None
        for hdu in hdul:
            if hdu.header.get('EXTNAME', '').strip().upper().startswith('AIPS CC'):
                cc_hdu = hdu
                break
        if cc_hdu is None:
            print('Could not extract clean rms since no CC table was found in .fits file.')
            return
        
        # Extract flux and positions
        flux = np.zeros(len(cc_hdu.data))
        xpos = np.zeros(len(cc_hdu.data))
        ypos = np.zeros(len(cc_hdu.data))

        for i in range(len(cc_hdu.data)):
            flux[i] = cc_hdu.data[i][0]
            xpos[i] = cc_hdu.data[i][1]*3.6E6
            ypos[i] = cc_hdu.data[i][2]*3.6E6
        del cc_hdu

    r = np.sqrt(xpos**2 + ypos**2)
    phi = np.zeros(len(r))

    # Adjust position angle values of CC components to consensus
      # definition
    for i, (x_, y_) in enumerate(zip(xpos, ypos)):
        if x_ >= 0 and y_ > 0:
            phi[i] = 180/np.pi * (np.arctan(x_ / y_))
        if x_ >= 0 and y_ < 0:
            phi[i] = 180 + 180/np.pi * (np.arctan(x_ / y_))
        if x_ < 0 and y_ > 0:
            phi[i] = 180/np.pi * (np.arctan(x_ / y_))
        if x_ < 0 and y_ < 0:
            phi[i] = -180 + 180/np.pi * (np.arctan(x_ / y_))
        if x_ > 0 and y_ == 0:
            phi[i] = 90.0
        if x_ < 0 and y_ == 0:
            phi[i] = -90.0
        if x_ == 0 and y_== 0:
            phi[i] = 0.0

    if os.path.exists(fits_path[0:-5]+"_tmp.mod"):
        os.remove(fits_path[0:-5]+"_tmp.mod")

    ascii.write([flux,
                r,
                phi],
                fits_path[0:-5]+"_tmp.mod",
                names=["Flux", "xpos", "ypos"],
                formats={'Flux':'.9f',
                        'xpos':'.5f',
                        'ypos':'.3f'},
                overwrite=True, format="fixed_width", delimiter=" ")

    with open(fits_path[0:-5]+"_tmp.mod", 'r') as f:
        t = f.readlines()
    
    if os.path.exists(fits_path[0:-5]+"_tmp.mod"):
        os.remove(fits_path[0:-5]+"_tmp.mod")
    
    with open(fits_path[0:-5]+"_CC.mod", 'w') as w:
        w.writelines(t[1:])
    
    f.close()
    w.close()

    rms = get_rms(fits_file=fits_path, uvf_file=uvf_path,
                  mod_file=fits_path[0:-5]+"_CC.mod",
                  shift=shift, uv_weight=uv_weight, error_weight=error_weight,
                  par_file=par_file, out_path=out_path,
                  difmap_path=difmap_path)
    return rms



def limit_formatter(x):
    """Format values with optional < > limits for LaTeX output."""
    if pd.isna(x):
        return '-'
    
    x = str(x).strip()

    # Plain limits like "<1.82" or ">3.14"
    if x.startswith('<') or x.startswith('>'):
        try:
            num = float(x[1:])
            return f"${x[0]}${num:.2f}"
        except ValueError:
            return x

    # Already-LaTeX limits like "$<$1.82"
    if x.startswith('$<$') or x.startswith('$>$'):
        try:
            num = float(x[3:])
            return f"{x[:3]}{num:.2f}"
        except ValueError:
            return x

    # Normal numeric value
    try:
        return f"${float(x):.2f}$"
    except ValueError:
        return x



def export_csv(df, filename):
    
    df_new = restructure_df(df)
    
    df_new.to_csv(filename, index=False)
    print(f"Saved results to {filename}")


