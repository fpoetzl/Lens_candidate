from astropy.io import fits
from astropy.time import Time
from glob import glob
import numpy as np
import pandas as pd
import os
import scipy
from scipy.optimize import curve_fit
import sys

from Lens_candidate import helpers
from Lens_candidate import plotters



class Image_dataset:
    '''
    An object class that represents an observation with an image, visibility,
    and modelfit data.
    '''
    def __init__(
        self,
        fits_file_path='',
        uvf_file_path='',
        mfit_file_path='',
        mfitid_file_path='',
        gain_err=0.1,
        difmap_path='/usr/local/difmap/difmap',
        difmap_weight=[0,-1],
        shift=[0,0],
        rms_method='image_fit'
    ):
        
        self.fits_file_path = fits_file_path
        self.uvf_file_path = uvf_file_path
        if mfit_file_path != '' and mfitid_file_path == '':
            self.mfitid_file_path = mfit_file_path + 'id'
        else:
            self.mfitid_file_path = mfitid_file_path
        if mfitid_file_path != '' and mfit_file_path == '':
            self.mfit_file_path = mfit_file_path.replace('id', '')
        else:
            self.mfit_file_path = mfit_file_path
        self.difmap_path = difmap_path
        self.difmap_weight = difmap_weight
        self.shift = np.array(shift)
        
        # check if paths exist:
        if not os.path.exists(fits_file_path):
            print('Error: no valid path to fits file')
        if not os.path.exists(uvf_file_path):
            print('Error: no valid path to uvfits file')
        if not os.path.exists(mfit_file_path):
            print('Error: no valid path to mfit file')
        if not os.path.exists(mfitid_file_path):
            print('Error: no valid path to mfitid file')
        
        # with fits.open(self.fits_file_path) as hdul:
        #     self.header = hdul[0].header.copy()
        self.header = fits.getheader(self.fits_file_path)
        try:
            self.source_name = self.header['OBJECT']
        except KeyError:
            self.source_name = ''
        self.epoch = self.get_epoch()
        self.freq = self.get_freq()
        self.obs_date = Time(self.epoch)
        self.mjd = self.obs_date.mjd
        self.gain_err = gain_err
        self.bmaj = self.header['BMAJ']
        self.bmin = self.header['BMIN']
        self.bpa = self.header['BPA']
        self.ps = self.header['CDELT2']*3.6E6
        
        self.comps = helpers.ReadComp(mfitid_file_path)
        for comp in self.comps:
            comp.gain_err = self.gain_err
            comp.fits_file = self.fits_file_path
            comp.uvfits_file = self.uvf_file_path
            comp.modelfit_file = self.mfit_file_path
            comp.mfitid_file = self.mfitid_file_path
        
        # self.gain_err = calc_err_flux(uvfits_file_name)
        
        for comp in self.comps:
            comp.shift(self.shift[0], self.shift[1])
            comp.is_shifted = False

        ### Get residual map rms by fitting Gaussians in the fits image ###
        self.rms_clean_fit = helpers.fit_image_comps(
            self.comps,
            IMAP_file=self.fits_file_path,
        )
        
        if rms_method == 'image_fit':
            rms_use = self.rms_clean_fit
        elif rms_method == 'difmap':
            rms_use = helpers.get_rms(
                fits_file=self.fits_file_path,
                uvf_file=self.uvf_file_path,
                mod_file=self.mfit_file_path,
                shift=self.shift,
                uv_weight=self.difmap_weight[0],
                error_weight=self.difmap_weight[1],
                par_file='',
                out_path='',
                difmap_path=self.difmap_path,
            )
        elif rms_method == 'individual':
            rms_use = None
        else:
            print('! Warning: invalid rms method provided. Valid options are '
                 +'"image_fit", "difmap" and "individual". Will use "image_fit".')
            rms_use = self.rms_clean_fit

        for comp in self.comps:
            comp.calc_errors(
                difmap_path=self.difmap_path,
                weight=difmap_weight,
                beam=[self.bmaj, self.bmin, self.bpa],
                rms_=rms_use,
                gain_err=self.gain_err,
                shift=self.shift,
            )
        
        outfile = self.mfit_file_path[:self.mfit_file_path.rfind('.')]+'.mfiterr'
        helpers.write_mfiterr(self.comps, outfile=outfile)

        CC_rms = helpers.get_clean_rms(
            fits_path=self.fits_file_path,
            uvf_path=self.uvf_file_path,
            shift=self.shift,
            uv_weight=self.difmap_weight[0],
            error_weight=self.difmap_weight[1],
            out_path=self.fits_file_path[0:-5]+'_CC.res',
            difmap_path=self.difmap_path,
        )
        self.CC_rms = CC_rms
        print('Clean (CC) rms from Difmap [Jy/beam]:')
        print(CC_rms)
        
        try:
            self.rms_theo = self.header['NOISE']
        except KeyError:
            self.rms_theo = np.nan
        try:
            self.rms_res = self.header['OB-NOISE']
        except KeyError:
            self.rms_res = np.nan
        del self.header
    
    def get_freq(self):
        # header = fits.get_header(self.fits_file_name)
        header = fits.getheader(self.fits_file_path)
        freq = header['CRVAL3']    # Hz
        return freq
    
    def get_epoch(self):
        # header = fits.get_header(self.fits_file_name)
        header = fits.getheader(self.fits_file_path)
        epoch = header['DATE-OBS']    # YYYY-MM-DD
        if epoch.endswith(')'):
            epoch = epoch[:-3]
        return epoch



class components:
    '''
    Defining a new Class to work with modelfit components.
    '''
    def __init__(self):
        self.flux = None  # total flux in Jy
        self.dist = None  # radial position in mas
        self.theta = None  # position angle (PA) in degrees
        self.major = None  # FWHM major axis in mas
        self.axratio = None  # axis ratio of ellipse
        self.minor = None  # FWHM minor axis in mas
        self.phi = None  # orientation of ellipse in degrees
        self.name = None  # component name
        self.flux_err = None
        self.Sp = None  # intensity of the pixel at the comp. position
        self.Sp_err = None
        self.dist_err = None
        self.theta_err = None
        self.major_err = None
        self.minor_err = None
        self.phi_err = None
        self.T_b_obs = None
        self.T_b_obs_err = None
        self.dlim = None
        self.gain_err = 0.1
        self.freq = None
        self.is_shifted = False
        
        self.fits_file = ''
        self.uvfits_file = ''
        self.modelfit_file = ''
        self.mfitid_file = ''
        self.par_file = ''

    def calc(self):
        self.minor = self.major*self.axratio
        self.theta_rot = 90 + self.theta
        self.a = np.divide(self.major, 2.)
        self.b = self.a*self.axratio

        self.ra = -self.dist*np.cos(self.theta_rot*np.pi/180.)
        self.dec = self.dist*np.sin(self.theta_rot*np.pi/180.)
  
    # Method to apply a shift in RA and Dec to the modelfit component
    def shift(self, shift_x, shift_y):  # shift in mas
        self.ra = self.ra + shift_x
        self.dec = self.dec + shift_y
        self.dist = np.sqrt(self.ra**2 + self.dec**2)
        if self.ra > 0 and self.dec > 0:
            self.theta = 180/np.pi * (np.arctan(self.ra / self.dec))
        if self.ra > 0 and self.dec < 0:
            self.theta = 180 + 180/np.pi * (np.arctan(self.ra / self.dec))
        if self.ra < 0 and self.dec > 0:
            self.theta = 180/np.pi * (np.arctan(self.ra / self.dec))
        if self.ra < 0 and self.dec < 0:
            self.theta = -180 + 180/np.pi * (np.arctan(self.ra / self.dec))
        self.theta_rot = 90 + self.theta
        self.is_shifted = True
        
        return self
    
    def calc_errors(
        self,
        difmap_path='/usr/local/difmap/difmap',
        comp_name=None,
        weight=[0,-1],
        gain_err=None,
        beam=None,
        rms_=None,
        shift=None
    ):
        if comp_name == None:
            comp_name = self.name
        if weight[0] > 0:
            uv_weight = 'u'
        else:
            uv_weight = 'n'
        if gain_err == None:
            gain_err = self.gain_err
        
        if self.fits_file != '' and self.uvfits_file != '' and self.modelfit_file != '' and self.mfitid_file != '':
            print('Calculate component errors using .uvfits, .fits, .mfit and'
                ' .mfitid files.')
            if self.name == None:
                print('Error: cannot compute uncertainties as no components' \
                    ' have been read yet.')
                return 0
            
            if beam == None:
                print('Error: cannot compute uncertainties without clean beam.')
                return 0
            
            if os.path.exists(difmap_path):
                S_p, rms = helpers.get_comp_peak_rms(
                    self.fits_file,
                    self.uvfits_file,
                    self.modelfit_file,
                    self.mfitid_file,
                    comp_name,
                    par_file=self.par_file,
                    difmap_path=difmap_path,
                    shift=shift
                    )
            else:
                print('! Warning: difmap path not found. Will not be able to'
                    ' compute individual component peak flux and rms.')
                if rms_ is None:
                    print('! Warning: tried to compute final rms with difmap but'
                        ' no difmap path found! Calculate component errors'
                        ' assuming 10 percent uncertainty.')
                    S_t = self.flux
                    fwhm = self.major
                    nu = self.freq
                    
                    fwhm_rad = fwhm/1000./3600.*np.pi/180.
                    sigma_S_t = S_t*0.1
                    sigma_fwhm = fwhm*0.1
                    sigma_fwhm_rad = sigma_fwhm/1000./3600.*np.pi/180.
                    
                    c = scipy.constants.c
                    k_B = scipy.constants.k
                    
                    T_b_obs = S_t*1E-26*c**2/(2*k_B*nu**2*fwhm_rad**2)
                    sigma_T_b_obs = np.sqrt(
                        (sigma_S_t*1E-26*c**2/(2*k_B*nu**2*fwhm_rad**2))**2
                        + (-2*sigma_fwhm_rad*S_t*1E-26*c**2
                            /(2*k_B*nu**2*fwhm_rad**3)
                        )**2
                    )
                    
                    ### Assign values to components variable initialized in 'read_components.py' ###
                    self.flux_err = sigma_S_t
                    self.Sp = S_t
                    self.Sp_err = sigma_S_t
                    self.dist_err = self.dist*0.1
                    self.theta_err = 10
                    self.major_err = sigma_fwhm
                    self.dlim = 0.0
                    self.T_b_obs = T_b_obs
                    self.T_b_obs_err = sigma_T_b_obs

                    return
            
            if rms_ != None:
                rms = rms_
            
            print('RMS used for error calculation [Jy/beam]:')
            print(round(rms,9))
            
            b_maj = beam[0]*3.6E6
            b_min = beam[1]*3.6E6
            
            ### Calculate errors ###
            SNR = S_p/rms
            sigma_p = rms*np.sqrt(1+SNR)
            SNR_p = S_p/sigma_p
            
            S_t = self.flux
            fwhm = self.major
            nu = self.freq
            
            # Other calculation (now obsolete): calculate error just from image
            # rms and gain error
            # sigma_t = np.sqrt((gain_err*S_t)**2 + rms**2)    
            sigma_t = sigma_p*np.sqrt(1+(S_t**2/S_p**2))    # According to Frank
            # Schinzel's PhD thesis
            
            '''
            IMPORTANT: previously, the gain error has been used as below.
            However, when we calculate the flux ratio and surface brightness 
            ratio later based on the total flux and its error, the gain error 
            should affect all components in the image in the same way: thus, 
            we can leave it out. For reporting the error on just the total 
            flux density, the gain error should be considered.
            '''
            # Add gain error in quadrature to reflect individual telescopes' 
            # fundamental gain uncertainty
            sigma_t = np.sqrt(sigma_t**2 + (gain_err*S_t)**2)
            
            # Calculate minimum resolvable size based on SNR at modelfit
            # positions
            if uv_weight == 'u' or uv_weight == 'U' or uv_weight == 'uniform':
                d_lim = 4./np.pi*np.sqrt(
                    np.pi*np.log(2)*b_maj*b_min*np.log((SNR+1)/SNR)
                )
            else:    # this is the condition for natural weight
                d_lim = 2./np.pi*np.sqrt(
                    np.pi*np.log(2)*b_maj*b_min*np.log((SNR+1)/SNR)
                )
            
            # Calculate other errors
            size = max(d_lim, fwhm)
            sigma_fwhm = size/SNR_p
            r = self.dist
            sigma_r = np.sqrt(b_maj*b_min + max(d_lim, fwhm)**2)/SNR_p
            sigma_phi = np.arctan(sigma_r/r) * 180/np.pi
            
            # Calculate observed brightness temperature
            # nu = np.round(header['CRVAL3']*1e-9,2)*1E9  # Hz
            c = scipy.constants.c
            k_B = scipy.constants.k
            fwhm_rad = size/1000./3600.*np.pi/180.  # radians
            sigma_fwhm_rad = sigma_fwhm/1000./3600.*np.pi/180.  # radians
            ### CONTRUCTION WORK ###
            Omega = np.pi*fwhm_rad**2/(4*np.log(2))
            ### CONTRUCTION WORK ###
            T_b_obs = S_t*1E-26*c**2/(2*k_B*nu**2*fwhm_rad**2)
            if fwhm >= d_lim:
                sigma_T_b_obs = np.sqrt(
                    (sigma_t*1E-26*c**2/(2*k_B*nu**2*fwhm_rad**2))**2
                    + (-2*sigma_fwhm_rad*S_t*1E-26*c**2
                        /(2*k_B*nu**2*fwhm_rad**3)
                    )**2
                )
            else:
                print('Component {0:s} is unresolved'.format(self.name))
                print('Calculated observed brightness temperature is a lower'
                    ' limit')
                sigma_T_b_obs = np.nan
            
            # Assign values to components variable initialized in 
            # 'read_components.py'
            self.flux_err = sigma_t
            self.Sp = S_p
            self.Sp_err = sigma_p
            self.dist_err = sigma_r
            self.theta_err = sigma_phi
            self.major_err = sigma_fwhm
            self.dlim = d_lim
            self.T_b_obs = T_b_obs
            self.T_b_obs_err = sigma_T_b_obs
            
        else:
            print('Calculate component errors assuming 10 percent uncertainty.')
            S_t = self.flux
            fwhm = self.major
            nu = self.freq
            
            fwhm_rad = fwhm/1000./3600.*np.pi/180.
            sigma_S_t = S_t*0.1
            sigma_fwhm = fwhm*0.1
            sigma_fwhm_rad = sigma_fwhm/1000./3600.*np.pi/180.
            
            c = scipy.constants.c
            k_B = scipy.constants.k
            
            T_b_obs = S_t*1E-26*c**2/(2*k_B*nu**2*fwhm_rad**2)
            sigma_T_b_obs = np.sqrt(
                (sigma_S_t*1E-26*c**2/(2*k_B*nu**2*fwhm_rad**2))**2
                + (-2*sigma_fwhm_rad*S_t*1E-26*c**2
                    /(2*k_B*nu**2*fwhm_rad**3)
                )**2
            )
            
            # Assign values to components variable initialized in 
            # 'read_components.py'
            self.flux_err = sigma_S_t
            self.Sp = S_t
            self.Sp_err = sigma_S_t
            self.dist_err = self.dist*0.1
            self.theta_err = 10
            self.major_err = sigma_fwhm
            self.dlim = 0.0
            self.T_b_obs = T_b_obs
            self.T_b_obs_err = sigma_T_b_obs



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
    
    def print_info(self):
        print(f'### Info on component {self.name} ###')
        if self.flux_err != None:
            print(f'# Flux: {self.flux:.4f} +/- {self.flux_err:.4f} Jy')
        else:
            print(f'# Flux: {self.flux:.4f} Jy')
        if self.dist_err != None:
            print(f'# Distance in polar coordinates: '
                f'{self.dist:.3f} +/- {self.dist_err:.3f} mas')
        else:
            print(f'# Distance in polar coordinates: {self.dist:.3f} mas')
        if self.theta_err != None:
            print(f'# Angle in polar coordinates: {self.theta:.1f} '
                f'+/- {self.theta_err:.1f} deg')
        else:
            print(f'# Angle in polar coordinates: {self.theta:.1f} deg')
        if self.major_err != None:
            print(f'# Major axis FWHM: {self.major:.5f} '
                f'+/- {self.major_err:.5f} mas')
        else:
            print(f'# Major axis FWHM: {self.major:.5f} mas')
        if self.minor != None:
            if self.minor_err != None:
                print(f'# Minor axis FWHM: {self.minor:.5f} '
                    f'+/- {self.minor_err:.5f} mas')
            else:
                print(f'# Minor axis FWHM: {self.minor:.5f} mas')
        else:
            print(f'# Axis ratio: {self.axratio:.2f}')
        if self.phi_err != None:
            print(f'# Major/minor axis position angle: {self.phi:.1f} '
                f'+/- {self.phi_err:.5f} deg')
        else:
            print(f'# Major/minor axis position angle: {self.phi:.1f} deg')



class lens_candidate():

    def __init__(
        self,
        source_name='',
        fits_files=[],
        uvf_files=[],
        mfit_files=[],
        mfitid_files=[],
        date_tolerance=1,
        freq_tolerance=1E9,
        z=-1,
        difmap_path='/usr/local/difmap/difmap',
        rms_method='image_fit',
    ):

        self.source_name = source_name
        self.z = z
        self.date_tolerance = date_tolerance
        self.freq_tolerance = freq_tolerance
        self.difmap_path = difmap_path
        self.rms_method = rms_method
        
        self.image_datasets = []
        
        self.df = pd.DataFrame()
        
        if (
            len(fits_files) != 0
             and len(uvf_files) != 0
             and len(mfit_files) != 0
             and len(mfitid_files) != 0
        ):
            self.image_datasets = [
                Image_dataset(
                    fits_file,
                    uvf_file,
                    mfit_file,
                    mfitid_file,
                    difmap_path=self.difmap_path,
                    rms_method=self.rms_method
                )
                for (fits_file, uvf_file, mfit_file, mfitid_file)
                in zip(fits_files, uvf_files, mfit_files, mfitid_files)
        ]
        else:
            self.image_datasets = []
        
        self.epochs = []
        self.mjds = []
        self.freqs = []
        
        for image_dataset in self.image_datasets:
            if self.name == '':
                self.name=image_dataset.name
            if not any(
                abs(num - image_dataset.freq)
                <= freq_tolerance for num in self.freqs
            ):
                self.freqs.append(image_dataset.freq)
            if not any(
                abs(num - image_dataset.mjd)
                <= date_tolerance for num in self.mjds
            ):
                self.epochs.append(image_dataset.epoch)
                self.mjds.append(image_dataset.mjd)
    
    
    
    def read_all(
        self,
        in_dir='./',
        shift=[0,0],
        difmap_weight=[0,-1],
        rms_method=None
    ):
        print('### READING DATA ###')
        if not in_dir.endswith("/"):
            in_dir += "/"
        
        if not os.path.exists(in_dir):
            print('Error: Input directory not found!')
            return 0
        
        if rms_method is None:
            rms_method = self.rms_method
        
        fits_files = np.sort(np.array([i for i in glob(in_dir+'*.fits')]))
        uvf_files = np.sort(np.array([i for i in glob(in_dir+'*.uvf*')]))
        mfit_files = np.sort(np.array([i for i in glob(in_dir+'*.mfit')]))
        mfitid_files = np.sort(np.array([i for i in glob(in_dir+'*.mfitid')]))
        
        print('# Reading fits files: #')
        print(fits_files)
        print('# Reading uvfits files: #')
        print(uvf_files)
        print('# Reading modelfit files: #')
        print(mfit_files)
        print('# Reading mfitid files: #')
        print(mfitid_files)
        
        if not len(fits_files) == len(uvf_files) == len(mfit_files):
            print('Error: non-matching number .fits, .uvfits or .mfit files '
                'in the specified folder. Data have not been read.')
            # print(fits_files)
            # print(uvf_files)
            # print(mfit_files)
            return 0
        
        if len(fits_files) == 0:
            print('Error: no fits images found!')
            return 0
        
        self.image_datasets = self.image_datasets + [
            Image_dataset(
                fits_file,
                uvf_file,
                mfit_file,
                mfitid_file,
                difmap_path=self.difmap_path,
                shift=shift,
                difmap_weight=difmap_weight,
                rms_method=rms_method,
            )
            for (
                fits_file,
                uvf_file,
                mfit_file,
                mfitid_file
            ) in zip(fits_files, uvf_files, mfit_files, mfitid_files)
        ]
        
        # for image_dataset in self.image_datasets:
            # if self.source_name == '':
                # self.source_name = image_dataset.source_name
            # if not any(abs(num - image_dataset.freq) <= self.freq_tolerance for num in self.freqs):
                # self.freqs.append(image_dataset.freq)
            # if not any(abs(num - image_dataset.mjd) <= self.date_tolerance for num in self.mjds):
                # self.epochs.append(image_dataset.epoch)
                # self.mjds.append(image_dataset.mjd)
    
    
    
    def add_image_dataset(
        self,
        fits_file='',
        uvf_file='',
        mfit_file='',
        mfitid_file='',
        shift=[0,0],
        difmap_weight=[0,-1],
        rms_method=None
    ):
        print('### ADDING IMAGE DATASET ###')
        if any(file == '' for file in [
            fits_file,
            uvf_file,
            mfit_file,
            mfitid_file
        ]):
            print('Error: need to provide .fits, .uvfits, .mfit and .mfitid '
                  'file!')
            return

        if rms_method is None:
            rms_method = self.rms_method
        
        self.image_datasets.append(
            Image_dataset(
                fits_file,
                uvf_file,
                mfit_file,
                mfitid_file,
                difmap_path=self.difmap_path,
                shift=shift,
                difmap_weight=difmap_weight,
                rms_method=rms_method,
                )
            )
        print('Added image_dataset')
    
    
    
    def print_image_datasets(self):
        print('Currently loaded datasets (epoch, freq):')
        for i, image_dataset in enumerate(self.image_datasets):
            print(i, image_dataset.epoch, image_dataset.freq)
    
    
    
    def get_image_datasets(self, by_index=None, epoch=None, freq=None):
        image_dataset_list = []
        if by_index == None:
            if epoch == None and freq == None:
                print('Please provide epoch or freq to extract dataset.')
                return
            else:
                for image_dataset in self.image_datasets:
                    if image_dataset.epoch == epoch or np.abs(image_dataset.freq - freq) <= self.freq_tolerance:
                        image_dataset_list.append(image_dataset)
        else:
            if type(by_index) == int:
                by_index = [by_index]
            for index in by_index:
                for i, image_dataset in enumerate(self.image_datasets):
                    if i == index:
                        image_dataset_list.append(image_dataset)
        
        return image_dataset_list
    
    
    
    def remove_image_dataset(self, by_index=None, epoch=None, freq=None):
        if by_index == None:
            if epoch == None and freq == None:
                print('Please provide epoch or freq to extract dataset.')
                return
            else:
                for i, image_dataset in enumerate(self.image_datasets):
                    if image_dataset.epoch == epoch or np.abs(image_dataset.freq - freq) <= self.freq_tolerance:
                        self.image_datasets.remove(image_dataset)
                        if not self.df.empty:
                            # self.df = self.df.drop(
                                # self.df[(self.df['Epoch'] == image_dataset.epoch)
                                 # & (self.df['Freq [Hz]'] == image_dataset.freq)].index
                                # )
                            self.df = self.df[~((self.df['Epoch'] == image_dataset.epoch) & (self.df['Freq [Hz]'] == image_dataset.freq))]
                            
        else:
            if type(by_index) == int:
                by_index = [by_index]
            for index in by_index:
                for i, image_dataset in enumerate(self.image_datasets):
                    if i == index:
                        if not self.df.empty:
                            # print(self.df['Epoch'])
                            # print(image_dataset.epoch)
                            # print(self.df['Epoch'] == image_dataset.epoch)
                            # print(self.df['Freq [Hz]'])
                            # print(image_dataset.freq)
                            # print(self.df['Freq [Hz]'] == image_dataset.freq)
                            # self.df = self.df.drop(
                                # self.df[(self.df['Epoch'] == image_dataset.epoch)
                                 # & (self.df['Freq [Hz]'] == image_dataset.freq)].index
                                # )
                            self.df = self.df[~((self.df['Epoch'] == image_dataset.epoch) & (self.df['Freq [Hz]'] == image_dataset.freq))]
            
            self.image_datasets = [image_dataset for i, image_dataset in enumerate(self.image_datasets) if i not in by_index]
    
    
    
    def set_same_source_name(self):
        names = []
        
        for i, image_dataset in enumerate(self.image_datasets):
            names.append(image_dataset.source_name)
        
        if names == []:
            return
        
        if len(set(names)) == 1:
            name = names[0]
        else:
            # Pick the longest string
            name = max(names, key=len)
        
        for i, image_dataset in enumerate(self.image_datasets):
            image_dataset.source_name = name
            self.source_name = name


    
    # def calc_FR(self):
    
    # def calc_SBR(self):
        
    # def calc_spix(self):
    
    # def calc_kinematics(self):
    
    
    
    def calc_all(self, outfile_path=''):
        
        dfs = []
        outpaths = []
        
        ### Group Image_data objects according to epoch within date_tolerance
        groups = []
        used = set()
        for i, image_dataset1 in enumerate(self.image_datasets):
            if i in used:
                continue

            group = [image_dataset1]
            used.add(i)

            for j, image_dataset2 in enumerate(self.image_datasets):
                if j in used:
                    continue
                if abs(image_dataset1.mjd - image_dataset2.mjd) <= self.date_tolerance:
                    group.append(image_dataset2)
                    used.add(j)
            
            # Sort this group by freq
            group.sort(key=lambda x: x.freq)
            groups.append(group)
        
        self.set_same_source_name()
        
        for g, group in enumerate(groups):
            save_index = len(dfs) - 1
            for i, image_dataset in enumerate(group):
                # Create output folder for each dataset
                outpath = image_dataset.source_name + '_' + image_dataset.epoch + '_' + str(round(image_dataset.freq/1E9,1)) + 'GHz' + '/'
                outpaths.append(outpath)
                os.makedirs(outpath, exist_ok=True)
                
                # Do main calculations
                results_fluxes, results_SBR_test, results_FR_test, results_sep_test = \
                    helpers.calc_lens_params(
                        image_dataset.comps,
                        outfile=outpath+'_test',
                        freq=image_dataset.freq,
                        gain_err=image_dataset.gain_err
                        )
                
                # Store results in dataframes
                df = helpers.build_results_df(
                    self.source_name,
                    image_dataset.epoch,
                    image_dataset.freq,
                    results_fluxes,
                    results_SBR_test,
                    results_FR_test,
                    results_sep_test
                    )
                
                # print('i', i)
                # print('len(group)-1', len(group)-1)
                
                # Calculate spectral index, if applicable
                if len(group) > 1 and i != len(group)-1:
                    alpha_A, alpha_A_err, alpha_B, alpha_B_err = \
                        helpers.calc_spix(group[i].comps, group[i+1].comps, group[i].freq, group[i+1].freq)
                    
                    df['alpha A'] = alpha_A
                    df['alpha A err'] = alpha_A_err
                    df['alpha B'] = alpha_B
                    df['alpha B err'] = alpha_B_err
                helpers.export_csv(df, outpath + 'lens_params.csv')
                
                dfs.append(df)
        
        # Combine dataframes
        combined_df = pd.concat(dfs, ignore_index=True)
        # Create output folder for summary results
        os.makedirs(image_dataset.source_name + '/', exist_ok=True)
        
        if outfile_path == '':
            outfile_path = './' + image_dataset.source_name + '/'
        elif type(outfile_path) == str and outfile_path[-1] != '/':
            outfile_path = outfile_path + '/'
        os.makedirs(outfile_path, exist_ok=True)
        
        # Sort dataframe and save to csv
        combined_df_sorted = combined_df.sort_values(by=["Epoch", "Freq [Hz]"], ascending=[True, True]).reset_index(drop=True)
        helpers.export_csv(combined_df_sorted, outfile_path + 'lens_params.csv')
        
        self.df = combined_df_sorted
    
    
    def plot_images_all(self, plt_xlim=[40,-40], plt_ylim=[-40,40],
                        do_shift_max=False, show=False, outfile_path=''):
        
        self.set_same_source_name()
        
        for i, image_dataset in enumerate(self.image_datasets):
            
            if outfile_path == '':
                outfile_path = './' + image_dataset.source_name + '_' + image_dataset.epoch + '_' + str(round(image_dataset.freq/1E9,1)) + 'GHz' + '/'
            elif type(outfile_path) == str and outfile_path[-1] != '/':
                outfile_path = outfile_path + '/'
            os.makedirs(outfile_path, exist_ok=True)
            
            plotters.plot_image_comps(
                image_dataset.comps,
                IMAP_file=image_dataset.fits_file_path,
                rms=image_dataset.CC_rms,
                plt_xlim=plt_xlim,
                plt_ylim=plt_ylim,
                do_shift_max=do_shift_max,
                srcname=image_dataset.source_name,
                outfile_path=outfile_path,
                show=show
            )
    
    
    def plot_image(self, image_dataset, plt_xlim=[40,-40], plt_ylim=[-40,40],
                   do_shift_max=True, show=False, outfile_path=''):
        
        self.set_same_source_name()
        
        if outfile_path == '':
            outfile_path = './' + image_dataset.source_name + '_' + image_dataset.epoch + '_' + str(round(image_dataset.freq/1E9,1)) + 'GHz' + '/'
        elif type(outfile_path) == str and outfile_path[-1] != '/':
            outfile_path = outfile_path + '/'
        os.makedirs(outfile_path, exist_ok=True)

        plotters.plot_image_comps(
            image_dataset.comps,
            IMAP_file=image_dataset.fits_file_path,
            rms=image_dataset.rms_clean_fit,
            plt_xlim=plt_xlim,
            plt_ylim=plt_ylim,
            do_shift_max=do_shift_max,
            srcname=image_dataset.source_name,
            outfile_path=outfile_path,
            show=show
            )
    
    
    
    def plot_fluxes_all(self, outfile_path='', show=False):
        
        if outfile_path == '':
            outfile_path = './' + self.source_name + '/'
        elif type(outfile_path) == str and outfile_path[-1] != '/':
            outfile_path = outfile_path + '/'
        os.makedirs(outfile_path, exist_ok=True)
        
        plotters.plot_fluxes(
            self.df,
            freq_tolerance=self.freq_tolerance,
            outfile_path=outfile_path,
            show=show
            )
    
    
    
    def plot_FR_all(self, outfile_path='', show=False):
        
        if outfile_path == '':
            outfile_path = './' + self.source_name + '/'
        elif type(outfile_path) == str and outfile_path[-1] != '/':
            outfile_path = outfile_path + '/'
        os.makedirs(outfile_path, exist_ok=True)
        
        plotters.plot_FR(
            self.df,
            freq_tolerance=self.freq_tolerance,
            outfile_path=outfile_path,
            show=show,
            )
    
    
    
    def plot_SBR_all(self, outfile_path='', show=False, SBR_thresh=4):
        
        if outfile_path == '':
            outfile_path = './' + self.source_name + '/'
        elif type(outfile_path) == str and outfile_path[-1] != '/':
            outfile_path = outfile_path + '/'
        os.makedirs(outfile_path, exist_ok=True)
        
        plotters.plot_SBR(
            self.df,
            freq_tolerance=self.freq_tolerance,
            outfile_path=outfile_path,
            show=show,
            SBR_thresh=SBR_thresh
            )
    
    
    
    def plot_separation_all(self, plot_line=False, outfile_path='', show=False):
        
        if outfile_path == '':
            outfile_path = './' + self.source_name + '/'
        elif type(outfile_path) == str and outfile_path[-1] != '/':
            outfile_path = outfile_path + '/'
        os.makedirs(outfile_path, exist_ok=True)
        
        plotters.plot_separation(
            self.df,
            freq_tolerance=self.freq_tolerance,
            outfile_path=outfile_path,
            show=show,
            plot_line=plot_line
            )
    
    
    def plot_Tb_all(self, outfile_path='', show=False):
        print('COMING SOON')
    
    
    
    def plot_all(self, outfile_path='', show=False, SBR_thresh=4, plot_line=False):
        
        if outfile_path == '':
            outfile_path = './' + self.source_name + '/'
        elif type(outfile_path) == str and outfile_path[-1] != '/':
            outfile_path = outfile_path + '/'
        os.makedirs(outfile_path, exist_ok=True)
        
        plotters.plot_fluxes(
            self.df,
            freq_tolerance=self.freq_tolerance,
            outfile_path=outfile_path,
            show=show
            )
        plotters.plot_FR(
            self.df,
            freq_tolerance=self.freq_tolerance,
            outfile_path=outfile_path,
            show=show
            )
        plotters.plot_SBR(
            self.df,
            freq_tolerance=self.freq_tolerance,
            outfile_path=outfile_path,
            show=show,
            SBR_thresh=SBR_thresh
            )
        plotters.plot_separation(
            self.df,
            freq_tolerance=self.freq_tolerance,
            outfile_path=outfile_path,
            show=show,
            plot_line=plot_line
            )
    
    
    
    # def plot_spix(self, out_dir='./'):
    
    # def plot_kinematics(self, out_dir='./'):
    
    
    
    def evaluate_FR(self, FR_sigmas=2, export_csv=False, outfile_path=''):
        print('### Evaluate separation criterion ###')
        if self.df.empty:
            print('Lens parameters not calculated yet, run calc_all() first!')
        else:
            self.df["FR OK?"] = 'YES'
            
            df = self.df.copy()
            
            # ensure epoch is datetime
            df['Epoch'] = pd.to_datetime(df['Epoch'])

            # unique frequencies sorted
            freqs = np.sort(df['Freq [Hz]'].unique())
            grouped_freqs = []
            while len(freqs) > 0:
                f0 = freqs[0]
                group = freqs[np.isclose(freqs, f0, atol=self.freq_tolerance)]
                grouped_freqs.append(group)
                freqs = freqs[~np.isclose(freqs, f0, atol=self.freq_tolerance)]

            rows = []
            for group in grouped_freqs:
                # get all rows that belong to this frequency group (all epochs)
                df_group = df[df['Freq [Hz]'].isin(group)].sort_values('Epoch')
                if len(df_group) < 2:
                    continue

                freq_min = np.min(group)  # representative frequency for the group
                # extract arrays
                epochs = pd.to_datetime(df_group['Epoch']).to_numpy()
                FRs = df_group['FR'].to_numpy()
                FR_errs = df_group['FR err'].to_numpy()
                idxs = df_group.index.to_numpy()

                # iterate consecutive pairs
                for i in range(len(FRs) - 1):
                    FR_diff = np.abs(FRs[i] - FRs[i+1])
                    comb_err = np.sqrt(FR_errs[i]**2 + FR_errs[i+1]**2)
                    FR_diff_tolerance = FR_sigmas*comb_err

                    if FR_diff > FR_diff_tolerance:
                        ep = self.df.loc[idxs[i], "Epoch"]
                        freq = self.df.loc[idxs[i], "Freq [Hz]"]
                        print(f'! FR criterion violated for dataset from {ep:s} at {freq/1e9:.1f} GHz')
                        self.df.loc[idxs[i], "FR OK?"] = 'NO'
                    # else:
                        # print('FR citerion NOT violated')
                            
        if export_csv == True:
            if outfile_path == '':
                outfile_path = './' + self.source_name + '/'
            elif type(outfile_path) == str and outfile_path[-1] != '/':
                outfile_path = outfile_path + '/'
            os.makedirs(outfile_path, exist_ok=True)
            outfile_path = outfile_path + 'lens_params_eval.csv'
            
            helpers.export_csv(self.df, outfile_path)
    
    
    
    def evaluate_separation(self, sep_sigmas=2, export_csv=False, outfile_path=''):
        print('### Evaluate separation criterion ###')
        if self.df.empty:
            print('Lens parameters not calculated yet, run calc_all() first!')
        else:
            self.df["Sep. OK?"] = 'YES'
            self.df["Kin. fit slope"] = np.nan
            self.df["Kin. fit slope err"] = np.nan
            self.df["Kin. fit y-int."] = np.nan
            self.df["Kin. fit y-int. err"] = np.nan
            
            df = self.df.copy()
            
            # ensure epoch is datetime
            df['Epoch'] = pd.to_datetime(df['Epoch'])
            df['MJD'] = Time(df['Epoch']).mjd

            # unique frequencies sorted
            freqs = np.sort(df['Freq [Hz]'].unique())
            grouped_freqs = []
            while len(freqs) > 0:
                f0 = freqs[0]
                group = freqs[np.isclose(freqs, f0, atol=self.freq_tolerance)]
                grouped_freqs.append(group)
                freqs = freqs[~np.isclose(freqs, f0, atol=self.freq_tolerance)]

            rows = []
            for group in grouped_freqs:
                # get all rows that belong to this frequency group (all epochs)
                df_group = df[df['Freq [Hz]'].isin(group)].sort_values('Epoch')
                if len(df_group) < 2:
                    continue

                freq_min = np.min(group)  # representative frequency for the group
                # extract arrays
                epochs = pd.to_datetime(df_group['Epoch']).to_numpy()
                seps = df_group['Sep. [mas]'].to_numpy()
                sep_errs = df_group['Sep. err [mas]'].to_numpy()
                idxs = df_group.index.to_numpy()
                
                # do kinematic linear fit #
                mjds = df_group['MJD'].to_numpy()
                line = lambda x, m, b: m*x + b
                if len(seps) > 1:
                    popt, pcov = curve_fit(line, mjds, seps, p0=[0,seps[0]])
                    slope = popt[0]
                    slope_err = np.sqrt(np.diag(pcov))[0]
                    y_int = popt[1]
                    y_int_err = np.sqrt(np.diag(pcov))[1]
                if slope/slope_err > sep_sigmas:
                    print(f'There seems to be significant ({sep_sigmas}-sigma) proper motion'
                          f' for frequency group around {freq_min/1e9:.1f} GHz. '
                          f'Slope is {slope*365:.3f} +/- {slope_err*365:.3f} mas/yr.')

                # iterate consecutive pairs
                for i in range(len(seps) - 1):
                    self.df.loc[idxs[i], "Kin. fit slope"] = slope
                    self.df.loc[idxs[i], "Kin. fit slope err"] = slope_err
                    self.df.loc[idxs[i], "Kin. fit y-int."] = y_int
                    self.df.loc[idxs[i], "Kin. fit y-int. err"] = y_int_err
                    
                    sep_diff = np.abs(seps[i] - seps[i+1])
                    comb_err = np.sqrt(sep_errs[i]**2 + sep_errs[i+1]**2)
                    sep_diff_tolerance = sep_sigmas*comb_err

                    if sep_diff > sep_diff_tolerance:
                        ep = self.df.loc[idxs[i], "Epoch"]
                        freq = self.df.loc[idxs[i], "Freq [Hz]"]
                        print(f'! Separation criterion violated for dataset from {ep:s} at {freq/1e9:.1f} GHz')
                        self.df.loc[idxs[i], "Sep. OK?"] = 'NO'
                    # else:
                        # print('Separation citerion NOT violated')
                            
        if export_csv == True:
            if outfile_path == '':
                outfile_path = './' + self.source_name + '/'
            elif type(outfile_path) == str and outfile_path[-1] != '/':
                outfile_path = outfile_path + '/'
            os.makedirs(outfile_path, exist_ok=True)
            outfile_path = outfile_path + 'lens_params_eval.csv'
            
            helpers.export_csv(self.df, outfile_path)
    
    
    
    def evaluate_spectra(self, spix_diff_max=0.5, use_err=False, export_csv=False, outfile_path=''):
        print('### Evaluating spectral criterion ###')
        if self.df.empty:
            print('Lens parameters not calculated yet, run calc_all() first!')
        else:
            self.df["alpha OK?"] = 'YES'
        
            alphas_A = np.array(self.df['alpha A'])
            alpha_errs_A = np.array(self.df['alpha A err'])
            alphas_B = np.array(self.df['alpha B'])
            alpha_errs_B = np.array(self.df['alpha B err'])
            
            # TO-DO: Incorporate flexible spix_diff_max based on 30 % flux ratio between frequencies.
            
            for i, alph in enumerate(alphas_A):
                
                if use_err == False:
                    if np.abs(alphas_A[i] - alphas_B[i]) > spix_diff_max:
                        ep = self.df["Epoch"][i]
                        freq = self.df["Freq [Hz]"][i]
                        print(f'! Spectral criterion violated for dataset from {ep:s} at {freq/1e9:.1f} GHz')
                        self.df["alpha OK?"][i] = 'NO'
                else:
                    min_alpha = np.nanmin(alphas_A[i], alphas_B[i])
                    max_alpha = np.nanmax(alphas_A[i], alphas_B[i])
                    min_alpha_err = [alpha_errs_A[i], alpha_errs_B[i]][np.argmin([alphas_A[i], alphas_B[i]])]
                    max_alpha_err = [alpha_errs_A[i], alpha_errs_B[i]][np.argmax([alphas_A[i], alphas_B[i]])]
                    
                    if np.abs(min_alpha + min_alpha_err - (max_alpha - max_alpha_err)) > spix_diff_max:
                        ep = self.df["Epoch"][i]
                        freq = self.df["Freq [Hz]"][i]
                        print(f'! Spectral criterion violated for dataset from {ep:s} at {freq/1e9:.1f} GHz')
                        self.df["alpha OK?"][i] = 'NO'
                            
        if export_csv == True:
            if outfile_path == '':
                outfile_path = './' + self.source_name + '/'
            elif type(outfile_path) == str and outfile_path[-1] != '/':
                outfile_path = outfile_path + '/'
            os.makedirs(outfile_path, exist_ok=True)
            outfile_path = outfile_path + 'lens_params_eval.csv'
            
            helpers.export_csv(self.df, outfile_path)
    
    
    
    def evaluate_SBR(self, SBR_thresh=4, export_csv=False, outfile_path=''):
        print('### Evaluating SBR criterion ###')
        if self.df.empty:
            print('Lens parameters not calculated yet, run calc_all() first!')
        else:
            self.df["SBR OK?"] = 'YES'
            
            SBRs = np.array(self.df['SBR'])
            SBR_errs = np.array(self.df['SBR err'])
            
            for i, SBR in enumerate(SBRs):
                if np.isnan(SBR_errs[i]) == False:
                    if SBR < 1:
                        if SBR + SBR_errs[i] < 1/SBR_thresh:
                            ep = self.df["Epoch"][i]
                            freq = self.df["Freq [Hz]"][i]
                            print(f'! SBR criterion violated for dataset from {ep:s} at {freq/1e9:.1f} GHz')
                            self.df["SBR OK?"][i] = 'NO'
                        # else:
                            # print('SBR citerion NOT violated for:')
                            # print(self.df['SBR'][i])
                    elif SBR >= 1:
                        if SBR - SBR_errs[i] > SBR_thresh:
                            ep = self.df["Epoch"][i]
                            freq = self.df["Freq [Hz]"][i]
                            print(f'! SBR criterion violated for dataset from {ep:s} at {freq/1e9:.1f} GHz')
                            self.df["SBR OK?"][i] = 'NO'
                        # else:
                            # print('SBR citerion NOT violated for:')
                            # print(self.df['SBR'][i])
                else:
                    if SBR < 1:
                        if SBR < 1/SBR_thresh and self.df['SBR is limit'][i] == 'upper':
                            ep = self.df["Epoch"][i]
                            freq = self.df["Freq [Hz]"][i]
                            print(f'! SBR criterion violated for dataset from {ep:s} at {freq/1e9:.1f} GHz')
                            self.df["SBR OK?"][i] = 'NO'
                        # else:
                            # print('SBR citerion NOT violated for:')
                            # print(self.df['SBR'][i])
                    elif SBR >= 1:
                        if SBR > SBR_thresh and self.df['SBR is limit'][i] == 'lower':
                            ep = self.df["Epoch"][i]
                            freq = self.df["Freq [Hz]"][i]
                            print(f'! SBR criterion violated for dataset from {ep:s} at {freq/1e9:.1f} GHz')
                            print(self.df['SBR'][i])
                            self.df["SBR OK?"][i] = 'NO'
                        # else:
                            # print('SBR citerion NOT violated for:')
                            # print(self.df['SBR'][i])
                            
        if export_csv == True:
            if outfile_path == '':
                outfile_path = './' + self.source_name + '/'
            elif type(outfile_path) == str and outfile_path[-1] != '/':
                outfile_path = outfile_path + '/'
            os.makedirs(outfile_path, exist_ok=True)
            outfile_path = outfile_path + 'lens_params_eval.csv'
            
            helpers.export_csv(self.df, outfile_path)
    
    
    
    def evaluate_all(self, FR_sigmas=2, SBR_thresh=4, spix_diff_max=0.5,
                     sep_sigmas=2, export_csv=False, outfile_path=''):
        '''
        Function to evaluate any outliers from lens expectation.
        '''
        if self.df.empty:
            print('Lens parameters not calculated yet, run calc_all() first!')
        else:
            self.evaluate_FR(FR_sigmas)
            
            self.evaluate_SBR(SBR_thresh)
            
            self.evaluate_separation(sep_sigmas)
            
            self.evaluate_spectra(spix_diff_max)
        
        if export_csv == True:
            if outfile_path == '':
                outfile_path = './' + self.source_name + '/'
            elif type(outfile_path) == str and outfile_path[-1] != '/':
                outfile_path = outfile_path + '/'
            os.makedirs(outfile_path, exist_ok=True)
            outfile_path = outfile_path + 'lens_params_eval.csv'
            
            helpers.export_csv(self.df, outfile_path)
    
    
    def check_variability(self, var_thresh=0.2, outfile_path=''):
        print('### Check component variability ###')
        if self.df.empty:
            print('! Error: lens parameters not calculated yet, run calc_all() first!')
        else:
            df = self.df.copy()
            
            if self.z == -1:
                print('! Warning: not correcting variability for redshift')
                z = 0
            else:
                z = self.z
            
            if outfile_path == '':
                outfile_path = './' + self.source_name + '/'
            elif type(outfile_path) == str and outfile_path[-1] != '/':
                outfile_path = outfile_path + '/'
            os.makedirs(outfile_path, exist_ok=True)
            
            # ensure epoch is datetime
            df['Epoch'] = pd.to_datetime(df['Epoch'])
            
            # unique frequencies sorted
            freqs = np.sort(df['Freq [Hz]'].unique())
            grouped_freqs = []
            while len(freqs) > 0:
                f0 = freqs[0]
                group = freqs[np.isclose(freqs, f0, atol=self.freq_tolerance)]
                grouped_freqs.append(group)
                freqs = freqs[~np.isclose(freqs, f0, atol=self.freq_tolerance)]

            rows = []
            for group in grouped_freqs:
                # get all rows that belong to this frequency group (all epochs)
                df_group = df[df['Freq [Hz]'].isin(group)].sort_values('Epoch')
                if len(df_group) < 2:
                    continue

                freq_min = np.min(group)  # representative frequency for the group
                epochs = pd.to_datetime(df_group['Epoch']).to_numpy(dtype='datetime64[D]')
                fluxes_A = df_group['Flux A [Jy]'].to_numpy()
                fluxes_B = df_group['Flux B [Jy]'].to_numpy()
                idxs = df_group.index.to_numpy()

                # iterate consecutive pairs
                delta_flux_A_list = []
                min_flux_A_list = []
                delta_flux_B_list = []
                min_flux_B_list = []
                delta_flux_tot_list = []
                min_flux_tot_list = []
                delta_time_list = []
                ep_list = []
                freq_list = []
                used_indices = []
                for i in range(len(epochs) - 1):
                    for j in range(len(epochs)):
                        if i != j and not [i,j] in used_indices and not [j,i] in used_indices:
                            delta_flux_A = np.abs(fluxes_A[i] - fluxes_A[j])
                            delta_flux_A_list.append(delta_flux_A)
                            min_flux_A_list.append(np.nanmin([fluxes_A[i], fluxes_A[j]]))
                            
                            delta_flux_B = np.abs(fluxes_B[i] - fluxes_B[j])
                            delta_flux_B_list.append(delta_flux_B)
                            min_flux_B_list.append(np.nanmin([fluxes_B[i], fluxes_B[j]]))
                            
                            delta_flux_tot = np.abs(fluxes_A[i]+fluxes_B[i] - (fluxes_A[j]+fluxes_B[j]))
                            delta_flux_tot_list.append(delta_flux_tot)
                            min_flux_tot_list.append(np.nanmin([fluxes_A[i]+fluxes_B[i], fluxes_A[j]+fluxes_B[j]]))
                            
                            delta_time = np.abs(epochs[i]-epochs[j]).astype('timedelta64[D]').astype(float)/365.25
                            delta_time_list.append(delta_time)
                            
                            ep_list.append([epochs[i],epochs[j]])
                            freq_list.append(freq_min)
                            
                            used_indices.append([i,j])
                
                delta_flux_A_list = np.array(delta_flux_A_list)
                min_flux_A_list = np.array(min_flux_A_list)
                delta_flux_B_list = np.array(delta_flux_B_list)
                min_flux_B_list = np.array(min_flux_B_list)
                delta_flux_tot_list = np.array(delta_flux_tot_list)
                min_flux_tot_list = np.array(min_flux_tot_list)
                delta_time_list = np.array(delta_time_list)
                    
                var_A = delta_flux_A_list/min_flux_A_list/delta_time_list*(1+z)
                var_B = delta_flux_B_list/min_flux_B_list/delta_time_list*(1+z)
                var_tot = delta_flux_tot_list/min_flux_tot_list/delta_time_list*(1+z)
                header_list = []
                
                for k in range(len(var_A)):
                    print(f'Total flux density maximum variability: {np.nanmax(var_tot[k])*100:.1f}'
                        f' %/yr between {ep_list[k][0]:s} and {ep_list[k][1]:s}'
                        f' at {freq_list[k]/1e9:.1f} GHz')
                    if var_tot[k] > var_thresh:
                        print(f'! Total flux density varies by more than {var_thresh*100:.0f}'
                            f' %/yr between {ep_list[k][0]:s} and {ep_list[k][1]:s}'
                            f' at {freq_list[k]/1e9:.1f} GHz: {var_tot[k]*100:.1f} %/yr')
                    print(f'Component A flux density maximum variability: {np.nanmax(var_A[k])*100:.1f}'
                        f' %/yr between {ep_list[k][0]:s} and {ep_list[k][1]:s}'
                        f' at {freq_list[k]/1e9:.1f} GHz')
                    if var_A[k] > var_thresh:
                        print(f'! Component A flux density varies by more than {var_thresh*100:.0f}'
                            f' %/yr between {ep_list[k][0]:s} and {ep_list[k][1]:s}'
                            f' at {freq_list[k]/1e9:.1f} GHz: {var_A[k]*100:.1f} %/yr')
                    print(f'Component B flux density maximum variability: {np.nanmax(var_B[k])*100:.1f}'
                        f' %/yr between {ep_list[k][0]:s} and {ep_list[k][1]:s}'
                        f' at {freq_list[k]/1e9:.1f} GHz')
                    if var_B[k] > var_thresh:
                        print(f'! Component B flux density varies by more than {var_thresh*100:.0f}'
                            f' %/yr between {ep_list[k][0]:s} and {ep_list[k][1]:s}'
                            f' at {freq_list[k]/1e9:.1f} GHz: {var_B[k]*100:.1f} %/yr')
                    header_list.append(str(ep_list[k][0])+' -> '+str(ep_list[k][1])+f' at {freq_min/1e9:.1f} GHz')
                
                rows = [np.round(var_tot,3), np.round(var_A,3), np.round(var_B,3)]
                row_names = [
                    'Total flux variability [/yr]',
                    'Component A variability [/yr]',
                    'Component B variability [/yr]'
                    ]

                df_out = pd.DataFrame(np.vstack(rows), columns=header_list, index=row_names)
                outfile = outfile_path + f'variability_{freq_min/1e9:.1f}GHz.csv'
                df_out.to_csv(outfile, index=True)
                print(f"Saved variability results to {outfile}")
    
    
    
    def export_latex(self, outfile_path=''):
        if self.df.empty:
            print('Error: cannot export .tex file because dataframe is empty. Run calc_all first.')
        else:
            print('### Saving LaTeX file ###')
            df = helpers.restructure_df(self.df)
            
            if outfile_path == '':
                outfile_path = './' + self.source_name + '/'
            elif type(outfile_path) == str and outfile_path[-1] != '/':
                outfile_path = outfile_path + '/'
            os.makedirs(outfile_path, exist_ok=True)
            outfile_path = outfile_path + 'lens_params.tex'
            
            #TO-DO: add custom format to change text color to red for values that
            #       violate lensing criteria
            
            df["SBR"] = df.apply(
                lambda row: (
                    f"$<${round(row['SBR'],2)}" if row["SBR is limit"] == "upper"
                    else f"$>${row['SBR']}" if row["SBR is limit"] == "lower"
                    else round(row['SBR'],2)
                    ),
                axis=1,
                )
            df["log(T_b,obs)"] = df.apply(
                lambda row: (
                    f"$<$ {round(row['log(T_b,obs)'],2)}" if row["T_b is limit"] == "upper"
                    else f"$>$ {row['log(T_b,obs)']}" if row["T_b is limit"] == "lower"
                    else round(row['log(T_b,obs)'],2)
                    ),
                axis=1,
                )
            
            if 'FR OK?' in df.columns:
                df = df.drop(columns=["FR OK?"])
            if 'SBR OK?' in df.columns:
                df = df.drop(columns=["SBR OK?"])
            if 'Sep. OK?' in df.columns:
                df = df.drop(columns=["Sep. OK?"])
            if 'alpha OK?' in df.columns:
                df = df.drop(columns=["alpha OK?"])
            
            # Drop unnecessary columns
            df = df.drop(columns=["T_b is limit"])
            df = df.drop(columns=["SBR is limit"])
            df = df.drop(columns=["SBR_max"])
            df = df.drop(columns=["SBR_max err"])
            df = df.drop(columns=["FR_max"])
            df = df.drop(columns=["FR_max err"])
            df = df.drop(columns=["log(T_b,obs,max)"])
            df = df.drop(columns=["log(T_b,obs,max) err"])
            
            # Adjust units for LaTeX printing
            for i, freq in enumerate(df["Freq [Hz]"]):
                 if type(freq) == float:
                    #  df["Freq [Hz]"][i] = freq/1E9
                     df.loc[i, "Freq [Hz]"] = freq/1E9
            for i, flux in enumerate(df["Flux [Jy]"]):
                 if type(flux) == float:
                    #  df["Flux [Jy]"][i] = df["Flux [Jy]"][i]*1E3
                    #  df["Flux err [Jy]"][i] = df["Flux err [Jy]"][i]*1E3
                     df.loc[i, "Flux [Jy]"] = df.loc[i, "Flux [Jy]"]*1E3
                     df.loc[i, "Flux err [Jy]"] = df.loc[i, "Flux err [Jy]"]*1E3
            # Adjust column names to LaTeX math format
            df = df.rename(columns={"Freq [Hz]": r"$\nu$ [GHz]"})
            df = df.rename(columns={"Flux [Jy]": r"$S_\nu$ [mJy]"})
            df = df.rename(columns={"Flux err [Jy]": r"$\sigma_{S_\nu}$ [mJy]"})
            df = df.rename(columns={"FR err": r"$\sigma_\mathrm{FR}$"})
            # df = df.rename(columns={"FR_max": r"$\mathrm{FR}_\mathrm{max}$"})
            # df = df.rename(columns={"FR_max err": r"$\sigma_\mathrm{FR_\mathrm{max}}$"})
            df = df.rename(columns={"log(T_b,obs)": r"$\log(T_\mathrm{b,obs})$"})
            df = df.rename(columns={"log(T_b,obs) err": r"$\sigma_{\log(T_\mathrm{b,obs})}$"})
            # df = df.rename(columns={"log(T_b,obs,max)": r"$\log(T_\mathrm{b,obs,max})$"})
            # df = df.rename(columns={"log(T_b,obs,max) err": r"$\sigma_{\log(T_\mathrm{b,obs,max})}$"})
            df = df.rename(columns={"SBR err": r"$\sigma_\mathrm{SBR}$"})
            # df = df.rename(columns={"SBR_max": r"$\mathrm{SBR}_\mathrm{max}$"})
            # df = df.rename(columns={"SBR_max err": r"$\sigma_{\mathrm{SBR}_\mathrm{max}}$"})
            df = df.rename(columns={"Sep. [mas]": r"${\Delta}r$ [mas]"})
            df = df.rename(columns={"Sep. err [mas]": r"$\sigma_{{\Delta}r}$ [mas]"})
            df = df.rename(columns={"alpha": r"$\alpha$"})
            df = df.rename(columns={"alpha err": r"$\sigma_\alpha$"})
            # Save to LaTeX
            formatters = {
                r"$\nu$ [GHz]": lambda x: f"{x:s}" if isinstance(x, str) else f"{x:.1f}",
                r"$S_\nu$ [mJy]": lambda x: f"{x:s}" if isinstance(x, str) else f"{x:.1f}",
                r"$\sigma_{S_\nu}$ [mJy]": lambda x: f"{x:s}" if isinstance(x, str) else f"{x:.1f}",
                "FR": lambda x: f"{x:s}" if isinstance(x, str) else f"{x:.2f}",
                r"$\sigma_\mathrm{FR}$": lambda x: f"{x:s}" if isinstance(x, str) else f"{x:.2f}",
                # r"$\mathrm{FR}_\mathrm{max}$": lambda x: f"{x:s}" if isinstance(x, str) else f"{x:.2f}",
                # r"$\sigma_\mathrm{FR_\mathrm{max}}$": lambda x: f"{x:s}" if isinstance(x, str) else f"{x:.2f}",
                r"$\log(T_\mathrm{b,obs})$": lambda x: helpers.limit_formatter(x),
                r"$\sigma_{\log(T_\mathrm{b,obs})}$": lambda x: helpers.limit_formatter(x),
                # r"$\log(T_\mathrm{b,obs,max})$": lambda x: helpers.limit_formatter(x),
                # r"$\sigma_{\log(T_\mathrm{b,obs,max})}$": lambda x: helpers.limit_formatter(x),
                "SBR": lambda x: helpers.limit_formatter(x),
                r"$\sigma_\mathrm{SBR}$": lambda x: helpers.limit_formatter(x),
                # r"$\mathrm{SBR}_\mathrm{max}$": lambda x: helpers.limit_formatter(x),
                # r"$\sigma_{\mathrm{SBR}_\mathrm{max}}$": lambda x: helpers.limit_formatter(x),
                r"${\Delta}r$ [mas]": lambda x: f"{x:s}" if isinstance(x, str) else f"{x:.2f}",
                r"$\sigma_{{\Delta}r}$ [mas]": lambda x: f"{x:s}" if isinstance(x, str) else f"{x:.2f}",
                r"$\alpha$": lambda x: f"{x:s}" if isinstance(x, str) else f"{x:.2f}",
                r"$\sigma_\alpha$": lambda x: f"{x:s}" if isinstance(x, str) else f"{x:.2f}",
                }
            col_format = "c" * len(df.columns)

            # for older pandas versions, now replaced with latex_table below
              # using df.style.to_latex 
            # latex_table = df.to_latex(
            #     index=False,
            #     escape=False,
            #     formatters=formatters,
            #     column_format=col_format
            #     ) 
            # latex_table = latex_table.replace("NaN", "-")

            latex_table = (
                df.style
                # .format(formatters, escape="latex", na_rep="-")
                .format(formatters, escape=None, na_rep="-")
                .hide(axis="index")
                .to_latex(column_format=col_format, hrules=True)
                )

            with open(outfile_path, "w") as f:
                f.write(
                    r"\documentclass{article}" + "\n" +
                    r"\usepackage[letterpaper,top=2cm,bottom=2cm,left=2.5cm,right=2.5cm,marginparwidth=1.5cm]{geometry}" + "\n" +
                    r"\usepackage{booktabs}" + "\n" +
                    r"\usepackage{adjustbox}" + "\n" +
                    r"\begin{document}" + "\n\n" +
                    r'\begin{table}[ht]' + "\n" +
                    r'\caption{Source}' + "\n" +
                    r'\vspace*{3mm}' + "\n" +
                    r'\adjustbox{width=1\textwidth}{%' + "\n" +
                    r'\label{tab:source}' + "\n" +
                    r'\centering''' + "\n" +
                    latex_table + "\n\n" +
                    r"}" + "\n" + r"\end{table}" + "\n\n" +
                    r"\end{document}"
                    )
            print(f'Saved file to {outfile_path}')
    
    
    
    def export_csv(self, outfile_path=''):
    
        df_new = helpers.restructure_df(self.df)
        
        df_new.to_csv(outfile_path, index=False)
        print(f"Saved results to {outfile_path}")
    
    
    
    def analyze(self):
        self.calc_all()
        self.plot_images_all()
        self.plot_all()
        self.evaluate_FR()
        self.evaluate_SBR()
        self.export_latex()
    
    
    
    def stack_by_freq(self):
        print('COMING SOON!')
    
    
    
    def overplot_freq_epoch(self):
        print('COMING SOON!')


