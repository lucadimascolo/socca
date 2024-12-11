from .utils import *
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D

import pyregion
import reproject

import matplotlib.pyplot as plt

# Support functions
# ========================================================
# Get mask from input HDU
# --------------------------------------------------------
def _hdu_mask(mask,hdu):
    data, _ = reproject.reproject_interp(mask,hdu.header)
    data[np.isnan(data)] = 0.00
    return np.where(data<1.00,0.00,1.00)

# Load image
# --------------------------------------------------------
def _img_loader(img,idx=0):
    if   isinstance(img,(fits.ImageHDU,fits.PrimaryHDU)):
        return img
    elif isinstance(img,fits.hdu.hdulist.HDUList):
        return img[idx]
    elif isinstance(img,str):
        img = fits.open(img)
        return img[idx]
    else:
        raise ValueError('img must be an ImageHDU or a string')

def _reduce_axes(hdu):
    head = hdu.header.copy()
    
    data = hdu.data.copy()
    data = data.squeeze()
    
    head['NAXIS'] = 2
    for idx in [3,4]:
        for key in ['NAXIS','CRPIX','CRVAL','CDELT','CUNIT','CTYPE']:
            head.pop(f'{key}{idx}',None)
        
        for jdx in range(1,5):
            for key in ['CD','PC']:
                head.pop(f'{key}{jdx}_{idx}',None)
                head.pop(f'{key}{idx}_{jdx}',None)
    
    return fits.PrimaryHDU(data=data,header=head)

# Coordinate grids
# --------------------------------------------------------
class WCSgrid:
    def __init__(self,hdu,wcs=None):
        headerWCS = hdu.header.copy()

        if wcs is None: wcs = WCS(headerWCS)

        gridmx, gridmy = np.meshgrid(np.arange(headerWCS['NAXIS1']),np.arange(headerWCS['NAXIS2']))
        gridwx, gridwy = wcs.all_pix2world(gridmx,gridmy,0)
        
        if (np.abs(gridwx.max()-gridwx.min()-3.6e2)<np.abs(2.00*headerWCS['CDELT1'])): 
            gridix = np.where(gridwx>headerWCS['CRVAL1']+headerWCS['CDELT1']*(headerWCS['NAXIS1']-headerWCS['CRPIX1']+1)+3.6e2)
            gridwx[gridix] = gridwx[gridix]-3.6e2
        
        self.x = jp.array(gridwx)
        self.y = jp.array(gridwy)

class FFTspec:
    def __init__(self,hdu):
        self.pulse = jp.fft.rfft2(jp.fft.ifftshift(jp.fft.ifft2(jp.full(hdu.data.shape,1.00+0.00j))).real)
        self.freq  = [jp.array(np.broadcast_to(np.fft.rfftfreq(hdu.data.shape[1])[None,:],self.pulse.shape)),
                      jp.array(np.broadcast_to(np.fft.fftfreq(hdu.data.shape[0])[:,None],self.pulse.shape))]
        self.head  = {key: hdu.header[key] for idx in [1,2] for key in [f'CRPIX{idx}',f'CRVAL{idx}',f'CDELT{idx}',f'NAXIS{idx}']}

    def shift(self,xc,yc):
        dx = (xc-self.head['CRVAL1'])*jp.cos(jp.deg2rad(self.head['CRVAL2']))
        dy = (yc-self.head['CRVAL2'])
        uphase = -2.00j*jp.pi*self.freq[0]*(self.head['CRPIX1']-2.50+dx/jp.abs(self.head['CDELT1']))
        vphase =  2.00j*jp.pi*self.freq[1]*(self.head['CRPIX2']-1.50+dy/jp.abs(self.head['CDELT2']))
        return uphase, vphase


# Image constructor
# ========================================================
#  Initialize image structure
#  --------------------------------------------------------
class Image:
    def __init__(self,img,noise=None,**kwargs):
        self.hdu = _img_loader(img,kwargs.get('img_idx',0))
        self.hdu = _reduce_axes(self.hdu)

        self.wcs = WCS(self.hdu.header)

        if self.hdu.header['CRPIX1']!=1.00+0.50*self.hdu.header['NAXIS1'] or \
           self.hdu.header['CRPIX2']!=1.00+0.50*self.hdu.header['NAXIS2']:

            crpix1 = 1.00+0.50*self.hdu.header['NAXIS1']
            crpix2 = 1.00+0.50*self.hdu.header['NAXIS2']
            crval1, crval2 = self.wcs.all_pix2world(crpix1,crpix2,1)

            self.hdu.header['CRPIX1'] = crpix1
            self.hdu.header['CRPIX2'] = crpix2
            self.hdu.header['CRVAL1'] = float(crval1)
            self.hdu.header['CRVAL2'] = float(crval2)
            
            self.wcs = WCS(self.hdu.header)

        if 'CDELT1' not in self.hdu.header or \
           'CDELT2' not in self.hdu.header:
            self.hdu.header['CDELT1'] = -np.hypot(self.hdu.header['CD1_1'],self.hdu.header['CD2_1'])
            self.hdu.header['CDELT2'] =  np.hypot(self.hdu.header['CD2_2'],self.hdu.header['CD2_2'])

        self.data = jp.array(self.hdu.data)
        self.grid = WCSgrid(self.hdu,self.wcs)
        self.fft  = FFTspec(self.hdu)

        self.mask = jp.ones(self.data.shape,dtype=int)
        self.mask = self.mask.at[jp.isnan(self.data)].set(0)
        
        self.sigma = self.getsigma(noise)
       
        if 'center' in kwargs and 'csize' in kwargs:
            self.cutout(center=kwargs['center'],csize=kwargs['csize'])

        if 'addmask' in kwargs:
            self.addmask(regions = kwargs['addmask'].get('regions'),
                         combine = kwargs['addmask'].get('combine',True),
                            mask = kwargs['addmask'].get('mask',None))
        
        self.psf = None
        if 'addpsf' in kwargs:
            self.addpsf(img = kwargs['addpsf'].get('img'),
                  normalize = kwargs['addpsf'].get('normalize',True),
                        idx = kwargs['addpsf'].get('idx',0))

#   Build elliptical distance grid
#   --------------------------------------------------------
    def getgrid(self,xc,yc,theta=0.00,e=0.00):
        sint = jp.sin(theta)
        cost = jp.cos(theta)

        xgrid = (-(self.grid.x-xc)*jp.cos(jp.deg2rad(yc))*sint-(self.grid.y-yc)*cost)
        ygrid = ( (self.grid.x-xc)*jp.cos(jp.deg2rad(yc))*cost-(self.grid.y-yc)*sint)
        
        return jp.hypot(xgrid,ygrid/(1.00-e))
    
#   Get cutout
#   --------------------------------------------------------
    def cutout(self,center,csize,getwcs=False):
        """
        center : tuple or SkyCoord
        csize  : int, array_like, or Quantity
        getwcs : bool, optional
            If True, return the WCS object of the cutout.
        """
        cutout_data  = Cutout2D(self.data,center,csize,wcs=self.wcs)
        cutout_mask  = Cutout2D(self.mask,center,csize,wcs=self.wcs)
        cutout_sigma = Cutout2D(self.sigma,center,csize,wcs=self.wcs)

        if self.psf is not None:
            cutout_psf = Cutout2D(self.psf,center,csize,wcs=self.wcs)
            self.addpsf(cutout_psf.data,normalize=False)

        cuthdu = fits.ImageHDU(data=cutout_data.data,header=cutout_data.wcs.to_header())
        
        self.hdu  = cuthdu 
        self.wcs  = cutout_data.wcs

        self.data  = jp.array(cutout_data.data);  del cutout_data
        self.mask  = jp.array(cutout_mask.data);  del cutout_mask
        self.sigma = jp.array(cutout_sigma.data); del cutout_sigma

        self.grid = WCSgrid(self.hdu,self.wcs)
        self.fft  = FFTspec(self.hdu)

#   Add mask
#   --------------------------------------------------------
    def addmask(self,regions,mask=None,combine=True):
        """
        regions : list
            List of regions to be masked. It can be
            a mix of strings, pyregion objects, np.arrays, and HDUs.
        mask : array_like, optional
            Mask to be added to the image.
        combine : bool, optional
            If True, combine with the existing mask. 
            If False, reset the mask.
        """
        if mask is not None:
            data = _hdu_mask(mask,self.hdu)
        else:
            data = np.ones(self.data.shape)

        if combine: data = data*self.mask.copy()

        hdu = fits.PrimaryHDU(data=data,header=self.wcs.to_header())

        for ri, r in enumerate(regions):
            if isinstance(r,str):
                reg = pyregion.open(r)
                idx = reg.get_mask(hdu=hdu)
                idx = idx.astype(bool)
                hdu.data[idx] = 0.00
            elif isinstance(r,pyregion.Shape):
                idx = r.get_mask(hdu=hdu).astype(bool)
                hdu.data[idx] = 0.00
            elif isinstance(r,np.ndarray):
                hdu.data = hdu.data*(r==1.00).astype(float)
            elif isinstance(r,fits.ImageHDU,fits.PrimaryHDU):
                data = _hdu_mask(mask,self.hdu)
                hdu.data = hdu.data*data    
        
        self.mask = hdu.data.astype(int).copy()

#   Add PSF
#   --------------------------------------------------------
    def addpsf(self,img,normalize=True,idx=0):
        """
        img : array_like, HDU, or str
            PSF image to be added to the image.
        normalize : bool, optional  
            If True, normalize the PSF image.
        idx : int, optional 
            Index of the HDU to be loaded.  
        """
        if isinstance(img,(np.ndarray,jp.ndarray)):
            kernel = img.copy()
        else:
            hdu = _img_loader(img,idx)
            kernel = hdu.data.copy()

        if normalize:
            kernel = kernel/np.sum(kernel)

        self.psf = kernel

        pad_width = [(0,max(0,s-k)) for s, k in zip(self.data.shape,kernel.shape)]
        self.psf_fft = np.pad(kernel,pad_width,mode='constant')
        self.psf_fft = jp.fft.rfft2(jp.fft.fftshift(self.psf_fft),s=self.psf_fft.shape)
        self.psf_fft = jp.abs(self.psf_fft)

#   --------------------------------------------------------

    def getsigma(self,noise):
        """
        noise : float or dict
            Noise level to be used in the image.
            If a float, it is used as the noise level.
            If a dict, it must have the key 'sigma' or 'sig' for the noise level.
            If a dict, it can have the key 'var' or 'variance' for the variance.
            If a dict, it can have the key 'wht', 'wgt', 'weight', 'weights', or 'invvar' for the inverse variance.
            If None, the Median Absolute Deviation (MAD) method is used to estimate the noise level.
        """
        if noise is None:
            print('Using MAD for estimating noise level')
            sigma = scipy.stats.median_abs_deviation(self.data[self.data!=0.00],axis=None,scale='normal')
            sigma = float(sigma)
            print(f'- noise level: {sigma:.2E}')
        elif isinstance(noise,dict):
            key = list(noise.keys())[0]
            if isinstance(noise[key],float):
                sigma = noise[key]
            else:
                sigma = _img_loader(noise[key],noise.get('idx',0)).data.copy()

            if key in ['var','variance']:
                sigma = np.sqrt(sigma)
            elif key in ['wht','wgt','weight','weights','invvar']:
                sigma = 1.00/np.sqrt(sigma)
            elif key not in ['sigma','sig','std','rms','stddev']:
                raise ValueError('unrecognized noise identifier]')
        else:
            raise ValueError('noise must be a float or a dictionary')

        if isinstance(sigma,float):
            sigma = np.full(self.data.shape,sigma)

        sigma[np.isinf(sigma)] = 0.00
        sigma[np.isnan(sigma)] = 0.00
        self.mask.at[sigma==0.00].set(0)
        
        return jp.array(sigma)