import matplotlib.pyplot as plt
import socca

import corner

import numpy as np
import os

from astropy.coordinates import SkyCoord
from astropy.convolution import Gaussian2DKernel

# Load the image to analyze
# `noise` controls the noise model; you can pass it in terms of rms, var, or weights.
# If you set it equal to None, it will compute a MAD-based rms from the input image.

img = 'input/exp_convoluted.fits'
img = socca.data.Image(img=img,noise=dict(std=2.00E-02))

# Here I am creating a dummy PSF. You can swap this with the dirty PDF
x,y = np.meshgrid(np.arange(img.data.shape[0]), np.arange(img.data.shape[1]))
kernel = Gaussian2DKernel(x_stddev=img.hdu.header['BMAJ']/(img.hdu.header['CDELT2']*2.355))
kernel = kernel.array

img.addpsf(img=kernel) # instead of an array, you can directly pass the HDU or the filename of the PSF image

mod = socca.models.Model()

prof0 = socca.models.Sersic()
prof0.xc    =    socca.priors.uniform(low =   334.38500, high =  334.38520)
prof0.yc    =    socca.priors.uniform(low =     0.29535, high =    0.29560)
prof0.re    = socca.priors.loguniform(low =    1.00E-07, high =   1.00E-02)
prof0.Ie    = socca.priors.loguniform(low =    1.00E-04, high =   1.00E+01)
prof0.ns    =    socca.priors.uniform(low =        0.25, high =   1.00E+01)
prof0.theta =    socca.priors.uniform(low = -0.25*np.pi, high = 0.25*np.pi)
prof0.e     =    socca.priors.uniform(low =        0.00, high =       0.95)
prof0.cbox  = 0.00
prof0.positive = True

mod.addcomponent(prof0)

# You can add any other components after building it simply as by calling mod.addcomponent() on it.
# There is also the extra option of linking parameters. I can tell you more about this once we have a call.

# this set of commands run nautilus sampler on your model
fit = socca.fitter(mod=mod,img=img)
fit.run(method='nautilus',dlogz=0.10,nlive=400,checkpoint='test.h5',resume=True,n_like_max=50000)

print(fit.logz,fit.refz,np.sqrt(2.00*(fit.logz-fit.refz)))
#corner.corner(fit.samples,labels=fit.labels,weights=fit.weights)
#plt.show(); plt.close()

## fit.dump('test.pickle')
## 
## # mraw – raw model
## # msmo - PSF-smoothed model
mraw, msmo, _ = fit.getmodel()

plt.subplot(141); plt.imshow(msmo)
plt.subplot(142); plt.imshow(img.data)
plt.subplot(143); plt.imshow(img.data-msmo)
plt.subplot(144); plt.imshow(img.psf_fft)
plt.show(); plt.close()
