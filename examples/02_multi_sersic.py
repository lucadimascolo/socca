import matplotlib.pyplot as plt
import socca

import corner

import numpy as np
import os

from astropy.coordinates import SkyCoord
from astropy.convolution import Gaussian2DKernel

noise = socca.noise.Normal()

img = 'input/exp_convoluted.fits'
img = socca.data.Image(img=img,noise=noise)

x_stddev = img.hdu.header['BMAJ']/img.hdu.header['CDELT2']/np.sqrt(8.00*np.log(2.00))
x,y = np.meshgrid(np.arange(img.data.shape[0]), np.arange(img.data.shape[0]))
kernel = Gaussian2DKernel(x_stddev=x_stddev)
kernel = kernel.array

img.addpsf(img=kernel)

mod = socca.models.Model()

# Bar component
prof0 = socca.models.Sersic()
prof0.parlist()
prof0.xc    = socca.priors.uniform(low = 334.38500,high = 334.38520)
prof0.yc    = socca.priors.uniform(low =   0.29535,high =   0.29560)
prof0.re    = socca.priors.loguniform(low = 1.00E-07, high = 1.00E-02)
prof0.Ie    = socca.priors.loguniform(low = 1.00E-04, high = 1.00E+01)
prof0.ns    = socca.priors.uniform(low =  0.25, high = 1.00)
prof0.theta = socca.priors.uniform(low = -0.25*np.pi, high = 0.25*np.pi)
prof0.e     = socca.priors.uniform(low =  0.00, high = 0.95)
prof0.cbox  = 0.00
prof0.positive = True

mod.addcomponent(prof0)

# Disk component
prof1 = socca.models.Sersic()
prof1.xc    = socca.priors.boundto(prof0,'xc')
prof1.yc    = socca.priors.boundto(prof0,'yc')
prof1.re    = socca.priors.loguniform(low = 1.00E-07, high = 1.00E-02)
prof1.Ie    = socca.priors.loguniform(low = 1.00E-04, high = 1.00E+01)
prof1.ns    = 1.00
prof1.theta = socca.priors.uniform(low = -0.25*np.pi, high = 0.25*np.pi)
prof1.e     = socca.priors.uniform(low =  0.00, high = 0.95)
prof1.cbox  = 0.00
prof1.positive = True

mod.addcomponent(prof1)

# Bulge component
prof2 = socca.models.Sersic()
prof2.xc    = socca.priors.boundto(prof0,'xc')
prof2.yc    = socca.priors.boundto(prof0,'yc')
prof2.re    = lambda src_01_re, src_02_re_scale: src_01_re*src_02_re_scale
prof2.addpar('re_scale',socca.priors.uniform(low=0.00,high=1.00))
prof2.Ie    = socca.priors.loguniform(low = 1.00E-04, high = 1.00E+01)
prof2.ns    = socca.priors.uniform(low =  1.00, high = 5.00)
prof2.theta = socca.priors.uniform(low = -0.25*np.pi, high = 0.25*np.pi)
prof2.e     = socca.priors.uniform(low =  0.00, high = 0.95)
prof2.cbox  = 0.00
prof2.positive = True

mod.addcomponent(prof2)

fit = socca.fitter(mod=mod,img=img)
fit.run(method='nautilus',dlogz=0.10,nlive=400,n_like_max=10000)
_, msmo, _ = fit.getmodel()

# ------------------------------------------------------------------------------------

plt.subplot(131); plt.imshow(img.data)
plt.subplot(132); plt.imshow(msmo)
plt.subplot(133); plt.imshow(img.data-msmo)
plt.show(); plt.close()

# ------------------------------------------------------------------------------------

if fit.method!='optimizer':
    for key in ['re','Ie']:
        idx = np.where(np.array(fit.labels)==f'src_00_{key}')[0][0]
        fit.samples[:,idx] = np.log10(fit.samples[:,idx])

    sigma = 10.00
    if sigma is None:
        edges = None
    else:
        edges = np.array([corner.quantile(s,[0.16,0.50,0.84],weights=fit.weights) for s in fit.samples.T])
        edges = np.array([[np.maximum(fit.samples[:,ei].min(),e[1]-sigma*(e[1]-e[0])),
                        np.minimum(fit.samples[:,ei].max(),e[1]+sigma*(e[2]-e[1]))] for ei, e in enumerate(edges)])
        
    corner.corner(fit.samples,weights=fit.weights,labels=fit.labels,range=edges)
    plt.savefig('test_corner.pdf',format='pdf',dpi=300); plt.close()
