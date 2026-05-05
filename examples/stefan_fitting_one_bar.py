import matplotlib.pyplot as plt
import socca
from astropy.io import fits

import corner

import numpy as np

from astropy.convolution import Gaussian2DKernel

import jax
import os
os.environ["JAX_PLATFORMS"] = "cpu" # or "cuda"


noise = socca.noise.Normal(sigma=0.015656*0.05)


img = fits.open("data/examples/stefan_rot_0_rdisk_half_inc_20_psf_small_noise_mid.fits")[0]
img.data = img.data.astype('float32')
img = socca.data.Image(img=img, noise=noise)

x_stddev = (
    img.hdu.header["PSF_SIG"]
    / img.hdu.header["CDELT2"]
    # / np.sqrt(8.00 * np.log(2.00))
)
x, y = np.meshgrid(np.arange(img.data.shape[0]), np.arange(img.data.shape[0]))
kernel = Gaussian2DKernel(x_stddev=x_stddev)
kernel = kernel.array

img.addpsf(img=kernel)

# Disk Component
disk_radial = socca.models.Sersic()
disk_radial.parlist()

radius = 3.00E-04  # degrees ~1arcsec.
disk_radial.xc = socca.priors.uniform(low=img.hdu.header['CRVAL1'] - radius,
                                      high=img.hdu.header['CRVAL1'] + radius)
disk_radial.yc = socca.priors.uniform(low=img.hdu.header['CRVAL2'] - radius,
                                      high=img.hdu.header['CRVAL2'] + radius)
disk_radial.Ie = socca.priors.loguniform(low=1.00e-04, high=1.00e02)
disk_radial.re = socca.priors.loguniform(low=1.00e-07, high=1.00e-02)
disk_radial.ns = 1.0  # Disk is exponential
disk_radial.theta = socca.priors.uniform(low=-0.5*np.pi, high=0.5*np.pi)

disk_vertical = socca.models.disk.vertical.HyperSecantHeight()
disk_vertical.zs = socca.priors.loguniform(low=1.00e-07, high=1.00e-03)  # 10% of radius
disk_vertical.inc = socca.priors.uniform(low=0., high=0.5*np.pi)  # 0-90 deg.

disk = socca.models.Disk(radial=disk_radial, vertical=disk_vertical)
disk.positive = True

# Bar component
bar_radial = socca.models.Sersic()
bar_radial.re = socca.priors.loguniform(low=1.00e-07, high=1.00e-02)
bar_radial.Ie = socca.priors.loguniform(low=1.00e-04, high=1.00e02)
bar_radial.ns = socca.priors.uniform(low=0.25, high=1.00)

bar_geom = socca.models.BarGeometry()
bar_geom.xc = socca.priors.boundto(disk, "xc")
bar_geom.yc = socca.priors.boundto(disk, "yc")
bar_geom.e = socca.priors.uniform(low=0.00, high=0.95)
bar_geom.inc = socca.priors.boundto(disk, "inc")
bar_geom.theta = socca.priors.boundto(disk, "theta")
bar_geom.rot = socca.priors.uniform(low=0., high=np.pi)  # additional rotation in disk frame.

bar = socca.models.Bar(radial=bar_radial, geometry=bar_geom)

mod = socca.models.Model()
mod.addcomponent(disk)
mod.addcomponent(bar)

fit = socca.fitter(mod=mod, img=img)
fit.run(method="pocomc", dlogz=0.10, vectorize=True, nlive=4, checkpoint='stefan_pocomc_checkpoint')
fit.dump('stefan_pocomc_fit_results.pickle')
_, msmo, _ = fit.getmodel()

# ----------------------------------------------------------------------

plt.subplot(131)
plt.imshow(img.data)
plt.subplot(132)
plt.imshow(msmo)
plt.subplot(133)
plt.imshow(img.data - msmo)
plt.show()
plt.close()

# ----------------------------------------------------------------------

if fit.method != "optimizer":
    for key in ["re", "Ie"]:
        idx = np.where(np.array(fit.labels) == f"src_00_{key}")[0][0]
        fit.samples[:, idx] = np.log10(fit.samples[:, idx])

    sigma = 10.00
    if sigma is None:
        edges = None
    else:
        edges = np.array(
            [
                corner.quantile(s, [0.16, 0.50, 0.84], weights=fit.weights)
                for s in fit.samples.T
            ]
        )
        edges = np.array(
            [
                [
                    np.maximum(
                        fit.samples[:, ei].min(), e[1] - sigma * (e[1] - e[0])
                    ),
                    np.minimum(
                        fit.samples[:, ei].max(), e[1] + sigma * (e[2] - e[1])
                    ),
                ]
                for ei, e in enumerate(edges)
            ]
        )

    corner.corner(
        fit.samples, weights=fit.weights, labels=fit.labels, range=edges
    )
    plt.savefig("test_corner.pdf", format="pdf", dpi=300)
    plt.close()
