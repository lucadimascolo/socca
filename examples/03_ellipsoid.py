import matplotlib.pyplot as plt
import socca

import corner

import numpy as np

from astropy.convolution import Gaussian2DKernel

noise = socca.noise.Normal()

img = "data/examples/exp_convoluted.fits"
img = socca.data.Image(img=img, noise=noise)

x_stddev = (
    img.hdu.header["BMAJ"]
    / img.hdu.header["CDELT2"]
    / np.sqrt(8.00 * np.log(2.00))
)
x, y = np.meshgrid(np.arange(img.data.shape[0]), np.arange(img.data.shape[0]))
kernel = Gaussian2DKernel(x_stddev=x_stddev)
kernel = kernel.array

img.addpsf(img=kernel)

mod = socca.models.Model()

# Disk component
disk_radial = socca.models.Sersic()
disk_radial.xc = socca.priors.uniform(low=334.38500, high=334.38520)
disk_radial.yc = socca.priors.uniform(low=0.29535, high=0.29560)
disk_radial.re = socca.priors.loguniform(low=1.00e-04, high=1.00e-02)
disk_radial.Ie = socca.priors.loguniform(low=1.00e-04, high=1.00e01)
disk_radial.ns = 1.00
disk_radial.theta = socca.priors.uniform(low=-0.50 * np.pi, high=0.50 * np.pi)

disk_vertical = socca.models.disk.vertical.HyperSecantHeight()
disk_vertical.zs = socca.priors.loguniform(low=1.00e-05, high=1.00e-03)
disk_vertical.inc = socca.priors.uniform(low=0.00, high=0.50 * np.pi)

disk = socca.models.Disk(radial=disk_radial, vertical=disk_vertical)
disk.positive = True

mod.addcomponent(disk)

# Bar component: tied to the disk's centroid, position angle, and
# inclination, with its own ellipticity, shape, and intrinsic rotation.
bar_radial = socca.models.Sersic()
bar_radial.re = socca.priors.loguniform(low=1.00e-05, high=1.00e-03)
bar_radial.Ie = socca.priors.loguniform(low=1.00e-04, high=1.00e01)
bar_radial.ns = socca.priors.uniform(low=0.25, high=1.00)

bar = socca.models.Bar(radial=bar_radial)
bar.xc = socca.priors.boundto(disk, "xc")
bar.yc = socca.priors.boundto(disk, "yc")
bar.theta = socca.priors.boundto(disk, "theta")
bar.e = socca.priors.uniform(low=0.00, high=0.95)
bar.inc = socca.priors.boundto(disk, "inc")
bar.rot = socca.priors.uniform(low=-0.50 * np.pi, high=0.50 * np.pi)
bar.positive = True

mod.addcomponent(bar)

fit = socca.fitter(mod=mod, img=img)
fit.run(method="nautilus", dlogz=0.10, nlive=400, n_like_max=10000)
_, msmo, _ = fit.getmodel()

# -------------------------------------------------------------------------

plt.subplot(131)
plt.imshow(img.data)
plt.subplot(132)
plt.imshow(msmo)
plt.subplot(133)
plt.imshow(img.data - msmo)
plt.show()
plt.close()

# -------------------------------------------------------------------------

if fit.method != "optimizer":
    for key in [
        "comp_00_radial.re",
        "comp_00_radial.Ie",
        "comp_01_radial.re",
        "comp_01_radial.Ie",
    ]:
        idx = np.where(np.array(fit.labels) == key)[0]
        if len(idx):
            fit.samples[:, idx[0]] = np.log10(fit.samples[:, idx[0]])

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
    plt.savefig("bar_corner.pdf", format="pdf", dpi=300)
    plt.close()
