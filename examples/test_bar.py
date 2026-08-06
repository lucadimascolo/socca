import matplotlib.pyplot as plt
import socca

import time

import numpy as np

noise = socca.noise.Normal()

img = "data/tutorial/tutorial_mock.fits"
img = socca.data.Image(img=img, noise=noise)

## Disk component

disk_radial = socca.models.Sersic()
disk_radial.Ie = 1e+1
disk_radial.re = 1e-3
disk_radial.ns = 1.0
disk_radial.xc = img.hdu.header['CRVAL1']
disk_radial.yc = img.hdu.header['CRVAL2']
disk_radial.theta = np.radians(60)  # Position angle Disk

disk_vertical = socca.models.disk.vertical.HyperSecantHeight()
disk_vertical.zs = disk_radial.re/10
disk_vertical.inc = np.radians(60)  # Inclination Disk
# disk_vertical.losdepth = disk_radial.re * 3

disk = socca.models.Disk(radial=disk_radial, vertical=disk_vertical)

## Bar component

bar_radial = socca.models.Sersic()
bar_radial.Ie = 1e+1
bar_radial.re = 2e-4
bar_radial.ns = 0.25

bar_radial.xc = socca.priors.boundto(disk, "xc")
bar_radial.yc = socca.priors.boundto(disk, "yc")
bar_radial.e = 0.7
bar_radial.theta = socca.priors.boundto(disk, "theta")

bar = socca.models.Bar(radial=bar_radial)
bar.inc = socca.priors.boundto(disk, "inc")
bar.rot = np.radians(30)  # Additional Position Angle Bar
# bar.losdepth = disk_vertical.losdepth * 2

print(bar_radial.re)
print(disk_radial.re)

mod = socca.models.Model()
mod.addcomponent(disk)
mod.addcomponent(bar)

mraw = mod.getmap(img, convolve=False)

plt.figure()
plt.imshow(mraw, origin='lower', cmap='inferno', vmin=0, vmax=np.max(mraw))
plt.colorbar()
plt.show()