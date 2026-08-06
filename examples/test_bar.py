import matplotlib.pyplot as plt
import socca


import numpy as np

noise = socca.noise.Normal()

img = "data/tutorial/tutorial_mock.fits"
img = socca.data.Image(img=img, noise=noise)

## Disk component

disk_radial = socca.models.Sersic()
disk_radial.Ie = 1e+1
disk_radial.re = 1.5e-2
disk_radial.ns = 1.0
disk_radial.xc = img.hdu.header['CRVAL1']
disk_radial.yc = img.hdu.header['CRVAL2']
disk_radial.theta = np.pi/3  # Position angle Disk
disk_radial.e = 0.5

disk_vertical = socca.models.disk.vertical.HyperSecantHeight()
disk_vertical.zs = 3e-4
disk_vertical.inc = np.pi/2  # Inclination Disk

disk = socca.models.Disk(radial=disk_radial, vertical=disk_vertical)

## Bar component

bar_radial = socca.models.Sersic()
bar_radial.Ie = 50e+1
bar_radial.re = 1.5e-2
bar_radial.ns = 0.5

bar_geom = socca.models.BarGeometry()
bar_geom.xc = img.hdu.header['CRVAL1']
bar_geom.yc = img.hdu.header['CRVAL2']
bar_geom.e = 0.8
bar_geom.inc = socca.priors.boundto(disk, "inc")
# bar_geom.inc = np.pi/2
bar_geom.theta = socca.priors.boundto(disk, "theta")
# bar_geom.theta = np.pi/3
bar_geom.rot = np.pi/2  # Additional Position Angle Bar

bar = socca.models.Bar(radial=bar_radial, geometry=bar_geom)

mod = socca.models.Model()
mod.addcomponent(disk)
mod.addcomponent(bar)

mraw = mod.getmap(img, convolve=False)

plt.figure()
plt.imshow(mraw, origin='lower', cmap='inferno')
plt.colorbar()
plt.show()