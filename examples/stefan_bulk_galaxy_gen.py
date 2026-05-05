import matplotlib
matplotlib.use('Agg')  # No display needed
import numpy as np
import os
import socca
from astropy.io import fits

# ── Parameter grid ──────────────────────────────────────────────────────────
ROTS   = [0, 90]                         # degrees
RDISKS = {"half": 0.5, "full": 1.0}      # as fractions of rdisk
INCS   = [0]                             # degrees
IES    = [0.25, 0.5, 1.0, 2.0, 3.0]      # Fractions of Ie disk

# Base parameters
RDISK_BASE = 1e-3
img_path   = "data/tutorial/tutorial_mock_256x256.fits"
base_dir   = "C:/Users/Stefan/Desktop/master thesis/Local_galaxy_images/256x256_different_contrast"

# ── Generation loop ──────────────────────────────────────────────────────────
noise = socca.noise.Normal()
img   = socca.data.Image(img=img_path, noise=noise)

for rot_deg in ROTS:
    for rdisk_label, rdisk_frac in RDISKS.items():
        for inc_deg in INCS:
            for Ie_frac in IES:
                socca.models.Component.idcls = 0

                out_dir  = os.path.join(base_dir, f"rot_{rot_deg}", f"rdisk_{rdisk_label}", f"inc_{inc_deg}", f"Ie_{Ie_frac}")
                out_file = os.path.join(out_dir, "galaxy.fits")
                os.makedirs(out_dir, exist_ok=True)

                print(f"Generating rot={rot_deg} | rdisk={rdisk_label} | inc={inc_deg} | Ie={Ie_frac}...", end=" ")

                # ── Disk ────────────────────────────────────────────────────────
                disk_radial       = socca.models.Sersic()
                disk_radial.Ie    = 1e+1
                disk_radial.re    = RDISK_BASE
                disk_radial.ns    = 1.0
                disk_radial.xc    = img.hdu.header['CRVAL1']
                disk_radial.yc    = img.hdu.header['CRVAL2']
                disk_radial.theta = np.radians(0)          # disk always N-S

                disk_vertical     = socca.models.disk.vertical.HyperSecantHeight()
                disk_vertical.zs  = disk_radial.re / 10
                disk_vertical.inc = np.radians(inc_deg)

                disk = socca.models.Disk(radial=disk_radial, vertical=disk_vertical)

                # ── Bar ─────────────────────────────────────────────────────────
                bar_re            = RDISK_BASE * rdisk_frac

                bar_radial        = socca.models.Sersic()
                bar_radial.Ie     = disk_radial.Ie * Ie_frac
                bar_radial.re     = bar_re
                bar_radial.ns     = 0.25

                bar_geom          = socca.models.BarGeometry()
                bar_geom.xc       = socca.priors.boundto(disk, "xc")
                bar_geom.yc       = socca.priors.boundto(disk, "yc")
                bar_geom.e        = 0.7
                bar_geom.inc      = socca.priors.boundto(disk, "inc")
                bar_geom.theta    = socca.priors.boundto(disk, "theta")
                bar_geom.rot      = np.radians(rot_deg)

                bar = socca.models.Bar(radial=bar_radial, geometry=bar_geom)

                # ── Model → FITS ─────────────────────────────────────────────────
                mod = socca.models.Model()
                mod.addcomponent(disk)
                mod.addcomponent(bar)

                mraw = mod.getmap(img, convolve=False)

                hdu = fits.PrimaryHDU(mraw, header=img.hdu.header.copy())
                hdu.writeto(out_file, overwrite=True)

                print("done.")

print("\nAll galaxy.fits files generated.")