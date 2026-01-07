from .utils import _img_loader, _reduce_axes
from . import noise as noisepdf

from astropy.io import fits
import jax.numpy as jp
import numpy as np

import inspect

from astropy.wcs import WCS
from astropy.nddata import Cutout2D

import pyregion
import reproject


# Support functions
# ========================================================
# Get mask from input HDU
# --------------------------------------------------------
def _hdu_mask(mask, hdu):
    data, _ = reproject.reproject_interp(mask, hdu.header)
    data[np.isnan(data)] = 0.00
    return np.where(data < 1.00, 0.00, 1.00)


# Coordinate grids
# --------------------------------------------------------
class WCSgrid:
    """
    WCS coordinate grid for image

    Parameters
    ----------
    hdu : fits.PrimaryHDU
        FITS HDU object containing image data and header.
    subgrid : int, optional
        Subgrid factor for sub-pixel sampling (default is 1).

    Attributes
    ----------
    x : jax.numpy.ndarray
        x-coordinate array in world coordinates.
    y : jax.numpy.ndarray
        y-coordinate array in world coordinates.
    """

    def __init__(self, hdu, subgrid=1):
        multis = (subgrid * subgrid, *hdu.data.shape)
        multix, multiy = np.zeros(multis), np.zeros(multis)

        header = hdu.header.copy()

        cdelt1 = np.abs(header["CDELT1"])
        cdelt2 = np.abs(header["CDELT2"])

        header["CRVAL1"] = header["CRVAL1"] - (
            0.50 - 0.50 / float(subgrid)
        ) * np.abs(header["CDELT1"])
        for isp in range(subgrid):
            header["CRVAL2"] = (
                header["CRVAL2"] - (0.50 - 0.50 / float(subgrid)) * cdelt2
            )
            for jsp in range(subgrid):
                multix[isp * subgrid + jsp], multiy[isp * subgrid + jsp] = (
                    self.getmesh(header=header)
                )
                header["CRVAL2"] = header["CRVAL2"] + cdelt2 / float(subgrid)
            header["CRVAL1"] = header["CRVAL1"] + cdelt1 / float(subgrid)
            header["CRVAL2"] = (
                header["CRVAL2"] - (0.5 + 0.5 / float(subgrid)) * cdelt2
            )

        self.x = jp.array(multix)
        self.y = jp.array(multiy)

    @staticmethod
    def getmesh(hdu=None, wcs=None, header=None):
        if (header is None) and (hdu is not None):
            headerWCS = hdu.header.copy()
        elif (header is not None) and (hdu is None):
            headerWCS = header.copy()
        else:
            raise ValueError("Either header or hdu should be defined.")

        if wcs is None:
            wcs = WCS(headerWCS)

        gridmx, gridmy = np.meshgrid(
            np.arange(headerWCS["NAXIS1"]), np.arange(headerWCS["NAXIS2"])
        )
        gridwx, gridwy = wcs.all_pix2world(gridmx, gridmy, 0)

        if np.abs(gridwx.max() - gridwx.min() - 3.6e2) < np.abs(
            2.00 * headerWCS["CDELT1"]
        ):
            gridix = np.where(
                gridwx
                > headerWCS["CRVAL1"]
                + headerWCS["CDELT1"]
                * (headerWCS["NAXIS1"] - headerWCS["CRPIX1"] + 1)
                + 3.6e2
            )
            gridwx[gridix] = gridwx[gridix] - 3.6e2

        return gridwx, gridwy


# FFT planner
# --------------------------------------------------------
class FFTspec:
    """
    FFT specification for image

    Parameters
    ----------
    hdu : fits.PrimaryHDU
        FITS HDU object containing image data and header.

    Attributes
    ----------
    pulse : jax.numpy.ndarray
        FFT of a unit pulse in image space.
    freq : list of jax.numpy.ndarray
        Frequency grids in u and v directions.
    head : dict
        Dictionary containing relevant header keywords.
    """

    def __init__(self, hdu):
        self.pulse = jp.fft.rfft2(
            jp.fft.ifftshift(
                jp.fft.ifft2(jp.full(hdu.data.shape, 1.00 + 0.00j))
            ).real
        )
        self.freq = [
            jp.array(
                np.broadcast_to(
                    np.fft.rfftfreq(hdu.data.shape[1])[None, :],
                    self.pulse.shape,
                )
            ),
            jp.array(
                np.broadcast_to(
                    np.fft.fftfreq(hdu.data.shape[0])[:, None],
                    self.pulse.shape,
                )
            ),
        ]
        self.head = {
            key: hdu.header[key]
            for idx in [1, 2]
            for key in [
                f"CRPIX{idx}",
                f"CRVAL{idx}",
                f"CDELT{idx}",
                f"NAXIS{idx}",
            ]
        }

    def shift(self, xc, yc):
        dx = (xc - self.head["CRVAL1"]) * jp.cos(
            jp.deg2rad(self.head["CRVAL2"])
        )
        dy = yc - self.head["CRVAL2"]
        uphase = (
            -2.00j
            * jp.pi
            * self.freq[0]
            * (self.head["CRPIX1"] - 1.00 + dx / jp.abs(self.head["CDELT1"]))
        )
        vphase = (
            2.00j
            * jp.pi
            * self.freq[1]
            * (self.head["CRPIX2"] - 1.00 + dy / jp.abs(self.head["CDELT2"]))
        )
        return uphase, vphase


# Image constructor
# ========================================================
# Initialize image structure
# --------------------------------------------------------
class Image:
    """
    A class for handling astronomical images with support for WCS, PSF, masking, and noise modeling.

    This class provides a comprehensive framework for working with astronomical images,
    including coordinate transformations, PSF convolution, exposure and response maps,
    region masking, and various noise models. It supports FFT-based operations and
    sub-pixel sampling through a configurable subgrid parameter.

    Parameters
    ----------
    img : str, HDU, or array_like
        Input image data. Can be a filename, FITS HDU object, or array.
    response : str, HDU, or array_like, optional
        Response map for the image. If None, uniform response is assumed.
    exposure : str, HDU, or array_like, optional
        Exposure map for the image. If None, uniform exposure is assumed.
    noise : noise object, optional
        Noise model instance (e.g., Normal, NormalALMA). Defaults to Normal().
    **kwargs : dict, optional
        Additional keyword arguments:

        - subgrid : int, optional
            Subgrid factor for sub-pixel sampling (default: 1).
        - img_idx : int, optional
            HDU index to load from the image file (default: 0).
        - exp_idx : int, optional
            HDU index to load from the exposure file (default: 0).
        - resp_idx : int, optional
            HDU index to load from the response file (default: 0).
        - center : tuple or SkyCoord, optional
            Center coordinates for cutout operation.
        - csize : int, array_like, or Quantity, optional
            Size for cutout operation.
        - addmask : dict, optional
            Dictionary with 'regions' and 'combine' keys for masking.
        - addpsf : dict, optional
            Dictionary with 'img', 'normalize', and 'idx' keys for PSF.

    Attributes
    ----------
    hdu : fits.PrimaryHDU
        FITS HDU object containing image data and header.
    wcs : astropy.wcs.WCS
        World Coordinate System object for coordinate transformations.
    data : jax.numpy.ndarray
        Image data array.
    mask : jax.numpy.ndarray
        Binary mask array (1 for valid pixels, 0 for masked).
    grid : WCSgrid
        Coordinate grid object with x and y coordinate arrays.
    fft : FFTspec
        FFT specification object for Fourier space operations.
    exp : jax.numpy.ndarray
        Exposure map array (applied after the beam convolution).
    resp : jax.numpy.ndarray
        Response map array (applied prior to the beam convolution).
    psf : ndarray or None
        Point spread function kernel.
    psf_fft : jax.numpy.ndarray or None
        FFT of the PSF kernel.
    noise : noise object
        Noise model instance.
    subgrid : int
        Subgrid sampling factor.

    Methods
    -------
    cutout(center, csize)
        Extract a cutout region from the image.
    addmask(regions, mask=None, combine=True)
        Add or update masking regions.
    addpsf(img, normalize=True, idx=0)
        Add a point spread function to the image.


    Notes
    -----
    - Missing CDELT keywords are computed from CD matrix elements.
    - NaN values in the input image are automatically masked.

    Examples
    --------
    >>> from socca.data import Image
    >>> img = Image('observation.fits', exposure='exposure.fits')
    >>> img.addpsf('psf.fits', normalize=True)
    >>> img.addmask(regions=['mask.reg'], combine=True)
    """

    def __init__(
        self, img, response=None, exposure=None, noise=None, **kwargs
    ):
        self.subgrid = kwargs.get("subgrid", 1)

        self.hdu = _img_loader(img, kwargs.get("img_idx", 0))
        self.hdu = _reduce_axes(self.hdu)

        self.wcs = WCS(self.hdu.header)

        if (
            self.hdu.header["CRPIX1"]
            != 1.00 + 0.50 * self.hdu.header["NAXIS1"]
            or self.hdu.header["CRPIX2"]
            != 1.00 + 0.50 * self.hdu.header["NAXIS2"]
        ):
            crpix1 = 1.00 + 0.50 * self.hdu.header["NAXIS1"]
            crpix2 = 1.00 + 0.50 * self.hdu.header["NAXIS2"]
            crval1, crval2 = self.wcs.all_pix2world(crpix1, crpix2, 1)

            self.hdu.header["CRPIX1"] = crpix1
            self.hdu.header["CRPIX2"] = crpix2
            self.hdu.header["CRVAL1"] = float(crval1)
            self.hdu.header["CRVAL2"] = float(crval2)

            self.wcs = WCS(self.hdu.header)

        if "CDELT1" not in self.hdu.header or "CDELT2" not in self.hdu.header:
            self.hdu.header["CDELT1"] = -np.hypot(
                self.hdu.header["CD1_1"], self.hdu.header["CD2_1"]
            )
            self.hdu.header["CDELT2"] = np.hypot(
                self.hdu.header["CD2_2"], self.hdu.header["CD2_2"]
            )

        self.data = jp.array(self.hdu.data)
        self.grid = WCSgrid(self.hdu, subgrid=self.subgrid)
        self.fft = FFTspec(self.hdu)

        self.mask = jp.ones(self.data.shape, dtype=int)
        self.mask = self.mask.at[jp.isnan(self.data)].set(0)

        if exposure is not None:
            self.exp = _img_loader(exposure, kwargs.get("exp_idx", 0))
            self.exp = _reduce_axes(self.exp)
            self.exp = self.exp.data.copy()
        else:
            self.exp = jp.ones(self.data.shape, dtype=float)

        if response is not None:
            self.resp = _img_loader(response, kwargs.get("resp_idx", 0))
            self.resp = _reduce_axes(self.resp)
            self.resp = self.resp.data.copy()
        else:
            self.resp = jp.ones(self.data.shape, dtype=float)

        if "center" in kwargs and "csize" in kwargs:
            self.cutout(center=kwargs["center"], csize=kwargs["csize"])

        if "addmask" in kwargs:
            self.addmask(
                regions=kwargs["addmask"].get("regions"),
                combine=kwargs["addmask"].get("combine", True),
            )

        self.psf = None
        if "addpsf" in kwargs:
            self.addpsf(
                img=kwargs["addpsf"].get("img"),
                normalize=kwargs["addpsf"].get("normalize", True),
                idx=kwargs["addpsf"].get("idx", 0),
            )

        if noise is None:
            self.noise = noisepdf.Normal()
        else:
            self.noise = noise

        if self.psf is None and self.noise.__class__.__name__ == "NormalRI":
            raise ValueError(
                "PSF must be defined for images with NormalRI noise."
            )

        noise_kwargs = list(inspect.signature(self.noise).parameters.keys())
        noise_kwargs = {k: eval(f"self.{k}") for k in noise_kwargs}
        self.noise(**noise_kwargs)
        self.mask = self.noise.mask.copy()

    #   Get cutout
    #   --------------------------------------------------------
    def cutout(self, center, csize):
        """
        center : tuple or SkyCoord
        csize  : int, array_like, or Quantity
        """
        cutout_data = Cutout2D(self.data, center, csize, wcs=self.wcs)
        cutout_mask = Cutout2D(self.mask, center, csize, wcs=self.wcs)
        cutout_exp = Cutout2D(self.exp, center, csize, wcs=self.wcs)
        cutout_resp = Cutout2D(self.resp, center, csize, wcs=self.wcs)

        self.data = jp.array(cutout_data.data)
        self.mask = jp.array(cutout_mask.data)
        del cutout_mask
        self.exp = jp.array(cutout_exp.data)
        del cutout_exp
        self.resp = jp.array(cutout_resp.data)
        del cutout_resp

        if self.psf is not None:
            center_psf = (self.hdu.header["CRPIX1"], self.hdu.header["CRPIX2"])
            cutout_psf = Cutout2D(self.psf, center_psf, csize, wcs=self.wcs)
            self.addpsf(cutout_psf.data, normalize=False)

        cuthdu = fits.PrimaryHDU(
            data=cutout_data.data, header=cutout_data.wcs.to_header()
        )

        crval = [center.ra.deg, center.dec.deg]
        crpix = cutout_data.wcs.all_world2pix(*crval, 1)

        cuthdu.header["CRPIX1"] = float(crpix[0])
        cuthdu.header["CRPIX2"] = float(crpix[1])
        cuthdu.header["CRVAL1"] = float(crval[0])
        cuthdu.header["CRVAL2"] = float(crval[1])

        self.hdu = cuthdu
        self.wcs = WCS(self.hdu.header)

        self.grid = WCSgrid(self.hdu, subgrid=self.subgrid)
        self.fft = FFTspec(self.hdu)

        if self.noise.__class__.__name__ == "Normal":
            self.noise(self.data, self.mask)
        else:
            raise NotImplementedError(
                "Cutout is only implemented for images with uncorrelated noise [Normal]."
            )

    #   Add mask
    #   --------------------------------------------------------
    def addmask(self, regions, combine=True):
        """
        regions : list
            List of regions to be masked. It can be
            a mix of strings, pyregion objects, np.arrays, and HDUs.
        combine : bool, optional
            If True, combine with the existing mask.
            If False, reset the mask.
        """

        mask = np.ones(self.data.shape)
        if combine:
            mask = mask * self.mask.copy()

        hdu = fits.PrimaryHDU(data=mask, header=self.wcs.to_header())

        for ri, r in enumerate(regions):
            if isinstance(r, str):
                reg = pyregion.open(r)
                idx = reg.get_mask(hdu=hdu)
                idx = idx.astype(bool)
                hdu.data[idx] = 0.00
            elif isinstance(r, pyregion.Shape):
                idx = r.get_mask(hdu=hdu).astype(bool)
                hdu.data[idx] = 0.00
            elif isinstance(r, np.ndarray):
                hdu.data = hdu.data * (r == 1.00).astype(float)
            elif isinstance(r, (fits.ImageHDU, fits.PrimaryHDU)):
                mask = _hdu_mask(mask, self.hdu)
                hdu.data = hdu.data * mask

        self.mask = hdu.data.astype(int).copy()

    #   Add PSF
    #   --------------------------------------------------------
    def addpsf(self, img, normalize=True, idx=0):
        """
        img : array_like, HDU, or str
            PSF image to be added to the image.
        normalize : bool, optional
            If True, normalize the PSF image.
        idx : int, optional
            Index of the HDU to be loaded.
        """
        if isinstance(img, (np.ndarray, jp.ndarray)):
            kernel = img.copy()
        else:
            hdu = _img_loader(img, idx)
            kernel = hdu.data.copy()

        if normalize:
            kernel = kernel / np.sum(kernel)

        kx, ky = kernel.shape
        dx, dy = self.data.shape

        if kx > dx:
            cx = (kx - dx) // 2
            kernel = kernel[cx : cx + dx, :]

        if ky > dy:
            cy = (ky - dy) // 2
            kernel = kernel[:, cy : cy + dy]

        self.psf = kernel

        pad_width = [
            (0, max(0, s - k)) for s, k in zip(self.data.shape, kernel.shape)
        ]
        self.psf_fft = np.pad(kernel, pad_width, mode="constant").astype(
            np.float64
        )
        self.psf_fft = jp.fft.rfft2(
            jp.fft.fftshift(self.psf_fft), s=self.psf_fft.shape
        )
        self.psf_fft = jp.abs(self.psf_fft)
