"""Image data handling and WCS coordinate grid utilities."""

from .utils import _img_loader, _reduce_axes
from . import noise as noisepdf

import jax
import jax.numpy as jp

import numpy as np

import inspect

from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D

import pyregion
import reproject


# Support functions
# ========================================================
# Get mask from input HDU
# --------------------------------------------------------
def _hdu_mask(mask, hdu):
    """
    Reproject a mask to match the WCS of a target HDU.

    Takes a mask HDU and reprojects it to match the coordinate system and
    shape of the target HDU. The reprojected mask is converted to binary
    values (0 or 1).

    Parameters
    ----------
    mask : fits.PrimaryHDU or fits.ImageHDU
        Input mask HDU to be reprojected.
    hdu : fits.PrimaryHDU or fits.ImageHDU
        Target HDU whose header defines the output WCS and shape.

    Returns
    -------
    numpy.ndarray
        Binary mask array with values 0.0 (masked) or 1.0 (valid).
        NaN values from reprojection are set to 0.0.
    """
    data, _ = reproject.reproject_interp(mask, hdu.header)
    data[np.isnan(data)] = 0.00
    return np.where(data < 1.00, 0.00, 1.00)


def pad_size(shape):
    """Compute the standard padded shape for convolution."""
    return tuple(2 * s - 1 for s in shape)


# Coordinate grids
# --------------------------------------------------------
class WCSgrid:
    """
    WCS coordinate grid for image.

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
        """
        Generate mesh coordinate arrays in world coordinate system.

        Creates 2D coordinate grids in world coordinates from either an HDU,
        WCS object, or FITS header. Handles 360-degree wraparound for RA
        coordinates near the discontinuity.

        Parameters
        ----------
        hdu : fits.PrimaryHDU or fits.ImageHDU, optional
            FITS HDU object containing header with WCS information.
        wcs : astropy.wcs.WCS, optional
            WCS object for coordinate transformation. If None, derived from
            header or HDU.
        header : fits.Header, optional
            FITS header containing WCS keywords.

        Returns
        -------
        tuple of numpy.ndarray
            (gridwx, gridwy) - 2D arrays of x and y world coordinates.

        Raises
        ------
        ValueError
            If neither header nor hdu is provided, or if both are provided.

        Notes
        -----
        Exactly one of `hdu` or `header` must be specified (not both).
        The method automatically handles RA coordinate discontinuity at 360°.
        """
        if (header is None) and (hdu is not None):
            headerWCS = hdu.header.copy()
        elif (header is not None) and (hdu is None):
            headerWCS = header.copy()
        else:
            raise ValueError("Either header or hdu should be defined.")

        if wcs is None:
            wcs = WCS(headerWCS)
        wcs = wcs.celestial

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
    FFT specification for image.

    Parameters
    ----------
    hdu : fits.PrimaryHDU
        FITS HDU object containing image data and header.

    Attributes
    ----------
    image_shape : tuple of int
        Shape of the original image ``(ny, nx)``.
    padded_shape : tuple of int
        Shape of the zero-padded array used for convolution.
    header : dict
        Dictionary containing relevant header keywords.
    center : jax.numpy.ndarray
        FFT of a unit impulse at the image centre, computed on the
        padded grid.
    freq : list of jax.numpy.ndarray
        Frequency grids in u and v directions, defined on the padded grid.
    pulse : jax.numpy.ndarray
        FFT of a unit pulse at the reference pixel position.
    """

    def __init__(self, hdu):
        self.image_shape = hdu.data.shape
        self.header = {
            key: hdu.header[key]
            for idx in [1, 2]
            for key in [
                f"CRPIX{idx}",
                f"CRVAL{idx}",
                f"CDELT{idx}",
                f"NAXIS{idx}",
            ]
        }

        self.padded_shape = pad_size(self.image_shape)

        self.center = jp.zeros(self.image_shape)
        self.center = self.center.at[
            self.image_shape[0] // 2, self.image_shape[1] // 2
        ].set(1.00)

        self.center = jp.fft.rfft2(self.center, self.padded_shape)

        self.freq = [
            jp.array(
                np.broadcast_to(
                    np.fft.rfftfreq(self.padded_shape[1])[None, :],
                    self.center.shape,
                )
            ),
            jp.array(
                np.broadcast_to(
                    np.fft.fftfreq(self.padded_shape[0])[:, None],
                    self.center.shape,
                )
            ),
        ]

        dx = self.image_shape[1] // 2 - self.header["CRPIX1"] + 1.00
        dy = self.image_shape[0] // 2 - self.header["CRPIX2"] + 1.00

        self.pulse = self.center * jp.exp(
            2.00j * jp.pi * (self.freq[0] * dx + self.freq[1] * dy)
        )

    def shift(self, xc, yc):
        """
        Compute Fourier shift for a point source at (xc, yc).

        Parameters
        ----------
        xc : float
            x-coordinate of the point source in world coordinates.
        yc : float
            y-coordinate of the point source in world coordinates.

        Returns
        -------
        jax.numpy.ndarray
            Complex array representing the Fourier shift for the point source.
        """
        dy = yc - self.header["CRVAL2"]
        dx = (xc - self.header["CRVAL1"]) * jp.cos(
            jp.deg2rad(self.header["CRVAL2"])
        )

        dx = dx / self.header["CDELT1"]
        dy = dy / self.header["CDELT2"]

        return jp.exp(-2.00j * jp.pi * (self.freq[0] * dx + self.freq[1] * dy))

    def ifft(self, data):
        """Compute inverse FFT to get image from Fourier-space data."""
        data_ = jp.fft.irfft2(data, self.padded_shape)

        start = tuple(
            (f - o) // 2 for f, o in zip(self.padded_shape, self.image_shape)
        )
        if self.image_shape[0] % 2 == 0:
            start = (start[0] + 1, start[1])
        if self.image_shape[1] % 2 == 0:
            start = (start[0], start[1] + 1)

        return jax.lax.dynamic_slice(data_, start, self.image_shape)


# Image constructor
# ========================================================
# Initialize image structure
# --------------------------------------------------------
class Image:
    """
    Image object for astronomical data.

    A class for handling astronomical images with support for WCS, PSF,
    masking, and noise modeling.

    This class provides a comprehensive framework for working with
    astronomical images, including coordinate transformations, PSF
    convolution, exposure and response maps, region masking, and various
    noise models. It supports FFT-based operations and sub-pixel sampling
    through a configurable subgrid parameter.

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

    Notes
    -----
    - Missing CDELT keywords are computed from CD matrix elements.
    - NaN values in the input image are automatically masked.
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
            self.exp.data[np.isnan(self.exp.data)] = 0.00
            self.exp = jp.array(self.exp.data.copy())
        else:
            self.exp = jp.ones(self.data.shape, dtype=float)

        if response is not None:
            self.resp = _img_loader(response, kwargs.get("resp_idx", 0))
            self.resp = _reduce_axes(self.resp)
            self.resp.data[np.isnan(self.resp.data)] = 0.00
            self.resp = jp.array(self.resp.data.copy())
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

    def _init_noise(self):
        """Initialise the noise model and populate the pixel mask.

        Inspects the noise model's call signature to forward the
        appropriate image attributes, then calls the model in-place so
        that ``self.noise.mask`` is populated and copied to
        ``self.mask``.

        Raises
        ------
        ValueError
            If the noise model is ``NormalRI`` but no PSF has been
            provided.
        """
        if self.psf is None and self.noise.__class__.__name__ == "NormalRI":
            raise ValueError(
                "PSF must be defined for images with NormalRI noise."
            )

        noise_kwargs = list(inspect.signature(self.noise).parameters.keys())
        noise_kwargs = {k: getattr(self, k) for k in noise_kwargs}
        self.noise(**noise_kwargs)
        self.mask = self.noise.mask.copy()

    #   Get cutout
    #   --------------------------------------------------------
    def cutout(self, center, csize):
        """
        Extract a cutout region from the image.

        Creates a smaller cutout of the image centered at the specified
        coordinates. All image attributes (data, mask, exposure, response,
        PSF, grid, FFT) are updated to reflect the cutout region.

        Parameters
        ----------
        center : tuple or astropy.coordinates.SkyCoord
            Center coordinates for the cutout. Can be a SkyCoord object
            or a tuple of (RA, Dec) values.
        csize : int, array_like, or astropy.units.Quantity
            Size of the cutout. Can be a single value (for square cutout),
            a tuple of (width, height), or a Quantity with angular units.

        Raises
        ------
        NotImplementedError
            If the noise model is not Normal (uncorrelated noise).

        Notes
        -----
        - Updates all image attributes including WCS, grid, and FFT specifications
        - If a PSF is present, it is also cutout to match the new image size
        - The image center (CRPIX/CRVAL) is updated to the cutout center
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
                "Cutout is only implemented for images with "
                "uncorrelated noise [Normal]."
            )

    #   Add mask
    #   --------------------------------------------------------
    def addmask(self, regions, combine=True):
        """
        Add or update masking regions to the image.

        Applies masking from various input formats (region files, pyregion
        objects, arrays, or HDUs). Masks can be combined with existing
        masks or replace them entirely.

        Parameters
        ----------
        regions : list
            List of regions to be masked. Elements can be:

            - str: Path to a region file (DS9/CRTF format)
            - pyregion.Shape: pyregion shape object
            - numpy.ndarray: Binary mask array (1=valid, 0=masked)
            - fits.ImageHDU or fits.PrimaryHDU: FITS mask HDU
        combine : bool, optional
            If True, combine with the existing mask (logical AND).
            If False, reset and replace the existing mask.
            Default is True.

        Notes
        -----
        - Multiple regions can be specified in a single call
        - For region files/shapes, pixels inside the region are masked (set to 0)
        - For array/HDU inputs, the mask is multiplied with existing mask
        - HDU masks are automatically reprojected to match image WCS
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
        Add a point spread function (PSF) to the image.

        Loads and prepares a PSF kernel for convolution operations. The PSF
        is normalized, resized if needed, and its FFT is precomputed for
        efficient convolution in Fourier space.

        Parameters
        ----------
        img : array_like, fits.HDU, or str
            PSF image input. Can be:

            - numpy or jax array: PSF kernel directly
            - fits.ImageHDU or fits.PrimaryHDU: FITS HDU with PSF
            - str: Path to FITS file containing PSF
        normalize : bool, optional
            If True, normalize the PSF to unit sum before use.
            Default is True.
        idx : int, optional
            HDU index to load if `img` is a filename.
            Default is 0 (primary HDU).

        Notes
        -----
        - If PSF dimensions exceed image dimensions, it is automatically cropped
        - The PSF FFT is precomputed and stored in `self.convolve.psf_fft`
        - PSF kernel is stored in `self.psf` for reference
        - PSF is zero-padded to match image dimensions before FFT
        """
        if isinstance(img, (np.ndarray, jp.ndarray)):
            kernel = img.copy()
        else:
            hdu = _img_loader(img, idx)
            hdu = _reduce_axes(hdu)
            kernel = np.asarray(hdu.data.astype(float).copy())

        if self.noise.__class__.__name__ == "NormalRI" and normalize:
            ValueError(
                "PSF normalization is not supported for NormalRI noise."
            )

        if normalize:
            kernel = kernel / np.sum(kernel)

        self.psf = kernel
        self.convolve = Convolve(self.psf, self.data.shape)


class Convolve:
    """FFT-based convolution operator with optional zero-padding."""

    def __init__(self, kernel, image_shape):
        """
        Initialize the convolution operator.

        Parameters
        ----------
        kernel : array_like
            PSF kernel array.
        image_shape : tuple of int
            Shape of the images to convolve.
        """
        self.kernel = kernel
        self.image_shape = image_shape

        self.padded_shape = pad_size(self.image_shape)

        self._fft = jp.fft.rfft2
        self._ifft = jp.fft.irfft2

        self.psf_fft = self._fft(self.kernel, self.padded_shape)

    def __call__(self, image):
        """
        Convolve an image with the PSF kernel.

        Parameters
        ----------
        image : jax.numpy.ndarray
            2-D image array to convolve.

        Returns
        -------
        jax.numpy.ndarray
            Convolved image with the same shape as the input.
        """
        _image = self._fft(image, self.padded_shape)
        _image = self._ifft(_image * self.psf_fft, self.padded_shape)

        start = tuple(
            (f - o) // 2 for f, o in zip(self.padded_shape, self.image_shape)
        )
        if self.image_shape[0] % 2 == 0:
            start = (start[0] + 1, start[1])
        if self.image_shape[1] % 2 == 0:
            start = (start[0], start[1] + 1)
        return jax.lax.dynamic_slice(_image, start, self.image_shape)
