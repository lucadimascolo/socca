"""Image data handling and WCS coordinate grid utilities."""

from .utils import _img_loader, _reduce_axes
from . import noise as noisepdf

import jax
import jax.numpy as jp

import numpy as np
from scipy.fft import next_fast_len

import copy
import inspect
import warnings

from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import wcs_to_celestial_frame
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D

import pyregion
import reproject


_pad_factor = 2.00


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


def pad_size(shape, factor=_pad_factor):
    """
    Compute the standard padded shape for convolution.

    Each axis is padded to at least ``factor * s - 1`` and rounded up to
    the next fast FFT length.
    """
    if factor < 1:
        raise ValueError("Padding factor must be >= 1.")
    return tuple(next_fast_len(int(factor * s - 1), real=True) for s in shape)


def _phase_ramp(shape, py, px):
    """
    Spectrum of a unit pulse at pixel ``(py, px)`` on an rfft2 grid.

    On even axes the Nyquist term takes the real part of the ramp, as for
    any real signal, so that sub-pixel shifts stay symmetric between axes.
    """
    ramp = []
    for n, p, freq in [
        (shape[0], py, np.fft.fftfreq),
        (shape[1], px, np.fft.rfftfreq),
    ]:
        f = freq(n)
        r = jp.exp(-2.00j * jp.pi * f * p)
        if n % 2 == 0:
            r = r.at[n // 2].set(jp.cos(jp.pi * p))
        ramp.append(r)
    return ramp[0][:, None] * ramp[1][None, :]


def _sym_center(kernel):
    """Single estimate of the centre of point symmetry of a stamp."""
    shape = tuple(2 * s for s in kernel.shape)
    spec = np.fft.fft2(kernel, shape)
    spec = spec * spec
    peak = np.unravel_index(np.argmax(np.fft.ifft2(spec).real), shape)
    center = np.array(peak, dtype=float) / 2.00
    fy = np.fft.fftfreq(shape[0])[:, None]
    fx = np.fft.fftfreq(shape[1])[None, :]
    spec = spec * np.exp(2.00j * np.pi * (fy * peak[0] + fx * peak[1]))
    for ax, n in enumerate(shape):
        a = np.moveaxis(spec, ax, 0)
        z = np.sum(a[1 : n // 2] * np.conj(a[: n // 2 - 1]))
        z = z + np.sum(a[n // 2 + 2 :] * np.conj(a[n // 2 + 1 : -1]))
        center[ax] -= np.angle(z) * n / (4.00 * np.pi)
    return center


def _psf_center(kernel, size=128, niter=8, tol=1e-9):
    """
    Centre of point symmetry ``(y, x)`` of a PSF stamp, in sub-pixels.

    Refined on tapered windows symmetric about the current estimate, so
    that truncation by the stamp edges does not bias it.
    """
    kernel = np.nan_to_num(np.asarray(kernel, dtype=float))
    if not np.any(kernel):
        raise ValueError("The PSF is empty.")
    nz = np.nonzero(np.abs(kernel) > 1e-12 * np.abs(kernel).max())
    center = _sym_center(kernel)
    for _ in range(niter):
        h = np.round(2.00 * center) / 2.00
        r = [
            min(hh, s - 1 - hh, size, max(hh - i.min(), i.max() - hh) + 1)
            for hh, s, i in zip(h, kernel.shape, nz)
        ]
        lo = [int(np.ceil(hh - rr)) for hh, rr in zip(h, r)]
        hi = [int(np.floor(hh + rr)) + 1 for hh, rr in zip(h, r)]
        if any(b - a < 3 for a, b in zip(lo, hi)):
            break
        sub = kernel[lo[0] : hi[0], lo[1] : hi[1]]
        w = np.ones(sub.shape)
        for ax, (a, b) in enumerate(zip(lo, hi)):
            t = 1.00 - np.abs(np.arange(a, b) - center[ax]) / (0.50 * (b - a))
            t = np.sin(np.pi * np.clip(t, 0.00, 0.50)) ** 2
            w = w * np.expand_dims(t, 1 - ax)
        new = _sym_center(sub * w) + np.array(lo)
        done = np.max(np.abs(new - center)) < tol
        center = new
        if done:
            break
    snap = np.abs(center - np.round(center)) < 1e-6
    center = np.where(snap, np.round(center), center)
    center = np.clip(center, 0.00, np.array(kernel.shape) - 1.00)
    return tuple(float(c) for c in center)


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
        wcs_obj = WCS(hdu.header).celestial
        for isp in range(subgrid):
            for jsp in range(subgrid):
                dx_sub = isp / subgrid - (subgrid - 1) / (2 * subgrid)
                dy_sub = jsp / subgrid - (subgrid - 1) / (2 * subgrid)
                multix[isp * subgrid + jsp], multiy[isp * subgrid + jsp] = (
                    self.getmesh(
                        header=hdu.header, wcs=wcs_obj, dx=dx_sub, dy=dy_sub
                    )
                )
        self.x = jp.array(multix)
        self.y = jp.array(multiy)

    @staticmethod
    def getmesh(hdu=None, wcs=None, header=None, dx=0.0, dy=0.0):
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
        dx : float, optional
            Sub-pixel offset in the x (column) direction, in pixels. Default 0.
        dy : float, optional
            Sub-pixel offset in the y (row) direction, in pixels. Default 0.

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
            np.arange(headerWCS["NAXIS1"]) + dx,
            np.arange(headerWCS["NAXIS2"]) + dy,
        )
        gridwx, gridwy = wcs.all_pix2world(gridmx, gridmy, 0)

        s = np.sort(gridwx.ravel())
        diffs = np.diff(s)
        if diffs.max() > 180:
            gridwx[gridwx > s[diffs.argmax()]] -= 360

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

        self.cd_inv = jp.array(
            np.linalg.inv(WCS(hdu.header).celestial.pixel_scale_matrix)
        )

        self.padded_shape = pad_size(self.image_shape)

        fshape = (self.padded_shape[0], self.padded_shape[1] // 2 + 1)

        self.freq = [
            jp.array(
                np.broadcast_to(
                    np.fft.rfftfreq(self.padded_shape[1])[None, :], fshape
                )
            ),
            jp.array(
                np.broadcast_to(
                    np.fft.fftfreq(self.padded_shape[0])[:, None], fshape
                )
            ),
        ]

        dx = self.header["CRPIX1"] - 1.00
        dy = self.header["CRPIX2"] - 1.00

        self.pulse = _phase_ramp(self.padded_shape, dy, dx)

    def shift(self, xc, yc):
        """
        Compute the spectrum of a unit point source at (xc, yc).

        Parameters
        ----------
        xc : float
            x-coordinate of the point source in world coordinates.
        yc : float
            y-coordinate of the point source in world coordinates.

        Returns
        -------
        jax.numpy.ndarray
            Complex spectrum of the point source on the padded grid.
        """
        dw = jp.array(
            [
                (xc - self.header["CRVAL1"])
                * jp.cos(jp.deg2rad(self.header["CRVAL2"])),
                yc - self.header["CRVAL2"],
            ]
        )
        dp = self.cd_inv @ dw
        return _phase_ramp(
            self.padded_shape,
            self.header["CRPIX2"] - 1.00 + dp[1],
            self.header["CRPIX1"] - 1.00 + dp[0],
        )

    def fft(self, data):
        """Compute FFT of the input data with zero-padding."""
        return jp.fft.rfft2(data, self.padded_shape)

    def ifft(self, data):
        """Compute inverse FFT to get image from Fourier-space data."""
        data_ = jp.fft.irfft2(data, self.padded_shape)
        return data_[: self.image_shape[0], : self.image_shape[1]]


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
            Dictionary with 'img', 'normalize', 'idx', and 'center' keys
            for PSF, passed to `addpsf`.

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

        ctypes = [str(self.hdu.header.get(f"CTYPE{i}", "")) for i in [1, 2]]
        if self.wcs.has_distortion or any("TPV" in c for c in ctypes):
            raise NotImplementedError(
                "Non-linear WCS distortions (SIP, TPV) are not supported. "
                "The pixel↔world transform uses the linear CD/PC matrix only."
            )

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
                self.hdu.header["CD1_2"], self.hdu.header["CD2_2"]
            )

        self.data = jp.array(
            self.hdu.data.astype(self.hdu.data.dtype.newbyteorder("="))
        )
        self.grid = WCSgrid(self.hdu, subgrid=self.subgrid)
        self.fft = FFTspec(self.hdu)

        self.mask = jp.ones(self.data.shape, dtype=int)
        self.mask = self.mask.at[jp.isnan(self.data)].set(0)

        if exposure is not None:
            self.exposure = _img_loader(exposure, kwargs.get("exp_idx", 0))
            self.exposure = _reduce_axes(self.exposure)
            self.exposure.data[np.isnan(self.exposure.data)] = 0.00
            self.exposure = jp.array(np.array(self.exposure.data, dtype=float))
        else:
            self.exposure = jp.ones(self.data.shape, dtype=float)

        if response is not None:
            self.response = _img_loader(response, kwargs.get("resp_idx", 0))
            self.response = _reduce_axes(self.response)
            self.response.data[np.isnan(self.response.data)] = 0.00
            self.response = jp.array(np.array(self.response.data, dtype=float))
        else:
            self.response = jp.ones(self.data.shape, dtype=float)

        if noise is None:
            self.noise = noisepdf.Normal()
        else:
            self.noise = noise

        self.psf = None

        if "center" in kwargs and "csize" in kwargs:
            self.cutout(center=kwargs["center"], csize=kwargs["csize"])

        if "addmask" in kwargs:
            self.addmask(
                regions=kwargs["addmask"].get("regions"),
                combine=kwargs["addmask"].get("combine", True),
            )

        if "addpsf" in kwargs:
            self.addpsf(
                img=kwargs["addpsf"].get("img"),
                normalize=kwargs["addpsf"].get("normalize", True),
                idx=kwargs["addpsf"].get("idx", 0),
                center=kwargs["addpsf"].get("center", "auto"),
            )

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
        - If a PSF is present, it is kept unchanged, with the same centre
        - The image center (CRPIX/CRVAL) is updated to the cutout center
        """
        if self.noise.__class__.__name__ != "Normal":
            raise NotImplementedError(
                "Cutout is only implemented for images with "
                "uncorrelated noise [Normal]."
            )

        if not isinstance(center, SkyCoord):
            center = SkyCoord(
                *center, unit="deg", frame=wcs_to_celestial_frame(self.wcs)
            )

        select = self.noise.select
        if select is not None and not isinstance(
            self.noise.kwargs[select], (float, int)
        ):
            nmap = self.noise.kwargs[select]
            if not isinstance(nmap, (np.ndarray, jp.ndarray)):
                nmap = _img_loader(nmap, self.noise.kwargs.get("idx", 0))
                nmap = _reduce_axes(nmap).data
            if np.shape(nmap) != self.data.shape:
                raise ValueError(
                    f"The noise map has shape {np.shape(nmap)}, "
                    f"the image {self.data.shape}."
                )
            self.noise = copy.copy(self.noise)
            self.noise.kwargs = dict(self.noise.kwargs)
            self.noise.kwargs[select] = Cutout2D(
                np.array(nmap, dtype=float),
                center,
                csize,
                wcs=self.wcs.celestial,
            ).data

        cutout_data = Cutout2D(
            self.data, center, csize, wcs=self.wcs.celestial
        )
        cutout_mask = Cutout2D(
            self.mask, center, csize, wcs=self.wcs.celestial
        )
        cutout_exp = Cutout2D(
            self.exposure, center, csize, wcs=self.wcs.celestial
        )
        cutout_resp = Cutout2D(
            self.response, center, csize, wcs=self.wcs.celestial
        )

        self.data = jp.array(np.array(cutout_data.data, dtype=float))
        self.mask = jp.array(np.array(cutout_mask.data, dtype=float))
        del cutout_mask
        self.exposure = jp.array(np.array(cutout_exp.data, dtype=float))
        del cutout_exp
        self.response = jp.array(np.array(cutout_resp.data, dtype=float))
        del cutout_resp

        if self.psf is not None:
            self.convolve = Convolve(
                self.psf, self.data.shape, center=self.convolve.center
            )

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
    def addpsf(self, img, normalize=True, idx=0, center="auto"):
        """
        Add a point spread function (PSF) to the image.

        Loads and prepares a PSF kernel for convolution operations. The PSF
        FFT is precomputed for efficient convolution in Fourier space.

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
        center : "auto", None, or tuple of float, optional
            PSF centre ``(y, x)`` in pixels of the stamp. "auto" (default)
            estimates it as the centre of point symmetry, to sub-pixel
            precision; None takes pixel ``n // 2`` along each axis, i.e.
            no recentring.

        Notes
        -----
        - The PSF is moved so that its centre lands on the source; only
          this translation is applied, and none for a whole-pixel centre
        - The PSF is stored unchanged (up to normalization) in `self.psf`
        - Only the part of the PSF that can reach the image is used
        - The PSF FFT is precomputed and stored in `self.convolve.psf_fft`
        """
        if isinstance(img, (np.ndarray, jp.ndarray)):
            kernel = np.asarray(img, dtype=float).copy()
        else:
            hdu = _img_loader(img, idx)
            hdu = _reduce_axes(hdu)
            kernel = np.asarray(hdu.data.astype(float).copy())

        if self.noise.__class__.__name__ == "NormalRI" and normalize:
            raise ValueError(
                "PSF normalization is not supported for NormalRI noise."
            )

        if normalize:
            kernel = kernel / np.sum(kernel)

        self.psf = jp.asarray(kernel)
        if isinstance(center, str):
            if center != "auto":
                raise ValueError('center must be "auto", None, or (y, x).')
            center = None
        elif center is None:
            center = tuple(k // 2 for k in kernel.shape)
        self.convolve = Convolve(self.psf, self.data.shape, center=center)


class Convolve:
    """FFT-based convolution operator with optional zero-padding."""

    def __init__(self, kernel, image_shape, center=None):
        """
        Initialize the convolution operator.

        Parameters
        ----------
        kernel : array_like
            PSF kernel array.
        image_shape : tuple of int
            Shape of the images to convolve.
        center : tuple of float, optional
            Position ``(y, x)`` of the PSF centre in the kernel. Defaults
            to its centre of point symmetry.

        Notes
        -----
        The kernel is rolled so that the pixel nearest to ``center`` sits
        at pixel ``(0, 0)`` of the padded grid, and the sub-pixel remainder
        is removed with a phase ramp, so that convolved spectra share the
        origin of ``FFTspec`` and are cropped from the origin of the
        padded grid.
        """
        self.kernel = kernel
        self.image_shape = image_shape

        self.padded_shape = pad_size(self.image_shape)

        self._fft = jp.fft.rfft2
        self._ifft = jp.fft.irfft2

        kshape = np.shape(kernel)
        if center is None:
            center = _psf_center(kernel)
        if np.ndim(center) != 1 or len(center) != 2:
            raise ValueError("center must be a (y, x) position.")
        if any(c < -0.50 or c > k - 0.50 for c, k in zip(center, kshape)):
            raise ValueError("PSF center must lie within the kernel.")
        self.center = tuple(float(c) for c in center)

        frac = [c - np.round(c) for c in self.center]
        center = [int(np.round(c)) for c in self.center]

        grid = pad_size(kshape)
        if any(frac):
            spec = jp.fft.rfft2(jp.asarray(kernel), grid)
            spec = spec * _phase_ramp(grid, -frac[0], -frac[1])
            full = jp.fft.irfft2(spec, grid)
        else:
            full = jp.zeros(grid).at[: kshape[0], : kshape[1]].set(kernel)
        full = jp.roll(full, (-center[0], -center[1]), axis=(0, 1))

        lag = [
            np.r_[0 : min(s, g // 2 + 1), -min(s - 1, (g - 1) // 2) : 0]
            for s, g in zip(image_shape, grid)
        ]
        iy, ix = np.ix_(lag[0], lag[1])
        padded = jp.zeros(self.padded_shape)
        padded = padded.at[
            iy % self.padded_shape[0], ix % self.padded_shape[1]
        ].set(full[iy % grid[0], ix % grid[1]])
        self.psf_fft = self._fft(padded)

    def crop(self, data_fft):
        """
        Inverse-FFT a Fourier-domain array and crop it to the image shape.

        Use this directly (instead of __call__) for data that has already
        been multiplied by self.psf_fft in Fourier space -- e.g. point
        sources rendered via FFTspec's phase-ramp pulses.

        Parameters
        ----------
        data_fft : jax.numpy.ndarray
            Fourier-domain array of shape self.psf_fft.shape.

        Returns
        -------
        jax.numpy.ndarray
            Image-space array with shape image_shape.
        """
        data = self._ifft(data_fft, self.padded_shape)
        return data[: self.image_shape[0], : self.image_shape[1]]

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
        return self.crop(_image * self.psf_fft)
