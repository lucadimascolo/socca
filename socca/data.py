"""Image data handling and WCS coordinate grid utilities."""

from .utils import _img_loader, _reduce_axes, _tangent, _wrap_ra
from . import noise as noisepdf

import jax
import jax.numpy as jp

import numpy as np
from scipy.fft import next_fast_len

import copy
import inspect
import os
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
    values (0 or 1): pixels whose interpolated mask value is below 1 (up to
    round-off) are masked.

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
    return np.where(data < 1.00 - 1.00e-06, 0.00, 1.00)


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


def _psf_center(kernel, size=128, niter=50, tol=1e-9):
    """
    Centre of point symmetry ``(y, x)`` of a PSF stamp, in sub-pixels.

    Refined on tapered windows symmetric about the current estimate, so
    that truncation by the stamp edges does not bias it.
    """
    kernel = np.nan_to_num(np.asarray(kernel, dtype=float))
    if not np.any(kernel):
        raise ValueError("The PSF is empty.")
    nz = np.nonzero(np.abs(kernel) > 1e-12 * np.abs(kernel).max())
    peak = np.unravel_index(np.argmax(kernel), kernel.shape)
    lo = [max(p - size, 0) for p in peak]
    hi = [p + size + 1 for p in peak]
    center = _sym_center(kernel[lo[0] : hi[0], lo[1] : hi[1]]) + np.array(lo)
    step = np.inf
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
        step = np.max(np.abs(new - center))
        center = new
        if step < tol:
            break
    if np.isfinite(step) and step > 1e-03:
        warnings.warn(
            f"The PSF centre estimate did not converge (last step {step:.1e} px)."
        )
    snap = np.abs(center - np.round(center)) < 1e-6
    center = np.where(snap, np.round(center), center)
    center = np.clip(center, 0.00, np.array(kernel.shape) - 1.00)
    return tuple(float(c) for c in center)


class SkyToPixel:
    """
    Differentiable world-to-pixel transform of an image WCS.

    Positions are first projected exactly onto the plane tangent at the
    field centre; a 2D Chebyshev series fitted to the full astropy WCS
    (any projection, including distortions) then maps them to pixels.

    Parameters
    ----------
    header : fits.Header
        Image header with a celestial WCS.
    shape : tuple of int
        Image shape ``(ny, nx)``.
    margin : float, optional
        Fraction of the image size covered beyond each edge.
    tol : float, optional
        Target accuracy of the fit, in pixels.
    """

    def __init__(self, header, shape, margin=0.50, tol=1e-07, maxdeg=14):
        wcs = WCS(header).celestial
        ny, nx = shape
        self.ra0, self.dec0 = (
            float(c) for c in wcs.all_pix2world(0.50 * nx, 0.50 * ny, 0)
        )

        for m in [margin, 0.25, 0.10, 0.02]:
            fit = self._fit(wcs, nx, ny, min(m, margin), tol, maxdeg)
            if fit[-1] < 1e-04:
                break
        self.lo, self.hi, cx, cy, deg, self.residual = fit
        if self.residual > 1e-02:
            raise ValueError(
                f"The WCS is reproduced to {self.residual:.1e} pixels only: "
                "the field is too wide to place point sources."
            )
        if self.residual > 1e-04:
            warnings.warn(
                f"The WCS is reproduced to {self.residual:.1e} pixels only."
            )
        cheb = np.polynomial.chebyshev
        self.coef = []
        for c in (cx, cy):
            c = c.reshape(deg + 1, deg + 1)
            self.coef.append(
                [jp.asarray(c)]
                + [jp.asarray(cheb.chebder(c, axis=ax)) for ax in (0, 1)]
            )

    def _fit(self, wcs, nx, ny, margin, tol, maxdeg, n=120):
        py, px = np.mgrid[
            -margin * ny : (1 + margin) * ny : complex(0, n),
            -margin * nx : (1 + margin) * nx : complex(0, n),
        ]
        px, py = px.ravel(), py.ravel()
        xi, eta = (
            np.asarray(c)
            for c in _tangent(
                *wcs.all_pix2world(px, py, 0), self.ra0, self.dec0
            )
        )
        keep = np.isfinite(xi) & np.isfinite(eta) & (np.hypot(xi, eta) < 1e03)
        px, py, xi, eta = px[keep], py[keep], xi[keep], eta[keep]
        self.lo = np.array([xi.min(), eta.min()])
        self.hi = np.array([xi.max(), eta.max()])
        u, v = self._scale(xi, eta)

        for deg in range(3, maxdeg + 1):
            basis = np.polynomial.chebyshev.chebvander2d(u, v, [deg, deg])
            cx = np.linalg.lstsq(basis, px, rcond=None)[0]
            cy = np.linalg.lstsq(basis, py, rcond=None)[0]
            residual = max(
                np.abs(basis @ cx - px).max(), np.abs(basis @ cy - py).max()
            )
            if residual < tol:
                break
        return self.lo, self.hi, cx, cy, deg, residual

    def _scale(self, xi, eta):
        u = 2.00 * (xi - self.lo[0]) / (self.hi[0] - self.lo[0]) - 1.00
        v = 2.00 * (eta - self.lo[1]) / (self.hi[1] - self.lo[1]) - 1.00
        return u, v

    @staticmethod
    def _chebyshev(t, n):
        terms = [jp.ones_like(t), t]
        for _ in range(2, n):
            terms.append(2.00 * t * terms[-1] - terms[-2])
        return jp.stack(terms[:n])

    def _series(self, c, u, v):
        tu = self._chebyshev(u, c.shape[0])
        tv = self._chebyshev(v, c.shape[1])
        return jp.einsum("i...,ij,j...->...", tu, c, tv)

    def __call__(self, ra, dec):
        """
        Pixel position ``(x, y)`` (0-based) of ``(ra, dec)`` in degrees.

        Outside the fitted area the transform is continued linearly.
        """
        u, v = self._scale(*_tangent(ra, dec, self.ra0, self.dec0))
        uc, vc = jp.clip(u, -1.00, 1.00), jp.clip(v, -1.00, 1.00)
        pos = []
        for c, cu, cv in self.coef:
            p = self._series(c, uc, vc)
            p = p + (u - uc) * self._series(cu, uc, vc)
            p = p + (v - vc) * self._series(cv, uc, vc)
            pos.append(p)
        return pos[0], pos[1]


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
        self.ra0, self.dec0 = (
            float(c)
            for c in wcs_obj.all_pix2world(
                0.50 * hdu.data.shape[1], 0.50 * hdu.data.shape[0], 0
            )
        )
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

        a0, d0 = np.deg2rad(self.ra0), np.deg2rad(self.dec0)
        self.frame = (
            np.array(
                [np.cos(d0) * np.cos(a0), np.cos(d0) * np.sin(a0), np.sin(d0)]
            ),
            np.array([-np.sin(a0), np.cos(a0), 0.00]),
            np.array(
                [
                    -np.sin(d0) * np.cos(a0),
                    -np.sin(d0) * np.sin(a0),
                    np.cos(d0),
                ]
            ),
        )
        self.tx, self.ty = _tangent(self.x, self.y, self.ra0, self.dec0)
        self.wide = bool(np.max(np.hypot(self.tx, self.ty)) > 10.00)
        if np.any(np.asarray(self.tx) >= 1.00e04):
            warnings.warn(
                "Pixels more than 90 degrees from the image centre are not "
                "modelled."
            )

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
        gridwx = _wrap_ra(gridwx, wcs.wcs.crval[0])

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
    world2pix : SkyToPixel
        World-to-pixel transform of the image WCS.
    freq : list of jax.numpy.ndarray
        Frequency grids in u and v directions, defined on the padded grid.
    pulse : jax.numpy.ndarray
        FFT of a unit pulse at pixel ``(0, 0)``.
    """

    def __init__(self, hdu):
        self.image_shape = hdu.data.shape
        self._header = hdu.header.copy()

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

        self.pulse = _phase_ramp(self.padded_shape, 0.00, 0.00)

    @property
    def world2pix(self):
        """World-to-pixel transform of the image WCS, built on first use."""
        if getattr(self, "_world2pix", None) is None:
            with jax.ensure_compile_time_eval():
                self._world2pix = SkyToPixel(self._header, self.image_shape)
        return self._world2pix

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
        px, py = self.world2pix(xc, yc)
        return _phase_ramp(self.padded_shape, py, px)

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
        if any("TPV" in c for c in ctypes):
            raise NotImplementedError(
                "TPV distortions are not supported by astropy.wcs."
            )

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
        - The WCS projection is kept; only the reference pixel is moved
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

        header = self.hdu.header.copy()
        header["NAXIS1"], header["NAXIS2"] = cutout_data.data.shape[::-1]
        header["CRPIX1"], header["CRPIX2"] = (
            float(c) for c in cutout_data.wcs.wcs.crpix
        )
        cuthdu = fits.PrimaryHDU(data=cutout_data.data, header=header)

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
        regions : list or region
            Region, or list of regions, to be masked. Regions can be:

            - str or path: Path to a region file (DS9/CRTF format)
            - pyregion.Shape or pyregion.ShapeList: pyregion shapes, e.g.
              from ``pyregion.parse``
            - array: Binary mask array (1=valid, 0=masked), or a stack of them
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

        hdu = fits.PrimaryHDU(data=mask, header=self.hdu.header.copy())

        single = (str, os.PathLike, pyregion.ShapeList, pyregion.Shape)
        if isinstance(regions, (*single, fits.ImageHDU, fits.PrimaryHDU)) or (
            isinstance(regions, (np.ndarray, jax.Array)) and regions.ndim == 2
        ):
            regions = [regions]

        for ri, r in enumerate(regions):
            if isinstance(r, (str, os.PathLike)):
                r = pyregion.open(os.fspath(r))
            if isinstance(r, pyregion.Shape):
                r = pyregion.ShapeList([r])
            if isinstance(r, pyregion.ShapeList):
                if r:
                    hdu.data[r.get_mask(hdu=hdu).astype(bool)] = 0.00
            elif isinstance(r, (np.ndarray, jax.Array)):
                if r.shape != hdu.data.shape:
                    raise ValueError(
                        f"Mask {ri} has shape {r.shape}, but the image has "
                        f"shape {hdu.data.shape}."
                    )
                hdu.data = hdu.data * (np.asarray(r) == 1.00).astype(float)
            elif isinstance(r, (fits.ImageHDU, fits.PrimaryHDU)):
                hdu.data = hdu.data * _hdu_mask(r, self.hdu)
            else:
                raise TypeError(
                    f"Cannot mask with a {type(r).__name__} (region {ri}): "
                    "use a region file, pyregion shapes, an array or an HDU."
                )

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

        if kernel.ndim != 2:
            raise ValueError(
                f"The PSF must be 2D, not of shape {kernel.shape}."
            )

        if self.noise.__class__.__name__ == "NormalRI" and normalize:
            raise ValueError(
                "PSF normalization is not supported for NormalRI noise."
            )

        if not np.all(np.isfinite(kernel)):
            warnings.warn("Non-finite PSF values are set to zero.")
            kernel = np.nan_to_num(kernel, nan=0.00, posinf=0.00, neginf=0.00)

        if normalize:
            if np.abs(np.sum(kernel)) < 1e-06 * np.sum(np.abs(kernel)):
                warnings.warn(
                    "The PSF sums to nearly zero; normalising it to unit sum "
                    "amplifies it greatly."
                )
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

        margin = 64 if any(frac) else 0
        lo = [max(c - s + 1 - margin, 0) for c, s in zip(center, image_shape)]
        hi = [
            min(c + s + margin, k)
            for c, s, k in zip(center, image_shape, kshape)
        ]
        kernel = jp.asarray(kernel)[lo[0] : hi[0], lo[1] : hi[1]]
        kshape = kernel.shape
        center = [c - a for c, a in zip(center, lo)]

        grid = pad_size(kshape)
        if any(frac):
            spec = jp.fft.rfft2(kernel, grid)
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
