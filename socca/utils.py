"""Helper utilities for image loading and array operations."""

import jax.numpy as jp
import numpy as np
from astropy.io import fits

__all__ = [
    "_img_loader",
    "_local_frame",
    "_offsets",
    "_reduce_axes",
    "_tangent",
    "_wrap_ra",
    "apodization",
]


# Wrap right ascension
# --------------------------------------------------------
def _wrap_ra(ra, ref):
    """Wrap `ra` (deg) onto the 360-degree branch centred on `ref`."""
    return ref + (ra - ref + 180.00) % 360.00 - 180.00


# Tangent-plane coordinates
# --------------------------------------------------------
def _tangent(ra, dec, ra0, dec0):
    """
    Gnomonic offsets (deg) of `(ra, dec)` on the plane tangent at `(ra0, dec0)`.

    The first axis points to increasing right ascension and the second to
    increasing declination. Positions more than 90 degrees away, which have
    no gnomonic projection, are placed at 1e4 degrees.
    """
    a, d, a0, d0 = (jp.deg2rad(c) for c in (ra, dec, ra0, dec0))
    cosd, sind, cosd0, sind0 = jp.cos(d), jp.sin(d), jp.cos(d0), jp.sin(d0)
    cosa = jp.cos(a - a0)
    cosc = sind0 * sind + cosd0 * cosd * cosa
    front = cosc > 0.00
    cosc = jp.where(front, cosc, 1.00)
    xi = jp.where(front, jp.rad2deg(cosd * jp.sin(a - a0) / cosc), 1.00e04)
    eta = jp.where(
        front, jp.rad2deg((cosd0 * sind - sind0 * cosd * cosa) / cosc), 1.00e04
    )
    return xi, eta


def _local_frame(grid, xc, yc):
    """
    Pixel offsets from ``(xc, yc)`` and their map to its local sky frame.

    Returns the offsets ``dx, dy`` (deg) of the grid pixels from ``(xc, yc)``
    on the plane tangent at the image centre, and the 2x2 matrix that turns
    them into east/north offsets on the plane tangent at ``(xc, yc)``. This
    is exact at ``(xc, yc)``, with a relative error of about the product of
    the distances (in radians) of a pixel from ``(xc, yc)`` and of
    ``(xc, yc)`` from the image centre. On wide images the offsets are
    computed exactly on the plane tangent at ``(xc, yc)`` instead.
    """
    if grid.wide:
        dx, dy = _tangent(grid.x, grid.y, xc, yc)
        return dx, dy, jp.eye(2)
    a, d = jp.deg2rad(xc), jp.deg2rad(yc)
    n = jp.stack([jp.cos(d) * jp.cos(a), jp.cos(d) * jp.sin(a), jp.sin(d)])
    east = jp.stack([-jp.sin(a), jp.cos(a), jp.zeros_like(a)])
    north = jp.stack(
        [-jp.sin(d) * jp.cos(a), -jp.sin(d) * jp.sin(a), jp.cos(d)]
    )
    n0, e0, f0 = (jp.asarray(v) for v in grid.frame)
    dn = n @ n0
    dx = grid.tx - jp.rad2deg((n @ e0) / dn)
    dy = grid.ty - jp.rad2deg((n @ f0) / dn)
    j = dn * jp.array([[east @ e0, east @ f0], [north @ e0, north @ f0]])
    return dx, dy, j


def _offsets(grid, xc, yc):
    """East/north offsets (deg) of the grid pixels from ``(xc, yc)``."""
    dx, dy, j = _local_frame(grid, xc, yc)
    return j[0, 0] * dx + j[0, 1] * dy, j[1, 0] * dx + j[1, 1] * dy


# Load image
# --------------------------------------------------------
def _img_loader(img, idx=0):
    """
    Load an image from various input formats.

    Flexible loader that handles FITS data in multiple formats: HDU objects,
    HDUList, or filename paths. Automatically extracts the appropriate HDU
    based on the provided index.

    Parameters
    ----------
    img : fits.PrimaryHDU, fits.ImageHDU, fits.HDUList, or str
        Input image to load. Can be:

        - fits.PrimaryHDU or fits.ImageHDU: Returned directly
        - fits.HDUList: HDU at index `idx` is extracted
        - str: Path to FITS file, opened and HDU at index `idx` extracted
    idx : int, optional
        Index of the HDU to extract from HDUList or FITS file.
        Default is 0 (primary HDU).

    Returns
    -------
    fits.PrimaryHDU or fits.ImageHDU
        The loaded HDU object.

    Raises
    ------
    ValueError
        If `img` is not a recognized type (HDU, HDUList, or string path).
    """
    if isinstance(img, (fits.ImageHDU, fits.PrimaryHDU)):
        return img
    elif isinstance(img, fits.hdu.hdulist.HDUList):
        return img[idx]
    elif isinstance(img, str):
        img = fits.open(img)
        return img[idx]
    else:
        raise ValueError("img must be an ImageHDU or a string")


# Reduce axes to 2D
# --------------------------------------------------------
def _reduce_axes(hdu):
    """
    Reduce multi-dimensional FITS data to 2D by removing extra axes.

    Squeezes the data array to remove singleton dimensions and cleans the
    FITS header by removing keywords associated with axes beyond the first
    two (spatial) dimensions. This is useful for reducing spectral cubes or
    other higher-dimensional data to 2D images.

    Parameters
    ----------
    hdu : fits.PrimaryHDU or fits.ImageHDU
        Input HDU with potentially higher-dimensional data (3D, 4D, etc.).

    Returns
    -------
    fits.PrimaryHDU
        New PrimaryHDU with 2D data array and cleaned header containing
        only keywords relevant to the first two axes.

    Notes
    -----
    - Data is squeezed using numpy.squeeze() to remove singleton dimensions
    - Removes NAXIS, CRPIX, CRVAL, CDELT, CUNIT, CTYPE keywords for axes 3 and 4
    - Removes CD and PC matrix elements involving axes 3 and 4
    - Sets NAXIS=2 in the output header
    """
    head = hdu.header.copy()

    data = hdu.data.copy()
    data = data.squeeze()

    head["NAXIS"] = 2
    for idx in [3, 4]:
        for key in ["NAXIS", "CRPIX", "CRVAL", "CDELT", "CUNIT", "CTYPE"]:
            head.pop(f"{key}{idx}", None)

        for jdx in range(1, 5):
            for key in ["CD", "PC"]:
                head.pop(f"{key}{jdx}_{idx}", None)
                head.pop(f"{key}{idx}_{jdx}", None)

    return fits.PrimaryHDU(data=data, header=head)


# Build an apodization window
# --------------------------------------------------------
def apodization(shape, width, profile="cos", alpha=1.00):
    """
    Build a 2D apodization window with a tapered border.

    Reproduces pixell's ``enmap.apod(map, width, profile, fill="zero")``:
    an array of ones with a taper applied within `width` pixels of each
    edge, separately (and multiplicatively) along both axes.

    Parameters
    ----------
    shape : tuple of int
        Shape of the output window, as (ny, nx).
    width : int or float
        Width of the tapered border, applied to all four edges. If
        `width` is a float in (0, 1), it is interpreted as a fraction of
        the respective axis length (rounded to the nearest pixel, so the
        y- and x-axis tapers may differ for a non-square `shape`);
        otherwise it is interpreted directly as a number of pixels. If
        the resolved pixel width is not positive, an array of ones is
        returned.
    profile : str, optional
        Shape of the taper. Either "cos" (raised-cosine, the default) or
        "lin" (linear ramp).
    alpha : float, optional
        Power to which the taper is raised, controlling how sharply it
        rises from 0 to 1. Default is 1.00 (unmodified taper); values
        above/below 1 make it steeper/gentler near the edge.

    Returns
    -------
    numpy.ndarray
        Array of shape `shape`, with values in [0, 1].

    Raises
    ------
    ValueError
        If `profile` is not one of "cos" or "lin".
    """
    if profile not in ("cos", "lin"):
        raise ValueError(f"Unknown apodization profile '{profile}'.")

    ny, nx = shape
    out = np.ones((ny, nx))

    def resolve(width, n):
        if 0.00 < width < 1.00:
            width = width * n
        return int(round(width))

    wy, wx = resolve(width, ny), resolve(width, nx)

    def taper(w):
        x = np.arange(1, w + 1) / (w + 1)
        if profile == "cos":
            return (0.50 * (1.00 - np.cos(np.pi * x))) ** alpha
        return x**alpha

    if wy > 0:
        prof = taper(wy)
        out[:wy, :] *= prof[:, None]
        out[-wy:, :] *= prof[::-1, None]
    if wx > 0:
        prof = taper(wx)
        out[:, :wx] *= prof[None, :]
        out[:, -wx:] *= prof[None, ::-1]
    return out
