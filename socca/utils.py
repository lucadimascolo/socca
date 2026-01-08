"""Helper utilities for image loading and array operations."""

from astropy.io import fits

__all__ = ["_img_loader", "_reduce_axes"]


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
