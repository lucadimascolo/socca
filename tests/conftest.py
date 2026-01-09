"""Pytest configuration and fixtures for socca tests."""

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS


@pytest.fixture
def simple_wcs_header():
    """Create a simple WCS header for testing."""
    header = fits.Header()
    header["NAXIS"] = 2
    header["NAXIS1"] = 64
    header["NAXIS2"] = 64
    header["CRPIX1"] = 32.5
    header["CRPIX2"] = 32.5
    header["CRVAL1"] = 180.0
    header["CRVAL2"] = 45.0
    header["CDELT1"] = -0.001
    header["CDELT2"] = 0.001
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    return header


@pytest.fixture
def simple_hdu(simple_wcs_header):
    """Create a simple FITS HDU with Gaussian data for testing."""
    np.random.seed(42)
    shape = (64, 64)
    y, x = np.mgrid[0 : shape[0], 0 : shape[1]]
    cx, cy = shape[1] / 2, shape[0] / 2
    sigma = 5.0
    data = 10.0 * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma**2))
    data += np.random.normal(0.0, 0.1, shape)
    return fits.PrimaryHDU(
        data=data.astype(np.float64), header=simple_wcs_header
    )


@pytest.fixture
def simple_image_data():
    """Create simple 2D image data for testing."""
    np.random.seed(42)
    shape = (32, 32)
    y, x = np.mgrid[0 : shape[0], 0 : shape[1]]
    cx, cy = shape[1] / 2, shape[0] / 2
    sigma = 3.0
    data = 5.0 * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma**2))
    data += np.random.normal(0, 0.05, shape)
    return data.astype(np.float64)


@pytest.fixture
def gaussian_psf():
    """Create a Gaussian PSF kernel for testing."""
    size = 11
    y, x = np.mgrid[0:size, 0:size]
    cx, cy = size / 2, size / 2
    sigma = 1.5
    psf = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma**2))
    psf /= psf.sum()
    return psf.astype(np.float64)


@pytest.fixture
def temp_fits_file(simple_hdu, tmp_path):
    """Create a temporary FITS file for testing file loading."""
    filepath = tmp_path / "test_image.fits"
    simple_hdu.writeto(filepath, overwrite=True)
    return str(filepath)


@pytest.fixture
def temp_psf_file(gaussian_psf, tmp_path, simple_wcs_header):
    """Create a temporary PSF FITS file for testing."""
    filepath = tmp_path / "test_psf.fits"
    header = simple_wcs_header.copy()
    header["NAXIS1"] = gaussian_psf.shape[1]
    header["NAXIS2"] = gaussian_psf.shape[0]
    hdu = fits.PrimaryHDU(data=gaussian_psf, header=header)
    hdu.writeto(filepath, overwrite=True)
    return str(filepath)


@pytest.fixture
def noise_sigma_map(simple_hdu):
    """Create a sigma noise map for testing."""
    return np.full(simple_hdu.data.shape, 0.1, dtype=np.float64)


@pytest.fixture
def uniform_mask(simple_hdu):
    """Create a uniform (all valid) mask for testing."""
    return np.ones(simple_hdu.data.shape, dtype=int)
