"""Tests for socca.utils module."""

import numpy as np
import pytest
from astropy.io import fits

from socca.utils import _img_loader, _reduce_axes


class TestImgLoader:
    """Tests for _img_loader function."""

    def test_load_primary_hdu(self, simple_hdu):
        """Test loading from a PrimaryHDU object."""
        result = _img_loader(simple_hdu)
        assert isinstance(result, fits.PrimaryHDU)
        assert result is simple_hdu

    def test_load_image_hdu(self, simple_hdu):
        """Test loading from an ImageHDU object."""
        image_hdu = fits.ImageHDU(
            data=simple_hdu.data.copy(), header=simple_hdu.header.copy()
        )
        result = _img_loader(image_hdu)
        assert isinstance(result, fits.ImageHDU)
        assert result is image_hdu

    def test_load_hdu_list(self, simple_hdu):
        """Test loading from an HDUList."""
        hdu_list = fits.HDUList([simple_hdu])
        result = _img_loader(hdu_list, idx=0)
        assert result is simple_hdu

    def test_load_hdu_list_with_index(self, simple_hdu):
        """Test loading a specific index from HDUList."""
        hdu0 = simple_hdu
        hdu1 = fits.ImageHDU(
            data=simple_hdu.data * 2, header=simple_hdu.header
        )
        hdu_list = fits.HDUList([hdu0, hdu1])
        result = _img_loader(hdu_list, idx=1)
        assert result is hdu1

    def test_load_from_file(self, temp_fits_file):
        """Test loading from a FITS file path."""
        result = _img_loader(temp_fits_file, idx=0)
        assert isinstance(result, (fits.PrimaryHDU, fits.ImageHDU))
        assert result.data is not None

    def test_invalid_input_raises_error(self):
        """Test that invalid input raises ValueError."""
        with pytest.raises(ValueError, match="img must be"):
            _img_loader([1, 2, 3])


class TestReduceAxes:
    """Tests for _reduce_axes function."""

    def test_reduce_3d_to_2d(self, simple_wcs_header):
        """Test reducing a 3D array to 2D."""
        data_3d = np.random.rand(1, 64, 64)
        header = simple_wcs_header.copy()
        header["NAXIS"] = 3
        header["NAXIS3"] = 1
        header["CRPIX3"] = 1
        header["CRVAL3"] = 1.0
        header["CDELT3"] = 1.0
        header["CTYPE3"] = "FREQ"

        hdu = fits.PrimaryHDU(data=data_3d, header=header)
        result = _reduce_axes(hdu)

        assert result.data.ndim == 2
        assert result.data.shape == (64, 64)
        assert result.header["NAXIS"] == 2
        assert "NAXIS3" not in result.header

    def test_reduce_4d_to_2d(self, simple_wcs_header):
        """Test reducing a 4D array to 2D."""
        data_4d = np.random.rand(1, 1, 64, 64)
        header = simple_wcs_header.copy()
        header["NAXIS"] = 4
        header["NAXIS3"] = 1
        header["NAXIS4"] = 1
        header["CRPIX3"] = 1
        header["CRVAL3"] = 1.0
        header["CDELT3"] = 1.0
        header["CTYPE3"] = "FREQ"
        header["CRPIX4"] = 1
        header["CRVAL4"] = 1.0
        header["CDELT4"] = 1.0
        header["CTYPE4"] = "STOKES"

        hdu = fits.PrimaryHDU(data=data_4d, header=header)
        result = _reduce_axes(hdu)

        assert result.data.ndim == 2
        assert result.data.shape == (64, 64)
        assert result.header["NAXIS"] == 2
        assert "NAXIS3" not in result.header
        assert "NAXIS4" not in result.header

    def test_2d_unchanged(self, simple_hdu):
        """Test that a 2D array is returned unchanged (except header cleanup)."""
        result = _reduce_axes(simple_hdu)
        assert result.data.shape == simple_hdu.data.shape
        np.testing.assert_array_equal(result.data, simple_hdu.data)

    def test_header_keywords_removed(self, simple_wcs_header):
        """Test that header keywords for axes 3 and 4 are removed."""
        data_3d = np.random.rand(1, 64, 64)
        header = simple_wcs_header.copy()
        header["NAXIS"] = 3
        header["NAXIS3"] = 1
        header["CRPIX3"] = 1
        header["CRVAL3"] = 1.0
        header["CDELT3"] = 1.0
        header["CTYPE3"] = "FREQ"
        header["CUNIT3"] = "Hz"
        header["CD3_3"] = 1.0
        header["PC3_3"] = 1.0

        hdu = fits.PrimaryHDU(data=data_3d, header=header)
        result = _reduce_axes(hdu)

        for key in ["CRPIX3", "CRVAL3", "CDELT3", "CTYPE3", "CUNIT3"]:
            assert key not in result.header

    def test_returns_primary_hdu(self, simple_hdu):
        """Test that result is always a PrimaryHDU."""
        result = _reduce_axes(simple_hdu)
        assert isinstance(result, fits.PrimaryHDU)
