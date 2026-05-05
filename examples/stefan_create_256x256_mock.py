"""
Create a 256x256 version of tutorial_mock.fits by padding the 128x128 image.
The reference pixel is adjusted to maintain the same world coordinate at the center.
"""
from astropy.io import fits
import numpy as np
import os

# Load the original 128x128 mock
original_path = "data/tutorial/tutorial_mock.fits"
output_path = "data/tutorial/tutorial_mock_256x256.fits"

# Open the original FITS file
with fits.open(original_path) as hdul:
    original_data = hdul[0].data.copy()
    original_header = hdul[0].header.copy()

print(f"Original shape: {original_data.shape}")
print(f"Original CRPIX1: {original_header.get('CRPIX1', 'N/A')}")
print(f"Original CRPIX2: {original_header.get('CRPIX2', 'N/A')}")

# Create 256x256 array with padding
padded_data = np.zeros((256, 256), dtype=original_data.dtype)

# Center the original 128x128 image (64 pixels padding on all sides)
padded_data[64:192, 64:192] = original_data

# Adjust WCS reference pixel
# The reference pixel shifts by 64 pixels in each direction
if 'CRPIX1' in original_header:
    original_header['CRPIX1'] += 64
if 'CRPIX2' in original_header:
    original_header['CRPIX2'] += 64

print(f"\nNew shape: {padded_data.shape}")
print(f"New CRPIX1: {original_header.get('CRPIX1', 'N/A')}")
print(f"New CRPIX2: {original_header.get('CRPIX2', 'N/A')}")

# Create new FITS HDU and save
hdu = fits.PrimaryHDU(data=padded_data, header=original_header)
hdu.writeto(output_path, overwrite=True)

print(f"\nSaved to: {output_path}")
