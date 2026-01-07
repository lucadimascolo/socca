from astropy.io import fits

__all__ = ["_img_loader", "_reduce_axes"]


# Load image
# --------------------------------------------------------
def _img_loader(img, idx=0):
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
