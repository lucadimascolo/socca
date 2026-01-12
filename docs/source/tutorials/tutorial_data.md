# Loading data

In **``socca``**, observational data are handled through the `Image` class, which provides a unified interface for loading astronomical images together with any associated supporting (meta)data, such as WCS information, exposure and response maps, masks, PSFs.

This section describes the expected input formats and the available options for loading and preprocessing data prior to model inference.

## Basic image loading

An image can be loaded by passing a FITS file (or an equivalent object) to `Image`:

```python
>>> from socca.data import Image
>>> img = Image("path/to/your/image.fits")
```

The input image can be alternatively provided as a filename pointing to a FITS file or a FITS `hdu` object. If a FITS file contains multiple HDUs, the first HDU is loaded by default. A different HDU can be selected using the `img_idx` keyword.

```{important}
Since **``socca``** relies on WCS information for coordinate transformations, the input image is expected to contain a header with valid WCS information.
```

## Defining the noise properties
Any `Image` instance must be associated with a noise model, passed via the `noise` argument. This is required to define the likelihood function used during inference. As shown in the "[Noise models](./tutorial_noise.md)" guide, not providing a noise model will default to using a `Normal` noise model with automatically estimated per-pixel standard deviation. Clearly, providing an explicit noise model tailored to the data at hand is strongly recommended. This can be done as follows:

```python
>>> from socca.data import Image
>>> from socca.noise import Normal
>>>
>>> noise = Normal(sigma=0.1)
>>>
>>> img = Image("path/to/your/image.fits", noise=noise)
``` 

## Linking a PSF model
If the observation has a non-negligible beam or point-spread function (PSF), it is important to provide a model for the PSF to the `Image` instance. This can be done by calling the `addpsf` method, which accepts either a FITS file, an `hdu` object, or a `numpy`/`jax.numpy` array containing the PSF model. For example:

```python
>>> img.addpsf("path/to/your/psf.fits")
```

As for the image data, if a FITS file contains multiple HDUs, the first HDU is loaded by default. A different HDU can be selected using the `idx` keyword.

```{important}
After loading, **``socca``** will automatically center the PSF model and normalize it to ensure that its integral sums to one. If the PSF image should not be normalized, it is possible to pass the `normalize=False` keyword when adding the PSF.
```

Alternatively, a PSF model can be provided during the initialization of the `Image` instance using the `addpsf` argument:

```python
>>> img = Image(
...     img="path/to/your/image.fits",
...     noise=noise,
...     addpsf=dict(img="path/to/your/psf.fits")
... )
```

```{warning}
The only requirement for the PSF model is that the corresponding input image has the same pixel scale of the main image, as **``socca``** does not perform any resampling of the PSF model to match the pixel scale of the input image. If the PSF model has a different size, though, it will be automatically padded or cropped to match the size of the input image. This is done to ensure that convolution operations can be performed efficiently in the Fourier domain.
```

## Adding a mask
If certain pixels in the input image should be excluded from the analysis, it is possible to provide a mask to the `Image` instance using the `addmask` method. The main input is a list of "regions" to be masked, which can be provided in various formats, including strings (for DS9 region files), `pyregion` objects, `numpy` arrays, or FITS HDUs. For example:

```python
>>> img.addmask(regions=[
...     "path/to/your/region.reg",
...     "path/to/another/region.fits"
... ])
```

or, equivalently:

```python
>>> img.addmask(regions=["path/to/your/region.reg"])
>>> img.addmask(regions=["path/to/another/region.fits"])
```

```{note}
Pixels with non-finite values (e.g., `NaN` or `Inf`) in the input image are automatically masked by **``socca``** upon loading.
```

By default, the new mask is combined with any existing mask. To reset the existing mask instead, the `combine=False` keyword can be used:

```python
>>> img.addmask(regions=["path/to/another/region.reg"], combine=False)
```

As with the PSF model, a mask can also be provided during the initialization of the `Image` instance using the `addmask` argument:

```python
>>> regions = [
...     "path/to/your/region.reg",
...     "path/to/another/region.fits"
... ]
>>>
>>> img = Image(
...     img="path/to/your/image.fits",
...     noise=noise,
...     addmask=dict(regions=regions),
... )
```

```{important}
In **``socca``**, the mask is interpreted in a boolean fashion, where pixels with a mask value of `1` are considered valid (i.e., included in the analysis), while pixels with a mask value of `0` are considered invalid (i.e., excluded from the analysis).
```

## Exposure and response maps
If the input image requires exposure or response corrections, these can be provided when initializing the `Image` instance using the `exposure` and `response` arguments, respectively. Conceptually, **``socca``** treats the response and exposure maps as multiplicative corrections to the model image, applied before and after PSF convolution, respectively.

Both the `exposure` and `response` arguments are loaded in the same way as the main image:

```python
>>> img = Image(
...     img="path/to/your/image.fits",
...     exposure="path/to/your/exposure/map.fits",
...     response="path/to/your/response/map.fits"
... )
```

```{warning}
The user should ensure that the exposure and response maps are compatible with the main image in terms of pixel scale, size, and WCS information, as **``socca``** does not perform any resampling or reprojecting of these maps.
```

## Cutout operation
**``socca``** provides the option to perform a cutout operation on the input image, allowing the user to run the model inference on a smaller region of interest. This can be done after the `Image` instance has been created, using the `cutout` method:

```python
>>> img.cutout(center=(150, 150), csize=100)
```
or during initialization using the `cutout` argument:

```python
>>> img = Image(
...     img="path/to/your/image.fits",
...     noise=noise,
...     center=(150, 150), 
...     csize=100
... )
```

Here, `center` specifies the center of the cutout region (in WCS coordinates) and can be provided as a tuple of pixel coordinates or as a [`SkyCoord` object](https://docs.astropy.org/en/stable/api/astropy.coordinates.SkyCoord.html). Instead, `csize` defines the size of the cutout (in pixels or angular units). The cutout operation will update the image data, WCS information, PSF, mask, exposure, and response maps accordingly. All the required operations are handled internally by **``socca``**, relying on the [`Cutout2D` method](https://docs.astropy.org/en/stable/api/astropy.nddata.Cutout2D.html) from [Astropy](https://www.astropy.org).
