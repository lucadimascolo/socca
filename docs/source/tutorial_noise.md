# Noise models

In **``socca``**, the likelihood function to be used during the inference process is fully specified by the noise model associated with the input data. This encodes the assumed statistical properties of the noise affecting the observations and therefore plays a central role in defining how model predictions are compared to data themselves. 
As shown in ["Getting started"](./tutorial_quickstart.md), any noise model should be linked to the input data by passing the noise object to the corresponding `socca.data.Image` instance:
```python
>>> img = Image(...,noise=noise)
```

At present, **``socca``** provides a limited but representative set of Gaussian noise models, covering the most common use cases encountered in astronomical imaging and map-based analyses. Below is a brief description of the noise models currently implemented in **``socca``**.

```{note}
Several noise models are under active development and testing, and will be added to **``socca``** in future releases. These include, among others, Poisson noise models for photon-counting data (e.g., X-ray observations), as well as noise models specifically designed for radio-interferometric observations.
```

```{important}
If the noise model is not explicitly specified when instantiating a `socca.data.Image` object, **``socca``** will use the `Normal` noise model by default and compute the per-pixel standard deviation from the median absolute deviation of all pixels in the input data that are not masked out.
```

## `Normal`: uncorrelated normal noise
The `Normal` noise model implements a multivariate normal likelihood, assuming that each spatial pixel in the input data is statistically independent from all others and affected by independent Gaussian noise with zero mean and known standard deviation. This is the simplest noise model available in **``socca``** and is appropriate when pixel-to-pixel correlations can be neglected and the noise variance is either uniform or known on a per-pixel basis.

The `Normal` noise model can be instantiated without any arguments. In such a case, the per-pixel standard deviation is automatically estimated from the median absolute deviation of all pixels in the input data, under the assumption that the noise is homogeneous across the image.

```python
>>> from socca.noise import Normal
>>> noise = Normal()
```

Alternatively, the noise amplitude can be specified explicitly using any of the following equivalent keyword arguments. Internally, all parametrizations are converted to a standard-deviation map.

| Argument   | Aliases                           | Meaning            | Relation |
|------------|-----------------------------------|--------------------|----------|
| `sigma`    | `sig`, `std`, `rms`, `stddev`     | Standard deviation | σ        |
| `variance` | `var`                             | Variance           | σ²       |
| `weight`   | `wht`, `wgt`, `weights`, `invvar` | Inverse variance   | 1 / σ²   |

These can be provided either as scalar values (assuming homogeneous noise levels) or as arrays with the same shape as the input data (allowing for spatially varying noise properties). If a string is provided for any of these arguments, it is interpreted as the path to a FITS file containing the corresponding maps. 

## `NormalCorrelated`: correlated normal noise
The `NormalCorrelated` noise model generalizes the uncorrelated Gaussian case by allowing for for spatial correlations between different pixels. The noise correlations are described explicitly through a full covariance matrix defined on the flattened data vector.

In this case, information about the covariance structure of the noise must always be provided explicitly when instantiating the model. This can be done by providing the full covariance matrix via the `cov` argument, or directly its inverse via the `icov` argument. In the former case, the inverse covariance matrix is computed internally using `jax.numpy.linalg.inv`. The inverse covariance matrix is used directly when evaluating the (normalized) likelihood.

```python
>>> from socca.noise import NormalCorrelated
>>>
>>> # From a covariance matrix
>>> noise_from_cov = NormalCorrelated(cov=covariance_matrix)
>>>
>>> # From an inverse covariance matrix
>>> noise_from_inv = NormalCorrelated(icov=inverse_covariance_matrix)
```

Both the `cov` and `icov` arguments should be provided as 2D `jax.numpy` arrays of shape `(nx*ny,nx*ny)`, where `(nx,ny)` is the shape of the input data. The two options are equivalent, and the choice is left to the user depending on which representation is more convenient to obtain. 

Alternatively, `NormalCorrelated` offers the possibility of estimating the noise covariance matrix internally from a set of noise realizations provided to **``socca``** via the `cube` argument.

```python
>>> from socca.noise import NormalCorrelated
>>> noise = NormalCorrelated(cube=noise_realizations)
```

Here, `noise_realizations` should be a 3D `jax.numpy` array of shape `(nr,nx,ny)`, where `nr` is the number of independent noise maps available. Once computed, the covariance matrix is smoothed using a simple convolutional kernel to reduce the impact of noise fluctuations. The amount of smoothing can be controlled via the `smooth` argument, denoting the number of iterations of the smoothing operation. By default, it is performed 3 times, using a simple 5-point stencil kernel. A custom smoothing kernel can also be provided via the `kernel` argument.

```{warning}
This noise model explicitly evaluates the likelihood using the full inverse covariance matrix and, therefore, is generally computationally more expensive than the uncorrelated `Normal` noise model.
```

## `NormalFourier`: uncorrelated Fourier-space normal noise
The `NormalFourier` noise model implements a Gaussian likelihood assuming that the noise is independent in Fourier space. Under this assumption, the noise covariance is diagonal in Fourier space and is defined directly on the Fourier coefficients of the data. This noise model is appropriate for approximately stationary noise processes, for which correlations in real space become simple in Fourier space. Typical use cases include map-based data originating from time-ordered or scan-based observations.

As in the case of `NormalCorrelated`, the noise properties can be specified either by providing the noise covariance in Fourier space via the `cov` argument, by providing its inverse via the `icov` argument, or by providing a set of noise realizations via the `cube` argument, from which the Fourier-space covariance is estimated internally. When a noise realization cube is provided, each realization is Fourier-transformed (optionally after apodization), and the covariance is computed as the mean squared modulus of the Fourier coefficients. The resulting covariance can optionally be smoothed before inversion.

```python
>>> from socca.noise import NormalFourier
>>>
>>> # From a Fourier-space covariance
>>> noise = NormalFourier(cov=fourier_covariance)
>>>
>>> # From an inverse Fourier-space covariance
>>> noise = NormalFourier(icov=inverse_fourier_covariance)
>>>
>>> # From a cube of noise realizations
>>> noise = NormalFourier(cube=noise_cube, ftype="real")
```

The optional argument `ftype` controls the type of Fourier transform used when estimating the covariance from the noise cube. By default, a real-to-complex 2D Fourier transform is used (`ftype="real"`), in order to minimize the memory usage and computational cost. If needed, it is possible to use a complex-to-complex 2D Fourier transform by specifying `ftype="complex"`. 

Other optional keyword arguments include `apod`, which specifies an apodization map applied to the data before Fourier transforming (though, by default, no apodization is applied). Finally, as in the case of `NormalCorrelated`, the `smooth` and `kernel` arguments control the optional smoothing of the Fourier-space covariance matrix when estimated from the noise realizations. As above, the default is 3 smoothing iterations using a 5-point stencil kernel.

```{warning}
The likelihood is evaluated by Fourier transforming the apodized residuals between model and data, weighting each Fourier mode by the inverse noise covariance, and summing over the Fourier modes with non-zero inverse covariance. Given the Fourier-space nature of this noise model, it is not possible to handle masked pixels in the input data.
```