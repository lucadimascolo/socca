# Getting started

This quickstart will walk you through a minimal end-to-end example of how to use **``socca``** to model an astronomical image. In this specific example, we will model a simulated toy image of a disk-like galaxy using a model comprising a Sérsic component and a point source.

You can access all the files you need to run this tutorial here:

Before getting into the nuts and bolts of using **``socca``** to model your favourite image, we first need to load **``socca``**:

```python
>>> import socca
```

Ready, set, go!

## Loading the data

As a first step, we build the data structure. The first element to define is the noise model, which is used by **``socca``** to determine the likelihood function for the inference. The simplest noise model (and the one injected in the mock data used here) assumes a normal distribution with statistically independent resolution elements (i.e. no pixel-to-pixel correlations). This can be defined as follows:

```python
>>> noise = socca.noise.Normal()
```

When no arguments are passed, `Normal` is configured to infer the standard deviation of the noise distribution directly from the input image. This estimate is based on the median absolute deviation of the pixel values. It is also possible to manually specify the standard deviation, variance, or inverse variance (i.e. weight) of the distribution, either as constant values across the image or by providing a corresponding map (see the ["Noise models"](./tutorial_noise.md) documentation page for more details).

We are now ready to load the image (and let **``socca``** handle much of the associated bookkeeping):

```python
>>> img = socca.data.Image(img='tutorial_mock.fits', noise=noise)
Using MAD for estimating noise level
- noise level: 4.11E-06
```

As mentioned above, **``socca``** uses the image itself to estimate the noise properties, measuring a root-mean-square noise level of 4.11×10<sup>-6</sup> (in image units).

## Defining a model

The next step is to define a model for the astronomical source(s) we want to fit.  
The basic building blocks of any model in **``socca``** are individual modular components, which can be combined linearly to describe the complex morphology of astronomical sources.

### Point source

As a first step, each component is defined independently and later combined into a single model. We start by defining a point-like component using the `Point` class from the `socca.models` module:

```python
>>> point = socca.models.Point()
```

The `point` object now represents a point-like model component that can be used to describe unresolved sources. To inspect the parameters associated with this model, we can use the `parameters()` method:

```python
>>> point.parameters()
xc   [deg] : None       | Right ascension
yc   [deg] : None       | Declination
Ic [image] : None       | Peak surface brightness
```

As shown above, the `Point` model is described by a total of three parameters. The respective units are reported in square brackets next to each name, while a short description of each parameter is provided on the right. The `None` values indicate that no priors or fixed values have been assigned yet. These can be specified either by defining probability distributions or by fixing parameters to constant values:

```python
>>> radius = 3.00E-03  # degrees
>>> point.xc = socca.priors.uniform(low = img.hdu.header['CRVAL1'] - radius,
                                   high = img.hdu.header['CRVAL1'] + radius)
>>> point.yc = socca.priors.uniform(low = img.hdu.header['CRVAL2'] - radius,
                                   high = img.hdu.header['CRVAL2'] + radius)
>>> point.Ic = socca.priors.loguniform(low = 1.00E-08, high = 1.00E-02)
```

Here we assign uniform priors to the centroid coordinates, centred for convenience on the reference coordinates of the input image. The intensity parameter is instead assigned a log-uniform prior, allowing the sampler to efficiently explore several orders of magnitude. For a more extensive overview of the available prior distributions, see the ["Priors and constraints"](./tutorial_priors.md) documentation page.

For the point source component, we are now done. We can move on to the Sérsic component.

### Sérsic profile

Setting up a Sérsic (or any other) component follows the same procedure as for the point source:

```python
>>> sersic = socca.models.Sersic()
>>> sersic.parameters()
xc    [deg] : None       | Right ascension of centroid
yc    [deg] : None       | Declination of centroid
theta [rad] : 0.0000E+00 | Position angle (east from north)
e        [] : 0.0000E+00 | Projected axis ratio
cbox     [] : 0.0000E+00 | Projected boxiness
re    [deg] : None       | Effective radius
Ie  [image] : None       | Surface brightness at re
ns       [] : 5.0000E-01 | Sérsic index
```

Inspecting the image, we see that the diffuse component is centred on the bright point source. We therefore fix the `point` and `sersic` components to share the same centroid. This is achieved by binding the `xc` and `yc` parameters of the Sérsic component to those of the point source:

```python
>>> sersic.xc = socca.priors.boundto(point, 'xc')
>>> sersic.yc = socca.priors.boundto(point, 'yc')
```

In some cases, it is convenient to fix parameters to specific values rather than sampling their posterior distributions. In **``socca``**, this can be done simply by assigning a numerical value. For example, we fix the Sérsic index to 1 (corresponding to an exponential profile):

```python
>>> sersic.ns = 1.00
```

We then define priors for the remaining free parameters:

```python
>>> sersic.theta = socca.priors.uniform(low = 0.00, high = np.pi)
>>> sersic.e     = socca.priors.uniform(low = 0.00, high = 1.00)
>>> sersic.re    = socca.priors.loguniform(low = 1.00E-08, high = 1.00E-02)
>>> sersic.Ie    = socca.priors.loguniform(low = 1.00E-08, high = 1.00E-02)
```

Since no prior or value is assigned to `cbox`, it remains fixed at its default value (`sersic.cbox = 0`).

### Building a composite model

In **``socca``**, each component is treated as an additive term. Constructing a composite model is therefore straightforward:

```python
>>> mod = socca.models.Model()
>>> mod.addcomponent(point)
>>> mod.addcomponent(sersic)
```

The `mod` object now represents the full model and provides predictions to the likelihood defined by `noise` for comparison with the input data.

## Your first fit with socca

We are now ready to run the inference step. We first initialise the fitter and then sample the posterior distribution:

```python
>>> fit = socca.fitter(mod=mod, img=img)
>>> fit.run()
Starting the nautilus sampler...
Finished  | 74     | 1        | 4        | 91200    | N/A    | 11106 | +174309.5
Elapsed time: 3.28 m
```

By default, **``socca``** uses the `Nautilus` nested sampling library. Additional options are available and fully integrated, including `dynesty`, `pocomc`, and optimisers from `scipy` (see the documentation for details).
