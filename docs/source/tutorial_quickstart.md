# Getting started

This quickstart will walk you through a minimal end‑to‑end example of how to use **``socca``** to model an astronomical image. In this specific example, we will model a simulated toy image of a disk-like galaxy using a model comprising a Sersic component and a point source.

You can access all the files you need for running this tutorial, here:

Before getting into the nuts and bolts of using socca for modelling your favorite image, though, we need to load **``socca``**:

```{code-block} python
>>> import socca
```

Ready, set, go!

## Loading the data
As first step, let's build the data structure. The first element to define is the noise structure, in order to inform **``socca``** on the likelihood probability to use for the inference. The simplest noise model (and the one injected in the mock data used here) follows a normal probability with statistically independent resolution elements (i.e., without any pixel-to-pixel cross-correlation). This can be defined as follows:

```{code-block} python
>>> noise = socca.noise.Normal()

```
When not passing any argument, `Normal` will be tuned to infer the standard deviation of the normal distribution directly from the input image. This is based on the median absolute deviation of the pixel distribution. However, it is possible to manually specificy the standard deviation, variance, or inverse variance (i.e., weight) of the distribution, either in the form of constant values for the whole image or by passing a corresponding map (see ["Noise models"](./tutorial_noise.md) documentation page for more details).

We are now ready to import the image (and let **``socca``** take care of a lot of bookkeeping):
```
>>> img = socca.data.Image(img='tutorial_mock.fits',noise=noise)
Using MAD for estimating noise level
- noise level: 4.11E-06
```

As mentioned above, **``socca``** is using the image itself to obtain an estimate of the noise properties, measuring a root-mean-square level for the noise of 4.11x10^-06 (in image units).


## Defining a model
The next step consists in defining a model for the astronomical source(s) we want to fit. 
The basic building blocks of any model in **``socca``** are individual modular components, that can be combined linearly to describe the complex morphology of astronomical sources. 

### Point source
As first step, each component is defined independently and later combined into a single model. Let's start by defining a point-like component using the `Point` class from the `socca.models` module:

```{code-block} python
>>> point = socca.models.Point()
```

The `point` object now represents a point-like model component that can be used to describe unresolved sources. To inspect the parameters associated with this model, we can use the `parameters()` method:

```{code-block} python
>>> point.parameters()
xc   [deg] : None       | Right ascension
yc   [deg] : None       | Declination
Ic [image] : None       | Peak surface brightness
```

As shown above, the `Point` model is described by total of three parameters. The respective units are reported in the square brackets next to each name, while the description of each parameter is provided on the right. Here, the `None` values indicate that the parameters have not been assigned any prior or value yet. This can be done either by considering probability distributions or by fixing them to specific values:

```{code-block} python
>>> radius = 3.00E-03 # degrees
>>> point.xc = socca.priors.uniform(img.hdu.header['CRVAL1']-radius,
                                    img.hdu.header['CRVAL1']+radius)
>>> point.yc = socca.priors.uniform(img.hdu.header['CRVAL2']-radius,
                                    img.hdu.header['CRVAL2']+radius)
>>> point.Ic = socca.priors.loguniform(1.00E-08, 1.00E-02)
```

Here, we are setting uniform priors for the centroid coordinates, for convenience centred on the central reference coordinates of the input image. The characteristic intensity is instead assigned a loguniform prior, in order to allow the sampler to efficiently span different order of magnitudes. For a more extensive overview of the available prior distributions, please refer to the ["Priors and constraints"](./tutorial_priors.md) documentation page. For what concerns the point source, instead, we are done. Let's move on to the Sérsic component.

### Sérsic profile
Setting up a Sérsic (or any other) component is not any different than the point source case just shown.

```{code-block} python
>>> sersic = socca.models.Sersic()
>>> sersis.parameters()
xc    [deg] : None       | Right ascension of centroid
yc    [deg] : None       | Declination of centroid
theta [rad] : 0.0000E+00 | Position angle (east from north)
e        [] : 0.0000E+00 | Projected axis ratio
cbox     [] : 0.0000E+00 | Projected boxiness
re    [deg] : None       | Effective radius
Ie  [image] : None       | Surface brightness at re
ns       [] : 5.0000E-01 | Sersic index
```

When inspect the image, we can see that the diffuse component is centred on the position of the bright point source. We can thus fix the `point` and `sersic` model terms to share the same centroid. In such case, we can then bind the two components by fixing the `xc` and `yc` parameters of the `sersic` to be equal to and vary together with the ones of `point`:

```{code-block} python
>>> sersic.xc = socca.priors.boundto(point,'xc')
>>> sersic.yc = socca.priors.boundto(point,'yc')
```

In some cases, it can be conveniente to fix some parameters to specific values instead of sampling its posterior. In **``socca``**, this can be easily achieved by setting the parameter equal to a numeric value. For instance, let's force the Sérsic index to be equal to 1 (i.e., as for an exponential profile):

```{code-block} python
>>> sersic.ns = 1.00
```

We can then proceed with setting the priors for all the remaining parameters

```{code-block} python
>>> sersic.theta = socca.priors.uniform(low = 0.00, high = np.pi)
>>> sersic.e     = socca.priors.uniform(low = 0.00, high = 1.00)
>>> sersic.re    = socca.priors.loguniform(low = 1.00E-08, high = 1.00E-02)
>>> sersic.Ie    = socca.priors.loguniform(low = 1.00E-08, high = 1.00E-02)
```

Of course, since we are not setting any priors or value for `cbox`, this will remain fixed to the default value (`sersic.cbox=0`).

### Building a composite model
**``socca``** considers each component as an additive term. Composing a model is thus a straightforward task:

```{code-block} python
>>> mod = socca.models.Model()
>>> mod.addcomponent(point)
>>> mod.addcomponent(sersic)
```

The `mod` is now the general class that will take care of providing the correct model to the likelihood defined by `noise` for comparison with the input data.

## Your first fit with socca

We are not all set for running the modelling step. We first inizialite the fitter, then let it sample the posterior:

```{code-block} python
>>> fit = socca.fitter(mod=mod,img=img)
>>> fit.run()
Starting the nautilus sampler...
Finished  | 74     | 1        | 4        | 91200    | N/A    | 11106 | +174309.5
Elapsed time: 3.28 m
```

And that's all. By default, **``socca``** will use the `Nautilus` nested sampling library. More options are however available and fully integrated in **``socca``** (e.g., `dynesty`, `pocomc`, as well `scipy`'s optimizer; see ...). 

