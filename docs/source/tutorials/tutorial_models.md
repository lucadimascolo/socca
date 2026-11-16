# Composite models


## Building a composite model
The model components introduced in "[Available model components](./tutorial_components.md)" (check it for a full description of every available components) can be used to build complex models by combining multiple components together:

```python
>>> from socca.models import Model
>>> model = Model()
>>> model.addcomponent(...)  # Add first component
>>> model.addcomponent(...)  # Add second component
>>> # ... add more components as needed ...
```

A more extensive example of how to build and fit composite models to image-space data can be found in the "[Getting started](./tutorial_quickstart.md)" tutorial.

## Obtaining images from models
Both the `Model` and individual component classes provide a `getmap()` method to generate two-dimensional model images evaluated on a specified coordinate grid.
This method requires an `Image` object (see the "[Loading data](./tutorial_data.md)" guide) to define the coordinate grid and, optionally, the PSF for convolution. Depending on whether it is called on an individual component or a composite model, the behavior of `getmap()` differs slightly.

### Individual components
For a single model component, the `getmap()` method returns the two-dimensional surface brightness distribution evaluated on the image grid:

```python
>>> from socca.models import Sersic
>>> from socca.data import Image
>>>
>>> # Load an image to define the coordinate grid
>>> img = Image('path/to/your/image.fits')
>>>
>>> # Create and configure a Sérsic component
>>> comp = Sersic()
>>> comp.xc = img.hdu.header['CRVAL1']
>>> comp.yc = img.hdu.header['CRVAL2']
>>> comp.re = 1.00E-03  # degrees
>>> comp.Ie = 1.00      # image units
>>> comp.ns = 1.00      # Sérsic index
>>>
>>> # Generate the model map
>>> model_map = comp.getmap(img)
```

By default, `getmap()` returns the unconvolved model. To obtain the PSF-convolved model (assuming a PSF has been provided when loading the image), set the `convolve` argument to `True`:

```python
>>> convolved_map = comp.getmap(img, convolve=True)
```

```{important}
All model parameters must be assigned numerical values before calling `getmap()`. If any parameter is set to `None` or contains a prior distribution instead of a fixed value, an error will be raised.
```

### Composite models
For composite models built using the `Model` class, the `getmap()` method works similarly to individual components when all parameters are assigned fixed numerical values:

```python
>>> model_map = model.getmap(img)                    # Unconvolved model
>>> model_map = model.getmap(img, convolve=True)     # PSF-convolved model
```

By default, the background component is not included in the output. To include it in the convolved model, set `addbackground=True`:

```python
>>> model_map = model.getmap(img, convolve=True, addbackground=True)
```

Clearly, since any **``socca``** composite model is a linear combination of multiple components, the resulting model image is equivalent to the sum of the individual component images (after convolution with the PSF, if requested):

```python
>>> # Create individual components
>>> comp1 = Sersic()
>>> comp1.xc, comp1.yc = 0.00, 0.00
>>> comp1.re, comp1.Ie, comp1.ns = 1.00E-03, 1.00E-01, 1.00
>>>
>>> comp2 = Exponential()
>>> comp2.xc, comp2.yc = 0.00, 0.00
>>> comp2.rs, comp2.Is = 2.00E-03, 5.00E-01
>>>
>>> # Build a composite model
>>> model = Model()
>>> model.addcomponent(comp1)
>>> model.addcomponent(comp2)
>>>
>>> # These two are equivalent:
>>> composite_map = model.getmap(img)
>>> summed_map = comp1.getmap(img) + comp2.getmap(img)
```

```{note}
If any parameter in the composite model is assigned a prior distribution rather than a fixed value, `getmap()` will raise an error. In such cases, use the `getmodel()` method instead, which accepts a parameter array as input.
```

### Accessing the inference model

During the fitting process, **``socca``** uses the `getmodel()` method internally, which requires a parameter array and returns multiple outputs:

```python
>>> model_raw, model_conv, model_background, \
...     mask_negative = model.getmodel(img, params)
```

where:
- `model_raw`: the raw (unconvolved) model, excluding background components
- `model_conv`: the smoothed (PSF-convolved) model, including all components
- `model_background`: the background component only
- `mask_negative`: a mask indicating pixels where positivity constraints were violated

The `params` array must contain the values of the free parameters (those with priors), in the order in which they appear in the model. This order follows how components were added to the model; the keys of all parameters are in `model.params`, and those of the free parameters in `[model.params[i] for i in model.paridx]` (the same as `fit.labels` for a fitter). The `parameters()` method of the `Model` object lists them by component:

```python
>>> from socca.models import Model, Sersic, Exponential
>>> import numpyro.distributions as dist
>>>
>>> # Build a composite model with priors
>>> comp1 = Sersic()
>>> comp1.xc, comp1.yc = 0.00, 0.00
>>> comp1.theta, comp1.e, comp1.cbox = 0.00, 0.00, 0.00
>>> comp1.re = dist.LogUniform(1.00E-04, 1.00E-02)
>>> comp1.Ie = dist.LogUniform(1.00E-02, 1.00E+02)
>>> comp1.ns = 1.00
>>>
>>> model = Model()
>>> model.addcomponent(comp1)
>>>
>>> model.parameters() # Print all parameters (fixed and free)

Sersic (comp_00)
================
xc    [deg] : 0.0000E+00                | Right ascension of centroid
yc    [deg] : 0.0000E+00                | Declination of centroid
theta [rad] : 0.0000E+00                | Position angle (east from north)
e        [] : 0.0000E+00                | Projected ellipticity (1 - axis ratio)
cbox     [] : 0.0000E+00                | Projected boxiness
re    [deg] : LogUniform [0.0001, 0.01] | Effective radius
Ie  [image] : LogUniform [0.01, 100]    | Surface brightness at re
ns       [] : 1.0000E+00                | Sersic index

>>> model.parameters(freeonly=True) # Print only free parameters (those with priors)

Sersic (comp_00)
================
re   [deg] : LogUniform [0.0001, 0.01] | Effective radius
Ie [image] : LogUniform [0.01, 100]    | Surface brightness at re

>>> model.params[5:7]
['comp_00_re', 'comp_00_Ie']
```

Each key follows the pattern `<component_id>_<parameter_name>`. For the sub-profiles of a `Disk`, an `Ellipsoid` or a bridge, the parameter name is dotted, e.g. `comp_00_radial.re` or `comp_00_vertical.zs`. This ordering is consistent throughout the inference process and must be respected when manually constructing parameter arrays for `getmodel()`.

In practice, when working with fitted models, the `fitter` object provides the more convenient `getmodel()` method (see the ["Running the model inference"](./tutorial_fitting.md) tutorial), which automatically uses the best-fit or median posterior parameters.

## Component ids and names

A component gets its id when it is added to a model: its `name`, if one was given at construction, or otherwise `comp_NN`, with the lowest number not yet used in that model (`comp_00`, `comp_01`, ...). A name must be a Python identifier; an integer or `comp_N` is normalized to `comp_NN`. Ids are unique within a model, and adding a component whose id is already used raises an error. A component that has not been added to a model has `id = None`.

```python
>>> halo = Exponential(name="halo", xc=0.00, yc=0.00, rs=2.00E-03, Is=0.50)
>>> model.addcomponent(halo)
>>> halo.id
'halo'
>>> model.params[-3:]
['halo_cbox', 'halo_rs', 'halo_Is']
```

Ids are also used to select components, e.g. in `fit.getmodel(component=...)`, which accepts ids, positions or the component objects themselves.

## Units

Parameters are stored in native units (degrees for positions and radii, radians for angles), but can be given in other units with `set_units`:

```python
>>> comp = Sersic()
>>> comp.set_units(re="arcsec", theta="deg")
>>> comp.re = 3.60  # arcsec
```

Priors, fixed values and ties are then expressed in the declared units, and periodic parameters have their period in those units (e.g. 180 for a position angle in degrees). On a `Disk`, an `Ellipsoid` or a bridge, the units of a sub-profile parameter can be declared on the sub-profile itself or on the wrapper with its dotted name, e.g. `disk.set_units(**{"radial.re": "arcsec"})`.

## Changing components after adding them

A model reads its components whenever it is used, so values, priors, ties, units and truncations changed after `addcomponent()` are taken into account. A fitter, instead, is built for the model as it was when the fitter was created: if the model changes afterwards, the fitter raises an error, and a new fitter must be created.
