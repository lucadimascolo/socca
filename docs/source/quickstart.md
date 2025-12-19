# Getting started

This quickstart will walk you through a minimal end‑to‑end example of how to use **``socca``** to model an astronomical image. In this specific example, we will model a simulated toy image of a disk-like galaxy using a model comprising a Sersic component and a point source.

## Loading the data
...

## Defining a model
The next step consists in defining a model for the astronomical source(s) we want to fit. 
The basic building blocks of any model in **``socca``** are individual modular components, that can be combined linearly to describe the complex morphology of astronomical sources. 

### Point source
As first step, each component is defined independently and later combined into a single model. Let's start by defining a point-like component using the `Point` class from the `socca.models` module:

```{code-block} python
>>> import socca
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
>>> prof.xc = socca.priors.uniform(...)
>>> prof.yc = socca.priors.uniform(...)
>>> prof.Ic = 1.00E-04
```

For a more detailed overview of the available prior distributions, please refer to the [priors documentation](./priors.md).

### Sérsic profile
...

### Building a composite model
