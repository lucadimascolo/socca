# Priors and constraints

Before running the model inference, it is necessary to assign prior probabilities to the parameters of a **``socca``** model. Hereafter, we will assume that we have a model component `comp` and that we want to assign a prior to its parameter `p`. In **``socca``**, there are several ways to do this.

## Probability distributions

**``socca``** provides an interface to a limited number of prior distributions. These include:
- `uniform(low, high)`: uniform probability distribution over the range `[low, high]` (the limits must be finite)
- `loguniform(low, high)`: log-uniform probability distribution over the range `[low, high]`
- `powerlaw(alpha, low, high)`: power-law probability distribution $p(x) \propto x^\alpha$ over the range `[low, high]`, with `0 < low < high` and a finite `alpha`
- `normal(loc, scale)`: normal probability distribution with mean `loc` and standard deviation `scale`
- `splitnormal(loc, losig, hisig)`: split-normal probability distribution with mean `loc` and lower and upper standard deviations `losig` and `hisig`, respectively  

For instance, we can set a uniform prior over the range `[low, high]` as:

```python
>>> from socca.priors import uniform
>>> # ... define component `comp` ...
>>> comp.p = uniform(low=low, high=high)
```

All the probability functions listed above are simple wrappers around `numpyro` distributions, and it is therefore possible to use them directly when defining model priors. The example above would translate to:

```python
>>> import numpyro.distributions as dist
>>> # ... define component `comp` ...
>>> comp.p = dist.Uniform(low=low, high=high)
```

## Fixed parameters

In **``socca``**, fixing a parameter to a given value is as simple as assigning a numerical value to it, e.g.:

```python
>>> comp.p = 1.00
```

This will be automatically recognized by **``socca``** as a fixed parameter and excluded from the inference process.

## Tied parameters

In some cases, it may be useful to tie parameters together. One common example is a composite model comprising concentric components. Let us assume that we have two model components, `comp1` and `comp2`, and that we want to allow the centroid coordinates `xc` and `yc` to vary with a uniform probability within the ranges `[x_low, x_high]` and `[y_low, y_high]`, respectively. We can then tie the centroid coordinates of the two components as follows:

```python
>>> from socca.priors import uniform, boundto
>>> # ... define component `comp1` ...
>>> comp1.xc = uniform(low=x_low, high=x_high)
>>> comp1.yc = uniform(low=y_low, high=y_high)
>>> # ... define component `comp2` ...
>>> comp2.xc = boundto(comp1, 'xc')
>>> comp2.yc = boundto(comp1, 'yc')
```

At each step of the inference process, this instructs **``socca``** to use exactly the same values for `comp2.xc` and `comp2.yc` as those sampled for `comp1.xc` and `comp1.yc`. This approach can be used for any parameter pair, including parameters belonging to the same model component. `boundto` follows the component object, not its id, and copies the physical value, converting units if the two parameters have different declared units. On a `Disk`, an `Ellipsoid` or a bridge, a bare name is looked up in the sub-profiles (e.g. `boundto(disk, "xc")` is `disk.radial.xc`); a name matching more than one sub-profile raises an error.

## Functional priors

A final class of priors is represented by functional expressions. This is useful when there is a complex interdependence between parameters or, for instance, when one parameter can be expressed as a scaled version of another.

As an example, consider two {ref}`Sérsic components <sersic>`, with the effective radius of the first component constrained to be smaller than that of the second. This can be implemented with `function`, which computes a parameter from other parameters:

```python
>>> from socca.models import Sersic
>>> from socca.priors import uniform, loguniform, function
>>>
>>> comp1 = Sersic()
>>> comp2 = Sersic()
>>>
>>> # ... define priors for comp1 parameters ...
>>> comp1.addparameter('re_scale')
>>> comp1.re_scale = uniform(low=0.00, high=1.00)
>>> comp1.re = function(lambda s, r: s * r, (comp1, 're_scale'), (comp2, 're'))
>>>
>>> # ... define priors for comp2 parameters ...
>>> comp2.re = loguniform(low=1.00E-08, high=1.00E-02)
```

First, we introduce a new parameter `re_scale`, which represents the ratio between the effective radii of the two components. In **``socca``**, additional parameters can be added to a model component using the `addparameter` method. Once added, the new parameter behaves exactly like any pre-defined model parameter.

Next, `function(func, *sources)` defines `comp1.re` from the source parameters, given as `(component, name)` pairs and passed to `func` in that order. The sources can belong to any component of the model, including the one being defined, or to the sub-profiles of a `Disk`, an `Ellipsoid` or a bridge (e.g. `(disk.radial, 're')`, or `(disk, 're')`). The function takes and returns the parameters in their declared units (see `set_units`).

```{note}
Earlier versions defined such ties with a bare `lambda`, whose argument names are model keys, e.g. `lambda comp_00_re_scale, comp_01_re: ...` (with `_` in place of the `.` of sub-profile keys). This still works, with a deprecation warning, but the names depend on the component ids, which are only assigned when the components are added to a model (see "[Composite models](./tutorial_models.md)"): use `function` instead.
```

## Positive constraints
In some cases, it may be useful to enforce positivity constraints on the surface brightness of a given model component throughout the image domain. This is particularly relevant in cases like the {ref}`PolyExponential <polyexponential>` or {ref}`PolyExpoRefact <polyexporefact>` models, where the polynomial terms can lead to negative surface brightness values. Such constraints can be implemented by setting the `positive` attribute of the model component to `True`, e.g.,:

```python
>>> from socca.models import PolyExponential
>>> comp = PolyExponential()
>>> # ... assign priors to parameters ...
>>> comp.positive = True
```

or by passing the `positive=True` keyword when adding the component to the composite model:

```python
>>> from socca.models import Model
>>> # ... define model component ...
>>> model = Model()
>>> model.addcomponent(comp,positive=True)
```