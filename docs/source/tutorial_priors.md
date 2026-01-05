# Priors and constraints

Before running the model inference, it is necessary to assign prior probabilities to the parameters of a **``socca``** model. Hereafter, we will assume that we have a model component `comp` and that we want to assign a prior to its parameter `p`. In **``socca``**, there are several ways to do this.

## Probability distributions

**``socca``** provides an interface to a limited number of prior distributions. These include:
- `uniform(low, high)`: uniform probability distribution over the range `[low, high]`
- `loguniform(low, high)`: log-uniform probability distribution over the range `[low, high]`
- `normal(loc, scale)`: normal probability distribution with mean `loc` and standard deviation `scale`
- `splitnormal(loc, losig, hisig)`: split-normal probability distribution with mean `loc` and lower and upper standard deviations `losig` and `hisig`, respectively  

For instance, we can set a uniform prior over the range `[low, high]` as:

```python
>>> from socca.priors import uniform
...
>>> comp.p = uniform(low=low, high=high)
```

All the probability functions listed above are simple wrappers around `numpyro` distributions, and it is therefore possible to use them directly when defining model priors. The example above would translate to:

```python
>>> import numpyro.distributions as dist
...
>>> comp.p = dist.Uniform(low=low, high=high)
```

## Fixed parameters

In **``socca``**, fixing a parameter to a given value is as simple as assigning a numerical value to it, e.g.:

```python
>>> comp.p = 1.00
```

This will be automatically recognized by **``socca``** as a fixed parameter and excluded from the inference process.

## Bounded parameters

In some cases, it may be useful to bind parameters together. One common example is a composite model comprising concentric components. Let us assume that we have two model components, `comp1` and `comp2`, and that we want to allow the centroid coordinates `xc` and `yc` to vary with a uniform probability within the ranges `[x_low, x_high]` and `[y_low, y_high]`, respectively. We can then bind the centroid coordinates of the two components as follows:

```python
>>> from socca.priors import uniform, boundto
...
>>> comp1.xc = uniform(low=x_low, high=x_high)
>>> comp1.yc = uniform(low=y_low, high=y_high)
...
>>> comp2.xc = boundto(comp1, 'xc')
>>> comp2.yc = boundto(comp1, 'yc')
```

At each step of the inference process, this instructs **``socca``** to use exactly the same values for `comp2.xc` and `comp2.yc` as those sampled for `comp1.xc` and `comp1.yc`. This approach can be used for any parameter pair, including parameters belonging to the same model component.

## Functional priors

A final class of priors is represented by functional expressions. This is useful when there is a complex interdependence between parameters or, for instance, when one parameter can be expressed as a scaled version of another.

As an example, consider two Sérsic components, with the effective radius of the first component constrained to be smaller than that of the second. This can be implemented as follows:

```python
>>> from socca.models import Sersic
>>> from socca.priors import uniform, loguniform
>>> comp1 = Sersic()
>>> comp2 = Sersic()
...
>>> comp1.addpar('re_scale')
>>> comp1.re_scale = uniform(low=0.00, high=1.00)
>>> comp1.re = lambda comp_01_re_scale, comp_02_re: comp_01_re_scale * comp_02_re
...
>>> comp2.re = loguniform(low=1.00E-08, high=1.00E-02)
```

First, we introduce a new parameter `re_scale`, which represents the ratio between the effective radii of the two components. In **``socca``**, additional parameters can be added to a model component using the `addpar` method. Once added, the new parameter behaves exactly like any pre-defined model parameter.

Next, we define how `comp1.re` and `comp2.re` are related via a `lambda` function. The arguments of the function have composite names: the first part (`comp_XX`) denotes the component identifier, which can be obtained by printing the `id` attribute of the model component:

```python
>>> print(comp1.id, comp2.id)
comp_01, comp_02
```

The second part of each argument corresponds to the specific parameter used in the expression. Component IDs are assigned automatically by **``socca``** upon component creation, following an incremental numbering scheme.
