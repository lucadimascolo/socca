# Priors and constraints

Before running the model inference, it is necessary to assign prior probabilities to any of the parameters of the **``socca``** model. Hereafter, we will assume to have a model component `comp` and that we want to set its parameter `p` to have a given prior. In **``socca``**, there are several options.

## Probability distributions
**``socca``** integrates the interface to a limited number of prior distributions. These include:
- `uniform(low, high)`: uniform probability distribution over the range `[low,high]`
- `loguniform(low, high)`: log-uniform probability distribution over the range `[low,high]`
- `normal(low, high)`: normal probability distribution with mean `loc` and standard deviation `scale`
- `splitnormal(loc, losig, hisig)`: split-normal probability distribution, with mean `loc` and lower and upper standard deviation `losig` and `hisig`, respectively.  

For instance, we can set a uniform prior over the range `[low,high]` as:

```python
>>> from socca.priors import uniform
...
>>> comp.p = uniform(low=low, high=high)
```

However, all the probability functions listed above are simple wrappers around `numpyro` distributions, and it is possible to directly use them for defining the model priors. The case shown above would translate to:

```python
>>> import numpyro.distributions as dist
...
>>> comp.p = dist.Uniform(low=low, high=high)
```

## Fixed parameters
In **``socca``**, fixing a parameter to a given value is as easy as setting the relevant parameter equal to a numerical value, e.g.:

```python
comp.p = 1.00
```

This will be automatically recognized by **``socca``** as a fixed parameter, and will be excluded from the inference process.

## Bounded parameters
In some cases, it might be useful to bind different parameters together. One common example is the case of a composite model, comprising concentric component. Let's assume to have two model components `comp1` and `comp2`, and to allow the centroid coordinates `xc` and `yc` to vary with uniform probability within the ranges `[x_low,x_high]` and `[y_low,y_high]`, respectively. We can then bind centroid coordinates of the two components as follows:

```python
>>> from socca.priors import uniform, boundto
...
>>> comp1.xc = uniform(low=x_low, high=x_high)
>>> comp1.yc = uniform(low=y_low, high=y_high)
...
>>> comp2.xc = boundto(comp1,'xc')
>>> comp1.yc = boundto(comp1,'yc')
```

At each step of the model inference, this will tell **``socca``** to use for `comp2.xc` and `comp2.yc` exactly the same values as `comp1.xc` and `comp1.yc`. Clearly, this approach can be used with any parameter pair, as well as over parameters from the same model component.

## Functional priors
A final class of priors is represented by functional expressions. This is useful in the case of a complex interdependence of different parameters or, for instance, when two parameters can be expressed as the scaled version of each other. Let's consider the latter example as an example and, for instance, consider two Sérsic components, with the first one having a effective radius smaller than the second component. This can be implemented as follows:

```python
>>> from socca.models import Sersic
>>> from socca.priors import uniform, loguniform
>>> comp1 = Sersic()
>>> comp2 = Sersic()
...
>>> comp1.addpar('re_scale')
>>> comp1.re_scale = uniform(low=0.00, high=1.00)
>>> comp1.re = lambda comp_01_re_scale, comp_02_re: comp_01_re_scale*comp_02_re
...
>>> comp2.re = loguniform(low=1.00E-08, high=1.00E-02)
```

First of all, we had to introduce a new parameter `re_scale`, denoting the ratio between the effective radii of the first and second component. As shown above, in **``socca``** it is possible to introduce any additional parameter to a given model component via the function `addpar`. The new parameter will then behave exactly as any other pre-included parameter.
Second, we have to define how the two effective radii `comp1.re` and `comp2.re` are related to each other. This is done via a `lambda` function. The arguments have composite names: the first part (`comp_XX`) denotes the id of the component, that can be obtained simply by printing out the `id` attribute of a given model term;

```python
>>> print(comp1.id,comp2.id)
comp_01, comp_02
```

The second part corresponds to the specific variable one wants to use in the expression. The `id` is assigned automatically by **``socca``**  to a given component as soon as this is created, using an incremental numbering system.