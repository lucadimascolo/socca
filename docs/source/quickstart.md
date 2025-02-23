# Quickstart

## Defining a model component

As an example, let's assume we aim at fitting an elliptical Sérsic profile to our data. It will be enough to important the corresponding component and set the priors for its parameters as follows:

```{code-block} python
>>> import socca
>>> prof = socca.models.Sersic()
>>> prof.print_params()

xc    [deg] : None
yc    [deg] : None
theta [rad] : None
e        [] : None
cbox     [] : None
re    [deg] : None
Ie  [image] : None
ns       [] : None
```

As you can see from the output, the model is described by total of eight parameters, whose respective units are reported in square brackets. The `None` values indicate that the parameters have not been assigned any prior or value yet. This can be done either by considering probability distributions or by providing fixed values:

```{code-block} python
>>> prof.xc    = socca.priors.uniform(...)
>>> prof.yc    = socca.priors.uniform(...)
>>> prof.theta = socca.priors.uniform(...)
>>> prof.e     = socca.priors.uniform(...)
>>> prof.cbox  = 0.00
>>> prof.re    = socca.priors.loguniform(...)
>>> prof.Ie    = socca.priors.loguniform(...)
>>> prof.ns    = 0.50
```

In this specific example, we are considering uniform prior distributions for all parameters but the half-light radius and intensity parameters `re` and `Ie`, for which we assume loguniform priors. The boxiness parameter `cbox` and the Sérsic index `ns` are instead fixed to `0.00` and `0.50`, respectively. As shown in this example, this can be acheived by setting a given variable equal to a float instead than to a probability distribution. For more details on the available options for the parameter priors, please refer to the [dedicated section](./priors.md).

As an alternative, it will be possible to directly build the model component by providing the priors as keyword arguments:

```{code-block} python
prof = socca.models.Sersic(xc = socca.priors.uniform(...),
                           yc = socca.priors.uniform(...),
                           theta = socca.priors.uniform(...),
                           e = socca.priors.uniform(...),
                           cbox = 0.00,
                           re = socca.priors.loguniform(...),
                           Ie = socca.priors.loguniform(...),
                           ns = 0.50)
```

## Building the global model

```{code-block} python
model = socca.models.Model(prof=prof)
```
or, alternatively,

```{code-block} python
model = socca.models.Model()
model.add_component(prof)
```
The latter approach is particularly useful when considering a multi-term composable model (see the [Building a composite model](./models/link.md) page for details).