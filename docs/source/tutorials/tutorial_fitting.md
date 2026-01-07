# Running the model inference

Once a model has been defined and priors have been assigned to all parameters, **``socca``** provides a unified interface for running the inference. This section describes the available sampling methods, their configuration options, and how to analyze and save the results.

## Initializing the fitter

Initializing and running the model inference is as simple as creating a `fitter` object and calling its `run()` method. Assuming that we have already defined a composite model `mod` and have an image `img` to fit, the inference can be started as follows:

```python
>>> from socca import fitter
>>> from socca.data import Image
>>> from socca.models import Model
>>> 
>>> mod = Model()
>>> #  ... add components to the model and set priors ...
>>>
>>> img = Image.load('path/to/your/image.fits')
>>> #  ... set up the image (PSF, noise model, mask, etc.) ...
>>>
>>> fit = fitter(mod=mod, img=img)
>>> fit.run()
```

The `fitter` object handles all the necessary bookkeeping, including the construction of the likelihood function based on the noise model and the mapping between the prior distributions and the model parameters.

```{caution}
An additional sampling backend based on [`numpyro`](https://num.pyro.ai/) NUTS ([No-U-Turn Sampler](https://num.pyro.ai/en/latest/mcmc.html#numpyro.infer.hmc.NUTS)) is included in the codebase but has not been thoroughly tested. It is left available for experimental use and development purposes only. Users interested in trying the `numpyro` backend should be aware that its behavior and results may not be fully validated.
```

## Available sampling methods

**``socca``** supports multiple backends for performing the model inference via Bayesian posterior sampling, which can be selected via the `method` argument when calling the `run()` method. The available options are:

| Method        | Description | Link |
|:-------------:|-------------|:----:|
| `'nautilus'`<br>(default) | Neural network-accelerated nested sampling | [↗](https://github.com/johannesulf/nautilus) |
| `'dynesty'`   | Dynamic nested sampling | [↗](https://github.com/joshspeagle/dynesty) |
| `'pocomc'`    | Preconditioned Monte Carlo sampling | [↗](https://github.com/minaskar/pocomc) |

Any of the options above can be selected by passing the corresponding string to the `method` argument of the `run()` method. To customize the behavior of each sampling method, any of the keyword arguments accepted by the respective sampler class can be passed to the `run()` method. These are automatically forwarded to the underlying sampler implementation. For a complete list of available options for each method, please refer to the respective documentation linked in the table above.
For instance, to use the `dynesty` sampler with 500 live points, one would do:

```python 
>>> fit.run(method='dynesty', nlive=500)
```

After the sampling is completed, the key results are stored in the following attributes of the `fitter` object:

| Attribute     | Description |
|---------------|-------------|
| `fit.samples` | Array of posterior samples with shape `(n_samples, n_params)` |
| `fit.weights` | Importance weights for each sample (normalized) |
| `fit.logz`    | Log-evidence estimate |

```{note}
The `fit.sampler` attribute provides direct access to the underlying sampler object (e.g., `nautilus.Sampler`, `dynesty.NestedSampler`, or `pocomc.Sampler`), which can be useful for accessing additional diagnostics or methods specific to each sampling library.
```

### Maximum a posteriori estimation

For fast point estimates, it is also possible to run a maximum a posteriori (MAP) optimization using the `'optimizer'` method, that uses the L-BFGS-B algorithm from `scipy.optimize.minimize`:

```python
>>> fit.run(method='optimizer')
```

```{important}
The optimization is designed to include the effect of priors on the posterior distribution. To do so, as in nested sampling, parameters are drawn from the unit hypercube and then projected onto the prior space using the inverse cumulative distribution function of the respective prior. The resulting parameters are then used to compute the prior-conditioned log-likelihood to be maximized.
```

The key argument for this method is `pinits`, which controls the starting point of the optimization:
| Option | Description |
|:------:|-------------|
| `'median'` (default) | Start from the 50th percentile of all prior distributions (i.e., the midpoint of the unit hypercube) |
| `'random'` | Start from a random point in the unit prior hypercube |
| array-like | A custom array of initial values with one value per free parameter |

As for the other sampling methods, additional keyword arguments accepted by `scipy.optimize.minimize` (such as `tol`, `options`, etc.) can be passed directly to the `run()` method and will be forwarded to the optimizer.

The optimization results are stored in `fit.results`, which is a `scipy.optimize.OptimizeResult` object providing access to the optimal parameters (`fit.results.x`), the final log-likelihood value (`-fit.results.fun`), and convergence information (`fit.results.success`, `fit.results.message`).

## Accessing the best-fit model

After running the inference, the `fitter` object provides a convenient `getmodel()` method to generate model images using the best-fit or median posterior parameters:

```python
>>> # Get all model components (raw, smoothed, background)
>>> model_raw, model_smoothed, model_background = fit.getmodel()
>>>
>>> # Get only the PSF-convolved model (excluding background)
>>> model_smoothed = fit.getmodel(what='smoothed')
```

The `getmodel()` method accepts the following arguments:

| Argument | Description |
|----------|-------------|
| `what` | Which model component(s) to return (see below) |
| `usebest` | If `True` (default), use median posterior parameters; if `False`, compute median of model realizations |
| `img` | Alternative `Image` object to evaluate the model on (default: use the fitted image) |
| `doresp` | Apply response/exposure correction (default: `False`) |
| `doexp` | Apply exposure map weighting (default: `False`) |

The `what` argument controls the output:

| Option | Description |
|:------:|-------------|
| `'all'` (default) | Returns a tuple of `(raw, smoothed, background)` |
| `'raw'` | Unconvolved model (excluding background) |
| `'smoothed'`<br>`'conv'`<br>`'convolved'` | PSF-convolved model (excluding background) |
| `'background'`<br>`'bkg'` | Background component only |

Multiple components can be requested as a list:

```python
>>> model_raw, model_background = fit.getmodel(what=['raw', 'background'])
```

When `usebest=True` (default), the model is computed using the weighted median of the posterior samples for each parameter. When `usebest=False`, the method instead computes the model for each posterior sample individually and returns the median of the resulting model realizations. The latter approach can be useful for capturing non-linear effects in the model, but is significantly slower.

For the optimizer method, `getmodel()` uses the optimal parameters found during optimization (`fit.results.x`), regardless of the `usebest` setting.

## Checkpointing and resuming

All sampling methods support saving the sampler state during the run, allowing interrupted runs to be resumed:

```python
>>> fit.run(checkpoint='my_fit_checkpoint')
```

By default, if a checkpoint file matching the provided name is found, **``socca``** will resume sampling from the last saved state. To force a fresh run:

```python
>>> fit.run(checkpoint='my_fit_checkpoint', resume=False)
```

```{note}
The checkpoint file format and location depend on the sampling backend. For `nautilus`, the sampler state is saved synchronously. For `dynesty` and `pocomc`, states are saved at regular intervals.
```

## Bayesian model comparison

For nested sampling methods, **``socca``** can compute Bayesian model comparison statistics:

```python
>>> fit.run(getzprior=True)  # Enable prior evidence estimation
>>> ln_BayesFac_raw, sigma_eff_raw, \
... ln_BayesFac_deboost, sigma_eff_deboost = fit.bmc()

null-model comparison
=====================
ln(Bayes factor) :  1.743E+05
effective sigma  :  5.907E+02

prior deboosted
=====================
ln(Bayes factor) :  1.725E+05
effective sigma  :  5.876E+02
```

The `getzprior=True` option runs an additional sampling step to estimate the prior evidence, which is used to compute the prior-corrected Bayes factor. The output includes:
- `ln_BayesFact_raw`: Raw log Bayes factor (vs. null model)
- `effect_sigma_raw`: Effective sigma significance
- `ln_BayesFact_deboost`: Prior-deboosted log Bayes factor
- `effect_sigma_deboost`: Prior-deboosted effective sigma


## Saving and loading results

To save the complete fitting results for later analysis:

```python
>>> fit.dump('my_fit_results.pickle')
```

This serializes the entire `fit` object, including the model, data, and all sampling results, using `dill`. To reload:

```python
>>> fit = socca.load('my_fit_results.pickle')
```

## Visualization

Finally, **``socca``** provides built-in plotting utilities accessible via `fit.plot`.

### Comparison plot

Generate a side-by-side comparison of data, model, and residuals using [APLpy](https://aplpy.github.io/):

```python
>>> fit.plot.comparison(name='comparison_figure')
```

The `comparison()` method accepts several customization options:

| Argument | Description |
|----------|-------------|
| `name` | Output filename (without extension). If `None`, displays interactively |
| `fmt` | Output format (default: `'pdf'`) |
| `fx`, `fy` | Figure scaling factors for width and height |
| `dpi` | Output resolution in dots per inch |
| `cmaps` | Colormap specification (see below) |
| `cmap_data` | Colormap for the data panel (default: `'magma'`) |
| `cmap_model` | Colormap for the model panel (default: `'magma'`) |
| `cmap_residual` | Colormap for the residuals panel (default: `'RdBu_r'`) |
| `gs_kwargs` | Dictionary of gridspec options (spacing, margins) |
| `model_kwargs` | Additional arguments passed to `fit.getmodel()` |

Colormaps can be specified in multiple ways via the `cmaps` argument:
- A single colormap name applied to all panels
- A list/tuple of three colormaps `[data_cmap, model_cmap, residual_cmap]`
- A dictionary with keys `'data'`, `'model'`, and `'residuals'`

```python
>>> # Custom colormaps
>>> fit.plot.comparison(name='figure', cmaps={'data': 'viridis', 
...                                           'model': 'viridis',
...                                           'residuals': 'coolwarm'})
```

### Corner plot

Visualize the posterior distributions using the [corner](https://corner.readthedocs.io/) library:

```python
>>> fit.plot.corner(name='corner_figure')
```

The `corner()` method provides several options for customizing the output:

| Argument | Description |
|----------|-------------|
| `name` | Output filename (without extension). If `None`, displays interactively |
| `fmt` | Output format (default: `'pdf'`) |
| `comp` | Components to include (see below) |
| `sigma` | Range in sigma units around the median for axis limits (default: `10.0`) |
| `edges` | Custom axis ranges as array of `[min, max]` pairs |
| `quantiles` | Quantiles to display on 1D histograms (default: `[0.16, 0.50, 0.84]`) |
| `bins` | Number of histogram bins (default: `40`) |
| `truths` | Array of true values to overplot (if known) |

The `comp` argument allows selecting which model components to include in the corner plot:
- `None` (default): Include all components
- Integer: Single component index (e.g., `comp=0` for the first component)
- List of integers: Multiple component indices (e.g., `comp=[0, 2]`)
- String: Component name (e.g., `comp='comp_00'`)
- Component object: The component instance itself

```python
>>> # Corner plot for only the first component
>>> fit.plot.corner(name='corner_comp0', comp=[0])
>>>
>>> # Corner plot for specific components
>>> fit.plot.corner(name='corner_selected', comp=[0, 2], sigma=5.0)
>>>
>>> # With known true values for validation
>>> fit.plot.corner(name='corner_truth', truths=[0.0, 0.0, 1e-3, 0.5, 1.0])
```

Any additional keyword arguments are passed directly to the underlying `corner.corner()` function, allowing full customization of the plot appearance.

See the ["Getting started"](./tutorial_quickstart.md) guide for example outputs of these plotting functions.
