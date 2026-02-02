# Visualizing the results

**``socca``** provides built-in plotting utilities accessible via `fit.plot`. These are specifically intended for visualizing the results of the fitting process, and thus require a completed `fit` object (see "[Running the model inference](./tutorial_fitting.md)").

## Comparison plot

Generate a side-by-side comparison of data, model, and residuals using [APLpy](https://aplpy.github.io/):

```python
>>> fit.plot.comparison(name='comparison_figure')
```

The `comparison()` method accepts several customization options:

| Argument | Description |
|----------|-------------|
| `name` | Output filename (without extension). If `None`, displays interactively |
| `component` | Model component(s) to include (see below) |
| `fmt` | Output format (default: `'pdf'`) |
| `fx`, `fy` | Figure scaling factors for width and height |
| `dpi` | Output resolution in dots per inch |
| `cmaps` | Colormap specification (see below) |
| `cmap_data` | Colormap for the data panel (default: `'magma'`) |
| `cmap_model` | Colormap for the model panel (default: `'magma'`) |
| `cmap_residual` | Colormap for the residuals panel (default: `'RdBu_r'`) |
| `gs_kwargs` | Dictionary of gridspec options (spacing, margins) |
| `model_kwargs` | Additional arguments passed to `fit.getmodel()` |

The `component` argument allows comparing the data against specific model components only:
- `None` (default): Include all model components
- Integer: Single component index (e.g., `component=0` for the first component)
- List of integers: Multiple component indices (e.g., `component=[0, 2]`)
- String: Component name (e.g., `component='comp_00'`)
- Component object: The component instance itself

```python
>>> # Compare data against only the first component
>>> fit.plot.comparison(name='comparison_comp0', component=0)
```

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

## Corner plot

Visualize the posterior distributions using the [corner](https://corner.readthedocs.io/) library:

```python
>>> fit.plot.corner(name='corner_figure')
```

The `corner()` method provides several options for customizing the output:

| Argument | Description |
|----------|-------------|
| `name` | Output filename (without extension). If `None`, displays interactively |
| `fmt` | Output format (default: `'pdf'`) |
| `component` | Components to include (see below) |
| `sigma` | Range in sigma units around the median for axis limits (default: `10.0`) |
| `edges` | Custom axis ranges as array of `[min, max]` pairs |
| `quantiles` | Quantiles to display on 1D histograms (default: `[0.16, 0.50, 0.84]`) |
| `bins` | Number of histogram bins (default: `40`) |
| `truths` | Array of true values to overplot (if known) |

The `sigma` argument sets the axis limits based on the specified number of standard deviations around the median of each parameter. If set to `None`, the limits are set automatically to encompass the full range of the samples. If `edges` is provided, it overrides the `sigma` setting with explicit min/max ranges for each parameter.

The `component` argument allows selecting which model components to include in the corner plot:
- `None` (default): Include all components
- Integer: Single component index (e.g., `component=0` for the first component)
- List of integers: Multiple component indices (e.g., `component=[0, 2]`)
- String: Component name (e.g., `component='comp_00'`)
- Component object: The component instance itself

```python
>>> # Corner plot for only the first component
>>> fit.plot.corner(name='corner_comp0', component=[0])
>>>
>>> # Corner plot for specific components
>>> fit.plot.corner(name='corner_selected', component=[0, 2], sigma=5.0)
>>>
>>> # With known true values for validation
>>> fit.plot.corner(name='corner_truth', truths=[0.0, 0.0, 1e-3, 0.5, 1.0])
```

Any additional keyword arguments are passed directly to the underlying `corner.corner()` function, allowing full customization of the plot appearance.

See the "[Getting started](./tutorial_quickstart.md)" guide for example outputs of these plotting functions.
