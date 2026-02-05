"""Visualization utilities for model fitting results."""

import matplotlib.pyplot as plt
import matplotlib.colorbar
import matplotlib

import numpy as np
import os
import warnings

import corner

from astropy.wcs.utils import wcs_to_celestial_frame
from astropy.coordinates import (
    ICRS,
    FK5,
    FK4,
    Galactic,
    HeliocentricTrueEcliptic,
    BarycentricTrueEcliptic,
    BaseEclipticFrame,
)
from astropy.visualization import ImageNormalize

try:
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = ["Helvetica"]
except Exception:
    pass


# Get axis labels from WCS frame
# Adapted from aplpy
# --------------------------------------------------------
def getframe(wcs):
    """
    Get axis labels from a WCS object based on its celestial frame.

    Parameters
    ----------
    wcs : astropy.wcs.WCS
        WCS object to extract frame information from.

    Returns
    -------
    xtext : str
        Label for the x-axis (e.g., 'RA (ICRS)', 'Galactic Longitude').
    ytext : str
        Label for the y-axis (e.g., 'Dec (ICRS)', 'Galactic Latitude').
    """
    frame = wcs_to_celestial_frame(wcs)
    if isinstance(frame, ICRS):
        xtext = "RA (ICRS)"
        ytext = "Dec (ICRS)"
    elif isinstance(frame, FK5):
        equinox = "{:g}".format(frame.equinox.jyear)
        xtext = "RA (J{0})".format(equinox)
        ytext = "Dec (J{0})".format(equinox)
    elif isinstance(frame, FK4):
        equinox = "{:g}".format(frame.equinox.byear)
        xtext = "RA (B{0})".format(equinox)
        ytext = "Dec (B{0})".format(equinox)
    elif isinstance(frame, Galactic):
        xtext = "Galactic Longitude"
        ytext = "Galactic Latitude"

    elif isinstance(
        frame,
        (HeliocentricTrueEcliptic, BarycentricTrueEcliptic, BaseEclipticFrame),
    ):
        xtext = "Ecliptic Longitude"
        ytext = "Ecliptic Latitude"
    else:
        cunit_x = wcs.wcs.cunit[0]
        cunit_y = wcs.wcs.cunit[1]

        cname_x = wcs.wcs.cname[0]
        cname_y = wcs.wcs.cname[1]

        ctype_x = wcs.wcs.ctype[0]
        ctype_y = wcs.wcs.ctype[1]

        xunit = " (%s)" % cunit_x if cunit_x not in ["", None] else ""
        yunit = " (%s)" % cunit_y if cunit_y not in ["", None] else ""

        if len(cname_x) > 0:
            xtext = cname_x + xunit
        else:
            if len(ctype_x) == 8 and ctype_x[4] == "-":
                xtext = ctype_x[:4].replace("-", "") + xunit
            else:
                xtext = ctype_x + xunit

        if len(cname_y) > 0:
            ytext = cname_y + yunit
        else:
            if len(ctype_y) == 8 and ctype_y[4] == "-":
                ytext = ctype_y[:4].replace("-", "") + yunit
            else:
                ytext = ctype_y + yunit

    return xtext, ytext


# Plotting utilities
# ========================================================
class Plotter:
    """
    Visualization interface for model fitting results.

    Provides methods for plotting posterior samples, corner plots, model
    images, residuals, and other diagnostic visualizations.
    """

    def __init__(self, fit):
        """
        Initialize the Plotter with a reference to a fitter object.

        Parameters
        ----------
        fit : fitter
            Fitter object containing the model, data, samples, and results
            to be plotted.
        """
        self.fit = fit

    # Corner plot of the posterior samples
    # --------------------------------------------------------
    def corner(
        self,
        name=None,
        component=None,
        fmt="pdf",
        sigma=10.00,
        edges=None,
        quantiles=[0.16, 0.50, 0.84],
        bins=40,
        **kwargs,
    ):
        """
        Create corner plots showing posterior distributions and correlations.

        Generates a corner (triangle) plot displaying 1D and 2D marginalized
        posterior distributions for model parameters. Plots can be customized
        to show specific components and parameter ranges.

        Parameters
        ----------
        name : str, optional
            Output filename (without extension). If None, displays plot
            interactively instead of saving. Default is None.
        component : str, int, list, or None, optional
            Component(s) to plot. Can be:

            - None: Plot all components (default)
            - str: Single component name (e.g., 'comp_00')
            - int: Component index
            - list: Multiple components (names, indices, or objects)
        fmt : str, optional
            Output file format (e.g., 'pdf', 'png'). Default is 'pdf'.
        sigma : float, optional
            Number of standard deviations to use for automatic axis limits.
            If None, uses full parameter ranges. Default is 10.0.
        edges : array_like or None, optional
            Explicit axis limits as [[xmin, xmax], ...] for each parameter.
            If None, computed automatically from `sigma`. Default is None.
        quantiles : list, optional
            Quantiles to display on 1D histograms. Default is [0.16, 0.5, 0.84]
            (median and 1-sigma intervals).
        bins : int, optional
            Number of bins for histograms. Default is 40.
        **kwargs : dict, optional
            Additional keyword arguments passed to corner.corner().
            See corner package documentation for available options.

        Notes
        -----
        - Automatically handles parameter units in axis labels
        - Uses weighted samples if available from nested sampling
        - Supports truths parameter via kwargs to show true parameter values
        """
        component = self.fit.mod._comp_filter(component)
        component = [f"comp_{ci:02d}" for ci in component]

        if edges is None:
            if sigma is None:
                edges = None
            else:
                edges = []
                for s in self.fit.samples.T:
                    q = corner.quantile(
                        s, [0.16, 0.50, 0.84], weights=self.fit.weights
                    )
                    qmin = np.maximum(s.min(), q[1] - sigma * (q[1] - q[0]))
                    qmax = np.minimum(s.max(), q[1] + sigma * (q[2] - q[1]))
                    edges.append([qmin, qmax])
                    del q, qmin, qmax
                edges = np.array(edges)

        labels = []
        for li, label in enumerate(self.fit.labels):
            if self.fit.units[li] is not None and len(self.fit.units[li]) > 0:
                labels.append(f"{label}\n[{self.fit.units[li]}]")
            else:
                labels.append(f"{label}\n")
        labels = np.array(labels)

        indices = np.array(
            [
                i
                for i, lbl in enumerate(labels)
                if any(lbl.startswith(c) for c in component)
            ],
            dtype=int,
        )

        if len(indices) == 0:
            raise ValueError(
                f"No parameters found for component(s) {component}. "
                f"Check that the component index exists."
            )

        truths = kwargs.pop("truths", None)
        if truths is not None:
            truths = np.array(truths)[indices]

        corner.corner(
            data=self.fit.samples[:, indices],
            weights=self.fit.weights,
            labels=labels[indices],
            range=edges[indices] if edges is not None else None,
            quantiles=quantiles,
            truths=truths,
            bins=bins,
            **kwargs,
        )

        if name is None:
            plt.show()
        else:
            root, ext = os.path.splitext(name)
            if ext.lower() == f".{fmt.lower()}":
                name = root
            plt.savefig(f"{name}.{fmt}", format=fmt, dpi=300)
        plt.close()

    # Comparison plot: data, model, residuals
    # --------------------------------------------------------
    def comparison(
        self,
        name=None,
        component=None,
        fmt="pdf",
        fx=1.00,
        fy=0.38,
        dpi=72.27 * 390.00 / 504.00,
        cmaps=None,
        cmap_data=None,
        cmap_model=None,
        cmap_residual=None,
        gs_kwargs=None,
        model_kwargs=None,
    ):
        """
        Create a three-panel comparison plot: data, model, and residuals.

        Generates a publication-quality figure showing the observed data,
        best-fit model, and residuals (data - model) side by side with
        appropriate colormaps and colorbars.

        Parameters
        ----------
        name : str, optional
            Output filename (with or without extension). Required for saving;
            if None, plot is saved with a default name.
        component : None, str, int, list, or Profile, optional
            Model component(s) to include in the comparison. Can be:

            - None: Include all model components (default)
            - str: Single component name (e.g., 'comp_00')
            - int: Component index (e.g., 0 for the first component)
            - list: Multiple components as names, indices, or Profile objects
            - Profile: Object with `id` attribute specifying the component

            This is useful for comparing data against individual model
            components. Default is None (all components).
        fmt : str, optional
            Output file format (e.g., 'pdf', 'png'). Default is 'pdf'.
        fx : float, optional
            Figure width scaling factor. Default is 1.0.
        fy : float, optional
            Figure height scaling factor. Default is 0.38.
        dpi : float, optional
            Dots per inch for output resolution. Default is ~55.97.
        cmaps : str, list, or dict, optional
            Colormap specification. Can be:

            - str: Single colormap applied to all panels
            - list: [data_cmap, model_cmap, residual_cmap]
            - dict: {'data': cmap, 'model': cmap, 'residuals': cmap}

            Defaults to ['magma', 'magma', 'RdBu_r'].
        cmap_data : str, optional
            Colormap for data panel (overrides cmaps). Default is 'magma'.
        cmap_model : str, optional
            Colormap for model panel (overrides cmaps). Default is 'magma'.
        cmap_residual : str, optional
            Colormap for residual panel (overrides cmaps). Default is 'RdBu_r'.
        gs_kwargs : dict, optional
            Keyword arguments for GridSpec layout (hspace, wspace, margins).
        model_kwargs : dict, optional
            Keyword arguments passed to getmodel() for model generation.
            The 'what' argument is automatically set to 'smoothed' and ignored
            if provided.

        Notes
        -----
        - Data and model panels share the same color scale
        - Residual panel uses a symmetric diverging colormap centered at 0
        - Uses APLpy for WCS-aware plotting
        - Automatically includes colorbars for data and residual panels
        - Output is saved at 300 DPI regardless of input dpi (used for sizing)

        Examples
        --------
        >>> # Standard comparison with all components
        >>> fit.plot.comparison('comparison.pdf')
        >>> # Compare data against a single component
        >>> fit.plot.comparison('component_0.pdf', component=0)
        """
        gs_kwargs = gs_kwargs or {}
        model_kwargs = model_kwargs or {}

        if "what" in model_kwargs:
            warnings.warn('"what" argument in model_kwargs is ignored.')
            del model_kwargs["what"]

        imgdata = np.asarray(self.fit.img.data)
        moddata = self.fit.getmodel(
            what="smoothed", component=component, **model_kwargs
        )

        figsize = (fx * 504.00 / dpi, fy * 504.00 / dpi)

        _gs_kwargs = dict(
            top=0.95,
            hspace=0.425,
            height_ratios=[1.00, 0.05],
            wspace=0.245,
            width_ratios=[1.00, 1.00, 1.00],
            bottom=0.130,
            left=0.105,
            right=1.000 - 0.105,
        )

        _gs_kwargs.update(gs_kwargs)

        fig = plt.figure(figsize=figsize, constrained_layout=False)
        gs = fig.add_gridspec(
            2,
            3,
            hspace=_gs_kwargs.get("hspace"),
            wspace=_gs_kwargs.get("wspace"),
            height_ratios=_gs_kwargs.get("height_ratios"),
            width_ratios=_gs_kwargs.get("width_ratios"),
        )
        gs.update(
            bottom=_gs_kwargs.get("bottom"),
            top=_gs_kwargs.get("top"),
            left=_gs_kwargs.get("left"),
            right=_gs_kwargs.get("right"),
        )

        axs = [
            fig.add_subplot(gs[0, gi], projection=self.fit.img.wcs)
            for gi in range(3)
        ]
        bxs = [fig.add_subplot(gs[1, :2]), fig.add_subplot(gs[1, 2])]

        titles = ["Data", "Model", "Residuals"]

        cmap_list = [None, None, None]
        if cmap_data is not None:
            cmap_list[0] = cmap_data
        if cmap_model is not None:
            cmap_list[1] = cmap_model
        if cmap_residual is not None:
            cmap_list[2] = cmap_residual

        if cmaps is not None:
            if isinstance(cmaps, dict):
                cmap_list[0] = cmaps.get("data", cmap_list[0])
                cmap_list[1] = cmaps.get("model", cmap_list[1])
                cmap_list[2] = cmaps.get(
                    "residuals", cmaps.get("residual", cmap_list[2])
                )
            elif isinstance(cmaps, (list, tuple)):
                for i in range(min(3, len(cmaps))):
                    if cmap_list[i] is None:
                        cmap_list[i] = cmaps[i]
            else:
                for i in range(3):
                    if cmap_list[i] is None:
                        cmap_list[i] = cmaps

        defaults = ["magma", "magma", "RdBu_r"]
        for i in range(3):
            if cmap_list[i] is None:
                cmap_list[i] = defaults[i]

        for mi, m in enumerate([imgdata, moddata, imgdata - moddata]):
            if mi < 2:
                vmin = imgdata.min()
                vmax = imgdata.max()
            else:
                vmin = -np.nanmax(np.abs(m))
                vmax = np.nanmax(np.abs(m))

            cmap = cmap_list[mi]

            norm = ImageNormalize(vmin=vmin, vmax=vmax)

            axs[mi].imshow(m, origin="lower", cmap=cmap, norm=norm)

            axs[mi].set_title(titles[mi], pad=10.00, fontsize=11)

            axs[mi].coords[0].display_minor_ticks(True)
            axs[mi].coords[1].display_minor_ticks(True)

            axs[mi].coords[0].set_ticklabel(size=9)
            axs[mi].coords[1].set_ticklabel(size=9)

            xlabel, ylabel = getframe(self.fit.img.wcs)

            axs[mi].coords[0].set_axislabel(xlabel)
            axs[mi].coords[1].set_axislabel(ylabel)

            if mi in [0, 2]:
                bax = bxs[0 if mi == 0 else 1]
                cbar = matplotlib.colorbar.ColorbarBase(
                    bax, cmap=cmap, norm=norm, orientation="horizontal"
                )
                cbar.set_label("Surface brightness [input units]", fontsize=11)

        axs[1].coords[1].set_ticklabel_visible(False)
        axs[2].coords[1].set_ticklabel_visible(False)

        if name is None:
            plt.show()
        else:
            root, ext = os.path.splitext(name)
            if ext.lower() == f".{fmt.lower()}":
                name = root
            plt.savefig(f"{name}.{fmt}", format=fmt, dpi=300)
        plt.close()

    # Autocorrelation time convergence plot
    # --------------------------------------------------------
    def autocorrelation(
        self,
        name=None,
        fmt="pdf",
        fx=0.50,
        fy=0.50,
        dpi=72.27 * 390.00 / 504.00,
        tau_factor=50,
        show_params=False,
    ):
        """
        Plot autocorrelation time convergence diagnostic for emcee runs.

        Creates a log-log plot showing how the integrated autocorrelation
        time estimates evolved during convergence-mode sampling. Useful for
        diagnosing whether the chain has converged.

        Parameters
        ----------
        name : str, optional
            Output filename (without extension). If None, displays plot
            interactively instead of saving. Default is None.
        fmt : str, optional
            Output file format (e.g., 'pdf', 'png'). Default is 'pdf'.
        fx : float, optional
            Figure width scaling factor. Default is 0.50.
        fy : float, optional
            Figure height scaling factor. Default is 0.50.
        dpi : float, optional
            Dots per inch for output resolution. Default is ~55.97.
        tau_factor : float, optional
            Factor used for convergence criterion (chain length > tau_factor
            * tau). A dashed line at N/tau_factor is shown. Default is 50.
        show_params : bool, optional
            If True, plot tau for each parameter individually. If False,
            only plot max(tau) across parameters. Default is False.

        Raises
        ------
        ValueError
            If tau_history is empty (run was not in convergence mode or
            tau never became reliable).

        Notes
        -----
        This plot is only available after running emcee with converge=True.
        The tau estimates become reliable when they cross below the dashed
        N/tau_factor line.
        """
        if (
            not hasattr(self.fit, "tau_history")
            or len(self.fit.tau_history) == 0
        ):
            raise ValueError(
                "No tau history available. Run with converge=True and "
                "ensure chain is long enough for tau estimates."
            )

        iterations = np.array([h[0] for h in self.fit.tau_history])
        tau_arrays = np.array([h[1] for h in self.fit.tau_history])

        figsize = (fx * 504.00 / dpi, fy * 504.00 / dpi)
        fig, ax = plt.subplots(figsize=figsize)

        if show_params:
            for pi in range(tau_arrays.shape[1]):
                label = self.fit.mod.params[self.fit.mod.paridx[pi]]
                ax.loglog(
                    iterations, tau_arrays[:, pi], "o-", label=label, ms=4
                )
        else:
            max_tau = np.max(tau_arrays, axis=1)
            ax.loglog(iterations, max_tau, "o-", label=r"max($\tau$)", ms=4)

        n_range = np.array([iterations.min(), iterations.max()])
        ax.plot(
            n_range,
            n_range / tau_factor,
            "--k",
            label=rf"$\tau = N/{tau_factor}$",
        )

        ax.set_xlabel("number of samples, $N$")
        ax.set_ylabel(r"$\tau$ estimates")
        ax.legend(fontsize=10)

        if name is None:
            plt.show()
        else:
            root, ext = os.path.splitext(name)
            if ext.lower() == f".{fmt.lower()}":
                name = root
            plt.savefig(f"{name}.{fmt}", format=fmt, dpi=300)
        plt.close()
