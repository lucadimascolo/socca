"""Visualization utilities for model fitting results."""

import matplotlib.pyplot as plt
import matplotlib.colorbar
import matplotlib

import numpy as np
import os
import warnings

import corner
import aplpy

from astropy.io import fits
from astropy.visualization import ImageNormalize

matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Helvetica"]


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

    #   Corner plot of the posterior samples
    #   --------------------------------------------------------
    def corner(
        self,
        name=None,
        comp=None,
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
        comp : str, int, list, or None, optional
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
        if comp is None:
            comp = [f"comp_{ci:02d}" for ci in range(self.fit.mod.ncomp)]
        elif isinstance(comp, str):
            comp = [comp]
        elif isinstance(comp, (list, tuple)):
            comp_ = []
            for c in comp:
                if isinstance(c, int):
                    comp_.append(f"comp_{int(c):02d}")
                elif isinstance(c, str):
                    if "comp" not in str(c):
                        comp_.append(f"comp_{str(c)}")
                    else:
                        comp_.append(str(c))
                elif hasattr(c, "id"):
                    comp_.append(f"comp_{int(c.id):02d}")

            comp = comp_
            del comp_

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
                if any(lbl.startswith(c) for c in comp)
            ]
        )

        truths = kwargs.pop("truths", None)
        if truths is not None:
            truths = np.array(truths)[indices]

        corner.corner(
            data=self.fit.samples[:, indices],
            weights=self.fit.weights,
            labels=labels[indices],
            range=edges[indices],
            quantiles=quantiles,
            truths=truths,
            bins=bins,
            **kwargs,
        )

        if name is None:
            plt.show()
        else:
            plt.savefig(f"{name}.{fmt}", format=fmt, dpi=300)
        plt.close()

    #   Comparison plot: data, model, residuals
    #   --------------------------------------------------------
    def comparison(
        self,
        name=None,
        fmt="pdf",
        fx=1.00,
        fy=0.38,
        dpi=72.27 * 390.00 / 504.00,
        cmaps=None,
        cmap_data=None,
        cmap_model=None,
        cmap_residual=None,
        gs_kwargs={},
        model_kwargs={},
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
        """
        if "what" in model_kwargs:
            warnings.warn('"what" argument in model_kwargs is ignored.')
            del model_kwargs["what"]

        imgdata = np.asarray(self.fit.img.data)
        moddata = self.fit.getmodel(what="smoothed", **model_kwargs)

        figsize = (fx * 504.00 / dpi, fy * 504.00 / dpi)

        _gs_kwargs = dict(
            hspace=0.475,
            height_ratios=[1.00, 0.05],
            wspace=0.245,
            width_ratios=[1.00, 1.00, 1.00],
            bottom=0.130,
            top=0.995,
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

        axs = [fig.add_subplot(gs[0, gi]) for gi in range(3)]
        axs = [a.get_position() for a in axs]
        axs = [[a.x0, a.y0, a.width, a.width * fx / fy] for a in axs]

        bxs = [fig.add_subplot(gs[1, :2]), fig.add_subplot(gs[1, 2])]

        bxs = [a.get_position() for a in bxs]
        bxs = [[a.x0, a.y0, a.width, a.height] for a in bxs]
        plt.close()

        fig = plt.figure(figsize=figsize, constrained_layout=False)

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

            hdu = fits.PrimaryHDU(m, header=self.fit.img.hdu.header)
            out = aplpy.FITSFigure(hdu, figure=fig, subplot=axs[mi])
            out.show_colorscale(cmap=cmap, vmin=vmin, vmax=vmax)
            out.image.set_norm(norm)

            if mi > 0:
                out.tick_labels.hide_y()
                out.axis_labels.hide_y()

            if mi in [0, 2]:
                bax = fig.add_axes(bxs[0 if mi == 0 else 1])
                cbar = matplotlib.colorbar.ColorbarBase(
                    bax, cmap=cmap, norm=norm, orientation="horizontal"
                )

                cbar.set_label("Surface brightness [input units]", fontsize=11)

            out.set_title(titles[mi], pad=10.00, fontsize=11)

            out.tick_labels.set_font(size=9)

        root, ext = os.path.splitext(name)
        if ext.lower() == f".{fmt.lower()}":
            name = root

        plt.savefig(f"{name}.{fmt}", format=fmt, dpi=300)
        plt.close()
