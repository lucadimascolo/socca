"""L-BFGS-B MLE and MAP optimization backend."""

import scipy.optimize

import jax
import jax.numpy as jp
import numpy as np

import inspect

from ...pool.mpi import MPI_COMM, MPI_RANK, MPI_SIZE
from .utils import warn_unused


#   Fitting method - optimizer
#   --------------------------------------------------------
def run_optimizer(self, pinits, **kwargs):
    """
    Run maximum likelihood optimization using scipy.optimize.

    Finds the maximum likelihood estimate (MLE) or maximum a
    posteriori (MAP) estimate using L-BFGS-B optimization with
    automatic differentiation via JAX.

    Parameters
    ----------
    pinits : array_like, str
        Initial parameter values. Can be:

        - array_like : specific initial values in parameter space
            (will be transformed to unit hypercube internally)
        - "median" : start from the median of each prior (0.5 in
            unit hypercube)
        - "random" : start from random values in unit hypercube
    **kwargs : dict
        Additional keyword arguments. Optimizer-specific options:

        - target : str, ``'mle'`` (default) or ``'map'``. Whether to
            maximize the likelihood alone (MLE) or the full posterior
            log-likelihood + log-prior (MAP).
        - init_step : float, largest first step in the unit hypercube
            (default: 0.05). L-BFGS-B's first step is minus the gradient,
            which for a log-likelihood usually jumps to the prior bounds,
            so a first pass runs on the objective rescaled to keep that
            step below init_step; the result is then refined with the
            unscaled objective.
        - nstarts : int, number of starting points (default: 1). Extra
            starts are drawn uniformly in the unit hypercube, and the
            best finite result is kept.
        - seed : int, random seed for the extra starts (default: 0)
        - tol : float, tolerance for termination (default: ftol=1e-15
            and gtol=1e-8 in options)
        - debug_nans : bool, raise on NaNs during the optimisation
            (default: False)
        - options : dict, solver-specific options passed to
            scipy.optimize.minimize

    Attributes Set
    --------------
    results : scipy.optimize.OptimizeResult
        Optimization result object containing:

        - x : optimal parameters in unit hypercube
        - fun : negative log-likelihood at optimum
        - success : whether optimization succeeded
        - message : description of termination cause

    Raises
    ------
    ValueError
        If pinits is a string other than "median" or "random".

    Notes
    -----
    The optimization is performed in the unit hypercube space
    with bounds [0, 1] for each parameter. The objective function
    is the negative log-likelihood, and gradients are computed
    automatically using JAX. The L-BFGS-B method is used for
    box-constrained optimization.

    References
    ----------
    Byrd, R. H,, Lu, P., Nocedal, J., SIAM J. Sci. Statist. Comput 16, 1190 (1995)
    Zhu, C., Byrd, R. H., Nocedal, J. TOMS, 23, 550 (1997)
    scipy.optimize.minimize documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    L-BFGS-B documentation: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html#optimize-minimize-lbfgsb

    """
    if MPI_SIZE > 1:
        MPI_COMM.bcast(None, root=0)
        if MPI_RANK != 0:
            raise SystemExit(0)
        raise ValueError("Optimizer does not support MPI parallelization.\n ")

    for key in ["pool", "ncores", "n_cores", "num_chains"]:
        if key in kwargs:
            raise ValueError(
                "Optimizer does not support multiprocessing (yet).\n "
            )

    debug_nans = kwargs.pop("debug_nans", False)
    init_step = kwargs.pop("init_step", 0.05)
    nstarts = kwargs.pop("nstarts", 1)
    seed = kwargs.pop("seed", 0)

    target = kwargs.pop("target", "mle")
    if target not in ("mle", "map"):
        raise ValueError("target must be 'mle' or 'map'.")

    opt_kwargs = {}
    for key in inspect.signature(scipy.optimize.minimize).parameters.keys():
        if key not in ["fun", "x0", "jac", "bounds", "method"]:
            if key in kwargs:
                opt_kwargs[key] = kwargs.pop(key)
    warn_unused(kwargs, "optimizer")
    # the default ftol is relative to |logL|, which grows with the pixels
    if "tol" not in opt_kwargs:
        options = {"ftol": 1e-15, "gtol": 1e-8}
        opt_kwargs["options"] = options | opt_kwargs.get("options", {})

    def _opt_prior(pp):
        return jp.array(self._prior_transform(pp))

    def _opt_func(pp):
        pars = _opt_prior(pp)
        ll = self._log_likelihood(pars)
        lp = self._log_prior(pars) if target == "map" else 0.0
        logobj = ll + lp
        return jp.where(
            jp.isfinite(logobj), -logobj, jp.finfo(pp.dtype).max / 2
        )

    _opt_func_jac = jax.jit(jax.value_and_grad(_opt_func))

    def opt_func_jac(pp):
        val, grad = _opt_func_jac(pp)
        grad = jp.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
        return val, grad

    if isinstance(pinits, (list, tuple, np.ndarray, jp.ndarray)):
        pinits = jp.asarray(pinits)
        for pi, p in enumerate(pinits):
            key = self.mod.params[self.mod.paridx[pi]]
            pinits = pinits.at[pi].set(self.mod.priors[key].cdf(p))
    else:
        if pinits == "median":
            pinits = jp.array([0.50 for _ in self.mod.paridx])
        elif pinits == "random":
            pinits = jp.array([np.random.rand() for _ in self.mod.paridx])
        else:
            raise ValueError(
                "Unknown pinits option. Use 'median', 'random', "
                "or provide an array-like object of initial values."
            )

    eps = 1e-6
    bounds = [(eps, 1.00 - eps) for _ in self.mod.paridx]

    def lbfgsb(fun, x0, **kwargs):
        return scipy.optimize.minimize(
            fun=fun,
            x0=x0,
            jac=True,
            bounds=bounds,
            method="L-BFGS-B",
            **kwargs,
        )

    def minimize(x0):
        gmax = float(jp.max(jp.abs(opt_func_jac(x0)[1])))
        scale = init_step / gmax if gmax > 0 else 1.00

        def scaled(pp):
            val, grad = opt_func_jac(pp)
            return val * scale, grad * scale

        first = lbfgsb(scaled, x0, options={"gtol": 1.00e-05 * scale})
        res = lbfgsb(opt_func_jac, first.x, **opt_kwargs)
        res.nit += first.nit
        res.nfev += first.nfev
        return res

    rng = np.random.default_rng(seed)
    starts = [np.asarray(pinits)] + [
        rng.uniform(eps, 1.00 - eps, len(self.mod.paridx))
        for _ in range(nstarts - 1)
    ]
    with jax.debug_nans(bool(debug_nans or jax.config.jax_debug_nans)):
        runs = [minimize(x0) for x0 in starts]
    self.results = min(runs, key=lambda r: (not np.isfinite(r.fun), r.fun))
    self.results.parameters = np.array(self._prior_transform(self.results.x))
