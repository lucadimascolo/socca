"""L-BFGS-B maximum likelihood and MAP optimization backend."""

import scipy.optimize

import jax
import jax.numpy as jp
import numpy as np

import inspect

from ...pool.mpi import MPI_COMM, MPI_RANK, MPI_SIZE


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
        Additional keyword arguments passed to scipy.optimize.minimize.
        Common options include:

        - tol : float, tolerance for termination
        - options : dict, solver-specific options

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

    opt_kwargs = {}
    for key in inspect.signature(scipy.optimize.minimize).parameters.keys():
        if key not in ["fun", "x0", "jac", "bounds", "method"]:
            if key in kwargs:
                opt_kwargs[key] = kwargs.pop(key)

    def _opt_prior(pp):
        return jp.array(self._prior_transform(pp))

    def _opt_func(pp):
        pars = _opt_prior(pp)
        return -self._log_likelihood(pars)

    opt_func_jac = jax.jit(jax.value_and_grad(_opt_func))

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

    bounds = [(0.00, 1.00) for _ in self.mod.paridx]

    self.results = scipy.optimize.minimize(
        fun=opt_func_jac,
        x0=pinits,
        jac=True,
        bounds=bounds,
        method="L-BFGS-B",
        **opt_kwargs,
    )
