# Parallelization

**``socca``** supports parallel likelihood evaluation to speed up sampling runs. Two parallelization modes are available: **multi-process** parallelization on a single machine, and **MPI-based** distributed parallelization across multiple nodes. Both modes are supported by the `nautilus`, `dynesty`, and `pocomc` sampling backends.

## Multi-process parallelization

The simplest way to parallelize a sampling run is to pass the `ncores` argument to the `run()` method. This spawns a pool of worker processes on the local machine, each evaluating the likelihood function independently:

```python
>>> fit = socca.fitter(img=img, mod=mod)
>>> fit.run(method='nautilus', ncores=4)
```

Under the hood, this creates a `MultiPool` from `socca.pool.mp`, using [loky](https://loky.readthedocs.io/) for robust process management. Workers do not re-import `__main__`, so no `if __name__ == "__main__":` guard _should_ be needed in user scripts.

The `ncores` argument is accepted by all three sampling backends:

```python
>>> fit.run(method='nautilus', ncores=4)
>>> fit.run(method='dynesty', ncores=4, nlive=500)
>>> fit.run(method='pocomc', ncores=4)
```

Alternatively, you can pass a pre-built pool object via the `pool` argument. An integer value for `pool` is treated identically to `ncores`:

```python
>>> fit.run(method='nautilus', pool=4)  # equivalent to ncores=4
```

Please note that specifying `ncores` and `pool` at the same time will raise a `ValueError`.


```{important}
Parallelization is most beneficial when the likelihood evaluation is computationally expensive (e.g., involving large images, complex models, or PSF convolution on fine grids). If the likelihood is fast to evaluate, the overhead of inter-process communication can outweigh the speedup, resulting in *slower* runs than serial execution. This is especially true for `dynesty` and `pocomc`, where the sampler frequently synchronizes with the worker pool. `nautilus` is less affected due to its batch-oriented point proposal, but even in that case the gains diminish for cheap likelihoods. As a rule of thumb, parallelization pays off when a single likelihood call takes at least a few milliseconds.
```

## MPI parallelization

For distributed computing across multiple nodes (e.g., on HPC clusters), **``socca``** supports MPI-based parallelization via [mpi4py](https://mpi4py.readthedocs.io/). MPI parallelization is activated automatically when the script is launched with `mpirun` or `mpiexec`:

```bash
mpirun -np 8 python my_fit_script.py
```

No changes to the Python script are required. **``socca``** detects the MPI environment by checking for standard environment variables set by common MPI launchers (Open MPI, MPICH, Intel MPI, SLURM). When multiple MPI ranks are detected, an `MPIPool` is created automatically and the likelihood evaluation is distributed across all ranks.

The rank-0 process (master) drives the sampler, while worker ranks enter a loop that receives batches of likelihood evaluations. After sampling completes, the results (`samples`, `weights`, `logw`, `logz`, `logz_prior`) are broadcast to all ranks.

### Installing MPI support

MPI support requires `mpi4py`:

```bash
pip install mpi4py
```

If `mpi4py` is not installed but the process is launched under an MPI environment, **``socca``** will raise an `ImportError` with installation instructions.

### Sampler compatibility

While all three backends support MPI, `nautilus` is the recommended choice for distributed runs. Dynesty's sequential point proposal does not benefit as much from MPI parallelization, and a warning is printed when this combination is detected:

```
Dynesty might not benefit from MPI parallelization due to its sequential
point proposal. Consider using method='nautilus' for MPI runs.
```

## NumPyro NUTS

The `numpyro` backend does not support MPI parallelization (an error is raised if attempted). However, the `ncores` or `pool` arguments can be passed to control the number of parallel MCMC chains:

```python
>>> fit.run(method='numpyro', ncores=4)  # runs 4 chains in parallel
```

## MPI-safe post-processing

When running under MPI, several post-processing methods are restricted to rank 0 to avoid redundant output or file conflicts. These include `parameters()`, `dump()`, `savemodel()`, and `bmc()`. On non-root ranks, these methods return `None` silently.

## Summary

| Method | Multi-process | MPI | Notes |
|:------:|:-------------:|:---:|-------|
| `'nautilus'` | `ncores` / `pool` | automatic | Recommended for MPI runs |
| `'dynesty'` | `ncores` / `pool` | automatic | MPI supported but less efficient |
| `'pocomc'` | `ncores` / `pool` | automatic | |
| `'numpyro'` | `ncores` (chains) | not supported | `ncores` sets number of chains |
| `'optimizer'` | not supported | not supported | Single-process only |
