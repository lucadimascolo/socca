"""Training and validation data generation for the gNFW emulator."""

import os
import warnings
from itertools import combinations, product

import h5py
import jax
import jax.numpy as jp
import numpy as np
from tqdm import tqdm

from .config import Profile
from ..model import getzero, integral


def generate_training_data(
    n_samples,
    n_radial,
    n_edge=0,
    n_corner=0,
    chunk_size=100,
    show_progress=True,
    output_path=None,
    seed=42,
    resume=True,
):
    """Generate labelled training data for the gNFW emulator.

    Computes the Abel-projected gNFW integral on a grid of randomly drawn
    parameter combinations. Supports boundary-aware sampling so that the
    emulator learns accurate behaviour at the edges of the training domain.

    Parameters
    ----------
    n_samples : int
        Number of interior (unconstrained) parameter combinations.
    n_radial : int
        Number of radial points per parameter combination.
    n_edge : int or float, optional
        Boundary face samples. A float < 1 is interpreted as a fraction of
        ``n_samples``. Default is 0 (no boundary samples).
    n_corner : int or float, optional
        Boundary edge/corner samples. A float < 1 is interpreted as a
        fraction of ``n_samples``. Default is 0.
    chunk_size : int, optional
        Number of parameter combinations processed per JAX call (controls
        memory usage). Default is 100.
    show_progress : bool, optional
        Show a tqdm progress bar. Default is True.
    output_path : str or None, optional
        If given, intermediate results and the final dataset are written to
        this HDF5 file. Enables resume behaviour.
    seed : int, optional
        PRNG seed. Default is 42.
    resume : bool, optional
        If True and ``output_path`` exists with a completed dataset, return
        the cached result without recomputing. Default is True.

    Returns
    -------
    dict
        Dictionary with keys ``x``, ``alpha``, ``beta``, ``gamma``,
        ``boundary_alpha``, ``boundary_beta``, ``boundary_gamma``, ``logy``.
        All values are 1-D JAX arrays of the same length.
    """
    rng = jax.random.PRNGKey(seed)
    rng, *subkeys = jax.random.split(rng, 5)

    if isinstance(n_edge, float) and n_edge < 1.0:
        n_edge = int(n_samples * n_edge)
    if isinstance(n_corner, float) and n_corner < 1.0:
        n_corner = int(n_samples * n_corner)

    # Each spec: (n, flag_alpha, flag_beta, flag_gamma)
    # flag = 0: free, -1: lower boundary, +1: upper boundary
    boundary_specs = [(n_samples, 0, 0, 0)]

    if n_edge > 0:
        for i in range(3):
            for sign in [-1, +1]:
                flags = [0, 0, 0]
                flags[i] = sign
                boundary_specs.append((n_edge, *flags))

    if n_corner > 0:
        for i, j in combinations(range(3), 2):
            for si, sj in product([-1, +1], repeat=2):
                flags = [0, 0, 0]
                flags[i] = si
                flags[j] = sj
                boundary_specs.append((n_corner, *flags))

        for signs in product([-1, +1], repeat=3):
            boundary_specs.append((n_corner, *signs))

    n_total = sum(spec[0] for spec in boundary_specs)
    n_pts = n_radial

    if not resume and output_path and os.path.exists(output_path):
        os.remove(output_path)

    if resume and output_path and os.path.exists(output_path):
        with h5py.File(output_path, "r") as f:
            if f.attrs.get("complete", False) and "boundary_alpha" in f:
                return {key: jp.array(f[key][:]) for key in f.keys()}

    alpha_chunks, beta_chunks, gamma_chunks = [], [], []
    flag_a_chunks, flag_b_chunks, flag_g_chunks = [], [], []

    rng, *param_keys = jax.random.split(rng, len(boundary_specs) * 3 + 1)
    ki = 0
    for spec in boundary_specs:
        n, fa, fb, fg = spec

        a = jax.random.uniform(
            param_keys[ki],
            (n,),
            minval=Profile.alpha[0],
            maxval=Profile.alpha[1],
        )
        b = jax.random.uniform(
            param_keys[ki + 1],
            (n,),
            minval=Profile.beta[0],
            maxval=Profile.beta[1],
        )
        g = jax.random.uniform(
            param_keys[ki + 2],
            (n,),
            minval=Profile.gamma[0],
            maxval=Profile.gamma[1],
        )
        ki += 3

        if fa == -1:
            a = jp.full((n,), Profile.alpha[0])
        elif fa == +1:
            a = jp.full((n,), Profile.alpha[1])

        if fb == -1:
            b = jp.full((n,), Profile.beta[0])
        elif fb == +1:
            b = jp.full((n,), Profile.beta[1])

        if fg == -1:
            g = jp.full((n,), Profile.gamma[0])
        elif fg == +1:
            g = jp.full((n,), Profile.gamma[1])

        alpha_chunks.append(a)
        beta_chunks.append(b)
        gamma_chunks.append(g)
        flag_a_chunks.append(jp.full((n,), fa, dtype=jp.float32))
        flag_b_chunks.append(jp.full((n,), fb, dtype=jp.float32))
        flag_g_chunks.append(jp.full((n,), fg, dtype=jp.float32))

    alpha = jp.concatenate(alpha_chunks)
    beta = jp.concatenate(beta_chunks)
    gamma = jp.concatenate(gamma_chunks)
    flag_alpha = jp.concatenate(flag_a_chunks)
    flag_beta = jp.concatenate(flag_b_chunks)
    flag_gamma = jp.concatenate(flag_g_chunks)

    rng, xkey = jax.random.split(rng)
    logx = jax.random.uniform(
        xkey,
        (n_total, n_radial - 1),
        minval=jp.log10(Profile.x[0]),
        maxval=jp.log10(Profile.x[1]),
    )
    x = jp.power(10.0, logx)
    x = jp.concatenate([x, jp.full((n_total, 1), Profile.x[1])], axis=1)

    _integral_chunk = jax.jit(jax.vmap(integral, in_axes=(0, 0, 0, 0)))
    n_chunks = (n_total + chunk_size - 1) // chunk_size

    n_done = 0
    if output_path:
        if os.path.exists(output_path):
            with h5py.File(output_path, "r") as f:
                if "y_raw" in f:
                    n_done = int(f.attrs.get("n_samples_done", 0))
        if n_done == 0:
            with h5py.File(output_path, "w") as f:
                f.create_dataset(
                    "y_raw",
                    shape=(n_total, n_pts),
                    dtype="float64",
                )
                f.attrs["n_samples_done"] = 0

    n_done_chunks = (
        (n_done + chunk_size - 1) // chunk_size if n_done > 0 else 0
    )
    iterator = range(n_chunks)
    if show_progress:
        iterator = tqdm(
            iterator,
            desc="Generating training data",
            initial=n_done_chunks,
            total=n_chunks,
        )

    y_chunks = []
    for c in iterator:
        s = c * chunk_size
        e = min(s + chunk_size, n_total)

        if e <= n_done:
            continue

        y_chunk = _integral_chunk(x[s:e], alpha[s:e], beta[s:e], gamma[s:e])

        if output_path:
            with h5py.File(output_path, "a") as f:
                f["y_raw"][s:e] = np.array(y_chunk)
                f.attrs["n_samples_done"] = e
        else:
            y_chunks.append(y_chunk)

    if output_path:
        with h5py.File(output_path, "r") as f:
            y_all = jp.array(f["y_raw"][:])
    else:
        y_all = jp.concatenate(y_chunks, axis=0)

    logy0 = jp.log10(getzero(alpha, beta, gamma))
    logy_all = jp.log10(y_all) - logy0[:, None]

    data = {
        "x": x.ravel(),
        "alpha": jp.repeat(alpha, n_pts),
        "beta": jp.repeat(beta, n_pts),
        "gamma": jp.repeat(gamma, n_pts),
        "boundary_alpha": jp.repeat(flag_alpha, n_pts),
        "boundary_beta": jp.repeat(flag_beta, n_pts),
        "boundary_gamma": jp.repeat(flag_gamma, n_pts),
        "logy": logy_all.ravel(),
    }

    valid = jp.isfinite(data["logy"])
    n_dropped = int(jp.sum(~valid))
    if n_dropped > 0:
        n_before = len(data["logy"])
        warnings.warn(
            f"Dropped {n_dropped}/{n_before} samples with non-finite "
            f"log(y) ({100 * n_dropped / n_before:.1f}%)"
        )

    result = {key: val[valid] for key, val in data.items()}

    if output_path:
        with h5py.File(output_path, "w") as f:
            for key, val in result.items():
                f.create_dataset(key, data=np.array(val))
            f.attrs["complete"] = True

    return result
