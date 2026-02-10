# Source Characterization using a Composable Analysis
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![stability-release-candidate](https://img.shields.io/badge/stability-release--candidate-48c9b0.svg)](https://github.com/mkenney/software-guides/blob/master/STABILITY-BADGES.md#release-candidate)
[![CI](https://github.com/lucadimascolo/socca/actions/workflows/ci.yml/badge.svg)](https://github.com/lucadimascolo/socca/actions/workflows/ci.yml)
[![Docs](https://github.com/lucadimascolo/socca/actions/workflows/docs.yml/badge.svg)](https://lucadimascolo.github.io/socca)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![JAX](https://img.shields.io/badge/accelerated%20by-JAX-923CAA)](https://docs.jax.dev/en/latest/index.html)
[![astropy](http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat)](http://www.astropy.org/)


`socca` is a minimal library for efficiently modelling image-space astronomical data. It is intended to be fast and flexible, taking advantage of the [`JAX`](https://github.com/google/jax) framework for performing just-in-time compilation and of state-of-the-art sampling algorithms ([`dynesty`](https://github.com/joshspeagle/dynesty), [`nautilus`](https://nautilus-sampler.readthedocs.io/en/latest/), [`pocomc`](https://github.com/minaskar/pocomc)) for the posterior exploration.

For detailed installation instructions, tutorials, and usage guides, check out the documentation at **[lucadimascolo.github.io/socca](https://lucadimascolo.github.io/socca)**.

Installation
---------------------------------------------------
> [!WARNING]
> `socca` has been built using `python=3.12`. Although the installation is not bound to this specific version and different releases could work fine, it is recommended to use the same version to avoid any compatibility issues.

To install `socca`, it should be enough to run

```
python -m pip install git+https://github.com/lucadimascolo/socca.git
```

This will download and install the latest version of `socca` as well as all the required dependencies. Once the installation is completed, you should be ready to get `socca` to crunch your data.


