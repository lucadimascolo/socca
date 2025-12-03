# Source Characterization using a Composable Analysis
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![stability-release-candidate](https://img.shields.io/badge/stability-pre--release-48c9b0.svg)](https://github.com/mkenney/software-guides/blob/master/STABILITY-BADGES.md#release-candidate)
[![astropy](http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat)](http://www.astropy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

`socca` is a minimal library for efficiently modelling image-space astronomical data. It is intended to be fast and flexible, taking advantage of the [`JAX`](https://github.com/google/jax) framework for performing just-in-time compilation and of state-of-the-art sampling algorithms ([`dynesty`](https://github.com/joshspeagle/dynesty), [`nautilus`](https://nautilus-sampler.readthedocs.io/en/latest/), [`pocomc`](https://github.com/minaskar/pocomc)) for the posterior exploration.

Installation
---------------------------------------------------
> [!WARNING]
> `socca` was built using `python=3.12`. Although the installation is not bound to this specific version and different releases could work fine, it is recommended to use the same version to avoid any compatibility issues.

To install `socca`, it should be enough to run

```
python -m pip install git+https://github.com/lucadimascolo/socca.git
```

This will download and install the latest version of `socca` as well as all the required dependencies. Once the installation is completed, you should be ready to get `socca` to crunch your data.


