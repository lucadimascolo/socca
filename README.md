# Source Characterization using a Composable Analysis
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3127/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![stability-alpha](https://img.shields.io/badge/stability-alpha-f4d03f.svg)](https://github.com/mkenney/software-guides/blob/master/STABILITY-BADGES.md#alpha)
[![astropy](http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat)](http://www.astropy.org/)


`socca` is a minimal library for efficiently modelling image-space astronomical data. It is intended to be fast and flexible, taking advantage of the [`JAX`](https://github.com/google/jax) framework for performing just-in-time compilation and of state-of-the-art nested sampling algorithms ([`dynesty`](https://github.com/joshspeagle/dynesty), [`nautilus`](https://nautilus-sampler.readthedocs.io/en/latest/)) for the posterior exploration.

Installation
---------------------------------------------------
> [!WARNING]
> `socca` was built using `python=3.12` and the installation is currently bound to this specific version. Although higher releases could work fine, it is recommended to use the same version to avoid any compatibility issues.

To install `socca`, it should be enough to run

```
python -m pip install git+https://github.com/lucadimascolo/socca.git
```

This will download and install the latest version of `socca` as well as all the required dependencies. Once the installation is completed, you should be ready get `socca` to crunch your data.


Notes
-----
...

Example
-------
This is a basic example of how to use `socca` for modelling an input image using a Sérsic profile.

```python
import socca
```

To do list
---------------------------------------------------
- [ ] noise model
- [ ] checkpointing and integration of `h5py` I/O framework
- [ ] extended models
- [ ] prior initialization
