# socca

<code>socca</code> is a minimal library for efficiently modelling image-space astronomical data. 
It is intended to be fast and flexible, taking advantage of the [<code>JAX</code>](https://github.com/google/jax) framework for performing just-in-time compilation and of state-of-the-art nested sampling algorithms ([<code>dynesty</code>](https://github.com/joshspeagle/dynesty), [<code>nautilus</code>](https://nautilus-sampler.readthedocs.io/en/latest/)) for the posterior exploration.

This code is broadly inspired by the excellent [<code>pysersic</code>](https://github.com/pysersic/pysersic) and [<code>astrophot</code>](https://github.com/Autostronomy/AstroPhot) libraries. We recommend their use if you require a more mature and thoroughly tested solution. <code>socca</code> is still in its infancy and many experimental features may undergo significant changes in the future. 

[![GitHub](https://img.shields.io/badge/GitHub-lucadimascolo%2Fsocca-blue.svg?style=flat)](https://github.com/lucadimascolo/socca)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![stability-release-candidate](https://img.shields.io/badge/stability-pre--release-48c9b0.svg)](https://github.com/mkenney/software-guides/blob/master/STABILITY-BADGES.md#release-candidate)
[![astropy](http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat)](http://www.astropy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

```{toctree}
:maxdepth: 2
:caption: Contents
./installation.md

