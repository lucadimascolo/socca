# socca

**``socca``** is a minimal library for efficiently modelling image-space astronomical data. It is intended to be fast and flexible, taking advantage of the [`JAX`](https://github.com/google/jax) framework for performing just-in-time compilation and of state-of-the-art sampling algorithms ([`dynesty`](https://github.com/joshspeagle/dynesty), [`nautilus`](https://nautilus-sampler.readthedocs.io/en/latest/), [`pocomc`](https://github.com/minaskar/pocomc)) for the posterior exploration.


This code is broadly inspired by the excellent [`pysersic`](https://github.com/pysersic/pysersic) and [`astrophot`](https://github.com/Autostronomy/AstroPhot) libraries. We recommend their use if you require a more mature and thoroughly tested solution. **``socca``** is still in its infancy and many experimental features may undergo significant changes in the future. 

[![GitHub](https://img.shields.io/badge/GitHub-lucadimascolo%2Fsocca-blue.svg?style=flat)](https://github.com/lucadimascolo/socca) [![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![stability-release-candidate](https://img.shields.io/badge/stability-pre--release-48c9b0.svg)](https://github.com/mkenney/software-guides/blob/master/STABILITY-BADGES.md#release-candidate)
[![astropy](http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat)](http://www.astropy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


## Author and license
Copyright (c) 2024 [Luca Di Mascolo](https://lucadimascolo.github.io) and [contributors](https://github.com/lucadimascolo/socca/graphs/contributors).

**``socca``** is an open-source library released under the MIT License. The full license terms can be found in the [LICENSE](https://github.com/lucadimascolo/socca/blob/main/LICENSE) file in the main repository.

```{toctree}
:hidden:
:maxdepth: 1
:caption: Information
./installation.md
./contribute.md
./citation.md
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Using socca
./tutorial_quickstart.md
./tutorial_noise.md
./tutorial_priors.md
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Discussions
./faq.md
```