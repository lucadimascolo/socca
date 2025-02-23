# socca

<code>socca</code> is a minimal library for efficiently modelling image-space astronomical data. 
It is intended to be fast and flexible, taking advantage of the [<code>JAX</code>](https://github.com/google/jax) framework for performing just-in-time compilation and of state-of-the-art nested sampling algorithms ([<code>dynesty</code>](https://github.com/joshspeagle/dynesty), [<code>nautilus</code>](https://nautilus-sampler.readthedocs.io/en/latest/)) for the posterior exploration.

This code is broadly inspired by the excellent [<code>pysersic</code>](https://github.com/pysersic/pysersic) and [<code>astrophot</code>](https://github.com/Autostronomy/AstroPhot) libraries. We recommend their use if you require a more mature and thoroughly tested solution. <code>socca</code> is still in its infancy and many experimental features may undergo significant changes in the future. 

[![GitHub](https://img.shields.io/badge/GitHub-lucadimascolo%2Fsocca-blue.svg?style=flat)](https://github.com/lucadimascolo/socca)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3127/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![stability-alpha](https://img.shields.io/badge/stability-alpha-f4d03f.svg)](https://github.com/mkenney/software-guides/blob/master/STABILITY-BADGES.md#alpha)
[![astropy](http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat)](http://www.astropy.org/)

## Contribute
<code>socca</code> is being actively developed and any contributions from the community are welcome. If you have any suggestions, feature requests, or bug reports, please open an issue via the [GitHub issue tracker](https://github.com/lucadimascolo/socca/issues)

## License and attribution
Copyright (c) 2024 [Luca Di Mascolo](https://lucadimascolo.github.io) and [contributors](https://github.com/lucadimascolo/socca/graphs/contributors).

<code>socca</code> is an open-source library released under the MIT License. The full license terms can be found in the [LICENSE](https://github.com/lucadimascolo/socca/blob/main/LICENSE) file in the main repository.


If you are going to include in a publication any results obtained using <code>socca</code>, please consider adding an hyperlink to the GitHub repository: [https://github.com/lucadimascolo/socca](https://github.com/lucadimascolo/socca).
