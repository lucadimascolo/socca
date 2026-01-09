# socca

**``socca``** is a minimal library for efficiently modelling image-space astronomical data. It is intended to be fast and flexible, taking advantage of the [`JAX`](https://github.com/google/jax) framework for performing just-in-time compilation and of state-of-the-art sampling algorithms ([`dynesty`](https://github.com/joshspeagle/dynesty), [`nautilus`](https://github.com/johannesulf/nautilus), [`pocomc`](https://github.com/minaskar/pocomc)) for the posterior exploration.


This code is broadly inspired by the excellent [`pysersic`](https://github.com/pysersic/pysersic) and [`astrophot`](https://github.com/Autostronomy/AstroPhot) libraries. We recommend their use if you require a more mature and thoroughly tested solution. **``socca``** is still in its infancy and many experimental features may undergo significant changes in the future. 

[![GitHub](https://img.shields.io/badge/GitHub-lucadimascolo%2Fsocca-blue.svg?style=flat-square)](https://github.com/lucadimascolo/socca)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg?style=flat-square)](https://www.python.org/downloads/)
[![stability-release-candidate](https://img.shields.io/badge/stability-pre--release-48c9b0.svg?style=flat-square)](https://github.com/mkenney/software-guides/blob/master/STABILITY-BADGES.md#release-candidate)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json&style=flat-square)](https://github.com/astral-sh/ruff)
[![CI](https://img.shields.io/github/actions/workflow/status/lucadimascolo/socca/ci.yml?style=flat-square&label=CI)](https://github.com/lucadimascolo/socca/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/github/actions/workflow/status/lucadimascolo/socca/docs.yml?style=flat-square&label=Docs)](https://lucadimascolo.github.io/socca)<br/>
[![JAX](https://img.shields.io/badge/accelerated%20by-JAX-923CAA?style=flat-square)](https://docs.jax.dev/en/latest/index.html)
[![astropy](http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat-square)](http://www.astropy.org/)

<br/>

```{admonition} Where to start?
:class: tip

⚙️ Clearly, first things first: head to the [installation guide](./installation.md) to get the package installed on your system.

🚀 To start using **``socca``**, we recommend going through the [quickstart tutorial](./tutorials/tutorial_quickstart.md) which will guide you through the main features of the library step by step. For a more in-depth overview of the different functionalities, check out the other guides in the "Using socca" section or the [API reference](./api/index.md).

🐛 Found a bug or want to contribute? Check out the [contribution guide](./contribute.md) for more information on how to get involved.

📚 Using **``socca``** in your research? Please see the [citation guide](./citation.md) for how to cite it in your publications.
```

<br/>

## Author and license
Copyright (c) 2024 [Luca Di Mascolo](https://lucadimascolo.github.io) and [contributors](https://github.com/lucadimascolo/socca/graphs/contributors).

**``socca``** is an open-source library released under the MIT License. The full license terms can be found in the [LICENSE](https://github.com/lucadimascolo/socca/blob/main/LICENSE) file in the main repository.

## Acknowledgements
Many thanks to Marloes van Asselt for testing early versions of the code and building the building the first prototypr of the `Disk` model component. Further thanks to Nic Zaparniuk for having extensively tested the code.

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
./tutorials/tutorial_quickstart.md
./tutorials/tutorial_noise.md
./tutorials/tutorial_data.md
./tutorials/tutorial_models.md
./tutorials/tutorial_priors.md
./tutorials/tutorial_fitting.md
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Additional resources
./api/index.md
./faq.md
```