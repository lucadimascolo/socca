# Changelog

## [1.4.0](https://github.com/lucadimascolo/socca/compare/v1.3.0...v1.4.0) (2026-08-07)


### Features

* add Bar constructor, profile, and kwarg builder ([6dab3e5](https://github.com/lucadimascolo/socca/commit/6dab3e5b4bf69ef3d963b6fafe17cc9345decfa6))
* add Bar.getmap for fixed-parameter model evaluation ([b68cc9a](https://github.com/lucadimascolo/socca/commit/b68cc9ae9d86a3be1351298fa74ce93d0b51940e))
* add BoundTo for tying parameters into namespaced sub-components ([505f90d](https://github.com/lucadimascolo/socca/commit/505f90de11dcb65a514329b2966b59f3a87f57ad))
* add xc/yc/theta/rs to BarGeometry and namespace Bar's units ([a7d4384](https://github.com/lucadimascolo/socca/commit/a7d43847124df52dcf091fa6128f561e7b9827e0))
* introduce BarGeometry helper and Bar.parameters()/parlist() ([cd744ad](https://github.com/lucadimascolo/socca/commit/cd744ad366eb6d8ee973074b90e4c3a633e759cd))
* register Bar in the models package and fix Disk/Bar id merging ([a27d10b](https://github.com/lucadimascolo/socca/commit/a27d10b1d8ff84fd7855cf26311af597b88fc42d))
* scaffold Bar model with line-of-sight grid construction ([8aa63fa](https://github.com/lucadimascolo/socca/commit/8aa63fa78fe8e614dc6e22afe031ea80650bbf57))
* switch Bar to ellipticity-based geometry and add BarGeometry defaults ([24ecb88](https://github.com/lucadimascolo/socca/commit/24ecb88f77d3b56d9335a142873ae6e8b1abe6d0))
* warn on out-of-range ellipticity/radius/angle-span parameters ([45ba3d8](https://github.com/lucadimascolo/socca/commit/45ba3d86534a341e9d037aa56a81173509e75db1))


### Bug Fixes

* apply Bar's intrinsic rotation before inclining, not after ([18bd6c7](https://github.com/lucadimascolo/socca/commit/18bd6c79166d1c2846807fbc9bb8b11c31be48e5))
* correct Bar position-angle rotation sign so PA starts along N/S ([1f0e7d8](https://github.com/lucadimascolo/socca/commit/1f0e7d82c061516827e6fab847096bc44a9640a7))
* correct double scaling of Bar's Sersic radius ([ad3a8a6](https://github.com/lucadimascolo/socca/commit/ad3a8a6c04c558a46f3382372ab05aaf8e6ea2e2))
* correct rot's description -- not simply an offset to theta ([9506650](https://github.com/lucadimascolo/socca/commit/950665011940435610123d34460af2c1887b93a2))
* correct the Ellipsoid profile equation in the components tutorial ([5cd7865](https://github.com/lucadimascolo/socca/commit/5cd7865bad7e7b12e19d417c364ea7dba7926c7d))
* make the boundto() tie marker private and callable ([98561bc](https://github.com/lucadimascolo/socca/commit/98561bc88909538610da8b520bb687f34dcbbb2f))
* recognize _BoundTo as tied in all .parameters() displays ([45bda29](https://github.com/lucadimascolo/socca/commit/45bda2957330b6e15ae78fe9b3b56e9826c11588))
* resolve boundto() ties when calling a component's getmap() directly ([da841f8](https://github.com/lucadimascolo/socca/commit/da841f8243f22599e4ac30c465b029c98cbe3154))
* rework Bar's PA/inclination/rotation convention in getgrid ([02c9476](https://github.com/lucadimascolo/socca/commit/02c94767c8d363895deb5b5e99350218d61a6f29))
* update variable name for the ellipticity ratio parameter ([7a69ff2](https://github.com/lucadimascolo/socca/commit/7a69ff29ad935844545bb66fabbd0063aab58fb4))


### Documentation

* add API reference page for socca.models.bar ([afac756](https://github.com/lucadimascolo/socca/commit/afac756986ed0482cc64c6e5735ade768195a704))
* document the Bar model in the components tutorial ([09d8853](https://github.com/lucadimascolo/socca/commit/09d885384f8c9c95e5d8abb8f1cc1f02d87bdd92))
* mention Bar in zoo()'s docstring ([5f080fa](https://github.com/lucadimascolo/socca/commit/5f080faf86fee2886a06aa3941c20d321dd05019))
* note the new range/span validation in the Ellipsoid tutorial ([d7f134e](https://github.com/lucadimascolo/socca/commit/d7f134e240a99e8606053533f31b8f68368f9f0e))
* point CONTRIBUTING.md at main, fix a stale issue-tracker link ([e7a8639](https://github.com/lucadimascolo/socca/commit/e7a8639d5d19d07e06dc4355a432d491789db3f4))
* rename Bar to Ellipsoid in examples/docs, note galactic bars as a use case ([c509576](https://github.com/lucadimascolo/socca/commit/c509576bae0deb26d3204a34481a8b47c9a57258))
* replace examples/test_bar.py with 03_bar.py ([4f0f69b](https://github.com/lucadimascolo/socca/commit/4f0f69b7d413a614a25ca547028f39f7154d4cf3))
* write real docstrings for Bar, closing out ruff's docstring debt ([73f16d9](https://github.com/lucadimascolo/socca/commit/73f16d9d28876ef5713646cb4bc8f0a92b813cd5))

## [1.3.0](https://github.com/lucadimascolo/socca/compare/v1.2.0...v1.3.0) (2026-07-30)


### Features

* add map/mle support to optimizer ([73db57f](https://github.com/lucadimascolo/socca/commit/73db57f55883435267f2a73625ab9a2a791936e5))
* add support for custom units ([be57273](https://github.com/lucadimascolo/socca/commit/be57273e40fbba9e7dccad7a8c8a7e598b43d59d))


### Bug Fixes

* **ci:** pin matplotlib&lt;3.9 to restore catppuccin compatibility ([482180a](https://github.com/lucadimascolo/socca/commit/482180ac70cd49f3740a584127db62ddd15a2e71))
* initialize noise before PSF setup ([a5d4ccd](https://github.com/lucadimascolo/socca/commit/a5d4ccda8ec793fb2b9c14e21735f91c7fb0fe22))
* normalize byte order for JAX array conversion ([e63bb3f](https://github.com/lucadimascolo/socca/commit/e63bb3fcf9c6db0a1912e46c14c173ec3c490e25))
* preserve inherited metadata dictionaries ([d9cbc45](https://github.com/lucadimascolo/socca/commit/d9cbc45c816226f7a9b66c69701e995dcca43c2c))


### Documentation

* add map/mle support ([3c521db](https://github.com/lucadimascolo/socca/commit/3c521dbedf54850789a8e3e970f6881a8ca3b014))
* fix gnfw equations ([74b6a0c](https://github.com/lucadimascolo/socca/commit/74b6a0c0416a7386151a975aad0c7db7e88907fe))

## [1.2.0](https://github.com/lucadimascolo/socca/compare/v1.1.0...v1.2.0) (2026-06-10)


### Features

* add support for pocomc vectorization ([776d104](https://github.com/lucadimascolo/socca/commit/776d104714e53888ccabb19b45d448acb270fcd0))


### Bug Fixes

* add guard for low=high values in (log)uniform priors ([08213e1](https://github.com/lucadimascolo/socca/commit/08213e1f170ade2227a6366e09076aa9b8604391))
* add missing raise for ValueError ([886a770](https://github.com/lucadimascolo/socca/commit/886a770e6db28b7f017284cc572954cb66e4a419))
* add support to optimizer in initialization and plotting ([2d3b851](https://github.com/lucadimascolo/socca/commit/2d3b85145494241202c3af9b35f4e1399393fed9))
* add support to PCi_j rotation in pixel-world transforms ([9c545b8](https://github.com/lucadimascolo/socca/commit/9c545b898a89e7a3e390a9d0e74c57f0a0e00b6a))
* avoid nans in gradient calculation ([ba628dd](https://github.com/lucadimascolo/socca/commit/ba628ddbdcbe9f74029d6d26901d3df40ef9547f))
* avoid per-sample GPU-to-CPU transfers in likelihood evaluation ([cb4af4f](https://github.com/lucadimascolo/socca/commit/cb4af4f8737585551fd7d973799b4ddf008a78d6))
* change to 0-indexed cutout re-centering for psf model ([112b6f1](https://github.com/lucadimascolo/socca/commit/112b6f1c07659f4777770b22ad98c12d0a941b13))
* correct CDELT2 calculation from CDX_2 values ([0dfeab7](https://github.com/lucadimascolo/socca/commit/0dfeab706e560b6133bde1ee3b70a79dd1018f51))
* initialize mbkg to empty list ([a6d200a](https://github.com/lucadimascolo/socca/commit/a6d200ac30472eed7ea32579c0e393ed247399dd))
* pin jax/jaxlib below 0.10.0 to avoid numpyro incompatibility ([4378cd1](https://github.com/lucadimascolo/socca/commit/4378cd1216f0cc87d913eb7eba6899e04b13495e))
* raise NotImplementedError for SIP/TPV distorted WCS ([e39235a](https://github.com/lucadimascolo/socca/commit/e39235a02aad3153b4137746bdf9c359cdd62d78))
* raise ValueError if sigma is not (float, int) ([3c5f671](https://github.com/lucadimascolo/socca/commit/3c5f6711e54e4b05d7421722dabc4b24543b1e3a))
* remove redundant type/value check ([fd3ac6b](https://github.com/lucadimascolo/socca/commit/fd3ac6b66fb85903758188467030420ecb03f9af))
* remove unused `e` and `cbox` from Disk model ([d333c7c](https://github.com/lucadimascolo/socca/commit/d333c7cd329cb74c00d1c131dff1722dfbcad286))
* replace eval with getattr for I values ([276e691](https://github.com/lucadimascolo/socca/commit/276e69123c0a342d986f3668da48f5b97341ff37))


### Documentation

* add vectorization argument to pocomc docs ([2215325](https://github.com/lucadimascolo/socca/commit/221532564980e5b8240c0d9870780fb0df268297))
* update Disk description ([82e5229](https://github.com/lucadimascolo/socca/commit/82e5229565cdbb3536ebe53a404cb106b1561db6))

## [1.1.0](https://github.com/lucadimascolo/socca/compare/v1.0.0...v1.1.0) (2026-04-17)


### Features

* add first implementation of truncation profile ([1b2d86b](https://github.com/lucadimascolo/socca/commit/1b2d86b10507755251b922c96fbe1e4c7acdef69))
* add removeparameter method ([9ccba5c](https://github.com/lucadimascolo/socca/commit/9ccba5c2597921a1c46ddc9ddf795724c979200e))
* export truncation module from radial subpackage ([3e67d3a](https://github.com/lucadimascolo/socca/commit/3e67d3a1ad20db085efddf09a992508422f3cf61))


### Bug Fixes

* accept FITS and arrays in getsigma ([b701863](https://github.com/lucadimascolo/socca/commit/b70186316dbf274c145f17954ab5d23ae3dc0c08))


### Documentation

* add truncation and split model sections ([d32b339](https://github.com/lucadimascolo/socca/commit/d32b3397aa41629359ca98f5305d298c2d1cae90))
