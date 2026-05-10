# Changelog

## [1.2.0](https://github.com/lucadimascolo/socca/compare/v1.1.0...v1.2.0) (2026-05-10)


### Features

* add support for pocomc vectorization ([776d104](https://github.com/lucadimascolo/socca/commit/776d104714e53888ccabb19b45d448acb270fcd0))


### Bug Fixes

* add guard for low=high values in (log)uniform priors ([08213e1](https://github.com/lucadimascolo/socca/commit/08213e1f170ade2227a6366e09076aa9b8604391))
* add missing raise for ValueError ([886a770](https://github.com/lucadimascolo/socca/commit/886a770e6db28b7f017284cc572954cb66e4a419))
* add support to optimizer in initialization and plotting ([2d3b851](https://github.com/lucadimascolo/socca/commit/2d3b85145494241202c3af9b35f4e1399393fed9))
* avoid per-sample GPU-to-CPU transfers in likelihood evaluation ([cb4af4f](https://github.com/lucadimascolo/socca/commit/cb4af4f8737585551fd7d973799b4ddf008a78d6))
* change to 0-indexed cutout re-centering for psf model ([112b6f1](https://github.com/lucadimascolo/socca/commit/112b6f1c07659f4777770b22ad98c12d0a941b13))
* correct CDELT2 calculation from CDX_2 values ([0dfeab7](https://github.com/lucadimascolo/socca/commit/0dfeab706e560b6133bde1ee3b70a79dd1018f51))
* initialize mbkg to empty list ([a6d200a](https://github.com/lucadimascolo/socca/commit/a6d200ac30472eed7ea32579c0e393ed247399dd))
* pin jax/jaxlib below 0.10.0 to avoid numpyro incompatibility ([4378cd1](https://github.com/lucadimascolo/socca/commit/4378cd1216f0cc87d913eb7eba6899e04b13495e))
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
