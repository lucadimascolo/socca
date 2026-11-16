# Migration notes

Some model conventions changed after **``socca``** 1.4.0. Models and fits saved with 1.4.0 or earlier cannot be loaded by this version: re-run the fit, or load them with 1.4.0 (`pip install socca==1.4.0`). To reuse parameter values from an earlier fit, convert them as below.

## `Disk`: radial amplitude

The vertical profiles are now normalized to unit integral along $z$, so that a face-on disk projects exactly onto its radial profile and the radial amplitude (e.g. `Is` of an `Exponential`) is the face-on surface brightness. Before, the vertical profiles had unit peak, so the radial amplitude was a 3D emissivity, degenerate with the scale height.

To convert an amplitude from an earlier fit, multiply it by the integral of the old vertical profile, with $z_s$ in degrees:

| Vertical profile | Factor |
| --- | --- |
| `HyperSecantHeight` ($\alpha = 2$, default) | $2 z_s$ |
| `HyperSecantHeight` (any $\alpha$) | $z_s\,B(\alpha/2, 1/2)$ |
| `ExponentialHeight` | $2 z_s$ |

## `Ellipsoid`: rotation

With $\mathrm{rot} = 0$, the major axis of an `Ellipsoid` now lies along the line of nodes, so $\theta$ is also its position angle, as for the other profiles. Before, it was perpendicular to it. To convert a rotation from an earlier fit, subtract $\pi/2$ from it: the maps agree exactly.

## `Ellipsoid`: amplitude

The density of an `Ellipsoid` is now its radial profile divided by $2\,(1 - e_{\mathrm{ratio}}\,e)\,r_s$, twice its shortest semi-axis at the scale radius $r_s$ of the profile (e.g. `re` for `Sersic`, `rs` for `Exponential`), in the same way a `Disk` divides its radial profile by the thickness of its vertical profile. The amplitude (e.g. `Ie`) is thus a surface brightness: seen face-on, an `Exponential` or `TopHat` ellipsoid has exactly that central surface brightness, and other profiles a value of the same order. Before, the radial profile was the density itself. A `CustomProfile`, having no scale radius, is still used as the density.

To convert an amplitude from an earlier fit, multiply it by $2\,(1 - e_{\mathrm{ratio}}\,e)\,r_s$, with $r_s$ in degrees.

## `gNFW` in `Disk` and `Ellipsoid`

Inside a `Disk` or an `Ellipsoid`, a `gNFW` profile now contributes its 3D density, normalized to `Ic` at $r = r_c$, instead of its projection. Fits of these models with an earlier version have no exact conversion.

## `Ellipsoid`: line-of-sight integration

`losdepth` now defaults to infinity and, if finite, caps the ellipsoidal radius $m$ rather than the line-of-sight coordinate $|z|$. `losbins` is now the number of quadrature nodes (default 128, was 200 trapezoid bins). Scripts that set a finite `losdepth` as a half-depth along the line of sight are not caught: remove it, or set it to the radius beyond which the density is negligible.

## Component ids and ties

A component's id is now its position in the model (`comp_00`, `comp_01`, ...), or the `name=` given to it, instead of a global counter of created components. Model keys, `fit.labels` and the argument names of lambda ties follow these ids, so they can differ from those of earlier versions.

Lambda ties are deprecated: use `socca.priors.function(func, (component, "name"), ...)`, which does not depend on argument names. Function and lambda ties now take and return parameters in their declared units (see `set_units`), while `boundto` copies the physical value. The string form of `boundto` was removed.

## Units and periods

Units declared with `set_units` on a sub-profile of a `Disk`, `Ellipsoid` or `Bridge`, or on a wrapper with a dotted name (e.g. `disk.set_units(**{"radial.re": "arcsec"})`), are now applied; before, they were ignored. A standalone `getmap()` also honours declared units.

Periods follow declared units (180 for an angle declared in degrees), and samplers treat a prior as periodic only when it spans a whole number of periods. The span warning is issued when the component is added to a model.

## Priors

`LogPowerLaw` was removed: use `powerlaw(alpha, low, high)`. `uniform()` refuses infinite limits.

## Truncations

Truncations of the sub-profiles of a `Disk`, `Ellipsoid` or `Bridge` are now applied; before, they were ignored. A truncation's `rt` and `wt` must be strictly positive and finite: to disable a truncation, set the profile's `truncation` to `None`.

## `Bridge`

The sub-profiles are now evaluated on absolute distances from the bridge axes, so Bridges with `Exponential`-family, `gNFW`, `Power` or `Sersic` sub-profiles, which could give invalid values before, and Bridges with truncated sub-profiles give different maps. The scale radius and amplitude of the sub-profiles are set by the Bridge from `rs`, `e` and `Is`, and setting them raises an error.

## Other changes

- A `Disk` drops the `e` and `cbox` of its radial profile, which it never used. Its `losdepth` must be finite, and `losbins` must be an integer of at least 2.
- `Disk.getgrid` takes the line-of-sight nodes `z` instead of `losdepth` and `losbins`, and `Ellipsoid.getgrid` returns the line-of-sight geometry `(rmin, a)`.
- A profile belongs to one component: wrapping it twice, wrapping a profile that is already in a model, or adding a wrapped profile to a model raises an error. Use `copy.copy` of the profile instead (a deep copy keeps the component id).
- A fitter raises an error if its model changed after the fitter was created: create a new fitter instead.
