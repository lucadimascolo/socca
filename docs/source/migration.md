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
