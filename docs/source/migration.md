# Migration notes

Some model conventions changed after **``socca``** 1.4.0. To reuse parameter values from an earlier fit, convert them as below.

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
