# Building a composite model
This section provides a brief overview of all the model components currently implemented in **``socca``**. These can be used to build complex models by combining multiple components together:

```python
>>> from socca.models import Model
>>> model = Model()
>>> model.addcomponent(...)  # Add first component
>>> model.addcomponent(...)  # Add second component
>>> # ... add more components as needed ...
```

A more extensive example of how to build and fit composite models to imaging data can be found in the ["Getting started"](./tutorial_quickstart.md) tutorial.

## Available model components

A quick list of all available model components can be obtained by calling the `zoo()` function:

```python
>>> from socca.models import zoo
>>> zoo()
Beta
gNFW
Sersic
Gaussian
Exponential
PolyExponential
PolyExpoRefact
ModExponential
Point
Background
Disk
```

### Radial profiles
A major class of model components implemented in **``socca``** are those based on radial profiles. These are two-dimensional surface brightness distributions that are axially symmetric around a centroid position, and whose radial dependence is described by a specific functional form. The radial profiles can be further generalized to account for projected ellipticity, boxiness, and position angle, allowing for a more flexible description of the source morphology.

Each radial profile component inherits from the abstract base class `Profile`, which defines the common interface and geometric parameters shared by all radial profiles. These include the centroid coordinates (`xc`, `yc`), position angle (`theta`, measured east from north), axis ratio (`e`), and boxiness (`cbox`).

```python
>>> from socca.models import Profile
>>> comp = Profile()
>>> comp.parameters()

Model parameters
================
xc    [deg] : None       | Right ascension of centroid
yc    [deg] : None       | Declination of centroid
theta [rad] : 0.0000E+00 | Position angle (east from north)
e        [] : 0.0000E+00 | Projected axis ratio
cbox     [] : 0.0000E+00 | Projected boxiness
```

These geometric parameters are used internally to build a generalized radial distance grid that accounts for ellipticity, position angle, and boxiness. For each pixel in the image, the distance from the centroid is computed as:

$$
r = \left( |x'|^{2+c} + \left|\frac{y'}{1-e}\right|^{2+c} \right)^{\frac{1}{2+c}},
$$

where $(x', y')$ are the rotated coordinates:

$$
\begin{bmatrix}
x' \\
y'
\end{bmatrix}
=
\begin{bmatrix}
-\cos(y_c)\sin(\theta) & -\cos(\theta) \\
\hspace{10pt}\cos(y_c)\cos(\theta) & -\sin(\theta)
\end{bmatrix}
\begin{bmatrix}
x - x_c \\
y - y_c
\end{bmatrix}
$$

This formulation ensures that the rotation is correctly applied in spherical coordinates, accounting for the cosine projection factor at the centroid declination $y_c$. The boxiness parameter $c$ allows for deviations from pure elliptical isophotes, with $c=0$ corresponding to perfect ellipses, while $c > 0$ produces boxy shapes and $c < 0$ yields disky isophotes (but $c > -1$ to ensure physical validity). 

```{caution}
`Profile` is an abstract base class and does not provide an implementation of any specific radial profile. All the radial profile components, however inherit from this base class and therefore share the same set of geometric parameters listed above.
```

#### Beta
The `Beta` class implements the $\beta$ model ([Cavaliere &  Fusco-Femiano 1976](https://ui.adsabs.harvard.edu/abs/1976A%26A....49..137C/abstract)) commonly used to describe the projected surface brightness profile of galaxy clusters under the assumption of an isothermal intracluster medium. The functional form of the profile is given by:

$$
I(r) = I_c \left(1 + \frac{r^2}{r_c^2}\right)^{-\beta}
$$

```python
>>> from socca.models import Beta
>>> comp = Beta()
>>> comp.parameters()

Model parameters
================
xc    [deg] : None       | Right ascension of centroid
yc    [deg] : None       | Declination of centroid
theta [rad] : 0.0000E+00 | Position angle (east from north)
e        [] : 0.0000E+00 | Projected axis ratio
cbox     [] : 0.0000E+00 | Projected boxiness
rc    [deg] : None       | Core radius
Ic  [image] : None       | Central surface brightness
beta     [] : 5.5000E-01 | Slope parameter
```

#### gNFW
The `gNFW` class implements the generalized Navarro-Frenk-White (gNFW) profile ([Nagai et al. 2007](https://ui.adsabs.harvard.edu/abs/2007ApJ...668....1N/abstract), [Mroczkowski et al. 2009](https://ui.adsabs.harvard.edu/abs/2009ApJ...694.1034M/abstract)) commonly used to describe the pressure profile of the intracluster medium in galaxy clusters. The projected two-dimensional profile is obtained by integrating the three-dimensional gNFW profile along the line of sight. In **``socca``**, this is implemented as follows:

$$
I(r) = I_c r_c \int_{r/r_c}^{+\infty} x^{-\gamma}(1+x^{\alpha})^{\frac{\gamma - \beta}{\alpha}} \tfrac{x}{\sqrt{x^2 - x_r^2}} \mathrm{d}x,
$$  

where $x$ is a dimensionless radius in units of the core radius $r_c$. The parameters $\alpha$, $\beta$, and $\gamma$ control the intermediate, outer, and inner slopes of the profile, respectively. The normalization $I_c$ has units of surface brightness in units of the input data per unit length.

The integral is computed numerically using Gauss-Kronrod adaptive quadrature, as implemented in [`quadax`](https://github.com/f0uriest/quadax). To improve computational efficiency, the profile is evaluated over a logarithmically spaced grid of dimensionless radii and then interpolated onto the two-dimensional coordinate grid on the fly during model evaluation. 

```python
>>> from socca.models import gNFW
>>> comp = gNFW()
>>> comp.parameters()

Model parameters
================
xc    [deg] : None       | Right ascension of centroid
yc    [deg] : None       | Declination of centroid
theta [rad] : 0.0000E+00 | Position angle (east from north)
e        [] : 0.0000E+00 | Projected axis ratio
cbox     [] : 0.0000E+00 | Projected boxiness
rc    [deg] : None       | Scale radius
Ic  [image] : None       | Characteristic surface brightness
alpha    [] : 1.0510E+00 | Intermediate slope
beta     [] : 5.4905E+00 | Outer slope
gamma    [] : 3.0810E-01 | Inner slope
```

(sersic)=
#### Sersic
The `Sersic` class implements the Sérsic profile ([Sérsic 1963](https://ui.adsabs.harvard.edu/abs/1963BAAA....6...41S)), a widely used empirical model for describing the surface brightness distribution of galaxies and other extended sources. The functional form of the profile is given by:

$$
I(r) = I_e \exp\left\{-b_n \left[ \left( \frac{r}{r_e} \right)^{1/n} - 1 \right]\right\},
$$

where $r_e$ is the effective radius enclosing half of the total flux, $I_e$ is the surface brightness at $r_e$, and $n$ is the Sérsic index controlling the concentration of the profile. The constant $b_n$ is defined such that $\Gamma(2n) = 2\gamma(2n, b_n)$, where $\Gamma$ and $\gamma$ are the complete and incomplete Gamma functions, respectively. For computational efficiency, the value of $b_n$ is interpolated at each step of the inference from a precomputed lookup table covering the range $n \in [0.25, 10]$.

The Sérsic profile encompasses several commonly used models as special cases, including the Gaussian ($n=0.5$), the exponential ($n = 1$), and the de Vaucouleurs ($n = 4$) profiles.

```python
>>> from socca.models import Sersic
>>> comp = Sersic()
>>> comp.parameters()

Model parameters
================
xc    [deg] : None       | Right ascension of centroid
yc    [deg] : None       | Declination of centroid
theta [rad] : 0.0000E+00 | Position angle (east from north)
e        [] : 0.0000E+00 | Projected axis ratio
cbox     [] : 0.0000E+00 | Projected boxiness
re    [deg] : None       | Effective radius
Ie  [image] : None       | Surface brightness at re
ns       [] : 5.0000E-01 | Sersic index
```

#### Gaussian
The `Gaussian` class implements a Gaussian surface brightness profile. The functional form of the profile is given by:

$$
I(r) = I_s \exp\left\{-\frac{1}{2}\left(\frac{r}{r_s}\right)^2\right\},
$$

where $I_s$ is the central surface brightness and $r_s$ is the scale radius (standard deviation of the Gaussian). The Gaussian profile is equivalent to a Sérsic profile with Sérsic index $n = 0.5$. The half-width at half-maximum (HWHM) is approximately $1.177 \, r_s$. Still, it is provided as a separate component for computational convenience and efficiency.

```python
>>> from socca.models import Gaussian
>>> comp = Gaussian()
>>> comp.parameters()

Model parameters
================
xc    [deg] : None       | Right ascension of centroid
yc    [deg] : None       | Declination of centroid
theta [rad] : 0.0000E+00 | Position angle (east from north)
e        [] : 0.0000E+00 | Projected axis ratio
cbox     [] : 0.0000E+00 | Projected boxiness
rs    [deg] : None       | Scale radius
Is  [image] : None       | Central surface brightness
```

#### Exponential
The `Exponential` class implements a simple exponential surface brightness profile, commonly used to describe galactic disks and other extended systems with smoothly declining radial profiles:

$$
I(r) = I_s \exp\left(-\frac{r}{r_s}\right),
$$

where $I_s$ is the central surface brightness and $r_s$ is the exponential scale radius. The exponential profile is equivalent to a Sérsic profile with Sérsic index $n = 1$, $I_e=I_s \exp\left({-b_{1}}\right)$, and $r_e = b_{1} r_s$, where $b_{1} \approx 1.67835$. As for `Gaussian`, this is implemented as a separate component for convenience.

```python
>>> from socca.models import Exponential
>>> comp = Exponential()
>>> comp.parameters()

Model parameters
================
xc    [deg] : None       | Right ascension of centroid
yc    [deg] : None       | Declination of centroid
theta [rad] : 0.0000E+00 | Position angle (east from north)
e        [] : 0.0000E+00 | Projected axis ratio
cbox     [] : 0.0000E+00 | Projected boxiness
rs    [deg] : None       | Scale radius
Is  [image] : None       | Central surface brightness
```

(polyexponential)=
#### PolyExponential
The `PolyExponential` class implements a generalized exponential surface brightness profile in which the radial decay is modulated by a polynomial function of radius. This profile was introduced by [Mancera Piña et al. (2024)](https://ui.adsabs.harvard.edu/abs/2024A%26A...689A.344M). The functional form of this model is given by:

$$
I(r) = I_s \left[ 1
+ \sum_{k=1}^{4} c_k \left(\frac{r}{r_c}\right)^k
\right] \exp\!\left(-\frac{r}{r_s}\right).
$$

where $I_s$ is the central surface brightness, $r_s$ is a characteristic scale radius, and the coefficients $\{c_k\}$ define the polynomial modulation of the exponential decay. This parameterization allows for flexible deviations from a pure exponential profile while preserving a smooth radial behavior.

```python
>>> from socca.models import PolyExponential
>>> comp = PolyExponential()
>>> comp.parameters()

Model parameters
================
xc    [deg] : None       | Right ascension of centroid
yc    [deg] : None       | Declination of centroid
theta [rad] : 0.0000E+00 | Position angle (east from north)
e        [] : 0.0000E+00 | Projected axis ratio
cbox     [] : 0.0000E+00 | Projected boxiness
rs    [deg] : None       | Scale radius
Is  [image] : None       | Central surface brightness
c1       [] : 0.0000E+00 | Polynomial coefficient 1
c2       [] : 0.0000E+00 | Polynomial coefficient 2
c3       [] : 0.0000E+00 | Polynomial coefficient 3
c4       [] : 0.0000E+00 | Polynomial coefficient 4
rc    [deg] : 2.7778E-04 | Reference radius for polynomial terms
```

(polyexporefact)=
#### PolyExpoRefact
The `PolyExpoRefact` class provides a refactored polynomial–exponential profile that is mathematically equivalent to `PolyExponential` but expressed in a form that improves parameter interpretability and reduces the degeneracies among the different polynomial terms:

$$
I(r) = \left[I_s+\sum_{k=1}^{4} I_k \left(\frac{r}{r_s}\right)^k\right] \exp\!\left(-\frac{r}{r_s}\right),
$$

```python
>>> from socca.models import PolyExpoRefact
>>> comp = PolyExpoRefact()
>>> comp.parameters()

Model parameters
================
xc    [deg] : None       | Right ascension of centroid
yc    [deg] : None       | Declination of centroid
theta [rad] : 0.0000E+00 | Position angle (east from north)
e        [] : 0.0000E+00 | Projected axis ratio
cbox     [] : 0.0000E+00 | Projected boxiness
rs    [deg] : None       | Scale radius
Is  [image] : None       | Central surface brightness
I1  [image] : 0.0000E+00 | Polynomial intensity coefficient 1
I2  [image] : 0.0000E+00 | Polynomial intensity coefficient 2
I3  [image] : 0.0000E+00 | Polynomial intensity coefficient 3
I4  [image] : 0.0000E+00 | Polynomial intensity coefficient 4
rc    [deg] : 2.7778E-04 | Reference radius for polynomial terms
```

#### ModExponential
The `ModExponential` class implements a modified exponential surface brightness profile in which the exponential term is altered by a power-law modification at large radii. The functional form of this model is given by:

$$
I(r) = I_s \exp\!\left(-\frac{r}{r_s}\right) \left(1 + \frac{r}{r_m}\right)^{\alpha}.
$$

```python
>>> from socca.models import ModExponential
>>> comp = ModExponential()
>>> comp.parameters()

Model parameters
================
xc    [deg] : None       | Right ascension of centroid
yc    [deg] : None       | Declination of centroid
theta [rad] : 0.0000E+00 | Position angle (east from north)
e        [] : 0.0000E+00 | Projected axis ratio
cbox     [] : 0.0000E+00 | Projected boxiness
rs    [deg] : None       | Scale radius
Is  [image] : None       | Central surface brightness
rm    [deg] : None       | Modification radius
alpha    [] : None       | Modification exponent
```

#### Custom radial profiles
In case none of the built-in `Profile` subclasses adequately describe the desired surface-brightness distribution, **``socca``** provides a simple interface for defining custom radial profiles through the `CustomProfile` class. In such a case, the user must provide two key ingredients:
- A list of parameter specifications (`parameters`), each in the form of a dictionary comprising the `name`, `unit`, and `description` fields for the corresponding variable. Each parameter specified in this way is automatically registered as an attribute of the model and integrated into the standard **``socca``** parameter handling. 
- A callable profile function (`profile`). The signature of the callable profile function must include the radial coordinate `r` as the first argument, followed by all the model parameters with name matching the argument names matching those specified in the parameter list. The function must return the surface brightness at radius `r` given the input parameter values. For consistency with the modelling framework, the profile function must be implemented using [`JAX`](https://jax.readthedocs.io/en/latest/) operations to ensure compatibility with `JAX`'s just-in-time compilation and automatic differentiation capabilities.

```python
>>> import jax.numpy as jp
>>> from socca.models import CustomProfile
>>>
>>> def custom_profile(r,scale):
>>>     return scale * jp.exp(-r)
>>>
>>> parameters = [{'name':'scale',
>>>                'unit':'image',
>>>                'description':'Scaling factor'}]
>>>
>>> comp = CustomProfile(parameters=parameters, profile=custom_profile)
>>> comp.parameters()

Model parameters
================
xc      [deg] : None       | Right ascension of centroid
yc      [deg] : None       | Declination of centroid
theta   [rad] : 0.0000E+00 | Position angle (east from north)
e          [] : 0.0000E+00 | Projected axis ratio
cbox       [] : 0.0000E+00 | Projected boxiness
scale [image] : None       | Scaling factor
```

### Disk with finite thickness
The `Disk` class implements a three-dimensional model for describing the emission from disk galaxies, including their finite thickness and inclination effects, particularly relevant when modeling edge-on systems. This is achieved by modeling the disk component by combining a radial surface brightness profile with a vertical structure model. The radial profile $s(r)$ describes the distribution of light in the disk plane, while a vertical profile $h(r,z)$ accounts for the thickness and inclination of the disk along the line of sight. The overall surface brightness distribution is obtained by integrating the three-dimensional model along the line of sight:

$$
I(r) = \int_{-l_{\mathrm{max}}}^{+l_{\mathrm{max}}} s(R) \, h(R,z) \, \mathrm{d}l,
$$

where $R$ and $z$ are the cylindrical coordinates in the disk frame, while $r$ and $l$ denote the projected radius and line-of-sight coordinate in the observer frame, respectively. The transformation between the two coordinate systems depends on the inclination angle of the disk with respect to the line-of-sight direction. A detailed discussion of the coordinate transformations involved in this calculation can be found in [van Asselt et al. (2026)](https://scixplorer.org/abs/2026arXiv260103339V).

```python
>>> from socca.models import Disk
>>> comp = Disk()
>>> comp.parameters()

Model parameters
================
radial.xc         [deg] : None       | Right ascension of centroid
radial.yc         [deg] : None       | Declination of centroid
radial.theta      [rad] : 0.0000E+00 | Position angle (east from north)
radial.e             [] : 0.0000E+00 | Projected axis ratio
radial.cbox          [] : 0.0000E+00 | Projected boxiness
radial.re         [deg] : None       | Effective radius
radial.Ie       [image] : None       | Surface brightness at re
radial.ns            [] : 5.0000E-01 | Sersic index
vertical.inc      [rad] : 0.0000E+00 | Inclination angle (0=face-on)
vertical.zs       [deg] : None       | Scale height

Hyperparameters
===============
vertical.losdepth [deg] : 2.7778E-03 | Half line-of-sigt extent for integration
vertical.losbins     [] : 2.0000E+02 | Number of points for line-of-sight integration
```

If compared to any of the radial profile components described in the previous section, it is possible to observe many differences in the model and parameter interface. First of all, the parameters are split between `radial` and `vertical` subgroups. The former contains all the parameters defining the radial profile on the disk plane. These are inherited from the reference radial profile class specified when initializing the `Disk` component. As it will be shown later on in more details, any of the radial profiles described in the previous section can be used. The `vertical` subset instead includes the parameters related to the vertical structure (including the inclination of the disk plane with respect to the line-of-sight direction, `inc`). More details on the specific meaning of each parameter will be provided below. As for the `radial` component, the `vertical` structure can be customized by the user.

Second, the `Disk` class implements a set of hyperparameters that do not enter the inference process, but tune key aspects of the model generation. In this specific case, they control the numerical accuracy of the line-of-sight integration. To ensure computational efficiency, the integration is currently computed via a trapezoid method (as implemented in [jax.numpy.trapezoid](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.trapezoid.html)) over a arbitrary finite extent $[-l_{\mathrm{max}}, +l_{\mathrm{max}}]$. The `Height` hyperparameters include the half line-of-sight extent `losdepth` (i.e., the $l_{\mathrm{max}}$ parameter in the equation above) and the number of integration points `losbins`. Increasing these values improves the accuracy of the model, at the cost of increased computational time.

```{important}
When accessing a specific variable, it is important to remember that both the radial and vertical parameters are stored as sub-attributes of the `radial` and `vertical` attributes of the `Disk` component, respectively. For instance, to access the effective radius parameter of the radial profile, one must use `comp.radial.re`, while the scale height of the vertical structure is stored in `comp.vertical.zs`.
```

#### Vertical profiles
The hyperparameters introduced above as well as the inclincation angle `inc` are inherited by `Disk` from the model for the vertical structures. These derived from a dedicated abstract base class, `Height`, which defines the common interface for all vertical profiles.

```python
>>> from socca.models import Height
>>> comp = Height()
>>> comp.parameters()

Model parameters
================
inc      [rad] : 0.0000E+00 | Inclination angle (0=face-on)

Hyperparameters
===============
losdepth [deg] : 2.7778E-03 | Half line-of-sigt extent for integration
losbins     [] : 2.0000E+02 | Number of points for line-of-sight integration
```

Currently, **``socca``** implements two vertical profiles: the power of a hyperbolic secant profile (`HyperSecantHeight`) and the exponential profile (`ExponentialHeight`). 

```{caution}
Please note that the `Height` subclasses can not be used as standalone model components, as they do not provide a complete two-dimensional surface brightness distribution. They can only be used in combination with a radial profile within the `Disk` component.
```

##### HyperSecantHeight
The `HyperSecantHeight` class implements a hyperbolic secant vertical profile raised to a power $\alpha$:

$$
h(z) = \mathrm{sech}^{\alpha}\left(|z|/z_s\right).
$$

Here, $z_s$ is the scale height of the disk, and $\alpha$ is an exponent that controls the sharpness of the vertical profile. By default, $\alpha = 2$, corresponding to the squared hyperbolic secant profile commonly used to describe the vertical structure of disk galaxies ([van der Kruit & Searle 1981](https://ui.adsabs.harvard.edu/abs/1981A&A....95..105V)).

```python
>>> from socca.models import HyperSecantHeight
>>> comp = HyperSecantHeight()
>>> comp.parameters()

Model parameters
================
inc      [rad] : 0.0000E+00 | Inclination angle (0=face-on)
zs       [deg] : None       | Scale height
alpha       [] : 2.0000E+00 | Exponent to the hyperbolic secant

Hyperparameters
===============
losdepth [deg] : 2.7778E-03 | Half line-of-sigt extent for integration
losbins     [] : 2.0000E+02 | Number of points for line-of-sight integration
```

##### ExponentialHeight
The `ExponentialHeight` class implements an exponential vertical profile. The functional form of this profile is given simply by:

$$
h(z) = \exp\left(-|z|/z_s\right).
$$

As in the previous case, $z_s$ is the scale height of the disk.

```python
>>> from socca.models import ExponentialHeight
>>> comp = ExponentialHeight()

Model parameters
================
inc      [rad] : 0.0000E+00 | Inclination angle (0=face-on)
zs       [deg] : None       | Scale height

Hyperparameters
===============
losdepth [deg] : 2.7778E-03 | Half line-of-sigt extent for integration
losbins     [] : 2.0000E+02 | Number of points for line-of-sight integration
```

#### Customizing the radial and vertical components
By default, the radial profile is modeled using a Sérsic profile, while the vertical structure is described by a squared hyperbolic secant function (`HyperSecantHeight` with $\alpha=2$). As mentioned above, both the radial and vertical profiles can however be customized by providing alternative profile classes when instantiating the `Disk` component. 

```python
>>> from socca.models import Disk, Exponential, ExponentialHeight
>>> comp = Disk(radial=Exponential(), vertical=ExponentialHeight())
```

### Other components
**``socca``** also includes a few additional model components that do not fall into the radial profile category or its finite-thickness disk generalization.

#### Point source
The `Point` class implements a point-like source, representing an unresolved object whose intrinsic spatial extent is negligible compared to the instrumental resolution. The surface brightness distribution is modeled as a Dirac delta function centered at the source position,

$$
I(x_c,y_c) = I_c\,\delta(x-x_c)\delta(y-y_c),
$$

where $I_{c}$ is the peak surface brightness (or total flux) and $(x_c,y_c)$ denote the source coordinates. 

```python
>>> from socca.models import Point
>>> comp = Point()
>>> comp.parameters()

Model parameters
================
xc   [deg] : None       | Right ascension
yc   [deg] : None       | Declination
Ic [image] : None       | Peak surface brightness
```

```{caution}
To allow for sub-pixel positioning of the point source, the `Point` component is computed directly in Fourier space as a constant function across all spatial frequencies of amplitude $I_c$ multipltied by a complex phase factor encoding the source position. The resulting model is then multiplied by the Fourier transform of the instrumental PSF, and transformed back to image space for comparison with the data. For this reason, fitting a point source without providing a PSF model might lead to artifacts in the resulting image.
```

#### Background
The `Background` class implements a two-dimensional polynomial background model defined on the image plane. This is intended to model large-scale background variations due to instrumental effects, sky emission, or residual systematics not captured by other model components. The surface brightness is modeled as a Cartesian polynomial in the sky coordinates $(x, y)$, expressed in units of a characteristic scale $r_s$:

$$
\begin{aligned}
I(x,y) =\;& a_0
+ a_{1x}\,\frac{x}{r_s}
+ a_{1y}\,\frac{y}{r_s} \\
&+ a_{2xx}\,\left(\frac{x}{r_s}\right)^2
+ a_{2xy}\,\frac{x y}{r_s^2}
+ a_{2yy}\,\left(\frac{y}{r_s}\right)^2 \\
&+ a_{3xxx}\,\left(\frac{x}{r_s}\right)^3
+ a_{3xxy}\,\frac{x^2 y}{r_s^3}
+ a_{3xyy}\,\frac{x y^2}{r_s^3}
+ a_{3yyy}\,\left(\frac{y}{r_s}\right)^3.
\end{aligned}
$$

```python
>>> from socca.models import Background
>>> comp = Background()
>>> comp.parameters()

Model parameters
================
rs [deg] : 2.7778E-04 | Reference radius for polynomial terms
a0    [] : None       | Polynomial coefficient 0
a1x   [] : 0.0000E+00 | Polynomial coefficient 1 in x
a1y   [] : 0.0000E+00 | Polynomial coefficient 1 in y
a2xx  [] : 0.0000E+00 | Polynomial coefficient 2 in x*x
a2xy  [] : 0.0000E+00 | Polynomial coefficient 2 in x*y
a2yy  [] : 0.0000E+00 | Polynomial coefficient 2 in y*y
a3xxx [] : 0.0000E+00 | Polynomial coefficient 3 in x*x*x
a3xxy [] : 0.0000E+00 | Polynomial coefficient 3 in x*x*y
a3xyy [] : 0.0000E+00 | Polynomial coefficient 3 in x*y*y
a3yyy [] : 0.0000E+00 | Polynomial coefficient 3 in y*y*y
```

```{warning}
By construction, the `Background` model is not convolved with the instrumental PSF, as it is assumed to vary smoothly on scales much larger than the PSF size. 
```

## Obtaining images from models
Both the `Model` and individual component classes provide a `getmap()` method to generate two-dimensional model images evaluated on a specified coordinate grid.
This method requires an `Image` object (see the ["Loading data"](./tutorial_data.md) guide) to define the coordinate grid and, optionally, the PSF for convolution. Depending on whether it is called on an individual component or a composite model, the behavior of `getmap()` differs slightly.

### Individual components
For a single model component, the `getmap()` method returns the two-dimensional surface brightness distribution evaluated on the image grid:

```python
>>> from socca.models import Sersic
>>> from socca.data import Image
>>>
>>> # Load an image to define the coordinate grid
>>> img = Image('path/to/your/image.fits')
>>>
>>> # Create and configure a Sérsic component
>>> comp = Sersic()
>>> comp.xc = img.hdu.header['CRVAL1']
>>> comp.yc = img.hdu.header['CRVAL2']
>>> comp.re = 1.00E-03  # degrees
>>> comp.Ie = 1.00      # image units
>>> comp.ns = 1.00      # Sérsic index
>>>
>>> # Generate the model map
>>> model_map = comp.getmap(img)
```

By default, `getmap()` returns the unconvolved model. To obtain the PSF-convolved model (assuming a PSF has been provided when loading the image), set the `convolve` argument to `True`:

```python
>>> convolved_map = comp.getmap(img, convolve=True)
```

```{important}
All model parameters must be assigned numerical values before calling `getmap()`. If any parameter is set to `None` or contains a prior distribution instead of a fixed value, an error will be raised.
```

### Composite models
For composite models built using the `Model` class, the `getmap()` method works similarly to individual components when all parameters are assigned fixed numerical values:

```python
>>> model_map = model.getmap(img)                    # Unconvolved model
>>> model_map = model.getmap(img, convolve=True)     # PSF-convolved model
```

By default, the background component is not included in the output. To include it in the convolved model, set `addbackground=True`:

```python
>>> model_map = model.getmap(img, convolve=True, addbackground=True)
```

Clearly, since any **``socca``** composite model is a linear combination of multiple components, the resulting model image is equivalent to the sum of the individual component images (after convolution with the PSF, if requested):

```python
>>> # Create individual components
>>> comp1 = Sersic()
>>> comp1.xc, comp1.yc = 0.00, 0.00
>>> comp1.re, comp1.Ie, comp1.ns = 1.00E-03, 1.00E-01, 1.00
>>>
>>> comp2 = Exponential()
>>> comp2.xc, comp2.yc = 0.00, 0.00
>>> comp2.rs, comp2.Is = 2.00E-03, 5.00E-01
>>>
>>> # Build a composite model
>>> model = Model()
>>> model.addcomponent(comp1)
>>> model.addcomponent(comp2)
>>>
>>> # These two are equivalent:
>>> composite_map = model.getmap(img)
>>> summed_map = comp1.getmap(img) + comp2.getmap(img)
```

```{note}
If any parameter in the composite model is assigned a prior distribution rather than a fixed value, `getmap()` will raise an error. In such cases, use the `getmodel()` method instead, which accepts a parameter array as input.
```

### Accessing the inference model

During the fitting process, **``socca``** uses the `getmodel()` method internally, which requires a parameter array and returns multiple outputs:

```python
>>> model_raw, model_conv, model_background, \
...     mask_negative = model.getmodel(img, params)
```

where:
- `model_raw`: the raw (unconvolved) model, excluding background components
- `model_conv`: the smoothed (PSF-convolved) model, including all components
- `model_background`: the background component only
- `mask_negative`: a mask indicating pixels where positivity constraints were violated

The `params` array must contain the parameter values in the order expected by the model. This order is determined by how components were added to the model and can be inspected via the `parameters()` method of the `Model` object:

```python
>>> from socca.models import Model, Sersic, Exponential
>>> import numpyro.distributions as dist
>>>
>>> # Build a composite model with priors
>>> comp1 = Sersic()
>>> comp1.xc, comp1.yc = 0.00, 0.00
>>> comp1.theta, comp1.e, comp1.cbox = 0.00, 0.00, 0.00
>>> comp1.re = dist.LogUniform(1.00E-04, 1.00E-02)
>>> comp1.Ie = dist.LogUniform(1.00E-02, 1.00E+02)
>>> comp1.ns = 1.00
>>>
>>> model = Model()
>>> model.addcomponent(comp1)
>>>
>>> model.parameters() # Print all parameters (fixed and free)

Model parameters
================
comp_00_xc    [deg] : 0.0000E+00
comp_00_yc    [deg] : 0.0000E+00
comp_00_theta [rad] : 0.0000E+00
comp_00_e        [] : 0.0000E+00
comp_00_cbox     [] : 0.0000E+00
comp_00_re    [deg] : Distribution: LogUniform
comp_00_Ie  [image] : Distribution: LogUniform
comp_00_ns       [] : 1.0000E+00

>>> model.parameters(freeonly=True) # Print only free parameters (those with priors)

Model parameters
================
comp_00_re    [deg] : Distribution: LogUniform
comp_00_Ie  [image] : Distribution: LogUniform
```

Each label follows the pattern `<component_id>_<parameter_name>`, where `component_id` is the default name (e.g., `comp_00`, `comp_01`, ...) assigned by **``socca``** when ini
initializing the corresponding component. This ordering is consistent throughout the inference process and must be respected when manually constructing parameter arrays for `getmodel()`.

In practice, when working with fitted models, the `fitter` object provides the more convenient `getmodel()` method (see the ["Running the model inference"](./tutorial_fitting.md) tutorial), which automatically uses the best-fit or median posterior parameters.