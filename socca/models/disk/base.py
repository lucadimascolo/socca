"""Disk model with finite thickness."""

from functools import partial
import inspect
import types

import warnings

import jax
import jax.numpy as jp
import numpyro.distributions

import numpy as np

from ..base import Component
from ..radial import Sersic
from .vertical import HyperSecantHeight


class Disk(Component):
    """
    3D disk model combining radial and vertical profiles with inclination.

    Creates a realistic disk galaxy model by combining a radial surface brightness
    profile with a vertical density distribution. The model accounts for disk
    inclination via line-of-sight integration.

    Parameters
    ----------
    radial : Profile, optional
        Radial profile defining in-plane surface brightness distribution.
        Default is Sersic(). Can be any 2D profile (Beta, Exponential, etc.).
    vertical : Height, optional
        Vertical profile defining scale height and vertical structure.
        Default is HyperSecantHeight(). Can be HyperSecantHeight or ExponentialHeight.
    **kwargs : dict
        Additional keyword arguments passed to Component.

    Attributes
    ----------
    radial : Profile
        Radial profile component.
    vertical : Height
        Vertical profile component.
    profile : callable
        Combined 3D density profile rho(r, z) = radial(r) * vertical(z).

    Notes
    -----
    The 3D disk model:
    1. Defines density as rho(r,z) = radial_profile(r) * vertical_profile(z)
    2. Integrates along the line of sight to obtain projected surface brightness
    3. Accounts for inclination by rotating the disk before integration
    4. Uses numerical integration (trapezoidal rule) along z-direction

    The inclination angle is stored in vertical.inc:

    - inc = 0: face-on (looking down on the disk)
    - inc = pi/2: edge-on (viewing disk from the side)

    Line-of-sight integration parameters (vertical.losdepth, vertical.losbins)
    control accuracy and computation time. Larger losdepth and more losbins
    improve accuracy for highly inclined disks.

    Examples
    --------
    >>> from socca.models import Disk, Exponential
    >>> from socca.models.disk.vertical import ExponentialHeight
    >>> # Create edge-on exponential disk
    >>> radial = Exponential(xc=180.5, yc=45.2, rs=0.02, Is=100)
    >>> vertical = ExponentialHeight(zs=0.002, inc=np.pi/2)
    >>> disk = Disk(radial=radial, vertical=vertical)
    """

    def __init__(
        self, radial=Sersic(), vertical=HyperSecantHeight(), **kwargs
    ):
        """
        Initialize a 3D disk model from radial and vertical profiles.

        Parameters
        ----------
        radial : Profile, optional
            Radial profile component. Default is Sersic().
        vertical : Height, optional
            Vertical profile component. Default is HyperSecantHeight().
        **kwargs : dict
            Additional keyword arguments.

        Notes
        -----
        The component ID is synchronized between the Disk, radial, and vertical
        components to ensure consistent parameter naming in composite models.
        """
        super().__init__(**kwargs)

        self.radial = radial
        self.vertical = vertical

        if self.radial.id != self.id:
            type(self).idcls -= 1
            idmin = np.minimum(
                int(self.radial.id.replace("comp_", "")),
                int(self.id.replace("comp_", "")),
            )
            self.id = f"comp_{idmin:02d}"
            self.radial.id = self.id

        self._rkw = [
            f"r_{key}"
            for key in list(
                inspect.signature(self.radial.profile).parameters.keys()
            )
            if key not in ["r", "z"]
        ]
        self._zkw = [
            f"z_{key}"
            for key in list(
                inspect.signature(self.vertical.profile).parameters.keys()
            )
            if key not in ["r", "z"]
        ]

        _profile = [
            "rfoo.profile(r,{0})".format(",".join(self._rkw)),
            "zfoo.profile(z,{0})".format(",".join(self._zkw)),
        ]
        _profile = "*".join(_profile)
        _profile = "lambda rfoo,zfoo,r,z,{0},{1}: {2}".format(
            ",".join(self._rkw), ",".join(self._zkw), _profile
        )

        self.profile = jax.jit(
            partial(eval(_profile), self.radial, self.vertical)
        )

        for key in self.radial.hyper:
            self.hyper.append(f"radial.{key}")

        for key in self.vertical.hyper:
            self.hyper.append(f"vertical.{key}")

        self.units.update(
            {
                f"radial.{key}": self.radial.units[key]
                for key in self.radial.units.keys()
            }
        )
        self.units.update(
            {
                f"vertical.{key}": self.vertical.units[key]
                for key in self.vertical.units.keys()
            }
        )

        self.description.update(
            {
                f"radial.{key}": self.radial.description[key]
                for key in self.radial.description.keys()
            }
        )
        self.description.update(
            {
                f"vertical.{key}": self.vertical.description[key]
                for key in self.vertical.description.keys()
            }
        )
        self._initialized = True

    def getmap(self, img, convolve=False):
        """
        Generate disk image via 3D line-of-sight integration.

        Computes the projected surface brightness by integrating the 3D density
        distribution along the line of sight, accounting for inclination.

        Parameters
        ----------
        img : Image
            Image object containing grid, PSF, and WCS information.
        convolve : bool, optional
            If True, convolve the model with the PSF. Default is False.

        Returns
        -------
        ndarray
            Projected disk image on the image grid.

        Raises
        ------
        ValueError
            If any parameter is a prior distribution or set to None.

        Warns
        -----
        UserWarning
            If convolve=True but no PSF is defined.

        Notes
        -----
        The algorithm:
        1. Creates 3D coordinate grids (r, z) accounting for inclination
        2. Evaluates 3D density rho(r, z) = radial(r) * vertical(z)
        3. Integrates along line of sight using trapezoidal rule
        4. Averages over sub-pixel sampling
        5. Optionally convolves with PSF

        Integration accuracy controlled by:

        - vertical.losdepth: extent of integration
        - vertical.losbins: number of integration points

        For edge-on disks, use larger losdepth (≥ 5*zs) and more losbins (≥ 200).

        Examples
        --------
        >>> from socca.models import Disk, Exponential
        >>> from socca.models.disk.vertical import ExponentialHeight
        >>> import numpy as np
        >>> from socca.data import Image
        >>> radial = Exponential(xc=180.5, yc=45.2, rs=0.02, Is=100)
        >>> vertical = ExponentialHeight(zs=0.002, inc=np.pi/4)  # 45 deg
        >>> disk = Disk(radial=radial, vertical=vertical)
        >>> img = Image('observation.fits')
        >>> disk_map = disk.getmap(img, convolve=True)
        """
        kwarg = {}
        for key in list(inspect.signature(self.profile).parameters.keys()):
            if key not in ["r", "z"]:
                if key.startswith("r_"):
                    val = getattr(self.radial, key.replace("r_", ""))
                elif key.startswith("z_"):
                    val = getattr(self.vertical, key.replace("z_", ""))
                else:
                    continue

                # Evaluate tied parameters (lambda functions)
                if callable(val):
                    sig = inspect.signature(val)
                    params = list(sig.parameters.keys())
                    if params:
                        args = [
                            getattr(self, p.replace(f"{self.id}_", ""))
                            for p in params
                        ]
                        val = val(*args)

                kwarg[key] = val

        # Add geometric parameters
        kwarg["xc"] = self.radial.xc
        kwarg["yc"] = self.radial.yc
        kwarg["theta"] = self.radial.theta
        kwarg["inc"] = self.vertical.inc
        kwarg["losdepth"] = self.vertical.losdepth
        kwarg["losbins"] = self.vertical.losbins

        for key in kwarg.keys():
            if isinstance(kwarg[key], numpyro.distributions.Distribution):
                raise ValueError(
                    "Priors must be fixed values, not distributions."
                )
            if kwarg[key] is None:
                raise ValueError(
                    f"keyword {key} is set to None. "
                    f"Please provide a valid value."
                )

        mgrid = self._evaluate(img, **kwarg)

        if convolve:
            if img.psf is None:
                warnings.warn(
                    "No PSF defined, so no convolution will be performed."
                )
            else:
                # mgrid = (
                #    jp.fft.rfft2(jp.fft.fftshift(mgrid), s=img.data.shape)
                #    * img.psf_fft
                # )
                # mgrid = jp.fft.ifftshift(
                #    jp.fft.irfft2(mgrid, s=img.data.shape)
                # ).real
                mgrid = img.convolve(mgrid)
        return mgrid

    def _build_kwargs(self, pars, comp_prefix):
        """
        Build keyword arguments for _evaluate from the full parameters dict.

        Parameters
        ----------
        pars : dict
            Full parameters dictionary with prefixed keys.
        comp_prefix : str
            Component prefix (e.g., 'comp_00').

        Returns
        -------
        dict
            Keyword arguments for _evaluate including geometric and profile
            parameters with r_ and z_ prefixes.
        """
        # Build profile parameters with r_ and z_ prefixes
        # Use self._rkw and self._zkw instead of inspect.signature since
        # the profile is a jax.jit(partial(...)) which doesn't preserve signature
        kwarg = {
            key.replace(f"{comp_prefix}_radial.", "r_"): pars[key]
            for key in pars
            if key.startswith(f"{comp_prefix}_radial.")
            and key.replace(f"{comp_prefix}_radial.", "r_") in self._rkw
        }
        kwarg.update(
            {
                key.replace(f"{comp_prefix}_vertical.", "z_"): pars[key]
                for key in pars
                if key.startswith(f"{comp_prefix}_vertical.")
                and key.replace(f"{comp_prefix}_vertical.", "z_") in self._zkw
            }
        )

        # Add geometric parameters
        kwarg["xc"] = pars[f"{comp_prefix}_radial.xc"]
        kwarg["yc"] = pars[f"{comp_prefix}_radial.yc"]
        kwarg["theta"] = pars[f"{comp_prefix}_radial.theta"]
        kwarg["inc"] = pars[f"{comp_prefix}_vertical.inc"]
        kwarg["losdepth"] = pars[f"{comp_prefix}_vertical.losdepth"]
        kwarg["losbins"] = pars[f"{comp_prefix}_vertical.losbins"]

        return kwarg

    def _evaluate(self, img, **kwarg):
        """
        Evaluate disk model on the given grid with explicit parameters.

        This internal method computes the projected disk surface brightness
        via line-of-sight integration using the provided geometric and profile
        parameters. It is used by both getmap() and Model.getmodel() to avoid
        code duplication.

        Parameters
        ----------
        img : Image
            Image object containing grid and WCS information.
        **kwarg : dict
            All parameters including geometric (xc, yc, theta, inc, losdepth,
            losbins) and profile-specific parameters with r_ and z_ prefixes.

        Returns
        -------
        ndarray
            2D array of projected surface brightness, averaged over subpixels.
        """
        xc = kwarg.pop("xc")
        yc = kwarg.pop("yc")
        theta = kwarg.pop("theta")
        inc = kwarg.pop("inc")
        losdepth = kwarg.pop("losdepth")
        losbins = kwarg.pop("losbins")

        rcube, zcube = self.getgrid(
            img.grid, xc, yc, losdepth, losbins, theta, inc
        )
        dx = 2.00 * losdepth / (losbins - 1)
        mgrid = self.profile(rcube, zcube, **kwarg)
        mgrid = jp.trapezoid(mgrid, dx=dx, axis=1)
        return jp.mean(mgrid, axis=0)

    @staticmethod
    @partial(jax.jit, static_argnames=["grid", "losbins"])
    def getgrid(grid, xc, yc, losdepth, losbins=200, theta=0.00, inc=0.00):
        """
        Compute 3D disk coordinates with rotation and inclination.

        Generates 3D coordinate grids (r, z) for disk model evaluation, accounting
        for position angle and inclination transformations needed for line-of-sight
        integration through an inclined disk.

        Parameters
        ----------
        grid : Grid
            Grid object with .x and .y celestial coordinate arrays (deg).
        xc : float
            Right ascension of disk center (deg).
        yc : float
            Declination of disk center (deg).
        losdepth : float
            Half-extent of line-of-sight integration (deg).
        losbins : int, optional
            Number of integration points along line of sight. Default is 200.
        theta : float, optional
            Position angle of disk major axis, east from north (rad). Default is 0.
        inc : float, optional
            Inclination angle (0 = face-on, pi/2 = edge-on) (rad). Default is 0.

        Returns
        -------
        rcube : ndarray
            4D array of radial distances from disk center (deg).
            Shape: (ssize, losbins, ysize, xsize).
        zcube : ndarray
            4D array of heights above/below disk midplane (deg).
            Shape: (ssize, losbins, ysize, xsize).

        Notes
        -----
        The transformation sequence:
        1. Center coordinates on (xc, yc)
        2. Apply spherical geometry correction (cos(dec))
        3. Rotate by position angle theta
        4. Create line-of-sight grid from -losdepth to +losdepth
        5. Apply inclination rotation
        6. Compute cylindrical radius r and height z

        The inclination is measured from face-on:

        - inc = 0: face-on, z-axis points toward observer
        - inc = pi/2: edge-on, disk in plane of sky

        The 4D arrays allow vectorized evaluation of the 3D density function
        across the entire image with sub-pixel sampling and line-of-sight
        integration.

        This function is JIT-compiled with static losbins for performance.
        """
        ssize, ysize, xsize = grid.x.shape

        zt = jp.linspace(-losdepth, losdepth, losbins)

        sint = jp.sin(theta)
        cost = jp.cos(theta)

        xt = (grid.x - xc) * jp.cos(jp.deg2rad(yc))
        yt = grid.y - yc

        xt, yt = -xt * sint - yt * cost, xt * cost - yt * sint

        xt = jp.broadcast_to(
            xt[:, None, :, :], (ssize, losbins, ysize, xsize)
        ).copy()
        yt = jp.broadcast_to(
            yt[:, None, :, :], (ssize, losbins, ysize, xsize)
        ).copy()
        zt = jp.broadcast_to(
            zt[None, :, None, None], (ssize, losbins, ysize, xsize)
        ).copy()

        sini = jp.sin(inc - 0.50 * jp.pi)
        cosi = jp.cos(inc - 0.50 * jp.pi)

        zt, yt = yt * cosi - zt * sini, yt * sini + zt * cosi

        return jp.sqrt(xt**2 + yt**2), zt

    def parameters(self):
        """
        Print formatted table of disk parameters from radial and vertical components.

        Displays parameters from both the radial and vertical profile components,
        organized as 'radial.parameter' and 'vertical.parameter'. Separates regular
        parameters from hyperparameters (integration settings).

        Notes
        -----
        Output format:

        Model parameters
        ================
        radial.xc        [deg]   : value | Right ascension of centroid
        radial.yc        [deg]   : value | Declination of centroid
        radial.rs        [deg]   : value | Scale radius
        vertical.zs      [deg]   : value | Scale height
        vertical.inc     [rad]   : value | Inclination angle (0=face-on)

        Hyperparameters
        ===============
        vertical.losdepth [deg]  : value | Half line-of-sight extent
        vertical.losbins  []     : value | Number of integration points

        Hyperparameters control the numerical integration accuracy but are not
        fitted parameters.

        Examples
        --------
        >>> from socca.models import Disk, Exponential
        >>> from socca.models.disk.vertical import ExponentialHeight
        >>> import numpy as np
        >>> radial = Exponential(xc=180.5, yc=45.2, rs=0.02, Is=100)
        >>> vertical = ExponentialHeight(zs=0.002, inc=np.pi/4)
        >>> disk = Disk(radial=radial, vertical=vertical)
        >>> disk.parameters()
        Model parameters
        ================
        radial.xc         [deg]    : 1.8050E+02 | Right ascension of centroid
        radial.yc         [deg]    : 4.5200E+01 | Declination of centroid
        ...
        """
        keyout = [key for key in self.units.keys() if key not in self.hyper]

        if len(keyout) > 0:
            maxlen = np.max(
                np.array(
                    [
                        len(f"{key} [{self.units[key]}]")
                        for key in keyout + self.hyper
                    ]
                )
            )

            print("\nModel parameters")
            print("=" * 16)
            for key in keyout:
                keylen = maxlen - len(f" [{self.units[key]}]")
                if key.startswith("radial."):
                    kvalue = getattr(self.radial, key.replace("radial.", ""))
                elif key.startswith("vertical."):
                    kvalue = getattr(
                        self.vertical, key.replace("vertical.", "")
                    )
                else:
                    kvalue = getattr(self, key)

                if kvalue is None:
                    kvalue = None
                elif isinstance(kvalue, numpyro.distributions.Distribution):
                    kvalue = f"Distribution: {kvalue.__class__.__name__}"
                elif isinstance(
                    kvalue, (types.LambdaType, types.FunctionType)
                ):
                    kvalue = "Tied parameter"
                else:
                    kvalue = f"{kvalue:.4E}"

                print(
                    f"{key:<{keylen}} [{self.units[key]}] : "
                    + f"{kvalue}".ljust(10)
                    + f" | {self.description[key]}"
                )

            if len(self.hyper) > 0:
                print("\nHyperparameters")
                print("=" * 15)
                for key in self.hyper:
                    keylen = maxlen - len(f" [{self.units[key]}]")
                    if key.startswith("radial."):
                        kvalue = getattr(
                            self.radial, key.replace("radial.", "")
                        )
                    elif key.startswith("vertical."):
                        kvalue = getattr(
                            self.vertical, key.replace("vertical.", "")
                        )
                    else:
                        kvalue = getattr(self, key)

                    if kvalue is None:
                        kvalue = None
                    elif isinstance(
                        kvalue, numpyro.distributions.Distribution
                    ):
                        kvalue = f"Distribution: {kvalue.__class__.__name__}"
                    elif isinstance(
                        kvalue, (types.LambdaType, types.FunctionType)
                    ):
                        kvalue = "Tied parameter"
                    else:
                        kvalue = f"{kvalue:.4E}"

                    print(
                        f"{key:<{keylen}} [{self.units[key]}] : "
                        + f"{kvalue}".ljust(10)
                        + f" | {self.description[key]}"
                    )
        else:
            print("No parameters defined.")

    def parlist(self):
        """
        Return list of parameter names from radial and vertical components.

        Collects all parameters from both the radial and vertical profile
        components, with names prefixed as 'radial.' and 'vertical.'.

        Returns
        -------
        list of str
            Combined list of parameter names from radial and vertical components.
            Names are prefixed with component type (e.g., 'radial.xc', 'vertical.zs').

        Notes
        -----
        This method is used internally by Model.addcomponent() when adding
        a Disk component to a composite model, ensuring all parameters from
        both sub-components are registered.

        Examples
        --------
        >>> from socca.models import Disk, Exponential
        >>> from socca.models.disk.vertical import ExponentialHeight
        >>> radial = Exponential(xc=180.5, yc=45.2, rs=0.02, Is=100)
        >>> vertical = ExponentialHeight(zs=0.002, inc=0.5)
        >>> disk = Disk(radial=radial, vertical=vertical)
        >>> disk.parlist()
        ['radial.xc', 'radial.yc', 'radial.theta', 'radial.e', 'radial.cbox',
         'radial.rs', 'radial.Is', 'vertical.losdepth', 'vertical.losbins',
         'vertical.inc', 'vertical.zs']
        """
        return list(self.units.keys())
