"""Truncation functions for radial surface brightness profiles."""

from abc import abstractmethod

import jax.numpy as jp

from .. import config


class Truncation:
    """Base class for radial truncation functions.

    Truncation objects are attached to a :class:`~socca.models.radial.base.Profile`
    to suppress emission beyond a characteristic radius.  Every subclass must
    implement :meth:`profile`, a static method that returns a multiplicative
    weight in ``[0, 1]`` as a function of projected radius.

    Parameters
    ----------
    **kwargs : dict
        Keyword arguments for truncation initialization:

        rt : float, optional
            Truncation radius in degrees.  Default is taken from
            :attr:`config.Truncation.rt`.
        wt : float, optional
            Truncation width (transition scale) in degrees.  Smaller values
            produce a sharper cut-off.  Default is taken from
            :attr:`config.Truncation.wt`.

    Attributes
    ----------
    rt : float
        Truncation radius in degrees.
    wt : float
        Truncation width in degrees.
    units : dict
        Physical units for each parameter (``rt`` and ``wt`` are both
        ``"deg"``).
    description : dict
        Human-readable description for each parameter.
    """

    def __init__(self, **kwargs):
        self.rt = kwargs.get("rt", config.Truncation.rt)
        self.wt = kwargs.get("wt", config.Truncation.wt)

        self.units = dict(rt="deg", wt="deg")
        self.description = dict(rt="Truncation radius", wt="Truncation width")

    @abstractmethod
    def profile(self, r):
        """Return the truncation weight at radius *r*.

        Parameters
        ----------
        r : array_like
            Projected elliptical radius in degrees.

        Returns
        -------
        array_like
            Multiplicative weight in ``[0, 1]``, same shape as *r*.
        """
        pass


class Exponential(Truncation):
    r"""Exponential truncation profile.

    Smoothly suppresses emission beyond the truncation radius using an
    exponential roll-off:

    .. math::

        T(r) = \frac{1 - e^{(r - r_t)/w_t}}{1 - e^{-r_t/w_t}}

    clipped to zero for :math:`r > r_t`.  The denominator normalises the
    weight to unity at :math:`r = 0`.

    Parameters
    ----------
    **kwargs : dict
        Passed to :class:`Truncation`.  Accepts ``rt`` and ``wt``.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def profile(r, rt, wt):
        """Evaluate the exponential truncation weight.

        Parameters
        ----------
        r : array_like
            Projected elliptical radius in degrees.
        rt : float
            Truncation radius in degrees.
        wt : float
            Truncation width in degrees.

        Returns
        -------
        array_like
            Truncation weight in ``[0, 1]``, same shape as *r*.
        """
        num = 1.00 - jp.exp((r - rt) / wt)
        den = 1.00 - jp.exp(-rt / wt)
        return jp.clip(num / den, 0.00, None)


class HyperTangent(Truncation):
    r"""Hyperbolic-tangent truncation profile.

    Smoothly suppresses emission beyond the truncation radius using a
    :math:`\tanh` roll-off:

    .. math::

        T(r) = \frac{1 - \tanh\!\left((r - r_t)/w_t\right)}
                     {1 - \tanh\!\left(-r_t/w_t\right)}

    The denominator normalises the weight to unity at :math:`r = 0`.
    Unlike the exponential variant the weight never reaches exactly zero,
    but decays rapidly for :math:`r \gg r_t`.

    Parameters
    ----------
    **kwargs : dict
        Passed to :class:`Truncation`.  Accepts ``rt`` and ``wt``.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def profile(r, rt, wt):
        """Evaluate the hyperbolic-tangent truncation weight.

        Parameters
        ----------
        r : array_like
            Projected elliptical radius in degrees.
        rt : float
            Truncation radius in degrees.
        wt : float
            Truncation width in degrees.

        Returns
        -------
        array_like
            Truncation weight, same shape as *r*.
        """
        num = 1.00 - jp.tanh((r - rt) / wt)
        den = 1.00 - jp.tanh(-rt / wt)
        return num / den
