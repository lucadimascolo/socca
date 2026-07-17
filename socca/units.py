"""Unit conversion utilities for socca model parameters."""

import astropy.units as au

_NON_CONVERTIBLE = {"image", ""}


def conversion_factor(from_unit_str, to_unit_str):
    """Return the multiplicative factor to convert ``from_unit_str`` to ``to_unit_str``.

    Only called for parameters with physically meaningful internal units
    (e.g. ``"deg"``, ``"rad"``).  Non-physical internal units (``"image"``,
    ``""``) are handled upstream in :meth:`Component.set_units` and never
    reach this function.

    Parameters
    ----------
    from_unit_str : str
        User-declared unit string (astropy-parseable, e.g. ``"arcsec"``).
    to_unit_str : str
        Internal unit string (e.g. ``"deg"``, ``"rad"``).

    Returns
    -------
    float
        Multiplicative factor: ``value_internal = value_user * factor``.

    Raises
    ------
    ValueError
        If either unit string is invalid or the two units are not convertible.
    """
    try:
        from_unit = au.Unit(from_unit_str)
        to_unit = au.Unit(to_unit_str)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid unit string: {e}") from e

    try:
        return float(from_unit.to(to_unit))
    except au.UnitConversionError as e:
        raise ValueError(
            f"Cannot convert '{from_unit_str}' to '{to_unit_str}': {e}"
        ) from e
