"""Tests for socca.units and unit conversion in models."""

import jax.numpy as jp
import numpy as np
import numpyro.distributions
import pytest

import socca.models as models
import socca.priors as priors
from socca.units import conversion_factor


@pytest.fixture(autouse=True)
def reset_idcls():
    """Reset Component ID counter before each test."""
    models.Component.idcls = 0


class TestConversionFactor:
    """Tests for socca.units.conversion_factor."""

    def test_arcsec_to_deg(self):
        """Arcsec → deg factor is 1/3600."""
        assert conversion_factor("arcsec", "deg") == pytest.approx(1 / 3600)

    def test_arcmin_to_deg(self):
        """Arcmin → deg factor is 1/60."""
        assert conversion_factor("arcmin", "deg") == pytest.approx(1 / 60)

    def test_deg_to_deg_identity(self):
        """Deg → deg factor is 1."""
        assert conversion_factor("deg", "deg") == pytest.approx(1.0)

    def test_deg_to_rad(self):
        """Deg → rad factor is pi/180."""
        assert conversion_factor("deg", "rad") == pytest.approx(np.pi / 180)

    def test_rad_to_rad_identity(self):
        """Rad → rad factor is 1."""
        assert conversion_factor("rad", "rad") == pytest.approx(1.0)

    def test_arcsec_to_rad(self):
        """Arcsec → rad factor is pi/180/3600."""
        assert conversion_factor("arcsec", "rad") == pytest.approx(
            np.pi / 180 / 3600
        )

    def test_invalid_from_unit_raises(self):
        """Non-parseable from-unit raises ValueError."""
        with pytest.raises(ValueError, match="Invalid unit"):
            conversion_factor("notaunit!!!", "deg")

    def test_incompatible_units_raises(self):
        """Physically incompatible units raise ValueError."""
        with pytest.raises(ValueError, match="Cannot convert"):
            conversion_factor("Jy", "deg")


class TestSetUnits:
    """Tests for Component.set_units."""

    def test_angular_params_stored(self):
        """Declared units are stored in _input_units."""
        g = models.Gaussian(xc=0.5, yc=0.5, rs=30.0, Is=1.0)
        g.set_units(xc="arcsec", yc="arcsec", rs="arcmin")
        assert g._input_units["xc"] == "arcsec"
        assert g._input_units["yc"] == "arcsec"
        assert g._input_units["rs"] == "arcmin"

    def test_native_units_dict_unchanged(self):
        """set_units does not mutate the native units dict."""
        g = models.Gaussian(xc=0.5, yc=0.5, rs=1.0, Is=1.0)
        g.set_units(xc="arcsec", rs="arcmin")
        assert g.units["xc"] == "deg"
        assert g.units["rs"] == "deg"

    def test_image_unit_warns_and_ignores(self):
        """Surface brightness parameters warn and are not stored."""
        g = models.Gaussian(xc=0.5, yc=0.5, rs=1.0, Is=1.0)
        with pytest.warns(UserWarning, match="surface brightness"):
            g.set_units(Is="mJy/arcsec2")
        assert "Is" not in g._input_units

    def test_dimensionless_unit_warns_and_ignores(self):
        """Dimensionless parameters warn and are not stored."""
        g = models.Gaussian(xc=0.5, yc=0.5, rs=1.0, Is=1.0)
        with pytest.warns(UserWarning, match="dimensionless"):
            g.set_units(e="deg")
        assert "e" not in g._input_units

    def test_unknown_parameter_raises(self):
        """Unknown parameter name raises ValueError."""
        g = models.Gaussian(xc=0.5, yc=0.5, rs=1.0, Is=1.0)
        with pytest.raises(ValueError, match="Unknown parameter"):
            g.set_units(nonexistent="arcsec")

    def test_invalid_unit_string_raises(self):
        """Non-parseable unit string raises ValueError."""
        g = models.Gaussian(xc=0.5, yc=0.5, rs=1.0, Is=1.0)
        with pytest.raises(ValueError):
            g.set_units(xc="notaunit!!!")

    def test_incompatible_unit_raises(self):
        """Unit incompatible with the internal unit raises ValueError."""
        g = models.Gaussian(xc=0.5, yc=0.5, rs=1.0, Is=1.0)
        with pytest.raises(ValueError, match="Cannot convert"):
            g.set_units(xc="Jy")

    def test_theta_rad_to_deg(self):
        """Theta can be declared in deg; internal unit stays rad."""
        g = models.Gaussian(xc=0.5, yc=0.5, rs=1.0, Is=1.0)
        g.set_units(theta="deg")
        assert g._input_units["theta"] == "deg"
        assert g.units["theta"] == "rad"


class TestModelConversions:
    """Tests for Model.conversions populated at addcomponent time."""

    def test_conversions_empty_without_set_units(self):
        """No set_units call leaves conversions dict empty."""
        g = models.Gaussian(xc=0.5, yc=0.5, rs=1.0, Is=1.0)
        mod = models.Model(g)
        assert mod.conversions == {}

    def test_conversion_factor_arcsec_stored(self):
        """Arcsec override stores the correct factor in Model.conversions."""
        g = models.Gaussian(xc=0.5, yc=0.5, rs=1.0, Is=1.0)
        g.set_units(xc="arcsec")
        mod = models.Model(g)
        assert "comp_00_xc" in mod.conversions
        assert mod.conversions["comp_00_xc"] == pytest.approx(1 / 3600)

    def test_conversion_factor_arcmin_stored(self):
        """Arcmin override stores the correct factor in Model.conversions."""
        g = models.Gaussian(xc=0.5, yc=0.5, rs=1.0, Is=1.0)
        g.set_units(rs="arcmin")
        mod = models.Model(g)
        assert mod.conversions["comp_00_rs"] == pytest.approx(1 / 60)

    def test_unoverridden_params_not_in_conversions(self):
        """Parameters without a declared override are absent from conversions."""
        g = models.Gaussian(xc=0.5, yc=0.5, rs=1.0, Is=1.0)
        g.set_units(xc="arcsec")
        mod = models.Model(g)
        assert "comp_00_yc" not in mod.conversions
        assert "comp_00_rs" not in mod.conversions

    def test_model_units_reflects_input_unit(self):
        """Model.units stores the user-declared unit for overridden parameters."""
        g = models.Gaussian(xc=0.5, yc=0.5, rs=1.0, Is=1.0)
        g.set_units(xc="arcsec")
        mod = models.Model(g)
        assert mod.units["comp_00_xc"] == "arcsec"
        assert mod.units["comp_00_yc"] == "deg"

    def test_multicomponent_independent_conversions(self):
        """Each component's conversions are stored independently."""
        g1 = models.Gaussian(xc=0.5, yc=0.5, rs=1.0, Is=1.0)
        g1.set_units(xc="arcsec")
        g2 = models.Gaussian(xc=0.5, yc=0.5, rs=1.0, Is=1.0)
        g2.set_units(rs="arcmin")
        mod = models.Model(g1)
        mod.addcomponent(g2)
        assert mod.conversions["comp_00_xc"] == pytest.approx(1 / 3600)
        assert mod.conversions["comp_01_rs"] == pytest.approx(1 / 60)
        assert "comp_01_xc" not in mod.conversions


class TestGetmodelWithUnits:
    """Tests for unit conversion applied inside Model.getmodel."""

    def _build_pars(self, mod, pp_array):
        """Replicate the getmodel pars-building logic for unit-testing."""
        pars = {}
        pp = jp.array(pp_array)
        for key in mod.params:
            if isinstance(mod.priors[key], (float, int)):
                pars[key] = mod.priors[key]
            elif isinstance(
                mod.priors[key], numpyro.distributions.Distribution
            ):
                pars[key], pp = pp[0], pp[1:]
        for key, factor in mod.conversions.items():
            if key in pars:
                pars[key] = pars[key] * factor
        return pars

    def test_fixed_arcsec_converted_to_deg(self):
        """Fixed arcsec value is converted to degrees before dispatch."""
        g = models.Gaussian(xc=3600.0, yc=0.5, rs=1.0, Is=1.0)
        g.set_units(xc="arcsec")
        mod = models.Model(g)
        pars = self._build_pars(mod, [])
        assert float(pars["comp_00_xc"]) == pytest.approx(1.0)

    def test_fixed_arcmin_converted_to_deg(self):
        """Fixed arcmin value is converted to degrees before dispatch."""
        g = models.Gaussian(xc=0.5, yc=0.5, rs=60.0, Is=1.0)
        g.set_units(rs="arcmin")
        mod = models.Model(g)
        pars = self._build_pars(mod, [])
        assert float(pars["comp_00_rs"]) == pytest.approx(1.0)

    def test_free_param_converted(self):
        """Sampled free parameter in arcsec is converted to degrees."""
        g = models.Gaussian(xc=None, yc=0.5, rs=1.0, Is=1.0)
        g.xc = priors.uniform(3599.0, 3601.0)
        g.set_units(xc="arcsec")
        mod = models.Model(g)
        pars = self._build_pars(mod, [3600.0])
        assert float(pars["comp_00_xc"]) == pytest.approx(1.0)

    def test_unconverted_param_unchanged(self):
        """Parameters without a declared override pass through unchanged."""
        g = models.Gaussian(xc=0.5, yc=2.0, rs=1.0, Is=1.0)
        g.set_units(xc="arcsec")
        mod = models.Model(g)
        pars = self._build_pars(mod, [])
        assert float(pars["comp_00_yc"]) == pytest.approx(2.0)

    def test_theta_deg_to_rad(self):
        """Theta declared in degrees is converted to radians."""
        g = models.Gaussian(xc=0.5, yc=0.5, rs=1.0, Is=1.0)
        g.theta = 180.0
        g.set_units(theta="deg")
        mod = models.Model(g)
        pars = self._build_pars(mod, [])
        assert float(pars["comp_00_theta"]) == pytest.approx(np.pi)

    def test_no_set_units_no_conversion(self):
        """Without set_units, values reach components unchanged."""
        g = models.Gaussian(xc=0.5, yc=0.5, rs=1.0, Is=1.0)
        mod = models.Model(g)
        pars = self._build_pars(mod, [])
        assert float(pars["comp_00_xc"]) == pytest.approx(0.5)
        assert float(pars["comp_00_rs"]) == pytest.approx(1.0)
