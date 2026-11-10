"""Tests for socca.models module."""

import copy
import gc
import importlib.util
import pickle
import weakref

import jax.numpy as jp
import numpy as np
import numpyro.distributions
import pytest
from astropy.io import fits

import socca.models as models
import socca.priors as priors
import socca.data as data
import socca.noise as noise


class TestModel:
    """Tests for Model class."""

    def test_empty_initialization(self):
        """Test creating an empty model."""
        mod = models.Model()
        assert mod.ncomp == 0
        assert mod.priors == {}
        assert mod.params == []

    def test_copy_is_independent(self):
        """Test that adding to a shallow copy leaves the original intact."""
        mod = models.Model(
            models.Beta(xc=180.0, yc=45.0, rc=0.005, Ic=1.0, beta=0.6)
        )
        new = copy.copy(mod)
        new.addcomponent(models.Point(xc=180.0, yc=45.0, Ic=1.0))
        assert (mod.ncomp, len(mod.components)) == (1, 1)
        assert mod.params[-1] == "comp_00_Ic"
        assert new.ncomp == 2 and new.params[-1] == "comp_01_Ic"

    def test_components_read_only(self):
        """Test that the component list cannot be changed in place."""
        mod = models.Model(
            models.Beta(xc=180.0, yc=45.0, rc=0.005, Ic=1.0, beta=0.6)
        )
        with pytest.raises(AttributeError):
            mod.components.pop()
        with pytest.raises(TypeError):
            mod.components[0] = models.Beta(
                xc=180.0, yc=45.0, rc=0.005, Ic=1.0, beta=0.6
            )
        assert mod.ncomp == 1

    def test_initialization_with_profile(self):
        """Test creating a model with initial profile."""
        beta = models.Beta(xc=180.0, yc=45.0, rc=0.01, Ic=1.0, beta=0.5)
        mod = models.Model(beta)
        assert mod.ncomp == 1
        assert len(mod.params) > 0

    def test_addcomponent(self):
        """Test adding components to model."""
        mod = models.Model()
        beta = models.Beta(xc=180.0, yc=45.0, rc=0.01, Ic=1.0, beta=0.5)
        mod.addcomponent(beta)
        assert mod.ncomp == 1
        assert "comp_00_xc" in mod.params
        assert "comp_00_yc" in mod.params

    def test_multiple_components(self):
        """Test adding multiple components."""
        mod = models.Model()
        beta = models.Beta(xc=180.0, yc=45.0, rc=0.01, Ic=1.0, beta=0.5)
        point = models.Point(xc=180.1, yc=45.1, Ic=100.0)
        mod.addcomponent(beta)
        mod.addcomponent(point)
        assert mod.ncomp == 2
        assert "comp_00_xc" in mod.params
        assert "comp_01_xc" in mod.params

    def test_none_parameter_raises_error(self):
        """Test that None parameters raise ValueError."""
        mod = models.Model()
        beta = models.Beta(xc=None, yc=45.0, rc=0.01, Ic=1.0, beta=0.5)
        with pytest.raises(ValueError, match="set to None"):
            mod.addcomponent(beta)

    def test_prior_indices(self):
        """Test that prior indices are tracked correctly."""
        mod = models.Model()
        beta = models.Beta(
            xc=priors.uniform(179.0, 181.0),
            yc=45.0,
            rc=priors.loguniform(0.001, 0.1),
            Ic=1.0,
            beta=0.5,
        )
        mod.addcomponent(beta)
        assert len(mod.paridx) == 2

    def test_tied_parameters(self):
        """Test tied parameter detection."""
        mod = models.Model()
        beta1 = models.Beta(xc=180.0, yc=45.0, rc=0.01, Ic=1.0, beta=0.5)
        mod.addcomponent(beta1)

        beta2 = models.Beta(
            xc=priors.boundto(beta1, "xc"),
            yc=45.1,
            rc=0.02,
            Ic=0.5,
            beta=0.6,
        )
        mod.addcomponent(beta2)
        assert any(mod.tied)

    def test_positivity_constraint(self):
        """Test positivity constraint handling."""
        beta = models.Beta(xc=180.0, yc=45.0, rc=0.01, Ic=1.0, beta=0.5)
        beta.positive = True
        mod = models.Model(beta)
        assert mod.positive[0] is True

    def test_positivity_override(self):
        """Test overriding positivity constraint."""
        beta = models.Beta(xc=180.0, yc=45.0, rc=0.01, Ic=1.0, beta=0.5)
        beta.positive = False
        mod = models.Model(beta, positive=True)
        assert mod.positive[0] is True

    def test_units_stored(self):
        """Test that units are stored for each parameter."""
        mod = models.Model()
        beta = models.Beta(xc=180.0, yc=45.0, rc=0.01, Ic=1.0, beta=0.5)
        mod.addcomponent(beta)
        assert "comp_00_xc" in mod.units
        assert mod.units["comp_00_xc"] == "deg"

    def test_type_stored(self):
        """Test that component types are stored."""
        mod = models.Model()
        mod.addcomponent(
            models.Beta(xc=180.0, yc=45.0, rc=0.01, Ic=1.0, beta=0.5)
        )
        mod.addcomponent(models.Point(xc=180.0, yc=45.0, Ic=1.0))
        assert mod.type == ["Beta", "Point"]


class TestModelGetmodel:
    """Tests for Model.getmodel method."""

    @pytest.fixture
    def simple_model_and_image(self, simple_hdu, gaussian_psf):
        """Create a simple model and image for testing."""
        img = data.Image(simple_hdu, noise=noise.Normal(sigma=0.1))
        img.addpsf(gaussian_psf)

        xc = simple_hdu.header["CRVAL1"]
        yc = simple_hdu.header["CRVAL2"]
        rc = 0.005

        beta = models.Beta(
            xc=priors.uniform(xc - 0.01, xc + 0.01),
            yc=yc,
            rc=rc,
            Ic=1.0,
            beta=0.5,
        )
        mod = models.Model(beta)
        return mod, img

    def test_getmodel_returns_tuple(self, simple_model_and_image):
        """Test that getmodel returns 4-tuple."""
        mod, img = simple_model_and_image
        pp = [img.hdu.header["CRVAL1"]]
        result = mod.getmodel(img, pp)
        assert len(result) == 4

    def test_getmodel_shapes(self, simple_model_and_image):
        """Test that getmodel outputs have correct shape."""
        mod, img = simple_model_and_image
        pp = [img.hdu.header["CRVAL1"]]
        mraw, msmo, mbkg, mneg = mod.getmodel(img, pp)
        assert mraw.shape == img.data.shape
        assert msmo.shape == img.data.shape
        assert mbkg.shape == img.data.shape
        assert mneg.shape == img.data.shape


class TestComponent:
    """Tests for Component base class."""

    def test_parlist(self):
        """Test parlist returns parameter names."""
        beta = models.Beta(xc=180.0, yc=45.0, rc=0.01, Ic=1.0, beta=0.5)
        params = beta.parlist()
        assert "xc" in params
        assert "yc" in params
        assert "rc" in params
        assert "Ic" in params
        assert "alpha" in params
        assert "beta" in params

    def test_addparameter(self):
        """Test adding custom parameter."""
        beta = models.Beta(xc=180.0, yc=45.0, rc=0.01, Ic=1.0, beta=0.5)
        beta.addparameter(
            "custom", value=42.0, units="custom_unit", description="Test param"
        )
        assert hasattr(beta, "custom")
        assert beta.custom == 42.0
        assert beta.units["custom"] == "custom_unit"


class TestProfile:
    """Tests for Profile base class."""

    def test_default_values(self):
        """Test default profile parameter values."""
        beta = models.Beta()
        assert beta.theta is not None or beta.theta == 0.0
        assert beta.e is not None or beta.e == 0.0
        assert beta.cbox is not None or beta.cbox == 0.0

    def test_e_accepts_tied_lambda(self):
        """Test that e can be set to a tied lambda without raising."""
        beta = models.Beta()
        beta.e = lambda comp_00_e1: comp_00_e1
        assert callable(beta.e)

    def test_e_accepts_boundto(self):
        """Test that e can be set to a _BoundTo without raising."""
        beta = models.Beta()
        other = models.Beta()
        beta.e = priors.boundto(other, "e")
        assert isinstance(beta.e, priors._BoundTo)

    def test_e_accepts_none(self):
        """Test that e can be set to None without raising."""
        beta = models.Beta()
        beta.e = None
        assert beta.e is None

    def test_e_still_raises_for_fixed_value_above_one(self):
        """Test that a fixed e >= 1 still raises ValueError."""
        beta = models.Beta()
        with pytest.raises(ValueError, match="greater than or equal to 1"):
            beta.e = 1.5

    def test_e_still_raises_for_fixed_value_below_zero(self):
        """Test that a fixed e < 0 still raises ValueError."""
        beta = models.Beta()
        with pytest.raises(ValueError, match="less than 0"):
            beta.e = -0.1

    def test_e_prior_may_reach_one(self, recwarn):
        """A prior with support up to e = 1 is accepted."""
        models.Beta().e = priors.uniform(0.0, 1.0)
        assert len(recwarn) == 0

    def test_e_still_raises_for_prior_out_of_range(self):
        """Test that a prior with out-of-range support still raises."""
        beta = models.Beta()
        with pytest.raises(ValueError, match="greater than 1"):
            beta.e = priors.uniform(0.0, 1.5)

    @staticmethod
    def _beta(theta):
        """Create a fixed Beta with the given theta."""
        return models.Beta(
            xc=0.0, yc=0.0, rc=1e-3, Ic=1.0, beta=0.6, theta=theta
        )

    def test_theta_narrow_prior_no_warning(self, recwarn):
        """A theta prior narrower than its period does not warn."""
        models.Model(self._beta(priors.uniform(-np.pi / 4, np.pi / 4)))
        assert len(recwarn) == 0

    def test_theta_exact_period_warns_periodic(self, recwarn):
        """A theta prior spanning its period warns when added to a model."""
        beta = self._beta(priors.uniform(0.00, np.pi))
        assert len(recwarn) == 0
        models.Model(beta)
        assert len(recwarn) == 1
        assert "periodic" in str(recwarn[0].message)
        assert recwarn[0].filename == __file__

    def test_theta_wider_than_period_warns_twice(self, recwarn):
        """A wider theta prior warns about multimodality and periodicity."""
        models.Model(self._beta(priors.uniform(-np.pi, np.pi)))
        messages = [str(w.message) for w in recwarn]
        assert any("multimodal" in m for m in messages)
        assert any("periodic" in m for m in messages)

    def test_theta_period_in_declared_units(self, recwarn):
        """Periods follow declared units: [0, 90] deg is not periodic."""
        beta = self._beta(priors.uniform(0.0, 90.0))
        beta.set_units(theta="deg")
        mod = models.Model(beta)
        assert len(recwarn) == 0
        assert mod.periodic["comp_00_theta"] == pytest.approx(180.0)
        beta = self._beta(priors.uniform(0.0, 180.0))
        beta.set_units(theta="deg")
        with pytest.warns(UserWarning, match="full period"):
            models.Model(beta)


class TestBeta:
    """Tests for Beta profile."""

    def test_initialization(self):
        """Test Beta profile initialization."""
        beta = models.Beta(xc=180.0, yc=45.0, rc=0.01, Ic=1.0, beta=0.5)
        assert beta.xc == 180.0
        assert beta.yc == 45.0
        assert beta.rc == 0.01
        assert beta.Ic == 1.0
        assert beta.beta == 0.5

    def test_profile_function(self):
        """Test Beta profile function."""
        r = jp.array([0.0, 0.01, 0.02, 0.05])
        result = models.Beta.profile(r, Ic=1.0, rc=0.01, alpha=2.0, beta=0.5)
        assert result.shape == r.shape
        assert float(result[0]) == 1.0
        assert float(result[1]) < 1.0

    def test_profile_central_value(self):
        """Test that profile is maximized at center."""
        r = jp.array([0.0])
        Ic = 10.0
        result = models.Beta.profile(r, Ic=Ic, rc=0.01, alpha=2.0, beta=0.5)
        assert float(result[0]) == pytest.approx(Ic)

    def test_profile_decreases_with_radius(self):
        """Test that profile decreases with radius."""
        r = jp.array([0.0, 0.01, 0.02, 0.05, 0.1])
        result = models.Beta.profile(r, Ic=1.0, rc=0.01, alpha=2.0, beta=0.5)
        for i in range(len(r) - 1):
            assert float(result[i]) >= float(result[i + 1])


class TestSersic:
    """Tests for Sersic profile."""

    def test_initialization(self):
        """Test Sersic profile initialization."""
        sersic = models.Sersic(xc=180.0, yc=45.0, re=0.01, Ie=1.0, ns=4.0)
        assert sersic.xc == 180.0
        assert sersic.re == 0.01
        assert sersic.Ie == 1.0
        assert sersic.ns == 4.0

    def test_profile_function(self):
        """Test Sersic profile function."""
        r = jp.array([0.0, 0.01, 0.02, 0.05])
        result = models.Sersic.profile(r, Ie=1.0, re=0.01, ns=1.0)
        assert result.shape == r.shape

    def test_profile_at_effective_radius(self):
        """Test that profile equals Ie at effective radius."""
        re = 0.01
        r = jp.array([re])
        Ie = 5.0
        result = models.Sersic.profile(r, Ie=Ie, re=re, ns=1.0)
        assert float(result[0]) == pytest.approx(Ie, rel=0.01)


class TestExponential:
    """Tests for Exponential profile."""

    def test_initialization(self):
        """Test Exponential profile initialization."""
        exp = models.Exponential(xc=180.0, yc=45.0, rs=0.01, Is=1.0)
        assert exp.xc == 180.0
        assert exp.rs == 0.01
        assert exp.Is == 1.0

    def test_profile_function(self):
        """Test Exponential profile function."""
        r = jp.array([0.0, 0.01, 0.02, 0.05])
        result = models.Exponential.profile(r, Is=1.0, rs=0.01)
        assert result.shape == r.shape
        assert float(result[0]) == pytest.approx(1.0)

    def test_profile_at_scale_radius(self):
        """Test profile at scale radius."""
        rs = 0.01
        r = jp.array([rs])
        Is = 1.0
        result = models.Exponential.profile(r, Is=Is, rs=rs)
        assert float(result[0]) == pytest.approx(Is / np.e)


class TestPoint:
    """Tests for Point source model."""

    def test_initialization(self):
        """Test Point source initialization."""
        point = models.Point(xc=180.0, yc=45.0, Ic=100.0)
        assert point.xc == 180.0
        assert point.yc == 45.0
        assert point.Ic == 100.0

    def test_parlist(self):
        """Test Point source parameter list."""
        point = models.Point(xc=180.0, yc=45.0, Ic=100.0)
        params = point.parlist()
        assert "xc" in params
        assert "yc" in params
        assert "Ic" in params


class TestBackground:
    """Tests for Background model."""

    def test_initialization_default(self):
        """Test Background initialization with default (constant)."""
        bkg = models.Background()
        assert hasattr(bkg, "a0")

    def test_initialization_with_params(self):
        """Test Background initialization with polynomial coefficients."""
        bkg = models.Background(a0=1.0, a1x=0.1, a1y=0.2)
        assert hasattr(bkg, "a0")
        assert hasattr(bkg, "a1x")
        assert hasattr(bkg, "a1y")
        params = bkg.parlist()
        assert "a0" in params

    def test_parlist(self):
        """Test Background parameter list."""
        bkg = models.Background(a0=5.0)
        params = bkg.parlist()
        assert "a0" in params


class TestProfileGetgrid:
    """Tests for Profile.getgrid static method."""

    @pytest.fixture
    def simple_grid(self, simple_hdu):
        """Create a simple grid for testing."""
        return data.WCSgrid(simple_hdu)

    def test_getgrid_returns_array(self, simple_grid, simple_hdu):
        """Test that getgrid returns an array."""
        xc = simple_hdu.header["CRVAL1"]
        yc = simple_hdu.header["CRVAL2"]
        result = models.Profile.getgrid(simple_grid, xc, yc)
        assert isinstance(result, jp.ndarray)

    def test_getgrid_circular(self, simple_grid, simple_hdu):
        """Test circular profile (e=0)."""
        xc = simple_hdu.header["CRVAL1"]
        yc = simple_hdu.header["CRVAL2"]
        result = models.Profile.getgrid(simple_grid, xc, yc, theta=0.0, e=0.0)
        assert result.shape == simple_grid.x.shape

    def test_getgrid_center_is_zero(self, simple_grid, simple_hdu):
        """Test that grid is approximately zero at center."""
        xc = simple_hdu.header["CRVAL1"]
        yc = simple_hdu.header["CRVAL2"]
        result = models.Profile.getgrid(simple_grid, xc, yc)
        center_idx = (
            0,
            simple_hdu.data.shape[0] // 2,
            simple_hdu.data.shape[1] // 2,
        )
        assert float(result[center_idx]) < 0.001


class TestZoo:
    """Tests for zoo() function."""

    def test_zoo_prints_models(self, capsys):
        """Test that zoo prints available models."""
        models.zoo()
        captured = capsys.readouterr()
        assert "Beta" in captured.out
        assert "gNFW" in captured.out
        assert "Power" in captured.out
        assert "TopHat" in captured.out
        assert "Sersic" in captured.out
        assert "Point" in captured.out
        assert "Background" in captured.out


class TestgNFW:
    """Tests for gNFW profile."""

    def test_numerics_change_detected(self, simple_hdu):
        """A fitter notices numerical settings changed after its creation."""
        import socca

        img = data.Image(simple_hdu, noise=noise.Normal(sigma=0.1))

        gnfw = models.gNFW(
            xc=180.0, yc=45.0, rc=0.01, alpha=1.0, beta=4.0, gamma=0.5
        )
        gnfw.Ic = priors.uniform(0.0, 2.0)
        fit = socca.fitter(img, models.Model(gnfw))
        gnfw.rz = np.logspace(-2, 0, 5)
        with pytest.raises(ValueError, match="model changed"):
            fit._checkmodel()

    def test_numerics_rebuild_profile(self):
        """Changing eps, rz or interpolate after a first use takes effect."""
        gnfw = models.gNFW(
            xc=180.0, yc=45.0, rc=0.01, Ic=1.0, alpha=1.0, beta=4.0, gamma=0.5
        )
        r = jp.linspace(1e-4, 0.05, 50)
        args = (1.0, 0.01, 1.0, 4.0, 0.5)
        before = np.asarray(gnfw.profile(r, *args))
        gnfw.rz = np.logspace(-7, 2, 200)
        after = np.asarray(gnfw.profile(r, *args))
        assert np.abs(after - before).max() > 0
        np.testing.assert_array_equal(
            copy.deepcopy(gnfw).profile(r, *args), after
        )
        with pytest.raises(ValueError, match="read-only"):
            gnfw.rz[-1] = 1.0  # in place, it would not recompile

    def test_emulator_false(self):
        """emulator=False means no emulator, like None."""
        gnfw = models.gNFW(emulator=False)
        assert gnfw._emulator_model is None
        assert copy.deepcopy(gnfw)._emulator_model is None

    def test_initialization(self):
        """Test gNFW profile initialization."""
        gnfw = models.gNFW(
            xc=180.0, yc=45.0, rc=0.01, Ic=1.0, alpha=1.0, beta=3.0, gamma=1.0
        )
        assert gnfw.xc == 180.0
        assert gnfw.rc == 0.01
        assert gnfw.alpha == 1.0
        assert gnfw.beta == 3.0
        assert gnfw.gamma == 1.0


class TestPower:
    """Tests for Power profile."""

    def test_initialization(self):
        """Test Power profile initialization."""
        power = models.Power(xc=180.0, yc=45.0, rc=0.01, Ic=1.0, alpha=2.0)
        assert power.xc == 180.0
        assert power.yc == 45.0
        assert power.rc == 0.01
        assert power.Ic == 1.0
        assert power.alpha == 2.0

    def test_profile_function(self):
        """Test Power profile function."""
        r = jp.array([0.005, 0.01, 0.02, 0.05])
        result = models.Power.profile(r, Ic=1.0, rc=0.01, alpha=2.0)
        assert result.shape == r.shape

    def test_profile_at_scale_radius(self):
        """Test that profile equals Ic at scale radius."""
        rc = 0.01
        r = jp.array([rc])
        Ic = 5.0
        result = models.Power.profile(r, Ic=Ic, rc=rc, alpha=2.0)
        assert float(result[0]) == pytest.approx(Ic)

    def test_profile_decreases_with_radius(self):
        """Test that profile decreases with radius for positive alpha."""
        r = jp.array([0.005, 0.01, 0.02, 0.05, 0.1])
        result = models.Power.profile(r, Ic=1.0, rc=0.01, alpha=2.0)
        for i in range(len(r) - 1):
            assert float(result[i]) >= float(result[i + 1])


class TestTopHat:
    """Tests for TopHat profile."""

    def test_initialization(self):
        """Test TopHat profile initialization."""
        tophat = models.TopHat(rc=0.01, Ic=5.0)
        assert tophat.rc == 0.01
        assert tophat.Ic == 5.0

    def test_profile_function(self):
        """Test TopHat profile function."""
        r = jp.array([0.0, 0.005, 0.01, 0.02])
        result = models.TopHat.profile(r, rc=0.01, Ic=1.0)
        assert result.shape == r.shape

    def test_profile_inside_cutoff(self):
        """Test that profile equals Ic inside cutoff radius."""
        r = jp.array([0.0, 0.005, 0.009])
        Ic = 5.0
        result = models.TopHat.profile(r, rc=0.01, Ic=Ic)
        for val in result:
            assert float(val) == pytest.approx(Ic)

    def test_profile_outside_cutoff(self):
        """Test that profile is 0 outside cutoff radius."""
        r = jp.array([0.011, 0.02, 0.05])
        result = models.TopHat.profile(r, rc=0.01, Ic=1.0)
        for val in result:
            assert float(val) == pytest.approx(0.0)


class TestSimpleBridge:
    """Tests for SimpleBridge model."""

    def test_initialization(self):
        """Test SimpleBridge default initialization."""
        bridge = models.SimpleBridge(xc=180.0, yc=45.0, rs=0.01, Is=1.0)
        assert bridge.xc == 180.0
        assert bridge.yc == 45.0
        assert bridge.rs == 0.01
        assert bridge.Is == 1.0
        assert bridge.theta == 0.0

    def test_parlist(self):
        """Test SimpleBridge parameter list."""
        bridge = models.SimpleBridge(xc=180.0, yc=45.0, rs=0.01, Is=1.0)
        params = bridge.parlist()
        assert "xc" in params
        assert "yc" in params
        assert "rs" in params
        assert "Is" in params
        assert "theta" in params
        assert "e" in params
        assert any("alpha" in p for p in params)
        assert any("beta" in p for p in params)

    def test_type_stored_in_model(self):
        """Test that bridge type is stored correctly in Model."""
        bridge = models.SimpleBridge(xc=180.0, yc=45.0, rs=0.01, Is=1.0)
        mod = models.Model(bridge)
        assert mod.type[0] == "SimpleBridge"

    def test_unsupported_point_type(self):
        """Test that Point raises TypeError."""
        with pytest.raises(TypeError, match="Point"):
            models.SimpleBridge(
                radial=models.Point(xc=180.0, yc=45.0, Ic=1.0),
                xc=180.0,
                yc=45.0,
                rs=0.01,
                Is=1.0,
            )

    def test_unsupported_background_type(self):
        """Test that Background raises TypeError."""
        with pytest.raises(TypeError, match="Background"):
            models.SimpleBridge(
                radial=models.Background(),
                xc=180.0,
                yc=45.0,
                rs=0.01,
                Is=1.0,
            )

    def test_default_radial_not_shared_between_instances(self):
        """Test that default-constructed SimpleBridges don't share sub-components."""
        b1 = models.SimpleBridge(xc=180.0, yc=45.0, rs=0.01, Is=1.0)
        b2 = models.SimpleBridge(xc=180.0, yc=45.0, rs=0.01, Is=1.0)
        assert b1.radial is not b2.radial
        assert b1.parallel is not b2.parallel


class TestMesaBridge:
    """Tests for MesaBridge model."""

    def test_initialization(self):
        """Test MesaBridge default initialization."""
        bridge = models.MesaBridge(xc=180.0, yc=45.0, rs=0.01, Is=1.0)
        assert bridge.xc == 180.0
        assert bridge.yc == 45.0
        assert bridge.rs == 0.01
        assert bridge.Is == 1.0

    def test_parlist(self):
        """Test MesaBridge parameter list."""
        bridge = models.MesaBridge(xc=180.0, yc=45.0, rs=0.01, Is=1.0)
        params = bridge.parlist()
        assert "xc" in params
        assert "yc" in params
        assert "rs" in params
        assert "Is" in params
        assert any("alpha" in p for p in params)

    def test_type_stored_in_model(self):
        """Test that bridge type is stored correctly in Model."""
        bridge = models.MesaBridge(xc=180.0, yc=45.0, rs=0.01, Is=1.0)
        mod = models.Model(bridge)
        assert mod.type[0] == "MesaBridge"

    def test_default_radial_not_shared_between_instances(self):
        """Test that default-constructed MesaBridges don't share sub-components."""
        b1 = models.MesaBridge(xc=180.0, yc=45.0, rs=0.01, Is=1.0)
        b2 = models.MesaBridge(xc=180.0, yc=45.0, rs=0.01, Is=1.0)
        assert b1.radial is not b2.radial
        assert b1.parallel is not b2.parallel


class TestDisk:
    """Tests for Disk model."""

    def test_default_initialization(self):
        """Test Disk default initialization."""
        disk = models.Disk()
        assert disk.radial.re is None
        assert disk.vertical.zs is None
        assert disk.vertical.inc == 0.0

    def test_parlist(self):
        """Test Disk parameter list."""
        disk = models.Disk()
        params = disk.parlist()
        assert "radial.re" in params
        assert "vertical.zs" in params
        assert "vertical.inc" in params

    def test_type_stored_in_model(self):
        """Test that Disk type is stored correctly in Model."""
        radial = models.Sersic(re=0.005, Ie=1.0, ns=1.0, xc=180.0, yc=45.0)
        vertical = models.disk.vertical.HyperSecantHeight(zs=0.0005)
        disk = models.Disk(radial=radial, vertical=vertical)
        mod = models.Model(disk)
        assert mod.type[0] == "Disk"

    def test_default_sub_components_not_shared_between_instances(self):
        """Test that default-constructed Disks don't share sub-components.

        Regression test: Disk.__init__ used to default radial/vertical via
        mutable default arguments (radial=Sersic(), vertical=Height()),
        evaluated once and shared across every Disk() call that doesn't
        pass its own -- so setting one Disk's radial.xc would silently
        leak into every other default-constructed Disk.
        """
        d1 = models.Disk()
        d2 = models.Disk()
        assert d1.radial is not d2.radial
        assert d1.vertical is not d2.vertical
        d1.radial.re = 0.005
        assert d2.radial.re is None

    @pytest.mark.parametrize(
        "vertical, kw",
        [
            ("HyperSecantHeight", dict(alpha=2.0)),
            ("HyperSecantHeight", dict(alpha=0.7)),
            ("ExponentialHeight", {}),
        ],
    )
    def test_vertical_unit_integral(self, vertical, kw):
        """Vertical profiles integrate to one along z."""
        zs = 2e-4
        z = np.linspace(-200 * zs, 200 * zs, 400001)
        prof = getattr(models.disk.vertical, vertical).profile
        h = np.asarray(prof(z, zs=zs, **kw))
        assert float(jp.trapezoid(h, z)) == pytest.approx(1.0, rel=1e-4)

    def test_face_on_disk_projects_onto_radial(self, simple_hdu):
        """A face-on disk has the surface brightness of its radial profile."""
        img = data.Image(simple_hdu)
        kw = dict(
            xc=simple_hdu.header["CRVAL1"],
            yc=simple_hdu.header["CRVAL2"],
            rs=0.005,
            Is=3.0,
        )
        vertical = models.disk.vertical.HyperSecantHeight(
            zs=0.001, inc=0.0, losdepth=0.04, losbins=2001
        )
        disk = models.Disk(radial=models.Exponential(**kw), vertical=vertical)
        got = np.asarray(disk.getmap(img))
        exp = np.asarray(models.Exponential(**kw).getmap(img))
        np.testing.assert_allclose(got, exp, rtol=1e-8)


class TestEllipsoid:
    """Tests for Ellipsoid model."""

    def test_default_initialization(self):
        """Test Ellipsoid default initialization."""
        ell = models.Ellipsoid()
        assert ell.radial.re is None
        assert ell.xc is None
        assert ell.yc is None
        assert ell.theta == 0.0
        assert ell.e == 0.0
        assert ell.eratio == 1.0
        assert ell.inc == 0.0
        assert ell.rot == 0.0
        assert ell.losdepth == pytest.approx(10.0 / 60.0 / 60.0)
        assert ell.losbins == 200

    def test_custom_radial(self):
        """Test Ellipsoid with an explicit radial profile."""
        radial = models.Sersic(re=0.001, Ie=10.0, ns=1.0)
        ell = models.Ellipsoid(radial=radial)
        assert ell.radial.re == 0.001
        assert ell.radial.Ie == 10.0
        assert ell.radial.ns == 1.0

    def test_parlist(self):
        """Test Ellipsoid parameter list."""
        ell = models.Ellipsoid()
        params = ell.parlist()
        assert "radial.re" in params
        assert "radial.Ie" in params
        assert "radial.ns" in params
        assert "xc" in params
        assert "yc" in params
        assert "theta" in params
        assert "e" in params
        assert "eratio" in params
        assert "inc" in params
        assert "rot" in params
        assert "losdepth" in params
        assert "losbins" in params

    def test_type_stored_in_model(self):
        """Test that Ellipsoid type is stored correctly in Model."""
        radial = models.Sersic(re=0.001, Ie=10.0, ns=1.0)
        ell = models.Ellipsoid(radial=radial, xc=180.0, yc=45.0)
        mod = models.Model(ell)
        assert mod.type[0] == "Ellipsoid"

    def test_radial_stripped_of_geometric_params(self):
        """Test that radial loses xc/yc/theta/e/cbox once owned by Ellipsoid."""
        ell = models.Ellipsoid()
        assert not hasattr(ell.radial, "xc")
        assert not hasattr(ell.radial, "yc")
        assert not hasattr(ell.radial, "theta")
        assert not hasattr(ell.radial, "e")
        assert not hasattr(ell.radial, "cbox")
        assert "xc" not in ell.radial.units
        assert "theta" not in ell.radial.units

    def test_inherits_from_radial_when_not_set_on_ellipsoid(self):
        """Test that xc/yc/theta/e set on radial are inherited by Ellipsoid."""
        radial = models.Sersic(
            re=0.001, Ie=10.0, ns=1.0, xc=180.0, yc=45.0, theta=0.5, e=0.3
        )
        ell = models.Ellipsoid(radial=radial)
        assert ell.xc == 180.0
        assert ell.yc == 45.0
        assert ell.theta == 0.5
        assert ell.e == 0.3

    def test_ellipsoid_value_wins_and_warns_when_both_set(self):
        """Test that Ellipsoid's own kwarg wins over radial's, with a warning."""
        radial = models.Sersic(re=0.001, Ie=10.0, ns=1.0, xc=180.0)
        with pytest.warns(UserWarning, match="both set"):
            ell = models.Ellipsoid(radial=radial, xc=200.0)
        assert ell.xc == 200.0

    def test_default_radial_not_shared_between_instances(self):
        """Test that default-constructed Ellipsoids don't share sub-components."""
        ell1 = models.Ellipsoid()
        ell2 = models.Ellipsoid()
        assert ell1.radial is not ell2.radial
        ell1.radial.re = 0.005
        assert ell2.radial.re is None

    def test_e_negative_warns(self):
        """Test that a negative e warns."""
        ell = models.Ellipsoid()
        with pytest.warns(UserWarning, match="less than 0"):
            ell.e = -0.1

    def test_e_too_large_warns(self):
        """Test that e >= 1 warns."""
        ell = models.Ellipsoid()
        with pytest.warns(UserWarning, match="greater than"):
            ell.e = 1.5

    def test_e_valid_no_warning(self, recwarn):
        """Test that a valid e in [0, 1) does not warn."""
        ell = models.Ellipsoid()
        ell.e = 0.5
        assert len(recwarn) == 0

    def test_eratio_too_large_warns(self):
        """Test that eratio > 1 warns."""
        ell = models.Ellipsoid()
        with pytest.warns(UserWarning, match="greater than"):
            ell.eratio = 1.5

    def test_eratio_negative_warns(self):
        """Test that eratio < 0 warns."""
        ell = models.Ellipsoid()
        with pytest.warns(UserWarning, match="less than 0"):
            ell.eratio = -0.1

    def test_eratio_valid_no_warning(self, recwarn):
        """Test that a valid eratio in [0, 1] does not warn."""
        ell = models.Ellipsoid()
        ell.eratio = 0.4
        assert len(recwarn) == 0

    @staticmethod
    def _ellipsoid(**kwargs):
        """Create a fixed Ellipsoid."""
        ell = models.Ellipsoid(radial=models.Sersic(re=1e-3, Ie=1.0))
        ell.xc, ell.yc = 0.0, 0.0
        for key, value in kwargs.items():
            setattr(ell, key, value)
        return ell

    def test_theta_wide_prior_warns(self):
        """A theta prior wider than its period warns when added."""
        ell = self._ellipsoid(theta=priors.uniform(-np.pi, np.pi))
        with pytest.warns(UserWarning, match="more than its period"):
            models.Model(ell)

    def test_theta_narrow_prior_no_warning(self, recwarn):
        """A theta prior narrower than its period does not warn when added."""
        models.Model(self._ellipsoid(theta=priors.uniform(-0.7, 0.7)))
        assert len(recwarn) == 0

    def test_theta_exact_period_warns_periodic(self, recwarn):
        """A theta prior spanning its period warns about periodic sampling."""
        models.Model(self._ellipsoid(theta=priors.uniform(0.00, np.pi)))
        assert len(recwarn) == 1
        assert "periodic" in str(recwarn[0].message)

    def test_rot_wide_prior_warns(self):
        """A rot prior wider than its period warns when added."""
        ell = self._ellipsoid(rot=priors.uniform(-np.pi, np.pi))
        with pytest.warns(UserWarning, match="more than its period"):
            models.Model(ell)

    def test_tied_e_does_not_warn_or_crash(self, recwarn):
        """Tying e via boundto() skips range validation and resolves."""
        src = self._ellipsoid(e=0.4)
        ell = self._ellipsoid()
        ell.e = priors.boundto(src, "e")
        assert ell._resolve("e") == 0.4
        assert len(recwarn) == 0

    def test_profile_major_axis_surface(self):
        """Test that the profile equals Ie on the major (x) axis surface."""
        re, e, eratio, Ie, ns = 0.001, 0.5, 1.0, 3.0, 1.0
        val = models.Ellipsoid().profile(
            re, 0.0, 0.0, e=e, eratio=eratio, Ie=Ie, re=re, ns=ns
        )
        assert float(val) == pytest.approx(Ie, rel=1e-3)

    def test_profile_first_minor_axis_surface(self):
        """Test that the profile equals Ie on the first minor (y) axis surface."""
        re, e, eratio, Ie, ns = 0.001, 0.5, 1.0, 3.0, 1.0
        rs1 = re * (1 - e)
        val = models.Ellipsoid().profile(
            0.0, rs1, 0.0, e=e, eratio=eratio, Ie=Ie, re=re, ns=ns
        )
        assert float(val) == pytest.approx(Ie, rel=1e-3)

    def test_profile_second_minor_axis_uses_eratio(self):
        """Test that the profile equals Ie on the eratio-scaled z-axis surface."""
        re, e, eratio, Ie, ns = 0.001, 0.6, 0.5, 3.0, 1.0
        rs2 = re * (1 - eratio * e)
        val = models.Ellipsoid().profile(
            0.0, 0.0, rs2, e=e, eratio=eratio, Ie=Ie, re=re, ns=ns
        )
        assert float(val) == pytest.approx(Ie, rel=1e-3)

    def test_eratio_default_makes_minor_axes_equal(self):
        """Test that eratio=1 (default) gives equal minor-axis scales."""
        re, e = 0.001, 0.6
        rs1 = re * (1 - e)
        rs2 = re * (1 - 1.0 * e)
        assert rs1 == pytest.approx(rs2)

    @staticmethod
    def _major_axis(m):
        """Major-axis angle of a map from its second moments (rad, mod pi)."""
        y, x = np.mgrid[: m.shape[0], : m.shape[1]]
        w = m / m.sum()
        dx, dy = x - (w * x).sum(), y - (w * y).sum()
        cxx, cyy = (w * dx * dx).sum(), (w * dy * dy).sum()
        return 0.5 * np.arctan2(2 * (w * dx * dy).sum(), cxx - cyy) % np.pi

    @pytest.mark.parametrize("inc", [0.0, 1.0])
    def test_rot_zero_puts_major_axis_along_theta(self, simple_hdu, inc):
        """With rot = 0 the major axis has position angle theta, as Sersic."""
        img = data.Image(simple_hdu)
        xc, yc = simple_hdu.header["CRVAL1"], simple_hdu.header["CRVAL2"]
        kw = dict(xc=xc, yc=yc, theta=0.5, e=0.5)
        ref = models.Sersic(re=0.005, Ie=1.0, ns=1.0, **kw)
        ell = models.Ellipsoid(
            radial=models.Sersic(re=0.005, Ie=1.0, ns=1.0),
            inc=inc,
            losdepth=0.03,
            **kw,
        )
        got = self._major_axis(np.asarray(ell.getmap(img)))
        exp = self._major_axis(np.asarray(ref.getmap(img)))
        assert got == pytest.approx(exp, abs=1e-2)


class TestBuildKwargsEvaluate:
    """Tests for _build_kwargs and _evaluate methods."""

    @pytest.fixture
    def simple_img(self, simple_hdu, gaussian_psf):
        """Create a simple Image for evaluation tests."""
        img = data.Image(simple_hdu, noise=noise.Normal(sigma=0.1))
        img.addpsf(gaussian_psf)
        return img

    def test_beta_build_kwargs(self, simple_hdu):
        """Test Beta._build_kwargs extracts correct parameters."""
        beta = models.Beta(xc=180.0, yc=45.0, rc=0.01, Ic=1.0, beta=0.5)
        pars = {
            "comp_00_xc": 180.0,
            "comp_00_yc": 45.0,
            "comp_00_theta": 0.0,
            "comp_00_e": 0.0,
            "comp_00_cbox": 0.0,
            "comp_00_rc": 0.01,
            "comp_00_Ic": 1.0,
            "comp_00_alpha": 2.0,
            "comp_00_beta": 0.5,
        }
        kwarg = beta._build_kwargs(pars, "comp_00")
        assert "xc" in kwarg
        assert "yc" in kwarg
        assert "rc" in kwarg
        assert "Ic" in kwarg
        assert "beta" in kwarg
        assert kwarg["xc"] == 180.0
        assert kwarg["rc"] == 0.01

    def test_beta_evaluate(self, simple_img):
        """Test Beta._evaluate produces correct shape."""
        beta = models.Beta(
            xc=simple_img.hdu.header["CRVAL1"],
            yc=simple_img.hdu.header["CRVAL2"],
            rc=0.005,
            Ic=1.0,
            beta=0.5,
        )
        kwarg = {
            "xc": simple_img.hdu.header["CRVAL1"],
            "yc": simple_img.hdu.header["CRVAL2"],
            "theta": 0.0,
            "e": 0.0,
            "cbox": 0.0,
            "rc": 0.005,
            "Ic": 1.0,
            "alpha": 2.0,
            "beta": 0.5,
        }
        result = beta._evaluate(simple_img, **kwarg)
        assert result.shape == simple_img.data.shape

    def test_point_build_kwargs(self):
        """Test Point._build_kwargs extracts correct parameters."""
        point = models.Point(xc=180.0, yc=45.0, Ic=100.0)
        pars = {
            "comp_00_xc": 180.0,
            "comp_00_yc": 45.0,
            "comp_00_Ic": 100.0,
        }
        kwarg = point._build_kwargs(pars, "comp_00")
        assert kwarg["xc"] == 180.0
        assert kwarg["yc"] == 45.0
        assert kwarg["Ic"] == 100.0

    def test_point_evaluate(self, simple_img):
        """Test Point._evaluate returns complex Fourier array on padded grid."""
        point = models.Point(
            xc=simple_img.hdu.header["CRVAL1"],
            yc=simple_img.hdu.header["CRVAL2"],
            Ic=100.0,
        )
        kwarg = {
            "xc": simple_img.hdu.header["CRVAL1"],
            "yc": simple_img.hdu.header["CRVAL2"],
            "Ic": 100.0,
        }
        result = point._evaluate(simple_img, **kwarg)
        padded = data.pad_size(simple_img.data.shape)
        expected_shape = (padded[0], padded[1] // 2 + 1)
        assert result.shape == expected_shape

    def test_background_build_kwargs(self):
        """Test Background._build_kwargs extracts parameters."""
        bkg = models.Background(a0=5.0)
        pars = {
            "comp_00_a0": 5.0,
            "comp_00_a1x": 0.0,
            "comp_00_a1y": 0.0,
            "comp_00_a2xx": 0.0,
            "comp_00_a2xy": 0.0,
            "comp_00_a2yy": 0.0,
            "comp_00_a3xxx": 0.0,
            "comp_00_a3xxy": 0.0,
            "comp_00_a3xyy": 0.0,
            "comp_00_a3yyy": 0.0,
            "comp_00_rs": 1.0,
        }
        kwarg = bkg._build_kwargs(pars, "comp_00")
        assert kwarg["a0"] == 5.0
        assert kwarg["rs"] == 1.0

    def test_background_evaluate(self, simple_img):
        """Test Background._evaluate produces correct shape."""
        bkg = models.Background(a0=5.0, rs=1.0)
        kwarg = {
            "a0": 5.0,
            "a1x": 0.0,
            "a1y": 0.0,
            "a2xx": 0.0,
            "a2xy": 0.0,
            "a2yy": 0.0,
            "a3xxx": 0.0,
            "a3xxy": 0.0,
            "a3xyy": 0.0,
            "a3yyy": 0.0,
            "rs": 1.0,
        }
        result = bkg._evaluate(simple_img, **kwarg)
        assert result.shape == simple_img.data.shape

    def test_bridge_build_kwargs_fallback(self):
        """Test Bridge._build_kwargs falls back for scale params."""
        bridge = models.SimpleBridge(xc=180.0, yc=45.0, rs=0.01, Is=1.0)
        cid = bridge.id
        pars = {
            f"{cid}_xc": 180.0,
            f"{cid}_yc": 45.0,
            f"{cid}_theta": 0.0,
            f"{cid}_Is": 1.0,
            f"{cid}_rs": 0.01,
            f"{cid}_e": 0.5,
            f"{cid}_radial.alpha": 2.0,
            f"{cid}_radial.beta": 0.5,
        }
        kwarg = bridge._build_kwargs(pars, cid)
        assert kwarg["xc"] == 180.0
        assert kwarg["yc"] == 45.0
        assert kwarg["Is"] == 1.0
        assert "r_Ic" in kwarg
        assert "r_rc" in kwarg
        assert "r_alpha" in kwarg
        assert "r_beta" in kwarg
        assert "z_Ic" in kwarg
        assert "z_rc" in kwarg

    def test_bridge_evaluate(self, simple_img):
        """Test Bridge._evaluate produces correct shape."""
        bridge = models.SimpleBridge(
            xc=simple_img.hdu.header["CRVAL1"],
            yc=simple_img.hdu.header["CRVAL2"],
            rs=0.005,
            Is=1.0,
        )
        cid = bridge.id
        pars = {
            f"{cid}_xc": simple_img.hdu.header["CRVAL1"],
            f"{cid}_yc": simple_img.hdu.header["CRVAL2"],
            f"{cid}_theta": 0.0,
            f"{cid}_Is": 1.0,
            f"{cid}_rs": 0.005,
            f"{cid}_e": 0.5,
            f"{cid}_radial.alpha": 2.0,
            f"{cid}_radial.beta": 0.5,
        }
        kwarg = bridge._build_kwargs(pars, cid)
        result = bridge._evaluate(simple_img, **kwarg)
        assert result.shape == simple_img.data.shape

    def test_ellipsoid_build_kwargs(self):
        """Test Ellipsoid._build_kwargs extracts radial and flat parameters."""
        radial = models.Sersic(re=0.001, Ie=10.0, ns=1.0)
        ell = models.Ellipsoid(radial=radial, xc=180.0, yc=45.0)
        cid = ell.id
        pars = {
            f"{cid}_radial.re": 0.001,
            f"{cid}_radial.Ie": 10.0,
            f"{cid}_radial.ns": 1.0,
            f"{cid}_xc": 180.0,
            f"{cid}_yc": 45.0,
            f"{cid}_theta": 0.0,
            f"{cid}_e": 0.5,
            f"{cid}_eratio": 0.4,
            f"{cid}_inc": 0.3,
            f"{cid}_rot": 0.1,
            f"{cid}_losdepth": 10.0 / 60.0 / 60.0,
            f"{cid}_losbins": 200,
        }
        kwarg = ell._build_kwargs(pars, cid)
        assert kwarg["re"] == 0.001
        assert kwarg["Ie"] == 10.0
        assert kwarg["ns"] == 1.0
        assert kwarg["xc"] == 180.0
        assert kwarg["yc"] == 45.0
        assert kwarg["e"] == 0.5
        assert kwarg["eratio"] == 0.4
        assert kwarg["inc"] == 0.3
        assert kwarg["rot"] == 0.1

    def test_ellipsoid_evaluate(self, simple_img):
        """Test Ellipsoid._evaluate produces correct shape."""
        radial = models.Sersic(re=0.0005, Ie=10.0, ns=1.0)
        ell = models.Ellipsoid(radial=radial)
        kwarg = {
            "re": 0.0005,
            "Ie": 10.0,
            "ns": 1.0,
            "xc": simple_img.hdu.header["CRVAL1"],
            "yc": simple_img.hdu.header["CRVAL2"],
            "theta": 0.0,
            "e": 0.5,
            "eratio": 0.5,
            "inc": 0.5,
            "rot": 0.0,
            "losdepth": 10.0 / 60.0 / 60.0,
            "losbins": 50,
        }
        result = ell._evaluate(simple_img, **kwarg)
        assert result.shape == simple_img.data.shape
        assert bool(jp.all(jp.isfinite(result)))


class TestGetmodelWithBridge:
    """Tests for Model.getmodel with Bridge components."""

    @pytest.fixture
    def bridge_model_and_image(self, simple_hdu, gaussian_psf):
        """Create a Bridge model and image for testing."""
        img = data.Image(simple_hdu, noise=noise.Normal(sigma=0.1))
        img.addpsf(gaussian_psf)

        xc = simple_hdu.header["CRVAL1"]
        yc = simple_hdu.header["CRVAL2"]

        bridge = models.SimpleBridge(
            xc=xc,
            yc=yc,
            rs=0.005,
            Is=priors.loguniform(0.1, 10.0),
            theta=0.0,
        )
        mod = models.Model(bridge)
        return mod, img

    def test_getmodel_returns_tuple(self, bridge_model_and_image):
        """Test that getmodel returns 4-tuple with Bridge."""
        mod, img = bridge_model_and_image
        pp = [1.0]
        result = mod.getmodel(img, pp)
        assert len(result) == 4

    def test_getmodel_shapes(self, bridge_model_and_image):
        """Test that getmodel outputs have correct shapes."""
        mod, img = bridge_model_and_image
        pp = [1.0]
        mraw, msmo, mbkg, mneg = mod.getmodel(img, pp)
        assert mraw.shape == img.data.shape
        assert msmo.shape == img.data.shape
        assert mbkg.shape == img.data.shape
        assert mneg.shape == img.data.shape


class TestGetmodelWithEllipsoid:
    """Tests for Model.getmodel with Ellipsoid components."""

    @pytest.fixture
    def ellipsoid_model_and_image(self, simple_hdu, gaussian_psf):
        """Create an Ellipsoid model and image for testing."""
        img = data.Image(simple_hdu, noise=noise.Normal(sigma=0.1))
        img.addpsf(gaussian_psf)

        xc = simple_hdu.header["CRVAL1"]
        yc = simple_hdu.header["CRVAL2"]

        radial = models.Sersic(
            re=0.005, ns=1.0, Ie=priors.loguniform(0.1, 10.0)
        )
        ell = models.Ellipsoid(radial=radial, xc=xc, yc=yc, e=0.5, inc=0.3)
        mod = models.Model(ell)
        return mod, img

    def test_radial_extra_parameter(self, simple_hdu):
        """A parameter added to the radial profile is not passed to it."""
        img = data.Image(simple_hdu, noise=noise.Normal(sigma=0.1))
        radial = models.Sersic(Ie=1.0, ns=1.0)
        radial.addparameter("scale", 2.0)
        radial.re = priors.function(lambda s: s * 0.0025, (radial, "scale"))
        ell = models.Ellipsoid(
            radial=radial, xc=180.0, yc=45.0, losdepth=0.005, losbins=51
        )
        np.testing.assert_allclose(
            models.Model(ell).getmap(img), ell.getmap(img)
        )

    def test_getmodel_returns_tuple(self, ellipsoid_model_and_image):
        """Test that getmodel returns 4-tuple with Ellipsoid."""
        mod, img = ellipsoid_model_and_image
        pp = [1.0]
        result = mod.getmodel(img, pp)
        assert len(result) == 4

    def test_getmodel_shapes(self, ellipsoid_model_and_image):
        """Test that getmodel outputs have correct shapes."""
        mod, img = ellipsoid_model_and_image
        pp = [1.0]
        mraw, msmo, mbkg, mneg = mod.getmodel(img, pp)
        assert mraw.shape == img.data.shape
        assert msmo.shape == img.data.shape
        assert mbkg.shape == img.data.shape
        assert mneg.shape == img.data.shape
        assert bool(jp.all(jp.isfinite(mraw)))

    @pytest.fixture
    def disk_and_tied_ellipsoid(self, simple_hdu, gaussian_psf):
        """Create a Disk and an Ellipsoid tied to it via boundto()."""
        img = data.Image(simple_hdu, noise=noise.Normal(sigma=0.1))
        img.addpsf(gaussian_psf)

        xc = simple_hdu.header["CRVAL1"]
        yc = simple_hdu.header["CRVAL2"]

        disk_radial = models.Sersic(
            re=0.005, Ie=5.0, ns=1.0, xc=xc, yc=yc, theta=0.4
        )
        disk_vertical = models.disk.vertical.HyperSecantHeight(
            zs=0.0005, inc=0.6
        )
        disk = models.Disk(radial=disk_radial, vertical=disk_vertical)

        ell_radial = models.Sersic(re=0.001, Ie=1.0, ns=0.5)
        ell = models.Ellipsoid(radial=ell_radial)
        ell.xc = priors.boundto(disk, "xc")
        ell.yc = priors.boundto(disk, "yc")
        ell.theta = priors.boundto(disk, "theta")
        ell.inc = priors.boundto(disk, "inc")
        ell.e = 0.6

        return disk, ell, img

    def test_getmodel_with_disk_and_tied_ellipsoid(
        self, disk_and_tied_ellipsoid
    ):
        """Test a composite Disk + boundto()-tied Ellipsoid via Model.getmodel."""
        disk, ell, img = disk_and_tied_ellipsoid
        mod = models.Model()
        mod.addcomponent(disk)
        mod.addcomponent(ell)

        mraw, msmo, mbkg, mneg = mod.getmodel(img, [])
        assert mraw.shape == img.data.shape
        assert bool(jp.all(jp.isfinite(mraw)))

    def test_direct_getmap_resolves_boundto_ties(
        self, disk_and_tied_ellipsoid
    ):
        """Test that Ellipsoid.getmap() called directly resolves boundto() ties.

        Regression test: calling a component's own getmap() (not through
        Model) previously failed to resolve boundto()-tied parameters.
        """
        _, ell, img = disk_and_tied_ellipsoid
        mraw = ell.getmap(img, convolve=False)
        assert mraw.shape == img.data.shape
        assert bool(jp.all(jp.isfinite(mraw)))

    def test_direct_getmap_matches_resolved_value(
        self, disk_and_tied_ellipsoid
    ):
        """Test that the tied xc used by getmap() matches the disk's value."""
        disk, ell, _ = disk_and_tied_ellipsoid
        assert ell._resolve("xc") == disk.radial.xc
        assert ell._resolve("theta") == disk.radial.theta
        assert ell._resolve("inc") == disk.vertical.inc


class TestModelComposition:
    """Tests for model composition operators."""

    def test_add_profiles(self):
        """Test adding profiles to create composite model."""
        beta = models.Beta(xc=180.0, yc=45.0, rc=0.01, Ic=1.0, beta=0.5)
        mod = models.Model(beta)
        point = models.Point(xc=180.1, yc=45.1, Ic=100.0)
        mod.addcomponent(point)
        assert mod.ncomp == 2


class TestTies:
    """Tie resolution by object, in dependency order, with cycle checks."""

    @pytest.fixture
    def img(self, simple_hdu):
        """Create an Image without PSF."""
        return data.Image(simple_hdu, noise=noise.Normal(sigma=0.1))

    @staticmethod
    def beta(**kwargs):
        """Create a fixed Beta profile."""
        pars = dict(xc=180.0, yc=45.0, rc=0.005, Ic=1.0, beta=0.6)
        pars.update(kwargs)
        return models.Beta(**pars)

    def test_plain_callable_refused(self):
        """A callable that is not a tie points to priors.function."""
        import functools

        double = functools.partial(np.multiply, 2.0)
        with pytest.raises(TypeError, match="priors.function"):
            models.Beta(xc=180.0, yc=45.0, rc=double, Ic=1.0, beta=0.6)
        beta = models.Beta(xc=180.0, yc=45.0, rc=0.005, Ic=1.0, beta=0.6)
        with pytest.raises(TypeError, match="priors.function"):
            beta.rc = double
        beta.rc = priors.function(double, (beta, "Ic"))

    def test_standalone_chain_linear(self):
        """A long chain of ties resolves each source once."""
        import time

        beta = models.Beta(xc=180.0, yc=45.0, rc=0.005, Ic=1.0, beta=0.6)
        beta.addparameter("p0", 2.0)
        for i in range(1, 24):
            beta.addparameter(
                f"p{i}",
                priors.function(lambda x: x + 1.0, (beta, f"p{i - 1}")),
            )
        start = time.perf_counter()
        assert beta._resolve("p23") == 25.0
        assert time.perf_counter() - start < 1.0  # 2^24 resolves took ~25 s

    def test_boundto_follows_object(self, img):
        """Binding follows the bound object, not its position."""
        a = self.beta(xc=180.003)
        b = self.beta()
        b.xc = priors.boundto(a, "xc")
        mod = models.Model(a)
        mod.addcomponent(b)
        ref = models.Model(self.beta(xc=180.003))
        ref.addcomponent(self.beta(xc=180.003))
        np.testing.assert_allclose(mod.getmap(img), ref.getmap(img))

    def test_boundto_order_independent(self, img):
        """A component may be bound to one added after it."""
        a = self.beta(xc=180.003)
        b = self.beta()
        b.xc = priors.boundto(a, "xc")
        mod = models.Model(b)
        mod.addcomponent(a)
        ref = models.Model(self.beta(xc=180.003))
        ref.addcomponent(self.beta(xc=180.003))
        np.testing.assert_allclose(mod.getmap(img), ref.getmap(img))

    def test_boundto_missing_object(self, img):
        """Binding to a component outside the model raises."""
        a = self.beta()
        b = self.beta()
        b.xc = priors.boundto(a, "xc")
        mod = models.Model(b)
        with pytest.raises(ValueError, match="not in this model"):
            mod.getmap(img)

    def test_boundto_sub_component(self):
        """Binding accepts a Disk sub-component."""
        disk = models.Disk(
            radial=models.Exponential(xc=180.002, yc=45.0, rs=0.005, Is=1.0),
            vertical=models.disk.vertical.HyperSecantHeight(zs=1e-3, inc=0.5),
        )
        b = self.beta()
        b.xc = priors.boundto(disk.radial, "xc")
        mod = models.Model(disk)
        mod.addcomponent(b)
        pars = {k: v for k, v in mod.priors.items() if not callable(v)}
        mod._resolve_ties(pars)
        assert pars["comp_01_xc"] == 180.002

    def test_chained_ties(self):
        """A tie to a tied parameter resolves regardless of order."""
        a = self.beta(xc=180.003)
        b = self.beta()
        c = self.beta()
        c.xc = lambda comp_01_xc: comp_01_xc
        b.xc = priors.boundto(a, "xc")
        mod = models.Model(a)
        mod.addcomponent(b)
        mod.addcomponent(c)
        pars = {k: v for k, v in mod.priors.items() if not callable(v)}
        mod._resolve_ties(pars)
        assert pars["comp_02_xc"] == 180.003

    def test_circular_ties(self, img):
        """Circular ties raise."""
        a = self.beta()
        b = self.beta()
        a.xc = lambda comp_01_xc: comp_01_xc
        b.xc = lambda comp_00_xc: comp_00_xc
        mod = models.Model(a)
        mod.addcomponent(b)
        with pytest.raises(ValueError, match="Circular tie"):
            mod.getmap(img)

    def test_unknown_tie_argument(self, img):
        """A lambda naming no model parameter raises."""
        a = self.beta()
        a.rc = lambda comp_05_rc: comp_05_rc
        mod = models.Model(a)
        with pytest.raises(KeyError, match="comp_05_rc"):
            mod.getmap(img)

    def test_bridge_internal_ties_by_position(self, img):
        """Bridge internal ties use the model prefix, also at position 1."""

        def bridge():
            return models.SimpleBridge(
                xc=180.0, yc=45.0, rs=0.005, Is=1.0, e=0.5, theta=0.0
            )

        mod = models.Model(self.beta(Ic=0.0))
        mod.addcomponent(bridge())
        np.testing.assert_allclose(
            mod.getmap(img), models.Model(bridge()).getmap(img)
        )
        np.testing.assert_allclose(mod.getmap(img), bridge().getmap(img))

    @pytest.mark.parametrize("kind", ["point", "background"])
    def test_standalone_boundto(self, img, kind):
        """Standalone getmap resolves boundto on Point and Background."""
        a = self.beta(xc=180.003)
        if kind == "point":
            c = models.Point(xc=180.0, yc=45.0, Ic=1.0)
            c.xc = priors.boundto(a, "xc")
            ref = models.Point(xc=180.003, yc=45.0, Ic=1.0)
        else:
            c = models.Background(a0=1.0)
            c.a0 = priors.boundto(a, "Ic")
            ref = models.Background(a0=1.0)
        np.testing.assert_allclose(c.getmap(img), ref.getmap(img))

    def test_standalone_own_lambda(self, img):
        """Standalone getmap resolves lambdas on the component's own pars."""
        a = self.beta(rc=0.004)
        models.Model(a)
        a.Ic = lambda comp_00_rc: comp_00_rc * 250
        np.testing.assert_allclose(
            a.getmap(img), self.beta(rc=0.004, Ic=1.0).getmap(img)
        )

    def test_standalone_disk_lambda(self, img):
        """Standalone Disk getmap resolves namespaced lambda arguments."""

        def disk(**kwargs):
            radial = models.Exponential(xc=180.0, yc=45.0, rs=0.005, Is=1.0)
            vertical = models.disk.vertical.HyperSecantHeight(zs=1e-3, inc=0.5)
            for key, value in kwargs.items():
                setattr(radial, key, value)
            return models.Disk(radial=radial, vertical=vertical)

        d = disk()
        models.Model(d)
        d.radial.Is = lambda comp_00_radial_rs: comp_00_radial_rs * 200
        np.testing.assert_allclose(d.getmap(img), disk().getmap(img))

    def test_standalone_foreign_lambda(self, img):
        """Standalone getmap raises clearly on another component's names."""
        a = self.beta()
        a.xc = lambda comp_99_xc: comp_99_xc
        with pytest.raises(ValueError, match="function"):
            a.getmap(img)
        models.Model(a)
        with pytest.raises(ValueError, match="Model.getmap"):
            a.getmap(img)

    def test_standalone_circular(self, img):
        """Standalone circular boundto raises."""
        a = self.beta()
        b = self.beta()
        a.xc = priors.boundto(b, "xc")
        b.xc = priors.boundto(a, "xc")
        with pytest.raises(ValueError, match="Circular tie"):
            a.getmap(img)

    def test_function_in_model(self, img):
        """function() with several sources resolves through a Model."""
        a = self.beta(rc=0.004)
        b = self.beta()
        b.addparameter("scale", 0.5)
        b.rc = priors.function(lambda s, r: s * r, (b, "scale"), (a, "rc"))
        mod = models.Model(b)
        mod.addcomponent(a)
        ref = models.Model(self.beta(rc=0.002))
        ref.addcomponent(self.beta(rc=0.004))
        np.testing.assert_allclose(mod.getmap(img), ref.getmap(img))
        np.testing.assert_allclose(
            b.getmap(img), self.beta(rc=0.002).getmap(img)
        )

    def test_function_cycle(self, img):
        """Circular function() ties raise."""
        a = self.beta()
        b = self.beta()
        a.rc = priors.function(lambda r: 2 * r, (b, "rc"))
        b.rc = priors.function(lambda r: 2 * r, (a, "rc"))
        mod = models.Model(a)
        mod.addcomponent(b)
        with pytest.raises(ValueError, match="Circular tie"):
            mod.getmap(img)

    def test_bridge_internals_fixed(self, img):
        """The sub-profiles' scale radius and amplitude cannot be set."""
        for name, value in [("rc", 0.003), ("Ic", 2.0)]:
            bridge = models.SimpleBridge(xc=180.0, yc=45.0, rs=0.005, Is=1.0)
            setattr(bridge.radial, name, value)
            with pytest.raises(ValueError, match="set by the SimpleBridge"):
                bridge.getmap(img)
            with pytest.raises(ValueError, match="set by the SimpleBridge"):
                models.Model(bridge).getmap(img)


class TestIds:
    """Component ids from model positions or names."""

    @staticmethod
    def beta(**kwargs):
        """Create a fixed Beta profile."""
        pars = dict(xc=180.0, yc=45.0, rc=0.005, Ic=1.0, beta=0.6)
        pars.update(kwargs)
        return models.Beta(**pars)

    def test_duplicate_keys(self):
        """Two parameters with the same model key are rejected."""
        pars = dict(xc=180.0, yc=45.0, rs=0.005, Is=1.0)
        a = models.Gaussian(name="core", **pars)
        a.addparameter("x_rs", 1.0)
        mod = models.Model(a)
        with pytest.raises(ValueError, match="share the model key core_x_rs"):
            mod.addcomponent(models.Gaussian(name="core_x", **pars))
        assert mod.ncomp == 1

    def test_failed_add_restores_id(self):
        """A component whose addition fails keeps its previous id."""
        pars = dict(xc=180.0, yc=45.0, rs=0.005, Is=1.0)
        a = models.Gaussian(name="core", **pars)
        a.addparameter("x_rs", 1.0)
        mod = models.Model(a)
        b = models.Gaussian(name="core_x", **pars)
        with pytest.raises(ValueError):
            mod.addcomponent(b)
        assert b.id is None and mod._prefix == ("core",)

    def test_positions_ignore_other_components(self):
        """Unused components and wrappers do not shift the numbering."""
        models.Beta()
        disk = models.Disk(
            radial=models.Exponential(xc=180.0, yc=45.0, rs=0.005, Is=1.0),
            vertical=models.disk.vertical.HyperSecantHeight(zs=1e-3, inc=0.5),
        )
        a = self.beta()
        mod = models.Model(disk)
        mod.addcomponent(a)
        assert (disk.id, a.id) == ("comp_00", "comp_01")
        assert "comp_00_radial.xc" in mod.params
        assert "comp_01_xc" in mod.params
        assert disk.radial.id is None

    @pytest.mark.parametrize(
        "name, cid",
        [
            ("core", "core"),
            (3, "comp_03"),
            ("comp_3", "comp_03"),
            (12, "comp_12"),
        ],
    )
    def test_names(self, name, cid):
        """Names are used as ids; numbers act as positions."""
        a = self.beta(name=name)
        mod = models.Model(a)
        assert a.id == cid
        assert f"{cid}_xc" in mod.params

    @pytest.mark.parametrize("name", ["1core", "a.b", "lambda", -1, 1.5, True])
    def test_invalid_names(self, name):
        """Names that cannot be used in keys raise."""
        with pytest.raises(ValueError, match="Invalid component name"):
            self.beta(name=name)

    def test_name_collision(self):
        """A name or number already used in the model raises."""
        mod = models.Model(self.beta())
        with pytest.raises(ValueError, match="'comp_00' is already used"):
            mod.addcomponent(self.beta(name=0))
        mod.addcomponent(self.beta(name="core"))
        with pytest.raises(ValueError, match="'core' is already used"):
            mod.addcomponent(self.beta(name="core"))
        assert mod.ncomp == 2

    def test_named_then_positional(self):
        """Unnamed components keep their position after named ones."""
        a = self.beta(name="core")
        b = self.beta()
        mod = models.Model(a)
        mod.addcomponent(b)
        assert b.id == "comp_01"

    def test_positional_skips_numbered_name(self):
        """An unnamed component skips an id already taken by a number."""
        mod = models.Model(self.beta(name=1))
        b = self.beta()
        mod.addcomponent(b)
        assert b.id == "comp_02"

    def test_same_object_twice(self):
        """Adding the same object twice raises."""
        a = self.beta()
        mod = models.Model(a)
        with pytest.raises(ValueError, match="already in the model"):
            mod.addcomponent(a)

    def test_rename(self):
        """Names are checked when set, and model members keep theirs."""
        a = models.Beta(xc=180.0, yc=45.0, rc=0.005, Ic=1.0, beta=0.6)
        with pytest.raises(ValueError, match="Invalid component name"):
            a.name = "1 bad"
        a.name = 3
        assert a.name == "comp_03"
        a.name = "halo"
        models.Model(a)
        with pytest.raises(ValueError, match="in a model as 'halo'"):
            a.name = "core"

    def test_second_model(self):
        """A component in a second model takes that model's id."""
        a = self.beta()
        first = models.Model(a)
        second = models.Model(self.beta())
        second.addcomponent(a)
        assert a.id == "comp_01"
        assert first._prefix == ("comp_00",)

    def test_comp_filter(self):
        """Selections by position, id, name and object."""
        a, b, c = self.beta(), self.beta(name="core"), self.beta()
        mod = models.Model(a)
        mod.addcomponent(b)
        mod.addcomponent(c)
        assert mod._comp_filter(None) == [0, 1, 2]
        assert mod._comp_filter([2, "core", "comp_00"]) == [2, 1, 0]
        assert mod._comp_filter(b) == [1]
        assert mod._comp_filter(np.int64(1)) == [1]
        for bad in [3, "comp_05", self.beta(), [1, "core"], [0, 0], [b, 1]]:
            with pytest.raises(ValueError):
                mod._comp_filter(bad)

    def test_lambda_warns(self):
        """Lambda ties warn; function and boundto do not."""
        a = self.beta()
        a.Ic = lambda comp_00_rc: comp_00_rc
        with pytest.warns(FutureWarning, match="priors.function"):
            models.Model(a)
        b = self.beta()
        b.Ic = priors.function(lambda r: r, (b, "rc"))
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            models.Model(b)

    @pytest.mark.parametrize(
        "value", [np.float32(0.004), np.float64(0.004), jp.asarray(0.004)]
    )
    def test_fixed_value_types(self, simple_hdu, value):
        """Fixed values of any numeric type enter the model."""
        img = data.Image(simple_hdu, noise=noise.Normal(sigma=0.1))
        a = self.beta(rc=value)
        b = self.beta()
        b.rc = priors.boundto(a, "rc")
        mod = models.Model(a)
        mod.addcomponent(b)
        ref = models.Model(self.beta(rc=0.004))
        ref.addcomponent(self.beta(rc=0.004))
        np.testing.assert_allclose(mod.getmap(img), ref.getmap(img), rtol=1e-6)


class TestListing:
    """Parameter listings grouped by component."""

    @staticmethod
    def model():
        """Create a model with fixed, free and tied parameters."""
        p = models.Point(xc=20.0, yc=-10.0, Ic=priors.uniform(0.0, 20.0))
        s = models.Sersic(name="core", re=1e-4, Ie=1.0, ns=1.0)
        s.xc = priors.boundto(p, "xc")
        s.yc = priors.function(lambda y, i: y + i, (p, "yc"), (p, "Ic"))
        s.addparameter("re_scale", 0.5, description="Scale of re")
        mod = models.Model(p)
        mod.addcomponent(s)
        return mod, p, s

    def test_model_blocks(self, capsys):
        """Each component is listed under its type and id."""
        mod, _, _ = self.model()
        mod.parameters()
        out = capsys.readouterr().out
        assert "Point (comp_00)" in out
        assert "Sersic (core)" in out
        assert "re_scale" in out and "Scale of re" in out
        assert "Uniform [0, 20]" in out
        assert "= comp_00.xc" in out
        assert "function of comp_00.yc, comp_00.Ic" in out

    def test_freeonly(self, capsys):
        """Only components with free parameters are listed with freeonly."""
        mod, _, _ = self.model()
        mod.parameters(freeonly=True)
        out = capsys.readouterr().out
        assert "Point (comp_00)" in out and "Sersic" not in out

    def test_repr(self):
        """The repr lists the components with their ids."""
        mod, _, _ = self.model()
        assert repr(mod).splitlines() == [
            "Model",
            "  Point (comp_00)",
            "  Sersic (core)",
        ]

    def test_component_title(self, capsys):
        """A component's title shows its id once it is in a model."""
        s = models.Sersic(xc=0.0, yc=0.0, re=1e-4, Ie=1.0)
        s.parameters()
        assert "\nSersic\n======\n" in capsys.readouterr().out
        models.Model(s)
        s.parameters()
        assert "\nSersic (comp_00)\n" in capsys.readouterr().out

    def test_wrapper_listing(self, capsys):
        """Disk lists its namespaced parameters and hyperparameters."""
        models.Disk().parameters()
        out = capsys.readouterr().out
        assert "radial.re" in out and "vertical.inc" in out
        assert "Hyperparameters" in out and "vertical.losbins" in out

    @pytest.mark.parametrize(
        "value, text",
        [
            (None, "None"),
            (np.float32(2.5), "2.5000E+00"),
            (priors.loguniform(1e-5, 1e-3), "LogUniform [1e-05, 0.001]"),
            (numpyro.distributions.HalfNormal(1.0), "HalfNormal [0, inf)"),
            (numpyro.distributions.Normal(0.0, 1.0), "Normal"),
            (lambda comp_00_rc: comp_00_rc, "lambda of comp_00_rc"),
        ],
    )
    def test_format_value(self, value, text):
        """Values, priors and lambdas are shown compactly."""
        from socca.models.base import _format_value

        assert _format_value(value) == text

    def test_source_outside_model(self):
        """Tie sources outside the model are marked."""
        mod, _, s = self.model()
        other = models.Point()
        s.xc = priors.boundto(other, "xc")
        from socca.models.base import _format_value

        assert (
            _format_value(s.xc, mod._source_label)
            == "= Point.xc (not in model)"
        )


class TestUnitsAndChecks:
    """Declared units, periods and parameter checks."""

    @pytest.fixture
    def img(self, simple_hdu):
        """Create an Image without PSF."""
        return data.Image(simple_hdu, noise=noise.Normal(sigma=0.1))

    @staticmethod
    def sersic(**kwargs):
        """Create a fixed Sersic profile."""
        pars = dict(xc=180.0, yc=45.0, re=0.004, Ie=1.0, ns=1.0)
        pars.update(kwargs)
        return models.Sersic(**pars)

    @staticmethod
    def disk(**kwargs):
        """Create a fixed Disk."""
        radial = models.Exponential(xc=180.0, yc=45.0, rs=0.004, Is=1.0)
        for key, value in kwargs.items():
            setattr(radial, key, value)
        vertical = models.disk.vertical.HyperSecantHeight(zs=1e-3, inc=0.5)
        return models.Disk(radial=radial, vertical=vertical)

    @pytest.mark.parametrize(
        "make",
        [
            lambda: models.Sersic(ns=numpyro.distributions.Normal(1.0, 0.5)),
            lambda: models.Sersic(e=numpyro.distributions.HalfNormal(0.3)),
        ],
    )
    def test_unbounded_priors_raise_cleanly(self, make):
        """Unbounded priors on bounded parameters raise ValueError."""
        with pytest.raises(ValueError, match="prior support includes"):
            make()

    def test_gnfw_unbounded_prior_warns(self):
        """Unbounded priors on gNFW slopes warn instead of crashing."""
        with pytest.warns(UserWarning, match="alpha prior support"):
            models.gNFW(alpha=numpyro.distributions.Normal(1.0, 0.2))

    def test_tied_setters(self):
        """Tied values are not range-checked."""
        a = self.sersic()
        b = self.sersic()
        b.ns = priors.boundto(a, "ns")
        b.e = priors.function(lambda x: x, (a, "e"))
        assert isinstance(b.ns, priors._BoundTo)
        assert isinstance(b.e, priors._Function)

    def test_disk_theta_periodic(self):
        """A Disk's radial.theta is periodic."""
        mod = models.Model(self.disk(theta=0.0))
        assert mod.periodic["comp_00_radial.theta"] == pytest.approx(np.pi)

    @pytest.mark.parametrize("when", ["before", "after"])
    def test_sub_profile_units(self, img, when):
        """Units set on a sub-profile apply, before or after wrapping."""
        if when == "before":
            radial = models.Exponential(xc=180.0, yc=45.0, rs=14.4, Is=1.0)
            radial.set_units(rs="arcsec")
            vertical = models.disk.vertical.HyperSecantHeight(zs=1e-3, inc=0.5)
            d = models.Disk(radial=radial, vertical=vertical)
        else:
            d = self.disk(rs=14.4)
            d.radial.set_units(rs="arcsec")
        mod = models.Model(d)
        assert mod.units["comp_00_radial.rs"] == "arcsec"
        ref = models.Model(self.disk()).getmap(img)
        np.testing.assert_allclose(mod.getmap(img), ref)
        np.testing.assert_allclose(d.getmap(img), self.disk().getmap(img))

    def test_wrapper_set_units_forwards(self):
        """set_units on a wrapper's dotted name reaches the sub-profile."""
        d = self.disk(rs=14.4)
        d.set_units(**{"radial.rs": "arcsec"})
        assert d.radial._input_units == {"rs": "arcsec"}

    def test_addcomponent_is_atomic(self):
        """A rejected component leaves the model unchanged."""
        mod = models.Model(self.sersic())
        before = (list(mod.params), list(mod.type), list(mod.positive))
        with pytest.raises(ValueError, match="set to None"):
            mod.addcomponent(models.Sersic(xc=0.0, yc=0.0))
        assert (mod.params, mod.type, mod.positive) == tuple(
            list(x) for x in before
        )
        assert mod.ncomp == 1

    def test_standalone_units(self, img):
        """Standalone getmap applies declared units."""
        s = self.sersic(re=14.4)
        s.set_units(re="arcsec")
        np.testing.assert_allclose(s.getmap(img), self.sersic().getmap(img))

    def test_function_in_declared_units(self, img):
        """Functions take and return values in declared units."""
        a = self.sersic(re=2.0)
        a.set_units(re="arcsec")
        b = self.sersic()
        b.set_units(re="arcsec")
        b.re = priors.function(lambda r: r + 12.4, (a, "re"))
        mod = models.Model(a)
        mod.addcomponent(b)
        ref = models.Model(self.sersic(re=2.0 / 3600))
        ref.addcomponent(self.sersic(re=14.4 / 3600))
        np.testing.assert_allclose(mod.getmap(img), ref.getmap(img))
        np.testing.assert_allclose(
            b.getmap(img), self.sersic(re=14.4 / 3600).getmap(img)
        )

    def test_boundto_across_units(self, img):
        """A binding copies the physical value between declared units."""
        a = self.sersic(re=0.24)
        a.set_units(re="arcmin")
        b = self.sersic()
        b.set_units(re="arcsec")
        b.re = priors.boundto(a, "re")
        mod = models.Model(a)
        mod.addcomponent(b)
        ref = models.Model(self.sersic())
        ref.addcomponent(self.sersic())
        np.testing.assert_allclose(mod.getmap(img), ref.getmap(img))
        np.testing.assert_allclose(b.getmap(img), self.sersic().getmap(img))

    def test_bridge_units(self, img):
        """Bridge internal ties stay correct with rs in declared units."""

        def bridge(rs):
            return models.SimpleBridge(
                xc=180.0, yc=45.0, rs=rs, Is=1.0, e=0.5, theta=0.0
            )

        b = bridge(18.0)
        b.set_units(rs="arcsec")
        ref = bridge(0.005).getmap(img)
        np.testing.assert_allclose(b.getmap(img), ref)
        np.testing.assert_allclose(models.Model(b).getmap(img), ref)


class TestRefactor:
    """Refactoring keeps the profile and the model unchanged."""

    @pytest.fixture
    def img(self, simple_hdu):
        """Create an Image without PSF."""
        return data.Image(simple_hdu, noise=noise.Normal(sigma=0.1))

    @staticmethod
    def polyexp(**kwargs):
        """Create a truncated PolyExpoRefact with fixed values."""
        prof = models.PolyExpoRefact(
            xc=180.0,
            yc=45.0,
            theta=0.5,
            e=0.4,
            cbox=0.2,
            Is=10.0,
            rs=0.004,
            rc=0.004,
            I1=1.0,
            I2=0.5,
            I3=0.0,
            I4=0.0,
            positive=False,
            name="disk",
        )
        prof.truncation = models.radial.truncation.HyperTangent()
        prof.rt, prof.wt = 0.008, 0.001
        for key, value in kwargs.items():
            setattr(prof, key, value)
        return prof

    def test_copy_keeps_everything(self, img):
        """The base refactor keeps every parameter, unit and self-tie."""
        b = models.Beta(
            xc=180.0, yc=45.0, rc=14.4, beta=0.7, theta=0.5, e=0.4, cbox=0.1
        )
        b.set_units(rc="arcsec")
        b.Ic = priors.function(lambda r: r / 14.4, (b, "rc"))
        models.Model(b)
        with pytest.warns(UserWarning, match="Nothing to refactor"):
            r = b.refactor()
        assert (r.theta, r.e, r.rc) == (0.5, 0.4, 14.4)
        assert r._input_units == {"rc": "arcsec"}
        assert r.id is None
        assert r.Ic.sources[0][0] is r
        r.rc = 7.2
        assert b.rc == 14.4
        np.testing.assert_allclose(b.refactor().getmap(img), b.getmap(img))

    def test_polyexp_fixed(self, img):
        """Fixed intensities give fixed coefficients and the same map."""
        prof = self.polyexp()
        new = prof.refactor()
        assert isinstance(new, models.PolyExponential)
        assert (new.c1, new.c2, new.c3, new.c4) == (0.1, 0.05, 0.0, 0.0)
        assert (new.cbox, new.positive, new.name) == (0.2, False, "disk")
        assert isinstance(
            new.truncation, models.radial.truncation.HyperTangent
        )
        np.testing.assert_allclose(new.getmap(img), prof.getmap(img))

    def test_polyexp_priors_and_ties(self, img):
        """Priors and ties on the intensities give the same model."""
        prof = self.polyexp(Is=priors.uniform(1.0, 20.0))
        prof.I1 = priors.uniform(0.0, 5.0)
        prof.I2 = priors.function(lambda i: 0.5 * i, (prof, "I1"))
        new = prof.refactor()
        old_model, new_model = models.Model(prof), models.Model(new)
        free = [new_model.params[i] for i in new_model.paridx]
        assert free == [old_model.params[i] for i in old_model.paridx]
        assert new.I2.sources[0][0] is new
        pp = np.array([12.0, 2.0])
        np.testing.assert_allclose(
            new_model.getmodel(img, pp)[1], old_model.getmodel(img, pp)[1]
        )

    def test_polyexp_extra_parameters(self):
        """Parameters added with addparameter are kept."""
        prof = self.polyexp()
        prof.addparameter("scale", 0.5, description="A scale")
        new = prof.refactor()
        assert new.scale == 0.5 and new.description["scale"] == "A scale"


class TestWrappedProfiles:
    """Tests for sub-profiles evaluated inside Disk, Ellipsoid and Bridge."""

    @pytest.fixture
    def img(self, simple_hdu):
        """Create an image centred on (180, 45)."""
        return data.Image(simple_hdu, noise=noise.Normal(sigma=0.1))

    @staticmethod
    def _trunc(rt):
        return models.radial.truncation.HyperTangent(rt=rt, wt=1e-4)

    def test_bridge_profiles_on_absolute_distances(self, img):
        """Bridge maps are finite and symmetric about the bridge axes."""
        for radial in [models.Beta(alpha=1.5), models.Exponential()]:
            bridge = models.SimpleBridge(
                radial=radial, xc=180.0, yc=45.0, rs=0.005, Is=1.0
            )
            mgrid = np.asarray(bridge.getmap(img))
            assert np.all(np.isfinite(mgrid))
            assert mgrid.max() <= 1.0 + 1e-12
            np.testing.assert_allclose(mgrid, mgrid[:, ::-1], rtol=1e-4)

    def test_mesabridge_getmap(self, img):
        """MesaBridge maps scale with Is."""
        kw = dict(xc=180.0, yc=45.0, rs=0.005)
        m1 = models.MesaBridge(Is=1.0, **kw).getmap(img)
        m2 = models.MesaBridge(Is=2.0, **kw).getmap(img)
        np.testing.assert_allclose(m2, 2.0 * m1)

    def test_disk_truncation(self, img):
        """A truncated Disk loses flux; a far truncation changes nothing."""

        def flux(rt):
            radial = models.Sersic(xc=180.0, yc=45.0, re=0.005, Ie=1.0)
            if rt is not None:
                radial.truncation = self._trunc(rt)
            vertical = models.disk.vertical.HyperSecantHeight(zs=5e-4)
            disk = models.Disk(radial=radial, vertical=vertical)
            return float(np.sum(disk.getmap(img)))

        assert flux(1e-3) < 0.5 * flux(None)
        np.testing.assert_allclose(flux(1.0), flux(None), rtol=1e-10)

    def test_bridge_truncation(self, img):
        """Truncating the parallel profile shortens the bridge."""

        def flux(rt):
            parallel = models.TopHat()
            if rt is not None:
                parallel.truncation = self._trunc(rt)
            bridge = models.SimpleBridge(
                parallel=parallel, xc=180.0, yc=45.0, rs=0.01, Is=1.0
            )
            return float(np.sum(bridge.getmap(img)))

        assert flux(2e-3) < 0.5 * flux(None)
        np.testing.assert_allclose(flux(1.0), flux(None), rtol=1e-10)

    def test_ellipsoid_truncation(self, img):
        """Ellipsoid truncation works on its own and inside a Model."""
        radial = models.Sersic(re=0.005, Ie=1.0, ns=1.0)
        radial.truncation = self._trunc(1e-3)
        ell = models.Ellipsoid(radial=radial, xc=180.0, yc=45.0, e=0.3)
        full = models.Ellipsoid(
            radial=models.Sersic(re=0.005, Ie=1.0, ns=1.0),
            xc=180.0,
            yc=45.0,
            e=0.3,
        )
        mgrid = ell.getmap(img)
        assert float(np.sum(mgrid)) < 0.5 * float(np.sum(full.getmap(img)))
        mraw = models.Model(ell).getmodel(img, [])[0]
        np.testing.assert_allclose(mraw, mgrid)

    @pytest.mark.parametrize(
        "radial",
        [
            models.Beta(rc=0.005, Ic=1.0),
            models.Exponential(rs=0.005, Is=1.0),
            models.gNFW(rc=0.005, Ic=1.0),
        ],
        ids=["Beta", "Exponential", "gNFW"],
    )
    def test_ellipsoid_any_radial(self, img, radial):
        """Ellipsoid accepts any radial profile, alone and in a Model."""
        ell = models.Ellipsoid(radial=radial, xc=180.0, yc=45.0, e=0.3)
        mgrid = ell.getmap(img)
        assert np.all(np.isfinite(mgrid)) and float(np.max(mgrid)) > 0
        mraw = models.Model(ell).getmodel(img, [])[0]
        np.testing.assert_allclose(mraw, mgrid)

    def test_ellipsoid_sersic_density(self):
        """The Sersic density matches the ellipsoidal-radius formula."""
        ell = models.Ellipsoid(
            radial=models.Sersic(re=0.005, Ie=2.0, ns=1.5), e=0.4, eratio=0.5
        )
        rng = np.random.default_rng(0)
        x, y, z = rng.uniform(-0.01, 0.01, (3, 50))
        rs1, rs2 = 0.005 * (1 - 0.4), 0.005 * (1 - 0.2)
        m = np.sqrt((x / 0.005) ** 2 + (y / rs1) ** 2 + (z / rs2) ** 2)
        expected = models.Sersic.profile(m, 2.0, 1.0, 1.5)
        got = ell.profile(x, y, z, e=0.4, eratio=0.5, Ie=2.0, re=0.005, ns=1.5)
        np.testing.assert_allclose(got, expected, rtol=1e-12)

    def test_truncation_swap_resets_units(self):
        """A new truncation's values are native, whatever the old units."""
        trunc = models.radial.truncation
        beta = models.Beta(
            xc=180.0,
            yc=45.0,
            rc=0.005,
            Ic=1.0,
            truncation=trunc.HyperTangent(rt=0.002, wt=0.0002),
        )
        beta.set_units(rt="arcsec")
        beta.rt = 7.2
        beta.truncation = trunc.Exponential(rt=0.002, wt=0.0002)
        mod = models.Model(beta)
        assert "comp_00_rt" not in mod.conversions
        assert beta.rt == 0.002

    @pytest.mark.parametrize("value", [np.inf, np.nan])
    @pytest.mark.parametrize("name", ["rt", "wt"])
    def test_truncation_parameters_finite(self, name, value):
        """An infinite or NaN truncation radius or width raises."""
        beta = models.Beta(rc=0.005, Ic=1.0, truncation=self._trunc(0.02))
        with pytest.raises(ValueError, match="positive and finite"):
            setattr(beta, name, value)

    def test_disk_losdepth_finite(self):
        """A Disk vertical profile refuses an infinite losdepth."""
        for value in (np.inf, np.nan, np.array(np.inf), jp.array(jp.inf)):
            with pytest.raises(ValueError, match="must be finite"):
                models.disk.vertical.HyperSecantHeight(zs=1e-3, losdepth=value)
        with pytest.raises(ValueError, match="must be a number"):
            models.disk.vertical.HyperSecantHeight(zs=1e-3, losdepth=True)
        models.Ellipsoid(losdepth=np.inf)  # the Ellipsoid default

    def test_truncation_object_finite(self):
        """A truncation object refuses infinite values, pointing to None."""
        with pytest.raises(ValueError, match="truncation to None"):
            models.radial.truncation.HyperTangent(rt=np.inf, wt=1e-3)

    @pytest.mark.parametrize(
        "make",
        [
            lambda **kw: models.disk.vertical.HyperSecantHeight(zs=1e-3, **kw),
            lambda **kw: models.Ellipsoid(**kw),
        ],
    )
    @pytest.mark.parametrize(
        "kw",
        [
            dict(losdepth=-0.01),
            dict(losdepth=0.0),
            dict(losdepth=np.nan),
            dict(losbins=1),
            dict(losbins=2.5),
            dict(losbins=True),
        ],
    )
    def test_line_of_sight_settings_checked(self, make, kw):
        """A non-positive losdepth, or losbins not an integer >= 2, raises."""
        with pytest.raises(ValueError, match="losdepth|losbins"):
            make(**kw)

    def test_truncation_parameters_positive(self):
        """Non-positive truncation radii and widths raise, priors included."""
        beta = models.Beta(rc=0.005, Ic=1.0, truncation=self._trunc(0.02))
        for name, value in [
            ("wt", -1e-3),
            ("wt", 0.0),
            ("wt", priors.uniform(-1e-3, 1e-3)),
            ("rt", priors.normal(0.02, 0.01)),
        ]:
            with pytest.raises(ValueError, match="strictly positive"):
                setattr(beta, name, value)
        beta.wt = priors.uniform(0.0, 1e-3)
        beta.rt = priors.loguniform(1e-3, 1e-1)
        with pytest.raises(ValueError, match="strictly positive"):
            models.radial.truncation.HyperTangent(wt=-1e-3)

    def test_truncation_set_after_wrapping(self, img):
        """A truncation added to a wrapped sub-profile is applied."""
        radial = models.Sersic(xc=180.0, yc=45.0, re=0.005, Ie=1.0)
        vertical = models.disk.vertical.HyperSecantHeight(zs=5e-4)
        disk = models.Disk(radial=radial, vertical=vertical)
        full = float(np.sum(disk.getmap(img)))
        disk.radial.truncation = self._trunc(1e-3)
        assert float(np.sum(disk.getmap(img))) < 0.5 * full
        mraw = models.Model(disk).getmodel(img, [])[0]
        np.testing.assert_allclose(mraw, disk.getmap(img))

        bridge = models.SimpleBridge(xc=180.0, yc=45.0, rs=0.01, Is=1.0)
        full = float(np.sum(bridge.getmap(img)))
        bridge.parallel.truncation = self._trunc(2e-3)
        assert float(np.sum(bridge.getmap(img))) < 0.5 * full
        mraw = models.Model(bridge).getmodel(img, [])[0]
        np.testing.assert_allclose(mraw, bridge.getmap(img))

    @pytest.mark.parametrize(
        "make",
        [
            lambda: models.Disk(
                radial=models.Sersic(xc=180.0, yc=45.0, re=0.005, Ie=1.0),
                vertical=models.disk.vertical.HyperSecantHeight(zs=5e-4),
            ),
            lambda: models.Ellipsoid(
                radial=models.Beta(rc=0.005, Ic=1.0), xc=180.0, yc=45.0
            ),
            lambda: models.SimpleBridge(xc=180.0, yc=45.0, rs=0.01, Is=1.0),
            lambda: models.MesaBridge(xc=180.0, yc=45.0, rs=0.01, Is=1.0),
            lambda: models.gNFW(xc=180.0, yc=45.0, rc=0.005, Ic=1.0),
            lambda: _custom(),
        ],
        ids=[
            "Disk",
            "Ellipsoid",
            "SimpleBridge",
            "MesaBridge",
            "gNFW",
            "Custom",
        ],
    )
    def test_pickle_roundtrip(self, img, make):
        """Components and models pickle and give the same maps."""
        comp = make()
        mgrid = comp.getmap(img)
        np.testing.assert_allclose(
            pickle.loads(pickle.dumps(comp)).getmap(img), mgrid
        )
        np.testing.assert_allclose(copy.deepcopy(comp).getmap(img), mgrid)
        mod = pickle.loads(pickle.dumps(models.Model(make())))
        np.testing.assert_allclose(mod.getmodel(img, [])[0], mgrid)


def _custom_profile(r, Ic, rc):
    return Ic * jp.exp(-r / rc)


def _custom():
    prof = models.CustomProfile(
        [dict(name="Ic"), dict(name="rc")], _custom_profile, xc=180.0, yc=45.0
    )
    prof.Ic, prof.rc = 1.0, 0.005
    return prof


class TestModelSync:
    """The model follows changes made to its components after adding them."""

    @pytest.fixture
    def img(self, simple_hdu):
        """Create an image centred on (180, 45)."""
        return data.Image(simple_hdu, noise=noise.Normal(sigma=0.1))

    @staticmethod
    def beta(**kw):
        """Return a fixed Beta at the image centre."""
        return models.Beta(xc=180.0, yc=45.0, rc=0.005, Ic=1.0, **kw)

    def test_values_priors_and_units(self, img):
        """Values, priors and declared units set after adding are used."""
        b = self.beta()
        mod = models.Model(b)
        b.Ic = 2.0
        np.testing.assert_allclose(mod.getmap(img), b.getmap(img))
        b.set_units(rc="arcsec")
        b.rc = 18.0
        np.testing.assert_allclose(mod.getmap(img), b.getmap(img))
        assert mod.units["comp_00_rc"] == "arcsec"
        b.Ic = priors.uniform(0.0, 3.0)
        mod.parameters()
        assert [mod.params[i] for i in mod.paridx] == ["comp_00_Ic"]

    def test_getmodel_follows_changes(self, img):
        """Model.getmodel uses values set after adding."""
        b = self.beta()
        mod = models.Model(b)
        b.Ic = 2.0
        mraw = mod.getmodel(img, [])[0]
        np.testing.assert_allclose(mraw, b.getmap(img))

    def test_ties_after_adding(self, img):
        """A tie set after adding is resolved."""
        a, b = self.beta(), self.beta()
        mod = models.Model(a)
        mod.addcomponent(b)
        b.xc = priors.function(lambda x: x + 0.002, (a, "xc"))
        ref = a.getmap(img) + models.Beta(
            xc=180.002, yc=45.0, rc=0.005, Ic=1.0
        ).getmap(img)
        np.testing.assert_allclose(mod.getmap(img), ref)

    @pytest.mark.parametrize("kind", ["Disk", "Ellipsoid", "SimpleBridge"])
    def test_late_truncation(self, img, kind):
        """A truncation set after adding is used by every wrapper."""
        if kind == "Disk":
            comp = models.Disk(
                radial=models.Sersic(xc=180.0, yc=45.0, re=0.005, Ie=1.0),
                vertical=models.disk.vertical.HyperSecantHeight(zs=5e-4),
            )
            sub = comp.radial
        elif kind == "Ellipsoid":
            comp = models.Ellipsoid(
                radial=models.Sersic(re=0.005, Ie=1.0), xc=180.0, yc=45.0
            )
            sub = comp.radial
        else:
            comp = models.SimpleBridge(xc=180.0, yc=45.0, rs=0.01, Is=1.0)
            sub = comp.parallel
        mod = models.Model(comp)
        full = float(np.sum(mod.getmap(img)))
        sub.truncation = models.radial.truncation.HyperTangent(
            rt=1e-3, wt=1e-4
        )
        cut = mod.getmap(img)
        assert float(np.sum(cut)) < 0.5 * full
        np.testing.assert_allclose(cut, comp.getmap(img))
        sub.truncation = None
        np.testing.assert_allclose(float(np.sum(mod.getmap(img))), full)
        assert not any(k.endswith((".rt", ".wt")) for k in mod.params)

    def test_fitter_follows_changes_before_creation(self, img):
        """A fitter sees the components as they are when it is created."""
        from socca import fitting

        b = self.beta()
        mod = models.Model(b)
        b.Ic = priors.uniform(0.0, 3.0)
        fit = fitting.fitter(img, mod)
        assert fit.labels == ["comp_00_Ic"]

    def test_fitter_rejects_later_changes(self, img):
        """Changing the model after creating a fitter raises at run()."""
        from socca import fitting

        b = self.beta()
        b.Ic = priors.uniform(0.0, 3.0)
        mod = models.Model(b)
        fit = fitting.fitter(img, mod)
        b.rc = 0.01
        with pytest.raises(ValueError, match="changed after this fitter"):
            fit.run(method="optimizer")

    def test_tie_outside_model_raises_at_fitter(self, img):
        """A tie to a component outside the model is caught early."""
        from socca import fitting

        a, b = self.beta(), self.beta()
        b.xc = priors.boundto(a, "xc")
        b.Ic = priors.uniform(0.0, 3.0)
        with pytest.raises(ValueError, match="not in this model"):
            fitting.fitter(img, models.Model(b))

    def test_span_warning_once(self):
        """Repeated syncs do not repeat warnings."""
        import warnings

        b = self.beta(theta=priors.uniform(-np.pi, np.pi))
        with pytest.warns(UserWarning, match="more than its period"):
            mod = models.Model(b)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            mod.parameters()
            repr(mod)

    @pytest.mark.parametrize("name", ["losdepth", "losbins"])
    def test_hyperparameters_are_not_fitted(self, name):
        """A prior on a hyperparameter raises."""
        disk = models.Disk(
            radial=self.beta(),
            vertical=models.disk.vertical.HyperSecantHeight(zs=5e-4),
        )
        setattr(disk.vertical, name, priors.uniform(1.0, 2.0))
        with pytest.raises(ValueError, match="hyperparameter"):
            models.Model(disk)


class TestOwnership:
    """A profile belongs to one component, and a model holds it once."""

    def test_wrap_twice(self):
        """Wrapping an already wrapped profile raises, before any change."""
        s = models.Sersic()
        models.Disk(radial=s)
        units = dict(s.units)
        for make in [
            lambda: models.Disk(radial=s),
            lambda: models.Ellipsoid(radial=s),
            lambda: models.SimpleBridge(radial=s),
        ]:
            with pytest.raises(ValueError, match="already part of a Disk"):
                make()
        assert s.units == units
        with pytest.raises(ValueError, match="used twice"):
            models.SimpleBridge(radial=(t := models.TopHat()), parallel=t)

    def test_copy_of_a_sub_profile(self):
        """A deep copy of a wrapped profile is free; a copied wrapper keeps its own."""
        disk = models.Disk()
        free = copy.deepcopy(disk.radial)
        assert free._wrapper is None
        models.Disk(radial=free)
        assert disk.radial._wrapper is disk
        other = copy.deepcopy(disk)
        assert other.radial._wrapper is other

    def test_model_rejects_sub_profiles(self):
        """A sub-profile cannot also be a model entry."""
        disk = models.Disk(
            radial=models.Sersic(xc=180.0, yc=45.0, re=0.005, Ie=1.0),
            vertical=models.disk.vertical.HyperSecantHeight(zs=5e-4),
        )
        mod = models.Model(disk)
        with pytest.raises(ValueError, match="part of a Disk"):
            mod.addcomponent(disk.radial)
        s = models.Sersic(xc=180.0, yc=45.0, re=0.005, Ie=1.0)
        mod = models.Model(s)
        models.Disk(
            radial=s, vertical=models.disk.vertical.HyperSecantHeight(zs=5e-4)
        )
        mod.parameters()
        with pytest.raises(ValueError, match="now part of a Disk"):
            mod.getmap(None)

    def test_reserved_truncation_names(self):
        """Profiles cannot define their own rt or wt."""
        with pytest.raises(ValueError, match="reserved"):
            models.CustomProfile([dict(name="rt")], lambda r, rt: r)
        with pytest.raises(ValueError, match="reserved"):
            models.Beta().addparameter("wt", 1.0)

    def test_truncation_parameters_are_not_removable(self):
        """The rt and wt parameters go with the truncation."""
        prof = models.Sersic(truncation=True)
        with pytest.raises(ValueError, match="truncation = None"):
            prof.removeparameter("rt")
        prof.truncation = None
        assert not hasattr(prof, "rt")

    @staticmethod
    def _disk():
        radial = models.Sersic(xc=180.0, yc=45.0, re=0.005, Ie=1.0)
        vertical = models.disk.vertical.HyperSecantHeight(zs=5e-4, inc=0.5)
        return models.Disk(radial=radial, vertical=vertical)

    def test_copies_of_sub_profiles_are_free(self):
        """Refactors, copies and pickles of a sub-profile leave the wrapper."""
        disk = self._disk()
        vertical = models.disk.vertical.HyperSecantHeight(zs=5e-4)
        with pytest.warns(UserWarning, match="Nothing to refactor"):
            radial = disk.radial.refactor()
        models.Disk(radial=radial, vertical=vertical)
        assert copy.copy(disk.radial)._wrapper is None
        assert pickle.loads(pickle.dumps(disk.radial))._wrapper is None
        assert copy.deepcopy(disk.radial)._wrapper is None

    def test_copies_of_wrappers_keep_their_subs(self):
        """A copied wrapper owns its copied subs; a shallow copy steals none."""
        disk = self._disk()
        deep = copy.deepcopy(disk)
        assert deep.radial._wrapper is deep
        assert disk.radial._wrapper is disk
        copy.copy(disk)
        assert disk.radial._wrapper is disk


class TestReviewChecks:
    """Listing, tie and selection checks."""

    def test_bridge_internal_radius_tie(self):
        """Tying to a Bridge's internal scale radius raises."""
        bridge = models.SimpleBridge()
        for sub in [bridge.radial, bridge.parallel]:
            with pytest.raises(
                ValueError, match="computed by its SimpleBridge"
            ):
                priors.boundto(sub, sub._scale_radius)
            with pytest.raises(
                ValueError, match="computed by its SimpleBridge"
            ):
                priors.function(lambda r: r, (sub, sub._scale_radius))
        priors.boundto(bridge, "rs")

    def test_prior_ranges_distinguishable(self, capsys):
        """Narrow priors print with enough digits to tell the bounds apart."""
        b = models.Beta(
            xc=priors.uniform(179.995, 180.005), rc=priors.uniform(0, 20)
        )
        b.parameters()
        out = capsys.readouterr().out
        assert "Uniform [179.995, 180.005]" in out
        assert "Uniform [0, 20]" in out

    def test_standalone_listing_declared_units(self, capsys):
        """Component listings show declared units."""
        b = models.Beta(rc=36.0)
        b.set_units(rc="arcsec")
        b.parameters()
        assert "rc [arcsec]" in capsys.readouterr().out

    def test_hyperparameters_listed_apart(self, capsys):
        """Model listings put losdepth and losbins under hyperparameters."""
        disk = models.Disk(
            radial=models.Sersic(xc=180.0, yc=45.0, re=0.005, Ie=1.0),
            vertical=models.disk.vertical.HyperSecantHeight(zs=5e-4),
        )
        models.Model(disk).parameters()
        out = capsys.readouterr().out
        head, _, tail = out.partition("Hyperparameters")
        assert "vertical.losbins" in tail and "vertical.losbins" not in head

    def test_ambiguous_lambda_argument(self):
        """A lambda argument matching two parameters raises."""
        a = models.Beta(name="x_radial", xc=180.0, yc=45.0, rc=0.01, Ic=1.0)
        disk = models.Disk(
            name="x",
            radial=models.Sersic(xc=180.0, yc=45.0, re=0.005, Ie=1.0),
            vertical=models.disk.vertical.HyperSecantHeight(zs=5e-4),
        )
        b = models.Beta(yc=45.0, rc=0.01, Ic=1.0)
        b.xc = lambda x_radial_xc: x_radial_xc
        mod = models.Model(a)
        mod.addcomponent(disk)
        mod.addcomponent(b)
        with pytest.raises(KeyError, match="several"):
            mod._tieorder()

    def test_named_lambda_standalone(self, simple_hdu):
        """A named component resolves its own lambda ties before adding."""
        img = data.Image(simple_hdu, noise=noise.Normal(sigma=0.1))
        b = models.Beta(name="core", xc=180.0, yc=45.0, rc=0.01)
        b.Ic = lambda core_rc: core_rc * 100.0
        ref = models.Beta(xc=180.0, yc=45.0, rc=0.01, Ic=1.0).getmap(img)
        np.testing.assert_allclose(b.getmap(img), ref)

    def test_prior_source_standalone(self, simple_hdu):
        """A function tie on a prior gives a clear standalone error."""
        img = data.Image(simple_hdu, noise=noise.Normal(sigma=0.1))
        a = models.Beta(xc=priors.uniform(179.9, 180.1))
        b = models.Beta(yc=45.0, rc=0.01, Ic=1.0)
        b.xc = priors.function(lambda x: x, (a, "xc"))
        with pytest.raises(ValueError, match="has a prior"):
            b.getmap(img)

    @pytest.mark.skipif(
        importlib.util.find_spec("flax") is None,
        reason="the gNFW emulator needs flax",
    )
    def test_gnfw_keeps_its_emulator(self, monkeypatch):
        """Unpickling a gNFW does not reload the emulator from disk."""
        g = models.gNFW(emulator=True)
        blob = pickle.dumps(g)
        monkeypatch.setattr(
            models.gNFW,
            "_load_emulator",
            staticmethod(lambda path: (_ for _ in ()).throw(OSError(path))),
        )
        new = pickle.loads(blob)
        assert new._emulator_model is not None


def test_compiled_profile_follows_replacements(simple_hdu):
    """Replacing a sub-profile's truncation recompiles the wrapper profile."""
    trunc = models.radial.truncation
    img = data.Image(simple_hdu, noise=noise.Normal(sigma=0.1))

    def disk(tr):
        radial = models.Sersic(xc=180.0, yc=45.0, re=0.005, Ie=1.0)
        radial.truncation = tr
        return models.Disk(
            radial=radial,
            vertical=models.disk.vertical.HyperSecantHeight(zs=5e-4),
        )

    comp = disk(trunc.HyperTangent(rt=1e-3, wt=1e-4))
    comp.getmap(img)
    comp.radial.truncation = trunc.Exponential(rt=1e-3, wt=1e-4)
    ref = disk(trunc.Exponential(rt=1e-3, wt=1e-4)).getmap(img)
    np.testing.assert_allclose(comp.getmap(img), ref)
    assert "_jit" not in pickle.loads(pickle.dumps(comp)).__dict__


def test_compiled_profile_keeps_its_truncation(simple_hdu):
    """The cache keeps its truncation alive, so no new one takes its address."""
    trunc = models.radial.truncation
    img = data.Image(simple_hdu, noise=noise.Normal(sigma=0.1))
    radial = models.Sersic(xc=180.0, yc=45.0, re=0.005, Ie=1.0)
    radial.truncation = trunc.HyperTangent(rt=1e-3, wt=1e-4)
    comp = models.Disk(
        radial=radial, vertical=models.disk.vertical.HyperSecantHeight(zs=5e-4)
    )
    comp.getmap(img)
    old = weakref.ref(radial.truncation)
    radial.truncation = trunc.Exponential(rt=1e-3, wt=1e-4)
    gc.collect()
    assert old() is not None
