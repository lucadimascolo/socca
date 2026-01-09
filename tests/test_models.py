"""Tests for socca.models module."""

import jax.numpy as jp
import numpy as np
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

    def test_unique_id(self):
        """Test that components get unique IDs."""
        comp1 = models.Beta(xc=180.0, yc=45.0, rc=0.01, Ic=1.0, beta=0.5)
        comp2 = models.Beta(xc=180.0, yc=45.0, rc=0.01, Ic=1.0, beta=0.5)
        assert comp1.id != comp2.id

    def test_parlist(self):
        """Test parlist returns parameter names."""
        beta = models.Beta(xc=180.0, yc=45.0, rc=0.01, Ic=1.0, beta=0.5)
        params = beta.parlist()
        assert "xc" in params
        assert "yc" in params
        assert "rc" in params
        assert "Ic" in params
        assert "beta" in params

    def test_addpar(self):
        """Test adding custom parameter."""
        beta = models.Beta(xc=180.0, yc=45.0, rc=0.01, Ic=1.0, beta=0.5)
        beta.addpar(
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
        result = models.Beta.profile(r, Ic=1.0, rc=0.01, beta=0.5)
        assert result.shape == r.shape
        assert float(result[0]) == 1.0
        assert float(result[1]) < 1.0

    def test_profile_central_value(self):
        """Test that profile is maximized at center."""
        r = jp.array([0.0])
        Ic = 10.0
        result = models.Beta.profile(r, Ic=Ic, rc=0.01, beta=0.5)
        assert float(result[0]) == pytest.approx(Ic)

    def test_profile_decreases_with_radius(self):
        """Test that profile decreases with radius."""
        r = jp.array([0.0, 0.01, 0.02, 0.05, 0.1])
        result = models.Beta.profile(r, Ic=1.0, rc=0.01, beta=0.5)
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
        assert "Sersic" in captured.out
        assert "Point" in captured.out
        assert "Background" in captured.out


class TestgNFW:
    """Tests for gNFW profile."""

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


class TestModelComposition:
    """Tests for model composition operators."""

    def test_add_profiles(self):
        """Test adding profiles to create composite model."""
        beta = models.Beta(xc=180.0, yc=45.0, rc=0.01, Ic=1.0, beta=0.5)
        mod = models.Model(beta)
        point = models.Point(xc=180.1, yc=45.1, Ic=100.0)
        mod.addcomponent(point)
        assert mod.ncomp == 2
