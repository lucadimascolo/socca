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

    @pytest.fixture(autouse=True)
    def reset_idcls(self):
        """Reset Component.idcls before each test."""
        models.Component.idcls = 0

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

    @pytest.fixture(autouse=True)
    def reset_idcls(self):
        """Reset Component.idcls before each test."""
        models.Component.idcls = 0

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

    @pytest.fixture(autouse=True)
    def reset_idcls(self):
        """Reset Component.idcls before each test."""
        models.Component.idcls = 0

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


class TestEllipsoid:
    """Tests for Ellipsoid model."""

    @pytest.fixture(autouse=True)
    def reset_idcls(self):
        """Reset Component.idcls before each test."""
        models.Component.idcls = 0

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

    def test_theta_wide_prior_warns(self):
        """Test that a theta prior spanning more than 180deg warns."""
        ell = models.Ellipsoid()
        with pytest.warns(UserWarning, match="180"):
            ell.theta = priors.uniform(-np.pi, np.pi)

    def test_theta_narrow_prior_no_warning(self, recwarn):
        """Test that a theta prior spanning less than 180deg does not warn."""
        ell = models.Ellipsoid()
        ell.theta = priors.uniform(-np.pi / 4, np.pi / 4)
        assert len(recwarn) == 0

    def test_rot_wide_prior_warns(self):
        """Test that a rot prior spanning more than 180deg warns."""
        ell = models.Ellipsoid()
        with pytest.warns(UserWarning, match="180"):
            ell.rot = priors.uniform(-np.pi, np.pi)

    def test_tied_e_does_not_warn_or_crash(self):
        """Test that tying e via boundto() doesn't trigger range validation."""
        disk = models.Disk()
        ell = models.Ellipsoid()
        ell.e = priors.boundto(disk, "inc")

    def test_profile_major_axis_surface(self):
        """Test that the profile equals Ie on the major (x) axis surface."""
        re, e, eratio, Ie, ns = 0.001, 0.5, 1.0, 3.0, 1.0
        val = models.Ellipsoid._ellipsoid_profile(
            re, 0.0, 0.0, Ie, re, e, eratio, ns
        )
        assert float(val) == pytest.approx(Ie, rel=1e-3)

    def test_profile_first_minor_axis_surface(self):
        """Test that the profile equals Ie on the first minor (y) axis surface."""
        re, e, eratio, Ie, ns = 0.001, 0.5, 1.0, 3.0, 1.0
        rs1 = re * (1 - e)
        val = models.Ellipsoid._ellipsoid_profile(
            0.0, rs1, 0.0, Ie, re, e, eratio, ns
        )
        assert float(val) == pytest.approx(Ie, rel=1e-3)

    def test_profile_second_minor_axis_uses_eratio(self):
        """Test that the profile equals Ie on the eratio-scaled z-axis surface."""
        re, e, eratio, Ie, ns = 0.001, 0.6, 0.5, 3.0, 1.0
        rs2 = re * (1 - eratio * e)
        val = models.Ellipsoid._ellipsoid_profile(
            0.0, 0.0, rs2, Ie, re, e, eratio, ns
        )
        assert float(val) == pytest.approx(Ie, rel=1e-3)

    def test_eratio_default_makes_minor_axes_equal(self):
        """Test that eratio=1 (default) gives equal minor-axis scales."""
        re, e = 0.001, 0.6
        rs1 = re * (1 - e)
        rs2 = re * (1 - 1.0 * e)
        assert rs1 == pytest.approx(rs2)


class TestBuildKwargsEvaluate:
    """Tests for _build_kwargs and _evaluate methods."""

    @pytest.fixture(autouse=True)
    def reset_idcls(self):
        """Reset Component.idcls before each test."""
        models.Component.idcls = 0

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

    @pytest.fixture(autouse=True)
    def reset_idcls(self):
        """Reset Component.idcls before each test."""
        models.Component.idcls = 0

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

    @pytest.fixture(autouse=True)
    def reset_idcls(self):
        """Reset Component.idcls before each test."""
        models.Component.idcls = 0

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
        Model) previously failed to resolve boundto()-tied parameters --
        see socca/priors.py's _BoundTo.resolve().
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
        assert ell.xc.resolve() == disk.radial.xc
        assert ell.theta.resolve() == disk.radial.theta
        assert ell.inc.resolve() == disk.vertical.inc


class TestModelComposition:
    """Tests for model composition operators."""

    def test_add_profiles(self):
        """Test adding profiles to create composite model."""
        beta = models.Beta(xc=180.0, yc=45.0, rc=0.01, Ic=1.0, beta=0.5)
        mod = models.Model(beta)
        point = models.Point(xc=180.1, yc=45.1, Ic=100.0)
        mod.addcomponent(point)
        assert mod.ncomp == 2
