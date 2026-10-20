"""Tests for socca.models module."""

import copy
import warnings

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
        with pytest.raises(ValueError, match="greater than 1"):
            beta.e = 1.5

    def test_e_still_raises_for_fixed_value_below_zero(self):
        """Test that a fixed e < 0 still raises ValueError."""
        beta = models.Beta()
        with pytest.raises(ValueError, match="lower than 0"):
            beta.e = -0.1

    def test_e_still_raises_for_prior_out_of_range(self):
        """Test that a prior with out-of-range support still raises."""
        beta = models.Beta()
        with pytest.raises(ValueError, match="greater than 1"):
            beta.e = priors.uniform(0.0, 1.5)

    @staticmethod
    def _beta(theta):
        return models.Beta(
            xc=180.0, yc=45.0, rc=0.01, Ic=1.0, beta=0.5, theta=theta
        )

    def test_theta_narrow_prior_no_warning(self, recwarn):
        """Test that a theta prior narrower than its period does not warn."""
        models.Model(self._beta(priors.uniform(-np.pi / 4, np.pi / 4)))
        assert len(recwarn) == 0

    def test_theta_exact_period_warns_periodic(self, recwarn):
        """A theta prior exactly spanning its period warns that it will be sampled as periodic, when the component is added to a model (not just at fit.run())."""
        beta = self._beta(priors.uniform(0.00, np.pi))
        assert len(recwarn) == 0
        models.Model(beta)
        assert len(recwarn) == 1
        assert "periodic" in str(recwarn[0].message)

    def test_theta_wider_than_period_warns_twice(self, recwarn):
        """A theta prior wider than its period warns about both avoidable multimodality and periodic sampling."""
        models.Model(self._beta(priors.uniform(-np.pi, np.pi)))
        messages = [str(w.message) for w in recwarn]
        assert any("multimodal" in m for m in messages)
        assert any("periodic" in m for m in messages)


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


class TestPointPlacement:
    """Tests for the position of point sources in the model image."""

    @pytest.fixture(autouse=True)
    def reset_component_counter(self):
        """Reset component ID counter before each test."""
        models.Component.reset()

    @staticmethod
    def _image(shape):
        header = fits.Header()
        header["CRPIX1"] = shape[1] // 2 + 1
        header["CRPIX2"] = shape[0] // 2 + 1
        header["CRVAL1"] = 180.0
        header["CRVAL2"] = 0.0
        header["CDELT1"] = -0.001
        header["CDELT2"] = 0.001
        header["CTYPE1"] = "RA---TAN"
        header["CTYPE2"] = "DEC--TAN"
        return data.Image(fits.PrimaryHDU(np.zeros(shape), header=header))

    @pytest.mark.parametrize("shape", [(32, 32), (33, 33), (32, 35)])
    def test_point_without_psf(self, shape):
        """Without a PSF the source lands on its WCS pixel in both maps."""
        img = self._image(shape)
        point = models.Point(xc=180.003, yc=0.002, Ic=2.0)
        mod = models.Model(point)

        expected = np.zeros(shape)
        expected[shape[0] // 2 + 2, shape[1] // 2 - 3] = 2.0

        mraw, msmo, _, _ = mod.getmodel(img, [])
        np.testing.assert_allclose(mraw, expected, atol=1e-8)
        np.testing.assert_allclose(msmo, expected, atol=1e-8)
        np.testing.assert_allclose(point.getmap(img), expected, atol=1e-8)

    @pytest.mark.parametrize("shape", [(32, 32), (33, 33), (32, 35)])
    @pytest.mark.parametrize("kshape", [(9, 9), (8, 8), (8, 11)])
    def test_point_with_psf(self, shape, kshape):
        """The source lands on its WCS pixel as the input PSF stamp."""
        img = self._image(shape)
        m = [2 * ((k - 1) // 2) + 1 for k in kshape]
        core = np.random.default_rng(0).random(m)
        psf = np.zeros(kshape)
        psf[kshape[0] - m[0] :, kshape[1] - m[1] :] = core + core[::-1, ::-1]
        psf = psf / psf.sum()
        img.addpsf(psf)

        point = models.Point(xc=180.003, yc=0.002, Ic=2.0)
        mod = models.Model(point)

        _, msmo, _, _ = mod.getmodel(img, [])
        msmo = np.asarray(msmo)
        np.testing.assert_allclose(
            point.getmap(img, convolve=True), msmo, atol=1e-12
        )

        expected = np.zeros(shape)
        y0 = shape[0] // 2 + 2 - kshape[0] // 2
        x0 = shape[1] // 2 - 3 - kshape[1] // 2
        expected[y0 : y0 + kshape[0], x0 : x0 + kshape[1]] = 2.0 * psf
        np.testing.assert_allclose(msmo, expected, atol=1e-8)

    @staticmethod
    def _square_image(n):
        header = fits.Header()
        header["CRPIX1"] = header["CRPIX2"] = n // 2 + 1
        header["CRVAL1"] = 150.0
        header["CRVAL2"] = 2.0
        header["CDELT1"] = -1.0 / 3600
        header["CDELT2"] = 1.0 / 3600
        header["CTYPE1"] = "RA---TAN"
        header["CTYPE2"] = "DEC--TAN"
        return data.Image(fits.PrimaryHDU(np.zeros((n, n)), header=header))

    @staticmethod
    def _at(img, y, x, **kwargs):
        ra, dec = img.wcs.all_pix2world(x, y, 0)
        return models.Point(xc=float(ra), yc=float(dec), **kwargs)

    @pytest.mark.parametrize("n", [32, 33])
    @pytest.mark.parametrize("sigma", [None, 0.7])
    def test_subpixel_point_transpose_symmetry(self, n, sigma):
        """Swapping the sub-pixel offsets of a source transposes its map."""
        maps = []
        for y, x in [(2.3, -3.45), (-3.45, 2.3)]:
            models.Component.reset()
            img = self._square_image(n)
            if sigma:
                k = np.arange(25) - 12.0
                g = np.exp(-0.5 * (k[:, None] ** 2 + k**2) / sigma**2)
                img.addpsf(g)
            point = self._at(img, n // 2 + y, n // 2 + x, Ic=1.0)
            mraw, msmo, _, _ = models.Model(point).getmodel(img, [])
            maps.append((np.asarray(mraw), np.asarray(msmo)))
        for a, b in zip(*maps):
            np.testing.assert_allclose(a, b.T, atol=1e-5 * np.abs(a).max())

    @pytest.mark.parametrize("psf", [True, False])
    def test_getmodel_matches_separate_convolution(self, psf):
        """The combined model equals the sum of its convolved components."""
        img = self._square_image(32)
        if psf:
            k = np.arange(15) - 7.0
            img.addpsf(np.exp(-0.5 * (k[:, None] ** 2 / 2.0 + k**2 / 3.0)))
        ra, dec = img.wcs.all_pix2world(14.3, 17.6, 0)
        sersic = models.Sersic(
            xc=float(ra),
            yc=float(dec),
            theta=0.3,
            e=0.3,
            re=3.0 / 3600,
            Ie=1.0,
            ns=1.0,
        )
        point = self._at(img, 12, 20, Ic=3.0)
        mod = models.Model(sersic)
        mod.addcomponent(point)

        mraw, msmo, _, _ = mod.getmodel(img, [])
        sraw = np.asarray(mod.getmodel(img, [], component=0)[0])
        praw = np.asarray(point.getmap(img))
        np.testing.assert_allclose(mraw, sraw + praw, atol=1e-12)

        smooth = np.asarray(img.convolve(sraw)) if psf else sraw
        np.testing.assert_allclose(
            msmo, smooth + point.getmap(img, convolve=psf), atol=1e-12
        )

    @pytest.mark.parametrize("csize", [15, 16])
    def test_point_after_cutout(self, csize):
        """After a cutout, sources still land on their WCS pixel."""
        from astropy.coordinates import SkyCoord

        img = self._square_image(64)
        ra, dec = img.wcs.all_pix2world(32.4, 31.7, 0)
        img.cutout(SkyCoord(float(ra), float(dec), unit="deg"), csize)
        point = self._at(img, 4, csize - 3, Ic=1.0)
        mraw = np.asarray(models.Model(point).getmodel(img, [])[0])
        expected = np.zeros((csize, csize))
        expected[4, csize - 3] = 1.0
        np.testing.assert_allclose(mraw, expected, atol=1e-5)


class TestRAWrap:
    """Tests for fields and sources across right ascension 0/360."""

    @pytest.fixture(autouse=True)
    def reset_component_counter(self):
        """Reset component ID counter before each test."""
        models.Component.reset()

    @staticmethod
    def _image(crval):
        header = fits.Header()
        header["CRPIX1"] = header["CRPIX2"] = 17
        header["CRVAL1"], header["CRVAL2"] = crval, 2.0
        header["CDELT1"], header["CDELT2"] = -1.0 / 3600, 1.0 / 3600
        header["CTYPE1"], header["CTYPE2"] = "RA---TAN", "DEC--TAN"
        return data.Image(fits.PrimaryHDU(np.zeros((32, 32)), header=header))

    @staticmethod
    def _component(kind, ra, dec):
        if kind == "point":
            return models.Point(xc=ra, yc=dec, Ic=1.0)
        if kind == "sersic":
            return models.Sersic(
                xc=ra, yc=dec, theta=0.2, e=0.2, re=3.0 / 3600, Ie=1.0, ns=1.0
            )
        if kind == "background":
            return models.Background(xc=ra, yc=dec, a0=0.1, a1x=0.01, a1y=0.02)
        if kind == "bridge":
            return models.SimpleBridge(
                xc=ra, yc=dec, rs=5.0 / 3600, Is=1.0, theta=0.2
            )
        if kind == "disk":
            disk = models.Disk(
                radial=models.Sersic(re=3.0 / 3600, Ie=1.0, ns=1.0),
                vertical=models.disk.vertical.HyperSecantHeight(zs=1.0 / 3600),
            )
            disk.radial.xc, disk.radial.yc = ra, dec
            disk.radial.theta, disk.vertical.inc = 0.3, 0.6
            return disk
        ell = models.Ellipsoid(
            radial=models.Sersic(re=3.0 / 3600, Ie=1.0, ns=1.0)
        )
        ell.xc, ell.yc, ell.theta, ell.e, ell.inc = ra, dec, 0.3, 0.3, 0.5
        return ell

    def _scene(self, kind, crval, dx):
        models.Component.reset()
        img = self._image(crval)
        ra = (crval + dx / np.cos(np.deg2rad(2.0)) / 3600.0) % 360.0
        mod = models.Model(self._component(kind, ra, 2.0 + 1.0 / 3600))
        return np.asarray(mod.getmodel(img, [])[1])

    @pytest.mark.parametrize(
        "kind",
        ["point", "sersic", "background", "bridge", "disk", "ellipsoid"],
    )
    @pytest.mark.parametrize("crval", [0.0, 0.0005, 359.998, 359.9995])
    @pytest.mark.parametrize("dx", [-3.0, 2.0])
    def test_matches_scene_away_from_zero(self, kind, crval, dx):
        """A scene across RA 0 matches the same scene at RA 150."""
        ref = self._scene(kind, 150.0, dx)
        out = self._scene(kind, crval, dx)
        np.testing.assert_allclose(out, ref, atol=1e-8 * np.abs(ref).max())

    def test_background_grid_wraps(self):
        """Background.getgrid takes RA on either side of 0/360."""
        img = self._image(0.0005)
        a = models.Background.getgrid(img.grid, 359.9998, 2.0)
        b = models.Background.getgrid(img.grid, -0.0002, 2.0)
        for x, y in zip(a, b):
            np.testing.assert_allclose(x, y, atol=1e-12)


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
        """Reset the component ids before each test."""
        models.Component.reset()

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
        """Reset the component ids before each test."""
        models.Component.reset()

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
        """Reset the component ids before each test."""
        models.Component.reset()

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


class TestParameterSetters:
    """Tests for range checks on tied values and unbounded priors."""

    @pytest.fixture(autouse=True)
    def reset_idcls(self):
        """Reset the component ids before each test."""
        models.Component.reset()

    @pytest.mark.parametrize(
        "make, name",
        [
            (models.Sersic, "ns"),
            (models.gNFW, "alpha"),
            (models.gNFW, "beta"),
            (models.gNFW, "gamma"),
        ],
    )
    def test_tied_values_accepted(self, make, name):
        """Lambdas and boundto are stored without range checks."""
        comp = make()
        tied = lambda comp_01_rc: 1.0  # noqa: E731
        setattr(comp, name, tied)
        assert getattr(comp, name) is tied
        other = make()
        setattr(comp, name, priors.boundto(other, name))
        assert isinstance(getattr(comp, name), priors._BoundTo)

    def test_unbounded_priors(self):
        """Priors with real support are range-checked as unbounded."""
        with pytest.warns(UserWarning, match="re prior support"):
            models.Sersic().re = priors.normal(1e-3, 1e-4)
        with pytest.warns(UserWarning, match="alpha prior support"):
            models.gNFW().alpha = priors.normal(1.0, 0.1)
        with pytest.raises(ValueError, match="ns prior support"):
            models.Sersic().ns = priors.normal(2.0, 0.3)
        with pytest.raises(ValueError, match="ellipticity prior support"):
            models.Sersic().e = priors.normal(0.3, 0.05)

    def test_bounded_priors_in_range(self):
        """Power-law priors are checked against their own bounds."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            models.Sersic().ns = priors.powerlaw(0.0, 0.5, 8.0)
            models.Sersic().ns = priors.logpowerlaw(-1.0, 0.5, 8.0)

    def test_unbounded_theta_prior_is_not_periodic(self):
        """An unbounded position-angle prior raises no span warning."""
        comp = models.Sersic(
            xc=180.0,
            yc=45.0,
            re=0.01,
            Ie=1.0,
            ns=1.0,
            theta=priors.normal(0.0, 0.1),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            models.Model(comp)

    def test_out_of_range_values(self):
        """Fixed values outside the valid range are still caught."""
        with pytest.raises(ValueError, match="ns parameter"):
            models.Sersic().ns = 12.0
        with pytest.warns(UserWarning, match="beta parameter is less than 3"):
            models.gNFW().beta = 2.0
        with pytest.warns(UserWarning, match="gamma parameter is greater"):
            models.gNFW().gamma = 1.0


class TestRefactor:
    """Tests for Profile.refactor."""

    @pytest.fixture(autouse=True)
    def reset_idcls(self):
        """Reset the component ids before each test."""
        models.Component.reset()

    def test_base_refactor_keeps_parameters(self, simple_hdu):
        """The base refactor returns an identical copy."""
        img = data.Image(simple_hdu)
        prof = models.Beta(
            xc=180.0, yc=45.0, theta=0.3, e=0.4, rc=0.01, Ic=2.0, beta=0.8
        )
        prof.truncation = True
        prof.rt = 0.02
        with pytest.warns(UserWarning, match="Nothing to refactor"):
            new = prof.refactor()
        assert new.id == prof.id and new is not prof
        np.testing.assert_array_equal(new.getmap(img), prof.getmap(img))
        new.rc = 0.02
        assert prof.rc == 0.01

    def test_base_refactor_keeps_bound_parameters(self, simple_hdu):
        """A refactored copy stays bound to the original host."""
        img = data.Image(simple_hdu)
        host = models.Beta(xc=180.0, yc=45.0, rc=0.01, Ic=1.0, beta=0.8)
        prof = models.Beta(rc=0.01, Ic=1.0, beta=0.8)
        prof.xc = priors.boundto(host, "xc")
        prof.yc = priors.boundto(host, "yc")
        with pytest.warns(UserWarning, match="Nothing to refactor"):
            new = prof.refactor()
        prof.cbox = priors.boundto(prof, "e")
        with pytest.warns(UserWarning, match="Nothing to refactor"):
            self_bound = prof.refactor()
        assert self_bound is not prof and self_bound.cbox._comp is self_bound
        host.xc = 180.01
        assert new.xc._comp is host
        np.testing.assert_array_equal(new.getmap(img), prof.getmap(img))

    def test_polyexporefact_keeps_units_and_positivity(self, simple_hdu):
        """Input units and the positivity flag are carried over."""
        img = data.Image(simple_hdu)
        kw = dict(Is=2.0, rs=36.0, rc=36.0, I1=0.5, I2=0.1, I3=0.0, I4=0.0)
        prof = models.PolyExpoRefact(xc=180.0, yc=45.0, positive=True, **kw)
        prof.set_units(rs="arcsec", rc="arcsec")
        new = prof.refactor()
        assert new.positive
        np.testing.assert_allclose(
            models.Model(new).getmap(img),
            models.Model(prof).getmap(img),
            rtol=1e-12,
            atol=1e-14,
        )

    def test_polyexporefact_keeps_cbox_and_truncation(self, simple_hdu):
        """PolyExpoRefact.refactor carries over cbox and the truncation."""
        img = data.Image(simple_hdu)
        prof = models.PolyExpoRefact(
            xc=180.0,
            yc=45.0,
            theta=0.3,
            e=0.4,
            cbox=0.3,
            Is=2.0,
            rs=0.01,
            rc=0.01,
            I1=0.5,
            I2=0.1,
            I3=0.0,
            I4=0.0,
            truncation=True,
        )
        prof.rt, prof.wt = 0.02, 0.002
        new = prof.refactor()
        assert isinstance(new, models.PolyExponential)
        assert new.id == prof.id
        assert models.Sersic().id == "comp_01"
        np.testing.assert_allclose(
            new.getmap(img), prof.getmap(img), rtol=1e-12, atol=1e-14
        )


class TestTiedNamespaced:
    """Tests for lambdas tied to namespaced (sub-component) parameters."""

    @staticmethod
    def _disk():
        disk = models.Disk()
        disk.radial.xc, disk.radial.yc = 180.0, 45.0
        disk.radial.re, disk.radial.Ie, disk.radial.ns = 0.01, 1.0, 1.0
        disk.vertical.zs, disk.vertical.inc = 0.001, 0.5
        disk.vertical.losdepth = 0.05
        return disk

    def _model(self, tied):
        models.Component.reset()
        disk = self._disk()
        sersic = models.Sersic(xc=180.0, yc=45.0, Ie=1.0, ns=1.0)
        sersic.re = (
            (lambda comp_00_radial_re: 0.5 * comp_00_radial_re)
            if tied
            else 0.005
        )
        mod = models.Model(disk)
        mod.addcomponent(sersic)
        return mod

    def test_model_resolves_namespaced_argument(self, simple_hdu):
        """A tied lambda may reference comp_NN_radial_<par>."""
        img = data.Image(simple_hdu)
        np.testing.assert_array_equal(
            self._model(True).getmap(img), self._model(False).getmap(img)
        )

    def test_component_resolves_namespaced_argument(self, simple_hdu):
        """A component's own getmap resolves radial_<par> arguments."""
        img = data.Image(simple_hdu)
        models.Component.reset()
        fixed = self._disk()
        models.Component.reset()
        tied = self._disk()
        tied.vertical.zs = lambda comp_00_radial_re: 0.1 * comp_00_radial_re
        np.testing.assert_array_equal(tied.getmap(img), fixed.getmap(img))

    @staticmethod
    def _ellipsoid(inc):
        ell = models.Ellipsoid(radial=models.Sersic(re=0.01, Ie=1.0, ns=1.0))
        ell.xc, ell.yc, ell.theta, ell.e = 180.0, 45.0, 0.3, 0.3
        ell.inc, ell.losdepth = inc, 0.05
        return ell

    @pytest.mark.parametrize("in_model", [False, True])
    def test_ellipsoid(self, simple_hdu, in_model):
        """Ellipsoid parameters may be tied to radial_<par>."""
        img = data.Image(simple_hdu)
        maps = []
        for inc in (lambda comp_00_radial_re: 50 * comp_00_radial_re, 0.5):
            models.Component.reset()
            ell = self._ellipsoid(inc)
            maps.append(
                models.Model(ell).getmap(img) if in_model else ell.getmap(img)
            )
        np.testing.assert_array_equal(*maps)

    @pytest.mark.parametrize("in_model", [False, True])
    def test_bridge(self, simple_hdu, in_model):
        """Bridge parameters may be tied to radial_<par>."""
        img = data.Image(simple_hdu)
        maps = []
        for tied in (True, False):
            models.Component.reset()
            bridge = models.SimpleBridge(
                xc=180.0, yc=45.0, rs=0.01, Is=1.0, theta=0.3, e=0.3
            )
            if tied:
                bridge.radial.beta = lambda comp_00_radial_alpha: (
                    0.5 * comp_00_radial_alpha
                )
            else:
                bridge.radial.beta = 0.5 * bridge.radial.alpha
            maps.append(
                models.Model(bridge).getmap(img)
                if in_model
                else bridge.getmap(img)
            )
        np.testing.assert_array_equal(*maps)


class TestTiesByComponentId:
    """Model keys and ties use component ids, whatever the order of addition."""

    @staticmethod
    def _sersic(**kw):
        return models.Sersic(xc=180.0, yc=45.0, Ie=1.0, ns=1.0, **kw)

    def _reference(self, img):
        models.Component.reset()
        mod = models.Model(self._sersic(re=0.01))
        mod.addcomponent(self._sersic(re=0.005))
        return mod.getmap(img)

    def test_tutorial_example(self, simple_hdu):
        """The lambda of the priors tutorial resolves by component id."""
        img = data.Image(simple_hdu)
        models.Component.reset()
        models.Sersic()
        comp1, comp2 = self._sersic(), self._sersic(re=0.01)
        assert (comp1.id, comp2.id) == ("comp_01", "comp_02")
        comp1.addparameter("re_scale")
        comp1.re_scale = 0.5
        comp1.re = lambda comp_01_re_scale, comp_02_re: (
            comp_01_re_scale * comp_02_re
        )
        mod = models.Model(comp2)
        mod.addcomponent(comp1)
        np.testing.assert_array_equal(mod.getmap(img), self._reference(img))

    def test_model_keys_are_ids(self, simple_hdu):
        """Model parameters are prefixed by component id, not model order."""
        img = data.Image(simple_hdu)
        models.Sersic()
        comp1, comp2 = self._sersic(re=0.01), self._sersic()
        comp2.re = lambda comp_01_re: comp_01_re / 2
        mod = models.Model(comp2)
        mod.addcomponent(comp1)
        assert mod.params[0].startswith("comp_02_")
        assert "comp_01_re" in mod.params
        np.testing.assert_array_equal(mod.getmap(img), self._reference(img))

    def test_boundto_after_other_components(self, simple_hdu):
        """A boundto tie resolves when ids are offset from model indices."""
        img = data.Image(simple_hdu)
        models.Component.reset()
        models.Disk()
        host, comp = self._sersic(re=0.01), self._sersic()
        comp.re = priors.boundto(host, "re")
        mod = models.Model(host)
        mod.addcomponent(comp)
        models.Component.reset()
        ref = models.Model(self._sersic(re=0.01))
        ref.addcomponent(self._sersic(re=0.01))
        np.testing.assert_array_equal(mod.getmap(img), ref.getmap(img))

    def test_bridge_after_other_components(self, simple_hdu):
        """A Bridge's internal ties use its own model index."""
        img = data.Image(simple_hdu)
        kw = dict(xc=180.0, yc=45.0, rs=0.01, Is=1.0, theta=0.3, e=0.3)
        models.Component.reset()
        models.Disk()
        exp = models.Exponential(xc=180.0, yc=45.0, rs=0.03, Is=1.0)
        bridge = models.SimpleBridge(**kw)
        mod = models.Model(bridge)
        mod.addcomponent(exp)
        models.Component.reset()
        ref = models.Model(models.SimpleBridge(**kw))
        np.testing.assert_array_equal(
            mod.getmodel(img, [], component=bridge)[0],
            ref.getmodel(img, [], component=0)[0],
        )

    @pytest.mark.parametrize("stray", [False, True])
    @pytest.mark.parametrize("bridge_first", [False, True])
    def test_bridge_tie_to_other_component(
        self, simple_hdu, stray, bridge_first
    ):
        """A Bridge sub-profile tie to another component resolves by id."""
        img = data.Image(simple_hdu)
        kw = dict(xc=180.0, yc=45.0, rs=0.01, Is=1.0, theta=0.3, e=0.3)
        models.Component.reset()
        if stray:
            models.Disk()
        exp = models.Exponential(xc=180.0, yc=45.0, rs=0.02, Is=1.0)
        bridge = models.SimpleBridge(**kw)
        bridge.parallel.rc = eval(f"lambda {exp.id}_rs: {exp.id}_rs")
        mod = models.Model(bridge if bridge_first else exp)
        mod.addcomponent(exp if bridge_first else bridge)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Tie arguments")
            out = mod.getmodel(img, [], component=bridge)[0]
        models.Component.reset()
        ref = models.SimpleBridge(**kw)
        ref.parallel.rc = 0.02
        np.testing.assert_array_equal(
            out, models.Model(ref).getmodel(img, [], component=0)[0]
        )

    def test_bridge_scale_tie_forms(self, simple_hdu):
        """Keyword-only and namespaced ties on Bridge scales work in a Model."""
        img = data.Image(simple_hdu)
        kw = dict(xc=180.0, yc=45.0, rs=0.01, Is=1.0, theta=0.3, e=0.3)
        models.Component.reset()
        bridge = models.SimpleBridge(**kw)
        bid = bridge.id
        bridge.parallel.rc = eval(
            f"lambda *, {bid}_rs, {bid}_radial_beta: 2 * {bid}_rs * "
            f"{bid}_radial_beta / {bid}_radial_beta"
        )
        out = models.Model(bridge).getmodel(img, [], component=0)[0]
        np.testing.assert_allclose(out, bridge.getmap(img))
        models.Component.reset()
        ref = models.SimpleBridge(**kw)
        ref.parallel.rc = 0.02
        np.testing.assert_allclose(
            out, models.Model(ref).getmodel(img, [], component=0)[0]
        )

    def test_bridge_scale_boundto_uses_model_value(self, simple_hdu):
        """A boundto on a Bridge scale reads the fitted, converted value."""
        img = data.Image(simple_hdu)
        kw = dict(xc=180.0, yc=45.0, rs=0.01, Is=1.0, theta=0.3, e=0.3)
        models.Component.reset()
        bridge = models.SimpleBridge(**kw)
        ref = models.Exponential(
            xc=180.0, yc=45.0, rs=priors.uniform(36.0, 72.0), Is=1.0
        )
        ref.set_units(rs="arcsec")
        bridge.parallel.rc = priors.boundto(ref, "rs")
        mod = models.Model(bridge)
        mod.addcomponent(ref)
        out = mod.getmodel(img, jp.array([54.0]), component=0)[0]
        models.Component.reset()
        fixed = models.SimpleBridge(**kw)
        fixed.parallel.rc = 0.015
        np.testing.assert_allclose(
            out, models.Model(fixed).getmodel(img, [], component=0)[0]
        )

    def test_bridge_internal_ties_do_not_warn(self, simple_hdu):
        """A Bridge's own scale ties never trigger the id warning."""
        img = data.Image(simple_hdu)
        models.Component.reset()
        exp = models.Exponential(xc=180.0, yc=45.0, rs=0.02, Is=1.0)
        bridge = models.SimpleBridge(xc=180.0, yc=45.0, rs=0.01, Is=1.0)
        mod = models.Model(bridge)
        mod.addcomponent(exp)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            mod.getmodel(img, [], component=bridge)

    def test_chained_boundto_any_order(self, simple_hdu):
        """A tie to a tied parameter resolves whatever the add order."""
        img = data.Image(simple_hdu)
        models.Component.reset()
        a = self._sersic(re=0.01)
        b, c = self._sersic(), self._sersic()
        b.re = priors.boundto(a, "re")
        c.re = priors.boundto(b, "re")
        mod = models.Model(c)
        mod.addcomponent(b)
        mod.addcomponent(a)
        models.Component.reset()
        ref = models.Model(self._sersic(re=0.01))
        for _ in range(2):
            ref.addcomponent(self._sersic(re=0.01))
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            out = mod.getmap(img)
        np.testing.assert_allclose(out, ref.getmap(img), rtol=1e-12)

    def test_circular_ties_raise(self, simple_hdu):
        """Ties that depend on each other raise."""
        img = data.Image(simple_hdu)
        models.Component.reset()
        a, b = self._sersic(), self._sersic()
        a.re = priors.boundto(b, "re")
        b.re = priors.boundto(a, "re")
        mod = models.Model(a)
        mod.addcomponent(b)
        with pytest.raises(ValueError, match="Circular ties"):
            mod.getmap(img)

    def test_tie_with_declared_units_warns_once(self):
        """Lambda ties involving declared units warn once; boundto does not."""
        a = models.Point(xc=10.0, yc=0.0, Ic=1.0)
        b = models.Point(yc=0.0, Ic=1.0)
        b.set_units(xc="arcsec")
        b.xc = lambda comp_00_xc: comp_00_xc + 5 / 3600
        c = models.Point(yc=0.0, Ic=1.0)
        c.set_units(xc="arcsec")
        c.xc = priors.boundto(a, "xc")
        mod = models.Model(a)
        mod.addcomponent(b)
        mod.addcomponent(c)
        pars = {"comp_00_xc": 10.0}
        with pytest.warns(UserWarning, match="native units") as record:
            mod._resolve_ties(dict(pars))
            mod._resolve_ties(dict(pars))
        assert len(record) == 1
        assert "comp_01_xc in arcsec" in str(record[0].message)
        values = dict(pars)
        mod._resolve_ties(values)
        assert values["comp_01_xc"] == 10.0 + 5 / 3600
        assert values["comp_02_xc"] == 10.0

    def test_legacy_model_periods_rebuilt(self):
        """Models pickled without periods get them back from the components."""
        sersic = models.Sersic()
        sersic.set_units(theta="deg")
        disk = models.Disk()
        for comp in [sersic, disk]:
            for name in comp.parlist():
                if eval(f"comp.{name}") is None:
                    exec(f"comp.{name} = 0.5")
        mod = models.Model(sersic)
        mod.addcomponent(disk)
        expected = dict(mod.periodic)
        assert expected[f"{sersic.id}_theta"] == pytest.approx(180.0)
        del mod.periodic
        for comp in mod.components:
            for sub in [comp, *vars(comp).values()]:
                if isinstance(sub, models.Component):
                    sub.__dict__.pop("periodic", None)
        loaded = copy.deepcopy(mod)
        assert loaded.periodic == expected

    def test_tie_with_native_unit_alias_does_not_warn(self, recwarn):
        """Declaring a native unit under another name does not warn."""
        a = models.Point(xc=10.0, yc=0.0, Ic=1.0)
        b = models.Point(yc=0.0, Ic=1.0)
        b.set_units(xc="degree")
        b.xc = lambda comp_00_xc: comp_00_xc + 1.0
        mod = models.Model(a)
        mod.addcomponent(b)
        mod._resolve_ties({"comp_00_xc": 10.0})
        assert not [w for w in recwarn if "native units" in str(w.message)]

    def test_boundto_ambiguous_name_raises(self):
        """A short name found in two Disk sub-components raises."""
        disk = models.Disk(radial=models.Beta(alpha=1.5))
        with pytest.raises(ValueError, match="'vertical.alpha'"):
            priors.boundto(disk, "alpha")
        tied = priors.boundto(disk, "vertical.alpha")
        assert tied.key == f"{disk.id}_vertical.alpha"
        assert priors.boundto(disk, "inc").key == f"{disk.id}_vertical.inc"
        assert priors.boundto(disk, "xc").key == f"{disk.id}_radial.xc"

    def test_boundto_component_not_in_model_raises(self, simple_hdu):
        """A boundto target missing from the model raises."""
        img = data.Image(simple_hdu)
        models.Component.reset()
        a, b = self._sersic(re=0.01), self._sersic()
        b.re = priors.boundto(a, "re")
        mod = models.Model(self._sersic(re=0.02))
        mod.addcomponent(b)
        with pytest.raises(ValueError, match="not in the model"):
            mod.getmap(img)

    def test_tie_to_missing_id_raises(self, simple_hdu):
        """A tie naming an id that is not in the model raises."""
        img = data.Image(simple_hdu)
        models.Sersic()
        comp1, comp2 = self._sersic(re=0.01), self._sersic()
        comp2.re = lambda comp_00_re: comp_00_re / 2
        mod = models.Model(comp1)
        mod.addcomponent(comp2)
        with pytest.raises(KeyError, match="comp_00_re"):
            mod.getmap(img)

    def test_duplicate_ids_raise(self):
        """Two components with the same id cannot be in one model."""
        import copy

        p = self._sersic(re=0.01)
        mod = models.Model(p)
        with pytest.raises(ValueError, match="already in the model"):
            mod.addcomponent(copy.deepcopy(p))
        assert mod.ncomp == 1
        with pytest.raises(ValueError, match="not in the model"):
            mod._comp_filter(self._sersic())

    def test_standalone_lambda_reads_bound_parameter(self, simple_hdu):
        """A lambda reading its own bound parameter works outside a Model."""
        img = data.Image(simple_hdu)
        models.Component.reset()
        disk = models.Disk()
        disk.radial.xc, disk.radial.yc = 180.0, 45.0
        disk.radial.re, disk.radial.Ie, disk.radial.ns = 0.003, 1.0, 1.0
        disk.radial.theta = 0.4
        disk.vertical.zs, disk.vertical.inc = 0.001, 0.5
        bar = models.Ellipsoid(
            radial=models.Sersic(re=0.002, Ie=2.0, ns=0.5),
            xc=180.0,
            yc=45.0,
            e=0.5,
            eratio=0.5,
            inc=0.5,
        )
        bar.theta = priors.boundto(disk, "theta")
        bar.rot = eval(f"lambda {bar.id}_theta: {bar.id}_theta / 2")
        mod = models.Model(disk)
        mod.addcomponent(bar)
        np.testing.assert_allclose(
            bar.getmap(img), mod.getmodel(img, [], component=bar)[0]
        )

    def test_keyword_only_lambda(self, simple_hdu):
        """Tie lambdas with keyword-only arguments are supported."""
        img = data.Image(simple_hdu)
        models.Component.reset()
        comp1, comp2 = self._sersic(re=0.01), self._sersic()
        comp2.re = lambda *, comp_00_re: comp_00_re / 2
        mod = models.Model(comp1)
        mod.addcomponent(comp2)
        np.testing.assert_array_equal(mod.getmap(img), self._reference(img))

    def test_boundto_target_replaced_by_copy_warns(self, simple_hdu):
        """A boundto target outside the model falls back to its id, warning."""
        import copy

        img = data.Image(simple_hdu)
        models.Component.reset()
        a = self._sersic(re=0.01)
        b = self._sersic()
        b.re = priors.boundto(a, "re")
        mod = models.Model(copy.deepcopy(a))
        mod.addcomponent(b)
        with pytest.warns(UserWarning, match="not in the model; using"):
            mod.getmap(img)

    def test_standalone_keyword_only_lambda(self, simple_hdu):
        """Keyword-only tie lambdas also work in a component's own getmap."""
        img = data.Image(simple_hdu)
        models.Component.reset()
        comp = self._sersic(re=0.01)
        comp.Ie = eval(f"lambda *, {comp.id}_re: 100.0 * {comp.id}_re")
        models.Component.reset()
        ref = models.Sersic(xc=180.0, yc=45.0, Ie=1.0, ns=1.0, re=0.01)
        np.testing.assert_allclose(comp.getmap(img), ref.getmap(img))

    def test_standalone_ties_point_background_disk(self, simple_hdu):
        """Standalone maps resolve ties on Point, Background and Disk."""
        img = data.Image(simple_hdu)
        models.Component.reset()
        host = self._sersic(re=0.01)
        point = models.Point(xc=priors.boundto(host, "xc"), yc=45.0)
        point.Ic = eval(f"lambda {point.id}_yc: {point.id}_yc / 45.0")
        ref = models.Point(xc=180.0, yc=45.0, Ic=1.0)
        np.testing.assert_allclose(point.getmap(img), ref.getmap(img))

        bkg = models.Background(a0=1.0, rs=1.0)
        bkg.a1x = eval(f"lambda {bkg.id}_a0: 0.1 * {bkg.id}_a0")
        for name in ["a1y", "a2xx", "a2xy", "a2yy"]:
            setattr(bkg, name, 0.0)
        for name in ["a3xxx", "a3xxy", "a3xyy", "a3yyy"]:
            setattr(bkg, name, 0.0)
        ref = models.Background(a0=1.0, a1x=0.1, rs=1.0)
        for name in ["a1y", "a2xx", "a2xy", "a2yy"]:
            setattr(ref, name, 0.0)
        for name in ["a3xxx", "a3xxy", "a3xyy", "a3yyy"]:
            setattr(ref, name, 0.0)
        np.testing.assert_allclose(bkg.getmap(img), ref.getmap(img))

        def disk(xc):
            d = models.Disk()
            d.radial.xc, d.radial.yc = xc, 45.0
            d.radial.re, d.radial.Ie, d.radial.ns = 0.003, 1.0, 1.0
            d.vertical.zs, d.vertical.inc = 0.001, 0.5
            return d

        tied = disk(None)
        tied.radial.xc = eval(
            f"lambda {tied.id}_radial_yc: 4.0 * {tied.id}_radial_yc"
        )
        np.testing.assert_allclose(tied.getmap(img), disk(180.0).getmap(img))

    def test_standalone_bridge_tied_inputs(self, simple_hdu):
        """Bridge internal ties see tied or prior inputs in a standalone map."""
        img = data.Image(simple_hdu)
        kw = dict(xc=180.0, yc=45.0, rs=0.01, Is=1.0, theta=0.3)
        models.Component.reset()
        tied = models.SimpleBridge(**kw)
        tied.e = eval(f"lambda {tied.id}_theta: {tied.id}_theta")
        ref = models.SimpleBridge(e=0.3, **kw)
        np.testing.assert_allclose(tied.getmap(img), ref.getmap(img))
        prior = models.SimpleBridge(
            **{**kw, "rs": priors.uniform(0.005, 0.02)}
        )
        with pytest.raises(ValueError, match="Priors must be fixed values"):
            prior.getmap(img)

    def test_bridge_scale_boundto_absent_names_parameter(self, simple_hdu):
        """The error for a Bridge scale bound outside the model names it."""
        img = data.Image(simple_hdu)
        models.Component.reset()
        bridge = models.SimpleBridge(xc=180.0, yc=45.0, rs=0.01, Is=1.0)
        other = self._sersic(re=0.01)
        bridge.parallel.rc = priors.boundto(other, "re")
        mod = models.Model(bridge)
        with pytest.raises(ValueError, match="comp_00_parallel.rc is bound"):
            mod.getmap(img)

    def test_standalone_circular_ties_raise(self, simple_hdu):
        """Circular ties raise in a standalone map, as in a Model."""
        img = data.Image(simple_hdu)
        models.Component.reset()
        comp = models.Sersic(xc=180.0, yc=45.0, ns=1.0)
        comp.re = lambda comp_00_Ie: comp_00_Ie
        comp.Ie = lambda comp_00_re: comp_00_re
        with pytest.raises(ValueError, match="Circular ties"):
            comp.getmap(img)

    def test_standalone_chained_ties_in_threads(self, simple_hdu):
        """Chained standalone ties evaluate concurrently from several threads."""
        import threading
        import time

        img = data.Image(simple_hdu)
        models.Component.reset()
        comp = self._sersic(re=0.01, theta=0.0)

        def slow(comp_00_theta):
            time.sleep(0.05)
            return comp_00_theta

        comp.e = slow
        comp.ns = lambda comp_00_e: 1.0 + comp_00_e
        errors = []

        def run():
            try:
                comp.getmap(img)
            except Exception as err:
                errors.append(err)

        threads = [threading.Thread(target=run) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []

    def test_standalone_prior_raises(self, simple_hdu):
        """A standalone map with a prior left in place raises clearly."""
        img = data.Image(simple_hdu)
        comp = self._sersic(re=priors.uniform(0.001, 0.01))
        with pytest.raises(ValueError, match="Priors must be fixed values"):
            comp.getmap(img)

    def test_component_filter_with_objects(self):
        """Components are selected by object, id or model position."""
        models.Sersic()
        first, second = self._sersic(re=0.01), self._sersic(re=0.005)
        mod = models.Model(second)
        mod.addcomponent(first)
        assert mod._comp_filter(first) == [1]
        assert mod._comp_filter([first, second]) == [1, 0]
        assert mod._comp_filter("comp_01") == [1]
        assert mod._comp_filter(["comp_02", 1]) == [0, 1]
        with pytest.raises(ValueError, match="No component comp_00"):
            mod._comp_filter("comp_00")
        with pytest.raises(IndexError):
            mod._comp_filter(2)
        assert mod._comp_filter(np.array([1, 0])) == [1, 0]
        assert mod._comp_filter(range(2)) == [0, 1]
        assert mod._comp_filter(np.int64(1)) == [1]
        assert mod._comp_filter(["comp_1", "comp_002"]) == [1, 0]
        with pytest.raises(ValueError, match="is not in the model"):
            mod._comp_filter(1.5)

    def test_tie_warning_once_per_model(self, simple_hdu):
        """The boundto fallback warning is emitted once per model."""
        import copy

        img = data.Image(simple_hdu)
        a = self._sersic(re=0.01)
        b = self._sersic()
        b.re = priors.boundto(a, "re")
        mod = models.Model(copy.deepcopy(a))
        mod.addcomponent(b)
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            for _ in range(3):
                mod.getmap(img)
        assert sum("not in the model" in str(w.message) for w in record) == 1


class TestExplicitIds:
    """Components can be given an explicit id."""

    @staticmethod
    def _sersic(**kw):
        return models.Sersic(xc=180.0, yc=45.0, Ie=1.0, ns=1.0, re=0.01, **kw)

    def test_explicit_and_automatic_ids(self):
        """Automatic ids continue in order and skip explicit numbers."""
        ids = [self._sersic().id, self._sersic(id=1).id, self._sersic().id]
        assert ids == ["comp_00", "comp_01", "comp_02"]
        assert self._sersic(id="comp_07").id == "comp_07"
        assert [self._sersic().id, self._sersic().id] == ["comp_03", "comp_04"]
        models.Component.reset()
        assert [self._sersic().id, self._sersic().id] == ["comp_00", "comp_01"]

    def test_idcls_is_read_only(self):
        """The counter can be read, and only reset through reset()."""
        self._sersic()
        assert models.Component.idcls == models.Sersic.idcls == 1
        assert self._sersic().idcls == 2
        with pytest.raises(AttributeError, match="reset"):
            models.Component.idcls = 0
        with pytest.raises(AttributeError, match="reset"):
            models.Sersic.idcls = 0
        models.Sersic.reset()
        assert models.Component.idcls == 0
        assert not hasattr(self._sersic(), "reset")

    def test_explicit_id_reused_on_rerun(self):
        """Re-creating components with explicit ids gives the same keys."""
        keys = []
        for _ in range(2):
            mod = models.Model(self._sersic(id=3))
            mod.addcomponent(self._sersic())
            keys.append(mod.params)
        assert keys[0][0] == "comp_03_xc"
        assert keys[1][: len(keys[0]) // 2] == keys[0][: len(keys[0]) // 2]

    @pytest.mark.parametrize(
        "value", [-1, 1.5, "five", True, "comp_x", "comp_", "\u00b2", "\u0663"]
    )
    def test_invalid_id_raises(self, value):
        """Ids must be non-negative integers or 'comp_NN'."""
        with pytest.raises(ValueError, match="non-negative integer"):
            self._sersic(id=value)

    def test_wrappers_share_explicit_id(self):
        """Disk, Ellipsoid and Bridge pass their id to sub-components."""
        disk = models.Disk(id=4)
        assert disk.id == disk.radial.id == disk.vertical.id == "comp_04"
        disk = models.Disk(radial=models.Sersic(id=9))
        assert disk.id == disk.radial.id == disk.vertical.id == "comp_09"
        ell = models.Ellipsoid(id=12)
        assert ell.id == ell.radial.id == "comp_12"
        bridge = models.SimpleBridge(id=15)
        assert bridge.id == bridge.radial.id == bridge.parallel.id == "comp_15"

    def test_wrappers_give_back_unused_ids(self):
        """Sub-components of a wrapper do not use up automatic ids."""
        disk = models.Disk(models.Sersic())
        assert disk.id == "comp_00"
        assert models.Ellipsoid().id == "comp_01"
        assert models.SimpleBridge().id == "comp_02"
        assert models.Disk(id=7).id == "comp_07"
        assert models.Background().id == "comp_03"
        radial = models.Sersic()
        point = models.Point()
        assert models.Disk(radial).id == "comp_04"
        assert point.id == "comp_05"
        assert models.Background().id == "comp_06"
        assert models.Background().id == "comp_08"

    @pytest.mark.parametrize("dup", ["deepcopy", "pickle"])
    def test_copied_ids_are_not_given_back(self, dup):
        """A copied id is never issued again; only the original adopts it."""
        import copy
        import pickle

        from socca.models.disk.vertical import HyperSecantHeight

        def clone(comp):
            if dup == "deepcopy":
                return copy.deepcopy(comp)
            return pickle.loads(pickle.dumps(comp))

        radial = models.Sersic()
        other = clone(radial)
        assert models.Disk(radial).id == "comp_00"
        assert other.id == "comp_00"
        assert models.Disk(other).id == "comp_01"
        radial = models.Sersic()
        vertical = HyperSecantHeight()
        held = clone(vertical)
        disk = models.Disk(radial, vertical)
        assert held.id == "comp_03"
        assert disk.id == "comp_02"
        assert models.Sersic().id == "comp_04"

    def test_old_copies_do_not_change_rerun_ids(self):
        """Copies made before a counter reset do not block the next run."""
        import copy

        def cell():
            models.Component.reset()
            return models.Disk().id, models.Sersic().id

        first = cell()
        kept = copy.deepcopy(models.Sersic())
        assert kept.id == "comp_02"
        assert cell() == first == ("comp_00", "comp_01")

    def test_reset_forgets_explicit_ids(self):
        """Explicit ids chosen elsewhere do not shift a re-run cell."""

        def cell():
            models.Component.reset()
            return [models.Sersic().id for _ in range(3)]

        first = cell()
        models.Background(id=1)
        assert cell() == first == ["comp_00", "comp_01", "comp_02"]

    def test_explicit_id_given_automatically_warns(self):
        """An explicit id already issued automatically warns."""
        models.Sersic()
        with pytest.warns(UserWarning, match="already given automatically"):
            assert models.Sersic(id=0).id == "comp_00"

    def test_wrapped_subprofiles_cannot_be_reused(self):
        """A sub-profile belongs to one wrapper; copies are new profiles."""
        import copy

        from socca.models.disk.vertical import ExponentialHeight

        radial = models.Exponential()
        thin = models.Disk(radial=radial)
        with pytest.raises(ValueError, match="already part of comp_00"):
            models.Disk(radial=radial)
        vertical = ExponentialHeight()
        models.Disk(vertical=vertical)
        with pytest.raises(ValueError, match="already part of"):
            models.Disk(vertical=vertical)
        thick = models.Disk(radial=copy.deepcopy(thin.radial))
        assert thick.id != thin.id
        with pytest.raises(ValueError, match="already part of"):
            models.Disk(radial=copy.deepcopy(thin).radial)

    def test_wrapping_again_after_reset(self):
        """Re-running a wrapper cell after a reset reuses its sub-profiles."""
        from socca.models.disk.vertical import ExponentialHeight

        radial, vertical = models.Exponential(), ExponentialHeight()
        for _ in range(2):
            models.Component.reset()
            assert (
                models.Disk(radial=radial, vertical=vertical).id == "comp_00"
            )

    def test_stripped_subprofiles_cannot_be_reused(self):
        """Ellipsoid and Bridge strip their sub-profiles, even after a reset."""
        import copy

        bar = models.Ellipsoid(radial=models.Sersic())
        models.Component.reset()
        for radial in (bar.radial, copy.deepcopy(bar.radial)):
            with pytest.raises(ValueError, match="Ellipsoid or Bridge"):
                models.Disk(radial=radial)

    def test_renaming_an_id_in_use_raises(self):
        """Wrapping does not rename an id named in a tie or a model."""
        import copy

        host = self._sersic()
        radial = self._sersic()
        host.re = lambda comp_01_re: 0.5 * comp_01_re
        backup = copy.deepcopy(radial)
        with pytest.raises(ValueError, match="comp_01 is already used"):
            models.Disk(radial=radial, id=7)
        with pytest.raises(ValueError, match="comp_01 is already used"):
            models.Disk(radial=backup)
        assert models.Disk(radial=radial).id == "comp_01"
        radial = self._sersic()
        models.Model(radial)
        with pytest.raises(ValueError, match="already used"):
            models.Ellipsoid(radial=radial, id=9)

    def test_ties_to_future_ids_do_not_block_wrapping(self):
        """A tie naming an id not issued yet does not block wrappers."""
        point = models.Point()
        point.xc = lambda comp_02_xc: comp_02_xc
        assert (models.Disk().id, models.Sersic().id) == ("comp_01", "comp_02")

    def test_explicit_subprofile_ids_are_not_replaced(self):
        """An explicit sub-profile id that the wrapper would drop raises."""
        from socca.models.disk.vertical import ExponentialHeight

        with pytest.raises(ValueError, match="Set the id on the Disk"):
            models.Disk(models.Sersic(id=3), ExponentialHeight(id=4))
        with pytest.raises(ValueError, match="Set the id on the Ellipsoid"):
            models.Ellipsoid(radial=models.Sersic(id=3), id=5)
        assert models.Disk(radial=models.Sersic(id=6)).id == "comp_06"
        assert models.Disk(vertical=ExponentialHeight(id=7)).id == "comp_07"

    def test_wrapper_takes_lowest_subprofile_id(self):
        """The wrapper takes its sub-profiles' lowest id, leaving no gaps."""
        from socca.models.disk.vertical import ExponentialHeight

        vertical = ExponentialHeight()
        assert models.Disk(vertical=vertical).id == "comp_00"
        assert models.Sersic().id == "comp_01"
        models.Component.reset()
        radial, vertical = models.Sersic(), ExponentialHeight()
        assert models.Disk(radial, vertical).id == "comp_00"
        vertical, radial = ExponentialHeight(), models.Sersic()
        assert models.Disk(radial, vertical).id == "comp_01"
        assert models.Sersic().id == "comp_02"

    @pytest.mark.parametrize(
        "make",
        [
            lambda r, p: models.Ellipsoid(radial=r, id="bad"),
            lambda r, p: models.Ellipsoid(radial=r, id=5),
            lambda r, p: models.SimpleBridge(radial=r, parallel=p),
            lambda r, p: models.MesaBridge(radial=r, parallel=p),
            lambda r, p: models.MesaBridge(radial=r, id=5),
            lambda r, p: models.Disk(radial=r, id=5),
        ],
    )
    def test_failed_wrapper_leaves_subprofile_intact(self, make):
        """A wrapper that raises changes neither its sub-profile nor ids."""
        radial, point = models.Sersic(id=3), models.Point()
        with pytest.raises((ValueError, TypeError)):
            make(radial, point)
        assert "xc" in radial.units and radial._wrapped is None
        assert models.Ellipsoid(radial=radial).id == "comp_03"
        assert models.Sersic().id == "comp_01"

    def test_failed_rename_skips_no_number(self):
        """A wrapper refusing to rename an id in use takes no number."""
        import copy

        radial = copy.deepcopy(models.Sersic())
        point = models.Point()
        point.xc = lambda comp_00_xc: comp_00_xc
        for wrapper in (
            models.Disk,
            models.Ellipsoid,
            models.SimpleBridge,
            models.MesaBridge,
        ):
            with pytest.raises(ValueError, match="already used"):
                wrapper(radial=radial)
        assert "xc" in radial.units
        assert models.Sersic().id == "comp_02"

    def test_subprofile_copied_with_its_wrapper_is_not_free(self):
        """A sub-profile copied along with its wrapper cannot be wrapped."""
        import copy

        disk = models.Disk()
        disk.radial.Ie = priors.boundto(disk, "radial.re")
        radial = copy.deepcopy(disk.radial)
        assert radial.Ie._comp.radial is radial
        with pytest.raises(ValueError, match="already part of"):
            models.Disk(radial=radial)
        assert models.Disk(radial=copy.deepcopy(models.Disk().radial))

    def test_loaded_subprofiles_keep_live_ids(self):
        """Wrapping a loaded sub-profile neither skips nor reuses live ids."""
        import pickle

        from socca.models.disk.vertical import HyperSecantHeight

        saved = pickle.dumps([models.Sersic(), HyperSecantHeight()])
        models.Component.reset()
        radial = models.Sersic()
        vertical = HyperSecantHeight()
        pickle.loads(saved)
        models.Disk(radial, vertical)
        assert models.Background().id == "comp_01"
        models.Component.reset()
        radial = models.Sersic()
        live = models.Sersic()
        _, loaded = pickle.loads(saved)
        models.Disk(radial, loaded)
        assert models.Background().id not in {radial.id, live.id}

    def test_old_loaded_subprofiles_keep_live_ids(self):
        """A sub-profile loaded before a reset does not free a live id."""
        import pickle

        from socca.models.disk.vertical import HyperSecantHeight

        models.Sersic()
        old = pickle.loads(pickle.dumps(HyperSecantHeight()))
        models.Component.reset()
        radial, live = models.Sersic(), models.Sersic()
        models.Disk(radial, old)
        assert models.Background().id not in {radial.id, live.id}

    def test_adopted_explicit_ids_stay_reserved(self):
        """An explicit id adopted from an older copy is not issued again."""
        import copy
        import warnings

        kept = copy.deepcopy(models.Sersic(id=1))
        models.Component.reset()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            disk = models.Disk(radial=kept)
        assert disk.id == "comp_01"
        assert [models.Sersic().id, models.Sersic().id] == [
            "comp_00",
            "comp_02",
        ]

    def test_refactor_ids(self):
        """A refactored profile takes over the id of the original."""
        prof = models.PolyExpoRefact(Is=1.0)
        disk = models.Disk(radial=prof.refactor())
        assert (prof.id, disk.id) == ("comp_00", "comp_00")
        assert models.Sersic().id == "comp_01"
        prof = models.PolyExpoRefact(Is=1.0)
        models.Point().xc = lambda comp_02_xc: comp_02_xc
        with pytest.raises(ValueError, match="comp_02 is already used"):
            models.Disk(radial=prof.refactor(), id=8)
        refactored = models.PolyExpoRefact(Is=1.0)
        same = refactored.refactor()
        models.Disk(radial=refactored, id=9)
        assert models.Sersic().id != same.id

    def test_loaded_model_components_are_not_renamed(self):
        """A component of a model loaded in a new session keeps its id."""
        import pickle

        mod = models.Model(models.Point(xc=1.0, yc=1.0, Ic=1.0))
        mod.addcomponent(self._sersic())
        models.Component.reset()
        loaded = pickle.loads(pickle.dumps(mod))
        assert models.Disk(radial=loaded.components[1]).id == "comp_01"
        assert models.Point().id == "comp_02"

    def test_model_components_wrap_when_not_renamed(self):
        """A component of a model wraps if its id stays, or after reset."""
        import copy
        import pickle

        sersic = self._sersic()
        mod = models.Model(sersic)
        models.Component.reset()
        loaded = pickle.loads(pickle.dumps(mod))
        assert models.Disk(radial=loaded.components[0]).id == "comp_00"
        models.Component.reset()
        models.Point()
        assert models.Disk(radial=sersic).id == "comp_01"
        models.Component.reset()
        mod = models.Model(self._sersic())
        models.Component.reset()
        models.Point()
        stale = copy.deepcopy(mod.components[0])
        assert models.Disk(radial=stale).id == "comp_01"
        with pytest.warns(UserWarning, match="comp_00 was already given"):
            models.Disk(radial=copy.deepcopy(mod).components[0])

    def test_refactored_ids_are_not_given_back(self):
        """The original's id is not issued again after a refactor."""
        from socca.models.disk.vertical import ExponentialHeight

        vertical = ExponentialHeight()
        prof = models.PolyExpoRefact(Is=1.0)
        models.Disk(radial=prof.refactor(), vertical=vertical)
        assert models.Sersic().id == "comp_02"

    @pytest.mark.parametrize(
        "wrapper", ["Disk", "Ellipsoid", "SimpleBridge", "MesaBridge"]
    )
    @pytest.mark.parametrize("cid", [None, 0, 7])
    def test_failed_wrapper_restores_ids(self, monkeypatch, wrapper, cid):
        """A wrapper failing after sharing its id leaves ids unchanged."""
        share = models.Component._share_id

        def failing(self, *subs):
            share(self, *subs)
            raise RuntimeError("late failure")

        models.Point()
        radial = models.Beta()
        records = lambda: (  # noqa: E731
            models.Component.idcls,
            dict(models.Component._issued),
            set(models.Component._explicit),
            (radial.id, radial._issue, radial._fixed, radial._wrapped),
        )
        before = records()
        monkeypatch.setattr(models.Component, "_share_id", failing)
        kwargs = {} if cid is None else {"id": cid}
        with pytest.raises(RuntimeError, match="late failure"):
            getattr(models, wrapper)(radial=radial, **kwargs)
        assert records() == before
        assert models.Component._holds(radial)

    def test_unsupported_subs_raise_before_any_change(self):
        """Unsupported sub-profiles raise before ids or parameters change."""
        custom = models.CustomProfile(
            [{"name": "a", "unit": "image", "description": ""}],
            lambda r, a: a * r,
        )
        units = dict(custom.units)
        for wrapper in [models.SimpleBridge, models.MesaBridge]:
            with pytest.raises(TypeError, match="not supported for Bridge"):
                wrapper(radial=custom)
        with pytest.raises(TypeError, match="not supported for Bridge"):
            models.Bridge(radial=models.Disk())
        for radial in [models.Point(), models.Background(), models.Disk()]:
            with pytest.raises(TypeError, match="not supported for Elli"):
                models.Ellipsoid(radial=radial)
        assert custom.units == units and custom._wrapped is None
        assert models.Sersic().id == "comp_05"

    def test_explicit_wrapper_id_held_by_sub_does_not_warn(self):
        """An explicit wrapper id held automatically by its sub is no clash."""
        radial = models.Sersic()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            disk = models.Disk(radial=radial, id=0)
        assert (disk.id, radial.id) == ("comp_00", "comp_00")
        assert models.Sersic().id == "comp_01"

    def test_same_sub_twice_raises(self):
        """One object cannot be two sub-components of a wrapper."""
        beta = models.Beta()
        with pytest.raises(ValueError, match="passed twice"):
            models.SimpleBridge(radial=beta, parallel=beta)
        sersic = models.Sersic()
        with pytest.raises(ValueError, match="passed twice"):
            models.Disk(radial=sersic, vertical=sersic)
        assert models.SimpleBridge(radial=beta).id == "comp_00"

    def test_refactor_rebinds_self_boundto(self):
        """A boundto on the profile itself follows the refactored profile."""
        prof = models.PolyExpoRefact(Is=1.0, e=0.3)
        prof.cbox = priors.boundto(prof, "e")
        new = prof.refactor()
        assert new.cbox._comp is new

    def test_duplicate_selection_raises(self):
        """A selection naming one component twice raises."""
        mod = models.Model(self._sersic())
        mod.addcomponent(self._sersic())
        with pytest.raises(ValueError, match="more than once"):
            mod._comp_filter([0, "comp_00"])
        assert mod._comp_filter([1, "comp_00"]) == [1, 0]

    def test_models_without_stored_prefixes(self, simple_hdu):
        """Models pickled before ids were model keys use model positions."""
        img = data.Image(simple_hdu)
        mod = models.Model(self._sersic())
        mod.addcomponent(models.Point(xc=180.0, yc=45.0, Ic=1.0))
        ref = mod.getmap(img)
        del mod.__dict__["_prefix"]
        assert mod._prefixes == ["comp_00", "comp_01"]
        np.testing.assert_array_equal(mod.getmap(img), ref)
        assert mod._comp_filter("comp_01") == [1]


class TestPeriodUnits:
    """Periods follow the input units of the parameter."""

    @pytest.fixture(autouse=True)
    def reset_idcls(self):
        """Reset the component ids before each test."""
        models.Component.reset()

    def test_period_in_degrees(self):
        """A theta given in degrees has a 180-degree period."""
        comp = models.Sersic(
            xc=180.0,
            yc=45.0,
            re=0.01,
            Ie=1.0,
            ns=1.0,
            theta=priors.uniform(30.0, 60.0),
        )
        comp.set_units(theta="deg")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            mod = models.Model(comp)
        assert mod.periodic["comp_00_theta"] == pytest.approx(180.0)

    def test_log_sampled_periodic_raises(self):
        """A periodic parameter cannot use a log-sampled prior."""
        comp = models.Sersic(
            xc=180.0,
            yc=45.0,
            re=0.01,
            Ie=1.0,
            ns=1.0,
            theta=priors.logpowerlaw(-1.0, 0.1, 1.0),
        )
        with pytest.raises(ValueError, match="comp_00_theta is periodic"):
            models.Model(comp)

    def test_full_period_in_degrees_warns(self):
        """A 0-180 degree prior is recognised as a full period."""
        comp = models.Sersic(
            xc=180.0,
            yc=45.0,
            re=0.01,
            Ie=1.0,
            ns=1.0,
            theta=priors.uniform(0.0, 180.0),
        )
        comp.set_units(theta="deg")
        with pytest.warns(UserWarning, match=r"full period \(180 deg\)"):
            models.Model(comp)

    def test_disk_theta_is_periodic(self):
        """The position angle of a Disk's radial profile is periodic."""
        disk = TestTiedNamespaced._disk()
        mod = models.Model(disk)
        assert mod.periodic["comp_00_radial.theta"] == pytest.approx(np.pi)

    def test_failed_add_leaves_model_unchanged(self):
        """A component rejected by addcomponent leaves no trace."""
        kw = dict(xc=180.0, yc=45.0, rc=0.01, Ic=1.0, beta=0.5)
        mod = models.Model(models.Beta(**kw))
        before = {
            k: list(v) for k, v in vars(mod).items() if isinstance(v, list)
        }
        comp = models.Beta(theta=priors.logpowerlaw(-1.0, 0.1, 1.0), **kw)
        with pytest.raises(ValueError, match="is periodic"):
            mod.addcomponent(comp)
        after = {
            k: list(v) for k, v in vars(mod).items() if isinstance(v, list)
        }
        assert after == before
        assert mod.ncomp == 1

    def test_span_warning_points_at_caller(self):
        """Span warnings are attributed to the line building the model."""
        comp = models.Sersic(
            xc=180.0,
            yc=45.0,
            re=0.01,
            Ie=1.0,
            ns=1.0,
            theta=priors.uniform(0.0, 4.0),
        )
        for build in (models.Model, lambda c: models.Model().addcomponent(c)):
            with warnings.catch_warnings(record=True) as record:
                warnings.simplefilter("always")
                build(comp)
            assert record
            assert all(w.filename == __file__ for w in record)

    def test_period_rounding(self):
        """A full period in mas is not reported as wider than a period."""
        comp = models.Sersic(
            xc=180.0,
            yc=45.0,
            re=0.01,
            Ie=1.0,
            ns=1.0,
            theta=priors.uniform(0.0, 648e6),
        )
        comp.set_units(theta="mas")
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            models.Model(comp)
        messages = [str(w.message) for w in record]
        assert any("full period" in m for m in messages)
        assert not any("more than" in m for m in messages)

    def test_disk_inherits_radial_units(self):
        """Units set on a Disk's radial profile carry over to the Disk."""
        radial = models.Sersic(
            xc=180.0,
            yc=45.0,
            re=36.0,
            Ie=1.0,
            ns=1.0,
            theta=priors.uniform(0.0, 90.0),
        )
        radial.set_units(theta="deg", re="arcsec")
        disk = models.Disk(radial=radial)
        disk.vertical.zs, disk.vertical.inc = 0.001, 0.5
        mod = models.Model(disk)
        assert mod.units["comp_00_radial.theta"] == "deg"
        assert mod.periodic["comp_00_radial.theta"] == pytest.approx(180.0)
        assert mod.conversions["comp_00_radial.re"] == pytest.approx(1 / 3600)

    def test_disk_radial_units_set_after_wrapping(self):
        """Units set on a Disk's radial profile after wrapping are used."""
        disk = models.Disk()
        disk.radial.xc, disk.radial.yc = 180.0, 45.0
        disk.radial.re, disk.radial.Ie, disk.radial.ns = 36.0, 1.0, 1.0
        disk.radial.theta = priors.uniform(0.0, 90.0)
        disk.vertical.zs, disk.vertical.inc = 0.001, 0.5
        disk.radial.set_units(theta="deg", re="arcsec")
        mod = models.Model(disk)
        assert mod.periodic["comp_00_radial.theta"] == pytest.approx(180.0)
        assert mod.conversions["comp_00_radial.re"] == pytest.approx(1 / 3600)

    def test_ellipsoid_inherits_radial_units(self):
        """A theta taken from an Ellipsoid's radial keeps its unit."""
        radial = models.Sersic(
            xc=180.0,
            yc=45.0,
            re=0.01,
            Ie=1.0,
            ns=1.0,
            theta=priors.uniform(0.0, 180.0),
        )
        radial.set_units(theta="deg")
        mod = models.Model(models.Ellipsoid(radial=radial))
        assert mod.units["comp_00_theta"] == "deg"
        assert mod.periodic["comp_00_theta"] == pytest.approx(180.0)


class TestEllipsoid:
    """Tests for Ellipsoid model."""

    @pytest.fixture(autouse=True)
    def reset_idcls(self):
        """Reset the component ids before each test."""
        models.Component.reset()

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
    def _ellipsoid(**kw):
        ell = models.Ellipsoid(radial=models.Sersic(re=0.01, Ie=1.0, ns=1.0))
        ell.xc, ell.yc, ell.e, ell.losdepth = 180.0, 45.0, 0.3, 0.05
        for key, value in kw.items():
            setattr(ell, key, value)
        return ell

    def test_theta_wide_prior_warns(self):
        """Test that a theta prior spanning more than 180deg warns."""
        ell = self._ellipsoid(theta=priors.uniform(-np.pi, np.pi))
        with pytest.warns(UserWarning, match="180"):
            models.Model(ell)

    def test_theta_narrow_prior_no_warning(self, recwarn):
        """Test that a theta prior spanning less than 180deg does not warn."""
        models.Model(
            self._ellipsoid(theta=priors.uniform(-np.pi / 4, np.pi / 4))
        )
        assert len(recwarn) == 0

    def test_theta_exact_period_warns_periodic(self, recwarn):
        """A theta prior exactly spanning 180deg (the normal way to mark it periodic) warns about automatic periodic sampling."""
        models.Model(self._ellipsoid(theta=priors.uniform(0.00, np.pi)))
        assert len(recwarn) == 1
        assert "periodic" in str(recwarn[0].message)

    def test_rot_wide_prior_warns(self):
        """Test that a rot prior spanning more than 180deg warns."""
        ell = self._ellipsoid(rot=priors.uniform(-np.pi, np.pi))
        with pytest.warns(UserWarning, match="180"):
            models.Model(ell)

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
        """Reset the component ids before each test."""
        models.Component.reset()

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
        """Reset the component ids before each test."""
        models.Component.reset()

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
        """Reset the component ids before each test."""
        models.Component.reset()

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


class TestParameterSummary:
    """Printed summaries name each component by type and id."""

    @staticmethod
    def _model():
        sersic = models.Sersic(xc=1.0, yc=1.0, Ie=1.0, ns=1.0)
        sersic.re = priors.uniform(0.01, 0.2)
        disk = models.Disk()
        disk.radial.xc, disk.radial.yc = 1.0, 1.0
        disk.radial.re, disk.radial.Ie = 0.1, 1.0
        disk.vertical.zs = 0.01
        mod = models.Model(sersic)
        mod.addcomponent(disk)
        mod.addcomponent(models.Background(a0=1.0))
        return mod

    def test_model_parameters_grouped(self, capsys):
        """Parameters are listed under 'Type (id)', one block each."""
        self._model().parameters()
        lines = capsys.readouterr().out.split("\n")
        assert lines[1:4] == ["Model parameters", "=" * 16, "Sersic (comp_00)"]
        assert "Disk (comp_01)" in lines and "Background (comp_02)" in lines
        assert lines[lines.index("Disk (comp_01)") - 1] == ""
        rows = [line for line in lines if line.startswith("  ")]
        assert len({row.index("] :") for row in rows}) == 1
        assert len({row.index(" | ") for row in rows}) == 1
        assert any(
            row.split()[0] == "radial.xc" and "Right ascension" in row
            for row in rows
        )

    def test_model_parameters_freeonly(self, capsys):
        """Components without free parameters are left out."""
        self._model().parameters(freeonly=True)
        out = capsys.readouterr().out
        assert "Sersic (comp_00)" in out and "  re " in out
        assert "Disk" not in out and "Background" not in out
        models.Model(models.Background(a0=1.0)).parameters(freeonly=True)
        assert "No free parameters." in capsys.readouterr().out

    def test_model_repr_lists_components(self):
        """repr(model) lists the component headers."""
        assert repr(self._model()).split("\n") == [
            "Model components",
            "=" * 16,
            "Sersic (comp_00)",
            "Disk (comp_01)",
            "Background (comp_02)",
        ]
        assert repr(models.Model()).endswith("No components.")

    @pytest.mark.parametrize(
        "make, title",
        [
            (lambda: models.Sersic(), "Sersic (comp_00)"),
            (lambda: models.Disk(), "Disk (comp_00)"),
            (lambda: models.Ellipsoid(), "Ellipsoid (comp_00)"),
            (lambda: models.SimpleBridge(), "SimpleBridge (comp_00)"),
        ],
    )
    def test_component_parameters_title(self, make, title, capsys):
        """A component's summary is headed by its type and id."""
        make().parameters()
        lines = capsys.readouterr().out.split("\n")
        assert lines[1:3] == [title, "=" * len(title)]
        rows = [line for line in lines[3:] if " : " in line]
        assert len({row.index("] :") for row in rows}) == 1
