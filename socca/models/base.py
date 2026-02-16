"""Base component class for model profiles."""

import types

import numpy as np
import numpyro.distributions

from . import config


class Component:
    """
    Base class for all model components.

    This abstract base class provides the infrastructure for model components,
    including parameter management, unique identification, and formatted output.
    All specific profile types (Beta, Sersic, Point, etc.) inherit from this class.

    Attributes
    ----------
    idcls : int (class variable)
        Class-level counter for assigning unique IDs to component instances.
    id : str
        Unique identifier for this component instance (e.g., 'comp_00', 'comp_01').
    positive : bool
        Whether to enforce positivity constraint on this component.
    hyper : list of str
        Names of hyperparameters (not fitted, control behavior).
    units : dict
        Mapping from parameter names to their physical units.
    description : dict
        Mapping from parameter names to their descriptions.
    """

    idcls = 0

    def __init__(self, **kwargs):
        """
        Initialize a component with unique ID and parameter tracking structures.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments for component initialization.

            positive : bool, optional
                Override default positivity constraint.

        Notes
        -----
        The component ID is automatically assigned and incremented for each
        new instance. This ensures unique identification when multiple components
        are combined in a model.
        """
        self.id = f"comp_{Component.idcls:02d}"
        Component.idcls += 1

        self.positive = kwargs.get("positive", config.Component.positive)
        self.hyper = []
        self.units = {}
        self.description = {}
        self._initialized = False

    def __setattr__(self, name, value):
        """Prevent adding undefined attributes after initialization."""
        if getattr(self, "_initialized", False) and not hasattr(self, name):
            raise AttributeError(
                f"Cannot add new attribute '{name}' to "
                f"{self.__class__.__name__}. "
                f"Use addparameter() to add new parameters."
            )
        super().__setattr__(name, value)

    #   Print model parameters
    #   --------------------------------------------------------
    def parameters(self):
        """
        Print a formatted table of component parameters and hyperparameters.

        Displays all parameters (excluding reserved keys) with their values,
        units, and descriptions in a human-readable format. Separates regular
        parameters from hyperparameters.

        Notes
        -----
        Output format:

        Model parameters
        ================
        parameter_name [units] : value | description

        Hyperparameters
        ===============
        hyperparam_name [units] : value | description

        - Parameters are model quantities to be fitted or fixed
        - Hyperparameters control computation but are not fitted
        - Values are shown in scientific notation (e.g., 1.5000E-02)
        - None values are displayed as "None"

        Examples
        --------
        >>> from socca.models import Beta
        >>> beta = Beta(xc=180.5, yc=45.2, rc=0.01, Ic=100, beta=0.5)
        >>> beta.parameters()
        Model parameters
        ================
        xc    [deg]    : 1.8050E+02 | Right ascension of centroid
        yc    [deg]    : 4.5200E+01 | Declination of centroid
        rc    [deg]    : 1.0000E-02 | Core radius
        Ic    [image]  : 1.0000E+02 | Central surface brightness
        beta  []       : 5.0000E-01 | Slope parameter
        theta [rad]    : 0.0000E+00 | Position angle (east from north)
        e     []       : 0.0000E+00 | Projected axis ratio
        cbox  []       : 0.0000E+00 | Projected boxiness
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
                keypar = getattr(self, key)

                if keypar is None:
                    keyval = None
                elif isinstance(keypar, numpyro.distributions.Distribution):
                    keyval = f"Distribution: {keypar.__class__.__name__}"
                elif isinstance(
                    keypar, (types.LambdaType, types.FunctionType)
                ):
                    keyval = "Tied keyparameter"
                else:
                    keyval = f"{keypar:.4E}"

                print(
                    f"{key:<{keylen}} [{self.units[key]}] : "
                    + f"{keyval}".ljust(10)
                    + f" | {self.description[key]}"
                )

            if len(self.hyper) > 0:
                print("\nHyperparameters")
                print("=" * 15)
                for key in self.hyper:
                    keylen = maxlen - len(f" [{self.units[key]}]")
                    kvalue = getattr(self, key)
                    kvalue = None if kvalue is None else f"{kvalue:.4E}"
                    print(
                        f"{key:<{keylen}} [{self.units[key]}] : "
                        + f"{kvalue}".ljust(10)
                        + f" | {self.description[key]}"
                    )

        else:
            print("No parameters defined.")

    def parlist(self):
        """
        Return a list of parameter names for this component.

        Returns
        -------
        list of str
            Names of all model parameters, as defined in the units dict.

        Notes
        -----
        This method is used internally by Model.addcomponent() to register
        parameters when adding a component to a composite model.

        Examples
        --------
        >>> from socca.models import Beta
        >>> beta = Beta(xc=180.5, yc=45.2)
        >>> beta.parlist()
        ['xc', 'yc', 'theta', 'e', 'cbox', 'rc', 'Ic', 'beta']
        """
        return list(self.units.keys())

    def addparameter(self, name, value=None, units="", description=""):
        """
        Add a new parameter to the component with metadata.

        Parameters
        ----------
        name : str
            Name of the parameter to add.
        value : float, int, Distribution, or callable, optional
            Parameter value. Can be a fixed value, a numpyro prior distribution,
            or a function for tied parameters. Default is None.
        units : str, optional
            Physical units of the parameter (e.g., 'deg', 'rad', 'image').
            Default is empty string.
        description : str, optional
            Human-readable description of the parameter. Default is empty string.

        Notes
        -----
        This method dynamically adds a parameter attribute to the component
        instance and registers its metadata (units and description).

        Examples
        --------
        >>> from socca.models import Component
        >>> comp = Component()
        >>> comp.addparameter('amplitude', value=1.0, units='Jy', description='Source flux')
        >>> comp.amplitude
        1.0
        >>> comp.units['amplitude']
        'Jy'
        """
        self._initialized = False
        setattr(self, name, value)
        self._initialized = True
        self.units.update({name: units})
        self.description.update({name: description})
