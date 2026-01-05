from dataclasses import dataclass


@dataclass(frozen=True)
class Component:
    positive : bool  = False

@dataclass(frozen=True)
class Profile:
    xc       : float = None
    yc       : float = None
    theta    : float = 0.00
    e        : float = 0.00
    cbox     : float = 0.00
    

@dataclass(frozen=True)
class Beta:
    rc   : float = None
    Ic   : float = None
    beta : float = 0.70*1.50-0.50

@dataclass(frozen=True)
class gNFW:
    rc    : float = None
    Ic    : float = None
    alpha : float = 1.0510E+00
    beta  : float = 5.4905E+00
    gamma : float = 3.0810E-01

@dataclass(frozen=True)
class Sersic:
    re : float = None
    Ie : float = None
    ns : float = 0.50


@dataclass(frozen=True)
class Exponential:
    rs : float = None
    Is : float = None


@dataclass(frozen=True)
class PolyExponential:
    ck : float = 0.00
    rc : float = 1.00/3.60E+03
    

@dataclass(frozen=True)
class PolyExpoRefact:
    I1 : float = 0.00
    I2 : float = 0.00
    I3 : float = 0.00
    I4 : float = 0.00
    rc : float = 1.00/3.60E+03
    

@dataclass(frozen=True)
class ModExponential:
    rm    : float = None
    alpha : float = None


@dataclass(frozen=True)
class Point:
    xc: float = None
    yc: float = None
    Ic: float = None


@dataclass(frozen=True)
class Background:
    positive : bool = False
    rs       : float = 1.00/60.00/60.00
    a0       : float = None
    a1x      : float = 0.00
    a1y      : float = 0.00
    a2xx     : float = 0.00
    a2xy     : float = 0.00
    a2yy     : float = 0.00
    a3xxx    : float = 0.00
    a3xxy    : float = 0.00
    a3xyy    : float = 0.00
    a3yyy    : float = 0.00


@dataclass(frozen=True)
class Height:
    inc      : float = 0.00
    losdepth : float = 10.00/60.00/60.00
    losbins  : int   = 200

@dataclass(frozen=True)
class HyperSecantHeight:
    zs    : float = None
    alpha : float = 2.00

@dataclass(frozen=True)
class ExponentialHeight:
    zs : float = None