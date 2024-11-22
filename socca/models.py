from .utils import *
import types

# Models
# ========================================================

class Model:
    def __init__(self):
        self.ncomp  = 0
        self.priors = {}
        self.params = []
        self.paridx = []
        self.profile = []
        self.tied = []

    def addcomponent(self,prof):
        for pi, p in enumerate(prof.__dict__.keys()):
            par = eval(f'prof.{p}')
            self.params.append( f'src_{self.ncomp:02d}_{p}')
            self.priors.update({f'src_{self.ncomp:02d}_{p}': par})
            if isinstance(par,scipy.stats._distn_infrastructure.rv_continuous_frozen):
                self.paridx.append(len(self.params)-1)
            
            if isinstance(par,(types.LambdaType, types.FunctionType)):
                self.tied.append(True)
            else:
                self.tied.append(False)

        self.profile.append(prof.profile)
        self.ncomp += 1

# General profile class
# --------------------------------------------------------
class Profile:
    def __init__(self,**kwargs):
        self.xc = kwargs.get('xc',None)
        self.yc = kwargs.get('yc',None)
        
        self.theta = kwargs.get('theta',None)
        self.e = kwargs.get('e',None)
    
    def addpar(self,name,value=None):
        setattr(self,name,value)

    @abstractmethod
    def profile(self,r):
        pass

# Sersic profile
# --------------------------------------------------------
from scipy.special import gammaincinv

n_ = np.linspace(0.25,10.00,1000)
b_ = gammaincinv(2.00*n_,0.5)

n_ = jp.array(n_)
b_ = jp.array(b_)
class Sersic(Profile):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.re = kwargs.get('re', None)
        self.ns = kwargs.get('ns', None)
        self.Ie = kwargs.get('Ie', None)

    @staticmethod
    @jax.jit
    def profile(r,Ie,re,ns):
        bn = jp.interp(ns,n_,b_)
        se = jp.power(r/re,1.00/ns)-1.00
        return Ie*jp.exp(-bn*se)
    