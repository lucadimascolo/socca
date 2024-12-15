from .utils import *
import types

# Models
# ========================================================

# General model structure
# --------------------------------------------------------
class Model:
    def __init__(self):
        self.ncomp  = 0
        self.priors = {}
        self.params = []
        self.paridx = []
        self.profile = []
        self.positive = []
        self.tied = []
        self.type = []

    def addcomponent(self,prof,positive=None):
        self.type.append(prof.__class__.__name__)
        self.positive.append(prof.positive if positive is None else positive)
        for pi, p in enumerate(prof.listpars()):
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

        self.positive = kwargs.get('positive',True)

    def listpars(self):
        okeys = ['positive']
        return [key for key in self.__dict__.keys() if key not in okeys]
        
    def addpar(self,name,value=None):
        setattr(self,name,value)

    @abstractmethod
    def profile(self,r):
        pass

    def getmap(self,img,convolve=False):
        rgrid = img.getgrid(self.xc,self.yc,self.theta,self.e)
        kwarg = {key: eval(f'self.{key}') for key in list(inspect.signature(self.profile).parameters.keys()) if key!='r'}
        
        for key in kwarg.keys():
            if isinstance(kwarg[key],scipy.stats._distn_infrastructure.rv_continuous_frozen):
                raise ValueError('Priors must be fixed values, not distributions.')
            if kwarg[key] is None:
                raise ValueError(f'keyword {key} is set to None. Please provide a valid value.')
            
        mgrid = self.profile(rgrid,**kwarg)

        if convolve:
            if img.psf is None:
                 warnings.warn('No PSF defined, so no convolution will be performed.')
            else:
                mgrid = jp.fft.rfft2(jp.fft.fftshift(mgrid),s=img.data.shape)*img.psf_fft
                mgrid = jp.fft.ifftshift(jp.fft.irfft2(mgrid,s=img.data.shape)).real
        return mgrid

    def refactor(self):
        warnings.warn('Nothing to refactor here.')
        return self.__class__(**self.__dict__)

# Sersic profile
# --------------------------------------------------------
from scipy.special import gammaincinv

n_ = np.linspace(0.25,10.00,1000)
b_ = gammaincinv(2.00*n_,0.5)

n_ = jp.array(n_)
b_ = jp.array(b_)
class Sersic(Profile):
    def __init__(self,**kwargs):
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
    

# Exponential profile
# --------------------------------------------------------
class Exponential(Profile):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.re = kwargs.get('re', None)
        self.Ie = kwargs.get('Ie', None)

    @staticmethod
    @jax.jit
    def profile(r,Ie,re):
        return Ie*jp.exp(-r/re)


# PolyExponential
# Mancera Piña et al., A&A, 689, A344 (2024)
# --------------------------------------------------------
class PolyExponential(Exponential):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.c1 = kwargs.get('c1',0.00)
        self.c2 = kwargs.get('c2',0.00)
        self.c3 = kwargs.get('c3',0.00)
        self.c4 = kwargs.get('c4',0.00)
        self.rc = kwargs.get('rc',1.00/3.60E+03)

    @staticmethod
    @jax.jit
    def profile(r,Ie,re,c1,c2,c3,c4,rc):
        factor = 1.00+c1*(r/rc)+c2*((r/rc)**2)+c3*((r/rc)**3)+c4*((r/rc)**4)
        return factor*Ie*jp.exp(-r/re)


# PolyExponential
# Mancera Piña et al., A&A, 689, A344 (2024)
# --------------------------------------------------------
class PolyExpoRefact(Exponential):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.I1 = kwargs.get('I1',0.00)
        self.I2 = kwargs.get('I2',0.00)
        self.I3 = kwargs.get('I3',0.00)
        self.I4 = kwargs.get('I4',0.00)
        self.rc = kwargs.get('rc',1.00/3.60E+03)

    @staticmethod
    @jax.jit
    def profile(r,Ie,re,I1,I2,I3,I4,rc):
        factor = Ie*jp.exp(-r/re)
        for ci in range(1,5):
            factor += eval(f'I{ci}')*jp.exp(-r/re)*((r/rc)**ci)
        return factor
    
    def refactor(self):
        kwargs = {key: eval(f'self.{key}') for key in ['xc','yc','theta','e','Ie','re','rc']}
        for ci in range(1,5): kwargs.update({f'c{ci}': eval(f'I{ci}')/kwargs['Ie']})

        return PolyExponential(**kwargs)


# Modified Exponential
# Mancera Piña et al., A&A, 689, A344 (2024)
# --------------------------------------------------------
class ModExponential(Exponential):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.alpha = kwargs.get('alpha', None)

    @staticmethod
    @jax.jit
    def profile(r,Ie,re,rm,alpha):
        return Ie*jp.exp(-r/re)*(1.00+r/rm)**alpha


# Point source
# --------------------------------------------------------
class Point:
    def __init__(self,**kwargs):
        self.xc = kwargs.get('xc',None)
        self.yc = kwargs.get('yc',None)
        self.Ic = kwargs.get('Ic',None)

    @staticmethod
    def profile(xc,yc,Ic):
        pass
    
    def getmap(self,img,convolve=False):
        kwarg = {key: eval(f'self.{key}') for key in ['xc','yc','Ic']}
        
        for key in kwarg.keys():
            if isinstance(kwarg[key],scipy.stats._distn_infrastructure.rv_continuous_frozen):
                raise ValueError('Priors must be fixed values, not distributions.')
            if kwarg[key] is None:
                raise ValueError(f'keyword {key} is set to None. Please provide a valid value.')
            
        uphase, vphase = img.fft.shift(self.xc,self.yc)
        mgrid = self.Ic*img.fft.pulse*jp.exp(-(uphase+vphase))

        if convolve:
            if img.psf is None:
                 warnings.warn('No PSF defined, so no convolution will be performed.')
            else:
                mgrid = mgrid*img.psf_fft
            mgrid = jp.fft.ifftshift(jp.fft.irfft2(mgrid,s=img.data.shape)).real
        return mgrid
    