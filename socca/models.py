from .utils import *
import types

# Support utilities
# ========================================================
# Print available models
# --------------------------------------------------------
def zoo():
    models = ['Sersic',
              'Exponential',
              'PolyExponential','PolyExpoRefact',
              'ModExponential',
              'Point','Background']
    for mi, m in enumerate(models):
        print(m)


# Models
# ========================================================
# General model structure
# --------------------------------------------------------
class Model:
    def __init__(self,prof=None,positive=None):
        self.ncomp  = 0
        self.priors = {}
        self.params = []
        self.paridx = []
        self.profile = []
        self.positive = []
        self.tied = []
        self.type = []

        if prof is not None:
            self.add_component(prof,positive)

    def add_component(self,prof,positive=None):
        self.type.append(prof.__class__.__name__)
        self.positive.append(prof.positive if positive is None else positive)
        for pi, p in enumerate(prof.parlist()):
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
        
        
# General composable term
# --------------------------------------------------------
class Component:
    def __init__(self):
        self.okeys = ['positive','units']
        
    def print_params(self):
        keyout =[]
        for key in self.__dict__.keys():
            if key not in self.okeys and key!='okeys':
                keyout.append(key)

        maxlen = np.max(np.array([len(f'{key} [{self.units[key]}]') for key in keyout]))
    
        for key in keyout:
            keylen = maxlen-len(f' [{self.units[key]}]')
            print(f'{key:<{keylen}} [{self.units[key]}] : {self.__dict__[key]}')
            
    def parlist(self):
        return [key for key in self.__dict__.keys() if key not in self.okeys]
        
    def addpar(self,name,value=None):
        setattr(self,name,value)


# General profile class
# --------------------------------------------------------
class Profile(Component):
    def __init__(self,**kwargs):
        super().__init__()
        self.xc = kwargs.get('xc',None)
        self.yc = kwargs.get('yc',None)
        
        self.theta = kwargs.get('theta',None)
        self.e     = kwargs.get('e',    None)
        self.cbox  = kwargs.get('cbox', None)

        self.positive = kwargs.get('positive',False)
        self.units = dict(xc='deg',yc='deg',theta='rad',e='',cbox='')

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
        self.Ie = kwargs.get('Ie', None)
        self.ns = kwargs.get('ns', None)

        self.units.update(dict(re='deg',Ie='image',ns=''))

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

        self.units.update(dict(re='deg',Ie='image'))
                                
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

        self.units.update(dict(rc='deg'))
        self.units.update({f'c{ci}':'' for ci in range(1,5)})

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

        self.units.update(dict(rc='deg'))
        self.units.update({f'I{ci}':'image' for ci in range(1,5)})

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
        self.rm = kwargs.get('rm', None)
        self.alpha = kwargs.get('alpha', None)

        self.units.update(dict(rm='deg',alpha=''))

    @staticmethod
    @jax.jit
    def profile(r,Ie,re,rm,alpha):
        return Ie*jp.exp(-r/re)*(1.00+r/rm)**alpha

# Thick disk model
# --------------------------------------------------------
class ThickDisk(Profile):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        delattr(self,'cbox')
        delattr(self,'e')

# Point source
# --------------------------------------------------------
class Point(Component):
    def __init__(self,**kwargs):
        self.xc = kwargs.get('xc',None)
        self.yc = kwargs.get('yc',None)
        self.Ic = kwargs.get('Ic',None)

        self.positive = kwargs.get('positive',False)
        self.units = dict(xc='deg',yc='deg',Ic='image')

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
    

# Bakcground
# ---------------------------------------------------
class Background(Component):
    def __init__(self,**kwargs):
        self.positive = kwargs.get('positive',False)
        self.rc = kwargs.get('rc',1.00/60.00/60.00)
        self.a0 = kwargs.get('a0',None)
        self.a1 = kwargs.get('a1',None)
        self.a2 = kwargs.get('a2',None)
        self.a3 = kwargs.get('a3',None)
        self.a4 = kwargs.get('a4',None)
        self.a5 = kwargs.get('a5',None)
        self.a6 = kwargs.get('a6',None)
        self.a7 = kwargs.get('a7',None)
        self.a8 = kwargs.get('a8',None)
        self.a9 = kwargs.get('a9',None)

        self.units = dict(rc='deg')
        self.units.update({f'a{ci}':'' for ci in range(10)})

    @staticmethod
    def profile(x,y,a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,rc):
        xc, yc = x/rc, y/rc
        factor  = a0
        factor += a1*xc + a2*yc 
        factor += a3*xc*yc + a4*xc**2 + a5*yc**2 
        factor += a6*xc**3 + a7*yc**3 + a8*xc**2*yc + a9*xc*yc**2
        return factor

    def getmap(self,img):
        xgrid, ygrid = img.getgrid()
        kwarg = {key: eval(f'self.{key}') for key in ['a0','a1','a2','a3','a4','a5','a6','a7','a8','a9','rc']}
        
        for key in kwarg.keys():
            if isinstance(kwarg[key], scipy.stats._distn_infrastructure.rv_continuous_frozen):
                raise ValueError('Priors must be fixed values, not distributions.')
            if kwarg[key] is None:
                raise ValueError(f'keyword {key} is set to None. Please provide a valid value.')
        
        return self.profile(xgrid,ygrid,**kwarg)