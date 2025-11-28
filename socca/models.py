from .utils import *
import types

# Support utilities
# ========================================================
# Print available models
# --------------------------------------------------------
def zoo():
    models = ['Beta',
              'Sersic',
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
            self.addcomp(prof,positive)

# Add a new component to the model
# --------------------------------------------------------
    def addcomponent(self,prof,positive=None):
        self.type.append(prof.__class__.__name__)
        self.positive.append(prof.positive if positive is None else positive)
        for pi, p in enumerate(prof.parlist()):
            par = eval(f'prof.{p}')
            self.params.append( f'src_{self.ncomp:02d}_{p}')
            self.priors.update({f'src_{self.ncomp:02d}_{p}': par})
            if isinstance(par,numpyro.distributions.Distribution):
                self.paridx.append(len(self.params)-1)
            
            if isinstance(par,(types.LambdaType, types.FunctionType)):
                self.tied.append(True)
            else:
                self.tied.append(False)
        self.profile.append(prof.profile)
        self.ncomp += 1

# Get the model map
# --------------------------------------------------------
    def getmap(self,img,pp,doresp=False,doexp=False):
        pars = {}
        for ki, key in enumerate(self.params):
            if isinstance(self.priors[key],(float,int)):
                pars[key] = self.priors[key]
            elif isinstance(self.priors[key],numpyro.distributions.Distribution):
                pars[key], pp = pp[0], pp[1:]
        
        for ki, key in enumerate(self.params):
            if self.tied[ki]:
                kwarg = list(inspect.signature(self.priors[key]).parameters.keys())
                kwarg = {k: pars[k] for k in kwarg}
                pars[key] = self.priors[key](**kwarg)
                del kwarg

        mbkg = jp.zeros(img.data.shape)
        mraw = jp.zeros(img.data.shape)
        mpts = jp.fft.rfft2(mraw,s=img.data.shape)

        mneg = jp.zeros(img.data.shape)

        for nc in range(self.ncomp):
            kwarg = {key.replace(f'src_{nc:02d}_',''): pars[key] for key in self.params \
                  if key.startswith(f'src_{nc:02d}') and \
                     key.replace(f'src_{nc:02d}_','') in list(inspect.signature(self.profile[nc]).parameters.keys())}

            if self.type[nc]=='Point':
                uphase, vphase = img.fft.shift(kwarg['xc'],kwarg['yc'])
                
                mone = kwarg['Ic']*img.fft.pulse*jp.exp(-(uphase+vphase))
                if self.positive[nc]: mneg = jp.where(mone<0.00,1.00,mneg)
                
                if doresp:
                    xpts = (kwarg['xc']-img.hdu.header['CRVAL1'])/jp.abs(img.hdu.header['CDELT1'])
                    ypts = (kwarg['yc']-img.hdu.header['CRVAL2'])/jp.abs(img.hdu.header['CDELT2'])

                    xpts = img.hdu.header['CRPIX1']-1+xpts*jp.cos(jp.deg2rad(img.hdu.header['CRVAL2']))
                    ypts = img.hdu.header['CRPIX2']-1+ypts
                    mone *= jax.scipy.ndimage.map_coordinates(img.resp,[jp.array([ypts]),jp.array([xpts])],order=1,mode='nearest')[0]
                
                mpts += mone.copy(); del mone

            elif self.type[nc]=='Background':
                yr = jp.mean(img.grid.y,axis=0)-img.hdu.header['CRVAL2']
                xr = jp.mean(img.grid.x,axis=0)-img.hdu.header['CRVAL1']
                xr = xr*jp.cos(jp.deg2rad(img.hdu.header['CRVAL2']))

                mone = self.profile[nc](xr,yr,**kwarg)
                if self.positive[nc]: mneg = jp.where(mone<0.00,1.00,mneg)

                mbkg += mone.copy(); del mone
                
            elif 'Thick' in self.type[nc]:
                pass
            else:
                rgrid = img.getgrid(pars[f'src_{nc:02d}_xc'],
                                    pars[f'src_{nc:02d}_yc'],
                                    pars[f'src_{nc:02d}_theta'],
                                    pars[f'src_{nc:02d}_e'],
                                    pars[f'src_{nc:02d}_cbox'])

                mone = self.profile[nc](rgrid,**kwarg)
                mone = jp.mean(mone,axis=0)
                if self.positive[nc]: mneg = jp.where(mone<0.00,1.00,mneg)

                mraw += mone.copy(); del mone
        
        msmo = mraw.copy()
        if doresp: msmo *= img.resp

        if img.psf is not None:
            msmo = (mpts+jp.fft.rfft2(jp.fft.fftshift(msmo),s=img.data.shape))*img.psf_fft
            msmo = jp.fft.ifftshift(jp.fft.irfft2(msmo,s=img.data.shape)).real
    
        mpts = jp.fft.ifftshift(jp.fft.irfft2(mpts,s=img.data.shape)).real
        
        if img.psf is None:
            msmo = msmo+mpts

        msmo = msmo+mbkg
        if doexp: 
            msmo *= img.exp
            mbkg *= img.exp

        return mraw+mpts, msmo, mbkg, mneg


# General composable term
# --------------------------------------------------------
class Component:
    idcls = 0
    def __init__(self):
        self.id = f'src_{type(self).idcls:02d}'
        type(self).idcls += 1

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
        
        self.theta = kwargs.get('theta',0.00)
        self.e     = kwargs.get('e',    0.00)
        self.cbox  = kwargs.get('cbox', 0.00)

        self.positive = kwargs.get('positive',False)
        self.units = dict(xc='deg',yc='deg',theta='rad',e='',cbox='')

    @abstractmethod
    def profile(self,r):
        pass

    def getmap(self,img,convolve=False):
        rgrid = img.getgrid(self.xc,self.yc,self.theta,self.e)
        kwarg = {key: getattr(self,key) for key in list(inspect.signature(self.profile).parameters.keys()) if key!='r'}
        
        for key in kwarg.keys():
            if isinstance(kwarg[key],numpyro.distributions.Distribution):
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

# Beta profile
# --------------------------------------------------------
class Beta(Profile):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.rc = kwargs.get('rc', None)
        self.Ic = kwargs.get('Ic', None)
        self.beta = kwargs.get('beta', None)

        self.units.update(dict(rc='deg',beta='',Ic='image'))

    @staticmethod
    @jax.jit
    def profile(r,Ic,rc,beta):
        return Ic*jp.power(1.00+(r/rc)**2,-beta)

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
        kwargs = {key: getattr(self,key) for key in ['xc','yc','theta','e','Ie','re','rc']}
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
        super().__init__()

        self.xc = kwargs.get('xc',None)
        self.yc = kwargs.get('yc',None)
        self.Ic = kwargs.get('Ic',None)

        self.positive = kwargs.get('positive',False)
        self.units = dict(xc='deg',yc='deg',Ic='image')

    @staticmethod
    def profile(xc,yc,Ic):
        pass
    
    def getmap(self,img,convolve=False):
        kwarg = {key: getattr(self,key) for key in ['xc','yc','Ic']}
        
        for key in kwarg.keys():
            if isinstance(kwarg[key],numpyro.distributions.Distribution):
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
        super().__init__()

        self.positive = kwargs.get('positive',False)
        self.rc = kwargs.get('rc',1.00/60.00/60.00)
        self.a0 = kwargs.get('a0',None)
        self.a1x = kwargs.get('a1x',0.00)
        self.a1y = kwargs.get('a1y',0.00)
        self.a2xx  = kwargs.get('a2xx', 0.00)
        self.a2xy  = kwargs.get('a2xy', 0.00)
        self.a2yy  = kwargs.get('a2yy', 0.00)
        self.a3xxx = kwargs.get('a3xxx',0.00)
        self.a3xxy = kwargs.get('a3xxy',0.00)
        self.a3xyy = kwargs.get('a3xyy',0.00)
        self.a3yyy = kwargs.get('a3yyy',0.00)

        self.units = dict(rc='deg')
        self.units.update({f'a{ci}':'' for ci in range(10)})

    @staticmethod
    def profile(x,y,a0,a1x,a1y,a2xx,a2xy,a2yy,a3xxx,a3xxy,a3xyy,a3yyy,rc):
        xc, yc = x/rc, y/rc
        factor  = a0
        factor += a1x*xc + a1y*yc
        factor += a2xy*xc*yc + a2xx*xc**2 + a2yy*yc**2
        factor += a3xxx*xc**3 + a3yyy*yc**3 + a3xxy*xc**2*yc + a3xyy*xc*yc**2
        return factor

    def getmap(self,img):
        xc, yc = self.img.wcs.crval
        xgrid, ygrid = img.getgrid(xc,yc,0.00,0.00)
        kwarg = {key: getattr(self,key) for key in ['a0','a1x','a1y','a2xx','a2xy','a2yy','a3xxx','a3xxy','a3xyy','a3yyy','rc']}
        
        for key in kwarg.keys():
            if isinstance(kwarg[key], numpyro.distributions.Distribution):
                raise ValueError('Priors must be fixed values, not distributions.')
            if kwarg[key] is None:
                raise ValueError(f'keyword {key} is set to None. Please provide a valid value.')
        
        return self.profile(xgrid,ygrid,**kwarg)