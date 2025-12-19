from xxlimited import foo
from ..utils import *
import types

from . import config

from quadax import quadgk

# Support utilities
# ========================================================
# Print available models
# --------------------------------------------------------
def zoo():
    models = ['Beta',
              'gNFW',
              'Sersic',
              'Exponential',
              'PolyExponential','PolyExpoRefact',
              'ModExponential',
              'Point','Background',
              'Disk']
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
        self.positive = []
        self.profile = []
        self.gridder = []
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

            if par is None:
                raise ValueError(f'Parameter {p} in component {self.ncomp:02d} is set to None. \
                                   Please provide a valid value or prior.')

            self.params.append( f'src_{self.ncomp:02d}_{p}')
            self.priors.update({f'src_{self.ncomp:02d}_{p}': par})
            if isinstance(par,numpyro.distributions.Distribution):
                self.paridx.append(len(self.params)-1)
            
            if isinstance(par,(types.LambdaType, types.FunctionType)):
                self.tied.append(True)
            else:
                self.tied.append(False)
        self.profile.append(prof.profile)
        self.gridder.append(prof.getgrid)
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
            if self.type[nc]=='Disk':
                kwarg = {key.replace(f'src_{nc:02d}_radial.','r_'): pars[key] for key in self.params \
                         if key.startswith(f'src_{nc:02d}') and \
                            key.replace(f'src_{nc:02d}_radial.','r_') in list(inspect.signature(self.profile[nc]).parameters.keys())}
                kwarg.update({key.replace(f'src_{nc:02d}_vertical.','z_'): pars[key] for key in self.params \
                             if key.startswith(f'src_{nc:02d}') and \
                                key.replace(f'src_{nc:02d}_vertical.','z_') in list(inspect.signature(self.profile[nc]).parameters.keys())})
            
                rcube, zcube = self.gridder[nc](img.grid,
                                                pars[f'src_{nc:02d}_radial.xc'],
                                                pars[f'src_{nc:02d}_radial.yc'],
                                                pars[f'src_{nc:02d}_vertical.zext'],
                                                pars[f'src_{nc:02d}_vertical.zsize'],
                                                pars[f'src_{nc:02d}_radial.theta'],
                                                pars[f'src_{nc:02d}_vertical.inc'])

                dx = 2.00*pars[f'src_{nc:02d}_vertical.zext']/(pars[f'src_{nc:02d}_vertical.zsize']-1)
                mone = self.profile[nc](rcube,zcube,**kwarg)
                mone = jp.trapezoid(mone,dx=dx,axis=1)
                mone = jp.mean(mone,axis=0)
                mraw += mone.copy(); del mone
            else:       
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
        
                else:
                    rgrid = self.gridder[nc](img.grid,
                                             pars[f'src_{nc:02d}_xc'],
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
    def __init__(self,**kwargs):
        self.id = f'src_{type(self).idcls:02d}'
        type(self).idcls += 1

        self.positive = kwargs.get('positive',config.Component.positive)
        self.okeys = ['id','positive','units','description']
        self.units = {}

        self.description = {}

#   Print model parameters  
#   --------------------------------------------------------
    def parameters(self):
        keyout =[]
        for key in self.__dict__.keys():
            if key not in self.okeys and key!='okeys':
                keyout.append(key)

        if len(keyout)>0:
            maxlen = np.max(np.array([len(f'{key} [{self.units[key]}]') for key in keyout]))
        
            for key in keyout:
                keylen = maxlen-len(f' [{self.units[key]}]')
                keyval = None if self.__dict__[key] is None else f'{self.__dict__[key]:.4E}'
                print(f'{key:<{keylen}} [{self.units[key]}] : '+f'{keyval}'.ljust(10) + f' | {self.description[key]}')
        else:
            print('No parameters defined.')

    def parlist(self):
        return [key for key in self.__dict__.keys() if key not in self.okeys and key!='okeys']
        
    def addpar(self,name,value=None):
        setattr(self,name,value)


# General profile class
# --------------------------------------------------------
class Profile(Component):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.xc = kwargs.get('xc',config.Profile.xc)
        self.yc = kwargs.get('yc',config.Profile.yc)
        
        self.theta = kwargs.get('theta',config.Profile.theta)
        self.e     = kwargs.get('e',    config.Profile.e)
        self.cbox  = kwargs.get('cbox', config.Profile.cbox)

        self.units.update(dict(xc='deg',yc='deg',theta='rad',e='',cbox=''))

        self.description.update(dict(xc = 'Right ascensio of centroid',
                                     yc = 'Declination of centroid',
                                  theta = 'Position angle (east from north)',
                                      e = 'Projected axis ratio',
                                   cbox = 'Projected boxiness'))

    @abstractmethod
    def profile(self,r):
        pass

    def getmap(self,img,convolve=False):
        kwarg = {key: getattr(self,key) for key in list(inspect.signature(self.profile).parameters.keys()) if key!='r'}

        rgrid = self.getgrid(img.grid,self.xc,self.yc,self.theta,self.e)

        for key in kwarg.keys():
            if isinstance(kwarg[key],numpyro.distributions.Distribution):
                raise ValueError('Priors must be fixed values, not distributions.')
            if kwarg[key] is None:
                raise ValueError(f'keyword {key} is set to None. Please provide a valid value.')
            
        mgrid = self.profile(rgrid,**kwarg)
        mgrid = jp.mean(mgrid,axis=0)
        
        if convolve:
            if img.psf is None:
                 warnings.warn('No PSF defined, so no convolution will be performed.')
            else:
                mgrid = jp.fft.rfft2(jp.fft.fftshift(mgrid),s=img.data.shape)*img.psf_fft
                mgrid = jp.fft.ifftshift(jp.fft.irfft2(mgrid,s=img.data.shape)).real
        return mgrid

    @staticmethod
    @partial(jax.jit,static_argnames=['grid'])
    def getgrid(grid,xc,yc,theta=0.00,e=0.00,cbox=0.00):
        sint = jp.sin(theta)
        cost = jp.cos(theta)

        xgrid = (-(grid.x-xc)*jp.cos(jp.deg2rad(yc))*sint-(grid.y-yc)*cost)
        ygrid = ( (grid.x-xc)*jp.cos(jp.deg2rad(yc))*cost-(grid.y-yc)*sint)
        
        xgrid = jp.abs(xgrid)**(cbox+2.00)
        ygrid = jp.abs(ygrid/(1.00-e))**(cbox+2.00)
        return jp.power(xgrid+ygrid,1.00/(cbox+2.00))
    
    def refactor(self):
        warnings.warn('Nothing to refactor here.')
        return self.__class__(**self.__dict__)


# Beta profile
# --------------------------------------------------------
class Beta(Profile):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.rc = kwargs.get('rc', config.Beta.rc)
        self.Ic = kwargs.get('Ic', config.Beta.Ic)
        self.beta = kwargs.get('beta', config.Beta.beta)

        self.units.update(dict(rc='deg',beta='',Ic='image'))

        self.description.update(dict(rc = 'Core radius',
                                     Ic = 'Central surface brightness',
                                   beta = 'Slope parameter'))
    @staticmethod
    @jax.jit
    def profile(r,Ic,rc,beta):
        return Ic*jp.power(1.00+(r/rc)**2,-beta)


# gNFW profile
# --------------------------------------------------------
class gNFW(Profile):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.rc = kwargs.get('rc', config.gNFW.rc)
        self.Ic = kwargs.get('Ic', config.gNFW.Ic)
        self.alpha = kwargs.get('alpha', config.gNFW.alpha)
        self.beta  = kwargs.get('beta',  config.gNFW.beta)
        self.gamma = kwargs.get('gamma', config.gNFW.gamma)

        self.rz  = kwargs.get('rz',jp.logspace(-7,2,1000))
        self.eps = kwargs.get('eps',1.00E-08)

        self.okeys.append('rz')
        self.okeys.append('eps')
        self.okeys.append('profile')

        self.units.update(dict(rc='deg',alpha='',beta='',gamma='',Ic='image'))

        self.description.update(dict(rc = 'Scale radius',
                                     Ic = 'Characteristic surface brightness',
                                  alpha = 'Intermediate slope',
                                   beta = 'Outer slope',
                                  gamma = 'Inner slope'))


        def _profile(r,Ic,rc,alpha,beta,gamma):
            return gNFW._profile(r,Ic,rc,alpha,beta,gamma,self.rz,self.eps)
        
        self.profile = jax.jit(_profile)

    @staticmethod
    def _profile(r,Ic,rc,alpha,beta,gamma,rz,eps=1.00E-08):

        def radial(u,alpha,beta,gamma):
            factor = 1.00+u**alpha
            factor = factor**((gamma-beta)/alpha)
            return factor/u**gamma  
              
        def integrand(u,uz):
            factor = radial(u,alpha,beta,gamma)
            return 2.00*factor*u/jp.sqrt(u**2-uz**2)

        def integrate(rzj):
            return quadgk(integrand,[rzj,jp.inf],args=(rzj,),epsabs=eps,epsrel=eps)[0]
        
        mz = Ic*jax.vmap(integrate)(rz)
        return jp.interp(r/rc,rz,mz)


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
        self.re = kwargs.get('re', config.Sersic.re)
        self.Ie = kwargs.get('Ie', config.Sersic.Ie)
        self.ns = kwargs.get('ns', config.Sersic.ns)

        self.units.update(dict(re='deg',Ie='image',ns=''))

        self.description.update(dict(re = 'Effective radius',
                                     Ie = 'Surface brightness at re',
                                     ns = 'Sersic index'))
        
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
        self.re = kwargs.get('re', config.Exponential.re)
        self.Ie = kwargs.get('Ie', config.Exponential.Ie)

        self.units.update(dict(re='deg',Ie='image'))
        self.description.update(dict(re = 'Scale radius',
                                     Ie = 'Surface brightness at re'))
                
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
        self.c1 = kwargs.get('c1',config.PolyExponential.c1)
        self.c2 = kwargs.get('c2',config.PolyExponential.c2)
        self.c3 = kwargs.get('c3',config.PolyExponential.c3)
        self.c4 = kwargs.get('c4',config.PolyExponential.c4)
        self.rc = kwargs.get('rc',config.PolyExponential.rc)

        self.units.update(dict(rc='deg'))
        self.units.update({f'c{ci}':'' for ci in range(1,5)})

        self.description.update(dict(c1 = 'Polynomial coefficient 1',
                                     c2 = 'Polynomial coefficient 2',
                                     c3 = 'Polynomial coefficient 3',
                                     c4 = 'Polynomial coefficient 4',
                                     rc = 'Reference radius for polynomial terms'))
        
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
        self.I1 = kwargs.get('I1',config.PolyExpoRefact.I1)
        self.I2 = kwargs.get('I2',config.PolyExpoRefact.I2)
        self.I3 = kwargs.get('I3',config.PolyExpoRefact.I3)
        self.I4 = kwargs.get('I4',config.PolyExpoRefact.I4)
        self.rc = kwargs.get('rc',config.PolyExpoRefact.rc)

        self.units.update(dict(rc='deg'))
        self.units.update({f'I{ci}':'image' for ci in range(1,5)})

        self.description.update(dict(I1 = 'Polynomial intensity coefficient 1',
                                     I2 = 'Polynomial intensity coefficient 2',
                                     I3 = 'Polynomial intensity coefficient 3',
                                     I4 = 'Polynomial intensity coefficient 4',
                                     rc = 'Reference radius for polynomial terms'))
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
        self.rm = kwargs.get('rm',config.ModExponential.rm)
        self.alpha = kwargs.get('alpha',config.ModExponential.alpha)

        self.units.update(dict(rm='deg',alpha=''))

        self.description.update(dict(rm = 'Modification radius',
                                  alpha = 'Modification exponent'))
        
    @staticmethod
    @jax.jit
    def profile(r,Ie,re,rm,alpha):
        return Ie*jp.exp(-r/re)*(1.00+r/rm)**alpha


# Point source
# --------------------------------------------------------
class Point(Component):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.xc = kwargs.get('xc',config.Point.xc)
        self.yc = kwargs.get('yc',config.Point.yc)
        self.Ic = kwargs.get('Ic',config.Point.Ic)

        self.units.update(dict(xc='deg',yc='deg',Ic='image'))

        self.description.update(dict(xc='Right ascension',
                                     yc='Declination',
                                     Ic='Peak surface brightness'))
        
    @staticmethod
    def profile(xc,yc,Ic):
        pass

    @staticmethod
    def getgrid():
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
        super().__init__(**kwargs)
        self.rc = kwargs.get('rc',config.Background.rc)
        self.a0 = kwargs.get('a0',config.Background.a0)
        self.a1x = kwargs.get('a1x',config.Background.a1x)
        self.a1y = kwargs.get('a1y',config.Background.a1y)
        self.a2xx  = kwargs.get('a2xx',config.Background.a2xx)
        self.a2xy  = kwargs.get('a2xy',config.Background.a2xy)
        self.a2yy  = kwargs.get('a2yy',config.Background.a2yy)
        self.a3xxx = kwargs.get('a3xxx',config.Background.a3xxx)
        self.a3xxy = kwargs.get('a3xxy',config.Background.a3xxy)
        self.a3xyy = kwargs.get('a3xyy',config.Background.a3xyy)
        self.a3yyy = kwargs.get('a3yyy',config.Background.a3yyy)

        self.units = dict(rc='deg')
        self.units.update({key: '' for key in self.__dict__.keys() if key not in self.okeys and key[0]=='a'})
        
        self.description.update(dict(rc    = 'Reference radius for polynomial terms',
                                     a0    = 'Polynomial coefficient 0',
                                     a1x   = 'Polynomial coefficient 1 in x',
                                     a1y   = 'Polynomial coefficient 1 in y',
                                     a2xx  = 'Polynomial coefficient 2 in x*x',
                                     a2xy  = 'Polynomial coefficient 2 in x*y',
                                     a2yy  = 'Polynomial coefficient 2 in y*y',
                                     a3xxx = 'Polynomial coefficient 3 in x*x*x',
                                     a3xxy = 'Polynomial coefficient 3 in x*x*y',
                                     a3xyy = 'Polynomial coefficient 3 in x*y*y',
                                     a3yyy = 'Polynomial coefficient 3 in y*y*y'))

    @staticmethod
    @jax.jit
    def profile(x,y,a0,a1x,a1y,a2xx,a2xy,a2yy,a3xxx,a3xxy,a3xyy,a3yyy,rc):
        xc, yc = x/rc, y/rc
        factor  = a0
        factor += a1x*xc + a1y*yc
        factor += a2xy*xc*yc + a2xx*xc**2 + a2yy*yc**2
        factor += a3xxx*xc**3 + a3yyy*yc**3 + a3xxy*xc**2*yc + a3xyy*xc*yc**2
        return factor
    
    @staticmethod
    @partial(jax.jit,static_argnames=['grid'])
    def getgrid(grid,xc,yc):
        return (grid.x-xc)*jp.cos(jp.deg2rad(yc)), grid.y-yc

    def getmap(self,img):
        xc, yc = self.img.wcs.crval
        xgrid, ygrid = self.getgrid(img.grid,xc,yc)
        klist = [key for key in self.__dict__.keys() if key not in self.okeys and key[0]=='a']+['rc']
        kwarg = {key: getattr(self,key) for key in klist}
        
        for key in kwarg.keys():
            if isinstance(kwarg[key], numpyro.distributions.Distribution):
                raise ValueError('Priors must be fixed values, not distributions.')
            if kwarg[key] is None:
                raise ValueError(f'keyword {key} is set to None. Please provide a valid value.')
        
        return self.profile(xgrid,ygrid,**kwarg)


# Vertical profile
# --------------------------------------------------------
class Height:
    def __init__(self,**kwargs):
        self.okeys = ['units']
        self.units = {}

        self.inc  = kwargs.get('inc',config.Height.inc)
        
        self.zext = kwargs.get('zext',config.Height.zext)
        self.zsize = kwargs.get('zsize',config.Height.zsize)
        self.units.update(dict(zext='deg', zsize='',inc='rad'))

        self.description = dict(zext = 'Half vertical extent for integration',
                               zsize = 'Number of vertical samples for integration',
                                 inc = 'Inclination angle (0=face-on)')
        
    @abstractmethod
    def profile(z):
        pass

# Hyperbolic cosine
# --------------------------------------------------------
class HyperCos(Height):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.zs = kwargs.get('zs',config.HyperCos.zs)
        self.units.update(dict(zs='deg'))
        self.description.update(dict(zs = 'Scale height'))

    @staticmethod
    @jax.jit
    def profile(z,zs):
        factor = jp.cosh(jp.abs(z)/zs)
        return 1.00/factor**2


# Disk model with finite thickness
# --------------------------------------------------------
class Disk(Component):
    def __init__(self,radial=Sersic(),vertical=HyperCos(),**kwargs):
        super().__init__(**kwargs)
        for key in ['radial','vertical','profile']:
            self.okeys.append(key)
        
        self.radial   = radial
        self.vertical = vertical

        if self.radial.id!=self.id:
            type(self).idcls -= 1
            idmin = np.minimum(int(self.radial.id.replace('src_','')),
                               int(self.id.replace('src_','')))
            self.id = f'src_{idmin:02d}'
            self.radial.id = self.id

        rkw = [f'r_{key}' for key in list(inspect.signature(self.radial.profile).parameters.keys()) if key not in ['r','z']]
        zkw = [f'z_{key}' for key in list(inspect.signature(self.vertical.profile).parameters.keys())  if key not in ['r','z']]

        profile_ = ['rfoo.profile(r,{0})'.format(','.join(rkw)),
                    'zfoo.profile(z,{0})'.format(','.join(zkw))]
        profile_ = '*'.join(profile_)
        profile_ = 'lambda rfoo,zfoo,r,z,{0},{1}: {2}'.format(','.join(rkw),','.join(zkw),profile_)

        self.profile = jax.jit(partial(eval(profile_),self.radial,self.vertical))

        print(self.vertical.units)

        self.units.update({f'radial.{key}': self.radial.units[key] for key in self.radial.units.keys()})
        self.units.update({f'vertical.{key}': self.vertical.units[key] for key in self.vertical.units.keys()})

        self.description.update({f'radial.{key}': self.radial.description[key] for key in self.radial.description.keys()})
        self.description.update({f'vertical.{key}': self.vertical.description[key] for key in self.vertical.description.keys()})

    def getmap(self,img,convolve=False):
        kwarg = {}
        for key in list(inspect.signature(self.profile).parameters.keys()):
            if key not in ['r','z']:
                if key.startswith('r_'):
                    kwarg[key] = getattr(self.radial,key.replace('r_',''))
                elif key.startswith('z_'):
                    kwarg[key] = getattr(self.vertical,key.replace('z_',''))
        
        for key in kwarg.keys():
            if isinstance(kwarg[key],numpyro.distributions.Distribution):
                raise ValueError('Priors must be fixed values, not distributions.')
            if kwarg[key] is None:
                raise ValueError(f'keyword {key} is set to None. Please provide a valid value.')

        rcube, zcube = self.getgrid(img.grid,
                                    self.radial.xc,
                                    self.radial.yc,
                                    self.vertical.zext,
                                    self.vertical.zsize,
                                    self.radial.theta,
                                    self.vertical.inc)

        dx = 2.00*self.vertical.zext/(self.vertical.zsize-1)

        mgrid = self.profile(rcube,zcube,**kwarg)
        mgrid = jp.trapezoid(mgrid,dx=dx,axis=1)
        mgrid = jp.mean(mgrid,axis=0)

        if convolve:
            if img.psf is None:
                 warnings.warn('No PSF defined, so no convolution will be performed.')
            else:
                mgrid = jp.fft.rfft2(jp.fft.fftshift(mgrid),s=img.data.shape)*img.psf_fft
                mgrid = jp.fft.ifftshift(jp.fft.irfft2(mgrid,s=img.data.shape)).real
        return mgrid

    @staticmethod
    @partial(jax.jit,static_argnames=['grid','zsize'])
    def getgrid(grid,xc,yc,zext,zsize=200,theta=0.00,inc=0.00):
        ssize, ysize, xsize = grid.x.shape

        zt = jp.linspace(-zext,zext,zsize)

        sint = jp.sin(theta)
        cost = jp.cos(theta)
        
        xt = (grid.x-xc)*jp.cos(jp.deg2rad(yc))
        yt = (grid.y-yc)

        xt, yt = -xt*sint-yt*cost,\
                  xt*cost-yt*sint

        xt = jp.broadcast_to(xt[   :,None,   :,   :],(ssize,zsize,ysize,xsize)).copy()
        yt = jp.broadcast_to(yt[   :,None,   :,   :],(ssize,zsize,ysize,xsize)).copy()
        zt = jp.broadcast_to(zt[None,   :,None,None],(ssize,zsize,ysize,xsize)).copy()

        sini = jp.sin(inc-0.50*jp.pi)
        cosi = jp.cos(inc-0.50*jp.pi)

        zt, yt = yt*cosi-zt*sini, \
                 yt*sini+zt*cosi
        
        return  jp.sqrt(xt**2+yt**2), zt
 
    def parameters(self):
        keyout =[]
        for key in self.radial.__dict__.keys():
            if key not in self.okeys and key!='okeys':
                keyout.append(f'radial.{key}')
        for key in self.vertical.__dict__.keys():
            if key not in self.okeys and key!='okeys':
                keyout.append(f'vertical.{key}')

        if len(keyout)>0:
            maxlen = np.max(np.array([len(f'{key} [{self.units[key]}]') for key in keyout]))
        
            for key in keyout:
                keylen = maxlen-len(f' [{self.units[key]}]')
                kvalue = self.radial.__dict__[key.replace('radial.','')] if 'radial' in key else \
                         self.vertical.__dict__[key.replace('vertical.','')]
                kvalue = None if kvalue is None else f'{kvalue:.4E}'
                print(f'{key:<{keylen}} [{self.units[key]}] : '+f'{kvalue}'.ljust(10) + f' | {self.description[key]}')
        else:
            print('No parameters defined.')

    def parlist(self):
        pars_  = [f'radial.{key}' for key in self.radial.__dict__.keys() if key not in self.okeys and key!='okeys']
        pars_ += [f'vertical.{key}' for key in self.vertical.__dict__.keys() if key not in self.okeys and key!='okeys']
        return pars_