from .utils import *
from .utils import _img_loader

import scipy.linalg
from astropy.convolution import convolve, CustomKernel

# Base PDF class
# ========================================================
class BaseNormal:
    def __init__(self,**kwargs):
        """
        Parameters
        ----------
        sigma : float or str, optional
            Standard deviation of the noise. Default is None, in which case the noise level is
            estimated using the Median Absolute Deviation (MAD) method.
        var : float or str, optional
            Variance of the noise. 
        wht : float or str, optional
            Weight (inverse variance) of the noise.

        If a string is provided, it is treated as a path to
        a FITS file containing the corresponding map.
        """
        self.options = {'sig': ['sigma','sig','std','rms','stddev'],
                        'var': ['var','variance'],
                        'wht': ['wht','wgt','weight','weights','invvar']}

        options = self.options['sig']+self.options['var']+self.options['wht']

        self.select = np.array([key for key in options if key in kwargs])
        if self.select.shape[0]>1:
            raise ValueError('Multiple noise identifiers found in kwargs. Please use only one of sigma, variance, or weight.')
        self.select = self.select[0] if self.select.shape[0]==1 else None

        self.kwargs = {key: kwargs[key] for key in options if key in kwargs}


#   Estimate/build noise statistics
#   --------------------------------------------------------
    def getsigma(self):
        if self.select is None:
            print('Using MAD for estimating noise level')
            sigma = scipy.stats.median_abs_deviation(self.data.at[self.mask].get(),axis=None,scale='normal',nan_policy='omit')
            sigma = float(sigma)
            print(f'- noise level: {sigma:.2E}')
        elif isinstance(self.select,str):
            if isinstance(self.kwargs[self.select],(float,int)):
                sigma = self.kwargs[self.select]
            else:
                sigma = _img_loader(self.kwargs[self.select],self.kwargs.get('idx',0)).data.copy()

            if self.select in self.options['var']:
                sigma = np.sqrt(sigma)
            elif self.select in self.options['wht']:
                self.mask.at[sigma==0.00].set(0)
                sigma = 1.00/np.sqrt(sigma)
            elif self.select not in self.options['sig']:
                raise ValueError('Unrecognized noise identifier')

        if isinstance(sigma,(float,int)):
            sigma = np.full(self.data.shape,sigma).astype(float)

        sigma[np.isinf(sigma)] = 0.00
        sigma[np.isnan(sigma)] = 0.00
        self.mask.at[sigma==0.00].set(0)
        
        return jp.array(sigma)
    

# Multi-variate normal noise (no correlation)
# ========================================================
class Normal(BaseNormal):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.kwargs.update({'idx': kwargs.get('idx',0)})

#   Set up noise model
#   --------------------------------------------------------
    def __call__(self,data,mask):
        """
        data : jax.numpy.ndarray
            Image data array.
        mask : jax.numpy.ndarray
            Image mask array.
        """
        self.mask = mask.astype(int).copy()
        self.data = data.copy()

        self.sigma = self.getsigma()

        self.mask  = self.mask==1.00
        self.data  = self.data.at[self.mask].get()
        self.sigma = self.sigma.at[self.mask].get()

        self.logpdf = jax.jit(lambda xr, xs: self._logpdf(xs,self.data,self.sigma))

#   Noise log-pdf/likelihood function   
#   --------------------------------------------------------
    @staticmethod
    def _logpdf(x,data,sigma):
        return jax.scipy.stats.norm.logpdf(x,loc=data,scale=sigma).sum()


# ALMA-like correlated normal noise
# ========================================================
class NormalALMA(BaseNormal):
    def __init__(self,donorm=True,**kwargs):
        super().__init__(**kwargs)
        self.donorm = donorm

        if not isinstance(self.kwargs[self.select],(float,int)):
            raise ValueError('NormalALMA supports only float/int inputs for the noise model.')
        
#   Set up noise model
#   --------------------------------------------------------
    def __call__(self,data,mask,psf_fft):
        self.mask = mask.astype(int).copy()
        self.data = data.copy()

        self.sigma = self.getsigma().mean()
        
        self.mask  = self.mask==1.00
        self.data  = self.data.at[self.mask].get()

        self.norm  = self.getnorm(psf_fft) if self.donorm else 0.00

        self.logpdf = jax.jit(lambda xr, xs: self._logpdf(xr,xs,self.data,self.sigma,self.norm))
    
#   Noise log-pdf/likelihood function
#   --------------------------------------------------------
    @staticmethod
    def _logpdf(xr,xs,data,sigma,norm=0.00):
        factor = xr*(xs-2.00*data)
        factor = jp.sum(factor)+norm
        return -0.50*factor/sigma**2
    
#   Likelihood normalization
#   --------------------------------------------------------
    def getnorm(self,psf_fft):
        refB = jp.fft.irfft2(psf_fft,s=self.mask.shape)
        covB = scipy.linalg.circulant(refB.ravel())
        covB = jp.array(covB)

        invB = jp.linalg.inv(covB)

        factor1 = jp.einsum('i,ij,j->',self.data,invB,self.data)/self.sigma**2
        factor2 = jp.linalg.slogdet(2.00*jp.pi*(self.sigma**2)*covB)[1]
        return factor1+factor2


# Correlated multi-variate normal noise
# To-Do: covariance from noise realizations
# ========================================================
class NormalCorr:
    def __init__(self,cov):
        self.cov = jp.asarray(cov.astype(float))
        self.icov = jp.linalg.inv(self.cov)

#   Set up noise model
#   --------------------------------------------------------
    def __call__(self,data,mask):
        self.mask = mask.copy()
        self.mask = self.mask==1.00
        self.data = data.at[self.mask].get()

        self.norm = float(jp.linalg.slogdet(2.00*jp.pi*self.cov)[1])
        self.logpdf = jax.jit(lambda xr, xs: self._logpdf(xs,self.data,self.icov)-0.50*self.norm)

#   Noise log-pdf/likelihood function
#   --------------------------------------------------------
    @staticmethod
    def _logpdf(x,data,icov):
        res = x-data
        return -0.50*jp.einsum('i,ij,j->',res,icov,res)

# Stationary noise
# ========================================================
class NormalStat:
    def __init__(self,cov=None,cube=None,ftype='real',**kwargs):
        if ftype not in ['real','rfft','full','fft']:
            raise ValueError("ftype must be either 'real'/'rfft' or 'full'/'fft'.")
        self.ftype = ftype

        if cube is not None and cov is None:
            self.apod = kwargs.get('apod',jp.ones(cube.shape))
            
            fft = jp.fft.rfft2  if ftype in ['real','rfft'] else jp.fft.fft2
            cov = fft(cube*self.apod,axes=(-2,-1))
            cov = np.mean(np.abs(cov)**2,axis=0)/np.mean(self.apod**2)

            smooth = kwargs.get('smooth',3)
            if smooth>0:
                kernel = kwargs.get('kernel',None)
                if kernel is None:
                    kernel = np.array([[0.00,1.00,0.00],
                                       [1.00,1.00,1.00],
                                       [0.00,1.00,0.00]])/5.00
                    kernel = CustomKernel(kernel)

                for _ in range(smooth): 
                    cov = jp.array(convolve(cov,kernel,boundary='wrap'))

        if cov is not None and cube is not None:
            warnings.warn('Both covariance matrix and noise cube provided. \
                           Using covariance matrix.')
        elif cov is None and cube is None:
            raise ValueError('Either covariance matrix or noise realization cube must be provided.')

        self.cov  = jp.asarray(cov.astype(float))
        self.icov = 1.00/self.cov

#   Set up noise model
#   --------------------------------------------------------
    def __call__(self,data,mask):
        self.mask = mask.copy()
        self.mask = self.mask==1.00
        self.data = data.copy()
        
        self.norm = 0.00
        self.logpdf = jax.jit(lambda xr, xs: self._logpdf(xs,self.data,self.icov,mask=self.mask,apod=self.apod,ftype=self.ftype)-0.50*self.norm)

#   Noise log-pdf/likelihood function
#   --------------------------------------------------------
    @staticmethod
    def _logpdf(x,data,icov,mask,apod,ftype='real'):
        fft = jp.fft.rfft2 if ftype in ['real','rfft'] else jp.fft.fft2

        xmap = jp.zeros(mask.shape)
        xmap = xmap.at[mask].set(x)

        chisq = fft((xmap-data)*apod,axes=(-2,-1))
        chisq = icov*jp.abs(chisq)**2
        return -0.50*jp.sum(chisq)
