from .utils import *


class PDF:
    def __init__(self,img):
        self.mask  = jp.where(img.mask==1.00)
        self.data  = img.data.at[self.mask].get()

# Multi-variate normal noise (no correlation)
# --------------------------------------------------------
class Normal(PDF):
    def __init__(self,img):
        super().__init__(img)
        self.sigma = img.sigma.at[self.mask].get()
        self.logpdf = jax.jit(lambda xr, xs: self._logpdf(xs,self.data,self.sigma))

    @staticmethod
    def _logpdf(x,data,sigma):
        return jax.scipy.stats.norm.logpdf(x,loc=data,scale=sigma)
    
# Correlated multi-variate normal noise
# --------------------------------------------------------
class NormalBeam(PDF):
    def __init__(self,img,eps=0.00):
        super().__init__(img)

      # Tikonov/Wiener damping
        if eps==0.00: 
            invB = 1.00/img.psf_fft 
        else:
            invB = eps*jp.abs(img.psf_fft).max()**2
            invB = jp.abs(img.psf_fft)**2+invB
            invB = jp.conj(img.psf_fft)/invB
        
        self.norm   = jp.fft.ifft2(jp.fft.fft2(img.data)*invB).real
        self.norm   = jp.sum(self.norm.at[self.mask].get())

        self.logpdf = jax.jit(lambda xr, xs: self._logpdf(xr,xs,self.data,self.norm))

    @staticmethod
    def _logpdf(xr,xs,data,norm):
        chi2 = norm+jp.sum(xr*xs)-2.00*jp.sum(data*xr)
        return -0.50*chi2