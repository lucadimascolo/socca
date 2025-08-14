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

        self.logpdf = jax.jit(lambda x: self._logpdf(x,self.data,self.sigma))

    @staticmethod
    def _logpdf(x,data,sigma):
        return jax.scipy.stats.norm.logpdf(x,loc=data,scale=sigma)
    
# Correlated multi-variate normal noise
# --------------------------------------------------------
class NormalCorr(PDF):
    def __init__(self,img):
        super().__init__(img)
        self.logpdf = jax.jit(lambda x: self._logpdf(x,self.data))

    @staticmethod
    def _logpdf(x,data):
        chi2 = 0 # jp.matmul(x,x)
        chi2 = chi2-2.00*jp.matmul(data,x)
        return -0.50*chi2