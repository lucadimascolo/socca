from .utils import *

import scipy.special

# Uniform
# --------------------------------------------------------
def uniform(low,high):
    if low>high:
        message = 'The lower limit must be smaller than the upper limit.'
        message = message+f'\nInput values > low={low} high={high}'
        raise ValueError(message) 
    return scipy.stats.uniform(loc=low,scale=high-low)

# Log-uniform
# --------------------------------------------------------
def loguniform(low,high):
    if low>high:
        message = 'The lower limit must be smaller than the upper limit.'
        message = message+f'\nInput values > low={low} high={high}'
        raise ValueError(message) 
    return scipy.stats.loguniform(a=low,b=high-low)

# Normal
# --------------------------------------------------------
def normal(loc,scale):
    return scipy.stats.norm(loc=loc,scale=scale)

# Split-normal
# --------------------------------------------------------
class splitnorm_gen(scipy.stats.rv_continuous):
    'splitnorm'
    def _pdf(self,x,losig,hisig):
        sig = hisig*np.heaviside(x,1.00)+losig*np.heaviside(-x,1.00)
        pdf = np.exp(-(x/sig)**2.00/2.00)
        return pdf*np.sqrt(2.00/np.pi)/(losig+hisig)

    def _ppf(self,q,losig,hisig):
        loppf =  losig*scipy.special.ndtri(q*(losig+hisig)/losig/2.00)
        hippf = -hisig*scipy.special.ndtri((1.00-q)*(losig+hisig)/hisig/2.00)
        return np.where(loppf<0.00,loppf,hippf)

splitnorm = splitnorm_gen(name='splitnorm')

# parameter bound to another component's parameter
# --------------------------------------------------------
def boundto(comp,var):
    if isinstance(comp,str):
        comp = eval(comp)

    x = f'{comp.id}_{var}'
    return eval(f'lambda {x}: {x}')