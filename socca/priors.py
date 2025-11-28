from .utils import *

# Uniform
# --------------------------------------------------------
def uniform(low,high):
    if low>high:
        message = 'The lower limit must be smaller than the upper limit.'
        message = message+f'\nInput values > low={low} high={high}'
        raise ValueError(message) 
   #return scipy.stats.uniform(loc=low,scale=high-low)
    return numpyro.distributions.Uniform(low,high)

# Log-uniform
# --------------------------------------------------------
def loguniform(low,high):
    if low>high:
        message = 'The lower limit must be smaller than the upper limit.'
        message = message+f'\nInput values > low={low} high={high}'
        raise ValueError(message) 
   #return scipy.stats.loguniform(a=low,b=high)
    return numpyro.distributions.LogUniform(low,high)

# Normal
# --------------------------------------------------------
def normal(loc,scale):
   #return scipy.stats.norm(loc=loc,scale=scale)
    return numpyro.distributions.Normal(loc,scale)
   
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

# Parameter bound to another component's parameter
# --------------------------------------------------------
def boundto(comp,var):
    if isinstance(comp,str):
        comp = eval(comp)

    x = f'{comp.id}_{var}'
    return eval(f'lambda {x}: {x}')


# pocoMC prior refactor for numpyro
# --------------------------------------------------------
class pocomcPrior:
    def __init__(self,dists,seed=0):
        self.dists = dists
        self.key = jax.random.PRNGKey(seed)
    
    def logpdf(self,x):
        logp = np.zeros(len(x))
        for i, dist in enumerate(self.dists): 
            logp += dist.log_prob(x[:,i])
        return logp

    def rvs(self,size=1):
        keys = jax.random.split(self.key,len(self.dists)+1)
        self.key, keys = keys[0], keys[1:]

        samples = [d.sample(k,(size,)) for d, k in zip(self.dists,keys)]
        return jp.column_stack(samples)
    
    @property
    def bounds(self):
        bounds = []
        for dist in self.dists:
            bounds.append([float(dist.icdf(0.00)),
                           float(dist.icdf(1.00))])
        return np.array(bounds)
    
    @property
    def dim(self):
        return len(self.dists)