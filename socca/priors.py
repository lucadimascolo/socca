from .utils import *

# Uniform
# --------------------------------------------------------
def uniform(low,high):
    if low>high:
        message = 'The lower limit must be smaller than the upper limit.'
        message = message+f'\nInput values > low={low} high={high}'
        raise ValueError(message) 
    return numpyro.distributions.Uniform(low,high)

# Log-uniform
# --------------------------------------------------------
def loguniform(low,high):
    if low>high:
        message = 'The lower limit must be smaller than the upper limit.'
        message = message+f'\nInput values > low={low} high={high}'
        raise ValueError(message) 
    return numpyro.distributions.LogUniform(low,high)

# Normal
# --------------------------------------------------------
def normal(loc,scale):
    return numpyro.distributions.Normal(loc,scale)
   
# Split-normal
# --------------------------------------------------------
class SplitNormal(numpyro.distributions.Distribution):
    """
    Split Normal (two-piece normal) distribution centered at 0.

    Parameters
    ----------
    losig : scale for x < 0
    hisig : scale for x >= 0
    """
    arg_constraints = {"losig": numpyro.distributions.constraints.positive,
                       "hisig": numpyro.distributions.constraints.positive,}
    support = numpyro.distributions.constraints.real
    reparametrized_params = ["losig", "hisig"]

    def __init__(self, losig, hisig, validate_args=None):
        self.losig, self.hisig = numpyro.distributions.util.promote_shapes(losig,hisig)
        batch_shape = jp.broadcast_shapes(jp.shape(self.losig),
                                          jp.shape(self.hisig),)
        super().__init__(batch_shape=batch_shape,
                         event_shape=(),
                         validate_args=validate_args)

    def log_prob(self, x):
        losig = self.losig
        hisig = self.hisig

        sig_prob = jp.where(x>=0.0,hisig,losig)
        log_norm = jp.log(jp.sqrt(2.0/jp.pi))-jp.log(losig+hisig)
        return log_norm-0.5*(x/sig_prob)**2

def splitnormal(loc,losig,hisig):
    return numpyro.distributions.TransformedDistribution(
        base_distribution = SplitNormal(losig,hisig),
               transforms = [numpyro.distributions.transforms.AffineTransform(loc,1.0)]
    )


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