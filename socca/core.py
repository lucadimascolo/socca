from .utils import *

import inspect

import dynesty
import nautilus

import dill

class fitter:
    def __init__(self,img,mod,noise='normal'):
        self.img = img
        self.mod = mod

        self.mask = jp.where(self.img.mask==1.00)

        data  = self.img.data.at[self.mask].get()
        sigma = self.img.sigma.at[self.mask].get()

        if noise=='normal':
            self.pdfnoise = lambda x: jax.scipy.stats.norm.logpdf(x,loc=data,scale=sigma).sum()

        if not hasattr(self.img,'shape'):
            setattr(self.img,'shape',self.img.data.shape)
        else:
            self.img.shape = self.img.data.shape

    def _prior_transform(self,pp):
        prior = []
        for pi, p in enumerate(pp):
            key = self.mod.params[self.mod.paridx[pi]]
            prior.append(self.mod.priors[key].ppf(p))
        return prior
    
    def _get_model(self,pp):
        pars = {}
        for ki, key in enumerate(self.mod.params):
            if isinstance(self.mod.priors[key],float):
                pars[key] = self.mod.priors[key]
            elif isinstance(self.mod.priors[key],scipy.stats._distn_infrastructure.rv_continuous_frozen):
                pars[key], pp = pp[0], pp[1:]
        
        for ki, key in enumerate(self.mod.params):
            if self.mod.tied[ki]:
                kwarg = list(inspect.signature(self.mod.priors[key]).parameters.keys())
                kwarg = {k: pars[k] for k in kwarg}
                pars[key] = self.mod.priors[key](**kwarg)
                del kwarg

        mraw = jp.zeros_like(self.img.grid.x)

        for nc in range(self.mod.ncomp):
            kwarg = {key.replace(f'src_{nc:02d}_',''): pars[key] for key in self.mod.params \
                     if key.startswith(f'src_{nc:02d}') and \
                        key.replace(f'src_{nc:02d}_','') in list(inspect.signature(self.mod.profile[nc]).parameters.keys())}

            rgrid = self.img.getgrid(pars[f'src_{nc:02d}_xc'],
                                     pars[f'src_{nc:02d}_yc'],
                                     pars[f'src_{nc:02d}_theta'],
                                     pars[f'src_{nc:02d}_e'])
     
            mraw += self.mod.profile[nc](rgrid,**kwarg)

        msmo = mraw.copy()
        if self.img.psf is not None:
            msmo = jp.fft.rfft2(jp.fft.fftshift(mraw),s=self.img.shape)*self.img.psf_fft
            msmo = jp.fft.ifftshift(jp.fft.irfft2(msmo,s=self.img.shape)).real

        return mraw, msmo

    def _log_likelihood(self,pp):
        _, mod = self._get_model(pp)
        mod = mod.at[self.mask].get()
        return self.pdfnoise(mod)

    def run(self,nlive=100,dlogz=0.01,method='dynesty'):
        self.method = method

        @jax.jit
        def log_likelihood(theta):
            return self._log_likelihood(theta)
        
        def prior_transform(utheta):
            return self._prior_transform(utheta)
        
        if self.method=='dynesty':

            ndims = len(self.mod.paridx)
            sampler = dynesty.NestedSampler(log_likelihood,prior_transform,
                                            ndim = ndims,
                                           nlive = nlive,
                                           bound = 'multi')
            sampler.run_nested(dlogz=dlogz)
        
            results = sampler.results

            self.samples = dynesty.utils.resample_equal(results['samples'],
                                                        results.importance_weights())
            
        elif self.method=='nautilus':
            prior = nautilus.Prior()
            for ki, key in enumerate(self.mod.params):
                if isinstance(self.mod.priors[key],scipy.stats._distn_infrastructure.rv_continuous_frozen):
                    prior.add_parameter(key,dist=self.mod.priors[key])

            def dict_likelihood(theta):
                pars = np.array([theta[self.mod.params[idx]] for idx in self.mod.paridx])
                return log_likelihood(pars)
            
            sampler = nautilus.Sampler(prior,dict_likelihood,n_live=nlive)
            sampler.run(f_live=dlogz,verbose=True)

            self.samples, log_w, _ = sampler.posterior()

            self.samples = dynesty.utils.resample_equal(self.samples,np.exp(log_w))
        
        self.labels = [self.mod.params[idx] for idx in self.mod.paridx]

    def dump(self,filename):
        odict = {key: self.__dict__[key] for key in self.__dict__.keys()}
        
        with open(filename,'wb') as f:
            dill.dump(odict,f,dill.HIGHEST_PROTOCOL)
        
    def load(self,filename):
        with open(filename,'rb') as f:
            odict = dill.load(f)
        self.__dict__.update(odict)

    def getmodel(self,usebest=True):
        if usebest:
            p = np.array([np.quantile(samp,0.50) for samp in self.samples.T])
            mraw, msmo = self._get_model(p)
        else:
            mraw, msmo = [], []
            for sample in self.samples:
                mraw_, msmo_ = self._get_model(sample)
                mraw.append(mraw_); del mraw_
                msmo.append(msmo_); del msmo_

            mraw = np.quantile(mraw,0.50,axis=0)
            msmo = np.quantile(msmo,0.50,axis=0)
        return mraw, msmo