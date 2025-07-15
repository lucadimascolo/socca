from .utils import *

import dynesty
import nautilus

import time
import dill
import os

# Fitter constructor
# ========================================================
# Initialize fitter structure
# --------------------------------------------------------
class fitter:
    def __init__(self,img,mod,noise='normal'):
        self.img = img
        self.mod = mod

        self.mask = jp.where(self.img.mask==1.00)

        data  = self.img.data.at[self.mask].get()
        sigma = self.img.sigma.at[self.mask].get()

        if noise=='normal':
            self.pdfnoise = lambda x: jax.scipy.stats.norm.logpdf(x,loc=data,scale=sigma)

        if not hasattr(self.img,'shape'):
            setattr(self.img,'shape',self.img.data.shape)
        else:
            self.img.shape = self.img.data.shape

        self.labels = [self.mod.params[idx] for idx in self.mod.paridx]

#   Transform prior hypercube
#   --------------------------------------------------------
    def _prior_transform(self,pp):
        prior = []
        for pi, p in enumerate(pp):
            key = self.mod.params[self.mod.paridx[pi]]
            prior.append(self.mod.priors[key].ppf(p))
        return prior
    
#   Compute total model
#   --------------------------------------------------------
    def _get_model(self,pp):
        return self.mod.getmap(self.img,pp)

#   Compute log-likelihood
#   --------------------------------------------------------
    def _log_likelihood(self,pp):
        _, mod, _, neg = self._get_model(pp)

        mod = mod.at[self.mask].get()
        pdf = self.pdfnoise(mod)
        return jp.where(neg.at[self.mask].get()==1,-jp.inf,pdf).sum()

#   Main sampler function
#   --------------------------------------------------------
    def run(self,nlive=100,dlogz=0.01,method='dynesty',checkpoint=None,resume=True,**kwargs):
        self.method = method

        @jax.jit
        def log_likelihood(theta):
            return self._log_likelihood(theta)
        
        def prior_transform(utheta):
            return self._prior_transform(utheta)
        
        if self.method=='dynesty':

            ndims = len(self.mod.paridx)

            if checkpoint is None:
                resume = False

            if ~resume or not os.path.exists(checkpoint):
                sampler = dynesty.NestedSampler(log_likelihood,prior_transform,
                                                ndim = ndims,
                                               nlive = nlive,
                                               bound = 'multi')
                sampler.run_nested(dlogz=dlogz,checkpoint_file=checkpoint)
            else:
                sampler = dynesty.NestedSampler.restore(checkpoint)
                sampler.run_nested(resume=True)

            self.sampler = sampler

            results = sampler.results

            self.samples = results['samples'].copy()
            self.weights = results.importance_weights().copy()
            self.logz = results['logz'][-1]
            
        elif self.method=='nautilus':
            prior = nautilus.Prior()
            for ki, key in enumerate(self.mod.params):
                if isinstance(self.mod.priors[key],scipy.stats._distn_infrastructure.rv_continuous_frozen):
                    prior.add_parameter(key,dist=self.mod.priors[key])

            def dict_likelihood(theta):
                pars = np.array([theta[self.mod.params[idx]] for idx in self.mod.paridx])
                return log_likelihood(pars)
            
            self.sampler = nautilus.Sampler(prior,dict_likelihood,n_live=nlive,filepath=checkpoint,resume=resume)
            
            toc = time.time()
            self.sampler.run(f_live=dlogz,verbose=True,**kwargs)
            tic = time.time()

            dt = tic-toc
            dt = '{0:.2f} s'.format(dt) if dt<60.00 else '{0:.2f} m'.format(dt/60.00)
            print(f'Elapsed time: {dt}')

            self.samples, logw, _ = self.sampler.posterior()
            self.weights = np.exp(logw).copy()
            self.logz = self.sampler.evidence()
    
#   Dump results
#   --------------------------------------------------------
    def dump(self,filename):
        odict = {key: self.__dict__[key] for key in self.__dict__.keys()}
        
        with open(filename,'wb') as f:
            dill.dump(odict,f,dill.HIGHEST_PROTOCOL)

#   Load results
#   --------------------------------------------------------
    def load(self,filename):
        with open(filename,'rb') as f:
            odict = dill.load(f)
        self.__dict__.update(odict)

#   Generate best-fit/median model
#   --------------------------------------------------------
    def getmodel(self,usebest=True,img=None):
        gm = self._get_model if img is None else lambda pp: self.mod.getmap(img,pp)

        if usebest:
            p = np.array([np.quantile(samp,0.50) for samp in self.samples.T])
            mraw, msmo, mbkg, _ = gm(p)
            msmo = msmo-mbkg
        else:
            mraw, msmo = [], []
            for sample in self.samples:
                mraw_, msmo_, mbkg_, _ = gm(sample)
                msmo_ = msmo_-mbkg_
                mraw.append(mraw_); del mraw_
                msmo.append(msmo_); del msmo_
                mbkg.append(mbkg_); del mbkg_

            mraw = np.quantile(mraw,0.50,axis=0)
            msmo = np.quantile(msmo,0.50,axis=0)
            mbkg = np.quantile(mbkg,0.50,axis=0)
        return mraw, msmo, mbkg