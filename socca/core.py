from pprint import pp
from shapely import bounds
from .utils import *

import dynesty
import nautilus
import pocomc

# Support functions
# ========================================================
# Compute importance weights for nested sampling
# --------------------------------------------------------
def get_imp_weights(logw,logz=None):
    if logz is None: logz = [logw.max()]
    if not hasattr(logz,'__len__'): logz = [logz]

    wt = logw-logz[-1]
    wt = wt-scipy.special.logsumexp(wt)
    return np.exp(wt)

# Fitter constructor
# ========================================================
# Initialize fitter structure
# --------------------------------------------------------
class fitter:
    def __init__(self,img,mod,noise):
        self.img = img
        self.mod = mod

        self.mask = jp.where(self.img.mask==1.00)
        self.pdfnoise = noise.logpdf

        if not hasattr(self.img,'shape'):
            setattr(self.img,'shape',self.img.data.shape)
        else:
            self.img.shape = self.img.data.shape

        self.labels = [self.mod.params[idx] for idx in self.mod.paridx]
    
#   Compute total model
#   --------------------------------------------------------
    def _get_model(self,pp):
        doresp = ~np.all(np.array(self.img.resp) == 1.00) #True
        doexp  = ~np.all(np.array(self.img.exp)  == 1.00) #True
        return self.mod.getmap(self.img,pp,doresp=doresp,doexp=doexp)

#   Compute log-likelihood
#   --------------------------------------------------------
    def _log_likelihood(self,pp):
        _, mod, _, neg = self._get_model(pp)

        mod = mod.at[self.mask].get()
        pdf = self.pdfnoise(mod)
        return jp.where(neg.at[self.mask].get()==1,-jp.inf,pdf).sum()

#   Prior probability distribution
#   --------------------------------------------------------
    def _log_prior(self,theta):
        prob = 0.00
        for idx in self.mod.paridx:
            key = self.mod.params[idx]
            prob += self.mod.priors[key].logpdf(theta[key])
        return prob

#   Transform prior hypercube
#   --------------------------------------------------------
    def _prior_transform(self,pp):
        prior = []
        for pi, p in enumerate(pp):
            key = self.mod.params[self.mod.paridx[pi]]
            prior.append(self.mod.priors[key].ppf(p))
        return prior
    
#   Main sampler function
#   --------------------------------------------------------
    def run(self,nlive=100,dlogz=0.01,method='dynesty',checkpoint=None,resume=True,**kwargs):
        self.method = method

        @jax.jit
        def log_likelihood(theta):
            return self._log_likelihood(theta)
        
        def log_prior(theta):
            return self._log_prior(theta)

        def prior_transform(utheta):
            return self._prior_transform(utheta)
        
        sampler_methods = {'dynesty': self._run_dynesty,
                          'nautilus': self._run_nautilus,
                            'pocomc': self._run_pocomc,
                         'optimizer': self._run_optimizer
        }
        
        if self.method in sampler_methods:
            sampler_kwargs = list(inspect.signature(sampler_methods[self.method]).parameters.keys())
            sampler_kwargs = [{key: eval(key) for key in sampler_kwargs if key!='kwargs'},eval('kwargs')]
            sampler_methods[self.method](**sampler_kwargs[0],**sampler_kwargs[1])
        else:
            raise ValueError(f"Unknown sampling method: {self.method}")

        self.logz_data = self.img.data.at[self.mask].get()
        self.logz_data = self.pdfnoise(jp.zeros(self.logz_data.shape)).sum()


#   Fitting method - Dynesty sampler
#   --------------------------------------------------------
    def _run_dynesty(self,log_likelihood,log_prior,prior_transform,nlive,dlogz,checkpoint,resume,**kwargs):
        ndims = len(self.mod.paridx)

        if checkpoint is None:
            resume = False

        print('\n* Running the main sampling step')
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

        print('\n* Sampling the prior probability')
        sampler_prior = dynesty.NestedSampler(log_prior,prior_transform,
                                            ndim = ndims,
                                           nlive = nlive,
                                           bound = 'multi')
        sampler_prior.run_nested(dlogz=dlogz)
        self.logz_prior = sampler.results['logz'][-1]

#   Fitting method - Nautilus sampler
#   --------------------------------------------------------
    def _run_nautilus(self,log_likelihood,log_prior,nlive,dlogz,checkpoint,resume,**kwargs):
        prior = nautilus.Prior()
        for ki, key in enumerate(self.mod.params):
            if isinstance(self.mod.priors[key],scipy.stats._distn_infrastructure.rv_continuous_frozen):
                prior.add_parameter(key,dist=self.mod.priors[key])

        def dict_likelihood(theta):
            pars = np.array([theta[self.mod.params[idx]] for idx in self.mod.paridx])
            return log_likelihood(pars)
        
        print('\n* Running the main sampling step')
        self.sampler = nautilus.Sampler(prior,dict_likelihood,n_live=nlive,filepath=checkpoint,resume=resume)
        
        toc = time.time()
        self.sampler.run(f_live=dlogz,verbose=True,**kwargs)
        tic = time.time()

        dt = tic-toc
        dt = '{0:.2f} s'.format(dt) if dt<60.00 else '{0:.2f} m'.format(dt/60.00)
        print(f'Elapsed time: {dt}')

        self.samples, self.logw, _ = self.sampler.posterior()
        self.logz = self.sampler.log_z

        self.weights = get_imp_weights(self.logw,self.logz)

        print('\n* Sampling the prior probability')
        sampler_prior = nautilus.Sampler(prior,log_prior,n_live=nlive)
        sampler_prior.run(f_live=dlogz,verbose=True,**kwargs)
        self.logz_prior = sampler_prior.log_z

#   Fitting method - PocoMC sampler
#   --------------------------------------------------------
    def _run_pocomc(self,log_likelihood,log_prior,nlive,checkpoint,resume,**kwargs):
        if checkpoint is None and resume: checkpoint = 'run'
        pocodir = '{0}_pocomc_dump'.format(checkpoint.replace('.hdf5','').replace('.h5',''))

        prior = []
        for ki, key in enumerate(self.mod.params):
            if isinstance(self.mod.priors[key],scipy.stats._distn_infrastructure.rv_continuous_frozen):
                prior.append(self.mod.priors[key])
        prior = pocomc.Prior(prior)

        print('\n* Running the main sampling step')
        self.sampler = pocomc.Sampler(likelihood = log_likelihood,
                                           prior = prior,
                                     n_effective = nlive,
                                        n_active = nlive//2,
                                       vectorize = False,
                                      output_dir = pocodir if resume else None,
                                         dynamic = True,
                                    random_state = 0)

        save_every = kwargs.get('save_every',10) if checkpoint is not None else None

        states_ = sorted(glob.glob(f'{pocodir}/*.state')) if resume else []
        self.sampler.run(save_every=save_every,resume_state_path=states_[-1] if len(states_) else None,progress=True)

        self.samples, self.logw, _, _ = self.sampler.posterior()
        self.logz, _ = self.sampler.evidence()

        self.weights = get_imp_weights(self.logw,self.logz)

        print('\n* Sampling the prior probability')
        sampler_prior = pocomc.Sampler(likelihood = log_prior,
                                           prior = prior,
                                     n_effective = nlive,
                                        n_active = nlive//2,
                                       vectorize = False,
                                         dynamic = True,
                                    random_state = 0)
        self.sampler.run(progress=True)
        self.logz_prior, _ = self.sampler.evidence()

#   Fitting method - optimizer
#   --------------------------------------------------------
    def _run_optimizer(self):
        pass

#   Compute standard Bayesian Model Selection estimators
#   --------------------------------------------------------
    def bmc(self,verbose=True):
        lnBF_raw = self.logz-self.logz_data
        seff_raw = np.sign(lnBF_raw)*np.sqrt(2.00*np.abs(lnBF_raw))

        lnBF_cor = lnBF_raw-self.logz_prior
        seff_cor = np.sign(lnBF_cor)*np.sqrt(2.00*np.abs(lnBF_cor))

        if verbose:
            print('\nnull-model comparison')
            print('-'*20)
            print(f'ln(Bayes factor) : {lnBF_raw:10.3E}')
            print(f'effective sigma  : {seff_raw:10.3E}')
            print('\nprior deboosted')
            print('-'*20)
            print(f'ln(Bayes factor) : {lnBF_cor:10.3E}')
            print(f'effective sigma  : {seff_cor:10.3E}\n')

        return lnBF_raw, seff_raw, lnBF_cor, seff_cor

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
    def getmodel(self,usebest=True,img=None,doresp=False,doexp=False):
        gm = lambda pp: self.mod.getmap(self.img if img is None else img,pp,doresp,doexp)

        if usebest:
            p = np.array([np.quantile(samp,0.50,method='inverted_cdf',weights=self.weights) for samp in self.samples.T])
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

            mraw = np.quantile(mraw,0.50,axis=0,method='inverted_cdf',weights=self.weights)
            msmo = np.quantile(msmo,0.50,axis=0,method='inverted_cdf',weights=self.weights)
            mbkg = np.quantile(mbkg,0.50,axis=0,method='inverted_cdf',weights=self.weights)
        return mraw, msmo, mbkg