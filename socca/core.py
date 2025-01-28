from .utils import *

import dynesty
import nautilus

import time
import dill


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

        mbkg = jp.zeros(self.img.shape)
        mraw = jp.zeros(self.img.shape)
        mpts = jp.fft.rfft2(mraw,s=self.img.shape)

        mneg = jp.zeros(self.img.shape)

        for nc in range(self.mod.ncomp):
            kwarg = {key.replace(f'src_{nc:02d}_',''): pars[key] for key in self.mod.params \
                  if key.startswith(f'src_{nc:02d}') and \
                     key.replace(f'src_{nc:02d}_','') in list(inspect.signature(self.mod.profile[nc]).parameters.keys())}

            if self.mod.type[nc]=='Background':
                yr = jp.mean(self.img.grid.y,axis=0)-self.img.hdu.header['CRVAL2']
                xr = jp.mean(self.img.grid.x,axis=0)-self.img.hdu.header['CRVAL1']
                xr = xr*jp.cos(jp.deg2rad(self.img.hdu.header['CRVAL2']))

                mone = self.mod.profile[nc](xr,yr,**kwarg)
                if self.mod.positive[nc]: mneg = jp.where(mone<0.00,1.00,mneg)

                mbkg += mone.copy(); del mone
            elif self.mod.type[nc]=='Point':
                uphase, vphase = self.img.fft.shift(kwarg['xc'],kwarg['yc'])
                
                mone = kwarg['Ic']*self.img.fft.pulse*jp.exp(-(uphase+vphase))
                if self.mod.positive[nc]: mneg = jp.where(mone<0.00,1.00,mneg)
                
                mpts += mone.copy(); del mone
            elif 'Thick' in self.mod.type[nc]:
                pass
            else:
                rgrid = self.img.getgrid(pars[f'src_{nc:02d}_xc'],
                                         pars[f'src_{nc:02d}_yc'],
                                         pars[f'src_{nc:02d}_theta'],
                                         pars[f'src_{nc:02d}_e'],
                                         pars[f'src_{nc:02d}_cbox'])

                mone = self.mod.profile[nc](rgrid,**kwarg)
                mone = jp.mean(mone,axis=0)
                if self.mod.positive[nc]: mneg = jp.where(mone<0.00,1.00,mneg)

                mraw += mone.copy(); del mone
        
        msmo = mraw.copy()
        if self.img.psf is not None:
            msmo = (mpts+jp.fft.rfft2(jp.fft.fftshift(mraw),s=self.img.shape))*self.img.psf_fft
            msmo = jp.fft.ifftshift(jp.fft.irfft2(msmo,s=self.img.shape)).real
    
        mpts = jp.fft.ifftshift(jp.fft.irfft2(mpts,s=self.img.shape)).real
        
        if self.img.psf is None:
            msmo = msmo+mpts

        return mraw+mpts, msmo+mbkg, mbkg, mneg

#   Compute log-likelihood
#   --------------------------------------------------------
    def _log_likelihood(self,pp):
        _, mod, _, neg = self._get_model(pp)

        mod = mod.at[self.mask].get()
        pdf = self.pdfnoise(mod)
        return jp.where(neg.at[self.mask].get()==1,-jp.inf,pdf).sum()

#   Main sampler function
#   --------------------------------------------------------
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
            toc = time.time()
            sampler.run(f_live=dlogz,verbose=True)
            tic = time.time()

            dt = tic-toc
            dt = '{0:.2f} s'.format(dt) if dt<60.00 else '{0:.2f} m'.format(dt/60.00)
            print(f'Elapsed time: {dt}')

            self.samples, log_w, _ = sampler.posterior()
            self.logz = sampler.evidence()
            self.samples = dynesty.utils.resample_equal(self.samples,np.exp(log_w))
        
        self.labels = [self.mod.params[idx] for idx in self.mod.paridx]

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
    def getmodel(self,usebest=True):
        if usebest:
            p = np.array([np.quantile(samp,0.50) for samp in self.samples.T])
            mraw, msmo, mbkg, _ = self._get_model(p)
            msmo = msmo-mbkg
        else:
            mraw, msmo = [], []
            for sample in self.samples:
                mraw_, msmo_, mbkg_, _ = self._get_model(sample)
                msmo_ = msmo_-mbkg_
                mraw.append(mraw_); del mraw_
                msmo.append(msmo_); del msmo_
                mbkg.append(mbkg_); del mbkg_

            mraw = np.quantile(mraw,0.50,axis=0)
            msmo = np.quantile(msmo,0.50,axis=0)
            mbkg = np.quantile(mbkg,0.50,axis=0)
        return mraw, msmo, mbkg