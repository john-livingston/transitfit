from collections import OrderedDict
import numpy as np
import pandas as pd
import copy
import lmfit
from emcee.utils import sample_ball
from astropy import constants as c
from astropy import units as u

from .linear import LinearSystematicsModel
from .util import init_batman, get_par, get_theta, timeit
from .util import t14_circ, t23_circ, arstar, max_k, inclination
from .model import residual_lm, logprob_lm
from .plot import plot_trace, plot_corner, plot_basic, plot_map, plot_mcmc


class TransitFit(object):

    def __init__(self, init_params, name, time, flux, unc=None, aux=None):

        """
        Analyze a transit dataset
        """
        self.name = name
        self.init_params = init_params
        self.t = time
        self.f = flux
        self.u = unc
        self.aux = aux
        self.sm = LinearSystematicsModel(aux, unc)
        self.exp_time = np.median(np.diff(time))
        self.tm = init_batman(self.t, self.init_params, exp_time=self.exp_time)
        self.priors = init_params['priors'] if 'priors' in init_params.keys() else {}
        self.map_par = None
        self.has_run = False

    @property
    def args(self):
        args = [self.t, self.f, self.tm, self.sm, self.priors]
        return args

    @property
    def t14(self):
        if self.map_par is not None:
            p, r, k, b = [self.map_par[i].value for i in ['p','r','k','b']]
            a = arstar(p, r)
            t14 = t14_circ(p, a, k, b)
            return t14

    @property
    def parameter_names(self):
        return list(self.map_par.keys()) if self.map_par is not None else None

    def log_probability(self, par):
        return logprob_lm(par, *self.args)

    def fit_map(self, lm_prefit=True, verbose=True, guess_t0=False):

        init_params = self.init_params
        par = get_par(get_theta(init_params), self.sm)
        if guess_t0:
            par['t0'].value = self.args[0].mean()
        args = self.args
        lm_logprob = -np.inf

        if lm_prefit:

            par['k'].vary = False
            par['r'].vary = False
            par['ls'].vary = False
            par['q1'].vary = False
            par['q2'].vary = False
            par['p'].vary = False

            res = lmfit.minimize(residual_lm, par, args=args[:4])
            if res.success:
                self.map_par = res.params.copy()
                print("Initial L-M least squares fit successful")
                if verbose:
                    print(lmfit.report_fit(res.params, show_correl=False))
                print("Transit depth: {0:.0f} [ppm]".format(res.params['k'].value ** 2 * 1e6))
                print("Transit duration: {0:.2f} [h]".format(self.t14 * 24))
                par = res.params.copy()
                lm_logprob = logprob_lm(par, *args)
                print("Log-probability: {}".format(lm_logprob))
            else:
                print("Initial L-M least squares fit failed")
            par['t0'].vary = False
            par['ls'].vary = True

        mini = lmfit.Minimizer(lambda *x : -logprob_lm(*x), par, fcn_args=args, nan_policy='propagate')
        try:
            res = mini.minimize(method='nelder-mead')
        except:
            print("Nelder-Mead failed, attempting Powell minimization")
            res = mini.minimize(method='powell')

        if verbose:
            print(res.success)
            print(lmfit.report_fit(res.params))
            print("Transit depth: {0:.0f} [ppm]".format(res.params['k'].value ** 2 * 1e6))
            print("Transit duration: {0:.2f} [h]".format(self.t14 * 24))
            map_logprob = logprob_lm(res.params, *args)
            print("Log-probability: {}".format(lm_logprob))
        if not res.success:
            print("WARNING: fit unsuccessful")
            self.map_par = par
        else:
            self.map_par = res.params.copy()

        self.map_par['t0'].vary = True
        self.map_par['k'].vary = True
        self.map_par['r'].vary = True
        self.map_par['q1'].vary = True
        self.map_par['q2'].vary = True
        self.map_par['p'].vary = True

    @timeit
    def fit_mcmc(self, steps=1000, nwalkers=100, two_stage=False, nproc=1, use_priors='all',
        verbose=True, vary_per=True):

        if self.map_par is not None:
            par = self.map_par.copy()
        else:
            par = get_par(get_theta(self.init_params), self.sm)

        if not vary_per:
            par['p'].vary = False

        theta = [v for k,v in par.items() if par[k].vary]
        ndim = len(theta)

        pos0 = sample_ball(theta, [1e-4]*ndim, nwalkers)

        args = copy.deepcopy(self.args)
        if use_priors == 'none':
            args[-1] = {}
        elif type(use_priors) is list or type(use_priors) is tuple:
            for prior in 'ld per t0 rho'.split():
                if prior not in use_priors:
                    args[-1].pop(prior)
        elif type(use_priors) == dict:
            args[-1] = use_priors

        mini = lmfit.Minimizer(logprob_lm, par, fcn_args=args)
        if two_stage:
            print("Running stage 1 MCMC (250 steps)...")
            res = mini.emcee(burn=0, steps=250, thin=1, pos=pos0, workers=nproc)
            highest_prob = np.argmax(res.lnprob)
            hp_loc = np.unravel_index(highest_prob, res.lnprob.shape)
            theta = res.chain[hp_loc]
            pos0 = sample_ball(theta, [1e-5]*ndim, nwalkers)

        print("Running production MCMC for {} steps...".format(steps))
        res = mini.emcee(burn=0, steps=steps, thin=1, pos=pos0, workers=nproc)
        self.res_mcmc = res

        highest_prob = np.argmax(res.lnprob)
        hp_loc = np.unravel_index(highest_prob, res.lnprob.shape)
        self.map_soln = res.chain[hp_loc]

        par = res.params.copy()
        par_vary = [k for k,v in par.items() if par[k].vary]
        for k,v in zip(par_vary, self.map_soln):
            par[k].set(v)

        self.map_par = par
        nbv = self.sm.nbv
        self.sm.parameter_vector = self.map_soln[-nbv:]

        if not self.map_par['p'].vary:
            par = self.map_par.copy()
            par_names = list(par.valuesdict().keys())
            idx = par_names.index('p')
            self.best = list(self.map_soln[:idx]) + [par['p'].value] + list(self.map_soln[idx:])
        else:
            self.best = self.map_soln

        if verbose:
            print(lmfit.report_fit(res.params, show_correl=False))

    def burn_thin(self, burn=500, thin=10):

        chain = self.res_mcmc.chain
        nwalkers, nsteps, ndim = chain.shape
        self.fc = chain[:,burn::thin,:].reshape(-1, ndim)

    def plot_trace(self, fp=None):

        par_names = 'T_0,P,Rp/R_\star,\\rho_\star,b,q_1,q_2,log(\sigma)'.split(',')
        if not self.map_par['p'].vary:
            par_names.pop(par_names.index('P'))
        par_names += self.sm.parameter_names
        par_names = [r'${}$'.format(i) for i in par_names]

        plot_trace(self.res_mcmc.chain, par_names, plot_idx=[0,1,2,3,4,5,6], fp=fp)

    def plot_corner(self, fp=None):

        fc = self.fc
        best = self.best
        par_names = 'T_0,P,Rp/R_\star,\\rho_\star,b,q_1,q_2,log(\sigma)'.split(',')

        if not self.map_par['p'].vary:
            idx = par_names.index('P')
            par_names.pop(idx)
            best.pop(idx)

        nmu = len(par_names)
        par_names += self.sm.parameter_names
        par_names = [r'${}$'.format(i) for i in par_names]

        plot_corner(fc[:,:nmu], best[:nmu], par_names[:nmu], fp=fp)

    def plot_basic(self, auxnames=None, fp=None):
        if auxnames is None:
            auxnames = self.sm.parameter_names
        plot_basic(self.t, self.f, self.aux, auxnames, fp=fp)

    def plot_map(self, c1='C9', c2='C1', plot_binned=True, binsize=8, sig=2, fp=None, axs=None):

        best = self.map_par
        t = self.t
        f = self.f
        sm = self.sm
        init_params = self.init_params
        plot_map(best, t, f, sm, init_params, plot_binned=plot_binned, binsize=binsize, sig=sig, fp=fp, axs=axs, c1=c1, c2=c2)

    def plot_mcmc(self, c1='C9', c2='C1', plot_binned=True, binsize=8, sig=2, nsamples=1000, fp=None, axs=None):

        fc = self.fc
        if not self.map_par['p'].vary:
            par = self.map_par.copy()
            par_names = list(par.valuesdict().keys())
            idx = par_names.index('p')
            p_s = np.repeat(self.map_par['p'].value, fc.shape[0])
            fc = np.c_[fc[:,:idx], p_s, fc[:,idx:]]
        best = list(self.map_par.valuesdict().values())
        t = self.t
        f = self.f
        sm = self.sm
        init_params = self.init_params
        plot_mcmc(fc, best, t, f, sm, init_params, plot_binned=plot_binned, binsize=binsize, sig=sig, nsamples=nsamples, fp=fp, axs=axs, c1=c1, c2=c2)

    def write_mcmc(self, basefp=''):

        par_names = self.parameter_names
        fc = self.fc

        df = pd.DataFrame(OrderedDict(zip(par_names, fc.T)))
        fp = '{}-mcmc-samples.csv.gz'.format(basefp)
        df.to_csv(fp, index=False, compression='gzip')

        d = OrderedDict()
        cols = par_names

        for name,col in zip(par_names, cols):
            a, b, c = df[name].quantile([0.16,0.5,0.84])
            keys = [col, col+'_lo', col+'_hi']
            vals = [b, b-a, c-b]
            for k,v in zip(keys, vals):
                d[k] = v
        df = pd.DataFrame(d, index=[self.name])
        fp = '{}-mcmc-results.csv'.format(basefp)
        df.to_csv(fp)

        df = pd.DataFrame(self.best_dict, index=[self.name])
        fp = '{}-mcmc-bestfit.csv'.format(basefp)
        df.to_csv(fp)

    def get_df(self, rstar, rstar_sig):

        par_names = list(self.map_par.valuesdict().keys())
        fc = self.fc
        if not self.map_par['p'].vary:
            par = self.map_par.copy()
            par_names = list(par.valuesdict().keys())
            idx = par_names.index('p')
            p_s = np.repeat(self.map_par['p'].value, fc.shape[0])
            fc = np.c_[fc[:,:idx], p_s, fc[:,idx:]]
        df = pd.DataFrame(OrderedDict(zip(par_names, fc.T)))
        df['rstar'] = np.random.randn(df.shape[0]) * rstar_sig + rstar
        df['pl_rad'] = df['k'] * df['rstar'] * (u.Rsun / u.Rearth).to(u.dimensionless_unscaled)
        df['a'] = df['p r'.split()].apply(lambda x: arstar(*x), axis=1)
        df['inc'] = df['a b'.split()].apply(lambda x: inclination(*x), axis=1)
        df['t14'] = df['p a k b'.split()].apply(lambda x: t14_circ(*x), axis=1)
        df['t23'] = df['p a k b'.split()].apply(lambda x: t23_circ(*x), axis=1)
        df['shape'] = df['t23'] / df['t14']
        df['max_k'] = max_k(df['shape'])
        # df['rhostar'] = df['p a'.split()].apply(lambda x: rhostar(*x), axis=1)
        return df
