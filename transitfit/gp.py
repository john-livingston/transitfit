import numpy as np
# import autograd.numpy as np
import pandas as pd
import celerite
from celerite.modeling import Model
from celerite import terms
from batman import TransitParams, TransitModel
from scipy import stats
import scipy.optimize as op
from functools import partial
from collections import OrderedDict
import matplotlib.pyplot as pl
from matplotlib import rcParams
rcParams["savefig.dpi"] = 200
rcParams["figure.dpi"] = 200
rcParams["xtick.direction"] = 'in'
rcParams["ytick.direction"] = 'in'

from .util import arstar, rhostar, inclination, q_to_u, fit_mcmc, binned
from .plot import plot_corner, plot_trace


class TransitMeanModel(Model):
    parameter_names = ('t0,p,k,r,b,q1,q2'.split(','))
    def get_value(self, t):
        return transit_model(self.get_parameter_vector(), t)

def neg_log_like(params, y, gp):
    gp.set_parameter_vector(params)
    return -gp.log_likelihood(y)

def grad_neg_log_like(params, y, gp):
    gp.set_parameter_vector(params)
    gll = gp.grad_log_likelihood(y)[1]
    return -gll

def transit_model(theta, t, exp_time=0.002):
    params = TransitParams()
    t0,p,k,r,b,q1,q2 = theta[:7]
    u1, u2 = q_to_u(q1, q2)
    a = arstar(p, r)
    i = inclination(a, b)
    params.t0 = t0                       #time of inferior conjunction
    params.per = p                      #orbital period
    params.rp = k                      #planet radius (in units of stellar radii)
    params.a = a                       #semi-major axis (in units of stellar radii)
    params.inc = i                     #orbital inclination (in degrees)
    params.ecc = 0.                      #eccentricity
    params.w = 90.                      #longitude of periastron (in degrees)
    params.u = [u1, u2]                #limb darkening coefficients
    params.limb_dark = "quadratic"
    m = TransitModel(params, t, supersample_factor=1, exp_time=exp_time)
    # m = TransitModel(params, t)
    return m.light_curve(params)

def log_prior(theta, ldp=None, pp=None, t0p=None, rhop=None, start_idx=0):
    t0,p,k,a,b,q1,q2 = theta[start_idx:start_idx+7]
    lp = 0
    if not 0 < k < 1:
        return -np.inf
    if a < 1:
        return -np.inf
    if not 0 <= b < 1+k:
        return -np.inf
    if not 0 < q1 < 1:
        return -np.inf
    if not 0 < q2 < 1:
        return -np.inf
    if ldp is not None:
        lp += np.log(stats.norm.pdf(q1, loc=ldp[0], scale=ldp[1]))
        lp += np.log(stats.norm.pdf(q2, loc=ldp[2], scale=ldp[3]))
    if rhop is not None:
        rho = rhostar(p, a)
        lp += np.log(stats.norm.pdf(rho, loc=rhop[0], scale=rhop[1]))
    if pp is not None:
        lp += np.log(stats.norm.pdf(p, loc=pp[0], scale=pp[1]))
    if t0p is not None:
        lp += np.log(stats.norm.pdf(t0, loc=t0p[0], scale=t0p[1]))
    if np.isnan(lp):
        lp = -np.inf
    return lp

def log_prob(theta, f, gp, ldp=None, rhop=None, pp=None, t0p=None, _log_prior=log_prior):
    gp.set_parameter_vector(theta)
    lp = gp.log_prior()
    lp += _log_prior(theta, ldp=ldp, rhop=rhop, pp=pp, t0p=t0p)
    if not np.isfinite(lp):
        return -np.inf
    return gp.log_likelihood(f) + lp

log_prior_gp_transit = partial(log_prior, start_idx=3)
log_prob_gp_transit = partial(log_prob, _log_prior=log_prior_gp_transit)


class GPTransitFit:

    def __init__(self, t, f, init_params):
        self.t = t
        self.f = f
        self.init_params = init_params

    def init_gp(self):

        bounds = [(-np.inf,np.inf), (0,np.inf), (0,1), (0,np.inf), (0,1), (0,1), (0,1)]
        t0,p,k,r,b,q1,q2 = [self.init_params.get(i) for i in 't0,p,k,r,b,q1,q2'.split(',')]
        mean_model = TransitMeanModel(t0=t0, p=p, k=k, r=r, b=b, q1=q1, q2=q2, bounds=bounds)
        lna, lnt, lns = -5, -3, np.log(self.f.std())
        kernel = terms.Matern32Term(log_sigma=lna, log_rho=lnt, bounds=[(-15,5),(-15,5)])
        kernel += terms.JitterTerm(log_sigma=lns, bounds=[(-10,0)])
        self.gp = celerite.GP(kernel, mean=mean_model, fit_mean=True)
        self.gp.compute(self.t)

    def fit_map(self):

        ini = self.gp.get_parameter_vector()
        args = self.f, self.gp
        neg_log_prob = lambda *x: -log_prob_gp_transit(*x)
        print(neg_log_prob(ini, *args))
        res = op.minimize(neg_log_prob, ini, args, method='nelder-mead')
        if not res.success:
            res = op.minimize(neg_log_prob, ini, args, method='powell')
        print(res.success)
        print(neg_log_prob(res.x, *args))

    def plot_map(self, plot_binned=True, binsize=10./60./24., sig=1, c1='C9', c2='C1',
    timeoffset=2450000, nmodel=300, fp=None, axs=None):

        t = self.t
        f = self.f

        data_color = '0.6'
        data_alpha = 0.5
        data_ms = 2
        data_lw = 0.5
        cr_alpha = 0.4

        if axs is None:
            fig, axs = pl.subplots(2, 1, figsize=(10,6), sharex=True, sharey=False)

        ti = np.linspace(t.min(), t.max(), nmodel)

        mu, var = self.gp.predict(f, t=ti, return_var=True)
        sig = np.sqrt(var)

        ax = axs[0]
        ax.plot(t, f, color=data_color, lw=data_lw, zorder=0)
        ax.plot(ti, mu, color=c1, lw=1, label='transit+systematics')
        ax.fill_between(ti, mu-sig, mu+sig, color=c1, alpha=0.5, lw=0)
        ax.xaxis.get_major_formatter().set_useOffset(timeoffset)
        ax.xaxis.offsetText.set_visible(False)
        ax.legend()

        ax = axs[1]
        mu_full, var = self.gp.predict(f, return_var=True)
        mu_tr = self.gp.mean.get_value(t)
        mu_sys = mu_full - mu_tr
        fcor = f - mu_sys

        mu_full, var = self.gp.predict(f, t=ti, return_var=True)
        sig = np.sqrt(var)
        mu_tr = self.gp.mean.get_value(ti)
        ax.plot(t, fcor, color=data_color, lw=data_lw, zorder=0)
        ax.plot(ti, mu_tr, color=c2, lw=1, label='transit')
        ax.fill_between(ti, mu_tr-sig, mu_tr+sig, color=c2, alpha=0.5, lw=0)
        # ax.plot(t, f-sys, color=data_color, lw=data_lw, zorder=0)
        if plot_binned:
            dfb = binned(t, fcor, binsize)
            x = dfb['x'].values
            y = dfb['y'].values
            yerr = dfb['stddev'].values
            ax.errorbar(x, y, yerr, marker='o', ms=3, linestyle='none', color='k', alpha=0.4)

        ax.xaxis.get_major_formatter().set_useOffset(timeoffset)
        ax.xaxis.offsetText.set_visible(False)
        ax.legend()

        pl.setp(axs, xlim=(t.min(), t.max()),
        xlabel=r'BJD$-${}'.format(timeoffset),
        ylabel='Relative flux')
        axs[0].get_figure().subplots_adjust(hspace=0)

        if fp is not None:
            axs[0].get_figure().savefig(fp, dpi=200)
            pl.close()
