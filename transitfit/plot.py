import numpy as np
from scipy import stats
import matplotlib.pyplot as pl
import corner

from .model import model_lm, model_tra_lm, model_sys_lm
from .model import model, model_tra, model_sys
from .util import binned, init_batman


def plot_trace(chain, names, burn=None, plot_idx=None, fp=None):

    if burn is not None:
        chain = chain[:,burn:,:].copy()

    if plot_idx is not None:
        chain = chain[:,:,plot_idx].copy()

    nwalkers, nsteps, ndim = chain.shape

    fig, axes = pl.subplots(ndim, 1, figsize=(10, ndim*1), sharex=True)
    for i in range(ndim):
        ax = axes[i]
        ax.plot(chain.T[i], "k", alpha=0.05)
        ax.set_xlim(0, nsteps)
        ax.set_ylabel(names[i], fontsize=16)
        ax.yaxis.set_label_coords(-0.1, 0.5)

    fig.subplots_adjust(hspace=0)
    if fp is not None:
        fig.savefig(fp, dpi=200)
        pl.close()

def plot_corner(fc, best, names, fp=None):

    hist_kwargs = dict(lw=1, alpha=1)
    title_kwargs = dict(fontdict=dict(fontsize=12))
    data_kwargs = dict(alpha=0.01)
    quantiles = 0.16, 0.5, 0.84

    corner.corner(fc, labels=names, truths=best,
                  quantiles=quantiles,
                  hist_kwargs=hist_kwargs,
                  title_kwargs=title_kwargs,
                  data_kwargs=data_kwargs,
                  smooth=1, smooth1d=1,
                  show_titles=True,
                  title_fmt='.4f')

    if fp is not None:
        pl.savefig(fp, dpi=200)
        pl.close()

def plot_basic(t, f, aux, auxnames, fp=None):

    naux = aux.shape[1]
    fig, axs = pl.subplots(naux+1, 1, figsize=(10,2*naux), sharex=True)

    ax = axs[0]
    ax.plot(t, f, 'k-')

    for i,vec in enumerate(aux.T):
        ax = axs[i+1]
        ax.plot(t, vec, 'k-')
        ax.set_ylabel(auxnames[i])

    pl.setp(ax, xlim=(t.min(), t.max()))

    if fp is not None:
        fig.savefig(fp)
        pl.close()

def plot_map(par, t, f, sm, init_params, plot_binned=True, binsize=16, sig=2, c1='C9', c2='C1',
    timeoffset=2450000, nmodel=300, fp=None, axs=None):

    data_color = '0.6'
    data_alpha = 0.5
    data_ms = 2
    data_lw = 0.5
    cr_alpha = 0.4

    if axs is None:
        fig, axs = pl.subplots(2, 1, figsize=(10,6), sharex=True, sharey=False)

    tmd = init_batman(t, init_params, exp_time=np.median(np.diff(t)))

    ti = np.linspace(t.min(), t.max(), nmodel)
    tmi = init_batman(ti, init_params, exp_time=np.median(np.diff(t)))

    ax = axs[0]
    ax.plot(t, f, color=data_color, lw=data_lw, zorder=0)
    ax.plot(t, model_lm(par, t, tmd, sm), color=c1, lw=1, label='transit+systematics')
    ax.xaxis.get_major_formatter().set_useOffset(timeoffset)
    ax.xaxis.offsetText.set_visible(False)
    ax.legend()

    ax = axs[1]
    sys = model_sys_lm(par, sm)
    ax.plot(t, f-sys, color=data_color, lw=data_lw, zorder=0)
    if plot_binned:
        t_b = binned(t, binsize)
        f_b = binned(f-sys, binsize)
        ax.plot(t_b, f_b, marker='o', ms=3, linestyle='none', color='k', alpha=0.4)
    ax.plot(ti, model_tra_lm(par, ti, tmi), color=c2, lw=1, label='transit')
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

def plot_mcmc(fc, best, t, f, sm, init_params, plot_binned=True, binsize=16, sig=2, c1='C9', c2='C1',
    timeoffset=2450000, nmodel=300, nsamples=10000, fp=None, axs=None):

    # cdfv = stats.norm.cdf([-3, 3, -2, 2, -1, 1]) * 100
    # cdfv = stats.norm.cdf([-2, 2, -1, 1]) * 100
    cdfv = stats.norm.cdf([-sig, sig]) * 100
    data_color = '0.6'
    data_alpha = 0.5
    data_ms = 2
    data_lw = 0.5
    cr_alpha = 0.4

    tmd = init_batman(t, init_params, exp_time=np.median(np.diff(t)))

    ti = np.linspace(t.min(), t.max(), nmodel)
    tmi = init_batman(ti, init_params, exp_time=np.median(np.diff(t)))

    if axs is None:
        fig, axs = pl.subplots(2, 1, figsize=(10,6), sharex=True, sharey=False)

    ax = axs[0]
    ax.plot(t, f, color=data_color, lw=data_lw, zorder=0)
    ms = [model(s, t, tmd, sm) for s in fc[np.random.randint(0, fc.shape[0], nsamples)]]
    pctl = np.percentile(ms, cdfv, axis=0)
    for i in range(int(cdfv.size/2)):
        lo, hi = pctl[i:i+2]
        ax.fill_between(t, lo, hi, color=c1, alpha=cr_alpha, lw=0)
    ax.plot(t, model(best, t, tmd, sm), color=c1, lw=1, label='transit+systematics')
    ax.xaxis.get_major_formatter().set_useOffset(timeoffset)
    ax.xaxis.offsetText.set_visible(False)
    ax.legend()

    ax = axs[1]
    sys = model_sys(best, t, sm)
    ax.plot(t, f-sys, color=data_color, lw=data_lw, zorder=0)
    ms = [model_tra(s, ti, tmi) for s in fc[np.random.randint(0, fc.shape[0], nsamples)]]
    pctl = np.percentile(ms, cdfv, axis=0)
    if plot_binned:
        t_b = binned(t, binsize)
        f_b = binned(f-sys, binsize)
        u_b = np.exp(best[7]) / np.sqrt(binsize)
        ax.errorbar(t_b, f_b, u_b, marker='o', ms=3, linestyle='none', color='k', alpha=0.4)
    for i in range(int(cdfv.size/2)):
        lo, hi = pctl[i:i+2]
        ax.fill_between(ti, lo, hi, color=c2, alpha=cr_alpha, lw=0)
    ax.plot(ti, model_tra(best, ti, tmi), color=c2, lw=1, label='transit')
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
