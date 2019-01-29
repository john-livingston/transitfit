import numpy as np
from scipy import stats
import lmfit
from batman import TransitParams, TransitModel

from .util import inclination, q_to_u, t14_circ, arstar

params = TransitParams()
params.limb_dark = "quadratic"


def model_tra_lm(par, t, tm):
    u1, u2 = q_to_u(par['q1'], par['q2'])
    a = arstar(par['p'], par['r'])
    i = inclination(a, par['b'].value)
    params.t0 = par['t0'] #time of inferior conjunction
    params.per = par['p'] #orbital period
    params.rp = par['k']  #planet radius (in units of stellar radii)
    params.a = a   #semi-major axis (in units of stellar radii)
    params.inc = i        #orbital inclination (in degrees)
    params.ecc = 0.       #eccentricity
    params.w = 90.        #longitude of periastron (in degrees)
    params.u = [u1, u2]   #limb darkening coefficients
    return tm.light_curve(params)

def model_sys_lm(par, sm):
    theta = [par.get(i) for i in sm.parameter_names]
    sm.parameter_vector = theta
    sys = sm.model
    return sys

def model_lm(par, t, TM, sm):
    return model_tra_lm(par, t, TM) + model_sys_lm(par, sm)

def residual_lm(par, t, f, TM, sm):
    m = model_lm(par, t, TM, sm)
    return f - m

def logprior_lm(par, priors={}):

    t0,p,k,r,b,q1,q2 = [par.get(i) for i in 't0 p k r b q1 q2'.split()]

    if p <= 0 or \
        q1 <= 0 or q1 >= 1 or \
        q2 <= 0 or q2 >= 1 or \
        r <= 0 or \
        b <= 0 or b > 1+k or \
        k < 0 or k > 1:
        return -np.inf

    lp = 0
    if len(priors) > 0:
        if 'ld' in priors.keys():
            # TODO: use q-space only if no limb-darkening priors are supplied?
            ldp = priors['ld']
            u1, u2 = q_to_u(q1, q2) # TODO: put priors on q1/q2 instead of u1/u2 (use new limbdark 'transform' option)
            lp += np.log(stats.norm.pdf(u1, loc=ldp[0], scale=ldp[1]))
            lp += np.log(stats.norm.pdf(u2, loc=ldp[2], scale=ldp[3]))
        if 'rho' in priors.keys():
            rhop = priors['rho']
            lp += np.log(stats.norm.pdf(r, loc=rhop[0], scale=rhop[1]))
        if 'per' in priors.keys():
            pp = priors['per']
            lp += np.log(stats.norm.pdf(p, loc=pp[0], scale=pp[1]))
        if 't0' in priors.keys():
            t0p = priors['t0']
            lp += np.log(stats.norm.pdf(t0, loc=t0p[0], scale=t0p[1]))
        if 't14' in priors.keys():
            a = arstar(p, r)
            t14 = t14_circ(p, a, k, b)
            t14p = priors['t14']
            lp += np.log(stats.norm.pdf(t14, loc=t14p[0], scale=t14p[1]))

    return lp

# ===================
# FIXME: still used by some plotting routines
# ===================

def model_tra(theta, t, tm, fix_p=False):
    t0,p,k,r,b,q1,q2 = theta[:7]
    if fix_p:
        p = fix_p
    a = arstar(p, r)
    i = inclination(a, b)
    u1, u2 = q_to_u(q1, q2)
    params.t0 = t0                       #time of inferior conjunction
    params.per = p                      #orbital period
    params.rp = k                      #planet radius (in units of stellar radii)
    params.a = a                       #semi-major axis (in units of stellar radii)
    params.inc = i                     #orbital inclination (in degrees)
    params.ecc = 0.                      #eccentricity
    params.w = 90.                      #longitude of periastron (in degrees)
    params.u = [u1, u2]                #limb darkening coefficients
    return tm.light_curve(params)

def model_sys(theta, t, sm):

    sm.parameter_vector = theta[8:]
    sys = sm.model

    return sys

def model(theta, t, tm, pld):
    return model_tra(theta, t, tm) + model_sys(theta, t, pld)

# ===================

def logprob_lm(par, t, f, TM, sm, priors):
    resid = residual_lm(par, t, f, TM, sm)
    s = np.exp(par['ls'])
    resid *= 1 / s
    resid *= resid
    resid += np.log(2 * np.pi * s**2)
    lnlike = -0.5 * np.sum(resid)
    logprob = lnlike + logprior_lm(par, priors)
    if np.isfinite(logprob):
        return logprob
    else:
        return -np.inf
