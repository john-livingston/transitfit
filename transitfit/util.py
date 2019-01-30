import time
import numpy as np
from astropy import constants as c
from astropy import units as u
import lmfit
from batman import TransitParams, TransitModel
import pandas as pd
from collections import OrderedDict

def timeit(method):
    """
    Decorator for timing method calls
    """
    def timed(*args, **kw):

        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        name = method.__name__.upper()
        print('{0} elapsed time: {1:.0f} sec'.format(name, te-ts))
        return result

    return timed

def mad_outliers(x, hi=5, lo=5):
    M = np.nanmedian(x)
    MAD = 1.4826 * np.nanmedian(np.abs(x - M))
    out = (x > M + hi * MAD) | (x < M - lo * MAD)
    return out

def bin_data(data, binsize_min):
    cols = ['c{}'.format(i) for i in range(data.shape[1])]
    df = pd.DataFrame(OrderedDict(zip(cols,data.T)))

    binsize = binsize_min / (60 * 24)
    bins = np.arange(df['c0'].min(), df['c0'].max(), binsize)
    groups = df.groupby(np.digitize(df['c0'], bins))
    df_binned = groups.mean().copy()
    return df_binned.values

def rho(logg, r):
    r = (r * u.R_sun).cgs
    g = 10 ** logg * u.cm / u.s ** 2
    rho = 3 * g / (r * c.G.cgs * 4 * np.pi)
    return rho.value

def q_to_u(q1, q2):
    """
    Maps limb-darkening q space to u space
    """
    u1 = 2 * np.sqrt(q1) * q2
    u2 = np.sqrt(q1) * (1 - 2*q2)
    return u1, u2

def u_to_q(u1, u2):
    """
    Maps limb-darkening u space to q space
    """
    q1 = (u1 + u2)**2
    q2 = u1 / (2 * (u1 + u2))
    return q1, q2

def inclination(a, b):
    """
    (a, b) --> inclination in degrees
    """
    return np.rad2deg(np.arccos(b / a))

def scaled_a(p, t14, k, i=90, b=0):
    """
    Computes the scaled semi-major axis (a/R*)
    """
    numer = np.sqrt( (k + 1)**2 - b**2 )
    denom = np.sin(np.deg2rad(i)) * np.sin(t14 * np.pi / p)
    return float(numer / denom)

def t14_circ(p, a, k, b):
    """
    Winn 2014 ("Transits and Occultations"), eq. 14
    """
    i = inclination(a, b)
    alpha = np.sqrt( (1 + k)**2 - b**2 )
    return (p / np.pi) * np.arcsin( alpha / np.sin(np.deg2rad(i)) / a )

def t23_circ(p, a, k, b):
    """
    Winn 2014 ("Transits and Occultations"), eq. 15
    """
    i = inclination(a, b)
    alpha = np.sqrt( (1 - k)**2 - b**2 )
    return (p / np.pi) * np.arcsin( alpha / np.sin(np.deg2rad(i)) / a )

def rhostar(p, a):
    """
    Mean stellar density, assuming a circular orbit.
    From eq.4 of http://arxiv.org/pdf/1311.1170v3.pdf.

    p : period [days]
    a : a/Rstar [dimensionless]

    Returns : mean stellar density [cgs]
    """
    Gcgs = 6.67408e-08
    ps = p * 86400
    rhostar = 3 * np.pi * a ** 3 / (Gcgs * ps ** 2)
    return rhostar

def arstar(p, rho):
    """
    Scaled orbital distance, assuming a circular orbit.
    From eq.4 of http://arxiv.org/pdf/1311.1170v3.pdf.

    p : period [days]
    rho : mean stellar density [cgs]

    Returns : a/Rstar [dimensionless]
    """
    Gcgs = 6.67408e-08
    ps = p * 86400
    arstar = (Gcgs * ps ** 2 * rho / (3 * np.pi)) ** (1/3.)
    return arstar

def max_k(tshape):
    """
    Seager & Mallen-Ornelas 2003, eq. 21
    """
    return (1 - tshape) / (1 + tshape)

def binned(a, binsize, fun=np.mean):
    """
    Simple binning function
    """
    return np.array([fun(a[i:i+binsize], axis=0) \
        for i in range(0, a.shape[0], binsize)])

def init_batman(t, init_params, exp_time=0.000694):

    a = inclination(init_params['p'], init_params['r'])
    i = inclination(a, init_params['b'])
    params = TransitParams()
    params.t0 = init_params['t0']     #time of inferior conjunction
    params.per = init_params['p']     #orbital period
    params.rp = init_params['k']      #planet radius (in units of stellar radii)
    params.a = a       #semi-major axis (in units of stellar radii)
    params.inc = i               #orbital inclination (in degrees)
    params.ecc = 0.                         #eccentricity
    params.w = 90.                          #longitude of periastron (in degrees)
    u1, u2 = q_to_u(init_params['q1'], init_params['q2'])
    params.u = [u1, u2]                     #limb darkening coefficients
    params.limb_dark = "quadratic"

    tm = TransitModel(params, t, supersample_factor=1, exp_time=exp_time)
    return tm

def get_init_params(per, t0, t14, rprs, rho=None, b=0.5, q1=0.5, q2=0.5):
    init_params = {}
    init_params['p'] = per
    init_params['t0'] = t0
    init_params['t14'] = t14
    init_params['k'] = rprs
    init_params['r'] = rho if rho is not None else 1.41
    init_params['b'] = b
    init_params['q1'] = q1
    init_params['q2'] = q2
    return init_params

def get_par(theta, sm):
    names = 't0 p k r b q1 q2 ls'.split()
    bounds = [
        (-np.inf,np.inf), #T0
        (0,np.inf), #P
        (0,1), #Rp/Rs
        (0,np.inf), #rhostar
        (0,1+theta[2]), #b
        (0,1), #q1
        (0,1), #q2
        (-10, 0) #log(sigma)
    ]
    par = lmfit.Parameters()
    for na,th,bo in zip(names, theta, bounds):
        par.add(na, th, min=bo[0], max=bo[1], vary=True)
    for na in sm.parameter_names:
        par.add(na, 0, vary=True)
    return par

def get_theta(params_dict):
    t0 = params_dict['t0']     #time of inferior conjunction
    p = params_dict['p']     #orbital period
    k = params_dict['k']      #planet radius (in units of stellar radii)
    r = params_dict['r']       #mean stellar density (cgs)
    b = params_dict['b']
    q1 = params_dict['q1']
    q2 = params_dict['q2']
    ls = params_dict['ls']
    return [t0,p,k,r,b,q1,q2,ls]
