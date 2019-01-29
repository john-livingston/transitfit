import numpy as np

class LinearSystematicsModel(object):

    def __init__(self, aux, unc=None, parameter_names=None):

        """
        Basic linear systematics model.

        Args:
            aux: auxiliary vectors (shape: (Ntime,Nvec))
            unc: (optional) array of flux uncertainties (shape: (Ntime))
                Note: a jitter term is required if these are not supplied
        """

        self._X = aux.copy()

        if parameter_names is None:
            self._pn = ['c{}'.format(i) for i in range(aux.shape[1])]
        else:
            self._pn = parameter_names

        self._pv = np.zeros(self._X.shape[1])
        self._unc = unc if unc is not None else np.zeros(aux.shape[0])

    @property
    def nbv(self):
        return len(self._pv)

    @property
    def parameter_vector(self):
        return self._pv

    @parameter_vector.setter
    def parameter_vector(self, value):
        self._pv = value

    @property
    def parameter_names(self):
        return self._pn

    @parameter_names.setter
    def parameter_names(self, value):
        self._pn = value

    @property
    def model(self):
        return np.dot(self._X, self._pv)

    def log_likelihood(self, residuals, jitter=None):
        r = self.model - residuals
        s = np.sqrt(self._unc**2 + jitter**2) if jitter is not None else self._unc
        r *= 1 / s
        r *= r
        r += np.log(2 * np.pi * s**2)
        loglike = -0.5 * np.sum(r)
        if not np.isfinite(loglike):
            return -np.inf
        return loglike
