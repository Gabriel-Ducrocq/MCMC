import numpy as np
import config
from scipy.stats import truncnorm


def sample_surrogate(var_latent, var_surrogate, l):
    return np.sqrt(var_latent+var_surrogate)*np.random.normal(size=2*l+1)


def sample_latent(surrogate, var_latent, var_surrogate, l):
    r = 1/((1/var_latent) + (1/var_surrogate))
    m = r*(1/var_surrogate)*surrogate
    return np.sqrt(r)*np.random.normal(size=2*l+1) + m



