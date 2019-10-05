import numpy as np
import config
import healpy as hp
import matplotlib.pyplot as plt
import gradientDescent
import utils


def compute_log_likelihood(x_pix, observations):
    return -(1/2)*np.sum(((observations - x_pix)**2)*(1/config.noise_covar))


def compute_CN_ratio(x_pix, y_pix, observations):
    return compute_log_likelihood(y_pix, observations) - compute_log_likelihood(x_pix, observations)


def crankNicolson(cls_, observations):
    accept = 0
    history = []
    s = hp.synalm(cls_, lmax=config.L_MAX_SCALARS)
    #h, s = gradientDescent.gradient_ascent(observations, cls_)
    s_pixel = hp.sphtfunc.alm2map(s, nside=config.NSIDE)
    for i in range(config.N_CN):
        history.append(s)
        prop = np.sqrt(1-config.beta_CN**2)*s + config.beta_CN*hp.sphtfunc.synalm(cls_, lmax=config.L_MAX_SCALARS)
        prop_pix = hp.sphtfunc.alm2map(prop, nside=config.NSIDE)
        r = compute_CN_ratio(s_pixel, prop_pix, observations)
        if np.log(np.random.uniform()) < r:
            s = prop
            s_pixel = prop_pix
            accept += 1

    print(accept/config.N_CN)
    return history, s



#### Second

def flatten_map(s):
    s_real = s.real[[i for i in range(len(s)) if i != 0 and i != 1 and i != (config.L_MAX_SCALARS+1)]]
    s_imag = s.imag[[i for i in range((config.L_MAX_SCALARS+2),len(s))]]
    s_flatten = np.concatenate((s_real, s_imag))
    return s_flatten


def unflat_map_to_pix(s):
    real_part = np.concatenate((np.zeros(2), s[:(config.L_MAX_SCALARS-1)] , np.zeros(1),
                                 s[(config.L_MAX_SCALARS-1):(config.dimension_sph-3)]))

    imag_part = np.concatenate((np.zeros(config.L_MAX_SCALARS+2), s[(config.dimension_sph-3):]))
    return real_part + 1j*imag_part


def extend_cls(cls):
    extended_cls = [cl for l in range(config.L_MAX_SCALARS+1) for cl in cls[l:]]
    extended_cls_real = (extended_cls[:(config.L_MAX_SCALARS+1)] + extended_cls[(config.L_MAX_SCALARS+2):])[2:]
    extended_cls_imag = extended_cls[(config.L_MAX_SCALARS+2):]
    extended_cls = extended_cls_real + extended_cls_imag
    return np.array(extended_cls)


def compute_log_likelihood2(s_pix, d):
    return -(1/2)*np.sum(((d-s_pix)**2)/config.noise_covar)


def compute_CN_ratio2(s_pix, s_pix_prop, d):
    return compute_log_likelihood2(s_pix_prop, d) - compute_log_likelihood2(s_pix, d)


def crankNicolson2(cls_, d):
    #s = hp.synalm(cls_, lmax=config.L_MAX_SCALARS)
    h, s = gradientDescent.gradient_ascent(d, cls_)
    s_pix = hp.sphtfunc.alm2map(s, nside=config.NSIDE)
    s = flatten_map(s)
    accepted = 0
    history = []
    for i in range(config.N_CN):
        history.append(s)
        s_prop = np.sqrt(1 - config.beta_CN**2)*s +config.beta_CN*flatten_map(hp.synalm(cls_, lmax=config.L_MAX_SCALARS))
        s_prop_pix = hp.alm2map(unflat_map_to_pix(s_prop), nside=config.NSIDE)
        r = compute_CN_ratio2(s_pix, s_prop_pix, d)
        if np.log(np.random.uniform()) < r:
            s = s_prop
            s_pix = s_prop_pix
            accepted += 1

    print(accepted/config.N_CN)
    return history, s



#### Good one

def compute_log_likelihood_good(s_pix, d):
    return -(1/2)*np.sum(((d-s_pix)**2)/config.noise_covar)


#def compute_log_likelihood_good(s, s_pix, d):
#    return (-1/2)*(np.dot(d.T, (1/config.noise_covar)*d)
#           - np.dot(s.T, utils.flatten_map(hp.map2alm((1/config.noise_covar)*d, lmax=config.L_MAX_SCALARS)))
#           - np.dot(d.T, (1/config.noise_covar)*s_pix)
#           + np.dot(s.T, utils.flatten_map(hp.map2alm((1/config.noise_covar)*s_pix, lmax=config.L_MAX_SCALARS))))


def compute_CN_ratio_good(s_pix, s_pix_prop, d):
    return compute_log_likelihood_good(s_pix_prop, d) - compute_log_likelihood_good(s_pix, d)


#def compute_CN_ratio_good(s, s_prop, s_pix, s_pix_prop, d):
#    return compute_log_likelihood_good(s_prop, s_pix_prop, d) - compute_log_likelihood_good(s, s_pix, d)



def crankNicolson_good(cls_, d):
    s = hp.synalm(cls_, lmax=config.L_MAX_SCALARS)
    s = utils.flatten_map(s)
    #h, s = gradientDescent.gradient_ascent_good(d, cls_)
    s_pix = hp.sphtfunc.alm2map(utils.unflat_map_to_pix(s), nside=config.NSIDE)
    #s = utils.flatten_map(s)
    accepted = 0
    history = []
    for i in range(config.N_CN):
        if i == 40000:
            config.beta_CN /= 9.5

        history.append(s)
        s_prop = np.sqrt(1 - config.beta_CN**2)*s +config.beta_CN*utils.flatten_map(hp.synalm(cls_, lmax=config.L_MAX_SCALARS))
        ### Ajouté le dénom !!!
        s_prop_pix = hp.alm2map(utils.unflat_map_to_pix(s_prop), nside=config.NSIDE)*(1/config.NSIDE)
        r = compute_CN_ratio_good(s_pix, s_prop_pix, d)
        if np.log(np.random.uniform()) < r:
            s = s_prop
            s_pix = s_prop_pix
            accepted += 1

    print(accepted/config.N_CN)
    return history, s
