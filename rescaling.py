import numpy as np
import config
from scipy.stats import truncnorm
import math


def generate_normal_real(d, cl,l):
    var_cl = np.array([cl/2 for _ in range(2*l+1)])
    var_cl[0] = cl
    var_noise = np.array([config.noise_covar/2 for _ in range(2*l+1)])
    var_noise[0] = config.noise_covar

    var = (1/((1/var_cl)+(1/var_noise)))
    mean_normal = var*(1/var_noise)*d
    fluc_map = np.sqrt(var)*np.random.normal(size=2*l+1)
    gen_map = fluc_map + mean_normal
    return gen_map, mean_normal, fluc_map


def propose_cl(cl_old):
    clip_low = -cl_old/np.sqrt(config.variance_rescale_prop)
    cl_new = truncnorm.rvs(a=clip_low, b=np.inf, loc=cl_old, scale=np.sqrt(config.variance_rescale_prop))
    return cl_new


def compute_log_proposal(cl_old, cl_new):
    clip_low = -cl_old / np.sqrt(config.variance_rescale_prop)
    return truncnorm.logpdf(cl_new, a=clip_low, b=np.inf, loc=cl_old, scale=np.sqrt(config.variance_rescale_prop))


def compute_log_likelihood(weiner_map, fluctuation_map, cosmo_variance, noise_variance, d):
    return -(1/2)*(np.sum(((d-weiner_map)**2)/noise_variance) + np.sum(((weiner_map)**2)/cosmo_variance)
                   + np.sum(((fluctuation_map)**2)/noise_variance))


def compute_log_MH_ratio(weiner_map_old, weiner_map_new, fluctuation_map_old, fluctuation_map_new, cosmic_var_old, cosmic_var_new,
                         noise_var, cl_new, cl_old, d):
    log_ratio = compute_log_likelihood(weiner_map_new, fluctuation_map_new, cosmic_var_new, noise_var, d) \
    - compute_log_likelihood(weiner_map_old, fluctuation_map_old, cosmic_var_old, noise_var, d) \
    + compute_log_proposal(cl_new, cl_old) - compute_log_proposal(cl_old, cl_new)

    #print("Log_ratio:")
    #print(log_ratio)
    return log_ratio


def metropolis(cl_old, weiner_map, fluctuation_map, d, l):
    old_signal_var = np.array([cl_old/2 for _ in range(2*l+1)])
    old_signal_var[0] = cl_old
    var_noise = np.array([config.noise_covar/2 for _ in range(2*l+1)])
    var_noise[0] = config.noise_covar
    cl_new = propose_cl(cl_old)
    new_signal_var = np.array([cl_new/2 for _ in range(2*l+1)])
    new_signal_var[0] = cl_new
    new_var = 1/((1/new_signal_var) + (1/var_noise))
    new_fluctuation_map = np.sqrt(cl_new/cl_old)*fluctuation_map
    new_weiner_map = new_var*(1/var_noise)*d

    log_r = compute_log_MH_ratio(weiner_map, new_weiner_map, fluctuation_map, new_fluctuation_map, old_signal_var, new_signal_var,
                                 var_noise, cl_new, cl_old, d)

    accept = 0
    if np.log(np.random.uniform()) < log_r:
        cl_old = cl_new
        weiner_map = new_weiner_map
        fluctuation_map = new_fluctuation_map
        accept += 1

    return cl_old, weiner_map, fluctuation_map, accept


def gibbs_rescale(cl_init, d, l):
    cl = cl_init
    total_accept = 0
    h_cl = []
    h_s = []
    for i in range(config.N_rescale):
        s, weiner, fluctuation = generate_normal_real(d, cl, l)
        cl, weiner, fluctuation, accept = metropolis(cl, weiner, fluctuation, d, l)

        h_cl.append(cl)
        h_s.append(weiner + fluctuation)
        total_accept += accept
        #print("\n")

    acceptance_rate = total_accept/config.N_rescale
    print(acceptance_rate)
    return h_cl, h_s, acceptance_rate



