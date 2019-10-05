import numpy as np
from scipy.stats import truncnorm
import config
import healpy as hp



def generate_normal_real(d, cl,l):
    var_cl = np.array([cl/2 for _ in range(2*l+1)])
    var_cl[0] = cl
    var_noise = np.array([config.noise_covar/2 for _ in range(2*l+1)])
    var_noise[0] = config.noise_covar

    var = 1/(1 + var_cl/var_noise)
    mean_normal = var*np.sqrt(var_cl)*(1/var_noise)*d
    gen_map = np.sqrt(var) * np.random.normal(size=2 * l + 1) + mean_normal
    return gen_map


def propose_cl(cl_old):
    clip_low = -cl_old/np.sqrt(config.var_NCGibbs_prop)
    cl_new = truncnorm.rvs(a=clip_low, b=np.inf, loc=cl_old, scale=np.sqrt(config.var_NCGibbs_prop))
    return cl_new


def compute_log_proposal(cl_old, cl_new):
    clip_low = -cl_old / np.sqrt(config.var_NCGibbs_prop)
    return truncnorm.logpdf(cl_new, a=clip_low, b=np.inf, loc=cl_old, scale=np.sqrt(config.var_NCGibbs_prop))



def compute_log_likelihood(cl, x, d, l):
    var_cl = np.array([cl/2 for _ in range(2*l+1)])
    var_cl[0] = cl
    var_noise = np.array([config.noise_covar/2 for _ in range(2*l+1)])
    var_noise[0] = config.noise_covar
    return -(1/2)*np.sum(((d-np.sqrt(var_cl)*x)**2)/var_noise)


def compute_log_MH_ratio(cl_old, cl_new, x, d, l):
    return compute_log_likelihood(cl_new, x, d, l) - compute_log_likelihood(cl_old, x, d, l) \
            + compute_log_proposal(cl_new, cl_old) - compute_log_proposal(cl_old, cl_new)


def metropolis(cl_old, x, d, l):
    old_signal_var = np.array([cl_old/2 for _ in range(2*l+1)])
    old_signal_var[0] = cl_old
    var_noise = np.array([config.noise_covar/2 for _ in range(2*l+1)])
    var_noise[0] = config.noise_covar
    cl_new = propose_cl(cl_old)
    new_signal_var = np.array([cl_new/2 for _ in range(2*l+1)])
    new_signal_var[0] = cl_new

    log_r = compute_log_MH_ratio(cl_old, cl_new, x, d, l)
    accept = 0
    if np.log(np.random.uniform()) < log_r:
        cl_old = cl_new

        accept += 1

    return cl_old, accept


def gibbs_nc(cl_init, d, l):
    cl = cl_init
    total_accept = 0
    h_cl = []
    h_s = []
    for i in range(config.N_nc_gibbs):
        if i % 1000 == 0:
            print("Non centered gibbs")
            print(i)

        x = generate_normal_real(d, cl, l)
        cl, accept = metropolis(cl, x, d, l)

        h_cl.append(cl)
        h_s.append(x)
        total_accept += accept
        #print("\n")

    acceptance_rate = total_accept/config.N_nc_gibbs
    print(acceptance_rate)
    return h_cl, h_s, acceptance_rate

