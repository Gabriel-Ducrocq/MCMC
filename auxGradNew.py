import numpy as np
import config
from scipy.stats import truncnorm


#def compute_log_lik(s, d, l):
#    return -(1/2)*( ( np.abs(d[0] - s[0])**2 + 2*np.sum((d[1:] - s[1:])**2) ) / config.noise_covar
#            + (2*l+1)* np.log(np.pi) + (2*l+1)*np.log(config.noise_covar) + np.log(2) )


def compute_log_lik(s, d, l):
    var = np.array([config.noise_covar/2 for _ in range(2*l+1)])
    var[0] = config.noise_covar
    return -(1/2)*np.sum(((d-s)**2)/var) - (1/2)*np.sum(np.log(var)) - ((2*l+1)/2)*np.log(2*np.pi)


def compute_grad_log_lik(s, d, l):
    var = np.array([config.noise_covar / 2 for _ in range(2 * l + 1)])
    var[0] = config.noise_covar
    grad = (d - s)/var
    return grad


def compute_g(z, s, d, l):
    grad_log = compute_grad_log_lik(s, d, l)
    return np.dot((z - s - (config.delta/4)*grad_log), grad_log)


def compute_log_auxiliary(z, cl, l):
    variance_cl = np.array([cl/2 for _ in range(2*l+1)])
    variance_cl[0] = cl
    variance = variance_cl + (config.delta/2)
    return -(1/2)*np.sum((z**2)/variance) - ((2*l+1)/2)*np.log(2*np.pi) - (1/2)*np.sum(np.log(variance))


def sample_auxiliary(s, d, l):
    grad_log = compute_grad_log_lik(s, d, l)
    return np.sqrt(config.delta/2)*np.random.normal(size=2*l+1) + s + (config.delta/2)*grad_log


def sample_latent(z, cl, l):
    variance_cl = np.array([cl/2 for _ in range(2*l+1)])
    variance_cl[0] = cl
    var_normal = (config.delta/2)*(1/(variance_cl + (config.delta/2)))*variance_cl
    return np.sqrt(var_normal)*np.random.normal(size=2*l+1) + (2/config.delta)*var_normal*z


def propose_cl(cl_old):
    clip_low = -cl_old/np.sqrt(config.variance_auxGrad_prop)
    cl_new = truncnorm.rvs(a=clip_low, b=np.inf, loc=cl_old, scale=np.sqrt(config.variance_auxGrad_prop))
    return cl_new


def compute_log_proposal(cl_old, cl_new):
    clip_low = -cl_old / np.sqrt(config.variance_auxGrad_prop)
    return truncnorm.logpdf(cl_new, a=clip_low, b=np.inf, loc=cl_old, scale=np.sqrt(config.variance_auxGrad_prop))

def compute_log_ratio(z, s, s_new, cl_old, cl_new, d, l):
    part1 = compute_log_lik(s_new, d, l) - compute_log_lik(s, d, l)
    part2 = compute_g(z, s_new, d, l) - compute_g(z, s, d, l)
    part3 = compute_log_auxiliary(z, cl_new, l) - compute_log_auxiliary(z, cl_old, l)
    part4 = compute_log_proposal(cl_new, cl_old) - compute_log_proposal(cl_old, cl_new)
    print("Likelihood ratio")
    print(part1)
    print("G ratio")
    print(part2)
    print("Auxiliary ratio")
    print(part3)
    print("Proposal ratio")
    print(part4)

    return part1 + part2 + part3 + part4



