import healpy as hp
import numpy as np
from classy import Class
import config
import json
import matplotlib.pyplot as plt
import pylab
import healpy as hp
from scipy.stats import truncnorm, invgamma
from statsmodels.graphics.tsaplots import plot_acf
import math


cosmo = Class()

LENSING = 'yes'
OUTPUT_CLASS = 'tCl pCl lCl'

COSMO_PARAMS_NAMES = ["n_s", "omega_b", "omega_cdm", "100*theta_s", "ln10^{10}A_s", "tau_reio"]
COSMO_PARAMS_MEAN = np.array([0.9665, 0.02242, 0.11933, 1.04101, 3.047, 0.0561])
COSMO_PARAMS_SIGMA = np.array([0.0038, 0.00014, 0.00091, 0.00029, 0.014, 0.0071])
LiteBIRD_sensitivities = np.array([36.1, 19.6, 20.2, 11.3, 10.3, 8.4, 7.0, 5.8, 4.7, 7.0, 5.8, 8.0, 9.1, 11.4, 19.6])


def generate_theta():
    return np.random.normal(COSMO_PARAMS_MEAN, COSMO_PARAMS_SIGMA)

def generate_cls(theta):
    params = {'output': OUTPUT_CLASS,
              'l_max_scalars': config.L_MAX_SCALARS,
              'lensing': LENSING}
    d = {name:val for name, val in zip(COSMO_PARAMS_NAMES, theta)}
    params.update(d)
    cosmo.set(params)
    cosmo.compute()
    cls = cosmo.lensed_cl(config.L_MAX_SCALARS)
    #10^12 parce que les cls sont exprimés en kelvin carré, du coup ça donne une stdd en 10^6
    cls["tt"] *= 1e12
    cosmo.struct_cleanup()
    cosmo.empty()
    return cls["tt"]


def create_graphics(l):
    results = np.load("results_"+str(l)+".npy")
    results = results.item()
    h_c_gibbs = results["centered_gibbs"]["h_cl"]
    h_nc_gibbs = results["non_centered_gibbs"]["h_cl"]
    auxGradient = results["auxGradient"]["h_cl"]
    rescalingGibbs = results["rescaling_gibbs"]["h_cl"]
    true_cl = results["cl_true"]
    noise_level = results["noise_level"]
    N_sample = results["centered_gibbs"]["N_sample"]

    fig3, ax3 = plt.subplots()
    ax3.plot(h_c_gibbs, label="Gibbs", alpha=0.3)
    ax3.plot(auxGradient, color="red", label="Auxiliary", alpha=0.3)
    ax3.plot(rescalingGibbs, color="green", label="Rescaling", alpha=0.3)
    ax3.plot(h_nc_gibbs, color="magenta", label="NonCentered", alpha=0.3)
    ax3.axhline(y=true_cl)
    ax3.legend(loc="upper right")
    ax3.set_title("Path N_sample " + str(N_sample) + ", SNR " + str(true_cl / noise_level) + ", l = " + str(l))

    fig4, ax4 = plt.subplots()
    ax4.plot(h_c_gibbs[::2][:2500], label="Gibbs", alpha=0.4)
    ax4.plot(auxGradient[::40][:2500], color="red", label="Auxiliary", alpha=0.4)
    ax4.plot(rescalingGibbs[:2500], color="green", label="Rescaling", alpha=0.4)
    ax4.plot(h_nc_gibbs[::2][:2500], color="magenta", label="NonCentered", alpha=0.4)
    ax4.axhline(y=true_cl)
    ax4.legend(loc="upper right")
    ax4.set_title("Thinned Path N_sample " + str(N_sample) + ", SNR " + str(true_cl / noise_level) + ", l = " + str(l))


    fig1, axes = plt.subplots(2,2)
    plot_acf(h_c_gibbs, ax=axes[0, 0],fft=True, lags=100, title="Centered Gibbs ACF")
    plot_acf(h_nc_gibbs, ax=axes[0, 1],fft=True, lags=100, title="Non centered Gibbs ACF")
    plot_acf(auxGradient, ax=axes[1, 0],fft=True, lags=100, title="Auxiliary Gradient ACF")
    plot_acf(rescalingGibbs, ax=axes[1, 1],fft=True, lags=100, title="Rescaling Gibbs ACF")
    fig1.suptitle("Autocorrelation plots per iteration for l = "+str(l)+", SNR = "+str(np.round(true_cl/noise_level, 5)))


    fig2, axes = plt.subplots(2,2)
    plot_acf(h_c_gibbs[::2], ax=axes[0, 0],fft=True, lags=100, title="Centered Gibbs ACF")
    plot_acf(h_nc_gibbs[::2], ax=axes[0, 1],fft=True, lags=100, title="Non centered Gibbs ACF")
    plot_acf(auxGradient[::40], ax=axes[1, 0],fft=True, lags=100, title="Auxiliary Gradient ACF")
    plot_acf(rescalingGibbs, ax=axes[1, 1],fft=True, lags=100, title="Rescaling Gibbs ACF")
    fig2.suptitle("Autocorrelation plots per 80 transf for l = "+str(l)+", SNR = "+str(np.round(true_cl/noise_level, 5)))
    plt.show()





""""
    js = {"centered_gibbs":{"h_cl":h_gibbs_cl, "N_sample":config.N_gibbs},
          "non_centered_gibbs":{"h_cl":h_nc_cl, "N_sample":config.N_nc_gibbs, "variance_proposal":config.var_NCGibbs_prop,
                                "acceptance_rate":accRate_nc},
          "rescaling_gibbs":{"h_cl":h_rescale_cl, "N_sample":config.N_rescale,"variance_proposal":config.variance_rescale_prop,
                             "acceptance_rate":accRate_rescale},
          "auxGradient":{"h_cl":h_cl, "N_sample":config.N_auxGrad, "variance_proposal":config.variance_auxGrad_prop,
                         "delta":config.delta, "acceptance_rate":accRate_auxGrad},
          "observations":d, "true_skymap":s_true, "l":l, "noise_level":config.noise_covar, "cl_true":true_cl,
          "NSIDE":config.NSIDE, "l_max":config.L_MAX_SCALARS}
"""

####A réécrire en réels
def compute_log_lik(d, s):
    return -(1/2)*np.sum((d-s)**2)/(config.noise_covar[0]/2) - (1/2)*len(d)*np.log(config.noise_covar[0]/2)\
           - (1/2)*len(d)*np.log(2*np.pi)


def compute_grad_log_lik(d, s):
    return 2*(d - s)/config.noise_covar[0]


def compute_g(z,y, d):
    grad_log = compute_grad_log_lik(d, y)
    first_factor = z - y - (config.delta/4)*grad_log
    second_factor = grad_log
    return np.dot(first_factor, second_factor)


def compute_log_auxiliary(z,cl):
    return -(1/2)*np.sum(z**2)/(cl+(config.delta/2)) - (1/2)*len(z)*np.log(2*np.pi) - (1/2)*len(z)*np.log((cl+(config.delta/2))/2)


def compute_log_proposal(cl_new, cl_old, l):
    variance = config.noise_covar[0]/(2*l+1)
    return -(1/2)*((cl_new - cl_old)**2)/variance


def propose_new_cl(cl_old,l):
    variance = (config.noise_covar[0]/(2*l+1))/10
    clip_low = -cl_old/np.sqrt(variance)
    cl_new = truncnorm.rvs(a=clip_low, b=np.inf, loc=cl_old, scale=np.sqrt(variance))
    return cl_new


def propose_new_latent(z, cl_new):
    variance_complex = (config.delta*cl_new)/(config.delta+2*cl_new)
    mean_normal = (2/config.delta)*variance_complex*z
    return np.sqrt(variance_complex/2)*np.random.normal(loc=mean_normal, scale=1)


def compute_log_ratio(x,y,z,cl_old, cl_new, d, l):
    print("Auxiliary log ratio:")
    print(compute_log_auxiliary(z, cl_new) - compute_log_auxiliary(z, cl_old))
    print("Likelihood log ratio")
    print(compute_log_lik(d, y) - compute_log_lik(d, x))
    print("Gradients log ratio")
    print(compute_g(z, y, d) - compute_g(z, x, d))
    print("Proposal log ratio:")
    print(compute_log_proposal(cl_old, cl_new,l)- compute_log_proposal(cl_new, cl_old, l))

    return compute_log_lik(d, y) - compute_log_lik(d, x) + compute_g(z, y, d) - compute_g(z, x, d) \
            + compute_log_auxiliary(z, cl_new) - compute_log_auxiliary(z, cl_old) + compute_log_proposal(cl_old, cl_new,l)\
            - compute_log_proposal(cl_new, cl_old, l)


def propose_auxiliary(x, d):
    mean_normal = x + (config.delta/2)*compute_grad_log_lik(d, x)
    return np.sqrt(config.delta/4)*np.random.normal(loc=mean_normal, scale=1)


def likelihood(x, sigma, l):
    return np.exp(-(((2*l+1)/2)*sigma)/(x+config.noise_covar))/(x+config.noise_covar)**((2*l+1)/2)


def likelihood_test(x, sigma, l, norm):
    return (1/norm)*np.exp(-(((2*l+1)/2)*sigma)/(x+config.noise_covar))/(x+config.noise_covar)**((2*l+1)/2)







##### Complex

def compute_auxGrad_log_lik(s, d):
    #Input:skymap and observations as complex
    return -(1/2)*np.sum(np.abs((d-s))**2)/config.noise_covar


def compute_auxGrad_grad_log_lik(s,d,l):
    #Input: skymap and observations as complex
    #Output: log grad as real (bijection)
    s_re, s_imag = s.real, s.imag[1:]
    s_flat = np.concatenate((s_re, s_imag))
    d_re, d_imag = d.real, d.imag[1:]
    d_flat = np.concatenate((d_re, d_imag))
    grad_log = 2*(d_flat - s_flat)/config.noise_covar
    grad_log[0] = (d_flat[0] - s_flat[0])/config.noise_covar
    output = grad_log[:l+1] + 1j*np.concatenate(([0], grad_log[l+1:]))
    return output


def compute_auxGrad_g(z,y, d, l):
    grad_log = compute_auxGrad_grad_log_lik(y, d, l)
    return np.dot((z - y - (config.delta/4)*grad_log), grad_log)


def compute_auxGrad_log_auxiliary(z, cl, l):
    #print("First")
    ##print(-(1 / 2) * ((np.abs(z[0]) ** 2 + np.sum(np.abs(z[1:]) ** 2)) / (cl + (config.delta / 2))))
    #print(-(1/2)*(np.abs(z[0]) ** 2 + np.sum(np.abs(z[1:]) ** 2))/ (cl + (config.delta / 2)))
    #print("---Sub first 1")
    #print((np.abs(z[0]) ** 2 + np.sum(np.abs(z[1:]) ** 2)))
    #print("---Sub first 2")
    #print(-(1 / 2)/ (cl + (config.delta / 2)))
    #print("Second")
    #print((2*l+1)*np.log(cl + (config.delta/2)))
    return -(1/2)*((np.abs(z[0])**2 + np.sum(np.abs(z[1:])**2))/(cl + (config.delta/2)) - (2*l+1)*np.log(cl + (config.delta/2)))


def compute_auxGrad_log_proposal(cl_new, cl_old):
    return -(1/2)*((cl_new - cl_old)**2)/config.variance_auxGrad_prop


def compute_auxGrad_log_ratio(z, s, y, cl_old, cl_new, l, d):
    part1 = compute_auxGrad_log_lik(y, d) - compute_auxGrad_log_lik(s, d)
    part2 = compute_auxGrad_g(z, y, d, l) - compute_auxGrad_g(z, s, d, l)
    part3 = compute_auxGrad_log_auxiliary(z, cl_new, l) - compute_auxGrad_log_auxiliary(z, cl_old, l)
    part4 = compute_auxGrad_log_proposal(cl_old, cl_new) - compute_auxGrad_log_proposal(cl_new, cl_old)
    print("Log lik ratio")
    print(part1)
    print("G part")
    print(part2)
    print("Auxiliary")
    print(part3)
    print("Proposal")
    print(part4)
    print("\n")
    return part1 + part2 + part3 + part4


def auxGrad_propose_auxiliary(s,d, l):
    mean_normal = s + (config.delta/2)*compute_auxGrad_grad_log_lik(s, d, l)
    z = mean_normal + np.random.normal(loc=0, scale= np.sqrt(config.delta/4), size=l+1)\
        + 1j*np.random.normal(loc=0, scale= np.sqrt(config.delta/4), size=l+1)

    z[0] = mean_normal[0] + np.random.normal(loc=0, scale= np.sqrt(config.delta/2), size=1)
    return z


def auxGrad_propose_new_latent(z, cl, l):
    var = (config.delta*cl)/(config.delta + 2*cl)
    mean_normal = (2/config.delta)*var*z
    generate = mean_normal + np.random.normal(loc=0, scale=np.sqrt(var/2), size=l+1)\
               + 1j*np.random.normal(loc=0, scale=np.sqrt(var/2), size=l+1)
    generate[0] = mean_normal[0] + np.random.normal(loc=0, scale=np.sqrt(var), size=1)
    return generate


def auxGrad_propose_new_cl(cl_old):
    clip_low = -cl_old/np.sqrt(config.variance_auxGrad_prop)
    cl_new = truncnorm.rvs(a=clip_low, b=np.inf, loc=cl_old, scale=np.sqrt(config.variance_auxGrad_prop))
    return cl_new



#Bij

def compute_auxGrad_log_lik_bij(s, d):
    return -(1/2)*((d[0]-s[0])**2)/config.noise_covar - (1/2)*np.sum((d[1:] - s[1:])**2)/(config.noise_covar/2)


def compute_auxGrad_grad_log_lik_bij(s,d):
    grad_log = 2*(d - s)/config.noise_covar
    grad_log[0] = (d[0] - s[0])/config.noise_covar
    return grad_log


def compute_auxGrad_g_bij(z,y, d):
    grad_log_y = compute_auxGrad_grad_log_lik_bij(y, d)
    return np.dot(z - y - (config.delta/4)*grad_log_y, grad_log_y)


def compute_auxGrad_log_auxiliary_bij(z, cl, l):
    var = [cl/2 for _ in range(2*l+1)]
    var[0] = cl
    var = np.array(var) + config.delta/2
    return -(1/2)*np.sum((z**2)/var) - np.sum(var)


def compute_auxGrad_log_proposal_bij(cl_new, cl_old):
    return -(1/2)*((cl_new - cl_old)**2)/config.variance_auxGrad_prop


def compute_auxGrad_log_ratio_bij(z, s, y, cl_old, cl_new, l, d):
    part1 = compute_auxGrad_log_lik_bij(y, d) - compute_auxGrad_log_lik_bij(s, d)
    part2 = compute_auxGrad_g_bij(z, y, d) - compute_auxGrad_g_bij(z, s, d)
    part3 = compute_auxGrad_log_auxiliary_bij(z, cl_new, l) - compute_auxGrad_log_auxiliary_bij(z, cl_old, l)
    part4 = compute_auxGrad_log_proposal_bij(cl_old, cl_new) - compute_auxGrad_log_proposal_bij(cl_new, cl_old)
    print("Log lik ratio")
    print(part1)
    print("G part")
    print(part2)
    print("Auxiliary")
    print(part3)
    print("Proposal")
    print(part4)
    print("\n")
    return part1 + part2 + part3 + part4


def auxGrad_propose_auxiliary_bij(s, d, l):
    mean_normal = s + (config.delta/2)*compute_auxGrad_grad_log_lik_bij(s, d)
    return np.sqrt(config.delta/2)*np.random.normal(loc=0, scale = 1, size = 2*l+1) + mean_normal


def auxGrad_propose_new_latent_bij(z, cl, l):
    var_cl = [cl/2 for _ in range(2*l+1)]
    var_cl[0] = cl
    var_cl = np.array(var_cl) + config.delta/2

    var = 1/( (1/var_cl) + (2/config.delta))
    mean_normal = (2/config.delta)*var*z
    print(mean_normal)
    print(var)
    return np.sqrt(var)*np.random.normal(size=2*l+1) + mean_normal


def auxGrad_propose_new_cl_bij(cl_old):
    clip_low = -cl_old/np.sqrt(config.variance_auxGrad_prop)
    cl_new = truncnorm.rvs(a=clip_low, b=np.inf, loc=cl_old, scale=np.sqrt(config.variance_auxGrad_prop))
    return cl_new

