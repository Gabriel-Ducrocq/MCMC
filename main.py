import utils
import config
import numpy as np
from auxGrad import auxGrad
import matplotlib.pyplot as plt
from gibbs import gibbs_real
from NonCentered_gibbs import gibbs_nc
import scipy.stats
import scipy.integrate
from statsmodels.graphics.tsaplots import plot_acf
from rescaling import gibbs_rescale
import json


def main(l):
    theta_ = utils.generate_theta()
    cls_ = utils.generate_cls(theta_)
    s_true = np.random.normal(loc=0, scale=np.sqrt(cls_[l] / 2), size=int(l+1)) \
           + 1j * np.random.normal(loc=0, scale=np.sqrt(cls_[l] / 2), size=int(l+1))
    d = s_true \
           + np.random.normal(loc=0, scale=np.sqrt(config.noise_covar / 2), size=int(l+1)) \
           + 1j * np.random.normal(loc=0, scale=np.sqrt(config.noise_covar / 2), size=int(l+1))

    s_true[0] = np.sqrt(cls_[l])*np.random.normal(size=1)
    d[0] = s_true[0] + np.sqrt(config.noise_covar)*np.random.normal(size=1)

    theta = utils.generate_theta()
    cls = utils.generate_cls(theta)
    cls_init = cls[l]
    s_init = np.random.normal(loc=0, scale=np.sqrt(cls_init/ 2), size=int(l+1)) \
           + 1j * np.random.normal(loc=0, scale=np.sqrt(cls_init / 2), size=int(l+1))
    s_init[0] = np.random.normal(loc=0, scale=np.sqrt(cls_init), size=1)


    d_flat = np.concatenate((d.real, d.imag[1:]))
    s_init = np.concatenate((s_init.real, s_init.imag[1:]))
    #cls_init += config.noise_covar
    #dd = d_flat[:l+1] + 1j*np.concatenate((np.array([0]), d_flat[l+1:]))
    #print(np.mean(d == dd))
    h_cl, h_s, accRate_auxGrad = auxGrad(cls_init, s_init, d_flat, l)
    print("\n")
    h_gibbs_cl, h_gibbs_s = gibbs_real(d_flat,cls_init, l)
    print("\n")
    h_rescale_cl, h_rescale_s, accRate_rescale = gibbs_rescale(cls_init, d_flat, l)
    print("\n")
    h_nc_cl, h_nc_s, accRate_nc = gibbs_nc(cls_init, d_flat, l)
    print("\n")
    return h_gibbs_cl, h_gibbs_s, h_cl, h_s, accRate_auxGrad, h_rescale_cl, h_rescale_s, accRate_rescale,\
           h_nc_cl, h_nc_s, accRate_nc, cls_[l], d_flat, s_true


if __name__ == "__main__":
    l = 100
    utils.create_graphics(l)

    """
    #h_cl, h_s, h_gibbs_cl, h_gibbs_s, true_cl, obs_cl = main(l)
    h_gibbs_cl, h_gibbs_s, h_cl, h_s,accRate_auxGrad, h_rescale_cl, h_rescale_s, accRate_rescale,\
    h_nc_cl, h_nc_s, accRate_nc, true_cl, d, s_true = main(l)
    #Le 0 est réel, mais python force un complex à partie imaginaire nulle !
    observed_cl = (np.abs(d[0]) ** 2 + 2*np.sum(np.abs(d[1:])**2))/ (2 * l + 1)

    print(d[0])
    print(config.noise_covar)
    print(true_cl)
    print(observed_cl)
    plt.plot(h_gibbs_cl, label="Gibbs", alpha=0.5)
    plt.plot(h_cl[::40], color="red", label="Auxiliary", alpha=0.5)
    plt.plot(h_rescale_cl, color="green", label="Rescaling", alpha=0.5)
    plt.plot(h_nc_cl, color="magenta", label="NonCentered", alpha=0.5)
    plt.axhline(y=true_cl)
    #plt.axhline(y=observed_cl, color="red", linestyle="--")
    plt.legend(loc="upper right")
    plt.title("Path N_sample " + str(config.N_gibbs) + ", SNR " + str(true_cl/config.noise_covar) + ", l = " + str(l))
    plt.show()


    norm, err = scipy.integrate.quad(utils.likelihood, a=0, b=np.inf, args=(observed_cl, l))
    y = []
    xs = np.arange(0, np.max(h_gibbs_cl+h_rescale_cl + h_nc_cl + h_cl), 0.001)
    for x in xs:
        y.append(utils.likelihood(x,observed_cl, l))


    scale = ((2*l+1)/2)*observed_cl
    true_posterior = scipy.stats.invgamma.rvs(a=(2*l-1)/2, scale=scale, loc=-config.noise_covar, size=100000)
    plt.hist(np.array(h_gibbs_cl), bins=50, density=True, alpha=0.5, color="blue", label="Gibbs")
    #plt.hist(np.array(true_posterior), bins=50, density=True, alpha=0.5, color="blue")
    #plt.plot(xs, np.array(y)/norm)
    plt.hist(np.array(h_cl[::40]), bins=25, density=True, alpha=0.5, color="red", label="Auxiliary")
    plt.hist(np.array(h_rescale_cl), bins=25, density=True, alpha=0.5, color="green", label="Rescale")
    plt.hist(np.array(h_nc_cl), bins=25, density=True, alpha=0.5, color="magenta", label="non centered")
    plt.axvline(x=true_cl, label="True cl")
    plt.axvline(x=observed_cl, color="red", linestyle="--", label="Observed cl")
    plt.legend(loc="upper right")
    plt.title("Histograms " + str(config.N_gibbs) + ", SNR " + str(true_cl / config.noise_covar) + ", l = " + str(l))
    plt.show()


    plot_acf(h_cl[::40], fft=True, lags=100, title="AuxGrad ACF")
    plot_acf(h_gibbs_cl, fft=True, lags=100, title="Gibbs ACF")
    plot_acf(h_rescale_cl, fft=True, lags=100, title="Rescale ACF")
    plot_acf(h_nc_cl, fft=True, lags=100, title="Non centered ACF")
    plt.show()

    print("Variance gibbs:")
    print(np.var(h_gibbs_cl))
    print("Variance auxiliary sampler:")
    print(np.var(h_cl))
    a1 = ((2*l+1)/2)**2
    a2 = ((2*l-3)/2)**2
    a3 = ((2*l-5)/2)
    sigma = (np.abs(d[0])**2 + 2*np.sum(np.abs(d[1:])**2))/(2*l+1)

    var = (a1*(sigma**2))/(a2*a3)
    print("True variance:")
    print(var)

    print("means")
    print(np.mean(h_gibbs_cl))
    print(np.mean(h_cl))
    #norm2, err = scipy.integrate.quad(utils.likelihood_test, a=0, b=np.inf, args=(observed_cl, l, norm))
    #print(norm2)

    js = {"centered_gibbs":{"h_cl":h_gibbs_cl, "N_sample":config.N_gibbs},
          "non_centered_gibbs":{"h_cl":h_nc_cl, "N_sample":config.N_nc_gibbs, "variance_proposal":config.var_NCGibbs_prop,
                                "acceptance_rate":accRate_nc},
          "rescaling_gibbs":{"h_cl":h_rescale_cl, "N_sample":config.N_rescale,"variance_proposal":config.variance_rescale_prop,
                             "acceptance_rate":accRate_rescale},
          "auxGradient":{"h_cl":h_cl, "N_sample":config.N_auxGrad, "variance_proposal":config.variance_auxGrad_prop,
                         "delta":config.delta, "acceptance_rate":accRate_auxGrad},
          "observations":d, "true_skymap":s_true, "l":l, "noise_level":config.noise_covar, "cl_true":true_cl,
          "NSIDE":config.NSIDE, "l_max":config.L_MAX_SCALARS}


    np.save("results_"+str(l)+".npy", js)
    """

## Winning configurations
#l = 50
#delta = 0.2
#variance_auxGrad_prop = 0.01

#l = 400
#delta = 0.05
#variance_auxGrad_prop = 0.00001