import numpy as np
from scipy.stats import invgamma
import config
import healpy as hp


def generate_normal_real(d, cl,l):
    var_cl = np.array([cl/2 for _ in range(2*l+1)])
    var_cl[0] = cl
    var_noise = np.array([config.noise_covar/2 for _ in range(2*l+1)])
    var_noise[0] = config.noise_covar

    var = (1/((1/var_cl)+(1/var_noise)))
    mean_normal = var*(1/var_noise)*d
    gen_map = np.sqrt(var)*np.random.normal(size=2*l+1) + mean_normal
    return gen_map


def generate_Inv_Gamma_real(s, l):
    #Idem: le 0 est réel, mais python force un complexe à partie imaginaire nulle.
    observed_Cl = (np.abs(s[0])**2 + 2*np.sum(s[1:]**2))/(2*l+1)
    alpha = (2*l-1)/2
    beta = (2*l+1)*(observed_Cl/2)
    return beta*invgamma.rvs(a=alpha)



def gibbs_real(d, cl_init, l):
    cl = cl_init
    history_cl = []
    history_s = []
    for i in range(config.N_gibbs):
        if i % 1000==0:
            print("Gibbs")
            print(i)

        s = generate_normal_real(d, cl, l)
        cl = generate_Inv_Gamma_real(s, l)
        history_cl.append(cl)
        history_s.append(s)

    return history_cl, history_s