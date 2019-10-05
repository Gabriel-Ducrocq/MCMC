import numpy as np
import healpy as hp

COSMO_PARAMS_NAMES = ["n_s", "omega_b", "omega_cdm", "100*theta_s", "ln10^{10}A_s", "tau_reio"]
COSMO_PARAMS_MEAN = np.array([0.9665, 0.02242, 0.11933, 1.04101, 3.047, 0.0561])
COSMO_PARAMS_SIGMA = np.array([0.0038, 0.00014, 0.00091, 0.00029, 0.014, 0.0071])
LiteBIRD_sensitivities = np.array([36.1, 19.6, 20.2, 11.3, 10.3, 8.4, 7.0, 5.8, 4.7, 7.0, 5.8, 8.0, 9.1, 11.4, 19.6])

LENSING = 'yes'
OUTPUT_CLASS = 'tCl pCl lCl'
observations = None

N_Stoke = 1
NSIDE = 512
Npix = 12*NSIDE**2
L_MAX_SCALARS=int(2*NSIDE)
dimension_sph = int((L_MAX_SCALARS*(L_MAX_SCALARS + 1)/2)+L_MAX_SCALARS+1)


def noise_covariance_in_freq(nside):
    ##Prendre les plus basses fréquences pour le bruit (là où il est le plus petit)
    cov = LiteBIRD_sensitivities ** 2 / hp.nside2resol(nside, arcmin=True) ** 2
    return cov


noise_covar_one_pix = noise_covariance_in_freq(NSIDE)
noise_covar = noise_covar_one_pix[7]
#noise_lm = hp.anafast(noise_covar)
#noise_covar = [noise_lm[1000]]
#print(noise_covar)
#noise_covar = np.random.choice(noise_covar_one_pix, Npix*N_Stoke)
N_metropolis = 1
N_gibbs = 100000


delta = 0.1
stdd_theta_prop = COSMO_PARAMS_SIGMA
N_auxGrad = 100000
variance_auxGrad_prop = 0.0001

N_rescale = 100000
variance_rescale_prop = 0.02

N_nc_gibbs = 100000
var_NCGibbs_prop = 0.001
###Bonnes configs:
#delta=0.1
#variance_auxGrad_prop=0.0001
#l=1000

#delta=0.1
#variance_auxGrad_prop=0.00001
#l=1000