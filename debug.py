import config
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import utils

l = 1000
theta_ = utils.generate_theta()
cls_ = utils.generate_cls(theta_)
cl = cls_[l]
idx = 1589

config.delta = 0.5

#### Fonctions pour contrôler que le résultat est correct


#Le bruit était mauvais !!!
def generate_normal(d, var_s,var_noise, l):
    var = 1/((1/var_s)+(1/var_noise))
    mean_normal = var*(1/var_noise)*d
    gen_map = np.sqrt(var)*np.random.normal(size=2*l+1) + mean_normal
    return gen_map

#Le bruit était mauvais aussi
def compute_likelihood(x, idx, d, var_s, var_noise):
    var = 1/((1/var_s[idx])+(1/var_noise[idx]))
    mean_normal = var*(1/var_noise[idx])*d[idx]
    return np.exp(-(1/2)*((x - mean_normal)**2)/var)/np.sqrt(2*np.pi*var)

#Le bruit était mauvais
def compute_likelihood2(x, idx, d, var_s, var_noise):
    return np.exp(-(1/2)*((d[idx]-x)**2)/var_noise[idx]) * np.exp(-(1/2)*(x**2)/var_s[idx])\
    /(np.sqrt(2*np.pi*var_noise[idx])*np.sqrt(2*np.pi*var_s[idx]))


#### Fonctions pour la mise en oeuvre de l'auxiliary gradient-based sampler

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


def sample_auxiliary(s, d, l):
    grad_log = compute_grad_log_lik(s, d, l)
    return np.sqrt(config.delta/2)*np.random.normal(size=2*l+1) + s + (config.delta/2)*grad_log


def sample_latent(z, cl, l):
    variance_cl = np.array([cl/2 for _ in range(2*l+1)])
    variance_cl[0] = cl
    var_normal = (config.delta/2)*(1/(variance_cl + (config.delta/2)))*variance_cl
    return np.sqrt(var_normal)*np.random.normal(size=2*l+1) + (2/config.delta)*var_normal*z


def compute_log_ratio(z, s, s_new, d, l):
    part1 = compute_log_lik(s_new, d, l) - compute_log_lik(s, d, l)
    part2 = compute_g(z, s_new, d, l) - compute_g(z, s, d, l)
    print("Likelihood ratio")
    print(part1)
    print("G ratio")
    print(part2)

    return part1 + part2


def metropolis(z, cl, s, d, l):
    s_new = sample_latent(z, cl, l)
    accepted = 0

    log_r = compute_log_ratio(z, s, s_new, d, l)
    if np.log(np.random.uniform()) < log_r:
        s = s_new
        accepted += 1

    print("Accepted")
    print(accepted)
    return s, accepted


def auxGrad(cl_init, s_init, d, l):
    cl = cl_init
    s = s_init
    total_accepted = 0
    h_s = []
    for i in range(config.N_auxGrad):
        if i % 1000 == 0:
            print("AuxGrad")
            print(i)

        z_new = sample_auxiliary(s, d, l)
        s, accepted = metropolis(z_new, cl, s, d, l)
        total_accepted += accepted

        h_s.append(s)
        print("\n")

    print("Acceptance rate:")
    print(total_accepted/config.N_auxGrad)
    return h_s




s_true = np.random.normal(loc=0, scale=np.sqrt(cl / 2), size=int(l+1)) \
       + 1j * np.random.normal(loc=0, scale=np.sqrt(cl / 2), size=int(l+1))
d = s_true \
       + np.random.normal(loc=0, scale=np.sqrt(config.noise_covar / 2), size=int(l+1)) \
       + 1j * np.random.normal(loc=0, scale=np.sqrt(config.noise_covar / 2), size=int(l+1))

s_true[0] = np.sqrt(cl)*np.random.normal(size=1)
d[0] = s_true[0] + np.sqrt(config.noise_covar)*np.random.normal(size=1)
d_flat = np.concatenate((d.real, d.imag[1:]))

s_init = np.random.normal(loc=0, scale=np.sqrt(cl / 2), size=int(l + 1)) \
         + 1j * np.random.normal(loc=0, scale=np.sqrt(cl / 2), size=int(l + 1))
s_init[0] = np.random.normal(loc=0, scale=np.sqrt(cl), size=1)
s_init = np.concatenate((s_init.real, s_init.imag[1:]))





var_s = np.array([cl/2 for _ in range(2*l+1)])
var_s[0] = cl
var_noise = np.array([config.noise_covar/2 for _ in range(2*l+1)])
var_noise[0] = config.noise_covar


h_normal = []
for i in range(10000):
    h_normal.append(generate_normal(d_flat, var_s, var_noise, l))

norm, err = scipy.integrate.quad(compute_likelihood2, a=-np.inf, b=np.inf, args=(idx, d_flat, var_s, var_noise))

h_s = auxGrad(cl, s_init, d_flat, l)
h_s = np.array(h_s)

plt.plot(h_s[:, idx], color="blue", label="AuxGrad", alpha=0.5)
plt.plot(np.array(h_normal)[:, idx], color="red", label="True posterior", alpha=0.5)
plt.legend(loc="upper right")
plt.title("Path " + str(config.N_auxGrad) + ", SNR " + str(cl / config.noise_covar) + ", l = " + str(l))
plt.show()

observations = np.array(h_normal)[:, idx]
x = np.arange(np.min(observations), np.max(observations), 0.001)
y = compute_likelihood(x, idx, d_flat, var_s, var_noise)
y2 = compute_likelihood2(x, idx, d_flat, var_s, var_noise)
plt.hist(observations, density=True, bins=25, alpha=0.5)
plt.hist(h_s[2000:, idx], density=True, bins=25, alpha=0.5, color="red")
plt.plot(x,y, color="green")
plt.plot(x,y2/norm, color="black", linestyle="--")
plt.axvline(x=np.mean(h_s[2000:, idx]), color="black", linestyle="--")
plt.axvline(x=np.mean(observations), color="black")
plt.title("Histograms " + str(config.N_auxGrad) + ", SNR " + str(cl / config.noise_covar) + ", l = " + str(l))
plt.show()
plt.show()

theo_var = 1/((1/var_s[idx])+(1/(config.noise_covar/2)))
print(np.sqrt(theo_var))
print(np.std(h_s[:, idx]))
print(np.std(observations))

