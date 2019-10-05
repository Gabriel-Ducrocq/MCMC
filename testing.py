import auxGradNew
import config
import numpy as np

##Testing the sample of the auxiliary variable

config.delta = 2
l = 10000
s = np.zeros(2*l+1)
d = np.ones(2*l+1)

r = auxGradNew.sample_auxiliary(s, d, l)
print("Average from sim")
print(np.mean(r[1:]))
print("Theory average")
print(2/config.noise_covar)

print("Variance from sim")
print(np.var(r[1:]))
print("Theory variance")
print("1")

print("\n")
config.delta = 1
l = 100000
s = np.ones(2*l+1)
d = np.ones(2*l+1)

r = auxGradNew.sample_auxiliary(s, d, l)
print("Average from sim")
print(np.mean(r[1:]))
print("Theory average")
print(1)

print("Variance from sim")
print(np.var(r[1:]))
print("Theory variance")
print(1/2)
print("\n\n\n")

######Testing the sampling of latent variable

l = 100000
cl = 2
z = np.zeros(2*l+1)
config.delta = 1
r = auxGradNew.sample_latent(z, cl, l)

print(config.delta)
print("Average from sim")
print(np.mean(r[1:]))
print("Theory average")
print(0)

print("Variance from sim")
print(np.var(r[1:]))
print("Theory variance")
print(1/3)
print("\n")


l = 100000
cl = 2
z = np.ones(2*l+1)
config.delta = 2
r = auxGradNew.sample_latent(z, cl, l)


print(config.delta)
print("Average from sim")
print(np.mean(r[1:]))
print("Theory average")
print(1/2)

print("Variance from sim")
print(np.var(r[1:]))
print("Theory variance")
print(1/2)
print("\n")


l = 1000000
cl = 2
z = 4*np.ones(2*l+1)
config.delta = 0.1
r = auxGradNew.sample_latent(z, cl, l)


print(config.delta)
print("Average from sim")
print(np.mean(r[1:]))
print("Theory average")
print((4*1/21)*(2/0.1))

print("Variance from sim")
print(np.var(r[1:]))
print("Theory variance")
print(1/21)
print("\n\n\n")


#### On teste maintenant le calcul de g

l = 1000000
cl = 2
z = 4*np.ones(2*l+1)
s = np.ones(2*l+1)
d = np.ones(2*l+1)
config.delta = 0.1
r = auxGradNew.compute_g(z, s, d, l)

theo_r = 0
print("Result should be 0:")
print(r)
print(theo_r == r)

print("\n")
l = 1000
z = np.ones(2*l+1)
s = np.ones(2*l+1)
d = 2*np.ones(2*l+1)
config.delta = 2
r = auxGradNew.compute_g(z, s, d, l)

theo_r = -(1/2)*(((2/config.noise_covar)**2)*(2*l) + (1/config.noise_covar)**2)
print(theo_r, r)
print(np.round(theo_r, 5) == np.round(r, 5))


print("\n")
l = 1000
z = np.ones(2*l+1)
s = np.ones(2*l+1)
d = 2*np.ones(2*l+1)
config.delta = 2
r = auxGradNew.compute_g(z, s, d, l)

theo_r = -(1/2)*(((2/config.noise_covar)**2)*(2*l) + (1/config.noise_covar)**2)
print(theo_r, r)
print(np.round(theo_r, 5) == np.round(r, 5))



print("\n")
l = 1000
z = np.ones(2*l+1)
s = np.zeros(2*l+1)
d = 2*np.ones(2*l+1)
config.delta = 4
r = auxGradNew.compute_g(z, s, d, l)

theo_r = (2/config.noise_covar) + 2*l*(4/config.noise_covar) - ((2/config.noise_covar)**2 + 2*l*((4/config.noise_covar)**2))
print(theo_r, r)
print(np.round(theo_r, 5) == np.round(r, 5))

print("\n\n\n")


####On teste le calcul de la log vraisemblance

l = 1000
s = np.ones(2*l+1)
d = np.ones(2*l+1)
config.delta = 4
config.noise_covar = 2

r = auxGradNew.compute_log_lik(s, d, l)
theo_r = -((2*l+1)/2)*np.log(2*np.pi) - (1/2)*np.log(2)

print(r)
print(theo_r)
print("Does the function matches the expected result:")
print(r == theo_r)


print("\n")

l = 10
s = np.zeros(2*l+1)
d = np.ones(2*l+1)
config.delta = 4
config.noise_covar = 2

r = auxGradNew.compute_log_lik(s, d, l)
theo_r = -(1/2)*((2*l) + 1/2) -((2*l+1)/2)*np.log(2*np.pi) - (1/2)*(np.log(2))

print(r)
print(theo_r)
print("Does the function matches the expected result:")
print(r == theo_r)


print("\n\n\n")



### On teste le gradient de la log vraisemblance

l = 100
s = np.zeros(2*l+1)
d = np.ones(2*l+1)
config.delta = 4
config.noise_covar = 2

theo_r = np.ones(len(d))
theo_r[0] = theo_r[0]/2
r = auxGradNew.compute_grad_log_lik(s,d,l)

print("Do we have the expected results ?")
print(np.mean(theo_r == r))

print("\n")

l = 100
s = np.ones(2*l+1)
d = 5*np.ones(2*l+1)
config.delta = 4
config.noise_covar = 1

theo_r = np.ones(len(d))*4*2
theo_r[0] = theo_r[0]/2
r = auxGradNew.compute_grad_log_lik(s,d,l)
print("Do we have the expected results ?")
print(np.mean(theo_r == r))

print("\n\n\n")

### Testing log auxiliary

l = 1000
s = np.ones(2*l+1)
z = np.ones(2*l+1)*5
cl = 0
config.delta = 2

theo_r = -(1/2)*(2*l+1)*(5**2) - ((2*l+1)/2)*np.log(2*np.pi)
r = auxGradNew.compute_log_auxiliary(z, cl, l)

print(r, theo_r)
print(r == theo_r)

print("\n")










