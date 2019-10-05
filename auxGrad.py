import numpy as np
import config
import auxGradNew
import utils


def metropolis(z, cl, s, d, l):
    cl_new = auxGradNew.propose_cl(cl)
    print("proposed cl")
    print(cl_new)
    s_new = auxGradNew.sample_latent(z, cl_new, l)
    accepted = 0

    log_r = auxGradNew.compute_log_ratio(z, s, s_new, cl, cl_new, d, l)
    if np.log(np.random.uniform()) < log_r:
        cl = cl_new
        s = s_new
        accepted += 1

    print("Accepted")
    print(accepted)
    return cl, s, accepted


def auxGrad(cl_init, s_init, d, l):
    cl = cl_init
    s = s_init
    total_accepted = 0
    h_cl = []
    h_s = []
    for i in range(config.N_auxGrad):
        if i % 1000 == 0:
            print("AuxGrad")
            print(i)

        z_new = auxGradNew.sample_auxiliary(s, d, l)
        cl, s, accepted = metropolis(z_new, cl, s, d, l)
        total_accepted += accepted

        h_cl.append(cl)
        h_s.append(s)
        print("\n")

    print("Acceptance rate:")
    acceptance_rate = total_accepted/config.N_auxGrad
    print(acceptance_rate)
    return h_cl, h_s, acceptance_rate

