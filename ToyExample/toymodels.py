import numpy as np


#how many models we want to have
total_models=11

def LDM(params,x):
    #x = (n,z)
    #params= parameters (volume, surface, asymmetry, Coulomb)
    n=x[0]
    z=x[1]

    return params[0]*(n+z) - params[1]*(n+z)**(2/3) - params[2]*((n-z)**2/(n+z)) - params[3]*((z**2)/((n+z)**(1/3)))

LDM_truth_params=[14,13.3,0.57,17]

def Truth(x):
    return LDM(LDM_truth_params,x)


#Noise level we will use to "wiggle" all the model predictions a bit
corruption_noise_Mass=0.05

# Fix random seed
np.random.seed(42)

