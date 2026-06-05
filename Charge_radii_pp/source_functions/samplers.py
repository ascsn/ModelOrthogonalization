import numpy as np
import pandas as pd

def gibbs_sampler(y, X, iterations,prior_info):
    #Make sure that "y" has the correct structure. If data is being centered, it should have the mean already substracted
    b_mean_prior, b_mean_cov, nu0, sigma20  =  prior_info
    #From A_First_Course_in_Bayesian_Statistical_Methods (page ~ 159), 
    #nu0 represent the effective prior samples and sigma2_0 represents the expected prior variance
    
    b_mean_cov_inv=np.linalg.inv(b_mean_cov)
    n = len(y) # We are still taking mass and radius to be the same size
    
    X_T_X=X.T.dot(X)
    X_T_X_inv = np.linalg.inv(X_T_X)

    b_data = X_T_X_inv.dot(X.T).dot(y)

    
    
    supermodel=X.dot(b_data)
    

    residuals = y - supermodel 

    
    
    sigma2 = np.sum(residuals**2) / len(residuals) 
    cov_matrix = sigma2 * X_T_X_inv
    
    samples = []
    
    for i in range(iterations):
        # Sample from the conditional posterior of bs given sigma2 and data

        
        cov_matrix = np.linalg.inv(X_T_X/sigma2
                                   + b_mean_cov_inv)
        
        mean_vector = cov_matrix.dot(        b_mean_cov_inv.dot(b_mean_prior)+ X.T.dot(y)/sigma2  )
        
        
        b_current = np.random.multivariate_normal(mean_vector, cov_matrix)

        
        
        
        # Sample from the conditional posterior of sigma2 given bs and data
        supermodel=X.dot(b_current)
        
        residuals = y - supermodel 
        
        
        shape_post = (nu0 + n)/2.
        scale_post = (nu0*sigma20 + np.sum(residuals**2))/2.0
        # sigma2 = 1 / np.random.gamma(shape_post, 1/scale_post)
        sigma2 = 1 / np.random.default_rng().gamma(shape_post, 1/scale_post)
        
        
        samples.append(np.append(b_current,np.sqrt(sigma2)))
    


    return np.array(samples)


def gibbs_sampler_student_t(y, X, iterations, prior_info, nu,
                            burn = 10000, stepsize = 0.01, centering_data=True):
    """
    Student-t noise MCMC (Metropolis-within-Gibbs), WITHOUT simplex constraints.

    Model (scale-mixture form):
      y_i | b, sigma^2, lambda_i  ~ N( (X^T b)_i,  sigma^2 / lambda_i )
      lambda_i                     ~ Gamma( (nu/2),  (nu/2) )   [shape, rate]
      sigma^2                      ~ InvGamma(nu0/2, (nu0*sigma20)/2 )

    Inputs
    ------
    y : (n,) array
    X : array so that X.T @ b has shape (n,)
    iterations : number of post-burn samples to keep
    prior_info : (nu0, sigma20)  =>  sigma^2 ~ InvGamma(nu0/2, (nu0*sigma20)/2)
    nu : Student-t degrees of freedom
    burn : burn-in iterations
    stepsize : proposal scale (uses diag(S_hat**2 * stepsize**2))
    centering_data : keep to match your original signature

    Returns
    -------
    samples : (iterations, len(b)+1) array; rows are [b..., sqrt(sigma2)]
    """
    if centering_data == False:
        print("Only available for centering data")
        return

    # Prior hyperparameters for sigma^2 (InvGamma)
    nu0, sigma20, S_hat, Vt_hat, bias0  = prior_info

    # Proposal covariance for b (match your structure)
    cov_matrix_step = np.diag(S_hat**2 * stepsize**2)

    n = len(y)

    # ---- Initialization (match your style) ----
    b_current = np.full(len(X.T), 0.0)
    supermodel_current = X.dot(b_current)
    residuals_current = y - supermodel_current

    # Your init for sigma2 (based on -SSR)
    log_likelihood_current = -np.sum(residuals_current**2)  # -SSR helper
    sigma2 = -log_likelihood_current / n if (-log_likelihood_current) > 0 else np.var(y)

    # Initialize latent precisions for the t-mixture
    lambdas = np.ones(n)

    samples = []
    acceptance = 0

    # Helper: quadratic form used in MH step for b (with current lambdas)
    # Up to constants: log p(y | b, σ^2, λ) ∝ -(1/(2σ^2)) * Σ λ_i r_i^2
    # We'll mirror your acceptance shape exp((ll_prop - ll_curr)/σ2), with
    # ll = -Σ λ_i r_i^2
    def quad_ll(resid, lambdas):
        return -np.sum(lambdas * resid**2)

    # ---------------- Burn-in ----------------
    for i in range(burn):
        # Propose b
        b_proposed = np.random.multivariate_normal(b_current, cov_matrix_step)
        supermodel_proposed = X.dot(b_proposed)
        residuals_proposed = y - supermodel_proposed

        # MH accept/reject for b (λ fixed)
        ll_curr = quad_ll(residuals_current, lambdas)
        ll_prop = quad_ll(residuals_proposed, lambdas)
        acceptance_prob = min(1.0, np.exp((ll_prop - ll_curr) / sigma2))
        if np.random.uniform() < acceptance_prob:
            b_current = np.copy(b_proposed)
            residuals_current = residuals_proposed

        # Gibbs: update lambdas | (b, σ^2, y)
        # shape = (ν+1)/2 ; rate = (ν + r_i^2/σ^2)/2  (numpy gamma uses shape, scale=1/rate)
        shape_lam = (nu + 1.0) / 2.0
        rate_lam = (nu + (residuals_current**2) / sigma2) / 2.0
        lambdas = np.random.default_rng().gamma(shape=shape_lam, scale=1.0 / rate_lam)

        # Gibbs: update sigma^2 | (b, λ, y)
        # InvGamma( (nu0+n)/2 , (nu0*sigma20 + Σ λ_i r_i^2)/2 )
        shape_post = (nu0 + n) / 2.0
        scale_post = (nu0 * sigma20 + np.sum(lambdas * residuals_current**2)) / 2.0
        sigma2 = 1.0 / np.random.default_rng().gamma(shape=shape_post, scale=1.0 / scale_post)

    # ---------------- Sampling ----------------
    for i in range(iterations):
        # Propose b
        b_proposed = np.random.multivariate_normal(b_current, cov_matrix_step)
        supermodel_proposed = X.dot(b_proposed)
        residuals_proposed = y - supermodel_proposed

        # MH accept/reject for b (λ fixed)
        ll_curr = quad_ll(residuals_current, lambdas)
        ll_prop = quad_ll(residuals_proposed, lambdas)
        acceptance_prob = min(1.0, np.exp((ll_prop - ll_curr) / sigma2))
        if np.random.uniform() < acceptance_prob:
            b_current = np.copy(b_proposed)
            residuals_current = residuals_proposed
            acceptance = acceptance + 1

        # Gibbs: update lambdas | (b, σ^2, y)
        shape_lam = (nu + 1.0) / 2.0
        rate_lam = (nu + (residuals_current**2) / sigma2) / 2.0
        lambdas = np.random.default_rng().gamma(shape=shape_lam, scale=1.0 / rate_lam)

        # Gibbs: update sigma^2 | (b, λ, y)
        shape_post = (nu0 + n) / 2.0
        scale_post = (nu0 * sigma20 + np.sum(lambdas * residuals_current**2)) / 2.0
        sigma2 = 1.0 / np.random.default_rng().gamma(shape=shape_post, scale=1.0 / scale_post)

        # Store sample: [b..., sqrt(sigma2)]
        samples.append(np.append(b_current, np.sqrt(sigma2)))

    print("percentage accepted:", round(acceptance / iterations * 100))
    return np.array(samples)


def gibbs_sampler_simplex(y, X, iterations,prior_info, centering_data=True, burn=10000, stepsize=0.001):
    if centering_data == False:
        print("Only available for centering data")
        return 
    #Make sure that "y" has the correct structure. If data is being centered, it should have the mean already substracted
#     b_mean_prior, b_mean_cov, nu0, sigma20  =  prior_info
    
    # What I change was to make S_hat, Vt_hat, and bias0 part of the prior info, conditional parameters.
    nu0, sigma20, S_hat, Vt_hat, bias0 =  prior_info   #Since our prior is only in \sigma, we are letting the \betas be free beyond the simplex
    
    #From A_First_Course_in_Bayesian_Statistical_Methods (page ~ 159), nu0 represent the effective prior samples and sigma2_0 represents the expected prior variance
    
    cov_matrix_step=np.diag(S_hat**2*stepsize**2)
    
    n = len(y)
    
    #Initializing the starting point at the average of all the models
    b_current = np.full(len(X),0)

    
    
    supermodel_current=X.T.dot(b_current)
    

    residuals_current = y - supermodel_current 

    
  
    log_likelihood_current= -np.sum(residuals_current**2)
    
    
    sigma2 = -log_likelihood_current/ len(residuals_current) 

    
    
    samples = []
    acceptance=0
    
    for i in range(burn):
        # Sample from the conditional posterior of bs given sigma2 and data       
        b_proposed =  np.random.multivariate_normal(b_current, cov_matrix_step)
        
        omegas_proposed=np.dot(b_proposed,Vt_hat) + bias0

        
            #Comment this line and uncomment the next if you want to check MCMC without the simplex  
        if  np.any(omegas_proposed < 0):
            pass
        
#         if 1>2:
#             pass
        
        else:
            
            supermodel_proposed=X.T.dot(b_proposed)
    
            residuals_proposed = y - supermodel_proposed 

            log_likelihood_proposed= -np.sum(residuals_proposed**2)
            
#           Calculate the acceptance probability
            acceptance_prob = min(1, np.exp(   (+log_likelihood_proposed - log_likelihood_current)/sigma2   ))

            # Accept or reject the proposal
            if np.random.uniform() < acceptance_prob:
                b_current = np.copy(b_proposed)
                log_likelihood_current=log_likelihood_proposed

        shape_post = (nu0 + n)/2.
        
        
#         scale_post = (nu0*sigma20 - log_likelihood_current/len(residuals_current))/2.0
        
        scale_post = (nu0*sigma20 - log_likelihood_current)/2.0

        sigma2 = 1 / np.random.default_rng().gamma(shape_post, 1/scale_post)
        
        
#         samples.append(np.append(b_current,np.sqrt(sigma2)))
        
        
  

    for i in range(iterations):
        # Sample from the conditional posterior of bs given sigma2 and data       
        b_proposed =  np.random.multivariate_normal(b_current, cov_matrix_step)
        
        omegas_proposed=np.dot(b_proposed,Vt_hat) + bias0

    #Comment this line and uncomment the next if you want to check MCMC without the simplex    
        if  np.any(omegas_proposed < 0):
            pass
        
#         if 1>2:
#             pass
        else:
            supermodel_proposed=X.T.dot(b_proposed)
    
            residuals_proposed = y - supermodel_proposed 

            log_likelihood_proposed= -np.sum(residuals_proposed**2)
            
#           Calculate the acceptance probability
            acceptance_prob = min(1, np.exp(   (+log_likelihood_proposed - log_likelihood_current)/sigma2   ))

            # Accept or reject the proposal
            if np.random.uniform() < acceptance_prob:
                b_current = np.copy(b_proposed)
                log_likelihood_current=log_likelihood_proposed
                acceptance=acceptance+1

        shape_post = (nu0 + n)/2.
        
        scale_post = (nu0*sigma20 - log_likelihood_current)/2.0

        sigma2 = 1 / np.random.default_rng().gamma(shape_post, 1/scale_post)
        
        
        samples.append(np.append(b_current,np.sqrt(sigma2)))        
        
        
        
    print("percentage accepted:", round(acceptance/iterations*100))
    return np.array(samples)

def gibbs_sampler_student_t_simplex(y, X, iterations, prior_info, nu,
                                    Vt_hat, bias0,
                                    burn = 10000, stepsize = 0.01,
                                    centering_data=True):
    """
    Student-t noise MCMC (Metropolis-within-Gibbs) with simplex constraint on weights.

    Scale-mixture form:
      y_i | b, sigma^2, lambda_i  ~ N( (X b)_i,  sigma^2 / lambda_i )
      lambda_i ~ Gamma((nu/2), (nu/2))     [shape, rate]
      sigma^2  ~ InvGamma(nu0/2, (nu0*sigma20)/2)

    Simplex constraint is enforced through:
      omegas = b @ Vt_hat + bias0
      if any(omegas < 0) : reject proposal for b

    Args
    ----
    y : (n,) array
    X : array so that X.dot(b) has shape (n,)
    iterations : int, post-burn samples to keep
    prior_info : (nu0, sigma20) for InvGamma(nu0/2, (nu0*sigma20)/2)
    nu : float, Student-t degrees of freedom
    Vt_hat : (p, m) array, right singular vectors (same used elsewhere)
    bias0 : (m,) array, base weights to ensure omegas >= 0 when feasible
    burn : int
    stepsize : float, proposal scale; proposal cov = diag(S_hat**2 * stepsize**2)
    centering_data : bool

    Returns
    -------
    samples : (iterations, p+1) array with rows [b..., sqrt(sigma2)]
    """
    if centering_data == False:
        print("Only available for centering data")
        return

    # Prior hyperparameters for sigma^2 (InvGamma)
    nu0, sigma20, S_hat, Vt_hat, bias0 = prior_info

    # Proposal covariance for b (match your structure)
    cov_matrix_step = np.diag(S_hat**2 * stepsize**2)

    n = len(y)

    # ---- Initialization (match your style) ----
    p = len(X.T)  # keep identical to your current function
    b_current = np.full(p, 0.0)

    supermodel_current = X.dot(b_current)
    residuals_current = y - supermodel_current

    # init for sigma2 (based on -SSR)
    log_likelihood_current = -np.sum(residuals_current**2)  # -SSR helper
    sigma2 = -log_likelihood_current / n if (-log_likelihood_current) > 0 else np.var(y)

    # latent precisions for t-mixture
    lambdas = np.ones(n)

    samples = []
    acceptance = 0

    # helper: quadratic form used in MH step for b (with current lambdas)
    # (proportional to log p(y | b, σ^2, λ), up to constants)
    def quad_ll(resid, lambdas):
        return -np.sum(lambdas * resid**2)

    # ---------------- Burn-in ----------------
    for i in range(burn):
        # Propose b
        b_proposed = np.random.multivariate_normal(b_current, cov_matrix_step)

        # Simplex check via omegas
        omegas_proposed = b_proposed @ Vt_hat + bias0
        if np.any(omegas_proposed < 0):
            # reject by doing nothing (same pattern as your Gaussian code)
            pass
        else:
            supermodel_proposed = X.dot(b_proposed)
            residuals_proposed = y - supermodel_proposed

            # MH accept/reject for b (λ fixed)
            ll_curr = quad_ll(residuals_current, lambdas)
            ll_prop = quad_ll(residuals_proposed, lambdas)
            acceptance_prob = min(1.0, np.exp((ll_prop - ll_curr) / sigma2))
            if np.random.uniform() < acceptance_prob:
                b_current = np.copy(b_proposed)
                residuals_current = residuals_proposed

        # Gibbs: update lambdas | (b, σ^2, y)
        shape_lam = (nu + 1.0) / 2.0
        rate_lam  = (nu + (residuals_current**2) / sigma2) / 2.0
        lambdas   = np.random.default_rng().gamma(shape=shape_lam, scale=1.0 / rate_lam)

        # Gibbs: update sigma^2 | (b, λ, y)
        shape_post = (nu0 + n) / 2.0
        scale_post = (nu0 * sigma20 + np.sum(lambdas * residuals_current**2)) / 2.0
        sigma2     = 1.0 / np.random.default_rng().gamma(shape=shape_post, scale=1.0 / scale_post)

    # ---------------- Sampling ----------------
    for i in range(iterations):
        # Propose b
        b_proposed = np.random.multivariate_normal(b_current, cov_matrix_step)

        # Simplex check via omegas
        omegas_proposed = b_proposed @ Vt_hat + bias0
        if np.any(omegas_proposed < 0):
            pass
        else:
            supermodel_proposed = X.dot(b_proposed)
            residuals_proposed = y - supermodel_proposed

            # MH accept/reject for b (λ fixed)
            ll_curr = quad_ll(residuals_current, lambdas)
            ll_prop = quad_ll(residuals_proposed, lambdas)
            acceptance_prob = min(1.0, np.exp((ll_prop - ll_curr) / sigma2))
            if np.random.uniform() < acceptance_prob:
                b_current = np.copy(b_proposed)
                residuals_current = residuals_proposed
                acceptance = acceptance + 1

        # Gibbs: update lambdas | (b, σ^2, y)
        shape_lam = (nu + 1.0) / 2.0
        rate_lam  = (nu + (residuals_current**2) / sigma2) / 2.0
        lambdas   = np.random.default_rng().gamma(shape=shape_lam, scale=1.0 / rate_lam)

        # Gibbs: update sigma^2 | (b, λ, y)
        shape_post = (nu0 + n) / 2.0
        scale_post = (nu0 * sigma20 + np.sum(lambdas * residuals_current**2)) / 2.0
        sigma2     = 1.0 / np.random.default_rng().gamma(shape=shape_post, scale=1.0 / scale_post)

        # Store sample: [b..., sqrt(sigma2)]
        samples.append(np.append(b_current, np.sqrt(sigma2)))

    print("percentage accepted:", round(acceptance / iterations * 100))
    return np.array(samples)