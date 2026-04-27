import numpy as np
import numpy.random as rng

import scipy.special as sspecial


# Samples latent location parameter (mu) from model
# -------------------------------------------------------
def sample_mu(model, model_type, X, Y, coords=None, N_mc=100):
    N_eval = len(Y)
    latent_mu, latent_var = 0, 1

    if model_type == 'GPLM':
        latent_params = model.predict(
            X_pred=X,
            gp_coords_pred=coords,
            predict_response=False,
            predict_var=True,
        )
        latent_mu, latent_var = latent_params['mu'], latent_params['var']

    elif model_type == 'GPB':
        latent_params = model.predict(
            data=X,
            gp_coords_pred=coords,
            predict_var=True,
            pred_latent=True
        )
        latent_mu = latent_params['fixed_effect'] + latent_params['random_effect_mean']
        latent_var = latent_params['random_effect_cov']
    
    elif model_type == 'ITB':
        test_group = np.zeros(N_eval)
        
        latent_params = model.predict(
            data=X, 
            group_data_pred=test_group,
            pred_latent=True
        )
        latent_mu = latent_params['fixed_effect']
        latent_var = 0.0

    mu_mc = rng.normal(
        loc=latent_mu,
        scale=np.sqrt(latent_var),
        size=(N_mc, N_eval),
    )

    return mu_mc


# Log score for independent linear model
# -------------------------------------------------------
# def ind_logscore(nll_func, params, X, Y):
def ind_logscore(nll_func, model_type, model_info, X, Y):
    N_eval = len(Y)
    nll = 0
    if model_type == 'lm':
        nll = nll_func(model_info['params'], X, Y)
    elif model_type == 'tb':
        test_group = np.zeros(N_eval)
        
        latent_params = model_info['model'].predict(
            data=X, 
            group_data_pred=test_group,
            pred_latent=True
        )
        loc_preds = latent_params['fixed_effect']
        
        nlls = -1.0 * nll_func(Y, loc_preds, model_info['aux_params'])
        nll = np.sum(nlls)
    
    ls = nll / N_eval

    return ls


# Log score for TB/GPLM/GPB models
# -------------------------------------------------------
def gp_logscore(model, model_type, X, Y, aux_params, ll_func, coords=None, N_mc=1000):
    
    mu_mc = sample_mu(model, model_type, X, Y, coords, N_mc)
    
    ll_matrix = ll_func(Y, mu_mc, aux_params) # log-likelihood given mu

    MC_log_py = sspecial.logsumexp(ll_matrix, axis=0) - np.log(N_mc)

    logscore = -np.mean(MC_log_py)
    
    return logscore


# Continuous-Ranked Probability Score (CRPS) Method
# -------------------------------------------------------
def crps(
    model,
    model_type,
    X,
    Y,
    aux_params,
    transform_func,
    coords=None,
    aux_transformed=True,
    N_mc=100,
    batch_size=5000,
):
    N_eval = len(Y)
    crps_sum = 0.0

    for start in range(0, N_eval, batch_size):
        end = min(start + batch_size, N_eval)

        X_batch = X[start:end]
        Y_batch = Y[start:end]
        coords_batch=None
        try: 
            coords_batch = coords[start:end]
        except:
            pass

        mu_mc_batch = None
        if model_type == 'GLM':
            loc_param_batch = (X_batch @ model).flatten()

            mu_mc_batch = np.broadcast_to(
                loc_param_batch[None, :],
                (N_mc, len(Y_batch)),
            )
        else:
            mu_mc_batch = sample_mu(
                model,
                model_type,
                X_batch,
                Y_batch,
                coords_batch,
                N_mc,
            )

        response_mc_batch = transform_func(
            mu_mc_batch,
            aux_params,
            aux_transformed=aux_transformed,
        )

        obs_crps_batch = get_response_CRPS(response_mc_batch, Y_batch)
        crps_sum += np.sum(obs_crps_batch)

    crps = crps_sum / N_eval

    return crps


# Helper function for CRPS
# -------------------------------------------------------
def get_response_CRPS(response, Y):

    # First term:
    # S1 = E|X - y|
    S1 = np.mean(np.abs(response - Y[None, :]), axis=0)

    # Second term:
    # S2 = (1/2) E|X - X'|
    #
    # For empirical samples x_(1) <= ... <= x_(N), we use:
    # sum_{j,k} |x_j - x_k| = 2 * sum_{i=1}^N (2i - N - 1) x_(i)
    #
    # Therefore:
    # (1/2N^2) sum_{j,k} |x_j - x_k|
    # = (1/N^2) sum_{i=1}^N (2i - N - 1) x_(i)

    n_mc = response.shape[0]
    response_sorted = np.sort(response, axis=0)

    weights = (2 * np.arange(1, n_mc + 1) - n_mc - 1)[:, None]
    S2 = np.sum(weights * response_sorted, axis=0) / (n_mc ** 2)

    return S1 - S2


# AIC / BIC
# -------------------------------------------------------
def AIC_BIC(nll, k, n):
    AIC = (2 * k) + (2 * nll)
    BIC = (np.log(n) * k) + (2 * nll)

    return AIC, BIC