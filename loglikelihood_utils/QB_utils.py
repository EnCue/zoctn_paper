import numpy as np

import scipy.special as sspecial

import jax
import jax.numpy as jnp
from jax.scipy import special as jspecial

jax.config.update("jax_enable_x64", True)

# ------------------------------------------------------------------
# LIKELIHOODS FOR LINEAR MODELS
# ------------------------------------------------------------------
def QB_nll(params, X, y):
    """
    params = [beta (p,)]
    X: (n, p)
    y: (n,) with y in [0, 1]
    """

    p = X.shape[1]
    beta = params[:p]

    eta = X @ beta

    ll = jnp.sum(
        -y * jax.nn.softplus(-eta)
        - (1.0 - y) * jax.nn.softplus(eta)
    )

    return -ll


def QB_predictive_mean(params, X):
    """
    Mean of Y where
    """
    params = np.asarray(params)
    X = np.asarray(X)
    p = X.shape[1]
    beta = params[:p]

    eta = X @ beta
    p = jspecial.expit(eta)

    EY = p

    return EY


# Stand-in for Quasi-B predictive density since Bernoulli QMLE yields no boundary masses 
# (only to be used for AE(p) calculation in simulation study)
def QB_predictive_distribution(X_test, beta, aux_params, include_interior=False, aux_transformed=False):
    N = X_test.shape[0]

    p0, p1 = np.zeros(N), np.zeros(N)

    return p0, p1

# ------------------------------------------------------------------
# LIKELIHOODS FOR EVALUATION (CRPS, Log-score)
# ------------------------------------------------------------------
def QB_ll_given_loc(y, location_par, aux_params=None):
    """
    Log-likelihood contributions under Bernoulli-logit:
        p = sigmoid(eta), eta = location_par
        log p(y|eta) = y*log(p) + (1-y)*log(1-p)

    For fractional y in [0,1], this is the Papke–Wooldridge quasi-likelihood
    objective (Bernoulli log-likelihood used as a quasi-likelihood).

    Parameters
    ----------
    y : array shape (N_eval,)
        Observations in [0,1]. May be binary {0,1} or fractional.
    location_par : array shape (N_mc, N_eval)
        Latent predictor (link scale).
    aux_params : unused (kept for API consistency)

    Returns
    -------
    ll : ndarray shape (N_mc, N_eval)
        Log-likelihood contributions per MC sample and eval point.
    """
    eta = np.asarray(location_par)

    # y_flat = np.asarray(y).reshape(-1)  # (N_eval,)
    # For quasi-likelihood we allow fractional y; just clip to [0,1] for safety
    y_clip = np.clip(y.T, 0.0, 1.0)

    # Stable form:
    # log(sigmoid(eta))     = -log(1 + exp(-eta)) = -logaddexp(0, -eta)
    # log(1 - sigmoid(eta)) = -log(1 + exp( eta)) = -logaddexp(0,  eta)
    log_p   = -np.logaddexp(0.0, -eta)
    log_1mp = -np.logaddexp(0.0,  eta)

    # broadcast y over MC dimension
    ll = y_clip * log_p + (1.0 - y_clip) * log_1mp
    return ll

def sample_QB(location_par, aux_params=None, aux_transformed=True):
    """
    Sample Y ~ Bernoulli(sigmoid(location_par)).

    location_par can be scalar, (N_eval,), or (N_mc, N_eval).
    Returns samples with same shape as location_par.
    """
    # if rng is None:
    #     rng = np.random.default_rng()

    eta = np.asarray(location_par)
    p = sspecial.expit(eta)
    # Draw Bernoulli: (U < p)
    # rng = np.random.default_rng()
    # u = rng.random(size=p.shape)
    u = np.random.uniform(size=p.shape)
    y = (u < p).astype(float)
    return y