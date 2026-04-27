import numpy as np

import scipy as sp
from scipy import stats
import scipy.special as sspecial

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax.scipy import special as jspecial

jax.config.update("jax_enable_x64", True)

EPS = 1e-12


# ------------------------------------------------------------------
# LIKELIHOODS FOR LINEAR MODELS
# ------------------------------------------------------------------

def BEINF_nll(params, X, y):
    """
    Zero-and-one inflated beta (BEINF) negative log-likelihood
    following Ospina & Ferrari.

    Parametrization used here:
      mu_i    = expit(X_i @ beta)
      alpha   = expit(raw_alpha)    # total discrete mass on {0,1}
      gamma   = expit(raw_gamma)    # conditional prob of 1 among boundary mass
      phi     = exp(raw_phi)        # beta precision

    Hence:
      P(Y=0) = alpha * (1 - gamma)
      P(Y=1) = alpha * gamma
      f(Y=y in (0,1)) = (1-alpha) * Beta(y; mu_i, phi)
    """
    p = X.shape[1]

    beta = params[:p]
    alpha = jspecial.expit(params[p])       # in (0,1)
    gamma = jspecial.expit(params[p + 1])   # in (0,1)
    phi   = jnp.exp(params[p + 2])          # > 0

    mu = jspecial.expit(X @ beta)

    m0 = (y == 0.0)
    m1 = (y == 1.0)
    mint = ~(m0 | m1)

    ll = 0.0

    # Mass at 0: alpha * (1 - gamma)
    p0 = alpha * (1.0 - gamma)
    ll0 = jnp.sum(jnp.where(m0, jnp.log(jnp.clip(p0, EPS, 1.0)), 0.0))
    ll = ll + ll0

    # Mass at 1: alpha * gamma
    p1 = alpha * gamma
    ll1 = jnp.sum(jnp.where(m1, jnp.log(jnp.clip(p1, EPS, 1.0)), 0.0))
    ll = ll + ll1

    # Interior beta density, multiplied by (1 - alpha)
    yy = jnp.clip(y, EPS, 1.0 - EPS)

    a = jnp.clip(mu * phi, EPS, jnp.inf)
    b = jnp.clip((1.0 - mu) * phi, EPS, jnp.inf)

    ll_int_each = (
        jnp.log1p(-alpha)
        + jsp.special.gammaln(phi)
        - jsp.special.gammaln(a)
        - jsp.special.gammaln(b)
        + (a - 1.0) * jnp.log(yy)
        + (b - 1.0) * jnp.log1p(-yy)
    )
    ll_int = jnp.sum(jnp.where(mint, ll_int_each, 0.0))
    ll = ll + ll_int

    return -ll


def BEINF_predictive_mean(params, X):
    """
    Predictive mean for the zero-and-one inflated beta (BEINF) model.

    Parametrization:
      mu_i    = expit(X_i @ beta)
      alpha   = expit(raw_alpha)    # total mass on {0,1}
      gamma   = expit(raw_gamma)    # conditional prob of 1 given boundary mass
      phi     = exp(raw_phi)        # precision (not needed for the mean)

    Then:
      E[Y_i | X_i] = alpha * gamma + (1 - alpha) * mu_i
    """
    params = np.asarray(params)
    X = np.asarray(X)

    p = X.shape[1]

    beta = params[:p]
    alpha = sspecial.expit(params[p])
    gamma = sspecial.expit(params[p + 1])
    # phi = np.exp(params[p + 2])   # not needed for predictive mean

    mu = sspecial.expit(X @ beta)

    return alpha * gamma + (1.0 - alpha) * mu


def BEINF_predictive_distribution(X, beta, aux_params, clip_y=1e-12, include_interior=True, aux_transformed=False):
    """
    Predictive distribution pieces for the zero-and-one inflated beta (BE-INF) model.

    Parameters
    ----------
    X : array-like, shape (n, p)
    beta : array-like, shape (p,)
    alpha : float
        Total boundary mass in (0,1), so:
          P(Y in {0,1} | x) = alpha
    gamma : float
        Conditional probability of 1 among boundary mass, so:
          P(Y=1 | Y in {0,1}, x) = gamma
    phi : float
        Beta precision parameter (> 0)
    clip_y : float, default=1e-12
        Numerical clipping for evaluating the interior density.

    Returns
    -------
    p0 : ndarray, shape (n,)
        P(Y=0 | x)
    p1 : ndarray, shape (n,)
        P(Y=1 | x)
    f_interior : callable
        Callable f_interior(y) returning the density on (0,1).
        If y has shape (m,), output has shape (m, n).
    """
    X = np.asarray(X)
    beta = np.asarray(beta)

    alpha, gamma, phi = aux_params[0], aux_params[1], aux_params[2]
    if not aux_transformed:
        alpha, gamma, phi = sspecial.expit(alpha), sspecial.expit(gamma), np.exp(phi)

    mu = sspecial.expit(X @ beta)   # shape (n,)

    # point masses
    p0 = alpha * (1.0 - gamma) * np.ones_like(mu)
    p1 = alpha * gamma * np.ones_like(mu)

    if include_interior:
        def f_interior(y):
            y = np.asarray(y)
            yy = np.clip(y, clip_y, 1.0 - clip_y)

            # reshape so output is (m, n) when y is (m,)
            yy2 = np.atleast_1d(yy)[..., None]      # (..., 1)
            mu2 = mu[None, ...]                     # (1, n)

            a = np.clip(mu2 * phi, clip_y, None)
            b = np.clip((1.0 - mu2) * phi, clip_y, None)

            # log Beta density + log(1-alpha)
            log_f = (
                np.log1p(-alpha)
                + sspecial.gammaln(phi)
                - sspecial.gammaln(a)
                - sspecial.gammaln(b)
                + (a - 1.0) * np.log(yy2)
                + (b - 1.0) * np.log1p(-yy2)
            )
            return np.exp(log_f)

        return p0, p1, f_interior
    else:
        return p0, p1



# ------------------------------------------------------------------
# CODE FOR (RANDOMIZED) PIT DIAGRAMS
# ------------------------------------------------------------------
def BEINF_randomized_pit(y, eta, aux_params, eps=1e-12, aux_transformed=True):
    """
    Randomized PIT for the zero-and-one inflated beta (BEINF) model.

    Parametrization:
      mu_i    = expit(X_i @ beta)
      alpha   = expit(raw_alpha)    # total boundary mass on {0,1}
      gamma   = expit(raw_gamma)    # conditional prob of 1 among boundary mass
      phi     = exp(raw_phi)        # beta precision

    Hence:
      P(Y=0) = alpha * (1 - gamma)
      P(Y=1) = alpha * gamma
      f(Y=y in (0,1)) = (1-alpha) * Beta(y; a_i, b_i)

    Parameters
    ----------
    params : array-like, shape (p + 3,)
        Parameter vector [beta..., raw_alpha, raw_gamma, raw_phi].
    X : array-like, shape (n, p)
        Design matrix.
    y : array-like, shape (n,)
        Observed responses in [0, 1].
    rng : np.random.Generator or None
        RNG for randomized PIT at atoms.
    eps : float
        Numerical stability constant.

    Returns
    -------
    pit : ndarray, shape (n,)
        Randomized PIT values.
    """
    
    rng = np.random.default_rng()

    alpha, gamma, phi = aux_params[0], aux_params[1], aux_params[2]
    if not aux_transformed:
        alpha = sspecial.expit(alpha)
        gamma = sspecial.expit(gamma)
        phi = np.exp(phi)

    mu = sspecial.expit(eta)
    mu = np.clip(mu, eps, 1.0 - eps)

    # boundary masses
    p0 = alpha * (1.0 - gamma)
    p1 = alpha * gamma

    # beta parameters for interior
    a = np.clip(mu * phi, eps, np.inf)
    b = np.clip((1.0 - mu) * phi, eps, np.inf)

    pit = np.empty_like(y, dtype=float)

    # y = 0
    mask0 = (y == 0.0)
    pit[mask0] = rng.uniform(0.0, p0, size=np.sum(mask0))

    # 0 < y < 1
    maskm = (y > 0.0) & (y < 1.0)
    if np.any(maskm):
        yy = np.clip(y[maskm], eps, 1.0 - eps)
        beta_cdf = sp.special.betainc(a[maskm], b[maskm], yy)
        pit[maskm] = p0 + (1.0 - alpha) * beta_cdf

    # y = 1
    mask1 = (y == 1.0)
    pit[mask1] = rng.uniform(1.0 - p1, 1.0, size=np.sum(mask1))

    return pit



# ------------------------------------------------------------------
# LIKELIHOODS FOR EVALUATION (CRPS, Log-score)
# ------------------------------------------------------------------

# Sampling procedure for BE-INF random variable
# -----------------------------------------------
def sample_BEINF(mu, aux_params, aux_transformed=True, eps=1e-12):
    """
    Sample from the zero-and-one inflated beta (BEINF) distribution.

    Parameters
    ----------
    mu : array-like
        Mean of the beta interior component, on the response scale, so mu in (0, 1).
        Can be shape (n_points, n_samples), (n_samples,), or scalar.
    aux_params : sequence of length 3
        [alpha, gamma, phi] if aux_transformed=True
        [raw_alpha, raw_gamma, raw_phi] if aux_transformed=False

        Parameterization:
          P(Y=0) = alpha * (1 - gamma)
          P(Y=1) = alpha * gamma
          P(Y in (0,1)) = 1 - alpha

        Interior beta component:
          Y | interior ~ Beta(mu * phi, (1 - mu) * phi)
    aux_transformed : bool, default=True
        Whether aux_params are already transformed to the constrained scale.
    eps : float, default=1e-12
        Small constant for numerical stability.

    Returns
    -------
    y : np.ndarray
        Samples with same shape as mu.
    """
    mu = np.asarray(mu)

    alpha, gamma, phi = aux_params[0], aux_params[1], aux_params[2]
    if not aux_transformed:
        alpha = sspecial.expit(alpha)
        gamma = sspecial.expit(gamma)
        phi = np.exp(phi)

    # Clamp for numerical safety
    mu = np.clip(mu, eps, 1.0 - eps)
    alpha = np.clip(alpha, eps, 1.0 - eps)
    gamma = np.clip(gamma, eps, 1.0 - eps)
    phi = max(phi, eps)

    # Mixture probabilities
    p0 = alpha * (1.0 - gamma)
    p1 = alpha * gamma
    pint = 1.0 - alpha

    # Draw mixture component for each entry in mu
    u = np.random.rand(*mu.shape)

    m0 = u < p0
    m1 = (u >= p0) & (u < p0 + p1)
    mint = ~(m0 | m1)

    y = np.empty_like(mu, dtype=float)

    # Boundary masses
    y[m0] = 0.0
    y[m1] = 1.0

    # Interior beta draws
    if np.any(mint):
        a = np.clip(mu[mint] * phi, eps, None)
        b = np.clip((1.0 - mu[mint]) * phi, eps, None)
        y[mint] = np.random.beta(a, b)

    return y
