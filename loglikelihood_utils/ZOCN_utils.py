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

SQRT_2PI = jnp.sqrt(2.0 * jnp.pi)

def normal_logpdf_standard_jax(z):
    # log ϕ(z) where ϕ is N(0,1) pdf
    return -0.5 * z * z - jnp.log(SQRT_2PI)

def normal_logpdf_jax(y, mu, sigma):
    z = (y - mu) / sigma
    return -jnp.log(sigma) + normal_logpdf_standard_jax(z)


def ZOCN_nll(params, X, y):
    """
    Latent y* ~ Normal(mu = X @ beta, sigma)
    Observed y = 0 if y* <= 0
             y = y* if 0 < y* < 1
             y = 1 if y* >= 1

    params = [beta (p,), log_sigma (1,)]
    """
    p = X.shape[1]
    beta = params[:p]
    sigma = jnp.exp(params[p])  # > 0

    mu = X @ beta

    m0 = (y == 0.0)
    m1 = (y == 1.0)
    mint = ~(m0 | m1)

    ll = 0.0

    # Mass at 0: P(y* <= 0) = Φ((0 - mu)/sigma)
    z0 = (0.0 - mu) / sigma
    p0 = jspecial.ndtr(z0)
    ll0 = jnp.sum(jnp.where(m0, jnp.log(jnp.clip(p0, EPS, 1.0)), 0.0))
    ll = ll + ll0

    # Mass at 1: P(y* >= 1) = 1 - Φ((1 - mu)/sigma)
    z1 = (1.0 - mu) / sigma
    p1 = 1.0 - jspecial.ndtr(z1)
    ll1 = jnp.sum(jnp.where(m1, jnp.log(jnp.clip(p1, EPS, 1.0)), 0.0))
    ll = ll + ll1

    # Interior density: log f(y | 0<y<1) = log Normal(y; mu, sigma)
    ll_int_each = normal_logpdf_jax(y, mu, sigma)
    ll_int = jnp.sum(jnp.where(mint, ll_int_each, 0.0))
    ll = ll + ll_int

    return -ll

# Predictive distribution for independent linear model
# -----------------------------------------------
def ZOCN_predictive_distribution(X, beta, aux_params, include_interior=True, aux_transformed=True):
    """
    Predictive distribution pieces for the [0,1] censored normal (Tobit-type):

      y* | X ~ Normal(mu, sigma^2),  mu = X beta
      y = 0 if y* <= 0
      y = y* if 0 < y* < 1
      y = 1 if y* >= 1

    Returns:
      p0: shape (n,)     mass at 0
      p1: shape (n,)     mass at 1
      f_interior: callable(y) -> array of shape (len(y), n) if y is 1D,
                                 or shape (..., n) if y is broadcastable.
    """
    X = np.asarray(X)
    beta = np.asarray(beta)

    sigma=aux_params[0]
    if not aux_transformed:
        sigma = np.exp(sigma)

    mu = X @ beta

    z0 = (0.0 - mu) / sigma
    z1 = (1.0 - mu) / sigma

    # point masses
    p0 = stats.norm.cdf(z0)
    p1 = 1.0 - stats.norm.cdf(z1)

    if include_interior:
        def f_interior(y):
            """
            Density for y in (0,1).
            If y has shape (m,), returns (m, n).
            """
            y = np.asarray(y)
            return stats.norm.pdf(y[..., None], loc=mu, scale=sigma)

        return p0, p1, f_interior
    else:
        return p0, p1


# Predictive mean of ZOC-N independent linear model
# -----------------------------------------------
def ZOCN_predictive_mean(params, X):
    """
    Conditional expectation E[Y | X, params] for the [0,1] censored normal model.

    params = [beta (p,), log_sigma (1,)]

    Uses:
      p0 = Φ((0-mu)/σ),  p1 = 1 - Φ((1-mu)/σ),  p_cont = Φ(z1)-Φ(z0)
      E[y* 1{0<y*<1}] = mu*p_cont + σ*(φ(z0) - φ(z1))
      E[Y] = 0*p0 + 1*p1 + E[y* 1{0<y*<1}]
           = p1 + mu*p_cont + σ*(φ(z0) - φ(z1))
    """
    params = np.asarray(params)
    X = np.asarray(X)

    p = X.shape[1]
    beta = params[:p]
    sigma = np.exp(params[p])

    mu = X @ beta

    z0 = (0.0 - mu) / sigma
    z1 = (1.0 - mu) / sigma

    Phi0 = stats.norm.cdf(z0)
    Phi1 = stats.norm.cdf(z1)

    p1 = 1.0 - Phi1
    p_cont = Phi1 - Phi0

    phi0 = stats.norm.pdf(z0)
    phi1 = stats.norm.pdf(z1)

    EY = p1 + mu * p_cont + sigma * (phi0 - phi1)
    return EY



# ------------------------------------------------------------------
# CODE FOR (RANDOMIZED) PIT DIAGRAMS
# ------------------------------------------------------------------
def ZOCN_randomized_pit(y, mu, aux_params, aux_transformed=True):
    """
    Randomized PIT for the [0,1]-censored normal (Tobit-type) model.

    Model:
      y* | X ~ Normal(mu, sigma^2),  mu = X beta
      y = 0 if y* <= 0
      y = y* if 0 < y* < 1
      y = 1 if y* >= 1

    Parameters
    ----------
    y : array-like, shape (n,)
        Observed responses in [0,1].
    X : array-like, shape (n, p)
        Design matrix.
    beta : array-like, shape (p,)
        Regression coefficients.
    sigma : float
        Latent normal standard deviation.
    rng : np.random.Generator or None
        Random number generator for randomized PIT at the atoms.

    Returns
    -------
    u : ndarray, shape (n,)
        Randomized PIT values.
    """
    rng = np.random.default_rng()
    
    sigma=aux_params[0]
    if not aux_transformed:
        sigma = np.exp(sigma)

    z0 = (0.0 - mu) / sigma
    z1 = (1.0 - mu) / sigma

    p0 = stats.norm.cdf(z0)
    p1 = 1.0 - stats.norm.cdf(z1)

    u = np.empty_like(y, dtype=float)

    # atom at 0
    mask0 = (y == 0.0)
    u[mask0] = rng.uniform(0.0, p0[mask0])

    # continuous interior
    maskm = (y > 0.0) & (y < 1.0)
    u[maskm] = stats.norm.cdf((y[maskm] - mu[maskm]) / sigma)

    # atom at 1
    mask1 = (y == 1.0)
    u[mask1] = rng.uniform(1.0 - p1[mask1], 1.0)

    # optional sanity check
    bad = ~(mask0 | maskm | mask1)
    if np.any(bad):
        raise ValueError("All observed y values must lie in [0,1].")

    return u


# ------------------------------------------------------------------
# LIKELIHOODS FOR EVALUATION (CRPS, Log-score)
# ------------------------------------------------------------------

# Sampling procedure for ZOC-N random variable
# -----------------------------------------------
def sample_ZOCN(location_par, aux_params, aux_transformed=True):
    """
    Sample from the zero_one_censored_normal (ZOCN) Tobit model.

    Model:
      y* ~ Normal(mu, sigma)
      y  = clip(y*, 0, 1)

    Parameters
    ----------
    location_par : array-like
        Location parameter(s) for the latent mean mu. Can be scalar or array.
        (In your GLM this would be mu = X @ beta; here we pass it directly.)
    aux_params : array-like
        aux_params[0] = sigma (>0)
        If aux_transformed=False, sigma is interpreted as log_sigma.
    aux_transformed : bool
        If False, exponentiate aux_params[0].
    rng : np.random.Generator or None
        Optional RNG for reproducibility.

    Returns
    -------
    y : ndarray
        Samples with same shape as location_par.
    """

    sigma = aux_params[0]
    if not aux_transformed:
        sigma = np.exp(sigma)

    mu = np.asarray(location_par)

    # Sample latent normal
    z = np.random.normal(mu, sigma)

    # Censor to [0, 1]
    y = np.clip(z, 0.0, 1.0)

    return y