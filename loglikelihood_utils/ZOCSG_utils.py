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

def gamma_logpdf_jax(z, k, theta):
    # log pdf of Gamma(shape=k, scale=theta) for z>0
    return ((k - 1.0) * jnp.log(z)) - (z / theta) - jspecial.gammaln(k) - (k * jnp.log(theta))

def gamma_cdf_jax(z, k, theta):
    # CDF of Gamma(shape=k, scale=theta)
    # regularized lower incomplete gamma: gammainc(k, z/theta)
    return jspecial.gammainc(k, z / theta)

def ZOCSG_nll(params, X, y):
    """
    params = [beta (p,), logk, logxi]
    X: (n,p), y: (n,)
    """
    p = X.shape[1]
    beta = params[:p]
    k = jnp.exp(params[p])       # > 0
    xi = jnp.exp(params[p + 1])       # > 0

    # Masks used for censored likelihood
    m0 = (y == 0)
    m1 = (y == 1.0)
    mint = ~(m0 | m1)

    eta = X @ beta
    mu = jnp.exp(eta)
    theta = mu / k

    ll = 0.0

    # Mass at 0: Phi(-mu/sigma)
    
    p0 = gamma_cdf_jax(xi, k, theta) # p0 = special.ndtr(z0)  # Phi
    ll0 = jnp.sum(jnp.where(m0, jnp.log(jnp.clip(p0, EPS, 1.0)), 0.0))
    ll = ll + ll0

    # Mass at 1: 1 - Phi((1-mu)/sigma)
    c = gamma_cdf_jax(xi + 1.0, k, theta)
    p1 = 1.0 - c
    ll1 = jnp.sum(jnp.where(m1, jnp.log(jnp.clip(p1, EPS, 1.0)), 0.0))
    ll = ll + ll1

    z = y + xi
    pint = gamma_logpdf_jax(z, k, theta)
    ll_int = jnp.sum(jnp.where(mint, pint, 0.0))
    ll = ll + ll_int

    return -ll

# Second-order automatic differentiation not available for jspecial.gammainc, so
# we estimate the parameter standard error using observation-wise NLL
# -----------------------------------------------
def ZOCSG_nll_i(params, x_i, y_i):
    p = x_i.shape[0]
    beta = params[:p]
    k  = jnp.exp(params[p])
    xi = jnp.exp(params[p + 1])

    eta = jnp.dot(x_i, beta)
    mu = jnp.exp(eta)
    theta = mu / k

    # three cases
    def case0(_):
        p0 = gamma_cdf_jax(xi, k, theta)
        return -jnp.log(jnp.clip(p0, EPS, 1.0))

    def case1(_):
        c = gamma_cdf_jax(xi + 1.0, k, theta)
        p1 = 1.0 - c
        return -jnp.log(jnp.clip(p1, EPS, 1.0))

    def caseint(_):
        z = y_i + xi
        ll = gamma_logpdf_jax(z, k, theta)
        return -ll

    return jax.lax.cond(
        y_i == 0.0, case0,
        lambda _: jax.lax.cond(y_i == 1.0, case1, caseint, operand=None),
        operand=None
    )

# Predictive distribution for independent linear model
# -----------------------------------------------
def ZOCSG_predictive_distribution(X, beta, aux_params, include_interior=True, aux_transformed=True):
    """
    Returns p0(x), p1(x), and a callable for interior density f(y|x) for y in (0,1).
    """
    X = np.asarray(X)
    k, xi = aux_params[0], aux_params[1]
    if not aux_transformed:
        k, xi = np.exp(k), np.exp(xi)
    
    eta = X @ beta
    mu = np.exp(eta)

    theta = mu / k

    p0 = gamma_cdf(xi, k, theta)
    p1 = 1.0 - gamma_cdf(xi + 1.0, k, theta)

    if include_interior:
        def f_interior(y):
            y = np.asarray(y)
            z = y + xi  
            logpdf = gamma_logpdf(z, k, theta)
            return np.exp(logpdf)

        return p0, p1, f_interior
    else:
        return p0, p1


# Predictive mean of ZOC-SG independent linear model
# -----------------------------------------------
def ZOCSG_predictive_mean(params, X):
    """
    Compute E[Y|x] for the zero/one-censored shifted gamma model.
    X: (n,p)
    beta: (p,)
    k, xi: scalars > 0
    """
    params = np.asarray(params)
    X = np.asarray(X)

    p = X.shape[1]
    beta = params[:p]
    k = jnp.exp(params[p])       # > 0
    xi = jnp.exp(params[p + 1])       # > 0

    mu = np.exp(X @ beta)      # latent mean of Z
    theta = mu / k             # scale

    a = xi
    b = xi + 1.0

    Fk_a = gamma_cdf(a, k, theta)
    Fk_b = gamma_cdf(b, k, theta)

    Fk1_a = gamma_cdf(a, k + 1.0, theta)
    Fk1_b = gamma_cdf(b, k + 1.0, theta)

    p1 = 1.0 - Fk_b  # P(Y=1) = P(Z>=b)

    M1_ab = theta * k * (Fk1_b - Fk1_a)  # ∫_a^b z f(z) dz

    EY = p1 + (M1_ab - xi * (Fk_b - Fk_a))
    # Numerically, clamp to [0,1] (optional but often sensible)
    return EY 



# ------------------------------------------------------------------
# CODE FOR (RANDOMIZED) PIT DIAGRAMS
# ------------------------------------------------------------------
def ZOCSG_randomized_pit(y, eta, aux_params, aux_transformed=True):
    """
    PIT / randomized PIT for the zero-one censored shifted gamma model.

    Parameters
    ----------
    y : array-like, shape (n,)
        Observed responses in [0, 1].
    eta : array-like, shape (n,)
        Linear predictor with mu = exp(eta).
    k : float
        Gamma shape.
    xi : float
        Shift.
    gamma_cdf : callable
        Gamma CDF function gamma_cdf(z, k, theta).
    randomized : bool
        If True, use randomized PIT at the atoms 0 and 1.
    rng : np.random.Generator or None
        Random number generator.

    Returns
    -------
    u : ndarray, shape (n,)
        PIT values.
    """
    rng = np.random.default_rng()

    k, xi = aux_params[0], aux_params[1]
    if not aux_transformed:
        k, xi = np.exp(k), np.exp(xi)

    mu = np.exp(eta)
    theta = mu / k

    p0 = gamma_cdf(xi, k, theta)
    p1 = 1.0 - gamma_cdf(xi + 1.0, k, theta)

    u = np.empty_like(y, dtype=float)

    # atom at 0
    mask0 = (y == 0.0)
    # if randomized:
    u[mask0] = rng.uniform(0.0, p0[mask0])
    # else:
    #     u[mask0] = p0[mask0]

    # atom at 1
    mask1 = (y == 1.0)
    # if randomized:
    u[mask1] = rng.uniform(1.0 - p1[mask1], 1.0)
    # else:
    #     u[mask1] = 1.0

    # interior
    maskm = (y > 0.0) & (y < 1.0)
    if np.any(maskm):
        z = y[maskm] + xi
        u[maskm] = gamma_cdf(z, k, theta[maskm])

    return u



# ------------------------------------------------------------------
# LIKELIHOODS FOR EVALUATION (CRPS, Log-score)
# ------------------------------------------------------------------
def gamma_logpdf(z, k, theta):
    """
    log pdf of Gamma(shape=k, scale=theta) for z>0
    Vectorized over z, theta; k can be scalar or array-broadcastable.
    """
    z = np.clip(z, EPS, np.inf)
    theta = np.clip(theta, EPS, np.inf)
    k = np.clip(k, EPS, np.inf)
    return ((k - 1.0) * np.log(z)) - (z / theta) - sspecial.gammaln(k) - (k * np.log(theta))

def gamma_cdf(z, k, theta):
    """
    CDF of Gamma(shape=k, scale=theta) at z:
      P(Z <= z) = gammainc(k, z/theta)
    SciPy's gammainc is the regularized lower incomplete gamma.
    """
    z = np.clip(z, 0.0, np.inf)
    theta = np.clip(theta, EPS, np.inf)
    k = np.clip(k, EPS, np.inf)
    return sspecial.gammainc(k, z / theta)


def ZOCSG_ll_given_loc(y, location_par, aux_params):
    """
    Zero-one censored shifted gamma log-likelihood contributions.

    Model:
      Z ~ Gamma(shape=k, scale=theta),  theta = mu/k,  mu = exp(location_par)
      Y = clip(Z - xi, 0, 1)

    Mixture:
      P(Y=0) = P(Z <= xi)
      P(Y=1) = P(Z >= 1+xi)
      For 0<y<1: log f(y) = log GammaPDF(y+xi; k, theta)

    Supports vectorized y and location_par broadcasting in the same style as
    ZOCTN_ll_given_mu.
    """
    k  = aux_params[0]   # > 0
    xi = aux_params[1]   # > 0

    (N_mc, N_eval) = location_par.shape

    # Prepare output
    out = np.zeros((N_mc, N_eval), dtype=float)

    # Identify cases (work with a flat (N_eval,) view like the Normal-based function)
    y0 = (y == 0).flatten()
    y1 = (y == 1).flatten()
    yint = ((~y0) & (~y1)).flatten()

    # mean and scale
    mu = np.exp(location_par)      # (N_mc, N_eval)
    theta = mu / k                 # (N_mc, N_eval)

    # Mass at 0: log P(Z <= xi)
    if np.any(y0):
        p0 = gamma_cdf(xi, k, theta)  # expected shape (N_mc, N_eval) via broadcasting
        log_p0 = np.log(np.clip(p0, EPS, 1.0))
        out = np.where(y0, log_p0, out)

    # Mass at 1: log P(Z >= 1+xi) = log(1 - CDF(1+xi))
    if np.any(y1):
        c = gamma_cdf(xi + 1.0, k, theta)
        c = np.clip(c, 0.0, 1.0 - EPS)          # keep strictly < 1 for log1p
        log_p1 = np.log1p(-c)                   # stable log(1 - c)
        out = np.where(y1, log_p1, out)

    # Interior pdf: log gamma_pdf(y+xi)
    if np.any(yint):
        y_clip = np.clip(y.T, EPS, 1.0 - EPS)  # (1, N_eval)
        z = y_clip + xi                                     # shift back to Gamma support
        log_f = gamma_logpdf(z, k, theta)                   # should broadcast to (N_mc, N_eval)
        out = np.where(yint, log_f, out)

    return out


# Sampling procedure for ZOC-SG random variable
# -----------------------------------------------
def sample_ZOCSG(location_par, aux_params, aux_transformed=True):
    """
    Sample from the zero_one_censored_shifted_gamma (ZOCSG) model.

    Model:
      Z ~ Gamma(shape=k, scale=theta),  theta = mu / k
      Y = clip(Z - xi, 0, 1)

    Parameters
    ----------
    location_par : array-like
        Log-mean parameter(s). Can be scalar or array.
    aux_params : array-like
        aux_params[0] = k  (>0)   shape
        aux_params[1] = xi (>0)   shift

    Returns
    -------
    y : ndarray
        Samples with same shape as location_par.
    """
    k  = aux_params[0]   # > 0
    xi = aux_params[1]   # > 0
    if not aux_transformed:
        k, xi = np.exp(k), np.exp(xi)

    location_par = np.asarray(location_par)

    # Mean and scale
    mu = np.exp(location_par)
    theta = mu / k

    # Sample latent gamma
    z = np.random.gamma(shape=k, scale=theta)

    # Shift and censor
    y = z - xi
    y = np.clip(y, 0.0, 1.0)

    return y