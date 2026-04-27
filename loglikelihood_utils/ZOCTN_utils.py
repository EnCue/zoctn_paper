import numpy as np

import scipy as sp
from scipy import stats
import scipy.special as sspecial

# JAX NEEDED FOR AUTOMATIC DIFFERENTIATION
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax.scipy import special as jspecial

jax.config.update("jax_enable_x64", True)

SQRT_2PI = jnp.sqrt(2.0 * jnp.pi)

EPS = 1e-12



# ----------------------------------------------------------
# CODE FOR LINEAR MODELS
# ----------------------------------------------------------
def normal_logpdf_standard(z):
    return -0.5 * z * z - jnp.log(SQRT_2PI)

def ZOCTN_nll(params, X, y):
    p = X.shape[1]
    beta = params[:p]
    sigma = jnp.exp(params[p])       # > 0
    a = jnp.exp(params[p + 1])       # > 0
    b = jnp.exp(params[p + 2])       # > 0

    m0 = (y == 0)
    m1 = (y == 1.0)
    mint = ~(m0 | m1)

    mu = X @ beta

    ll = 0.0

    # Mass at 0: Phi(-mu/sigma)
    z0 = -mu / sigma
    p0 = jspecial.ndtr(z0)  # Phi
    ll0 = jnp.sum(jnp.where(m0, jnp.log(jnp.clip(p0, EPS, 1.0)), 0.0))
    ll = ll + ll0

    # Mass at 1: 1 - Phi((1-mu)/sigma)
    z1 = (1.0 - mu) / sigma
    p1 = 1.0 - jspecial.ndtr(z1)
    ll1 = jnp.sum(jnp.where(m1, jnp.log(jnp.clip(p1, EPS, 1.0)), 0.0))
    ll = ll + ll1

    # Interior density
    yy = jnp.clip(y, EPS, 1.0 - EPS)
    # x = g^{-1}(y) = logistic((logit(y)-log(a))/b)
    x = jspecial.expit((jspecial.logit(yy) - jnp.log(a)) / b)
    x = jnp.clip(x, 1e-15, 1.0 - 1e-15)

    z = (x - mu) / sigma

    ll_int_each = (
        -jnp.log(sigma)
        + normal_logpdf_standard(z)
        + jnp.log(x) + jnp.log1p(-x)
        - jnp.log(b)
        - (jnp.log(y) + jnp.log1p(-yy))
    )
    ll_int = jnp.sum(jnp.where(mint, ll_int_each, 0.0))
    ll = ll + ll_int

    return -ll


# Returns predictive distribution for independent linear model
def ZOCTN_predictive_distribution(X, beta, aux_params, clip_y=1e-12, include_interior=True, aux_transformed=True):
    """
    Predictive distribution pieces for the zero-one-censored transformed-normal model.

    Returns:
      p0: (n,)  P(Y=0|x)
      p1: (n,)  P(Y=1|x)
      f_interior: callable(y) returning density on (0,1).
                  If y is shape (m,), output is (m, n).
    """
    X = np.asarray(X)
    beta = np.asarray(beta)

    mu = X @ beta
    sigma, a, b = aux_params[0], aux_params[1], aux_params[2]
    if not aux_transformed:
        sigma, a, b = np.exp(sigma), np.exp(a), np.exp(b)

    # masses
    p0 = sspecial.ndtr(-mu / sigma)
    p1 = 1.0 - sspecial.ndtr((1.0 - mu) / sigma)

    if include_interior:
        def f_interior(y):
            y = np.asarray(y)
            yy = np.clip(y, clip_y, 1.0 - clip_y)

            # inverse transform to x in (0,1)
            x = sspecial.expit((sspecial.logit(yy) - np.log(a)) / b)   # (..., 1) broadcast to (..., n)
            x = np.clip(x, 1e-15, 1.0 - 1e-15)

            z = (x - mu) / sigma

            # fY = (1/sigma) * phi(z) * x(1-x) / (b*y(1-y))
            # compute in log-space for stability
            log_f = (
                -np.log(sigma)
                + normal_logpdf_standard(z)
                + np.log(x) + np.log1p(-x)
                - np.log(b)
                - (np.log(yy) + np.log1p(-yy))
            )
            return np.exp(log_f)

        return p0, p1, f_interior
    else:
        return p0, p1


# Mean of predictive ZOC-TN distribution for independent linear model
def ZOCTN_predictive_mean(params, X, n_quad=64, eps=1e-12): # X, beta, sigma, a, b,
    """
    Mean of Y = logistic(log(a) + b*logit(W)),
    where W = clamp(Z,0,1), Z ~ N(mu, sigma^2).
    """
    params = np.asarray(params)
    X = np.asarray(X)
    p = X.shape[1]

    beta = params[:p]
    sigma = np.exp(params[p])       # > 0
    a = np.exp(params[p + 1])       # > 0
    b = np.exp(params[p + 2])       # > 0

    mu = X @ beta
    sigma = np.asarray(sigma, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    # g(x)
    def g(x):
        return sspecial.expit(np.log(a) + b * sp.special.logit(x))

    # Mass at Y = 1
    p_one = stats.norm.cdf((mu - 1.0) / sigma)

    # Integration bounds in standardized coordinates
    t0 = (0.0 - mu) / sigma
    t1 = (1.0 - mu) / sigma

    # Gauss–Legendre nodes/weights on [-1, 1]
    nodes, weights = np.polynomial.legendre.leggauss(n_quad)

    # Map nodes to [t0, t1]
    half = 0.5 * (t1 - t0)
    mid  = 0.5 * (t1 + t0)
    t = mid[..., None] + half[..., None] * nodes

    z = mu[..., None] + sigma[..., None] * t

    # Integral over (0,1)
    integral = half * np.sum(
        weights * g(z) * stats.norm.pdf(t),
        axis=-1
    )

    return p_one + integral



# ----------------------------------------------------------
# CODE FOR (RANDOMIZED) PIT RELIABILITY DIAGRAMS
# ----------------------------------------------------------
def ZOCTN_randomized_pit(y, mu, aux_params, clip_y=1e-12, aux_transformed=True):
    """
    Randomized PIT for the zero-one-censored transformed-normal model.

    Model:
      X* | X ~ Normal(mu, sigma^2),   mu = X beta

      Y = 0,                         if X* <= 0
      Y = g(X*),                     if 0 < X* < 1
      Y = 1,                         if X* >= 1

    where for x in (0,1),
      g(x) = expit(log(a) + b * logit(x))

    so
      g^{-1}(y) = expit((logit(y) - log(a)) / b)

    Parameters
    ----------
    y : array-like, shape (n,)
        Observed responses in [0, 1].
    mu : array-like, shape (n, )
        Conditional means of latent normal
    aux_params : array-like, shape (sigma, a, b)
        Auxiliary parameters for model.
    aux_transformed: bool
        Whether auxiliary parameters are strictly positive

    Returns
    -------
    pit : ndarray, shape (n,)
        Randomized PIT values.
    """
    rng = np.random.default_rng()

    # mu = X @ beta
    sigma, a, b = aux_params[0], aux_params[1], aux_params[2]
    if not aux_transformed:
        sigma, a, b = np.exp(sigma), np.exp(a), np.exp(b)

    # point masses from censoring of latent X*
    p0 = sspecial.ndtr(-mu / sigma)
    p1 = 1.0 - sspecial.ndtr((1.0 - mu) / sigma)

    pit = np.empty_like(y, dtype=float)

    # y = 0 atom
    mask0 = (y == 0.0)
    pit[mask0] = rng.uniform(0.0, p0[mask0])

    # y = 1 atom
    mask1 = (y == 1.0)
    pit[mask1] = rng.uniform(1.0 - p1[mask1], 1.0)

    # 0 < y < 1 interior
    maskm = (y > 0.0) & (y < 1.0)
    if np.any(maskm):
        yy = np.clip(y[maskm], clip_y, 1.0 - clip_y)

        # inverse transform: y -> x in (0,1)
        x = sspecial.expit((sspecial.logit(yy) - np.log(a)) / b)
        x = np.clip(x, 1e-15, 1.0 - 1e-15)

        if b > 0:
            # increasing transform:
            # F_Y(y) = P(Y <= y) = P(X* <= x(y))
            pit[maskm] = sspecial.ndtr((x - mu[maskm]) / sigma)
        else:
            # decreasing transform:
            # F_Y(y) = P(X* <= 0) + P(x(y) <= X* < 1)
            #        = p0 + Φ((1-mu)/σ) - Φ((x(y)-mu)/σ)
            pit[maskm] = (
                p0[maskm]
                + sspecial.ndtr((1.0 - mu[maskm]) / sigma)
                - sspecial.ndtr((x - mu[maskm]) / sigma)
            )

    return pit



# ----------------------------------------------------------
# CODE FOR LOG-SCORE CALCULATION
# ----------------------------------------------------------
def _logphi(t):
    """log N(0,1) pdf at t"""
    return -0.5 * ((t**2) + np.log(SQRT_2PI))

def ZOCTN_ll_given_mu(y, mu, aux_params):
    """
    Log probability of Y under:
      Z ~ N(mu, sigma^2)
      W = clip(Z, 0, 1)
      Y = g(W)
      g(x) = logistic(log(a) + b * logit(x)), x in (0,1), a>0, b>0

    Resulting distribution:
      P(Y=0) = Phi(-mu/sigma)
      P(Y=1) = Phi(-(1-mu)/sigma)
      For 0<y<1:
        x = g^{-1}(y) = logistic((logit(y) - log(a))/b)
        f_Y(y) = (1/sigma) * phi((x - mu)/sigma) * x(1-x) / (b * y(1-y))

    Supports vectorized y and mu broadcasting.
    """

    sigma = aux_params[0]       # > 0
    a = aux_params[1]       # > 0
    b = aux_params[2]       # > 0
    
    # mu: (N_mc, N_eval) matrix, rows representing MC samples from GP given an observation (column)
    (N_mc, N_eval) = mu.shape

    # Prepare output with broadcasting
    out = np.zeros((N_mc, N_eval))

    # Identify cases
    y0 = (y == 0).flatten()
    y1 = (y == 1).flatten()
    yint = ((~y0) & (~y1)).flatten()

    # Point mass at 0: log Phi(-mu/sigma)
    if np.any(y0):
        z0 = (-mu) / sigma
        out = np.where(y0, sspecial.log_ndtr(z0), out)

    # Point mass at 1: log Phi(-(1-mu)/sigma)
    if np.any(y1):
        z1 = (-(1.0 - mu)) / sigma  
        out = np.where(y1, sspecial.log_ndtr(z1), out)

    # Continuous density on (0,1)
    if np.any(yint):
        y_clip = np.clip(y.T, EPS, 1.0 - EPS)  
        x = sspecial.expit((sspecial.logit(y_clip) - np.log(a)) / b)
        x = np.clip(x, EPS, 1.0 - EPS)

        z = (x - mu) / sigma  

        log_f = (
            -np.log(sigma)
            + _logphi(z)
            + np.log(x) + np.log1p(-x)
            - np.log(b)
            - np.log(y_clip) - np.log1p(-y_clip)
        )
        out = np.where(yint, log_f, out)

    return out


def sample_ZOCTN(mu, aux_params, aux_transformed=True):
    """
    
    """
    sigma, a, b = aux_params[0], aux_params[1], aux_params[2]
    if not aux_transformed:
        sigma, a, b = np.exp(sigma), np.exp(a), np.exp(b)
    
    # Sampling 1 latent per location parameter
    z = np.random.normal(mu, sigma)

    # Censoring
    w = np.clip(z, 0.0, 1.0)

    # Transforming censored latent
    y = sspecial.expit(np.log(a) + b * sspecial.logit(w))

    return y