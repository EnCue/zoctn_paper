import numpy as np

import scipy as sp
from scipy import stats
import scipy.special as sspecial

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax.scipy import special as jspecial

# TFP NEEDED FOR INCOMPLETE BETA IMPLEMENTATION (AUTOMATIC-DIFFERENTIABLE)
import tensorflow_probability.substrates.jax as tfp_jax

jax.config.update("jax_enable_x64", True)

EPS = 1e-12


# ------------------------------------------------------------------
# LIKELIHOODS FOR LINEAR MODELS
# ------------------------------------------------------------------

# Helper functions
def beta_logpdf_jax(t, a, b):
    # log f_B(t; a,b) for t in (0,1)
    return (a - 1.0) * jnp.log(t) + (b - 1.0) * jnp.log1p(-t) - jsp.special.betaln(a, b)

def beta_cdf_jax(t, a, b):
    # F_B(t; a,b) using regularized incomplete beta I_t(a,b)
    return tfp_jax.math.betainc(a, b, t) # tfps_betainc(a, b, t)


# JAX-ified NLL method
# -----------------------------------------------
def ZOCTB_nll(params, X, y):
    """
    params = [beta (p,), logk, logxi]
    X: (n,p), y: (n,)
    """
    p = X.shape[1]
    beta = params[:p]
    phi = jnp.exp(params[p])       # > 0
    u = jnp.exp(params[p + 1])       # > 0

    # Masks used for censored likelihood
    m0 = (y == 0)
    m1 = (y == 1.0)
    mint = ~(m0 | m1)

    eta = X @ beta
    mu = jsp.special.expit(eta)
    mu = jnp.clip(mu, EPS, 1.0 - EPS)

    a = mu * phi
    b = (1.0 - mu) * phi

    denom = 1.0 + 2.0 * u
    t0 = u / denom
    t1 = (1.0 + u) / denom

    ll = 0.0

    # Mass at 0: Phi(-mu/sigma)
    p0 = beta_cdf_jax(t0, a, b)
    ll0 = jnp.sum(jnp.where(m0, jnp.log(jnp.clip(p0, EPS, 1.0)), 0.0))
    ll = ll + ll0

    # Mass at 1: 1 - Phi((1-mu)/sigma)
    c = beta_cdf_jax(t1, a, b)
    p1 = 1.0 - c
    ll1 = jnp.sum(jnp.where(m1, jnp.log(jnp.clip(p1, EPS, 1.0)), 0.0))
    ll = ll + ll1

    # Interior pdf
    t = (y + u) / denom
    t = jnp.clip(t, EPS, 1.0 - EPS)
    pint = beta_logpdf_jax(t, a, b) - jnp.log(denom)
    ll_int = jnp.sum(jnp.where(mint, pint, 0.0))
    ll = ll + ll_int

    return -ll


# Predictive distribution for independent linear model
# -----------------------------------------------
def ZOCTB_predictive_distribution(X, beta, aux_params, clip_mu=1e-12, include_interior=True, aux_transformed=True):
    """
    Predictive distribution pieces for the zero-one-censored transformed-beta model.

    Model:
      mu(x) = sigmoid(X beta)
      a = mu * phi,  b = (1-mu) * phi
      t0 = u/(1+2u), t1 = (1+u)/(1+2u)
      P(Y=0) = F_B(t0; a,b)
      f_Y(y) = f_B(t; a,b) / (1+2u),  t=(y+u)/(1+2u) for y in (0,1)
      P(Y=1) = 1 - F_B(t1; a,b)

    Returns:
      p0: shape (n,)
      p1: shape (n,)
      f_interior: callable(y) -> array of shape (len(y), n) if y is 1D,
                  or shape (..., n) if y is broadcastable.
    """
    X = np.asarray(X)
    beta = np.asarray(beta)

    phi, u = aux_params[0], aux_params[1]
    if not aux_transformed:
        phi, u = np.exp(phi), np.exp(u)

    # mean regression
    eta = X @ beta
    mu = sp.special.expit(eta)

    # beta parameters
    a = mu * phi
    b = (1.0 - mu) * phi

    # transform constants
    denom = 1.0 + 2.0 * u
    t0 = u / denom
    t1 = (1.0 + u) / denom

    # CDFs for masses
    p0 = sp.special.betainc(a, b, t0)
    p1 = 1.0 - sp.special.betainc(a, b, t1)

    if include_interior:
        def f_interior(y):
            """
            Interior density for y in (0,1).
            If y is shape (m,), returns (m, n).
            """
            y = np.asarray(y)

            # map y -> t
            t = (y + u) / denom  # broadcast y over observations

            # for numerical stability in log terms, keep away from 0/1
            t = np.clip(t, 1e-15, 1.0 - 1e-15)

            # log beta pdf: (a-1)log t + (b-1)log(1-t) - betaln(a,b)
            logpdf = (a - 1.0) * np.log(t) + (b - 1.0) * np.log1p(-t) - sp.special.betaln(a, b)

            # Jacobian: dt/dy = 1/(1+2u)
            return np.exp(logpdf) / denom

        return p0, p1, f_interior
    else:
        return p0, p1

# Predictive mean of ZOC-TB independent linear model
# -----------------------------------------------
def ZOCTB_predictive_mean(params, X): # X, beta, phi, u
    """
    Mean of Y where:
      P(Y=0) = F_B(t0; a,b),   t0 = u/(1+2u)
      f_Y(y) = f_B(t; a,b)/(1+2u),  y in (0,1) with t = (y+u)/(1+2u)
      P(Y=1) = 1 - F_B(t1; a,b), t1 = (1+u)/(1+2u)
    and a = mu*phi, b = (1-mu)*phi, mu = sigmoid(location_par).

    Returns E[Y | phi,u,location_par].
    """
    params = np.asarray(params)
    X = np.asarray(X)
    p = X.shape[1]
    beta = params[:p]
    phi = np.exp(params[p])       # > 0
    u = np.exp(params[p + 1])       # > 0

    eta = X @ beta
    mu = sp.special.expit(eta)  # 1/(1+exp(-location_par))
    a = mu * phi
    b = (1 - mu) * phi

    # thresholds in Beta-space
    denom = 1.0 + 2.0 * u
    t0 = u / denom
    t1 = (1.0 + u) / denom

    # probabilities of atoms
    F0 = stats.beta.cdf(t0, a, b)   # P(T <= t0) = P(Y=0)
    F1 = stats.beta.cdf(t1, a, b)   # P(T <= t1)
    p1 = 1.0 - F1                  # P(Y=1) = P(T > t1)
    p_cont = F1 - F0               # P(t0 < T < t1)

    # E[T * 1{t0<T<t1}] for T~Beta(a,b):
    #   ∫_{t0}^{t1} t f_{a,b}(t) dt = (a/(a+b)) * (F_{a+1,b}(t1) - F_{a+1,b}(t0))
    ET_trunc = (a / (a + b)) * (stats.beta.cdf(t1, a + 1.0, b) - stats.beta.cdf(t0, a + 1.0, b))

    # On the continuous region: Y = (1+2u)T - u
    E_cont = (1.0 + 2.0 * u) * ET_trunc - u * p_cont

    EY = p1 + E_cont

    return EY



# ------------------------------------------------------------------
# CODE FOR (RANDOMIZED) PIT DIAGRAMS
# ------------------------------------------------------------------
def ZOCTB_randomized_pit(y, eta, aux_params, clip_mu=1e-12, aux_transformed=True):
    """
    Randomized PIT for the zero-one-censored transformed-beta model.

    Parameters
    ----------
    y : array-like, shape (n,)
        Observed responses in [0, 1].
    mu : array-like, shape (n,)
        Mean parameter in (0, 1).
    phi : float
        Precision parameter (> 0).
    u : float
        Transformation/censoring parameter.
    rng : np.random.Generator or None
        Random number generator.
    clip_mu : float
        Numerical clipping for mu.

    Returns
    -------
    pit : ndarray, shape (n,)
        Randomized PIT values.
    """
    rng = np.random.default_rng()

    phi, u = aux_params[0], aux_params[1]
    if not aux_transformed:
        phi, u = np.exp(phi), np.exp(u)

    mu = sp.special.expit(eta)

    # clip mu for numerical stability
    mu = np.clip(mu, clip_mu, 1.0 - clip_mu)

    # beta parameters
    a = mu * phi
    b = (1.0 - mu) * phi

    # transform constants
    denom = 1.0 + 2.0 * u
    t0 = u / denom
    t1 = (1.0 + u) / denom

    # point masses
    p0 = sp.special.betainc(a, b, t0)
    p1 = 1.0 - sp.special.betainc(a, b, t1)

    pit = np.empty_like(y, dtype=float)

    # y = 0 atom
    mask0 = (y == 0.0)
    pit[mask0] = rng.uniform(0.0, p0[mask0])

    # 0 < y < 1 interior
    maskm = (y > 0.0) & (y < 1.0)
    if np.any(maskm):
        t = (y[maskm] + u) / denom
        t = np.clip(t, 1e-15, 1.0 - 1e-15)
        pit[maskm] = sp.special.betainc(a[maskm], b[maskm], t)

    # y = 1 atom
    mask1 = (y == 1.0)
    pit[mask1] = rng.uniform(1.0 - p1[mask1], 1.0)

    return pit



# ------------------------------------------------------------------
# CODE FOR EVALUATION (CRPS, Log-score)
# ------------------------------------------------------------------
def beta_logpdf(t, a, b):
    # log f_B(t; a,b) for t in (0,1)
    t = np.clip(t, EPS, 1.0 - EPS)
    return (a - 1.0) * np.log(t) + (b - 1.0) * np.log1p(-t) - sspecial.betaln(a, b)

def beta_cdf(t, a, b):
    # F_B(t; a,b) using regularized incomplete beta I_t(a,b)
    t = np.clip(t, EPS, 1.0 - EPS)
    return sspecial.betainc(a, b, t)  # regularized incomplete beta


def ZOCTB_ll_given_loc(y, location_par, aux_params):
    """
    SciPy/NumPy version of the zero_one_censored_transformed_beta log-likelihood,
    written in a style comparable to your JAX ZOCTB_nll, but vectorized like ZOCTN_ll_given_mu.

    y: array shape (N_eval,) (or broadcastable to that)
    location_par: array shape (N_mc, N_eval)
    aux_params: [phi (>0), u (>0)]
    returns: log-likelihood contributions, shape (N_mc, N_eval)
    """
    phi = aux_params[0]   # > 0
    u   = aux_params[1]   # > 0

    # location_par: (N_mc, N_eval)
    (N_mc, N_eval) = location_par.shape

    # Masks used for censored likelihood (like your JAX code)
    # y_flat = np.asarray(y).reshape(-1)   # (N_eval,)
    m0 = (y == 0).flatten()
    m1 = (y == 1).flatten()
    mint = ((~m0) & (~m1)).flatten() # mint = ~(m0 | m1)

    # Mean on (0,1)
    mu = sspecial.expit(location_par)
    mu = np.clip(mu, EPS, 1.0 - EPS)

    # Beta shapes
    a = mu * phi
    b = (1.0 - mu) * phi

    denom = 1.0 + 2.0 * u
    t0 = u / denom
    t1 = (1.0 + u) / denom

    # Accumulate ll like the JAX implementation, but per-(mc,eval)
    ll = np.zeros((N_mc, N_eval), dtype=float)

    # Mass at 0: log F_B(t0; a,b)
    if np.any(m0):
        p0 = beta_cdf(t0, a, b)
        ll0 = np.log(np.clip(p0, EPS, 1.0))
        ll = ll + np.where(m0, ll0, 0.0)

    # Mass at 1: log(1 - F_B(t1; a,b))
    if np.any(m1):
        c  = beta_cdf(t1, a, b)
        p1 = 1.0 - c
        ll1 = np.log(np.clip(p1, EPS, 1.0))
        ll = ll + np.where(m1, ll1, 0.0)

    # Interior pdf: log f_B(t;a,b) - log(denom), where t=(y+u)/denom
    if np.any(mint):
        # shape (1, N_eval) so it broadcasts across N_mc like your ZOCTN code
        y_clip = np.clip(y.T, EPS, 1.0 - EPS) # ?
        t = (y_clip + u) / denom
        t = np.clip(t, EPS, 1.0 - EPS)

        pint = beta_logpdf(t, a, b) - np.log(denom)
        ll = ll + np.where(mint, pint, 0.0)

    return ll

def sample_ZOCTB(location_par, aux_params, aux_transformed=True):
    """
    Draw samples under:
      T ~ Beta(a,b), a=mu*phi, b=(1-mu)*phi, mu=expit(location_par)
      Y = clip( (1+2u)*T - u, 0, 1 )

    location_par can be scalar or array-like.
    Returns array of same shape as location_par.
    """
    phi = aux_params[0]
    u = aux_params[1]
    if not aux_transformed:
        phi, u = np.exp(phi), np.exp(u)
    denom = 1.0 + 2.0 * u

    mu_beta = sspecial.expit(location_par)
    a = np.clip(mu_beta * phi, EPS, np.inf)
    b = np.clip((1.0 - mu_beta) * phi, EPS, np.inf)

    # Sample T ~ Beta(a,b)
    t = np.random.beta(a, b)

    # Affine transform then censor to [0,1]
    y_raw = denom * t - u
    y = np.clip(y_raw, 0.0, 1.0)
    return y