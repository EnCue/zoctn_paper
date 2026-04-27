import time

import numpy as np
from scipy import special as sspecial
import pandas as pd

import jax
import jax.numpy as jnp

import sys
sys.path.append("PATH TO EVAL UTILS FOLDER")
from eval_utils import AIC_BIC, crps
sys.path.append("PATH TO FIT LM UTILS FOLDER")
from fit_lm import fit_ILM

jax.config.update("jax_enable_x64", True)

def add_intercept(X):
    return np.column_stack([np.ones(X.shape[0]), X])


# Simulate covariates -> sample response
# -----------------------------------------------
def simulate_dataset(
    sample_func,
    n_train=1000,
    n_test=1000,
    beta_true=np.array([0.50, 0.005, -0.015, 0.01]),  # includes intercept
    aux_params=None,
    mixture_dist=False
):
    # rng = np.random.default_rng()

    # 3 normal features
    Xtr_raw = np.random.normal(size=(n_train, 3))
    Xte_raw = np.random.normal(size=(n_test, 3))

    # add intercept column
    Xtr = add_intercept(Xtr_raw)
    Xte = add_intercept(Xte_raw)

    mu_tr = Xtr @ beta_true
    mu_te = Xte @ beta_true

    ytr, yte = None, None
    if not mixture_dist:
        ytr = sample_func(mu_tr, aux_params, aux_transformed=True) 
        yte = sample_func(mu_te, aux_params, aux_transformed=True)
    else:
        ytr = sample_func(Xtr, mu_tr, aux_params, aux_transformed=True)
        yte = sample_func(Xte, mu_te, aux_params, aux_transformed=True)

    return Xtr, ytr, Xte, yte


# Generate data -> evaluate models on training/testing datasets
# -----------------------------------------------
def evaluate_models_on_simdata(
    model_specs,
    sample_func,
    data_density,
    n_train=1000,
    beta_true=np.array([0.50, 0.005, -0.015, 0.01]),  # intercept + 3 slopes
    aux_params_true=None,
    mixture_dist=False,
    maxiter=500
):
    # ------------------------------------------------------
    # 1) Simulate ZOC-TN data
    # ------------------------------------------------------
    X_train, y_train, X_test, y_test = simulate_dataset(
        sample_func,
        n_train=n_train,
        n_test=n_train,
        beta_true=beta_true,  # includes intercept
        aux_params=aux_params_true,
        mixture_dist=mixture_dist
    )

    p0_test, p1_test = None, None
    if mixture_dist:
        p0_test, p1_test = boundary_probs_pdf_spec(X_test, return_pc=False)
    else:
        p0_test, p1_test = data_density(X_test, beta_true, aux_params_true, include_interior=False, aux_transformed=True)
    
    results = []
    fitted = {}
    pits = {}

    # ------------------------------------------------------
    # 2) Fit each model and evaluate
    # ------------------------------------------------------
    for name, spec in model_specs.items():

        # Parameter count
        p = X_train.shape[1]
        k = p + spec["n_aux"]

        t_start = time.time()
        fit = fit_ILM(
            nll=spec["nll"],
            X_np=X_train,
            y_np=y_train.flatten(),
            n_aux_params=spec["n_aux"],
            maxiter=maxiter
        )
        t_end = time.time()

        fitting_time = t_end - t_start

        params_hat = fit["params_hat"]
        beta, aux_params = params_hat[:p], params_hat[p:]
        fitted[name] = fit

        # -------- IN-SAMPLE METRICS
        # NLL on train/test
        train_nll = float(spec["nll"](jnp.asarray(params_hat), jnp.asarray(X_train), jnp.asarray(y_train)))
        # Classical train AIC/BIC
        _, bic = AIC_BIC(train_nll, k=k, n=n_train)

        # -------- OUT-OF-SAMPLE METRICS ------
        # Mean predictions
        yhat_test = spec["pred_mean"](params_hat, X_test)

        # MSE
        mse_test = float(np.mean((y_test - yhat_test) ** 2))
        
        # Log-score
        nll_test = float(spec["nll"](jnp.asarray(params_hat), jnp.asarray(X_test), jnp.asarray(y_test)))
        ls_test = nll_test / n_train

        # CRPS
        crps_test = crps(beta, 'GLM', X_test, y_test.flatten(), aux_params, spec['sample_func'], aux_transformed=False)

        # Boundary calibration
        bd0_score, bd1_score = None, None
        if spec['pred_density']:
            p0_hat, p1_hat = spec['pred_density'](X_test, beta, aux_params, include_interior=False, aux_transformed=False)

            d0, d1 = p0_test - p0_hat, p1_test - p1_hat
            bd0_score = np.sum(np.abs(d0))
            bd1_score = np.sum(np.abs(d1))

        # PIT values
        pit_vec = None
        if spec['pit']:
            eta_preds = X_test @ beta
            pit_vec = spec['pit'](y_test.flatten(), eta_preds, aux_params, aux_transformed=False)

        pits[name] = pit_vec

        results.append({
            "model": name,
            "fitting_time": fitting_time,
            "BIC": bic,
            "MSE": mse_test,
            "LS": ls_test,
            "CRPS": crps_test,
            "bd0_score": bd0_score,
            "bd1_score": bd1_score,
        })

    results_df = pd.DataFrame(results).sort_values(
        ["BIC"]
    ).reset_index(drop=True)

    results_df.set_index('model', inplace=True)

    return {
        "results": results_df,
        "pits": pits,
        "fitted": fitted,
    }



# ------------------------------------------------------------------
# CODE FOR SAMPLING MIXTURE DISTRIBUTIONS (Right-Skew / W-Shape)
# ------------------------------------------------------------------

# SAMPLING METHODS FOR MIXTURE DGPS
def _clip_unit(z, eps=1e-10):
    return np.clip(z, eps, 1.0 - eps)

def boundary_probs_pdf_spec(X, return_pc=True):
    """
    X_raw: (n, 3) with columns x1, x2, x3 (no intercept)
    Returns p0, p1, pc according to the PDF specification.
    """
    x0 = X[:, 0] # Intercept
    x1 = X[:, 1]
    x2 = X[:, 2]
    x3 = X[:, 3]

    omega0 = -4.5 * x0 - 3.0 * x1 + 1.0 * x2 - 0.5 * x3
    omega1 = -2.0 * x0 + 1.0 * x1 - 0.5 * x2 + 0.25 * x3

    e0 = np.exp(omega0)
    e1 = np.exp(omega1)
    denom = 1.0 + e0 + e1

    p0 = e0 / denom
    p1 = e1 / denom
    pc = 1.0 - p0 - p1

    if return_pc:
        return p0, p1, pc
    else:
        return p0, p1


def MDR_sample(X, mu, aux_params=None, aux_transformed=True):
    """
    MD_R:
      - exact PDF boundary masses p0(x), p1(x)
      - interior is right-skewed / concentrated near 1

    aux_params options:
      kappa: concentration for Beta interior (default 18.0)
      shift: positive shift to move interior mean toward 1 (default 1.25)
      heteroskedastic: bool, if True allow concentration to vary with x1
      kappa_slope: slope for log-kappa variation if heteroskedastic=True
    """
    aux_params = aux_params or {}

    mu = np.asarray(mu)
    n = len(mu)

    p0, p1, pc = boundary_probs_pdf_spec(X)

    kappa = aux_params[0]
    
    # Interior mean near 1, driven by mu
    m = sspecial.expit(mu)
    m = _clip_unit(m)

    alpha = m * kappa
    beta = (1.0 - m) * kappa

    u = np.random.uniform(size=n)
    y = np.empty(n)

    idx0 = u < p0
    idx1 = (u >= p0) & (u < p0 + p1)
    idxc = ~(idx0 | idx1)

    y[idx0] = 0.0
    y[idx1] = 1.0
    y[idxc] = np.random.beta(alpha[idxc], beta[idxc])

    return y

def MDW_sample(X, mu, aux_params=None, aux_transformed=True):
    """
    MD_W:
      - exact PDF boundary masses p0(x), p1(x)
      - W-shaped interior via 3-component Beta mixture

    aux_params options:
      centers: 3 component means, default [0.15, 0.50, 0.85]
      kappas:  3 component concentrations, default [60, 110, 60]
      weight_strength: how strongly mu tilts left/right weights, default 1.5
      center_weight: baseline boost for middle component, default 1.25
    """
    aux_params = aux_params or {}

    mu = np.asarray(mu)
    n = len(mu)

    p0, p1, pc = boundary_probs_pdf_spec(X)

    centers = aux_params[0] 
    kappas = aux_params[1] 
    weight_strength = aux_params[2]
    center_weight = aux_params[3]

    if centers.shape != (3,) or kappas.shape != (3,):
        raise ValueError("centers and kappas must have length 3.")

    # Component-specific Beta parameters
    a = centers * kappas
    b = (1.0 - centers) * kappas

    # mu-dependent mixture weights:
    # lower mu -> more left mode
    # higher mu -> more right mode
    # middle mode always present
    g = np.tanh(weight_strength * mu)

    s1 = np.exp(-g)              # left component
    s2 = np.exp(center_weight)   # middle component
    s3 = np.exp(+g)              # right component
    S = s1 + s2 + s3

    w1 = s1 / S
    w2 = s2 / S
    w3 = s3 / S

    u = np.random.uniform(size=n)
    y = np.empty(n)

    idx0 = u < p0
    idx1 = (u >= p0) & (u < p0 + p1)
    idxc = ~(idx0 | idx1)

    y[idx0] = 0.0
    y[idx1] = 1.0

    if np.any(idxc):
        v = np.random.uniform(size=idxc.sum())

        c1 = v < w1[idxc]
        c2 = (v >= w1[idxc]) & (v < w1[idxc] + w2[idxc])
        c3 = ~(c1 | c2)

        yc = np.empty(idxc.sum())

        if np.any(c1):
            yc[c1] = np.random.beta(a[0], b[0], size=c1.sum())
        if np.any(c2):
            yc[c2] = np.random.beta(a[1], b[1], size=c2.sum())
        if np.any(c3):
            yc[c3] = np.random.beta(a[2], b[2], size=c3.sum())

        y[idxc] = yc

    return y