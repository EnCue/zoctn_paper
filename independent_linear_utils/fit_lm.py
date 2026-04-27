
import jax
import jax.numpy as jnp

from jaxopt import ScipyMinimize

jax.config.update("jax_enable_x64", True)


# JAX-ified independent linear model fitting for given dataset and model NLL
# ------------------------------------------------------
def fit_ILM(nll, X_np, y_np, n_aux_params, maxiter=500, tol=1e-8, return_std=False, std_approx=False, nll_i=None, ridge=1e-10):
    X = jnp.asarray(X_np, dtype=jnp.float64)
    y = jnp.asarray(y_np, dtype=jnp.float64)

    nll_vg = jax.jit(jax.value_and_grad(nll))

    p = X.shape[1]
    d = p + n_aux_params

    params0 = jnp.ones((d,), dtype=X.dtype)
    

    solver = ScipyMinimize(fun=nll_vg, method="L-BFGS-B", value_and_grad=True,
                        maxiter=maxiter, options={'gtol': tol})

    res = solver.run(params0, X, y)
    params_hat = res.params

    if return_std:
        # se_hat = None
        if not std_approx:
            hess_fn = jax.jit(jax.hessian(lambda theta_hat, X, y: nll(theta_hat, X, y)))
            H = hess_fn(params_hat, X, y)

            # Symmetrize for numerical stability
            H = 0.5 * (H + H.T)

            # Regularize a bit in case of near-singularity
            H_reg = H + ridge * jnp.eye(d, dtype=H.dtype)

            # Cov = inv(H); use solve rather than explicit inverse
            cov_hat = jnp.linalg.solve(H_reg, jnp.eye(d, dtype=H.dtype))
            se_hat = jnp.sqrt(jnp.clip(jnp.diag(cov_hat), a_min=0.0))

            return {
                "params_hat": params_hat,
                "state": res.state,
                'se_hat': se_hat,
            }
        else:
            # OPG
            se_opg, cov_opg, info_opg = opg_cov_se(nll_i, params_hat, X, y, ridge)
            return {
                "params_hat": params_hat,
                "state": res.state,
                'se_hat': se_opg,
            }
    else:
        return {
            "params_hat": params_hat,
            "state": res.state,
        }

# OPG method for parameter S.E. estimation
# -------------------------------------------------------------
def opg_cov_se(nll_i, params_hat, X, y, ridge=1e-8):
    """
    Outer-product-of-gradients covariance and standard errors.
    """
    # Gradient of single-observation NLL
    grad_i = jax.vmap(
        jax.grad(nll_i),
        in_axes=(None, 0, 0),
    )(params_hat, X, y)   # shape: (n, d)

    # OPG information matrix
    I = grad_i.T @ grad_i
    I = 0.5 * (I + I.T)   # symmetrize

    # Regularize in case of near-singularity
    I_reg = I + ridge * jnp.eye(I.shape[0], dtype=I.dtype)

    # Covariance = inverse information
    cov = jnp.linalg.solve(I_reg, jnp.eye(I.shape[0], dtype=I.dtype))
    se = jnp.sqrt(jnp.clip(jnp.diag(cov), a_min=0.0))

    return se, cov, I