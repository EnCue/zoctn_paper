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
        loc_preds = loc_preds.reshape(1, N_eval)
        
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


# Portfolio Loss Evaluation Method (Shortfall and 0.99-q loss)
# -------------------------------------------------------

def pinball_loss(simulated_losses, true_loss, alpha=0.99):
    q = np.quantile(simulated_losses, alpha)
    diff = true_loss - q

    if diff < 0:
        return (1 - alpha) * (q - true_loss)
    else:
        return alpha * (true_loss - q)

def sim_portfolio_loss(sample_func, model_type, model, aux_params, X, Y, UPB_before_default, coords=None, N_sim=10000):
    N_points = X.shape[0]

    sim_lgd = None
    if model_type == 'glm':
        loc = (X @ model).reshape((1, N_points))
        loc_mc = np.tile(loc, (N_sim, 1))

        sim_lgd = sample_func(loc_mc, aux_params, aux_transformed=False)
    else:
        loc_mc = sample_mu(model, model_type, X, Y, coords, N_sim)

        sim_lgd = sample_func(loc_mc, aux_params, aux_transformed=True)


    sim_loss_dist = sim_lgd @ UPB_before_default.reshape((N_points, 1))
    mean_sim_loss = np.mean(sim_loss_dist)

    true_loss = np.dot(Y.flatten(), UPB_before_default.flatten())

    avg_shortfall = true_loss - mean_sim_loss

    q99_loss = pinball_loss(sim_loss_dist, true_loss)

    return avg_shortfall, q99_loss


# PIT Reliability Residual Plotting
# -------------------------------------------------------
import matplotlib.pyplot as plt

def plot_pit_residuals_two_panel(
    pit_by_model,
    left_models,
    right_models,
    model_colors,
    model_ls,
    display_names,
    titles=("MODEL CLASS A", "MODEL CLASS B"),
    figsize=(12, 5),
    linewidth=2.0,
    ideal_linewidth=1.2,
    show_counts=False,
    n_subsample=1000,
    symmetric_ylim=True,
    ypad=0.02,
    random_state=123,
):
    """
    Plot PIT reliability residual diagrams in two panels.

    Residual = empirical CDF(PIT) - PIT.

    Each panel has its own legend placed to the right.
    """

    rng = np.random.default_rng(random_state)
    all_models = list(dict.fromkeys(left_models + right_models))

    curves = {}
    ymin, ymax = np.inf, -np.inf

    # ---- Precompute curves + shared y-scale ----
    for model_name in all_models:

        u = np.asarray(pit_by_model[model_name])
        u = u[np.isfinite(u)]
        u = np.clip(u, 0.0, 1.0)

        if len(u) == 0:
            continue

        n_total = len(u)

        if n_subsample is not None and len(u) > n_subsample:
            idx = rng.choice(len(u), size=n_subsample, replace=False)
            u = u[idx]

        u_sorted = np.sort(u)
        ecdf = np.arange(1, len(u_sorted) + 1) / len(u_sorted)
        residual = ecdf - u_sorted

        curves[model_name] = {
            "x": u_sorted,
            "y": residual,
            "n_total": n_total,
            "n_display": len(u_sorted),
        }

        ymin = min(ymin, residual.min())
        ymax = max(ymax, residual.max())

    if symmetric_ylim:
        lim = max(abs(ymin), abs(ymax)) + ypad
        ylim = (-lim, lim)
    else:
        ylim = (ymin - ypad, ymax + ypad)

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True, dpi=200)

    def _plot_panel(ax, model_names, title):
        for model_name in model_names:
            if model_name not in curves:
                continue

            d = curves[model_name]

            label = (
                f"{display_names[model_name]} (n={d['n_total']}, shown={d['n_display']})"
                if show_counts
                else display_names[model_name]
            )

            line, = ax.step(
                d["x"],
                d["y"],
                where="post",
                color=model_colors[model_name],
                linestyle=model_ls[model_name],
                linewidth=linewidth,
                label=label,
            )

            # Improve dashed visibility
            if model_ls[model_name] == ":":
                line.set_dashes([1.5, 2.5])
            elif model_ls[model_name] == "--":
                line.set_dashes([6, 3])

        ax.axhline(
            0,
            color="black",
            linestyle="--",
            linewidth=ideal_linewidth,
            # label="Ideal",
        )

        ax.set_xlim(0, 1)
        ax.set_ylim(*ylim)
        ax.set_xlabel("PIT")
        ax.set_ylabel("Empirical CDF - PIT")
        ax.set_title(title)

        # ---- Legend to the right of this panel ----
        ax.legend(
            frameon=False,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
        )

    _plot_panel(axes[0], left_models, titles[0])
    _plot_panel(axes[1], right_models, titles[1])

    # Leave space for both legends
    fig.tight_layout(rect=[0, 0, 0.85, 1])

    return fig, axes