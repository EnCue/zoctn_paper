import numpy as np
import pickle
import pandas as pd

# from scipy import stats
import scipy as sp
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

import gpboost as gpb

import sys
# sys.path.append("PATH TO eval_utils")")
from eval_utils import ind_logscore, gp_logscore, crps, AIC_BIC # sim_portfolio_loss <- Import for portfolio loss metrics

# sys.path.append("PATH TO fit_lm")
from fit_lm import fit_ILM


# ------------------------------------------------------------------
# CODE FOR EVALUATING MODEL TYPES: ILM, ITB, GPLM, GPB
# ------------------------------------------------------------------

TIME_VERSIONING_DATA_PATH = "data/processed_data/time_versioning/"
TRAINED_MODELS_DATA_PATH = "trained_models/"

# EVALUATION FOR GPB MODELS
# -------------------------------------------------------
def eval_gpb(location_type, likelihood, year, feature_cols, target_col, coords_cols, ll_given_mu, sample_func, pit_func):
    test_df = pd.read_feather(TIME_VERSIONING_DATA_PATH+f"{year}_snapshot/testing_data.feather")
    X_test_df, Y_test_df = test_df[feature_cols], test_df[target_col]

    # Include for portfolio loss metrics if test dataset contain "upb_before_default" column (UPB taken year before default)
    # UPB_before_default_df = test_df['upb_before_default']
    # UPB_before_default = UPB_before_default_df.to_numpy()

    coords_test_df = X_test_df[coords_cols]
    coords_test = coords_test_df.to_numpy()
    
    gpb_type = ""
    if location_type == 'spatial':
        gpb_type = "sgpb"
    elif location_type == 'spatio-temporal':
        gpb_type = "stgpb"
        X_test_df = X_test_df.drop(['year_default'], axis=1)


    categorical_cols = ["occupancy", "loan_purpose", "first_time_homebuyer", "MSA", "number_of_borrowers"]
    X_test, Y_test = pd.get_dummies(X_test_df, columns=categorical_cols, drop_first=True).to_numpy(), Y_test_df.to_numpy()

    # Training model
    booster = gpb.Booster(model_file=TRAINED_MODELS_DATA_PATH + f"{gpb_type}s/{likelihood}_{year}.json")
    model = booster.gp_model

    aux_params = None
    if likelihood == 'zoctn':
        sigma = model.get_aux_pars()['sigma'].iloc[0]
        a = model.get_aux_pars()['asymmetry'].iloc[0]
        b = model.get_aux_pars()['skewness'].iloc[0]
        aux_params = [sigma, np.exp(a), b]

    # Evaluation
    eval_dict = {}

    model_os_preds = booster.predict(
        data=X_test, 
        gp_coords_pred=coords_test
    )

    Yhat_OS = model_os_preds['response_mean']
    eval_dict['MSE'] = mean_squared_error(Y_test, Yhat_OS)
    
    eval_dict['LS'] = gp_logscore(booster, 'GPB', X_test, Y_test.flatten(), aux_params, ll_given_mu, coords_test, N_mc=1000)

    eval_dict['CRPS'] = crps(booster, 'GPB', X_test, Y_test.flatten(), aux_params, sample_func, coords=coords_test, N_mc=200)

    # Include for portfolio loss metrics
    # avg_shortfall, q99_loss = sim_portfolio_loss(sample_func, 'GPB', booster, aux_params, X_test, Y_test, UPB_before_default, coords=coords_test, N_sim=10000)
    # eval_dict['OS_Shortfall'] = avg_shortfall
    # eval_dict['OS_q99'] = q99_loss

    latent_params = booster.predict(
        data=X_test, 
        gp_coords_pred=coords_test,
        pred_latent=True, 
    )
    mu_preds = latent_params['fixed_effect'] + latent_params['random_effect_mean']
    
    pit_u = pit_func(Y_test.flatten(), mu_preds, aux_params, aux_transformed=True)
    eval_dict['PIT_u'] = pit_u

    return eval_dict


# EVALUATION FOR GPLM MODELS
# -------------------------------------------------------
def eval_gplm(location_type, likelihood, year, feature_cols, target_col, coords_cols, ll_given_mu, sample_func, pit_func):
    test_df = pd.read_feather(TIME_VERSIONING_DATA_PATH+f"{year}_snapshot/testing_data.feather")
    X_test_df, Y_test_df = test_df[feature_cols], test_df[target_col]

    # Include for portfolio loss metrics if test dataset contain "upb_before_default" column (UPB taken year before default)
    # UPB_before_default_df = test_df['upb_before_default']
    # UPB_before_default = UPB_before_default_df.to_numpy()

    coords_test_df = X_test_df[coords_cols]
    coords_test = coords_test_df.to_numpy()

    gplm_type = ""
    if location_type == 'spatial':
        gplm_type = 'sgplm'
    if location_type == 'spatio-temporal':
        # Drop year_default
        gplm_type = 'stgplm'
        X_test_df = X_test_df.drop(['year_default'], axis=1)


    categorical_cols = ["occupancy", "loan_purpose", "first_time_homebuyer", "MSA", "number_of_borrowers"]
    X_test, Y_test = pd.get_dummies(X_test_df, columns=categorical_cols, drop_first=True).to_numpy(), Y_test_df.to_numpy()
    Xaug_test = np.c_[np.ones(X_test.shape[0]), X_test]

    # Evaluation
    eval_dict = {}

    model = gpb.GPModel(model_file=TRAINED_MODELS_DATA_PATH + f"{gplm_type}s/{likelihood}_{year}.json")

    aux_params = None
    if likelihood == 'zoctn':
        sigma = model.get_aux_pars()['sigma'].iloc[0]
        a = model.get_aux_pars()['asymmetry'].iloc[0]
        b = model.get_aux_pars()['skewness'].iloc[0]
        aux_params = [sigma, np.exp(a), b]

    model_os_preds = model.predict(
        X_pred=Xaug_test,
        gp_coords_pred=coords_test,
        predict_response=True
    )

    Yhat_OS = model_os_preds['mu']
    eval_dict['MSE'] = mean_squared_error(Y_test, Yhat_OS)
    
    eval_dict['LS'] = gp_logscore(model, 'GPLM', Xaug_test, Y_test.flatten(), aux_params, ll_given_mu, coords_test, N_mc=1000)

    eval_dict['CRPS'] = crps(model, 'GPLM', Xaug_test, Y_test.flatten(), aux_params, sample_func, coords=coords_test)

    # Include for portfolio loss metrics
    # avg_shortfall, q99_loss = sim_portfolio_loss(sample_func, 'GPLM', model, aux_params, Xaug_test, Y_test, UPB_before_default, coords=coords_test, N_sim=10000)
    # eval_dict['OS_Shortfall'] = avg_shortfall
    # eval_dict['OS_q99'] = q99_loss

    latent_params = model.predict(
        X_pred=Xaug_test,
        gp_coords_pred=coords_test,
        predict_response=False,
    )
    mu_preds = latent_params['mu']
    
    pit_u = pit_func(Y_test.flatten(), mu_preds, aux_params, aux_transformed=True)
    eval_dict['PIT_u'] = pit_u

    return eval_dict


# EVALUATION FOR ITB MODELS
# -------------------------------------------------------
def eval_itb(model_dict, year, feature_cols, target_col):
    # Importing data
    test_df = pd.read_feather(TIME_VERSIONING_DATA_PATH+f"{year}_snapshot/testing_data.feather")
    X_test_df, Y_test_df = test_df[feature_cols], test_df[target_col]

    # Include for portfolio loss metrics if test dataset contain "upb_before_default" column (UPB taken year before default)
    # UPB_before_default_df = test_df['upb_before_default']
    # UPB_before_default = UPB_before_default_df.to_numpy()

    categorical_cols = ["occupancy", "loan_purpose", "first_time_homebuyer", "MSA", "number_of_borrowers"]
    X_test, Y_test = pd.get_dummies(X_test_df, columns=categorical_cols, drop_first=True).to_numpy(), Y_test_df.to_numpy()
    
    (N_test, P_test) = X_test.shape

    test_group = np.zeros(N_test)

    eval_dict = {}
    
    for nll_name in model_dict.keys():
        ll_given_loc, sample_func = model_dict[nll_name]['ll_given_loc'], model_dict[nll_name]['sample_func']
        pit_func = model_dict[nll_name]['pit_func']
        booster = gpb.Booster(model_file=TRAINED_MODELS_DATA_PATH + f"itbs/{nll_name}_{year}.json")
        model = booster.gp_model

        aux_params = None
        if nll_name == 'zoctn':
            sigma = model.get_aux_pars()['sigma'].iloc[0]
            a = model.get_aux_pars()['asymmetry'].iloc[0]
            b = model.get_aux_pars()['skewness'].iloc[0]
            aux_params = [sigma, np.exp(a), b]


        if nll_name == 'zero_one_censored_shifted_gamma':
            k = model.get_aux_pars()['shape'].iloc[0]
            xi = model.get_aux_pars()['xi'].iloc[0]
            aux_params = [k, xi]


        # Predicting on test set
        model_os_preds = booster.predict(
            data=X_test, 
            group_data_pred=test_group
        )

        # Computing performance metrics
        # --------------------------------
        
        # MSE
        os_mse = mean_squared_error(Y_test, model_os_preds['response_mean'])
        
        # Log score
        model_info = {'model': booster, 'aux_params': aux_params}
        os_ls = ind_logscore(ll_given_loc, 'tb', model_info, X_test, Y_test.flatten())

        # CRPS
        os_crps = crps(booster, 'ITB', X_test, Y_test.flatten(), aux_params, sample_func, N_mc=100)

        # Include for portfolio loss metrics
        # avg_shortfall, q99_loss = sim_portfolio_loss(sample_func, 'tb', model, aux_params, X_test, Y_test, UPB_before_default, N_sim=10000)

        pit_u = None
        if pit_func:
            latent_params = booster.predict(
                data=X_test, 
                group_data_pred=test_group,
                pred_latent=True,
            )
            mu_preds = latent_params['fixed_effect']
            pit_u = pit_func(Y_test.flatten(), mu_preds, aux_params, aux_transformed=True)
            
        eval_dict[nll_name] = {
            'MSE': os_mse,
            'LS': os_ls,
            'CRPS': os_crps,
            # 'Shortfall' = avg_shortfall,
            # 'q99' = q99_loss,
            'PIT_u': pit_u,
        }

    return eval_dict


# EVALUATION FOR ILMs
# -------------------------------------------------------
def eval_glm(nlls, year, feature_cols, target_col, stage_2=False):
    
    train_df = pd.read_feather(
        TIME_VERSIONING_DATA_PATH + f"{year}_snapshot/training_data.feather"
    )
    test_df = pd.read_feather(
        TIME_VERSIONING_DATA_PATH + f"{year}_snapshot/testing_data.feather"
    )

    X_train_df = train_df[feature_cols]
    Y_train = train_df[target_col].to_numpy()

    X_test_df = test_df[feature_cols]
    Y_test = test_df[target_col].to_numpy()

    categorical_cols = ["occupancy", "loan_purpose", "first_time_homebuyer", "MSA", "number_of_borrowers"]

    # One-hot encode train and test separately
    X_train_dum = pd.get_dummies(
        X_train_df,
        columns=categorical_cols,
        drop_first=True
    )

    X_test_dum = pd.get_dummies(
        X_test_df,
        columns=categorical_cols,
        drop_first=True
    )

    # Convert to numpy
    X_train_np = X_train_dum.to_numpy()
    X_test_np = X_test_dum.to_numpy()

    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train_np)

    # Apply training-set scaling coefficients to test data
    X_test_std = scaler.transform(X_test_np)

    # Add intercept column after scaling
    Xaug_train = np.c_[np.ones(X_train_std.shape[0]), X_train_std]
    Xaug_test = np.c_[np.ones(X_test_std.shape[0]), X_test_std]
    
    (N_train, Paug_train) = Xaug_train.shape

    eval_dict = {}
    
    for nll_name in nlls.keys():
        nll_func, sample_func, N_aux_params = nlls[nll_name]['obj'], nlls[nll_name]['sample_func'], nlls[nll_name]['N_aux_params']

        fitted_model = {}
        with open(TRAINED_MODELS_DATA_PATH + f'ilms/{nll_name}_{year}.pickle', 'rb') as f:
            fitted_model = pickle.load(f)
        params_hat = fitted_model['params_hat']
        beta = params_hat[:Paug_train]
        aux_params = params_hat[Paug_train:]

        if stage_2:
            pit_func = nlls[nll_name]['pit_func']

            # NLL
            os_nll = nll_func(params_hat, Xaug_test, Y_test.flatten())

            # MSE Stats
            pred_mu_func = nlls[nll_name]['pred_mu']
            Yhat_test = pred_mu_func(params_hat, Xaug_test)
            
            os_mse = mean_squared_error(Y_test, Yhat_test)

            model_info = {'params': params_hat}
            os_ls = ind_logscore(nll_func, 'lm', model_info, Xaug_test, Y_test.flatten())

            os_crps = crps(beta, 'GLM', Xaug_test, Y_test.flatten(), aux_params, sample_func, aux_transformed=False)

            # Include for portfolio loss metrics
            # avg_shortfall, q99_loss = sim_portfolio_loss(sample_func, 'glm', beta, aux_params, Xaug_test, Y_test, UPB_before_default, N_sim=10000)

            pit_u = None
            if pit_func:
                mu_preds = Xaug_test @ beta
                pit_u = pit_func(Y_test.flatten(), mu_preds, aux_params, aux_transformed=False)
            
            eval_dict[nll_name] = {
                'NLL': os_nll,
                'MSE': os_mse,
                'LS': os_ls,
                'CRPS': os_crps,
                # 'OS_Shortfall' = avg_shortfall,
                # 'OS_q99' = q99_loss,
                'PIT_u': pit_u,
            }
        else:
            is_nll = fitted_model['state'].fun_val
            AIC, BIC = AIC_BIC(is_nll, Paug_train + N_aux_params, N_train)
            eval_dict[nll_name] = {
                'IS_NLL': is_nll,
                'AIC': AIC,
                'BIC': BIC,
            }

    return eval_dict 


# IN-SAMPLE PERFORMANCE PLOTTING
# -------------------------------------------------------
import matplotlib.pyplot as plt

# AIC Plot
def plot_is_ilm_aic(eval_dict, years, N_is):
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12, 6), dpi=200)
    # Bar chart with number of observations per year
    ax_counts = axes.twinx()

    ax_counts.bar(
        years,
        N_is,
        color='darkgray', alpha=0.2, edgecolor="darkgray",
        # alpha=0.3,
        width=0.8,
        zorder=0,
        label="Yearly Defaults"
    )

    # Log scale for counts on right axis
    ax_counts.set_yscale("log")
    
    ax_counts.set_ylabel("Number of Defaults (log scale)", color="black")
    ax_counts.grid(False)

    qb_aic = [eval_dict[y]['QB']['AIC'] for y in years]
    zocn_aic = [eval_dict[y]['ZOCN']['AIC'] for y in years]
    beinf_aic = [eval_dict[y]['BEINF']['AIC'] for y in years]
    zocsg_aic = [eval_dict[y]['ZOCSG']['AIC'] for y in years]
    zoctb_aic = [eval_dict[y]['ZOCTB']['AIC'] for y in years]
    zoctn_aic = [eval_dict[y]['ZOCTN']['AIC'] for y in years]

    # ---- AIC plot ----
    # Benchmark QB
    axes.plot(years, qb_aic, linestyle='solid', color='tab:blue', label="Quasi-B")
    # ZOCN
    axes.plot(years, zocn_aic, linestyle='solid', color='tab:purple', label="ZOC-N")
    # BE-INF
    axes.plot(years, beinf_aic, linestyle='solid', color='tab:pink', label="BE-INF")
    # ZOCSG
    axes.plot(years, zocsg_aic, linestyle='solid', color='tab:orange', label="ZOC-SG")
    # ZOCTB
    axes.plot(years, zoctb_aic, linestyle='solid', color='tab:green', label="ZOC-TB")
    # ZOCSG
    axes.plot(years, zoctn_aic, linestyle='solid', color='tab:red', label="ZOC-TN")

    # axes.set_title("Model AIC")
    axes.set_xlabel("Year", fontsize=10)
    axes.set_ylabel("AIC", fontsize=10)
    axes.legend(loc='upper left')

    plt.show()

# Average performance table
def ilm_is_performance(eval_dict, years):
    qb_aic, qb_bic = [eval_dict[y]['QB']['AIC'] for y in years], [eval_dict[y]['QB']['BIC'] for y in years]
    zocn_aic, zocn_bic = [eval_dict[y]['ZOCN']['AIC'] for y in years], [eval_dict[y]['ZOCN']['BIC'] for y in years]
    beinf_aic, beinf_bic = [eval_dict[y]['BEINF']['AIC'] for y in years], [eval_dict[y]['BEINF']['BIC'] for y in years]
    zocsg_aic, zocsg_bic = [eval_dict[y]['ZOCSG']['AIC'] for y in years], [eval_dict[y]['ZOCSG']['BIC'] for y in years]
    zoctb_aic, zoctb_bic = [eval_dict[y]['ZOCTB']['AIC'] for y in years], [eval_dict[y]['ZOCTB']['BIC'] for y in years]
    zoctn_aic, zoctn_bic = [eval_dict[y]['ZOCTN']['AIC'] for y in years], [eval_dict[y]['ZOCTN']['BIC'] for y in years]

    qb_nll = [eval_dict[y]['QB']['IS_NLL'] for y in years]
    zocn_nll = [eval_dict[y]['ZOCN']['IS_NLL'] for y in years]
    beinf_nll = [eval_dict[y]['BEINF']['IS_NLL'] for y in years]
    zocsg_nll = [eval_dict[y]['ZOCSG']['IS_NLL'] for y in years]
    zoctb_nll = [eval_dict[y]['ZOCTB']['IS_NLL'] for y in years]
    zoctn_nll = [eval_dict[y]['ZOCTN']['IS_NLL'] for y in years]

    beinf_bic = [eval_dict[y]['BEINF']['BIC'] for y in years]

    d = {
        'NLL (Final)': [qb_nll[-1], zocn_nll[-1], beinf_nll[-1], zocsg_nll[-1], zoctb_nll[-1], zoctn_nll[-1]],
        'AIC (Final)': [qb_aic[-1], zocn_aic[-1], beinf_aic[-1], zocsg_aic[-1], zoctb_aic[-1], zoctn_aic[-1]],
        'BIC (Final)': [qb_bic[-1], zocn_bic[-1], beinf_bic[-1], zocsg_bic[-1], zoctb_bic[-1], zoctn_bic[-1]],
    }
    likelihoods = ['Bernoulli', 'ZOC-Normal', 'BEINF', 'ZOCS-Gamma', 'ZOCT-Beta', 'ZOCT-Normal']
    is_df = pd.DataFrame(data=d, index=likelihoods)

    return is_df

# ILM fitted coefficients
def ilm_coeffs(nll_dict, feature_cols, target_cols, categorical_cols):
    train_df = pd.read_feather(TIME_VERSIONING_DATA_PATH+f"2022_snapshot/training_data.feather")
    test_df = pd.read_feather(TIME_VERSIONING_DATA_PATH+f"2022_snapshot/testing_data.feather")

    train_idx, test_idx = train_df.index.to_numpy(), test_df.index.to_numpy()
    data = pd.concat([train_df, test_df], ignore_index=False)

    X_df, Y_df = data[feature_cols], data[target_cols]


    X_np, Y_np = pd.get_dummies(X_df, columns=categorical_cols, drop_first=True).to_numpy(), Y_df.to_numpy()

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_np)
    # Augmenting feature matrices
    Xaug_std = np.c_[np.ones(X_std.shape[0]), X_std]

    fit_models = {}

    for nll in nll_dict:
        nll_func, N_aux_params = nll_dict[nll]['obj'], nll_dict[nll]['N_aux_params']
        
        new_model = {}
        if nll == 'ZOCSG':
            ZOCSG_nll_i = nll_dict[nll]['nll_i']
            new_model = fit_ILM(nll_func, Xaug_std, Y_np.flatten(), N_aux_params, return_std=True, std_approx=True, nll_i=ZOCSG_nll_i)
        else:
            new_model = fit_ILM(nll_func, Xaug_std, Y_np.flatten(), N_aux_params, return_std=True)
        
        fit_models[nll] = new_model
    
    return fit_models

# Display estimated coefficients and standard errors as dataframes
def display_coeffs(coeff_dict):

    feature_labels = ['intercept', "X_centroid", "Y_centroid", 'credit_score', 'nr_units', 'insurance_percent',
                'original_debt_to_income', 'original_loan_to_value', 'original_upb',
                'ir_spread', 'n_months', 'ltv_at_default', 'gdp_growth',
                'ln_income_per_capita', 'ln_expenditures_per_capita',
                'unemployment_rate', 'hpi_growth', 'gdp_construction_growth',
                'gross_operating_surplus_growth', 'inflation_rate', 'DJIA_growth',
                'occupancy_P', 'occupancy_S', 'loan_purpose_N',
                'loan_purpose_P', 'first_time_homebuyer_1', 'MSA_1',
                'number_of_borrowers_2'] 

    qb_aux_labels = []
    beinf_aux_labels = ['alpha', 'gamma', 'phi']
    zocn_aux_labels = ['sigma']
    zocsg_aux_labels = ['k', 'xi']
    zoctb_aux_labels = ['phi', 'u']
    zoctn_aux_labels = ['sigma', 'a', 'b']

    def mle_dict(params, labels, std=None, std_opg=None):
        
        d = {
            "Parameter": params,
            "Std (Hessian)": std, 
        }
        mle_df = pd.DataFrame(data=d, index=labels)

        return mle_df


    qb_coeff_beta = coeff_dict['QB']['params_hat']
    qb_coeff = qb_coeff_beta 

    qb_std_beta = coeff_dict['QB']['se_hat']
    qb_std = qb_std_beta 

    qb_labels = feature_labels
    qb_mle_df = mle_dict(qb_coeff, qb_labels, std=qb_std)


    # BE-INF
    beinf_coeff_beta = coeff_dict['BEINF']['params_hat'][:-3]
    beinf_coeff_raw_aux = coeff_dict['BEINF']['params_hat'][-3:]
    # (expit, expit, exp)
    beinf_coeff_aux = np.array([sp.special.expit(np.exp(beinf_coeff_raw_aux[0])), sp.special.expit(np.exp(beinf_coeff_raw_aux[1])), np.exp(beinf_coeff_raw_aux[2])])
    beinf_coeff = np.concatenate((beinf_coeff_beta, beinf_coeff_aux), axis=0)

    beinf_std_beta = coeff_dict['BEINF']['se_hat'][:-3]
    beinf_std_raw_aux = coeff_dict['BEINF']['se_hat'][-3:]
    # Multiply by the estimate rule for delta method with g() = exp()
    beinf_std_aux = np.array([
        beinf_coeff_aux[0] * (1 - beinf_coeff_aux[0]) * beinf_std_raw_aux[0],
        beinf_coeff_aux[1] * (1 - beinf_coeff_aux[1]) * beinf_std_raw_aux[1],
        beinf_coeff_aux[2] * beinf_std_raw_aux[2]
        ])
    beinf_std = np.concatenate((beinf_std_beta, beinf_std_aux), axis=0)

    beinf_labels = feature_labels + beinf_aux_labels
    beinf_mle_df = mle_dict(beinf_coeff, beinf_labels, beinf_std)


    # ZOCN
    zocn_coeff_beta = coeff_dict['ZOCN']['params_hat'][:-1]
    zocn_coeff_raw_aux = coeff_dict['ZOCN']['params_hat'][-1:]
    # (exp)
    zocn_coeff_aux = np.array([np.exp(zocn_coeff_raw_aux[0])])
    zocn_coeff = np.concatenate((zocn_coeff_beta, zocn_coeff_aux), axis=0)

    zocn_std_beta = coeff_dict['ZOCN']['se_hat'][:-1]
    zocn_std_raw_aux = coeff_dict['ZOCN']['se_hat'][-1:]
    # Multiply by the estimate rule for delta method with g() = exp()
    zocn_std_aux = np.array([zocn_coeff_aux[0] * zocn_std_raw_aux[0]])
    zocn_std = np.concatenate((zocn_std_beta, zocn_std_aux), axis=0)

    zocn_labels = feature_labels + zocn_aux_labels
    zocn_mle_df = mle_dict(zocn_coeff, zocn_labels, std=zocn_std)


    # ZOCTN
    zoctn_coeff_beta = coeff_dict['ZOCTN']['params_hat'][:-3]
    zoctn_coeff_raw_aux = coeff_dict['ZOCTN']['params_hat'][-3:]
    # (exp, id, exp)
    zoctn_coeff_aux = np.array([np.exp(zoctn_coeff_raw_aux[0]), zoctn_coeff_raw_aux[1], np.exp(zoctn_coeff_raw_aux[2])])
    zoctn_coeff = np.concatenate((zoctn_coeff_beta, zoctn_coeff_aux), axis=0)

    zoctn_std_beta = coeff_dict['ZOCTN']['se_hat'][:-3]
    zoctn_std_raw_aux = coeff_dict['ZOCTN']['se_hat'][-3:]
    # Multiply by the estimate rule for delta method with g() = exp()
    zoctn_std_aux = np.array([zoctn_coeff_aux[0] * zoctn_std_raw_aux[0], zoctn_std_raw_aux[1], zoctn_coeff_aux[2] * zoctn_std_raw_aux[2]])
    zoctn_std = np.concatenate((zoctn_std_beta, zoctn_std_aux), axis=0)

    zoctn_labels = feature_labels + zoctn_aux_labels
    zoctn_mle_df = mle_dict(zoctn_coeff, zoctn_labels, std=zoctn_std)


    # ZOCT-B
    zoctb_coeff_beta = coeff_dict['ZOCTB']['params_hat'][:-2]
    zoctb_coeff_raw_aux = coeff_dict['ZOCTB']['params_hat'][-2:]
    # (exp, id, exp)
    zoctb_coeff_aux = np.array([np.exp(zoctb_coeff_raw_aux[0]), np.exp(zoctb_coeff_raw_aux[1])])
    zoctb_coeff = np.concatenate((zoctb_coeff_beta, zoctb_coeff_aux), axis=0)

    zoctb_std_beta = coeff_dict['ZOCTB']['se_hat'][:-2]
    zoctb_std_raw_aux = coeff_dict['ZOCTB']['se_hat'][-2:]
    # Multiply by the estimate rule for delta method with g() = exp()
    zoctb_std_aux = np.array([zoctb_coeff_aux[0] * zoctb_std_raw_aux[0], zoctb_coeff_aux[1] * zoctb_std_raw_aux[1]])
    zoctb_std = np.concatenate((zoctb_std_beta, zoctb_std_aux), axis=0)

    zoctb_labels = feature_labels + zoctb_aux_labels
    zoctb_mle_df = mle_dict(zoctb_coeff, zoctb_labels, std=zoctb_std)


    # ZOCS-G
    zocsg_coeff_beta = coeff_dict['ZOCSG']['params_hat'][:-2]
    zocsg_coeff_raw_aux = coeff_dict['ZOCSG']['params_hat'][-2:]
    # (exp, exp)
    zocsg_coeff_aux = np.array([np.exp(zocsg_coeff_raw_aux[0]), np.exp(zocsg_coeff_raw_aux[1])])
    zocsg_coeff = np.concatenate((zocsg_coeff_beta, zocsg_coeff_aux), axis=0)

    zocsg_std_beta = coeff_dict['ZOCSG']['se_hat'][:-2]
    zocsg_std_raw_aux = coeff_dict['ZOCSG']['se_hat'][-2:]
    # Multiply by the estimate rule for delta method with g() = exp()
    zocsg_std_aux = np.array([zocsg_coeff_aux[0] * zocsg_std_raw_aux[0], zocsg_coeff_aux[1] * zocsg_std_raw_aux[1]])
    zocsg_std = np.concatenate((zocsg_std_beta, zocsg_std_aux), axis=0)

    zocsg_labels = feature_labels + zocsg_aux_labels
    zocsg_mle_df = mle_dict(zocsg_coeff, zocsg_labels, std=zocsg_std)

    return qb_mle_df, zocn_mle_df, beinf_mle_df, zoctn_mle_df, zoctb_mle_df, zocsg_mle_df

# Generate suspended rootograms
def suspended_rootogram(y_obs, E0, E_int, E1, bins, bins_hist, ax=None, title="", twinx=False, start=False, end=False):
    if ax is None:
        ax = plt.gca()

    # Histogram underlay
    Obs_histogram, _ = np.histogram(y_obs, bins=bins_hist)
    bins_hist_int = bins_hist[1:-1]
    O_hist_int = Obs_histogram[1:-1]

    freq_int = np.sqrt(O_hist_int)

    left_hist_int, right_hist_int = bins_hist_int[:-1], bins_hist_int[1:]
    centers_hist_int = 0.5 * (left_hist_int + right_hist_int)
    widths_hist_int = right_hist_int - left_hist_int

    freq_bd = np.sqrt(Obs_histogram[[0, -1]])

    h_histogram = 1/32
    left_hist_bd, right_hist_bd = np.array([-h_histogram, 1.0]), np.array([0, 1 + h_histogram])
    centers_bd = np.array([0.0, 1.0])

    ax2 = ax
    if twinx:
        ax2 = ax.twinx()
    
    ax2.bar(centers_hist_int, freq_int, width=widths_hist_int, color='darkgray', alpha=0.2, edgecolor="darkgray", linewidth=1, align="center")
    ax2.bar(centers_bd, freq_bd, width=h_histogram, color='dimgray', alpha=0.2, edgecolor="dimgray", linewidth=1, align="center")
    if twinx and end:
        ax2.set_ylabel(r"Histogram ($\sqrt{Frequency}$)", color="darkgray")

    # Rootogram
    O, _ = np.histogram(y_obs, bins=bins)

    bins_int = bins[1:-1]
    O_int = O[1:-1]

    delta_int = np.sqrt(O_int) - np.sqrt(E_int)

    left_int, right_int = bins_int[:-1], bins_int[1:]
    centers_int = 0.5 * (left_int + right_int)
    widths_int = right_int - left_int

    O_bd = O[[0, -1]]
    E_bd = [E0, E1]

    delta_bd = np.sqrt(O_bd) - np.sqrt(E_bd)

    h = 1/16
    left_bd, right_bd = np.array([-h, 1.0]), np.array([0, 1 + h])
    centers_bd = np.array([0.0, 1.0]) 
    widths_bd = right_bd - left_bd

    ax.bar(centers_int, delta_int, width=widths_int, color='mediumpurple', alpha=0.8, edgecolor="rebeccapurple", linewidth=1, align="center")
    ax.bar(centers_bd, delta_bd, width=h, color='orange', alpha=0.5, edgecolor="darkorange", linewidth=1, align="center")

    ax.axhline(0.0, linewidth=1, color='rebeccapurple')
    ax.set_title(title)
    ax.set_xlabel("LGD")
    
    if start:
        if twinx:
            ax.set_ylabel(r"Rootogram ($\sqrt{Frequency}$)", color="darkgray")  
        else:
            ax.set_ylabel(r"$\sqrt{Frequency}$")
    return ax

# EXAMPLE USAGE:
# ------------------------------------
# Import bins
# rootogram_bins = {}

# output_path = "rootogram_bins.pkl"
# with open(output_path, "rb") as f:
#     rootogram_bins = pickle.load(f)
#
# import matplotlib.gridspec as gridspec

# bins_histogram_int = np.linspace(EPS_plt, 1.0 - EPS_plt, 33)
# bins_histogram = np.concatenate(([0], bins_histogram_int, [1]))

# fig = plt.figure(figsize=(12, 8), dpi=300)
# gs = gridspec.GridSpec(2, 6, figure=fig)

# # ---- Bottom row (3 plots across full width) ----
# ax1 = fig.add_subplot(gs[1, 0:2])
# ax1 = hanging_rootogram(Y_np, rootogram_bins['ZOCSG']['E0'], rootogram_bins['ZOCSG']['E_int'], rootogram_bins['ZOCSG']['E1'], bins, bins_histogram, ax=ax1, title="ZOC-SG", start=True)
# # ax1.label_outer()
# ax2 = fig.add_subplot(gs[1, 2:4], sharey=ax1)
# ax2 = hanging_rootogram(Y_np, rootogram_bins['ZOCTB']['E0'], rootogram_bins['ZOCTB']['E_int'], rootogram_bins['ZOCTB']['E1'], bins, bins_histogram, ax=ax2, title="ZOC-TB")
# # ax2.label_outer()
# ax3 = fig.add_subplot(gs[1, 4:6], sharey=ax1)
# ax3 = hanging_rootogram(Y_np, rootogram_bins['ZOCTN']['E0'], rootogram_bins['ZOCTN']['E_int'], rootogram_bins['ZOCTN']['E1'], bins, bins_histogram, ax=ax3, title="ZOC-TN",  end=True)
# # ax3.label_outer()

# # ---- Top row (2 centered plots) ----
# ax4 = fig.add_subplot(gs[0, 0:2], sharey=ax1)
# ax4 = hanging_rootogram(Y_np, rootogram_bins['QB']['E0'], rootogram_bins['QB']['E_int'], rootogram_bins['QB']['E1'], bins, bins_histogram, ax=ax4, title="Quasi-B", start=True)
# # ax4.label_outer()
# ax5 = fig.add_subplot(gs[0, 2:4], sharey=ax1)
# ax5 = hanging_rootogram(Y_np, rootogram_bins['ZOCN']['E0'], rootogram_bins['ZOCN']['E_int'], rootogram_bins['ZOCN']['E1'], bins, bins_histogram, ax=ax5, title="ZOC-N")

# ax6 = fig.add_subplot(gs[0, 4:6], sharey=ax1)
# ax6 = hanging_rootogram(Y_np, BEINF_E0, BEINF_E_int, BEINF_E1, bins, bins_histogram, ax=ax6, title="BE-INF", end=True)


# plt.tight_layout()
# plt.show()


def plot_os_zoctn_performance(years, ilm_results, itb_results, gplm_results, gpb_results):
    fig, axes = plt.subplots(
        3, 1, figsize=(12, 8), sharex=True,
        gridspec_kw={"hspace": 0.18},
        dpi=200
    )

    # Common style
    line_kw = dict(linewidth=2, markersize=4)
    grid_kw = dict(alpha=0.35, linewidth=1)

    # Top: MSE
    axes[0].plot(years, ilm_results['ZOCTN']['MSE'], marker="^", color='darkorange', label="LM", **line_kw)
    axes[0].plot(years, gplm_results['spatial']['MSE'], marker="o", color='gold', label="SGPLM", **line_kw)
    axes[0].plot(years, gplm_results['spatio-temporal']['MSE'], marker="*", color='yellow', label="STGPLM", **line_kw)

    axes[0].plot(years, itb_results['zoctn']['MSE'], marker="^", color='indigo', label="TB", **line_kw)
    axes[0].plot(years, gpb_results['spatial']['MSE'], marker="o", color='darkorchid', label="SGPB", **line_kw)
    axes[0].plot(years, gpb_results['spatio-temporal']['MSE'], marker="*", color='violet', label="STGPB", **line_kw)
    axes[0].set_ylabel("MSE", fontsize=10)
    axes[0].grid(True, **grid_kw)
    axes[0].tick_params(labelsize=8)

    # Middle: Log Score
    axes[1].plot(years, ilm_results['ZOCTN']['LS'], marker="^", color='darkorange', label="LM", **line_kw)
    axes[1].plot(years, gplm_results['spatial']['LS'], marker="o", color='gold', label="SGPLM", **line_kw)
    axes[1].plot(years, gplm_results['spatio-temporal']['LS'], marker="*", color='yellow', label="STGPLM", **line_kw)

    axes[1].plot(years, itb_results['zoctn']['LS'], marker="^", color='indigo', label="TB", **line_kw)
    axes[1].plot(years, gpb_results['spatial']['LS'], marker="o", color='darkorchid', label="SGPB", **line_kw)
    axes[1].plot(years, gpb_results['spatio-temporal']['LS'], marker="*", color='violet', label="STGPB", **line_kw)
    axes[1].set_ylabel("Log-score", fontsize=10)
    axes[1].grid(True, **grid_kw)
    axes[1].tick_params(labelsize=8)

    # Bottom: CRPS
    axes[2].plot(years, ilm_results['ZOCTN']['CRPS'], marker="^", color='darkorange', label="LM", **line_kw)
    axes[2].plot(years, gplm_results['spatial']['CRPS'], marker="o", color='gold', label="SGPLM", **line_kw)
    axes[2].plot(years, gplm_results['spatio-temporal']['CRPS'], marker="*", color='yellow', label="STGPLM", **line_kw)

    axes[2].plot(years, itb_results['zoctn']['CRPS'], marker="^", color='indigo', label="TB", **line_kw)
    axes[2].plot(years, gpb_results['spatial']['CRPS'], marker="o", color='darkorchid', label="SGPB", **line_kw)
    axes[2].plot(years, gpb_results['spatio-temporal']['CRPS'], marker="*", color='violet', label="STGPB", **line_kw)
    axes[2].set_ylabel("CRPS", fontsize=10)
    axes[2].set_xlabel("Year", fontsize=10)
    axes[2].grid(True, **grid_kw)
    axes[2].tick_params(labelsize=8)

    # X-axis formatting
    axes[2].set_xticks(years)
    axes[2].set_xticklabels(years, rotation=0)

    # Legend centered at top, like the paper
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, 0.95),
        columnspacing=1.5,
        handlelength=1.2
    )

    fig.subplots_adjust(top=0.86, bottom=0.10, left=0.12, right=0.98)
    # fig.suptitle("ZOC-TN Model Performance Across Years", fontsize=15)

    plt.show()