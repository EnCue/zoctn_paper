import numpy as np

import pandas as pd

# from scipy import stats
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

import gpboost as gpb

import sys
sys.path.append("PATH TO EVAL UTILS FOLDER")
from eval_utils import ind_logscore, gp_logscore, crps, AIC_BIC

sys.path.append("PATH TO FIT LM UTILS FOLDER")
from fit_lm import fit_ILM


# ------------------------------------------------------------------
# CODE FOR EVALUATING MODEL TYPES: ILM, ITB, GPLM, GPB
# ------------------------------------------------------------------

TIME_VERSIONING_DATA_PATH = "PATH TO ANNUAL LGD DATA SPLITS"
TRAINED_MODELS_DATA_PATH = "PATH TO TRAINED GPBOOST MODELS (JSON FILES)"


# EVALUATION FOR GPB MODELS
# -------------------------------------------------------
def eval_gpb(location_type, likelihood, year, feature_cols, target_col, coords_cols, ll_given_mu, sample_func):
    test_df = pd.read_feather(TIME_VERSIONING_DATA_PATH+f"{year}_snapshot/testing_data.feather")
    X_test_df, Y_test_df = test_df[feature_cols], test_df[target_col]

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
    eval_dict['OS_MSE'] = mean_squared_error(Y_test, Yhat_OS)
    
    eval_dict['OS_LS'] = gp_logscore(booster, 'GPB', X_test, Y_test.flatten(), aux_params, ll_given_mu, coords_test, N_mc=1000)

    eval_dict['OS_CRPS'] = crps(booster, 'GPB', X_test, Y_test.flatten(), aux_params, sample_func, coords=coords_test, N_mc=200)

    return eval_dict


# EVALUATION FOR GPLM MODELS
# -------------------------------------------------------
def eval_gplm(location_type, likelihood, year, feature_cols, target_col, coords_cols, ll_given_mu, sample_func):
    test_df = pd.read_feather(TIME_VERSIONING_DATA_PATH+f"{year}_snapshot/testing_data.feather")
    X_test_df, Y_test_df = test_df[feature_cols], test_df[target_col]

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
    eval_dict['OS_MSE'] = mean_squared_error(Y_test, Yhat_OS)
    
    eval_dict['OS_LS'] = gp_logscore(model, 'GPLM', Xaug_test, Y_test.flatten(), aux_params, ll_given_mu, coords_test, N_mc=1000)

    eval_dict['OS_CRPS'] = crps(model, 'GPLM', Xaug_test, Y_test.flatten(), aux_params, sample_func, coords=coords_test)

    return eval_dict


# EVALUATION FOR ITB MODELS
# -------------------------------------------------------
def eval_itb(model_info, year, feature_cols, target_col):
    # Importing data
    test_df = pd.read_feather(TIME_VERSIONING_DATA_PATH+f"{year}_snapshot/testing_data.feather")
    X_test_df, Y_test_df = test_df[feature_cols], test_df[target_col]

    categorical_cols = ["occupancy", "loan_purpose", "first_time_homebuyer", "MSA", "number_of_borrowers"]
    X_test, Y_test = pd.get_dummies(X_test_df, columns=categorical_cols, drop_first=True).to_numpy(), Y_test_df.to_numpy()
    
    (N_test, P_test) = X_test.shape

    test_group = np.zeros(N_test)

    eval_dict = {}
    
    for nll_name in model_info.keys():
        ll_given_loc, sample_func = model_info[nll_name]['ll_given_loc'], model_info[nll_name]['sample_func']
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

        eval_dict[nll_name] = {
            'OS_MSE': os_mse,
            'OS_LS': os_ls,
            'OS_CRPS': os_crps,
        }

    return eval_dict


# EVALUATION FOR ILMs
# -------------------------------------------------------
def eval_glm(nlls, year, feature_cols, target_col, stage_2=False):
    train_df = pd.read_feather(TIME_VERSIONING_DATA_PATH+f"{year}_snapshot/training_data.feather")
    test_df = pd.read_feather(TIME_VERSIONING_DATA_PATH+f"{year}_snapshot/testing_data.feather")

    train_idx, test_idx = train_df.index.to_numpy(), test_df.index.to_numpy()
    data = pd.concat([train_df, test_df], ignore_index=False)

    X_df, Y_df = data[feature_cols], data[target_col]

    categorical_cols = ["occupancy", "loan_purpose", "first_time_homebuyer", "MSA", "number_of_borrowers"]

    X_np, Y_np = pd.get_dummies(X_df, columns=categorical_cols, drop_first=True).to_numpy(), Y_df.to_numpy()

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_np)
    # Augmenting feature matrices
    Xaug_std = np.c_[np.ones(X_std.shape[0]), X_std]

    Xaug_train, Y_train, Xaug_test, Y_test = Xaug_std[train_idx, :], Y_np[train_idx], Xaug_std[test_idx, :], Y_np[test_idx]
    
    (N_train, Paug_train) = Xaug_train.shape

    eval_dict = {}
    
    for nll_name in nlls.keys():
        nll_func, sample_func, N_aux_params = nlls[nll_name]['obj'], nlls[nll_name]['sample_func'], nlls[nll_name]['N_aux_params']

        fitted_model = fit_ILM(nll_func, Xaug_train, Y_train.flatten(), N_aux_params)
        params_hat = fitted_model['params_hat']
        beta = params_hat[:Paug_train]
        aux_params = params_hat[Paug_train:]

        # NLL Stats
        is_nll = fitted_model['state'].fun_val
        AIC, BIC = AIC_BIC(is_nll, Paug_train + N_aux_params, N_train)
        os_nll = nll_func(params_hat, Xaug_test, Y_test.flatten())

        if stage_2:
            # MSE Stats
            pred_mu_func = nlls[nll_name]['pred_mu']
            Yhat_test = pred_mu_func(params_hat, Xaug_test)
            
            os_mse = mean_squared_error(Y_test, Yhat_test)

            model_info = {'params': params_hat}
            os_ls = ind_logscore(nll_func, 'lm', model_info, Xaug_test, Y_test.flatten())

            os_crps = crps(beta, 'GLM', Xaug_test, Y_test.flatten(), aux_params, sample_func, aux_transformed=False)

            eval_dict[nll_name] = {
                'params_hat': fitted_model['params_hat'],
                'IS_NLL': is_nll,
                'AIC': AIC,
                'BIC': BIC,
                'OS_NLL': os_nll,
                'OS_MSE': os_mse,
                'OS_LS': os_ls,
                'OS_CRPS': os_crps,
            }
        else:
            
            eval_dict[nll_name] = {
                'IS_NLL': is_nll,
                'AIC': AIC,
                'BIC': BIC,
                'OS_NLL': os_nll,
            }

    return eval_dict 