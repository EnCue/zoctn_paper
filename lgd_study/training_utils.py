import numpy as np
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

import gpboost as gpb

import sys
sys.path.append("PATH TO FIT LM UTILS FOLDER")
from eval_utils import fit_ILM

TIME_VERSIONING_DATA_PATH = "data/processed_data/time_versioning/"
TRAINED_MODELS_DATA_PATH = "trained_models/"


# Fitting routine for GP linear models
# -------------------------------------------------------
def train_gplm(location_type, likelihood, year, feature_cols, target_col, coords_cols, categorical_cols):
    train_df = pd.read_feather(TIME_VERSIONING_DATA_PATH+f"{year}_snapshot/training_data.feather")

    X_train_df, Y_train_df = train_df[feature_cols], train_df[target_col]

    coords_train_df = X_train_df[coords_cols]
    coords_train = coords_train_df.to_numpy()

    model = None
    gplm_type = ""
    if location_type == 'spatial':
        gplm_type = 'sgplm'
        model = gpb.GPModel(
            gp_coords=coords_train,
            cov_function="matern",
            cov_fct_shape=1.5,
            gp_approx="vecchia", num_neighbors=20,
            likelihood=likelihood,
        )
    elif location_type == 'spatiotemporal':
        # Drop year_default
        gplm_type = 'stgplm'
        X_train_df = X_train_df.drop(['year_default'], axis=1)

        model = gpb.GPModel(
            gp_coords=coords_train,
            cov_function="matern_space_time",
            cov_fct_shape=1.5,
            gp_approx="vecchia", num_neighbors=20,
            likelihood=likelihood,
        )

    # categorical_cols = ["occupancy", "loan_purpose", "first_time_homebuyer", "MSA", "number_of_borrowers"]
    X_train, Y_train = pd.get_dummies(X_train_df, columns=categorical_cols, drop_first=True).to_numpy(), Y_train_df.to_numpy()
    Xaug_train = np.c_[np.ones(X_train.shape[0]), X_train]

    # Training model
    model.fit(y=Y_train.flatten(), X=Xaug_train)

    model.save_model(TRAINED_MODELS_DATA_PATH + f"{gplm_type}s/{likelihood}_{year}.json")


# Fitting routine for GP tree-boosting models
# -------------------------------------------------------
def train_gpb(location_type, likelihood, year, feature_cols, target_col, coords_cols, categorical_cols, params, N_rounds):
    train_df = pd.read_feather(TIME_VERSIONING_DATA_PATH+f"{year}_snapshot/training_data.feather")

    X_train_df, Y_train_df = train_df[feature_cols], train_df[target_col]

    coords_train_df = X_train_df[coords_cols]
    coords_train = coords_train_df.to_numpy()
    
    model = None
    gpb_type = ""
    if location_type == 'spatial':
        gpb_type = 'sgpb'
        model = gpb.GPModel(
            gp_coords=coords_train,
            cov_function="matern",
            cov_fct_shape=1.5,
            gp_approx="vecchia", num_neighbors=20,
            likelihood=likelihood, 
        )
    elif location_type == 'spatiotemporal':
        gpb_type = 'stgpb'
        X_train_df = X_train_df.drop(['year_default'], axis=1)

        model = gpb.GPModel(
            gp_coords=coords_train,
            cov_function="matern_space_time",
            cov_fct_shape=1.5,
            gp_approx="vecchia", num_neighbors=20,
            likelihood=likelihood, 
        )

    X_train, Y_train = pd.get_dummies(X_train_df, columns=categorical_cols, drop_first=True).to_numpy(), Y_train_df.to_numpy()
    data_train = gpb.Dataset(data=X_train, label=Y_train.flatten())

    # Training model
    booster = gpb.train(params=params, train_set=data_train, gp_model=model, num_boost_round=N_rounds)

    booster.save_model(TRAINED_MODELS_DATA_PATH + f"{gpb_type}s/{likelihood}_{year}.json")


# Fitting routine for independent tree-boosting models
# -------------------------------------------------------
def train_itb(nlls, year, feature_cols, target_col, categorical_cols):
    train_df = pd.read_feather(TIME_VERSIONING_DATA_PATH+f"{year}_snapshot/training_data.feather")

    X_train_df, Y_train_df = train_df[feature_cols], train_df[target_col]

    X_train, Y_train = pd.get_dummies(X_train_df, columns=categorical_cols, drop_first=True).to_numpy(), Y_train_df.to_numpy()
    
    (N_train, P_train) = X_train.shape

    for nll_name in nlls.keys():
        hyperparams, N_rounds = nlls[nll_name]['hyperparams'], nlls[nll_name]['N_rounds']

        model = None
        if nll_name == 'binomial_logit':
            qb_weights = np.ones(N_train)
            model = gpb.GPModel(
                num_data=N_train,
                likelihood=nll_name,
                weights=qb_weights,
            )
        else:
            model = gpb.GPModel(
                num_data=N_train,
                likelihood=nll_name,
            )

        data_train = gpb.Dataset(data=X_train, label=Y_train.flatten())

        # Training model
        booster = gpb.train(
            params=hyperparams, 
            train_set=data_train, 
            gp_model=model, 
            num_boost_round=N_rounds
        )

        booster.save_model(TRAINED_MODELS_DATA_PATH + f"itbs/{nll_name}_{year}.json")


# Fitting routine for independent linear models
# -------------------------------------------------------
def train_ilm(nlls, year, feature_cols, target_col, categorical_cols):
    train_df = pd.read_feather(TIME_VERSIONING_DATA_PATH+f"{year}_snapshot/training_data.feather")
    # test_df = pd.read_feather(TIME_VERSIONING_DATA_PATH+f"{year}_snapshot/testing_data.feather")

    # train_idx, test_idx = train_df.index.to_numpy(), test_df.index.to_numpy()
    # data = pd.concat([train_df, test_df], ignore_index=False)

    X_df, Y_df = train_df[feature_cols], train_df[target_col]

    X_np, Y_np = pd.get_dummies(X_df, columns=categorical_cols, drop_first=True).to_numpy(), Y_df.to_numpy()

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_np)
    # Augmenting feature matrices
    Xaug_std = np.c_[np.ones(X_std.shape[0]), X_std]

    # Xaug_train, Y_train = Xaug_std[train_idx, :], Y_np[train_idx]
    
    for nll_name in nlls.keys():
        nll_func, N_aux_params = nlls[nll_name]['obj'], nlls[nll_name]['N_aux_params']
        
        fitted_model = fit_ILM(nll_func, Xaug_std, Y_np.flatten(), N_aux_params)

        with open(TRAINED_MODELS_DATA_PATH + f'ilms/{nll_name}_{year}_fixed.pickle', 'wb') as handle:
            pickle.dump(fitted_model, handle)