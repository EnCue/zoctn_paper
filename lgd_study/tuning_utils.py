import pickle

import gpboost as gpb
import numpy as np

import pandas as pd

TIME_VERSIONING_DATA_PATH = "data/processed_data/time_versioning/"
TUNED_HPS_DATA_PATH = "tuned_parameters/"

def tune_vanillaboost(likelihood, year, feature_cols, categorical_cols, target_col, hp_grid, N_rnd_grid=100, seed=None):
    # Loading data
    train_df = pd.read_feather(TIME_VERSIONING_DATA_PATH+f"{year}_snapshot/training_data.feather")
    valid_df = pd.read_feather(TIME_VERSIONING_DATA_PATH+f"{year}_snapshot/testing_data.feather")

    train_idx, valid_idx = train_df.index.to_numpy(), valid_df.index.to_numpy()

    data = pd.concat([train_df, valid_df], ignore_index=True)
    folds = [(train_idx, valid_idx)]

    X_df, Y_df = data[feature_cols], data[target_col]

    X_np, Y_np = pd.get_dummies(X_df, columns=categorical_cols, drop_first=True).to_numpy(), Y_df.to_numpy()

    N = X_np.shape[0]

    model = None
    if likelihood == 'binomial_logit':
        qb_weights = np.ones(N)
        model = gpb.GPModel(
            num_data=N,
            likelihood=likelihood, 
            weights=qb_weights
        )
    else:
        model = gpb.GPModel(
            num_data=N,
            likelihood=likelihood,
        )
    dataset = gpb.Dataset(data=X_np, label=Y_np.flatten())
    
    hp_grid['max_bin'] = [250, 500, 1000, np.min([10000, N])]
    other_params = {'verbose': 0} # avoid trace information when training models
    metric = "mse" # Define metric

    opt_params = gpb.grid_search_tune_parameters(param_grid=hp_grid, params=other_params,
                                                train_set=dataset, gp_model=model,
                                                num_try_random=N_rnd_grid, folds=folds, 
                                                num_boost_round=800, early_stopping_rounds=20,
                                                verbose_eval=1, metric=metric, seed=seed)
    

    return opt_params 

def tune_gpboost(location_type, likelihood, year, feature_cols, categorical_cols, target_col, coords_cols, hp_grid=None, set_params=None, N_rnd_grid=100, seed=None):
    # Loading data
    train_df = pd.read_feather(TIME_VERSIONING_DATA_PATH+f"{year}_snapshot/training_data.feather")
    valid_df = pd.read_feather(TIME_VERSIONING_DATA_PATH+f"{year}_snapshot/testing_data.feather")

    train_idx, valid_idx = train_df.index.to_numpy(), valid_df.index.to_numpy()
    folds = [(train_idx, valid_idx)]
    data = pd.concat([train_df, valid_df], ignore_index=True)

    X_df, Y_df = data[feature_cols], data[target_col]
    coords_df = X_df[coords_cols]

    coords_np = coords_df.to_numpy()

    model = None
    if location_type == 'spatial': 
        model = gpb.GPModel(
            gp_coords=coords_np,
            cov_function="matern", 
            cov_fct_shape=1.5,
            gp_approx="vecchia", num_neighbors=20,
            matrix_inversion_method="cholesky",
            likelihood=likelihood, 
            # num_parallel_threads=16, Sometimes necessary depending on hardware
        )

        X_np, Y_np = pd.get_dummies(X_df, columns=categorical_cols, drop_first=True).to_numpy(), Y_df.to_numpy()
        dataset = gpb.Dataset(data=X_np, label=Y_np.flatten())
        
        N = X_np.shape[0]
        hp_grid['max_bin'] = [250, 500, 1000, np.min([10000, N])]
        other_params = {'verbose': 0} # avoid trace information when training models
        metric = "mse" # Define metric

        # For SGPB models we tune all eligible tree-boosting hyperparameters
        opt_params = gpb.grid_search_tune_parameters(param_grid=hp_grid, params=other_params,
                                                    train_set=dataset, gp_model=model,
                                                    num_try_random=N_rnd_grid, folds=folds, 
                                                    num_boost_round=800, early_stopping_rounds=20,
                                                    verbose_eval=1, metric=metric, seed=seed)
        
        return opt_params
    elif location_type == 'spatiotemporal':
        # Drop year_default from feature matrix
        X_df = X_df.drop(['year_default'], axis=1)

        model = gpb.GPModel(
            gp_coords=coords_np[train_idx, :],
            cov_function="matern_space_time",
            cov_fct_shape=1.5,
            gp_approx="vecchia", num_neighbors=20,
            matrix_inversion_method="cholesky",
            likelihood=likelihood, 
            # num_parallel_threads=16, Sometimes necessary depending on hardware
        )
        model.set_prediction_data(gp_coords_pred=coords_np[valid_idx, :])

        X_np, Y_np = pd.get_dummies(X_df, columns=categorical_cols, drop_first=True).to_numpy(), Y_df.to_numpy()
        data_train = gpb.Dataset(X_np[train_idx, :], Y_np.flatten()[train_idx])
        data_eval = gpb.Dataset(X_np[valid_idx, :], Y_np.flatten()[valid_idx], reference=data_train)
        
        # For STGPB models we tune only the number of trees (other params are set)
        evals_result = {}  # Optional: record eval data for plotting
        set_params['metric'] = 'mse'
        bst = gpb.train(params=set_params, train_set=data_train, num_boost_round=800,
                        gp_model=model, valid_sets=data_eval, early_stopping_rounds=20, 
                        evals_result=evals_result, verbose_eval=False, seed=seed)

        N_trees = bst.best_iteration
        return N_trees 


def tune_over_years(model_type, likelihood, year_range, feature_cols, categorical_cols, target_col, location_type="", coord_cols=None, seed=None):
    hp_grid = { 
        'learning_rate': [0.001, 0.01, 0.1, 1, 10], 
        'min_data_in_leaf': [1, 10, 100, 1000],
        'max_depth': [-1],
        'num_leaves': 2**np.arange(2,10),
        'lambda_l2': [0, 0.1, 1, 10], # 100
        'feature_fraction': [0.70, 0.8, 0.9],
        'line_search_step_length': [True, False]
    }
    N_rnd_grid = 100

    for y in year_range:
        tuning_data = {}
        if model_type == 'ind':
            tuning_data = tune_vanillaboost(likelihood, y-1, feature_cols, categorical_cols, target_col, hp_grid, N_rnd_grid, seed=seed)
        elif model_type == "gp":
            if location_type == 'spatial':
                tuning_data = tune_gpboost(location_type, likelihood, y-1, feature_cols, categorical_cols, target_col, coord_cols, hp_grid=hp_grid, N_rnd_grid=N_rnd_grid, seed=seed)
            elif location_type == 'spatiotemporal':
                # We tune only N_trees for STGPBs, all other hps are imported from SGPB models
                sgpb_tuning_dict = {}
                with open(TUNED_HPS_DATA_PATH + f'spatialgp_zoctn_hyperparams_{y}.pickle', 'rb') as handle:
                    sgpb_tuning_dict = pickle.load(handle)
            
                sgpb_params = sgpb_tuning_dict['best_params']
                N_trees = tune_gpboost(location_type, likelihood, y-1, feature_cols, categorical_cols, target_col, coord_cols, set_params=sgpb_params, N_rnd_grid=N_rnd_grid, seed=seed)
                tuning_data = {
                    'best_params': sgpb_tuning_dict,
                    'best_iter': N_trees,
                }
            
        with open(TUNED_HPS_DATA_PATH + f'{location_type}{model_type}_{likelihood}_hyperparams_{y}.pickle', 'wb') as handle:
            pickle.dump(tuning_data, handle)