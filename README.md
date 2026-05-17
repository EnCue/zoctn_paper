# A Censored Transformed Model for Proportional Outcomes with Boundary Mass with an Application to Spatio-Temporal Mortgage Loss Given Default Modeling

This repository contains the Python code used to produce the results presented in the paper **"A Censored Transformed Model for Proportional Outcomes with Boundary Mass with an Application to Spatio-Temporal Mortgage Loss Given Default Modeling."**

In particular, code is organized toward two objectives 1) the simulation study presented in Section 3 of the paper, and 2) the empirical application to mortgage loss given default forecasting.

The independent tree-boosted as well as Gaussian-process random effects (linear and tree-boosted) models are implemented using the GPBoost Python package: https://github.com/fabsig/GPBoost. Independent linear models implementations are handled using JAX (https://github.com/jax-ml/jax) and TensorFlow Probability (https://github.com/tensorflow/probability).

## Simulation Study
Full specification of the independent linear models considered in the article are given in the respective model's Python file (e.g. `loglikelihood_utils/ZOCTN_utils.py`). `independent_linear_utils/fit_lm.py` provides an MLE fitting routine. A brief demonstration notebook is provided to reproduce select figures from Section 3 (`simulation_study/simulation_study.ipynb`).

## LGD Application

The LGD application relies on two ancillary processes: dataset generation, and model fitting.

### Dataset Generation

Data for the LGD application is taken from the publicly available single family loan-level dataset (SFLLD) published and maintained by Freddie Mac. The retrieval procedure for this dataset is described in https://github.com/pkuendig/SpaceTimeML. Additionally used is the average interest rate issued by Freddie Mac for 30-year fixed-rate mortgages from the Federal Reserve Bank of St. Louis (https://fred.stlouisfed.org/series/MORTGAGE30US).

Once these have been downloaded and organized the `data_extraction.ipynb` notebook walks through the data processing procedure used to reproduce the final LGD dataset used in the article.

To perform one-year-ahead prediction, the dataset is partitioned into yearly folds containing defaults occurring in a given year. These expanding window datasets are saved in `lgd_study/data/process_data/time_versioning/`.

### Model Hyperparameter Tuning \& Training

Methods to perform tuning as described in the paper for tree-boosting hyperparameters is provided in `tuning_utils.py`. Tuning is performed via a random grid-search with $N=100$ samples.

Once tuned tree-boosting configurations have been found and saved to `lgd_study/tuned_parameters`, the models can be fit using methods in `training_utils.py`. 

Demo code for both tuning and training can be found in `lgd_study/model_fitting.ipynb`. Trained models are saved to `lgd_study/trained_models/`.

### Model Evaluation

Evaluation and comparison is presented in `lgd_study/model_evaluation.ipynb`, and interpretation (e.g. SHAP figures) can be found in `lgd_study/model_interpretation.ipynb`

