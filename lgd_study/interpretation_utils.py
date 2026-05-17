import numpy as np
import gpboost as gpb
import pandas as pd

import os
import matplotlib.pyplot as plt
import shap


TIME_VERSIONING_DATA_PATH = "data/processed_data/time_versioning/"
TRAINED_MODELS_DATA_PATH = "trained_models/"


def generate_shap_plots(
    model,
    X,
    feature_names=None,
    output_dir="shap_figures",
    sample_size=10000,
    random_state=42,
    max_display=8,
    numeric_features=None,
    interaction_method="auto",   # "auto", "strongest", or dict
    class_index=None,
    show=False,
):
    """
    Produce and save SHAP figures for GP tree-boosted model

    Parameters
    ----------
    model : fitted tree-based model
    X : pandas.DataFrame or numpy.ndarray
        Feature matrix used for SHAP analysis.
    feature_names : list[str] or None
        Required if X is a numpy array. Ignored if X is a DataFrame.
    output_dir : str
        Where to save output PNG files.
    sample_size : int
        Number of rows to subsample for SHAP computation.
    random_state : int
        Reproducibility seed.
    max_display : int
        Number of top features to show in Figure 8.
    numeric_features : list[str] or None
        Which features count as numeric for Figure 9. If None and X is a DataFrame,
        inferred from numeric dtypes.
    interaction_method : str or dict
        - "auto": let SHAP choose the coloring feature in dependence plots
        - "strongest": choose strongest interacting feature via SHAP utility
        - dict: map like {"feature_a": "feature_b", ...}
    class_index : int or None
        If SHAP returns one array per class, pick which class to plot.
    show : bool
        Whether to display plots interactively.
    """

    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1) Normalize X into a DataFrame so labels carry through cleanly
    # ------------------------------------------------------------------
    if isinstance(X, pd.DataFrame):
        X_df = X.copy()
    else:
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feature_names)

    # infer numeric features if not provided
    if numeric_features is None:
        numeric_features = X_df.select_dtypes(include=[np.number]).columns.tolist()

    # ------------------------------------------------------------------
    # 2) Subsample rows 
    # ------------------------------------------------------------------
    n = len(X_df)
    rng = np.random.default_rng(random_state)
    idx = rng.choice(n, size=min(sample_size, n), replace=False)
    X_sub = X_df.iloc[idx].copy()

    # ------------------------------------------------------------------
    # 3) Compute SHAP values
    # ------------------------------------------------------------------
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sub)

    # Handle binary/multiclass outputs that may return a list
    if isinstance(shap_values, list):
        if class_index is None:
            # common convention: for binary classification, use positive class
            class_index = 1 if len(shap_values) > 1 else 0
        shap_values = shap_values[class_index]

    shap_values = np.asarray(shap_values)


    # ------------------------------------------------------------------
    # 4) Rank features by mean absolute SHAP value
    # ------------------------------------------------------------------
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance = pd.Series(mean_abs_shap, index=X_sub.columns).sort_values(ascending=False)

    top_features = importance.head(max_display).index.tolist()
    top_numeric_features = [f for f in top_features if f in numeric_features]

    # ------------------------------------------------------------------
    # 5) SHAP Importance 
    # ------------------------------------------------------------------
    summary_cmap = plt.get_cmap("plasma")
    plt.figure()
    shap.summary_plot(
        shap_values,
        X_sub,
        plot_type="dot",      # beeswarm-style summary
        max_display=max_display,
        cmap=summary_cmap,
        show=False,
    )
    plt.tight_layout()
    fig8_path = os.path.join(output_dir, "SHAP_summary_top8.png")
    plt.savefig(fig8_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

    # ------------------------------------------------------------------
    # 6) SHAP dependence plots for top numeric features
    # ------------------------------------------------------------------
    top_numeric_features = top_numeric_features[:6]

    dependence_paths = []

    def choose_interaction_feature(feat_name):
        if isinstance(interaction_method, dict):
            return interaction_method.get(feat_name, "auto")

        if interaction_method == "auto":
            return "auto"

        if interaction_method == "strongest":
            try:
                # approximate_interactions returns feature indices ordered by likely interaction strength
                inds = shap.utils.approximate_interactions(
                    feat_name, shap_values, X_sub
                )
                if len(inds) > 0:
                    return X_sub.columns[inds[0]]
            except Exception:
                pass
            return "auto"

        return "auto"

    # ------------------------------------------------------------------
    # 7) Combined SHAP dependence grid
    # ------------------------------------------------------------------
    # Re-render into one multi-panel figure if we have up to 6 numeric features.
    if len(top_numeric_features) > 0:
        n_panels = len(top_numeric_features)
        ncols = 2
        nrows = int(np.ceil(n_panels / ncols))

        fig = plt.figure(figsize=(12, 3.8 * nrows))

        for i, feat in enumerate(top_numeric_features, start=1):
            plt.subplot(nrows, ncols, i)
            interaction_feat = choose_interaction_feature(feat)

            dep_cmap = plt.get_cmap("plasma")

            shap.dependence_plot(
                feat,
                shap_values,
                X_sub,
                interaction_index=interaction_feat,
                ax=plt.gca(),
                show=False,
                cmap=dep_cmap,
                x_jitter=0
            )

            ax = plt.gca()

            if feat == "original_upb":
                ax.ticklabel_format(axis="x", style="scientific", scilimits=(5, 5), useMathText=True)

        plt.tight_layout()
        fig9_path = os.path.join(output_dir, "SHAP_dependence_grid.png")
        plt.savefig(fig9_path, dpi=200, bbox_inches="tight")
        if show:
            plt.show()
        plt.close()
    else:
        fig9_path = None