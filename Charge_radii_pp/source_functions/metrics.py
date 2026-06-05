import numpy as np
import pandas as pd

def rndm_m_random_calculator(filtered_model_predictions, samples, Vt_hat):
    """
    Generates posterior predictive samples and credible intervals.

    Args:
        filtered_model_predictions (numpy.ndarray): Model predictions.
        samples (numpy.ndarray): Gibbs samples `[beta, sigma]`.
        Vt_hat (numpy.ndarray): Normalized right singular vectors.

    Returns:
        tuple[numpy.ndarray, list[numpy.ndarray]]:
            - `rndm_m` (numpy.ndarray): Posterior predictive samples.
            - `[lower, median, upper]` (list[numpy.ndarray]): Credible interval arrays.
    """
    np.random.seed(142858)
    rng = np.random.default_rng()

    theta_rand_selected = rng.choice(samples, 50000, replace=False)

    # Extract betas and noise std deviations
    betas = theta_rand_selected[:, :-1]  # shape: (10000, num_models - 1)
    noise_stds = theta_rand_selected[:, -1]  # shape: (10000,)

    # Compute model weights: shape (10000, num_models)
    default_weights = np.full(Vt_hat.shape[1], 1 / Vt_hat.shape[1])
    model_weights_random = (
        betas @ Vt_hat + default_weights
    )  # broadcasting default_weights

    # Generate noiseless predictions: shape (10000, num_data_points)
    yvals_rand_radius = (
        model_weights_random @ filtered_model_predictions.T
    )  # dot product

    # Add Gaussian noise with std = noise_stds (assume diagonal covariance)
    # We'll use broadcasting: noise_stds[:, None] * standard normal noise
    noise = rng.standard_normal(yvals_rand_radius.shape) * noise_stds[:, None]
    rndm = yvals_rand_radius + noise

    # Compute credible intervals
    lower_radius = np.percentile(rndm, 2.5, axis=0)
    median_radius = np.percentile(rndm, 50, axis=0)
    upper_radius = np.percentile(rndm, 97.5, axis=0)

    return rndm, [lower_radius, median_radius, upper_radius]

def coverage(percentiles, rndm_m, models_output, truth_column):
    """
    Calculates coverage percentages for credible intervals.

    Args:
        percentiles (list[int]): Percentiles to evaluate (e.g., `[5, 10, ..., 95]`).
        rndm_m (numpy.ndarray): Posterior samples of predictions.
        models_output (pandas.DataFrame): DataFrame containing true values.
        truth_column (str): Name of column with true values.

    Returns:
        list[float]: Coverage percentages for each percentile.
    """
    #  How often the model’s credible intervals actually contain the true value
    data_total = len(rndm_m.T)  # Number of data points
    M_evals = len(rndm_m)  # Number of samples
    data_true = models_output[truth_column].tolist()

    coverage_results = []

    for p in percentiles:
        count_covered = 0
        for i in range(data_total):
            # Sort model evaluations for the i-th data point
            sorted_evals = np.sort(rndm_m.T[i])
            # Find indices for lower and upper bounds of the credible interval
            lower_idx = int((0.5 - p / 200) * M_evals)
            upper_idx = int((0.5 + p / 200) * M_evals) - 1
            # Check if the true value y[i] is within this interval
            if sorted_evals[lower_idx] <= data_true[i] <= sorted_evals[upper_idx]:
                count_covered += 1
        coverage_results.append(count_covered / data_total * 100)

    return coverage_results

# This function calculates the RMSE of the any model with respect to the truth column in the models output dataframe.
def rmse(pred, truth):
    pred = np.asarray(pred)
    truth = np.asarray(truth)
    return np.sqrt(np.mean((pred - truth) ** 2))

# This function calculates the RMSE of the supermodel and the individual models for the train, validation, test, and full datasets.
def model_rmse_by_split(models_selected, train_df, val_df, test_df, full_df, truth_col='truth'):
    results = {}

    for model in models_selected:
        results[model] = {
            'train': rmse(train_df[model], train_df[truth_col]),
            'validation': rmse(val_df[model], val_df[truth_col]),
            'test': rmse(test_df[model], test_df[truth_col]),
            'full': rmse(full_df[model], full_df[truth_col]),
        }

    return pd.DataFrame(results).T

def fit_pc_least_squares_weights(train_df, model_cols, Vt, n_components, truth_col='truth'):
    """
    Fit least-squares BMC/supermodel weights using the first n_components PCs.

    n_components = 0 corresponds to the uniform-average model.
    """
    X_train = train_df[model_cols].to_numpy()
    y_train = train_df[truth_col].to_numpy()

    n_models = len(model_cols)
    default_weights = np.full(n_models, 1.0 / n_models)

    # p = 0: no PCs kept, just uniform model average
    if n_components == 0:
        return default_weights

    Vt_hat = Vt[:n_components, :]  # shape: (p, n_models)

    # mean model prediction for each isotope
    mean_train = X_train @ default_weights

    # centered model-output matrix
    X_centered = X_train - mean_train[:, None]

    # PC features
    X_pc = X_centered @ Vt_hat.T  # shape: (n_train, p)

    # target relative to mean prediction
    y_centered = y_train - mean_train

    # least-squares fit in PC space
    beta, *_ = np.linalg.lstsq(X_pc, y_centered, rcond=None)

    # translate PC coefficients back to model weights
    weights = default_weights + beta @ Vt_hat

    return weights


def evaluate_weights_rmse(df, model_cols, weights, truth_col='truth'):
    """
    Evaluate RMSE of fixed model weights on a dataframe.
    """
    X = df[model_cols].to_numpy()
    truth = df[truth_col].to_numpy()

    pred = X @ weights
    return rmse(pred, truth)


def pc_rmse_curve(
    train_df,
    val_df,
    test_df,
    full_df,
    model_cols,
    Vt,
    truth_col='truth',
    max_components=None
):
    """
    Compute RMSE as a function of number of PCs kept.

    Returns dataframe with columns:
      n_components, full, train, validation, test
    """
    n_models = len(model_cols)

    if max_components is None:
        max_components = n_models

    results = []

    for p in range(0, max_components + 1):
        weights = fit_pc_least_squares_weights(
            train_df=train_df,
            model_cols=model_cols,
            Vt=Vt,
            n_components=p,
            truth_col=truth_col
        )

        row = {
            'n_components': p,
            'full': evaluate_weights_rmse(full_df, model_cols, weights, truth_col),
            'train': evaluate_weights_rmse(train_df, model_cols, weights, truth_col),
            'validation': evaluate_weights_rmse(val_df, model_cols, weights, truth_col),
            'test': evaluate_weights_rmse(test_df, model_cols, weights, truth_col),
        }

        results.append(row)

    return pd.DataFrame(results)
