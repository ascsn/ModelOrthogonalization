import numpy as np
import pandas as pd

from pre_processing import filtered_models_output_extraction

"""
This code set up funcions that calculate the BMC predictions
"""


class BMCPredictor:
    """
    Reusable posterior predictive engine:
    - preselects posterior draws once
    - precomputes model weights once
    - per dataset: multiply + add noise + percentiles
    """
    def __init__(self, samples, Vt_hat, n_draws=50000, seed=142858):
        self.seed = int(seed)
        rng = np.random.default_rng(self.seed)

        if n_draws > samples.shape[0]:
            raise ValueError(f"n_draws={n_draws} > #samples={samples.shape[0]}")

        # choose posterior draws once
        idx = rng.choice(samples.shape[0], n_draws, replace=False)
        theta = samples[idx]

        # split into betas and sigma
        self.betas = theta[:, :-1]
        self.noise_stds = theta[:, -1]

        # sanity check shapes
        if self.betas.shape[1] != Vt_hat.shape[0]:
            raise ValueError(
                f"Shape mismatch: betas has {self.betas.shape[1]} columns "
                f"but Vt_hat has {Vt_hat.shape[0]} rows."
            )

        # precompute weights once
        n_models = Vt_hat.shape[1]
        default_weights = np.full(n_models, 1.0 / n_models)
        self.weights = self.betas @ Vt_hat + default_weights  # (n_draws, n_models)

    def credible_intervals(self, model_predictions, seed_offset=0, add_noise=True,
                           percentiles=(2.5, 50.0, 97.5)):
        """
        model_predictions: array (n_points, n_models) in the SAME model order as Vt_hat expects.
        returns: (lower, median, upper), each (n_points,)
        """
        X = np.asarray(model_predictions)
        if X.ndim != 2:
            raise ValueError("model_predictions must be 2D: (n_points, n_models)")
        if X.shape[1] != self.weights.shape[1]:
            raise ValueError(
                f"Model count mismatch: X has {X.shape[1]} models, "
                f"weights expect {self.weights.shape[1]}"
            )

        # noiseless predictive draws: (n_draws, n_points)
        y = self.weights @ X.T

        if add_noise:
            rng = np.random.default_rng(self.seed + int(seed_offset))
            y = y + rng.standard_normal(y.shape) * self.noise_stds[:, None]

        lo, med, hi = np.percentile(y, percentiles, axis=0)
        return lo, med, hi
    
""" 
This function gives you the models output that you have for your Z number. 
"""

def build_filtered_outputs_per_Z(models_output, train_idx, val_idx, test_idx,
                                 stable_coordinates_df, Z_values, N_range=(0, 300)):
    """
    Returns dict:
      per_Z[Z] = {'all':..., 'train':..., 'val':..., 'test':..., 'stable':...}
    """
    per_Z = {}
    for Z in Z_values:
        all_df, train_df, val_df, test_df, stable_df = filtered_models_output_extraction(
            models_output=models_output,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            stable_coordinates_df=stable_coordinates_df,
            Z_range=(Z, Z),
            N_range=N_range
        )
        per_Z[Z] = {
            'all': all_df,
            'train': train_df,
            'val': val_df,
            'test': test_df,
            'stable': stable_df
        }
    return per_Z

def build_bmc_intervals_per_Z(per_Z_data, predictor, model_cols):
    """
    Adds BMC intervals per Z:
      bmc_per_Z[Z] = {'lower':..., 'median':..., 'upper':..., 'interval_half_width':...}
    """
    bmc_per_Z = {}
    for Z, d in per_Z_data.items():
        all_df = d['all']

        # IMPORTANT: column order must match Vt_hat's model order
        X = all_df[model_cols].to_numpy()  # (n_points, n_models)

        lo, med, hi = predictor.credible_intervals(X, seed_offset=Z, add_noise=True)
        # The idea is to roughtly estimate the predictive std dev as (hi - lo)/2, 
        # which is half the width of the 95% credible interval.
        interval_half_width = (hi - lo)/2  # crude estimate of predictive std dev for this Z
        bmc_per_Z[Z] = {'lower': lo, 'median': med, 'upper': hi, 'interval_half_width': interval_half_width}
    return bmc_per_Z

def build_bmc_error_dataframe(per_Z_data, bmc_per_Z, truth_col='truth'):
    """
    Build a flat dataframe for all isotopes with BMC median error and absolute error.
    per_Z_data: output from build_filtered_outputs_per_Z
    bmc_per_Z: output from build_bmc_intervals_per_Z
    truth_col: name of the experimental truth column in your data
    """
    dfs = []
    for Z, d in per_Z_data.items():
        df = d['all'].copy()
        if Z not in bmc_per_Z:
            raise KeyError(f"Missing BMC results for Z={Z}")
        df['BMC_median'] = bmc_per_Z[Z]['median']
        df['error'] = df['BMC_median'] - df[truth_col]
        df['abs_error'] = np.abs(df['error'])
        # This interval_half_width is a crude estimate of the predictive uncertainty for this Z,
        # derived from the BMC credible interval width, which we will use to determine which isotopes 
        # are "problematic" (e.g. abs_error > 2*interval_half_width).
        df['interval_half_width'] = bmc_per_Z[Z]['interval_half_width']
        dfs.append(df[['Z', 'N', 'error', 'abs_error', 'interval_half_width']])
    
    return pd.concat(dfs, ignore_index=True)