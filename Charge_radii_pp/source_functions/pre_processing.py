import numpy as np
import pandas as pd

""". 
This file contains functions that are used to pre-process the data before feeding 
into the Bayesian model. The functions include:
- filtered_NZ_extraction: This function takes only even-even nuclei with N,Z >= 8
- unified_NZ_extraction: This function extracts nuclei that exist in every models
- selected_models_data_sets_extraction: This function creates dataframe that 
  contains all the model predictions of a property of interest
- models_output_extraction: This function rearrange the columns of the 
  selected_models_data_sets and separate into intrapolation and extrapolation part
- filtered_models_output_extraction: This function extract [all, train, val, test] 
slices filtered by Z and N ranges, then sorted.
- filtered_predictions_all_elements: This function is a lightweight orchestrator 
  that picks the Zs to process (defaults to all in models_output), for each Z, 
  calls the two extractors (which already handle N-range & sorting), 
  and returns a dict: Z -> {'all','train','val','test','stable'} DataFrames.
- filtered_models_output_stable_extraction: This function extract models output 
  dataframe for stable isotope for a specific element
- filtered_models_output_dict_extraction: This function helps to create the 
  filtered models output for different isotopes at the same time and store 
  them in a dictionary
- isotope_chain_forward_differences: This function computes forward differences 
  within each Z-chain only. For each fixed Z, it sorts by N, 
  computes (next - prev) differences, and stores the diff at the 
  "next" isotope label (N_next, Z). 
  If require_step=True, only keeps pairs with N_next - N_prev == step 
  (useful for even-even chains with step=2). 
"""

def filtered_NZ_extraction(filtered_NZ):
    #We want to create a function that takes only even-even nuclei with N,Z >= 8
    filtered_NZ_new = []
    for isotope in filtered_NZ:
        if ((isotope[0] >= 8) & (isotope[1] >=8)) & ((isotope[0]%2 == 0) & (isotope[1]%2 == 0)): # Sort the even-even isotope with proton and neutron number 8 or above
            filtered_NZ_new.append(isotope)
    filtered_NZ_new  = np.array(filtered_NZ_new)
    return filtered_NZ_new

def unified_NZ_extraction(models_data_sets, models_selected, filtered_NZ):
    #We want to create a function that extracts nuclei that exist in every models
    for model in models_selected:
        filtered_NZ_new = []
        for isotope in filtered_NZ:
            if ( (isotope[0] == models_data_sets[model]['N']) & (isotope[1] == models_data_sets[model]['Z']) ).any(): # Choose nuclei that are contained in each model
                filtered_NZ_new.append(isotope)
        filtered_NZ = np.array(filtered_NZ_new) #update our new list of isotope and repeat this for every model
    filtered_NZ_df = pd.DataFrame({'N' : filtered_NZ.T[0], 'Z' : filtered_NZ.T[1]})
    return filtered_NZ_new, filtered_NZ_df

def selected_models_data_sets_extraction(models_data_sets, models_selected, filtered_NZ_df, property):
    #We want to create dataframe that contains all the model predictions of a property of interest
    selected_models_data_sets = pd.DataFrame(filtered_NZ_df.copy()) #Initiate the dataframe by the proton and neutron number
    for model in models_selected:
        # Convert the model's dataset to a dataframe (if not already)
        model_df = pd.DataFrame(models_data_sets[model])

        # Merge with selected_models_data_sets to keep only matching isotopes
        merged_df = pd.merge(selected_models_data_sets, model_df, on=['N', 'Z'], how='inner')

        # Update selected_models_data_sets to include only the common isotopes
        selected_models_data_sets = merged_df[['N', 'Z'] + [col for col in selected_models_data_sets.columns if col not in ['N', 'Z']]]

        # Add the model's predictions for the property
        selected_models_data_sets[model] = merged_df[property]

    return selected_models_data_sets

def models_output_extraction(selected_models_data_sets, train_coordinates, validation_coordinates, test_coordinates):
    #Technically rearrange the columns of the selected_models_data_sets and separate into intrapolation and extrapolation part
    models_output = selected_models_data_sets.copy()
    models_output['A'] = models_output['N'] + models_output['Z'] # Put the mass number in our dataframe
    cols = list(models_output.columns)
    cols[2], cols[-1] = cols[-1], cols[2] # Rearranging columns 
    models_output = models_output[cols]

    models_output_train = models_output.iloc[train_coordinates] # Training regions
    models_output_validation = models_output.iloc[validation_coordinates] # Validation regions
    models_output_test = models_output.iloc[test_coordinates] # Test regions
    return [models_output, models_output_train, models_output_validation, models_output_test]


def clip_sort(df, N_min, N_max, Z_min, Z_max):
        mask_N = df['N'].between(N_min, N_max)
        out_N = df.loc[mask_N].copy()
        
        mask_Z = out_N['Z'].between(Z_min, Z_max)
        out_Z = out_N.loc[mask_Z].copy()

        return out_Z

def filtered_models_output_extraction(models_output, train_idx, val_idx, test_idx, \
                                      stable_coordinates_df, Z_range, N_range = None):
    """
    Extract [all, train, val, test] slices filtered by Z and N ranges, then sorted.

    Parameters
    ----------
    models_output : pd.DataFrame
        Must contain columns 'Z', 'N', each model column, and 'truth'.
    train_coordinates / validation_coordinates / test_coordinates : sequence[int]
        Positional indices for .iloc to define splits.
    Z_range : (int, int) or None
        Inclusive range of proton numbers to keep. None => no Z filter.
        Use (Z, Z) to get a single element.
    N_range : (int, int)
        Inclusive neutron-number range to keep.

    Returns
    -------
    list[pd.DataFrame] : [all_df, train_df, val_df, test_df]
    """

    if Z_range == None:
        raise ValueError('Must specify the range of Z')
    
    if N_range is None:
        N_range = (0, 300)

    Z_min, Z_max = Z_range[0], Z_range[1]
    N_min, N_max = N_range[0], N_range[1]

    models_output_stable = pd.merge(models_output, stable_coordinates_df, on = ['N', 'Z'], how = 'inner')


    all_df  = clip_sort(models_output, N_min, N_max, Z_min, Z_max)
    train_df = clip_sort(models_output.iloc[train_idx], N_min, N_max, Z_min, Z_max)
    val_df   = clip_sort(models_output.iloc[val_idx], N_min, N_max, Z_min, Z_max)
    test_df  = clip_sort(models_output.iloc[test_idx], N_min, N_max, Z_min, Z_max)
    stable_df = clip_sort(models_output_stable, N_min, N_max, Z_min, Z_max)

    return [all_df, train_df, val_df, test_df, stable_df]


def filtered_predictions_all_elements(models_output, train_idx, val_idx, test_idx, \
                                            stable_coordinates_df, Zs=None):
    """
    Lightweight orchestrator:
    - Picks the Zs to process (defaults to all in models_output).
    - For each Z, calls your two extractors (which already handle N-range & sorting).
    - Returns a dict: Z -> {'all','train','val','test','stable'} DataFrames.
    """
    if Zs is None:
        Zs = sorted(models_output['Z'].unique())

    predictions_dict = {}

    for Z in Zs:
        [all_df, train_df, val_df, test_df, stable_df]= filtered_models_output_extraction(
            models_output,
            train_idx,
            val_idx,
            test_idx,
            stable_coordinates_df,
            Z_range = (Z,Z),
            N_range = None
        )

        predictions_dict[Z] = {
            'all': all_df,
            'train': train_df,
            'val': val_df,
            'test': test_df,
            'stable': stable_df
        }
    return predictions_dict



def filtered_models_output_stable_extraction(Selected_element, models_output, stable_coordinates):
    # Extract models output dataframe for stable isotope for a specific element
    Z_range = (Selected_element, Selected_element) # Choose element we want to analyze
    N_range = (0,300)
    stable_isotopes = []
    for isotope in stable_coordinates:
        if isotope[1] == Selected_element: #Choose isotope that has the same proton number
            stable_isotopes.append(isotope)
    stable_isotopes = np.array(stable_isotopes)
    
    filtered_models_output_stable = models_output[(models_output['Z'] >= Z_range[0]) & (models_output['Z'] <= Z_range[1])\
                                                 & (models_output['N'] >= N_range[0]) & (models_output['N'] <= N_range[1]) & models_output['N'].isin(stable_isotopes.T[0])]
    return filtered_models_output_stable

def filtered_models_output_dict_extraction(Selected_elements, key_name, models_output, train_coordinates, validation_coordinates, test_coordinates, stable_coordinates):
    # This functions helps me to create the filtered models output for different isotopes at the same time and store them in a dictionary
    filtered_models_output_dict = {} # Initiate dictionary
    for j in Selected_elements:
        key = f"{key_name}" + f"{int(j)}" # name is the key-value pair in the dictionary
        # The line below store the list of filtered models output for a specific element in the dictionary
        filtered_models_output_dict[key] = filtered_models_output_extraction(models_output, train_coordinates, validation_coordinates, test_coordinates, stable_coordinates,[int(j), int(j)])
        # The line below add to the list we have above the stable isotopes into our list
        filtered_models_output_dict[key].append(filtered_models_output_stable_extraction(int(j), models_output, stable_coordinates))
    filtered_models_output_dict.keys() # This is just to check if I had the right name for the list
#     return filtered_models_output_dict

def isotope_chain_forward_differences(
    dataframe: pd.DataFrame,
    N_col: str = "N",
    Z_col: str = "Z",
    step: int = 2,                 # step=2 for even-even chains; use 1 if you want consecutive N
    require_step: bool = True,     # if True, only diff when N_next - N_prev == step
    keep_prev_labels: bool = True
) -> pd.DataFrame:
    """
    Forward differences within each Z-chain only.

    For each fixed Z:
      sort by N,
      compute (next - prev) differences,
      store the diff at the "next" isotope label (N_next, Z).

    If require_step=True, only keep pairs with N_next - N_prev == step
    (useful for even-even chains with step=2).
    """
    df = dataframe.copy()
    d = df.sort_values([Z_col, N_col], kind="mergesort").reset_index(drop=True)

    value_cols = [c for c in d.columns if c not in (N_col, Z_col)]

    if len(d) < 2:
        return d.iloc[0:0].copy()

    out_parts = []

    for Z, g in d.groupby(Z_col, sort=False):
        g = g.sort_values(N_col, kind="mergesort").reset_index(drop=True)
        if len(g) < 2:
            continue

        # compute forward diffs
        diffs = g[value_cols].iloc[1:].to_numpy() - g[value_cols].iloc[:-1].to_numpy()

        # labels stored at the "next" isotope
        out = g[[N_col, Z_col]].iloc[1:].copy()
        out[value_cols] = diffs

        if keep_prev_labels:
            out[f"{N_col}_prev"] = g[N_col].iloc[:-1].to_numpy()
            out[f"{Z_col}_prev"] = g[Z_col].iloc[:-1].to_numpy()

        if require_step:
            prevN = g[N_col].iloc[:-1].to_numpy()
            nextN = g[N_col].iloc[1:].to_numpy()
            mask = (nextN - prevN) == step
            out = out.loc[mask].copy()

        out_parts.append(out)

    if not out_parts:
        # return empty with expected columns
        cols = [N_col, Z_col] + value_cols
        if keep_prev_labels:
            cols += [f"{N_col}_prev", f"{Z_col}_prev"]
        return pd.DataFrame(columns=cols)

    return pd.concat(out_parts, ignore_index=True)