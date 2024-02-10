import numpy as np
import pandas as pd


def h5_data_loader(models, observable, residuals = False):
    """Loads selected_data.h5, and creates a DataFrame with theoretical values of observables + exp values
    
    Inputs:
        models: A list of models whose theoretical values to include
        observable: Observable name ("BE" for binding energies, "TwoPSE" for two proton sep. eng)
        residuals: Boolean, return residuals or no
    Output: Pandas.DataFrame with theoretical vales, experimental values, and (Z,N) pairs
    """
    # Load ZN + BMEX data
    ZN = pd.read_csv('.\data\ZN.dat', delim_whitespace=True)
    # model_data = pd.read_hdf(".\data\selected_data.h5", key = 'AME2020', mode = 'r')
    model_data = pd.read_hdf("./data/selected_data.h5", key = 'AME2020', mode = 'r')
    
    # Which ZN from ZN.dat are in selected_Data.h5
    data_dict = {}
    index = model_data[["Z","N"]].isin(ZN).sum(axis = 1) == 2
    data_dict["exp"] = model_data[index][observable].values
    data_pd = pd.DataFrame(data_dict)

    # Populating data_pd
    for model in models:
        # model_data = pd.read_hdf(".\data\selected_data.h5", key = model, mode = 'r')
        model_data = pd.read_hdf("./data/selected_data.h5", key = model, mode = 'r')

        index_list = []
        ZN[model] = False
        data_pd[model] = 0
        for index, row in ZN.iterrows():
            N = row['N']
            Z = row['Z']

            if((model_data['Z'] == Z) & (model_data['N'] == N)).any():
                index_list = index_list + [index]
                ZN.loc[index,model] = True
                data_pd.loc[index,model] = model_data[(model_data['Z'] == Z) & (model_data['N'] == N)][observable].values

    data_pd = data_pd.loc[ZN.all(axis = 1),:]
    data_pd["Z"] = ZN.loc[ZN.all(axis = 1),"Z"]
    data_pd["N"] = ZN.loc[ZN.all(axis = 1),"N"]
    
    #Residuals y_exp - y_th
    if residuals:
        data_pd[models] = - data_pd[models].subtract(data_pd["exp"], axis = 0)
    return data_pd