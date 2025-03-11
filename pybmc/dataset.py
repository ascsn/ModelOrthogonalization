import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Dataset:
    """
    The main idea of this class is to handle data construction and organization of the set of models
    that we choose to analyze. What should this class contain:
    + Methods that help us extract model data from the data source.
    + Methods that select valid domains (e.g., even-even nuclei).
    + Methods that make our data self-consistent among all models.
    """

    models_selected_default = ['ME2', 'MEdelta', 'PC1', 'NL3S', 'SKMS', 'SKP', 'SLY4', 'SV', 'UNEDF0', 'UNEDF1']
    keys_default = ['N', 'Z', 'BE', 'ChRad']

    def __init__(self, data_source=None, models=None, keys = None):
        """
        Initialize the dataset with models and extract relevant data.

        :param data_source: Path to the HDF5 file containing the data.
        :param models: List of model names to extract data for.
        :param keys: Variable arguments specifying which data attributes to extract.
        """
        self.data_source = data_source if data_source else "./data/selected_data.h5"
        self.models = models if models else Dataset.models_selected_default
        self.keys = keys if keys else Dataset.keys_default
        self.raw_data_sets = self.extract_data(self.data_source, self.models, self.keys)
        print('The initial instance of this class will contain attributes:\
               data_source, models, keys, raw_data_sets')

    def domain_synchronization(self, models_data_sets = None, models_selected = None ):
        '''
        In terms of class structure, let's first synchronized our data first without appplying any
        condition on the isotope (like N > 8 or N even etc). 
        '''
        # This line of code initialize the isotopes that we are using
        initial_isotope = self.isotope_initialization()
        # This function are not yet made optimal as much as possible as the functions that I am writing still 
        # depends on 
        synchronized_domain = self.unified_NZ_extraction(initial_isotope, models_data_sets, models_selected)
        return np.array(synchronized_domain)
        
    def extract_data(self, data_source=None, models=None, keys=None):
        """
        Extracts data from the given HDF5 file and organizes it into a dictionary.
        """
        # Use instance attributes if arguments are not provided
        data_source = data_source if data_source is not None else self.data_source
        models = models if models is not None else self.models
        keys = keys if keys is not None else self.keys

        data_dict = {}
        for model in models:
            data_values = pd.read_hdf(data_source, key=model)
            data_dict[model] = {key: data_values[key] for key in keys}
        
        return data_dict

    def isotope_initialization(self):
        """
        Extracts and returns a numpy array of values of nuclei's properties (like 'N', 'Z', etc.) 
        for the first model. This is used to initialize the list that will be looped over 
        to extract the final data for calculations.

        :param properties: Variable number of property names to extract.
        :return: numpy array of properties for the first model.
        """
        # Check if all provided properties exist in the data for the first model
        first_model = self.models[0]
        # if first_model not in self.raw_data_sets:
        #     raise ValueError(f"Model '{first_model}' not found in the extracted data.")

        # for prop in properties:
        #     if prop not in self.raw_data_sets[first_model]:
        #         raise ValueError(f"Required key '{prop}' not found in the data.")

        # Extract the properties dynamically and create a numpy array
        filtered_data_initial = np.array(
            [self.raw_data_sets[first_model]['N'].tolist(), self.raw_data_sets[first_model]['Z'].tolist()]
        ).T
        
        return filtered_data_initial

    
    def unified_NZ_extraction(self, filtered_NZ, models_data_sets = None, models_selected = None):

        models_data_sets = models_data_sets if models_data_sets is not None else self.raw_data_sets
        models_selected = models_selected if models_selected is not None else self.models
        
        #We want to create a function that extracts nuclei that exist in every models
        for model in models_selected:
            filtered_NZ_new = []
            for isotope in filtered_NZ:
                if ( (isotope[0] == models_data_sets[model]['N']) & (isotope[1] == models_data_sets[model]['Z']) ).any(): # Choose nuclei that are contained in each model
                    filtered_NZ_new.append(isotope)
            filtered_NZ = np.array(filtered_NZ_new) #update our new list of isotope and repeat this for every model
        # filtered_NZ_df = pd.DataFrame({'N' : filtered_NZ.T[0], 'Z' : filtered_NZ.T[1]})
        return filtered_NZ_new
    
    def selected_models_data_sets_extraction(self, filtered_NZ_df, models_data_sets = None, models_selected = None, property = None):
        models_data_sets = models_data_sets if models_data_sets is not None else self.raw_data_sets
        models_selected = models_selected if models_selected is not None else self.models

        
        #We want to create dataframe that contains all the model predictions of a property of interest
        selected_models_data_sets = pd.DataFrame(filtered_NZ_df) #Initiate the dataframe by the proton and neutron number
        for model in models_selected:
            merged_df = pd.merge(filtered_NZ_df, pd.DataFrame(models_data_sets[model]), on = ['N', 'Z'], how  = 'inner') # Merge the isotope with their corresponding predictions
            selected_models_data_sets[model] = merged_df[property] # Choose the properties we want to perfrom BMM from
        return selected_models_data_sets


load_data = Dataset(None, None, ['N', 'Z', 'BE', 'ChRad']) # We are using default arguments here to extract data
print(list(load_data.__dict__.keys()))
filtered_NZ = load_data.domain_synchronization()


