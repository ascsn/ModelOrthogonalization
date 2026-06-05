import pandas as pd

# This function loads the charge radii data from a CSV file 
# and renames the columns for consistency.
def load_charge_radii(path):
    df = pd.read_csv(path)
    df.rename(columns={'z': 'Z', 'n': 'N'}, inplace=True)
    return df

def load_charge_radii_fy(path, output_col):
    df = pd.read_csv(path)
    df.rename(columns = {'correctedRMS' : output_col}, inplace= True)
    df = df[['N', 'Z', output_col]]
    return df
