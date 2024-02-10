import numpy as np
import pandas as pd

ZN = pd.read_csv('NZFull2020.txt', delim_whitespace=True)

#delete selected_data.h5 if it exists

import os

if os.path.exists('selected_data.h5'):
    os.remove('selected_data.h5')

# download file from internet if it doesn't exist locally

if not os.path.isfile('bmexdb-10-07-2023.h5'):

    import urllib.request

    url = 'https://github.com/massexplorer/bmex-masses/raw/main/data/7-10-23.h5'
    urllib.request.urlretrieve(url, 'bmexdb-10-07-2023.h5')

db = 'bmexdb-10-07-2023.h5'

models = ['AME2020', 'ME2', 'MEdelta', 'PC1', 'NL3S', 'SKMS', 'SKP', 'SLY4', 'SV', 'UNEDF0', \
        'UNEDF1', 'UNEDF2', 'FRDM12', 'HFB24', 'BCPM', 'D1M']

for model in models:

    print("Processing model: ", model)

    bmexdb = pd.read_hdf(db, model)

    selected_data = bmexdb.merge(ZN, on=['Z','N'])

    # the following adds N-2 data points

    # for index, row in ZN.iterrows():

    #     # N = row['N']-2
    #     N = row['N']
    #     Z = row['Z']

    #     if(not ((selected_data['Z'] == Z) & (selected_data['N'] == N)).any()):
    #         selected_data = selected_data._append(bmexdb[(bmexdb['Z'] == Z) & (bmexdb['N'] == N)])


    selected_data = selected_data.reset_index(drop=True)

    # export selected data to csv

    # selected_data.to_csv('selected_data.csv', index=False)

    # export selected data to h5

    selected_data.to_hdf('selected_data.h5', key=model, mode='a')
