# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 21:06:59 2022

@author: Kareem
"""

import re
import pickle
import os
from scipy.signal import decimate
import numpy as np

def save_to_file(file_name:str, dictionary):
    with open(f'{file_name}.pkl', 'wb') as f:
        pickle.dump(dictionary, f)

def load_file(fine_name:str):
    with open(f'{fine_name}.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)
    return loaded_dict

save_name = 'all_records'

all_records_dict = load_file(save_name)


for jdx, event in enumerate(all_records_dict):
    records = event['record']
    print(f'event no. {jdx}')
    for idx, record in enumerate(records['records']):
        ds_factor = int(0.02 // records['rates'][idx])
        if ds_factor > 1:
            new_record = decimate(record, ds_factor).tolist()
            records['records'][idx] = new_record
            records['points'][idx] = len(new_record)
            records['rates'][idx] = 0.02
    all_records_dict[jdx]['record'] = records
    
save_name = 'all_records_decimated'
save_to_file(save_name, all_records_dict)


new_records = load_file('all_records_decimated')
old_records = load_file('all_records')