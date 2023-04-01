import numpy as np
import csv
from copy import deepcopy
import pickle as pkl

class CustomCSVLogger:

    def __init__(self,
                 file_name: str,
                 initiate=True):
    
        self.file_name = file_name
        self.initiate = initiate
    
    
    def write_to_log(self,
            loss: dict,
            ):
        keys = loss.keys()
        if self.initiate:
            write_mode = 'w'
        else:
            write_mode = 'a'
            
        with open(f'{self.file_name}.csv', write_mode, newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            if self.initiate:
                dict_writer.writeheader()
                self.initiate = False
                
            dict_writer.writerow(loss)

        
class ModelMonitor:
    
    def __init__(self,
                 file_name:str,
                 monitor_list:list = ['val_loss'],
                 mode='min',
                 disc='',
                 ):
                 
        modes = ['min', 'max']
        if mode not in modes:
            raise ValueError("Invalid sim type. Expected one of: %s" % modes)
            
        if mode == 'min':
            self.best_metric = 1E6
        else:
            self.best_metric = -1E6
            
        self.file_name = file_name
        self.mode = mode
        self.disc = disc
        self.monitor_list = monitor_list
        
    def update(self, model, metrics:dict):
    
        new_metric = sum(metrics[k] for k in self.monitor_list)
        
        if self.mode == 'min':
            if new_metric < self.best_metric:
                model.save_weights(self.file_name)
                print(f'New best weights found {self.disc}, weights are saved')
                self.best_metric = deepcopy(new_metric)

        elif self.mode == 'max':
            if new_metric > self.best_metric:
                model.save_weights(self.file_name)
                print(f'New best weights found {self.disc}, weights are saved')
                self.best_metric = deepcopy(new_metric)
                     
        
    
def save_pkl(filepath, obj):
    with open(f'{filepath}.pickle', 'wb') as handle:
        pkl.dump(obj, handle, protocol=pkl.HIGHEST_PROTOCOL)
    return 1


def load_pkl(filepath):
    with open(f'{filepath}.pickle', 'rb') as handle:
        dic = pkl.load(handle)
    return dic
    
def create_kl_weight_list(final, warmup_epochs, start=0, total_epochs=1000, mode='sigmoid'):
    modes = ['sigmoid', 'linear', 'constant']
    if mode not in modes:
        raise ValueError("Invalid sim type. Expected one of: %s" % modes)
        
    if mode == 'sigmoid':
        warmup = [start+final*logistic(x, x1=warmup_epochs) for x in range(warmup_epochs)]
    
    elif mode == 'linear':
        step = (final-start) / warmup_epochs
        warmup =[start + step*x for x in range(warmup_epochs)]

    else:
        warmup = [final for x in range(warmup_epochs)]
        
    constant_weights = [final for x in range(total_epochs-warmup_epochs)]
    warmup.extend(constant_weights)
    
    return warmup


def merge_dicts(dict1, dict2):
    res = {**dict1, **dict2}
    return res

# for KL weight warmup
def logistic(x, x1=200, epsilon=1E-4):
    y0 = epsilon
    y1 = 1 - epsilon
    b = np.log(y0 / (1 - y0))  # x0 = 0
    a = (np.log(y1 / (1 - y1)) - b) / x1
    y = 1 / (1 + np.exp(-(x * a + b)))
    return y
    
def save_to_csv(file_name: str, list_of_dicts: list):
    keys = list_of_dicts[0].keys()
    with open(f'{file_name}.csv', 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(list_of_dicts)