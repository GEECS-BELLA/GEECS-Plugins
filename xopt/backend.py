
import os
import yaml
import time
import numpy as np
import pandas as pd
import shelve
import sys

from xopt.evaluator import Evaluator
from xopt.generators.bayesian import ExpectedImprovementGenerator
from xopt import Xopt

import torch

import sys
import time

class MyBackend:
    def __init__(self):
        self.devices = {}  # A dict that will store the initialized device controls
        self.config_params = {
            'normalize': None,  # default value
            'opt_method': 'bayes_ucb',  # default value
            'opt_steps': None,
            'shots_per_step':None,
            'opt_target_device': None,
            'opt_target_var_name':None,
            'disable_sets': False
        }
        
        self.yaml_config= {}
        self.yaml_string: None
        self.X = None
        self.optimization_status=''
        
    def set_config_params(self, params):
        self.config_params.update(params)
        print(params)

    def load_xopt_base_config(self, filename: str = "bayes_ucb", directory: str = "config_files/base_xop_optimization_configs") -> dict:
        """
        Loads an xopt yaml config from a given directory based on the filename.

        Args:
        - filename (str): The name of the file without the .yaml extension. Default is "bayes_ucb".
        - directory (str): The directory where the yaml file is located. Default is "config_files/base_xop_optimization_configs".

        Returns:
        - dict: The parsed data from the yaml file.
        """

        with open(os.path.join(directory, f"{filename}.yaml"), 'r') as file:
            data = yaml.safe_load(file)
        return data
    
    
    def configure_yaml(self,backend_vocs):
        
        self.yaml_config = self.load_xopt_base_config(self.config_params['opt_method'])
                
        self.yaml_config['evaluator']['function_kwargs'] = {'normalize': self.config_params['normalize'],'shots_per_step': self.config_params['shots_per_step'],'disable_sets': self.config_params['disable_sets']}
                
        for tag in backend_vocs.keys():
            print(tag)
            self.yaml_config['vocs']['variables'][tag]=backend_vocs[tag]
            print(self.yaml_config['vocs']['variables'][tag])
    
        if self.config_params['normalize']:
            for tag in backend_vocs.keys():
                self.yaml_config['vocs']['variables'][tag]=[-1.0,1.0]
            keys = self.yaml_config['vocs']['variables'].keys()
      
                
        return self.yaml_config
        
    def initialize_xopt(self):
        self.yaml_string = yaml.dump(self.yaml_config)
        self.X = Xopt.from_yaml(self.yaml_string)
        print(self.X)
            
        if self.config_params['opt_method'] == 'bayes_ucb':
            # print initial number of points to be generated
            n_initial=3
            self.X.random_evaluate(n_initial)
            print('optimization initialized')
            
    def xopt_step(self):
        try:
            # Attempt to execute code that might raise an exception
            self.X.step()
            df = self.X.data
            #print(df)
            return df
        except Exception as e:
            # Handle exceptions
            print(f"An error occurred during optimization step: {e}")
            # Optionally, return a default value or re-raise the exception
            return None  # or raise e

        
        

    

    
    
