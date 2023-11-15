
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

from geecs_functions import GeecsXoptInterface
geecs_interface = GeecsXoptInterface()


class MyBackend:
    def __init__(self):
        self.devices = {}  # A dict that will store the initialized device controls
        self.config_params = {
            'normalize': True,  # default value
            'opt_method': 'bayes'  # default value
        }
        self.yaml_config={}
        self.X = None
        
        #some parameters for setting up the simulation case
        self.optPosition = np.array([18.45, 0.6])
        self.numParts = 200000

        self.startDist = np.transpose([
            np.random.normal(self.optPosition[0], 0.4, self.numParts),
            np.random.normal(self.optPosition[1], 0.4, self.numParts)
        ])

    def set_config_params(self, params):
        self.config_params.update(params)
    
    def configure_yaml(self,backend_vocs):
        print("backend vocs")
        print(backend_vocs)
        
        
        #define the xopt configuration
        YAML = """
        xopt:
            dump_file: dump.yaml
        generator:
            name:
        evaluator:
            # function: geecs_functions.geecs_measurement
            function: __main__.geecs_measurement
            # function: self.geecs_interface.geecs_measurement
            

        vocs:
            variables:
                {}
            objectives: {f: "MAXIMIZE"}

        """

        self.yaml_config = yaml.safe_load(YAML)
        
        self.yaml_config['evaluator']['function_kwargs'] = {'normalize': self.config_params['normalize']}
        # self.yaml_config['evaluator']['function_kwargs'] = {'devices': self.devices}
        
        
        
        for tag in backend_vocs.keys():
            print(tag)
            self.yaml_config['vocs']['variables'][tag]=backend_vocs[tag]
            print(self.yaml_config['vocs']['variables'][tag])
    
        if self.config_params['normalize']:
            for tag in backend_vocs.keys():
                self.yaml_config['vocs']['variables'][tag]=[-1.0,1.0]
            keys = self.yaml_config['vocs']['variables'].keys()
            
        # New dictionary for initial points
        init_point = {}

        # Calculate the average of the bounds and assign it to the corresponding key
        for key, bounds in backend_vocs.items():
            average = sum(bounds) / len(bounds)
            init_point[key] = average
            
        if self.config_params['opt_method'] == 'bayes':
            self.yaml_config['generator']['name'] = 'upper_confidence_bound'
            self.yaml_config['generator']['n_initial'] = 2
            self.yaml_config['generator']['acq'] = {'beta':0.1}
            self.yaml_config['xopt']['dump_file'] = 'bayes.yaml'
            
            
        elif self.config_params['opt_method'] == 'nelder':
            self.yaml_config['generator']['name'] = 'neldermead'
            self.yaml_config['generator']['adaptive'] = True
            self.yaml_config['generator']['xatol'] = 0.1
            self.yaml_config['generator']['fatol'] = 0.05
            self.yaml_config['generator']['initial_point'] = init_point
            self.yaml_config['xopt']['dump_file'] = 'nelder.yaml'
            
        if self.yaml_config['generator']['name']=='neldermead':
            if self.config_params['normalize']:
                initial_point = self.yaml_config['generator']['initial_point']

                normalized_initial_point = {}
                for key in keys:
                    normalized_initial_point[key] = geecs_interface.normalize_controls(key, initial_point[key])

                self.yaml_config['generator']['initial_point'] = normalized_initial_point
                
        print(self.yaml_config)
        return self.yaml_config
        
    def initialize_xopt(self):
        print(self.X)
        self.X = Xopt(config=self.yaml_config)
        print(self.X)
            
        if self.config_params['opt_method'] == 'bayes':
            # print initial number of points to be generated
            print(self.X.generator.options.n_initial)

            # call X.step() to generate + evaluate initial points
            self.X.step()

            # inspect the gathered data
            self.X.data
            
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

        
        

    

    
    
