#! /usr/bin/python

#----

"""Import Imagine Optic's Python interface
"""
import os, sys
sys.path.append('./../..')
import wavekit_py as wkpy

"""Relative path from executable to input datafile
"""
config_file_path = './../DATAS/config_file_haso.dat'

"""CameraSet constructor from a haso configuration file
"""
camera_set = wkpy.CameraSet(config_file_path = config_file_path)
"""Get the list of the specific features
"""
feature_list_size = camera_set.get_specific_features_list_size()
for i in range(feature_list_size):
    feature_name = camera_set.get_specific_feature_name(i)
    print('specific feature name : '+feature_name)

"""Get parameter list
"""
parameter_list_size = camera_set.get_parameters_list_size()
for i in range(parameter_list_size):
    parameter_name = camera_set.get_parameter_name(i)
    parameter_type = camera_set.get_parameter_type(parameter_name)
    parameter_has_limits = camera_set.get_parameter_option(parameter_name)[1]
    if(parameter_type == wkpy.E_TYPES.INT and parameter_has_limits):
        limit_min, limit_max = camera_set.get_parameter_limits(parameter_name)
        print('Parameter name : '+parameter_name+' - min : '+str(limit_min)+' - max : '+str(limit_max))
