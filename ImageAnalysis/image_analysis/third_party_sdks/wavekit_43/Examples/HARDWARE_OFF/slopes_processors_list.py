#! /usr/bin/python

#----

"""Import Imagine Optic's Python interface
"""
import os, sys
sys.path.append('./../..')
import wavekit_py as wkpy

NB_SLOPES_TO_PROCESS = 5

hasoslopes_file_path = './../DATAS/data_phase_computation.has'

"""Create hasodatas array from files
"""
hasodata_array = []
for i in range(NB_SLOPES_TO_PROCESS):
    hasodata_file_path = './../DATAS/data_phase_computation_'+str(i)+'.has'
    hasodata_array.append(wkpy.HasoData(has_file_path = hasodata_file_path))

"""SlopesPostProcessorList default constructor
"""
slopespostprocessorlist = wkpy.SlopesPostProcessorList()
"""Create hasoslopes used for the processing
"""
hasoslopes = wkpy.HasoSlopes(has_file_path = hasoslopes_file_path)

"""Insert the needed processors to the list
"""
slopespostprocessorlist.insert_adder(
    0,
    hasoslopes,
    'adder'
    )
slopespostprocessorlist.insert_scaler(
    1,
    1.75,
    'scaler'
    )
slopespostprocessorlist.insert_substractor(
    2,
    hasoslopes,
    'substractor'
    )

"""Apply this processors list to a batch of files
"""
for i in range(NB_SLOPES_TO_PROCESS):
    hasodata_array[i].apply_slopes_post_processor_list(slopespostprocessorlist)
    hasodata_file_path = './../OUT_FILES/data_phase_computation_'+str(i)+'_processed.has'
    hasodata_array[i].save(
        hasodata_file_path,
        'slopes processors list exemple',
        ''
        )
    print('Haso slopes '+str(i)+' processed')
    print('Haso slopes saved to file '+str(hasodata_file_path))
