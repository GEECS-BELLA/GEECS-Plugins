# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 15:24:33 2024

@author: kjensen11, kjensen@lbl.gov
"""

# configuration distances in [m]
data = {('U_S4', 'UC_ALineEBeam3'): 0.94,
        ('U_S3', 'U_S4'): 0.94}

# composite distances
data[('U_S3', 'UC_ALineEBeam3')] = data[('U_S3', 'U_S4')] + data[('U_S4', 'UC_ALineEBeam3')]

def get_experimental_config_distance(start_loc, end_loc):

    distance = data.get((start_loc, end_loc), None)

    if distance is None:
        raise Exception("No experimental configuration distance found.")

    else:
        return data[(start_loc, end_loc)]
