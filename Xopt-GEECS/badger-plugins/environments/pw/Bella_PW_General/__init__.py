try:  # The top import works in Badger with symlinks, the bottom import lets PyCharm know what is being imported
    from ..base_geecs_env import Environment as geecs_Environment
except ImportError:
    from ...geecs_general.base_geecs_env import Environment as geecs_Environment

class Environment(geecs_Environment):
    name = 'Bella_PW_General'
    variables = {
        'HEX-PL1-1:ypos': [-8,2],
        'HEX-PL1-1:wangle': [-2,2],
        'HEX-PL1-1:zpos': [-5,0],
        'HEX-PL1-1:vangle': [0,1],
        'STAGE-2BL-Compressor:position.axis 1': [-4.5,6],
        'STAGE-HPD-Tel:position': [180,600],
        'STAGE-1BL-Compression:position.axis 1': [-5,6],
        'MCD-1BL-ESP302M4M5:position.axis 1': [-1.2,-1.15],
        'MCD-1BL-ESP302M4M5:position.axis 2': [-0.93,-0.89],
        'MCD-1BL-ESP302M4M5:position.axis 3': [-0.8,0.84],
        'PRC-PL1-HighPressure:pressure.device 1': [50,150],
        'PRC-PL1-HighPressure:pressure.device 2': [50,150],
        'MCD-1BL-ESP302M2M4:position.axis 1': [0.98,1.02],
        'MCD-1BL-ESP302M2M4:position.axis 2': [-1.4,-1.18],
        'MCD-1BL-ESP302M2M4:position.axis 3': [-0.78,-0.74],
        'HEX-PL1-2:ypos': [-0.5,1.1],
        'HEX-PL1-2:zpos': [-1.6,-0.2]
    }
    observables = ['CAM-PL1-1-SideView:meancounts',
                   'CAM-HPD-CCD:maxcounts','CAM-HPD-M3Near:MeanCounts']
    some_parameter: str = 'test'

