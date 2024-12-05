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
        'stage-2bl-compressor:position.axis 1': [-4.5,6],
        'stage-hpd-tel:position': [180,600],
        'stage-1bl-compression:position.axis 1': [-5,6],
        'mcd-1bl-esp302m4m5:position.axis 1': [-1.2,-1.15],
        'mcd-1bl-esp302m4m5:position.axis 2': [-0.93,-0.89],
        'mcd-1bl-esp302m4m5:position.axis 3': [-0.8,0.84],
        'prc-pl1-highpressure:pressure.device 1': [50,150],
        'prc-pl1-highpressure:pressure.device 2': [50,150],
        'mcd-1bl-esp302m2m4:position.axis 1': [0.98,1.02],
        'mcd-1bl-esp302m2m4:position.axis 2': [-1.4,-1.18],
        'mcd-1bl-esp302m2m4:position.axis 3': [-0.78,-0.74],
        'hex-pl1-2:ypos': [-0.8,1],
        'hex-pl1-2:zpos': [-1.6,-1.4],

    }
    observables = ['CAM-PL1-1-SideView:meancounts',
                   'CAM-HPD-CCD:maxcounts']
    some_parameter: str = 'test'

