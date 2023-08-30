from geecs_python_api.controls.devices.geecs_device import GeecsDevice
from geecs_python_api.controls.api_defs import VarAlias
from geecs_python_api.controls.experiment import Experiment


class HttExp(Experiment):
    # Singleton
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'instance'):
            cls.instance = super(HttExp, cls).__new__(cls)
            cls.instance.__initialized = False
        return cls.instance

    def __init__(self):
        # Singleton
        if self.__initialized:
            return
        self.__initialized = True

        super().__init__('Thomson')

        # Devices
        self.pressure = GeecsDevice('HTT-HighPressureDAQ')
        self.pressure.var_spans = {VarAlias('HPDAQao'): (0., 7.)}
        self.pressure.build_var_dicts()
        self.pressure.var_pressure = self.pressure.var_names_by_index.get(0)[0]
        self.pressure.subscribe_var_values()

        self.devs = {
            'pressure': self.pressure
        }


if __name__ == '__main__':
    Experiment.initialize('Thomson')

    # htt = HtuExp()
    # htt.initialize(htt.exp_name)
    # htt.close()

    print('done')
