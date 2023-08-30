from typing import Optional
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

    def __init__(self, get_info: bool = True):
        # Singleton
        if self.__initialized:
            return
        self.__initialized = True

        super().__init__('Thomson', get_info)

        self.pressure: Optional[GeecsDevice] = None

    def connect(self, pressure: bool = True):
        # Devices
        if isinstance(self.pressure, GeecsDevice):
            self.pressure.close()

        if pressure:
            self.pressure = GeecsDevice('HTT-HighPressureDAQ')
            self.pressure.var_spans = {VarAlias('HPDAQao'): (0., 7.)}
            self.pressure.build_var_dicts()
            self.pressure.var_pressure = self.pressure.var_names_by_index.get(0)[0]
            self.pressure.subscribe_var_values()
        else:
            self.pressure = None

        # Dictionary
        self.devs = {
            'pressure': self.pressure
        }  # handy to manipulate devices by batch


if __name__ == '__main__':
    # Experiment.get_info('Thomson')

    htt = HttExp(get_info=True)
    htt.connect()
    htt.close()

    print('done')
