import time
from geecs_api.devices import HTU
from geecs_api.experiment import Experiment


class HtuExp(Experiment):
    # Singleton
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'instance'):
            cls.instance = super(HtuExp, cls).__new__(cls)
            cls.instance.__initialized = False
        return cls.instance

    def __init__(self):
        # Singleton
        if self.__initialized:
            return
        self.__initialized = True

        super().__init__('Undulator')

        # Devices
        self.jet = HTU.GasJet(self.exp_devs)
        self.jet.stage.subscribe_var_values()
        self.jet.pressure.subscribe_var_values()

        self.devs = {
            'jet': self.jet
        }


if __name__ == '__main__':
    htu = HtuExp()

    # htu.jet.stage.set_position('X', 7.6)
    # htu.jet.pressure.set_pressure(0.)
    # htu.jet.pressure.set_trigger(False)
    htu.jet.pressure.get_trigger()

    time.sleep(1.0)
    print(f'Stage state:\n\t{htu.jet.stage.state}')
    print(f'Pressure state:\n\t{htu.jet.pressure.state}')

    print(f'Stage setpoints:\n\t{htu.jet.stage.setpoints}')
    print(f'Pressure setpoints:\n\t{htu.jet.pressure.setpoints}')

    htu.cleanup()
