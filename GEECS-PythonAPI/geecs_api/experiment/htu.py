import time
from geecs_api.devices.HTU import GasJet, Laser
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
        self.laser = Laser(self.exp_devs)
        self.jet = GasJet(self.exp_devs)

        self.devs = {
            'laser': self.laser,
            'jet': self.jet
        }


if __name__ == '__main__':
    htu = HtuExp()

    # htu.jet.stage.set_position('X', 7.6)
    # htu.jet.pressure.set_pressure(0.)
    # htu.jet.trigger.set_status(False)
    # htu.jet.blade.set_depth(-17.)

    htu.laser.compressor.get_separation()

    time.sleep(1.0)
    print(f'Compressor state:\n\t{htu.laser.compressor.state}')
    # print(f'Stage state:\n\t{htu.jet.stage.state}')
    # print(f'Pressure state:\n\t{htu.jet.pressure.state}')
    # print(f'Trigger state:\n\t{htu.jet.trigger.state}')
    # print(f'Blade state:\n\t{htu.jet.blade.state}')

    # print(f'Stage setpoints:\n\t{htu.jet.stage.setpoints}')

    htu.cleanup()
