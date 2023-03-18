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

        self.devs = {
            'jet': self.jet
        }


if __name__ == '__main__':
    htu = HtuExp()
    htu.jet.stage.subscribe_var_values()

    time.sleep(1.0)
    print(f'Jet state:\n\t{htu.devs["jet"].stage.gets}')
    print(f'Jet config:\n\t{htu.devs["jet"].stage.sets}')

    htu.cleanup()
