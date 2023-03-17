import time
from geecs_api.devices import HTU
from geecs_api.interface import GeecsDatabase


class HtuExp:
    def __init__(self):
        self.exp_devs = GeecsDatabase.find_experiment_variables('Undulator')
        self.jet = HTU.GasJet(self.exp_devs)

        self.devs = {
            'jet': self.jet
        }

    def cleanup(self):
        for dev in self.devs.values():
            try:
                dev.cleanup()
            except Exception:
                pass


if __name__ == '__main__':
    htu = HtuExp()
    htu.jet.subscribe_var_values()

    time.sleep(1.0)
    print(f'Jet state:\n\t{htu.devs["jet"].state}')
    print(f'Jet setpoints:\n\t{htu.devs["jet"].config}')

    htu.cleanup()
