import time
from geecs_api.devices.HTU import GasJet, Laser, Transport
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
        self.e_transport = Transport(self.exp_devs)

        self.devs = {
            'laser': self.laser,
            'jet': self.jet,
            'e_transport': self.e_transport
        }

    def shutdown(self):
        self.jet.stage.set_position('Y', -5., exec_timeout=30, sync=False)
        self.jet.pressure.set_pressure(0., exec_timeout=30, sync=False)

        self.e_transport.hexapod.set_position('Y', -22., exec_timeout=120, sync=False)

        self.laser.seed_laser.amp4_shutter.insert(True, exec_timeout=30, sync=False)
        self.laser.pump_laser.set_lamp_timing(700., exec_timeout=30, sync=False)
        self.laser.pump_laser.shutters.insert('North', 1, True, exec_timeout=30, sync=False)
        self.laser.pump_laser.shutters.insert('South', 1, True, exec_timeout=30, sync=False)
        self.laser.pump_laser.shutters.insert('North', 2, True, exec_timeout=30, sync=False)
        self.laser.pump_laser.shutters.insert('South', 2, True, exec_timeout=30, sync=False)
        self.laser.pump_laser.shutters.insert('North', 3, True, exec_timeout=30, sync=False)
        self.laser.pump_laser.shutters.insert('South', 3, True, exec_timeout=30, sync=False)
        self.laser.pump_laser.shutters.insert('North', 4, True, exec_timeout=30, sync=False)
        self.laser.pump_laser.shutters.insert('South', 5, True, exec_timeout=30, sync=False)


if __name__ == '__main__':
    htu = HtuExp()

    time.sleep(1.0)
    # htu.jet.stage.set_position('X', 7.5)
    # htu.jet.pressure.set_pressure(290.)
    # htu.jet.trigger.run(False)
    # htu.jet.blade.set_depth(-17.1)
    # htu.laser.compressor.get_separation()

    time.sleep(1.0)
    print(f'Stage state:\n\t{htu.jet.stage.state}')
    print(f'Pressure state:\n\t{htu.jet.pressure.state}')
    print(f'Trigger state:\n\t{htu.jet.trigger.state}')
    print(f'Blade state:\n\t{htu.jet.blade.state}')
    print(f'Compressor state:\n\t{htu.laser.compressor.state}')
    print(f'Seed shutter (Amp4) state:\n\t{htu.laser.seed_laser.amp4_shutter.state}')
    print(f'Pump laser state:\n\t{htu.laser.pump_laser.state}')
    print(f'Pump shutters state:\n\t{htu.laser.pump_laser.shutters.state}')
    print(f'Hexapod state:\n\t{htu.e_transport.hexapod.state}')

    # print(f'Stage setpoints:\n\t{htu.jet.stage.setpoints}')

    htu.cleanup()
