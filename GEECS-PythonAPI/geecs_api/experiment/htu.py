from geecs_api.api_defs import exec_async
from geecs_api.devices.HTU import Laser, GasJet, Transport, Diagnostics
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
        self.transport = Transport(self.exp_devs)
        self.diagnostics = Diagnostics(self.exp_devs)

        self.devs = {
            'laser': self.laser,
            'jet': self.jet,
            'diagnostics': self.diagnostics,
            'transport': self.transport
        }  # handy to manipulate devices by batch

    def shutdown(self) -> bool:
        exec_async(self.jet.pressure.set_pressure, args=(0., 30))
        exec_async(self.jet.stage.set_position, ('Y', -5., 30))

        is_amp4_in = self.laser.seed.amp4_shutter.insert(exec_timeout=30)

        exec_async(self.laser.pump.set_lamp_timing, (700., 30))

        is_gaia_in = self.laser.pump.shutters.insert('North', 1, exec_timeout=30)
        is_gaia_in &= self.laser.pump.shutters.insert('South', 1, exec_timeout=30)
        is_gaia_in &= self.laser.pump.shutters.insert('North', 2, exec_timeout=30)
        is_gaia_in &= self.laser.pump.shutters.insert('South', 2, exec_timeout=30)
        is_gaia_in &= self.laser.pump.shutters.insert('North', 3, exec_timeout=30)
        is_gaia_in &= self.laser.pump.shutters.insert('South', 3, exec_timeout=30)
        is_gaia_in &= self.laser.pump.shutters.insert('North', 4, exec_timeout=30)
        is_gaia_in &= self.laser.pump.shutters.insert('South', 4, exec_timeout=30)

        is_hexapod_out = self.transport.hexapod.move_out(exec_timeout=120)

        is_pressure_zero = abs(self.jet.pressure.state_psi()) < 0.1
        is_jet_out = abs(self.jet.stage.state_y() + 5.) < 0.01
        is_lamp_timing_off = abs(self.laser.pump.state_lamp_timing() - 700.) < 1

        return is_pressure_zero and is_jet_out and is_amp4_in and is_lamp_timing_off and is_gaia_in and is_hexapod_out


if __name__ == '__main__':
    htu = HtuExp()
    htu.cleanup()
