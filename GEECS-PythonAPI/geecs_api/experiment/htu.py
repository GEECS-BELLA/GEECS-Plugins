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
        self.laser = Laser()
        self.jet = GasJet()
        self.diagnostics = Diagnostics()
        self.transport = Transport()

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

        # lamp_timing: float = 600.
        # exec_async(self.laser.pump.set_lamp_timing, (lamp_timing, 30))

        is_gaia_in = self.laser.pump.shutters.insert(1, 'North', exec_timeout=30)
        is_gaia_in &= self.laser.pump.shutters.insert(1, 'South', exec_timeout=30)
        is_gaia_in &= self.laser.pump.shutters.insert(2, 'North', exec_timeout=30)
        is_gaia_in &= self.laser.pump.shutters.insert(2, 'South', exec_timeout=30)
        is_gaia_in &= self.laser.pump.shutters.insert(3, 'North', exec_timeout=30)
        is_gaia_in &= self.laser.pump.shutters.insert(3, 'South', exec_timeout=30)
        is_gaia_in &= self.laser.pump.shutters.insert(4, 'North', exec_timeout=30)
        is_gaia_in &= self.laser.pump.shutters.insert(4, 'South', exec_timeout=30)

        is_hexapod_out = self.transport.hexapod.move_out(exec_timeout=120)

        is_pressure_zero = abs(self.jet.pressure.state_psi()) < 0.1
        is_jet_out = abs(self.jet.stage.state_y() + 5.) < 0.01
        # is_lamp_timing_off = abs(self.laser.pump.state_lamp_timing() - lamp_timing) < 1
        is_lamp_timing_off = True

        return is_pressure_zero and is_jet_out and is_amp4_in and is_lamp_timing_off and is_gaia_in and is_hexapod_out


if __name__ == '__main__':
    htu = HtuExp()
    htu.cleanup()
