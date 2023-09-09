from typing import Optional
from geecs_python_api.controls.api_defs import exec_async
from geecs_python_api.controls.devices.HTU import Laser, GasJet, Diagnostics, Transport
from geecs_python_api.controls.experiment import Experiment


class HtuExp(Experiment):
    # Singleton
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'instance'):
            cls.instance = super(HtuExp, cls).__new__(cls)
            cls.instance.__initialized = False
        return cls.instance

    def __init__(self, get_info: bool = True):
        # Singleton
        if self.__initialized:
            return
        self.__initialized = True

        super().__init__('Undulator', get_info)

        self.laser: Optional[Laser] = None
        self.jet: Optional[GasJet] = None
        self.diagnostics: Optional[Diagnostics] = None
        self.transport: Optional[Transport] = None

    def connect(self, laser: bool = True, jet: bool = True, diagnostics: bool = True, transport: bool = True):
        # Devices
        if isinstance(self.laser, Laser):
            self.laser.close()
        self.laser = Laser() if laser else None

        if isinstance(self.jet, GasJet):
            self.jet.close()
        self.jet = GasJet() if jet else None

        if isinstance(self.diagnostics, Diagnostics):
            self.diagnostics.close()
        self.diagnostics = Diagnostics() if diagnostics else None

        if isinstance(self.transport, Transport):
            self.transport.close()
        self.transport = Transport() if transport else None

        # Dictionary
        self.devs = {
            'laser': self.laser,
            'jet': self.jet,
            'diagnostics': self.diagnostics,
            'transport': self.transport
        }  # handy to manipulate devices by batch

    def beta_shutdown(self) -> bool:
        if isinstance(self.laser, Laser):
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
        else:
            is_amp4_in = is_gaia_in = False

        if isinstance(self.transport, Transport):
            is_hexapod_out = self.transport.pmq.move_out(exec_timeout=120)
        else:
            is_hexapod_out = False

        if isinstance(self.jet, GasJet):
            is_pressure_zero = abs(self.jet.pressure.state_psi()) < 0.1
            is_jet_out = abs(self.jet.stage.state_y() + 5.) < 0.01
        else:
            is_pressure_zero = is_jet_out = False

        # is_lamp_timing_off = abs(self.laser.pump.state_lamp_timing() - lamp_timing) < 1
        is_lamp_timing_off = True

        return is_pressure_zero and is_jet_out and is_amp4_in and is_lamp_timing_off and is_gaia_in and is_hexapod_out


if __name__ == '__main__':
    # Experiment.get_info('Undulator')

    htu = HtuExp(get_info=True)
    htu.connect()
    htu.close()

    print('done')
