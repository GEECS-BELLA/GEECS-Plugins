import socket
from typing import Optional
from pathlib import Path
from dotenv import dotenv_values
from geecs_python_api.controls.devices import GeecsDevice
from geecs_python_api.controls.interface import GeecsDatabase, api_error


class Experiment:
    def __init__(self, name: str, get_info: bool = True):
        self.exp_name: str = name
        self.devs: dict[str, GeecsDevice] = {}

        self.base_path: Path
        try:
            env_dict: dict = dotenv_values()
            self.base_path = Path(env_dict['DATA_BASE_PATH'])
        except Exception:
            self.base_path = Path(r'Z:\data')

        self.is_offline = (self.base_path.drive.lower() == 'c:')
        if get_info and not self.is_offline:
            GeecsDevice.exp_info = GeecsDatabase.collect_exp_info(self.exp_name)

    def close(self):
        for dev in self.devs.values():
            try:
                if dev is not None:
                    dev.close()
            except Exception:
                pass

    def wait_for_all_devices(self, timeout: Optional[float] = None) -> bool:
        synced = True
        for dev in self.devs.values():
            synced &= dev.wait_for_all_devices(timeout)

        return synced

    def send_preset(self, preset: str):
        if self.devs and preset:
            MCAST_GRP = '234.5.6.8'
            MCAST_PORT = 58432
            MULTICAST_TTL = 4

            sock = None
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
                sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, MULTICAST_TTL)
                sock.sendto(f'preset>>{preset}>>{socket.gethostbyname(socket.gethostname())}'.encode(),
                            (MCAST_GRP, MCAST_PORT))

            except Exception:
                api_error.error(f'Failed to send preset "{preset}"', 'UdpHandler class, method "send_preset"')

            finally:
                try:
                    sock.close()
                except Exception:
                    pass

    @staticmethod
    def get_info(exp_name: str, is_local: bool = False):
        if not is_local:
            GeecsDevice.exp_info = GeecsDatabase.collect_exp_info(exp_name)
