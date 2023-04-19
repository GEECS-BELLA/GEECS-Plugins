import socket
from typing import Optional, Any
from geecs_api.devices import GeecsDevice
from geecs_api.interface import GeecsDatabase, api_error


class Experiment:
    def __init__(self, name: str):
        self.exp_name: str = name
        self.devs: dict[str, GeecsDevice] = {}
        self.exp_info = GeecsDatabase.collect_exp_info(self.exp_name)

    def cleanup(self):
        for dev in self.devs.values():
            try:
                dev.cleanup()
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
