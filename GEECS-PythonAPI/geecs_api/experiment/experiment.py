import socket
from geecs_api.devices import GeecsDevice
from geecs_api.interface import GeecsDatabase, api_error


class Experiment:
    def __init__(self, name: str):
        self.exp_name: str = name
        self.exp_devs = GeecsDatabase.find_experiment_variables(self.exp_name)
        self.exp_guis = GeecsDatabase.find_experiment_guis(self.exp_name)
        self.devs: dict[str, GeecsDevice] = {}

    def cleanup(self):
        for dev in self.devs.values():
            try:
                dev.cleanup()
            except Exception:
                pass

    @staticmethod
    def send_preset(preset: str):
        if preset:
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
