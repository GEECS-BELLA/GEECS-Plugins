import geecs_api.interface as gi


class GeecsDevice:
    def __init__(self, name='', debug=False):
        dev_name = name
        dev_tcp = gi.TcpSubscriber()
        dev_udp = gi.UdpHandler()
        dev_ip = dev_port = ''

        if name:
            dev_ip, dev_port = gi.GDb.find_device(name)
            if debug:
                if dev_ip:
                    print(f'Device found: {dev_ip}, {dev_port}')
                else:
                    print('Device not found')


if __name__ == '__main__':
    my_dev = GeecsDevice('U_ESP_JetXYZ', debug=True)
    my_dev.
