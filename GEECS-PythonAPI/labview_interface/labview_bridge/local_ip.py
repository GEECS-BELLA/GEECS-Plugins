import subprocess
import re


def id_ethernet_ports():
    """ Identify active ports. """
    ip_cfg = subprocess.check_output("ipconfig /all").decode('utf-8')
    name = ''
    ports = ()

    while True:
        port = re.search(r'[^\r\n]+:\r\n', ip_cfg)

        if port is None:
            break

        description = ip_cfg[:port.start(0)]
        dhcp_line = re.search(r'DHCP Enabled[^\r\n]+', description)

        if dhcp_line is not None:
            dhcp_line = dhcp_line.group(0)
            # dhcp_line = description[dhcp_line.start(0):dhcp_line.end(0)]
            # dhcp_status = re.search(r'[^\s]+$', dhcp_line)
            # dhcp_status = dhcp_line[dhcp_status.start(0):dhcp_status.end(0)].lower() == 'yes'
            dhcp_status = re.search(r'[^\s]+$', dhcp_line).group(0).lower() == 'yes'

            ip_line = re.search(r"IPv4 Address[.\s:]*[0-9.]+", description)
            if ip_line is not None:
                ip_line = description[ip_line.start(0):ip_line.end(0)]
                ip_value = re.search(r'[0-9.]+$', ip_line).group(0)
                # ip_value = ip_line[ip_value.start(0):ip_value.end(0)]
            else:
                ip_value = None

            subnet_line = re.search(r'Subnet Mask[.\s:]*[0-9.]+', description)
            if subnet_line is not None:
                subnet_line = description[subnet_line.start(0):subnet_line.end(0)]
                subnet_value = re.search(r'[0-9.]+$', subnet_line).group(0)
                # subnet_value = subnet_line[subnet_value.start(0):subnet_value.end(0)]
            else:
                subnet_value = None

            name = re.match(r'[^:]+', name).group(0)

            if (ip_value is not None) and (subnet_value is not None):
                ports = (*ports, dict(DHCP=dhcp_status, IPv4=ip_value, Subnet=subnet_value, Name=name))

        name = port.group(0)
        ip_cfg = ip_cfg[port.end(0):]

    return ports


def test_internet_connection(port='localhost', attempts=4, timeout_ms=500):
    """ Test if port is connected to the internet. """
    ping_return = subprocess.check_output(f'ping -n {attempts} -w {timeout_ms} -S {port} google.com').decode('utf-8')

    return re.search('Reply from', ping_return) is not None


def local_ip():
    """ Find active interface. """
    ports = id_ethernet_ports()

    if ports is None:
        return 'localhost'

    for port in ports:
        try:
            port_connected = test_internet_connection(port["IPv4"], attempts=1)
        except:
            port_connected = False

        if port_connected:
            return port["IPv4"]

    return 'localhost'


if __name__ == "__main__":
    ethernet_ports = id_ethernet_ports()

    if len(ethernet_ports) > 0:
        for i_port in range(len(ethernet_ports)):
            try:
                print(f'port {ethernet_ports[i_port]["IPv4"]} connected: '
                      f'{test_internet_connection(ethernet_ports[i_port]["IPv4"], attempts=1)}')
            except:
                print(f'port {ethernet_ports[i_port]["IPv4"]} connected: False')

    print(f'local IP: {local_ip()}')
