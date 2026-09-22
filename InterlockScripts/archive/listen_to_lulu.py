"""Standalone raw-socket UDP broadcast listener (archived debug script)."""

import socket

SERVER_PORT = 5005


def main():
    """Listen for and print UDP broadcasts on SERVER_PORT."""
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    client_socket.bind(("", SERVER_PORT))

    print(f"Listening for broadcasts on port {SERVER_PORT}...")

    try:
        while True:
            data, server_address = client_socket.recvfrom(1024)
            print(f"Received from server: {data.decode()}")
    except KeyboardInterrupt:
        print("Client interrupted. Exiting...")
    finally:
        client_socket.close()


if __name__ == "__main__":
    main()
