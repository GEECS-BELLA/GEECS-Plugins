"""
TCP Client for monitoring interlock flag status
Connects to the interlock server and displays real-time status
"""
import socket
import time
import struct

def run_interlock_client(host="192.168.14.14", port=5001):
    """Connect to interlock server and display status"""
    print(f"Connecting to interlock server at {host}:{port}...")
    
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
            client.connect((host, port))
            print("Connected! Monitoring interlock status...\n")
            
            print("=" * 80)
            while True:
                # First read the 4-byte length prefix
                length_data = b''
                while len(length_data) < 4:
                    chunk = client.recv(4 - len(length_data))
                    if not chunk:
                        print("Connection closed by server")
                        return
                    length_data += chunk

                # unpacl the length
                message_length = struct.unpack('>I', length_data)[0]
                # print(f"DEBUG: Expecting {message_length} bytes")
                
                # Safety check for unreasonable message sizes
                if message_length > 10000:  # 10KB should be plenty for status messages
                    print(f"ERROR: Message length {message_length} is unreasonably large!")
                    print("This likely means server and client are out of sync.")
                    print("Please restart both server and client.")
                    return

                # Now read exactly message_length bytes
                message_data = b''
                while len(message_data) < message_length:
                    chunk = client.recv(message_length - len(message_data))
                    if not chunk:
                        print("Connection closed by server")
                        return
                    message_data += chunk
                
                # print(f"DEBUG: Received {len(message_data)} bytes: {message_data}")
                
                #decode and display the message
                status_line = message_data.decode('utf-8').strip()
                timestamp = time.strftime("%H:%M:%S")
                
                # Parse multi-device status
                devices = status_line.split(" | ")
                
                # Display timestamp and status
                print(f"[{timestamp}] ", end="")
                for dev_status in devices:
                    print(f"{dev_status} ", end="")
                print(flush=True)
                    
    except ConnectionRefusedError:
        print("Error: Could not connect to server. Is it running?")
    except KeyboardInterrupt:
        print("\nClient stopped by user")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    run_interlock_client()
