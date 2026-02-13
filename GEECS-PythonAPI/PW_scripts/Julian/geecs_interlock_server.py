import socket
import threading
import time
from typing import Dict, Callable, Any # Added Any for type hinting in monitor functions
import struct


class InterlockServer:
    """
    TCP server that broadcasts interlock status flags to clients.
    
    Usage:
        server = InterlockServer(host="IP ADDRESS", port="PORT")
        server.set_interlock("device1", True)
        server.set_interlock("device2", False)

        # or register some custom device monitoring functions
        server.register_monitor("device1", my_custom_check, interval=0.5)
    """

    def __init__(self, host="192.168.14.14", port=5001):
        self.host = host
        self.port = port
        self.interlock_flags: Dict[str, bool] = {}
        self.flags_lock = threading.Lock()
        self.server_running = False
        self._server_thread = None
        self._monitor_threads = []

    def set_interlock(self, name: str, is_active: bool):
        """
        Set an interlock flag manually.

        Args:
            name: indentifier for the interlock
            is_active: True if the interlock should trigger
        """
        with self.flags_lock:
            old_state = self.interlock_flags.get(name, False)
            self.interlock_flags[name] = is_active
            if is_active != old_state:
                status = "ACTIVE" if is_active else "NOT_ACTIVE"
                print(f"[{name}] Interlock {status}")

    def get_interlock(self, name: str) -> bool:
        """ 
        Get the current state of an interlock flag.

        """
        with self.flags_lock:
            return self.interlock_flags.get(name, False)
        
    def get_all_interlocks(self) -> Dict[str, bool]:
        """ Gett all interlock flags. """
        with self.flags_lock:
            return self.interlock_flags.copy()
        
    def register_monitor(self, name: str, check_func: Callable[[], bool], interval: float = 0.5):
        """ 
        Register a monitoring function that periodically checks a condition and updates the interlock flag.

        Args:
            name: identifier for the interlock
            check_func: returns True if interlock should trigger
            interval: acquisition interval (sec)
        """
        def monitor_loop():
            self.interlock_flags[name] = False # Initialize
            while self.server_running:
                try:
                    result = check_func()
                    self.set_interlock(name, result)
                except Exception as e:
                    print(f"Error in monitor '{name}': {e}")
                    self.set_interlock(name, True)  # Set interlock active on error (check w Tony)
                time.sleep(interval)

        thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitor_threads.append(thread)
        if self.server_running:
            thread.start()


### TCP Server Code Below - No need to modify unless you want to change the protocol or add authentication, etc. ###


    def _handle_client(self, conn, addr):
        """Handle a connected client and send status updates."""
        print(f"Client connected: {addr}")
        try:
            while self.server_running:
                with self.flags_lock:
                    status_lines = []
                    for name, flag in self.interlock_flags.items():
                        status = "WARNING! Interlock conditions not met." if flag else "SAFE"
                        status_lines.append(f"{name}: {status}")
                    message = " | ".join(status_lines) if status_lines else "No monitors active"
                
                # encode the message
                message_bytes = message.encode('utf-8')

                # send length as 4-byte integer followed by message
                length_prefix = struct.pack('>I', len(message_bytes))
                conn.sendall(length_prefix + message_bytes)

                time.sleep(0.5) # Adjust the sending interval as needed
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            try:
                conn.close()
            except:
                pass
            print(f"Client disconnected: {addr}")
    
    def _server_loop(self):
        """Main server loop."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((self.host, self.port))
            s.listen()
            print(f"Interlock server listening on {self.host}:{self.port}")
            
            while self.server_running:
                try:
                    s.settimeout(1.0)
                    conn, addr = s.accept()
                    client_thread = threading.Thread(
                        target=self._handle_client, 
                        args=(conn, addr), 
                        daemon=True
                    )
                    client_thread.start()
                except socket.timeout:
                    continue
                except Exception as e:
                    if self.server_running:
                        print(f"Server error: {e}")
    
    def start(self):
        """Start the interlock server."""
        if self.server_running:
            print("Server is already running")
            return
        
        self.server_running = True
        
        # Start all registered monitor threads
        for thread in self._monitor_threads:
            thread.start()
        
        # Start server thread
        self._server_thread = threading.Thread(target=self._server_loop, daemon=True)
        self._server_thread.start()
        print(f"Interlock server started with {len(self.interlock_flags)} monitor(s)")
    
    def stop(self):
        """Stop the interlock server."""
        print("Stopping interlock server...")
        self.server_running = False
        if self._server_thread:
            self._server_thread.join(timeout=2)
        print("Interlock server stopped")