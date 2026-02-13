"""
Generic Interlock Server
Allows custom logic to set interlock flags that are broadcast to TCP clients.
"""
import socket
import threading
import time
from typing import Dict, Callable, Any


class InterlockServer:
    """
    A generic TCP server that broadcasts interlock status flags.
    
    Usage:
        server = InterlockServer(host="127.0.0.1", port=5001)
        server.start()
        
        # Set interlock flags from your custom logic
        server.set_interlock("device1", True)
        server.set_interlock("device2", False)
        
        # Or register monitoring functions
        server.register_monitor("device1", my_check_function, interval=0.5)
    """
    
    def __init__(self, host="127.0.0.1", port=5001):
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
            name: Name/identifier for the interlock
            is_active: True if interlock should trigger, False if OK
        """
        with self.flags_lock:
            old_state = self.interlock_flags.get(name, False)
            self.interlock_flags[name] = is_active
            if is_active != old_state:
                status = "ACTIVE" if is_active else "OK"
                print(f"[{name}] Interlock {status}")
    
    def get_interlock(self, name: str) -> bool:
        """Get the current state of an interlock flag."""
        with self.flags_lock:
            return self.interlock_flags.get(name, False)
    
    def get_all_interlocks(self) -> Dict[str, bool]:
        """Get all interlock flags."""
        with self.flags_lock:
            return self.interlock_flags.copy()
    
    def register_monitor(self, name: str, check_func: Callable[[], bool], interval: float = 0.5):
        """
        Register a monitoring function that runs periodically.
        
        Args:
            name: Name/identifier for this monitor
            check_func: Function that returns True if interlock should trigger, False if OK
            interval: How often to run the check (seconds)
        """
        def monitor_loop():
            self.interlock_flags[name] = False  # Initialize
            while self.server_running:
                try:
                    result = check_func()
                    self.set_interlock(name, result)
                except Exception as e:
                    print(f"Error in monitor '{name}': {e}")
                    self.set_interlock(name, True)  # Set interlock on error
                time.sleep(interval)
        
        thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitor_threads.append(thread)
        if self.server_running:
            thread.start()
    
    def _handle_client(self, conn, addr):
        """Handle a connected client and send status updates."""
        print(f"Client connected: {addr}")
        try:
            while self.server_running:
                with self.flags_lock:
                    status_lines = []
                    for name, flag in self.interlock_flags.items():
                        status = "INTERLOCK_ACTIVE" if flag else "OK"
                        status_lines.append(f"{name}: {status}")
                    message = " | ".join(status_lines) + "\n" if status_lines else "No monitors active\n"
                
                conn.sendall(message.encode('utf-8'))
                time.sleep(0.5)
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


# Example helper for camera-based interlocks
def create_camera_check(camera, variable_name: str, threshold: float, above: bool = True):
    """
    Create a check function for camera-based interlocks.
    
    Args:
        camera: GeecsDevice camera object
        variable_name: Variable to monitor (e.g., 'MaxCounts')
        threshold: Threshold value
        above: If True, trigger when value > threshold, else when value < threshold
    
    Returns:
        Function that returns True if interlock should trigger
    """
    def check():
        value = camera.get(variable_name)
        if value is None:
            return False
        return (value > threshold) if above else (value < threshold)
    return check
