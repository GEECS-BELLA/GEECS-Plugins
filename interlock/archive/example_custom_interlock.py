"""
Example 2: Using InterlockServer with custom logic (not cameras)
"""
import time
import random
from interlock_server import InterlockServer

# Create server
server = InterlockServer(host="127.0.0.1", port=5001)

# Example 1: Custom check function - temperature monitoring
def check_temperature():
    """Simulate temperature check"""
    temp = random.uniform(20, 100)
    return temp > 80  # Interlock if temperature exceeds 80

# Example 2: Custom check function - pressure monitoring
def check_pressure():
    """Simulate pressure check"""
    pressure = random.uniform(0, 150)
    return pressure > 120  # Interlock if pressure exceeds 120

# Example 3: File-based interlock
def check_file_exists():
    """Interlock if a specific file exists"""
    import os
    return os.path.exists("C:\\temp\\emergency_stop.txt")

# Register all monitors
server.register_monitor("Temperature", check_temperature, interval=1.0)
server.register_monitor("Pressure", check_pressure, interval=0.5)
server.register_monitor("EmergencyStop", check_file_exists, interval=2.0)

# Start server
server.start()

print("Custom interlock server running with 3 monitors:")
print("  - Temperature (random simulation)")
print("  - Pressure (random simulation)")
print("  - Emergency stop file check (C:\\temp\\emergency_stop.txt)")
print("\nPress Ctrl+C to stop.")

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    server.stop()
