"""
Example 3: Manual interlock control - set flags directly from your script
"""
import time
from interlock_server import InterlockServer

# Create server
server = InterlockServer(host="127.0.0.1", port=5001)
server.start()

print("Manual interlock server running.")
print("Setting interlock flags programmatically...\n")

# Manually control interlocks from your script logic
for i in range(20):
    # Your custom logic here - can be anything!
    
    # Example: Set interlock based on time
    if i % 5 == 0:
        server.set_interlock("ProcessA", True)
        print(f"[{i}] ProcessA: INTERLOCK TRIGGERED")
    else:
        server.set_interlock("ProcessA", False)
    
    # Example: Set interlock based on computation
    value = i * 10
    if value > 100:
        server.set_interlock("ProcessB", True)
        print(f"[{i}] ProcessB: INTERLOCK TRIGGERED (value={value})")
    else:
        server.set_interlock("ProcessB", False)
    
    # Example: Combine multiple conditions
    is_critical = (i > 10) and (i % 2 == 0)
    server.set_interlock("CriticalState", is_critical)
    if is_critical:
        print(f"[{i}] CriticalState: INTERLOCK TRIGGERED")
    
    time.sleep(1)

print("\nDemo complete. Server still running. Press Ctrl+C to stop.")
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    server.stop()
