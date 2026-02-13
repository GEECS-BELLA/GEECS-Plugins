# Interlock Server - Generic Abstraction

A flexible TCP-based interlock system that broadcasts status flags to clients. Works with any custom logic, not just cameras.

## Quick Start

### 1. Camera-based interlock (original use case)
```cmd
python example_camera_interlock.py
```

### 2. Custom logic interlock
```cmd
python example_custom_interlock.py
```

### 3. Manual control
```cmd
python example_manual_interlock.py
```

### 4. Monitor with client
```cmd
python interlock_client.py
```

## API Usage

### Basic Setup
```python
from interlock_server import InterlockServer

server = InterlockServer(host="127.0.0.1", port=5001)
server.start()
```

### Option 1: Manual Control
Set interlock flags directly from your script:
```python
server.set_interlock("my_check", True)   # Trigger interlock
server.set_interlock("my_check", False)  # Clear interlock
```

### Option 2: Register Monitoring Functions
Auto-monitor with periodic checks:
```python
def my_check_function():
    # Your custom logic
    return value > threshold  # True = interlock active

server.register_monitor("my_check", my_check_function, interval=0.5)
```

### Option 3: Camera Helper
```python
from interlock_server import create_camera_check

check = create_camera_check(camera, 'MaxCounts', threshold=140)
server.register_monitor("Camera1", check, interval=0.5)
```

## Examples

**Temperature monitoring:**
```python
def check_temp():
    temp = read_temperature_sensor()
    return temp > 80

server.register_monitor("Temperature", check_temp, interval=1.0)
```

**File-based emergency stop:**
```python
def check_estop():
    return os.path.exists("emergency_stop.txt")

server.register_monitor("EmergencyStop", check_estop, interval=2.0)
```

**Complex logic:**
```python
def complex_check():
    a = get_value_a()
    b = get_value_b()
    return (a > 100 and b < 50) or is_critical_condition()

server.register_monitor("ComplexLogic", complex_check)
```

## Files

- `interlock_server.py` - Core `InterlockServer` class
- `interlock_client.py` - TCP client for monitoring
- `example_camera_interlock.py` - Camera monitoring example
- `example_custom_interlock.py` - Custom logic example
- `example_manual_interlock.py` - Manual control example
