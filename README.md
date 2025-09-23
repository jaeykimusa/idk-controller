# idk-controller

# Go2 Robot Simulation System

A modular client-server architecture for controlling the Unitree Go2 robot in MuJoCo simulation.

## Architecture Overview

The system consists of two main components:

1. **Simulation Server** (`simulation_server.py`): Runs the MuJoCo simulation and listens for control commands
2. **Control Client** (`control_client.py`): Sends commands to control the robot

## Installation

### Prerequisites
```bash
pip install mujoco numpy scipy
```

## Usage

### Step 1: Start the Simulation Server

Open a terminal and run:
```bash
python simulation_server.py
```

This will:
- Start the MuJoCo simulation viewer
- Open a socket server on `localhost:5555`
- Wait for client connections

### Step 2: Connect the Control Client

Open another terminal and run:
```bash
python control_client.py
```

## Control Commands

### Interactive Mode Commands

When running the client in interactive mode, you can use:

- `stand [speed]` - Make the robot stand up (speed: 0.1-2.0, default: 0.5)
- `sit [speed]` - Make the robot sit down
- `neutral` - Move to neutral position (all joints at 0)
- `stretch` - Move to stretch position
- `stop` - Stop current motion immediately
- `reset` - Reset simulation to initial state
- `state` - Get current robot state and joint positions
- `custom <j1> <j2> ... <j12>` - Set custom joint angles (12 values in radians)
- `demo` - Run an automated demo sequence
- `exit` or `quit` - Exit the client
- `shutdown` - Shutdown the server and exit

### Example Interactive Session

```
Command> stand
[RESPONSE] Standing up

Command> state
[ROBOT STATE]
  State: standing
  Controller Active: True
  Trajectory Progress: 45.2%

Command> sit 0.3
[RESPONSE] Sitting down

Command> demo
[DEMO] Starting demo sequence...
```

### Batch Mode

You can also run commands in batch mode:

```bash
# Run a sequence of commands
python control_client.py --batch stand sit stand reset

# Run the demo
python control_client.py --demo
```

### Connect to Remote Server

If running on different machines:

```bash
# On server machine
python simulation_server.py

# On client machine (replace with server's IP)
python control_client.py --host 192.168.1.100 --port 5555
```

## Customization

### Adding New Positions

In `simulation_server.py`, add to the `positions` dictionary:

```python
self.positions = {
    'stand': np.array([0, 0.9, -1.8] * 4),
    'sit': self._get_sit_position(),
    'neutral': np.zeros(12),
    'stretch': np.array([0, 0.4, -0.8] * 4),
    'your_pose': np.array([...]),  # Add your custom pose
}
```

### Adjusting Control Parameters

Modify PD gains in `TrajectoryController`:

```python
self.controller = TrajectoryController(kp=30, kd=0.5)  # Adjust kp and kd
```

### Creating Custom Controllers

You can extend the `TrajectoryController` class to implement different control strategies:

```python
class WalkingController(TrajectoryController):
    def compute_control(self, q_current, qd_current, dt):
        # Implement walking gait
        pass
```

## Communication Protocol

The client and server communicate using JSON messages over TCP sockets.

### Command Format

```json
{
    "type": "stand",
    "speed": 0.5
}
```

### Response Format

```json
{
    "status": "success",
    "message": "Standing up"
}
```

### State Query Response

```json
{
    "status": "success",
    "state": {
        "robot_state": "standing",
        "joint_positions": [...],
        "joint_velocities": [...],
        "controller_active": true,
        "trajectory_progress": 0.452
    }
}
```

## Extending the System

### Adding New Commands

1. In `simulation_server.py`, add to `process_command()`:
```python
elif cmd_type == 'your_command':
    # Your command logic
    self.send_response({"status": "success", "message": "Command executed"})
```

2. In `control_client.py`, add a method:
```python
def your_command(self):
    response = self.send_and_receive({'type': 'your_command'})
    if response:
        print(f"[RESPONSE] {response.get('message')}")
```

### Adding Sensors

You can extend the state information to include sensor data:

```python
def get_sensor_data(self):
    return {
        "imu": self.sim.data.sensordata[...],
        "contact_forces": self.sim.data.cfrc_ext[...],
        # Add more sensors
    }
```

## Troubleshooting

### Connection Issues
- Ensure the server is running before starting the client
- Check firewall settings if using remote connections
- Verify the correct host and port settings

### Simulation Issues
- Update the XML path in `simulation_server.py` to your robot model location
- Ensure MuJoCo is properly installed and licensed

### Control Issues
- Adjust PD gains if the robot is unstable
- Reduce speed parameter if movements are too fast
- Check joint limits in your robot model

## Architecture Benefits

1. **Modularity**: Simulation and control logic are separated
2. **Flexibility**: Easy to add new controllers and commands
3. **Remote Control**: Can control simulation from different machines
4. **Extensibility**: Simple to add sensors, new gaits, or AI controllers
5. **Testing**: Can test control algorithms without modifying simulation code

## Future Enhancements

- Add walking gait controller
- Implement ROS integration
- Add GUI control interface
- Include sensor feedback visualization
- Support multiple robot clients
- Add recording and playback functionality