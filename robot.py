# robot.py

from dataclasses import dataclass, asdict
import numpy as np

@dataclass
class RobotState:
    """Simple state representation."""
    time: float
    base_position: np.ndarray  # [x, y, z]
    base_orientation: np.ndarray  # [roll, pitch, yaw]
    joint_positions: np.ndarray  # 12 joints
    joint_velocities: np.ndarray
    # foot_contacts: np.ndarray  # [FL, FR, RL, RR]

@dataclass  
class Command:
    """Command from controller."""
    type: str  # 'torque', 'stand', 'walk', 'jump'
    data: dict  # Command-specific data

class Robot:
    def __init__(self):
        global sim_running
        self.sim_running = False

