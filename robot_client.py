# robot_client.py
"""Robot control client - send commands and receive state."""

import zmq
import numpy as np
import pickle
import time
from dataclasses import dataclass
from typing import Optional
from robot import RobotState, Robot, Command


class RobotClient:
    """Client to communicate with simulation server."""
    
    def __init__(self, host: str = "localhost", port: int = 5555):
        self.ctrl_running = True
        # ZeroMQ setup - REQ (Request) socket
        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{host}:{port}")
        print(f"Connected to simulation at {host}:{port}")
        self.last_state = None
    
    def send_command(self, cmd: Command) -> dict:
        """
        Send command and receive state.
        
        Returns:
            Current robot state as dictionary
        """
        # Send command
        self.socket.send(pickle.dumps(cmd))
        
        # Receive state
        msg = self.socket.recv()
        state = pickle.loads(msg)
        self.last_state = state
        
        return state
    
    # ========== High-Level Commands ==========
    
    def stand(self, height: float = 0.3) -> dict:
        """Command robot to stand at specified height."""
        cmd = Command(type='stand', data={'height': height})
        return self.send_command(cmd)
    
    def send_torques(self, torques: np.ndarray) -> dict:
        """Send joint torques."""
        cmd = Command(type='torque', data={'torques': torques.tolist()})
        return self.send_command(cmd)
    
    def zero_torques(self) -> dict:
        """Set all torques to zero."""
        cmd = Command(type='zero', data={})
        return self.send_command(cmd)
    
    def get_state(self) -> dict:
        """Get current state (sends zero command)."""
        return self.zero_torques()
    
    def close(self):
        """Close connection."""
        self.socket.close()

    def stop(self):
        """Send stop command to simulation."""
        self.ctrl_running = False
        cmd = Command(type='stop', data={})
        return self.send_command(cmd)


# ========== Simple API ==========

def create_robot(host="localhost", port=5555) -> RobotClient:
    """Create robot client (simple interface)."""
    return RobotClient(host, port)


# ========== Example Usage ==========

def example_interactive():
    """Interactive control example."""
    robot = create_robot()
    
    print("\n" + "="*50)
    print("Robot Control Interface")
    print("="*50)
    
    try:
        # Stand at default height
        print("\n[1] Standing at 0.3m...")
        state = robot.stand(height=0.3)
        print(f"    Position: {state['base_position']}")
        time.sleep(3)
        
        # Stand higher
        print("\n[2] Standing at 0.4m...")
        state = robot.stand(height=0.4)
        print(f"    Position: {state['base_position']}")
        time.sleep(3)
        
        # Stand lower
        print("\n[3] Standing at 0.25m...")
        state = robot.stand(height=0.25)
        print(f"    Position: {state['base_position']}")
        time.sleep(3)
        
        # Zero torques
        print("\n[4] Releasing (zero torques)...")
        robot.zero_torques()
        time.sleep(2)
        
    except KeyboardInterrupt:
        print("\n✓ Control stopped")
    finally:
        robot.close()


def example_continuous_control():
    """Example: Continuous control loop."""
    robot = create_robot()
    
    print("\nRunning continuous control for 10 seconds...")
    print("Press Ctrl+C to stop\n")
    
    start_time = time.time()
    
    try:
        while time.time() - start_time < 10.0:
            # Get current state
            state = robot.get_state()
            
            # Compute torques (example: simple PD)
            target_joints = np.array([0.0, 0.8, -1.6] * 4)
            current_joints = np.array(state['joint_positions'])
            
            kp = 20.0
            kd = 2.0
            
            torques = kp * (target_joints - current_joints) - kd * np.array(state['joint_velocities'])
            
            # Send torques
            robot.send_torques(torques)
            
            # Print status
            z = state['base_position'][2]
            print(f"Time: {state['time']:.2f}s | Height: {z:.3f}m", end='\r')
            
            time.sleep(0.002)  # 500 Hz control
            
    except KeyboardInterrupt:
        print("\n✓ Control stopped")
    finally:
        robot.zero_torques()
        robot.close()

def main():
    robot = create_robot()

    def stop():
        robot.stop()
        running = False


    while robot.ctrl_running is True:
        try:
            command = input(">>> ")
            eval(command)
        except Exception as e:
            print(f"Error: {e}")
    
    print("Mujoco simulation disconnected.")
    print("Controller closed.")

if __name__ == "__main__":
    # Run interactive example
    main()