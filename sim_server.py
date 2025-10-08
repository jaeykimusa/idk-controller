# sim_server.py
"""Simulation server - runs MuJoCo and listens for commands."""

import zmq
import numpy as np
import mujoco
import mujoco.viewer
import pickle
import time
from dataclasses import dataclass, asdict
from scipy.spatial.transform import Rotation as R

from pathlib import Path
from robot import RobotState, Robot, Command

class MujocoServer:
    """MuJoCo simulation server."""
    
    def __init__(self, xml_path: str, port: int = 5555):
        # Load MuJoCo
        self.robot = Robot()
        self.robot.sim_running = True
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        
        # ZeroMQ setup - REP (Reply) socket
        context = zmq.Context()
        self.socket = context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{port}")
        print(f"✓ Simulation server started on port {port}")
        
        # Current torque command
        self.u_cmd = np.zeros(12)
        
    def get_state(self) -> RobotState:
        """Extract current state."""
        return RobotState(
            time=self.data.time,
            base_position=self.data.qpos[0:3].copy(),
            base_orientation=R.from_quat([self.data.qpos[4], self.data.qpos[5], self.data.qpos[6], self.data.qpos[3]]).as_euler('xyz', degrees=False),
            joint_positions=self.data.qpos[7:19].copy(),
            joint_velocities=self.data.qvel[6:18].copy(),
            # foot_contacts=self._detect_contacts(),
        )
    
    def apply_command(self, cmd: Command):
        """Apply command to simulation."""
        if cmd.type == 'torque':
            self.u_cmd = np.array(cmd.data['torques'])
        elif cmd.type == 'zero':
            self.u_cmd = np.zeros(12)
        elif cmd.type == 'stop':
            self.robot.sim_running = False
    
    def step(self):
        """Step simulation."""
        # Apply torques
        self.data.ctrl[:12] = self.u_cmd
        
        # Step
        mujoco.mj_step(self.model, self.data)
        self.viewer.sync()
    
    def run(self):
        """Main server loop."""
        print("Waiting for commands...\n")
        try:
            while self.robot.sim_running is True:
                # Check for command (non-blocking with timeout)
                if self.socket.poll(timeout=1):  # 1ms timeout
                    # Receive command
                    msg = self.socket.recv()
                    cmd = pickle.loads(msg)
                    
                    # Apply command
                    self.apply_command(cmd)
                    print(f"Received: {cmd.type}", end='\r')
                    
                    # Send state back
                    state = self.get_state()
                    self.socket.send(pickle.dumps(asdict(state)))
                
                # Step simulation
                self.step()

            # This runs after 'running' becomes False and loop exits
            print("Mujoco simulation is closed.")
            self.viewer.close()
                
        except KeyboardInterrupt:
            print("\nSimulation stopped")
            self.viewer.close()
    
    # def _detect_contacts(self) -> np.ndarray:
    #     """Simple contact detection."""
    #     contacts = np.zeros(4)
    #     # Implement based on your model
    #     return contacts


if __name__ == "__main__":    
    REPO_ROOT = Path(__file__).resolve().parent.parent
    SCENE_XML_PATH = REPO_ROOT / "idk-controller" / "robot_description" / "go2" / "scene.xml"

    sim_server = MujocoServer(str(SCENE_XML_PATH))
    sim_server.run()