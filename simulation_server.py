#!/usr/bin/env python3
"""
simulation_server.py - MuJoCo Simulation Server for Go2 Robot
Run this to start the simulation and listen for control commands
"""

import mujoco
from mujoco import viewer
import numpy as np
import socket
import json
import threading
import time
from scipy.spatial.transform import Rotation as R
from enum import Enum
from typing import Optional, Dict, Any


class RobotState(Enum):
    IDLE = "idle"
    STANDING = "standing"
    SITTING = "sitting"
    WALKING = "walking"


class MujocoSimulation:
    def __init__(self, xml_path: str):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.dt = self.model.opt.timestep
        self.viewer = None
        
    def nextStep(self):
        mujoco.mj_step(self.model, self.data)
        
    def inputControl(self, cmd: np.ndarray):
        self.data.ctrl = cmd
        
    def getQ(self) -> np.ndarray:
        q = self.data.qpos
        q_orient = R.from_quat([q[4], q[5], q[6], q[3]]).as_euler('xyz', degrees=False)
        q = np.concatenate((q[0:3], q_orient, q[7:]))
        return q
    
    def getQD(self) -> np.ndarray:
        return self.data.qvel
    
    def getJointPositions(self) -> np.ndarray:
        return self.data.qpos[-12:]
    
    def getJointVelocities(self) -> np.ndarray:
        return self.data.qvel[-12:]
    
    def setJointPositions(self, joint_positions: np.ndarray):
        self.data.qpos[-12:] = joint_positions
        
    def reset(self):
        mujoco.mj_resetData(self.model, self.data)


class TrajectoryController:
    """Base controller with trajectory interpolation"""
    def __init__(self, kp: float = 30, kd: float = 0.5):
        self.kp = kp
        self.kd = kd
        self.alpha = 0.0
        self.alpha_speed = 0.5
        self.q_start = None
        self.q_target = None
        self.active = False
        
    def start_trajectory(self, q_start: np.ndarray, q_target: np.ndarray, speed: float = 0.5):
        self.q_start = q_start.copy()
        self.q_target = q_target.copy()
        self.alpha = 0.0
        self.alpha_speed = speed
        self.active = True
        
    def update(self, dt: float) -> bool:
        """Update interpolation. Returns True if complete."""
        if not self.active:
            return True
        
        self.alpha = min(self.alpha + self.alpha_speed * dt, 1.0)
        if self.alpha >= 1.0:
            self.active = False
            return True
        return False
    
    def get_target_position(self) -> Optional[np.ndarray]:
        if not self.active or self.q_start is None or self.q_target is None:
            return None
        return self.q_start * (1 - self.alpha) + self.q_target * self.alpha
    
    def compute_control(self, q_current: np.ndarray, qd_current: np.ndarray, dt: float) -> np.ndarray:
        self.update(dt)
        q_desired = self.get_target_position()
        
        if q_desired is None:
            return np.zeros_like(q_current)
            
        qd_desired = np.zeros_like(qd_current)
        return self.kp * (q_desired - q_current) + self.kd * (qd_desired - qd_current)


class Go2SimulationServer:
    def __init__(self, xml_path: str, host: str = 'localhost', port: int = 5555):
        # Simulation
        self.sim = MujocoSimulation(xml_path)
        
        # Controllers
        self.controller = TrajectoryController(kp=30, kd=0.5)
        
        # Predefined positions
        self.positions = {
            'stand': np.array([0, 0.9, -1.8] * 4),
            'sit': self._get_sit_position(),
            'neutral': np.zeros(12),
            'stretch': np.array([0, 0.4, -0.8] * 4),
        }
        
        # Server state
        self.state = RobotState.IDLE
        self.running = True
        self.command_queue = []
        self.lock = threading.Lock()
        
        # Socket setup
        self.host = host
        self.port = port
        self.server_socket = None
        self.client_socket = None
        
    def _get_sit_position(self) -> np.ndarray:
        """Generate sitting position"""
        q_sit = np.zeros(12)
        q_sit[1::3] += np.pi / 2  # Hip joints
        q_sit[2::3] -= np.pi - 0.2  # Knee joints
        return q_sit
    
    def start_server(self):
        """Start socket server"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)
        self.server_socket.settimeout(1.0)  # Non-blocking with timeout
        print(f"[SERVER] Listening on {self.host}:{self.port}")
        
    def handle_client(self):
        """Handle incoming client connections and commands"""
        try:
            if self.client_socket is None:
                try:
                    self.client_socket, addr = self.server_socket.accept()
                    self.client_socket.settimeout(0.1)
                    print(f"[SERVER] Client connected from {addr}")
                    self.send_response({"status": "connected", "message": "Go2 Simulation Server Ready"})
                except socket.timeout:
                    return
                    
            if self.client_socket:
                try:
                    data = self.client_socket.recv(1024)
                    if data:
                        command = json.loads(data.decode('utf-8'))
                        self.process_command(command)
                    else:
                        # Client disconnected
                        print("[SERVER] Client disconnected")
                        self.client_socket.close()
                        self.client_socket = None
                except socket.timeout:
                    pass
                except ConnectionResetError:
                    print("[SERVER] Client connection reset")
                    self.client_socket.close()
                    self.client_socket = None
        except Exception as e:
            print(f"[SERVER] Error handling client: {e}")
            
    def send_response(self, response: Dict[str, Any]):
        """Send response to client"""
        if self.client_socket:
            try:
                msg = json.dumps(response).encode('utf-8')
                self.client_socket.send(msg)
            except Exception as e:
                print(f"[SERVER] Error sending response: {e}")
                
    def process_command(self, command: Dict[str, Any]):
        """Process incoming command"""
        cmd_type = command.get('type', '')
        
        print(f"[SERVER] Received command: {cmd_type}")
        
        with self.lock:
            if cmd_type == 'stand':
                self.execute_stand(command.get('speed', 0.5))
                
            elif cmd_type == 'sit':
                self.execute_sit(command.get('speed', 0.5))
                
            elif cmd_type == 'move_to':
                position = command.get('position', 'neutral')
                self.execute_move_to(position, command.get('speed', 0.5))
                
            elif cmd_type == 'custom_position':
                joints = np.array(command.get('joints', [0]*12))
                self.execute_custom_position(joints, command.get('speed', 0.5))
                
            elif cmd_type == 'stop':
                self.controller.active = False
                self.state = RobotState.IDLE
                self.send_response({"status": "success", "message": "Stopped"})
                
            elif cmd_type == 'reset':
                self.sim.reset()
                self.controller.active = False
                self.state = RobotState.IDLE
                self.send_response({"status": "success", "message": "Reset complete"})
                
            elif cmd_type == 'get_state':
                state_info = {
                    "robot_state": self.state.value,
                    "joint_positions": self.sim.getJointPositions().tolist(),
                    "joint_velocities": self.sim.getJointVelocities().tolist(),
                    "controller_active": self.controller.active,
                    "trajectory_progress": self.controller.alpha
                }
                self.send_response({"status": "success", "state": state_info})
                
            elif cmd_type == 'shutdown':
                self.running = False
                self.send_response({"status": "success", "message": "Shutting down"})
                
            else:
                self.send_response({"status": "error", "message": f"Unknown command: {cmd_type}"})
                
    def execute_stand(self, speed: float = 0.5):
        """Execute stand command"""
        current_pos = self.sim.getJointPositions()
        target_pos = self.positions['stand']
        self.controller.start_trajectory(current_pos, target_pos, speed)
        self.state = RobotState.STANDING
        self.send_response({"status": "success", "message": "Standing up"})
        
    def execute_sit(self, speed: float = 0.5):
        """Execute sit command"""
        current_pos = self.sim.getJointPositions()
        target_pos = self.positions['sit']
        self.controller.start_trajectory(current_pos, target_pos, speed)
        self.state = RobotState.SITTING
        self.send_response({"status": "success", "message": "Sitting down"})
        
    def execute_move_to(self, position: str, speed: float = 0.5):
        """Move to a predefined position"""
        if position in self.positions:
            current_pos = self.sim.getJointPositions()
            target_pos = self.positions[position]
            self.controller.start_trajectory(current_pos, target_pos, speed)
            self.send_response({"status": "success", "message": f"Moving to {position}"})
        else:
            self.send_response({"status": "error", "message": f"Unknown position: {position}"})
            
    def execute_custom_position(self, joints: np.ndarray, speed: float = 0.5):
        """Move to custom joint positions"""
        if len(joints) == 12:
            current_pos = self.sim.getJointPositions()
            self.controller.start_trajectory(current_pos, joints, speed)
            self.send_response({"status": "success", "message": "Moving to custom position"})
        else:
            self.send_response({"status": "error", "message": "Invalid joint array (need 12 values)"})
            
    def simulation_loop(self):
        """Main simulation loop"""
        with viewer.launch_passive(self.sim.model, self.sim.data) as v:
            # Configure camera
            v.cam.azimuth = 135
            v.cam.elevation = -20
            v.cam.distance = 3
            v.cam.lookat[:] = [0, 0, 0.3]
            
            print("[SERVER] Simulation started")
            
            while v.is_running() and self.running:
                # Handle network commands
                self.handle_client()
                
                # Update control
                with self.lock:
                    q = self.sim.getJointPositions()
                    qd = self.sim.getJointVelocities()
                    control = self.controller.compute_control(q, qd, self.sim.dt)
                    self.sim.inputControl(control)
                
                # Step simulation
                self.sim.nextStep()
                v.sync()
                
    def run(self):
        """Run the server"""
        try:
            self.start_server()
            self.simulation_loop()
        finally:
            if self.client_socket:
                self.client_socket.close()
            if self.server_socket:
                self.server_socket.close()
            print("[SERVER] Shutdown complete")


def main():
    # Update this path to your robot model
    xml_path = '/Users/jaeykim/mpc/ai_trot_gait/unitree_robotics_a1/scene2.xml'
    
    server = Go2SimulationServer(xml_path)
    server.run()


if __name__ == "__main__":
    main()