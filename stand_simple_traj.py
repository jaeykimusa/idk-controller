import mujoco
from mujoco import viewer
import os
import numpy as np
from scipy.spatial.transform import Rotation as R
import time

def loadRobot() -> mujoco.MjModel:
    # model_path = os.path.join(os.path.dirname(__file__), "go2.xml")
    xml_path = '/Users/jaeykim/idk-controller/description/scene.xml'
    return mujoco.MjModel.from_xml_path(xml_path)

def loadData(robot: mujoco.MjModel) -> mujoco.MjData:
    return mujoco.MjData(robot)


class Mujoco:
    def __init__(self):
        self.model = loadRobot()
        self.data = loadData(self.model)
        self.dt = self.model.opt.timestep  # 0.002

    def loadRobot(self) -> mujoco.MjModel:
        # model_path = os.path.join(os.path.dirname(__file__), "go2.xml")
        xml_path = '/Users/jaeykim/mpc/ai_trot_gait/unitree_robotics_a1/scene2.xml'
        self.model = mujoco.MjModel.from_xml_path(xml_path)
    
    def loadData(self, robot: mujoco.MjModel) -> mujoco.MjData:
        self.data = mujoco.MjData(robot)

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
    
    def getQDD(self) -> np.ndarray:
        return self.data.qacc
    
    def getTau(self) -> np.ndarray:
        return self.data.ctrl
    
    def getJointPositions(self) -> np.ndarray:
        return self.data.qpos[-12:]
    
    def getJointVelocities(self) -> np.ndarray:
        return self.data.qvel[-12:]
    
    def setJointPositions(self, joint_positions: np.ndarray):
        """Set the last 12 joint positions"""
        self.data.qpos[-12:] = joint_positions


class Go2Controller:
    def __init__(
            self,
            q0: np.ndarray,
            qd0: np.ndarray,
            qdd0: np.ndarray,
            tau0: np.ndarray,
            kp: float,
            kd: float,
            alpha_speed: float = 1.0,  # Speed of interpolation
        ):
        self.q = q0
        self.qd = qd0
        self.qdd = qdd0
        self.tau = tau0
        self.kp = kp
        self.kd = kd
        
        # Trajectory generation parameters
        self.alpha = 0.0  # Interpolation parameter (0 to 1)
        self.alpha_speed = alpha_speed
        
        # Set initial position (crouched/folded position)
        self.q_initial = self.get_initial_position(q0)
        
        # Set desired position (standing position from the model's default)
        # You can customize this to your desired standing position
        self.q_desired = np.array([0, 0.75, -1.5] * 4)  # Your original desired position
        
    def get_initial_position(self, q0: np.ndarray) -> np.ndarray:
        """Generate initial crouched position similar to the first code"""
        q_initial = q0[-12:].copy()
        
        # Modify every 3rd joint starting from index 1 (hip joints)
        # This corresponds to: q_initial[-12 + 1 :: 3] += np.pi / 2
        q_initial[1::3] += np.pi / 2
        
        # Modify every 3rd joint starting from index 2 (knee joints)  
        # This corresponds to: q_initial[-12 + 2 :: 3] -= np.pi - 0.2
        q_initial[2::3] -= np.pi - 0.2
        
        return q_initial
    
    def update_alpha(self, dt: float) -> None:
        """Update the interpolation parameter alpha"""
        if self.alpha < 1.0:
            self.alpha = min(self.alpha + self.alpha_speed * dt, 1.0)
    
    def interpolate(self, from_: np.ndarray, to_: np.ndarray, alpha: float) -> np.ndarray:
        """Linear interpolation between two arrays"""
        assert 0 <= alpha <= 1, "alpha must be between 0 and 1"
        return from_ * (1 - alpha) + to_ * alpha
    
    def update_state(self, q: np.ndarray, qd: np.ndarray, qdd: np.ndarray, tau: np.ndarray):
        self.q = q
        self.qd = qd
        self.qdd = qdd
        self.tau = tau

    def getJointPDControlWithTrajectory(self, dt: float) -> np.ndarray:
        """Get PD control with trajectory generation"""
        # Update interpolation parameter
        self.update_alpha(dt)
        
        # Get current joint positions and velocities
        q_current = self.q[-12:]
        qd_current = self.qd[-12:]
        
        # Interpolate between initial and desired positions
        q_target = self.interpolate(
            from_=self.q_initial,
            to_=self.q_desired,
            alpha=self.alpha
        )
        
        # Desired velocity is zero (we want smooth motion)
        qd_target = np.zeros_like(qd_current)
        
        # Compute PD control
        tau = self.kp * (q_target - q_current) + self.kd * (qd_target - qd_current)
        
        return tau
    
    def getJointPDControl(self) -> np.ndarray:
        """Original PD control without trajectory generation"""
        q = self.q[-12:]
        qd = self.qd[-12:]
        q_desired = np.array([0, 0.9, -1.8] * 4)
        qd_desired = np.zeros_like(q[-12:])
        tau = self.kp * (q_desired - q) + self.kd * (qd_desired - qd)
        return tau
    
    def is_trajectory_complete(self) -> bool:
        """Check if the trajectory interpolation is complete"""
        return self.alpha >= 1.0


def main():
    sim = Mujoco()
    
    # Get initial state
    q0 = sim.getQ()
    qd0 = sim.getQD()
    qdd0 = sim.getQDD()
    tau0 = sim.getTau()
    
    # Create controller with trajectory generation
    ctrl = Go2Controller(
        q0=q0,
        qd0=qd0,
        qdd0=qdd0,
        tau0=tau0,
        kp=30,
        kd=0.5,
        alpha_speed=0.25 # Adjust this to control how fast the robot stands up
    )
    
    # Set initial crouched position
    initial_joint_positions = ctrl.q_initial
    sim.setJointPositions(initial_joint_positions)

    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        
        # Default configure camera
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -20
        viewer.cam.distance = 3
        viewer.cam.lookat[:] = [0, 0, 0.3]
        
        time.sleep(1)  # Wait a moment to see the initial position
        # Run simulation
        while viewer.is_running():
            # Update controller state
            q = sim.getQ()
            qd = sim.getQD()
            qdd = sim.getQDD()
            tau = sim.getTau()
            ctrl.update_state(q, qd, qdd, tau)
            
            # Get control with trajectory generation
            control = ctrl.getJointPDControlWithTrajectory(sim.dt)
            sim.inputControl(control)
            
            # Step simulation
            sim.nextStep()
            viewer.sync()
            
            # Optional: Print progress
            if not ctrl.is_trajectory_complete():
                print(f"Trajectory progress: {ctrl.alpha:.2%}")


if __name__ == "__main__":
    main()