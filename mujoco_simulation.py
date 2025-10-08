import mujoco 
import numpy as np
from scipy.spatial.transform import Rotation as R


GO2_XML_PATH = '/Users/jaeykim/mpc/ai_trot_gait/unitree_robotics_a1/scene2.xml'

class Mujoco:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(GO2_XML_PATH)
        self.data = mujoco.MjData(self.model)
        self.dt = self.model.opt.timestep  # 0.002

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
    
    def getU(self) -> np.ndarray:
        return self.data.ctrl
    
    def getState(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        q = self.getQ()
        qd = self.getQD()
        qdd = self.getQDD()
        u = self.getU()
        return q, qd, qdd, u

    def getJointPositions(self) -> np.ndarray:
        return self.data.qpos[-12:]
    
    def getJointVelocities(self) -> np.ndarray:
        return self.data.qvel[-12:]
    
    def setJointPositions(self, joint_positions: np.ndarray):
        """Set the last 12 joint positions"""
        self.data.qpos[-12:] = joint_positions

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)

    def getJacobian(self, body_name: str, point: np.ndarray = None):
        """
        Compute the Jacobian (position and rotation) of a point on a body.
        
        Args:
            body_name: Name of the body in the MJCF model (e.g., "FR_foot").
            point: 3D point in world coordinates (defaults to body origin).

        Returns:
            jacp (3, nv), jacr (3, nv)
        """
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)

        if body_id < 0:
            raise ValueError(f"Body '{body_name}' not found in model.")

        # If point is not given, use body origin (world position)
        if point is None:
            xpos = self.data.xpos[body_id]   # (3,)
            point = np.copy(xpos)

        nv = self.model.nv
        jacp = np.zeros((3, nv))
        jacr = np.zeros((3, nv))

        mujoco.mj_jac(self.model, self.data, jacp, jacr, point, body_id)

        return jacp, jacr