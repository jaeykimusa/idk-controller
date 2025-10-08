# pmodel.py
# Unitree Go2 robot model for Pinocchio

from pathlib import Path
import numpy as np
import pinocchio as pin


# === Configuration ===
REPO_ROOT = Path(__file__).resolve().parent.parent
URDF_PATH = REPO_ROOT / "idk-controller" / "robot_description" / "go2" / "go2.urdf"

# Default "neutral" configuration for Unitree Go2 (custom)
DEFAULT_Q0 = np.array([
    0.0, 0.0, 0.3,       # base position
    0.0, 0.0, 0.0,       # base orientation (rpy)
    0.0, 0.8, -1.6,      # FR leg
    0.0, 0.8, -1.6,      # FL leg
    0.0, 0.8, -1.6,      # RR leg
    0.0, 0.8, -1.6,      # RL leg
])


class PinocchioModel:
    """Wrapper class for initializing and managing a Pinocchio robot model."""

    def __init__(self, q0: np.ndarray | None = None):
        """Initialize the robot model and data.

        Args:
            q0: Optional initial joint configuration (18×1 or 18, numpy array).
        """
        self.model, self.data = self._load_model()
        self.q0 = self._initialize_q(q0)

    # --- Private methods -----------------------------------------------------

    def _load_model(self):
        """Load robot model from URDF and create its data container."""
        root_joint = pin.JointModelComposite(2)
        root_joint.addJoint(pin.JointModelTranslation())
        root_joint.addJoint(pin.JointModelSphericalZYX())

        model = pin.buildModelFromUrdf(str(URDF_PATH), root_joint)
        data = model.createData()
        return model, data

    def _initialize_q(self, q0: np.ndarray | None):
        """Validate and set initial configuration q0."""
        if q0 is None:
            q = DEFAULT_Q0.copy()
        else:
            if not isinstance(q0, np.ndarray):
                raise TypeError("Parameter 'q0' must be a NumPy array.")
            if q0.shape not in [(18,), (18, 1)]:
                raise ValueError(f"Parameter 'q0' must have shape (18,) or (18,1), got {q0.shape}.")
            q = q0.reshape(18, 1)

        # Update kinematics
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        return q

    # --- Public utility ------------------------------------------------------

    def print_info(self):
        """Print basic robot information."""
        print(f"Model DOF (nq): {self.model.nq}")
        print(f"Velocity DOF (nv): {self.model.nv}")
        print("Joint names:", self.model.names)


# --- Standalone test --------------------------------------------------------
def main():
    robot = PinocchioModel()
    robot.print_info()
    print("Initial configuration q0:\n", robot.q0.T)



if __name__ == "__main__":
    main()
