

import os

import numpy as np
import pinocchio as pin

from pathlib import Path


# Get path relative to your script
REPO_ROOT = Path(__file__).parent.parent
ROBOT_FILES_PATH = REPO_ROOT / "idk-controller" / "robot_description"
# Load URDF
urdf_path = ROBOT_FILES_PATH / "go2" / "go2.urdf"

# Load the model
model = pin.buildModelFromUrdf(str(urdf_path))
data = model.createData()

# The model will have a freeflyer root joint (quaternion-based)
# Configuration: [x, y, z, qx, qy, qz, qw, joint_angles...]
print(f"Model DOF: {model.nq}")
print(f"Velocity DOF: {model.nv}")
print(f"Joint names: {[name for name in model.names]}")

exit()

q0 = pin.neutral(model)

print(q0)

for name, value in zip(model.names[1:], q0):
    print(f"{name}: {value}")