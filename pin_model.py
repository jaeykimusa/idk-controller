import pinocchio as pin
import numpy as np

# model = pin.buildModelFromUrdf("/Users/jaeykim/idk-controller/description/go2.urdf")

model = pin.

data = model.createData()

print(model)

q0 = pin.neutral(model)

print(q0)

for name, value in zip(model.names[1:], q0):
    print(f"{name}: {value}")