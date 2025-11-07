import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("pendulum.xml")
data = mujoco.MjData(model)

with mujoco.viewer.launch(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
