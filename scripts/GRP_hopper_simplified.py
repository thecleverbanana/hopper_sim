import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from models.GRP_hopper_simplified import simplified_GRP_hopper


# ========================================================
#                  Forward Simulation
# ========================================================
def simulate_step(hopper, X, u, mode, dt):
    if mode == "flight":
        dx,_ = hopper.flight_state(X, u)
        Xn = X + dx * dt

        # foot above ground
        if Xn[2] < 0.0:
            Xn[2] = 0.0
            Xn[3] = 0.0

    else:
        dx,_ = hopper.stance_state(X, u)
        Xn = X + dx * dt

        Xn[2] = 0.0
        Xn[3] = 0.0

        if Xn[0] < 0.0:
            Xn[0] = 0.0
            Xn[1] = max(0.0, Xn[1])

    return Xn


# ========================================================
#                    Create Hopper + MPC
# ========================================================
hopper = simplified_GRP_hopper(mb=1.0, mf=0.2, l0=0.3)

H  = 25
dt = 0.02

x_current = np.array([0.30, 0.0, 0.0, 0.0])

init_mode_seq = ["stance"] * H

mpc = simplified_GRP_hopper.NLPController(
    hopper=hopper,
    H=H,
    dt=dt,
    x0=x_current,
    mode_seq=init_mode_seq,
    u_min=-50.0, u_max=50.0,
    R_u=1e-2,
    Q_l=20.0,
    Q_b_mid=5.0,
    min_body_height=0.1,
    Q_bh=50.0,
    Q_bv=20.0,
    Q_fh=5.0,           # new: encourage foot to lift
    Q_fv=2.0,
    body_ref=0.35,
    stance_steps=8,
)

# ========================================================
#                    Simulation Loop
# ========================================================
N_steps = 400
warm = None
state_current = "stance"

xs_b, xs_f, us, modes = [], [], [], []

for t in range(N_steps):
    # 1) mode predicted by MPC
    mode_seq = mpc.predict_mode_sequence(x_current, state_current)
    mpc.update_mode_sequence(mode_seq)

    # 2) solve MPC
    u, warm = mpc.compute(x_current, warm_start=warm, return_warm_start=True)

    # 3) update mode: use MPC prediction (key fix)
    state_current = mode_seq[0]

    # 4) forward simulate
    x_current = simulate_step(hopper, x_current, u, state_current, dt)

    xs_b.append(x_current[0])
    xs_f.append(x_current[2])
    us.append(u)
    modes.append(1 if state_current=="flight" else 0)


# ========================================================
#                         Plot
# ========================================================
time = np.arange(N_steps)*dt

plt.figure(figsize=(10,6))
plt.subplot(3,1,1)
plt.plot(time, xs_b, label="body height")
plt.plot(time, xs_f, label="foot height")
plt.legend()
plt.title("Hopper MPC: Height")

plt.subplot(3,1,2)
plt.plot(time, us)
plt.title("Control Force")

plt.subplot(3,1,3)
plt.plot(time, modes)
plt.title("Mode (0=stance,1=flight)")

plt.tight_layout()
plt.show()
