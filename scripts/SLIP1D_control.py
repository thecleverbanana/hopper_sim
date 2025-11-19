"""
NLP Control for 1D SLIP Model using IPOPT

Demonstrates optimal control of a 1D spring-loaded inverted pendulum hopper
using Model Predictive Control (MPC) with IPOPT solver.
"""

import os
import sys

# Handle path setup
cwd = os.getcwd()
if cwd.endswith("scripts"):
    os.chdir("..")
sys.path.insert(0, os.getcwd())

import numpy as np
import matplotlib.pyplot as plt
from models.SLIP1D import SLIP1D, NLPController1D

# ============================================================
#                    Model Parameters
# ============================================================
m = 1.0          # Body mass (kg)
r0 = 0.3         # Nominal leg length (m)
k = 200.0        # Spring stiffness (N/m)
g = 9.81         # Gravity (m/s^2)

# ============================================================
#                    MPC/NLP Parameters
# ============================================================
H = 30           # Prediction horizon
dt_mpc = 0.02    # MPC time step (s)
dt_sim = 0.001   # Simulation time step (s)
t_max = 3.0      # Simulation duration (s)

# Control bounds
u_min = -800.0
u_max = 800.0

# Cost weights (tuned for reaching target height ~0.5m)
R_u = 2e-5       # Control effort weight (small to allow control)
Q_z = 400.0      # Height tracking weight (running) - high
Q_v = 4.0        # Velocity weight (running)
Q_zT = 1000.0    # Height tracking weight (terminal) - high
Q_vT = 12.0      # Velocity weight (terminal)
z_target = 0.5   # Target height (m)

# ============================================================
#                    Simulation Setup
# ============================================================
N_steps = int(t_max / dt_sim)
t = np.linspace(0, t_max, N_steps)
control_update_freq = int(dt_mpc / dt_sim)

# Initial state: [z, z_dot]
z0 = r0 * 0.85   # Start compressed (allows spring to push up)
z_dot0 = 0.0     # Start at rest (controller will provide energy)
x0 = np.array([z0, z_dot0])

print("="*60)
print("1D SLIP Model - NLP/MPC Control (IPOPT)")
print("="*60)
print(f"Model: m={m} kg, r0={r0} m, k={k} N/m")
print(f"MPC: H={H}, dt_mpc={dt_mpc} s, dt_sim={dt_sim} s")
print(f"Cost weights: Q_z={Q_z}, Q_v={Q_v}, Q_zT={Q_zT}, Q_vT={Q_vT}, R_u={R_u}")
print(f"Target height: {z_target} m")
print(f"Initial state: z={z0:.3f} m, z_dot={z_dot0:.3f} m/s")
print("="*60 + "\n")

# Initialize model
slip = SLIP1D(mass=m, leg_length=r0, k=k, g=g, verbose=True)

# Determine initial mode
current_mode = "stance" if x0[0] < r0 else "flight"

# Initial mode sequence (will be updated during simulation)
init_mode_seq = ["stance"] * H if current_mode == "stance" else ["flight"] * H

# Initialize NLP controller
nlp_controller = NLPController1D(
    slip_model=slip,
    H=H,
    dt=dt_mpc,
    x0=x0,
    mode_seq=init_mode_seq,
    u_min=u_min,
    u_max=u_max,
    R_u=R_u,
    Q_z=Q_z,
    Q_v=Q_v,
    Q_zT=Q_zT,
    Q_vT=Q_vT,
    z_target=z_target,
)

# Storage arrays
x = np.zeros((N_steps, 2))  # State history
u = np.zeros(N_steps)       # Control history
mode = np.zeros(N_steps)    # Mode: 0=stance, 1=flight
x[0] = x0
mode[0] = 0 if current_mode == "stance" else 1

# Warm start for MPC
warm_start = None
u_current = 0.0

# Simulation loop
for i in range(1, N_steps):
    state = x[i-1]
    z, z_dot = state
    
    # ============================================================
    # MPC Update (only at control update frequency)
    # ============================================================
    if i % control_update_freq == 0:
        # Build mode sequence based on current state
        if current_mode == "stance":
            # Estimate stance duration (roughly based on spring period)
            stance_steps = int(min(12, H))
            mode_seq = ["stance"] * stance_steps + ["flight"] * int(H - stance_steps)
        else:  # flight
            # Estimate flight duration
            flight_estimate = int(min(8, H // 2))
            mode_seq = ["flight"] * flight_estimate + ["stance"] * int(H - flight_estimate)
        
        nlp_controller.update_mode_sequence(mode_seq)
        
        # Solve MPC
        try:
            u_current, warm_start = nlp_controller.compute(
                state, warm_start=warm_start, return_warm_start=True
            )
        except Exception as e:
            print(f"Warning: MPC solve failed at step {i}: {e}")
            # Use previous control or zero
            u_current = u_current if i > control_update_freq else 0.0
    
    u[i] = u_current
    
    # ============================================================
    # Forward Simulation
    # ============================================================
    if current_mode == "stance":
        # Stance phase: spring-loaded dynamics
        x_dot = slip.stance_dynamics(0, state, u[i])
        x[i] = state + x_dot * dt_sim
        
        # Check for take-off: leg extends beyond rest length
        if x[i][0] >= r0 and x[i][1] > 0:
            current_mode = "flight"
            x[i][0] = r0  # Set exactly at rest length for take-off
        
    else:  # flight
        # Flight phase: ballistic motion (no control in flight)
        x_dot = slip.flight_dynamics(0, state)
        x[i] = state + x_dot * dt_sim
        
        # Check for touch-down: body reaches rest length height
        if x[i][0] <= r0 and x[i][1] < 0:
            current_mode = "stance"
            x[i][0] = r0  # Set exactly at rest length for touch-down
    
    mode[i] = 0 if current_mode == "stance" else 1
    
    # Safety: prevent negative height
    if x[i][0] < 0:
        x[i][0] = 0.0
        x[i][1] = 0.0
        current_mode = "stance"
        mode[i] = 0

# ============================================================
#                    Plotting
# ============================================================
fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

# Plot 1: Height trajectory
axs[0].plot(t, x[:, 0], label='Body Height $z$', linewidth=2, color='blue')
axs[0].axhline(z_target, linestyle='--', label=f'Target Height $z_{{ref}}={z_target}$ m', 
               linewidth=1.5, color='red', alpha=0.7)
axs[0].axhline(r0, linestyle=':', label=f'Leg Rest Length $r_0={r0}$ m', 
               linewidth=1, color='gray', alpha=0.5)
axs[0].axhline(0, linestyle='-', label='Ground', linewidth=1, color='brown', alpha=0.5)
axs[0].set_ylabel('Height (m)')
axs[0].set_title('1D SLIP Model - NLP/MPC Control (IPOPT)', fontsize=12, fontweight='bold')
axs[0].legend(loc='upper right')
axs[0].grid(True, alpha=0.3)
axs[0].set_ylim([-0.05, max(0.7, np.max(x[:, 0]) * 1.1)])

# Plot 2: Velocity
axs[1].plot(t, x[:, 1], label='Vertical Velocity $\dot{z}$', linewidth=2, color='green')
axs[1].axhline(0, linestyle='--', color='black', alpha=0.3)
axs[1].set_ylabel('Velocity (m/s)')
axs[1].set_title('Vertical Velocity', fontsize=10)
axs[1].legend()
axs[1].grid(True, alpha=0.3)

# Plot 3: Control input
axs[2].plot(t, u, label='Control Input $u$', linewidth=1.5, color='orange')
axs[2].axhline(0, linestyle='--', color='black', alpha=0.3)
axs[2].set_ylabel('Force (N)')
axs[2].set_title('Control Input (NLP/MPC)', fontsize=10)
axs[2].legend()
axs[2].grid(True, alpha=0.3)

# Plot 4: Mode (stance/flight)
axs[3].plot(t, mode, color='tab:red', drawstyle='steps-post', 
            label='Mode', linewidth=2)
axs[3].set_xlabel('Time (s)')
axs[3].set_ylabel('Mode')
axs[3].set_yticks([0, 1])
axs[3].set_yticklabels(['Stance', 'Flight'])
axs[3].set_title('Phase Transitions', fontsize=10)
axs[3].legend()
axs[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================
#                    Performance Metrics
# ============================================================
print("\n=== SIMULATION RESULTS ===")
print(f"Final height: {x[-1, 0]:.4f} m")
print(f"Target height: {z_target:.4f} m")
print(f"Height error: {abs(x[-1, 0] - z_target):.4f} m")
print(f"Max height reached: {np.max(x[:, 0]):.4f} m")
print(f"Min height reached: {np.min(x[:, 0]):.4f} m")
print(f"Number of hops: {int(np.sum(np.diff(mode) != 0) / 2)}")
print(f"RMS control effort: {np.sqrt(np.mean(u**2)):.2f} N")
print(f"Max control input: {np.max(np.abs(u)):.2f} N")
print(f"Mean absolute height error: {np.mean(np.abs(x[:, 0] - z_target)):.4f} m")
print(f"\nNLP/MPC Advantages:")
print(f"  - Optimal control strategy (exploits spring dynamics)")
print(f"  - Predictive planning (H={H} steps ahead)")
print(f"  - Better energy efficiency")
print(f"  - Coordinated stance/flight transitions")
