import os
import sys
# Get current working directory
cwd = os.getcwd()

# Only go up one level if we're currently inside 'scripts'
if cwd.endswith("scripts"):
    os.chdir("..")
print("Current working directory:", os.getcwd())

# Add current directory to Python path
sys.path.insert(0, os.getcwd())

import yaml
import numpy as np
import matplotlib.pyplot as plt
from models.simple_active_hopper_spring import SpringLoadedHopper, PDController

# ============================================================
#                    Load Configuration
# ============================================================
with open("cfg/simple_active_hopper_spring.yaml", "r") as f:
    cfg = yaml.safe_load(f)

m_b = cfg["body_mass"]
m_l = cfg["leg_mass"]
l0 = cfg["spring_length"]
k = cfg["spring_constant"]
g = cfg["gravity"]
x0 = np.array(cfg["x0"])

# PD Controller parameters
pd_cfg = cfg["pd"]
kp = pd_cfg["kp"]
kd = pd_cfg["kd"]

# ============================================================
#                    Simulation Parameters
# ============================================================
sim_cfg = cfg["simulation"]
dt_sim = float(sim_cfg["dt"])
t_max = float(sim_cfg["t_max"])
N_steps = int(t_max / dt_sim)
t = np.linspace(0, t_max, N_steps)

# Initial state
state_current = "stance" if x0[2] <= 0 else "flight"

# ============================================================
#                    Create Hopper and Controller
# ============================================================
print("="*60)
print("PD Controller Simulation - Spring-Loaded Hopper")
print("="*60)
print(f"Body mass: {m_b} kg, Leg mass: {m_l} kg")
print(f"Spring constant: {k} N/m, Spring rest length: {l0} m")
print(f"PD Controller: kp={kp}, kd={kd}")
print("\n=== PD CONTROL LIMITATIONS DEMONSTRATION ===")
print("PD controller controls height directly, ignoring spring dynamics.")
print("This leads to:")
print("  - Poor energy management (doesn't exploit spring)")
print("  - Inefficient hopping (fighting against spring)")
print("  - Difficulty coordinating leg for takeoff/landing")
print("  - Oscillations and overshoot")
print("  - Requires hacks (extra downward force) to work")
print("="*60 + "\n")

# Create hopper model and PD controller
hopper = SpringLoadedHopper(m_body=m_b, m_leg=m_l, l0=l0, k=k, g=g)
controller = PDController(kp=kp, kd=kd)

# ============================================================
#                    Simulation Setup
# ============================================================
# Storage arrays
x = np.zeros((N_steps, 4))  # State history
x[0] = x0
state_arr = np.zeros(N_steps)  # Mode history: 0=stance, 1=flight
state_arr[0] = 1 if state_current == "flight" else 0

# Track metrics to demonstrate limitations
height_ref_hist = []
height_error_hist = []
control_effort_hist = []
spring_energy_hist = []

# ============================================================
#                    Simulation Loop
# ============================================================
for i in range(1, N_steps):
    t_curr = t[i]
    
    # Height reference profile - set target height for hopping
    height_ref = 0.6  # Target apex height
    
    # During stance, we need to compress spring to store energy
    # During flight, we want to reach target height
    if state_current == "stance":
        # In stance: need to compress spring first, then let it expand
        l_current = x[i-1][0] - x[i-1][2]  # current leg length
        
        if l_current < l0:  # Spring is compressed - let it expand
            # Spring is compressed, let it push up naturally
            # Use PD to guide but don't fight the spring
            height_dot_ref = 0.0
            u = controller.compute(x[i-1], height_ref, height_dot_ref)
            # Reduce control to let spring do the work
            u = u * 0.3
        else:  # Spring not compressed enough - push down to compress
            # Push DOWN to compress spring (this is the hack!)
            # PD wants to push UP, but we need DOWN to compress
            height_dot_ref = 0.0
            u_pd = controller.compute(x[i-1], height_ref, height_dot_ref)
            # Override PD: push down to compress spring
            u = -80  # Strong downward force to compress spring
    else:  # flight
        # In flight: try to reach target height
        height_dot_ref = 0.0
        u = controller.compute(x[i-1], height_ref, height_dot_ref)
    
    # Store metrics
    height_ref_hist.append(height_ref)
    height_error_hist.append(height_ref - x[i-1][0])
    control_effort_hist.append(u**2)
    
    # Spring energy: 0.5 * k * (l0 - l)^2
    l = x[i-1][0] - x[i-1][2]
    spring_compression = max(0, l0 - l)
    spring_energy_hist.append(0.5 * k * spring_compression**2)

    # ============================================================
    # Forward Simulation
    # ============================================================
    if state_current == "flight":
        x_dot, F_sub = hopper.flight_state(x[i-1], u)
        x[i] = x[i-1] + x_dot * dt_sim

        # Touchdown: leg hits ground
        if x[i-1][2] > 0 and x[i][2] <= 0 and x[i-1][3] < 0:
            x[i][2] = 0.0
            x[i][3] = 0.0
            state_current = "stance"

    elif state_current == "stance":
        result = hopper.stance_state(x[i-1], u)
        x_dot = result[0]
        F_sub = result[1] if len(result) >= 2 else 0.0
        x[i] = x[i-1] + x_dot * dt_sim
        
        # Ensure foot stays on ground during stance
        if x[i][2] < 0:
            x[i][2] = 0.0
            x[i][3] = 0.0

        # Lift-off conditions:
        # 1. Body height exceeds spring rest length (spring extended)
        # 2. Body is moving upward with sufficient velocity
        # 3. Ground reaction force is zero or negative (leg can lift)
        l_current = x[i][0] - x[i][2]  # current leg length
        body_moving_up = x[i][1] > 0.05  # body has upward velocity
        
        # Lift off when spring extends beyond rest length and body is moving up
        # OR when ground reaction becomes zero/negative (leg unloads)
        if (l_current > l0 and body_moving_up) or (F_sub <= 0 and x[i][1] > 0):
            state_current = "flight"
            # Give leg initial upward velocity when lifting off
            x[i][3] = x[i][1] * 0.5  # leg follows body with some velocity

    state_arr[i] = 1 if state_current == "flight" else 0

# ============================================================
#                    Results and Visualization
# ============================================================
print("\n" + "="*60)
print("Simulation Complete!")
print("="*60)
print(f"Mean absolute height error: {np.mean(np.abs(height_error_hist)):.4f} m")
print(f"RMS height error: {np.sqrt(np.mean(np.array(height_error_hist)**2)):.4f} m")
print(f"Total control effort: {np.sum(control_effort_hist):.2f}")
print(f"Mean spring energy: {np.mean(spring_energy_hist):.4f} J")
print(f"Max body height: {np.max(x[:, 0]):.4f} m")
print(f"Number of hops (state transitions): {int(np.sum(np.diff(state_arr) != 0) / 2)}")
print(f"\nNote: PD controller requires hacks (extra downward force) to make hopping work!")
print("      A proper controller would coordinate leg and exploit spring naturally.")
print("="*60 + "\n")

# --- Plotting ---
fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

# Plot 1: Trajectories and height reference
axs[0].plot(t, x[:, 0], label='Body Height $x_b$', linewidth=2, color='blue')
axs[0].plot(t[1:], height_ref_hist, '--', label='Height Reference', linewidth=1.5, alpha=0.7, color='red')
axs[0].plot(t, x[:, 2], label='Leg Position $x_l$', alpha=0.6, color='orange')
axs[0].axhline(l0, linestyle=':', color='gray', label=f'Spring Rest Length $l_0={l0}$', alpha=0.5)
axs[0].axhline(0, linestyle='-', color='brown', label='Ground', linewidth=1, alpha=0.5)
axs[0].set_ylabel('Height (m)')
axs[0].set_title('PD Height Control - Hopping Behavior (with Limitations)', fontsize=12, fontweight='bold')
axs[0].legend(loc='upper right')
axs[0].grid(True, alpha=0.3)
axs[0].set_ylim([-0.1, max(1.0, np.max(x[:, 0]) * 1.1)])

# Plot 2: Height tracking error (demonstrates poor tracking)
axs[1].plot(t[1:], height_error_hist, label='Height Error', color='red', linewidth=1.5)
axs[1].axhline(0, linestyle='--', color='black', alpha=0.3)
axs[1].set_ylabel('Tracking Error (m)')
axs[1].set_title('PD Control Limitation: Poor Reference Tracking', fontsize=10)
axs[1].legend()
axs[1].grid(True, alpha=0.3)

# Plot 3: Control effort (demonstrates inefficiency)
axs[2].plot(t[1:], control_effort_hist, label='Control Effort $u^2$', color='orange', linewidth=1.5)
axs[2].set_ylabel('Control Effort')
axs[2].set_title('PD Control Limitation: High Control Effort (Fighting Spring)', fontsize=10)
axs[2].legend()
axs[2].grid(True, alpha=0.3)

# Plot 4: State transitions and spring energy
ax2_twin = axs[3].twinx()
axs[3].plot(t, state_arr, color='tab:red', drawstyle='steps-post', label='State', linewidth=2)
ax2_twin.plot(t[1:], spring_energy_hist, color='green', label='Spring Energy', linewidth=1.5, alpha=0.7)
axs[3].set_xlabel('Time (s)')
axs[3].set_ylabel('State', color='tab:red')
ax2_twin.set_ylabel('Spring Energy (J)', color='green')
axs[3].set_yticks([0, 1])
axs[3].set_yticklabels(['Stance', 'Flight'])
axs[3].tick_params(axis='y', labelcolor='tab:red')
ax2_twin.tick_params(axis='y', labelcolor='green')
axs[3].set_title('State Transitions & Spring Energy (PD Cannot Exploit Spring Efficiently)', fontsize=10)
axs[3].grid(True, alpha=0.3)

# Add text annotation about limitations
fig.text(0.02, 0.02, 
         'PD Control Limitations:\n'
         '• Ignores spring dynamics → inefficient energy use\n'
         '• No feedforward → poor tracking during transitions\n'
         '• Cannot coordinate leg for optimal takeoff/landing\n'
         '• High control effort due to fighting spring forces',
         fontsize=9, verticalalignment='bottom',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
plt.show()

print("\n✅ Compare with NLP controller to see the performance difference!")
