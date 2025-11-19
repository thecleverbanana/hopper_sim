#!/usr/bin/env python3
"""
Comparison script: PD vs NLP Controller Performance
Runs both controllers and compares metrics
"""

import os
import sys
import numpy as np
import yaml

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.simple_active_hopper_spring import SpringLoadedHopper, PDController, NLPController


def run_pd_simulation(cfg, dt_sim, t_max):
    """Run PD controller simulation"""
    # Physical parameters
    mb = float(cfg["body_mass"])
    ml = float(cfg["leg_mass"])
    l0 = float(cfg["spring_length"])
    k = float(cfg["spring_constant"])
    g = float(cfg["gravity"])
    x0 = np.array(cfg["x0"])
    
    # PD parameters
    pd_cfg = cfg["pd"]
    kp = float(pd_cfg["kp"])
    kd = float(pd_cfg["kd"])
    
    # Initialize
    hopper = SpringLoadedHopper(mb, ml, l0, k, g)
    controller = PDController(kp=kp, kd=kd)
    
    # Simulation
    N = int(t_max / dt_sim)
    t = np.linspace(0, t_max, N)
    x = np.zeros((N, 4))
    x[0] = x0
    u_hist = np.zeros(N)
    height_error_hist = []
    control_effort_hist = []
    spring_energy_hist = []
    
    state_current = "stance"
    x_target = float(cfg["nlp"]["x_target"])
    
    for i in range(1, N):
        x_current = x[i-1].copy()
        height_ref = x_target
        height_dot_ref = 0.0
        
        # PD control
        u = controller.compute(x_current, height_ref, height_dot_ref)
        
        # Simulate
        if state_current == "stance":
            result = hopper.stance_state(x_current, u)
            x_dot = result[0]
            F_sub = result[1] if len(result) >= 2 else 0.0
        else:
            result = hopper.flight_state(x_current, u)
            x_dot = result[0]
            F_sub = 0.0
        
        x_next = x_current + dt_sim * np.array(x_dot)
        
        # Ground constraint
        if x_next[2] < 0:
            x_next[2] = 0.0
            x_next[3] = 0.0
        
        # Mode transition
        if state_current == "stance":
            l = x_next[0] - x_next[2]
            if l > l0 and x_next[1] > 0.05:
                state_current = "flight"
        else:
            if x_next[2] <= 0:
                state_current = "stance"
        
        x[i] = x_next
        
        # Metrics
        height_error_hist.append(height_ref - x_next[0])
        control_effort_hist.append(u**2)
        l = x_next[0] - x_next[2]
        spring_compression = max(0.0, l0 - l)
        spring_energy_hist.append(0.5 * k * spring_compression**2)
    
    return {
        'mean_height_error': np.mean(np.abs(height_error_hist)),
        'rms_height_error': np.sqrt(np.mean(np.array(height_error_hist)**2)),
        'total_control_effort': np.sum(control_effort_hist),
        'mean_spring_energy': np.mean(spring_energy_hist),
        'max_height': np.max(x[:, 0]),
        'height_error_hist': height_error_hist,
        'control_effort_hist': control_effort_hist,
    }


def run_nlp_simulation(cfg, dt_sim, t_max):
    """Run NLP controller simulation"""
    # Physical parameters
    mb = float(cfg["body_mass"])
    ml = float(cfg["leg_mass"])
    l0 = float(cfg["spring_length"])
    k = float(cfg["spring_constant"])
    g = float(cfg["gravity"])
    x0 = np.array(cfg["x0"])
    
    # NLP parameters
    nlp_cfg = cfg["nlp"]
    H = int(nlp_cfg["H"])
    dt_mpc = float(nlp_cfg["dt_control"])
    u_min = float(nlp_cfg["u_min"])
    u_max = float(nlp_cfg["u_max"])
    R_u = float(nlp_cfg["R_u"])
    Q_x = float(nlp_cfg["Q_x"])
    Q_v = float(nlp_cfg["Q_v"])
    Q_xT = float(nlp_cfg["Q_xT"])
    Q_vT = float(nlp_cfg["Q_vT"])
    x_target = float(nlp_cfg["x_target"])
    
    # Initialize
    hopper = SpringLoadedHopper(mb, ml, l0, k, g)
    init_mode_seq = ["stance"] * H
    nlp_controller = NLPController(
        hopper=hopper, H=H, dt=dt_mpc, x0=x0, mode_seq=init_mode_seq,
        u_min=u_min, u_max=u_max, R_u=R_u, Q_x=Q_x, Q_v=Q_v,
        Q_xT=Q_xT, Q_vT=Q_vT, x_target=x_target
    )
    
    # Simulation
    N = int(t_max / dt_sim)
    t = np.linspace(0, t_max, N)
    x = np.zeros((N, 4))
    x[0] = x0
    u_hist = np.zeros(N)
    height_error_hist = []
    control_effort_hist = []
    spring_energy_hist = []
    
    state_current = "stance"
    control_update_freq = int(dt_mpc / dt_sim)
    warm_start = None
    mpc_counter = 0
    
    for i in range(1, N):
        x_current = x[i-1].copy()
        t_current = t[i-1]
        
        # MPC update
        if i % control_update_freq == 0:
            # Predict mode sequence
            if state_current == "stance":
                stance_steps = int(min(15, H * 0.6))
                mode_seq = ["stance"] * stance_steps + ["flight"] * int(H - stance_steps)
            else:
                flight_estimate = int(min(12, H * 0.5))
                mode_seq = ["flight"] * flight_estimate + ["stance"] * int(H - flight_estimate)
            
            nlp_controller.update_mode_sequence(mode_seq)
            
            try:
                u_current, warm_start = nlp_controller.compute(
                    x_current, t_current=t_current, warm_start=warm_start, return_warm_start=True
                )
                mpc_counter += 1
            except:
                u_current = u_hist[i-1] if i > 1 else 0.0
                warm_start = None
        else:
            u_current = u_hist[i-1] if i > 1 else 0.0
        
        # Simulate
        if state_current == "stance":
            result = hopper.stance_state(x_current, u_current)
            x_dot = result[0]
            F_sub = result[1] if len(result) >= 2 else 0.0
        else:
            result = hopper.flight_state(x_current, u_current)
            x_dot = result[0]
            F_sub = 0.0
        
        x_next = x_current + dt_sim * np.array(x_dot)
        
        # Ground constraint
        if x_next[2] < 0:
            x_next[2] = 0.0
            x_next[3] = 0.0
        
        # Mode transition
        if state_current == "stance":
            l = x_next[0] - x_next[2]
            if l > l0 and x_next[1] > 0.05:
                state_current = "flight"
        else:
            if x_next[2] <= 0:
                state_current = "stance"
        
        x[i] = x_next
        
        # Metrics
        height_error_hist.append(x_target - x_next[0])
        control_effort_hist.append(u_current**2)
        l = x_next[0] - x_next[2]
        spring_compression = max(0.0, l0 - l)
        spring_energy_hist.append(0.5 * k * spring_compression**2)
    
    return {
        'mean_height_error': np.mean(np.abs(height_error_hist)),
        'rms_height_error': np.sqrt(np.mean(np.array(height_error_hist)**2)),
        'total_control_effort': np.sum(control_effort_hist),
        'mean_spring_energy': np.mean(spring_energy_hist),
        'max_height': np.max(x[:, 0]),
        'mpc_calls': mpc_counter,
        'height_error_hist': height_error_hist,
        'control_effort_hist': control_effort_hist,
    }


# ============================================================
# Main Comparison
# ============================================================
print("="*70)
print("PD vs NLP Controller Performance Comparison")
print("="*70)

# Load config
config_path = os.path.join(os.path.dirname(__file__), "..", "cfg", "simple_active_hopper_spring.yaml")
with open(config_path, 'r') as f:
    cfg = yaml.safe_load(f)

sim_cfg = cfg["simulation"]
dt_sim = float(sim_cfg["dt"])
t_max = float(sim_cfg["t_max"])

print(f"\nSimulation parameters:")
print(f"  Duration: {t_max} s")
print(f"  Time step: {dt_sim} s")
print(f"  Target height: {cfg['nlp']['x_target']} m")
print()

# Run PD controller
print("Running PD controller...")
pd_results = run_pd_simulation(cfg, dt_sim, t_max)

# Run NLP controller
print("Running NLP controller...")
nlp_results = run_nlp_simulation(cfg, dt_sim, t_max)

# Compare results
print("\n" + "="*70)
print("PERFORMANCE COMPARISON")
print("="*70)
print(f"{'Metric':<30} {'PD Controller':<20} {'NLP Controller':<20} {'Winner':<15}")
print("-"*70)

metrics = [
    ('Mean Height Error (m)', 'mean_height_error', 'lower'),
    ('RMS Height Error (m)', 'rms_height_error', 'lower'),
    ('Total Control Effort', 'total_control_effort', 'lower'),
    ('Mean Spring Energy (J)', 'mean_spring_energy', 'lower'),
    ('Max Height (m)', 'max_height', 'closer_to_target'),
]

target_height = float(cfg["nlp"]["x_target"])

for metric_name, key, better in metrics:
    pd_val = pd_results[key]
    nlp_val = nlp_results[key]
    
    if better == 'lower':
        winner = "NLP" if nlp_val < pd_val else "PD"
        improvement = ((pd_val - nlp_val) / pd_val * 100) if pd_val > 0 else 0
    elif better == 'closer_to_target':
        pd_diff = abs(pd_val - target_height)
        nlp_diff = abs(nlp_val - target_height)
        winner = "NLP" if nlp_diff < pd_diff else "PD"
        improvement = ((pd_diff - nlp_diff) / pd_diff * 100) if pd_diff > 0 else 0
    else:
        winner = "Tie"
        improvement = 0
    
    print(f"{metric_name:<30} {pd_val:<20.4f} {nlp_val:<20.4f} {winner:<15} ({improvement:+.1f}%)")

print("-"*70)
print(f"\nMPC calls: {nlp_results.get('mpc_calls', 'N/A')}")

# Overall winner
pd_score = (pd_results['mean_height_error'] + pd_results['rms_height_error'] + 
             pd_results['total_control_effort'] / 1e6)
nlp_score = (nlp_results['mean_height_error'] + nlp_results['rms_height_error'] + 
             nlp_results['total_control_effort'] / 1e6)

print("\n" + "="*70)
if nlp_score < pd_score:
    print("✅ NLP CONTROLLER WINS!")
    print(f"   Overall score: PD={pd_score:.4f}, NLP={nlp_score:.4f}")
    print(f"   Improvement: {((pd_score - nlp_score) / pd_score * 100):.1f}%")
else:
    print("⚠️  NLP controller needs tuning")
    print(f"   Overall score: PD={pd_score:.4f}, NLP={nlp_score:.4f}")
print("="*70)

