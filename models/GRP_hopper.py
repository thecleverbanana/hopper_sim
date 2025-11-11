# https://ieeexplore.ieee.org/document/7989248
import numpy as np
from scipy.optimize import minimize

class simplified_GRP_hopper:
    def __init__(self, mb, mf, k, c, l0, g=9.81):
        self.mb = mb   # body mass
        self.mf = mf   # foot/leg mass
        self.k = k     # spring stiffness
        self.c = c     # damping coefficient
        self.l0 = l0   # rest leg length
        self.g = g     # gravity

    def flight_state(self, X, u):
        x_b, x_b_dot, x_f, x_f_dot = X
        g = self.g

        # relative displacement and velocity
        delta_l = self.l0 - (x_b - x_f)
        delta_ldot = x_f_dot - x_b_dot

        # spring-damper force
        F_spring = self.k * delta_l + self.c * delta_ldot

        # body and foot accelerations
        x_b_ddot = -g + (F_spring + u) / self.mb
        x_f_ddot = -g - F_spring / self.mf

        F_sub = 0.0  # no substrate force in flight

        return np.array([x_b_dot, x_b_ddot, x_f_dot, x_f_ddot]), F_sub

    def stance_state(self, X, u, substrate='rigid'):
        x_b, x_b_dot, x_f, x_f_dot = X
        g = self.g

        # relative length and velocity
        delta_l = self.l0 - (x_b - x_f)
        delta_ldot = x_f_dot - x_b_dot

        # spring-damper force in leg
        F_leg = self.k * delta_l + self.c * delta_ldot

        if substrate == 'rigid':
            # rigid ground: foot fixed (x_f = 0)
            x_f = 0.0
            x_f_dot = 0.0
            x_f_ddot = 0.0
            # equilibrium of foot mass
            F_sub = self.mf * g + F_leg

        elif substrate == 'granular':
            raise NotImplementedError
            # Example granular resistance model
            F_sg = self.mf * g + F_leg  # solid-ground equivalent
            F_p = 0.5 * self.mf * g     # quasi-static depth component
            F_v = 0.1 * self.mf * x_f_dot**2 * np.sign(x_f_dot)  # velocity-dependent
            M_added = 0.2 * self.mf * abs(x_f_dot)               # added mass (example)

            F_sub = (self.mf / (self.mf + M_added)) * (F_p + F_v) \
                    - (M_added / (self.mf + M_added)) * F_sg
            x_f_ddot = (-self.mf * g - F_leg + F_sub) / self.mf
     
        # body equation of motion
        x_b_ddot = (-self.mb * g + F_leg + u) / self.mb

        return np.array([x_b_dot, x_b_ddot, x_f_dot, x_f_ddot]), F_sub

    class PDController:
        # Classic PD Controller for Testing Only
        def __init__(self, kp, kd):
            self.kp = kp
            self.kd = kd

        def compute(self, X, l_ref, ldot_ref):
            x_b, x_b_dot, x_f, x_f_dot = X

            # Leg length and its rate
            l = x_b - x_f
            ldot = x_b_dot - x_f_dot

            # PD control error
            error = l_ref - l
            derror = ldot_ref - ldot

            # Control input (actuator force)
            u = self.kp * error + self.kd * derror
            return u
        
    class NLPController:
        def __init__(self, hopper, state='stance', horizon=5, dt=0.01):
            self.hopper = hopper
            self.state = state
            self.horizon = horizon
            self.dt = dt

        def simulate_dynamics(self, X, u):
            """One-step forward simulation based on current phase."""
            if self.state == 'flight':
                dX, _ = self.hopper.flight_state(X, u)
            else:
                dX, _ = self.hopper.stance_state(X, u)
            return X + dX * self.dt

        def cost_function(self, u_seq, X0, l_ref):
            """
            Cost over the prediction horizon:
            penalize deviation of leg length from reference and control effort.
            """
            X = np.copy(X0)
            cost = 0.0
            for u in u_seq:
                # propagate dynamics
                X = self.simulate_dynamics(X, u)
                x_b, x_f = X[0], X[2]
                l = x_b - x_f  # leg length

                # cost: leg length tracking + control effort regularization
                cost += (l - l_ref)**2 + 0.01 * (u**2)
            return cost

        def compute(self, X0, l_ref):
            """Compute optimal control input to minimize tracking error."""
            u_init = np.zeros(self.horizon)
            bounds = [(-200, 200)] * self.horizon  # actuator force limits

            res = minimize(
                self.cost_function,
                u_init,
                args=(X0, l_ref),
                bounds=bounds,
                method='SLSQP',
                options={'maxiter': 150, 'ftol': 1e-5, 'disp': False}
            )

            # return first control input if optimization succeeded
            return res.x[0] if res.success else 0.0

