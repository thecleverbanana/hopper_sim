# https://ieeexplore.ieee.org/document/7989248
import numpy as np
import cyipopt

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
        

    # ---------- NEW: NLP controller with multiple shooting ----------
    class NLPController(cyipopt.Problem):
        """
        Multiple-shooting NLP for one horizon, using trapezoidal integration.

        State at node k:
            x_k = [x_b, x_b_dot, x_f, x_f_dot]

        Control at node k:
            u_k (scalar force)

        Dynamics constraint (for each interval k):
            x_{k+1} - x_k
            - dt/2 * ( f(x_k,u_k) + f(x_{k+1},u_{k+1}) ) = 0

        Cost:
            J = sum_k dt/2 * ( g(x_k,u_k) + g(x_{k+1},u_{k+1}) ),
            with g(x,u) = Q_l (l - l_ref)^2 + R_u u^2, l = x_b - x_f.
        """

        def __init__(
            self,
            hopper,            # instance of simplified_GRP_hopper
            H,                 # number of intervals
            dt,                # time step (assume constant here)
            x0,                # initial state (4,)
            mode_seq,          # list of length H, each 'flight' or 'stance'
            u_min, u_max,      # scalar bounds on u
            l_ref,             # desired leg length
            Q_l=1.0,
            R_u=1e-3,
        ):
            self.hopper = hopper
            self.H = H
            self.dt = dt
            self.nx = 4
            self.nu = 1
            self.mode_seq = list(mode_seq)
            assert len(self.mode_seq) == H, "mode_seq length must equal H"
            self.u_min = u_min
            self.u_max = u_max
            self.l_ref = l_ref
            self.Q_l = Q_l
            self.R_u = R_u

            # Number of decision variables:
            #   states x_0...x_H (H+1 nodes)
            #   controls u_0...u_H (H+1 nodes)
            self.Nx_total = (H + 1) * self.nx
            self.Nu_total = (H + 1) * self.nu
            n_var = self.Nx_total + self.Nu_total

            # Number of equality constraints (4 per interval)
            m_constr = H * self.nx

            # ----- variable bounds -----
            w_L = -1e6 * np.ones(n_var)
            w_U =  1e6 * np.ones(n_var)

            # Fix initial state: x_0 = x0
            w_L[0:self.nx] = x0
            w_U[0:self.nx] = x0

            # Control bounds for all u_k
            u_start = self.Nx_total
            w_L[u_start:] = u_min
            w_U[u_start:] = u_max

            # ----- constraint bounds: all dynamics defects = 0 -----
            c_L = np.zeros(m_constr)
            c_U = np.zeros(m_constr)

            self.lb = w_L
            self.ub = w_U
            self.cl = c_L
            self.cu = c_U
            self.n = n_var
            self.m = m_constr

            super().__init__(n=n_var, m=m_constr, lb=w_L, ub=w_U, cl=c_L, cu=c_U)

        # ---- helpers ----

        def _unpack(self, w):
            """
            Split big vector w into:
                X: (H+1, 4) states
                U: (H+1, 1) controls
            """
            x_flat = w[:self.Nx_total]
            u_flat = w[self.Nx_total:]
            X = x_flat.reshape((self.H + 1, self.nx))
            U = u_flat.reshape((self.H + 1, self.nu))
            return X, U

        def _running_cost(self, x, u):
            """Return scalar cost for state/control pair."""
            x_b, _, x_f, _ = map(float, x)
            l = x_b - x_f
            # ensure scalar u
            u_val = float(np.asarray(u).ravel()[0])
            cost = self.Q_l * (l - self.l_ref) ** 2 + self.R_u * (u_val ** 2)
            return float(cost)

        def objective(self, w):
            """Trapezoidal integration of running cost; always returns float."""
            X, U = self._unpack(w)
            J = 0.0
            for k in range(self.H):
                gk   = float(self._running_cost(X[k],   U[k]))
                gkp1 = float(self._running_cost(X[k+1], U[k+1]))
                # print(type(gk), type(gkp1), gk, gkp1)
                J += float(0.5 * self.dt * (gk + gkp1))
            return float(J)

        def gradient(self, w):
            """
            Finite-difference gradient of objective.
            (Simple but OK for small problems; can be replaced by analytic later.)
            """
            eps = 1e-8
            f0 = self.objective(w)
            grad = np.zeros_like(w)
            for i in range(len(w)):
                wp = w.copy()
                wp[i] += eps
                grad[i] = (self.objective(wp) - f0) / eps
            return grad

        def constraints(self, w):
            """
            Dynamics constraints for each interval k:

                x_{k+1} - x_k
                - dt/2 * ( f(x_k, u_k) + f(x_{k+1}, u_{k+1}) ) = 0

            stacked for k = 0,...,H-1 into a vector of length H*nx.
            """
            X, U = self._unpack(w)
            c = np.zeros(self.H * self.nx)

            for k in range(self.H):
                xk   = X[k]
                xkp1 = X[k+1]
                uk   = U[k, 0]
                ukp1 = U[k+1, 0]

                mode = self.mode_seq[k]
                if mode == 'flight':
                    fk,   _ = self.hopper.flight_state(xk,   uk)
                    fkp1, _ = self.hopper.flight_state(xkp1, ukp1)
                elif mode == 'stance':
                    fk,   _ = self.hopper.stance_state(xk,   uk)
                    fkp1, _ = self.hopper.stance_state(xkp1, ukp1)
                else:
                    raise ValueError(f"Unknown mode '{mode}' at step {k}")

                defect = xkp1 - xk - 0.5 * self.dt * (fk + fkp1)
                c[k*self.nx:(k+1)*self.nx] = defect

            return c

        def jacobian(self, w):
            """
            Finite-difference Jacobian of constraints:

                J_ij = d c_i / d w_j

            returned as a flattened (row-major) array, as required by cyipopt.
            """
            eps = 1e-8
            c0 = self.constraints(w)
            m = len(c0)
            n = len(w)
            J = np.zeros((m, n))
            for j in range(n):
                wp = w.copy()
                wp[j] += eps
                cj = self.constraints(wp)
                J[:, j] = (cj - c0) / eps
            return J.ravel()

            # ---- Solve one-step MPC problem ----
        def compute(self, x_current, l_ref=None, warm_start=None):
            """
            Solve the NLP given the current state, return first control action u0.
            Optionally override l_ref for dynamic leg-length tracking.
            """
            if l_ref is not None:
                self.l_ref = l_ref

            # Update initial state bounds
            self.lb[:self.nx] = x_current
            self.ub[:self.nx] = x_current

            # Initial guess for IPOPT
            if warm_start is None:
                w0 = np.zeros(self.n + self.m)
                # simple initialization
                X0 = np.tile(x_current, self.H + 1)
                U0 = np.zeros((self.H + 1, self.nu))
                w0 = np.concatenate([X0, U0.ravel()])
            else:
                w0 = warm_start

            nlp = cyipopt.Problem(
                n=self.n,
                m=self.m,
                problem_obj=self,
                lb=self.lb,
                ub=self.ub,
                cl=self.cl,
                cu=self.cu,
            )

            nlp.add_option('max_iter', 200)
            nlp.add_option('print_level', 0)
            nlp.add_option('tol', 1e-3)

            w_opt, info = nlp.solve(w0)

            X_opt, U_opt = self._unpack(w_opt)
            u0 = U_opt[0, 0]
            return u0


