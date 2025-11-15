# https://ieeexplore.ieee.org/document/7989248
import numpy as np
import cyipopt

class simplified_GRP_hopper:
    def __init__(self, mb, mf, l0, e=0.1, g=9.81):
        self.mb = mb   # body mass
        self.mf = mf   # foot/leg mass
        self.l0 = l0   # rest leg length (not used in simplified dynamics)
        self.g = g     # gravity
        self.e = e     # restitution coefficient

    def flight_state(self, X, u):
        x_b, x_b_dot, x_f, x_f_dot = X
        g = self.g

        # Direct control force (no spring-damper)
        x_b_ddot = -g + u / self.mb
        x_f_ddot = -g - u / self.mf

        F_sub = 0.0  # no substrate force in flight

        return np.array([x_b_dot, x_b_ddot, x_f_dot, x_f_ddot]), F_sub
    
    def stance_state(self, X, u, substrate='rigid'):
        x_b, x_b_dot, x_f, x_f_dot = X
        mb = self.mb
        g  = self.g

        if substrate == 'rigid':
            x_f_ddot = 0.0
            x_f_dot  = 0.0

            x_b_ddot_free = (u - mb * g) / mb

            if x_b <= 0 and x_b_ddot_free < 0:
                x_b_ddot = 0.0
                F_sub = mb * g - u
                if F_sub < 0:
                    F_sub = 0.0

            else:
                x_b_ddot = x_b_ddot_free
                F_sub = 0.0

            return np.array([x_b_dot, x_b_ddot, x_f_dot, x_f_ddot]), F_sub

        elif substrate == 'granular':
            raise NotImplementedError("Granular substrate not implemented")

    def detect_liftoff(self, X, F_sub):
        x_b, v_b, x_f, v_f = X

        # Liftoff when spring force drops to foot weight or below, and body moving upward
        # This matches the logic: F_spring <= m_l * g and x[i][1] > 0
        return (v_b > 0) and (F_sub <= 0) and (x_b >= 0)

   
    def apply_liftoff_impulse(self, X):
        x_b, v_b, x_f, v_f = X

        mb = self.mb   # body mass
        mf = self.mf   # leg/foot mass

        # relative velocity before leaving ground
        v_rel = v_b - v_f     # foot is 0 in stance

        # impulse-based velocity update (1D elastic/inelastic separation)
        v_b_new = v_b - (1 + self.e) * (mf / (mb + mf)) * v_rel
        v_f_new = v_f + (1 + self.e) * (mb / (mb + mf)) * v_rel

        # enforce non-penetration
        if x_b < 0:
            x_b = 0.0
        if v_b < 0:
            v_b = 0.0

        # update and return
        return np.array([x_b, v_b_new, x_f, v_f_new])

    def detect_touchdown(self, X):
        x_b, v_b, x_f, v_f = X
        return (x_f <= 0.0) and (v_f < 0.0)

    def apply_touchdown_impulse(self, X):
        x_b, v_b, x_f, v_f = X

        v_f_new = 0.0
        x_f_new = 0.0
        return np.array([x_b, v_b, x_f_new, v_f_new])

    def jacobian_dynamics(self, X, u, mode):
        mb = self.mb
        mf = self.mf

        f_x = np.zeros((4, 4))
        f_u = np.zeros((4,))

        f_x[0, 1] = 1.0

        f_u[1] = 1.0 / mb

        if mode == "flight":
            f_x[2, 3] = 1.0
            f_u[3] = -1.0 / mf

        else:
            pass

        return f_x, f_u
    
    class NLPController(cyipopt.Problem):
        def __init__(self, hopper, H, dt, x0, mode_seq,
                        u_min, u_max,
                        R_u, Q_terminal, body_ref):

            self.hopper = hopper
            self.H = H
            self.dt = dt
            self.nx = 4
            self.nu = 1

            self.mode_seq = mode_seq

            # cost weights
            self.R_u        = R_u
            self.Q_terminal = Q_terminal  
            self.body_ref   = body_ref

            # decision vars: X(0..H), U(0..H)
            self.Nx_total = (H + 1) * self.nx
            self.Nu_total = (H + 1) * self.nu
            n = self.Nx_total + self.Nu_total
            m = H * self.nx

            # bounds
            lb = -1e6 * np.ones(n)
            ub =  1e6 * np.ones(n)

            # initial state fixed
            lb[:self.nx] = x0
            ub[:self.nx] = x0

            # control bounds
            lb[self.Nx_total:] = u_min
            ub[self.Nx_total:] = u_max

            self.lb = lb
            self.ub = ub
            self.cl = np.zeros(m)
            self.cu = np.zeros(m)
            self.n = n
            self.m = m

            super().__init__(n=n, m=m, lb=lb, ub=ub, cl=self.cl, cu=self.cu)

        # --------------------------------------------------
        def _unpack(self, w):
            X = w[:self.Nx_total].reshape(self.H + 1, self.nx)
            U = w[self.Nx_total:].reshape(self.H + 1, self.nu)
            return X, U

        # --------------------------------------------------
        # Running cost: ONLY effort now
        # --------------------------------------------------
        def _running_cost(self, x, u):
            u = float(u)
            return self.R_u * (u**2)

        # --------------------------------------------------
        # Objective function with terminal cost
        # --------------------------------------------------
        def objective(self, w):
            X, U = self._unpack(w)
            J = 0.0

            # running cost
            for k in range(self.H):
                J += 0.5 * self.dt * (
                    self._running_cost(X[k],   U[k]) +
                    self._running_cost(X[k+1], U[k+1])
                )

            # terminal cost: final body height x_b(H)
            x_b_H = X[self.H, 0]
            J += self.Q_terminal * (x_b_H - self.body_ref)**2

            return J

        # --------------------------------------------------
        # trapezoidal dynamics constraints
        # --------------------------------------------------
        def constraints(self, w):
            X, U = self._unpack(w)
            dt = self.dt
            c  = np.zeros(self.m)

            for k in range(self.H):
                xk = X[k]
                xkp1 = X[k+1]
                uk = float(U[k])
                ukp1 = float(U[k+1])
                
                mode = self.mode_seq[k]
                
                if mode == "flight":
                    fk, _   = self.hopper.flight_state(xk,   uk)
                    fkp1, _ = self.hopper.flight_state(xkp1, ukp1)
                else:  # stance
                    fk, _   = self.hopper.stance_state(xk,   uk)
                    fkp1, _ = self.hopper.stance_state(xkp1, ukp1)

                fk = np.asarray(fk)
                fkp1 = np.asarray(fkp1)

                defect = xkp1 - xk - 0.5 * dt * (fk + fkp1)
                c[k*4:(k+1)*4] = defect

            return c

        # --------------------------------------------------
        # analytic Jacobian of constraints
        # --------------------------------------------------
        def jacobian(self, w):
            X, U = self._unpack(w)

            H  = self.H
            nx = self.nx
            dt = self.dt
            Nx_total = self.Nx_total

            # final Jacobian (m constraints × n variables)
            J = np.zeros((self.m, self.n))

            for k in range(H):
                row0 = k * nx

                # unpack states/inputs
                xk, xkp1 = X[k], X[k+1]
                uk, ukp1 = float(U[k]), float(U[k+1])
                mode     = self.mode_seq[k]

                # get local Jacobians from hopper model
                fk_x,   fk_u   = self.hopper.jacobian_dynamics(xk,   uk,   mode)
                fkp1_x, fkp1_u = self.hopper.jacobian_dynamics(xkp1, ukp1, mode)

                # trapezoidal linearization blocks
                A_k   = -np.eye(nx) - 0.5 * dt * fk_x
                A_kp1 =  np.eye(nx) - 0.5 * dt * fkp1_x
                B_k   = -0.5 * dt * fk_u.reshape(nx, 1)
                B_kp1 = -0.5 * dt * fkp1_u.reshape(nx, 1)

                J[row0:row0+nx, k*nx:(k+1)*nx] = A_k
                J[row0:row0+nx, (k+1)*nx:(k+2)*nx] = A_kp1
                J[row0:row0+nx, Nx_total + k : Nx_total + k + 1] = B_k
                J[row0:row0+nx, Nx_total + k + 1 : Nx_total + k + 2] = B_kp1

            return J.ravel()

        # --------------------------------------------------
        # mode sequence prediction
        # --------------------------------------------------
        def predict_mode_sequence(self, x_current, state_current, H, dt_control):
            mode_seq = []
            state = state_current

            # state copy
            x = np.array(x_current, dtype=float)

            for k in range(H):

                u_pred = 0.0  # prediction uses small control

                if state == "flight":
                    # -------- FLIGHT ------
                    x_dot, F_sub = self.hopper.flight_state(x, u_pred)
                    x_next = x + x_dot * dt_control

                    # touchdown detection
                    if self.hopper.detect_touchdown(x_next):
                        state = "stance"
                        
                        # snap foot to ground
                        x_next[2] = 0.0
                        x_next[3] = 0.0

                else:
                    # -------- STANCE ------
                    x_dot, F_sub = self.hopper.stance_state(x, u_pred)
                    x_next = x + x_dot * dt_control

                    # enforce stance constraints
                    x_next[2] = 0.0
                    x_next[3] = 0.0

                    # liftoff detection (pre-impulse check)
                    if self.hopper.detect_liftoff(x_next, F_sub):
                        
                        # apply impulsive velocity update
                        x_next = self.hopper.apply_liftoff_impulse(x_next)

                        state = "flight"

                mode_seq.append(state)
                x = x_next

            return mode_seq

        # --------------------------------------------------
        # update mode sequence
        # --------------------------------------------------
        def update_mode_sequence(self, new_mode_seq):
            if len(new_mode_seq) != self.H:
                raise ValueError(f"Mode sequence must have length {self.H}, got {len(new_mode_seq)}")
            self.mode_seq = list(new_mode_seq)

        # Dummy Gradient Function
        # --------------------------------------------------
        def gradient(self, w):
            return np.zeros_like(w)

        # --------------------------------------------------
        # main MPC solve
        # --------------------------------------------------
        def compute(self, x_current, warm_start=None):
            # enforce initial condition
            self.lb[:4] = x_current
            self.ub[:4] = x_current

            # warm start
            if warm_start is None:
                X0 = np.tile(x_current, self.H+1)
                U0 = np.zeros((self.H+1,1))
                w0 = np.concatenate([X0, U0.ravel()])
            else:
                w0 = warm_start

            # IPOPT
            nlp = cyipopt.Problem(
                n=self.n,
                m=self.m,
                problem_obj=self,
                lb=self.lb, ub=self.ub,
                cl=self.cl, cu=self.cu
            )

            # fast settings
            nlp.add_option("hessian_approximation","limited-memory")
            nlp.add_option("linear_solver","mumps")
            nlp.add_option("tol",5e-3)
            nlp.add_option("max_iter",80)
            nlp.add_option("print_level",0)

            # solve
            w_opt, _ = nlp.solve(w0)
            X_opt, U_opt = self._unpack(w_opt)

            return float(U_opt[0,0]), w_opt