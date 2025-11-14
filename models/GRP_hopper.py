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

    class NLPController(cyipopt.Problem):
        def __init__(
            self,
            hopper,
            H,
            dt,
            x0,
            mode_seq,
            u_min, u_max,
            l_ref,
            Q_l=1.0,
            R_u=1e-3,
            Q_bh=0.0,
            Q_fh=0.0,
            Q_bd=0.0,
            body_ref=0.3,
            foot_ref=0.0,
        ):

            # -----------------------------
            # store parameters
            # -----------------------------
            self.hopper = hopper
            self.H = H
            self.dt = dt
            self.nx = 4
            self.nu = 1

            # mode sequence (must be length H)
            self.mode_seq = list(mode_seq)
            assert len(self.mode_seq) == H

            # cost parameters
            self.l_ref = l_ref
            self.Q_l = Q_l
            self.R_u = R_u
            self.Q_bh = Q_bh
            self.Q_fh = Q_fh
            self.Q_bd = Q_bd
            self.body_ref = body_ref
            self.foot_ref = foot_ref

            # -----------------------------
            # Decision variables = X + U
            # X: (H+1)*4
            # U: (H+1)*1
            # -----------------------------
            self.Nx_total = (H + 1) * self.nx
            self.Nu_total = (H + 1) * self.nu
            n_var = self.Nx_total + self.Nu_total

            # constraints = defects for each interval (H * 4)
            m_constr = H * self.nx

            # -----------------------------
            # Variable bounds
            # -----------------------------
            w_L = -1e6 * np.ones(n_var)
            w_U =  1e6 * np.ones(n_var)

            # Initial state fixed
            w_L[:self.nx] = x0
            w_U[:self.nx] = x0

            # Control bounds
            u_start = self.Nx_total
            w_L[u_start:] = u_min
            w_U[u_start:] = u_max

            # constraint bounds = 0
            c_L = np.zeros(m_constr)
            c_U = np.zeros(m_constr)

            self.lb = w_L
            self.ub = w_U
            self.cl = c_L
            self.cu = c_U
            self.n = n_var
            self.m = m_constr

            super().__init__(n=n_var, m=m_constr, lb=w_L, ub=w_U, cl=c_L, cu=c_U)


        # -----------------------------------------------------
        # Helper: unpack X and U
        # -----------------------------------------------------
        def _unpack(self, w):
            X = w[:self.Nx_total].reshape((self.H + 1, self.nx))
            U = w[self.Nx_total:].reshape((self.H + 1, self.nu))
            return X, U


        # -----------------------------------------------------
        # Running cost
        # -----------------------------------------------------
        def _running_cost(self, x, u):

            x_b, x_b_dot, x_f, x_f_dot = x
            u = float(u)

            l = x_b - x_f

            return (
                self.Q_l  * (l - self.l_ref)**2 +
                self.R_u  * (u**2) +
                self.Q_bh * (x_b - self.body_ref)**2 +
                self.Q_fh * (x_f - self.foot_ref)**2 +
                self.Q_bd * (x_b_dot**2)
            )


        # -----------------------------------------------------
        # Objective (trapezoidal integration)
        # -----------------------------------------------------
        def objective(self, w):
            X, U = self._unpack(w)
            J = 0.0
            for k in range(self.H):
                J += 0.5 * self.dt * (
                    self._running_cost(X[k],   U[k]) +
                    self._running_cost(X[k+1], U[k+1])
                )
            return float(J)


        # -----------------------------------------------------
        # Gradient (finite diff)
        # -----------------------------------------------------
        def gradient(self, w):
            eps = 1e-8
            f0 = self.objective(w)
            grad = np.zeros_like(w)
            for i in range(len(w)):
                w2 = w.copy()
                w2[i] += eps
                grad[i] = (self.objective(w2) - f0) / eps
            return grad


        # -----------------------------------------------------
        # Dynamics constraints
        # -----------------------------------------------------
        def constraints(self, w):
            X, U = self._unpack(w)
            c = np.zeros(self.H * self.nx)

            for k in range(self.H):

                xk = X[k]
                xkp1 = X[k+1]
                uk = float(U[k])
                ukp1 = float(U[k+1])

                mode = self.mode_seq[k]

                if mode == "flight":
                    fk, _   = self.hopper.flight_state(xk,   uk)     # take only x_dot
                    fkp1, _ = self.hopper.flight_state(xkp1, ukp1)
                else:  # stance
                    fk, _   = self.hopper.stance_state(xk,   uk)
                    fkp1, _ = self.hopper.stance_state(xkp1, ukp1)

                fk   = np.asarray(fk)
                fkp1 = np.asarray(fkp1)

                defect = xkp1 - xk - 0.5 * self.dt * (fk + fkp1)
                c[k*self.nx:(k+1)*self.nx] = defect

            return c



        # -----------------------------------------------------
        # Jacobian (finite diff)
        # -----------------------------------------------------
        def jacobian(self, w):
            eps = 1e-8
            c0 = self.constraints(w)
            m = len(c0)
            n = len(w)

            J = np.zeros((m, n))
            for j in range(n):
                w2 = w.copy()
                w2[j] += eps
                c2 = self.constraints(w2)
                J[:, j] = (c2 - c0) / eps

            return J.ravel()


        # -----------------------------------------------------
        # Solve MPC and return u0
        # -----------------------------------------------------
        def compute(self, x_current, l_ref=None, body_ref=None, warm_start=None):

            # -------------------------------------------------
            # Update references (body height / leg length)
            # -------------------------------------------------
            if l_ref is not None:
                self.l_ref = l_ref
            if body_ref is not None:
                self.body_ref = body_ref

            # -------------------------------------------------
            # Enforce initial state constraint x_0 = x_current
            # -------------------------------------------------
            self.lb[:self.nx] = x_current
            self.ub[:self.nx] = x_current

            # -------------------------------------------------
            # Warm-start strategy
            # -------------------------------------------------
            if warm_start is None:
                # naive warm start
                X0 = np.tile(x_current, self.H + 1)
                U0 = np.zeros((self.H + 1, self.nu))
                w0 = np.concatenate([X0, U0.ravel()])
            else:
                w0 = warm_start

            # -------------------------------------------------
            # Build NLP instance
            # -------------------------------------------------
            nlp = cyipopt.Problem(
                n=self.n,
                m=self.m,
                problem_obj=self,
                lb=self.lb,
                ub=self.ub,
                cl=self.cl,
                cu=self.cu,
            )

            # -------------------------------------------------
            # FAST IPOPT SETTINGS (correct for cyipopt)
            # -------------------------------------------------
            # L-BFGS Hessian (no exact Hessian needed)
            nlp.add_option("hessian_approximation", "limited-memory")
            nlp.add_option("limited_memory_max_history", 20)
            nlp.add_option("limited_memory_initialization", "scalar2")

            # Linear solver
            nlp.add_option("linear_solver", "mumps")

            # Convergence tolerances
            nlp.add_option("tol", 1e-3)
            nlp.add_option("acceptable_tol", 5e-3)

            # Max iterations
            nlp.add_option("max_iter", 150)

            # Quiet solver
            nlp.add_option("print_level", 0)

            # -------------------------------------------------
            # Solve NLP
            # -------------------------------------------------
            w_opt, info = nlp.solve(w0)

            X_opt, U_opt = self._unpack(w_opt)

            # Return u_0 only
            return float(U_opt[0, 0])

      