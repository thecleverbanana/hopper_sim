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

    def jacobian_dynamics(self, X, u, mode):
        # X = [x_b, x_b_dot, x_f, x_f_dot]
        x_b, x_b_dot, x_f, x_f_dot = X

        k = self.k
        c = self.c
        mb = self.mb
        mf = self.mf
        l0 = self.l0
        g  = self.g

        # leg terms (not actually needed for df, but harmless)
        delta_l    = l0 - (x_b - x_f)
        delta_ldot = x_f_dot - x_b_dot

        # ∂F_leg/∂state (F = k*delta_l + c*delta_ldot)
        dF_dx_b     = -k
        dF_dx_bdot  = -c
        dF_dx_f     =  k
        dF_dx_fdot  =  c

        # f = [ x_b_dot,
        #       x_b_ddot,
        #       x_f_dot,
        #       x_f_ddot ]
        f_x = np.zeros((4,4))
        f_u = np.zeros((4,))

        # common entries
        # row 0: dx_b/dt = x_b_dot
        f_x[0,1] = 1.0

        # row 1: dx_b_dot/dt = -g + (F + u)/mb
        f_x[1,0] = dF_dx_b     / mb
        f_x[1,1] = dF_dx_bdot  / mb
        f_x[1,2] = dF_dx_f     / mb
        f_x[1,3] = dF_dx_fdot  / mb
        f_u[1]   = 1.0 / mb

        if mode == "flight":
            # row 2: dx_f/dt = x_f_dot
            f_x[2,3] = 1.0

            # row 3: dx_f_dot/dt = -g - F/mf
            f_x[3,0] = -dF_dx_b    / mf
            f_x[3,1] = -dF_dx_bdot / mf
            f_x[3,2] = -dF_dx_f    / mf
            f_x[3,3] = -dF_dx_fdot / mf

        else:  # stance: foot fixed (x_f = 0, x_f_dot = 0)
            # row 2,3 = 0 already
            pass

        return f_x, f_u
    
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
            R_u,
            l_ref,
            Q_l=1.0,
            Q_bh=0.0,
            Q_fh=0.0,
            Q_bd=0.0,
            Ql_T=0.0,
            Qbh_T=0.0,
            Qfh_T=0.0,
            Qbd_T=0.0,
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
            self.Ql_T = Ql_T
            self.Qbh_T = Qbh_T
            self.Qfh_T = Qfh_T
            self.Qbd_T = Qbd_T
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
        # Objective (trapezoidal integration) with terminal cost
        # -----------------------------------------------------
        # def objective(self, w):
        #     X, U = self._unpack(w)
        #     J = 0.0

        #     # running cost
        #     for k in range(self.H):
        #         J += self.dt * self._running_cost(X[k], U[k])

        #     # terminal cost
        #     x_T = X[self.H]
        #     x_b, x_b_dot, x_f, x_f_dot = x_T
        #     l = x_b - x_f

        #     J += (
        #         self.Ql_T  * (l - self.l_ref)**2 +
        #         self.Qbh_T * (x_b - self.body_ref)**2 +
        #         self.Qfh_T * (x_f - self.foot_ref)**2 +
        #         self.Qbd_T * (x_b_dot**2)
        #     )

        #     return float(J)

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
            c = np.zeros(self.H * self.nx) # [H*4] 1-d tensor

            for k in range(self.H):

                xk = X[k] #[4]
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

                fk   = np.asarray(fk) #[4]
                fkp1 = np.asarray(fkp1)

                defect = xkp1 - xk - 0.5 * self.dt * (fk + fkp1) #[4]
                c[k*self.nx:(k+1)*self.nx] = defect 

            return c

        # -----------------------------------------------------
        # Analytic Jacobian of constraints
        # -----------------------------------------------------

        def jacobian(self, w):
            X, U = self._unpack(w)
            H = self.H
            dt = self.dt

            # Jacobian is m × n flattened
            J = np.zeros((self.m, self.n))

            # convenience
            nx = self.nx
            nu = self.nu
            Nx_total = self.Nx_total

            # loop over each interval
            for k in range(H):
                row0 = k * nx

                # extract variables
                xk     = X[k]
                xkp1   = X[k+1]
                uk     = float(U[k])
                ukp1   = float(U[k+1])

                mode = self.mode_seq[k]

                # ------------------------------------------------------------------
                # 1. compute f(x_k, u_k) and f(x_{k+1}, u_{k+1})
                # ------------------------------------------------------------------
                if mode == "flight":
                    fk, _ = self.hopper.flight_state(xk, uk)
                    fkp1, _ = self.hopper.flight_state(xkp1, ukp1)
                else:
                    fk, _ = self.hopper.stance_state(xk, uk)
                    fkp1, _ = self.hopper.stance_state(xkp1, ukp1)

                fk   = np.asarray(fk)
                fkp1 = np.asarray(fkp1)

                # ------------------------------------------------------------------
                # 2. analytic Jacobian of dynamics f_x and f_u
                # flight_state and stance_state must return analytic derivatives
                # ------------------------------------------------------------------
                fk_x, fk_u = self.hopper.jacobian_dynamics(xk, uk, mode)
                fkp1_x, fkp1_u = self.hopper.jacobian_dynamics(xkp1, ukp1, mode)

                # trapezoidal coefficients
                A_k   = -np.eye(nx) - 0.5 * dt * fk_x
                B_k   = -0.5 * dt * fk_u.reshape(nx, 1)
                A_kp1 =  np.eye(nx) - 0.5 * dt * fkp1_x
                B_kp1 = -0.5 * dt * fkp1_u.reshape(nx, 1)

                # ------------------------------------------------------------------
                # 3. scatter these into Jacobian matrix
                # ------------------------------------------------------------------
                # column indices
                xk_col     = k * nx
                xkp1_col   = (k+1) * nx
                uk_col     = Nx_total + k * nu
                ukp1_col   = Nx_total + (k+1) * nu

                # fill blocks
                J[row0:row0+nx, xk_col:xk_col+nx]       = A_k
                J[row0:row0+nx, xkp1_col:xkp1_col+nx]   = A_kp1
                J[row0:row0+nx, uk_col:uk_col+nu]       = B_k
                J[row0:row0+nx, ukp1_col:ukp1_col+nu]   = B_kp1

            return J.ravel()


        # -----------------------------------------------------
        # Solve MPC and return u0
        # -----------------------------------------------------
        def compute(self, x_current, l_ref=None, body_ref=None, warm_start=None, return_warm_start=False):

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
            # Warm-start strategy (IMPROVED)
            # -------------------------------------------------
            if warm_start is None:
                # naive warm start
                X0 = np.tile(x_current, self.H + 1)
                U0 = np.zeros((self.H + 1, self.nu))
                w0 = np.concatenate([X0, U0.ravel()])
            else:
                # Shift previous solution for better warm-start
                # This uses the "tail" of the previous trajectory
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

            # Convergence tolerances (relaxed for speed)
            nlp.add_option("tol", 5e-3)              # Relaxed from 1e-3
            nlp.add_option("acceptable_tol", 1e-2)   # Relaxed from 5e-3

            # Max iterations (reduced for speed)
            nlp.add_option("max_iter", 100)          # Reduced from 150

            # Quiet solver
            nlp.add_option("print_level", 0)

            # -------------------------------------------------
            # Solve NLP
            # -------------------------------------------------
            w_opt, info = nlp.solve(w0)

            X_opt, U_opt = self._unpack(w_opt)

            # Return u_0 and optionally the full solution for warm-starting
            if return_warm_start:
                return float(U_opt[0, 0]), w_opt
            else:
                return float(U_opt[0, 0])

      