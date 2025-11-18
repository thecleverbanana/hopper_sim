import numpy as np
import cyipopt

class simplified_GRP_hopper:
    def __init__(self, mb, mf, l0, g=9.81):
        self.mb = mb   # body mass
        self.mf = mf   # foot/leg mass
        self.l0 = l0   # rest leg length
        self.g = g     # gravity

    # -----------------------------
    # FLIGHT
    # -----------------------------
    def flight_state(self, X, u):
        x_b, x_b_dot, x_f, x_f_dot = X
        g = self.g

        x_b_ddot = -g + u / self.mb
        x_f_ddot = -g - u / self.mf
        F_sub = 0.0 

        return np.array([x_b_dot, x_b_ddot, x_f_dot, x_f_ddot]), F_sub

    # -----------------------------
    # STANCE
    # -----------------------------
    def stance_state(self, X, u, substrate='rigid'):
        x_b, x_b_dot, x_f, x_f_dot = X
        mb = self.mb
        g = self.g

        if substrate == 'rigid':
            # Foot is locked to ground during stance
            x_f_ddot = 0.0
            x_f_dot  = 0.0

            # Compute free body acceleration
            x_b_ddot_free = (u - mb * g) / mb

            # Ground penetration prevention: body cannot go below foot
            # Also prevent body from going below ground itself
            min_height = max(x_f, 0.0)
            
            if x_b <= min_height and x_b_ddot_free < 0.0:
                # Body at/below minimum height and accelerating down -> stop it
                x_b_ddot = 0.0
                x_b_dot = max(0.0, x_b_dot)  # Also zero out downward velocity
                F_sub = mb * g - u
                if F_sub < 0.0:
                    F_sub = 0.0
            else:
                x_b_ddot = x_b_ddot_free
                F_sub = 0.0

            return np.array([x_b_dot, x_b_ddot, x_f_dot, x_f_ddot]), F_sub

        elif substrate == 'granular':
            raise NotImplementedError("Granular substrate not implemented")

    # -----------------------------
    # Jacobian of dynamics
    # -----------------------------
    def jacobian_dynamics(self, X, u, mode):
        mb = self.mb
        mf = self.mf

        f_x = np.zeros((4, 4))
        f_u = np.zeros((4,))

        # x_b dynamics
        f_x[0, 1] = 1.0          # d(x_b_dot)/d(x_b_dot) = 1
        f_u[1]    = 1.0 / mb     # d(x_b_ddot)/d(u)

        if mode == "flight":
            f_x[2, 3] = 1.0      # d(x_f_dot)/d(x_f_dot) = 1
            f_u[3]    = -1.0 / mf
        else:
            pass

        return f_x, f_u

    # =========================================================
    #      NLP Controller (MPC via IPOPT)
    # =========================================================
    class NLPController(cyipopt.Problem):
        """
        Extremely simplified MPC for hopper apex control:
        - Running cost: R_u * u^2
        - Terminal cost: Q_h*(x_b(H)-body_ref)^2 + Q_v*(x_b_dot(H))^2
        """

        def __init__(self,hopper,H,dt,x0,mode_seq,u_min,u_max,R_u,Q_bh,Q_bv,Q_fh,Q_fv,Q_l,body_ref,body_vref,foot_ref,foot_vref):
            # Store
            self.hopper = hopper
            self.l0 = self.hopper.l0
            self.H = H
            self.dt = dt
            self.nx = 4
            self.nu = 1

            self.mode_seq = list(mode_seq)
            assert len(self.mode_seq) == H

            # cost weights
            self.R_u = R_u
            self.Q_bh = Q_bh
            self.Q_bv = Q_bv
            self.Q_fh = Q_fh
            self.Q_fv = Q_fv
            self.Q_l = Q_l
            self.body_ref = body_ref
            self.body_vref = body_vref
            self.foot_ref = foot_ref
            self.foot_vref = foot_vref

            # Decision variable sizes
            self.Nx_total = (H + 1) * self.nx
            self.Nu_total = (H + 1) * self.nu
            n_var = self.Nx_total + self.Nu_total

            # Constraints: H * nx defects
            m_constr = H * self.nx

            # Variable bounds
            w_L = -1e6 * np.ones(n_var)
            w_U =  1e6 * np.ones(n_var)
            
            # Physical constraints on states (for all timesteps)
            for k in range(H + 1):
                idx = k * self.nx
                # Body position: must be above ground
                w_L[idx + 0] = 0.0001  # x_b >= 0 (small epsilon to avoid numerical issues)
                # Body velocity: reasonable bounds
                w_L[idx + 1] = -10.0   # x_b_dot >= -10 m/s
                w_U[idx + 1] = 10.0    # x_b_dot <= 10 m/s
                # Foot position: must be above ground
                w_L[idx + 2] = 0.0     # x_f >= 0
                # Foot velocity: reasonable bounds
                w_L[idx + 3] = -10.0   # x_f_dot >= -10 m/s
                w_U[idx + 3] = 10.0    # x_f_dot <= 10 m/s

            # Fix initial state x_0 (overrides the above for k=0)
            w_L[:self.nx] = x0
            w_U[:self.nx] = x0

            # Control bounds
            u_start = self.Nx_total
            w_L[u_start:] = u_min
            w_U[u_start:] = u_max

            # Constraint bounds = 0
            c_L = np.zeros(m_constr)
            c_U = np.zeros(m_constr)

            # Save
            self.lb = w_L
            self.ub = w_U
            self.cl = c_L
            self.cu = c_U
            self.n  = n_var
            self.m  = m_constr

            super().__init__(n=n_var, m=m_constr, lb=w_L, ub=w_U, cl=c_L, cu=c_U)


        # -----------------------------------------------------
        # Helper
        # -----------------------------------------------------
        def _unpack(self, w):
            X = w[:self.Nx_total].reshape((self.H + 1, self.nx))
            U = w[self.Nx_total:].reshape((self.H + 1, self.nu))
            return X, U


        # -----------------------------------------------------
        # Running Cost: R_u * u^2 (very simple)
        # -----------------------------------------------------
        def _running_cost(self, x, u):
            return self.R_u * (float(u)**2)

        def _running_cost_grad(self, x, u):
            # dL/dx = 0,   dL/du = 2 R_u u
            dL_dx = np.zeros(4)
            dL_du = 2.0 * self.R_u * float(u)
            return dL_dx, dL_du


        # -----------------------------------------------------
        # Objective with running + terminal costs
        # -----------------------------------------------------
        def objective(self, w):
            X, U = self._unpack(w)
            dt = self.dt
            H = self.H
            J = 0.0

            # Control effort (running cost)
            for k in range(self.H + 1):
                J += dt * self.R_u * (U[k, 0] ** 2)

            # Leg length soft constraint (all timesteps)
            for k in range(self.H + 1):
                x_b = X[k, 0]
                x_f = X[k, 2]
                delta_l = x_b - x_f - self.l0
                J += dt * self.Q_l * (delta_l ** 2)    

            # -----------------------------------------
            # Running costs for body/foot tracking
            # -----------------------------------------
            for k in range(H + 1):
                x_b = X[k, 0]
                x_f = X[k, 2]
                
                # Encourage body to rise toward target (running cost)
                J += dt * 0.1 * self.Q_bh * (x_b - self.body_ref)**2
                
                # Encourage foot to lift toward target (running cost)
                J += dt * 0.1 * self.Q_fh * (x_f - self.foot_ref)**2

            # -----------------------------------------
            # Terminal state costs (stronger weight at horizon)
            # -----------------------------------------
            x_bH    = X[H, 0]
            x_bdotH = X[H, 1]
            x_fH    = X[H, 2]
            x_fdotH = X[H, 3]

            # body height target
            J += self.Q_bh * (x_bH - self.body_ref)**2

            # body velocity target (apex condition)
            J += self.Q_bv * (x_bdotH - self.body_vref)**2

            # foot height target
            J += self.Q_fh * (x_fH - self.foot_ref)**2

            # foot velocity target
            J += self.Q_fv * (x_fdotH - self.foot_vref)**2

            return float(J)


        # -----------------------------------------------------
        # Gradient (analytic)
        # -----------------------------------------------------
        def gradient(self, w):
            X, U = self._unpack(w)
            dt = self.dt
            H = self.H

            dJ_dX = np.zeros_like(X)
            dJ_dU = np.zeros_like(U)

            # =====================================================
            # 1. Control effort: R_u * u^2
            # =====================================================
            for k in range(H + 1):
                _, dLdu = self._running_cost_grad(X[k], U[k])
                dJ_dU[k, 0] += dt * dLdu

            # =====================================================
            # 2. Leg length soft constraint over all k
            #    J += dt * Q_l (x_b - x_f - l0)^2
            # =====================================================
            for k in range(H + 1):
                x_b = X[k, 0]
                x_f = X[k, 2]
                delta_l = x_b - x_f - self.l0

                # ∂/∂x_b
                dJ_dX[k, 0] += dt * 2.0 * self.Q_l * delta_l
                # ∂/∂x_f
                dJ_dX[k, 2] += dt * (-2.0 * self.Q_l * delta_l)

            # =====================================================
            # 3. Running costs for body/foot tracking
            # =====================================================
            for k in range(H + 1):
                x_b = X[k, 0]
                x_f = X[k, 2]
                
                # Body height running cost gradient
                dJ_dX[k, 0] += dt * 0.1 * 2.0 * self.Q_bh * (x_b - self.body_ref)
                
                # Foot height running cost gradient
                dJ_dX[k, 2] += dt * 0.1 * 2.0 * self.Q_fh * (x_f - self.foot_ref)

            # =====================================================
            # 4. Terminal costs at H (stronger weight)
            # =====================================================
            x_bH    = X[H, 0]
            x_bdotH = X[H, 1]
            x_fH    = X[H, 2]
            x_fdotH = X[H, 3]

            # body height gradient
            dJ_dX[H, 0] += 2.0 * self.Q_bh * (x_bH - self.body_ref)

            # body velocity gradient
            dJ_dX[H, 1] += 2.0 * self.Q_bv * (x_bdotH - self.body_vref)

            # foot height gradient
            dJ_dX[H, 2] += 2.0 * self.Q_fh * (x_fH - self.foot_ref)

            # foot velocity gradient
            dJ_dX[H, 3] += 2.0 * self.Q_fv * (x_fdotH - self.foot_vref)

            # =====================================================
            # 5. Flatten result for IPOPT
            # =====================================================
            grad_X = dJ_dX.reshape(-1)
            grad_U = dJ_dU.reshape(-1)
            return np.concatenate([grad_X, grad_U])

        # -----------------------------------------------------
        # Dynamics constraints (trapezoidal)
        # -----------------------------------------------------
        def constraints(self, w):
            X, U = self._unpack(w)
            dt = self.dt
            c = np.zeros(self.m)

            for k in range(self.H):
                xk     = X[k]
                xkp1   = X[k+1]
                uk     = float(U[k])
                ukp1   = float(U[k+1])
                mode   = self.mode_seq[k]

                # dynamics
                if mode == "flight":
                    fk, _   = self.hopper.flight_state(xk, uk)
                    fkp1, _ = self.hopper.flight_state(xkp1, ukp1)
                else:
                    fk, _   = self.hopper.stance_state(xk, uk)
                    fkp1, _ = self.hopper.stance_state(xkp1, ukp1)

                defect = xkp1 - xk - 0.5 * dt * (np.array(fk) + np.array(fkp1))
                c[k*self.nx:(k+1)*self.nx] = defect

            return c

        # -----------------------------------------------------
        # Jacobian of constraints (analytic)
        # -----------------------------------------------------
        def jacobian(self, w):
            X, U = self._unpack(w)
            dt = self.dt
            nx = self.nx
            nu = self.nu
            Nx_total = self.Nx_total

            J = np.zeros((self.m, self.n))

            for k in range(self.H):
                row0 = k * nx
                xk   = X[k]
                xkp1 = X[k+1]
                uk   = float(U[k])
                ukp1 = float(U[k+1])
                mode = self.mode_seq[k]

                if mode == "flight":
                    fk, _   = self.hopper.flight_state(xk, uk)
                    fkp1, _ = self.hopper.flight_state(xkp1, ukp1)
                else:
                    fk, _   = self.hopper.stance_state(xk, uk)
                    fkp1, _ = self.hopper.stance_state(xkp1, ukp1)

                fk_x, fk_u     = self.hopper.jacobian_dynamics(xk, uk, mode)
                fkp1_x, fkp1_u = self.hopper.jacobian_dynamics(xkp1, ukp1, mode)

                A_k   = -np.eye(nx) - 0.5 * dt * fk_x
                B_k   = -0.5 * dt * fk_u.reshape(nx, 1)

                A_kp1 =  np.eye(nx) - 0.5 * dt * fkp1_x
                B_kp1 = -0.5 * dt * fkp1_u.reshape(nx, 1)

                xk_col   = k * nx
                xkp1_col = (k+1) * nx
                uk_col   = Nx_total + k * nu
                ukp1_col = Nx_total + (k+1) * nu

                J[row0:row0+nx, xk_col:xk_col+nx]     = A_k
                J[row0:row0+nx, xkp1_col:xkp1_col+nx] = A_kp1

                J[row0:row0+nx, uk_col:uk_col+nu]     = B_k
                J[row0:row0+nx, ukp1_col:ukp1_col+nu] = B_kp1

            return J.ravel()

        def predict_mode_sequence(self,x_current,state_current, stance_duration=0.20, u_guess=0.0, eps_touch=1e-4):
            H  = self.H
            dt = self.dt
            hopper = self.hopper

            mode_seq = []
            x = np.array(x_current, dtype=float)
            current = state_current

            t_stance = 0.0

            for k in range(H):

                if current == "flight":
                    # flight rollout
                    f, _ = hopper.flight_state(x, u_guess)
                    x_next = x + dt * f

                    # touchdown event
                    if (x[2] > 0.0) and (x_next[2] <= eps_touch):
                        current = "stance"
                        t_stance = 0.0

                    x = x_next

                else:  # stance
                    f, _ = hopper.stance_state(x, u_guess)
                    x_next = x + dt * f

                    t_stance += dt

                    # liftoff after fixed stance duration
                    if t_stance >= stance_duration:
                        current = "flight"

                    x = x_next

                mode_seq.append(current)

            return mode_seq


        # -----------------------------------------------------
        # Update mode sequence
        # -----------------------------------------------------
        def update_mode_sequence(self, mode_seq):
            if len(mode_seq) != self.H:
                raise ValueError("Invalid mode_seq length")
            for m in mode_seq:
                if m not in ["flight", "stance"]:
                    raise ValueError("Invalid mode")
            self.mode_seq = list(mode_seq)


        # -----------------------------------------------------
        # Solve with warm-start shift
        # -----------------------------------------------------
        def compute(self, x_current, warm_start=None, return_warm_start=False):
            # Fix initial state x_0
            self.lb[:self.nx] = x_current
            self.ub[:self.nx] = x_current

            # Warm-start
            if warm_start is None:
                X0 = np.tile(x_current, self.H + 1)
                U0 = np.zeros((self.H + 1, self.nu))
                w0 = np.concatenate([X0, U0.reshape(-1)])
            else:
                X_prev, U_prev = self._unpack(warm_start)
                X_new = np.zeros_like(X_prev)
                U_new = np.zeros_like(U_prev)

                X_new[0] = x_current
                X_new[1:] = X_prev[1:]
                U_new[:-1] = U_prev[1:]
                U_new[-1]  = U_prev[-1]

                w0 = np.concatenate([X_new.reshape(-1), U_new.reshape(-1)])

            # IPOPT Problem
            nlp = cyipopt.Problem(
                n=self.n,
                m=self.m,
                problem_obj=self,
                lb=self.lb,
                ub=self.ub,
                cl=self.cl,
                cu=self.cu
            )

            # IPOPT Options
            nlp.add_option("print_level", 0)
            nlp.add_option("hessian_approximation", "limited-memory")
            nlp.add_option("linear_solver", "mumps")
            nlp.add_option("tol", 1e-3)
            nlp.add_option("acceptable_tol", 5e-3)
            nlp.add_option("max_iter", 150)

            # Solve
            w_opt, info = nlp.solve(w0)
            _, U_opt = self._unpack(w_opt)

            if return_warm_start:
                return float(U_opt[0, 0]), w_opt
            else:
                return float(U_opt[0, 0])
