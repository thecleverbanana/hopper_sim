import numpy as np
import cyipopt

class simplified_GRP_hopper:
    def __init__(self, mb, mf, l0, g=9.81):
        self.mb = mb
        self.mf = mf
        self.l0 = l0
        self.g = g

    # =====================================================
    #                    FLIGHT DYNAMICS
    # =====================================================
    def flight_state(self, X, u):
        x_b, x_b_dot, x_f, x_f_dot = X
        g = self.g

        x_b_ddot = -g + u / self.mb
        x_f_ddot = -g - u / self.mf
        F_sub = 0.0

        return np.array([x_b_dot, x_b_ddot, x_f_dot, x_f_ddot]), F_sub

    # =====================================================
    #                    STANCE DYNAMICS
    # =====================================================
    def stance_state(self, X, u, substrate='rigid'):
        x_b, x_b_dot, x_f, x_f_dot = X
        mb = self.mb
        g = self.g

        if substrate == 'rigid':
            # stance: foot locked
            x_f = 0.0
            x_f_dot = 0.0
            x_f_ddot = 0.0

            x_b_ddot_free = (u - mb * g) / mb

            # prevent body penetrating ground
            if x_b <= 0.0 and x_b_ddot_free < 0.0:
                x_b_ddot = 0.0
                x_b_dot = max(0.0, x_b_dot)
                F_sub = mb * g - u
                F_sub = max(F_sub, 0.0)
            else:
                x_b_ddot = x_b_ddot_free
                F_sub = 0.0

            return np.array([x_b_dot, x_b_ddot, x_f_dot, x_f_ddot]), F_sub

        else:
            raise NotImplementedError

    # =====================================================
    #             LINEARIZED DYNAMICS JACOBIAN
    # =====================================================
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

        return f_x, f_u

    # =====================================================
    #                    NLP / MPC
    # =====================================================
    class NLPController(cyipopt.Problem):
        """
        Simplified NMPC with fixed-pattern mode sequence.
        """

        def __init__(
            self,
            hopper,
            H,
            dt,
            x0,
            mode_seq,
            u_min,
            u_max,
            R_u,
            Q_l,
            Q_b_mid=0.0,
            min_body_height=0.05,
            Q_bh=0.0,
            Q_bv=0.0,
            Q_fh=0.0,
            Q_fv=0.0,
            body_ref=0.3,
            body_vref=0.0,
            foot_ref=0.0,
            foot_vref=0.0,
            stance_steps=8,
        ):
            self.hopper = hopper
            self.l_ref = hopper.l0
            self.H = H
            self.dt = dt
            self.nx = 4
            self.nu = 1

            self.stance_steps = int(stance_steps)
            self.mode_seq = list(mode_seq)

            self.R_u = R_u
            self.Q_l = Q_l
            self.Q_b_mid = Q_b_mid
            self.min_body_height = min_body_height

            self.Q_bh = Q_bh
            self.Q_bv = Q_bv
            self.Q_fh = Q_fh
            self.Q_fv = Q_fv

            self.body_ref = body_ref
            self.body_vref = body_vref
            self.foot_ref = foot_ref
            self.foot_vref = foot_vref

            self.Nx_total = (H + 1) * self.nx
            self.Nu_total = (H + 1) * self.nu
            n_var = self.Nx_total + self.Nu_total

            m_constr = H * self.nx

            w_L = -1e6 * np.ones(n_var)
            w_U =  1e6 * np.ones(n_var)

            w_L[:self.nx] = x0
            w_U[:self.nx] = x0

            u_start = self.Nx_total
            w_L[u_start:] = u_min
            w_U[u_start:] = u_max

            c_L = np.zeros(m_constr)
            c_U = np.zeros(m_constr)

            self.lb = w_L
            self.ub = w_U
            self.cl = c_L
            self.cu = c_U
            self.n = n_var
            self.m = m_constr

            super().__init__(n=n_var, m=m_constr,
                             lb=w_L, ub=w_U,
                             cl=c_L, cu=c_U)

        # --------------------------------------
        def _unpack(self, w):
            X = w[:self.Nx_total].reshape((self.H+1, self.nx))
            U = w[self.Nx_total:].reshape((self.H+1, self.nu))
            return X, U

        # --------------------------------------
        def objective(self, w):
            X, U = self._unpack(w)
            dt = self.dt
            H = self.H

            J = 0.0

            for k in range(H+1):
                u = U[k,0]
                x_b = X[k,0]
                x_f = X[k,2]

                J += dt * self.R_u * (u*u)

                l = x_b - x_f
                J += dt * self.Q_l * (l - self.l_ref)**2

                J += dt * self.Q_b_mid * (x_b - self.min_body_height)**2

            x_bH, x_bvH, x_fH, x_fvH = X[H]

            J += self.Q_bh*(x_bH - self.body_ref)**2
            J += self.Q_bv*(x_bvH - self.body_vref)**2
            J += self.Q_fh*(x_fH - self.foot_ref)**2
            J += self.Q_fv*(x_fvH - self.foot_vref)**2

            return float(J)

        # --------------------------------------
        def gradient(self, w):
            X, U = self._unpack(w)
            dt = self.dt
            H = self.H

            dX = np.zeros_like(X)
            dU = np.zeros_like(U)

            for k in range(H+1):
                u = U[k,0]
                x_b = X[k,0]
                x_f = X[k,2]
                l = x_b - x_f

                dU[k,0] += dt * 2*self.R_u*u

                dX[k,0] += dt * 2*self.Q_l*(l - self.l_ref)
                dX[k,2] += dt * 2*self.Q_l*(l - self.l_ref)*(-1)

                dX[k,0] += dt * 2*self.Q_b_mid*(x_b - self.min_body_height)

            x_bH, x_bvH, x_fH, x_fvH = X[H]

            dX[H,0] += 2*self.Q_bh*(x_bH - self.body_ref)
            dX[H,1] += 2*self.Q_bv*(x_bvH - self.body_vref)
            dX[H,2] += 2*self.Q_fh*(x_fH - self.foot_ref)
            dX[H,3] += 2*self.Q_fv*(x_fvH - self.foot_vref)

            return np.concatenate([dX.reshape(-1), dU.reshape(-1)])

        # --------------------------------------
        def constraints(self, w):
            X, U = self._unpack(w)
            dt = self.dt

            c = np.zeros(self.m)

            for k in range(self.H):
                row = k*self.nx
                xk = X[k]
                xkp1 = X[k+1]
                uk = float(U[k])
                ukp1 = float(U[k+1])

                mode = self.mode_seq[k]

                if mode == "flight":
                    fk,_ = self.hopper.flight_state(xk, uk)
                    fkp1,_ = self.hopper.flight_state(xkp1, ukp1)
                else:
                    fk,_ = self.hopper.stance_state(xk, uk)
                    fkp1,_ = self.hopper.stance_state(xkp1, ukp1)

                defect = xkp1 - xk - 0.5*dt*(fk + fkp1)
                c[row:row+self.nx] = defect

            return c

        # --------------------------------------
        def jacobian(self, w):
            X, U = self._unpack(w)
            dt = self.dt
            nx = self.nx
            nu = self.nu
            Nx_total = self.Nx_total

            J = np.zeros((self.m, self.n))

            for k in range(self.H):
                row = k*nx
                xk = X[k]
                xkp1 = X[k+1]
                uk = float(U[k])
                ukp1 = float(U[k+1])

                mode = self.mode_seq[k]

                fk_x, fk_u = self.hopper.jacobian_dynamics(xk, uk, mode)
                fkp1_x, fkp1_u = self.hopper.jacobian_dynamics(xkp1, ukp1, mode)

                A_k   = -np.eye(nx) - 0.5*dt*fk_x
                B_k   = -0.5*dt*fk_u.reshape(nx,1)

                A_kp1 =  np.eye(nx) - 0.5*dt*fkp1_x
                B_kp1 = -0.5*dt*fkp1_u.reshape(nx,1)

                xk_col = k*nx
                xkp1_col = (k+1)*nx
                uk_col = Nx_total + k*nu
                ukp1_col = Nx_total + (k+1)*nu

                J[row:row+nx, xk_col:xk_col+nx] = A_k
                J[row:row+nx, xkp1_col:xkp1_col+nx] = A_kp1
                J[row:row+nx, uk_col:uk_col+nu] = B_k
                J[row:row+nx, ukp1_col:ukp1_col+nu] = B_kp1

            return J.ravel()

        # --------------------------------------
        def predict_mode_sequence(self, x_current, state_current):
            """
            Fixed, stable pattern:
              stance_steps then flight, or vice versa.
            """
            H = self.H
            Ns = min(self.stance_steps, H)

            seq = []
            if state_current == "stance":
                seq.extend(["stance"] * Ns)
                seq.extend(["flight"] * (H - Ns))
            else:
                seq.extend(["flight"] * Ns)
                seq.extend(["stance"] * (H - Ns))

            return seq

        # --------------------------------------
        def update_mode_sequence(self, mode_seq):
            self.mode_seq = list(mode_seq)

        # --------------------------------------
        def compute(self, x_current, warm_start=None, return_warm_start=False):
            self.lb[:self.nx] = x_current
            self.ub[:self.nx] = x_current

            if warm_start is None:
                X0 = np.tile(x_current, self.H+1)
                U0 = np.zeros((self.H+1, self.nu))
                w0 = np.concatenate([X0, U0.reshape(-1)])
            else:
                Xp, Up = self._unpack(warm_start)

                Xn = np.zeros_like(Xp)
                Un = np.zeros_like(Up)
                Xn[0] = x_current
                Xn[1:] = Xp[1:]
                Un[:-1] = Up[1:]
                Un[-1] = Up[-1]

                w0 = np.concatenate([Xn.reshape(-1), Un.reshape(-1)])

            nlp = cyipopt.Problem(
                n=self.n, m=self.m,
                problem_obj=self,
                lb=self.lb, ub=self.ub,
                cl=self.cl, cu=self.cu
            )

            nlp.add_option("print_level", 0)
            nlp.add_option("hessian_approximation", "limited-memory")
            nlp.add_option("linear_solver", "mumps")
            nlp.add_option("tol", 1e-3)
            nlp.add_option("acceptable_tol", 5e-3)
            nlp.add_option("max_iter", 150)

            w_opt, info = nlp.solve(w0)
            _, U_opt = self._unpack(w_opt)

            if return_warm_start:
                return float(U_opt[0,0]), w_opt
            else:
                return float(U_opt[0,0])
