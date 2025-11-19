import numpy as np
import cyipopt


class SpringLoadedHopper:
    """
    Spring-loaded active hopper model.
    State: [x_b, x_b_dot, x_l, x_l_dot]
    """
    def __init__(self, m_body=1.0, m_leg=0.2, l0=0.3, k=200.0, g=9.81):
        self.m_body = m_body
        self.m_leg = m_leg
        self.l0 = l0
        self.k = k
        self.g = g
    
    def flight_state(self, X, u):
        """
        Flight phase dynamics.
        Spring force acts between body and leg.
        """
        x_b, x_b_dot, x_l, x_l_dot = X
        
        # Spring force: F_spring = k * (l0 - l) where l = x_b - x_l
        l = x_b - x_l
        F_spring = self.k * (self.l0 - l)
        
        # Body acceleration: gravity + spring force + control
        x_b_ddot = (-self.m_body * self.g + F_spring + u) / self.m_body
        
        # Leg acceleration: gravity - spring force - control
        x_l_ddot = (-self.m_leg * self.g - F_spring - u) / self.m_leg
        
        return np.array([x_b_dot, x_b_ddot, x_l_dot, x_l_ddot]), 0.0  # F_sub = 0 in flight

    def stance_state(self, X, u):
        """
        Stance phase dynamics.
        Foot is fixed at ground (x_l = 0).
        Spring force acts between body and foot.
        """
        x_b, x_b_dot, x_l, x_l_dot = X

        # RIGID GROUND: foot fixed at x_l = 0
        x_l = 0.0
        x_l_dot = 0.0
        
        # Spring force: F_spring = k * (l0 - l) where l = x_b - x_l = x_b
        # When x_b < l0, spring is compressed, F_spring > 0 (upward on body)
        # When x_b > l0, spring is extended, F_spring < 0 (downward on body)
        l = x_b - x_l  # = x_b since x_l = 0
        F_spring = self.k * (self.l0 - l)
        
        # Body acceleration: gravity + spring force + control
        x_b_ddot = (-self.m_body * self.g + F_spring + u) / self.m_body
        
        # Ground reaction force: balances leg mass and control
        F_sub = self.m_leg * self.g + u
        
        x_l_ddot = 0.0
        
        return np.array([x_b_dot, x_b_ddot, x_l_dot, x_l_ddot]), F_sub, F_spring
    
    def jacobian_dynamics(self, X, u, mode):
        """Linearized dynamics Jacobian"""
        k = self.k
        mb = self.m_body
        mf = self.m_leg
        
        f_x = np.zeros((4, 4))
        f_u = np.zeros((4,))
        
        f_x[0, 1] = 1.0  # dx_b/dx_b_dot
        f_x[2, 3] = 1.0  # dx_l/dx_l_dot
        
        if mode == "flight":
            # Spring force: F_spring = k * (l0 - (x_b - x_l))
            f_x[1, 0] = -k / mb  # dx_b_ddot/dx_b
            f_x[1, 2] = k / mb   # dx_b_ddot/dx_l
            f_x[3, 0] = k / mf   # dx_l_ddot/dx_b
            f_x[3, 2] = -k / mf  # dx_l_ddot/dx_l
            
            f_u[1] = 1.0 / mb
            f_u[3] = -1.0 / mf
        else:  # stance
            # In stance: x_l = 0 fixed, so only x_b matters
            f_x[1, 0] = -k / mb  # dx_b_ddot/dx_b
            f_u[1] = 1.0 / mb
        
        return f_x, f_u


class PDController:
    """
    Simple PD controller for height control.
    
    LIMITATIONS:
    - Only considers body height, ignores spring dynamics
    - Cannot coordinate leg properly for energy-efficient hopping
    - No feedforward compensation for spring forces
    - Struggles with phase transitions (stance/flight)
    
    This controller demonstrates why optimal control is needed!
    """
    def __init__(self, kp=800.0, kd=100.0):
        self.kp = kp
        self.kd = kd
    
    def compute(self, X, height_ref, height_dot_ref=0.0):
        """
        Compute PD control input.
        
        Args:
            X: State [x_b, x_b_dot, x_l, x_l_dot]
            height_ref: Target body height
            height_dot_ref: Target body velocity (default 0)
        
        Returns:
            u: Control force
        """
        x_b, x_b_dot, _, _ = X
        e = height_ref - x_b
        e_dot = height_dot_ref - x_b_dot
        u = self.kp * e + self.kd * e_dot
        return u


class NLPController(cyipopt.Problem):
    """
    Nonlinear Programming (NLP) controller using IPOPT.
    Implements Model Predictive Control (MPC) for optimal hopping.
    """
    def __init__(
        self,
        hopper,
        H,
        dt,
        x0,
        mode_seq,
        u_min=-200.0,
        u_max=200.0,
        R_u=1e-3,
        R_du=1e-4,  # Control rate penalty for smoothness
        Q_x=30.0,
        Q_v=5.0,
        Q_xT=80.0,
        Q_vT=10.0,
        Q_lc=0.0,  # Limit cycle stabilization weight
        x_target=0.6,
        enable_limit_cycle=False,  # Enable limit cycle stabilization
    ):
        """
        Initialize NLP controller.
        
        Args:
            hopper: SpringLoadedHopper instance
            H: Prediction horizon length
            dt: Time step for MPC
            x0: Initial state [x_b, x_b_dot, x_l, x_l_dot]
            mode_seq: List of modes for horizon ["stance", "flight", ...]
            u_min, u_max: Control bounds
            R_u: Control effort weight
            Q_x: Running cost weight for height tracking
            Q_v: Running cost weight for velocity
            Q_xT: Terminal cost weight for height
            Q_vT: Terminal cost weight for velocity
            x_target: Target body height
        """
        # -----------------------------
        # Store parameters
        # -----------------------------
        self.hopper = hopper
        self.H = H
        self.dt = dt
        self.nx = 4
        self.nu = 1
        self.mode_seq = list(mode_seq)
        assert len(self.mode_seq) == H, f"Mode sequence length {len(self.mode_seq)} != H {H}"
        
        # Cost parameters
        self.R_u = R_u
        self.R_du = R_du  # Control rate penalty for smoothness
        self.Q_x = Q_x
        self.Q_v = Q_v
        self.Q_xT = Q_xT
        self.Q_vT = Q_vT
        self.Q_lc = Q_lc  # Limit cycle stabilization weight
        self.x_target = x_target  # Default constant target
        
        # Reference trajectory (can be function or array)
        self.x_ref_trajectory = None  # Will be set via update_reference()
        self.x_ref_function = None  # Callable function x_ref(t)
        self.t_current = 0.0  # Current simulation time (updated in compute())
        
        # Limit cycle stabilization
        self.enable_limit_cycle = enable_limit_cycle
        self.x_poincare_prev = None  # Previous Poincaré section state (for limit cycle)
        
        # -----------------------------
        # Decision variables: X + U
        # X: (H+1)*nx, U: (H+1)*nu
        # -----------------------------
        self.Nx_total = (H + 1) * self.nx
        self.Nu_total = (H + 1) * self.nu
        n_var = self.Nx_total + self.Nu_total
        
        # Constraints: H dynamics constraints (nx each)
        m_constr = H * self.nx
        
        # -----------------------------
        # Variable bounds
        # -----------------------------
        w_L = -1e6 * np.ones(n_var)
        w_U = 1e6 * np.ones(n_var)
        
        # Fix initial state
        w_L[:self.nx] = x0
        w_U[:self.nx] = x0
        
        # Control bounds
        u_start = self.Nx_total
        w_L[u_start:] = u_min
        w_U[u_start:] = u_max
        
        # Constraint bounds (defect = 0)
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
    # Helper: unpack decision variables
    # -----------------------------------------------------
    def _unpack(self, w):
        """Unpack decision variables into states and controls"""
        X = w[:self.Nx_total].reshape(self.H + 1, self.nx)
        U = w[self.Nx_total:].reshape(self.H + 1, self.nu)
        return X, U
    
    # -----------------------------------------------------
    # Cost Functions
    # -----------------------------------------------------
    def _running_cost(self, x, v, u, u_prev=None, x_ref=None, v_ref=None):
        """Running cost at a single time step"""
        if x_ref is None:
            x_ref = self.x_target
        if v_ref is None:
            v_ref = 0.0  # Default: zero velocity reference
        cost = (
            self.R_u * u**2 +
            self.Q_x * (x - x_ref)**2 +
            self.Q_v * (v - v_ref)**2  # Track velocity reference for smoothness
        )
        # Add control rate penalty for smoothness
        if u_prev is not None:
            cost += self.R_du * (u - u_prev)**2
        return cost
    
    def _terminal_cost(self, xT, vT, x_ref=None):
        """Terminal cost"""
        if x_ref is None:
            x_ref = self.x_target
        return (
            self.Q_xT * (xT - x_ref)**2 +
            self.Q_vT * vT**2
        )
    
    def _get_reference(self, k, t_current=0.0):
        """
        Get reference height at time step k.
        
        Args:
            k: Time step index (0 to H)
            t_current: Current simulation time
        
        Returns:
            x_ref: Reference height at this time step
        """
        if self.x_ref_function is not None:
            # Use callable function
            t_k = t_current + k * self.dt
            return self.x_ref_function(t_k)
        elif self.x_ref_trajectory is not None:
            # Use pre-computed trajectory array
            if k < len(self.x_ref_trajectory):
                return self.x_ref_trajectory[k]
            else:
                # Extend last value if horizon exceeds trajectory
                return self.x_ref_trajectory[-1]
        else:
            # Default: constant target
            return self.x_target
    
    def _get_reference_velocity(self, k, t_current=0.0):
        """
        Get reference velocity at time step k (for smooth tracking).
        Uses finite difference to estimate velocity from reference trajectory.
        
        Args:
            k: Time step index (0 to H)
            t_current: Current simulation time
        
        Returns:
            v_ref: Reference velocity at this time step
        """
        dt = self.dt
        if k == 0:
            # At first step, use forward difference
            x_ref_0 = self._get_reference(0, t_current)
            x_ref_1 = self._get_reference(1, t_current)
            return (x_ref_1 - x_ref_0) / dt
        elif k == self.H:
            # At last step, use backward difference
            x_ref_H = self._get_reference(self.H, t_current)
            x_ref_Hm1 = self._get_reference(self.H - 1, t_current)
            return (x_ref_H - x_ref_Hm1) / dt
        else:
            # Use central difference for smoother estimate
            x_ref_kp1 = self._get_reference(k + 1, t_current)
            x_ref_km1 = self._get_reference(k - 1, t_current)
            return (x_ref_kp1 - x_ref_km1) / (2 * dt)
    
    def objective(self, w):
        """Objective function: minimize control effort and tracking error"""
        X, U = self._unpack(w)
        dt = self.dt
        H = self.H
        
        J = 0.0
        
        # Running cost
        for k in range(H):
            x, v = X[k, 0], X[k, 1]
            u = U[k, 0]
            u_prev = U[k-1, 0] if k > 0 else 0.0  # Previous control for rate penalty
            x_ref_k = self._get_reference(k, self.t_current)
            v_ref_k = self._get_reference_velocity(k, self.t_current)
            J += dt * self._running_cost(x, v, u, u_prev, x_ref_k, v_ref_k)
        
        # Terminal cost
        xT, vT = X[H, 0], X[H, 1]
        x_ref_T = self._get_reference(H, self.t_current)
        J += self._terminal_cost(xT, vT, x_ref_T)
        
        # Limit cycle stabilization: penalize deviation from previous Poincaré section
        # This encourages convergence to a periodic orbit
        if self.enable_limit_cycle and self.x_poincare_prev is not None:
            # Use terminal state as current Poincaré section (assuming it's at touchdown)
            xT_full = X[H]
            # Only penalize height and velocity (not leg states which reset at touchdown)
            J += self.Q_lc * (
                (xT_full[0] - self.x_poincare_prev[0])**2 +
                (xT_full[1] - self.x_poincare_prev[1])**2
            )
        
        return float(J)
    
    def gradient(self, w):
        """Analytical gradient of objective (correct implementation)"""
        X, U = self._unpack(w)
        dt = self.dt
        H = self.H
        
        grad_X = np.zeros_like(X)
        grad_U = np.zeros_like(U)
        
        # Running cost gradients
        for k in range(H):
            x, v = X[k, 0], X[k, 1]  # Body height and velocity
            u = U[k, 0]
            u_prev = U[k-1, 0] if k > 0 else 0.0
            x_ref_k = self._get_reference(k, self.t_current)
            v_ref_k = self._get_reference_velocity(k, self.t_current)
            
            # dJ/dx (height) - only affects body height (index 0)
            grad_X[k, 0] += dt * 2 * self.Q_x * (x - x_ref_k)
            # dJ/dv (velocity) - only affects body velocity (index 1)
            grad_X[k, 1] += dt * 2 * self.Q_v * (v - v_ref_k)
            # dJ/du (control)
            grad_U[k, 0] += dt * 2 * self.R_u * u
            # dJ/du (control rate penalty)
            if k > 0:
                grad_U[k, 0] += dt * 2 * self.R_du * (u - u_prev)
                grad_U[k-1, 0] -= dt * 2 * self.R_du * (u - u_prev)
        
        # Terminal cost gradients
        xT, vT = X[H, 0], X[H, 1]
        x_ref_T = self._get_reference(H, self.t_current)
        grad_X[H, 0] += 2 * self.Q_xT * (xT - x_ref_T)
        grad_X[H, 1] += 2 * self.Q_vT * vT
        
        # Limit cycle stabilization gradients
        if self.enable_limit_cycle and self.x_poincare_prev is not None:
            xT_full = X[H]
            grad_X[H, 0] += 2 * self.Q_lc * (xT_full[0] - self.x_poincare_prev[0])
            grad_X[H, 1] += 2 * self.Q_lc * (xT_full[1] - self.x_poincare_prev[1])
        
        return np.concatenate([grad_X.reshape(-1), grad_U.reshape(-1)])
    
    # -----------------------------------------------------
    # Constraint Functions
    # -----------------------------------------------------
    def _get_dynamics(self, x, u, mode):
        """Get dynamics for given state, control, and mode"""
        if mode == "flight":
            f, _ = self.hopper.flight_state(x, u)
        else:  # stance
            result = self.hopper.stance_state(x, u)
            f = result[0]  # Extract dynamics (first element)
        return f
    
    def constraints(self, w):
        """Dynamics constraints: x_{k+1} = x_k + dt * f(x_k, u_k)"""
        X, U = self._unpack(w)
        dt = self.dt
        H = self.H
        nx = self.nx
        
        c = np.zeros(self.m)
        
        for k in range(H):
            xk = X[k]
            xkp1 = X[k+1]
            uk = U[k, 0]
            ukp1 = U[k+1, 0]
            
            mode = self.mode_seq[k]
            
            # Get dynamics at both endpoints (trapezoidal integration)
            fk = self._get_dynamics(xk, uk, mode)
            fkp1 = self._get_dynamics(xkp1, ukp1, mode)
            
            # Defect constraint: x_{k+1} - x_k - 0.5*dt*(f_k + f_{k+1}) = 0
            defect = xkp1 - xk - 0.5 * dt * (fk + fkp1)
            c[k*nx:(k+1)*nx] = defect
        
        return c
    
    def jacobian(self, w):
        """Analytical Jacobian of constraints (correct implementation)"""
        X, U = self._unpack(w)
        H = self.H
        dt = self.dt
        
        # Jacobian is m × n (2D matrix, will be flattened at end)
        J = np.zeros((self.m, self.n))
        
        # Convenience variables
        nx = self.nx
        nu = self.nu
        Nx_total = self.Nx_total
        
        # Loop over each interval
        for k in range(H):
            row0 = k * nx
            
            # Extract variables
            xk = X[k]
            xkp1 = X[k+1]
            uk = float(U[k, 0])
            ukp1 = float(U[k+1, 0])
            
            mode = self.mode_seq[k]
            
            # Get dynamics at endpoints
            if mode == "flight":
                fk, _ = self.hopper.flight_state(xk, uk)
                fkp1, _ = self.hopper.flight_state(xkp1, ukp1)
            else:  # stance
                result_k = self.hopper.stance_state(xk, uk)
                result_kp1 = self.hopper.stance_state(xkp1, ukp1)
                fk = result_k[0]
                fkp1 = result_kp1[0]
            
            fk = np.asarray(fk)
            fkp1 = np.asarray(fkp1)
            
            # Get analytical dynamics Jacobians
            fk_x, fk_u = self.hopper.jacobian_dynamics(xk, uk, mode)
            fkp1_x, fkp1_u = self.hopper.jacobian_dynamics(xkp1, ukp1, mode)
            
            # Trapezoidal integration coefficients
            A_k = -np.eye(nx) - 0.5 * dt * fk_x
            B_k = -0.5 * dt * fk_u.reshape(nx, 1)
            A_kp1 = np.eye(nx) - 0.5 * dt * fkp1_x
            B_kp1 = -0.5 * dt * fkp1_u.reshape(nx, 1)
            
            # Column indices
            xk_col = k * nx
            xkp1_col = (k + 1) * nx
            uk_col = Nx_total + k * nu
            ukp1_col = Nx_total + (k + 1) * nu
            
            # Fill Jacobian blocks
            J[row0:row0+nx, xk_col:xk_col+nx] = A_k
            J[row0:row0+nx, xkp1_col:xkp1_col+nx] = A_kp1
            J[row0:row0+nx, uk_col:uk_col+nu] = B_k
            J[row0:row0+nx, ukp1_col:ukp1_col+nu] = B_kp1
        
        return J.ravel()
    
    # -----------------------------------------------------
    # Mode Sequence Management
    # -----------------------------------------------------
    def update_mode_sequence(self, mode_seq):
        """Update mode sequence for MPC horizon"""
        assert len(mode_seq) == self.H, f"Mode sequence length {len(mode_seq)} != H {self.H}"
        self.mode_seq = list(mode_seq)
    
    # -----------------------------------------------------
    # Limit Cycle Stabilization
    # -----------------------------------------------------
    def update_poincare_state(self, x_poincare):
        """
        Update Poincaré section state for limit cycle stabilization.
        Call this when system crosses Poincaré section (e.g., at touchdown).
        
        Args:
            x_poincare: State at Poincaré section [x_b, x_b_dot, x_l, x_l_dot]
        """
        self.x_poincare_prev = np.array(x_poincare)
    
    def enable_limit_cycle_stabilization(self, Q_lc, enable=True):
        """
        Enable/disable limit cycle stabilization.
        
        Args:
            Q_lc: Weight for limit cycle cost
            enable: Whether to enable limit cycle stabilization
        """
        self.enable_limit_cycle = enable
        self.Q_lc = Q_lc
    
    # -----------------------------------------------------
    # Reference Trajectory Management
    # -----------------------------------------------------
    def update_reference(self, x_ref=None, x_ref_function=None):
        """
        Update reference trajectory for tracking.
        
        Args:
            x_ref: Array of reference heights for horizon [x_ref[0], ..., x_ref[H]]
                   or None to use constant target
            x_ref_function: Callable function x_ref(t) that returns reference height at time t
                           or None to use array/constant
        """
        if x_ref_function is not None:
            self.x_ref_function = x_ref_function
            self.x_ref_trajectory = None
        elif x_ref is not None:
            self.x_ref_trajectory = np.asarray(x_ref)
            self.x_ref_function = None
        else:
            # Reset to constant target
            self.x_ref_trajectory = None
            self.x_ref_function = None
    
    # -----------------------------------------------------
    # MPC Solve
    # -----------------------------------------------------
    def _prepare_warm_start(self, x_current, warm_start):
        """Prepare warm start guess"""
        if warm_start is None:
            X0 = np.tile(x_current, self.H + 1)
            U0 = np.zeros((self.H + 1, self.nu))
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
        return w0
    
    def compute(self, x_current, t_current=0.0, warm_start=None, return_warm_start=False):
        """
        Solve MPC and return first control input.
        
        Args:
            x_current: Current state [x_b, x_b_dot, x_l, x_l_dot]
            t_current: Current simulation time (for trajectory tracking)
            warm_start: Previous solution for warm-starting (optional)
            return_warm_start: If True, return solution for next warm-start
        
        Returns:
            u: First control input (or (u, w_opt) if return_warm_start=True)
        """
        # Store current time for cost function evaluation
        self.t_current = t_current
        
        # Update initial state bounds
        self.lb[:self.nx] = x_current
        self.ub[:self.nx] = x_current
        
        # Prepare warm start
        w0 = self._prepare_warm_start(x_current, warm_start)
        
        # Solve NLP
        nlp = cyipopt.Problem(
            n=self.n, m=self.m,
            problem_obj=self,
            lb=self.lb, ub=self.ub,
            cl=self.cl, cu=self.cu
        )
        
        nlp.add_option("print_level", 0)
        nlp.add_option("hessian_approximation", "limited-memory")
        nlp.add_option("limited_memory_max_history", 20)
        nlp.add_option("limited_memory_initialization", "scalar2")
        nlp.add_option("tol", 1e-3)  # Tolerance for accuracy
        nlp.add_option("acceptable_tol", 5e-3)  # Acceptable tolerance for early termination
        nlp.add_option("max_iter", 150)  # Enough iterations for convergence
        nlp.add_option("mu_strategy", "adaptive")  # Adaptive barrier parameter
        nlp.add_option("bound_relax_factor", 0.0)  # No bound relaxation
        
        w_opt, info = nlp.solve(w0)
        _, U_opt = self._unpack(w_opt)
        
        if return_warm_start:
            return float(U_opt[0, 0]), w_opt
        else:
            return float(U_opt[0, 0])
