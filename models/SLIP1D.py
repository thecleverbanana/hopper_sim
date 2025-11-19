"""
1D Spring-Loaded Inverted Pendulum (SLIP) Model

A simplified 1D vertical hopper model with:
- State: [z, z_dot] (height and vertical velocity)
- Spring force acts vertically
- Flight phase: ballistic motion
- Stance phase: spring dynamics with foot fixed at ground
"""

import numpy as np
from scipy.integrate import solve_ivp
from math import pi as PI
import cyipopt

# State indices
Z, Z_DOT = (0, 1)


class SLIP1D:
    """
    1D Spring-Loaded Inverted Pendulum model.
    
    State: [z, z_dot]
        z: vertical position of body CoM (m)
        z_dot: vertical velocity of body CoM (m/s)
    
    Parameters:
        mass: body mass (kg)
        leg_length: nominal leg length r0 (m)
        k: spring stiffness (N/m)
        g: gravitational acceleration (m/s^2)
    """
    g = 9.81

    def __init__(self, mass, leg_length, k, g=9.81, verbose=False):
        """
        Initialize 1D SLIP model.
        
        Args:
            mass: Body mass (kg)
            leg_length: Nominal leg length r0 (m)
            k: Spring stiffness (N/m)
            g: Gravitational acceleration (m/s^2)
            verbose: Print model parameters
        """
        self.m = mass
        self.r0 = leg_length
        self.k = k
        self.g = g
        self.verbose = verbose
        if verbose:
            print(str(self))

    def flight_dynamics(self, t, state):
        """
        Flight phase dynamics (ballistic motion).
        
        Args:
            t: Time (not used, required by solve_ivp)
            state: [z, z_dot]
        
        Returns:
            [z_dot, z_ddot]
        """
        z, z_dot = state
        z_ddot = -self.g
        return np.array([z_dot, z_ddot])

    def stance_dynamics(self, t, state, u=0.0):
        """
        Stance phase dynamics (spring-loaded).
        
        Args:
            t: Time (not used, required by solve_ivp)
            state: [z, z_dot]
            u: Control input (force applied to body, N)
        
        Returns:
            [z_dot, z_ddot]
        """
        z, z_dot = state
        
        # Spring compression: delta = r0 - z (when z < r0, spring is compressed)
        # Spring force: F_spring = k * (r0 - z) (positive when compressed)
        F_spring = self.k * (self.r0 - z)
        
        # Body acceleration: gravity + spring force + control
        z_ddot = (-self.m * self.g + F_spring + u) / self.m
        
        return np.array([z_dot, z_ddot])

    def get_flight_trajectory(self, t, take_off_state):
        """
        Compute flight trajectory analytically.
        
        Args:
            t: Time array (s)
            take_off_state: [z_TO, z_dot_TO] at take-off
        
        Returns:
            trajectory: (2, len(t)) array [z, z_dot] over time
        """
        assert take_off_state.shape[0] == 2, 'Provide a valid (2,) take-off state [z, z_dot]'
        z_TO, z_dot_TO = take_off_state

        trajectory = np.zeros((2, t.shape[0]))
        trajectory[0, :] = z_TO + z_dot_TO * t - 0.5 * self.g * t ** 2
        trajectory[1, :] = z_dot_TO - self.g * t

        return trajectory

    def get_stance_trajectory(self, touch_down_state, u=0.0, dt=0.005, t_max=None):
        """
        Compute stance trajectory using numerical integration.
        
        Args:
            touch_down_state: [z_TD, z_dot_TD] at touch-down
            u: Control input (constant force, N)
            dt: Integration time step (s)
            t_max: Maximum integration time (s). If None, uses spring natural period.
        
        Returns:
            t: Time array (s)
            trajectory: (2, len(t)) array [z, z_dot] over time
        """
        assert touch_down_state.shape[0] == 2, 'Provide a valid (2,) touch-down state [z, z_dot]'

        if t_max is None:
            t_max = 2 / self.spring_natural_freq

        def stance_dynamics_wrapper(t, x):
            return self.stance_dynamics(t, x, u)

        def take_off_detection(t, x, *args):
            """Detect when leg returns to nominal length (take-off)"""
            return x[0] - self.r0

        to_event = take_off_detection
        to_event.terminal = True
        to_event.direction = 1  # Only trigger when z crosses r0 from below

        def ground_contact_detection(t, x, *args):
            """Detect when body hits ground"""
            return x[0]

        ground_event = ground_contact_detection
        ground_event.terminal = True
        ground_event.direction = -1  # Only trigger when z crosses 0 from above

        t_eval = np.linspace(0, t_max, int(t_max / dt))
        
        solution = solve_ivp(
            fun=stance_dynamics_wrapper,
            t_span=(0, t_max),
            t_eval=t_eval,
            y0=touch_down_state,
            events=[to_event, ground_event],
            first_step=0.0001
        )
        
        t = solution.t
        trajectory = solution.y
        
        # Include take-off event state if detected
        if solution.status == 1 and len(solution.t_events[0]) > 0:
            t = np.append(t, solution.t_events[0][0])
            trajectory = np.hstack((trajectory, solution.y_events[0][:, 0:1]))

        return t, trajectory

    def predict_touch_down(self, take_off_state):
        """
        Predict touch-down state from take-off state.
        
        Args:
            take_off_state: [z_TO, z_dot_TO] at take-off
        
        Returns:
            touch_down_state: [z_TD, z_dot_TD] at touch-down (z_TD = r0)
            time_of_flight: Time from take-off to touch-down (s)
        """
        z_TO, z_dot_TO = take_off_state
        
        # Solve: z_TD = z_TO + z_dot_TO * t - 0.5 * g * t^2 = r0
        # This is: 0.5 * g * t^2 - z_dot_TO * t + (r0 - z_TO) = 0
        
        a = 0.5 * self.g
        b = -z_dot_TO
        c = self.r0 - z_TO
        
        discriminant = b**2 - 4 * a * c
        
        if discriminant < 0:
            # No real solution - body won't reach r0 height
            # Return instant touch-down
            return np.array([z_TO, z_dot_TO]), 0.0
        
        t1 = (-b + np.sqrt(discriminant)) / (2 * a)
        t2 = (-b - np.sqrt(discriminant)) / (2 * a)
        
        # Choose positive time
        time_of_flight = max(t1, t2) if t1 > 0 and t2 > 0 else (t1 if t1 > 0 else t2)
        
        if time_of_flight <= 0:
            # Instant touch-down
            return np.array([z_TO, z_dot_TO]), 0.0
        
        z_dot_TD = z_dot_TO - self.g * time_of_flight
        touch_down_state = np.array([self.r0, z_dot_TD])
        
        return touch_down_state, time_of_flight

    def get_apex_height(self, state):
        """
        Compute apex height from current state.
        
        Args:
            state: [z, z_dot]
        
        Returns:
            apex_height: Maximum height reached (m)
            time_to_apex: Time to reach apex (s)
        """
        z, z_dot = state
        
        if z_dot <= 0:
            # Already at or past apex
            return z, 0.0
        
        time_to_apex = z_dot / self.g
        apex_height = z + z_dot * time_to_apex - 0.5 * self.g * time_to_apex**2
        
        return apex_height, time_to_apex

    def jacobian_flight(self, state):
        """
        Linearized flight dynamics Jacobian.
        
        Args:
            state: [z, z_dot]
        
        Returns:
            f_x: (2, 2) Jacobian matrix
        """
        f_x = np.zeros((2, 2))
        f_x[0, 1] = 1.0  # dz/dz_dot
        # f_x[1, :] = [0, 0] since z_ddot = -g (constant)
        return f_x

    def jacobian_stance(self, state, u=0.0):
        """
        Linearized stance dynamics Jacobian.
        
        Args:
            state: [z, z_dot]
            u: Control input (N)
        
        Returns:
            f_x: (2, 2) State Jacobian
            f_u: (2,) Control Jacobian
        """
        z, z_dot = state
        
        f_x = np.zeros((2, 2))
        f_x[0, 1] = 1.0  # dz/dz_dot
        f_x[1, 0] = -self.k / self.m  # dz_ddot/dz (from spring force)
        
        f_u = np.zeros(2)
        f_u[1] = 1.0 / self.m  # dz_ddot/du
        
        return f_x, f_u

    @property
    def spring_natural_freq(self):
        """Natural frequency of spring-mass system (Hz)"""
        return 1 / (2 * PI) * np.sqrt(self.k / self.m)

    @property
    def spring_natural_period(self):
        """Natural period of spring-mass system (s)"""
        return 2 * PI * np.sqrt(self.m / self.k)

    def __str__(self):
        return 'SLIP1D m=%.2f[kg] r0=%.3f[m] k=%.1f[N/m] f_n=%.2f[Hz]' % (
            self.m, self.r0, self.k, self.spring_natural_freq
        )

    def __repr__(self):
        return str(self)


class NLPController1D(cyipopt.Problem):
    """
    Nonlinear Programming (NLP) Controller for 1D SLIP Model using IPOPT.
    
    Implements Model Predictive Control (MPC) with:
    - Prediction horizon H
    - Mode-aware dynamics (stance/flight)
    - Cost function: tracking error + control effort
    - Constraints: dynamics consistency
    """
    
    def __init__(
        self,
        slip_model,
        H,
        dt,
        x0,
        mode_seq,
        u_min=-500.0,
        u_max=500.0,
        R_u=1e-3,
        Q_z=50.0,
        Q_v=10.0,
        Q_zT=100.0,
        Q_vT=20.0,
        z_target=0.5,
    ):
        """
        Initialize NLP controller for 1D SLIP.
        
        Args:
            slip_model: SLIP1D instance
            H: Prediction horizon length
            dt: Time step for MPC
            x0: Initial state [z, z_dot]
            mode_seq: List of modes for horizon ["stance", "flight", ...]
            u_min, u_max: Control bounds
            R_u: Control effort weight
            Q_z: Running cost weight for height tracking
            Q_v: Running cost weight for velocity
            Q_zT: Terminal cost weight for height
            Q_vT: Terminal cost weight for velocity
            z_target: Target height
        """
        # Store parameters
        self.slip = slip_model
        self.H = H
        self.dt = dt
        self.nx = 2  # State dimension [z, z_dot]
        self.nu = 1  # Control dimension
        self.mode_seq = list(mode_seq)
        assert len(self.mode_seq) == H, f"Mode sequence length {len(self.mode_seq)} != H {H}"
        
        # Cost parameters
        self.R_u = R_u
        self.Q_z = Q_z
        self.Q_v = Q_v
        self.Q_zT = Q_zT
        self.Q_vT = Q_vT
        self.z_target = z_target
        
        # Decision variables: X + U
        # X: (H+1)*nx, U: (H+1)*nu
        self.Nx_total = (H + 1) * self.nx
        self.Nu_total = (H + 1) * self.nu
        n_var = self.Nx_total + self.Nu_total
        
        # Constraints: H dynamics constraints (nx each)
        m_constr = H * self.nx
        
        # Variable bounds
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
    
    def _unpack(self, w):
        """Unpack decision variables into states and controls"""
        X = w[:self.Nx_total].reshape(self.H + 1, self.nx)
        U = w[self.Nx_total:].reshape(self.H + 1, self.nu)
        return X, U
    
    def _running_cost(self, z, v, u):
        """Running cost at a single time step"""
        return (
            self.R_u * u**2 +
            self.Q_z * (z - self.z_target)**2 +
            self.Q_v * v**2
        )
    
    def _terminal_cost(self, zT, vT):
        """Terminal cost"""
        return (
            self.Q_zT * (zT - self.z_target)**2 +
            self.Q_vT * vT**2
        )
    
    def objective(self, w):
        """Objective function: minimize control effort and tracking error"""
        X, U = self._unpack(w)
        dt = self.dt
        H = self.H
        
        J = 0.0
        
        # Running cost
        for k in range(H):
            z, v = X[k, 0], X[k, 1]
            u = U[k, 0]
            J += dt * self._running_cost(z, v, u)
        
        # Terminal cost
        zT, vT = X[H, 0], X[H, 1]
        J += self._terminal_cost(zT, vT)
        
        return float(J)
    
    def gradient(self, w):
        """Gradient of objective (finite difference)"""
        eps = 1e-8
        grad = np.zeros_like(w)
        f0 = self.objective(w)
        
        for i in range(len(w)):
            wp = w.copy()
            wp[i] += eps
            grad[i] = (self.objective(wp) - f0) / eps
        
        return grad
    
    def _get_dynamics(self, x, u, mode):
        """Get dynamics for given state, control, and mode"""
        if mode == "flight":
            f = self.slip.flight_dynamics(0, x)
        else:  # stance
            f = self.slip.stance_dynamics(0, x, u)
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
        """Jacobian of constraints (finite difference)"""
        eps = 1e-8
        c0 = self.constraints(w)
        J = np.zeros((len(c0), len(w)))
        
        for i in range(len(w)):
            wp = w.copy()
            wp[i] += eps
            J[:, i] = (self.constraints(wp) - c0) / eps
        
        return J.ravel()
    
    def update_mode_sequence(self, mode_seq):
        """Update mode sequence for MPC horizon"""
        assert len(mode_seq) == self.H, f"Mode sequence length {len(mode_seq)} != H {self.H}"
        self.mode_seq = list(mode_seq)
    
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
    
    def compute(self, x_current, warm_start=None, return_warm_start=False):
        """
        Solve MPC and return first control input.
        
        Args:
            x_current: Current state [z, z_dot]
            warm_start: Previous solution for warm-starting (optional)
            return_warm_start: If True, return solution for next warm-start
        
        Returns:
            u: First control input (or (u, w_opt) if return_warm_start=True)
        """
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
        nlp.add_option("tol", 1e-4)  # Tighter tolerance
        nlp.add_option("max_iter", 300)  # More iterations
        nlp.add_option("mu_strategy", "adaptive")
        nlp.add_option("bound_relax_factor", 0.0)
        
        w_opt, info = nlp.solve(w0)
        _, U_opt = self._unpack(w_opt)
        
        if return_warm_start:
            return float(U_opt[0, 0]), w_opt
        else:
            return float(U_opt[0, 0])
