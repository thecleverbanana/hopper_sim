import numpy as np
import cyipopt
import matplotlib.pyplot as plt


# ============================================================
#   更真实的 1D SLIP Hopper：单质点 + 无质量腿
#   状态 X = [x_b, v_b]
#   腿长 = x_b,  弹簧压缩 = max(l0 - x_b, 0)
# ============================================================
class RealSLIP_Hopper:
    def __init__(self, m=1.0, l0=0.4, k=1500.0, g=9.81):
        self.m  = m
        self.l0 = l0
        self.k  = k
        self.g  = g

    # -----------------------------------------
    # 连续动力学（contact-implicit）
    # -----------------------------------------
    def dynamics(self, X, u):
        x, v = X
        m, g, k, l0 = self.m, self.g, self.k, self.l0

        # 弹簧压缩：只在 x < l0 时起作用
        compression = max(l0 - x, 0.0)
        F_spring = k * compression

        # 垂直加速度：重力 + 弹簧 + 控制力
        a = (-m * g + F_spring + u) / m

        return np.array([v, a])

    # -----------------------------------------
    # 前向积分（Euler）
    # -----------------------------------------
    def step(self, X, u, dt):
        dX = self.dynamics(X, u)
        Xn = X + dX * dt

        # 防止数值误差穿地
        if Xn[0] < 0.0:
            Xn[0] = 0.0
            Xn[1] = max(0.0, Xn[1])

        return Xn

    # ============================================================
    #  内嵌 NMPC 控制器（IPOPT）
    # ============================================================
    class MPC(cyipopt.Problem):
        def __init__(
            self,
            hopper,
            H,
            dt,
            x0,
            u_min=-300.0,
            u_max=300.0,
            R_u=1e-3,
            Q_x=30.0,
            Q_v=5.0,
            Q_xT=80.0,
            Q_vT=10.0,
            x_target=0.6,
        ):
            """
            OCP:
              min ∑ (R_u u^2 + Q_x (x-x_ref)^2 + Q_v v^2) + 终端项
              s.t.  x_{k+1} = x_k + dt * f(x_k, u_k)
            """

            self.hopper   = hopper
            self.H        = H
            self.dt       = dt
            self.nx       = 2
            self.nu       = 1

            self.R_u  = R_u
            self.Q_x  = Q_x
            self.Q_v  = Q_v
            self.Q_xT = Q_xT
            self.Q_vT = Q_vT
            self.x_target = x_target

            # ---------- 决策变量 ----------
            # X: (H+1)*2, U: H*1  （这里控制只用前 H 步）
            self.Nx_total = (H + 1) * self.nx
            self.Nu_total = H * self.nu
            n_var = self.Nx_total + self.Nu_total

            # 约束：H 个 step，每个 2 维
            m_constr = H * self.nx

            self.n = n_var
            self.m = m_constr

            # ---------- 变量上下界 ----------
            lb = -1e6 * np.ones(n_var)
            ub =  1e6 * np.ones(n_var)

            # 固定初始状态
            lb[0:self.nx] = x0
            ub[0:self.nx] = x0

            # 控制量上下界
            u_start = self.Nx_total
            lb[u_start:] = u_min
            ub[u_start:] = u_max

            # 约束上下界（defect = 0）
            cl = np.zeros(m_constr)
            cu = np.zeros(m_constr)

            self.lb = lb
            self.ub = ub
            self.cl = cl
            self.cu = cu

            super().__init__(n=n_var, m=m_constr, lb=lb, ub=ub, cl=cl, cu=cu)

        # -----------------------------------------
        def _unpack(self, w):
            X = w[:self.Nx_total].reshape(self.H + 1, self.nx)
            U = w[self.Nx_total:].reshape(self.H, self.nu)
            return X, U

        # -----------------------------------------
        def objective(self, w):
            X, U = self._unpack(w)
            dt = self.dt
            H  = self.H

            J = 0.0

            for k in range(H):
                x, v = X[k]
                u    = U[k, 0]

                J += dt * (
                    self.R_u * u**2 +
                    self.Q_x * (x - self.x_target)**2 +
                    self.Q_v * v**2
                )

            # 终端项
            xT, vT = X[H]
            J += self.Q_xT * (xT - self.x_target)**2
            J += self.Q_vT * vT**2

            return float(J)

        # -----------------------------------------
        def gradient(self, w):
            # 为简单起见，直接用有限差分
            eps = 1e-8
            grad = np.zeros_like(w)
            f0 = self.objective(w)
            for i in range(len(w)):
                wp = w.copy()
                wp[i] += eps
                grad[i] = (self.objective(wp) - f0) / eps
            return grad

        # -----------------------------------------
        def constraints(self, w):
            X, U = self._unpack(w)
            dt = self.dt
            H  = self.H
            nx = self.nx

            c = np.zeros(self.m)

            for k in range(H):
                xk = X[k]
                xkp1 = X[k+1]
                u = U[k, 0]

                fk = self.hopper.dynamics(xk, u)
                xk_euler = xk + dt * fk

                defect = xkp1 - xk_euler
                c[k*nx:(k+1)*nx] = defect

            return c

        # -----------------------------------------
        def jacobian(self, w):
            # 同样用有限差分
            eps = 1e-8
            c0 = self.constraints(w)
            J = np.zeros((len(c0), len(w)))
            for i in range(len(w)):
                wp = w.copy()
                wp[i] += eps
                J[:, i] = (self.constraints(wp) - c0) / eps
            return J

        # -----------------------------------------
        def compute(self, x0, warm=None):
            # 更新初始状态的 bound
            self.lb[0:self.nx] = x0
            self.ub[0:self.nx] = x0

            if warm is None:
                w0 = np.zeros(self.n)
                w0[:self.nx] = x0
            else:
                w0 = warm

            solver = cyipopt.Problem(
                n=self.n, m=self.m,
                problem_obj=self,
                lb=self.lb, ub=self.ub,
                cl=self.cl, cu=self.cu
            )

            solver.add_option("print_level", 0)
            solver.add_option("tol", 1e-3)
            solver.add_option("max_iter", 80)
            solver.add_option("hessian_approximation", "limited-memory")

            w_opt, info = solver.solve(w0)
            X_opt, U_opt = self._unpack(w_opt)

            return float(U_opt[0, 0]), w_opt


# ============================================================
#                      DEMO: 跳跃仿真
# ============================================================
if __name__ == "__main__":
    hopper = RealSLIP_Hopper(
        m=1.0,
        l0=0.4,
        k=1500.0,
        g=9.81
    )

    dt = 0.005
    H  = 30

    # 初始位置接近自然腿长稍高一点
    x = np.array([0.45, 0.0])   # [height, velocity]

    # 目标 apex 高度（比 l0 高不少）
    x_target = 0.6

    mpc = RealSLIP_Hopper.MPC(
        hopper=hopper,
        H=H,
        dt=dt,
        x0=x,
        u_min=-400.0,
        u_max=400.0,
        R_u=1e-3,
        Q_x=20.0,
        Q_v=5.0,
        Q_xT=80.0,
        Q_vT=20.0,
        x_target=x_target,
    )

    N = 1000
    warm = None

    x_hist = []
    v_hist = []
    u_hist = []
    contact_hist = []

    for k in range(N):
        u, warm = mpc.compute(x, warm)

        # 一步真实仿真
        x = hopper.step(x, u, dt)

        x_hist.append(x[0])
        v_hist.append(x[1])
        u_hist.append(u)

        # 是否接触地面：x < l0 即认为在 stance
        contact_hist.append(1 if x[0] < hopper.l0 else 0)

    t = np.arange(N) * dt

    plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 1)
    plt.plot(t, x_hist, label="body height")
    plt.axhline(hopper.l0, linestyle="--", label="leg rest length")
    plt.axhline(x_target, linestyle=":", label="target height")
    plt.legend()
    plt.ylabel("x [m]")

    plt.subplot(3, 1, 2)
    plt.plot(t, u_hist)
    plt.ylabel("u [N]")

    plt.subplot(3, 1, 3)
    plt.step(t, contact_hist, where="post")
    plt.ylabel("contact")
    plt.xlabel("time [s]")

    plt.tight_layout()
    plt.show()
