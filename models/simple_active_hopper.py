import numpy as np

def flight_state(X, m_body, m_leg, g, u):
    x_b, x_b_dot, x_l, x_l_dot = X
    x_b_ddot = (-m_body * g + u) / m_body
    x_l_ddot = (-m_leg * g - u) / m_leg
    return np.array([x_b_dot, x_b_ddot, x_l_dot, x_l_ddot]), 0.0  # F_sub = 0 in flight


def stance_state(X, m_body, m_leg, g, u):
    x_b, x_b_dot, x_l, x_l_dot = X

    # RIGID GROUND: foot fixed at x_l = 0
    x_l = 0.0
    x_l_dot = 0.0
    x_b_ddot = (-m_body * g + u) / m_body

    # constraint equation on leg: m_l * 0 = -m_l g - u + F_sub
    F_sub = m_leg * g + u

    x_l_ddot = 0.0

    return np.array([x_b_dot, x_b_ddot, x_l_dot, x_l_ddot]), F_sub

class PDController:
    def __init__(self, kp, kd):
        self.kp = kp
        self.kd = kd

    def compute(self, X, l_ref, ldot_ref):
        x_b, x_b_dot, x_l, x_l_dot = X
        l = x_b - x_l
        ldot = x_b_dot - x_l_dot
        error = l_ref - l
        derror = ldot_ref - ldot
        return self.kp * error + self.kd * derror