import numpy as np

def flight_state(X, k, l0, m_body, m_leg, g):
    x_b, x_b_dot, x_l, x_l_dot = X
    x_b_ddot = (-m_body * g + k*(l0-x_b+x_l)) / m_body
    x_l_ddot = (-m_leg * g - k*(l0-x_b+x_l)) / m_leg
    return np.array([x_b_dot, x_b_ddot, x_l_dot, x_l_ddot])

def stance_state(X, k, l0, m_body, m_leg, g):
    x_b, x_b_dot, x_l, x_l_dot = X
    F_spring = k * (l0 - (x_b - x_l))
    x_b_ddot = (-m_body * g + F_spring) / m_body

    x_l_ddot = 0
    x_l_dot = 0
    x_l = 0

    return np.array([x_b_dot, x_b_ddot, x_l_dot, x_l_ddot]), F_spring

