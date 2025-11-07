import numpy as np

def hopper_compression(t, x, m_b, m_l, L0, k, g):
    yb, yb_dot, _, _ = x
    spring_force = k * (yb - L0)           # negative when compressed
    yb_ddot = -spring_force / m_b - g      # negative spring_force → upward accel
    return [yb_dot, yb_ddot, 0, 0]


def hopper_flight(t, x, m_b, m_l, L0, k, g):
    yb, yb_dot, yl, yl_dot = x
    yb_ddot = -g
    yl_ddot = -g
    return [yb_dot, yb_ddot, yl_dot, yl_ddot]


def hopper_uploading(t, x, m_b, m_l, L0, k, g):
    yb, yb_dot, yl, yl_dot = x
    spring_force = k * (L0 - (yb - yl))    # positive when compressed
    yb_ddot =  spring_force / m_b - g
    yl_ddot = -spring_force / m_l - g
    return [yb_dot, yb_ddot, yl_dot, yl_ddot]


