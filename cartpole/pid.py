import numpy as np

class PID:
    def __init__(self, kp, ki, kd, dt, umin=-10.0, umax=10.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.dt = dt
        self.umin, self.umax = umin, umax
        self.integral = 0.0
        self.prev_err = 0.0

    def reset(self):
        self.integral = 0.0
        self.prev_err = 0.0

    def update(self, err):
        self.integral += err * self.dt
        derr = (err - self.prev_err) / self.dt
        self.prev_err = err
        u = self.kp * err + self.ki * self.integral + self.kd * derr
        return float(np.clip(u, self.umin, self.umax))