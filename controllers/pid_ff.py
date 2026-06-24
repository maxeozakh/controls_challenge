import math
from statistics import mean
from controllers import BaseController

class Controller(BaseController):
  """
  PID + future forward controller
  """
  def __init__(self,):
    self.p = 0.195
    self.i = 0.100
    self.d = -0.053
    self.error_integral = 0
    self.prev_error = 0
    
    self.k_f_lataccel = 0.4
    self.k_f_roll = 0.3
    self.k_roll = 0.4
    self.n_future = 3

    self.step = 0 

  def controllable_speed(self, x, k=0.0005):
    return math.exp(-k*x)

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    roll_lataccel, v_ego, a_ego = state
    f_lataccel, f_roll_lataccel, f_v_ego, f_a_ego = future_plan
    # if self.step == 80:
    #   self.error_integral = 0
    #   self.prev_error = 0

    error = (target_lataccel - current_lataccel)
    self.error_integral += error
    error_diff = error - self.prev_error
    self.prev_error = error

    # us
    v_ego_norm = self.controllable_speed(v_ego)
    steer_pid = (self.p * error * v_ego_norm + self.i * self.error_integral + self.d * error_diff)

    avg_f_lataccel= 0 if len(f_lataccel) < self.n_future else mean(f_lataccel[:self.n_future])
    avg_f_roll_lataccel = 0 if len(f_roll_lataccel) < self.n_future else mean(f_roll_lataccel[:self.n_future])

    steer_f_roll = self.k_f_roll * avg_f_roll_lataccel
    steer_ff = self.k_f_lataccel * avg_f_lataccel - steer_f_roll

    steer_roll = self.k_roll * roll_lataccel
    steer = steer_ff + steer_pid - steer_roll

    self.step += 1
    return steer