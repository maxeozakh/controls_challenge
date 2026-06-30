from . import BaseController
import math
import numpy as np

LAT_ACCEL_COST = 50.0
DEL_T = 0.1

class Controller(BaseController):
  """
  A simple PID controller
  """
  def __init__(self,):
    self.p = 0.195
    self.i = 0.100
    self.d = -0.053
    self.error_integral = 0
    self.prev_error = 0
    self.steer_candidates = np.arange(-1,1,.01)
    self.n_steps = 4

    self.k_steer = 1
    self.k_roll = .1
    self.k_speed = 0.005

  def controllable_speed(self, x, k):
    return math.exp(-k*x)
    
  def get_cost(self, l_target, l_current, l_modeled):
    lat_cost = (l_target - l_modeled) ** 2 * 100
    jerk_cost = ((l_modeled - l_current) / DEL_T) ** 2 * 100
    cost = lat_cost * LAT_ACCEL_COST + jerk_cost
    return cost

  def simp_model(self, steer, current_lataccel, roll_lataccel, v_ego):
    delta = (self.k_steer * steer + self.k_roll * roll_lataccel) * self.controllable_speed(v_ego, self.k_speed)
    cand_lataccel = current_lataccel + delta
    return cand_lataccel

  def find_decent_steer(self, target_lataccel, current_lataccel, state, future_plan):
    roll_lataccel, v_ego, a_ego = state
    best_cand = 0.0
    smallest_cost = 9999999999999999
    f_lataccel, f_roll_lataccel, f_v_ego, f_a_ego = future_plan

    N = min(self.n_steps, len(f_lataccel))
    for steer_candidate in self.steer_candidates:
      cost_candidate = 0.0
      l_current = current_lataccel
      l_target = target_lataccel
      l_roll = roll_lataccel

      for k in range(N):
        l_modeled = self.simp_model(steer_candidate, l_current, l_roll, v_ego)
        cost = self.get_cost(l_target, l_current, l_modeled)

        l_current = l_modeled
        l_target = f_lataccel[k]
        l_roll = f_roll_lataccel[k]

        cost_candidate += cost

      best_cand = steer_candidate if cost_candidate < smallest_cost else best_cand
      smallest_cost = cost_candidate if cost_candidate < smallest_cost else smallest_cost 


    return best_cand
      
      

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    f_lataccel, f_roll_lataccel, f_v_ego, f_a_ego = future_plan

    steer = self.find_decent_steer(target_lataccel, current_lataccel, state, future_plan)
    
    return steer

    # error = (target_lataccel - current_lataccel)
    # self.error_integral += error
    # error_diff = error - self.prev_error
    # self.prev_error = error
    # return self.p * error + self.i * self.error_integral + self.d * error_diff
