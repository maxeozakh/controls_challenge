from . import BaseController
import random
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
    self.steer_candidates = np.arange(-1,1,.02)

  def simp_model(self, steer, target_lataccel, current_lataccel):
    cand_lataccel = current_lataccel + steer
    lat_cost = (target_lataccel - cand_lataccel) ** 2 * 100
    jerk_cost = ((cand_lataccel - current_lataccel) / DEL_T) ** 2 * 100
    cost = lat_cost * LAT_ACCEL_COST + jerk_cost
    return cost

  def pick_best_candidate(self, target_lataccel, current_lataccel):
    best_cand = 0.0
    smallest_cost = 9999999999999999

    for c in self.steer_candidates:
      cost = self.simp_model(c, target_lataccel, current_lataccel)
      best_cand = c if cost < smallest_cost else best_cand
      smallest_cost = cost if cost < smallest_cost else smallest_cost 

    return best_cand
      
      

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    roll_lataccel, v_ego, a_ego = state
    f_lataccel, f_roll_lataccel, f_v_ego, f_a_ego = future_plan

    steer = self.pick_best_candidate(target_lataccel, current_lataccel)
    
    return steer

    # error = (target_lataccel - current_lataccel)
    # self.error_integral += error
    # error_diff = error - self.prev_error
    # self.prev_error = error
    # return self.p * error + self.i * self.error_integral + self.d * error_diff
