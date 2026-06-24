goal is to produce desired lateral acceleration via steering, without crazy turns

potential directions:
* PID + FF as a baseline, to get familiar with the context
* MPC -- seems like dominant, honest approach
* PPO -- just curious to make it work
* evolution -- just curious


# PID + FF as a baseline, to get familiar with the context
## todo
- introduce roll_lataccel

## successes
- tune steering by normalized v_ego 
    - get a slight boost 
## failures
- seems to me that warmup issue is not beatable with PID setup
    - tried to introduce re-init at step 80 to not accumulate data on the fake perfect driving -- metrics goes worse
    ```
    # if self.step == 80:
    #   self.error_integral = 0
    #   self.prev_error = 0
    ```
