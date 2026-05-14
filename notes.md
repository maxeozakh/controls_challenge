# to try
cartpole (because it's easiest)
- PID
- MPC

moonlander
- MPC
- MLP + feature enginnering
- PPO

controlls challenge itself
- all above 

# terminology
- lateral movement -- 
- bicycle model -- 
- autoregressive model --

# model params
- `v_ego`            -- car velocity, aka current speed
- `a_ego`            -- forward acceleration, aka current speed *change*, 
- `road_lataccel == roll_lataccel` -- how much the road is pushes us sideways, aka sideways acceleration
- `current_lataccel` -- how hard the car is turning sideways right now
- `steer_actions`    -- steer input

# data
## state
roll_lataccel, v_ego, a_ego

## future plan
lataccel, roll_lataccel, v_ego, a_ego,

# evals
-- `lataccel_cost` -- how accurate steering, tracking error
-- `jerk_cost`     -- how raw steering

# commands
python tinyphysics.py --model_path ./models/tinyphysics.onnx --data_path ./data/00000.csv --debug --controller pid

# loop

each timestep we receive:
- target_lataccel
- current_lateccel
- state: roll_lataccel, v_ego, a_ego
- future plan: lataccel, roll_lataccel, v_ego, a_ego,
