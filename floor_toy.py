import numpy as np

# ----------------------------------------------------------------------
# Constants copied in spirit from the real challenge (tinyphysics.py)
# ----------------------------------------------------------------------
T = 400              # number of scored steps (real one is 500-100 = 400)
DEL_T = 0.1          # 10 Hz
LAT_MULT = 50.0      # lataccel cost multiplier in the total
NOISE_STD = 0.044    # the simulator's "dice": jitter added AFTER your steer

# ----------------------------------------------------------------------
# The thing we must follow: a smooth driving maneuver, known in advance.
# (kept smooth so its OWN jerk is tiny, to isolate the noise effect)
# ----------------------------------------------------------------------
t = np.arange(T)
target = 1.5 * np.sin(0.02 * t) + 0.6 * np.sin(0.005 * t)


# ----------------------------------------------------------------------
# THE COST  (identical structure to the challenge)
#   lataccel_cost : how far realized motion is from the target
#   jerk_cost     : how jittery the realized motion is
# ----------------------------------------------------------------------
def cost(target, realized):
    lat = np.mean((target - realized) ** 2) * 100
    jerk = np.mean((np.diff(realized) / DEL_T) ** 2) * 100
    total = lat * LAT_MULT + jerk
    return lat, jerk, total


# ----------------------------------------------------------------------
# THE "CAR" (plant).  Your steering sets the EXPECTED motion.
# Then the simulator rolls a die and adds jitter -> REALIZED motion.
# ----------------------------------------------------------------------
GAIN = 1.0  # steer -> expected lataccel (trivial plant, so we focus on noise/cost)

def plant_step(steer, rng, noise_std):
    expected = GAIN * steer                      # <-- YOUR steering decision lands here
    jitter   = rng.normal(0.0, noise_std)        # <-- RANDOM NOISE, added after you act
    realized = expected + jitter                 # <-- what the cost actually sees
    return realized


# ----------------------------------------------------------------------
# A controller. Perfect feed-forward: steer so expected == target.
# ----------------------------------------------------------------------
def perfect_controller(target_now):
    return target_now / GAIN                     # <-- our steering command


def rollout(noise_std, seed=0):
    rng = np.random.default_rng(seed)
    realized = np.zeros(T)
    for i in range(T):
        steer = perfect_controller(target[i])    # decide
        realized[i] = plant_step(steer, rng, noise_std)  # car responds (+ maybe noise)
    return realized


# ======================================================================
# DEMO 1 — NO dice. Follow the target exactly.
# Only cost left is the target's own movement. Near zero.
# ======================================================================
realized = rollout(noise_std=0.0)
lat, jerk, total = cost(target, realized)
print("1) no noise, follow target exactly:")
print(f"   lat={lat:.3f}  jerk={jerk:.3f}  TOTAL={total:.3f}\n")


# ======================================================================
# DEMO 2 — NO dice, but the CHEATER'S trick: don't follow exactly,
# instead inject the mathematically optimal *smoothed* trajectory.
# This is the closed-form min of:  12.5*sum(c-target)^2 + 25.06*sum(c-c_prev)^2
# It beats demo 1 because shaving jerk is worth more than matching perfectly.
# ======================================================================
A = 100 * LAT_MULT / T          # weight on tracking  -> 12.5
B = 100 / DEL_T**2 / (T - 1)    # weight on jerk      -> 25.06
print(f"   (cost weights:  tracking A={A:.3f},  jerk B={B:.3f})\n")

# Build the linear system (A*I + B*L) c = A*target, L = path Laplacian (the jerk term)
L = np.zeros((T, T))
for i in range(T - 1):
    L[i, i]     += 1; L[i, i+1] -= 1
    L[i+1, i+1] += 1; L[i+1, i] -= 1
c_star = np.linalg.solve(A * np.eye(T) + B * L, A * target)
lat, jerk, total = cost(target, c_star)
print("2) no noise, CHEATER injects optimal smoothed trajectory:")
print(f"   lat={lat:.3f}  jerk={jerk:.3f}  TOTAL={total:.3f}   <- the metric's own floor\n")


# ======================================================================
# DEMO 3 — WITH dice. Perfect steering, but the simulator jitters.
# The jitter you cannot remove blows up the jerk cost: the NOISE FLOOR.
# ======================================================================
totals = []
for seed in range(20):
    realized = rollout(noise_std=NOISE_STD, seed=seed)
    _, _, total = cost(target, realized)
    totals.append(total)
lat, jerk, _ = cost(target, rollout(noise_std=NOISE_STD, seed=0))
print("3) WITH noise, perfect steering (an honest controller's best case):")
print(f"   lat={lat:.3f}  jerk={jerk:.3f}  TOTAL≈{np.mean(totals):.3f}  (+/-{np.std(totals):.3f})")
print("   -> you steered perfectly, yet the dice alone cost you this much.")
