"""
view_policy.py — Live viewer for a policy being trained in another terminal.

Watches a weights file on disk and runs rendered episodes using the latest weights.
Run alongside `policy_gradient.py` to watch your policy improve in real time.

Usage:
    python view_policy.py
    python view_policy.py --weights latest_weights.pt --hidden 16
"""
import argparse
import os
import time

import numpy as np
import torch

from cartpole import CartPoleEnv


def policy_forward(state, params):
    """Same forward pass as in policy_gradient.py. Kept here so viewer has no
    dependency on the training script being importable."""
    W1, b1, W2, b2 = params
    h = torch.relu(state @ W1 + b1)
    logits = h @ W2 + b2
    return torch.softmax(logits, dim=-1)


def load_weights(path):
    """Load weight tensors from disk. Returns None if file missing or unreadable."""
    if not os.path.exists(path):
        return None
    try:
        return torch.load(path, weights_only=True)
    except Exception:
        # Possible if we read mid-write despite atomic rename (unlikely but safe).
        return None


def run_demo(params, max_steps=500):
    """Run one rendered episode with the given weights. Returns total reward."""
    env = CartPoleEnv(render_mode="human")
    obs, _ = env.reset()
    total_reward = 0.0
    try:
        with torch.no_grad():
            for _ in range(max_steps):
                state = torch.from_numpy(obs).float()
                probs = policy_forward(state, params)
                action = int(np.argmax(probs.numpy()))  # greedy — see note below
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                if terminated or truncated:
                    break
    finally:
        env.close()
    return total_reward


def main():
    parser = argparse.ArgumentParser(description="View a policy being trained live.")
    parser.add_argument("--weights", type=str, default="latest_weights.pt")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--wait", type=float, default=1.0,
                        help="Seconds to wait between episodes (also lets new weights land).")
    args = parser.parse_args()

    print(f"Viewer waiting for weights file: {args.weights}")
    print("(Run policy_gradient.py in another terminal.)")

    # Wait until the weights file appears
    while not os.path.exists(args.weights):
        time.sleep(0.5)

    episode = 0
    while True:
        episode += 1
        params = load_weights(args.weights)
        if params is None:
            time.sleep(0.5)
            continue
        reward = run_demo(params, max_steps=args.max_steps)
        print(f"Demo episode {episode}: reward = {reward:.1f}")
        time.sleep(args.wait)


if __name__ == "__main__":
    main()