"""
REINFORCE on CartPole, vanilla policy gradient 

rawly:
1. Create the environment
2. Create the policy network (4 inputs → 2 outputs = action probabilities)
3. Create an optimizer (Adam, SGD, whatever)
4. Repeat for many episodes:
   a. Reset the environment, get initial observation
   b. Play one episode:
       - For each step:
         * Pass observation through policy network → get probabilities
         * Sample an action from those probabilities
         * Take the action, observe (reward, next observation, done)
         * Remember: (the log-probability of the action we took, the reward)
   c. Compute the discounted returns R_t for each step
   d. (Optionally) standardize returns
   e. Compute the loss = -mean( log_p(action) * R )
   f. Backprop, take an optimizer step
   g. Print episode reward, repeat
"""

import argparse
from typing import Optional

import numpy as np
import torch

from cartpole import CartPoleEnv


def make_policy(state_dim: int = 4, hidden_dim: int = 16,
               action_dim: int = 2, seed: int = 0):
   """
   prep policy's weight tensors
   """
   g = torch.Generator().manual_seed(seed)

   # layer 1: state 4 -> hidden 16
   W1 = torch.randn(state_dim, hidden_dim, generator=g) * 0.1
   b1 = torch.zeros(hidden_dim)

   # layer 2: hidden 16 -> logits 2 
   W2 = torch.randn(hidden_dim, action_dim, generator=g) * 0.1
   b2 = torch.zeros(action_dim)

   params = [W1, b1, W2, b2]
   for p in params:
      p.requires_grad_(True)

   return params

def policy_forward(state: torch.Tensor, params: list) -> torch.Tensor:
   """
   forward pass, returns action probabilities
   shape (2, ) -- probability of each action, summing to 1
   """
   W1, b1, W2, b2 = params

   # layer 1: linear + relu
   h = state @ W1 + b1
   h = torch.relu(h)

   # layer 2: linear -> logits
   logits = h @ W2 + b2 

   probs = torch.softmax(logits, dim=-1)
   return probs

def run_episode(env: CartPoleEnv, params: list, max_steps: int = 500, seed: Optional[int] = None):
   obs, _ = env.reset(seed=seed)
   log_probs = []
   rewards = []

   for _ in range(max_steps):
      state = torch.from_numpy(obs).float()

      probs = policy_forward(state, params)

      action = np.random.choice(2, p=probs.detach().numpy())

      log_prob = torch.log(probs[action])
      log_probs.append(log_prob)

      obs, reward, terminated, truncated, _ = env.step(action)

      rewards.append(reward)

      if terminated or truncated:
         break

   return log_probs, rewards

def compute_returns(rewards: list, gamma: float = 0.99) -> torch.Tensor:
   """
   compute weighted list of rewards
   """

   returns = []
   R = 0.0
   for r in reversed(rewards):
      R = r + gamma * R
      returns.append(R)
   returns.reverse()

   return torch.tensor(returns, dtype=torch.float32)

def standardize(returns: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
   """
   encouraging and discouraging roughly half of the performed actions
   """
   return (returns - returns.mean()) / (returns.std() + eps)

def get_loss(log_probs: list, returns: torch.Tensor) -> torch.Tensor:
   log_probs_t = torch.stack(log_probs)
   loss = -(returns * log_probs_t).mean()
   return loss

def compute_loss(log_probs: list, returns: torch.Tensor) -> torch.Tensor:
   log_probs_t = torch.stack(log_probs)
   loss = -(returns * log_probs_t).mean()
   return loss

def train(
   num_episodes: int = 1000,
   lr: float = 0.01,
   gamma: float = 0.99,
   hidden_dim: int = 16,
   max_steps: int = 500,
   seed: int = 0,
   log_every: int = 10,
   render_every: int = 0
) -> list:
   env = CartPoleEnv()
   params = make_policy(state_dim=4, hidden_dim=hidden_dim, action_dim=2,seed=seed)

   episode_rewards = []
   recent_rewards = []

   for ep in range(num_episodes):
      log_probs, rewards = run_episode(env, params, max_steps=max_steps)
      returns = compute_returns(rewards, gamma=gamma)
      returns = standardize(returns)

      loss = compute_loss(log_probs, returns)

      loss.backward()

      with torch.no_grad():
         for p in params:
            p -= lr * p.grad
            p.grad.zero_()
      
      total_reward = sum(rewards)
      episode_rewards.append(total_reward)
      recent_rewards.append(total_reward)

      if len(recent_rewards) > 50:
         recent_rewards.pop(0)
      
      if (ep + 1) % log_every == 0:
         avg = sum(recent_rewards) / len(recent_rewards)
         print(f"Episode {ep + 1:4d} | reward={total_reward:6.1f} | avg(last 50)={avg:6.1f} | loss={loss.item():+.4f}")
      
   save_weights(params)
   env.close()
   return episode_rewards

def save_weights(params, path="latest_weights.pt"):
    """Save current policy weights atomically (avoid half-written reads)."""
    # Strip out the gradients; we only want the values.
    state = [p.detach().clone() for p in params]
    # Write to a temp path, then rename — guarantees the viewer never reads a partial file.
    tmp_path = path + ".tmp"
    torch.save(state, tmp_path)
    import os
    os.replace(tmp_path, path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train REINFORCE on CartPole.")
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train(
        num_episodes=args.episodes,
        lr=args.lr,
        gamma=args.gamma,
        hidden_dim=args.hidden,
        seed=args.seed,
    )