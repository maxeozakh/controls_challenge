import numpy as np
import torch
from cartpole import CartPoleEnv

state_dim = 4
hidden_dim = 16
action_dim = 2
weights_scale = 0.1
W1 = (torch.randn(state_dim, hidden_dim) * weights_scale).requires_grad_(True)
W2 = (torch.randn(hidden_dim, action_dim) * weights_scale).requires_grad_(True)

env = CartPoleEnv()
n_episodes = 10000
total_rewards = []
recent_rewards = []
log_every = 10
lr = 0.2

for episode in range(n_episodes):
   n_episode_steps = 500
   log_probs = []
   rewards = []
   obs, _ = env.reset()
   for step in range(n_episode_steps):
      state = torch.from_numpy(obs).float()
      h = state @ W1
      hidden = torch.relu(h)
      logits = hidden @ W2
      probs = torch.softmax(logits, dim=0)

      # TODO: cringe, how to do it simpier? why in this way?
      action = np.random.choice(2, p=probs.detach().numpy())
      log_prob = torch.log(probs[action])
      log_probs.append(log_prob)
      obs, reward, terminated, truncated, _ = env.step(action)
      rewards.append(reward)
      if terminated or truncated:
         break

   returns = []
   R = 0.0
   gamma = 0.99
   for r in reversed(rewards):
      R = r + gamma * R
      returns.append(R)
   returns.reverse()

   returns = torch.tensor(returns, dtype=torch.float32)
   returns = (returns - returns.mean()) / (returns.std() + 0.001)

   log_probs_t = torch.stack(log_probs)
   loss = -(returns * log_probs_t).mean()
   loss.backward()

   with torch.no_grad():
      for p in [W1, W2]:
         p -= lr * p.grad
         p.grad.zero_()

   episode_reward = sum(rewards)
   total_rewards.append(episode_reward)
   recent_rewards.append(episode_reward)
   if len(recent_rewards) > 50:
      recent_rewards.pop(0)
   if (episode + 1) % log_every == 0:
      avg = sum(recent_rewards) / len(recent_rewards)
      print(f"episode {episode + 1:4d} | reward={episode_reward:6.1f} | "
            f"avg(last 50)={avg:6.1f} | loss={loss.item():+.4f}")

env.close()
