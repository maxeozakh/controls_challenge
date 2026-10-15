import torch
import numpy as np
import torch.nn as nn
import gymnasium as gym

state_dim = 4
hidden_dim = 16
action_dim = 2
n_episodes = 10000
total_rewards = []
recent_rewards = []
log_every = 10
lr = 1e-2

policy_layers = [nn.Linear(state_dim, hidden_dim),nn.ReLU(),nn.Linear(hidden_dim, action_dim)]
policy_model = nn.Sequential(*policy_layers)
policy_optimizer = torch.optim.Adam(policy_model.parameters(), lr)

critic_layers = [nn.Linear(state_dim, hidden_dim),nn.ReLU(),nn.Linear(hidden_dim, 1)]
critic_model = nn.Sequential(*critic_layers)
critic_optimizer = torch.optim.Adam(critic_model.parameters(), lr)

env = gym.make("CartPole-v1")
for episode in range(n_episodes):
   n_episode_steps = 500
   log_probs = []
   rewards = []
   states = []
   obs, _ = env.reset()
   # ===== ROLLOUT ===== #
   for step in range(n_episode_steps):
      state = torch.from_numpy(obs).float()
      states.append(state)
      logits = policy_model(state)
      probs = torch.softmax(logits, dim=0)

      # TODO: cringe, how to do it simpier? why in this way?
      action = np.random.choice(2, p=probs.detach().numpy())
      log_prob = torch.log(probs[action])
      log_probs.append(log_prob)
      obs, reward, terminated, truncated, _ = env.step(action)
      rewards.append(reward)
      if terminated or truncated:
         break
   # ================= #
   states = torch.stack(states)
   critic_out = critic_model(states)
   critic_out = critic_out.squeeze(-1)


   returns = []
   R = 0.0
   gamma = 0.99
   for r in reversed(rewards):
      R = r + gamma * R
      returns.append(R)
   returns.reverse()

   returns = torch.tensor(returns, dtype=torch.float32)

   # regard to naming, it is a standard RL term:
   # value -- how much return you expect from state S if you just follow policy
   # action value -- -||- if you take a specific action in state S first, then follow policy
   # *advantage* -- action value minus value, so advantage of specific action
   advantage = returns - critic_out.detach()
   # TODO: understand it numerically
   advantage = (advantage - advantage.mean()) / (advantage.std() + 0.001)
   # returns = (returns - returns.mean()) / (returns.std() + 0.001)

   log_probs = torch.stack(log_probs)

   loss = -(advantage * log_probs).mean()
   loss.backward()

   value_loss = ((critic_out - returns)**2).mean()
   value_loss.backward()

   policy_optimizer.step()
   policy_optimizer.zero_grad()

   critic_optimizer.step()
   critic_optimizer.zero_grad()

   episode_reward = sum(rewards)
   total_rewards.append(episode_reward)
   recent_rewards.append(episode_reward)
   if len(recent_rewards) > 50:
      recent_rewards.pop(0)
   if (episode + 1) % log_every == 0:
      avg = sum(recent_rewards) / len(recent_rewards)
      print(f"episode {episode + 1:4d} | reward={episode_reward:6.1f} | "
            f"avg(last 50)={avg:6.1f} | loss={loss.item():+.4f} | value_loss={value_loss.item():.4f}")

env.close()
