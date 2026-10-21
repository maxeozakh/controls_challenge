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
   log_probs, rewards, states, entropies = [], [], [], []
   last_state_Value = None

   obs, _ = env.reset()
   # ===== ROLLOUT ===== #
   for step in range(n_episode_steps):
      state = torch.from_numpy(obs).float()
      logits = policy_model(state)
      probs = torch.softmax(logits, dim=0)
      # TODO: cringe, how to do it simpier? why in this way?
      action = np.random.choice(2, p=probs.detach().numpy())
      log_prob = torch.log(probs[action])
      log_probs.append(log_prob)
      states.append(state)
      obs, reward, terminated, truncated, _ = env.step(action)
      rewards.append(reward)

      entropy = -torch.sum(probs.detach()*torch.log(probs.detach())).item()
      entropies.append(entropy)

      if truncated and not terminated:
         last_obs = torch.from_numpy(obs).float()
         last_state_Value = critic_model(last_obs)
         last_state_Value = last_state_Value.detach().item()
      else:
         last_state_Value = 0
      if terminated or truncated:
         break
   # ================= #
   states = torch.stack(states)
   # V's
   critic_out = critic_model(states)
   critic_out = critic_out.squeeze(-1)


   advantages = []
   gamma = 0.99
   # 0.95 is ~ "listen mostly to the next 10-20 steps"
   lam = 0.95
   critic_out_detached = critic_out.detach()
   last_advantage = 0
   for t in reversed(range(len(rewards))):
      V_t = critic_out_detached[t]
      V_next = critic_out_detached[t+1] if t != len(rewards)-1 else last_state_Value
     
      surprise = rewards[t] + gamma * V_next - V_t
      last_advantage = surprise + gamma * lam * last_advantage
      advantages.append(last_advantage)

   advantages.reverse()
   advantages = torch.stack(advantages)
   value_targets = advantages + critic_out_detached

   # regard to naming, it is a standard RL term:
   # value -- how much return you expect from state S if you just follow policy
   # action value -- -||- if you take a specific action in state S first, then follow policy
   # *advantage* -- action value minus value, so advantage of specific action
   # TODO: understand it numerically
   advantages_norm = (advantages - advantages.mean()) / (advantages.std() + 0.001)

   log_probs = torch.stack(log_probs)

   loss = -(advantages_norm * log_probs).mean()
   loss.backward()

   value_loss = ((critic_out - value_targets)**2).mean()
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
      print(f"{episode + 1:4d} | r={episode_reward:6.1f} | "
            f"entropy={torch.tensor(entropies).mean():.2f} | "f"avg(last 50)={avg:6.1f} | loss={loss.item():+.2f} | value_loss={value_loss.item():.2f}")

env.close()
