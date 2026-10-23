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
   rollout_log_probs, rewards, states, entropies, actions = [], [], [], [], []
   last_step_Value = None

   obs, _ = env.reset()
   # ===== ROLLOUT ===== #
   for step in range(n_episode_steps):
      with torch.no_grad():
         state = torch.from_numpy(obs).float()
         states.append(state)

         logits = policy_model(state)
         probs = torch.softmax(logits, dim=0)
         action = np.random.choice(2, p=probs.numpy())
         actions.append(action)

         log_prob = torch.log(probs[action])
         rollout_log_probs.append(log_prob)

         obs, reward, terminated, truncated, _ = env.step(action)
         rewards.append(reward)


         if truncated and not terminated:
            last_obs = torch.from_numpy(obs).float()
            last_step_Value = critic_model(last_obs)
            last_step_Value = last_step_Value.item()
         else:
            last_step_Value = 0
         if terminated or truncated:
            break
   # ================= #

   states = torch.stack(states)
   actions = torch.tensor(actions, dtype=torch.long)

   # on the topic of running policy model twice:
   # "A sports analogy: run 1 is playing the match. 
   # Run 2 is watching the recording afterwards and asking, 
   # for each moment, 'how likely was I to make that move, 
   # and should I make it more or less likely?'"
   logits = policy_model(states)
   log_probs = torch.log_softmax(logits, dim=-1)
   actions_log_prob = log_probs[torch.arange(actions.shape[0]), actions]

   # V's
   critic_out = critic_model(states)
   critic_out = critic_out.squeeze(-1)

   


   # 0.95 is ~ "listen mostly to the next 10-20 steps"
   lam = 0.95
   gamma = 0.99
   critic_out_detached = critic_out.detach()
   advantages, last_advantage = [], 0
   for t in reversed(range(len(rewards))):
      V_t = critic_out_detached[t]
      V_next = critic_out_detached[t+1] if t != len(rewards)-1 else last_step_Value
     
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

   policy_loss = -(advantages_norm * actions_log_prob).mean()

   rollout_log_probs = torch.stack(rollout_log_probs) 
   # print(actions_log_prob - rollout_log_probs)

   policy_loss.backward()

   value_loss = ((critic_out - value_targets)**2).mean()
   value_loss.backward()

   policy_optimizer.step()
   policy_optimizer.zero_grad()

   critic_optimizer.step()
   critic_optimizer.zero_grad()

   episode_reward = sum(rewards)
   total_rewards.append(episode_reward)
   recent_rewards.append(episode_reward)

   # we need to adjust that after we moved it out of rollout, right?
   probs = log_probs.detach().exp()
   entropy = -torch.sum(probs*log_probs.detach(), dim=-1).mean().item()
   entropies.append(entropy)

   if len(recent_rewards) > 50:
      recent_rewards.pop(0)
   if (episode + 1) % log_every == 0:
      avg = sum(recent_rewards) / len(recent_rewards)
      print(f"{episode + 1:4d} | r={episode_reward:6.1f} | "
            f"entropy={entropy:.2f} | "f"avg(last 50)={avg:6.1f} | loss={policy_loss.item():+.2f} | value_loss={value_loss.item():.2f}")

env.close()
