# # # # import numpy as np

# # # # state_space = [(i, j) for i in range(3) for j in range(3)]
# # # # val_pi_s = {}
# # # # rewards = {(0,2): 1, (2,2):-1}
# # # # episode_lst = []
# # # # num_visits = {}

# # # # print(state_space)
# # # # for liner_st in state_space:
# # # #     for state in liner_st:
# # # #         num_visits[state] = 0
# # # #         val_pi_s[state] = 0
        
# # # #         if state in rewards.keys():
# # # #             val_pi_s[state] = rewards.get(state)
            
            
# # # # print(num_visits, val_pi_s)


# # # # state = [(row, col) for row in range(5) for col in range(5) ]
# # # # curr_state = (0,1)

# # # # grid_display = []

# # # # for i, cell in enumerate(state):
# # # #     if cell == curr_state:
# # # #         grid_display.append(" A ")
# # # #     elif cell == (0, 0):
# # # #         grid_display.append(" S ")
# # # #     elif cell == (4, 4):
# # # #         grid_display.append(" G ")
# # # #     else:
# # # #         grid_display.append(" . ")
        
# # # #     if (i + 1) % 5 == 0:
# # # #         print("".join(grid_display))
# # # #         grid_display = []



# # # """
# # # # = Done

# # # To Do:
# # # # 1. K armed bandit
# # # # 2. MDP
# # # # 3. Monte Carlo
# # #     # MC PRED
# # #     # MC CONTROL
# # # # 4. Temporal Difference (TD)
# # # """


# # from collections import defaultdict
# # import numpy as np

# # class MonteCarloControl:
# #     def __init__(self, env, gamma=0.9):
# #         self.env = env
# #         self.gamma = gamma
# #         self.policy_selector = PolicySelection(self.env.action_space, "e-greedy", epsilon=0.5)
# #         self.Q = defaultdict(lambda: np.zeros(len(self.env.action_space)))
# #         self.C = defaultdict(lambda: np.zeros(len(self.env.action_space)))
# #         self.policy = defaultdict(int)

# #     def gen_episode(self, policy_type):
# #         episode = []
# #         state = self.env.reset()  # Must return (0,0)
# #         while True:
# #             action = self.policy_selector.select_action(state, policy_type, self.Q)
# #             next_state, reward, done = self.env.step(action)
# #             episode.append((state, action, reward))
# #             state = next_state
# #             if done:
# #                 break
# #         return episode

# #     def off_policy_mc_control(self, num_episodes=10000):
# #         for _ in range(num_episodes):
# #             episode = self.gen_episode(policy_type="e-greedy")
# #             G = 0.0
# #             W = 1.0

# #             for t in reversed(range(len(episode))):
# #                 state, action_idx, reward = episode[t]
# #                 G = self.gamma * G + reward

# #                 # Update C and Q
# #                 self.C[state][action_idx] += W
# #                 self.Q[state][action_idx] += (W / self.C[state][action_idx]) * (G - self.Q[state][action_idx])

# #                 # Update target policy (greedy)
# #                 best_action_idx = np.argmax(self.Q[state])
# #                 self.policy[state] = best_action_idx

# #                 # Stop if action differs from target policy
# #                 if action_idx != self.policy[state]:
# #                     break

# #                 # Compute behavior policy probability
# #                 b_a_s = self.policy_selector.get_policy_probab(state, action_idx, self.Q)
# #                 if b_a_s == 0:
# #                     break

# #                 # Update importance sampling weight
# #                 print(f"b_a_s: {b_a_s}, W: {W}")  # Debug print
# #                 W *= 1.0 / b_a_s

# #         print(self.policy, self.Q)
# #         return self.policy, self.Q

# #     def print_q_policy(self):
# #         grid_size = self.env.size
# #         action_map = {0: '↑', 1: '↓', 2: '→', 3: '←'}
# #         for row in range(grid_size):
# #             row_str = []
# #             for col in range(grid_size):
# #                 state = (row, col)
# #                 q_values = self.Q[state]
# #                 best_action = np.argmax(q_values)
# #                 row_str.append(action_map[best_action])
# #             print(" | ".join(row_str))
# #         print("-" * (4 * grid_size))

# # class PolicySelection:
# #     def __init__(self, action_space, policy_type, epsilon=0.1):
# #         self.action_space = action_space
# #         self.epsilon = epsilon
# #         self.policy_type = policy_type

# #     def select_action(self, state, policy_type, Q):
# #         if policy_type == "e-greedy":
# #             if np.random.rand() < self.epsilon:
# #                 return np.random.choice(self.action_space)
# #             return np.argmax(Q[state])
# #         raise ValueError("Policy not supported")

# #     def get_policy_probab(self, state, action, Q):
# #         num_actions = len(self.action_space)
# #         best_action = np.argmax(Q[state])
# #         if action == best_action:
# #             return 1 - self.epsilon + self.epsilon / num_actions
# #         return self.epsilon / num_actions

# # class GridEnv:
# #     def __init__(self, size=5):
# #         self.size = size
# #         self.action_space = [0, 1, 2, 3]  # ↑, ↓, →, ←
# #         self.goal = (4, 4)
# #         self.state = (0, 0)

# #     def reset(self):
# #         self.state = (0, 0)
# #         return self.state

# #     def step(self, action):
# #         row, col = self.state
# #         if action == 0:  # ↑
# #             next_row = max(0, row - 1)
# #             next_col = col
# #         elif action == 1:  # ↓
# #             next_row = min(self.size - 1, row + 1)
# #             next_col = col
# #         elif action == 2:  # →
# #             next_row = row
# #             next_col = min(self.size - 1, col + 1)
# #         elif action == 3:  # ←
# #             next_row = row
# #             next_col = max(0, col - 1)
# #         self.state = (next_row, next_col)
# #         reward = 1.0 if self.state == self.goal else -0.01
# #         done = self.state == self.goal
# #         return self.state, reward, done

# # # Test the implementation
# # if __name__ == "__main__":
# #     env = GridEnv()
# #     agent = MonteCarloControl(env)
# #     policy, Q = agent.off_policy_mc_control(num_episodes=10000)
# #     agent.print_q_policy()
























# # import os
# # import sys
# # import numpy as np
# # from collections import defaultdict

# # os.makedirs("Monte_Carlo/Results", exist_ok=True)
# # from utils.policy import PolicySelection

# # sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# # class MonteCarloControl:
# #     def __init__(self, env, gamma = 0.9, epsilon= 0.9):
# #         self.env = env                                          # Environment
# #         policy_type = "e-greedy"
# #         self.policy_selector = PolicySelection(action_space=self.env.action_space,policy_type=policy_type, epsilon=0.9)
# #         self.gamma = gamma                                      # Discount Factor
# #         self.returns = defaultdict(list)                        # Stores returns after averaging
# #         self.C = defaultdict(float)                             # Stores sum of weighted returns
# #         self.num_action = len(self.env.action_space)            # Number of possible action
# #         self.Q = defaultdict(lambda: np.zeros(self.num_action)) # Stores Q Values
# #         self.policy = defaultdict(lambda: np.random.randint(self.num_action))
        
        
# #     def action_selector(self, state, policy_type, Q_Value=None):
# #         """ Code to Select Policy Type"""
# #         return self.policy_selector.select_action(state, policy_type, Q_Value)
        
            
# #     def gen_episode(self, policy_type):
# #         episode = []
# #         state = self.env.reset()
        
# #         while True:
# #             action = self.action_selector(state, policy_type, self.Q)
                                
# #             next_state, reward, done = self.env.step(self.env.action_space.index(action))
# #             episode.append((state, action, reward))
# #             state = next_state
            
# #             if done:
# #                 break
            
# #         return episode
    
# #     def mc_control_exploring_start(self, num_episodes = 1000):
# #         for _ in range(num_episodes):
            
# #             ## Can update it to choose random action too #Default (0, 0)
# #             # start = self.env.reset()            
# #             # action = np.random.randint(self.num_action)
# #             episode = self.gen_episode(policy_type="random")
# #             G = 0
# #             visited_state = set()

# #             for t in reversed(range(len(episode))):
# #                 state, action, reward = episode[t]
# #                 G = self.gamma * G + reward
# #                 action_idx = self.env.action_space.index(action)
                
# #                 if (state, action_idx) not in visited_state:
# #                     visited_state.add((state, action_idx))
# #                     self.returns[(state, action_idx)].append(G)
# #                     self.Q[(state, action_idx)] = np.mean(self.returns[(state, action_idx)])
# #                     self.policy[state] = np.argmax(self.Q[(state, action_idx)])
            
# #         return self.Q
            
            
# #     def print_q_policy(self):
# #         grid_size = self.env.size
# #         action_map = {0: '↑', 1: '↓', 2: '→', 3: '←'}
# #         for row in range(grid_size):
# #             row_str = []
# #             for col in range(grid_size):
# #                 state = (row, col)
# #                 q_values = self.Q[state]
# #                 best_action = np.argmax(q_values)
# #                 row_str.append(action_map[best_action])
# #             print(" | ".join(row_str))
# #         print("-" * (4 * grid_size))


# import os, sys
# import importlib

# module_name = "phase_05_actor_critic.3_redq.REDQ"
# mod_ = importlib.import_module(module_name)
# print(mod_)



from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

from isaacsim.asset