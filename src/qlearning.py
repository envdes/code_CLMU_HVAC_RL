import gymnasium as gym
import numpy as np
import random
import pickle
import time
import os
from typing import Tuple, Callable
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict

"""
example of a Q-learning agent that uses a discretized observation space
"""
def discretize(obs):
    """
    Discretize the observation space and convert it into a 1D index.
    """
    obs[2] = obs[2] * 10
    obs = np.round(obs)    # 对 obs 的所有元素取整

    index = (obs[0] * 100000 +
             obs[1] * 10000 +
             obs[3] * 100 +
             obs[4] * 10 +
             obs[2] * 1
             )
    return index

class QLearning:
    def __init__(self, 
                 exp_name: str = os.path.basename(__file__)[: -len(".py")],
                 env_id: str = "CartPole-v1",
                 learning_rate: float = 0.1,
                 gamma: float = 0.99, 
                 epsilon: float = 1.0, 
                 max_epsilon: float = 1.0, 
                 min_epsilon: float = 0.01, 
                 decay_rate: float = 0.01,
                 discretize: Callable = discretize,
                 track: bool = False,
                 project_name: str = "qlearning",
                 wandb_entity: str = None,
                 seed: int = 1,
                 ):
        
        """
        Args:
            env: gym.Env
                The environment to train the agent on
            bins: tuple
                The number of bins to discretize the observation space into
            learning_rate: float
                The learning rate for updating the Q-table
            gamma: float
                The discount rate for future rewards
            epsilon: float
                The exploration rate
            max_epsilon: float
                The maximum exploration rate
            min_epsilon: float
                The minimum exploration rate
            decay_rate: float
                The decay rate for the exploration rate
        
        """
        
        self.env_id = env_id
        self.env = gym.make(self.env_id)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate
        self.track = track
        self.project_name = project_name
        self.exp_name = exp_name
        self.wandb_entity = wandb_entity
        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        self.q_table = defaultdict(lambda: np.zeros(self.env.action_space.n))
        #self.q_table = np.zeros(bins + (self.env.action_space.n,))
        self.discretize = discretize if discretize is not None else None

    def choose_action(self, state):
        discretized_state = self.discretize(state)
        if np.random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.q_table[discretized_state])

    def update_q_table(self, state, action, reward, next_state):
        discretized_state = self.discretize(state)
        discretized_next_state = self.discretize(next_state)
        best_next_action = np.argmax(self.q_table[discretized_next_state])
        td_target = reward + self.gamma * self.q_table[discretized_next_state][best_next_action]
        td_error = td_target - self.q_table[discretized_state][action]
        self.q_table[discretized_state][action] += self.learning_rate * td_error
        #td_target = reward + self.gamma * self.q_table[discretized_next_state + (best_next_action,)]
        #td_error = td_target - self.q_table[discretized_state + (action,)]
        #self.q_table[discretized_state + (action,)] += self.learning_rate * td_error

    def decay_epsilon(self, episode):
        self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.decay_rate * episode)

    def train(self, num_episodes=10000, max_steps=200000):
        
        run_name = f"{self.project_name}/{self.env_id}__{self.exp_name}__{self.seed}__{int(time.time())}"
        if self.track:
            import wandb
            wandb.init(
                project=self.project_name,
                entity=self.wandb_entity,
                sync_tensorboard=True,
                config=vars(self),
                name=run_name,
                monitor_gym=True,
                save_code=True,
            )
        writer = SummaryWriter(f"tensorboard/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(self).items()])),
        )
        global_step = 0
        
        for episode in range(num_episodes):
            state,_ = self.env.reset()
            rewards = 0
            for step in range(max_steps):
                global_step += 1
                action = self.choose_action(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                self.update_q_table(state, action, reward, next_state)
                state = next_state
                rewards += reward
                if terminated or truncated:
                    writer.add_scalar("charts/episodic_return", rewards, global_step)
                    writer.add_scalar("charts/episodic_length", step+1, global_step)
                    break

            self.decay_epsilon(episode)
            
        self.evaluate(num_episodes=3, max_steps=200000, writer=writer)
        
        self.model_path = f"q_table/{run_name}/{self.exp_name}.pkl"
        if not os.path.exists(f"q_table/{run_name}"):
            os.makedirs(f"q_table/{run_name}", exist_ok=True)
        self.save_q_table(self.model_path)
        
        writer.close()
        self.env.close()
        if self.track:
            wandb.finish()
        return self.model_path

    def evaluate(self, num_episodes=3, max_steps=200000, writer=None):
        
        for episode in range(num_episodes):
            total_rewards = 0
            state,_ = self.env.reset()
            for step in range(max_steps):
                action = self.choose_action(state)
                state, reward, terminated, truncated, info = self.env.step(action)
                total_rewards += reward
                if terminated or truncated:
                    if writer is not None:
                        writer.add_scalar("EvalRewards", total_rewards, episode)
                    break

        average_reward = total_rewards / num_episodes
        print(f"Average reward: {average_reward}")

    def print_q_table(self):
        print("Final Q-Table Values")
        print(self.q_table)
        
    def save_q_table(self, filename):
        #np.save(filename, self.q_table)
        with open(filename, 'wb') as f:
            pickle.dump(dict(self.q_table), f)
        
    def load_q_table(self, filename):
        #self.q_table = np.load(filename)
        self.q_table = defaultdict(lambda: np.zeros(self.env.action_space.n))
        with open(filename, 'rb') as f:
            self.q_table.update(pickle.load(f))