# this is a PPO implementation for continuous action space environments
# the algorithm is based on the paper https://arxiv.org/abs/1707.06347
# and implementation is based on the Cleanrl library https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py
# the modification using mo_gymnasium is adapted from the morl-baselines library https://github.com/LucasAlegre/morl-baselines

# ! Junjie Yu makes the following changes to the original code:
# 2024/07/17
# 1 make value network to be a multi output network
# 2 use the reward_dim to adapt to multi output value network
# 3 use mo_gymnasium to wrap the mo envs
# 4 add reward_dim and reward_weight to the class
# 5 delete the print info
# 6 add the interation mean episodic return to the tensorboard and wandb

# import python base libraries
import os
import random
import time
from dataclasses import dataclass
from typing import Callable, Union

# import third party libraries
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
from src.utils import log_config

# ! JJ
def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        if capture_video and idx == 0:
            env = env = gym.make(env_id, render_mode="rgb_array")
        else:
            env = env = gym.make(env_id)
        # ! JJ
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        # ! JJ
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
        # ! JJ
    return thunk

# ! JJ
def evaluate(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    Model: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
    capture_video: bool = False, # ! JJ
    gamma: float = 0.99
):
    envs = envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, capture_video, run_name, gamma)])
    agent = Model(envs).to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()

    obs, _ = envs.reset()
    episodic_returns = []
    ACTIONS = []
    while len(episodic_returns) < eval_episodes:
        obs, _ = envs.reset()
        dis_vec = 0
        gamma = gamma
        done = False
        A = []
        while not done:
            actions, _, _, _ = agent.get_action_and_value(torch.Tensor(obs).to(device))
            next_obs, rewards, termination, truncation, infos = envs.step(actions.cpu().numpy())
            done = np.logical_or(termination, truncation)
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if "episode" not in info:
                        continue
                    #print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
                    episodic_returns += [info["episode"]["r"]]
            dis_vec += rewards.ravel() * gamma
            gamma *= gamma
            obs = next_obs
            A += [actions.cpu().numpy()]
            
        ACTIONS += [A]
    return (np.array(episodic_returns, dtype=np.float32),
           np.array(ACTIONS, dtype=np.float32))
# ! JJ


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class RLAgent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            # ! JJ
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

class PPO_continuous_action():
    def __init__(self,
        exp_name: str = os.path.basename(__file__)[: -len(".py")],
        agent_id: int = 0,
        seed: int = 1,
        torch_deterministic: bool = True,
        cuda: bool = True,
        track: bool = False,
        project_name: str = "CLMU",
        wandb_entity: str = None,
        capture_video: bool = False,
        save_model: bool = False,
        env_id: str = "CLMU-v0",
        total_timesteps: int = 1000000,
        learning_rate: float = 3e-4,
        num_envs: int = 1,
        num_steps: int = 2048,
        anneal_lr: bool = True,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        num_minibatches: int = 32,
        update_epochs: int = 10,
        norm_adv: bool = True,
        clip_coef: float = 0.2,
        clip_vloss: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        target_kl: float = None,
        batch_size: int = 0,
        minibatch_size: int = 0,
        num_iterations: int = 0):
        
        """
        A class to run PPO algorithm on continuous action space environments, adapted from CleanRL library.
        
        exp_name (str): the name of this experiment
        agent_id (int): the id of the agent
        seed (int): seed of the experiment
        torch_deterministic (bool): if toggled, `torch.backends.cudnn.deterministic=False`
        cuda (bool): if toggled, cuda will be enabled by default
        track (bool): if toggled, this experiment will be tracked with Weights and Biases
        project_name (str): the wandb's project name
        wandb_entity (str): the entity (team) of wandb's project
        capture_video (bool): whether to capture videos of the agent performances (check out `videos` folder)
        save_model (bool): whether to save model into the `runs/{run_name}` folder
        env_id (str): the id of the environment
        total_timesteps (int): total timesteps of the experiments
        learning_rate (float): the learning rate of the optimizer
        num_envs (int): the number of parallel game environments
        num_steps (int): the number of steps to run in each environment per policy rollout
        anneal_lr (bool): Toggle learning rate annealing for policy and value networks
        gamma (float): the discount factor gamma
        gae_lambda (float): the lambda for the general advantage estimation
        num_minibatches (int): the number of mini-batches
        update_epochs (int): the K epochs to update the policy
        norm_adv (bool): Toggles advantages normalization
        clip_coef (float): the surrogate clipping coefficient
        clip_vloss (bool): Toggles whether or not to use a clipped loss for the value function, as per the paper.
        ent_coef (float): coefficient of the entropy
        vf_coef (float): coefficient of the value function
        max_grad_norm (float): the maximum norm for the gradient clipping
        target_kl (float): the target KL divergence threshold
        batch_size (int): the batch size (computed in runtime)
        minibatch_size (int): the mini-batch size (computed in runtime)
        num_iterations (int): the number of iterations (computed in runtime)
        """
        
        self.exp_name = exp_name
        self.agent_id = agent_id
        self.seed = seed
        self.torch_deterministic = torch_deterministic
        self.cuda = cuda
        self.track = track
        self.project_name = project_name
        self.wandb_entity = wandb_entity
        self.capture_video = capture_video
        self.save_model = save_model
        self.env_id = env_id
        self.total_timesteps = total_timesteps
        self.learning_rate = learning_rate
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.anneal_lr = anneal_lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.num_minibatches = num_minibatches
        self.update_epochs = update_epochs
        self.norm_adv = norm_adv
        self.clip_coef = clip_coef
        self.clip_vloss = clip_vloss
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size
        self.num_iterations = num_iterations
        
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.cuda else "cpu")
        
        self.__init_reproducibility()
        
        # setup the environment and agent
        self.setup_envs()
        self.setup_agent()
        
        self.logger = log_config()
        
        
    def get_config(self):
        """
        Returns the configuration of the PPO_continuous_action class.
        """
        return vars(self)
        
    def set_reward_weight(self, reward_weight):
        """
        Set the reward weight of the model.
        
        reward_weight (list): the reward weight of the model
        """
        self.reward_weight = reward_weight if isinstance(reward_weight, np.ndarray) else np.array(reward_weight)
        
    def __init_reproducibility(self):
        """
        Initialize the reproducibility of the experiment.
        """
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = self.torch_deterministic
        
    def setup_envs(self, run_name=None):
        """
        Set up the environments.
        """
        self.envs = gym.vector.SyncVectorEnv(
            [make_env(self.env_id, i, self.capture_video, run_name, self.gamma) for i in range(self.num_envs)]
        )
        assert isinstance(self.envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
        
        return self.envs
    
    def setup_agent(self, envs=None):
        """
        Set up the agent.
        """
        envs = self.envs if envs is None else envs
        self.agent = RLAgent(envs=envs).to(self.device)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=self.learning_rate, eps=1e-5)
        
        return self.agent, self.optimizer
        
    def train(self, num_iterations=None):
    
        self.batch_size = int(self.num_envs * self.num_steps)
        self.minibatch_size = int(self.batch_size // self.num_minibatches)
        self.num_iterations = self.total_timesteps // self.batch_size if self.total_timesteps is not None else self.num_iterations
        self.num_iterations = self.num_iterations if num_iterations is None else num_iterations
        
        run_name = f"{self.project_name}/{self.agent_id}_{self.env_id}__{self.exp_name}__{self.seed}__{int(time.time())}" 
        
        # -------------------------set up logging----------------------------
        if self.track:
            import wandb
            wandb.tensorboard.patch(root_logdir=f"tensorboard/{run_name}")
            wandb.init(
                project=self.project_name,
                entity=self.wandb_entity,
                sync_tensorboard=True,
                config=self.get_config(),
                name=run_name,
                monitor_gym=True,
                save_code=True,
            )
        writer = SummaryWriter(f"tensorboard/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in self.get_config().items()])),
        )


        # ------------------------replay storage--------------------
        # ALGO Logic: Storage setup
        obs = torch.zeros((self.num_steps, self.num_envs) + self.envs.single_observation_space.shape).to(self.device)
        actions = torch.zeros((self.num_steps, self.num_envs) + self.envs.single_action_space.shape).to(self.device)
        logprobs = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        # ! JJ
        rewards = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        dones = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        # ! JJ
        values = torch.zeros((self.num_steps, self.num_envs)).to(self.device)


        # ----------------------training iteration-----------------------
        # TRY NOT TO MODIFY: start the game
        global_step = 0
        start_time = time.time()
        next_obs, _ = self.envs.reset(seed=self.seed)
        next_obs = torch.Tensor(next_obs).to(self.device)
        next_done = torch.zeros(self.num_envs).to(self.device)

        for iteration in range(1, self.num_iterations + 1):
            # Annealing the rate if instructed to do so.
            if self.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / self.num_iterations
                lrnow = frac * self.learning_rate
                self.optimizer.param_groups[0]["lr"] = lrnow

            # =========================collect data============================
            for step in range(0, int(self.num_steps/self.num_envs)):
                global_step += self.num_envs
                obs[step] = next_obs
                dones[step] = next_done

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, _, value = self.agent.get_action_and_value(next_obs)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, terminations, truncations, infos = self.envs.step(action.cpu().numpy())
                next_done = np.logical_or(terminations, truncations)
                rewards[step] = torch.tensor(reward).to(self.device)#.view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(self.device), torch.Tensor(next_done).to(self.device)

                if "final_info" in infos:
                    for info in infos["final_info"]:
                        if info and "episode" in info:
                            writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                            writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

            # ============================compute advantage========================
            # bootstrap value if not done
            with torch.no_grad():
                next_value = self.agent.get_value(next_obs)#.reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(self.device)
                lastgaelam = 0
                for t in reversed(range(self.num_steps)):
                    if t == self.num_steps - 1:
                        nextnonterminal = torch.ones_like(next_value) - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = torch.ones_like(next_value) - dones[t + 1]
                        nextvalues = values[t + 1]
                    # TD error
                    delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
                    # GAE
                    advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values

            # ================================training data==============================
            # flatten the batch
            b_obs = obs.reshape((-1,) + self.envs.single_observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + self.envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # ================================training===================================
            # Optimizing the policy and value network
            b_inds = np.arange(self.batch_size)
            clipfracs = []
            for epoch in range(self.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, self.batch_size, self.minibatch_size):
                    end = start + self.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if self.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if self.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -self.clip_coef,
                            self.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    # Entropy loss
                    entropy_loss = entropy.mean()
                    
                    # total loss
                    loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                if self.target_kl is not None and approx_kl > self.target_kl:
                    break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            writer.add_scalar("charts/learning_rate", self.optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)
            writer.add_scalar("return/mean_rewards", np.mean(rewards.cpu().numpy()), global_step)
            # ! JJ
            #print("SPS:", int(global_step / (time.time() - start_time)))
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

            if self.track:
                wandb.log(
                    {
                        f"agent{self.agent_id}_vloss": v_loss.item(),
                        f"agent{self.agent_id}_pgloss": pg_loss.item(),
                        f"agent{self.agent_id}_entropy": entropy_loss.item(),
                        f"agent{self.agent_id}_approxkl": approx_kl.item(),
                        f"agent{self.agent_id}_clipfrac": np.mean(clipfracs),
                        f"agent{self.agent_id}_explained_variance": explained_var,
                        f"agent{self.agent_id}_lr": self.optimizer.param_groups[0]["lr"],
                        f"agent{self.agent_id}_mean_rewards": np.mean(rewards.cpu().numpy()),
                        "charts/global_step": global_step,
                    }
                )


        self.model_path = f"ppo_model/{run_name}/{self.exp_name}.ppo"
        if self.save_model:
            
            if not os.path.exists(os.path.dirname(self.model_path)):
                os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            torch.save(self.agent.state_dict(), self.model_path)
            #print(f"model saved to {self.model_path}")
            self.logger.info(f"model saved to {self.model_path}")
            # ! JJ
            #from cleanrl_utils.evals.ppo_eval import evaluate

            episodic_returns, ACTIONS = evaluate(
                self.model_path,
                make_env,
                self.env_id,
                eval_episodes=3,
                run_name=f"{run_name}-eval",
                Model=RLAgent,
                device=self.device,
                gamma=self.gamma,
            )
            
            for idx, episodic_return in enumerate(episodic_returns):       
                writer.add_scalar("eval/scale_episodic_return", episodic_return, idx)
                #writer.add_scalar("eval/actions", ACTIONS[idx], idx)
            # ! JJ
            
        self.model_path = f"ppo_model/{run_name}/{self.exp_name}.ppo"

        self.envs.close()
        writer.close()
        wandb.finish() if self.track else None
        

    def eval(self, 
             model_path: str = None, 
             env_id: str = None, 
             eval_episodes: list = None,
             devicde: torch.device = None,
             gama: float = None):
        
        """
        Evaluate the model.
        
        Args:
            model_path (str): the path of the model
            env_id (str): the id of the environment
            eval_episodes (list): the number of evaluation episodes
            devicde (torch.device): the device to run the evaluation
            gama (float): the gamma value
        
        Returns:
            episodic_returns (list): the episodic returns
            disc_vec_retruns (list): the discounted vector returns
            ACTIONS (list): the actions
        """
        
        model_path = self.model_path if model_path is None else model_path
        env_id = self.env_id if env_id is None else env_id
        devicde = self.device if devicde is None else devicde
        gama = self.gamma if gama is None else gama
        
        episodic_returns, disc_vec_retruns, ACTIONS  = evaluate(
                model_path,
                make_env,
                env_id,
                eval_episodes=eval_episodes,
                run_name="eval",
                Model=RLAgent,
                device=self.device,
                gamma=self.gamma,
            )
        return episodic_returns, disc_vec_retruns, ACTIONS
    
    def save(self, model_path: str = None):
        """
        Save the model.
        """
        model_path = self.model_path if model_path is None else model_path
        
        if not os.path.exists(os.path.dirname(model_path)):
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        torch.save(self.agent.state_dict(), model_path)
        #print(f"model saved to {self.model_path}")
        self.logger.info(f"model saved to {model_path}")
        
    def load(self, model_path: str = None):
        """
        Load the model.
        """
        model_path = self.model_path if model_path is None else model_path
        self.agent.load_state_dict(torch.load(model_path, map_location=self.device))
        #print(f"model loaded from {model_path}")
        self.logger.info(f"model loaded from {model_path}")