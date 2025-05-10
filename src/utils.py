import logging
import random
import torch
import numpy as np
import gymnasium as gym

""""
def log_config():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - gpmorl - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    return logger
"""

def log_config(log_file='ppo.log'):
    """
    Configure the logging to output messages to both console and file.

    Parameters:
    log_file (str): The path to the log file. Defaults to 'application.log'.

    Returns:
    logger (logging.Logger): Configured logger instance.
    """
    # Create a logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Create a console handler for output to console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create a file handler for output to file
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Define the log format
    formatter = logging.Formatter('%(asctime)s - rl_sac - %(levelname)s - %(message)s')
    
    # Set formatter for both handlers
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Add handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

def seed_everything(seed=1):
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.backends.cudnn.deterministic = True
    
def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env.action_space.seed(seed)
        return env

    return thunk