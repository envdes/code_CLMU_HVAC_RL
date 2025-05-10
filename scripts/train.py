# training script for the iroof environment: actions = 5, data_path = ['EPDM', 'IRON', 'PVC', 'SHINGLES', 'ZINC']
# city: ["beijing", "newyork", "hongkong", "singapore"]

import sys
sys.path.append('..')

# Import the necessary libraries
from src.ppo_continuous_action import PPO_continuous_action as PPO
from src.sac_continuous_action import SAC_continuous_action as SAC
from src.dqn import DQN
from src.qlearning import QLearning
from src.clmuxenv import action_space_Discrete, action_space_Continuous
from src.utils import log_config, seed_everything

# Import the thrid party libraries
import gymnasium as gym
from gymnasium.envs.registration import register
import numpy as np
import warnings
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Train the RL agent for the iroof environment')
    parser.add_argument('--algo', type=str, default='ppo', help='Algorithm')
    parser.add_argument('--city', type=str, default='beijing', help='City name')
    parser.add_argument('--total_timesteps', type=int, default=48*365*50, help='Total timesteps')
    parser.add_argument('--gamma', type=float, default=0.99, help='Gamma')
    parser.add_argument('--num_steps', type=int, default=48*365, help='Number of steps')
    parser.add_argument('--num_envs', type=int, default=1, help='Number of environments')
    parser.add_argument('--project_name', type=str, default='clmux', help='Project name')
    parser.add_argument('--track', type=bool, default=True, help='Track')
    parser.add_argument('--save_model', type=bool, default=True, help='Save model')
    parser.add_argument('--seed', type=int, default=1, help='Seed')
    parser.add_argument('--step_num', type=int, default=48*365, help='Number of steps') 
    return parser.parse_args()

os.chdir('..')

warnings.filterwarnings("ignore")

def workflow(args):

    logger = log_config(log_file='iroof.log')

    logger.info(f'Start training the {args.algo} agent for the iroof environment')
    
    if args.algo in ['ppo', 'sac']:
        action_space = action_space_Continuous
    else:
        action_space = action_space_Discrete

    city = args.city
    surfdata = f"data/clmu_input/surfdata_{city}.nc"
    forcing = f"data/hac_off/{city}/default.nc"
    epochnum = args.step_num
    
    if city == 'london':
        forcing_time_range = ["2013", "2013"]
    else:
        forcing_time_range = ['2022', '2022']
    
    # Register the environment
    register(
        id=f'clmux-{city}',
        entry_point="src.clmuxenv:clmux_gym",
        kwargs = dict(
                    envid = f'clmux-{city}',
                    surfdata = surfdata,
                    forcing = forcing,
                    epochnum = epochnum,
                    action_space = action_space,
                    forcing_time_range = forcing_time_range,
        )
    )

    logger.info('Registering the iroof environment')
    logger.info('Environment variables:')

    logger.info(f'Step number: {epochnum}') 
    logger.info(f'Environment ID: clmux-{city}')

    total_timesteps=args.total_timesteps
    gamma=args.gamma
    num_steps=args.num_steps
    num_envs=args.num_envs
    project_name=args.project_name
    track=args.track
    save_model=args.save_model
    seed = args.seed

    logger.info(f'Training the {args.algo} agent')
    logger.info('Training parameters:')
    logger.info(f'Total timesteps: {total_timesteps}')
    logger.info(f'Gamma: {gamma}')
    logger.info(f'Number of steps: {num_steps}')
    logger.info(f'Number of environments: {num_envs}')
    logger.info(f'Project name: {project_name}')
    logger.info(f'Track: {track}')
    logger.info(f'Save model: {save_model}')
    logger.info(f'Seed: {seed}')


    if args.algo == 'ppo':
        agent = PPO(env_id=f'clmux-{args.city}',
                    seed=seed, total_timesteps=total_timesteps, gamma=gamma,
                    num_steps=num_steps, num_envs=num_envs,
                    project_name=project_name,
                    track=track, save_model=save_model)
        modelpath = agent.train()
    
    if args.algo == 'sac':
        agent = SAC(env_id=f'clmux-{args.city}',
                    seed=seed, total_timesteps=total_timesteps, gamma=gamma, num_envs=num_envs,
                    project_name=project_name,
                    track=track, save_model=save_model)
        modelpath = agent.train()
    
    if args.algo == 'dqn':
        agent = DQN(env_id=f'clmux-{args.city}', 
            seed=seed, total_timesteps=total_timesteps, gamma=gamma,
            project_name=project_name,
            track=track, save_model=save_model)
        modelpath = agent.train()
    elif args.algo == 'qlearning':
        num_episodes = int(total_timesteps/args.step_num)
        agent = QLearning(env_id=f'clmux-{args.city}',
                          gamma=gamma, seed=seed, project_name=project_name, track=track)
        modelpath = agent.train(num_episodes=num_episodes)

    logger.info(f'Trained model saved at {modelpath}')
    logger.info('Training completed')
    
if __name__ == '__main__':
    seed_everything()
    args = parse_args()
    workflow(args)
    