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
from pythermalcomfort.models import pmv_ppd_iso

# Import the thrid party libraries
import gymnasium as gym
from gymnasium.envs.registration import register
import numpy as np
import pandas as pd
import warnings
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Train the RL agent for the iroof environment')
    parser.add_argument('--algo', type=str, default='ppo', help='Algorithm')
    parser.add_argument('--city', type=str, default='beijing', help='City name')
    parser.add_argument('--reward_weight', type=float, default=0.5, help='Reward weight')
    parser.add_argument('--total_timesteps', type=int, default=48*181*20, #48*365*50
                        help='Total timesteps')
    parser.add_argument('--gamma', type=float, default=0.99, help='Gamma')
    parser.add_argument('--num_steps', type=int, default=48*181, help='Number of steps')
    parser.add_argument('--num_envs', type=int, default=1, help='Number of environments')
    parser.add_argument('--project_name', type=str, default='clmux-w-var', help='Project name')
    parser.add_argument('--track', type=bool, default=True, help='Track')
    parser.add_argument('--save_model', type=bool, default=True, help='Save model')
    parser.add_argument('--seed', type=int, default=42, help='Seed')
    parser.add_argument('--step_num', type=int, default=48*181,#48*365,
                        help='Number of steps') 
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
    reward_weight = args.reward_weight
    # max energy consumption for each city (W/m^2)
    # 25.677637100219727, 53.58064651489258, 42.56001281738281, 13.664632797241211, 4.108735084533691
    max_energy_dict = {
        "london": 1000.0,
    }
    max_energy = max_energy_dict[city]
    
    ele_price = pd.read_csv('/home/junjieyu/Github/RL_CLMU/data/plotting_w/london_avgprice.csv')
    
    
    def reward_function(info, ele_price_v, col, met):
        # ref: https://www.sciencedirect.com/science/article/pii/S0378778824013574?via%3Dihub
        # r = -w*lambda_P*P - (1-w)*lambda_T*(|T-T_up| + |T-T_low|)
        # w: weight of power consumption
        # lambda_P: scaling constants for power consumption
        # P: power consumption
        # lambda_T: scaling constants for temperature
        # T: temperature
        # T_up: upper limit of temperature
        # T_low: lower limit of temperature
        # r: reward
        # action: action
        # info: info
        w = reward_weight
        
        #col = 1.0 # ref: https://pythermalcomfort.readthedocs.io/en/latest/documentation/clothing.html#dynamic-clothing
        #met = 1.2 # ref: https://www.sciencedirect.com/science/article/pii/S0378778824013574?via%3Dihub
        
        COP_cooling = -0.14 * (info['taf [K]'] - 273.15) + 7.31
        COP_heating = 0.07 * (info['taf [K]'] - 273.15) + 3.20
        
        lambda_P = 1/max_energy

        #Edemand = Esite/COP/Peff
        P_ac = info['eflx_urban_ac [W/m**2]']/COP_cooling#/0.9/0.96
        P_heat = info['eflx_urban_heat [W/m**2]']/COP_heating#/0.9/0.96
        E_cost = (P_ac + P_heat)*ele_price_v * 0.5 # 0.5 hour per timestep
        # nomalized reward:
        # minimize energy consumption and temperature deviation
        # morn = (w - min(w)) / (max(w) - min(w))
        pmv = pmv_ppd_iso(
                    tdb=info['t_building [K]'] - 273.15,
                    tr=info['t_mean_radiant [K]'] - 273.15,
                    vr=0.1,
                    rh=50,
                    met=met,
                    clo=col,
                    model="7730-2005",
                    limit_inputs=False
                ).pmv
        
        if abs(pmv) <= 0.5:
            thermal_discomfort = 0
        else:
            diff = min(abs(pmv - (-0.5)), abs(pmv - 0.5))
            thermal_discomfort = - diff ** 2
        
        r = - w * lambda_P * E_cost + (1 - w) * thermal_discomfort
        
        return r, E_cost, thermal_discomfort
    
    
    surfdata = f"data/clmu_input/surfdata_{city}.nc"
    # /home/junjieyu/Github/RL_CLMU/data/hac_on_wasteheat/london/default.nc
    forcing = f"/home/junjieyu/Github/RL_CLMU/data/hac_on_wasteheat/{city}/default.nc"
    epochnum = args.step_num
    
    if city == 'london':
        forcing_time_range = ["2012-11", "2013-04"]
    else:
        forcing_time_range = ['2022', '2022']
    
    # Register the environment
    register(
        id=f'clmux-{city}-{reward_weight}',
        entry_point="src.clmuxenv:clmux_gym",
        kwargs = dict(
                    envid = f'clmux-{city}-{reward_weight}',
                    surfdata = surfdata,
                    forcing = forcing,
                    epochnum = epochnum,
                    action_space = action_space,
                    forcing_time_range = forcing_time_range,
                    reward_function = reward_function,
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
        agent = PPO(env_id=f'clmux-{args.city}-{args.reward_weight}',
                    seed=seed, total_timesteps=total_timesteps, gamma=gamma,
                    num_steps=num_steps, num_envs=num_envs,
                    project_name=project_name,
                    track=track, save_model=save_model)
        modelpath = agent.train()
    
    if args.algo == 'sac':
        agent = SAC(env_id=f'clmux-{args.city}-{args.reward_weight}',
                    seed=seed, total_timesteps=total_timesteps, gamma=gamma, num_envs=num_envs,
                    project_name=project_name, learning_starts = 48*181,#48*365,
                    track=track, save_model=save_model, autotune=False)
        modelpath = agent.train()
    
    if args.algo == 'dqn':
        agent = DQN(env_id=f'clmux-{args.city}-{args.reward_weight}', 
            seed=seed, total_timesteps=total_timesteps, gamma=gamma,
            project_name=project_name,
            track=track, save_model=save_model, learning_starts = 48*181,#48*365,
            )
        modelpath = agent.train()
    elif args.algo == 'qlearning':
        num_episodes = int(total_timesteps/args.step_num)
        agent = QLearning(env_id=f'clmux-{args.city}-{args.reward_weight}',
                          gamma=gamma, seed=seed, project_name=project_name, track=track)
        modelpath = agent.train(num_episodes=num_episodes)

    logger.info(f'Trained model saved at {modelpath}')
    logger.info('Training completed')
    
if __name__ == '__main__':
    seed_everything()
    args = parse_args()
    workflow(args)
    