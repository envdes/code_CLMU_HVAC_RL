import sys
sys.path.append('..')

import gymnasium as gym
from gymnasium.envs.registration import register
from src.clmuxenv import action_space_Discrete, action_space_Continuous
import torch
import numpy as np
import xarray as xr
from src.sac_continuous_action import Actor
import argparse
import warnings
import os

os.chdir('..')
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description='get the sac model to netcdf file')
    parser.add_argument('--model_path', type=str, 
                        default='sac_model/clmux/_clmux-london__sac_continuous_action__1__1727035402/sac_continuous_action.sac', 
                        help='Model path')
    parser.add_argument('--ouptut_path', type=str,
                        default='model.nc',
                        help='Output path')
    return parser.parse_args()

args = parse_args()
model_path = args.model_path
output_path = args.ouptut_path



# initialize a environment to create a sac model
# ------------------------------
city = "london"
surfdata = f"data/clmu_input/surfdata_{city}.nc"
forcing = f"data/hac_off/{city}/default.nc"
epochnum = 4800

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
                action_space = action_space_Continuous,
                forcing_time_range = forcing_time_range,
    )
)
# ------------------------------


env = gym.make(f'clmux-{city}')
env = gym.vector.SyncVectorEnv([lambda: env])

# 创建模型实例并初始化权重
model = Actor(env)

#model_path = "/home/junjieyu/Github/CLMUX/sac_model/clmux/_clmux-london__sac_continuous_action__1__1727035402/sac_continuous_action.sac"
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
# get the model weights
model_weights = model.state_dict()

# dict to store the model weights
data_dict = {}

# transform the model weights to numpy arrays
for param_name, param_value in model_weights.items():
    # move the tensor to cpu and convert to numpy array
    param_np = param_value.cpu().numpy()
    
    # store the numpy array in the data_dict
    if param_name == "action_scale":
        data_dict[param_name] = (["dim_action_scale_0", "dim_action_scale_1"], param_np)  # action_scale 是二维的
    elif param_name == "action_bias":
        data_dict[param_name] = (["dim_action_bias_0", "dim_action_bias_1"], param_np)  # action_bias 是二维的
    else:
        # store the numpy array in the data_dict
        dims = ["dim_" + param_name + "_" + str(i) for i in range(param_np.ndim)]
        data_dict[param_name] = (dims, param_np)

ds = xr.Dataset(data_dict)

# save to netcdf file
#ds.to_netcdf("model_weights_and_bias_xarray.nc")
ds.to_netcdf(output_path)
print("Model weights and bias saved to", output_path)