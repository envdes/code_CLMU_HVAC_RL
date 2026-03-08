import sys
sys.path.append('..')

import gymnasium as gym
from gymnasium.envs.registration import register
from src.clmuxenv import action_space_Discrete, action_space_Continuous
import torch
import numpy as np
import xarray as xr
#from src.sac_continuous_action import Actor
from src.dqn import QNetwork
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
forcing = f"/home/junjieyu/Github/CLMUX_0.5/data/hac_off/{city}/default.nc"
epochnum = 4800

if city == 'london':
    forcing_time_range = ["2012", "2012"]
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
                action_space = action_space_Discrete,
                forcing_time_range = forcing_time_range,
    )
)
# ------------------------------


env = gym.make(f'clmux-{city}')
env = gym.vector.SyncVectorEnv([lambda: env])

# 创建模型实例并初始化权重
model = QNetwork(env)

#model_path = "/home/junjieyu/Github/CLMUX/sac_model/clmux/_clmux-london__sac_continuous_action__1__1727035402/sac_continuous_action.sac"
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
# get the model weights
model_weights = model.state_dict()

def export_to_fortran(model, file_path="/home/junjieyu/Github/RL_CLMU/data/model.bin"):
    state_dict = model.state_dict()
    # 按照网络层序排列 keys
    keys = [
        'network.0.weight', 'network.0.bias',
        'network.2.weight', 'network.2.bias',
        'network.4.weight', 'network.4.bias'
    ]
    
    with open(file_path, "wb") as f:
        for key in keys:
            # 确保转化为 float32 (Fortran 的 real)
            data = state_dict[key].t().contiguous().detach().cpu().numpy().astype(np.float64)
            
            # 关键：PyTorch 权重是 (out_dim, in_dim)
            # numpy 默认行优先存储，写入二进制后，
            # Fortran 按列优先读取 (out_dim, in_dim) 矩阵，正好对应。
            data.tofile(f)
            print(f"Exported {key} with shape {data.shape}")

# 假设你的模型变量名为 model
export_to_fortran(model, file_path=output_path)
print("Model weights and bias saved to", output_path)