import sys
sys.path.append('..')

import gymnasium as gym
from clmux.clmux import bem
from typing import Union, Any, SupportsFloat, Callable
from random import random
import xarray as xr
import numpy as np
import pandas as pd

#action_space_Continuous = gym.spaces.Box(low=np.array([273.15+20, 273.15+10, 0.3]), 
#                                  high=np.array([273.15+35, 273.15+25, 0.5]), 
#                                  dtype=np.float32, seed=0)

action_space_Continuous = gym.spaces.Box(low=np.array([273.15+16]), 
                                  high=np.array([273.15+24]), 
                                  dtype=np.float32, seed=0)

action_space_Discrete = gym.spaces.Discrete(9, seed=0)

ele_price = pd.read_csv('/home/junjieyu/Github/RL_CLMU/data/plotting_w/london_avgprice.csv')
#observation_space = gym.spaces.Box(low=np.array([273.15+15, 273.15+10, 0.3, 273.15-20, 273.15-20,
#                                                 ele_price.min().values[0], ele_price.min().values[0], ele_price.min().values[0], ele_price.min().values[0], ele_price.min().values[0],
#                                                 # time features can be added here
#                                                    -1.0, -1.0, -1.0, -1.0, 
#                                                    273.15-20
#                                                 ]), 
#                                       high=np.array([273.15+35, 273.15+30, 0.5, 273.15+40, 273.15+40,
#                                                      ele_price.max().values[0], ele_price.max().values[0], ele_price.max().values[0], ele_price.max().values[0], ele_price.max().values[0],
#                                                      # time features can be added here
#                                                        1.0, 1.0, 1.0, 1.0,
#                                                        273.15+40
#                                                      ]), 
#                                       dtype=np.float32, seed=0)
observation_space = gym.spaces.Box(low=np.array([273.15+15, 273.15-20, 273.15-20,
                                                 0.0, 0.0, 0.0, 0.0, 0.0,
                                                 # time features can be added here
                                                    -1.0, -1.0, -1.0, -1.0
                                                 ]), 
                                      high=np.array([273.15+30, 273.15+40, 273.15+40,
                                                     1.0, 1.0, 1.0, 1.0, 1.0,
                                                     # time features can be added here
                                                       1.0, 1.0, 1.0, 1.0
                                                     ]), 
                                      dtype=np.float32, seed=0)

def reward_function(info):
    # ref: https://ugr-sail.github.io/sinergym/compilation/main/pages/rewards.html
    # ref2: https://www.sciencedirect.com/science/article/pii/S2666546820300203
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
    w = 0.5
    lambda_T = 1/23
    lambda_P = 1/50
    T1 = 27 + 273.15  # upper limit of temperature in K
    T2 = 19 + 273.15  # lower limit of temperature in K
    #Edemand = Esite/COP/Peff
    P_ac = info['eflx_urban_ac [W/m**2]']/3.6/0.43
    P_heat = info['eflx_urban_heat [W/m**2]']/0.9/0.96
    r = -w*lambda_P*(P_ac + P_heat) - (1-w)*lambda_T*(abs(info['t_building [K]'] - T1) + abs(info['t_building [K]'] - T2))
    return r

def get_dynamic_parameters(month, hour):
    # --- 1. 判断是否是睡觉时间 (例如 23:00 - 07:00) ---
    is_sleeping = (hour >= 23) or (hour < 7)

    # --- 2. 设定 Met (代谢率) ---
    if is_sleeping:
        current_met = 0.7  # 睡觉模式：产热极低
    else:
        current_met = 1.2  # 日间模式：正常静坐

    # --- 3. 设定 Clo (服装/被褥) ---
    if is_sleeping:
        # 【关键修正】睡觉必须盖被子！
        # 冬天被子厚一点 (2.5), 夏天薄被子 (1.5)
        if month in [12, 1, 2]:
            current_clo = 2.5  # 厚棉被
        elif month in [6, 7, 8]:
            current_clo = 1.5  # 毯子/薄被
        else:
            current_clo = 2.0  # 普通被子
    else:
        # 白天穿衣服 (之前的逻辑)
        if month in [11, 12, 1, 2, 3, 4]:
            current_clo = 1.0  # 冬装
        elif month in [6, 7, 8]:
            current_clo = 0.5  # 夏装
        else:
            current_clo = 0.75 # 秋装
            
    return current_clo, current_met

def get_input(time_step : int,
              surface : xr.Dataset,
              forcing : xr.Dataset,
              urban_hac : str = "on",
              urban_explicit_ac : str = "on",
              p_ac : float = 1.0,
              vent_ach : float = 0.3,
              t_roof_inner_bef = 291.80765,
              t_sunw_inner_bef = 291.96564,
              t_shdw_inner_bef = 291.96564,
              t_floor_bef = 291.9698,
              t_building_bef = 290.44763,
              t_building_max = 380,
              t_building_min = 285.1000061
              ):
    dtime = 1800.0 # time step in seconds
    urban_hac = urban_hac
    urban_explicit_ac = urban_explicit_ac
    p_ac = p_ac
    ht_roof = surface['HT_ROOF'].sel(numurbl=3).isel(lsmlat=0, lsmlon=0).values
    vent_ach = vent_ach
    canyon_hwr = surface['CANYON_HWR'].sel(numurbl=3).isel(lsmlat=0, lsmlon=0).values
    wtlunit_roof = surface['WTLUNIT_ROOF'].sel(numurbl=3).isel(lsmlat=0, lsmlon=0).values
    zi_roof = surface['THICK_ROOF'].sel(numurbl=3).isel(lsmlat=0, lsmlon=0).values/10 * 0.5
    z_roof = 0

    tssbef_roof = forcing['TSOI'].sel(column=71).isel(levgrnd=9).isel(time=time_step).values
    t_soisno_roof = forcing['TSOI'].sel(column=71).isel(levgrnd=9).isel(time=time_step+1).values
    tk_roof = surface['TK_ROOF'].sel(nlevurb=9, numurbl=3).isel(lsmlat=0, lsmlon=0).values
    zi_sunw = surface['THICK_WALL'].sel(numurbl=3).isel(lsmlat=0, lsmlon=0).values/10 * 0.5
    z_sunw = 0

    tssbef_sunw = forcing['TSOI'].sel(column=72).isel(levgrnd=9).isel(time=time_step).values
    t_soisno_sunw = forcing['TSOI'].sel(column=72).isel(levgrnd=9).isel(time=time_step+1).values
    tk_sunw = surface['TK_WALL'].sel(nlevurb=9, numurbl=3).isel(lsmlat=0, lsmlon=0).values
    tk_shdw = surface['TK_WALL'].sel(nlevurb=9, numurbl=3).isel(lsmlat=0, lsmlon=0).values
    
    zi_shdw = surface['THICK_WALL'].sel(numurbl=3).isel(lsmlat=0, lsmlon=0).values/10 * 0.5
    z_shdw = 0
    
    tssbef_shdw = forcing['TSOI'].sel(column=73).isel(levgrnd=9).isel(time=time_step).values
    t_soisno_shdw = forcing['TSOI'].sel(column=73).isel(levgrnd=9).isel(time=time_step+1).values
    taf = forcing['TSA'].sel(pft=71).isel(time=time_step+1).values

    return dtime, urban_hac, urban_explicit_ac, p_ac, ht_roof, \
        t_building_max, t_building_min, vent_ach, canyon_hwr, wtlunit_roof,\
        zi_roof, z_roof, tssbef_roof, t_soisno_roof, tk_roof,\
        zi_sunw, z_sunw, tssbef_sunw, t_soisno_sunw, tk_sunw,\
        zi_shdw, z_shdw, tssbef_shdw, t_soisno_shdw, tk_shdw,\
        t_roof_inner_bef, t_sunw_inner_bef, t_shdw_inner_bef, t_floor_bef, t_building_bef, taf


class clmux_gym(gym.Env):
    """
    Custom Environment that follows gym interface.
    This is a CESM env that will run CESM/CLM and return the reward, state, action
    """

    def __init__(self, 
                envid: str,
                surfdata: str,
                forcing: str,
                epochnum: int,
                forcing_time_range: list = None,
                action_space: gym.Space = action_space_Continuous,
                observation_space: gym.Space = observation_space,
                reward_function: Callable[[Any, Any], SupportsFloat] = reward_function,
                seed: Union[int, None] = None,
                 ):
        super(clmux_gym, self).__init__()
        
        self.envid = envid
        self.action_space = action_space
        self.observation_space = observation_space
        self.reward_function = reward_function
        
        self.spec = self.Spec()
        self.spec.id = envid
        
        self.surfdata = xr.open_dataset(surfdata)
        self.forcing = xr.open_dataset(forcing)
        
        if forcing_time_range is not None:
            self.forcing = self.forcing.sel(time=slice(forcing_time_range[0], forcing_time_range[1]))
        
        self.forcing = self.forcing.assign_coords({"column": self.forcing.cols1d_itype_col.values})
        self.forcing = self.forcing.assign_coords({"pft": self.forcing.pfts1d_itype_col.values})
        
        self.datalen = len(self.forcing.time)
        
        self.bem = bem()
        
        self.epochnum = epochnum if epochnum is not None else 48*365
        
        self.time = self.forcing.time.dt.round('min')
        self.year = self.time.dt.year
        self.month = self.time.dt.month
        self.day = self.time.dt.day
        self.dayofyear = self.time.dt.dayofyear
        self.hour = self.time.dt.hour + self.time.dt.minute / 60.0
        self.ele_price = pd.read_csv('/home/junjieyu/Github/RL_CLMU/data/plotting_w/london_avgprice.csv')
        self.ele_price = self.ele_price.set_index(['month', 'day', 'hour'])
        
    def step(self, action):

        if isinstance(self.action_space, gym.spaces.Box):
            #self.ac_set_point = action[0]
            #self.heat_set_point = action[1]
            #self.vent_ach = action[2]
            self.ac_set_point = 25 + 273.15
            self.heat_set_point = action[0]
            self.vent_ach = 0.3
        elif isinstance(self.action_space, gym.spaces.Discrete):
            #self.ac_set_point = 25.0 + 273.15 if action in [0,1,2,3] else 100.0 + 273.15
            #self.heat_set_point = 20.0 + 273.15 if action in [0,1,4,5] else -50.0 + 273.15
            #self.vent_ach = 0.5 if action in [0,2,4,6] else 0.3
            self.ac_set_point = 25.0 + 273.15 
            self.heat_set_point = 16.0 + 273.15 + action
            self.vent_ach = 0.3
        else:
            raise ValueError("Action space not supported")
        
        input = get_input(time_step = self.time_step,
                          surface = self.surfdata,
                          forcing = self.forcing,
                          t_roof_inner_bef = self.t_roof_inner_bef,
                          t_sunw_inner_bef = self.t_sunw_inner_bef,
                          t_shdw_inner_bef = self.t_shdw_inner_bef,
                          t_floor_bef = self.t_floor_bef,
                          t_building_bef = self.t_building_bef,
                          t_building_max = self.ac_set_point,
                          t_building_min = self.heat_set_point,
                          vent_ach=self.vent_ach)
        
        self.t_roof_inner_bef,self.t_sunw_inner_bef,\
        self.t_shdw_inner_bef,self.t_floor_bef,self.t_building_bef,\
        info = self.bem.bem(*input)
        
        # get electricity price
        cele_price_v = self.ele_price.loc[(int(self.month[self.time_step+1].values), 
                                         int(self.day[self.time_step+1].values), 
                                         round(float(self.hour[self.time_step+1].values),1))].values[0]
        
        col, met = get_dynamic_parameters(int(self.month[self.time_step+1].values), 
                                         int(self.hour[self.time_step+1].values))
        #reward = self.reward_function(info, cele_price_v, col, met)
        reward, E_cost, thermal_discomfort = self.reward_function(info, cele_price_v, col, met)
        
        time_step_v1 = self.time_step + 1 + 1 if (self.time_step + 1 + 1 < self.datalen) else self.time_step+1+1 - self.datalen
        time_step_v2 = self.time_step + 2 + 1 if (self.time_step + 2 + 1 < self.datalen) else self.time_step+2+1 - self.datalen
        time_step_v3 = self.time_step + 3 + 1 if (self.time_step + 3 + 1 < self.datalen) else self.time_step+3+1 - self.datalen
        time_step_v4 = self.time_step + 4 + 1 if (self.time_step + 4 + 1 < self.datalen) else self.time_step+4+1 - self.datalen
        
        fele_price_v1 = self.ele_price.loc[(int(self.month[time_step_v1].values), 
                                         int(self.day[time_step_v1].values), 
                                         round(float(self.hour[time_step_v1].values),1))].values[0]
        fele_price_v2 = self.ele_price.loc[(int(self.month[time_step_v2].values), 
                                         int(self.day[time_step_v2].values), 
                                         round(float(self.hour[time_step_v2].values),1))].values[0]
        fele_price_v3 = self.ele_price.loc[(int(self.month[time_step_v3].values), 
                                         int(self.day[time_step_v3].values),
                                         round(float(self.hour[time_step_v3].values),1))].values[0]
        fele_price_v4 = self.ele_price.loc[(int(self.month[time_step_v4].values), 
                                         int(self.day[time_step_v4].values),
                                         round(float(self.hour[time_step_v4].values),1))].values[0]
        
        # time features can be added here
        hour_embedding = np.array([np.sin(2 * np.pi * self.hour[self.time_step] / 24),
                                   np.cos(2 * np.pi * self.hour[self.time_step] / 24)])
        # 判断是否为闰年
        is_leap_year = (self.year[self.time_step] % 4 == 0) & ((self.year[self.time_step] % 100 != 0) | (self.year[self.time_step] % 400 == 0))
        days_in_year = 366 if is_leap_year else 365
        dayofyear_embedding = np.array([np.sin(2 * np.pi * self.dayofyear[self.time_step] / days_in_year),
                                       np.cos(2 * np.pi * self.dayofyear[self.time_step] / days_in_year)])
        
        if isinstance(self.action_space, gym.spaces.Discrete):
            #info['ac_set_point'] = 1 if action in [0,1,2,3] else 0
            #info['heat_set_point'] = 1 if action in [0,1,4,5] else 0
            #info['vent_ach'] = 0.5 if action in [0,2,4,6] else 0.3
            info['ac_set_point'] = 25.0 + 273.15 
            info['heat_set_point'] = (self.heat_set_point - (16.0 + 273.15)) / (24.0 - 16.0)
            
            info['vent_ach'] = 0.3
        else:
            #info['ac_set_point'] = self.ac_set_point 
            #info['heat_set_point'] = self.heat_set_point
            #info['vent_ach'] = self.vent_ach
            info['heat_set_point'] = (self.heat_set_point - (16.0 + 273.15)) / (24.0 - 16.0)
        
        info['heat_set_point_log'] = self.heat_set_point
        info['cele_price_v'] = cele_price_v
        info['fele_price_v1'] = fele_price_v1
        info['fele_price_v2'] = fele_price_v2
        info['fele_price_v3'] = fele_price_v3
        info['fele_price_v4'] = fele_price_v4
        info['hour_embedding'] = hour_embedding
        info['dayofyear_embedding'] = dayofyear_embedding
        info['E_cost'] = E_cost
        info['thermal_discomfort'] = thermal_discomfort
        #self.observation = np.array([self.ac_set_point, self.heat_set_point,
        #                                self.vent_ach, self.t_building_bef, self.taf,
        #                                cele_price_v, fele_price_v1, fele_price_v2, fele_price_v3, fele_price_v4,
        #                                hour_embedding[0], hour_embedding[1], dayofyear_embedding[0], dayofyear_embedding[1]
        #                                ])
        #265 306 # add 2 / sub 2 to get max and min
        #270 300 # add 2 / sub 2 to get max and min
        #0.0 50 for price # add 0 / round to 50 to get max and min
        #self.observation = np.array([info['ac_set_point'], info['heat_set_point'], info['vent_ach'],
        #                                (self.t_building_bef-270)/(300-270), (self.taf-265)/(306-265),
        #                                (cele_price_v-0.0)/(50.0-0.0), (fele_price_v1-0.0)/(50.0-0.0), (fele_price_v2-0.0)/(50.0-0.0), (fele_price_v3-0.0)/(50.0-0.0), (fele_price_v4-0.0)/(50.0-0.0),
        #                                hour_embedding[0], hour_embedding[1], dayofyear_embedding[0], dayofyear_embedding[1],
        #                                info['t_mean_radiant [K]']-270/(300-270)
        #                                ])
        
        self.observation = np.array([info['heat_set_point'], #info['heat_set_point'], info['vent_ach'],
                                        (self.t_building_bef-270)/(300-270), (self.taf-265)/(306-265),
                                        (cele_price_v-0.0)/(50.0-0.0), (fele_price_v1-0.0)/(50.0-0.0), (fele_price_v2-0.0)/(50.0-0.0), (fele_price_v3-0.0)/(50.0-0.0), (fele_price_v4-0.0)/(50.0-0.0),
                                        hour_embedding[0], hour_embedding[1], dayofyear_embedding[0], dayofyear_embedding[1]
                                        ])
        # normalize observation
        # self.observation = (self.observation - self.observation_space.low) / (self.observation_space.high - self.observation_space.low)
        self.time_step += 1
        self.step_count += 1
        
        terminated = self.step_count >= self.epochnum 
        truncated = self.time_step + 1 >= self.datalen
        
        return self.observation, reward, terminated, truncated, info


    def reset(self, seed=None, options=None):
        
        if seed is not None:
            np.random.seed(seed)
            self.seed = seed
        
        #self.time_step = np.random.randint(0, self.datalen//self.epochnum) if options is None else options.get("time_step", 0)
        
        #self.time_step = self.time_step * self.epochnum + 1
        self.time_step = 0
        
        #if self.time_step == 0:
        #    self.time_step = 1
            
        self.step_count = 0
            
        self.ac_set_point = 25.0 + 273.15 if options is None else options.get("ac_set_point", 25.0 + 273.15)
        self.heat_set_point = 15.0 + 273.15 if options is None else options.get("heat_set_point", 15.0 + 273.15)
        self.vent_ach = 0.3 if options is None else options.get("vent_ach", 0.3)
        
        
        self.t_roof_inner_bef = 273.15 + 16.0 if options is None else options.get("t_roof_inner_bef", 273.15 + 16.0)
        self.t_sunw_inner_bef = 273.15 + 16.0 if options is None else options.get("t_sunw_inner_bef", 273.15 + 16.0)
        self.t_shdw_inner_bef = 273.15 + 16.0 if options is None else options.get("t_shdw_inner_bef", 273.15 + 16.0)
        self.t_floor_bef = 273.15 + 16.0 if options is None else options.get("t_floor_bef", 273.15 + 16.0)
        self.t_building_bef = 273.15 + 16.0 if options is None else options.get("t_building_bef", 273.15 + 16.0)
        self.taf = self.forcing['TSA'].sel(pft=71).isel(time=self.time_step+1).values
        
        cele_price_v = self.ele_price.loc[(int(self.month[self.time_step+1].values), 
                                         int(self.day[self.time_step+1].values), 
                                         round(float(self.hour[self.time_step+1].values),1))].values[0]
        time_step_v1 = self.time_step + 1 + 1 if (self.time_step + 1 + 1 < self.datalen) else self.time_step+1+1 - self.datalen
        time_step_v2 = self.time_step + 2 + 1 if (self.time_step + 2 + 1 < self.datalen) else self.time_step+2+1 - self.datalen
        time_step_v3 = self.time_step + 3 + 1 if (self.time_step + 3 + 1 < self.datalen) else self.time_step+3+1 - self.datalen
        time_step_v4 = self.time_step + 4 + 1 if (self.time_step + 4 + 1 < self.datalen) else self.time_step+4+1 - self.datalen
        fele_price_v1 = self.ele_price.loc[(int(self.month[time_step_v1].values), 
                                         int(self.day[time_step_v1].values), 
                                         round(float(self.hour[time_step_v1].values),1))].values[0]
        fele_price_v2 = self.ele_price.loc[(int(self.month[time_step_v2].values), 
                                         int(self.day[time_step_v2].values), 
                                         round(float(self.hour[time_step_v2].values),1))].values[0]
        fele_price_v3 = self.ele_price.loc[(int(self.month[time_step_v3].values), 
                                         int(self.day[time_step_v3].values),
                                         round(float(self.hour[time_step_v3].values),1))].values[0]
        fele_price_v4 = self.ele_price.loc[(int(self.month[time_step_v4].values), 
                                         int(self.day[time_step_v4].values),
                                         round(float(self.hour[time_step_v4].values),1))].values[0]
        
        hour_embedding = np.array([np.sin(2 * np.pi * self.hour[self.time_step] / 24),
                                   np.cos(2 * np.pi * self.hour[self.time_step] / 24)])
        # 判断是否为闰年
        is_leap_year = (self.year[self.time_step] % 4 == 0) & ((self.year[self.time_step] % 100 != 0) | (self.year[self.time_step] % 400 == 0))
        days_in_year = 366 if is_leap_year else 365
        dayofyear_embedding = np.array([np.sin(2 * np.pi * self.dayofyear[self.time_step] / days_in_year),
                                       np.cos(2 * np.pi * self.dayofyear[self.time_step] / days_in_year)])
        #self.observation = np.array([self.ac_set_point, self.heat_set_point, self.vent_ach, 
        #                             self.t_building_bef, self.taf,
        #                             cele_price_v, fele_price_v1, fele_price_v2, fele_price_v3, fele_price_v4,
        #                             # time features can be added here
        #                             hour_embedding[0], hour_embedding[1], dayofyear_embedding[0], dayofyear_embedding[1]
        #                             ])
        if isinstance(self.action_space, gym.spaces.Discrete):
            #ac_set_point_obs = 1 if random() < 0.5 else 0
            #heat_set_point_obs = 0 if ac_set_point_obs==1 else 0
            #vent_ach_obs = 0.3 if random() < 0.5 else 0.5
            heat_set_point_obs = (self.heat_set_point - 16.0 - 273.15) / (24.0 - 16.0)
        else:
            ac_set_point_obs = (self.ac_set_point - 10.0) / (30.0 - 10.0)
            heat_set_point_obs = (self.heat_set_point - 16.0) / (24.0 - 16.0)
            vent_ach_obs = (self.vent_ach - 0.3) / (0.5 - 0.3)
        #self.observation = np.array([ac_set_point_obs, heat_set_point_obs, vent_ach_obs,
        #                             (self.t_building_bef-270)/(300-270), (self.taf-265)/(306-265),
        #                             (cele_price_v-0.0)/(50-0.0), (fele_price_v1-0.0)/(50-0.0), (fele_price_v2-0.00)/(50-0.0), (fele_price_v3-0.0)/(50-0.00), (fele_price_v4-0.0)/(50-0.0),
        #                             hour_embedding[0], hour_embedding[1], dayofyear_embedding[0], dayofyear_embedding[1],
        #                            (16-270)/(300-270)
        #                             ])
        self.observation = np.array([heat_set_point_obs, #heat_set_point_obs, vent_ach_obs,
                                     (self.t_building_bef-270)/(300-270), (self.taf-265)/(306-265),
                                     (cele_price_v-0.0)/(50-0.0), (fele_price_v1-0.0)/(50-0.0), (fele_price_v2-0.00)/(50-0.0), (fele_price_v3-0.0)/(50-00), (fele_price_v4-0.0)/(50-0.0),
                                     hour_embedding[0], hour_embedding[1], dayofyear_embedding[0], dayofyear_embedding[1]
                                    ])
        # normalize observation
        # self.observation = (self.observation - self.observation_space.low) / (self.observation_space.high - self.observation_space.low)
        info = {
            "ac_set_point": self.ac_set_point,
            "heat_set_point": self.heat_set_point,
            "vent_ach": self.vent_ach,
            "t_building_bef": self.t_building_bef,
            "taf": self.taf,
            "cele_price_v": cele_price_v,
            "fele_price_v1": fele_price_v1,
            "fele_price_v2": fele_price_v2,
            "fele_price_v3": fele_price_v3,
            "fele_price_v4": fele_price_v4
        }
        
        return self.observation, info
    


    def render(self, mode='human'):
        pass
        

    def close(self):
        pass
    
    class Spec:
        def __init__(self):
            self.id = None  # type: str