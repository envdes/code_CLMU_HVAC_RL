import sys
sys.path.append('..')

import gymnasium as gym
from clmux.clmux import bem
from typing import Union, Any, SupportsFloat, Callable
from random import random
import xarray as xr
import numpy as np

action_space_Continuous = gym.spaces.Box(low=np.array([273.15+25, 273.15+10, 0.3]), 
                                  high=np.array([273.15+35, 273.15+20, 0.5]), 
                                  dtype=np.float32, seed=0)

action_space_Discrete = gym.spaces.Discrete(8, seed=0)

observation_space = gym.spaces.Box(low=np.array([273.15+25, 273.15+10, 0.3, 273.15-50, 273.15-50]), 
                                       high=np.array([273.15+35, 273.15+20, 0.5, 273.15+50, 273.15+50]), 
                                       dtype=np.float32, seed=0)

def reward_function(info):
    # ref: https://ugr-sail.github.io/sinergym/compilation/main/pages/rewards.html
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
    w = 0.1
    #Edemand = Esite/COP/Peff
    P_ac = info['eflx_urban_ac [W/m**2]']/3.6/0.43
    P_heat = info['eflx_urban_heat [W/m**2]']/0.9/0.96
    r = -w*1*(P_ac + P_heat) - (1-w)*1*(abs(info['t_building [K]'] - 24 - 273.15) + abs(info['t_building [K]'] - 18 - 273.15))
    return r

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
        
        
    def step(self, action):

        if isinstance(self.action_space, gym.spaces.Box):
            self.ac_set_point = action[0]
            self.heat_set_point = action[1]
            self.vent_ach = action[2]
        elif isinstance(self.action_space, gym.spaces.Discrete):
            self.ac_set_point = 26.0 + 273.15 if action in [0,1,2,3] else 55.0 + 273.15
            self.heat_set_point = 15.0 + 273.15 if action in [0,1,4,5] else -15.0 + 273.15
            self.vent_ach = 0.5 if action in [0,2,4,6] else 0.3
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
        
        reward = self.reward_function(info)
        
        info['ac_set_point'] = self.ac_set_point
        info['heat_set_point'] = self.heat_set_point
        info['vent_ach'] = self.vent_ach
        
        self.observation = np.array([self.ac_set_point, self.heat_set_point,
                                        self.vent_ach, self.t_building_bef, self.taf])
        
        self.time_step += 1
        self.step_count += 1
        
        terminated = self.step_count >= self.epochnum 
        truncated = self.time_step + 1 >= self.datalen
        
        return self.observation, reward, terminated, truncated, info


    def reset(self, seed=None, options=None):
        
        if seed is not None:
            np.random.seed(seed)
            self.seed = seed
        
        self.time_step = np.random.randint(0, self.datalen//self.epochnum) if options is None else options.get("time_step", 0)
        
        self.time_step = self.time_step * self.epochnum + 1
        
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
        
        self.observation = np.array([self.ac_set_point, self.heat_set_point, self.vent_ach, 
                                     self.t_building_bef, self.taf])
        
        info = {
            "ac_set_point": self.ac_set_point,
            "heat_set_point": self.heat_set_point,
            "vent_ach": self.vent_ach,
            "t_building_bef": self.t_building_bef,
            "taf": self.taf
        }
        
        return self.observation, info
    


    def render(self, mode='human'):
        pass
        

    def close(self):
        pass
    
    class Spec:
        def __init__(self):
            self.id = None  # type: str