# how to use this script?
# 1. install the pyclmuapp package
# 2. download the surface data and forcing data
# 3. run this script: sudo python get_clmu.py > get_clmu.log

from pyclmuapp import usp_clmu
import os
import shutil
import xarray as xr
import numpy as np
# if the docker container is not start, you can use the following code to start the container
# $ pyclmuapp --has_container False --container_type docker --init True


coordinates=[
    ["39.9041999", "116.4073963"],
    ["22.396428", "114.109497"],
    ["40.71427", "-74.00597"],
    ["1.36666666", "103.8"]]

cities = ["beijing", "hongkong", "newyork", "singapore"]

surface_data = {
    city: f"clmu_input/surfdata_{city}.nc" for city in cities
}
surface_data["london"] = "clmu_input/surfdata_london.nc"

forcing_data = {
    city: f"clmu_input/era5_forcing_{coord[0]}_{coord[1]}_30_2011_1_2023_12.nc" 
    for coord, city in zip(coordinates, cities)
}

forcing_data["london"] = "clmu_input/forcing.nc"

print("surface_data", surface_data)
print("forcing_data", forcing_data)


def get_data(usp,city,hac, heating_set_point=None):

    # initialize
    usp = usp_clmu(
            pwd=os.getcwd(),
            container_type='docker')

    shutil.copytree('fsrc', usp.input_path+'/usp/SourceMods/src.clm', dirs_exist_ok=True)
    #os.system(f"cp usp_RL.sh {usp.input_path}/usp/usp.sh")
    # check surface
    surf_data=usp.check_surf(usr_surfdata=surface_data[city])
    # check the domain
    usp.check_domain()
    # check the forcing
    usp.check_forcing(
        usr_forcing=forcing_data[city])

    os.system(f"cp usp.sh ./inputfolder/usp/usp.sh")

    if city == "london":
        RUN_STARTDATE = "2002-01-01"
    else:
        RUN_STARTDATE = "2011-01-01"
        
    if hac == "off":
        URBAN_HAC = "OFF"
    elif hac == "on":
        URBAN_HAC = "ON"
    else:
        URBAN_HAC = "ON_WASTEHEAT"

    if heating_set_point is not None:
        ds = xr.open_dataset(f"/home/junjieyu/Github/RL_CLMU/data/clmu_input/surfdata_{city}.nc")
        ds['LT_BUILDING_MAX'] = ds['T_BUILDING_MIN']
        # ref: https://www.sciencedirect.com/science/article/pii/S2210670724006358#bib0071
        ds['LT_BUILDING_MAX'].values = np.zeros(ds['LT_BUILDING_MAX'].values.shape) + 24 + 273.15
        ds['T_BUILDING_MIN'].values = np.zeros(ds['T_BUILDING_MIN'].shape)  + heating_set_point + 273.15
        ds['ALB_ROOF_DIR'].attrs['long_name'] = "Direct solar albedo of roof surface"
        ds['ALB_ROOF_DIR'].attrs['units'] = "fraction"
        ds['ALB_ROOF_DIF'].attrs['long_name'] = "Diffuse solar albedo of roof surface"
        ds['ALB_ROOF_DIF'].attrs['units'] = "fraction"
        if os.path.exists(f"surfdata_{city}_{heating_set_point}.nc"):
            os.remove(f"surfdata_{city}_{heating_set_point}.nc")
        ds.to_netcdf(f"surfdata_{city}_{heating_set_point}.nc")

        surf_data=usp.check_surf(usr_surfdata=f"surfdata_{city}_{heating_set_point}.nc")

    if heating_set_point is not None:
        case_name = f"default_heating_{heating_set_point}"
    else:
        case_name = "default"
    usp_case = usp.run(
                output_prefix= f"{city}_clm.nc",
                case_name = case_name, 
                RUN_STARTDATE = RUN_STARTDATE,
                STOP_OPTION = "nyears", 
                STOP_N = "12",
                iflog = True,
                logfile = "log.log",
                hist_type = "PFTS",
                urban_hac = URBAN_HAC,
                crun_type="usp-exec"#"case", when docker container is not start)
            )
    print("default", ":", usp_case)
    
    if not os.path.exists(f"hac_{hac}/{city}"):
        os.makedirs(f"hac_{hac}/{city}", exist_ok=True)
    
    if heating_set_point is not None:
        os.system(f"cp {usp_case[0]} hac_{hac}/{city}/default_heating_{heating_set_point}.nc")
    else:
        os.system(f"cp {usp_case[0]} hac_{hac}/{city}/default.nc")

    usp.case_clean(case_name="default")

        
if __name__ == "__main__":
    # initialize
    usp = usp_clmu(
        pwd=os.getcwd(),
        container_type='docker')
    usp.docker("stop") # stop the docker container
    usp.docker("rm") # remove the docker container
    # before running container, you need the image
    # usp.docker("pull") # to pull the docker image if you don't have it
    usp.docker("run") # run the docker container
    
    #for hac in ["on_wasteheat", "on", "off"]:
    #    for city in surface_data.keys():
    #        if city == "london":
    #            print(city, hac)
    #            get_data(usp,city,hac)
    heat_set_points = [18, 20, 22, 24]
    for heating_set_point in heat_set_points:
        for city in surface_data.keys():
            if city == "london":
                for hac in ["on_wasteheat", "on"]:
                    print(city, hac, heating_set_point)
                    get_data(usp,city,hac, heating_set_point=heating_set_point)
    
    usp.docker("stop") # stop the docker container
    usp.docker("rm") # remove the docker container
    os.system(f"rm -rf {usp.input_path}")
    os.system(f"rm -rf {usp.output_path}")
    os.system(f"rm -rf {usp.log_path}")
    os.system(f"rm -rf {usp.scripts_path}")