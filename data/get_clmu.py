# how to use this script?
# 1. install the pyclmuapp package
# 2. download the surface data and forcing data
# 3. run this script: sudo python get_clmu.py > get_clmu.log

from pyclmuapp import usp_clmu
import os

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



def get_data(usp,city,hac):

    # initialize
    usp = usp_clmu(
            pwd=os.getcwd(),
            container_type='docker')

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

    usp_case = usp.run(
                output_prefix= f"{city}_clm.nc",
                case_name = "default", 
                RUN_STARTDATE = RUN_STARTDATE,
                STOP_OPTION = "nyears", 
                STOP_N = "12",
                iflog = True,
                logfile = "log.log",
                hist_type = "PFTS",
                urban_hac = URBAN_HAC,
                run_tyep="usp-exec"#"case", when docker container is not start)
            )
    print("default", ":", usp_case)
    
    if not os.path.exists(f"hac_{hac}/{city}"):
        os.makedirs(f"hac_{hac}/{city}", exist_ok=True)
    
    os.system(f"cp {usp_case[0]} hac_{hac}/{city}/default.nc")

    usp.case_clean(case_name="default")
        
if __name__ == "__main__":
    # initialize
    usp = usp_clmu(
        pwd=os.getcwd(),
        container_type='docker')

    # before running container, you need the image
    # usp.docker("pull") # to pull the docker image if you don't have it
    usp.docker("run") # run the docker container
    
    for hac in ["on_wasteheat", "on", "off"]:
        for city in surface_data.keys():
            print(city, hac)
            get_data(usp,city,hac)
    
    usp.docker("stop") # stop the docker container
    usp.docker("rm") # remove the docker container