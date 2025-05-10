from pyclmuapp import usp_clmu
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description='run clmu model with sac actor')
    parser.add_argument('--nc_model', type=str, 
                        default='model.nc', 
                        help='Model path')
    parser.add_argument('--surf', type=str,
                        default='surfdata_london.nc',
                        help='Surface data')
    parser.add_argument('--forcing', type=str,
                        default='forcing_london.nc',
                        help='Forcing data')
    parser.add_argument('--case_name', type=str,
                        default='london_sac',
                        help='Case name')
    parser.add_argument('--RUN_STARTDATE', type=str,
                        default='2002-01-01',
                        help='Start date')
    parser.add_argument('--STOP_OPTION', type=str,
                        default='nyears',
                        help='Stop option')
    parser.add_argument('--STOP_N', type=str,
                        default='12',
                        help='Stop number')
                        
    return parser.parse_args()

args = parse_args()
model_path = args.nc_model
surfdata = args.surf
forcing = args.forcing
case_name = args.case_name
RUN_STARTDATE = args.RUN_STARTDATE
STOP_OPTION = args.STOP_OPTION
STOP_N = args.STOP_N


# initialize
usp = usp_clmu(
    pwd=os.getcwd(),
    container_type='docker')

# before running container, you need the image
# usp.docker("pull") # to pull the docker image if you don't have it

usp.docker("run") # run the docker container

# check surface
# here we use the default surface data, which is the london uk-kin data
# lat = 51.5116, lon = -0.1167
usp.check_surf(usr_surfdata=surfdata)
# check the domain
# the domain file will be revised according to the surface data if usr_domain is not provided
# do this after check_surf
# because the surfdata should be provided to read the domain file
usp.check_domain()
# check the forcing
# this forcing derived from urban-plumber forcing data
usp.check_forcing(usr_forcing=forcing)

os.system(f"cp usp_RL.sh {usp.input_path}/usp/usp.sh")
os.system(f"cp {model_path} {usp.input_path}/model.nc")
os.system(f"mkdir -p {usp.input_path}/usp/SourceMods/src.clm")
os.system(f"cp ../src/clmu/clm_varpar.F90 {usp.input_path}/usp/SourceMods/src.clm/clm_varpar.F90")
os.system(f"cp ../src/clmu/SurfaceAlbedoMod.F90 {usp.input_path}/usp/SourceMods/src.clm/SurfaceAlbedoMod.F90")
os.system(f"cp ../src/clmu/sac_actor.f90 {usp.input_path}/usp/SourceMods/src.clm/sac_actor.F90")
os.system(f"cp ../src/clmu/TemperatureType.F90 {usp.input_path}/usp/SourceMods/src.clm/TemperatureType.F90")
os.system(f"cp ../src/clmu/UrbBuildTempOleson2015Mod.F90 {usp.input_path}/usp/SourceMods/src.clm/UrbBuildTempOleson2015Mod.F90")


usp_res = usp.run(
            output_prefix= "_clm.nc",
            case_name = case_name,
            RUN_STARTDATE = RUN_STARTDATE,
            STOP_OPTION = STOP_OPTION,
            STOP_N = STOP_N,
            iflog = True,
            logfile = "log.log",
            run_tyep="usp-exec"#"case", when docker container is not start)
        )

print(usp_res)  # print the result

os.system(f"cp {usp_res[0]} clmu_sac_output/{case_name}_clmu_sac.nc")

usp.case_clean()

usp.docker("stop")
usp.docker("rm")