from pyclmuapp import usp_clmu
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description='run clmu model with dqn')
    parser.add_argument('--nc_model', type=str, 
                        default='model.nc', 
                        help='Model path')
    parser.add_argument('--price_ele', type=str, 
                        default='/home/junjieyu/Github/RL_CLMU/data/plotting_w/london_avgprice.csv', 
                        help='Price electricity data path')
    parser.add_argument('--surf', type=str,
                        default='surfdata_london.nc',
                        help='Surface data')
    parser.add_argument('--urban_hac', type=str,
                        default='ON_WASTEHEAT',
                        help='Urban HAC')
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
price_ele = args.price_ele
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
usp.docker("stop")
usp.docker("rm")
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
os.system(f"cp {model_path} {usp.input_path}/model.bin")
os.system(f"cp {price_ele} {usp.input_path}/london_avgprice.csv")
os.system(f"mkdir -p {usp.input_path}/usp/SourceMods/src.clm")
os.system(f"cp ../src/clmu_dqn/clm_varpar.F90 {usp.input_path}/usp/SourceMods/src.clm/clm_varpar.F90")
os.system(f"cp ../src/clmu_dqn/SurfaceAlbedoMod.F90 {usp.input_path}/usp/SourceMods/src.clm/SurfaceAlbedoMod.F90")
os.system(f"cp ../src/clmu_dqn/dqn.F90 {usp.input_path}/usp/SourceMods/src.clm/dqn.F90")
os.system(f"cp ../src/clmu_dqn/TemperatureType.F90 {usp.input_path}/usp/SourceMods/src.clm/TemperatureType.F90")
os.system(f"cp ../src/clmu_dqn/UrbBuildTempOleson2015Mod.F90 {usp.input_path}/usp/SourceMods/src.clm/UrbBuildTempOleson2015Mod.F90")
os.system(f"cp ../src/clmu_dqn/UrbanFluxesMod.F90 {usp.input_path}/usp/SourceMods/src.clm/UrbanFluxesMod.F90")

URBAN_HAC = args.urban_hac

if URBAN_HAC == "off":
    URBAN_HAC = "OFF"
elif URBAN_HAC == "on":
    URBAN_HAC = "ON"
else:
    URBAN_HAC = "ON_WASTEHEAT"

usp_res = usp.run(
            output_prefix= "_clm.nc",
            case_name = f"{case_name}_{URBAN_HAC}",
            RUN_STARTDATE = RUN_STARTDATE,
            STOP_OPTION = STOP_OPTION,
            STOP_N = STOP_N,
            iflog = True,
            logfile = "dqn.log",
            urban_hac = URBAN_HAC,
            crun_type="usp-exec"#"case", when docker container is not start)
        )

print(usp_res)  # print the result

os.makedirs("clmu_dqn_output", exist_ok=True)
os.system(f"cp {usp_res[0]} clmu_dqn_output/{case_name}_clmu_dqn_{URBAN_HAC}.nc")
usp.case_clean()
os.system(f"rm -rf {usp.input_path}")
os.system(f"rm -rf {usp.output_path}")
os.system(f"rm -rf {usp.log_path}")
os.system(f"rm -rf {usp.scripts_path}")

usp.docker("stop")
usp.docker("rm")