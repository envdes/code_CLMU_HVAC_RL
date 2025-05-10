#! /bin/bash
# This script is used to create a new case with a user-defined dataset
# Junjie YU, 2024-03-13, Manchester

# --------------------------------Input of ENV variables--------------------------------
# ! note:  export the path of plumber and the input data beafore running this scripts
# export plumber=/p/scratch/CESMDATAROOT/inputdata/Urban-PLUMBER
# export CASESRPITS=/p/project/clm5.0/cime/scripts
# platform: UoM csf3; Linux; MacOS.
# * How to use? bash plumber.sh -h for help

# --------------------------------ENV--------------------------------
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
export OMPI_ALLOW_RUN_AS_ROOT=1
export CASESRPITS=/p/project/clm5.0/cime/scripts 
export USER=root 
export PROJECT=/p/project
export SCRATCH=/p/scratch 
export CESMDATAROOT=/p/scratch/CESMDATAROOT 
export CSMDATA=/p/scratch/CESMDATAROOT/inputdata 
export CASESCRIPT=/p/project/clm5.0/cime/scripts
usage() {
    echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --ATMDOM_FILE <ATMDOM_FILE>  The domain file of the atmosphere"
      echo "  --SURF <SURF>  The surface dataset"
      echo "  --FORCING_FILE <FORCING_FILE>  The forcing file"
      echo "  --case_name <case_name>  The name of the case"
      echo "  --RUN_STARTDATE <RUN_STARTDATE>  The start date of the run"
      echo "  --START_TOD <START_TOD>  The start time of day"
      echo "  --STOP_OPTION <STOP_OPTION>  The stop option"
      echo "  --STOP_N <STOP_N>  The stop number"
      echo "  --CLIMATE_VAR <CLIMATE_VAR>  The climate variable"
      echo "  --START <START>  The start year"
      echo "  --END <END>  The end year"
      echo "  --RUN_TYPE <RUN_TYPE>  The run type"
      echo "  --RUN_REFCASE <RUN_REFCASE>  The reference case"
      echo "  --RUN_REFDATE <RUN_REFDATE>  The reference date"
      echo "  --RUN_REFTOD <RUN_REFTOD>  The reference time of day"
      echo "  --urban_hac <urban_hac>  The urban heat and anthropogenic heat"
      exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
         --ATMDOM_FILE)
            ATMDOM_FILE=$2
            shift 2
            ;;
         --SURF)
            SURF=$2
            shift 2
            ;;
         --FORCING_FILE)
            FORCING_FILE=$2
            shift 2
            ;;
         --case_name)
            case_name=$2
            shift 2
            ;;
         --RUN_STARTDATE)
            RUN_STARTDATE=$2
            shift 2
            ;;
         --START_TOD)
            START_TOD=$2
            shift 2
            ;;
         --STOP_OPTION)
            STOP_OPTION=$2
            shift 2
            ;;
         --STOP_N)
            STOP_N=$2
            shift 2
            ;;
         --hist_type)
            hist_type=$2
            shift 2
            ;;
         --hist_nhtfrq)
            hist_nhtfrq=$2
            shift 2
            ;;
         --hist_mfilt)
            hist_mfilt=$2
            shift 2
            ;;
         --START)
            START=$2
            shift 2
            ;;
         --END)
            END=$2
            shift 2
            ;;
         --RUN_TYPE)
            RUN_TYPE=$2
            shift 2
            ;;
         --RUN_REFCASE)
            RUN_REFCASE=$2
            shift 2
            ;;
         --RUN_REFDATE)
            RUN_REFDATE=$2
            shift 2
            ;;
        --RUN_REFTOD)
            RUN_REFTOD=$2
            shift 2
            ;;
         --urban_hac)
            urban_hac=$2
            shift 2
            ;;
         *)
            usage
            ;;
    esac
done


# Set the environment variables
# --------------------------------Input--------------------------------
export ctsm_input=/p/clmuapp
export ATMDOM_PATH=${ctsm_input}/usp
#export ATMDOM_FILE=domain.lnd.domain.lnd.1x1_${GRIDNAME}_noocean.nc_domain.ocn_noocean.nc.210525.nc
export FORCING_PATH=${ctsm_input}/usp
export SURF=${ctsm_input}/usp/${SURF}
#export FORCING_FILE=CTSM_DATM_${GRIDNAME}_${FORCING_DATE}.nc
# --------------------------------Input--------------------------------

# --------------------------------Setting--------------------------------
# Create a new case
cd ${CASESRPITS}
if [ -d ${case_name} ]; then
   echo "The case ${case_name} already exists."
   cd ${case_name}
else
   ./create_newcase --case ${case_name} --res CLM_USRDAT --compset 2000_DATM%1PT_CLM50%SP_SICE_SOCN_SROF_SGLC_SWAV --run-unsupported
   cd ${case_name}
   echo "domainfile='$ATMDOM_PATH/$ATMDOM_FILE'" >> user_nl_datm
   echo "fsurdat='${SURF}'" >> user_nl_clm
   echo "hist_avgflag_pertape='A'" >> user_nl_clm
   echo "hist_nhtfrq=1000000000" >> user_nl_clm
   echo "hist_mfilt=1" >> user_nl_clm
   cp ${ctsm_input}/usp/SourceMods/src.clm/* ${case_name}/SourceMods/src.clm/

   ./xmlchange DATM_MODE=CLM1PT
   # Set up the case
   ./xmlchange NTASKS=1
   ./xmlchange CALENDAR=GREGORIAN # or NOLEAP
   #./xmlchange CLM_USRDAT_NAME=${GRIDNAME}
   #./xmlchange DIN_LOC_ROOT_CLMFORC=${FORCING_PATH}
   ./case.setup
   ./case.build

fi

#./preview_namelists
# Modify the DATA namelist
# ref: https://www2.cesm.ucar.edu/models/cesm1.2/data8/doc/x310.html

# Modify the CLM namelist
echo "fsurdat='${SURF}'" >> user_nl_clm_jj
echo "hist_empty_htapes=.true." >> user_nl_clm_jj
variables=$hist_type
formatted_variables=$(echo $variables | sed "s/\([^,]\+\)/'\1'/g")
hist_type1d_pertape=$formatted_variables
IFS=',' read -ra hist_array <<< "$hist_type1d_pertape"

length=${#hist_array[@]}

hist_nhtfrq_l="${hist_nhtfrq},"
for (( i=1; i<=length; i++ ))
do
    new_hist_nhtfrq+=${hist_nhtfrq_l}
done

hist_mfilt_l="${hist_mfilt},"
for (( i=1; i<=length; i++ ))
do
    new_hist_mfilt+=${hist_mfilt_l}
done

new_hist_dov2xy=$(printf '.false.,'%.0s $(seq 1 $length))
new_hist_nhtfrq=${new_hist_nhtfrq%,}
new_hist_mfilt=${new_hist_mfilt%,}
new_hist_dov2xy=${new_hist_dov2xy%,}

#variables=$hist_nhtfrq
#new_hist_nhtfrq=$variables
#
#variables=$hist_mfilt
#new_hist_mfilt=$variables

echo "hist_nhtfrq=${new_hist_nhtfrq}" >> user_nl_clm_jj
echo "hist_mfilt=${new_hist_mfilt}" >> user_nl_clm_jj
echo "hist_dov2xy=${new_hist_dov2xy}" >> user_nl_clm_jj
echo "hist_type1d_pertape=' '" >> user_nl_clm_jj
echo "urban_hac='${urban_hac}'" >> user_nl_clm_jj

if [ $length -eq 1 ]; then
    echo "hist_fincl1='AC_SET_POINT', 'HEAT_SET_POINT', 'VENT_ACH', 'TFLOOR', 'TSHDW_INNER', 'TSUNW_INNER', 'TROOF_INNER', 'FSA', 'FIRA', 'URBAN_HEAT', 'WASTEHEAT', 'EFLX_LH_TOT', 'FSH', 'FGR', 'URBAN_AC', 'RAIN', 'SNOW', 'SNOCAN', 'H2OCAN', 'QVEGT', 'QVEGE', 'QSOIL', 'QOVER', 'QH2OSFC', 'QDRAI', 'QDRAI_PERCH', 'SOILLIQ', 'SOILICE', 'H2OSNO', 'H2OSFC', 'HIA','SWBGT','WBT', 'HUMIDEX', 'TWS', 'TRAFFICFLUX', 'TBUILD', 'COSZEN', 'ZWT', 'QFLX_EVAP_TOT', 'RH2M','TG','TSKIN','TBOT','FIRE','FLDS','FSDS','FSR','FGEV','TSOI','ERRSOI','SABV','SABG','FSDSVD','FSDSND','FSDSVI','FSDSNI','FSRVD','FSRND','FSRVI','FSRNI','TSA','FCTR','FCEV','QBOT','Q2M','H2OSOI','SWup','LWup','Rnet','Qh','Qle','Qstor','Qtau','Wind','Qair','Tair','PSurf','Rainf','SWdown','LWdown','SoilAlpha_U','TBUILD','TWS','TRAFFICFLUX','COSZEN'" >> user_nl_clm_jj

elif [ $length -eq 2 ]; then
    echo "hist_fincl1='AC_SET_POINT', 'HEAT_SET_POINT', 'VENT_ACH', 'TFLOOR', 'TSHDW_INNER', 'TSUNW_INNER', 'TROOF_INNER', 'FSA', 'FIRA', 'URBAN_HEAT', 'WASTEHEAT', 'EFLX_LH_TOT', 'FSH', 'FGR', 'URBAN_AC', 'RAIN', 'SNOW', 'SNOCAN', 'H2OCAN', 'QVEGT', 'QVEGE', 'QSOIL', 'QOVER', 'QH2OSFC', 'QDRAI', 'QDRAI_PERCH', 'SOILLIQ', 'SOILICE', 'H2OSNO', 'H2OSFC', 'HIA','SWBGT','WBT', 'HUMIDEX', 'TWS', 'TRAFFICFLUX', 'TBUILD', 'COSZEN', 'ZWT', 'QFLX_EVAP_TOT', 'RH2M','TG','TSKIN','TBOT','FIRE','FLDS','FSDS','FSR','FGEV','TSOI','ERRSOI','SABV','SABG','FSDSVD','FSDSND','FSDSVI','FSDSNI','FSRVD','FSRND','FSRVI','FSRNI','TSA','FCTR','FCEV','QBOT','Q2M','H2OSOI','SWup','LWup','Rnet','Qh','Qle','Qstor','Qtau','Wind','Qair','Tair','PSurf','Rainf','SWdown','LWdown','SoilAlpha_U','TBUILD','TWS','TRAFFICFLUX','COSZEN'" >> user_nl_clm_jj
    echo "hist_fincl2='AC_SET_POINT', 'HEAT_SET_POINT', 'VENT_ACH', 'TFLOOR', 'TSHDW_INNER', 'TSUNW_INNER', 'TROOF_INNER', 'FSA', 'FIRA', 'URBAN_HEAT', 'WASTEHEAT', 'EFLX_LH_TOT', 'FSH', 'FGR', 'URBAN_AC', 'RAIN', 'SNOW', 'SNOCAN', 'H2OCAN', 'QVEGT', 'QVEGE', 'QSOIL', 'QOVER', 'QH2OSFC', 'QDRAI', 'QDRAI_PERCH', 'SOILLIQ', 'SOILICE', 'H2OSNO', 'H2OSFC', 'HIA','SWBGT','WBT', 'HUMIDEX', 'TWS', 'TRAFFICFLUX', 'TBUILD', 'COSZEN', 'ZWT', 'QFLX_EVAP_TOT', 'RH2M','TG','TSKIN','TBOT','FIRE','FLDS','FSDS','FSR','FGEV','TSOI','ERRSOI','SABV','SABG','FSDSVD','FSDSND','FSDSVI','FSDSNI','FSRVD','FSRND','FSRVI','FSRNI','TSA','FCTR','FCEV','QBOT','Q2M','H2OSOI','SWup','LWup','Rnet','Qh','Qle','Qstor','Qtau','Wind','Qair','Tair','PSurf','Rainf','SWdown','LWdown','SoilAlpha_U','TBUILD','TWS','TRAFFICFLUX','COSZEN'" >> user_nl_clm_jj

else
    echo "hist_fincl1='AC_SET_POINT', 'HEAT_SET_POINT', 'VENT_ACH', 'TFLOOR', 'TSHDW_INNER', 'TSUNW_INNER', 'TROOF_INNER', 'FSA', 'FIRA', 'URBAN_HEAT', 'WASTEHEAT', 'EFLX_LH_TOT', 'FSH', 'FGR', 'URBAN_AC', 'RAIN', 'SNOW', 'SNOCAN', 'H2OCAN', 'QVEGT', 'QVEGE', 'QSOIL', 'QOVER', 'QH2OSFC', 'QDRAI', 'QDRAI_PERCH', 'SOILLIQ', 'SOILICE', 'H2OSNO', 'H2OSFC', 'HIA','SWBGT','WBT', 'HUMIDEX', 'TWS', 'TRAFFICFLUX', 'TBUILD', 'COSZEN', 'ZWT', 'QFLX_EVAP_TOT', 'RH2M','TG','TSKIN','TBOT','FIRE','FLDS','FSDS','FSR','FGEV','TSOI','ERRSOI','SABV','SABG','FSDSVD','FSDSND','FSDSVI','FSDSNI','FSRVD','FSRND','FSRVI','FSRNI','TSA','FCTR','FCEV','QBOT','Q2M','H2OSOI','SWup','LWup','Rnet','Qh','Qle','Qstor','Qtau','Wind','Qair','Tair','PSurf','Rainf','SWdown','LWdown','SoilAlpha_U','TBUILD','TWS','TRAFFICFLUX','COSZEN'" >> user_nl_clm_jj
    echo "hist_fincl2='AC_SET_POINT', 'HEAT_SET_POINT', 'VENT_ACH', 'TFLOOR', 'TSHDW_INNER', 'TSUNW_INNER', 'TROOF_INNER', 'FSA', 'FIRA', 'URBAN_HEAT', 'WASTEHEAT', 'EFLX_LH_TOT', 'FSH', 'FGR', 'URBAN_AC', 'RAIN', 'SNOW', 'SNOCAN', 'H2OCAN', 'QVEGT', 'QVEGE', 'QSOIL', 'QOVER', 'QH2OSFC', 'QDRAI', 'QDRAI_PERCH', 'SOILLIQ', 'SOILICE', 'H2OSNO', 'H2OSFC', 'HIA','SWBGT','WBT', 'HUMIDEX', 'TWS', 'TRAFFICFLUX', 'TBUILD', 'COSZEN', 'ZWT', 'QFLX_EVAP_TOT', 'RH2M','TG','TSKIN','TBOT','FIRE','FLDS','FSDS','FSR','FGEV','TSOI','ERRSOI','SABV','SABG','FSDSVD','FSDSND','FSDSVI','FSDSNI','FSRVD','FSRND','FSRVI','FSRNI','TSA','FCTR','FCEV','QBOT','Q2M','H2OSOI','SWup','LWup','Rnet','Qh','Qle','Qstor','Qtau','Wind','Qair','Tair','PSurf','Rainf','SWdown','LWdown','SoilAlpha_U','TBUILD','TWS','TRAFFICFLUX','COSZEN'" >> user_nl_clm_jj
    echo "hist_fincl3='AC_SET_POINT', 'HEAT_SET_POINT', 'VENT_ACH', 'TFLOOR', 'TSHDW_INNER', 'TSUNW_INNER', 'TROOF_INNER', 'FSA', 'FIRA', 'URBAN_HEAT', 'WASTEHEAT', 'EFLX_LH_TOT', 'FSH', 'FGR', 'URBAN_AC', 'RAIN', 'SNOW', 'SNOCAN', 'H2OCAN', 'QVEGT', 'QVEGE', 'QSOIL', 'QOVER', 'QH2OSFC', 'QDRAI', 'QDRAI_PERCH', 'SOILLIQ', 'SOILICE', 'H2OSNO', 'H2OSFC', 'HIA','SWBGT','WBT', 'HUMIDEX', 'TWS', 'TRAFFICFLUX', 'TBUILD', 'COSZEN', 'ZWT', 'QFLX_EVAP_TOT', 'RH2M','TG','TSKIN','TBOT','FIRE','FLDS','FSDS','FSR','FGEV','TSOI','ERRSOI','SABV','SABG','FSDSVD','FSDSND','FSDSVI','FSDSNI','FSRVD','FSRND','FSRVI','FSRNI','TSA','FCTR','FCEV','QBOT','Q2M','H2OSOI','SWup','LWup','Rnet','Qh','Qle','Qstor','Qtau','Wind','Qair','Tair','PSurf','Rainf','SWdown','LWdown','SoilAlpha_U','TBUILD','TWS','TRAFFICFLUX','COSZEN'" >> user_nl_clm_jj
fi

cp user_nl_clm_jj user_nl_clm
rm user_nl_clm_jj

# Modify the DATM namelist
./xmlchange LND_DOMAIN_FILE=${ATMDOM_FILE}
./xmlchange LND_DOMAIN_PATH=${ATMDOM_PATH}
./xmlchange ATM_DOMAIN_FILE=${ATMDOM_FILE}
./xmlchange ATM_DOMAIN_PATH=${ATMDOM_PATH}
./xmlchange DATM_CLMNCEP_YR_START=${START}
./xmlchange DATM_CLMNCEP_YR_END=${END}
./xmlchange DATM_CLMNCEP_YR_ALIGN=${START}
# Modify the DATA namelist
# ref: https://www2.cesm.ucar.edu/models/cesm1.2/data8/doc/x310.html
cat <<EOF > user_datm.streams.txt.CLM1PT.CLM_USRDAT
<?xml version="1.0"?>
<file id="stream" version="1.0">
<dataSource>
   GENERIC
</dataSource>
<domainInfo>
  <variableNames>
     time    time
        xc      lon
        yc      lat
        area    area
        mask    mask
  </variableNames>
  <filePath>
     $ATMDOM_PATH
  </filePath>
  <fileNames>
     $ATMDOM_FILE
  </fileNames>
</domainInfo>
<fieldInfo>
   <variableNames>
     Zbot     z
        Prectmms precn
        Wind     wind
        LWdown   lwdn
        PSurf    pbot
        Qair     shum
        Tair     tbot
        SWdown   swdn
   </variableNames>
   <filePath>
    $FORCING_PATH
   </filePath>
   <fileNames>
    $FORCING_FILE
   </fileNames>
   <offset>
      0
   </offset>
</fieldInfo>
</file>
EOF

# Modify the DATM namelist

./xmlchange RUN_STARTDATE=${RUN_STARTDATE}
./xmlchange START_TOD=${START_TOD}
./xmlchange STOP_OPTION=${STOP_OPTION}
./xmlchange STOP_N=${STOP_N}

echo "RUN_TYPE=${RUN_TYPE}"
if [ "${RUN_TYPE}" = "coldstart" ];
then
   echo "--------------coldstart-----------------"
    ./xmlchange CLM_FORCE_COLDSTART=on
fi

if [ "${RUN_TYPE}" = "branch" ];
then
   # Set up the case of restart
   echo "----------------branch-----------------"
   #export RUN_REFDIR=/p/scratch/CESMDATAROOT/CaseOutputs/${RUN_REFCASE}/run
   export RUN_REFDIR=/p/scratch/CESMDATAROOT/Archive/rest/${RUN_REFDATE}-${RUN_REFTOD}
   ./xmlchange RUN_TYPE=branch
   ./xmlchange RUN_REFDIR=${RUN_REFDIR}
   ./xmlchange RUN_REFDATE=${RUN_REFDATE}
   ./xmlchange RUN_REFTOD=${RUN_REFTOD}
   ./xmlchange GET_REFCASE=TRUE
   ./xmlchange RUN_REFCASE=${RUN_REFCASE}
fi

./preview_namelists

./case.submit