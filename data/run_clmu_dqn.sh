#!bin/bash
# how to run: sudo bash run_clmu_dqn.sh > run_clmu_dqn.log
# activate conda env
source /home/junjieyu/miniconda3/bin/activate /home/junjieyu/miniconda3/envs/pyclmuapp

echo "Starting run at: `date`"
rm -rf /home/junjieyu/Github/RL_CLMU/data/inputfolder /home/junjieyu/Github/RL_CLMU/data/logfolder /home/junjieyu/Github/RL_CLMU/data/scriptsfolder


#cities=("beijing" "hongkong" "newyork" "singapore" "london")
cities=("london-0.3" "london-0.5" "london-0.7")

# use the absolute path of the model will be better
#frocing_paths=("era5_forcing_39.9041999_116.4073963_30_2011_1_2023_12.nc" \
#"era5_forcing_22.396428_114.109497_30_2011_1_2023_12.nc" \
#"era5_forcing_40.71427_-74.00597_30_2011_1_2023_12.nc" \
#"era5_forcing_1.36666666_103.8_30_2011_1_2023_12.nc" \
#"forcing.nc")
frocing_paths=("forcing.nc" \
                "forcing.nc" \
                "forcing.nc")

urban_hac=("ON_WASTEHEAT" "on")

#start_date=("2011-01-01" "2011-01-01" "2011-01-01" "2011-01-01" "2002-01-01")
start_date=("2002-01-01" "2002-01-01" "2002-01-01")
STOP_N=12  # 12 years
for i in {0..2}
do
    echo "Start getting clmu for ${cities[i]} at: `date`"
    # use the absolute path of the model will be better
    # use the get_sac_model.py to get the nc_model
    # #--surf "clmu_input/surfdata_${cities[i]}.nc" \
    python run_clmu_dqn.py --nc_model "dqn_models/${cities[i]}.bin" \
        --surf "clmu_input/surfdata_london.nc" \
        --forcing "clmu_input/${frocing_paths[i]}" \
        --case_name "${cities[i]}_dqn" \
        --RUN_STARTDATE ${start_date[i]} \
        --urban_hac "ON_WASTEHEAT" \
        --STOP_N ${STOP_N}
done

for i in {0..2}
do
    echo "Start getting clmu for ${cities[i]} at: `date`"
    
    python run_clmu_dqn.py --nc_model "dqn_models/${cities[i]}.bin" \
        --surf "clmu_input/surfdata_london.nc" \
        --forcing "clmu_input/${frocing_paths[i]}" \
        --case_name "${cities[i]}_dqn" \
        --RUN_STARTDATE ${start_date[i]} \
        --urban_hac "on"\
        --STOP_N ${STOP_N}
done

echo "End run at: `date`"