#!bin/bash
# how to run: sudo bash run_clmu_sac.sh
# activate conda env
source /home/junjieyu/miniconda3/bin/activate /home/junjieyu/miniconda3/envs/pyclmuapp

echo "Starting run at: `date`"


cities=("beijing" "hongkong" "newyork" "singapore" "london")

# use the absolute path of the model will be better
frocing_paths=("era5_forcing_39.9041999_116.4073963_30_2011_1_2023_12.nc" \
"era5_forcing_22.396428_114.109497_30_2011_1_2023_12.nc" \
"era5_forcing_40.71427_-74.00597_30_2011_1_2023_12.nc" \
"era5_forcing_1.36666666_103.8_30_2011_1_2023_12.nc" \
"forcing.nc")

start_date=("2011-01-01" "2011-01-01" "2011-01-01" "2011-01-01" "2002-01-01")

for i in {0..4}
do
    echo "Start getting clmu for ${cities[i]} at: `date`"
    # use the absolute path of the model will be better
    # use the get_sac_model.py to get the nc_model
    python run_clmu_sac.py --nc_model "sac_models/${cities[i]}.nc" \
        --surf "clmu_input/surfdata_${cities[i]}.nc" \
        --forcing "clmu_input/${frocing_paths[i]}" \
        --case_name "${cities[i]}_sac" \
        --RUN_STARTDATE ${start_date[i]} 
done

echo "End run at: `date`"