#!bin/bash
# how to run: sudo bash get_sac_model.sh
# activate conda env
source /home/junjieyu/miniconda3/bin/activate /home/junjieyu/miniconda3/envs/pyclmuapp

echo "Starting run at: `date`"


cities=("beijing" "hongkong" "newyork" "singapore" "london")

# use the absolute path of the model will be better
model_paths=("/home/junjieyu/Github/CLMUX_0.5/sac_model/clmux-0.5/clmux-beijing__sac_continuous_action__42__1753199563/sac_continuous_action.sac" \
"/home/junjieyu/Github/CLMUX_0.5/sac_model/clmux-0.5/clmux-hongkong__sac_continuous_action__42__1753199562/sac_continuous_action.sac" \
"/home/junjieyu/Github/CLMUX_0.5/sac_model/clmux-0.5/clmux-newyork__sac_continuous_action__42__1753199563/sac_continuous_action.sac" \
"/home/junjieyu/Github/CLMUX_0.5/sac_model/clmux-0.5/clmux-singapore__sac_continuous_action__42__1753199563/sac_continuous_action.sac" \
"/home/junjieyu/Github/CLMUX_0.5/sac_model/clmux-0.5/clmux-london__sac_continuous_action__42__1753199562/sac_continuous_action.sac")

for i in {0..4}
do
    echo "Start getting clmu sac modle for ${cities[i]} at: `date`"
    # use the absolute path of the model will be better
    python get_sac_model.py --model_path ${model_paths[i]} --ouptut_path "data/sac_models/${cities[i]}.nc" 
done

echo "End run at: `date`"