#!bin/bash
# how to run: sudo bash get_sac_model.sh
# activate conda env
source /home/junjieyu/miniconda3/bin/activate /home/junjieyu/miniconda3/envs/pyclmuapp

echo "Starting run at: `date`"


cities=("beijing" "hongkong" "newyork" "singapore" "london")

# use the absolute path of the model will be better
model_paths=("sac_model/clmux/_clmux-beijing__sac_continuous_action__1__1727035402/sac_continuous_action.sac" \
"sac_model/clmux/_clmux-hongkong__sac_continuous_action__1__1727035402/sac_continuous_action.sac" \
"sac_model/clmux/_clmux-newyork__sac_continuous_action__1__1727035402/sac_continuous_action.sac" \
"sac_model/clmux/_clmux-singapore__sac_continuous_action__1__1727035402/sac_continuous_action.sac" \
"sac_model/clmux/_clmux-london__sac_continuous_action__1__1727035402/sac_continuous_action.sac")

for i in {0..4}
do
    echo "Start getting clmu sac modle for ${cities[i]} at: `date`"
    # use the absolute path of the model will be better
    python get_sac_model.py --model_path ${model_paths[i]} --ouptut_path "data/sac_models/${cities[i]}.nc" 
done

echo "End run at: `date`"