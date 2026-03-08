#!bin/bash
# how to run: sudo bash get_sac_model.sh
# activate conda env
source /home/junjieyu/miniconda3/bin/activate /home/junjieyu/miniconda3/envs/pyclmuapp

echo "Starting run at: `date`"


#cities=("beijing" "hongkong" "newyork" "singapore" "london")
cities=("london-0.3" "london-0.5" "london-0.7")

# use the absolute path of the model will be better
model_paths=("/home/junjieyu/Github/RL_CLMU/sac_model/clmux-w-var/clmux-london-0.3__sac_continuous_action__42/sac_continuous_action.sac" \
"/home/junjieyu/Github/RL_CLMU/sac_model/clmux-w-var/clmux-london-0.5__sac_continuous_action__42/sac_continuous_action.sac" \
"/home/junjieyu/Github/RL_CLMU/sac_model/clmux-w-var/clmux-london-0.7__sac_continuous_action__42/sac_continuous_action.sac")

for i in {0..2}
do
    echo "Start getting clmu sac modle for ${cities[i]} at: `date`"
    # use the absolute path of the model will be better
    python get_sac_model.py --model_path ${model_paths[i]} --ouptut_path "data/sac_models/${cities[i]}.nc" 
done

echo "End run at: `date`"