#!bin/bash
# how to run: sudo bash get_dqn_model.sh
# activate conda env
source /home/junjieyu/miniconda3/bin/activate /home/junjieyu/miniconda3/envs/pyclmuapp

echo "Starting run at: `date`"


#cities=("beijing" "hongkong" "newyork" "singapore" "london")
cities=("london-0.3" "london-0.5" "london-0.7")

# use the absolute path of the model will be better
model_paths=("/home/junjieyu/Github/RL_CLMU/dqn_model/clmux-w-var/clmux-london-0.3__dqn__42/dqn.dqn" \
"/home/junjieyu/Github/RL_CLMU/dqn_model/clmux-w-var/clmux-london-0.5__dqn__42/dqn.dqn" \
"/home/junjieyu/Github/RL_CLMU/dqn_model/clmux-w-var/clmux-london-0.7__dqn__42/dqn.dqn")

for i in {0..2}
do
    echo "Start getting clmu dqn model for ${cities[i]} at: `date`"
    # use the absolute path of the model will be better
    python get_dqn_model.py --model_path ${model_paths[i]} --ouptut_path "data/dqn_models/${cities[i]}.bin" 
done

echo "End run at: `date`"