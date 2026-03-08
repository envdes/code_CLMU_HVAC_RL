#!/bin/bash
#Junjie Yu, 2024-08-08

# how to use ?
# bash train_w.sh > train_w.log

# activate conda env
source /home/junjieyu/miniconda3/bin/activate /home/junjieyu/miniconda3/envs/pyclmuapp

echo "Starting run at: `date`"
rm -rf /home/junjieyu/Github/RL_CLMU/dqn_model
rm -rf /home/junjieyu/Github/RL_CLMU/sac_model
rm -rf /home/junjieyu/Github/RL_CLMU/q_table
rm -rf /home/junjieyu/Github/RL_CLMU/tensorboard/clmux-w-var
rm -rf /home/junjieyu/Github/RL_CLMU/wandb
#cities=("beijing" "hongkong" "newyork" "singapore" "london")
cities=("london")
reward_weight=(0.3 0.5 0.7)
algo=("sac" "dqn" "qlearning")
#algo=("sac")
task_limit=15  # Number of concurrent tasks
counter=0

for ao in "${algo[@]}"; do
    for c in "${cities[@]}"; do
        for rw in "${reward_weight[@]}"; do
            echo "Processing city: ${c} with algorithm: ${ao} and reward weight: ${rw}"
            python train_w.py --city "${c}" --algo "${ao}" --reward_weight "${rw}" &
            ((counter++))

            # If counter reaches the task limit, wait for the current batch to finish
            if [[ $counter -eq $task_limit ]]; then
                wait
                counter=0  # Reset counter after waiting
            fi
        done
    done
    #wait  # Wait for remaining processes if any
    echo "Finished processing algo: ${ao}"
done

echo "End run at: $(date)"
