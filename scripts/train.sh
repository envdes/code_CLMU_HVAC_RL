#!/bin/bash
#Junjie Yu, 2024-08-08

# how to use ?
# bash train.sh > train.log

# activate conda env
source /home/junjieyu/miniconda3/bin/activate /home/junjieyu/miniconda3/envs/pyclmuapp

echo "Starting run at: `date`"

cities=("beijing" "hongkong" "newyork" "singapore" "london")
algo=("ppo" "sac" "dqn" "qlearning")
task_limit=10  # Number of concurrent tasks
counter=0

for ao in "${algo[@]}"; do
    for c in "${cities[@]}"; do
        echo "Processing city: ${c} with algorithm: ${ao}"
        python train.py --city "${c}" --algo "${ao}" &
        ((counter++))

        # If counter reaches the task limit, wait for the current batch to finish
        if [[ $counter -eq $task_limit ]]; then
            wait
            counter=0  # Reset counter after waiting
        fi
    done
    #wait  # Wait for remaining processes if any
    echo "Finished processing algo: ${ao}"
done

echo "End run at: $(date)"
