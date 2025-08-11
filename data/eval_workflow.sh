#!bin/bash
# how to run: sudo bash eval_workflow.sh
# activate conda env
source /home/junjieyu/miniconda3/bin/activate /home/junjieyu/miniconda3/envs/pyclmuapp

sudo bash get_sac_model.sh
sudo bash run_clmu_sac.sh
sudo bash run_clmu_sac_transfer.sh

