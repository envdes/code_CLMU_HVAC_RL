#!/bin/bash
# how to run: sudo bash dqn_workflow.sh
# activate conda env
source /home/junjieyu/miniconda3/bin/activate /home/junjieyu/miniconda3/envs/pyclmuapp

docker stop myclmu
docker rm myclmu
rm -rf /home/junjieyu/Github/RL_CLMU/data/inputfolder /home/junjieyu/Github/RL_CLMU/data/logfolder \
            /home/junjieyu/Github/RL_CLMU/data/scriptsfolder /home/junjieyu/Github/RL_CLMU/data/outputfolder \
            /home/junjieyu/Github/RL_CLMU/data/sac_models

#mkdir /home/junjieyu/Github/RL_CLMU/data/sac_models

#cd /home/junjieyu/Github/RL_CLMU/scripts
#bash train_w.sh > train_w.log
rm -rf /home/junjieyu/Github/RL_CLMU/data/clmu_dqn_output
mkdir /home/junjieyu/Github/RL_CLMU/data/clmu_dqn_output
cd /home/junjieyu/Github/RL_CLMU/data
#sudo bash get_sac_model.sh
#sudo bash run_clmu_sac.sh
mkdir dqn_models
bash get_dqn_model.sh
bash run_clmu_dqn.sh > run_clmu_dqn.log
#sudo bash run_clmu_dqn_realprice.sh > run_clmu_dqn_realprice.log
#sudo bash run_clmu_sac_transfer.sh

