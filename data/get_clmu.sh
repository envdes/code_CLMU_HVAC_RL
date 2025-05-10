#!bin/bash
# how to run: sudo bash get_clmu.sh
# activate conda env
source /home/junjieyu/miniconda3/bin/activate /home/junjieyu/miniconda3/envs/pyclmuapp

echo "Starting run at: `date`"

python get_clmu.py > get_hac_clmu.log

echo "End run at: `date`"