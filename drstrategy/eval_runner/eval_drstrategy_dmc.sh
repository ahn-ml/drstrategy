#!/usr/bin/sh
# SAGA
wandb=True
project_name=icml24_eval # icml24_eval
group=drstrategy # drstrategy, lexa, gc_director, lexa_explore_lbs, lexa_explore_p2e,  
num_seed_frames=0
num_eval_episodes=3
seed=1
task=walk
skilled_explore=True
SE_maxstep=300
CUDA_VISIBLE_DEVICES=0 python eval.py configs=dmc_pixels agent=p2eDrStrategy domain=dmcyoga task=walker_$task \
    num_seed_frames=$num_seed_frames \
    num_eval_episodes=$num_eval_episodes \
    agent.use_skilled_explore=$skilled_explore \
    agent.skilled_explore.max_step_for_landmark_executor=$SE_maxstep \
    seed=$seed use_wandb=$wandb group=$group project_name=$project_name tags=\"$HostName,GPU_$ServerNum,$task,$group,seed_$seed\" \
    eval_dir=/root/baseline/$task/$group/$seed