#!/usr/bin/sh
# SAGA
wandb=True
group=drstrategy # drstrategy, lexa, gc_director, lexa_explore_lbs, lexa_explore_p2e,  
num_train_frames=7000010
num_seed_frames=2500
log_every_frames=1000
save_model_every_frames=50000
seed=1
task=robokitchen
skilled_explore=True
landmark2landmark=True
train_achiever=True
run_achiever=True
achiever_sample=random # traj or random
non_episodic=False
landmark_dim=128
code_dim=16
reset_behavior_every_frames=150
SE_maxstep=25
CUDA_VISIBLE_DEVICES=0 python pretrain.py configs=robokitchen_pixels agent=drstrategy domain=robokitchen task=robokitchen_$task \
    num_train_frames=$num_train_frames \
    num_seed_frames=$num_seed_frames\
    log_every_frames=$log_every_frames \
    save_model_every_frames=$save_model_every_frames \
    agent.use_skilled_explore=$skilled_explore \
    agent.use_landmark2landmark=$landmark2landmark \
    agent.use_achieve=$train_achiever \
    run_achiever=$run_achiever \
    achiever_sample=$achiever_sample \
    non_episodic=$non_episodic \
    agent.landmark_dim=$landmark_dim \
    agent.code_dim=$code_dim \
    reset_behavior_every_frames=$reset_behavior_every_frames \
    agent.skilled_explore.max_step_for_landmark_executor=$SE_maxstep \
    snapshot_dir=/root/baseline/$task/$group/$seed \
    seed=$seed use_wandb=$wandb group=$group tags=\"$HostName,GPU_$ServerNum,$task,$group,seed_$seed\"
