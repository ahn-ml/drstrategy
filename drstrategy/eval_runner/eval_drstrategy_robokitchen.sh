wandb=True
group=drstrategy # drstrategy, lexa, gc_director, lexa_explore_lbs, lexa_explore_p2e,  
num_seed_frames=0
num_eval_episodes=10
seed=1
task=robokitchen
skilled_explore=True
SE_maxstep=25
CUDA_VISIBLE_DEVICES=0 python eval.py configs=robokitchen_pixels agent=drstrategy domain=robokitchen task=robokitchen_$task \
    num_seed_frames=$num_seed_frames \
    num_eval_episodes=$num_eval_episodes \
    agent.use_skilled_explore=$skilled_explore \
    agent.skilled_explore.max_step_for_landmark_executor=$SE_maxstep \
    seed=$seed use_wandb=$wandb group=$group tags=\"$HostName,GPU_$ServerNum,$task,$group,seed_$seed\" \
    eval_dir=/root/baseline/$task/$group/$seed