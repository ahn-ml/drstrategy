import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

import os

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path

import gym
import hydra
import numpy as np
import torch
import wandb
from dm_env import specs

import envs
import utils
from logger import Logger
from replay import ReplayBuffer, make_replay_loader

torch.backends.cudnn.benchmark = True

from dmc_benchmark import PRIMAL_TASKS


def make_agent(obs_type, obs_spec, action_spec, num_expl_steps, cfg):
    cfg.obs_type = obs_type
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    cfg.num_expl_steps = num_expl_steps
    return hydra.utils.instantiate(cfg)


def make_agent(obs_space, action_spec, cur_config, cfg):
    from copy import deepcopy
    cur_config = deepcopy(cur_config)
    del cur_config.agent
    return hydra.utils.instantiate(cfg, cfg=cur_config, obs_space=obs_space, act_spec=action_spec)

class Workspace:
    def __init__(self, cfg, savedir=None, workdir=None):
        self.workdir = Path.cwd() if workdir is None else workdir
        print(f'workspace: {self.workdir}')

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

        # create logger
        self.logger = Logger(self.workdir,
                             use_tb=cfg.use_tb,
                             use_wandb=cfg.use_wandb)
        # create envs
        task = cfg.task if cfg.task != 'none' else PRIMAL_TASKS[self.cfg.domain] # -> which is the URLB default
        frame_stack = 1
        img_size = 64

        self.train_env = envs.make(task, cfg.obs_type, frame_stack,
                                  cfg.action_repeat, cfg.seed, img_size=img_size, cfg=cfg)
        self.eval_env = envs.make(task, cfg.obs_type, frame_stack,
                                 cfg.action_repeat, cfg.seed, img_size=img_size, cfg=cfg)

        cfg = self.update_discrete_action_space(cfg, self.train_env.action_spec())
        self.cfg = cfg


        # # create agent
        self.agent = make_agent(self.train_env.obs_space,
                                self.train_env.action_spec(), cfg, cfg.agent)
        # get meta specs
        meta_specs = self.agent.get_meta_specs()
        # create replay buffer
        if 'rnavmemorymaze3D' in task:
            data_specs = (self.train_env.observation_spec(),
                self.train_env.action_spec(),
                specs.Array((1,), np.float32, 'reward'),
                specs.Array((1,), np.float32, 'discount'),
                specs.Array((3,), np.float32, 'position'),
                specs.Array((2,), np.float32, 'direction'),)
        elif 'rnav' in task:
            data_specs = (self.train_env.observation_spec(),
                        self.train_env.action_spec(),
                        specs.Array((1,), np.float32, 'reward'),
                        specs.Array((1,), np.float32, 'discount'),
                        specs.Array((3,), np.float32, 'position'))
        elif 'robokitchen' in task:
            data_specs = (self.train_env.observation_spec(),
                        self.train_env.action_spec(),
                        specs.Array((1,), np.float32, 'reward'),
                        specs.Array((1,), np.float32, 'discount'),
                        specs.Array((30,), np.float64, 'position'))
        elif 'walker' in task or 'quadruped' in task:
            position_dim = 9 if 'walker' in task else 23
            data_specs = (self.train_env.observation_spec(),
                        self.train_env.action_spec(),
                        specs.Array((1,), np.float32, 'reward'),
                        specs.Array((1,), np.float32, 'discount'),
                        specs.Array((position_dim,), np.float64, 'position'))
        else:
            data_specs = (self.train_env.observation_spec(),
                        self.train_env.action_spec(),
                        specs.Array((1,), np.float32, 'reward'),
                        specs.Array((1,), np.float32, 'discount'))

        # create replay storage
        self.replay_storage = ReplayBuffer(data_specs, meta_specs,
                                                  self.workdir / 'buffer',
                                                  length=cfg.batch_length, **cfg.replay,
                                                  device=cfg.device)

        # create replay buffer
        self.replay_loader = make_replay_loader(self.replay_storage,
                                                cfg.batch_size, #
                                                )
        self._replay_iter = None

        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

        # visited positions
        self._visited_positions_file_path = os.path.join(self.workdir, "visited_positions.txt")
        
        self.run_achiever = self.cfg.run_achiever
        self.non_episodic = self.cfg.non_episodic


    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    def _update_visited_positions(self, time_step):
        if "position" in time_step.keys():
            if "direction" in time_step.keys():
                with open(self._visited_positions_file_path, "a") as f:
                    f.write(str(time_step["position"][0]) + "," + str(time_step["position"][2]) + "," + str(time_step["direction"][0]) + "," + str(time_step["direction"][1]) + "\n")
                f.close()
                return
            with open(self._visited_positions_file_path, "a") as f:
                f.write(str(time_step["position"][0]) + "," + str(time_step["position"][2]) + "\n")
            f.close()

    def eval(self):
        step, episode, total_episode, total_reward, success = 0, 0, 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)
        meta = self.agent.init_meta()
        #log_for_video = []
        while eval_until_episode(episode):
            for room_idx in range(len(self.eval_env.goal_positions)): # number of rooms
                for goal_idx in range(len(self.eval_env.goal_positions[room_idx])):
                    time_step = self.eval_env.reset()
                    goal = self.eval_env.get_goal(room_idx=room_idx, goal_idx=goal_idx)
                    self.agent.reset_behavior(self.eval_env, "achiever", self.agent.get_goal_embedding(goal))
                    #if episode == 0 and "position" in time_step.keys():
                    #    _log_for_video = {"position": [time_step["position"][0], time_step["position"][2]], "observation": time_step["observation"]}
                    #    _log_for_video.update(self.agent.get_skilled_explore_stats())
                    #    log_for_video.append(_log_for_video)
                    agent_state = None
                    while not time_step['is_last']:
                        with torch.no_grad(), utils.eval_mode(self.agent):
                            action, agent_state = self.agent.act(time_step, # time_step.observation
                                                    meta,
                                                    self.global_step,
                                                    eval_mode=True,
                                                    state=agent_state,
                                                    behavior_type="achiever")
                        time_step = self.eval_env.step(action)
                        #if episode == 0 and "position" in time_step.keys():
                        #    _log_for_video = {"position": [time_step["position"][0], time_step["position"][2]], "observation": time_step["observation"]}
                        #    _log_for_video.update(self.agent.get_skilled_explore_stats())
                        #    log_for_video.append(_log_for_video)
                        total_reward += time_step['reward']
                        step += 1
                        # to check the agent is near to the goal
                        if self.eval_env.is_goal_achieved(time_step["position"], direction=time_step.get("direction", None)):
                            success += 1
                            time_step['is_last'] = True
                        
                    total_episode += 1

            episode += 1

        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', total_reward / total_episode)
            log('episode_length', step * self.cfg.action_repeat / total_episode)
            log('episode', self.global_episode)
            log('step', self.global_step)
            log('success_rate', success / total_episode)
            room_goal_ratio = self.eval_env.get_room_goal_ratio()
            for i, _ratio in enumerate(room_goal_ratio):
                log(f'success_rate_for_goals_in_room{i}', _ratio)

        #image, video = self.agent.report_trajectories_with_landmarks(self.train_env, self._visited_positions_file_path, current_trajectory=log_for_video)
        #self.logger.log_image(image, self.global_frame)
        #self.logger.log_video(video, self.global_frame)

    def train(self):
        # predicates
        train_until_step = utils.Until(self.cfg.num_train_frames, self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_seed_frames, self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames, self.cfg.action_repeat)
        train_every_n_steps = self.cfg.train_every_actions // self.cfg.action_repeat
        should_train_step = utils.Every(train_every_n_steps * self.cfg.action_repeat, self.cfg.action_repeat)
        should_log_scalars = utils.Every(self.cfg.log_every_frames, self.cfg.action_repeat)
        should_log_recon = utils.Every(self.cfg.recon_every_frames, self.cfg.action_repeat)
        should_reset_behavior = utils.Every(self.cfg.reset_behavior_every_frames, self.cfg.action_repeat)
        should_save_model = utils.Every(self.cfg.save_model_every_frames, self.cfg.action_repeat)

        episode_step, episode_reward, success = 0, 0, 0
        time_step = self.train_env.reset()
        behavior_type, goal = "random", None
        self.agent.reset_behavior(self.train_env, behavior_type, goal)
        agent_state = None
        meta = self.agent.init_meta()
        self.replay_storage.add(time_step, meta)
        metrics = None
        while train_until_step(self.global_step):
            if time_step['is_last']:
                self._global_episode += 1
                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat
                    with self.logger.log_and_dump_ctx(self.global_frame, ty='train') as log:
                        log('fps', episode_frame / elapsed_time)
                        log('total_time', total_time)
                        log('episode_reward', episode_reward)
                        log('episode_length', episode_frame)
                        log('episode', self.global_episode)
                        log('success_rate', success)
                        log('buffer_size', len(self.replay_storage))
                        log('step', self.global_step)
                    self.logger.log_global_step(self.global_step)
                    # save last model
                    self.save_last_model()

                # reset env
                if self.non_episodic:
                    prev_pos = time_step["position"]
                    time_step = self.train_env.reset(pos=[prev_pos[0], prev_pos[2]])
                else:
                    time_step = self.train_env.reset()
                if seed_until_step(self.global_step):
                    behavior_type = "random"
                else:
                    if self.run_achiever:                    
                        behavior_type = "explorer" if self._global_episode % 2 == 0 else "achiever"
                    else:
                        behavior_type = "explorer"
                if behavior_type == "achiever":
                    goal, goal_position = self.agent.get_goal_from_replay_buffer(next(self.replay_iter))
                    self.agent.reset_behavior(self.train_env, behavior_type, goal)
                else:
                    goal, goal_position = None, None
                self._update_visited_positions(time_step) # for visualization
                agent_state = None # Resetting agent's latent state
                meta = self.agent.init_meta()
                self.replay_storage.add(time_step, meta)
                # try to save snapshot
                # if self.global_frame in self.cfg.snapshots:
                    # self.save_snapshot()
                episode_step = 0
                episode_reward = 0
                success = 0
            
            if should_reset_behavior(self.global_step) and behavior_type == "explorer":
                self.agent.reset_behavior(self.train_env, behavior_type, goal)

            # try to evaluate
            # if eval_every_step(self.global_step) and self.global_step > 0:
            #     self.logger.log('eval_total_time', self.timer.total_time(), self.global_frame)
            #     self.eval()

            meta = self.agent.update_meta(meta, self.global_step, time_step)
            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                if seed_until_step(self.global_step) or behavior_type == "random": # even though we are over the seed phase, we still want to prefill the buffer with random actions before starting a new episode
                    action =  self.train_env.act_space['action'].sample()
                else:
                    action, agent_state = self.agent.act(time_step, meta, self.global_step, eval_mode=False, state=agent_state, behavior_type=behavior_type)
                    
            # if behavior_type == "achiever":
            #     if self.train_env.is_goal_achieved(time_step["position"], goal_position, direction=time_step.get("direction", None)):
            #         success = 1

            # try to update the agent
            if not seed_until_step(self.global_step):
                if should_train_step(self.global_step):
                    metrics = self.agent.update(next(self.replay_iter), self.global_step)[1]
                if should_log_scalars(self.global_step):
                    # if 'rnav' in self.cfg.task:
                    #     metrics.update(self.agent.get_visitation_stats(self.train_env, self._visited_positions_file_path))
                    self.logger.log_metrics(metrics, self.global_frame, ty='train')
                # if self.global_step > 0 and should_log_recon(self.global_step):
                #     videos = self.agent.report(next(self.replay_iter))
                #     self.logger.log_video(videos, self.global_frame)
                #     image = self.agent.report_trajectories_with_landmarks(self.train_env, self._visited_positions_file_path)
                #     self.logger.log_image(image, self.global_frame)

            # take env step
            time_step = self.train_env.step(action)
            self._update_visited_positions(time_step) # for visualization
            episode_reward += time_step['reward']
            self.replay_storage.add(time_step, meta)
            episode_step += 1
            self._global_step += 1

            if should_save_model(self.global_step):
                self.save_pt_snapshot()

    @utils.retry
    def save_snapshot(self):
        snapshot = self.get_snapshot_dir() / f'snapshot_{self.global_frame}.pt'
        keys_to_save = ['agent', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)
            
    @utils.retry
    def save_pt_snapshot(self):
        snapshot = self.get_save_snapshot_dir() / f'snapshot_{self.global_frame}.pt'
        keys_to_save = ['agent', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def setup_wandb(self):
        self.wandb_run_id = utils.setup_wandb(self.cfg, workdir=self.workdir)
        
    @utils.retry
    def save_last_model(self):
        snapshot = self.root_dir / 'last_snapshot.pt'
        if snapshot.is_file():
            temp = Path(str(snapshot).replace("last_snapshot.pt", "second_last_snapshot.pt"))
            os.replace(snapshot, temp)
        keys_to_save = ['agent', '_global_step', '_global_episode']
        if self.cfg.use_wandb:
            keys_to_save.append('wandb_run_id')
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def load_snapshot(self):
        try:
            snapshot = self.root_dir / 'last_snapshot.pt'
            with snapshot.open('rb') as f:
                payload = torch.load(f)
        except:
            snapshot = self.root_dir / 'second_last_snapshot.pt'
            with snapshot.open('rb') as f:
                payload = torch.load(f)
        for k,v in payload.items():
            setattr(self, k, v)
            if k == 'wandb_run_id':
                assert wandb.run is None
                utils.setup_wandb(self.cfg, workdir=self.workdir, load=True, id=v)

    def get_snapshot_dir(self):
        snap_dir = self.cfg.snapshot_dir
        snapshot_dir = self.workdir / Path(snap_dir)
        snapshot_dir.mkdir(exist_ok=True, parents=True)
        return snapshot_dir
    
    def get_save_snapshot_dir(self):
        snap_dir = self.cfg.snapshot_dir
        snapshot_dir = Path(snap_dir)
        snapshot_dir.mkdir(exist_ok=True, parents=True)
        return snapshot_dir

    def update_discrete_action_space(self, cfg, act_spec):
        discrete = 'int' in act_spec.dtype.name
        if cfg.actor.dist == 'auto':
            cfg.actor.dist = 'onehot' if discrete else 'trunc_normal'
        if cfg.actor_grad == 'auto':
            cfg.actor_grad = 'reinforce' if discrete else 'dynamics'
        return cfg


@hydra.main(config_path='.', config_name='pretrain')
def main(cfg):
    root_dir = Path.cwd() if cfg.resume_dir == 'none' else Path(cfg.resume_dir)
    if cfg.resume_dir != 'none':
        workspace = Workspace(cfg, workdir=root_dir)
    else:
        workspace = Workspace(cfg)
    workspace.root_dir = root_dir
    snapshot = workspace.root_dir / 'last_snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()
    if cfg.use_wandb and wandb.run is None:
        # otherwise it was resumed
        workspace.setup_wandb()
    workspace.train()

if __name__ == '__main__':
    main()
