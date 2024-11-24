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
import torch.nn.functional as F

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

        # evaluation results
        self._eval_result_file_path = os.path.join(self.workdir, "results.txt")
        
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

    def _update_eval_results(self, results):
        with open(self._eval_result_file_path, "a") as f:
            f.write(str(results["epoch"]) + "," + str(results["SE"]) + "," + str(results["LE"]) + "\n")
        f.close()
            
    def find_closest_landmark_idx(self, _goal):
        zero_traj_len=3
        with torch.no_grad():
            # zero action trajectory
            embed = self.agent.get_goal_embedding(_goal)
            action = torch.zeros((1, zero_traj_len,) + self.agent.act_spec.shape, device=self.agent.device)
            is_first = torch.zeros(action.shape[:-1], device=self.agent.device)
            is_first[:, 0] = 1
            embed = embed[None, ...].repeat(1, zero_traj_len, 1)
            post, _ = self.agent.wm.rssm.observe(embed, action, is_first, state=None)
            ze = self.agent.landmark_module.landmark_encoder(post['deter'][0,-1][None, ...]).mean
            zq, idx = self.agent.landmark_module.emb(ze, training=False)
        return idx, post

    def eval(self, eval_type="SE"):
        step, episode, total_episode, total_reward, success, landmark2landmark_success = 0, 0, 0, 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)
        meta = self.agent.init_meta()
        if 'robokitchen' in self.cfg.task:
            self.eval_env.reset_statics()
        while eval_until_episode(episode):
            for room_idx in range(len(self.eval_env.goal_positions)): # number of rooms
                for goal_idx in range(len(self.eval_env.goal_positions[room_idx])):
                    step = 0
                    time_step = self.eval_env.reset() # In memory-maze it is required to reset before get_goal and warping. This will fix the error of memory-maze
                    goal = self.eval_env.get_goal(room_idx=room_idx, goal_idx=goal_idx)
                    if eval_type == "SE":
                        target_landmark, _ = self.find_closest_landmark_idx(goal)
                        meta['landmark'] = np.zeros(self.agent.landmark_dim, dtype=np.float32)
                        meta['landmark'][target_landmark] = 1.0
                        target_landmark_deter = self.agent.landmark_module.emb.weight.T[target_landmark]
                        with torch.no_grad():
                            target_landmark_deter = self.agent.landmark_module.landmark_decoder(target_landmark_deter).mean
                    _behavior_type = "achiever" if eval_type == "LE" else "landmark2landmark"
                    self.agent.reset_behavior(self.eval_env, _behavior_type, self.agent.get_goal_embedding(goal))
                    agent_state = None
                    while not time_step['is_last']:
                        with torch.no_grad(), utils.eval_mode(self.agent):
                            action, agent_state = self.agent.act(time_step, # time_step.observation
                                                    meta,
                                                    self.global_step,
                                                    eval_mode=True,
                                                    state=agent_state,
                                                    behavior_type=_behavior_type)
                        time_step = self.eval_env.step(action)
                        total_reward += time_step['reward']
                        step += 1
                        
                        if eval_type == "SE" and _behavior_type == "landmark2landmark":
                            is_landmark_achieved = F.mse_loss(agent_state['latent']['deter'].squeeze(), target_landmark_deter.squeeze()) < 0.07
                            if step > self.cfg.agent.skilled_explore.max_step_for_landmark_executor or is_landmark_achieved:
                                if is_landmark_achieved:
                                    landmark2landmark_success += 1
                                _behavior_type = "achiever"
                                self.agent.reset_behavior(self.eval_env, _behavior_type, self.agent.get_goal_embedding(goal))
                                                
                        # check the agent is near to the goal
                        if self.eval_env.is_goal_achieved(time_step["position"], direction=time_step.get("direction", None)):
                            success += 1
                            time_step['is_last'] = True
                        
                    total_episode += 1

            episode += 1
        
        num_8, num_6, num_4, num_2, num_0 = 0, 0, 0, 0, 0
        if 'kitchen' in self.cfg.task:
            room_goal_ratio = self.eval_env.get_room_goal_ratio()
            for (name, _ratio) in zip(self.eval_env.goal_configs, room_goal_ratio):
                if _ratio > 0.8:
                    num_8 += 1
                if _ratio > 0.6:
                    num_6 += 1
                if _ratio > 0.4:
                    num_4 += 1
                if _ratio > 0.2:
                    num_2 += 1
                if _ratio > 0:
                    num_0 += 1
                
            
        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', total_reward / total_episode)
            log('episode_length', step * self.cfg.action_repeat / total_episode)
            log('episode', self.global_episode)
            log('step', self.global_step)
            log('total_success_rate', success / total_episode)
            log('number_of_tasks_8', num_8)
            log('number_of_tasks_6', num_6)
            log('number_of_tasks_4', num_4)
            log('number_of_tasks_2', num_2)
            log('number_of_tasks_0', num_0)
            room_goal_ratio = self.eval_env.get_room_goal_ratio()
            if 'rnav' in self.cfg.task:
                for i, _ratio in enumerate(room_goal_ratio):
                    log(f'success_rate_for_goals_in_room{i}', _ratio)
            elif 'kitchen' in self.cfg.task:
                for (name, _ratio) in zip(self.eval_env.goal_configs, room_goal_ratio):
                    real_name = '_'.join(name)
                    log(f'success_rate_{real_name}', _ratio)

        success_rate = success / total_episode
        landmark2landmark_success_rate = landmark2landmark_success / total_episode
        return success_rate, landmark2landmark_success_rate
    
    def eval_snapshots(self):
        files = [file for file in os.listdir(self.cfg.eval_dir) if file.endswith('pt')]
        files = sorted(files, key=lambda x: os.path.getmtime(os.path.join(Path(self.cfg.eval_dir), x)))
        for file_name in files:
            _epoch = file_name.split("_")[1].split(".")[0]
            snapshot = Path(os.path.join(self.cfg.eval_dir, file_name))
            if snapshot.exists():
                print(f'resuming: {snapshot}')
                self.load_snapshot(_snapshot = snapshot)
            
            LE_success_rate, SE_sucess_rate, landmark2landmark_success_rate = 0, 0, 0
            if self.cfg.agent.use_skilled_explore:
                SE_sucess_rate, landmark2landmark_success_rate = self.eval(eval_type="SE")
            LE_success_rate, _ = self.eval(eval_type="LE")
            self._update_eval_results({"epoch": _epoch, "SE": SE_sucess_rate, "LE": LE_success_rate})
            print(f"epoch: {_epoch}, SE success rate: {SE_sucess_rate}, LE success rate: {LE_success_rate}")
            with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
                log('episode', self.global_episode)
                log('step', self.global_step)
                log('epoch', int(_epoch))
                log('SE_success_rate', SE_sucess_rate)
                log('SE_arrival_to_landmark_rate', landmark2landmark_success_rate)
                log('LE_success_rate', LE_success_rate)
    
    def load_snapshot(self, _snapshot):
        try:
            snapshot = Path(_snapshot)
            with snapshot.open('rb') as f:
                payload = torch.load(f)
        except:
            snapshot = self.root_dir / 'second_last_snapshot.pt'
            with snapshot.open('rb') as f:
                payload = torch.load(f)
        for k,v in payload.items():
            setattr(self, k, v)
    
    def setup_wandb(self):
        self.wandb_run_id = utils.setup_wandb(self.cfg, workdir=self.workdir)
        
    # From Sungwon @swy99 https://github.com/hany606/zeroshot/commit/0e31309ed5262deef9be01c1612481c501cb63e3
    def update_discrete_action_space(self, cfg, act_spec):
        discrete = 'int' in act_spec.dtype.name
        if cfg.actor.dist == 'auto':
            cfg.actor.dist = 'onehot' if discrete else 'trunc_normal'
        if cfg.actor_grad == 'auto':
            cfg.actor_grad = 'reinforce' if discrete else 'dynamics'
        return cfg


@hydra.main(config_path='.', config_name='eval')
def main(cfg):
    root_dir = Path.cwd() 
    workspace = Workspace(cfg)
    workspace.root_dir = root_dir
    if cfg.use_wandb and wandb.run is None:
        workspace.setup_wandb()
    workspace.eval_snapshots()

if __name__ == '__main__':
    main()
