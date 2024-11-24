import torch.nn as nn
import torch

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import io
from matplotlib.ticker import LogLocator
from matplotlib.colors import LogNorm

from collections import OrderedDict
import numpy as np
from dm_env import specs

import utils
import agent.net_utils as common
from agent.mb_utils import *
from agent.landmark_behavior import *
from agent.landmark_rep import *

# taken from https://github.com/orybkin/lexa/blob/master/lexa/tools.py#L806C1-L816C75
def get_future_goal_idxs(seq_len, bs):
    cur_idx_list = []
    goal_idx_list = []
    #generate indices grid
    for cur_idx in range(seq_len):
      for goal_idx in range(cur_idx, seq_len):
        cur_idx_list.append(np.concatenate([np.ones((bs,1), dtype=np.int32)*cur_idx, np.arange(bs).reshape(-1,1)], axis = -1))
        goal_idx_list.append(np.concatenate([np.ones((bs,1), dtype=np.int32)*goal_idx, np.arange(bs).reshape(-1,1)], axis = -1))
    return np.concatenate(cur_idx_list,0), np.concatenate(goal_idx_list,0)

# taken from https://github.com/orybkin/lexa/blob/master/lexa/tools.py#L818C1-L823C31  
def get_future_goal_idxs_neg_sampling(num_negs, seq_len, bs, batch_len):
    cur_idxs = np.random.randint((0,0), (seq_len, bs), size=(num_negs,2))
    goal_idxs = np.random.randint((0,0), (seq_len, bs), size=(num_negs,2))
    for i in range(num_negs):
      goal_idxs[i,1] = np.random.choice([j for j in range(bs) if j//batch_len != cur_idxs[i,1]//batch_len])
    return cur_idxs, goal_idxs
  
class DrStrategyAgent(nn.Module):
  def __init__(self, name, cfg, obs_space, act_spec, **kwargs):
    super().__init__()
    self.name = name
    self.cfg = cfg
    self.cfg.update(**kwargs)
    self.obs_space = obs_space
    self.act_spec = act_spec
    self.tfstep = None
    self._use_amp = (cfg.precision == 16)
    self.device = cfg.device
    self.act_dim = act_spec.shape[0]

    # World model
    self.wm = WorldModel(cfg, obs_space, self.act_dim, self.tfstep)
    self.wm.recon_landmarks = True

    # Landmarks
    self.landmark_dim = kwargs['landmark_dim']
    self.landmark_pbe = utils.PBE(utils.RMS(self.device), kwargs['knn_clip'], kwargs['knn_k'], kwargs['knn_avg'], kwargs['knn_rms'], self.device)

    self.landmark_module = LandmarkModule(self.cfg, self.landmark_dim, code_dim=kwargs['code_dim'], code_resampling=kwargs['code_resampling'], resample_every=kwargs['resample_every'])
    self.wm.landmark_module = self.landmark_module

    self._landmark_behavior = LandmarkActorCritic(self.cfg, self.act_spec, self.tfstep, self.landmark_dim, self.landmark_module)
    if cfg.use_landmark2landmark:
      self._landmark_behavior = LandmarkToLandmarkActorCritic(self.cfg, self.act_spec, self.tfstep, self.landmark_dim, self.landmark_module, statetostate=False)
    
    # Explorer
    if cfg.use_skilled_explore:
      self.use_skilled_explore = True
      self._env_behavior = SingleMetaCtrlAC(cfg, "explorer", self.act_spec, self.landmark_dim, self.tfstep, self._landmark_behavior, self.landmark_module).to(self.device)
    else:
      self.use_skilled_explore = False
      self._env_behavior = ActorCritic(cfg, self.act_spec, self.tfstep)
    self.lbs = common.MLP(self.wm.inp_size, (1,), **self.cfg.reward_head).to(self.device)
    self.lbs_opt = common.Optimizer('lbs', self.lbs.parameters(), **self.cfg.model_opt, use_amp=self._use_amp)
    self.lbs.train()
    
    # Achiever
    if cfg.use_achieve:
      self.use_achieve = True
      if cfg.use_skilled_achieve:
        self.use_skilled_achieve = True
        self._env_behavior2 = SingleMetaCtrlAC(cfg, "achiever", self.act_spec, self.landmark_dim, self.tfstep, self._landmark_behavior, self.landmark_module, goal_conditioned=True, goal_dim=self.wm.embed_dim).to(self.device)
      else:
        self.use_skilled_achieve = False
        self._env_behavior2 = ActorCritic(cfg, self.act_spec, self.tfstep, goal_conditioned=True, goal_dim=self.wm.embed_dim).to(self.device)
      # Temporal Distance Predictor
      self.tdp = common.MLP(self.wm.embed_dim*2, (1,), **self.cfg.reward_head).to(self.device)
      self.tdp_opt = common.Optimizer('tdp', self.tdp.parameters(), **self.cfg.model_opt, use_amp=self._use_amp)
      self.tdp.train()
      self.td_cur_idxs, self.td_goal_idxs = get_future_goal_idxs(seq_len=cfg.imag_horizon, bs=cfg.batch_size*cfg.batch_length)
    else:
      self.use_achieve = False

    # Adaptation
    self.num_init_frames = kwargs['num_init_frames']
    self.update_task_every_step = self.update_landmark_every_step = kwargs['update_landmark_every_step']
    self.is_ft = False

    # Common
    self.to(self.device)
    self.requires_grad_(requires_grad=False)

  def init_meta(self):
      return self.init_meta_discrete()

  def get_meta_specs(self):
      return (specs.Array((self.landmark_dim,), np.float32, 'landmark'),)

  def init_meta_discrete(self):
      landmark = np.zeros(self.landmark_dim, dtype=np.float32)
      landmark[np.random.choice(self.landmark_dim)] = 1.0
      meta = OrderedDict()
      meta['landmark'] = landmark
      return meta

  def update_meta(self, meta, global_step, time_step):
      if global_step % self.update_landmark_every_step == 0:
          return self.init_meta()
      return meta

  def act(self, obs, meta, step, eval_mode, state, behavior_type="explorer"):
    # Infer current state
    obs = {k : torch.as_tensor(np.copy(v), device=self.device).unsqueeze(0) for k, v in obs.items()}
    meta = {k : torch.as_tensor(np.copy(v), device=self.device).unsqueeze(0) for k, v in meta.items()}
    if state is None:
      latent = self.wm.rssm.initial(len(obs['reward']))
      action = torch.zeros((len(obs['reward']),) + self.act_spec.shape, device=self.device)
    else:
      latent, action = state["latent"], state["action"]
    embed = self.wm.encoder(self.wm.preprocess(obs))
    should_sample = (not eval_mode) or (not self.cfg.eval_state_mean)
    latent, _ = self.wm.rssm.obs_step(latent, action, embed, obs['is_first'], should_sample)
    feat = self.wm.rssm.get_feat(latent)

    if not self.is_ft:
      if behavior_type == "landmark2landmark":
        landmark = meta["landmark"]
        inp = torch.cat([feat, landmark], dim=-1)
        actor = self._landmark_behavior.actor(inp)
        action = actor.mean if eval_mode else actor.sample()
        new_state = {"latent": latent, "action": action, "landmark": landmark}
        return action.cpu().numpy()[0], new_state
      if behavior_type == "explorer":
        if self.use_skilled_explore:
          action = self._env_behavior.act(feat, latent["deter"], eval_mode=eval_mode)
        else:
          action = self._env_behavior.act(feat, eval_mode=eval_mode)
      elif behavior_type == "achiever":
        if self.use_skilled_achieve:
          action = self._env_behavior2.act(feat, latent["deter"], eval_mode=eval_mode)
        else:
          action = self._env_behavior2.act(feat, eval_mode=eval_mode)
      new_state = {"latent": latent, "action": action}
      return action.cpu().numpy()[0], new_state

  def pbe_reward_fn(self, seq):
    rep = seq['deter']
    B, T, _ = rep.shape
    reward = self.landmark_pbe(rep.reshape(B*T, -1), cdist=True, apply_log=False).reshape(B, T, 1)
    return reward.detach()

  def code_reward_fn(self, seq):
    T, B, _ = seq['landmark'].shape
    landmark_target = seq['landmark'].reshape(T*B, -1)
    vq_landmark = landmark_target @ self.landmark_module.emb.weight.T
    state_pred = self.landmark_module.landmark_decoder(vq_landmark).mean.reshape(T, B, -1)
    reward = -torch.norm(state_pred - seq['deter'], p=2, dim=-1).reshape(T, B, 1)
    return reward

  def landmark_mi_fn(self, seq):
    ce_rw   = self.code_reward_fn(seq)
    ent_rw  = self.pbe_reward_fn(seq)
    return ent_rw + ce_rw

  def update_lbs(self, outs):
    metrics = dict()
    B, T, _ = outs['feat'].shape
    feat, kl = outs['feat'].detach(), outs['kl'].detach()
    feat = feat.reshape(B*T, -1)
    kl = kl.reshape(B*T, -1)

    loss = -self.lbs(feat).log_prob(kl).mean()
    metrics.update(self.lbs_opt(loss, self.lbs.parameters()))
    metrics['lbs_loss'] = loss.item()
    return metrics

  def update_behavior(self, state=None, outputs=None, metrics={}, data=None):
    if outputs is not None:
      post = outputs['post']
      is_terminal = outputs['is_terminal']
    else:
      data = self.wm.preprocess(data)
      embed = self.wm.encoder(data)
      post, _ = self.wm.rssm.observe(
          embed, data['action'], data['is_first'])
      is_terminal = data['is_terminal']
    #
    start = {k: stop_gradient(v) for k,v in post.items()}
    # Train landmark (module + AC)
    start['feat'] = stop_gradient(self.wm.rssm.get_feat(start))
    metrics.update(self.landmark_module.update(start))
    metrics.update(self._landmark_behavior.update(
        self.wm, start, is_terminal, self.landmark_mi_fn))
    return start, metrics

  def update_wm(self, data, step):
    metrics = {}
    state, outputs, mets = self.wm.update(data, state=None)
    outputs['is_terminal'] = data['is_terminal']
    metrics.update(mets)
    return state, outputs, metrics

  # inspired by https://github.com/orybkin/lexa/blob/master/lexa/gcdreamer_imag.py#L157-L188
  def update_tdp(self, start, goal):
    metrics = dict()
    is_terminal = torch.zeros([start["deter"].shape[0], 1, 1]).to(self.device)
    seq = self.wm.imagine(self._env_behavior2.actor, start, is_terminal, self.cfg.imag_horizon, goal_cond=goal)
    def _helper(cur_idxs, goal_idxs, distance):
      cur_states = {k: v[cur_idxs[:,0],cur_idxs[:,1]] for k, v in seq.items()}
      goal_states = {k: v[goal_idxs[:,0],goal_idxs[:,1]] for k, v in seq.items()}
      pred = self.tdp(torch.cat([self.wm.heads["emb_decoder"](self.wm.rssm.get_feat(cur_states))["embed"].mean, self.wm.heads["emb_decoder"](self.wm.rssm.get_feat(goal_states))["embed"].mean], dim=-1)).mean
      distance = distance/self.cfg.imag_horizon
      return F.mse_loss(pred.squeeze(), torch.Tensor(distance).to(self.device))
    # positives
    idxs = np.random.choice(np.arange(len(self.td_cur_idxs)), self.cfg.td_num_positives)
    loss = _helper(self.td_cur_idxs[idxs], self.td_goal_idxs[idxs], self.td_goal_idxs[idxs][:,0] - self.td_cur_idxs[idxs][:,0])
    # negatives
    if self.cfg.td_neg_sampling_factor > 0:
      seq_len, bs = seq["deter"].shape[:2]
      num_negs = int(self.cfg.td_neg_sampling_factor*self.cfg.td_num_positives)
      neg_cur_idxs, neg_goal_idxs = get_future_goal_idxs_neg_sampling(num_negs, seq_len, bs, self.cfg.batch_length)
      loss += _helper(neg_cur_idxs, neg_goal_idxs, torch.ones(num_negs)*seq_len)
    metrics.update(self.tdp_opt(loss, self.tdp.parameters()))
    metrics['temporal_distance_loss'] = loss.item()
    return metrics
  
  def update(self, data, step):
    # Train WM
    metrics = {}
    state, outputs, mets = self.wm.update(data, state=None)
    metrics.update(mets)
    start = outputs['post']
    start = {k: stop_gradient(v) for k,v in start.items()}
    if not self.is_ft:
      # Train landmark (module + AC)
      start['feat'] = stop_gradient(self.wm.rssm.get_feat(start))
      metrics.update(self.landmark_module.update(start))
      metrics.update(self._landmark_behavior.update(self.wm, start, data['is_terminal'], self.landmark_mi_fn))

      # LBS exploration
      with common.RequiresGrad(self.lbs):
        with torch.cuda.amp.autocast(enabled=self._use_amp):
            metrics.update(self.update_lbs(outputs))
      reward_fn = lambda seq: self.lbs(seq['feat']).mean
      mets = self._env_behavior.update(self.wm, start, data['is_terminal'], reward_fn)
      metrics.update({f"explorer_{k}": v for k, v in mets.items()})

      # Goal-conditioned achievement
      if self.use_achieve: 
        goal = self.wm.get_goal(data, sample=self.cfg.achiever_sample)
        with common.RequiresGrad(self.tdp):
          with torch.cuda.amp.autocast(enabled=self._use_amp):
            metrics.update(self.update_tdp(start, goal))
        reward_fn = lambda seq: -1*self.tdp(torch.cat([self.wm.heads["emb_decoder"](self.wm.rssm.get_feat(seq))["embed"].mean, seq['goal']], dim=-1)).mean
        mets = self._env_behavior2.update(self.wm, start, data['is_terminal'], reward_fn, goal)
        metrics.update({f"achiever_{k}": v for k, v in mets.items()})
      
      # update additional metrics for skilled exploration
      if self.use_skilled_explore:
        metrics.update({"arriving_ratio_to_landmark": self._env_behavior.num_arriving_at_landmark/(self._env_behavior.num_arriving_at_landmark + self._env_behavior.num_failing_to_arrive_at_landmark+1e-7)})
    else:
      raise ValueError(f"Finetuning mode is not supported yet")
    return state, metrics

  def init_from(self, other):
      # WM
      print(f"Copying the pretrained world model")
      utils.hard_update_params(other.wm.rssm, self.wm.rssm)
      utils.hard_update_params(other.wm.encoder, self.wm.encoder)
      utils.hard_update_params(other.wm.heads['decoder'], self.wm.heads['decoder'])

      # Landmark
      print(f"Copying the pretrained landmark modules")
      utils.hard_update_params(other._landmark_behavior.actor, self._landmark_behavior.actor)
      utils.hard_update_params(other.landmark_module, self.landmark_module)
      if getattr(self.landmark_module, 'emb', False):
        self.landmark_module.emb.weight.data.copy_(other.landmark_module.emb.weight.data)

  def reset_behavior(self, env, behavior_type="explorer", goal=None):
    if behavior_type == "explorer":
      self._env_behavior.reset(self.wm, env) if self.use_skilled_explore else self._env_behavior.reset()
    if behavior_type == "achiever":
      self._env_behavior2.reset(self.wm, env, goal) if self.use_skilled_achieve else self._env_behavior2.reset(goal)

  def get_landmark_curiosities(self):
    if hasattr(self._env_behavior, "landmark_curiosities"):
      return self._env_behavior.landmark_curiosities
    else:
      return None

  def get_skilled_explore_stats(self):
    if self.use_skilled_explore:
      selected_landmark = self._env_behavior.selected_landmark
      if selected_landmark is not None:
        # For reconstructing landmarks
        latent_landmark = selected_landmark @ self.landmark_module.emb.weight.T # (1, D)
        latent_landmark = latent_landmark.reshape(-1, 1, latent_landmark.shape[-1]) # (1, 1, D)
        deter = self.landmark_module.landmark_decoder(latent_landmark).mean # (1, 1, D)
        stats = self.wm.rssm._suff_stats_ensemble(deter)
        index = torch.randint(0, self.wm.rssm._ensemble, ())
        stats = {k: v[index] for k, v in stats.items()}
        dist = self.wm.rssm.get_dist(stats)
        stoch = dist.sample()
        prior = {'stoch': stoch, 'deter': deter, **stats}
        recon_landmark = self.wm.heads["decoder"](self.wm.rssm.get_feat(prior))["observation"].mean + 0.5 # [0,1]
        recon_landmark = recon_landmark.squeeze().detach().cpu().numpy() # (C, H, W)
        selected_landmark = selected_landmark.squeeze().detach().cpu().numpy()
      else:
        recon_landmark = None
      return {"selected_landmark": selected_landmark, "recon_landmark": recon_landmark, "act_on_actor": self._env_behavior.act_on_actor, "arrive_at_landmark": self._env_behavior.arrive_at_landmark}
    else:
      return {}

  def get_goal_embedding(self, goal):
    with torch.no_grad():
      goal = torch.Tensor(np.array([goal])).permute(0,3,1,2).to(self.device) / 255.0 - 0.5 # same to the preprocessing in wm
      embed = self.wm.encoder({"observation": goal})
    return embed
      
  def get_goal_from_replay_buffer(self, data):
    with torch.no_grad():
      goal, goal_position = self.wm.get_goal(data, on_real=True) # [1,D], [3]
    return goal, goal_position.detach().cpu().numpy()
    
  def report(self, data):
    report = {}
    data = self.wm.preprocess(data)
    for key in self.wm.heads['decoder'].cnn_keys:
      name = key.replace('/', '_')
      report[f'openl_{name}'] = self.wm.video_pred(data, key)
    return report

  def get_visitation_stats(self, env, pos_file_path):
    histogram_scale = self.cfg.histogram_scale
    metrics = {}
    # get visitation ratio
    min_x, max_x, min_z, max_z = env.min_x, env.max_x, env.min_z, env.max_z
    bins_x = histogram_scale * (max_x - min_x)
    bins_z = histogram_scale * (max_z - min_z)
    pos_traj, _, _ = np.histogram2d([min_x], [min_z], bins=[int(bins_x), int(bins_z)], range=[[min_x, max_x], [min_z, max_z]])
    pos_traj = pos_traj.T * 0.0
    with open(pos_file_path, "r") as f:
      for line in f.readlines():
        line_elements = line.split(",")
        x, z = line_elements[0], line_elements[1]
        x, z = float(x), float(z)
        _pos_traj, _, _ = np.histogram2d([x], [z], bins=[int(bins_x), int(bins_z)], range=[[min_x, max_x], [min_z, max_z]])
        pos_traj += _pos_traj.T
    metrics.update({"visitation_ratio": np.sum(pos_traj > 0) / np.prod(pos_traj.shape)})
    # get room visitation
    room_visitations = np.zeros(len(env.rooms))
    with open(pos_file_path, "r") as f:
      for line in f.readlines():
        line_elements = line.split(",")
        x, z = line_elements[0], line_elements[1]
        x, z = float(x), float(z)
        for room_id in range(len(env.rooms)):
          if x >= env.rooms[room_id][0] and x <= env.rooms[room_id][1] and z >= env.rooms[room_id][2] and z <= env.rooms[room_id][3]:
            metrics.update({f"room_{room_id}_visitation": 1})
            room_visitations[room_id] = 1
            break
    for room_id in range(len(env.rooms)):
      if room_visitations[room_id] == 0:
        metrics.update({f"room_{room_id}_visitation": 0})
    metrics.update({"visited_rooms": np.sum(room_visitations)})
    return metrics
    
  def report_trajectory_heatmap(self, env, pos_file_path):
    min_x, max_x, min_z, max_z = env.min_x, env.max_x, env.min_z, env.max_z
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    levels = np.logspace(-1, 5, 1000)
    with open(pos_file_path, "r") as f:
      lines = f.readlines()
    data = [line.strip().split(',') for line in lines]
    x_values = np.array([float(line[0]) for line in data])
    y_values = np.array([float(line[1]) for line in data])
    
    hist, x_edges, y_edges = np.histogram2d(x_values, y_values, bins=(45, 45), range=[[min_x, max_x], [min_z, max_z]])
    X, Y = np.meshgrid(x_edges[:-1], y_edges[:-1])
    contour = ax.contourf(X, Y, hist.T, cmap='nipy_spectral',interpolation="nearest", norm=LogNorm(), levels=levels)
    fig.colorbar(contour, ax=ax, label='Density')  # Use fig.colorbar with the specific axis 'ax'
    
    ax.set_title('trajectory heatmap')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    img_pil = Image.open(buf)
    img_np = np.array(img_pil).astype(np.float32) / 255.0 # [0, 1]
    plt.close(fig)
    buf.close()
    return {"trajectory_heatmap": torch.Tensor(img_np)}  

  def report_trajectories_with_landmarks(self, env, pos_file_path, current_trajectory=None):
    histogram_scale = self.cfg.histogram_scale
    min_x, max_x, min_z, max_z = env.min_x, env.max_x, env.min_z, env.max_z
    topdown_view = env.get_topdown_view()
    map_height, map_width, _ = topdown_view.shape
    def draw_topdown_view(ax, selected_landmarks=None, without_landmarks=False):
      x_coords, z_coords = [], []
      with open(pos_file_path, "r") as f:
        for line in f.readlines():
          line_elements = line.split(",")
          x, z = line_elements[0], line_elements[1]
          x, z = float(x), float(z)
          x_coords.append(int((x - min_x) / (max_x - min_x) * map_width))
          z_coords.append(int((z - min_z) / (max_z - min_z) * map_height))
      ax.imshow(topdown_view)
      ax.scatter(x_coords, z_coords, c="blue", s=3, alpha=0.1, label="Trajectory")
      if not without_landmarks:
        if selected_landmarks is None:
          vq_landmarks = self.landmark_module.emb.weight.T
        else:
          vq_landmarks = selected_landmarks @ self.landmark_module.emb.weight.T
        state_preds = self.landmark_module.landmark_decoder(vq_landmarks).mean
        landmark_pos_estimation = self.wm.heads["pos_decoder"](state_preds)["position"].mean.cpu().numpy()
        landmark_pos_x = np.clip(landmark_pos_estimation[:, 0], min_x, max_x)
        landmark_pos_z = np.clip(landmark_pos_estimation[:, 2], min_z, max_z)
        x_coords, z_coords = [], []
        for x, z in zip(landmark_pos_x, landmark_pos_z):
          x_coords.append(int((x - min_x) / (max_x - min_x) * map_width))
          z_coords.append(int((z - min_z) / (max_z - min_z) * map_height))
        ax.scatter(x_coords, z_coords, c="red", s=70, marker="*", label="Landmark Position")
        landmark_poses = [[x, z] for x, z in zip(landmark_pos_x, landmark_pos_z)]
      else:
        landmark_poses = None
      ax.xaxis.set_visible(False)
      ax.yaxis.set_visible(False)
      ax.set_title("Top-down View", fontsize=20)
      #ax.legend(loc="upper left")
      return landmark_poses
    

    def draw_heatmap_old(_fig, _ax):
      # histogram
      bins_x = histogram_scale * (max_x - min_x)
      bins_z = histogram_scale * (max_z - min_z)
      pos_traj, _, _ = np.histogram2d([min_x], [min_z], bins=[int(bins_x), int(bins_z)], range=[[min_x, max_x], [min_z, max_z]])
      pos_traj = pos_traj.T * 0.0
      with open(pos_file_path, "r") as f:
        for line in f.readlines():
          line_elements = line.split(",")
          x, z = line_elements[0], line_elements[1]
          x, z = float(x), float(z)
          _pos_traj, _, _ = np.histogram2d([x], [z], bins=[int(bins_x), int(bins_z)], range=[[min_x, max_x], [min_z, max_z]])
          pos_traj += _pos_traj.T
      cax = _ax.imshow(pos_traj, cmap="jet", interpolation="nearest", norm=LogNorm())# nipy_spectral, OrRd, jet
      plt.colorbar(cax, ax=_ax, label='Number of Visits', ticks=LogLocator())
      _ax.xaxis.set_visible(False)
      _ax.yaxis.set_visible(False)
      _ax.set_title("Trajectory Histogram", fontsize=20)
      buf = io.BytesIO()
      plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
      buf.seek(0)
      img_pil = Image.open(buf)
      img_np = np.array(img_pil).astype(np.float32) / 255.0 # [0, 1]
      img_np = (1 - img_np[:,:,3:]) * np.ones_like(img_np)[:,:,:3] + img_np[:,:,3:] * img_np[:,:,:3] # apply transparency
      plt.close(_fig)
      buf.close()
      return img_np 
    
    if current_trajectory is None:
      fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
      # top-down view
      ax = axes[0]
      landmark_poses = draw_topdown_view(ax)
      # histogram
      ax = axes[1]
      img_np = draw_heatmap_old(fig, ax)
      # landmark reconstruction and agent view on estimated landmark position
      latent_landmark = self.landmark_module.emb.weight.T # (N, D)
      latent_landmark = latent_landmark.reshape(-1, 1, latent_landmark.shape[-1]) # (N, 1, D)
      deter = self.landmark_module.landmark_decoder(latent_landmark).mean # (N, 1, D)
      stats = self.wm.rssm._suff_stats_ensemble(deter)
      index = torch.randint(0, self.wm.rssm._ensemble, ())
      stats = {k: v[index] for k, v in stats.items()}
      dist = self.wm.rssm.get_dist(stats)
      stoch = dist.sample()
      prior = {'stoch': stoch, 'deter': deter, **stats}
      recon_landmark = self.wm.heads["decoder"](self.wm.rssm.get_feat(prior))["observation"].mean + 0.5 # [0,1]
      recon_landmark = recon_landmark.squeeze().permute(0,2,3,1).detach().cpu().numpy() # (N, H, W, C)
      latent_landmarks = self.landmark_module.emb.weight.T
      state_preds = self.landmark_module.landmark_decoder(latent_landmarks).mean
      estimated_landmark_poses = self.wm.heads["pos_decoder"](state_preds)["position"].mean.cpu().numpy()
      view_from_landmark_pos = []
      for _landmark_pos in estimated_landmark_poses:
        view_from_landmark_pos.append(env.render_on_pos(_landmark_pos)/255.0)
      view_from_landmark_pos = np.array(view_from_landmark_pos) # (N, H, W, C)
      landmark_recon = np.concatenate([recon_landmark, view_from_landmark_pos], axis=1) # (N, 2*H, W, C)
      landmark_recon = np.concatenate([_landmark_recon for _landmark_recon in landmark_recon], axis=1) # (2*H, W*N, C)
      return {"trajectory_history_with_landmarks": torch.Tensor(img_np), "landmark_recon": torch.Tensor(landmark_recon)}
    else:
      # top-down view
      fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
      ax = axes
      x_coords, z_coords, selected_landmarks = [], [], []
      for _traj in current_trajectory:
        x, z = _traj["position"]
        x_coords.append(int((x - min_x) / (max_x - min_x) * map_width))
        z_coords.append(int((z - min_z) / (max_z - min_z) * map_height))
        if "selected_landmark" in _traj.keys() and _traj["selected_landmark"] is not None:
          found = False
          for _selected_landmark in selected_landmarks:
            if np.all(_selected_landmark == _traj["selected_landmark"]):
              found = True
          if not found:
            selected_landmarks.append(_traj["selected_landmark"])
      if len(selected_landmarks) > 0:
        selected_landmarks = torch.Tensor(np.array(selected_landmarks)).to(self.device)

      if len(selected_landmarks) > 0:
        landmark_poses = draw_topdown_view(ax, selected_landmarks=selected_landmarks)
      else:
        landmark_poses = draw_topdown_view(ax, without_landmarks=True) # no landmarks, return None
      ax.scatter(x_coords, z_coords, c="green", s=50, marker="8", label="Agent", alpha=0.9)
      buf = io.BytesIO()
      plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
      buf.seek(0)
      img_pil = Image.open(buf)
      img_np = np.array(img_pil).astype(np.float32) / 255.0 # [0, 1]
      img_np = (1 - img_np[:,:,3:]) * np.ones_like(img_np)[:,:,:3] + img_np[:,:,3:] * img_np[:,:,:3] # apply transparency
      plt.close(fig)
      buf.close()
      # video with agent view and landmark reconstruction
      video = []
      zero_padding = 10
      for _traj in current_trajectory:
        _topdown_view = np.copy(topdown_view)[::4, ::4] / 255.0 # downsample
        map_height, map_width, _ = _topdown_view.shape
        x = int((_traj["position"][0] - min_x) / (max_x - min_x) * map_width)
        z = int((_traj["position"][1] - min_z) / (max_z - min_z) * map_height)
        x_range = [x-3 if x-3 >= 0 else 0, x+3 if x+3 < map_width else map_width-1]
        z_range = [z-3 if z-3 >= 0 else 0, z+3 if z+3 < map_height else map_height-1]
        _topdown_view[z_range[0]:z_range[1], x_range[0]:x_range[1]] = [0, 1, 0] # green
        # agent view
        agent_view = np.zeros((_traj["observation"].shape[1]+zero_padding*2, _traj["observation"].shape[2]+zero_padding*2, 3), dtype=np.float32)
        if "act_on_actor" in _traj.keys() and _traj["act_on_actor"]:
          agent_view[zero_padding//2:-zero_padding//2, zero_padding//2:-zero_padding//2, :] = [0, 1, 0] if _traj["arrive_at_landmark"] else [1, 0, 0] # green if arrive at landmark, red if not
        agent_view[zero_padding:-zero_padding, zero_padding:-zero_padding, :] = _traj["observation"].transpose(1, 2, 0) / 255.0 # (C, H, W) -> (H, W, C)
        # landmark reconstruction
        if "recon_landmark" in _traj.keys() and _traj["recon_landmark"] is not None and not _traj["act_on_actor"]:
          landmark_recon = np.zeros((_traj["recon_landmark"].shape[1]+zero_padding*2, _traj["recon_landmark"].shape[2]+zero_padding*2, 3), dtype=np.float32)
          landmark_recon[zero_padding:-zero_padding, zero_padding:-zero_padding, :] = _traj["recon_landmark"].transpose(1, 2, 0) # (C, H, W) -> (H, W, C)
        else:
          landmark_recon = None
        # concatenate
        additional_view = np.concatenate([agent_view, landmark_recon], axis=0) if landmark_recon is not None else agent_view
        if _topdown_view.shape[0] > additional_view.shape[0]:
          additional_view = np.concatenate([additional_view, np.zeros((_topdown_view.shape[0]-additional_view.shape[0], additional_view.shape[1], 3), dtype=np.float32)], axis=0)
        else:
          _topdown_view = np.concatenate([_topdown_view, np.zeros((additional_view.shape[0]-_topdown_view.shape[0], _topdown_view.shape[1], 3), dtype=np.float32)], axis=0)
        _frame = np.concatenate([_topdown_view, additional_view], axis=1)
        video.append(_frame.reshape(1, *_frame.shape)) # (1, H, W, C)
      video = np.concatenate(video, axis=0) # (T, H, W, C)
      return {"trajectories": torch.Tensor(img_np)}, {"trajectories_video": torch.Tensor(video).permute(0,3,1,2).unsqueeze(0)}
