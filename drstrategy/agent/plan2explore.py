# Adapted from: https://github.dev/mazpie/mastering-urlb

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from agent.drstrategy import DrStrategyAgent, stop_gradient
import agent.net_utils as common

class Disagreement(nn.Module):
  def __init__(self, obs_dim, action_dim, hidden_dim, n_models=5, pred_dim=None):
    super().__init__()
    if pred_dim is None: pred_dim = obs_dim
    self.ensemble = nn.ModuleList([
        nn.Sequential(nn.Linear(obs_dim + action_dim, hidden_dim),
                      nn.ReLU(), nn.Linear(hidden_dim, pred_dim))
        for _ in range(n_models)
    ])

  def forward(self, obs, action, next_obs):
    #import ipdb; ipdb.set_trace()
    assert obs.shape[0] == next_obs.shape[0]
    assert obs.shape[0] == action.shape[0]

    errors = []
    for model in self.ensemble:
      next_obs_hat = model(torch.cat([obs, action], dim=-1))
      model_error = torch.norm(next_obs - next_obs_hat,
                                dim=-1,
                                p=2,
                                keepdim=True)
      errors.append(model_error)

    return torch.cat(errors, dim=1)

  def get_disagreement(self, obs, action):
    assert obs.shape[0] == action.shape[0]

    preds = []
    for model in self.ensemble:
      next_obs_hat = model(torch.cat([obs, action], dim=-1))
      preds.append(next_obs_hat)
    preds = torch.stack(preds, dim=0)
    return torch.var(preds, dim=0).mean(dim=-1)


class Plan2ExploreDrStrategy(DrStrategyAgent):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    print("-> Plan2Explore Agent")
    in_dim = self.wm.inp_size
    pred_dim = self.wm.embed_dim
    self.hidden_dim = pred_dim
    self.reward_free = True

    self.disagreement = Disagreement(in_dim, self.act_dim,
                                      self.hidden_dim, pred_dim=pred_dim).to(self.device)

    # optimizers
    self.disagreement_opt = common.Optimizer('disagreement', self.disagreement.parameters(), **self.cfg.model_opt, use_amp=self._use_amp)
    
    self.lbs = None
    
    self.disagreement.train()
    self.requires_grad_(requires_grad=False)

  def update_disagreement(self, obs, action, next_obs, step):
    metrics = dict()

    error = self.disagreement(obs, action, next_obs)

    loss = error.mean()

    metrics.update(self.disagreement_opt(loss, self.disagreement.parameters()))

    metrics['disagreement_loss'] = loss.item()

    return metrics

  def compute_intr_reward(self, seq):
    obs, action = seq['feat'][:-1], stop_gradient(seq['action'][1:])
    intr_rew = torch.zeros(list(seq['action'].shape[:-1]) + [1], device=self.device)
    if len(action.shape) > 2:
      B, T, _ = action.shape
      obs = obs.reshape(B*T, -1)
      action = action.reshape(B*T, -1)
      reward = self.disagreement.get_disagreement(obs, action).reshape(B, T, 1)
    else:
      reward = self.disagreement.get_disagreement(obs, action).unsqueeze(-1)
    intr_rew[1:] = reward
    return intr_rew

  def update(self, data, step):
    B, T, _ = data['action'].shape
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

      # ------------ Modified ------------
      T = T-1
      inp = stop_gradient(outputs['feat'][:, :-1]).reshape(B*T, -1)
      action = data['action'][:, 1:].reshape(B*T, -1)
      out = stop_gradient(outputs['embed'][:,1:]).reshape(B*T,-1)
      with common.RequiresGrad(self.disagreement):
        with torch.cuda.amp.autocast(enabled=self._use_amp):
          metrics.update(self.update_disagreement(inp, action, out, step))  
      reward_fn = self.compute_intr_reward
      # ----------------------------------
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
      #self.reward_smoothing =  self.reward_smoothing and (not (data['reward'] > 1e-4).any())
      #self._env_behavior.reward_smoothing = self.reward_smoothing

      ## Train task AC
      #if not self.reward_smoothing:
      #  reward_fn = lambda seq: self.wm.heads['reward'](seq['feat']).mean
      #  metrics.update(self._env_behavior.update(self.wm, start, data['is_terminal'], reward_fn))
    return state, metrics
