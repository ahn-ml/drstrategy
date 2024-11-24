import torch.nn as nn
import torch
import torch.distributions as D
import torch.nn.functional as F
import numpy as np

import utils
import agent.net_utils as common
from agent.mb_utils import stop_gradient
from collections import deque

class LandmarkActorCritic(nn.Module):
  def __init__(self, config, act_spec, tfstep, landmark_dim, landmark_module, solved_meta=None, imagine_obs=False):
    super().__init__()
    self.cfg = config
    self.act_spec = act_spec
    self.tfstep = tfstep
    self._use_amp = (config.precision == 16)
    self.device = config.device
    self._use_deter = config.use_deter_for_landmark_executor
    self._landmark_module = landmark_module

    self.imagine_obs = imagine_obs
    self.solved_meta = solved_meta
    self.landmark_dim = landmark_dim
    inp_size = config.rssm.deter
    if config.rssm.discrete:
      inp_size += config.rssm.stoch * config.rssm.discrete
    else:
      inp_size += config.rssm.stoch

    if self._use_deter:
      inp_size += config.rssm.deter
    else:
      inp_size += landmark_dim
    self.actor = common.MLP(inp_size, act_spec.shape[0], **self.cfg.actor)
    self.critic = common.MLP(inp_size, (1,), **self.cfg.critic)
    if self.cfg.slow_target:
      self._target_critic = common.MLP(inp_size, (1,), **self.cfg.critic)
      self._updates = 0
    else:
      self._target_critic = self.critic
    self.actor_opt = common.Optimizer('landmark_actor', self.actor.parameters(), **self.cfg.actor_opt, use_amp=self._use_amp)
    self.critic_opt = common.Optimizer('landmark_critic', self.critic.parameters(), **self.cfg.critic_opt, use_amp=self._use_amp)
    self.rewnorm = common.StreamNorm(**self.cfg.landmark_reward_norm, device=self.device)

  def get_landmark_inp(self, landmark):
    if self._use_deter:
      latent_landmark = landmark @ self._landmark_module.emb.weight.T # (1, D)
      return self._landmark_module.landmark_decoder(latent_landmark).mean # (1, D)
    else:
      return landmark
    
  def _get_feat_ac(self, seq):
    return torch.cat([seq['feat'], self.get_landmark_inp(seq['landmark'])], dim=-1)
  
  def reset(self):
    return
  
  def act(self, feat, target_landmark, eval_mode=False, return_dist=False):
    if return_dist:
      return self.actor(torch.cat([feat, self.get_landmark_inp(target_landmark)], dim=-1))
    return self.actor(torch.cat([feat, self.get_landmark_inp(target_landmark)], dim=-1)).mean if eval_mode else self.actor(torch.cat([feat, self.get_landmark_inp(target_landmark)], dim=-1)).sample()
  
  def update(self, world_model, start, is_terminal, reward_fn):
    metrics = {}
    hor = self.cfg.imag_horizon
    with common.RequiresGrad(self.actor):
      with torch.cuda.amp.autocast(enabled=self._use_amp):
        B,T , _ = start['deter'].shape
        if self.solved_meta is not None:
          img_landmark = torch.from_numpy(self.solved_meta['landmark']).repeat(B*T, 1).to(self.device)
        else:
          img_landmark = F.one_hot(torch.randint(0, self.landmark_dim, size=(B*T,), device=self.device), num_classes=self.landmark_dim).float()

        seq = world_model.imagine(self.actor, start, is_terminal, hor, landmark_cond=img_landmark, get_landmark_inp=self.get_landmark_inp)
        if self.imagine_obs:
          with torch.no_grad():
            seq['observation'] = world_model.heads['decoder'](seq['feat'].detach())['observation'].mean
        reward = reward_fn(seq)
        seq['reward'], mets1 = self.rewnorm(reward)
        mets1 = {f'landmark_reward_{k}': v for k, v in mets1.items()}
        target, mets2 = self.target(seq)
        actor_loss, mets3 = self.actor_loss(seq, target)
      metrics.update(self.actor_opt(actor_loss, self.actor.parameters()))
    with common.RequiresGrad(self.critic):
      with torch.cuda.amp.autocast(enabled=self._use_amp):
        seq = {k: stop_gradient(v) for k,v in seq.items()}
        critic_loss, mets4 = self.critic_loss(seq, target)
      metrics.update(self.critic_opt(critic_loss, self.critic.parameters()))
    metrics.update(**mets1, **mets2, **mets3, **mets4)
    self.update_slow_target()  # Variables exist after first forward pass.
    return metrics

  def actor_loss(self, seq, target): #, step):
    self.tfstep = 0
    metrics = {}
    policy = self.actor(stop_gradient(self._get_feat_ac(seq)[:-2]))
    if self.cfg.actor_grad == 'dynamics':
      objective = target[1:]
    elif self.cfg.actor_grad == 'reinforce':
      baseline = self._target_critic(self._get_feat_ac(seq)[:-2]).mean
      advantage = stop_gradient(target[1:] - baseline)
      objective = policy.log_prob(stop_gradient(seq['action'][1:-1]))[:,:,None] * advantage
    elif self.cfg.actor_grad == 'both':
      baseline = self._target_critic(self._get_feat_ac(seq)[:-2]).mean
      advantage = stop_gradient(target[1:] - baseline)
      objective = policy.log_prob(stop_gradient(seq['action'][1:-1]))[:,:,None] * advantage
      mix = utils.schedule(self.cfg.actor_grad_mix, self.tfstep)
      objective = mix * target[1:] + (1 - mix) * objective
      metrics['landmark_actor_grad_mix'] = mix
    else:
      raise NotImplementedError(self.cfg.actor_grad)
    ent = policy.entropy()[:,:,None]
    ent_scale = utils.schedule(self.cfg.landmark_actor_ent, self.tfstep)
    objective += ent_scale * ent
    weight = stop_gradient(seq['weight'])
    actor_loss = -(weight[:-2] * objective).mean()
    metrics['landmark_actor_ent'] = ent.mean()
    metrics['landmark_actor_ent_scale'] = ent_scale
    return actor_loss, metrics

  def critic_loss(self, seq, target):
    dist = self.critic(self._get_feat_ac(seq)[:-1])
    target = stop_gradient(target)
    weight = stop_gradient(seq['weight'])
    critic_loss = -(dist.log_prob(target)[:,:,None] * weight[:-1]).mean()
    metrics = {'landmark_critic': dist.mean.mean() }
    return critic_loss, metrics

  def target(self, seq):
    reward = seq['reward']
    disc = seq['discount']
    value = self._target_critic(self._get_feat_ac(seq)).mean
    # Skipping last time step because it is used for bootstrapping.
    target = common.lambda_return(
        reward[:-1], value[:-1], disc[:-1],
        bootstrap=value[-1],
        lambda_=self.cfg.discount_lambda,
        axis=0)
    metrics = {}
    metrics['landmark_critic_slow'] = value.mean()
    metrics['landmark_critic_target'] = target.mean()
    return target, metrics

  def update_slow_target(self):
    if self.cfg.slow_target:
      if self._updates % self.cfg.slow_target_update == 0:
        mix = 1.0 if self._updates == 0 else float(
            self.cfg.slow_target_fraction)
        for s, d in zip(self.critic.parameters(), self._target_critic.parameters()):
          d.data = mix * s.data + (1 - mix) * d.data
      self._updates += 1

class SingleMetaCtrlAC(nn.Module):
  def __init__(self, config, agent_type, act_spec, landmark_dim, tfstep, landmark_executor, landmark_module, goal_conditioned=False, goal_dim=0):
    super().__init__()
    self.cfg = config
    self._agent_type = agent_type
    if self._agent_type == "explorer":
      config_skilled_agent = config.skilled_explore
    elif self._agent_type == "achiever":
      config_skilled_agent = config.skilled_achieve
    else:
      raise NotImplementedError(f"agent_type {self._agent_type} is not implemented")
    self._max_step_for_landmark_executor = config_skilled_agent.max_step_for_landmark_executor
    self._threshold_for_landmark_executor = config_skilled_agent.threshold_for_landmark_executor
    self._go_landmark = config_skilled_agent.go_landmark
    self.act_spec = act_spec
    self.landmark_dim = landmark_dim
    self.tfstep = tfstep
    self.landmark_executor = landmark_executor
    self.landmark_module = landmark_module
    self._goal_conditioned = goal_conditioned
    self._use_amp = (config.precision == 16)
    self.device = config.device

    inp_size = config.rssm.deter
    if config.rssm.discrete:
      inp_size += config.rssm.stoch * config.rssm.discrete
    else:
      inp_size += config.rssm.stoch
    if goal_conditioned:
      inp_size += goal_dim
    self.actor = common.MLP(inp_size, act_spec.shape[0], **self.cfg.actor)
    self.critic = common.MLP(inp_size, (1,), **self.cfg.critic)
    if self.cfg.slow_target:
      self._target_critic = common.MLP(inp_size, (1,), **self.cfg.critic)
      self._updates = 0
    else:
      self._target_critic = self.critic
    self.actor_opt = common.Optimizer('actor', self.actor.parameters(), **self.cfg.actor_opt, use_amp=self._use_amp)
    self.critic_opt = common.Optimizer('critic', self.critic.parameters(), **self.cfg.critic_opt, use_amp=self._use_amp)
    self.rewnorm = common.StreamNorm(**self.cfg.reward_norm, device=self.device)

    # Meta-Controller
    self.selected_landmark = None
    self.act_cnt = 0
    self.act_on_actor = False
    if self._agent_type == "explorer":
      self.landmark_curiosities = np.array([1/landmark_dim] * landmark_dim) # initialized with large values
    self.stats_max_widnowsize = 100
    self.stats_arriving_at_landmark = deque(maxlen=self.stats_max_widnowsize)

  @property
  def num_arriving_at_landmark(self):
    return sum(self.stats_arriving_at_landmark)
  @property
  def num_failing_to_arrive_at_landmark(self):
    return self.stats_max_widnowsize - sum(self.stats_arriving_at_landmark)

  def _get_feat_ac(self, seq):
    if 'goal' in seq.keys():
      return torch.cat([seq['feat'], seq['goal']], dim=-1)
    else:
      return seq['feat']
    
  def _collect_top_k_nearest_landmarks(self, world_model, goal, top_k_landmarks=1):
    decoded_landmarks = self.landmark_module.landmark_decoder(self.landmark_module.emb.weight.T).mean
    deter_goal = world_model.get_deter_from_embed(goal)
    dist = -1 * torch.norm(decoded_landmarks[None,] - deter_goal[:,None], dim=-1)
    _, argmin = torch.topk(dist, top_k_landmarks)
    return argmin
  
  def _select_landmark(self, world_model, goal):
    if self._agent_type == "explorer":
      selected_landmark = np.eye(self.landmark_dim)[np.random.choice(self.landmark_dim, p=self.landmark_curiosities)]
      return torch.from_numpy(selected_landmark).unsqueeze(0).float().to(self.device)
    elif self._agent_type == "achiever":
      decoded_landmarks = self.landmark_module.landmark_decoder(self.landmark_module.emb.weight.T).mean
      deter_goal = world_model.get_deter_from_embed(goal)
      dist = -1 * torch.norm(decoded_landmarks[None,] - deter_goal[:,None], dim=-1)
      _, argmin = torch.topk(dist, 1)
      return torch.eye(self.landmark_dim).float().to(self.device)[argmin].squeeze(1)
      
  def reset(self, world_model, env, goal=None):
    assert (goal is None) ^ self._goal_conditioned, "Goal conditioned agent must have goal and non-goal conditioned agent must not have goal"
    self.goal = goal
    # select landmark at the beginning of episode
    self.selected_landmark = self._select_landmark(world_model, goal)
    latent_landmark = self.selected_landmark @ self.landmark_module.emb.weight.T # (1, D)
    latent_landmark = latent_landmark.reshape(-1, 1, latent_landmark.shape[-1]) # (1, 1, D)
    self.decoded_landmark = self.landmark_module.landmark_decoder(latent_landmark).mean # (1, 1, D) # for checking if the agent arrives at landmark
    # reset
    if self._go_landmark:
      landmark_pos_decoder = world_model.heads["pos_decoder"]
      landmark_pos = landmark_pos_decoder(self.decoded_landmark)["position"].mean.squeeze().cpu().numpy()
      env.warp_to(landmark_pos)
      self.act_cnt = 0
      self.act_on_actor = True # start from selected landmark with actor
      self.arrive_at_landmark = False # it is not used in this case
      self.landmark_executor.reset()
    else:
      self.act_cnt = 0
      self.act_on_actor = False
      self.arrive_at_landmark = False
      self.landmark_executor.reset()

  def act(self, feat, deter, eval_mode=False):
    # act
    if self.act_on_actor: # use actor
      inp = torch.cat([feat, self.goal], dim=-1) if self._goal_conditioned else feat
      act = self.actor(inp).mean if eval_mode else self.actor(inp).sample()
    else: # use landmark executor
      act = self.landmark_executor.act(feat, self.selected_landmark, eval_mode=eval_mode)
    # check if the agent arrives at landmark  
    if not self.act_on_actor:
      if self.act_cnt > self._max_step_for_landmark_executor: # if the agent cannot reach the landmark in max step, then use actor
        self.act_on_actor = True
        self.arrive_at_landmark = False
        self.stats_arriving_at_landmark.append(0)
      if F.mse_loss(deter.squeeze(), self.decoded_landmark.squeeze()) < self._threshold_for_landmark_executor: # if the agent is near to landmark, then use actor
        self.act_on_actor = True
        self.arrive_at_landmark = True
        self.stats_arriving_at_landmark.append(1)
    self.act_cnt += 1
    return act

  def update(self, world_model, start, is_terminal, reward_fn, goal=None):
    metrics = {}
    hor = self.cfg.imag_horizon
    with common.RequiresGrad(self.actor):
      with torch.cuda.amp.autocast(enabled=self._use_amp):
        if self._agent_type == "explorer": # exploring from every landmark
          # landmark selection
          latent_landmarks = self.landmark_module.emb.weight.T
          latent_landmarks = latent_landmarks.reshape(-1, 1, latent_landmarks.shape[-1])
        elif self._agent_type == "achiever": # exploring from the top-k nearest landmarks to goal
          latent_landmarks = self._select_landmark(world_model, goal)
          latent_landmarks = latent_landmarks @ self.landmark_module.emb.weight.T # (1, D)
          latent_landmarks = latent_landmarks.reshape(-1, 1, latent_landmarks.shape[-1]) # (1, 1, D)
        deter = self.landmark_module.landmark_decoder(latent_landmarks).mean
        stats = world_model.rssm._suff_stats_ensemble(deter)
        index = torch.randint(0, world_model.rssm._ensemble, ())
        stats = {k: v[index] for k, v in stats.items()}
        dist = world_model.rssm.get_dist(stats)
        stoch = dist.sample()
        start_on_landmark = {"deter": deter, "stoch": stoch, **stats}
        sos_is_terminal = torch.zeros([self.landmark_dim, 1, 1]).to(self.device)
        start = {k: v.reshape([-1,1]+list(v.shape[2:])) for k, v in start.items()}
        is_terminal = is_terminal.reshape(-1,1,1)
        shape = start['deter'].shape
        start_on_landmark = {k: torch.cat([v, start[k]], dim=0) for k, v in start_on_landmark.items()} # to explore on random states
        sos_is_terminal = torch.cat([sos_is_terminal, is_terminal], dim=0)
        # imagine
        seq = world_model.imagine(self.actor, start_on_landmark, sos_is_terminal, hor, goal_cond=goal)
        reward = reward_fn(seq)
        seq['reward'], mets1 = self.rewnorm(reward)
        mets1 = {f'reward_{k}': v for k, v in mets1.items()}
        target, mets2 = self.target(seq)
        target = target #target[:, -shape[0]:] # target
        seq = seq #{k: v[:, -shape[0]:] for k, v in seq.items()} #seq
        actor_loss, mets3 = self.actor_loss(seq, target)
        metrics.update(self.actor_opt(actor_loss, self.actor.parameters()))
        if self._agent_type == "explorer": # update curiosity
          for s_idx in range(self.landmark_dim):
            self.landmark_curiosities[s_idx] = self.landmark_curiosities[s_idx] * 0.0 + 1.0 * target[:, s_idx].mean().cpu().detach().numpy()
          self.landmark_curiosities = self.landmark_curiosities - np.min(self.landmark_curiosities) # to avoid negative values
          self.landmark_curiosities = self.landmark_curiosities / np.sum(self.landmark_curiosities) # to make sum to 1
    with common.RequiresGrad(self.critic):
      with torch.cuda.amp.autocast(enabled=self._use_amp):
        seq = {k: stop_gradient(v) for k,v in seq.items()}
        critic_loss, mets4 = self.critic_loss(seq, target)
      metrics.update(self.critic_opt(critic_loss, self.critic.parameters()))
    metrics.update(**mets1, **mets2, **mets3, **mets4)
    self.update_slow_target()  # Variables exist after first forward pass.
    return metrics

  def actor_loss(self, seq, target): #, step):
    self.tfstep = 0
    metrics = {}
    policy = self.actor(stop_gradient(self._get_feat_ac(seq)[:-2]))
    if self.cfg.actor_grad == 'dynamics':
      objective = target[1:]
    elif self.cfg.actor_grad == 'reinforce':
      baseline = self._target_critic(self._get_feat_ac(seq)[:-2]).mean
      advantage = stop_gradient(target[1:] - baseline)
      objective = policy.log_prob(stop_gradient(seq['action'][1:-1]))[:,:,None] * advantage
    elif self.cfg.actor_grad == 'both':
      baseline = self._target_critic(self._get_feat_ac(seq)[:-2]).mean
      advantage = stop_gradient(target[1:] - baseline)
      objective = policy.log_prob(stop_gradient(seq['action'][1:-1]))[:,:,None] * advantage
      mix = utils.schedule(self.cfg.actor_grad_mix, self.tfstep)
      objective = mix * target[1:] + (1 - mix) * objective
      metrics['actor_grad_mix'] = mix
    else:
      raise NotImplementedError(self.cfg.actor_grad)
    ent = policy.entropy()[:,:,None]
    ent_scale = utils.schedule(self.cfg.actor_ent, self.tfstep)
    objective += ent_scale * ent
    weight = stop_gradient(seq['weight'])
    actor_loss = -(weight[:-2] * objective).mean()
    metrics['actor_ent'] = ent.mean()
    metrics['actor_ent_scale'] = ent_scale
    return actor_loss, metrics

  def critic_loss(self, seq, target):
    dist = self.critic(self._get_feat_ac(seq)[:-1])
    target = stop_gradient(target)
    weight = stop_gradient(seq['weight'])
    critic_loss = -(dist.log_prob(target)[:,:,None] * weight[:-1]).mean()
    metrics = {'critic': dist.mean.mean() }
    return critic_loss, metrics

  def target(self, seq):
    reward = seq['reward']
    disc = seq['discount']
    value = self._target_critic(self._get_feat_ac(seq)).mean
    # Skipping last time step because it is used for bootstrapping.
    target = common.lambda_return(
        reward[:-1], value[:-1], disc[:-1],
        bootstrap=value[-1],
        lambda_=self.cfg.discount_lambda,
        axis=0)
    metrics = {}
    metrics['critic_slow'] = value.mean()
    metrics['critic_target'] = target.mean()
    return target, metrics

  def update_slow_target(self):
    if self.cfg.slow_target:
      if self._updates % self.cfg.slow_target_update == 0:
        mix = 1.0 if self._updates == 0 else float(
            self.cfg.slow_target_fraction)
        for s, d in zip(self.critic.parameters(), self._target_critic.parameters()):
          d.data = mix * s.data + (1 - mix) * d.data
      self._updates += 1
      
class LandmarkToLandmarkActorCritic(nn.Module):
  def __init__(self, config, act_spec, tfstep, landmark_dim, landmark_module, solved_meta=None, imagine_obs=False, statetostate=False):
    super().__init__()
    self.cfg = config
    self.act_spec = act_spec
    self.tfstep = tfstep
    self._use_amp = config.precision == 16
    self.device = config.device
    self._use_deter = config.use_deter_for_landmark_executor
    self.landmark_module = landmark_module

    self.imagine_obs = imagine_obs
    self.solved_meta = solved_meta
    self.landmark_dim = landmark_dim
    inp_size = config.rssm.deter
    if config.rssm.discrete:
      inp_size += config.rssm.stoch * config.rssm.discrete
    else:
      inp_size += config.rssm.stoch

    if not statetostate: # if it is False 
      # \pi(a|s,z)     
      if self._use_deter:
        inp_size += config.rssm.deter
      else:
        inp_size += landmark_dim

    else:
      inp_size += inp_size
      # inp_size += landmark_dim
    self.actor = common.MLP(inp_size, act_spec.shape[0], **self.cfg.actor)
    self.critic = common.MLP(inp_size, (1,), **self.cfg.critic)
    if self.cfg.slow_target:
      self._target_critic = common.MLP(inp_size, (1,), **self.cfg.critic)
      self._updates = 0
    else:
      self._target_critic = self.critic
        
    self.actor_opt = common.Optimizer( "landmark_actor", self.actor.parameters(), **self.cfg.actor_opt, use_amp=self._use_amp)
    self.critic_opt = common.Optimizer( "landmark_critic", self.critic.parameters(), **self.cfg.critic_opt, use_amp=self._use_amp)
    self.rewnorm = common.StreamNorm(**self.cfg.landmark_reward_norm, device=self.device)
    
  def update_finalizer(self, world_model, landmark_actor, start, is_terminal, reward_fn):
    metrics = {}
    hor = 20
    with common.RequiresGrad(self.actor):
      with torch.cuda.amp.autocast(enabled=self._use_amp):
        B, T, _ = start["deter"].shape  # B, T = 50, 50 torch.Size([50, 50, 200])
        z_e = self.landmark_module.landmark_encoder(start["deter"]).mean  # torch.Size([50, 50, 16])
        z_e = z_e.view(B * T, -1)
        emb, landmark_chosen = self.landmark_module.emb(z_e, training=False)  # torch.Size([2500, 16]), torch.Size([2500])
        start_landmark = torch.eye(self.landmark_dim)
        start_landmark = start_landmark[landmark_chosen]
        start_landmark = start_landmark.to(self.device).float()
        
        vq_landmark = start_landmark @ self.landmark_module.emb.weight.T 
        x = deter = self.landmark_module.landmark_decoder(vq_landmark).mean # x: [2500, 200]
        stats = world_model.rssm._suff_stats_ensemble(x)
        index = torch.randint(0, world_model.rssm._ensemble, ()) 
        stats = {k: v[index] for k, v in stats.items()}
        dist = world_model.rssm.get_dist(stats)
        stoch = dist.sample()
        landmark_to_state = {'stoch': stoch, 'deter': deter, **stats} # torch.Size([2500, 32, 32])   start: torch.Size([50, 50, 32, 32]) 
        landmark_to_state = {k: v.reshape([B, T] + list(v.shape[1:])) for k, v in landmark_to_state.items()} # torch.Size([50, 50, 32, 32]) 
        
        start_cond = {k: torch.cat((v[:, hor:, ...], v[:, T-hor*2:T-hor, ...]), dim=1) for k, v in start.items()} # torch.Size([50, 50, 32, 32]) 
        
        seq = world_model.imagine(self.actor, landmark_to_state, is_terminal, self.cfg.imag_horizon, landmark_cond=start_landmark, state_cond=start_cond,)
        
        if self.imagine_obs:
          with torch.no_grad():
            seq["observation"] = world_model.heads["decoder"](seq["feat"].detach())["observation"].mean
        reward = reward_fn(seq)
        seq["reward"], mets1 = self.rewnorm(reward)
        mets1 = {f"landmark_reward_{k}": v for k, v in mets1.items()}
        target, mets2 = self.target(seq)
        actor_loss, mets3 = self.actor_loss(seq, target)
      metrics.update(self.actor_opt(actor_loss, self.actor.parameters()))
    with common.RequiresGrad(self.critic):
      with torch.cuda.amp.autocast(enabled=self._use_amp):
        seq = {k: stop_gradient(v) for k, v in seq.items()}
        critic_loss, mets4 = self.critic_loss(seq, target)
      metrics.update(self.critic_opt(critic_loss, self.critic.parameters()))
    metrics.update(**mets1, **mets2, **mets3, **mets4)
    self.update_slow_target()  # Variables exist after first forward pass.
    return metrics
  
  def update(self, world_model, start, is_terminal, reward_fn):
    # Just for consistency
    metrics = {}
    hor = self.cfg.imag_horizon #15
    with common.RequiresGrad(self.actor):
      with torch.cuda.amp.autocast(enabled=self._use_amp):
        B, T, _ = start["deter"].shape  # B, T
        z_e = self.landmark_module.landmark_encoder(start["deter"]).mean  # B, T, 16
        z_e = z_e.view(B * T, -1) # B*T, 16
        emb, landmark_chosen = self.landmark_module.emb(z_e, training=False)  # B*T, 16, B*T
        landmark_chosen = landmark_chosen.view(B, T)  # B,T
        unique_chosen_landmarks = torch.unique(landmark_chosen)
        mets_landmark = {'unique_landmark_num_per_batch': unique_chosen_landmarks.size(0)} # get the num of unique landmarks
        # From: ---------->---|
        # to  : ------->---|-->
        # _landmark_chosen = torch.cat((landmark_chosen[:, hor:], landmark_chosen[:, T-2*hor:T-hor]), dim=1)  # B, hor: (From the hor till the end) & B, :hor (from begining till hor)
        # _landmark_chosen[_landmark_chosen == landmark_chosen] = torch.randint(self.cfg.landmark_dim, ((_landmark_chosen == landmark_chosen).sum(), ), device=self.device)
        # That gives better results
        _landmark_chosen = torch.cat((landmark_chosen[:, hor:], landmark_chosen[:, :hor]), dim=1)  # B, hor: (From the hor till the end) & B, :hor (from begining till hor)
        # Note: it is a problematic as it chooses from landmarks from the unique landmarks over the batch
        _landmark_chosen[_landmark_chosen == landmark_chosen] = unique_chosen_landmarks[torch.randint(unique_chosen_landmarks.size(0), (1,)).item()]
        _landmark_chosen = _landmark_chosen.view(B * T)
        img_landmark = torch.eye(self.landmark_dim)
        img_landmark = img_landmark[_landmark_chosen]
        img_landmark = img_landmark.to(self.device).float()
        seq = world_model.imagine(self.actor, start, is_terminal, self.cfg.imag_horizon, landmark_cond=img_landmark, get_landmark_inp=self.get_landmark_inp)
        if self.imagine_obs:
          with torch.no_grad():
            seq["observation"] = world_model.heads["decoder"](seq["feat"].detach())["observation"].mean
        reward = reward_fn(seq)
        seq["reward"], mets1 = self.rewnorm(reward)
        mets1 = {f"landmark_reward_{k}": v for k, v in mets1.items()}
        target, mets2 = self.target(seq)
        actor_loss, mets3 = self.actor_loss(seq, target)
      metrics.update(self.actor_opt(actor_loss, self.actor.parameters()))
    with common.RequiresGrad(self.critic):
      with torch.cuda.amp.autocast(enabled=self._use_amp):
        seq = {k: stop_gradient(v) for k, v in seq.items()}
        critic_loss, mets4 = self.critic_loss(seq, target)
      metrics.update(self.critic_opt(critic_loss, self.critic.parameters()))
    metrics.update(**mets1, **mets2, **mets3, **mets4, **mets_landmark)
    self.update_slow_target()  # Variables exist after first forward pass.
    return metrics

  def actor_loss(self, seq, target):
    self.tfstep = 0
    metrics = {}
    policy = self.actor(stop_gradient(self._get_feat_ac(seq)[:-2]))
    if self.cfg.actor_grad == "dynamics":
      objective = target[1:]
    elif self.cfg.actor_grad == "reinforce":
      baseline = self._target_critic(self._get_feat_ac(seq)[:-2]).mean
      advantage = stop_gradient(target[1:] - baseline)
      objective = policy.log_prob(stop_gradient(seq["action"][1:-1]))[:, :, None] * advantage
    elif self.cfg.actor_grad == "both":
      baseline = self._target_critic(self._get_feat_ac(seq)[:-2]).mean
      advantage = stop_gradient(target[1:] - baseline)
      objective = policy.log_prob(stop_gradient(seq["action"][1:-1]))[:, :, None] * advantage
      mix = utils.schedule(self.cfg.actor_grad_mix, self.tfstep)
      objective = mix * target[1:] + (1 - mix) * objective
      metrics["landmark_actor_grad_mix"] = mix
    else:
      raise NotImplementedError(self.cfg.actor_grad)
    ent = policy.entropy()[:, :, None]
    ent_scale = utils.schedule(self.cfg.landmark_actor_ent, self.tfstep)
    objective += ent_scale * ent
    weight = stop_gradient(seq["weight"])
    actor_loss = -(weight[:-2] * objective).mean()
    metrics["landmark_actor_ent"] = ent.mean()
    metrics["landmark_actor_ent_scale"] = ent_scale
    return actor_loss, metrics

  def critic_loss(self, seq, target):
    dist = self.critic(self._get_feat_ac(seq)[:-1])
    target = stop_gradient(target)
    weight = stop_gradient(seq["weight"])
    critic_loss = -(dist.log_prob(target)[:, :, None] * weight[:-1]).mean()
    metrics = {"landmark_critic": dist.mean.mean()}
    return critic_loss, metrics

  def target(self, seq):
    reward = seq["reward"]
    disc = seq["discount"]
    value = self._target_critic(self._get_feat_ac(seq)).mean
    # Skipping last time step because it is used for bootstrapping.
    target = common.lambda_return(
        reward[:-1], value[:-1], disc[:-1],
        bootstrap=value[-1],
        lambda_=self.cfg.discount_lambda,
        axis=0)
    metrics = {}
    metrics["landmark_critic_slow"] = value.mean()
    metrics["landmark_critic_target"] = target.mean()
    return target, metrics

  def update_slow_target(self):
    if self.cfg.slow_target:
      if self._updates % self.cfg.slow_target_update == 0:
        mix = 1.0 if self._updates == 0 else float(self.cfg.slow_target_fraction)
        for s, d in zip(self.critic.parameters(), self._target_critic.parameters()):
          d.data = mix * s.data + (1 - mix) * d.data
      self._updates += 1

  def get_landmark_inp(self, landmark):
    if self._use_deter:
      latent_landmark = landmark @ self.landmark_module.emb.weight.T # (1, D)
      return self.landmark_module.landmark_decoder(latent_landmark).mean # (1, D)
    else:
      return landmark

  def act(self, feat, target_landmark, eval_mode=False, return_dist=False, deter=None):
    # target_landmark is one-hot encoding
    if return_dist:
      return self.actor(torch.cat([feat, self.get_landmark_inp(target_landmark)], dim=-1))
    return self.actor(torch.cat([feat, self.get_landmark_inp(target_landmark)], dim=-1)).mean if eval_mode else self.actor(torch.cat([feat, self.get_landmark_inp(target_landmark)], dim=-1)).sample()

  def actor_act(self, feat, target_landmark, eval_mode=False, return_dist=False, deter=None):
    if return_dist:
      return self.actor(torch.cat([feat, self.get_landmark_inp(target_landmark)], dim=-1))
    return self.actor(torch.cat([feat, self.get_landmark_inp(target_landmark)], dim=-1)).mean if eval_mode else self.actor(torch.cat([feat, self.get_landmark_inp(target_landmark)], dim=-1)).sample()
  
  def reset(self):
    return
      
  def _get_feat_ac(self, seq):
    if 'state_cond' in seq.keys():
      return torch.cat([seq["feat"], seq["state_cond"]], dim=-1)
    return torch.cat([seq['feat'], self.get_landmark_inp(seq['landmark'])], dim=-1)
  
