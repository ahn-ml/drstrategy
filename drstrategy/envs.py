from collections import OrderedDict, deque
from typing import Any, NamedTuple
import os

import dm_env
import numpy as np
from dm_control import suite
from dm_control.suite.wrappers import action_scale, pixels
from dm_env import StepType, specs

import custom_dmc_tasks as cdmc
from custom_dmc_tasks.yoga_utils import shortest_angle, quat2euler

import gym
import pickle
from copy import deepcopy

from itertools import combinations

class ImageToPyTorch(gym.ObservationWrapper):
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0,
                                                high=255,
                                                shape=(obs_shape[::-1]),
                                                dtype=np.uint8)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)



class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        return getattr(self, attr)


class FlattenJacoObservationWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env
        self._obs_spec = OrderedDict()
        wrapped_obs_spec = env.observation_spec().copy()
        if 'front_close' in wrapped_obs_spec:
            spec = wrapped_obs_spec['front_close']
            # drop batch dim
            self._obs_spec['pixels'] = specs.BoundedArray(shape=spec.shape[1:],
                                                          dtype=spec.dtype,
                                                          minimum=spec.minimum,
                                                          maximum=spec.maximum,
                                                          name='pixels')
            wrapped_obs_spec.pop('front_close')

        for key, spec in wrapped_obs_spec.items():
            assert spec.dtype == np.float64
            assert type(spec) == specs.Array
        dim = np.sum(
            np.fromiter((np.int(np.prod(spec.shape))
                         for spec in wrapped_obs_spec.values()), np.int32))

        self._obs_spec['observations'] = specs.Array(shape=(dim,),
                                                     dtype=np.float32,
                                                     name='observations')

    def _transform_observation(self, time_step):
        obs = OrderedDict()

        if 'front_close' in time_step.observation:
            pixels = time_step.observation['front_close']
            time_step.observation.pop('front_close')
            pixels = np.squeeze(pixels)
            obs['pixels'] = pixels

        features = []
        for feature in time_step.observation.values():
            features.append(feature.ravel())
        obs['observations'] = np.concatenate(features, axis=0)
        return time_step._replace(observation=obs)

    def reset(self):
        time_step = self._env.reset()
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ActionRepeatWrapper(dm_env.Environment):
    def __init__(self, env, num_repeats):
        self._env = env
        self._num_repeats = num_repeats

    def step(self, action):
        reward = 0.0
        discount = 1.0
        for i in range(self._num_repeats):
            time_step = self._env.step(action)
            reward += (time_step.reward or 0.0) * discount
            discount *= time_step.discount
            if time_step.last():
                break

        return time_step._replace(reward=reward, discount=discount)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class FrameStackWrapper(dm_env.Environment):
    def __init__(self, env, num_frames, pixels_key='pixels'):
        self._env = env
        self._num_frames = num_frames
        self._frames = deque([], maxlen=num_frames)
        self._pixels_key = pixels_key

        wrapped_obs_spec = env.observation_spec()
        assert pixels_key in wrapped_obs_spec

        pixels_shape = wrapped_obs_spec[pixels_key].shape
        # remove batch dim
        if len(pixels_shape) == 4:
            pixels_shape = pixels_shape[1:]
        self._obs_spec = specs.BoundedArray(shape=np.concatenate(
            [[pixels_shape[2] * num_frames], pixels_shape[:2]], axis=0),
                                            dtype=np.uint8,
                                            minimum=0,
                                            maximum=255,
                                            name='observation')

    def _transform_observation(self, time_step):
        assert len(self._frames) == self._num_frames
        obs = np.concatenate(list(self._frames), axis=0)
        return time_step._replace(observation=obs)

    def _extract_pixels(self, time_step):
        pixels = time_step.observation[self._pixels_key]
        # remove batch dim
        if len(pixels.shape) == 4:
            pixels = pixels[0]
        return pixels.transpose(2, 0, 1).copy()

    def reset(self):
        time_step = self._env.reset()
        pixels = self._extract_pixels(time_step)
        for _ in range(self._num_frames):
            self._frames.append(pixels)
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        pixels = self._extract_pixels(time_step)
        self._frames.append(pixels)
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ActionDTypeWrapper(dm_env.Environment):
    def __init__(self, env, dtype):
        self._env = env
        wrapped_action_spec = env.action_spec()
        self._action_spec = specs.BoundedArray(wrapped_action_spec.shape,
                                               dtype,
                                               wrapped_action_spec.minimum,
                                               wrapped_action_spec.maximum,
                                               'action')

    def step(self, action):
        action = action.astype(self._env.action_spec().dtype)
        return self._env.step(action)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._action_spec

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ObservationDTypeWrapper(dm_env.Environment):
    def __init__(self, env, dtype):
        self._env = env
        self._dtype = dtype
        wrapped_obs_spec = env.observation_spec()['observations']
        self._obs_spec = specs.Array(wrapped_obs_spec.shape, dtype,
                                     'observation')

    def _transform_observation(self, time_step):
        obs = time_step.observation['observations'].astype(self._dtype)
        return time_step._replace(observation=obs)

    def reset(self):
        time_step = self._env.reset()
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ExtendedTimeStepWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env

    def reset(self):
        time_step = self._env.reset()
        return self._augment_time_step(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._augment_time_step(time_step, action)

    def _augment_time_step(self, time_step, action=None):
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        return ExtendedTimeStep(observation=time_step.observation,
                                step_type=time_step.step_type,
                                action=action,
                                reward=time_step.reward or 0.0,
                                discount=time_step.discount or 1.0)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)

# Especially for Memory-Maze environment
class OneHotActionMMZ(dm_env.Environment):
    '''
    This is based on OneHotAction wrapper in DreamerV2.
    Reference: https://github.com/danijar/dreamerv2/blob/main/dreamerv2/common/envs.py#L399
    This helps converting one-hot action to discrete action.
    
    Ex)
        Environment takes only one of 0, 1, 2, 3 as action.
        env
        wrapped_env = OneHotAction(env)
        
        env.step(2) # GOOD
        env.step([0,0,1,0]) # ERROR
        wrapped_env.step([0,0,1,0]) # GOOD, this is same as env.step(2)
    '''
    def __init__(self, env, key='action'):
        act_spec = env.action_spec()
        assert 'int' in act_spec.dtype.name # check discrete
        if not(isinstance(act_spec, specs.DiscreteArray)) and len(act_spec.shape) > 0 and act_spec.shape[0] > 1:
            raise NotImplementedError('Multi-dimensional discrete action space is not implemented')

        self._env = env
        self._key = key
        self._random = np.random.RandomState()
        self.first_error = False

    def __getattr__(self, name):
        # A problem with memory-maze class
        if name.startswith('__'):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)
    
    def observation_spec(self):
        return self._env.observation_spec()
    
    def action_spec(self):
        spec = self._env.action_spec()
        n = spec.maximum - spec.minimum + 1
        shape = (n,)
        return specs.BoundedArray(
            name="action",
            shape=shape,
            dtype=spec.dtype,
            minimum=0,
            maximum=1,
        )

    @property
    def act_space(self):
        act_space = self._env.act_space
        def sample():
            n = act_space['action'].n
            reference = np.zeros((n,), dtype=act_space['action'].dtype)
            reference[np.random.randint(0, n)] = 1
            return reference
        act_space['action'].sample = sample
        return act_space

    def step(self, action):
        index = np.argmax(action).astype(int)
        reference = np.zeros_like(action)
        reference[index] = 1
        # Changed as the implemented OneHotDist, if we sampled the mean from it, it will return the categorical probabilities
        # if np.sum(action) == 1:
        #     new_action = np.zeros_like(action)
        #     new_action[np.argmax(action)] = 1
        #     action = new_action
        # # 
        if not np.allclose(reference, action):
            # it means that the action is not one-hot
            # It happens when we get the mode of the OneHotDist -> it gives the categorical logits
            if not self.first_error:
                print(f"Warning: Invalid one-hot action: {action}, it will appear only once")
            new_action = np.zeros_like(action)
            new_action[np.argmax(action)] = 1
            action = new_action
            self.first_error = True
            # raise ValueError(f'Invalid one-hot action:\n{action}')
        return self._env.step(index)

    def reset(self, *args, **kwargs):
        ret = self._env.reset(*args, **kwargs)
        goal = None
        if isinstance(ret, tuple):
            state, goal = ret[0], ret[1]
        else:
            state = ret
        state['action'] = np.zeros(self.action_spec().shape, dtype=self.action_spec().dtype)
        if goal is None:
            return state
        else:
            return state, goal

    def _sample_action(self):
        actions = self.action_spec().shape[0]
        index = self._random.randint(0, actions)
        reference = np.zeros(actions, dtype=np.float32)
        reference[index] = 1.0
        return reference


class DMC:
  def __init__(self, env):
    self._env = env
    self._ignored_keys = []

  @property
  def obs_space(self):
    spaces = {
        'observation': self._env.observation_spec(),
        'reward': gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
        'is_first': gym.spaces.Box(0, 1, (), dtype=np.bool),
        'is_last': gym.spaces.Box(0, 1, (), dtype=np.bool),
        'is_terminal': gym.spaces.Box(0, 1, (), dtype=np.bool),
    }
    return spaces

  @property
  def act_space(self):
    spec = self._env.action_spec()
    action = gym.spaces.Box((spec.minimum)*spec.shape[0], (spec.maximum)*spec.shape[0], shape=spec.shape, dtype=np.float32)
    return {'action': action}

  def step(self, action):
    time_step = self._env.step(action)
    assert time_step.discount in (0, 1)
    obs = {
        'reward': time_step.reward,
        'is_first': False,
        'is_last': time_step.last(),
        'is_terminal': time_step.discount == 0,
        'observation': time_step.observation,
        'action' : action,
        'discount': time_step.discount
    }
    return obs

  def reset(self):
    time_step = self._env.reset()
    obs = {
        'reward': 0.0,
        'is_first': True,
        'is_last': False,
        'is_terminal': False,
        'observation': time_step.observation,
        'action' : np.zeros_like(self.act_space['action'].sample()),
        'discount': time_step.discount
    }
    return obs

  def __getattr__(self, name):
    if name == 'obs_space':
        return self.obs_space
    if name == 'act_space':
        return self.act_space
    return getattr(self._env, name)


class DMCYoga:
  def __init__(self, env, domain):
    self._size = (64, 64)
    self.domain = domain
    self._camera = env._camera
    os.environ["MUJOCO_GL"] = "egl"
    self._env = env
    self._ignored_keys = []
    from custom_dmc_tasks.yoga_utils import get_dmc_benchmark_goals
    self.goal_positions = get_dmc_benchmark_goals(self.domain)
    self.room_goal_cnt = [0 for _ in range(len(self.goal_positions))]
    self.room_goal_success = [0 for _ in range(len(self.goal_positions))]
    self.rendered_goals = self.render_all_goals() # render all of them all at once and keep them in memory
    
    
  def get_room_goal_ratio(self):
    return np.array(self.room_goal_success) / (np.array(self.room_goal_cnt) + 1e-6)

  def render_all_goals(self):
    self._env.reset()
    rooms_list = self.goal_positions
    goals_list = lambda idx: self.goal_positions[idx]
    print(f'[All] render_all_goals')
    render_all_goals_list = []
    for room_idx in range(len(rooms_list)):
      render_all_goals_list.append([])
      for goal_idx in range(len(goals_list(room_idx))):
        goal = self.render_goal(self.goal_positions[room_idx][goal_idx])
        render_all_goals_list[-1].append(goal)
    self._env.reset()
    return render_all_goals_list


  @property
  def obs_space(self):
    spaces = {
        'observation': self._env.observation_spec(),
        'reward': gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
        'is_first': gym.spaces.Box(0, 1, (), dtype=np.bool),
        'is_last': gym.spaces.Box(0, 1, (), dtype=np.bool),
        'is_terminal': gym.spaces.Box(0, 1, (), dtype=np.bool),
    }
    return spaces

  @property
  def act_space(self):
    spec = self._env.action_spec()
    action = gym.spaces.Box((spec.minimum)*spec.shape[0], (spec.maximum)*spec.shape[0], shape=spec.shape, dtype=np.float32)
    return {'action': action}

  def step(self, action):
    time_step = self._env.step(action)
    assert time_step.discount in (0, 1)
    obs = {
        'reward': time_step.reward,
        'is_first': False,
        'is_last': time_step.last(),
        'is_terminal': time_step.discount == 0,
        'observation': time_step.observation,
        "position": self._env.physics.data.qpos.copy(),
        'action' : action,
        'discount': time_step.discount
    }
    return obs

  def reset(self):
    time_step = self._env.reset()
    obs = {
        'reward': 0.0,
        'is_first': True,
        'is_last': False,
        'is_terminal': False,
        'observation': time_step.observation,
        "position": self._env.physics.data.qpos.copy(),
        'action' : np.zeros_like(self.act_space['action'].sample()),
        'discount': time_step.discount
    }
    return obs


  def __getattr__(self, name):
    if name == 'obs_space':
        return self.obs_space
    if name == 'act_space':
        return self.act_space
    return getattr(self._env, name)

  def get_excluded_qpos(self):
    # Returns the indices of qpos elements that correspond to global coordinates
    task_type = self.domain
    if task_type == 'walker':
      return [1, 5, 8] # global position and ankles
    if task_type == 'quadruped':
      return [0, 1]

  # from LEXA but modified to have the same interface as our environment
  def is_goal_achieved(self, pos, goal_pos=None, threshold=None, direction=None):
    if goal_pos is None:
      goal_pos = self._goal_position

    task_type = self.domain
    ex = self.get_excluded_qpos()
    distance = goal_pos - self._env.physics.data.qpos
    distance = np.linalg.norm(distance) - np.linalg.norm(distance[ex])
    reward = -distance

    if task_type == 'walker' :
      def get_su(_goal):
        dist = np.abs(pos - _goal)
        dist = dist[..., [0, 2, 3, 4, 6, 7]]
        dist[...,1] = shortest_angle(dist[...,1])
        return dist.max(-1)

      goal = goal_pos
      distance = min(get_su(goal), get_su(goal[..., [0, 1, 2, 6, 7, 8, 3, 4, 5]]))
      # return -distance, (distance < 0.7).astype(np.float32)
      return (distance < 0.7).astype(np.float32)

    if task_type == 'quadruped':
      def get_su(state, goal):
        dist = np.abs(state - goal)
        dist[..., [1, 2, 3]] = shortest_angle(dist[..., [1, 2, 3]])
        if self.goal_idx in [0, 1, 2, 5, 6, 7, 8, 11]:
          dist = dist[..., [0,1,2,3,4,8,12,16]]
        if self.goal_idx in [12, 13]:
          dist = dist[..., [0,1,2,3]]
        return dist.max(-1)

      def rotate(s, times=1):
        # Invariance goes as follows: add 1.57 to azimuth, circle legs 0,1,2,3 -> 1,2,3,0
        s = s.copy()
        for i in range(times):
          s[..., 1] = s[..., 1] + 1.57
          s[..., -16:] = np.roll(s[..., -16:], 12)
        return s

      def normalize(s):
        return np.concatenate((s[..., 2:3], quat2euler(s[..., 3:7]), s[..., 7:]), -1)

      state = normalize(pos)
      goal = normalize(goal_pos)
      distance = min(get_su(state, goal), get_su(rotate(state, 1), goal), get_su(rotate(state, 2), goal), get_su(
        rotate(state, 3), goal))
      # return -distance, (distance < 0.7).astype(np.float32)
      success = (distance < 0.7).astype(np.float32)
      self.room_goal_success[self._goal_room] += int(success)
      return success


  # Adapted from LEXA
  def get_goal(self, room_idx=None, goal_idx=None):
    self._goal_room = np.random.randint(len(self.rooms)) if room_idx is None else room_idx
    self.goal_idx = np.random.randint(len(self.goal_positions[self._goal_room])) if goal_idx is None else goal_idx
    self._goal_position = self.goal_positions[self._goal_room][self.goal_idx]
    self.room_goal_cnt[self._goal_room] += 1
    return self.rendered_goals[self._goal_room][self.goal_idx]
  
  def render(self):
    return self._env.physics.render(*self._size, camera_id=self._camera)
  
  

  # from LEXA
  def render_goal(self, goal):
    # current_pose = self._env.physics.get_state()
    size = self._env.physics.get_state().shape[0] - goal.shape[0]
    self._env.physics.set_state(np.concatenate((goal, np.zeros([size]))))
    self._env.step(np.zeros_like(self.act_space['action'].sample()))
    goal_img = self.render()
    # self._env.physics.set_state(current_pose)
    # self._env.step(np.zeros_like(self.act_space['action'].sample()))
    return goal_img


class MemoryMaze:
    def __init__(self, task, discrete_actions=False, no_wall_patterns=False, different_floor_textures=False, override_high_walls=False, sky=False, frame_stack=1, action_repeat=1, workdir=None, time_limit=None):
        os.environ['MUJOCO_GL'] = 'osmesa'
        from memory_maze.custom_task import FourRooms7x7, FourRooms15x15, EightRooms30x30, Maze7x7, Maze15x15
        from memory_maze.gym_wrappers import GymWrapper
        if '4x7x7' in task:
            self.layout = FourRooms7x7()
        elif '4x15x15' in task:
            self.layout = FourRooms15x15()
        elif '8x30x30' in task:
            self.layout = EightRooms30x30()
        elif 'mzx7x7' in task:
            self.layout = Maze7x7()
        elif 'mzx15x15' in task:
            self.layout = Maze15x15()
        self.max_num_steps = self.layout.max_num_steps
        env_time_limit = self.max_num_steps if time_limit is None else time_limit 
        self.discrete_actions = discrete_actions
        print(f"Create Memory-Maze environment with\n\tWall Patterns: {not no_wall_patterns}\n\tContinous: {not discrete_actions}\n\tTask: {task}\n\tDifferent Floors: {different_floor_textures}\n\tHigh Walls: {override_high_walls}\n\tSky: {sky}\n\tTime limit: {env_time_limit}")
        env = self._init_env(image_only_obs=False,
                             global_observables=True,
                             discrete_actions=self.discrete_actions,
                             top_camera=False,
                             good_visibility=False,
                             no_wall_patterns=no_wall_patterns,
                             different_floor_textures=different_floor_textures,
                             override_high_walls=override_high_walls,
                             sky=sky,
                             workdir=workdir,
                             time_limit=env_time_limit)
        if not self.discrete_actions:
            env = ActionDTypeWrapper(env, np.float32)
        env = ActionRepeatWrapper(env, action_repeat)

        # env = FrameStackWrapper(env, frame_stack, 'image')
        self._env = GymWrapper(env)
        self.no_wall_patterns = no_wall_patterns
        self.different_floor_textures = different_floor_textures
        
        
        self._compute_min_max_coords()
        self._compute_rooms()
        self.invert_origin = self.layout.invert_origin # it is made to invert the origin point from bottom-left to top-left
        self.get3D = lambda x: [x[0], 0, x[1]]
        self.num_steps = 0

        self.init_qpos = self._physics.data.qpos.copy()
        self.len_x, self.len_y = self.layout.len_x, self.layout.len_y
        self.goal_poses = self.layout.goal_poses
        self.room_goal_cnt = [0 for _ in range(len(self.rooms))]
        self.room_goal_success = [0 for _ in range(len(self.rooms))]
        
        self._goal_poses_for_render = self.layout.goal_poses_for_render
        self.resetted = False
    
    def get_room_goal_ratio(self):
        return np.array(self.room_goal_success) / (np.array(self.room_goal_cnt) + 1e-6)

    @property
    def goal_positions(self):
        return self.goal_poses

    def _init_env(self, **kwargs):
        from memory_maze.custom_task import C_memory_maze_fixed_layout
        workdir = kwargs.pop('workdir', None)
        n_targets = self.layout.layout.count('G')
        kwargs.update({'allow_same_color_targets': True})
        self._mm_env = C_memory_maze_fixed_layout(
            entity_layer=self.layout.layout,
            n_targets=n_targets,
            time_limit=kwargs.pop('time_limit'),
            target_color_in_image=False,
            seed=42,
            ret_extra_top_view=True,
            **kwargs,
        )
        self._physics = self._mm_env.env.env.env.env.physics
        return self._mm_env

            
    def _compute_min_max_coords(self):
        shape = self._env.observation_space['maze_layout'].shape
        self.min_x, self.max_x, self.min_z, self.max_z = self.layout.get_min_max_coords(shape)
    
    def _compute_rooms(self):
        self.rooms = self.layout.get_rooms()

        
    def warp_to(self, pose):
        # pose: [x, y, rot_z] -> rot_z: positive means rotation clockwise, negative means anti-clockwise
        self._mm_env.env.env.env.env.physics.set_state(np.array([pose[0], pose[1], 0, pose[2], 0, 0, 0, 0, 0, 0]))
        zero_action = np.array([0,0]) if not self.discrete_actions else np.array(0)
        return self.step(zero_action, no_inc_num_steps=True)

    def render_on_pos(self, pos):
        # as pos only was for position
        # return self.render_on_pose([pos[0], pos[2], 0])[0]
        return np.zeros((64,64,3))

    def render_on_pose(self, pose):
        # pose: [x,y, rot]
        # goal is returned as HWC
        current_pose = self._mm_env.env.env.env.env.physics.get_state()
        obs = self.warp_to(pose)
        _ = self.warp_to(current_pose)
        return obs['observation'].transpose(1,2,0), np.concatenate([obs['position'], obs['direction']], axis=-1)    # {"pos": obs["position"], "dir": obs["direction"]}
    
    def is_goal_achieved(self, pos, goal_pos=None, threshold=0.1, direction=None, return_diff=False):
        # pos: [x, 0, y]
        if goal_pos is None:
            goal_pos = self._goal_pose
        # assert direction is not None
        pose = list(deepcopy(pos))
        if direction is not None:
            # pose.extend(list(direction))
            direction = np.arctan2(direction[1], direction[0])
            direction_g =  np.arctan2(goal_pos[4], goal_pos[3])
        # else:
        goal_pos = deepcopy(goal_pos)
        goal_pos = goal_pos[:3]
        
        diff = np.sum(np.abs(np.array(pose, dtype=pos.dtype) - np.array(goal_pos, dtype=pos.dtype)))
        diff_direction = np.abs(direction - direction_g) if direction is not None else 0
        # diff = np.linalg.norm(np.array(pose, dtype=pos.dtype)  - np.array(goal_pos, dtype=pos.dtype))
        result = False
        if diff < threshold and diff_direction <= np.pi/4:
            if self._goal_is_given:
                self.room_goal_success[self._goal_room] += 1
            result = True
        diff = {"position": diff, "direction": diff_direction}
        return (result, diff) if return_diff else result
    
    def get_goal(self, room_idx=None, goal_idx=None):
        self._goal_is_given = True
        self._goal_room = np.random.randint(len(self.rooms)) if room_idx is None else room_idx
        _goal_idx = np.random.randint(len(self.goal_positions[self._goal_room])) if goal_idx is None else goal_idx
        goal_pose4render = self._goal_poses_for_render[self._goal_room][_goal_idx]
        self._goal_pose = self.goal_positions[self._goal_room][_goal_idx]
        self.room_goal_cnt[self._goal_room] += 1
        return self.render_on_pose(goal_pose4render)[0]


    def observation_spec(self):
        v = self._env.observation_space["image"]
        shape = v.shape
        new_shape = (shape[-1], shape[0], shape[1])
        spec = OrderedDict()
        spec['observation'] = specs.BoundedArray(
            name="observation",
            shape=new_shape,
            dtype=v.dtype,
            minimum=v.low.transpose(2, 0, 1),
            maximum=v.high.transpose(2, 0, 1),
        )
        # spec['observation'] = specs.BoundedArray(
        #     name="observation",
        #     shape=shape,
        #     dtype=v.dtype,
        #     minimum=v.low,#.transpose(2, 0, 1),
        #     maximum=v.high#.transpose(2, 0, 1),
        # )
        
        return spec['observation']

    # TODO: refactor
    def action_spec(self):
        # self._env.action_space.dtype = np.float32
        env_action_space_old = self._env.action_space
        if self.discrete_actions:
            env_action_space = gym.spaces.Discrete(env_action_space_old.n)
            return specs.DiscreteArray(
                name="action",
                num_values=env_action_space.n,
                dtype=env_action_space.dtype,
            )
        else:
            env_action_space = gym.spaces.Box(env_action_space_old.low, env_action_space_old.high, env_action_space_old.shape, dtype=env_action_space_old.dtype)
            return specs.BoundedArray(
                name="action",
                shape=env_action_space.shape,
                dtype=env_action_space.dtype,
                minimum=env_action_space.low,
                maximum=env_action_space.high,
            )

    @property
    def obs_space(self):
        observation = self._env.observation_space["image"]
        shape = observation.shape
        observation = gym.spaces.Box(0, 255, (shape[-1], shape[0], shape[1]), dtype=observation.dtype)
        spaces = {
            "observation": observation,
            "reward": gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
            "is_first": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_last": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_terminal": gym.spaces.Box(0, 1, (), dtype=bool),
        }
        return spaces

    @property
    def act_space(self):
        return {"action": self._env.action_space}

    def step(self, action, no_inc_num_steps=False):
        # action is not dictionary
        action = action.astype(self._env.action_space.dtype)
        time_step = self._env.step(action) # obs, reward, done, info
        obs = time_step[0]
        obs_image = obs['image'] # 64,64,3
        self.obs_image = obs_image.transpose(2, 0, 1) # 3, 64, 64
        self.topdown_view = obs["top_view"]
        self.agent_pose = {"agent_pos": obs['agent_pos'], "agent_dir": obs['agent_dir']}

        if not no_inc_num_steps:
            self.num_steps += 1

        # assert time_step.discount in (0, 1)
        obs = {
            "reward": time_step[1],
            "is_first": False,
            "is_last": self.num_steps >= self.max_num_steps, #time_step[2],
            "is_terminal": False,
            "observation": self.obs_image,
            "position": np.array(self.get3D(self.invert_origin(obs["agent_pos"])), dtype=np.float32), # flip it to top-left
            "direction": np.array(obs["agent_dir"], dtype=np.float32), 
            "action": action,
            "discount": 0.99,
        }
        return obs

    # TODO: make it better
    # TODO: change this to have a consistent reset() no extra arguement, no extra returns 
    def reset(self, with_goal=False, pos=None):
        non_episodic = not (pos is None) # if pos is not None -> it is non_episodic
        if (not self.resetted) or (not non_episodic):
            obs = self._env.reset()
            self.resetted = True
            self.obs_image = obs["image"].transpose(2, 0, 1)

        else:
            # This makes some shift for some reason and cannot fix it
            # current_pose = self._mm_env.env.env.env.env.physics.get_state()
            # self._mm_env.env.env.env.env.physics.set_state(current_pose)
            # zero_action = np.array([0,0], dtype=np.float32) if not self.discrete_actions else np.array(0, dtype=np.float32)
            # obs =  self.step(zero_action, no_inc_num_steps=True)
            obs = {"image": self.obs_image, "top_view": self.topdown_view, **self.agent_pose}          
            self.obs_image = obs["image"]
        self.topdown_view = obs["top_view"]
        self.agent_pose = {"agent_pos": obs['agent_pos'], "agent_dir": obs['agent_dir']}
        self.num_steps = 0

        obs = {
            "reward": 0.0,
            "is_first": True,
            "is_last": False,
            "is_terminal": False,
            "observation": self.obs_image,
            "position": np.array(self.get3D(self.invert_origin(obs["agent_pos"])), dtype=np.float32),
            "direction": np.array(obs["agent_dir"], dtype=np.float32),
            "discount": 0.99,
        }
        # TODO: refactor
        if not self.discrete_actions:
            obs["action"] = np.zeros_like(self.act_space["action"].sample())
        
        if with_goal:
            self._goal_is_given = True
            self._goal_room = np.random.randint(len(self.rooms))
            self._goal_pose = self.goal_positions[self._goal_room][np.random.randint(len(self.goal_positions[self._goal_room]))]
            self.room_goal_cnt[self._goal_room] += 1
            return obs, self.render_on_pos(self._goal_pose)
        else:
            self._goal_is_given = False
            return obs
    
    def get_topdown_view(self):
        return self.layout.cut_topdown_view(self.topdown_view) 


    def __getattr__(self, name):
        if name == "obs_space":
            return self.obs_space
        if name == "act_space":
            return self.act_space
        return getattr(self._env, name)



class RoomNav:
    def __init__(self, name, obs_level=1, continuous=True, size=64):

        import drstrategy_envs.miniworld
        from gym.wrappers import ResizeObservation

        room_size = 15
        door_size = 2.5
        ev = ResizeObservation(env=gym.make(f"MiniWorld-{name}-v0", obs_level=obs_level, continuous=continuous, room_size=room_size, door_size=door_size), shape=size)
        self.goal_positions = []
        if name in ["NineRooms", "SpiralNineRooms"]:
            nrows, ncols = 3, 3
        elif name in ["TwentyFiveRooms", "SpiralTwentyFiveRooms"]:
            nrows, ncols = 5, 5
        elif name == "ThreeRooms":
            nrows, ncols = 1, 3
        elif name == "FiveRooms":
            nrows, ncols = 1, 5
        elif name == "OneRoom":
            nrows, ncols = 1, 1
        else:
            raise NotImplementedError(f"Unknown room name {name}")
        self.rooms = []
        for i in range(nrows):
            for j in range(ncols):
                self.rooms.append([room_size*j, room_size*(j+0.95), room_size*i, room_size*(i+0.95)])
                if name in ["NineRooms", "SpiralNineRooms"]:
                    self.goal_positions.append([
                        [room_size*(j + 0.5) - 0.5, 0.0, room_size*(i + 0.5) - 0.5],
                        [room_size*(j + 0.3) - 0.5, 0.0, room_size*(i + 0.7) - 0.5]
                    ])
                else:
                    self.goal_positions.append([
                        [room_size*(j + 0.5) - 0.5, 0.0, room_size*(i + 0.5) - 0.5]
                    ])
        self.room_goal_cnt = [0 for _ in range(len(self.rooms))]
        self.room_goal_success = [0 for _ in range(len(self.rooms))]

        self._env = ImageToPyTorch(ev)
        
    def get_room_goal_ratio(self):
        return np.array(self.room_goal_success) / (np.array(self.room_goal_cnt) + 1e-6)

    def warp_to(self, pos):
        pos_x, pos_z = pos[0], pos[2]
        offset = 0.5
        # to manipulate if the estimated position is outside the room
        flag_pos_x_in_room, flag_pos_z_in_room = False, False
        for room in self.rooms:
            if pos_x > room[0]+offset and pos_x < room[1]-offset:
                flag_pos_x_in_room = True
            if pos_z > room[2]+offset and pos_z < room[3]-offset:
                flag_pos_z_in_room = True
            if flag_pos_x_in_room and flag_pos_z_in_room:
                break
        if not flag_pos_x_in_room:
            room_poses_x = []
            for room in self.rooms:
                room_poses_x.append(room[0]+offset)
                room_poses_x.append(room[1]-offset)
            # find the closest position
            room_poses_x = np.array(room_poses_x)
            pos_x = room_poses_x[np.argmin(np.abs(room_poses_x - pos_x))]
        if not flag_pos_z_in_room:
            room_poses_z = []
            for room in self.rooms:
                room_poses_z.append(room[0]+offset)
                room_poses_z.append(room[1]-offset)
            # find the closest position
            room_poses_z = np.array(room_poses_z)
            pos_z = room_poses_z[np.argmin(np.abs(room_poses_z - pos_z))]
        self._env.place_agent(pos=[pos_x, 0.0, pos_z])

    def render_on_pos(self, pos):
        # pos: [x,0,z]
        current_pos = self._env.agent.pos
        self.warp_to(pos)
        obs = self._env.render_top_view(POMDP=True)
        self._env.place_agent(pos=current_pos)
        import cv2
        obs = cv2.resize(obs, self.shape[::-1], interpolation=cv2.INTER_AREA)
        return obs
    
    def is_goal_achieved(self, pos, goal_pos=None, threshold=0.1, direction=None):
        if goal_pos is None:
            goal_pos = self._goal_position
        if np.sum(np.abs(pos - goal_pos)) < threshold:
            if self._goal_is_given:
                self.room_goal_success[self._goal_room] += 1
            return True
        return False

    def get_goal(self, room_idx=None, goal_idx=None):
        self._goal_is_given = True
        self._goal_room = np.random.randint(len(self.rooms)) if room_idx is None else room_idx
        _goal_idx = np.random.randint(len(self.goal_positions[self._goal_room])) if goal_idx is None else goal_idx
        self._goal_position = self.goal_positions[self._goal_room][_goal_idx]
        self.room_goal_cnt[self._goal_room] += 1
        return self.render_on_pos(self._goal_position)

    def observation_spec(self):
        v = self._env.observation_space
        return specs.BoundedArray(
            name="observation",
            shape=v.shape,
            dtype=v.dtype,
            minimum=v.low,
            maximum=v.high,
        )

    def action_spec(self):
        return specs.BoundedArray(
            name="action",
            shape=self._env.action_space.shape,
            dtype=self._env.action_space.dtype,
            minimum=self._env.action_space.low,
            maximum=self._env.action_space.high,
        )

    @property
    def obs_space(self):
        spaces = {
            "observation": self._env.observation_space,
            "reward": gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
            "is_first": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_last": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_terminal": gym.spaces.Box(0, 1, (), dtype=bool),
        }
        return spaces

    @property
    def act_space(self):
        action = self._env.action_space
        return {"action": action}

    def step(self, action):
        time_step = self._env.step(action) # obs, reward, done, info
        # assert time_step.discount in (0, 1)
        obs = {
            "reward": time_step[1],
            "is_first": False,
            "is_last": time_step[2],
            "is_terminal": False,
            "observation": time_step[0],
            "position": np.array(time_step[3]["pos"], dtype=np.float32),
            "action": action,
            "discount": 0.99,
        }
        return obs

    def reset(self, with_goal=False, pos=None):
        # notice: pos=[x,z], while the return position is [x,0,z]   
        if pos is not None:
            obs = self._env.reset(pos=pos)
        else:
            obs = self._env.reset()
        pos = np.array(self._env.agent.pos, dtype=np.float32)

        obs = {
            "reward": 0.0,
            "is_first": True,
            "is_last": False,
            "is_terminal": False,
            "observation": obs,
            "position": pos,
            "action": np.zeros_like(self.act_space["action"].sample()),
            "discount": 0.99,
        }
        if with_goal:
            self._goal_is_given = True
            self._goal_room = np.random.randint(len(self.rooms))
            self._goal_position = self.goal_positions[self._goal_room][np.random.randint(len(self.goal_positions[self._goal_room]))]
            self.room_goal_cnt[self._goal_room] += 1
            return obs, self.render_on_pos(self._goal_position)
        else:
            self._goal_is_given = False
            return obs

    def get_topdown_view(self):
        topdown_view = self._env.render_top_view(POMDP=False, frame_buffer=self._env.topdown_fb)
        # remove zero paddings
        summed = np.sum(topdown_view, axis=1)
        for idx in range(summed.shape[0]):
            if np.sum(summed[idx]) != 0:
                break
        topdown_view = topdown_view[idx:-idx, :, :]
        summed = np.sum(topdown_view, axis=0)
        for idx in range(summed.shape[0]):
            if np.sum(summed[idx]) != 0:
                break
        topdown_view = topdown_view[:, idx:(-idx+2), :]
        return topdown_view

    def __getattr__(self, name):
        if name == "obs_space":
            return self.obs_space
        if name == "act_space":
            return self.act_space
        return getattr(self._env, name)
    
class RoboKitchen:
    def __init__(self, action_repeat=2, size=(64, 64)):

        os.environ["MUJOCO_GL"] = "egl"
        import drstrategy_envs.robokitchen

        self._env = gym.make('kitchen-lexa-v0')
        self._env.sim_robot.renderer._camera_settings = dict(distance=1.86, lookat=[-0.3, .5, 2.], azimuth=90, elevation=-60)
        self._size = size
        self._action_repeat = action_repeat

        self.goal_idx = 0
        self.obs_element_goals, self.obs_element_indices, self.goal_configs = self.get_kitchen_benchmark_goals()
        self.goals = list(range(len(self.obs_element_goals)))
        self.goal_positions = [[i for i in range(len(self.obs_element_goals))]]
        self.achieved_goals = [0 for i in range(len(self.obs_element_goals))]
        self.visited_goals_cnt = [0 for i in range(len(self.obs_element_goals))]
        self.current_on_goal = False

    def observation_spec(self,):
        v = self.obs_space['observation']
        return specs.BoundedArray(name='observation', shape=v.shape, dtype=v.dtype, minimum=v.low, maximum=v.high)

    def action_spec(self,):
        return specs.BoundedArray(name='action',
            shape=self._env.action_space.shape, dtype=self._env.action_space.dtype, minimum=self._env.action_space.low, maximum=self._env.action_space.high)

    @property
    def obs_space(self):
        spaces = {
            "observation": gym.spaces.Box(0, 255, (3,) + self._size, dtype=np.uint8),
            "reward": gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
            "is_first": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_last": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_terminal": gym.spaces.Box(0, 1, (), dtype=bool),
            "state": self._env.observation_space,
        }
        return spaces

    @property
    def act_space(self):
        action = self._env.action_space
        return {"action": action}

    def step(self, action):
        total_reward = 0.0
        state = None; done=False
        for _ in range(self._action_repeat):
            state, reward, done, info = self._env.step(action)
            reward = self.compute_reward()
            total_reward += reward
            if done:
                break
        obs = {
            "observation": np.moveaxis(state, 2, 0),
            "position": self._env.sim.data.qpos.copy(),
            "reward": total_reward,
            "is_first": False,
            "is_last": done,
            "is_terminal": False,
            "action": action,
            "discount": 0.99,
        }

        return obs


    def reset(self):
        observation = self._env.reset()
        obs = {
            "reward": 0.0,
            "is_first": True,
            "is_last": False,
            "is_terminal": False,
            "observation": np.moveaxis(observation, 2, 0),
            "position": self._env.sim.data.qpos.copy(),
            "action": np.zeros_like(self.act_space["action"].sample()),
            'discount' : 1
        }

        self.goal_idx = np.random.randint(len(self.goals))
        self.goal = self.goals[self.goal_idx]
        return obs

    def compute_reward(self, goal=None):
        if goal is None:
            goal = self.goal
        qpos = self._env.sim.data.qpos.copy()

        if len(self.obs_element_indices[goal]) > 9 :
            return  -np.linalg.norm(qpos[self.obs_element_indices[goal]][9:] - self.obs_element_goals[goal][9:])
        else:
            return -np.linalg.norm(qpos[self.obs_element_indices[goal]] - self.obs_element_goals[goal])

    def reset_statics(self):
        tmp_visited_goals_cnt = []
        tmp_achieved_goals = []
        for i in range(len(self.visited_goals_cnt)):
            tmp_visited_goals_cnt.append(0)
            tmp_achieved_goals.append(0)
        self.achieved_goals = tmp_achieved_goals
        self.visited_goals_cnt = tmp_visited_goals_cnt

    def get_goal(self, room_idx=None, goal_idx=None):
        self.current_on_goal = True
        if goal_idx is None:
            self.visited_goals_cnt[self.goal_idx] += 1
        else:
            self.goal_idx = goal_idx
            self.goal = self.goals[self.goal_idx]
            self.visited_goals_cnt[self.goal_idx] += 1

        backup_qpos = self._env.sim.data.qpos.copy()
        backup_qvel = self._env.sim.data.qvel.copy()

        qpos = self.init_qpos.copy()
        qpos[self.obs_element_indices[self.goal_idx]] = self.obs_element_goals[self.goal_idx]
        self._env.set_state(qpos, np.zeros(len(self._env.init_qvel)))
        goal_obs = self._env.render('rgb_array', width=self._env.imwidth, height=self._env.imheight)
        self._env.set_state(backup_qpos, backup_qvel)

        return goal_obs

    def is_goal_achieved(self, pos, goal_pos=None, threshold=None, direction=None):
        qpos = pos
        if goal_pos is None:
            goal = self.goal
            goal_qpos = self.init_qpos.copy()
            goal_qpos[self.obs_element_indices[goal]] = self.obs_element_goals[goal]
            per_obj_success = {
            'bottom_burner' : ((qpos[9]<-0.38) and (goal_qpos[9]<-0.38)) or ((qpos[9]>-0.38) and (goal_qpos[9]>-0.38)),
            'top_burner':    ((qpos[13]<-0.38) and (goal_qpos[13]<-0.38)) or ((qpos[13]>-0.38) and (goal_qpos[13]>-0.38)),
            'light_switch':  ((qpos[17]<-0.25) and (goal_qpos[17]<-0.25)) or ((qpos[17]>-0.25) and (goal_qpos[17]>-0.25)),
            'slide_cabinet' :  abs(qpos[19] - goal_qpos[19])<0.1,
            'hinge_cabinet' :  abs(qpos[21] - goal_qpos[21])<0.2,
            'microwave' :      abs(qpos[22] - goal_qpos[22])<0.2,
            'kettle' : np.linalg.norm(qpos[23:25] - goal_qpos[23:25]) < 0.2
            }
            task_objects = self.goal_configs[goal]
            task_rel_success = 1
            for _obj in task_objects:
                task_rel_success *= per_obj_success[_obj]
                
            all_obj_success = 1
            for _obj in per_obj_success:
                all_obj_success *= per_obj_success[_obj]
                
            before_on_goal = self.current_on_goal
            # if (int(task_rel_success) == 1) and (self.current_on_goal):
            #     self.achieved_goals[self.goal_idx] += 1
            #     self.current_on_goal = False
            if (int(all_obj_success) == 1) and (self.current_on_goal):
                self.achieved_goals[self.goal_idx] += 1
                self.current_on_goal = False
            return bool(int(all_obj_success)) and before_on_goal

        else:
            goal_qpos = goal_pos.copy()
            if 1 - np.abs(np.dot(pos, goal_pos.T)) < 1e-2:
                return True
            else:
                return False

    def get_room_goal_ratio(self):
        return np.array(self.achieved_goals) / (np.array(self.visited_goals_cnt) + 1e-6)


    def get_kitchen_benchmark_goals(self):
        object_goal_vals = {
            'bottom_burner' :  [-0.88, -0.01],
            'light_switch' :  [ -0.69, -0.05],
            'slide_cabinet':  [0.37],
            'hinge_cabinet':   [0., 0.5],
            'microwave'    :   [-0.5],
            'kettle'       :   [-0.23, 0.75, 1.62]}

        object_goal_idxs = {
            'bottom_burner' :  [9, 10],
            'light_switch' :  [17, 18],
            'slide_cabinet':  [19],
            'hinge_cabinet':  [20, 21],
            'microwave'    :  [22],
            'kettle'       :  [23, 24, 25]}

        base_task_names = [ 'bottom_burner', 'light_switch', 'slide_cabinet','hinge_cabinet', 'microwave', 'kettle' ]
        goal_configs = []
        #single task
        for i in range(6):
            goal_configs.append( [base_task_names[i]])

        #two tasks
        for i,j  in combinations([1,2,3,5], 2) :
            goal_configs.append( [base_task_names[i], base_task_names[j]] )

        obs_element_goals = [] ; obs_element_indices = []
        for objects in goal_configs:
            _goal = np.concatenate([object_goal_vals[obj] for obj in objects])
            _goal_idxs = np.concatenate([object_goal_idxs[obj] for obj in objects])

            obs_element_goals.append(_goal)
            obs_element_indices.append(_goal_idxs)

        return obs_element_goals, obs_element_indices, goal_configs

    def __getattr__(self, name):
        if name == "obs_space":
            return self.obs_space
        if name == "act_space":
            return self.act_space
        return getattr(self._env, name)


class SparseMetaWorld:
    def __init__(
        self,
        name,
        seed=None,
        action_repeat=1,
        size=(64, 64),
        camera=None,
    ):
        import metaworld

        os.environ["MUJOCO_GL"] = "egl"

        # Construct the benchmark, sampling tasks
        self.ml1 = metaworld.ML1(f'{name}-v2', seed=seed)

        # Create an environment with task `pick_place`
        env_cls = self.ml1.train_classes[f'{name}-v2']
        self._env = env_cls()
        self._env._freeze_rand_vec = False
        self._size = size
        self._action_repeat = action_repeat
        self._camera = camera
        self._seed = seed
        self._tasks = self.ml1.test_tasks
        if name == 'reach':
            with open(f'../../../mw_tasks/reach_harder/{seed}.pickle', 'rb') as handle:
                self._tasks = pickle.load(handle)

    def observation_spec(self,):
        v = self.obs_space['observation']
        return specs.BoundedArray(name='observation', shape=v.shape, dtype=v.dtype, minimum=v.low, maximum=v.high)

    def action_spec(self,):
        return specs.BoundedArray(name='action',
            shape=self._env.action_space.shape, dtype=self._env.action_space.dtype, minimum=self._env.action_space.low, maximum=self._env.action_space.high)

    @property
    def obs_space(self):
        spaces = {
            "observation": gym.spaces.Box(0, 255, (3,) + self._size, dtype=np.uint8),
            "reward": gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
            "is_first": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_last": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_terminal": gym.spaces.Box(0, 1, (), dtype=bool),
            "state": self._env.observation_space,
            "success": gym.spaces.Box(0, 1, (), dtype=bool),
        }
        return spaces

    @property
    def act_space(self):
        action = self._env.action_space
        return {"action": action}

    def step(self, action):
        reward = 0.0
        success = 0.0
        for _ in range(self._action_repeat):
            state, rew, done, info = self._env.step(action)
            success += float(info["success"])
            reward += float(info["success"])
        success = min(success, 1.0)
        assert success in [0.0, 1.0]
        obs = {
            "reward": reward,
            "is_first": False,
            "is_last": False,  # will be handled by timelimit wrapper
            "is_terminal": False,  # will be handled by per_episode function
            "observation": self._env.sim.render(
                *self._size, mode="offscreen", camera_name=self._camera
            ).transpose(2, 0, 1).copy(),
            "state": state,
            'action' : action,
            "success": success,
            'discount' : 1
        }
        return obs

    def reset(self):
        # Set task to ML1 choices
        task_id = np.random.randint(0,len(self._tasks))
        return self.reset_with_task_id(task_id)

    def reset_with_task_id(self, task_id):
        if self._camera == "corner2":
            self._env.model.cam_pos[2][:] = [0.75, 0.075, 0.7]

        # Set task to ML1 choices
        task = self._tasks[task_id]
        self._env.set_task(task)

        state = self._env.reset()
        # This ensures the first observation is correct in the renderer
        self._env.sim.render(*self._size, mode="offscreen", camera_name=self._camera)
        for site in self._env._target_site_config:
            self._env._set_pos_site(*site)
        self._env.sim._render_context_offscreen._set_mujoco_buffers()

        obs = {
            "reward": 0.0,
            "is_first": True,
            "is_last": False,
            "is_terminal": False,
            "observation": self._env.sim.render(
                *self._size, mode="offscreen", camera_name=self._camera
            ).transpose(2, 0, 1).copy(),
            "state": state,
            'action' : np.zeros_like(self.act_space['action'].sample()),
            "success": False,
            'discount' : 1
        }
        return obs

    def __getattr__(self, name):
        if name == 'obs_space':
            return self.obs_space
        if name == 'act_space':
            return self.act_space
        return getattr(self._env, name)

class TimeLimit:

  def __init__(self, env, duration):
    self._env = env
    self._duration = duration
    self._step = None

  def __getattr__(self, name):
    if name.startswith('__'):
      raise AttributeError(name)
    try:
      return getattr(self._env, name)
    except AttributeError:
      raise ValueError(name)

  def step(self, action):
    assert self._step is not None, 'Must reset environment.'
    obs = self._env.step(action)
    self._step += 1
    if self._duration and self._step >= self._duration:
      obs['is_last'] = True
      self._step = None
    return obs

  def reset(self):
    self._step = 0
    return self._env.reset()

  def reset_with_task_id(self, task_id):
    self._step = 0
    return self._env.reset_with_task_id(task_id)

def _make_jaco(obs_type, domain, task, frame_stack, action_repeat, seed, img_size, exorl=False):
    env = cdmc.make_jaco(task, obs_type, seed, img_size, exorl=exorl)
    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, action_repeat)
    env = FlattenJacoObservationWrapper(env)
    env._size = (img_size, img_size)
    return env


def _make_dmc(obs_type, domain, task, frame_stack, action_repeat, seed, img_size, exorl=False):
    visualize_reward = False
    if (domain, task) in suite.ALL_TASKS:
        env = suite.load(domain,
                         task,
                         task_kwargs=dict(random=seed),
                         environment_kwargs=dict(flat_observation=True),
                         visualize_reward=visualize_reward)
    else:
        env = cdmc.make(domain,
                        task,
                        task_kwargs=dict(random=seed),
                        environment_kwargs=dict(flat_observation=True),
                        visualize_reward=visualize_reward)
    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, action_repeat)
    if obs_type == 'pixels':
        # zoom in camera for quadruped
        camera_id = dict(quadruped=2).get(domain, 0)
        render_kwargs = dict(height=img_size, width=img_size, camera_id=camera_id)
        env = pixels.Wrapper(env,
                             pixels_only=True,
                             render_kwargs=render_kwargs)
        env._size = (img_size, img_size)
        env._camera = camera_id
    return env


def make(name, obs_type, frame_stack, action_repeat, seed, img_size=84, exorl=False, cfg=None):
    assert obs_type in ['states', 'pixels']
    domain, task = name.split('_', 1)
    if domain == 'mw':
        return TimeLimit(SparseMetaWorld(task, seed=seed, action_repeat=action_repeat, size=(img_size,img_size), camera='corner2'), 250)
    elif domain == "rnav2dpomdp":
        os.environ["PYOPENGL_PLATFORM"] = "egl"
        return RoomNav(task, size=(img_size, img_size))
    elif "rnavmemorymaze" in domain:
        time_limit = float('inf') if cfg is not None and cfg.non_episodic else None
        env = MemoryMaze(task, discrete_actions='disc' in domain.lower(), 
                               no_wall_patterns='nowalltexture' in domain.lower(),
                               different_floor_textures='difffloortexture' in domain.lower(),
                               override_high_walls='highwalls' in domain.lower(),
                               sky='sky' in domain.lower(),
                               frame_stack=frame_stack, 
                               action_repeat=action_repeat,
                               time_limit=time_limit)
        if env.discrete_actions:
            env = OneHotActionMMZ(env)
        return env
    # elif "dmcyoga" in domain:
    #     env = DMCYOGA()

    elif domain == "robokitchen":
        return TimeLimit(RoboKitchen(size=(img_size, img_size), action_repeat=action_repeat), 75)

    else:
        domain = dict(cup='ball_in_cup', point='point_mass').get(domain, domain)
        make_fn = _make_jaco if domain == 'jaco' else _make_dmc
        env = make_fn(obs_type, domain, task, frame_stack, action_repeat, seed, img_size, exorl=exorl)

        if obs_type == 'pixels':
            env = FrameStackWrapper(env, frame_stack)
        else:
            env = ObservationDTypeWrapper(env, np.float32)

        env = action_scale.Wrapper(env, minimum=-1.0, maximum=+1.0)
        env = ExtendedTimeStepWrapper(env)

        # return DMC(env)
        return DMCYoga(env, domain)
