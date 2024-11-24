from memory_maze.maze import *
from dm_control.locomotion.arenas import covering, labmaze_textures, mazes
import os
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import tempfile

def extract_unique_numbers(string):
    return set(string)


custom_colors = [
#     (213,189,114),
    # (241,231,197),
    (242,242,240),
    (193,99,66),
    (74,125,191),
    # (101,140,110),
    # (90,66,47),
#     (17,30,39),
    (3,115,62),
    (220,20,60),
    # (202,70,11),
#     (16,11,111),
    # (192,99,66),
    # (2,255,112),
    # (57,204,204),
    # (177,14,201),
    # (182,236,153),
    (170,170,170),
    # (240,19,190),
    # (254,195,0),
    # (212,207,254),
    # (74,134,253),
    (199,216,34),
    (218,112,214),
    # (126,77,95),
    (255,99,68),
    # (150,142,252),
]
# Convert the RGB values to the 0-1 range
colors_normalized = [(r/255, g/255, b/255) for r, g, b in custom_colors]

# Create a discrete colormap
custom_cmap = mcolors.ListedColormap(colors_normalized)

# Adapted from: https://github.com/danijar/director/tree/main/embodied/envs
class WallNoTexture(labmaze_textures.WallTextures):
  def _build(self, color=[0.8, 0.8, 0.8], model='labmaze_style_01'):
    self._mjcf_root = mjcf.RootElement(model=model)
    self._textures = [self._mjcf_root.asset.add(
        'texture', type='2d', name='wall', builtin='flat',
        rgb1=color, width=100, height=100)]
    
class CWallTexture(labmaze_textures.WallTextures):
  def _build(self, color=[0.8, 0.8, 0.8], model='labmaze_style_01'):
    labmaze_textures = labmaze_assets.get_wall_texture_paths('style_01')
    texture_path = labmaze_textures['blue']
    self._mjcf_root = mjcf.RootElement(model=model)
    im_frame = Image.open(texture_path)
    im_frame = im_frame.convert('RGB')
    np_frame = np.array(im_frame)
    new_color = np.array(color)
    non_black_mask = (np_frame[:, :, :3] > [70, 70, 70]).any(axis=-1)
    np_frame[non_black_mask] = new_color
    image = Image.fromarray(np_frame)
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
        # Save the image to the temporary file
        image.save(temp_file.name)
        valid_name = temp_file.name.split('/')[-1]
        self._textures= [self._mjcf_root.asset.add(  # type: ignore
                'texture', type='2d', name=f'wall_{valid_name}',
                file=temp_file.name)]

    

class CFixedFloorTexture(labmaze_textures.FloorTextures):
    """Selects a single texture instead of a collection to sample from."""

    def _build(self, style, texture_names):
        labmaze_textures = labmaze_assets.get_floor_texture_paths(style)
        self._mjcf_root = mjcf.RootElement(model='labmaze_' + style)
        self._textures = []
        if isinstance(texture_names, str):
            texture_names = [texture_names]
        for texture_name in texture_names:
            if texture_name not in labmaze_textures:
                raise ValueError(f'`texture_name` should be one of {labmaze_textures.keys()}: got {texture_name}')
            texture_path = labmaze_textures[texture_name]
#             self._textures.append(self._mjcf_root.asset.add(
#                 'texture', type='skybox', name=texture_name, builtin='gradient',
#                 rgb1=[0,0,0], rgb2=[30,20,10], width=100, height=100))

#             self._textures.append(self._mjcf_root.asset.add(
#                 'texture', type='cube', name='texgeom', builtin='gradient', mark="cross", markrgb=[100,100,100], random="0.01",
#                 rgb1=[0,0,0], rgb2=[30,20,10], width=127, height=1278))
            
            self._textures.append(self._mjcf_root.asset.add(  # type: ignore
                'texture', type='2d', name=texture_name,
                file=texture_path.format(texture_name)))
    # TODO: it does not work for now
    def append(self, texture):
        self._textures.extend(texture._textures)
        
class CFixedFloorTexture2(labmaze_textures.FloorTextures):
    """Selects a single texture instead of a collection to sample from."""

    def _build(self, style, colors):
        labmaze_textures = labmaze_assets.get_floor_texture_paths(style)
        self._mjcf_root = mjcf.RootElement(model='labmaze_' + style)
        self._textures = []
        for i,color in enumerate(colors):
#             self._textures.append(self._mjcf_root.asset.add(
#                 'texture', type='skybox', name=texture_name, builtin='gradient',
#                 rgb1=[0,0,0], rgb2=[30,20,10], width=100, height=100))

#             self._textures.append(self._mjcf_root.asset.add(
#                 'texture', type='cube', name='texgeom', builtin='gradient', mark="cross", markrgb=[100,100,100], random="0.01",
#                 rgb1=[0,0,0], rgb2=[30,20,10], width=127, height=1278))
            self._textures.append(self._mjcf_root.asset.add(  # type: ignore
                'texture', type='2d', name='floor'+str(i), builtin='flat',
                rgb1=color, width=100, height=100))
    
#             self._textures.append(self._mjcf_root.asset.add(  # type: ignore
#                 'texture', type='2d', name=texture_name,
#                 file=texture_path.format(texture_name)))
    # TODO: it does not work for now
    def append(self, texture):
        self._textures.extend(texture._textures)

        
class CFixedFloorTexture3(labmaze_textures.FloorTextures):
    """Selects a single texture instead of a collection to sample from."""

    def _build(self, style, color):
        self._mjcf_root = mjcf.RootElement(model='labmaze_' + style)
        self._textures = [self._mjcf_root.asset.add(
            'texture', type='2d', name='floor', builtin='flat',
            rgb1=color, width=100, height=100)]
    
class FixedSkyBox(labmaze_textures.SkyBox):
  """Represents a texture asset for the sky box."""

  def _build(self, style):
    labmaze_textures = labmaze_assets.get_sky_texture_paths('sky_03')
    left = labmaze_textures.left
    right = labmaze_textures.right
    up = labmaze_textures.up
    down = labmaze_textures.down
    back = labmaze_textures.back
    front = labmaze_textures.front
    self._mjcf_root = mjcf.RootElement(model='labmaze_' + style)
    self._texture = self._mjcf_root.asset.add(
        'texture', type='skybox', name='texture',
        fileleft=left, fileright=right,
        fileup=up, filedown=down,
        filefront=front, fileback=back)

  @property
  def mjcf_model(self):
    return self._mjcf_root

  @property
  def texture(self):
    return self._texture


class CMazeWithTargetsArenaFixedLayout(MazeWithTargetsArenaFixedLayout):
    def _block_variations(self):
        nblocks = 3
        _DEFAULT_FLOOR_CHAR = '.'
        floor_chars = _DEFAULT_FLOOR_CHAR + string.ascii_uppercase #['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        n, m = self._maze.variations_layer.shape[:2]
        mblocks = m * nblocks // n
        if mblocks <= 1:
            mblocks = 1
        elif 10 % mblocks == 0:
            mblocks -= 1
        ivar = 0
        for i in range(nblocks):
            for j in range(mblocks):
                i_from = i * n // nblocks
                i_to = (i + 1) * n // nblocks
                j_from = j * m // mblocks
                j_to = (j + 1) * m // mblocks
                self._change_block_char(i_from, i_to, j_from, j_to, floor_chars[ivar])
                ivar = (ivar + 1) % 10
    def _change_block_char(self, i1, i2, j1, j2, char):
        grid = self._maze.variations_layer
        i, j = np.where(grid[i1:i2, j1:j2] == '.')
        grid[i + i1, j + j1] = char
    
    def regenerate(self, random_state):
        """Generates a new maze layout.

        Patch of MazeWithTargets.regenerate() which uses random_state.
        """        
        self._maze.regenerate()
        # logging.debug('GENERATED MAZE:\n%s', self._maze.entity_layer)
        self._find_spawn_and_target_positions()

        if self._text_maze_regenerated_hook:
            self._text_maze_regenerated_hook()

        # Remove old texturing planes.
        for geom_name in self._texturing_geom_names:
            del self._mjcf_root.worldbody.geom[geom_name]
        self._texturing_geom_names = []

        # Remove old texturing materials.
        for material_name in self._texturing_material_names:
            del self._mjcf_root.asset.material[material_name]
        self._texturing_material_names = []

        # Remove old actual-wall geoms.
        self._maze_body.geom.clear()
        
        #random_state_fixed_wall = random_state  
        random_state_fixed_wall = np.random.RandomState(2)
        # next = iter(wall_textures)

        self._current_wall_texture = {
            # wall_char: wall_textures  # PATCH: use random_state for wall textures
            # Modified
            wall_char: wall_textures[0]#random_state_fixed_wall.choice(wall_textures)  # PATCH: use random_state for wall textures
            for wall_char, wall_textures in self._wall_textures.items()
        }
        for wall_char in self._wall_textures:
            self._make_wall_geoms(wall_char)
        self._make_floor_variations()

    def _make_floor_variations(self, build_tile_geoms_fn=None):
        """Fork of mazes.MazeWithTargets._make_floor_variations().

        Makes the room floors different if possible, instead of sampling randomly.
        """
        _DEFAULT_FLOOR_CHAR = '.'

        main_floor_texture = self._floor_textures[0]
        if len(self._floor_textures) > 1:
            room_floor_textures = self._floor_textures[1:]
        else:
            room_floor_textures = [main_floor_texture]

        possible_variations = _DEFAULT_FLOOR_CHAR + string.ascii_uppercase
        self._block_variations()
        variations_layer = self._maze.variations_layer

        for i_var, variation in enumerate(_DEFAULT_FLOOR_CHAR + string.ascii_uppercase):
            if variation not in variations_layer:#self._maze.variations_layer:
                break

            if build_tile_geoms_fn is None:
                # Break the floor variation down to odd-sized tiles.
                tiles = covering.make_walls(variations_layer,#self._maze.variations_layer,
                                            wall_char=variation,
                                            make_odd_sized_walls=True)
            else:
                tiles = build_tile_geoms_fn(wall_char=variation)

            if variation == _DEFAULT_FLOOR_CHAR:
                variation_texture = main_floor_texture
            else:
                variation_texture = room_floor_textures[i_var % len(room_floor_textures)]

            for i, tile in enumerate(tiles):
                tile_mid = covering.GridCoordinates(
                    (tile.start.y + tile.end.y - 1) / 2,
                    (tile.start.x + tile.end.x - 1) / 2)
                tile_pos = np.array([(tile_mid.x - self._x_offset) * self._xy_scale,
                                     -(tile_mid.y - self._y_offset) * self._xy_scale,
                                     0.0])
                tile_size = np.array([(tile.end.x - tile_mid.x - 0.5) * self._xy_scale,
                                      (tile.end.y - tile_mid.y - 0.5) * self._xy_scale,
                                      self._xy_scale])
                if variation == _DEFAULT_FLOOR_CHAR:
                    tile_name = 'floor_{}'.format(i)
                else:
                    tile_name = 'floor_{}_{}'.format(variation, i)
                self._tile_geom_names[tile.start] = tile_name
                self._texturing_material_names.append(tile_name)
                self._texturing_geom_names.append(tile_name)
                material = self._mjcf_root.asset.add(
                    'material', name=tile_name, texture=variation_texture,
                    texrepeat=(2 * tile_size[[0, 1]] / self._xy_scale))
                self._mjcf_root.worldbody.add(
                    'geom', name=tile_name, type='plane', material=material,
                    pos=tile_pos, size=tile_size, contype=0, conaffinity=0)
                
class CMemoryMazeTask(random_goal_maze.NullGoalMaze):
    # Adapted from dm_control.locomotion.tasks.RepeatSingleGoalMaze

    def __init__(self,
                 walker,
                 maze_arena,
                 n_targets=3,
                 target_radius=0.3,
                 target_height_above_ground=0.0,
                 target_reward_scale=1.0,
                 enable_global_task_observables=False,
                 camera_resolution=64,
                 physics_timestep=DEFAULT_PHYSICS_TIMESTEP,
                 control_timestep=DEFAULT_CONTROL_TIMESTEP,
                 allow_same_color_targets=False,
                 ):
        super().__init__(
            walker=walker,
            maze_arena=maze_arena,
            randomize_spawn_position=False, #True (Modified)
            randomize_spawn_rotation=False, #True (Modified)
            contact_termination=False,
            enable_global_task_observables=enable_global_task_observables,
            physics_timestep=physics_timestep,
            control_timestep=control_timestep
        )
        self._target_reward_scale = target_reward_scale
        self._targets = []
        for i in range(n_targets):
            if allow_same_color_targets:
                color = TARGET_COLORS[i % len(TARGET_COLORS)]
            else:
                color = TARGET_COLORS[i]
            target = target_sphere.TargetSphere(
                radius=target_radius,
                height_above_ground=target_radius + target_height_above_ground,
                rgb1=tuple(color * 1.0),
                rgb2=tuple(color * 1.0),
            )
            self._targets.append(target)
            self._maze_arena.attach(target)
        self._current_target_ix = 0
        self._rewarded_this_step = False
        self._targets_obtained = 0

        if enable_global_task_observables:
            # Add egocentric vectors to targets
            xpos_origin_callable = lambda phys: phys.bind(walker.root_body).xpos

            def _target_pos(physics, target):
                return physics.bind(target.geom).xpos

            for i in range(n_targets):
                # Absolute target position
                walker.observables.add_observable(
                    f'target_abs_{i}',
                    observable_lib.Generic(functools.partial(_target_pos, target=self._targets[i])),
                )
                # Relative target position
                walker.observables.add_egocentric_vector(
                    f'target_rel_{i}',
                    observable_lib.Generic(functools.partial(_target_pos, target=self._targets[i])),
                    origin_callable=xpos_origin_callable)

        self._task_observables = super().task_observables

        def _current_target_index(_):
            return self._current_target_ix

        def _current_target_color(_):
            if allow_same_color_targets:
                return TARGET_COLORS[self._current_target_ix % len(TARGET_COLORS)]
            else:
                return TARGET_COLORS[self._current_target_ix]

        self._task_observables['target_index'] = observable_lib.Generic(_current_target_index)
        self._task_observables['target_index'].enabled = True
        self._task_observables['target_color'] = observable_lib.Generic(_current_target_color)
        self._task_observables['target_color'].enabled = True

        self._walker.observables.egocentric_camera.height = camera_resolution
        self._walker.observables.egocentric_camera.width = camera_resolution
        self._maze_arena.observables.top_camera.height = camera_resolution
        self._maze_arena.observables.top_camera.width = camera_resolution

    @property
    def task_observables(self):
        return self._task_observables

    @property
    def name(self):
        return 'memory_maze'

    def initialize_episode_mjcf(self, rng: RandomState):
        self._maze_arena.regenerate(rng)  # Bypass super()._initialize_episode_mjcf(), because it ignores rng
        while True:
            ok = self._place_targets(rng)
            if not ok:
                # Could not place targets - regenerate the maze
                self._maze_arena.regenerate(rng)
                continue
            break
        self._pick_new_target(rng)

    def initialize_episode(self, physics, rng: RandomState):
        super().initialize_episode(physics, rng)
        self._rewarded_this_step = False
        self._targets_obtained = 0

    def after_step(self, physics, rng: RandomState):
        super().after_step(physics, rng)
        self._rewarded_this_step = False
        for i, target in enumerate(self._targets):
            if target.activated:
                if i == self._current_target_ix:
                    self._rewarded_this_step = True
                    self._targets_obtained += 1
                    self._pick_new_target(rng)
                target.reset(physics)  # Resets activated=False

    def should_terminate_episode(self, physics):
        return super().should_terminate_episode(physics)

    def get_reward(self, physics):
        if self._rewarded_this_step:
            return self._target_reward_scale
        return 0.0

    def _place_targets(self, rng: RandomState) -> bool:
        possible_positions = list(self._maze_arena.target_positions)
        # rng.shuffle(possible_positions) #modified
        if len(possible_positions) < len(self._targets):
            # Too few rooms - need to regenerate the maze
            return False
        for target, pos in zip(self._targets, possible_positions):
            mjcf.get_attachment_frame(target.mjcf_model).pos = pos
        return True

    def _pick_new_target(self, rng: RandomState):
        while True:
            ix = rng.randint(len(self._targets))
            if self._targets[ix].activated:
                continue  # Skip the target that the agent is touching
            self._current_target_ix = ix
            break


# ---------------------------------------------------–-------------------------
from memory_maze.tasks import *
def C_memory_maze_fixed_layout(
    entity_layer,
    n_targets,
    time_limit,
    control_freq=DEFAULT_CONTROL_FREQ,
    discrete_actions=True,
    image_only_obs=False,
    target_color_in_image=True,
    global_observables=False,
    top_camera=False,
    good_visibility=False,
    show_path=False,
    camera_resolution=64,
    seed=None,
    allow_same_color_targets=False,
    ret_extra_top_view=False,
    no_wall_patterns=False,
    different_floor_textures=True,
    override_high_walls=False,
    sky=False
):
    random_state = np.random.RandomState(seed)
    walker = RollingBallWithFriction(camera_height=0.3, add_ears=top_camera)
    if not no_wall_patterns:
        wall_textures = {'*': CWallTexture([0.8, 0.8, 0.8])}
#         cmap = custom_cmap
#         for index in range(len(cmap.colors)):
#             wall_textures[str(index)] = CWallTexture([int(i*255) for i in cmap(index)[:3]])
        # Adapted from: https://github.com/danijar/director/tree/main/embodied/envs
        cmap = plt.get_cmap('tab20')
        # To keep the light colors away from the darker colors
        for index in range(10):
            wall_textures[str(index)] = CWallTexture([int(i*255) for i in cmap(index*2)[:3]])
        for index, idx1 in enumerate([i for i in range(1, 20, 2)]):
            wall_textures[str(index+10)] = CWallTexture([int(i*255) for i in cmap(idx1*2)[:3]])
        # wall_textures = dict({
        #         '*': FixedWallTexture('style_01', 'yellow'),
        #         '0': FixedWallTexture('style_01', 'cerise'),
        #         '1': FixedWallTexture('style_01', 'blue'),
        #         '2': FixedWallTexture('style_01', 'green_bright'),
        #         '3': FixedWallTexture('style_01', 'green'),
        #         '4': FixedWallTexture('style_01', 'purple'),
        #         '5': FixedWallTexture('style_01', 'red'),
        #         '6': FixedWallTexture('style_02', 'purple'),
        #         '7': FixedWallTexture('style_02', 'dblue'),
        #         '8': FixedWallTexture('style_02', 'blue_bright'),
        #         '9': FixedWallTexture('style_02', 'lgreen'),
        #         '10': FixedWallTexture('style_02', 'yellow_bright'),
        #         '11': FixedWallTexture('style_03', 'cyan'),
        #         '12': FixedWallTexture('style_03', 'gray_bright'),
        #         '13': FixedWallTexture('style_03', 'orange'),
        #         '14': FixedWallTexture('style_03', 'gray'),
        #         '15': FixedWallTexture('style_03', 'orange_bright'),
        #         '16': FixedWallTexture('style_04', 'red_bright'),
        #     })

#     wall_textures=dict({
#             '*': FixedWallTexture('style_01', 'yellow'),  # default wall
#         }, **{str(i): labmaze_textures.WallTextures('style_01') for i in range(10)}  # variations
#         ),

    else:
        wall_textures = {'*': WallNoTexture([0.8, 0.8, 0.8])}
        cmap = plt.get_cmap('tab20')
    #     To keep the light colors away from the darker colors
        for index in range(10):
            wall_textures[str(index)] = WallNoTexture(cmap(index*2)[:3])
        for index, idx1 in enumerate([i for i in range(1, 20, 2)]):
            wall_textures[str(index+10)] = WallNoTexture(cmap(idx1)[:3])
    if different_floor_textures:
#         floor_textures = CFixedFloorTexture('style_01', ['blue', 'orange_bright', 'blue_bright', 'orange', 'blue_team', 'red_team']) #'style_02', ['blue_bright', 'blue', 'green_bright', 'green', 'orange'])
        floor_colors = []
        cmap = plt.get_cmap('tab20')
        # To keep the light colors away from the darker colors
        for index in range(10):
            floor_colors.append([int(i*255) for i in cmap(index*2)[:3]])
        for index, idx1 in enumerate([i for i in range(1, 20, 2)]):
            floor_colors.append([int(i*255) for i in cmap(idx1*2)[:3]])
#         floor_textures = {'*': CFixedFloorTexture3('style_01', floor_colors[0])}
#         for i in range(1, len(floor_colors)):
#             floor_textures[str(i)] = CFixedFloorTexture3('style_01', floor_colors[i])
        floor_textures = CFixedFloorTexture2('style_01', floor_colors)
    else:
        floor_textures = CFixedFloorTexture('style_01', ['blue'])
    
    normal_height = 1.5 if not good_visibility else 0.4
    wall_height = 20 if override_high_walls else normal_height
    arena = CMazeWithTargetsArenaFixedLayout(
        entity_layer=entity_layer,
        num_objects=n_targets,
        xy_scale=2.0,
        z_height=wall_height,
        floor_textures=floor_textures,
        wall_textures=wall_textures,
        skybox_texture=FixedSkyBox("sky_03") if sky else None, # None
        random_state=random_state,
    )
#     CFixedFloorTexture('style_01', ['blue', 'blue_bright', 'orange_bright', 'orange', 'red_team']).append(CFixedFloorTexture('style_02', ['green_bright', 'green', 'orange']))

    task = CMemoryMazeTask(
        walker=walker,
        maze_arena=arena,
        n_targets=n_targets,
        target_radius=0.6,
        target_height_above_ground=0.5 if good_visibility else -0.4,
        enable_global_task_observables=True,  # Always add to underlying env, but not always expose in RemapObservationWrapper
        control_timestep=1.0 / control_freq,
        camera_resolution=camera_resolution,
        allow_same_color_targets=allow_same_color_targets,
    )

    if top_camera or ret_extra_top_view:
        task.observables['top_camera'].enabled = True


    env = composer.Environment(
        time_limit=time_limit - 1e-3,  # subtract epsilon to make sure ep_length=time_limit*fps
        task=task,
        random_state=random_state,
        strip_singleton_obs_buffer_dim=True)

    obs_mapping = {
        'image': 'walker/egocentric_camera' if not top_camera else 'top_camera',
        'target_color': 'target_color',
    }
    
    if ret_extra_top_view:
        obs_mapping['top_view'] = 'top_camera'
    
    if global_observables:
        env = TargetsPositionWrapper(env, task._maze_arena.xy_scale, task._maze_arena.maze.width, task._maze_arena.maze.height)
        env = AgentPositionWrapper(env, task._maze_arena.xy_scale, task._maze_arena.maze.width, task._maze_arena.maze.height)
        env = MazeLayoutWrapper(env)
        obs_mapping = dict(obs_mapping, **{
            'agent_pos': 'agent_pos',
            'agent_dir': 'agent_dir',
            'targets_vec': 'targets_vec',
            'targets_pos': 'targets_pos',
            'target_vec': 'target_vec',
            'target_pos': 'target_pos',
            'maze_layout': 'maze_layout',
        })

    env = RemapObservationWrapper(env, obs_mapping)

    if target_color_in_image:
        env = TargetColorAsBorderWrapper(env)

    if show_path:
        env = PathToTargetWrapper(env)
        env = DrawMinimapWrapper(env)

    if image_only_obs:
        # assert target_color_in_image, 'Image-only observation only makes sense with target_color_in_image'
        env = ImageOnlyObservationWrapper(env)

    if discrete_actions:
        env = DiscreteActionSetWrapper(env, [
            np.array([0.0, 0.0]),  # noop
            np.array([-1.0, 0.0]),  # forward
            np.array([0.0, -1.0]),  # left
            np.array([0.0, +1.0]),  # right
            np.array([-1.0, -1.0]),  # forward + left
            np.array([-1.0, +1.0]),  # forward + right
        ])

    return env

def Cmemory_maze_four_rooms_7x7_fixed_layout(**kwargs):
    return C_memory_maze_fixed_layout(
        entity_layer=kwargs.get("layout",CFOUR_ROOMS_7x7_LAYOUT),
        n_targets=kwargs.get('n_targets', 4),
        time_limit=kwargs.get('time_limit', 500),
        target_color_in_image=False,
        seed=42,
        ret_extra_top_view=True,
        **kwargs,
    )

CFOUR_ROOMS_7x7_LAYOUT = """
*********
*P      *
* G * G *
*   *   *
* ***** *
*   *   *
* G * G *
*       *
*********
"""[1:]

CFOUR_ROOMS_15x15_SimpleLAYOUT = """
*****************
*P              *
*            G  *
*       *       *
*       *       *
*   G   *   G   *
*       *       *
*       *       *
*   *********   *
*       *       *
*       *       *
*   G   *   G   *
*       *       *
*       *       *
*  G            *
*               *
*****************
"""[1:]

CFOUR_ROOMS_15x15_LAYOUT = """
*****************
*P      *       *
*       *       *
*               *
*   G       G   *
*               *
*       *       *
*       *       *
***   *****   ***
*       *       *
*       *       *
*               *
*   G       G   *
*               *
*       *       *
*       *       *
*****************
"""[1:]

# Not finished
CNine_ROOMS_15x15_LAYOUT = """
****************
*    *    *    *
*      G       *
* G          G *
*    *    *    *
**  ***  ***  **
*    *    *    *
*   G      G   *
*       G      *
*    *    *    *
**  ***  ***  **
*    *    *    *
*  G         G *
*      G       *
*    *    *    *
****************
"""[1:]



CEIGHT_ROOMS_30x30_LAYOUT = """
*****************
*P      *       *
*       *       *
*               *
*   G       G   *
*               *
*       *       *
*       *       *
***   *****   ***
*       *       *
*       *       *
*               *
*   G       G   *
*               *
*       *       *
*       *       *
***   *****   ***
*       *       *
*       *       *
*               *
*   G       G   *
*               *
*       *       *
*       *       *
***   *****   ***
*       *       *
*       *       *
*               *
*   G       G   *
*               *
*       *       *
*       *       *
*****************
"""[1:]


CMaze_7x7_LAYOUT = """
*********
*P  *G  *
*** *** *
*G*     *
* *** ***
*     *G*
* ***   *
* G*G   *
*********
"""[1:]

CMaze_15x15_LAYOUT = """
*****************
***P      *     *
*** *  G* *   * *
*       * *  G* *
*   * *** ***** *
*         *     *
* ***   * * *** *
*G      *   *   *
*** *** * * *  G*
*** *     * *   *
*** * *   * * * *
*     *G      * *
* *** * * *  G* *
*  G      *   * *
*   *   * *** * *
*   *  G*       *
*****************
"""[1:]


class Layout:
    def __init__(self) -> None:
        self.layout = None
        self.rooms = None
        self.min_x, self.max_x, self.min_y, self.max_y = None, None, None, None
        self.invert_origin = None
        self.max_num_steps = 500
        self.len_x = None
        self.len_y = None

        
    def get_rooms(self):
        return self.rooms
    
    def get_min_max_coords(self, shape):
        # -1.5, 8.5, 8.5, -1.5
        self.min_x = -0.5 - 1
        self.max_x = -0.5 + shape[1]+2
        self.min_y = -0.5 - 1
        self.max_y = -0.5 + shape[0]+2
        return self.min_x, self.max_x, self.min_y, self.max_y
    
    def cut_topdown_view(self, topdown_view):
        return topdown_view
    
class FourRooms7x7(Layout):
    # 7.5
    def __init__(self) -> None:
        super().__init__()
        self.layout = CFOUR_ROOMS_7x7_LAYOUT
        # self.rooms = [[-0.5, 3.5, 3.5, 7.5], [3.5, 7.5, 3.5, 7.5], [-0.5, 3.5, -0.5, 3.5], [3.5, 7.5, -0.5, 3.5]]    
        # invert directions of y axis as the origin be from the top-left corner instead of bottom-left
        # invert(self.rooms)
        self.rooms = [[-1.5, 4.5, -1.5, 4.5], [4.5, 8.5, -1.5, 4.5], [-1.5, 4.5, 4.5, 8.5], [4.5, 8.5, 4.5, 8.5]]
        self.invert_origin = lambda p: [p[0], -p[1]+7]  # bad solution but works for now # equation of a line passing by (-1.5, 8.5) and (8.5, -1.5)
        self.max_num_steps = 500
        self.len_x = 7
        self.len_y = 7
        self.goal_poses_for_render = [
            [[-4.5,1,1.5]], # red
            [[4.5,1,1.7]], # green
            [[-4.5,-1,1.5]], # blue
            [[4.5,-1,1.7]], # pruple
        ]
        self.goal_poses = [
            [[0.8830412, 0.0, 2.4073212, 0.7027669, 0.7114202]], # red
            [[6.0740557, 0.0, 2.4066296, -0.6445081, 0.7645975]], # green
            [[1.3114746, 0.0, 4.7681146, 0.099178575, -0.9950696]], # blue
            [[5.5873814, 0.0, 4.76201, -0.31318974, -0.9496906]], # pruple
        ]
        

# Legacy
class FourRoomsSimple15x15(Layout):
    # 15.5
    def __init__(self) -> None:
        super().__init__()
        self.layout = CFOUR_ROOMS_15x15_SimpleLAYOUT
        # self.rooms = [[-0.5, 3.5, 3.5, 7.5], [3.5, 7.5, 3.5, 7.5], [-0.5, 3.5, -0.5, 3.5], [3.5, 7.5, -0.5, 3.5]]    
        # invert directions of y axis as the origin be from the top-left corner instead of bottom-left
        # invert(self.rooms)
        self.rooms = [[-1.5, 8.5, -1.5, 8.5], [8.5, 16.5, -1.5, 8.5], [-1.5, 8.5, 8.5, 16.5], [8.5, 16.5, 8.5, 16.5]]
        # self.invert_origin = lambda p: [p[0], -p[1]+7]  # bad solution but works for now # equation of a line passing by (-1.5, 8.5) and (8.5, -1.5)
        self.invert_origin = lambda p: [p[0], -p[1]+15]
        self.max_num_steps = 1000
        self.len_x = 15
        self.len_y = 15
        self.goal_poses = [
            [[5,14,0.4]], # red
            [[7,10,1.5], # blue
            [7,-2,1.5]], # yellow
            [[-9,-8,1.5], # beige
            [-9,-2,1.5]], # blue
            [[-9,8,1.3]], # green
        ]

class FourRooms15x15(Layout):
    # 15.5
    def __init__(self) -> None:
        super().__init__()
        self.layout = CFOUR_ROOMS_15x15_LAYOUT
        # self.rooms = [[-0.5, 3.5, 3.5, 7.5], [3.5, 7.5, 3.5, 7.5], [-0.5, 3.5, -0.5, 3.5], [3.5, 7.5, -0.5, 3.5]]    
        # invert directions of y axis as the origin be from the top-left corner instead of bottom-left
        # invert(self.rooms)
        self.rooms = [[-1.5, 8.5, -1.5, 8.5], [8.5, 16.5, -1.5, 8.5], [-1.5, 8.5, 8.5, 16.5], [8.5, 16.5, 8.5, 16.5]]
        # self.invert_origin = lambda p: [p[0], -p[1]+7]  # bad solution but works for now # equation of a line passing by (-1.5, 8.5) and (8.5, -1.5)
        self.invert_origin = lambda p: [p[0], -p[1]+15]
        self.max_num_steps = 1000
        self.len_x = 15
        self.len_y = 15
        self.goal_poses_for_render = [
            # [5,14,0.4], # red
            # [7,10,1.5], # blue
            # [7,-2,1.5], # yellow
            # [-9,-8,1.5], # beige
            # [-9,-2,1.5], # blue
            # [-9,8,1.3], # green
            # Shifted
            # [[-9,10,1.5]], # red
            # [[7,10,1.5]], # green
            # [[9,-10,-1.65]], # purple
            # [[-7,-10,-1.65]], # blue
            [[-8,6,-1.57079633]], # red
            [[6,8,0]], # green
            [[-8,-6,1.57079633]], # blue
            [[8,-6,1.57079633]], # purple
        ]
        self.goal_poses = [
            # Shifted
            # [[3.0, 0.0, 2.5, 0.0707372, -0.997495]],
            # [[11.0, 0.0, 2.5, 0.0707372, -0.997495]],
            # [[12.0, 0.0, 12.5, -0.07912089, 0.99686503]],
            # [[4.0, 0.0, 12.5, -0.07912089, 0.99686503]],
            [[3.5, 0.0, 4.5, 0.0, 1.0]],
            [[10.5, 0.0, 3.5, 1.0, 0.0]],
            [[3.5, 0.0, 10.5, -3.2051033e-09, -1.0]],
            [[11.5, 0.0, 10.5, -3.2051033e-09, -1.0]],
        ]

# Not finished
class NineRooms15x15(Layout):
    # Each room is 4x4
    # 13.5
    def __init__(self) -> None:
        super().__init__()
        self.layout = CNine_ROOMS_15x15_LAYOUT
        # self.rooms = [[-0.5, 3.5, 3.5, 7.5], [3.5, 7.5, 3.5, 7.5], [-0.5, 3.5, -0.5, 3.5], [3.5, 7.5, -0.5, 3.5]]    
        # invert directions of y axis as the origin be from the top-left corner instead of bottom-left
        # invert(self.rooms)
        self.rooms = [[-1.5, 8.5, -1.5, 8.5], [8.5, 16.5, -1.5, 8.5], [-1.5, 8.5, 8.5, 16.5], [8.5, 16.5, 8.5, 16.5]]
        # self.invert_origin = lambda p: [p[0], -p[1]+7]  # bad solution but works for now # equation of a line passing by (-1.5, 8.5) and (8.5, -1.5)
        self.invert_origin = lambda p: [p[0], -p[1]+14]
        self.max_num_steps = 1000
        self.len_x = 14
        self.len_y = 14
        self.goal_poses_for_render = [
#             [5,14,0.4], # red
#             [7,10,1.5], # blue
#             [7,-2,1.5], # yellow
#             [-9,-8,1.5], # beige
#             [-9,-2,1.5], # blue
#             [-9,8,1.3], # green
            [[-9,10,1.5]], # red
            [[7,10,1.5]], # green
            [[9,-10,-1.65]], # purple
            [[-7,-10,-1.65]], # blue
        ]
        self.goal_poses = [
            [[3.0, 0.0, 2.5, 0.0707372, -0.997495]],
            [[11.0, 0.0, 2.5, 0.0707372, -0.997495]],
            [[12.0, 0.0, 12.5, -0.07912089, 0.99686503]],
            [[4.0, 0.0, 12.5, -0.07912089, 0.99686503]],

        ]

class EightRooms30x30(Layout):
    # 31.5
    def __init__(self) -> None:
        super().__init__()
        self.layout = CEIGHT_ROOMS_30x30_LAYOUT
        # min_x, max_x, min_y, max_y
        # rooms = [[0, 7.5], [23, 31], [7.5,15], [23,31],
        #          [0, 7.5], [15.5, 23], [7.5,15], [15.5, 23],
        #          [0, 7.5], [7.5, 15.5], [7.5,15], [7.5, 15.5],
        #         [0, 7.5], [0, 7.5], [7.5,15], [0, 7.5],
        #        ]
        # invert directions of y axis as the origin to be from the top-left corner instead of bottom-left
        # invert(self.rooms)
        self.rooms = [  [0, 7.5, 0, 8], [7.5, 15, 0, 8],
                        [0, 7.5, 8, 15.5], [7.5, 15, 8, 15.5],
                        [0, 7.5, 15.5, 23.5], [7.5, 15, 15.5, 23.5],
                        [0, 7.5, 23.5, 31], [7.5, 15, 23.5, 31]
                        ]
        # self.invert_origin = lambda p: [p[0], -p[1]+7]  # bad solution but works for now # equation of a line passing by (-1.5, 8.5) and (8.5, -1.5)
        self.invert_origin = lambda p: [p[0],-p[1]+31]#[p[0], -p[1]+31]
        self.max_num_steps = 2000
        self.len_x = 31
        self.len_y = 31
        self.goal_poses_for_render = [
            [[-9,26, 1.5]], # red 0
            [[7,26, 1.5]], # green 1
            [[-10,9, 0]], # blue 2 
            [[10,9, 3.14]], #purple 3
            [[-10,-9, 0]], #yellow 4
            [[10,-9, 3.14]], #beige 5
            [[-7,-26, -1.5]], #red 6
            [[7,-26, -1.5]], #green 7
            ] 
        
        self.goal_poses = [
            [[3.0, 0.0, 2.5, 0.0707372, -0.997495]],
            [[11.0, 0.0, 2.5, 0.0707372, -0.997495]],
            [[2.5, 0.0, 11.0, 1.0, 0.0]],
            [[12.5, 0.0, 11.0, -0.99999875, -0.0015926529]],
            [[2.5, 0.0, 20.0, 1.0, 0.0]],
            [[12.5, 0.0, 20.0, -0.99999875, -0.0015926529]],
            [[4.0, 0.0, 28.5, 0.0707372, 0.997495]],
            [[11.0, 0.0, 28.5, 0.0707372, 0.997495]],
        ]

    def cut_topdown_view(self, topdown_view):
        return topdown_view[3:61, 17:47]

class Maze7x7(Layout):
    # 7.5
    def __init__(self) -> None:
        super().__init__()
        self.layout = CMaze_7x7_LAYOUT
        # self.rooms = [[-0.5, 3.5, 3.5, 7.5], [3.5, 7.5, 3.5, 7.5], [-0.5, 3.5, -0.5, 3.5], [3.5, 7.5, -0.5, 3.5]]    
        # invert directions of y axis as the origin be from the top-left corner instead of bottom-left
        # invert(self.rooms)
        self.rooms = [[-1.5, 4.5, -1.5, 4.5], [4.5, 8.5, -1.5, 4.5], [-1.5, 4.5, 4.5, 8.5], [4.5, 8.5, 4.5, 8.5]]
        self.invert_origin = lambda p: [p[0], -p[1]+7]  # bad solution but works for now # equation of a line passing by (-1.5, 8.5) and (8.5, -1.5)
        self.max_num_steps = 500
        self.len_x = 7
        self.len_y = 7
        self.goal_poses_for_render = [
            [[-6,0.2,-1.57079633]], # green
            [[3.8,6,3.14159265]], # red
            [[-5.8,-6,0], [1.8, -6, 3.14159265]], # pruple, yellow
            [[6,-3.8,-1.57079633]], # blue
        ]
        self.goal_poses = [
            [[0.5, 0.0, 3.4, 0.0, 1.0]],
            [[5.4, 0.0, 0.5, -1.0, -3.5897931e-09]],
            [[0.6, 0.0, 6.5, 1.0, 0.0],[4.4, 0.0, 6.5, -1.0, -3.5897931e-09]],
            [[6.5, 0.0, 5.4, 0.0, 1.0] ],
        ]
    
class Maze15x15(Layout):
    # 15.5
    def __init__(self) -> None:
        super().__init__()
        self.layout = CMaze_15x15_LAYOUT
        # self.rooms = [[-0.5, 3.5, 3.5, 7.5], [3.5, 7.5, 3.5, 7.5], [-0.5, 3.5, -0.5, 3.5], [3.5, 7.5, -0.5, 3.5]]    
        # invert directions of y axis as the origin be from the top-left corner instead of bottom-left
        # invert(self.rooms)
        self.rooms = [[-1.5, 8.5, -1.5, 8.5], [8.5, 16.5, -1.5, 8.5], [-1.5, 8.5, 8.5, 16.5], [8.5, 16.5, 8.5, 16.5]]
        # self.invert_origin = lambda p: [p[0], -p[1]+7]  # bad solution but works for now # equation of a line passing by (-1.5, 8.5) and (8.5, -1.5)
        self.invert_origin = lambda p: [p[0], -p[1]+15]
        self.max_num_steps = 1000
        self.len_x = 15
        self.len_y = 15
        self.goal_poses_for_render = [
            # shifted
            # [[-7,10,0], [-14,4,1.57079633]], # red/blueroom # blue/greenroom
            # [[ 7,10,0], [14,4,1.57079633]], # green/lightgreenroom # purple/redroom
            # [[0,-6,3.14], [-7,-10,3.14], [-2,-10,1.57]], # yellow/purpleroom # red/purpleroom # green/purpleroom
            # [[6,-7,0]], # beige/greyroom
            [[-4,12,0], [-14,4,1.57079633]], # red/blueroom # blue/greenroom
            [[ 8,10,0], [12,0,0]], # green/lightgreenroom # purple/redroom
            [[0,-6,3.14], [-8,-10,3.14], [-2,-12,1.57]], # yellow/purpleroom # red/purpleroom # green/purpleroom
            [[8,-8,0]], # beige/greyroom

        ]
        self.goal_poses = [
            # [[4.0, 0.0, 2.5, 1.0, 0.0],[0.5, 0.0, 5.5, -3.2051033e-09, -1.0]],
            # [[11.0, 0.0, 2.5, 1.0, 0.0],[14.5, 0.0, 5.5, -3.2051033e-09, -1.0]],
            # [[7.5, 0.0, 10.5, -0.99999875, -0.0015926529],[4.0, 0.0, 12.5, -0.99999875, -0.0015926529],[6.5, 0.0, 12.5, 0.0007963267, -0.9999997]],
            # [[10.5, 0.0, 11.0, 1.0, 0.0]]
            [[5.5, 0.0, 1.5, 1.0, 0.0],[0.5, 0.0, 5.5, -3.2051033e-09, -1.0]],
            [[11.5, 0.0, 2.5, 1.0, 0.0],[13.5, 0.0, 7.5, 1.0, 0.0]],
            [[7.5, 0.0, 10.5, -0.99999875, -0.0015926529],[3.5, 0.0, 12.5, -0.99999875, -0.0015926529],[6.5, 0.0, 13.5, 0.0007963267, -0.9999997]],
            [[11.5, 0.0, 11.5, 1.0, 0.0]]

        ]

def Cmemory_maze_fixed_layout(**kwargs):
    return C_memory_maze_fixed_layout(
        entity_layer=kwargs.pop("layout",CFOUR_ROOMS_7x7_LAYOUT),
        n_targets=kwargs.pop('n_targets', 4),
        time_limit=kwargs.pop('time_limit', 500),
        target_color_in_image=False,
        seed=42,
        ret_extra_top_view=True,
        # different_floor_textures=True,
        **kwargs,
    )
