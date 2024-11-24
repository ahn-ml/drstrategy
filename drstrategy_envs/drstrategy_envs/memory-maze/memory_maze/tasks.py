import numpy as np
from dm_control import composer
from dm_control.locomotion.arenas import labmaze_textures

from memory_maze.maze import *
from memory_maze.oracle import DrawMinimapWrapper, PathToTargetWrapper
from memory_maze.wrappers import *

# Slow control (4Hz), so that agent without HRL has a chance.
# Native control would be ~20Hz, so this corresponds roughly to action_repeat=5.
DEFAULT_CONTROL_FREQ = 4.0


def memory_maze_single_room_3x3(**kwargs):
    return _memory_maze(
        maze_size=3,
        n_targets=3,
        time_limit=float('inf'),
        max_rooms=1,
        room_min_size=3,
        room_max_size=3,
        targets_per_room=3,
        target_color_in_image=False,
        seed=42,
        **kwargs,
    )


def memory_maze_two_rooms_3x7(**kwargs):
    return _memory_maze(
        maze_size=(7, 3),
        n_targets=6,
        time_limit=float('inf'),
        max_rooms=2,
        room_min_size=3,
        room_max_size=3,
        targets_per_room=3,
        target_color_in_image=False,
        seed=42,
        **kwargs,
    )


TWO_ROOMS_3x7_LAYOUT = """
*********
*   *   *
*   *   *
*       *
*********
"""[1:]


TWO_ROOMS_3x7_LAYOUT_RANDOM_GOALS = """
*********
*PGG*GGP*
*PPG*GPP*
*PP   PP*
*********
"""[1:]


def memory_maze_two_rooms_3x7_fixed_layout(**kwargs):
    return _memory_maze_fixed_layout(
        entity_layer=TWO_ROOMS_3x7_LAYOUT,
        n_targets=6,
        time_limit=float('inf'),
        target_color_in_image=False,
        seed=42,
        **kwargs,
    )


def memory_maze_two_rooms_3x7_fixed_layout_random_goals(**kwargs):
    return _memory_maze_fixed_layout(
        entity_layer=TWO_ROOMS_3x7_LAYOUT_RANDOM_GOALS,
        n_targets=3,
        time_limit=float('inf'),
        target_color_in_image=False,
        seed=42,
        **kwargs,
    )


FOUR_ROOMS_7x7_LAYOUT = """
*********
*G      *
* G *G  *
*   *  G*
* ***** *
*   *G  *
*G  *   *
*       *
*********
"""[1:]


FOUR_ROOMS_7x7_LAYOUT_RANDOM_GOALS = """
*********
*PP   PP*
*PPG*GPP*
* GG*GG *
* ***** *
* GG*GG *
*PPG*GPP*
*PP   PP*
*********
"""[1:]


TEN_ROOMS_7x19_LAYOUT_RANDOM_GOALS = """
*********************
*PP   PP*PP   PP*GGP*
*PPG*GPG*GPG*GPG*GPP*
* GG*PP   PP*PP   P *
* ***************** *
* GG*PP   PP*PP   P *
*PPG*GPG*GPG*GPG*GPP*
*PP   PP*PP   PP*GGP*
*********************
"""[1:]


TWENTY_ROOMS_7x39_LAYOUT_RANDOM_GOALS = """
*****************************************
*PP   PP*PP   PP*PP   PP*PP   PP*PP   PP*
*PPG*GPG*GPG*GPG*GPG*GPG*GPG*GPG*GPG*GPP*
* GG*PP   PP*PP   PP*PP   PP*PP   PP*GG *
* ************************************* *
* GG*PP   PP*PP   PP*PP   PP*PP   PP*GG *
*PPG*GPG*GPG*GPG*GPG*GPG*GPG*GPG*GPG*GPP*
*PP   PP*PP   PP*PP   PP*PP   PP*PP   PP*
*****************************************
"""[1:]


def memory_maze_four_rooms_7x7_fixed_layout(**kwargs):
    return _memory_maze_fixed_layout(
        entity_layer=FOUR_ROOMS_7x7_LAYOUT,
        n_targets=6,
        time_limit=float('inf'),
        target_color_in_image=False,
        seed=42,
        **kwargs,
    )


def memory_maze_four_rooms_7x7_fixed_layout_random_goals(**kwargs):
    return _memory_maze_fixed_layout(
        entity_layer=FOUR_ROOMS_7x7_LAYOUT_RANDOM_GOALS,
        n_targets=6,
        time_limit=float('inf'),
        target_color_in_image=False,
        seed=42,
        **kwargs,
    )


def memory_maze_ten_rooms_7x19_fixed_layout_random_goals(**kwargs):
    return _memory_maze_fixed_layout(
        entity_layer=TEN_ROOMS_7x19_LAYOUT_RANDOM_GOALS,
        n_targets=15,
        time_limit=float('inf'),
        target_color_in_image=False,
        seed=42,
        **kwargs,
    )


def memory_maze_twenty_rooms_7x39_fixed_layout_random_goals(**kwargs):
    return _memory_maze_fixed_layout(
        entity_layer=TWENTY_ROOMS_7x39_LAYOUT_RANDOM_GOALS,
        n_targets=30,
        time_limit=float('inf'),
        target_color_in_image=False,
        seed=42,
        **kwargs,
    )


def memory_maze_9x9(**kwargs):
    """
    Maze based on DMLab30-explore_goal_locations_small
    {
        mazeHeight = 11,  # with outer walls
        mazeWidth = 11,
        roomCount = 4,
        roomMaxSize = 5,
        roomMinSize = 3,
    }
    """
    return _memory_maze(9, 3, 250, **kwargs)


def memory_maze_11x11(**kwargs):
    return _memory_maze(11, 4, 500, **kwargs)


def memory_maze_13x13(**kwargs):
    return _memory_maze(13, 5, 750, **kwargs)


def memory_maze_15x15(**kwargs):
    """
    Maze based on DMLab30-explore_goal_locations_large
    {
        mazeHeight = 17,  # with outer walls
        mazeWidth = 17,
        roomCount = 9,
        roomMaxSize = 3,
        roomMaxSize = 3,
    }
    """
    return _memory_maze(15, 6, 1000, max_rooms=9, room_max_size=3, **kwargs)


def _memory_maze(
    maze_size,  # measured without exterior walls
    n_targets,
    time_limit,
    max_rooms=6,
    room_min_size=3,
    room_max_size=5,
    targets_per_room=1,
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
):
    random_state = np.random.RandomState(seed)
    walker = RollingBallWithFriction(camera_height=0.3, add_ears=top_camera)
    if isinstance(maze_size, int):
        maze_size = (maze_size, maze_size)
    arena = MazeWithTargetsArena(
        x_cells=maze_size[0] + 2,  # inner size => outer size
        y_cells=maze_size[1] + 2,
        xy_scale=2.0,
        z_height=1.5 if not good_visibility else 0.4,
        max_rooms=max_rooms,
        room_min_size=room_min_size,
        room_max_size=room_max_size,
        spawns_per_room=1,
        targets_per_room=targets_per_room,
        floor_textures=FixedFloorTexture('style_01', ['blue', 'blue_bright']),
        wall_textures=dict({
            '*': FixedWallTexture('style_01', 'yellow'),  # default wall
        }, **{str(i): labmaze_textures.WallTextures('style_01') for i in range(10)}  # variations
        ),
        skybox_texture=None,
        random_seed=random_state.randint(2147483648),
    )

    task = MemoryMazeTask(
        walker=walker,
        maze_arena=arena,
        n_targets=n_targets,
        target_radius=0.6,
        target_height_above_ground=0.5 if good_visibility else -0.6,
        enable_global_task_observables=True,  # Always add to underlying env, but not always expose in RemapObservationWrapper
        control_timestep=1.0 / control_freq,
        camera_resolution=camera_resolution,
    )

    if top_camera:
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


def _memory_maze_fixed_layout(
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
    allow_same_color_targets=True,
):
    random_state = np.random.RandomState(seed)
    walker = RollingBallWithFriction(camera_height=0.3, add_ears=top_camera)
    arena = MazeWithTargetsArenaFixedLayout(
        entity_layer=entity_layer,
        num_objects=n_targets,
        xy_scale=2.0,
        z_height=1.5 if not good_visibility else 0.4,
        floor_textures=FixedFloorTexture('style_01', ['blue', 'blue_bright']),
        wall_textures=dict({
            '*': FixedWallTexture('style_01', 'yellow'),  # default wall
        }, **{str(i): labmaze_textures.WallTextures('style_01') for i in range(10)}  # variations
        ),
        skybox_texture=None,
        random_state=random_state,
    )

    task = MemoryMazeTask(
        walker=walker,
        maze_arena=arena,
        n_targets=n_targets,
        target_radius=0.6,
        target_height_above_ground=0.5 if good_visibility else -0.6,
        enable_global_task_observables=True,  # Always add to underlying env, but not always expose in RemapObservationWrapper
        control_timestep=1.0 / control_freq,
        camera_resolution=camera_resolution,
        allow_same_color_targets=allow_same_color_targets,
    )

    if top_camera:
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
