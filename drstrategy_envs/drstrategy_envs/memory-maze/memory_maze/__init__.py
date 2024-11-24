import os

# NOTE: Env MUJOCO_GL=egl is necessary for headless hardware rendering on GPU,
# but breaks when running on a CPU machine. Alternatively set MUJOCO_GL=osmesa.
if 'MUJOCO_GL' not in os.environ:
    os.environ['MUJOCO_GL'] = 'egl'

from . import tasks

try:
    # Register gym environments, if gym is available

    from typing import Callable
    from functools import partial as f

    import dm_env
    import gym
    from gym.envs.registration import register

    from .gym_wrappers import GymWrapper

    def _make_gym_env(dm_task: Callable[[], dm_env.Environment], **kwargs):
        dmenv = dm_task(**kwargs)
        return GymWrapper(dmenv)

    sizes = {
        '9x9': tasks.memory_maze_9x9,
        '11x11': tasks.memory_maze_11x11,
        '13x13': tasks.memory_maze_13x13,
        '15x15': tasks.memory_maze_15x15,
    }

    for key, dm_task in sizes.items():
        # Image-only obs space
        register(id=f'MemoryMaze-{key}-v0', entry_point=f(_make_gym_env, dm_task, image_only_obs=True))  # Standard
        register(id=f'MemoryMaze-{key}-Vis-v0', entry_point=f(_make_gym_env, dm_task, image_only_obs=True, good_visibility=True))  # Easily visible targets
        register(id=f'MemoryMaze-{key}-HD-v0', entry_point=f(_make_gym_env, dm_task, image_only_obs=True, camera_resolution=256))  # High-res camera
        register(id=f'MemoryMaze-{key}-Top-v0', entry_point=f(_make_gym_env, dm_task, image_only_obs=True, camera_resolution=256, top_camera=True))  # Top-down camera

        # Extra global observables (dict obs space)
        register(id=f'MemoryMaze-{key}-ExtraObs-v0', entry_point=f(_make_gym_env, dm_task, global_observables=True))
        register(id=f'MemoryMaze-{key}-ExtraObs-Vis-v0', entry_point=f(_make_gym_env, dm_task, global_observables=True, good_visibility=True))
        register(id=f'MemoryMaze-{key}-ExtraObs-Top-v0', entry_point=f(_make_gym_env, dm_task, global_observables=True, camera_resolution=256, top_camera=True))

        # Oracle observables with shortest path shown
        register(id=f'MemoryMaze-{key}-Oracle-v0', entry_point=f(_make_gym_env, dm_task, image_only_obs=True, global_observables=True, show_path=True))
        register(id=f'MemoryMaze-{key}-Oracle-Top-v0', entry_point=f(_make_gym_env, dm_task, image_only_obs=True, global_observables=True, show_path=True, camera_resolution=256, top_camera=True))
        register(id=f'MemoryMaze-{key}-Oracle-ExtraObs-v0', entry_point=f(_make_gym_env, dm_task, global_observables=True, show_path=True))

        # High control frequency
        register(id=f'MemoryMaze-{key}-HiFreq-v0', entry_point=f(_make_gym_env, dm_task, image_only_obs=True, control_freq=40))
        register(id=f'MemoryMaze-{key}-HiFreq-Vis-v0', entry_point=f(_make_gym_env, dm_task, image_only_obs=True, control_freq=40, good_visibility=True))
        register(id=f'MemoryMaze-{key}-HiFreq-HD-v0', entry_point=f(_make_gym_env, dm_task, image_only_obs=True, control_freq=40, camera_resolution=256))

    register(
        id=f'MemoryMaze-single-room-3x3-v0',
        entry_point=f(
            _make_gym_env,
            tasks.memory_maze_single_room_3x3,
            image_only_obs=True,
        ),
    )
    register(
        id=f'MemoryMaze-single-room-3x3-Top-v0',
        entry_point=f(
            _make_gym_env,
            tasks.memory_maze_single_room_3x3,
            image_only_obs=True,
            camera_resolution=256,
            top_camera=True,
        ),
    )
    register(
        id=f'MemoryMaze-two-rooms-3x7-v0',
        entry_point=f(
            _make_gym_env,
            tasks.memory_maze_two_rooms_3x7,
            image_only_obs=True,
        ),
    )
    register(
        id=f'MemoryMaze-two-rooms-3x7-Top-v0',
        entry_point=f(
            _make_gym_env,
            tasks.memory_maze_two_rooms_3x7,
            image_only_obs=True,
            camera_resolution=256,
            top_camera=True,
        ),
    )
    register(
        id=f'MemoryMaze-two-rooms-3x7-fixed-layout-v0',
        entry_point=f(
            _make_gym_env,
            tasks.memory_maze_two_rooms_3x7_fixed_layout,
            image_only_obs=True,
        ),
    )
    register(
        id=f'MemoryMaze-two-rooms-3x7-fixed-layout-Top-v0',
        entry_point=f(
            _make_gym_env,
            tasks.memory_maze_two_rooms_3x7_fixed_layout,
            image_only_obs=True,
            camera_resolution=256,
            top_camera=True,
        ),
    )
    register(
        id=f'MemoryMaze-four-rooms-7x7-fixed-layout-v0',
        entry_point=f(
            _make_gym_env,
            tasks.memory_maze_four_rooms_7x7_fixed_layout,
            image_only_obs=True,
        ),
    )
    register(
        id=f'MemoryMaze-four-rooms-7x7-fixed-layout-Top-v0',
        entry_point=f(
            _make_gym_env,
            tasks.memory_maze_four_rooms_7x7_fixed_layout,
            image_only_obs=True,
            camera_resolution=256,
            top_camera=True,
        ),
    )
    register(
        id=f'MemoryMaze-four-rooms-7x7-fixed-layout-random-goals-v0',
        entry_point=f(
            _make_gym_env,
            tasks.memory_maze_four_rooms_7x7_fixed_layout_random_goals,
            image_only_obs=True,
        ),
    )
    register(
        id=f'MemoryMaze-four-rooms-7x7-fixed-layout-random-goals-Top-v0',
        entry_point=f(
            _make_gym_env,
            tasks.memory_maze_four_rooms_7x7_fixed_layout_random_goals,
            image_only_obs=True,
            camera_resolution=256,
            top_camera=True,
        ),
    )
    register(
        id=f'MemoryMaze-twenty-rooms-7x39-fixed-layout-random-goals-v0',
        entry_point=f(
            _make_gym_env,
            tasks.memory_maze_twenty_rooms_7x39_fixed_layout_random_goals,
            image_only_obs=True,
        ),
    )
    register(
        id=f'MemoryMaze-twenty-rooms-7x39-fixed-layout-random-goals-Top-v0',
        entry_point=f(
            _make_gym_env,
            tasks.memory_maze_twenty_rooms_7x39_fixed_layout_random_goals,
            image_only_obs=True,
            camera_resolution=480,
            top_camera=True,
        ),
    )


except ImportError:
    print('memory_maze: gym environments not registered.')
    raise
