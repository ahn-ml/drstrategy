import argparse
from pathlib import Path

import numpy as np

from memory_maze import helpers, tasks


def check_room(timestep):
    x = timestep.observation['agent_pos'][0]
    if x <= 3:
        room = 'left'
    elif x >= 4:
        room = 'right'
    else:
        raise AssertionError('Agent is not in any room.')
    return room


def check_boundary(timestep, room):
    agent_pos = timestep.observation['agent_pos']
    agent_dir = timestep.observation['agent_dir']
    x, y = agent_pos[0], agent_pos[1]
    dx, dy = agent_dir[0], agent_dir[1]
    if room == 'left':
        left_close = x < 1 and dx < 0
        right_close = x > 2 and dx > 0
    else:
        left_close = x < 5 and dx < 0
        right_close = x > 6 and dx > 0
    bottom_close = y < 1 and dy < 0
    top_close = y > 2 and dy > 0
    return left_close, right_close, bottom_close, top_close


def turn_to_target(env, timestep, target_pos, max_steps):
    obs_list = []
    action_list = []
    step_counter = 0

    agent_pos = timestep.observation['agent_pos']
    agent_dir = timestep.observation['agent_dir']

    target_dir = target_pos - agent_pos
    target_dir = target_dir / np.linalg.norm(target_dir)
    target_dir_normal = np.array([-target_dir[1], target_dir[0]])

    if np.inner(agent_dir, target_dir_normal) < 0:
        action = 2
    else:
        action = 3

    while np.inner(agent_dir, target_dir) < np.cos(np.pi / 18):
        if step_counter >= max_steps:
            raise AssertionError('Failed to turn to target.')
        timestep = env.step(action)
        assert not timestep.last()
        agent_dir = timestep.observation['agent_dir']
        obs_list.append(timestep.observation['image'])
        action_list.append(action)
        step_counter += 1

    return timestep, obs_list, action_list, step_counter


def move_one_step_toward_target(env, timestep, target_pos, max_steps):
    timestep, obs_list, action_list, step_counter = turn_to_target(env, timestep, target_pos, max_steps - 1)

    action = 1
    timestep = env.step(action)
    assert not timestep.last()
    obs_list.append(timestep.observation['image'])
    action_list.append(action)
    step_counter += 1

    return timestep, obs_list, action_list, step_counter


def explore_room(env, timestep, num_steps):
    obs_list = []
    action_list = []
    step_counter = 0
    room = check_room(timestep)

    while True:
        if (step_counter == 0 and np.random.randint(1, 4) != 1) or (step_counter > 0 and action == 1):
            left_close, right_close, bottom_close, top_close = check_boundary(timestep, room)
            agent_dir = timestep.observation['agent_dir']
            dx, dy = agent_dir[0], agent_dir[1]
            if (left_close and bottom_close) or (left_close and top_close) or (right_close and bottom_close) or (right_close and top_close):
                action = np.random.randint(2, 4)
                action_repeat = np.random.randint(8, 13)
            elif left_close and dy < 0:
                action = 2
                action_repeat = np.random.randint(5, 8)
            elif left_close and dy >= 0:
                action = 3
                action_repeat = np.random.randint(5, 8)
            elif right_close and dy < 0:
                action = 3
                action_repeat = np.random.randint(5, 8)
            elif right_close and dy >= 0:
                action = 2
                action_repeat = np.random.randint(5, 8)
            elif bottom_close and dx < 0:
                action = 3
                action_repeat = np.random.randint(5, 8)
            elif bottom_close and dx >= 0:
                action = 2
                action_repeat = np.random.randint(5, 8)
            elif top_close and dx < 0:
                action = 2
                action_repeat = np.random.randint(5, 8)
            elif top_close and dx >= 0:
                action = 3
                action_repeat = np.random.randint(5, 8)
            else:
                action = np.random.randint(2, 4)
                action_repeat = np.random.randint(0, 3)
        else:
            action = 1
            action_repeat = np.random.randint(2, 5)

        for _ in range(action_repeat):
            timestep = env.step(action)
            assert not timestep.last()
            obs_list.append(timestep.observation['image'])
            action_list.append(action)
            step_counter += 1
            if step_counter == num_steps:
                break
            if action == 1:
                assert check_room(timestep) == room
                left_close, right_close, bottom_close, top_close = check_boundary(timestep, room)
                if left_close or right_close or bottom_close or top_close:
                    break

        if step_counter == num_steps:
            return timestep, obs_list, action_list, step_counter


def switch_room(env, timestep):
    obs_list = []
    action_list = []
    step_counter = 0
    room = check_room(timestep)

    agent_pos = timestep.observation['agent_pos']
    if agent_pos[1] >= 0.5:
        if room == 'left':
            target_pos = np.array([3.0, 0.0])
        else:
            target_pos = np.array([4.0, 0.0])

        move_steps = 0
        while timestep.observation['agent_pos'][1] >= 0.5:
            if move_steps >= 25:
                raise AssertionError('Failed to move to target.')
            timestep, local_obs_list, local_action_list, local_step_counter = move_one_step_toward_target(env, timestep, target_pos, max_steps=15)
            obs_list.extend(local_obs_list)
            action_list.extend(local_action_list)
            step_counter += local_step_counter
            move_steps += 1

    if room == 'left':
        target_pos = np.array([7.0, 0.5])
    else:
        target_pos = np.array([0.0, 0.5])

    move_steps = 0
    while (room == 'left' and timestep.observation['agent_pos'][0] < 4.5) or (room == 'right' and timestep.observation['agent_pos'][0] > 2.5):
        if move_steps >= 25:
            raise AssertionError('Failed to switch room.')
        timestep, local_obs_list, local_action_list, local_step_counter = move_one_step_toward_target(env, timestep, target_pos, max_steps=15)
        obs_list.extend(local_obs_list)
        action_list.extend(local_action_list)
        step_counter += local_step_counter
        move_steps += 1

    return timestep, obs_list, action_list, step_counter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--top_camera', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--episode_start', type=int, default=0)
    parser.add_argument('--num_episodes', type=int, default=500)
    parser.add_argument('--episode_len', type=int, default=250)
    parser.add_argument('--save_dir', type=str, default='/common/home/fd213/datasets/memory-maze-3x7-reset-new-250')
    args = parser.parse_args()

    kwargs = {}
    if args.top_camera:
        args.save_dir = f'{args.save_dir}-top-down'
        kwargs['camera_resolution'] = 256

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    env = tasks.memory_maze_two_rooms_3x7_fixed_layout_random_goals(global_observables=True, top_camera=args.top_camera, **kwargs)
    num_actions = 6

    seed = args.seed * (args.episode_start + 1)
    np.random.seed(seed)

    episode_idx = args.episode_start
    while episode_idx < args.episode_start + args.num_episodes:
        timestep = env.reset()
        obs_list = [timestep.observation['image']]
        action_list = []
        step_counter = 0

        switch_room_steps = [
            np.random.randint(40, 51),
            np.random.randint(90, 101),
            150,
            np.random.randint(190, 201),
        ]

        try:
            for switch_room_step in switch_room_steps:
                timestep, local_obs_list, local_action_list, local_step_counter = explore_room(env, timestep, num_steps=switch_room_step - step_counter)
                obs_list.extend(local_obs_list)
                action_list.extend(local_action_list)
                step_counter += local_step_counter

                timestep, local_obs_list, local_action_list, local_step_counter = switch_room(env, timestep)
                obs_list.extend(local_obs_list)
                action_list.extend(local_action_list)
                step_counter += local_step_counter

            timestep, local_obs_list, local_action_list, local_step_counter = explore_room(env, timestep, num_steps=args.episode_len - step_counter)
            obs_list.extend(local_obs_list)
            action_list.extend(local_action_list)
            step_counter += local_step_counter

            observations = np.stack(obs_list, axis=0)
            actions = np.stack(action_list, axis=0)
            actions = helpers.to_onehot(actions, num_actions)
            dummy_action = np.zeros_like(actions[:1])
            actions = np.concatenate((dummy_action, actions), axis=0)
            data = {'image': observations, 'action': actions}
            path = save_dir / f'episode-{episode_idx:06}.npz'
            with path.open('wb') as f:
                np.savez_compressed(f, **data)
            print(f'Episode {episode_idx:06} done.')
            episode_idx += 1
        except AssertionError:
            continue


if __name__ == '__main__':
    main()
